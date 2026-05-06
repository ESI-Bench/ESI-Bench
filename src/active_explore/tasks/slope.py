from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import torch as th


TASK_NAME = "slope"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

NUM_STEPS = 30
SAMPLE_EVERY = 2
CAMERA_ORDER = [2, 1, 3]

VALID_ACTIONS = {
    "put_on_slope",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "turn_up",
    "turn_down",
    "stop",
}

ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "answer", "reasoning", "confidence"],
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "confidence", "reasoning"],
}


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def display_category(value: Any) -> str:
    return normalize_text(value).replace("_", " ")


def keyed_list_to_map(items: object) -> dict[str, Any]:
    if isinstance(items, dict):
        return {normalize_text(key): value for key, value in items.items() if normalize_text(key)}
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item["_key"])] = item
    return output


def objects_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("objects"))


def task_object(payload: dict[str, Any]) -> dict[str, Any]:
    return objects_map(payload).get("task_obj") or {}


def object_category(payload: dict[str, Any]) -> str:
    return normalize_text(payload.get("obj_category") or task_object(payload).get("category"))


def object_model(payload: dict[str, Any]) -> str:
    return normalize_text(payload.get("obj_model") or task_object(payload).get("model"))


def object_label(payload: dict[str, Any]) -> str:
    return display_category(object_category(payload))


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")) or "unknown_scene", normalize_text(payload.get("room")) or "unknown_room"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    if normalize_text(payload.get("question_id")):
        return normalize_text(payload["question_id"])
    run_idx = payload.get("run_idx")
    return f"run_{int(run_idx):03d}" if run_idx is not None else source_path.stem


def pose_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [pose for pose in payload.get("camera_poses") or [] if isinstance(pose, dict)]


def first_pose_by_view(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = {}
    for pose in pose_records(payload):
        view_idx = pose.get("view_idx")
        if view_idx is None:
            continue
        view_label = int(view_idx) + 1
        output.setdefault(view_label, pose)
    return output


def correct_answer(payload: dict[str, Any]) -> str:
    answer = normalize_answer(payload.get("_ground_truth"))
    if answer in {"yes", "no"}:
        return answer
    return "no" if bool(payload.get("ground_truth_slid")) or bool(payload.get("ground_truth_fallen")) else "yes"


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"yes", "stable", "not slide", "does not slide", "will not slide", "stay stable"}:
        return "yes"
    if text in {"no", "unstable", "slide", "slides", "fallen", "falls", "will slide", "will fall"}:
        return "no"
    if "not slide" in text or "stay stable" in text or "stays stable" in text:
        return "yes"
    if "slide" in text or "fall" in text:
        return "no"
    return "unsure"


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    if not object_category(payload) or not object_model(payload):
        return {"skip_reason": "missing_task_object_category_or_model"}
    if not payload.get("slope") or not payload.get("slope_scale") or not payload.get("slope_quaternion"):
        return {"skip_reason": "missing_slope_geometry"}
    obj = task_object(payload)
    if obj.get("pos_on_floor") is None:
        return {"skip_reason": "missing_task_object_pos_on_floor"}
    if not pose_records(payload):
        return {"skip_reason": "missing_camera_poses"}
    return {}


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "PrimitiveObject",
            "name": "slope",
            "primitive_type": "Cube",
            "fixed_base": True,
            "scale": payload["slope_scale"],
            "position": [0.0, 0.0, 50.0],
            "orientation": payload["slope_quaternion"],
            "visual_only": False,
            "rgba": [0.6, 0.55, 0.45, 1.0],
        },
        {
            "type": "DatasetObject",
            "name": "task_obj",
            "category": object_category(payload),
            "model": object_model(payload),
            "position": [0.0, 0.0, 50.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    ]


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_view = first_pose_by_view(payload)
    chosen_view = None
    for view in CAMERA_ORDER:
        if view in by_view:
            chosen_view = view
            break
    if chosen_view is None and by_view:
        chosen_view = sorted(by_view)[0]
    pose = by_view[chosen_view] if chosen_view is not None else pose_records(payload)[0]
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"view": pose, "view_label": chosen_view, "selection": "preferred_slope_view"},
    )


def apply_friction(obj, static_friction: float, dynamic_friction: float) -> None:
    try:
        import omnigibson.lazy as lazy

        mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
            prim_path=f"{obj.prim_path}/Looks/{obj.name}_friction_mat",
            name=f"{obj.name}_friction_mat",
            static_friction=float(static_friction),
            dynamic_friction=float(dynamic_friction),
            restitution=0.1,
        )
        for link in obj.links.values():
            for mesh in link.collision_meshes.values():
                mesh.apply_physics_material(mat)
    except Exception:
        pass


def check_visibility(obj_names: list[str]) -> dict[str, bool]:
    for _ in range(10):
        og.sim.step()
    try:
        raw = og.sim._viewer_camera._annotators["seg_instance"].get_data()
        visible_str = " ".join(str(value) for value in raw["info"]["idToLabels"].values())
        return {name: name in visible_str for name in obj_names}
    except Exception:
        return {name: True for name in obj_names}


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = task_state if task_state is not None else {}
    slope = env.scene.object_registry("name", "slope")
    task_obj = env.scene.object_registry("name", "task_obj")
    if slope is None or task_obj is None:
        raise ValueError("Slope environment missing slope or task_obj")

    slope.set_position_orientation(
        position=th.tensor(payload["slope"]["position"], dtype=th.float32),
        orientation=th.tensor(payload["slope_quaternion"], dtype=th.float32),
    )
    slope.keep_still()
    for _ in range(10):
        og.sim.step()

    obj = task_object(payload)
    task_obj.set_position_orientation(
        position=th.tensor(obj["pos_on_floor"], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    task_obj.keep_still()
    for _ in range(20):
        og.sim.step()

    apply_friction(slope, payload.get("static_friction", 0.5), payload.get("dynamic_friction", 0.5))
    apply_friction(task_obj, payload.get("static_friction", 0.5), payload.get("dynamic_friction", 0.5))

    by_view = first_pose_by_view(payload)
    chosen_view = camera_info.get("view_label")
    chosen_pose = camera_info.get("view")
    visibility = None
    for view in CAMERA_ORDER:
        pose = by_view.get(view)
        if not pose:
            continue
        og.sim._viewer_camera.set_position_orientation(
            position=np.array(pose["position"], dtype=float),
            orientation=np.array(pose["quaternion_xyzw"], dtype=float),
        )
        for _ in range(5):
            og.sim.render()
        visibility = check_visibility(["slope", "task_obj"])
        if visibility.get("slope") and visibility.get("task_obj"):
            chosen_view = view
            chosen_pose = pose
            break

    state.update(
        {
            "slope": slope,
            "task_obj": task_obj,
            "obj_pos_init": obj.get("pos_init"),
            "placed_on_slope": False,
            "chosen_view": chosen_view,
            "camera_visibility": visibility,
        }
    )
    if chosen_pose:
        return {
            "camera_override": {
                "position": chosen_pose["position"],
                "quaternion_xyzw": chosen_pose["quaternion_xyzw"],
            },
            "chosen_view": chosen_view,
            "visibility": visibility,
        }
    return {}


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    obj = object_label(payload)
    return (
        f"You are a spatial and physical reasoning agent. Your goal is to determine whether a {obj} will stay stable and not slide on the slope.\n"
        "You can take the following actions:\n"
        f"  - put_on_slope: place the {obj} on the slope and observe what happens\n"
        "  - move_forward | move_backward | move_left | move_right | move_up | move_down\n"
        "  - turn_left | turn_right | turn_up | turn_down | stop\n\n"
        "Use move/turn actions only if you cannot observe the object clearly. In most cases use put_on_slope.\n"
        "After put_on_slope, you will receive a sequence of physics simulation frames showing the object motion over time.\n\n"
        "Output ONLY valid JSON in this exact format:\n"
        '{"action": "<action>", "reasoning": "<one or two sentences>", "answer": "yes" or "no" or "unsure", "confidence": <float 0-1>}\n\n'
        '"yes" = the object stays stable and will not slide. "no" = the object slides or falls. '
        f'Output answer="unsure" and confidence < {threshold} if not yet sure.'
    )


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> str:
    return (
        f"Choose whether the {object_label(payload)} stays stable on the slope. "
        "Return JSON with answer exactly 'yes' or 'no', confidence, and reasoning."
    )


def parse_model_output(parsed: dict[str, Any], payload: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    action = normalize_text(parsed.get("action")).lower() or "put_on_slope"
    if action == "<end>":
        action = "stop"
    if action not in VALID_ACTIONS:
        action = "put_on_slope"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        **parsed,
        "action": action,
        "answer": normalize_answer(parsed.get("answer")),
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def should_stop(
    parsed: dict[str, Any],
    history: list[dict[str, Any]],
    step: int,
    max_steps: int,
    min_steps: int,
    threshold: float,
) -> tuple[bool, str]:
    action = normalize_text(parsed.get("action"))
    answer = normalize_answer(parsed.get("answer"))
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if action == "put_on_slope":
        return False, ""
    if action == "stop":
        return True, "model_stop"
    if answer in {"yes", "no"} and confidence >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return False


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_answer(item.get("answer"))
        if answer in {"yes", "no"}:
            return answer, int(item["step"])
    if history:
        return "unsure", int(history[-1]["step"])
    return "unsure", -1


def execute_task_action(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    action: str,
    pos: np.ndarray,
    quat: np.ndarray,
    task_state: dict[str, Any] | None = None,
    step: int | None = None,
    step_image_dir: Path | None = None,
) -> dict[str, Any]:
    state = task_state or {}
    if normalize_text(action).lower() != "put_on_slope":
        return {"handled": False}
    slope = state.get("slope") or env.scene.object_registry("name", "slope")
    task_obj = state.get("task_obj") or env.scene.object_registry("name", "task_obj")
    if slope is None or task_obj is None:
        return {"handled": True, "operation": "put_on_slope", "success": False, "reason": "missing_slope_or_task_obj"}

    og.sim._viewer_camera.set_position_orientation(position=np.array(pos, dtype=float), orientation=np.array(quat, dtype=float))
    slope_bmin, slope_bmax = [x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x) for x in slope.aabb]
    slope_mid = (slope_bmin + slope_bmax) / 2.0
    task_obj.set_position_orientation(
        position=th.tensor([slope_mid[0], slope_mid[1], slope_mid[2] + 0.05], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    task_obj.keep_still()

    frame_dir = step_image_dir or Path(".")
    frame_dir.mkdir(parents=True, exist_ok=True)
    all_frame_paths: list[str] = []
    for frame_idx in range(1, NUM_STEPS + 1):
        og.sim.step()
        og.sim.render()
        rgb = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
        frame_path = frame_dir / f"place_step_{int(step or 0):03d}_frame_{frame_idx:03d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        all_frame_paths.append(str(frame_path))
    sampled_paths = all_frame_paths[::SAMPLE_EVERY]

    state["placed_on_slope"] = True
    state["all_frame_paths"] = all_frame_paths
    state["sampled_frame_paths"] = sampled_paths
    return {
        "handled": True,
        "operation": "put_on_slope",
        "success": True,
        "extra_image_paths": sampled_paths,
        "all_frame_paths": all_frame_paths,
        "sampled_frame_paths": sampled_paths,
        "position": pos.tolist(),
        "quaternion_xyzw": quat.tolist(),
    }


def post_action_query(
    model_client,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    image_path: Path,
    history: list[dict[str, Any]],
    action_result: dict[str, Any],
    config: Any,
    task_state: dict[str, Any] | None = None,
    reference_image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if action_result.get("operation") != "put_on_slope" or not action_result.get("all_frame_paths"):
        return {}
    frame_paths = [Path(path) for path in action_result["all_frame_paths"]]
    prior_history = history[:-1]
    contents: list[Any] = []
    if prior_history:
        summary = "\n".join(
            f"Step {item['step']}: action={item['action']} answer={item['answer']} conf={item['confidence']:.2f} reasoning={item['reasoning']}"
            for item in prior_history
        )
        contents.append("Action history so far:\n" + summary)
    contents.extend(frame_paths)
    contents.append(
        f"You just placed the {object_label(payload)} on the slope. "
        f"The {len(frame_paths)} images above show {NUM_STEPS} physics simulation frames from one camera view. "
        "Observe the object's motion over time. Does it stay stable and not slide?"
    )
    parsed, raw_text, finish_reason = model_client.generate_json(
        contents=contents,
        system_instruction=build_system_prompt(payload, config.threshold, config.min_steps, camera_info, task_state),
        response_schema=ACTION_RESPONSE_SCHEMA,
        max_output_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        fallback={"action": "put_on_slope", "answer": "unsure", "confidence": 0.0, "reasoning": "post-action parse fallback"},
    )
    parsed = parse_model_output(parsed, payload, task_state)
    answer = parsed["answer"]
    confidence = float(parsed["confidence"])
    reasoning = parsed["reasoning"]
    history_update = {
        "answer": answer,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_output_post_action": raw_text,
        "finish_reason_post_action": finish_reason,
    }
    if answer in {"yes", "no"} and confidence >= config.threshold:
        return {
            "handled": True,
            "history_update": history_update,
            "final_answer": {
                "answer": answer,
                "answer_step": int(history[-1]["step"]) if history else -1,
                "confidence": confidence,
                "reasoning": reasoning,
                "steps": int(history[-1]["step"]) if history else 0,
                "stopped_by": "post_put_on_slope_confidence_threshold",
            },
        }
    return {"handled": True, "history_update": history_update}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = task_state or {}
    answer = normalize_answer(final_answer.get("answer"))
    gt = correct_answer(payload)
    slide_dist = None
    task_obj = state.get("task_obj")
    obj_pos_init = state.get("obj_pos_init")
    if task_obj is not None and obj_pos_init is not None:
        try:
            pos_final = task_obj.get_position_orientation()[0].cpu().numpy()
            slide_dist = float(np.linalg.norm(pos_final[:2] - np.array(obj_pos_init, dtype=float)[:2]))
        except Exception:
            slide_dist = None
    return {
        "answer": answer,
        "ground_truth": gt,
        "correct": answer == gt,
        "obj_category": object_category(payload),
        "slope_angle_deg": payload.get("slope_angle_deg"),
        "gt_slid": payload.get("ground_truth_slid"),
        "gt_fallen": payload.get("ground_truth_fallen"),
        "placed_on_slope": bool(state.get("placed_on_slope")),
        "chosen_view": state.get("chosen_view"),
        "slide_dist_m": round(slide_dist, 5) if slide_dist is not None else None,
        "static_friction": payload.get("static_friction"),
        "dynamic_friction": payload.get("dynamic_friction"),
    }
