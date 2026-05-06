from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import torch as th
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.usd_utils import create_joint, delete_or_deactivate_prim


TASK_NAME = "storage"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
SIZE_ORDER = ["small", "fit", "big"]
SIZE_DISPLAY = {"small": "small", "fit": "medium", "big": "large"}
PLACEMENT_RETRIES = 30

VALID_CAMERA_ACTIONS = {
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
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "reasoning", "answer", "confidence"],
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


def normalize_label(value: Any) -> str:
    return normalize_text(value).lower().replace("_", " ")


def keyed_list_to_map(items: object) -> dict[str, Any]:
    if isinstance(items, dict):
        return {normalize_text(key): value for key, value in items.items() if normalize_text(key)}
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item["_key"])] = item
    return output


def scale_list(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value), float(value), float(value)]
    if isinstance(value, list):
        return [float(item) for item in value]
    return [1.0, 1.0, 1.0]


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    if source_json is not None and "__" in source_json.stem:
        return {"skip_reason": "storage_active_single_only_skips_variant_json"}
    return {}


def containee(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("containee")
    if not isinstance(item, dict):
        raise ValueError("Missing containee in storage JSON")
    return item


def containers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("containers")
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    output = [item for item in items if isinstance(item, dict) and normalize_text(item.get("fit_check") or item.get("_key"))]
    if not output:
        raise ValueError("Missing containers in storage JSON")
    return sorted(output, key=lambda item: SIZE_ORDER.index(item.get("fit_check") or item.get("_key")) if (item.get("fit_check") or item.get("_key")) in SIZE_ORDER else 99)


def initial_pose_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("initial_poses"))


def container_object_name(container: dict[str, Any]) -> str:
    idx = container.get("idx")
    if idx is None:
        ft = normalize_text(container.get("fit_check") or container.get("_key"))
        idx = {"small": 0, "fit": 1, "big": 2}.get(ft, len(ft))
    return f"obj_container_{idx}"


def container_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {container_object_name(item): item for item in containers(payload)}


def container_display_names(payload: dict[str, Any]) -> dict[str, str]:
    items = containers(payload)
    cats = {normalize_text(item.get("fit_check") or item.get("_key")): normalize_label(item.get("cat")) for item in items}
    counts = {}
    for cat in cats.values():
        counts[cat] = counts.get(cat, 0) + 1
    output = {}
    for item in items:
        ft = normalize_text(item.get("fit_check") or item.get("_key"))
        cat = normalize_label(item.get("cat"))
        display = f"{SIZE_DISPLAY.get(ft, ft)} {cat}" if counts.get(cat, 0) > 1 else cat
        output[container_object_name(item)] = display
    return output


def display_to_container_name(payload: dict[str, Any]) -> dict[str, str]:
    return {display: name for name, display in container_display_names(payload).items()}


def target_container_name(payload: dict[str, Any]) -> str | None:
    target_ft = normalize_text(containee(payload).get("target_container"))
    if not target_ft:
        gt = normalize_label(payload.get("_ground_truth"))
        for name, display in container_display_names(payload).items():
            if gt and (gt == display or gt in display or display in gt):
                return name
        return None
    for item in containers(payload):
        if normalize_text(item.get("fit_check") or item.get("_key")) == target_ft:
            return container_object_name(item)
    return None


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    item = containee(payload)
    output = [
        {
            "type": "DatasetObject",
            "name": "obj_containee",
            "category": item["cat"],
            "model": item["model"],
            "position": [150.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale": scale_list(item.get("scale")),
        }
    ]
    for container in containers(payload):
        idx = int(container.get("idx", len(output) - 1))
        output.append(
            {
                "type": "DatasetObject",
                "name": container_object_name(container),
                "category": container["cat"],
                "model": container["model"],
                "position": [170.0 + idx * 5.0, 100.0, 100.0],
                "orientation": SQUARE_ORI,
                "scale": scale_list(container.get("scale")),
            }
        )
    return output


def orbital_views(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("orbital_views_before") or []
    if isinstance(raw, dict):
        return [value for value in raw.values() if isinstance(value, dict)]
    return [item for item in raw if isinstance(item, dict)]


def _pose_center(pose: dict[str, Any]) -> np.ndarray:
    if pose.get("position"):
        return np.array(pose["position"], dtype=float)
    lo = np.array(pose.get("aabb_min", [0.0, 0.0, 0.0]), dtype=float)
    hi = np.array(pose.get("aabb_max", [0.0, 0.0, 0.0]), dtype=float)
    return (lo + hi) * 0.5


def _fallback_camera_from_initial_poses(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poses = initial_pose_map(payload)
    points = [_pose_center(pose) for key, pose in poses.items() if key.startswith("obj_container") or key == "obj_containee"]
    if not points:
        raise ValueError("Missing storage camera and initial poses")
    center = np.mean(np.stack(points), axis=0)
    extent = np.max(np.stack(points), axis=0) - np.min(np.stack(points), axis=0)
    radius = max(float(np.linalg.norm(extent[:2])), 1.5)
    pos = np.array([center[0], center[1] - radius - 0.8, max(center[2] + 1.2, 1.2)], dtype=float)
    # Same coarse oblique viewer-camera convention used by several generated orbital views.
    quat = np.array([0.65328148, 0.27059805, 0.27059805, 0.65328148], dtype=float)
    return pos, quat, {"selection": "fallback_from_initial_poses", "target_center": center.tolist()}


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    views = orbital_views(payload)
    if views:
        chosen = None
        for view in views:
            visible = view.get("exist_obj_container") or []
            all_containers_visible = all(bool(item.get("value")) for item in visible if isinstance(item, dict))
            if view.get("exist_obj_containee") and all_containers_visible:
                chosen = view
                break
        chosen = chosen or views[0]
        return (
            np.array(chosen["eye"], dtype=float),
            np.array(chosen["quaternion_xyzw"], dtype=float),
            {"view": chosen, "selection": "first_all_visible_or_first_orbital"},
        )
    return _fallback_camera_from_initial_poses(payload)


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    item_label = normalize_label(containee(payload).get("cat"))
    container_list = "; ".join(container_display_names(payload).values())
    return "\n".join(
        [
            "You are a spatial reasoning agent.",
            f"Your goal is to determine which container the {item_label} best fits inside.",
            "",
            f"Object to place: {item_label}",
            f"Containers: {container_list}",
            "",
            "You can pick up the object and place it into containers to physically test the fit.",
            "The next image or extra action image may show the result of your action.",
            "",
            "CONFIDENCE RULES:",
            "- An object counts as inside a container if any part of it is inside and it is stable; it does not need to be fully submerged.",
            "- An object sitting completely on top of a container, not inside at all, does not count.",
            f"- Output confidence >= {threshold:.2f} only if you can clearly see the object is at least partially inside a container and stable.",
            "- If unsure, keep exploring with actions.",
            "",
            "Output EXACTLY one valid JSON object and nothing else:",
            '{"action": "<action>", "reasoning": "<description>", "answer": "<container or unsure>", "confidence": <float>}',
            "",
            "Available actions:",
            "pick up <object> | put <object> inside <container> | move_forward | move_backward | move_left | move_right | move_up | move_down | turn_left | turn_right | turn_up | turn_down | stop",
            "",
            "Guidance:",
            "- Usually prefer physical tests over camera movement when the objects are visible.",
            "- If the object fits in multiple containers, choose the smaller suitable one for efficiency.",
            "- If the object is upright on top of a container and not tilted or partially inside, it is not inside.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    item_label = normalize_label(containee(payload).get("cat"))
    container_list = "; ".join(container_display_names(payload).values())
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"You must choose which container the {item_label} best fits inside.",
            f"Containers: {container_list}",
            "Do not answer unsure.",
            'Output EXACTLY: {"answer": "<container>", "confidence": <float>, "reasoning": "<brief explanation>"}',
        ]
    )


def _parse_answer_to_container(answer: Any, payload: dict[str, Any]) -> str | None:
    text = normalize_label(answer)
    if not text or text.startswith("unsure") or text in {"not sure", "unknown"}:
        return None
    mapping = display_to_container_name(payload)
    for display, name in sorted(mapping.items(), key=lambda item: -len(item[0])):
        display_norm = normalize_label(display)
        if display_norm in text or text in display_norm:
            return name
    for name, container in container_by_name(payload).items():
        ft = normalize_label(container.get("fit_check") or container.get("_key"))
        cat = normalize_label(container.get("cat"))
        if ft and ft in text:
            return name
        if cat and cat in text and len({normalize_label(c.get("cat")) for c in containers(payload)}) == len(containers(payload)):
            return name
    return None


def parse_model_output(
    parsed: dict[str, Any],
    payload: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    answer = normalize_text(parsed.get("answer")) or "unsure"
    parsed_container = _parse_answer_to_container(answer, payload) if isinstance(payload, dict) else None
    return {
        **parsed,
        "action": action,
        "answer": answer,
        "parsed_container": parsed_container,
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
    action = normalize_text(parsed.get("action")).lower()
    if action == "stop" and parsed.get("parsed_container"):
        return True, "model_stop"
    if parsed.get("parsed_container") and float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and normalize_label(answer) not in {"unsure", "not sure", "unknown"}:
            return answer, int(item["step"])
    return "unsure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_label(answer) in {"", "unsure", "not sure", "unknown"}


def _get_aabb(obj) -> tuple[np.ndarray, np.ndarray]:
    bmin, bmax = [value.cpu().numpy() if hasattr(value, "cpu") else np.array(value) for value in obj.aabb]
    return bmin, bmax


def _is_inside(small_obj, cont_obj) -> bool:
    s_min, s_max = _get_aabb(small_obj)
    c_min, c_max = _get_aabb(cont_obj)
    tol = 0.05
    xy = (
        s_min[0] >= c_min[0] - tol
        and s_max[0] <= c_max[0] + tol
        and s_min[1] >= c_min[1] - tol
        and s_max[1] <= c_max[1] + tol
    )
    z = s_min[2] < c_max[2] - 0.05
    return bool(xy and z)


def _find_eye_camera_key(robot) -> str:
    for key in getattr(robot, "_sensors", {}):
        if "eyes:Camera:0" in key:
            return key
    return ""


def _step_sim(n: int) -> None:
    for _ in range(int(n)):
        og.sim.step()


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    task_state = task_state if task_state is not None else {}
    scene = env.scene
    poses = initial_pose_map(payload)
    for name in ["obj_containee", *container_by_name(payload).keys()]:
        obj = scene.object_registry("name", name)
        pose = poses.get(name)
        if obj is not None and pose:
            obj.set_position_orientation(
                position=th.tensor(pose["position"], dtype=th.float32),
                orientation=th.tensor(pose["quaternion_xyzw"], dtype=th.float32),
            )
            obj.keep_still()
    _step_sim(30)

    eye_camera_key = ""
    if env.robots:
        robot = env.robots[0]
        eye_camera_key = _find_eye_camera_key(robot)
        if eye_camera_key:
            try:
                pos = robot._sensors[eye_camera_key].get_position()
                pos[2] -= 0.3
                pos[0] += 0.05
                pos[1] += 0.2
                robot._sensors[eye_camera_key].set_position(pos)
                _step_sim(30)
            except Exception:
                pass
    task_state["eye_camera_key"] = eye_camera_key
    return {"eye_camera_key": eye_camera_key}


def _mock_grasp(env, obj, task_state: dict[str, Any]) -> dict[str, Any]:
    if not env.robots:
        return {"handled": True, "operation": "pick_up", "success": False, "error": "missing_robot"}
    robot = env.robots[0]
    grasp_point = robot.get_eef_position(robot.default_arm)
    obj.visual_only = True
    obj.set_position_orientation(grasp_point, SQUARE_ORI)
    obj.keep_still()
    og.sim.step()
    joint_prim_path = f"{robot.eef_links[robot.default_arm].prim_path}/ag_constraint"
    joint = create_joint(
        prim_path=joint_prim_path,
        joint_type="FixedJoint",
        body0=robot.eef_links[robot.default_arm].prim_path,
        body1=obj.root_link.prim_path,
        enabled=True,
        exclude_from_articulation=True,
    )
    task_state["ag_joint_prim_path"] = str(joint.GetPrimPath())
    _step_sim(10)
    return {"handled": True, "operation": "pick_up", "object": obj.name, "success": True}


def _mock_release(obj, task_state: dict[str, Any]) -> None:
    joint_path = task_state.pop("ag_joint_prim_path", None)
    if joint_path:
        try:
            delete_or_deactivate_prim(joint_path)
        except Exception:
            pass
    obj.set_position_orientation([100.0, 100.0, 100.0], SQUARE_ORI)
    obj.visual_only = False
    obj.keep_still()
    _step_sim(10)


def _capture_robot_eye(env, task_state: dict[str, Any], path: Path) -> Path | None:
    if not env.robots:
        return None
    key = normalize_text(task_state.get("eye_camera_key"))
    if not key:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            og.sim.render()
        img = env.robots[0]._sensors[key].get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return path
    except Exception:
        return None


def _physical_state(scene, payload: dict[str, Any]) -> str | None:
    obj = scene.object_registry("name", "obj_containee")
    if obj is None:
        return None
    for name in container_by_name(payload):
        cont = scene.object_registry("name", name)
        if cont is not None and _is_inside(obj, cont):
            return name
    return None


def _resolve_container_from_text(text: str, payload: dict[str, Any]) -> str | None:
    return _parse_answer_to_container(text, payload)


def execute_task_action(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    action: str,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    task_state: dict[str, Any] | None = None,
    step: int | None = None,
    step_image_dir: Path | None = None,
) -> dict[str, Any]:
    task_state = task_state if task_state is not None else {}
    action_lower = normalize_text(action).lower()
    scene = env.scene
    extras: list[str] = []

    if action_lower.startswith("pick up "):
        obj_desc = action_lower[len("pick up ") :].strip()
        item_label = normalize_label(containee(payload).get("cat"))
        if item_label not in obj_desc and obj_desc not in item_label and "object" not in obj_desc:
            return {"handled": True, "operation": "pick_up", "success": False, "error": "unknown_object"}
        obj = scene.object_registry("name", "obj_containee")
        result = _mock_grasp(env, obj, task_state) if obj is not None else {"handled": True, "operation": "pick_up", "success": False}
        if step_image_dir is not None:
            eye_path = _capture_robot_eye(env, task_state, step_image_dir / f"step_{step:03d}_robot_eye_pickup.png")
            if eye_path:
                extras.append(str(eye_path))
        return {**result, "physical_state": _physical_state(scene, payload), "extra_image_paths": extras}

    if action_lower.startswith("put ") and " inside " in action_lower:
        _obj_part, container_part = action_lower[len("put ") :].split(" inside ", 1)
        cont_name = _resolve_container_from_text(container_part, payload)
        obj = scene.object_registry("name", "obj_containee")
        cont = scene.object_registry("name", cont_name) if cont_name else None
        success = False
        attempts = 0
        if obj is not None and cont is not None:
            _mock_release(obj, task_state)
            for attempts in range(1, PLACEMENT_RETRIES + 1):
                try:
                    ok = sample_kinematics("onTop", obj, cont, use_last_ditch_effort=True, use_trav_map=False)
                except Exception:
                    ok = False
                _step_sim(30)
                if ok and _is_inside(obj, cont):
                    success = True
                    break
                obj.set_position_orientation([300.0, 310.0, 100.0], SQUARE_ORI)
                obj.keep_still()
                _step_sim(5)
        if step_image_dir is not None:
            eye_path = _capture_robot_eye(env, task_state, step_image_dir / f"step_{step:03d}_robot_eye_place.png")
            if eye_path:
                extras.append(str(eye_path))
        return {
            "handled": True,
            "operation": "put_inside",
            "object": "obj_containee",
            "container": cont_name,
            "success": success,
            "attempts": attempts,
            "physical_state": _physical_state(scene, payload),
            "extra_image_paths": extras,
        }

    return {"handled": False}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt_name = target_container_name(payload)
    predicted_name = _parse_answer_to_container((final_answer or {}).get("answer"), payload)
    displays = container_display_names(payload)
    return {
        "task_type": "storage",
        "question": normalize_text(payload.get("_question")),
        "gt_cont_name": gt_name,
        "gt_display": displays.get(gt_name, "") if gt_name else "",
        "predicted_container": predicted_name,
        "predicted_display": displays.get(predicted_name, "") if predicted_name else "",
        "container_choices": [{"name": name, "display": display} for name, display in displays.items()],
        "correct": predicted_name == gt_name if predicted_name and gt_name else None,
    }
