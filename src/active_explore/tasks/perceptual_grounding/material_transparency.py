from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import torch as th
from scipy.spatial.transform import Rotation


TASK_NAME = "material_transparency"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

VALID_ACTIONS = [
    "pickup",
    "pour",
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
]

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
    return normalize_text(value).replace("_", " ")


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower().replace("_", " ")
    if text in {"yes", "y", "true", "inside", "in"}:
        return "yes"
    if text in {"no", "n", "false", "outside", "not inside", "out"}:
        return "no"
    if "not sure" in text or "unsure" in text or "unknown" in text or not text:
        return "not sure"
    if "not inside" in text or "outside" in text:
        return "no"
    if "inside" in text:
        return "yes"
    return "not sure"


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


def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    fwd = np.array(target, dtype=float) - np.array(eye, dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    fwd /= norm
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    cam_up /= np.linalg.norm(cam_up)
    return Rotation.from_matrix(np.column_stack([right, cam_up, -fwd])).as_quat()


def aabb_numpy(obj) -> tuple[np.ndarray, np.ndarray]:
    bmin, bmax = obj.aabb
    bmin = bmin.cpu().numpy() if hasattr(bmin, "cpu") else np.asarray(bmin)
    bmax = bmax.cpu().numpy() if hasattr(bmax, "cpu") else np.asarray(bmax)
    return bmin.astype(float), bmax.astype(float)


def object_pose(obj) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = obj.get_position_orientation()
    pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
    quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
    return pos.astype(float), quat.astype(float)


def set_object_pose(obj, pos: np.ndarray, quat: np.ndarray, visual_only: bool = True) -> None:
    if visual_only:
        try:
            obj.visual_only = True
        except Exception:
            pass
    obj.set_position_orientation(
        position=th.tensor(np.asarray(pos, dtype=float), dtype=th.float32),
        orientation=th.tensor(np.asarray(quat, dtype=float), dtype=th.float32),
    )
    try:
        obj.keep_still()
    except Exception:
        pass


def small_object_inside_container(obj_small, obj_container, margin: float = 0.01) -> bool:
    c_bmin, c_bmax = aabb_numpy(obj_container)
    s_bmin, s_bmax = aabb_numpy(obj_small)
    s_center = (s_bmin + s_bmax) / 2.0
    inside_xy = bool(
        c_bmin[0] - margin <= s_center[0] <= c_bmax[0] + margin
        and c_bmin[1] - margin <= s_center[1] <= c_bmax[1] + margin
    )
    inside_z = bool(c_bmin[2] - margin <= s_center[2] <= c_bmax[2] + margin)
    return inside_xy and inside_z


def make_action_camera(obj_container, obj_small) -> dict[str, list[float]]:
    c_bmin, c_bmax = aabb_numpy(obj_container)
    s_bmin, s_bmax = aabb_numpy(obj_small)
    bmin = np.minimum(c_bmin, s_bmin)
    bmax = np.maximum(c_bmax, s_bmax)
    center = (bmin + bmax) / 2.0
    span = np.maximum(bmax - bmin, 0.05)
    distance = max(0.70, float(max(span[0], span[1])) * 5.0)
    eye = np.array(
        [
            center[0] + distance * 0.35,
            center[1] - distance,
            max(bmax[2] + 0.75, center[2] + 0.65),
        ],
        dtype=float,
    )
    target = np.array([center[0], center[1], min(bmax[2], center[2] + 0.08)], dtype=float)
    quat = look_at_quat(eye, target)
    return {"position": eye.tolist(), "quaternion_xyzw": quat.tolist(), "target": target.tolist()}


def set_viewer_camera(camera: dict[str, Any] | None) -> None:
    if not camera:
        return
    og.sim._viewer_camera.set_position_orientation(
        position=np.array(camera["position"], dtype=float),
        orientation=np.array(camera["quaternion_xyzw"], dtype=float),
    )
    for _ in range(4):
        og.sim.render()


def capture_viewer_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        og.sim.render()
    rgb = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return str(path)


def objects_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("objects"))


def camera_pose_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    poses = payload.get("camera_poses")
    if isinstance(poses, dict):
        return [value for value in poses.values() if isinstance(value, dict)]
    if isinstance(poses, list):
        return [pose for pose in poses if isinstance(pose, dict)]
    return []


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def small_obj_label(payload: dict[str, Any]) -> str:
    return normalize_label(payload.get("small_obj_cat") or objects_map(payload).get("obj_small", {}).get("category"))


def container_label(payload: dict[str, Any]) -> str:
    return normalize_label(payload.get("container_cat") or objects_map(payload).get("obj_container", {}).get("category"))


def correct_answer(payload: dict[str, Any]) -> str:
    return normalize_answer(payload.get("_ground_truth"))


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for key in ("obj_container", "obj_small"):
        obj = objects_map(payload).get(key)
        if not obj:
            raise ValueError(f"Missing {key} in transparent JSON")
        output.append(
            {
                "type": "DatasetObject",
                "name": key,
                "category": obj["category"],
                "model": obj["model"],
                "scale": scale_list(obj.get("scale")),
                "position": obj["position"],
                "orientation": obj["quaternion_xyzw"],
            }
        )
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poses = camera_pose_records(payload)
    if not poses:
        raise ValueError("Missing camera_poses in transparent JSON")
    pose = None
    for candidate in poses:
        if candidate.get("_key") == "0.png":
            pose = candidate
            break
    pose = pose or poses[0]
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"view": pose, "selection": "0.png_or_first_camera_pose"},
    )


def postprocess_env(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj_container = env.scene.object_registry("name", "obj_container")
    obj_small = env.scene.object_registry("name", "obj_small")
    if obj_container is None or obj_small is None:
        return {}
    action_camera = make_action_camera(obj_container, obj_small)
    state = task_state if task_state is not None else {}
    state["action_camera"] = action_camera
    return {"camera_override": action_camera, "action_camera": action_camera}


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    small_obj = small_obj_label(payload)
    container = container_label(payload)
    return "\n".join(
        [
            "You are a meticulous spatial reasoning expert.",
            f"Your goal is to determine if the {small_obj} is inside the {container}.",
            "",
            "STRICT CONFIDENCE RULES:",
            "1. PROXIMITY IS NOT CONTAINMENT: Side-view occlusion is a trap. If the object is behind the container, it can look like it is inside.",
            "2. Do not output confidence above 0.5 for yes unless you have strong proof.",
            f"3. Proof requires seeing the base of the {small_obj} sitting on the interior bottom surface of the {container} from a top-down angle.",
            "4. DISPROOF SEARCH: Look for a gap. If you see any space between the objects, answer no with high confidence.",
            "5. REASONING REQUIREMENT: Describe the rim of the container and where the object sits relative to it.",
            "6. If you cannot see inside the rim, you are not allowed to be sure.",
            "7. ACTION ADVICE: Use move_up and turn_down to look into the container.",
            "",
            "You may also use these physical actions:",
            "- pickup: lift the container upward to reveal whether the small object moves with it or remains separate.",
            "- pour: move the small object upward and out past the container rim to test containment.",
            "",
            "Output ONLY valid JSON:",
            '{"action": "<name>", "reasoning": "<proof-based-description>", "answer": "yes|no|not sure", "confidence": <float>}',
            "",
            "Available actions:",
            "move_forward | move_backward | move_left | move_right | move_up | move_down | turn_left | turn_right | turn_up | turn_down | pickup | pour | stop",
            f"Confidence threshold to stop: {threshold:.2f}.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    small_obj = small_obj_label(payload)
    container = container_label(payload)
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"You must decide whether the {small_obj} is inside the {container}.",
            "Do not answer not sure.",
            'Output EXACTLY: {"answer": "yes|no", "confidence": <float>, "reasoning": "<brief explanation>"}',
        ]
    )


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
    if action == "<end>":
        action = "stop"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    answer = normalize_answer(parsed.get("answer"))
    return {
        **parsed,
        "action": action,
        "answer": answer,
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
    if normalize_text(parsed.get("action")).lower() == "stop":
        return True, "model_stop"
    if float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_answer(item.get("answer"))
        if answer in {"yes", "no"}:
            return answer, int(item["step"])
    return "not sure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_answer(answer) == "not sure"


def _pickup_container(obj_container, obj_small) -> dict[str, Any]:
    start_pos, quat = object_pose(obj_container)
    small_start_pos, small_quat = object_pose(obj_small)
    _, s_bmax = aabb_numpy(obj_small)
    lift_total = 0.30
    end_pos = start_pos.copy()
    end_pos[2] = max(start_pos[2] + lift_total, float(s_bmax[2]) + 0.18)
    lift_delta = end_pos - start_pos
    small_inside = small_object_inside_container(obj_small, obj_container)
    small_end_pos = small_start_pos + lift_delta if small_inside else small_start_pos.copy()
    set_object_pose(obj_container, end_pos, quat, visual_only=True)
    if small_inside:
        set_object_pose(obj_small, small_end_pos, small_quat, visual_only=True)
    for _ in range(8):
        og.sim.step()
        set_object_pose(obj_container, end_pos, quat, visual_only=True)
        if small_inside:
            set_object_pose(obj_small, small_end_pos, small_quat, visual_only=True)
    return {
        "handled": True,
        "operation": "pickup_container",
        "success": True,
        "lift_m": float(end_pos[2] - start_pos[2]),
        "start_position": start_pos.tolist(),
        "end_position": end_pos.tolist(),
        "small_object_inside_before_pickup": small_inside,
        "small_object_start_position": small_start_pos.tolist(),
        "small_object_end_position": small_end_pos.tolist(),
        "small_object_lifted": small_inside,
    }


def _pour_small_obj(obj_small, obj_container) -> dict[str, Any]:
    start_pos, start_quat = object_pose(obj_small)
    c_bmin, c_bmax = aabb_numpy(obj_container)
    s_bmin, s_bmax = aabb_numpy(obj_small)
    obj_half_width = max(0.03, float(max(s_bmax[0] - s_bmin[0], s_bmax[1] - s_bmin[1])) / 2.0)
    end_pos = start_pos.copy()
    end_pos[0] = float(c_bmax[0]) + obj_half_width + 0.08
    end_pos[1] = float((c_bmin[1] + c_bmax[1]) / 2.0)
    end_pos[2] = float(c_bmax[2]) + max(0.08, float(s_bmax[2] - s_bmin[2]) / 2.0)
    set_object_pose(obj_small, end_pos, start_quat, visual_only=True)
    for _ in range(8):
        og.sim.step()
        set_object_pose(obj_small, end_pos, start_quat, visual_only=True)
    final_bmin, _final_bmax = aabb_numpy(obj_small)
    cleared = bool(final_bmin[2] > c_bmax[2] and end_pos[0] > c_bmax[0])
    return {
        "handled": True,
        "operation": "pour_small_obj",
        "success": cleared,
        "start_position": start_pos.tolist(),
        "end_position": end_pos.tolist(),
        "container_top_z": float(c_bmax[2]),
    }


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
    action = normalize_text(action).lower()
    scene = env.scene
    state = task_state if task_state is not None else {}
    if action == "pickup":
        obj_container = scene.object_registry("name", "obj_container")
        obj_small = scene.object_registry("name", "obj_small")
        if obj_container is None or obj_small is None:
            return {"handled": True, "operation": "pickup_container", "success": False, "error": "missing_obj_container"}
        action_camera = state.get("action_camera") or make_action_camera(obj_container, obj_small)
        set_viewer_camera(action_camera)
        extra_paths = []
        if step_image_dir is not None:
            extra_paths.append(capture_viewer_image(Path(step_image_dir) / f"action_step_{int(step or 0):03d}_pickup_before.png"))
        result = _pickup_container(obj_container, obj_small)
        set_viewer_camera(action_camera)
        if step_image_dir is not None:
            extra_paths.append(capture_viewer_image(Path(step_image_dir) / f"action_step_{int(step or 0):03d}_pickup_after.png"))
        return {
            **result,
            "position": action_camera["position"],
            "quaternion_xyzw": action_camera["quaternion_xyzw"],
            "extra_image_paths": extra_paths,
        }
    if action == "pour":
        obj_container = scene.object_registry("name", "obj_container")
        obj_small = scene.object_registry("name", "obj_small")
        if obj_container is None or obj_small is None:
            return {"handled": True, "operation": "pour_small_obj", "success": False, "error": "missing_object"}
        action_camera = state.get("action_camera") or make_action_camera(obj_container, obj_small)
        set_viewer_camera(action_camera)
        extra_paths = []
        if step_image_dir is not None:
            extra_paths.append(capture_viewer_image(Path(step_image_dir) / f"action_step_{int(step or 0):03d}_pour_before.png"))
        result = _pour_small_obj(obj_small, obj_container)
        set_viewer_camera(action_camera)
        if step_image_dir is not None:
            extra_paths.append(capture_viewer_image(Path(step_image_dir) / f"action_step_{int(step or 0):03d}_pour_after.png"))
        return {
            **result,
            "position": action_camera["position"],
            "quaternion_xyzw": action_camera["quaternion_xyzw"],
            "extra_image_paths": extra_paths,
        }
    return {"handled": False}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = correct_answer(payload)
    predicted = normalize_answer((final_answer or {}).get("answer"))
    return {
        "task_type": "transparent",
        "question": normalize_text(payload.get("_question")),
        "small_obj": small_obj_label(payload),
        "container": container_label(payload),
        "correct_answer": gt,
        "predicted_answer": predicted if predicted != "not sure" else None,
        "correct": predicted == gt if predicted in {"yes", "no"} and gt in {"yes", "no"} else None,
    }
