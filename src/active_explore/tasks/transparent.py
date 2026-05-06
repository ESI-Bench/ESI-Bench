from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import omnigibson as og
import torch as th


TASK_NAME = "transparent"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

VALID_ACTIONS = {
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
    "pickup",
    "pour",
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


def _pickup_container(obj_container) -> dict[str, Any]:
    lift_step = 0.02
    lift_total = 0.30
    steps = int(lift_total / lift_step)
    for _ in range(steps):
        pos, ori = obj_container.get_position_orientation()
        new_z = float(pos.cpu().numpy()[2] if hasattr(pos, "cpu") else pos[2]) + lift_step
        obj_container.set_position_orientation(
            position=th.tensor([pos[0], pos[1], new_z], dtype=th.float32),
            orientation=ori,
        )
        og.sim.step()
    return {"handled": True, "operation": "pickup_container", "success": True, "lift_m": lift_total}


def _pour_small_obj(obj_small, obj_container) -> dict[str, Any]:
    lift_step = 0.02
    c_bmin, c_bmax = [value.cpu().numpy() if hasattr(value, "cpu") else np.array(value) for value in obj_container.aabb]
    cont_top_z = float(c_bmax[2])
    cleared = False
    for _ in range(300):
        pos, ori = obj_small.get_position_orientation()
        s_bmin, _s_bmax = [value.cpu().numpy() if hasattr(value, "cpu") else np.array(value) for value in obj_small.aabb]
        if float(s_bmin[2]) >= cont_top_z:
            cleared = True
            break
        new_z = float(pos.cpu().numpy()[2] if hasattr(pos, "cpu") else pos[2]) + lift_step
        obj_small.set_position_orientation(
            position=th.tensor([pos[0], pos[1], new_z], dtype=th.float32),
            orientation=ori,
        )
        og.sim.step()
        current_z = float(obj_small.get_position_orientation()[0][2])
        if abs(current_z - new_z) > lift_step * 0.5:
            break

    pos, _ori = obj_small.get_position_orientation()
    s_bmin, s_bmax = [value.cpu().numpy() if hasattr(value, "cpu") else np.array(value) for value in obj_small.aabb]
    if float(s_bmin[2]) >= cont_top_z:
        place_x = float(c_bmax[0]) + 0.02 + float(s_bmax[0] - s_bmin[0]) / 2.0
        obj_small.set_position_orientation(
            position=th.tensor([place_x, pos[1], pos[2]], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        cleared = True
    return {"handled": True, "operation": "pour_small_obj", "success": cleared}


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
    if action == "pickup":
        obj_container = scene.object_registry("name", "obj_container")
        if obj_container is None:
            return {"handled": True, "operation": "pickup_container", "success": False, "error": "missing_obj_container"}
        return _pickup_container(obj_container)
    if action == "pour":
        obj_container = scene.object_registry("name", "obj_container")
        obj_small = scene.object_registry("name", "obj_small")
        if obj_container is None or obj_small is None:
            return {"handled": True, "operation": "pour_small_obj", "success": False, "error": "missing_object"}
        return _pour_small_obj(obj_small, obj_container)
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
