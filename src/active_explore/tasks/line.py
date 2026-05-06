from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TASK_NAME = "line"
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
    "stop",
}


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower().replace("_", " ")
    if text in {"yes", "y", "true", "in a line", "collinear"}:
        return "yes"
    if text in {"no", "n", "false", "not in a line", "not collinear", "non-collinear", "non collinear"}:
        return "no"
    if "not sure" in text or "unsure" in text or "unknown" in text or not text:
        return "not sure"
    if "not collinear" in text or "not in a line" in text:
        return "no"
    if "collinear" in text or "in a line" in text or "straight line" in text:
        return "yes"
    return "not sure"


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


def pose_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    poses = payload.get("camera_poses")
    if isinstance(poses, dict):
        return [value for value in poses.values() if isinstance(value, dict)]
    if isinstance(poses, list):
        return [pose for pose in poses if isinstance(pose, dict)]
    return []


def task_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects = objects_map(payload)
    output = []
    for key in ("obj1", "obj2", "obj3"):
        obj = objects.get(key)
        if not obj:
            raise ValueError(f"Missing {key} in line JSON")
        output.append(obj)
    return output


def object_labels(payload: dict[str, Any]) -> list[str]:
    return [display_category(obj.get("category")) for obj in task_objects(payload)]


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


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for key, obj in zip(("obj1", "obj2", "obj3"), task_objects(payload), strict=True):
        spec = {
            "type": "DatasetObject",
            "name": normalize_text(obj.get("name")) or key,
            "category": obj["category"],
            "model": obj["model"],
            "position": obj["position"],
            "orientation": obj["quaternion_xyzw"],
        }
        if obj.get("scale") is not None:
            spec["scale"] = obj["scale"]
        output.append(spec)
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poses = pose_records(payload)
    if not poses:
        raise ValueError("Missing camera_poses in line JSON")
    pose = poses[0]
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {
            "view_index": 0,
            "view": pose,
            "selection": "first_camera_pose",
        },
    )


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    obj1, obj2, obj3 = object_labels(payload)
    return "\n".join(
        [
            "You are an embodied spatial reasoning expert controlling a camera in a 3D indoor scene.",
            f"TASK: Determine whether the {obj1}, {obj2}, and {obj3} are arranged in a straight line on the floor.",
            "",
            "STRICT RULES:",
            "1. Visual parallax can make objects appear collinear from one angle but not another.",
            "2. Actively verify from multiple distinct viewpoints before committing.",
            "3. Use move_up with turn_down for a more top-down view when useful.",
            "4. Answer yes only if the three object centers appear collinear on the floor.",
            "5. Answer no if one object is clearly off the line formed by the other two.",
            "6. If you need more evidence, answer 'not sure' and keep exploring.",
            "",
            "Output EXACTLY one valid JSON object and nothing else:",
            "{",
            '  "action": "<action_name>",',
            '  "answer": "<yes, no, or not sure>",',
            '  "reasoning": "<brief explanation>",',
            '  "confidence": <float 0.0-1.0>',
            "}",
            "",
            "Available camera actions:",
            "  move_forward | move_backward | move_left | move_right | move_up | move_down",
            "  turn_left | turn_right | turn_up | turn_down | stop",
            "",
            f"Confidence threshold to conclude: {threshold:.2f}.",
            "Use stop only when you are ready to finish with a conclusive yes/no answer.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    obj1, obj2, obj3 = object_labels(payload)
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"You must decide whether the {obj1}, {obj2}, and {obj3} are in a straight line on the floor.",
            "Do not answer 'not sure'.",
            'Output EXACTLY: {"answer": "<yes or no>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
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
        "conclusive": answer in {"yes", "no"},
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
    if float(parsed.get("confidence", 0.0)) >= threshold and bool(parsed.get("conclusive")):
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


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predicted = normalize_answer((final_answer or {}).get("answer"))
    target = normalize_answer(payload.get("_ground_truth"))
    return {
        "task_type": "line",
        "question": normalize_text(payload.get("_question")),
        "objects": object_labels(payload),
        "predicted_answer": predicted if predicted != "not sure" else None,
        "ground_truth": target,
        "correct": predicted == target if predicted in {"yes", "no"} and target in {"yes", "no"} else None,
    }
