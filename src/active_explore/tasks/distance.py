from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np


TASK_NAME = "distance"
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


def normalize_category(value: Any) -> str:
    return normalize_text(value).lower().replace(" ", "_")


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


def visibility_map(payload: dict[str, Any], key: str) -> dict[int, bool]:
    output: dict[int, bool] = {}
    for item in payload.get(key) or []:
        if not isinstance(item, dict):
            continue
        indices = item.get("_indices") or []
        if not indices:
            continue
        output[int(indices[0])] = bool(item.get("value"))
    return output


def pose_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    poses = payload.get("camera_poses") or []
    return [pose for pose in poses if isinstance(pose, dict)]


def near_category(payload: dict[str, Any]) -> str:
    return normalize_category(payload.get("near_category") or objects_map(payload).get("obj_near", {}).get("category"))


def far_category(payload: dict[str, Any]) -> str:
    return normalize_category(payload.get("far_category") or objects_map(payload).get("obj_far", {}).get("category"))


def ref_category(payload: dict[str, Any]) -> str:
    return normalize_category(payload.get("ref_category") or objects_map(payload).get("obj_ref", {}).get("category"))


def choice_order(payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> list[str]:
    state_order = (task_state or {}).get("choice_order")
    if isinstance(state_order, list) and len(state_order) >= 2:
        return [normalize_category(item) for item in state_order[:2]]
    return [near_category(payload), far_category(payload)]


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    candidates = [near_category(payload), far_category(payload)]
    seed_source = f"{payload.get('seed', '')}:{payload.get('question_id', '')}:{source_json.stem if source_json else ''}"
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return {
        "choice_order": candidates,
        "shuffle_seed": seed,
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


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for key in ("obj_ref", "obj_near", "obj_far"):
        obj = objects_map(payload).get(key)
        if not obj or obj.get("model") == "scene_native" or not obj.get("model"):
            continue
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
        raise ValueError("Missing camera_poses in distance JSON")
    exist_near = visibility_map(payload, "exist_near")
    exist_far = visibility_map(payload, "exist_far")
    chosen_index = None
    for index, _pose in enumerate(poses):
        if exist_near.get(index, False) and exist_far.get(index, False):
            chosen_index = index
            break
    if chosen_index is None:
        chosen_index = 0
    pose = poses[chosen_index]
    near_visible = exist_near.get(chosen_index, False)
    far_visible = exist_far.get(chosen_index, False)
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {
            "view_index": chosen_index,
            "view": pose,
            "selection": "first_near_far_visible" if near_visible and far_visible else "first_pose_fallback",
            "near_visible": near_visible,
            "far_visible": far_visible,
        },
    )


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    first, second = choice_order(payload, task_state)
    ref_label = display_category(ref_category(payload))
    first_label = display_category(first)
    second_label = display_category(second)
    return "\n".join(
        [
            "You are an embodied spatial reasoning expert controlling a camera in a 3D indoor scene.",
            f"TASK: Determine which object is closer to the {ref_label}: the {first_label} or the {second_label}.",
            "",
            "STRICT RULES:",
            "1. Do not jump to conclusions. Distance judgments from one view can be distorted by perspective.",
            "2. Actively explore from multiple viewpoints before committing.",
            "3. Use move_up with turn_down when a more top-down view is useful.",
            "4. Move backward or sideways when the two candidate objects are at unfair depths in the image.",
            "5. Only give a conclusive answer when confidence is high.",
            "6. If you need more evidence, answer 'not sure' and keep exploring.",
            "",
            "Output EXACTLY one valid JSON object and nothing else:",
            "{",
            '  "action": "<action_name>",',
            '  "answer": "<one of the listed answers>",',
            '  "reasoning": "<brief explanation>",',
            '  "confidence": <float 0.0-1.0>',
            "}",
            "",
            "Answer choices:",
            f"  - {first_label}",
            f"  - {second_label}",
            "  - not sure",
            "",
            "Available camera actions:",
            "  move_forward | move_backward | move_left | move_right | move_up | move_down",
            "  turn_left | turn_right | turn_up | turn_down | stop",
            "",
            f"Confidence threshold to conclude: {threshold:.2f}.",
            "Use stop only when you are ready to finish with a conclusive answer.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    first, second = choice_order(payload, task_state)
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            "You must commit to the object that is closer to the reference.",
            "Do not answer 'not sure'.",
            "Answer choices:",
            f"  - {display_category(first)}",
            f"  - {display_category(second)}",
            'Output EXACTLY: {"answer": "<selected object>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
        ]
    )


def _answer_to_category(answer: Any, payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> str:
    text = normalize_category(answer)
    if text in {"", "not_sure", "unsure", "unknown"}:
        return "not_sure"
    for category in choice_order(payload, task_state):
        norm = normalize_category(category)
        if text == norm or text in norm or norm in text:
            return norm
    if "near" in text:
        return near_category(payload)
    if "far" in text:
        return far_category(payload)
    return "not_sure"


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
    answer = _answer_to_category(parsed.get("answer"), payload, task_state) if isinstance(payload, dict) else normalize_category(parsed.get("answer"))
    return {
        **parsed,
        "action": action,
        "answer": answer or "not_sure",
        "conclusive": answer not in {"", "not_sure", "unsure", "unknown"},
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
        answer = normalize_category(item.get("answer"))
        if answer and answer not in {"not_sure", "unsure", "unknown"}:
            return answer, int(item["step"])
    return "not_sure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_category(answer) in {"", "not_sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predicted = normalize_category((final_answer or {}).get("answer"))
    target = near_category(payload)
    far = far_category(payload)
    return {
        "task_type": "distance",
        "question": normalize_text(payload.get("_question")),
        "ref_category": ref_category(payload),
        "near_category": target,
        "far_category": far,
        "predicted_category": predicted if predicted != "not_sure" else None,
        "ground_truth": target,
        "correct": predicted == target if predicted and predicted != "not_sure" else None,
        "question_order": choice_order(payload, task_state),
    }
