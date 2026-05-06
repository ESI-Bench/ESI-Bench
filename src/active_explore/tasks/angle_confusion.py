from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TASK_NAME = "angle_confusion"
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
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item.get("_key"))] = item
    return output


def objects_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("objects"))


def view_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    views = payload.get("views") or []
    return [view for view in views if isinstance(view, dict)]


def visibility_values(payload: dict[str, Any]) -> dict[int, bool]:
    output = {}
    for item in payload.get("exist_target_obj") or []:
        if not isinstance(item, dict):
            continue
        indices = item.get("_indices") or []
        if not indices:
            continue
        output[int(indices[0])] = bool(item.get("value"))
    return output


def target_object(payload: dict[str, Any]) -> dict[str, Any]:
    target = objects_map(payload).get("target")
    if not target:
        raise ValueError("Missing target object in angle_confusion JSON")
    return target


def choices(payload: dict[str, Any], include_not_sure: bool = True) -> list[dict[str, str]]:
    target = normalize_category(payload.get("_ground_truth") or target_object(payload).get("category"))
    distractors = [normalize_category(item) for item in payload.get("target_confusable_with") or [] if normalize_text(item)]
    categories = [target]
    if distractors:
        categories.append(distractors[0])
    elif normalize_text(payload.get("_question")):
        parts = normalize_text(payload["_question"]).split(":", 1)[-1].replace("?", "").split(" or ")
        for part in parts:
            cat = normalize_category(part)
            if cat and cat not in categories:
                categories.append(cat)
            if len(categories) >= 2:
                break
    categories = categories[:2]
    if len(categories) < 2:
        raise ValueError("Missing angle_confusion answer choices")
    base = [
        {"letter": chr(ord("A") + index), "category": category, "display": display_category(category)}
        for index, category in enumerate(categories)
    ]
    if include_not_sure:
        base.append({"letter": chr(ord("A") + len(base)), "category": "not_sure", "display": "not sure"})
    return base


ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "choice_letter": {"type": "string"},
        "choice_text": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "choice_letter", "choice_text", "reasoning", "confidence"],
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
    return source_path.stem


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    target = target_object(payload)
    return [
        {
            "type": "DatasetObject",
            "name": "target_obj",
            "category": target.get("requested_category") or target["category"],
            "model": target["model"],
            "position": target["position"],
            "orientation": target["quaternion_xyzw"],
            "scale": target.get("scale") or [1, 1, 1],
        }
    ]


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    visibility = visibility_values(payload)
    records = view_records(payload)
    chosen_index = None
    for index, view in enumerate(records):
        if view.get("type") == "topdown":
            continue
        if visibility.get(index, False):
            chosen_index = index
            break
    if chosen_index is None:
        for index, view in enumerate(records):
            if view.get("type") != "topdown":
                chosen_index = index
                break
    if chosen_index is None:
        raise ValueError("Missing non-topdown initial view in angle_confusion JSON")
    view = records[chosen_index]
    return (
        np.array(view["position"], dtype=float),
        np.array(view["quaternion_xyzw"], dtype=float),
        {"view_index": chosen_index, "view": view},
    )


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    choice_lines = "\n".join(f"{item['letter']}. {item['display']}" for item in choices(payload, include_not_sure=True))
    return "\n".join(
        [
            "You are an embodied visual recognition agent controlling a camera in a 3D indoor scene.",
            "TASK: Identify the category of the target object near the center of the initial view.",
            "This is a viewpoint-disambiguation task: actively explore different viewpoints of the same object.",
            "Keep tracking that same central object region across steps.",
            "",
            "You will receive up to 5 recent past views and then the CURRENT view (always last).",
            "At every step, choose ONE camera action and provide your current best answer.",
            "Output EXACTLY one valid JSON object and nothing else:",
            "{",
            '  "action": "<action_name>",',
            '  "choice_letter": "<A, B, or not-sure option>",',
            '  "choice_text": "<selected option text>",',
            '  "reasoning": "<one short sentence>",',
            '  "confidence": <float 0.0-1.0>',
            "}",
            "",
            "Answer choices:",
            choice_lines,
            "",
            "Available camera actions:",
            "  move_forward | move_backward | move_left | move_right | move_up | move_down",
            "  turn_left | turn_right | turn_up | turn_down | stop",
            "",
            "Rules:",
            "  - Output ONLY valid JSON, nothing else.",
            "  - Always include all five fields.",
            "  - Pick exactly one listed choice.",
            f"  - If confidence reaches or exceeds {threshold:.2f}, the episode may stop after this step.",
            "  - Prefer small turns and lateral moves to inspect multiple sides of the object.",
            "  - Keep the target object in frame; if it leaves, recover it immediately.",
            "  - Base your reasoning on the observed views, not on prior assumptions.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    choice_lines = "\n".join(f"{item['letter']}. {item['display']}" for item in choices(payload, include_not_sure=False))
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            "You must commit to one of the target category choices.",
            "Do not answer 'not sure'.",
            "Answer choices:",
            choice_lines,
            "Output EXACTLY one JSON object:",
            '{"answer": "<selected category or choice letter>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
        ]
    )


def _choice_to_category(parsed: dict[str, Any], payload: dict[str, Any]) -> str | None:
    by_letter = {item["letter"].upper(): item for item in choices(payload, include_not_sure=True)}
    letter = normalize_text(parsed.get("choice_letter") or parsed.get("answer")).upper()
    text = normalize_category(parsed.get("choice_text") or parsed.get("answer"))
    if letter in by_letter:
        return by_letter[letter]["category"]
    if text:
        for item in by_letter.values():
            if text in {normalize_category(item["display"]), item["category"]}:
                return item["category"]
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
    if action not in VALID_ACTIONS:
        action = "move_forward"
    predicted = _choice_to_category(parsed, payload) if isinstance(payload, dict) else normalize_category(parsed.get("answer"))
    answer = predicted or "not_sure"
    return {
        **parsed,
        "action": action,
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
        "predicted_category": answer,
    }


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    if float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_category(item.get("answer"))
        if answer and answer != "not_sure":
            return answer, int(item["step"])
    return "not_sure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_category(answer) in {"", "not_sure", "unknown", "unsure"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = normalize_category(payload.get("_ground_truth") or target_object(payload).get("category"))
    predicted = normalize_category((final_answer or {}).get("answer"))
    return {
        "task_type": "angle_confusion",
        "question": normalize_text(payload.get("_question")),
        "target_category": target,
        "distractor_categories": [normalize_category(item) for item in payload.get("target_confusable_with") or []],
        "predicted_category": predicted if predicted != "not_sure" else None,
        "correct": predicted == target if predicted and predicted != "not_sure" else None,
    }
