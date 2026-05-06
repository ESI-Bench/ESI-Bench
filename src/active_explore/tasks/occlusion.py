from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


TASK_NAME = "occlusion"
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


def view_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    views = payload.get("views") or []
    if isinstance(views, dict):
        return [value for value in views.values() if isinstance(value, dict)]
    return [view for view in views if isinstance(view, dict)]


def target_object(payload: dict[str, Any]) -> dict[str, Any]:
    target = objects_map(payload).get("target")
    if not target:
        raise ValueError("Missing target object in occlusion JSON")
    return target


def occluder_object(payload: dict[str, Any]) -> dict[str, Any]:
    occluder = objects_map(payload).get("occluder")
    if not occluder:
        raise ValueError("Missing occluder object in occlusion JSON")
    return occluder


def target_category(payload: dict[str, Any]) -> str:
    return normalize_category(payload.get("_ground_truth") or target_object(payload).get("requested_category") or target_object(payload).get("category"))


def _choices_from_question(question: str) -> list[str]:
    text = normalize_text(question)
    if not text:
        return []
    tail = text.split(":", 1)[-1]
    tail = re.sub(r"\?$", "", tail).strip()
    parts = re.split(r"\s+or\s+", tail, flags=re.IGNORECASE)
    output = []
    for part in parts:
        cleaned = normalize_category(re.sub(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$", "", part))
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output[:2]


def choice_categories(payload: dict[str, Any]) -> list[str]:
    target = target_category(payload)
    categories = _choices_from_question(normalize_text(payload.get("_question")))
    if target and target not in categories:
        categories.insert(0, target)

    for distractor in payload.get("target_confusable_with") or []:
        category = normalize_category(distractor)
        if category and category != target and category not in categories:
            categories.append(category)
            break

    categories = categories[:2]
    if target and target not in categories:
        categories = [target] + categories[:1]
    if len(categories) < 2:
        raise ValueError("Missing occlusion answer choices")
    return categories


def choices(payload: dict[str, Any], include_not_sure: bool = True) -> list[dict[str, str]]:
    base = [
        {"letter": chr(ord("A") + index), "category": category, "display": display_category(category)}
        for index, category in enumerate(choice_categories(payload))
    ]
    if include_not_sure:
        base.append({"letter": chr(ord("A") + len(base)), "category": "not_sure", "display": "not sure"})
    return base


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for key, fallback_name in (("target", "target_obj"), ("occluder", "occluder_obj")):
        obj = objects_map(payload).get(key)
        if not obj:
            raise ValueError(f"Missing {key} object in occlusion JSON")
        spec = {
            "type": "DatasetObject",
            "name": normalize_text(obj.get("name")) or fallback_name,
            "category": obj.get("requested_category") or obj["category"],
            "model": obj["model"],
            "position": obj["position"],
            "orientation": obj["quaternion_xyzw"],
        }
        scale = obj.get("applied_scale") or obj.get("scale")
        if scale is not None:
            spec["scale"] = scale
        output.append(spec)
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = view_records(payload)
    if not records:
        raise ValueError("Missing views in occlusion JSON")
    view = records[0]
    return (
        np.array(view["position"], dtype=float),
        np.array(view["quaternion_xyzw"], dtype=float),
        {
            "view_index": 0,
            "view": view,
            "selection": "first_view",
        },
    )


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    choice_lines = "\n".join(f"{item['letter']}. {item['display']}" for item in choices(payload, include_not_sure=True))
    occluder_display = display_category(occluder_object(payload).get("requested_category") or occluder_object(payload).get("category"))
    return "\n".join(
        [
            "You are an embodied visual recognition agent controlling a camera in a 3D indoor scene.",
            "TASK: Identify the category of the partially occluded target object.",
            f"A {occluder_display} partially blocks the target near the center of the initial view.",
            "Track that same central target region across steps and identify the hidden object, not the foreground blocker.",
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
            "  - Prefer small lateral moves and turns to reveal more of the occluded target.",
            "  - Keep the target region in frame; if it leaves, recover it immediately.",
            "  - Avoid repeating the same action back-to-back without a meaningful viewpoint change.",
            "  - Base your reasoning on the observed views, not on prior assumptions.",
            "  - You may say 'not sure' if genuinely uncertain. Continue exploring to gather evidence.",
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
            "You must identify the partially occluded target object.",
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


def should_stop(
    parsed: dict[str, Any],
    history: list[dict[str, Any]],
    step: int,
    max_steps: int,
    min_steps: int,
    threshold: float,
) -> tuple[bool, str]:
    confidence = float(parsed.get("confidence", 0.0))
    answer = normalize_category(parsed.get("answer"))
    if confidence >= threshold and answer != "not_sure":
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
    target = target_category(payload)
    predicted = normalize_category((final_answer or {}).get("answer"))
    distractors = [category for category in choice_categories(payload) if category != target]
    return {
        "task_type": "occlusion",
        "question": normalize_text(payload.get("_question")),
        "target_category": target,
        "occluder_category": normalize_category(occluder_object(payload).get("requested_category") or occluder_object(payload).get("category")),
        "distractor_category": distractors[0] if distractors else None,
        "choices": choices(payload, include_not_sure=False),
        "predicted_category": predicted if predicted != "not_sure" else None,
        "correct": predicted == target if predicted and predicted != "not_sure" else None,
    }
