from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils import compute_exact_match, normalize_answer_for_eval, normalize_options, normalize_text


TASK_NAME = "cognitivemap"
FULL_SCENE = True
DEFAULT_MODEL = "gpt-5"

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


def normalize_answer_text(value: object) -> str:
    return normalize_text(value).lower()


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], "full_scene"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    raw = normalize_text(payload.get("question_id")) or source_path.stem
    return raw.replace("\\", "/").split("/")[-1]


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> dict[str, Any]:
    return {}


def extract_primary_camera_pose(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = payload.get("question_data", {})
    render = question_data.get("render", {}) or {}
    candidates = [
        (render.get("initial_view") or {}).get("camera_pose"),
        (question_data.get("initial_view") or {}).get("camera_pose"),
        (render.get("overview_view") or {}).get("camera_pose"),
        (question_data.get("overview_view") or {}).get("camera_pose"),
        (question_data.get("source_view") or {}).get("camera_pose"),
    ]
    for pose in candidates:
        if isinstance(pose, dict) and pose.get("position") and pose.get("quaternion_xyzw"):
            return pose

    room_views = render.get("room_views") or question_data.get("room_views") or {}
    if isinstance(room_views, dict):
        for room_name in sorted(room_views.keys()):
            room_record = room_views.get(room_name) or {}
            views = room_record.get("views")
            if isinstance(views, dict):
                for view_name in sorted(views.keys()):
                    pose = (views[view_name] or {}).get("camera_pose")
                    if isinstance(pose, dict) and pose.get("position") and pose.get("quaternion_xyzw"):
                        return pose
            if isinstance(views, list):
                for view in views:
                    pose = (view or {}).get("camera_pose")
                    if isinstance(pose, dict) and pose.get("position") and pose.get("quaternion_xyzw"):
                        return pose

    camera_poses = payload.get("camera_poses") or {}
    if isinstance(camera_poses, dict):
        for pose in camera_poses.values():
            if isinstance(pose, dict) and pose.get("position") and pose.get("quaternion_xyzw"):
                return pose

    raise ValueError("Missing camera pose in cognitivemap question JSON")


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose = extract_primary_camera_pose(payload)
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"camera_pose": pose},
    )


def get_task_context(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = payload.get("question_data", {})
    return {
        "question": normalize_text(question_data.get("question") or payload.get("_question")),
        "task_type": normalize_text(payload.get("task_type") or question_data.get("task_type")),
        "options": question_data.get("options"),
        "ground_truth": question_data.get("answer") if question_data.get("answer") is not None else payload.get("_ground_truth"),
        "question_data": question_data,
    }


def build_option_lines(options: object) -> list[str]:
    normalized = normalize_options(options)
    if isinstance(normalized, list) and normalized and all(isinstance(item, list) for item in normalized):
        return [
            f"Step {index} options: {', '.join(str(x) for x in step_options)}"
            for index, step_options in enumerate(normalized, start=1)
        ]
    if isinstance(normalized, list) and normalized:
        return [f"Options: {', '.join(str(x) for x in normalized)}"]
    return []


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
) -> str:
    ctx = get_task_context(payload)
    normalized_options = normalize_options(ctx["options"])
    answer_format = (
        '"<step_1_choice> | <step_2_choice> | ..."'
        if isinstance(normalized_options, list) and normalized_options and all(isinstance(item, list) for item in normalized_options)
        else '"<best answer or not sure>"'
    )
    lines = [
        "You are an embodied spatial reasoning agent exploring a 3D indoor scene for a cognitive map task.",
        f"Task type: {ctx['task_type']}",
        f"Question: {ctx['question']}",
        *build_option_lines(ctx["options"]),
        "",
        "You will receive recent views followed by the CURRENT view (always last).",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop>",',
        '  "reasoning": "<brief explanation>",',
        f'  "answer": {answer_format},',
        '  "confidence": <float 0.0-1.0>',
        "}",
        "",
        "Rules:",
        "  - Output ONLY valid JSON.",
        "  - Use exploration actions when the current view is insufficient.",
        "  - Past views are supporting context; decide primarily from the current image plus accumulated evidence.",
        "  - When the task has multiple steps, keep the answer order aligned with the steps and separate choices with ' | '.",
        f"  - Before step {min_steps}, confidence should usually remain <= 0.5 unless the answer is extremely obvious.",
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.",
        "  - Prefer moves and turns that expose room identity, adjacency, passages, and room-layout cues.",
    ]
    return "\n".join(lines)


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> str:
    ctx = get_task_context(payload)
    normalized_options = normalize_options(ctx["options"])
    answer_format = (
        '"<step_1_choice> | <step_2_choice> | ..."'
        if isinstance(normalized_options, list) and normalized_options and all(isinstance(item, list) for item in normalized_options)
        else '"<best answer>"'
    )
    lines = [
        "Exploration budget is exhausted.",
        f"Question: {ctx['question']}",
        *build_option_lines(ctx["options"]),
        "You must output one final answer.",
        "Do not answer 'not sure'.",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        f'  "answer": {answer_format},',
        '  "confidence": <float 0.0-1.0>,',
        '  "reasoning": "<brief explanation>"',
        "}",
    ]
    return "\n".join(lines)


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    action = normalize_answer_text(parsed.get("action")) or "move_forward"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        **parsed,
        "action": action,
        "answer": normalize_text(parsed.get("answer")) or "not sure",
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    if float(parsed.get("confidence", 0.0)) >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if parsed.get("action") == "stop":
        return True, "model_stop"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    if not history:
        return "not sure", -1
    latest = history[-1]
    return normalize_text(latest.get("answer")) or "not sure", int(latest["step"])


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return stop_reason == "max_steps" or not normalize_text(answer) or normalize_text(answer).lower() == "not sure"


def score(payload: dict[str, Any], final_answer: dict[str, Any], camera_info: dict[str, Any] | None = None) -> dict[str, Any]:
    ctx = get_task_context(payload)
    prediction = (final_answer or {}).get("answer")
    ground_truth = ctx["ground_truth"]
    options = ctx["options"]
    exact_match = compute_exact_match(prediction, ground_truth, options)
    return {
        "task_type": ctx["task_type"],
        "question": ctx["question"],
        "options": options,
        "ground_truth": ground_truth,
        "ground_truth_normalized": normalize_answer_for_eval(ground_truth, options),
        "prediction_normalized": normalize_answer_for_eval(prediction, options),
        "exact_match": exact_match,
        "correct": bool(exact_match),
    }
