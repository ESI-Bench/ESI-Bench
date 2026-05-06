from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from utils import normalize_text, resolve_path


TASK_NAME = "unobserved_changes"
FULL_SCENE = True
DEFAULT_MODEL = "gemini-2.5-flash"

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


def normalize_choice(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^[a-z]\s*[\).:\-]\s*", "", text)
    text = text.strip("\"' ")
    return re.sub(r"\s+", " ", text)


def canonicalize_prediction(predicted_answer: object, options: list[str]) -> str:
    raw = normalize_choice(predicted_answer)
    if not raw:
        return ""
    option_map = {normalize_choice(option): option for option in options}
    if raw in option_map:
        return option_map[raw]
    for normalized_option, option in option_map.items():
        if raw == f"answer: {normalized_option}" or raw.endswith(f": {normalized_option}"):
            return option
    for normalized_option, option in option_map.items():
        if normalized_option and normalized_option in raw:
            return option
    return normalize_text(predicted_answer)


def compute_accuracy(predicted_answer: object, ground_truth: object) -> float:
    return 1.0 if normalize_choice(predicted_answer) == normalize_choice(ground_truth) else 0.0


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")), normalize_text(payload.get("room")) or "full_scene"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    raw = normalize_text(payload.get("question_id")) or source_path.stem
    return raw.replace("\\", "/").split("/")[-1]


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _question_data(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("question_data") or {}


def _keyed_list_to_map(items: object) -> dict[str, Any]:
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item.get("_key"))] = item
    return output


def _phase_description_map(payload: dict[str, Any]) -> dict[str, str]:
    phase_description = _question_data(payload).get("phase_description") or {}
    direct = {
        "phase_1": normalize_text(phase_description.get("phase_1")),
        "phase_2": normalize_text(phase_description.get("phase_2")),
    }
    phase_items = _keyed_list_to_map(phase_description.get("phase"))
    for key in ("phase_1", "phase_2"):
        if not direct[key] and isinstance(phase_items.get(key), dict):
            direct[key] = normalize_text(phase_items[key].get("value"))
    return direct


def _phase_content_map(box: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    direct = {
        "phase1_content": box.get("phase1_content"),
        "phase2_content": box.get("phase2_content"),
    }
    phase_items = _keyed_list_to_map(box.get("phase_content"))
    for key in ("phase1_content", "phase2_content"):
        if direct[key] is None and isinstance(phase_items.get(key), dict):
            item = dict(phase_items[key])
            if item.get("category") is None or item.get("model") is None:
                direct[key] = None
            else:
                direct[key] = item
    return direct


def _content_payload_to_runtime(content: dict[str, Any] | None) -> dict[str, Any] | None:
    if content is None:
        return None
    return {
        "category": content.get("category"),
        "display_name": content.get("display_name"),
        "representative_model": content.get("model"),
        "bbox_size_m": content.get("bbox_size_m"),
        "sampling_source": content.get("sampling_source"),
    }


def build_states_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    for box in _question_data(payload).get("boxes") or []:
        container = box.get("container") or {}
        phase_content = _phase_content_map(box)
        states.append(
            {
                "box_index": int(box.get("box_index", len(states))),
                "position_label": normalize_text(box.get("position_label")) or f"box {len(states)}",
                "change_type": normalize_text(box.get("change_type")) or "no_change",
                "phase1_content": _content_payload_to_runtime(phase_content.get("phase1_content")),
                "phase2_content": _content_payload_to_runtime(phase_content.get("phase2_content")),
                "container_name": normalize_text(container.get("name")),
                "container_category": normalize_text(container.get("category")),
                "container_model": normalize_text(container.get("model")),
                "container_placement": container.get("placement"),
                "container_bbox": container.get("bbox"),
            }
        )
    if not states:
        raise ValueError("Question JSON does not contain any boxes.")
    return states


def _gt_view_phase_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    render = _question_data(payload).get("render") or {}
    gt_view = render.get("gt_view") or {}
    if isinstance(gt_view.get("image1"), dict) or isinstance(gt_view.get("image2"), dict):
        return {
            key: value
            for key in ("image1", "image2")
            if isinstance((value := gt_view.get(key)), dict)
        }
    return _keyed_list_to_map(gt_view.get("image"))


def _first_gt_image(payload: dict[str, Any], phase_key: str) -> dict[str, Any] | None:
    entry = _gt_view_phase_entries(payload).get(phase_key)
    images = (entry or {}).get("images") or []
    for image in images:
        if isinstance(image, dict) and image.get("image_path"):
            return image
    return None


def _reference_image_paths(payload: dict[str, Any], source_json: Path, config=None) -> tuple[list[Path], dict[str, Any] | None, str | None]:
    data_root = getattr(config, "json_root", None)
    image1 = _first_gt_image(payload, "image1")
    image2 = _first_gt_image(payload, "image2")
    if image1 is None or image2 is None:
        return [], None, "missing_phase_reference_images"
    path1 = resolve_path(image1.get("image_path"), source_json, data_root=data_root)
    path2 = resolve_path(image2.get("image_path"), source_json, data_root=data_root)
    if path1 is None or path2 is None:
        return [], None, "unresolved_phase_reference_images"
    pose = image2.get("camera_pose")
    if not pose or not pose.get("position") or not pose.get("quaternion_xyzw"):
        return [], None, "missing_phase2_camera_pose"
    return [path1, path2], pose, None


def preprocess(payload: dict[str, Any], source_json: Path, config=None) -> dict[str, Any]:
    reference_paths, pose, skip_reason = _reference_image_paths(payload, source_json, config=config)
    if skip_reason:
        return {"skip_reason": skip_reason}
    return {
        "source_json": str(source_json),
        "reference_image_paths": [str(path) for path in reference_paths],
        "initial_camera_pose": pose,
    }


def reference_image_paths(payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> list[Path]:
    return [Path(path) for path in (task_state or {}).get("reference_image_paths", [])]


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Pipeline calls this before preprocess, so read directly from the keyed-list
    # image2 entry here as well.
    image2 = _first_gt_image(payload, "image2")
    pose = (image2 or {}).get("camera_pose")
    if not pose or not pose.get("position") or not pose.get("quaternion_xyzw"):
        raise ValueError("Missing phase-2 camera pose in unobserved_changes JSON")
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"camera_pose": pose},
    )


def postprocess_env(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    states = build_states_from_payload(payload)
    if task_state is not None:
        task_state["unobserved_change_states"] = states
        task_state["phase_descriptions"] = _phase_description_map(payload)
    return {
        "box_count": len(states),
        "reference_driven": True,
    }


def get_context(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = _question_data(payload)
    phase_description = _phase_description_map(payload)
    boxes = []
    for box in question_data.get("boxes") or []:
        container = box.get("container") or {}
        boxes.append(
            {
                "label": normalize_text(box.get("position_label")) or f"box {int(box.get('box_index', 0))}",
                "container_category": normalize_text(container.get("category")),
                "change_type": normalize_text(box.get("change_type")),
            }
        )
    return {
        "task_type": normalize_text(payload.get("task_type") or question_data.get("task_type")),
        "question": normalize_text(question_data.get("question")),
        "options": [normalize_text(opt) for opt in question_data.get("options", []) if normalize_text(opt)],
        "phase_1": phase_description["phase_1"],
        "phase_2": phase_description["phase_2"],
        "boxes": boxes,
        "ground_truth": normalize_text(question_data.get("answer")),
    }


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_context(payload)
    lines = [
        "You are an embodied visual reasoning agent for unobserved scene changes.",
        f"Task type: {ctx['task_type']}",
        f"Question: {ctx['question']}",
    ]
    if ctx["phase_1"]:
        lines.append(f"Phase 1 description: {ctx['phase_1']}")
    if ctx["phase_2"]:
        lines.append(f"Phase 2 description: {ctx['phase_2']}")
    if ctx["options"]:
        lines.append("Options: " + ", ".join(ctx["options"]))
    if ctx["boxes"]:
        box_line = ", ".join(f"{item['label']} ({item['container_category'] or 'box'})" for item in ctx["boxes"])
        lines.append(f"Box identities in the scene: {box_line}")
    lines.extend([
        "",
        "Important scene setup:",
        "  - The first reference image is the original Phase 1 image.",
        "  - The second reference image is the original Phase 2 image.",
        "  - The reference images are authoritative for the before/after change.",
        "  - The simulator view is extra spatial context and may not contain generated hidden contents.",
        "",
        "You will receive those two reference images plus recent exploration views, with the CURRENT view always last.",
        "Use the reference images to understand the before/after change, and use exploration views only when they add useful spatial context.",
        "",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop>",',
        '  "reasoning": "<brief explanation>",',
        '  "answer": "<one option exactly as written or not sure>",',
        '  "confidence": <float 0.0-1.0>',
        "}",
        "",
        "Rules:",
        "  - Output only valid JSON.",
        "  - The answer should match one listed option exactly when confident.",
        "  - If the current view is insufficient, choose a movement action instead of stopping.",
        "  - Do not assume simulator-only details override the two reference images.",
        "  - Use turn actions when rotation is more helpful than translation.",
        f"  - Before step {min_steps}, confidence should usually remain <= 0.5 unless the answer is extremely obvious.",
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.",
        "  - If uncertain, answer 'not sure' and continue exploring instead of guessing too early.",
    ])
    return "\n".join(lines)


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_context(payload)
    lines = ["Exploration budget is exhausted.", f"Question: {ctx['question']}"]
    if ctx["options"]:
        lines.append("Options: " + ", ".join(ctx["options"]))
    lines.extend([
        "Choose the single best option using the Phase 1 image, the Phase 2 image, and the exploration evidence.",
        "Do not answer 'not sure'.",
        "Output EXACTLY one JSON object and nothing else:",
        '{"answer": "<one listed option exactly as written>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
    ])
    return "\n".join(lines)


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    ctx_options = parsed.get("_options") if isinstance(parsed.get("_options"), list) else []
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    answer = canonicalize_prediction(parsed.get("answer"), ctx_options) if ctx_options else normalize_text(parsed.get("answer"))
    if normalize_choice(answer) in {"", "not sure", "unsure", "unknown"}:
        answer = "not sure"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        **parsed,
        "action": action,
        "answer": answer,
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
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and answer.lower() != "not sure":
            return answer, int(item["step"])
    if history:
        return normalize_text(history[-1].get("answer")) or "not sure", int(history[-1]["step"])
    return "not sure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_choice(answer) in {"", "not sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = get_context(payload)
    predicted_answer = canonicalize_prediction((final_answer or {}).get("answer"), ctx["options"])
    accuracy = compute_accuracy(predicted_answer, ctx["ground_truth"]) if predicted_answer else 0.0
    return {
        "task_type": ctx["task_type"],
        "question": ctx["question"],
        "options": ctx["options"],
        "ground_truth": ctx["ground_truth"],
        "predicted_answer": predicted_answer,
        "accuracy": accuracy,
        "correct": bool(accuracy),
        "reference_images": (task_state or {}).get("reference_image_paths", []),
    }
