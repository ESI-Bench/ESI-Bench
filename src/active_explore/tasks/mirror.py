from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.objects.dataset_object import DatasetObject

from utils import compute_exact_match, normalize_answer_for_eval, normalize_options, normalize_text


TASK_NAME = "mirror"
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


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")), normalize_text(payload.get("room")) or "full_scene"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    raw = normalize_text(payload.get("question_id")) or source_path.stem
    return raw.replace("\\", "/").split("/")[-1]


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _question_data(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("question_data") or {}


def _get_item_orientation(item: dict[str, Any]) -> list[float]:
    orientation = item.get("quaternion_xyzw")
    if orientation is None:
        orientation = item.get("orientation")
    if orientation is None:
        orientation = [0.0, 0.0, 0.0, 1.0]
    return [float(value) for value in orientation]


def _step_sim(steps: int) -> None:
    for _ in range(max(int(steps), 0)):
        og.sim.step()


def _add_or_update_mirror(scene, payload: dict[str, Any]) -> str:
    mirror_setup = payload.get("mirror_setup") or {}
    scene_setup = _question_data(payload).get("scene_setup") or {}
    mirror_pose = scene_setup.get("mirror") or {}
    mirror_name = normalize_text(mirror_setup.get("name")) or "render_mirror_main"
    mirror_obj = scene.object_registry("name", mirror_name)
    if mirror_obj is None:
        mirror_obj = DatasetObject(
            name=mirror_name,
            category=normalize_text(mirror_setup.get("category")) or "mirror",
            model=normalize_text(mirror_setup.get("model")) or "tytkbq",
            visual_only=True,
        )
        scene.add_object(mirror_obj)
    try:
        mirror_obj.visual_only = True
    except Exception:
        pass
    if mirror_pose.get("position") and mirror_pose.get("quaternion_xyzw"):
        mirror_obj.set_position_orientation(
            position=th.tensor(mirror_pose["position"], dtype=th.float32),
            orientation=th.tensor(mirror_pose["quaternion_xyzw"], dtype=th.float32),
        )
    return mirror_name


def _add_question_objects(scene, payload: dict[str, Any]) -> list[str]:
    added_names = []
    for item in _question_data(payload).get("placement") or []:
        name = normalize_text(item.get("name"))
        category = normalize_text(item.get("category"))
        model = normalize_text(item.get("model"))
        position = item.get("real_position")
        if not name or not category or not model or position is None:
            continue
        existing = scene.object_registry("name", name)
        if existing is not None:
            try:
                scene.remove_object(existing)
            except Exception:
                pass
        obj = DatasetObject(name=name, category=category, model=model, visual_only=True)
        scene.add_object(obj)
        try:
            obj.visual_only = True
        except Exception:
            pass
        obj.set_position_orientation(
            position=th.tensor(position, dtype=th.float32),
            orientation=th.tensor(_get_item_orientation(item), dtype=th.float32),
        )
        added_names.append(name)
    _step_sim(20)
    return added_names


def postprocess_env(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene = env.scene
    mirror_name = _add_or_update_mirror(scene, payload)
    dynamic_names = _add_question_objects(scene, payload)
    if task_state is not None:
        task_state["mirror_name"] = mirror_name
        task_state["dynamic_object_names"] = dynamic_names
    return {"mirror_name": mirror_name, "dynamic_object_names": dynamic_names}


def _camera_pose(payload: dict[str, Any]) -> dict[str, Any]:
    qd = _question_data(payload)
    pose = (qd.get("render") or {}).get("camera_pose")
    if not pose:
        pose = (qd.get("scene_setup") or {}).get("camera")
    if not pose or not pose.get("position") or not pose.get("quaternion_xyzw"):
        raise ValueError("Missing mirror camera pose")
    return pose


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose = _camera_pose(payload)
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"camera_pose": pose},
    )


def get_allowed_answers(payload: dict[str, Any]) -> list[str]:
    options = _question_data(payload).get("options") or []
    if options:
        return [normalize_text(option) for option in options if normalize_text(option)]
    task_type = normalize_text(payload.get("task_type"))
    if task_type == "mirror_object_reality":
        return ["Yes", "No"]
    if task_type == "mirror_distance":
        return ["A", "B"]
    return []


def get_task_context(payload: dict[str, Any]) -> dict[str, Any]:
    qd = _question_data(payload)
    return {
        "task_type": normalize_text(payload.get("task_type")),
        "question": normalize_text(qd.get("question")),
        "options": qd.get("options"),
        "allowed_answers": get_allowed_answers(payload),
        "ground_truth": qd.get("answer"),
    }


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_task_context(payload)
    lines = [
        "You are an embodied spatial reasoning agent exploring a 3D indoor scene with a mirror.",
        f"Task type: {ctx['task_type']}",
        f"Question: {ctx['question']}",
    ]
    if ctx["allowed_answers"]:
        lines.append("Options: " + ", ".join(ctx["allowed_answers"]))
    lines.extend([
        "",
        "You will receive recent views followed by the CURRENT view (always last).",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop>",',
        '  "reasoning": "<brief explanation>",',
        '  "answer": "<best current answer or not sure>",',
        '  "confidence": <float 0.0-1.0>',
        "}",
        "",
        "Rules:",
        "  - Output ONLY valid JSON.",
        "  - You are actively exploring, so if the current view is not enough, choose a movement action.",
        f"  - Before step {min_steps}, confidence should usually remain <= 0.5 unless the answer is extremely obvious.",
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.",
        "  - Prefer actions that reveal the mirror relation from a new angle.",
        "  - Use turn actions to change viewpoint, not only move actions.",
        "  - Past views are only supporting context; decide primarily from the current image.",
    ])
    return "\n".join(lines)


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_task_context(payload)
    lines = ["Exploration budget is exhausted.", f"Question: {ctx['question']}"]
    if ctx["allowed_answers"]:
        lines.append("You must choose exactly one final answer from: " + ", ".join(ctx["allowed_answers"]))
    lines.extend([
        "Do not answer 'not sure'.",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "answer": "<single final answer>",',
        '  "confidence": <float 0.0-1.0>,',
        '  "reasoning": "<brief explanation>"',
        "}",
    ])
    return "\n".join(lines)


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
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
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and answer.lower() != "not sure":
            return answer, int(item["step"])
    if history:
        return normalize_text(history[-1].get("answer")) or "not sure", int(history[-1]["step"])
    return "not sure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_text(answer).lower() in {"", "not sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
