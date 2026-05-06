from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


TASK_NAME = "action"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]

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


def keyed_list_to_map(items: object) -> dict[str, Any]:
    if isinstance(items, dict):
        return {normalize_text(key): value for key, value in items.items() if normalize_text(key)}
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item["_key"])] = item
    return output


def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    fwd = np.array(target, float) - np.array(eye, float)
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    fwd /= n
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    cam_up /= np.linalg.norm(cam_up)
    return Rotation.from_matrix(np.column_stack([right, cam_up, -fwd])).as_quat()


def pick_best_view(qa_per_view, stage0_initial_views, n_objects):
    exist_map = {v["view_idx"]: v for v in stage0_initial_views}
    qa_map = {v["view_idx"]: v for v in qa_per_view}
    required = ["obj_fixed", "obj_A_correct", "obj_A_wrong"]
    if n_objects == 3:
        required += ["obj_C_correct", "obj_C_wrong"]

    def all_visible(idx):
        view = exist_map.get(idx, {})
        return all(view.get(f"exist_{key}", False) for key in required)

    def has_opposite(idx):
        return qa_map.get(idx, {}).get("opposite_sides", False)

    for idx in [0, 2, 1, 3]:
        if all_visible(idx) and has_opposite(idx):
            return idx, qa_map[idx], exist_map[idx]
    for idx in [0, 2, 1, 3]:
        if all_visible(idx):
            return idx, qa_map[idx], exist_map[idx]
    return 0, qa_map.get(0, qa_per_view[0]), exist_map.get(0, stage0_initial_views[0])


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_objects = int(payload["n_objects"])
    view_idx, qa, view_data = pick_best_view(payload["qa_per_view"], payload["stage0_initial_views"], n_objects)
    geo = payload["orbital_geometry"]
    centre_xy = geo["centre_xy"]
    radius = geo["radius"]
    height = geo["height"]
    centre_z = geo["centre_z"]
    azimuth = 2.0 * np.pi * view_idx / 4
    eye = np.array([centre_xy[0] + radius * np.cos(azimuth), centre_xy[1] + radius * np.sin(azimuth), height], dtype=float)
    quat = look_at_quat(eye, [centre_xy[0], centre_xy[1], centre_z])
    return eye, quat, {"view_idx": view_idx, "qa": qa, "view_data": view_data}


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    initial_poses = keyed_list_to_map(payload.get("initial_poses"))
    for index, (name, info) in enumerate(keyed_list_to_map(payload.get("objects")).items()):
        pose = initial_poses.get(name) or {}
        output.append({
            "type": "DatasetObject",
            "name": name,
            "category": info.get("cat") or info.get("category"),
            "model": info["model"],
            "scale": info.get("scale"),
            "position": pose.get("position", [150.0 + index * 5, 100.0, 100.0]),
            "orientation": pose.get("quaternion_xyzw", SQUARE_ORI),
        })
    return output


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return f"{payload.get('scene', 'scene')}_{payload.get('room', 'room')}_{payload.get('run_idx', source_path.stem)}"


def _task_display(payload: dict[str, Any], camera_info: dict[str, Any]):
    hierarchy = payload["hierarchy"]
    n_objects = int(payload["n_objects"])
    qa = camera_info["qa"]
    a_name = hierarchy["obj_A_cat"].replace("_", " ")
    b_name = hierarchy["obj_fixed_cat"].replace("_", " ")
    c_name = hierarchy.get("obj_C_cat", "")
    c_display = c_name.replace("_", " ") if c_name else None
    rel_a_correct = qa["rel_A_correct"]
    rel_a_wrong = qa["rel_A_wrong"]
    rel_c_correct = qa.get("rel_C_correct")
    rel_c_wrong = qa.get("rel_C_wrong")
    display = {
        "obj_A_correct": f"{a_name} to the {rel_a_correct} of B",
        "obj_A_wrong": f"{a_name} to the {rel_a_wrong} of B",
        "obj_fixed": f"{b_name} (B, center)",
    }
    if n_objects == 3:
        display["obj_C_correct"] = f"{c_display} to the {rel_c_correct} of B"
        display["obj_C_wrong"] = f"{c_display} to the {rel_c_wrong} of B"
    return display, qa, n_objects


def object_display_names(payload: dict[str, Any], camera_info: dict[str, Any]) -> dict[str, str]:
    display, _qa, _n_objects = _task_display(payload, camera_info)
    return display


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> dict[str, Any]:
    return {}


def build_task_question(payload: dict[str, Any], camera_info: dict[str, Any]) -> tuple[str, str]:
    hierarchy = payload["hierarchy"]
    display, qa, n_objects = _task_display(payload, camera_info)
    a_name = hierarchy["obj_A_cat"].replace("_", " ")
    b_name = hierarchy["obj_fixed_cat"].replace("_", " ")
    c_name = hierarchy.get("obj_C_cat")
    if n_objects == 2:
        question = (
            f"There are two {a_name}s and one {b_name} (B) in the scene. "
            f"One {a_name} is to the {qa['rel_A_correct']} of B, one to the {qa['rel_A_wrong']}. "
            f"Which {a_name} should B be placed ON TOP OF as its base? "
            f"Answer: '{qa['rel_A_correct']}' or '{qa['rel_A_wrong']}'."
        )
        answer_format = f"'{qa['rel_A_correct']}' or '{qa['rel_A_wrong']}'"
    else:
        c_display = c_name.replace("_", " ") if c_name else "object"
        question = (
            f"There are two {a_name}s, one {b_name} (B), and two {c_display}s. "
            f"One {a_name} is to the {qa['rel_A_correct']} of B, one to the {qa['rel_A_wrong']}. "
            f"One {c_display} is to the {qa['rel_C_correct']} of B, one to the {qa['rel_C_wrong']}. "
            f"Which {a_name} should B be placed ON TOP OF? Which {c_display} should go ON TOP of B?"
        )
        answer_format = f"two directions e.g. '{qa['rel_A_correct']}, {qa['rel_C_correct']}'"
    return question, answer_format


def build_system_prompt(payload: dict[str, Any], threshold: float, min_steps: int, camera_info: dict[str, Any] | None = None) -> str:
    camera_info = camera_info or {}
    display, _qa, _n_objects = _task_display(payload, camera_info)
    question, answer_format = build_task_question(payload, camera_info)
    obj_list = "\n".join(f"  - {value}" for value in display.values())
    return (
        "You are an embodied physical reasoning agent.\n\n"
        f"TASK: {question}\n\n"
        f"OBJECTS (use these exact names):\n{obj_list}\n\n"
        "ACTIONS:\n"
        "  pick up <object>\n"
        "  place <object> on top of <object>\n"
        "  put back <object>\n"
        "  move_forward | move_backward | move_left | move_right | move_up | move_down\n"
        "  turn_left | turn_right | turn_up | turn_down | stop\n\n"
        "RULES:\n"
        "  - Physically test placements before committing when possible.\n"
        "  - If a fit is wrong, use put back and retry.\n"
        f"  - Before step {min_steps}, do not stop unless the answer is extremely obvious.\n"
        f"  - Stop when confidence is at least {threshold:.2f} and you have a parsed answer.\n"
        "  - Output ONLY valid JSON with action, reasoning, answer, confidence.\n"
        f'  - answer must be <{answer_format}> or "unsure".'
    )


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> str:
    question, answer_format = build_task_question(payload, camera_info or {})
    return (
        "Exploration budget is exhausted.\n"
        f"TASK: {question}\n"
        f"You must choose a final answer in this format: {answer_format}.\n"
        "Output ONLY valid JSON: "
        '{"answer": "<final answer>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}'
    )


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        **parsed,
        "action": normalize_text(parsed.get("action")).lower() or "move_forward",
        "answer": normalize_text(parsed.get("answer")) or "unsure",
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def parse_answer(answer: str, payload: dict[str, Any], camera_info: dict[str, Any]):
    if not answer or "unsure" in answer.lower():
        return None
    _display, qa, n_objects = _task_display(payload, camera_info)
    found = re.findall(r"\b(left|right|above|below)\b", answer.lower())
    if n_objects == 2:
        return (found[0], None) if found else None
    if len(found) >= 2:
        return found[0], found[1]
    if len(found) == 1:
        return found[0], None
    return None


def finalize_answer(answer: str, payload: dict[str, Any] | None = None, camera_info: dict[str, Any] | None = None):
    if payload is None or camera_info is None:
        return answer
    return parse_answer(answer, payload, camera_info)


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    confidence = float(parsed.get("confidence", 0.0))
    action = normalize_text(parsed.get("action")).lower()
    answer = normalize_text(parsed.get("answer"))
    if confidence >= threshold and step >= min_steps and answer and "unsure" not in answer.lower():
        return True, "confidence_threshold"
    if action == "stop":
        return True, "model_stop"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and "unsure" not in answer.lower():
            return answer, int(item["step"])
    if history:
        return normalize_text(history[-1].get("answer")) or "unsure", int(history[-1]["step"])
    return "unsure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return not normalize_text(answer) or "unsure" in normalize_text(answer).lower()


def score(payload: dict[str, Any], final_answer: dict[str, Any], camera_info: dict[str, Any] | None = None) -> dict[str, Any]:
    camera_info = camera_info or {}
    parsed = parse_answer((final_answer or {}).get("answer", ""), payload, camera_info)
    _display, qa, n_objects = _task_display(payload, camera_info)
    correct = False
    if parsed is not None and parsed[0].lower() == qa["rel_A_correct"].lower():
        correct = n_objects == 2 or (parsed[1] is not None and parsed[1].lower() == qa["rel_C_correct"].lower())
    return {
        "ground_truth": {"A": qa["rel_A_correct"], "C": qa.get("rel_C_correct")},
        "final_parsed_A": parsed[0] if parsed else None,
        "final_parsed_C": parsed[1] if parsed and n_objects == 3 else None,
        "correct": correct,
    }
