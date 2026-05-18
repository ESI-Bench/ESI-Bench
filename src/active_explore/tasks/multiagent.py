from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


TASK_NAME = "multiagent"
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


def list_records(items: object) -> list[dict[str, Any]]:
    if isinstance(items, dict):
        return [value for value in items.values() if isinstance(value, dict)]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def object_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list_records(payload.get("objects_meta"))


def observer_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list_records(payload.get("observer_poses"))


def observer_step_number(record: dict[str, Any]) -> int:
    key = normalize_text(record.get("_key"))
    match = re.search(r"step_(\d+)", key)
    return int(match.group(1)) if match else -1


def target_category(payload: dict[str, Any]) -> str:
    return normalize_text(payload.get("object_category") or payload.get("task_category"))


def true_count(payload: dict[str, Any]) -> int:
    return int(payload.get("true_count"))


def robot_final_xy(payload: dict[str, Any]) -> np.ndarray:
    navigation = payload.get("navigation") or {}
    if navigation.get("final_xy"):
        return np.array(navigation["final_xy"][:2], dtype=float)
    waypoints = payload.get("path_waypoints") or []
    if waypoints:
        return np.array(waypoints[-1][:2], dtype=float)
    raise ValueError("Missing robot final xy in multiagent JSON")


def robot_final_quat(payload: dict[str, Any]) -> np.ndarray:
    step_log = payload.get("step_log") or []
    if step_log and isinstance(step_log[-1], dict) and step_log[-1].get("robot_quat_xyzw"):
        return np.array(step_log[-1]["robot_quat_xyzw"], dtype=float)
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    if not target_category(payload):
        return {"skip_reason": "missing_object_category"}
    try:
        true_count(payload)
    except Exception:
        return {"skip_reason": "missing_true_count"}
    if not observer_records(payload):
        return {"skip_reason": "missing_observer_poses"}
    try:
        robot_xy = robot_final_xy(payload)
    except Exception:
        return {"skip_reason": "missing_robot_final_xy"}
    missing_models = [item.get("_key") for item in object_records(payload) if not item.get("model")]
    if missing_models:
        return {"skip_reason": f"missing_object_models:{','.join(map(str, missing_models))}"}
    return {
        "robot_final_xy": robot_xy.tolist(),
        "robot_final_quat": robot_final_quat(payload).tolist(),
        "robot_start_z": float(payload.get("robot_start_z", 0.1)),
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
    for index, item in enumerate(object_records(payload)):
        if not item.get("position") or not item.get("quaternion_xyzw") or not item.get("model"):
            continue
        name = normalize_text(item.get("_key") or item.get("name")) or f"hidden_obj_{index}"
        output.append(
            {
                "type": "DatasetObject",
                "name": name,
                "category": item.get("category") or target_category(payload),
                "model": item["model"],
                "position": item["position"],
                "orientation": item["quaternion_xyzw"],
            }
        )
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = sorted(observer_records(payload), key=observer_step_number)
    if not records:
        raise ValueError("Missing observer_poses in multiagent JSON")
    pose = records[-1]
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {
            "view_index": len(records) - 1,
            "view": pose,
            "selection": "last_observer_pose",
            "observer_key": pose.get("_key"),
        },
    )


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    task_state = task_state or {}
    if not getattr(env, "robots", None):
        return {"robot_placed": False, "reason": "no_robot"}
    robot = env.robots[0]
    xy = np.array(task_state.get("robot_final_xy") or robot_final_xy(payload), dtype=float)
    z = float(task_state.get("robot_start_z", payload.get("robot_start_z", 0.1)))
    quat = np.array(task_state.get("robot_final_quat") or robot_final_quat(payload), dtype=float)
    robot.set_position_orientation(
        position=th.tensor([float(xy[0]), float(xy[1]), z], dtype=th.float32),
        orientation=th.tensor(quat, dtype=th.float32),
    )
    sync_error = None
    try:
        import omnigibson as og

        og.sim.step()
    except Exception as exc:
        sync_error = str(exc)
    return {
        "robot_placed": True,
        "robot_final_xy": xy.tolist(),
        "robot_start_z": z,
        "robot_final_quat": quat.tolist(),
        "sync_error": sync_error,
    }


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    category = display_category(target_category(payload))
    robot_xy = np.array((task_state or {}).get("robot_final_xy") or robot_final_xy(payload), dtype=float)
    proximity_thresh = float(payload.get("proximity_thresh", 1.0))
    return "\n".join(
        [
            "You are an embodied spatial reasoning expert navigating a 3D scene.",
            f"TASK: Count how many '{category}' objects are on the robot's walk path.",
            "The answer is one of: 0, 1, 2, or 3.",
            "",
            "Situation:",
            "- You are viewing from an observer camera looking toward the robot.",
            f"- The robot ended its walk near world position ({robot_xy[0]:.2f}, {robot_xy[1]:.2f}).",
            f"- Count '{category}' objects within about {proximity_thresh:.2f}m of the robot path.",
            "",
            "Movement rules:",
            "1. Move toward the robot's final position step by step.",
            "2. After every move_forward or move_backward, use turn_down to scan the ground/floor area, then turn back up before moving again.",
            "3. Turn left and right periodically to check both sides of the path.",
            "4. Only commit after you have gotten close to the robot and scanned thoroughly.",
            "5. The same object may appear in multiple views; keep correspondence and avoid double counting.",
            "",
            "Output EXACTLY one valid JSON object and nothing else:",
            "{",
            '  "action": "<action_name>",',
            '  "answer": "<0, 1, 2, 3, or unsure>",',
            '  "reasoning": "<brief explanation>",',
            '  "confidence": <float 0.0-1.0>',
            "}",
            "",
            "Available camera actions:",
            "  move_forward | move_backward | move_left | move_right | move_up | move_down",
            "  turn_left | turn_right | turn_up | turn_down | stop",
            f"Default to 'unsure' unless confidence reaches {threshold:.2f} and you have scanned thoroughly.",
        ]
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    category = display_category(target_category(payload))
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"You must commit to a final count of '{category}' objects near the robot path.",
            "Answer must be exactly 0, 1, 2, or 3. Do not answer 'unsure'.",
            'Output EXACTLY: {"answer": <0, 1, 2, or 3>, "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
        ]
    )


def normalize_answer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value <= 3:
        return value
    text = normalize_text(value).lower()
    if text in {"", "unsure", "not sure", "unknown", "none", "null"}:
        return None
    match = re.search(r"\b([0-3])\b", text)
    if match:
        return int(match.group(1))
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
    if action == "<end>":
        action = "stop"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    answer = normalize_answer(parsed.get("answer"))
    return {
        **parsed,
        "action": action,
        "answer": answer if answer is not None else "unsure",
        "conclusive": answer is not None,
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
    if bool(parsed.get("conclusive")) and float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[Any, int]:
    for item in reversed(history):
        answer = normalize_answer(item.get("answer"))
        if answer is not None:
            return answer, int(item["step"])
    return "unsure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: Any, stop_reason: str) -> bool:
    return normalize_answer(answer) is None


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predicted = normalize_answer((final_answer or {}).get("answer"))
    target = true_count(payload)
    return {
        "task_type": "multiagent",
        "question": normalize_text(payload.get("_question")),
        "object_category": target_category(payload),
        "predicted_count": predicted,
        "true_count": target,
        "correct": predicted == target if predicted is not None else None,
    }
