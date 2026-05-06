from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import omnigibson as og
import omnigibson.object_states as object_states
import torch as th


TASK_NAME = "counting"
DEFAULT_MODEL = "gemini-2.5-flash"
SOFTACC_K = 3

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


def normalize_answer(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_count_answer(value: Any, allow_not_sure: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return str(int(round(value))) if np.isfinite(value) else ""
    text = normalize_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if allow_not_sure and lowered in {"not sure", "unsure", "unknown", "i don't know", "dont know"}:
        return "not sure"
    match = re.search(r"-?\d+", text)
    if match:
        try:
            return str(int(match.group(0)))
        except Exception:
            pass
    return text


def get_context(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = payload.get("question_data", {})
    render = question_data.get("render", {})
    count_object = question_data.get("count_object") or {}
    return {
        "question": normalize_text(question_data.get("question")),
        "count_target": normalize_text(question_data.get("count_target") or count_object.get("category")),
        "count_target_display": normalize_text(count_object.get("display_name") or question_data.get("count_target")),
        "render_target_category": normalize_text(render.get("target_category")),
        "task_type": normalize_text(payload.get("task_type") or question_data.get("task_type")),
        "options": [normalize_text(opt) for opt in question_data.get("options", []) if normalize_text(opt)],
    }


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
) -> str:
    ctx = get_context(payload)
    semantic_target = ctx["count_target_display"] or ctx["count_target"]
    lines = [
        "You are an embodied visual counting agent exploring a 3D indoor scene.",
        f"Task type: {ctx['task_type']}",
        f"Question: {ctx['question']}",
    ]
    if semantic_target:
        lines.append(f"Requested semantic target category: {semantic_target}")
    if semantic_target and ctx["render_target_category"] and ctx["render_target_category"] != ctx["count_target"]:
        lines.append(
            "Important: in this simulator reconstruction, the visible proxy asset category for the target is "
            f"'{ctx['render_target_category']}', while the question category is '{semantic_target}'."
        )
    if ctx["options"]:
        lines.append("Dataset options (for reference only): " + ", ".join(ctx["options"]))
    lines.extend([
        "",
        "You will receive recent views followed by the CURRENT view (always last).",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop|pick up <object>|place <object> on top of <object>|put back <object>>",',
        '  "reasoning": "<brief explanation>",',
        '  "answer": "<positive integer count or not sure>",',
        '  "confidence": <float 0.0-1.0>',
        "}",
        "",
        "Rules:",
        "  - Output ONLY valid JSON.",
        "  - Count only the requested target category across the whole room.",
        "  - Avoid double-counting the same object across different views.",
        "  - Do not answer 0 because the scene is guaranteed to contain at least one target object.",
        "  - If the current view is insufficient, choose a movement action instead of stopping.",
        f"  - Before step {min_steps}, confidence should usually remain <= 0.5 unless the answer is extremely obvious.",
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.",
        "  - Prefer exploring corners, occluded regions, and likely hiding places from new angles.",
    ])
    return "\n".join(lines)


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> str:
    ctx = get_context(payload)
    semantic_target = ctx["count_target_display"] or ctx["count_target"]
    lines = ["Exploration budget is exhausted.", f"Question: {ctx['question']}"]
    if semantic_target:
        lines.append(f"Count target: {semantic_target}")
    if semantic_target and ctx["render_target_category"] and ctx["render_target_category"] != ctx["count_target"]:
        lines.append(f"Visible proxy asset category: {ctx['render_target_category']}")
    lines.extend([
        "You must output one final positive integer count.",
        "Do not answer 0.",
        "Do not answer 'not sure'.",
        "Output EXACTLY one JSON object and nothing else:",
        '{"answer": "<positive integer>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
    ])
    return "\n".join(lines)


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    action = normalize_answer(parsed.get("action")) or "move_forward"
    answer = normalize_count_answer(parsed.get("answer"), allow_not_sure=True) or "not sure"
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


def extract_primary_camera_pose(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = payload.get("question_data", {})
    render = question_data.get("render", {}) or {}
    primary_image = normalize_text(render.get("image"))
    camera_poses = render.get("camera_poses") or {}
    if isinstance(camera_poses, list):
        pose_by_name = {
            normalize_text(pose.get("_key") or pose.get("image") or pose.get("filename")): pose
            for pose in camera_poses
            if isinstance(pose, dict)
        }
        if primary_image:
            pose = pose_by_name.get(Path(primary_image).name)
            if pose:
                return pose
        for pose in camera_poses:
            if isinstance(pose, dict) and pose:
                return pose
    elif primary_image:
        pose = camera_poses.get(Path(primary_image).name)
        if pose:
            return pose
    if isinstance(camera_poses, dict) and camera_poses:
        first_key = next(iter(camera_poses))
        if camera_poses[first_key]:
            return camera_poses[first_key]
    fallback_pose = render.get("camera_pose") or question_data.get("camera_pose") or {}
    if fallback_pose:
        return fallback_pose
    raise ValueError("Missing camera pose in counting question JSON")


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose = extract_primary_camera_pose(payload)
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"camera_pose": pose},
    )


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    render = payload.get("question_data", {}).get("render", {})
    resolved = render.get("resolved_objects") or {}
    output = []
    for group_name in ("containers", "confusers", "targets"):
        for item in resolved.get(group_name, []) or []:
            name = normalize_text(item.get("name"))
            category = normalize_text(item.get("category"))
            model = normalize_text(item.get("model"))
            position = item.get("position") or item.get("requested_position")
            if not name or not category or not model or position is None:
                continue
            output.append({
                "type": "DatasetObject",
                "name": name,
                "category": category,
                "model": model,
                "position": position,
                "orientation": item.get("quaternion_xyzw") or item.get("orientation") or [0.0, 0.0, 0.0, 1.0],
                "visual_only": True,
            })
    return output


def tensor_to_tuple3(value) -> tuple[float, float, float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    vals = value.tolist() if hasattr(value, "tolist") else list(value)
    return float(vals[0]), float(vals[1]), float(vals[2])


def quaternion_xyzw_to_front_xy(quat_xyzw) -> tuple[float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(-np.cos(yaw)), float(-np.sin(yaw))


def object_specs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    render = payload.get("question_data", {}).get("render", {})
    resolved = render.get("resolved_objects") or {}
    output = []
    for group_name in ("containers", "confusers", "targets"):
        output.extend(resolved.get(group_name, []) or [])
    return output


def maybe_set_open_state(obj, open_state_value) -> None:
    if open_state_value is None:
        return
    try:
        states = getattr(obj, "states", {})
        if object_states.Open in states:
            states[object_states.Open].set_value(bool(open_state_value))
    except Exception:
        pass


def find_container_reveal_links(container_obj) -> list[object]:
    links = list((getattr(container_obj, "links", {}) or {}).values())
    if not links:
        return []

    container_category = str(getattr(container_obj, "category", "")).lower()

    def link_record(link):
        try:
            bbox_min, bbox_max = [tensor_to_tuple3(x) for x in link.aabb]
            center_x = float((bbox_min[0] + bbox_max[0]) * 0.5)
            center_y = float((bbox_min[1] + bbox_max[1]) * 0.5)
            center_z = float((bbox_min[2] + bbox_max[2]) * 0.5)
            extent_x = float(bbox_max[0] - bbox_min[0])
            extent_y = float(bbox_max[1] - bbox_min[1])
            extent_z = float(bbox_max[2] - bbox_min[2])
            footprint = max(extent_x, 0.0) * max(extent_y, 0.0)
        except Exception:
            center_x = center_y = center_z = 0.0
            extent_x = extent_y = extent_z = 0.0
            footprint = 0.0
        return center_x, center_y, center_z, extent_x, extent_y, extent_z, footprint

    named = []
    for link in links:
        link_name = str(getattr(link, "name", "")).lower()
        if any(keyword in link_name for keyword in ("lid", "cover", "door", "top", "panel")):
            named.append(link)
    if named:
        return named

    try:
        bbox_min, bbox_max = [tensor_to_tuple3(x) for x in container_obj.aabb]
        container_width = max(float(bbox_max[0] - bbox_min[0]), 1e-4)
        container_depth = max(float(bbox_max[1] - bbox_min[1]), 1e-4)
        container_height = max(float(bbox_max[2] - bbox_min[2]), 1e-4)
        container_center_z = float((bbox_min[2] + bbox_max[2]) * 0.5)
    except Exception:
        return []

    if "box" in container_category or "microwave" in container_category:
        try:
            _, container_quat = container_obj.get_position_orientation()
            if hasattr(container_quat, "detach"):
                container_quat = container_quat.detach().cpu().tolist()
            front_xy = quaternion_xyzw_to_front_xy(container_quat)
        except Exception:
            front_xy = (1.0, 0.0)

        try:
            container_pos, _ = container_obj.get_position_orientation()
            if hasattr(container_pos, "detach"):
                container_pos = container_pos.detach().cpu().tolist()
        except Exception:
            container_pos = [0.0, 0.0, 0.0]

        box_candidates = []
        for link in links:
            center_x, center_y, _center_z, extent_x, extent_y, extent_z, _footprint = link_record(link)
            if max(extent_x, extent_y, extent_z) <= 1e-4:
                continue
            rel_x = float(center_x - container_pos[0])
            rel_y = float(center_y - container_pos[1])
            front_score = rel_x * front_xy[0] + rel_y * front_xy[1]
            front_thickness = abs(front_xy[0]) * extent_x + abs(front_xy[1]) * extent_y
            lateral_width = abs(front_xy[1]) * extent_x + abs(front_xy[0]) * extent_y
            if lateral_width < 0.35 * min(container_width, container_depth):
                continue
            if extent_z < 0.35 * container_height:
                continue
            thickness_ratio = front_thickness / max(min(container_width, container_depth), 1e-4)
            if thickness_ratio > 0.45:
                continue
            box_candidates.append((front_score, -thickness_ratio, lateral_width * extent_z, link))
        box_candidates.sort(key=lambda item: (-item[0], item[1], item[2]), reverse=True)
        if box_candidates:
            return [box_candidates[0][3]]

    heuristic = []
    for link in links:
        _center_x, _center_y, center_z, _extent_x, _extent_y, extent_z, footprint = link_record(link)
        if center_z <= container_center_z or extent_z <= 0.0:
            continue
        heuristic.append((footprint, -extent_z, link))
    heuristic.sort(reverse=True)
    return [item[2] for item in heuristic[:1]]


def hide_reveal_links(container_obj) -> list[tuple[object, bool]]:
    hidden = []
    for link in find_container_reveal_links(container_obj):
        try:
            hidden.append((link, bool(link.visible)))
            link.visible = False
        except Exception:
            pass
    return hidden


def hide_hidden_in_box_microwave_doors(scene, payload: dict[str, Any]) -> list[tuple[object, bool]]:
    if payload.get("task_type") != "hidden_in_box":
        return []

    resolved = payload.get("question_data", {}).get("render", {}).get("resolved_objects") or {}
    target_container_names = {
        normalize_text(item.get("name"))
        for item in (resolved.get("containers") or [])
        if normalize_text(item.get("name"))
    }

    candidates = []
    seen_ids: set[int] = set()
    for container_name in sorted(target_container_names):
        obj = scene.object_registry("name", container_name)
        if obj is not None and id(obj) not in seen_ids:
            seen_ids.add(id(obj))
            candidates.append(obj)

    for obj in scene.objects:
        name = str(getattr(obj, "name", "")).lower()
        category = str(getattr(obj, "category", "")).lower()
        if "microwave" not in name and "microwave" not in category:
            continue
        if id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))
        candidates.append(obj)

    hidden = []
    for obj in candidates:
        hidden.extend(hide_reveal_links(obj))
    return hidden


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any] | None = None) -> dict[str, Any]:
    scene = env.scene
    created_objects = {}
    for item in object_specs(payload):
        name = normalize_text(item.get("name"))
        if not name:
            continue
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        try:
            obj.visual_only = True
        except Exception:
            pass
        maybe_set_open_state(obj, item.get("open_state"))
        created_objects[name] = obj

    for _ in range(20):
        og.sim.step()

    inside_edges = []
    for item in object_specs(payload):
        container_name = normalize_text(item.get("container_name"))
        obj_name = normalize_text(item.get("name"))
        if not container_name or not obj_name:
            continue
        obj = created_objects.get(obj_name) or scene.object_registry("name", obj_name)
        container_obj = created_objects.get(container_name) or scene.object_registry("name", container_name)
        if obj is None or container_obj is None:
            continue
        try:
            states = getattr(obj, "states", {})
            if object_states.Inside in states:
                states[object_states.Inside].set_value(container_obj, True)
                inside_edges.append({"object": obj_name, "container": container_name})
        except Exception:
            pass

    hidden_links = hide_hidden_in_box_microwave_doors(scene, payload)
    if hidden_links:
        for _ in range(5):
            og.sim.step()
    for _ in range(10):
        og.sim.step()
    return {
        "visual_only_objects": sorted(created_objects),
        "inside_edges": inside_edges,
        "hidden_reveal_links": len(hidden_links),
    }


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return payload["scene"], payload["room"]


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def ground_truth(payload: dict[str, Any]) -> str:
    return normalize_count_answer(payload.get("question_data", {}).get("answer"), allow_not_sure=False)


def finalize_answer(answer: str) -> str:
    return normalize_count_answer(answer, allow_not_sure=False)


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    confidence = float(parsed.get("confidence", 0.0))
    action = normalize_answer(parsed.get("action"))
    if confidence >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if action == "stop":
        return True, "model_stop"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_count_answer(item.get("answer"), allow_not_sure=True)
        if answer and answer != "not sure":
            return answer, int(item["step"])
    if history:
        last = normalize_count_answer(history[-1].get("answer"), allow_not_sure=True) or "not sure"
        return last, int(history[-1]["step"])
    return "not sure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_answer(answer) == "not sure" or not normalize_text(answer)


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred = finalize_answer((final_answer or {}).get("answer"))
    gt = ground_truth(payload)
    pred_num = int(pred) if re.fullmatch(r"-?\d+", pred or "") else None
    gt_num = int(gt) if re.fullmatch(r"-?\d+", gt or "") else None
    abs_error = abs(pred_num - gt_num) if pred_num is not None and gt_num is not None else None
    soft_score = max(0.0, 1.0 - abs_error / float(SOFTACC_K)) if abs_error is not None else None
    return {
        "ground_truth": gt,
        "ground_truth_number": gt_num,
        "predicted_number": pred_num,
        "absolute_error": abs_error,
        "soft_score": soft_score,
        "correct": pred == gt if pred else None,
    }
