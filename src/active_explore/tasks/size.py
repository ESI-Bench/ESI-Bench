from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.transform_utils import quat2euler
from omnigibson.utils.usd_utils import create_joint, delete_or_deactivate_prim
from scipy.spatial.transform import Rotation


TASK_NAME = "size"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

MOVE_STEP = 0.25
MAX_NAV_ATTEMPTS = 50
AG_JOINT_PRIM = None

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


def objects_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("objects"))


def object_entry(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return objects_map(payload).get(key) or {}


def task_label(payload: dict[str, Any]) -> str:
    return display_category(payload.get("task_category") or object_entry(payload, "task_obj1").get("category"))


def ref_label(payload: dict[str, Any], index: int) -> str:
    return display_category(object_entry(payload, f"ref_obj{index}").get("category"))


def canonical_answer(payload: dict[str, Any], index: int) -> str:
    return f"near the {ref_label(payload, index)}"


def normalize_answer(payload: dict[str, Any], value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"", "not sure", "unsure", "unknown", "none"}:
        return "not sure"
    ref1 = ref_label(payload, 1).lower()
    ref2 = ref_label(payload, 2).lower()
    if ref1 and (ref1 in text or text in ref1):
        return canonical_answer(payload, 1)
    if ref2 and (ref2 in text or text in ref2):
        return canonical_answer(payload, 2)
    if "task_obj1" in text or "ref_obj1" in text:
        return canonical_answer(payload, 1)
    if "task_obj2" in text or "ref_obj2" in text:
        return canonical_answer(payload, 2)
    return "not sure"


def pose_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [pose for pose in payload.get("camera_poses") or [] if isinstance(pose, dict)]


def visibility_lookup(payload: dict[str, Any], key: str) -> dict[tuple[int, int], bool]:
    output: dict[tuple[int, int], bool] = {}
    for item in payload.get(key) or []:
        if not isinstance(item, dict):
            continue
        indices = item.get("_indices") or []
        if len(indices) >= 2:
            output[(int(indices[0]), int(indices[1]))] = bool(item.get("value"))
    return output


def pair_visible(payload: dict[str, Any], pair_index: int, view_index: int) -> bool:
    task_vis = visibility_lookup(payload, "exist_task_obj")
    ref_vis = visibility_lookup(payload, "exist_ref_obj")
    return bool(task_vis.get((pair_index, view_index), False) and ref_vis.get((pair_index, view_index), False))


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")) or "unknown_scene", normalize_text(payload.get("room")) or "unknown_room"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    objects = objects_map(payload)
    for key in ("task_obj1", "task_obj2", "ref_obj1", "ref_obj2"):
        obj = objects.get(key) or {}
        if not obj.get("category") or not obj.get("model") or obj.get("position") is None:
            return {"skip_reason": f"missing_{key}_category_model_or_position"}
    if not pose_records(payload):
        return {"skip_reason": "missing_camera_poses"}
    return {}


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for key in ("ref_obj1", "ref_obj2", "task_obj1", "task_obj2"):
        obj = object_entry(payload, key)
        if not obj or obj.get("model") == "scene_native" or not obj.get("model"):
            continue
        spec = {
            "type": "DatasetObject",
            "name": normalize_text(obj.get("name")) or key,
            "category": obj["category"],
            "model": obj["model"],
            "position": obj["position"],
            "orientation": obj.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0],
        }
        scale = obj.get("scale")
        if isinstance(scale, (int, float)) and float(scale) != 1.0:
            spec["scale"] = [float(scale), float(scale), float(scale)]
        elif isinstance(scale, list):
            spec["scale"] = scale
        output.append(spec)
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poses = pose_records(payload)
    chosen_index = None
    chosen_pair = None
    for index, _pose in enumerate(poses):
        if pair_visible(payload, 1, index):
            chosen_index = index
            chosen_pair = 1
            break
        if pair_visible(payload, 2, index):
            chosen_index = index
            chosen_pair = 2
            break
    if chosen_index is None:
        for index, pose in enumerate(poses):
            if pose.get("type") == "side":
                chosen_index = index
                break
    if chosen_index is None:
        chosen_index = 0
    pose = poses[chosen_index]
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {
            "view_index": chosen_index,
            "view": pose,
            "selection": "first_task_ref_pair_visible" if chosen_pair else "first_side_or_pose_fallback",
            "visible_pair": chosen_pair,
        },
    )


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = task_state if task_state is not None else {}
    robot = env.robots[0] if getattr(env, "robots", None) else None
    if robot is not None:
        for key, sensor in getattr(robot, "_sensors", {}).items():
            if "eyes:Camera:0" in key:
                position = sensor.get_position()
                position[2] -= 0.3
                position[0] += 0.05
                position[1] += 0.2
                sensor.set_position(position)
                for _ in range(100):
                    og.sim.step()
                state["robot_eye_sensor"] = key
                break
    state["floor_name"] = normalize_text(payload.get("floor_name") or payload.get("floor"))
    return {}


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any],
    task_state: dict[str, Any] | None = None,
) -> str:
    task = task_label(payload)
    ref1 = ref_label(payload, 1)
    ref2 = ref_label(payload, 2)
    return (
        f"You are an embodied spatial reasoning expert. TASK: You see two copies of a '{task}'. "
        f"One is near the {ref1} and one is near the {ref2}. Determine which copy is LARGER.\n\n"
        "Strict rules:\n"
        "1. Do not conclude from a single view. Perspective distortion is common.\n"
        "2. Actively move for better viewpoints: move_up plus turn_down for top-down views, move_backward for comparison, and sideways moves for parallax.\n"
        "3. You may inspect a task object with a robot-eye view. The pickup actions are valid only when both the target task object and its paired reference object are visible in the current viewer-camera frame.\n"
        f"4. Available pickup actions: pick_up_obj_near_{ref1}, pick_up_obj_near_{ref2}.\n"
        f"5. Only give a conclusive answer when confidence >= {threshold} and you have checked multiple viewpoints.\n"
        "6. To finish, use action 'stop'. Do not output '<end>'.\n"
        "7. JSON only. Output exactly: "
        '{"action": "<name>", "reasoning": "<str>", "answer": "<answer or not sure>", "confidence": <float>}.\n'
        f"Answer must be one of: 'near the {ref1}', 'near the {ref2}', or 'not sure'.\n"
        "Available movement actions: move_forward, move_backward, move_left, move_right, move_up, move_down, turn_left, turn_right, turn_up, turn_down, stop."
    )


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> str:
    task = task_label(payload)
    ref1 = ref_label(payload, 1)
    ref2 = ref_label(payload, 2)
    return (
        f"Final step. Which '{task}' is larger: the one near the {ref1}, or the one near the {ref2}? "
        f"Return JSON only with answer exactly 'near the {ref1}' or 'near the {ref2}'. 'not sure' is not allowed."
    )


def parse_model_output(parsed: dict[str, Any], payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    action = normalize_text(parsed.get("action")).lower()
    if action == "<end>":
        action = "stop"
    ref1_action = f"pick_up_obj_near_{ref_label(payload, 1).lower()}"
    ref2_action = f"pick_up_obj_near_{ref_label(payload, 2).lower()}"
    if action not in VALID_ACTIONS and action not in {ref1_action, ref2_action}:
        action = "move_forward"
    return {
        "action": action,
        "answer": normalize_answer(payload, parsed.get("answer")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasoning": normalize_text(parsed.get("reasoning")),
    }


def should_stop(
    parsed: dict[str, Any],
    history: list[dict[str, Any]],
    step: int,
    max_steps: int,
    min_steps: int,
    threshold: float,
) -> tuple[bool, str]:
    answer = normalize_text(parsed.get("answer")).lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if parsed.get("action") == "stop":
        return True, "model_stop"
    if answer not in {"", "not sure", "unsure", "unknown"} and confidence >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and answer.lower() not in {"not sure", "unsure", "unknown"}:
            return answer, int(item["step"])
    if history:
        return "not sure", int(history[-1]["step"])
    return "not sure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_text(answer).lower() in {"", "not sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = normalize_answer(payload, final_answer.get("answer"))
    ground_truth = normalize_answer(payload, payload.get("_ground_truth") or canonical_answer(payload, 2))
    return {
        "answer": answer,
        "ground_truth": ground_truth,
        "correct": answer == ground_truth,
        "task_category": object_entry(payload, "task_obj1").get("category"),
        "ref1_category": object_entry(payload, "ref_obj1").get("category"),
        "ref2_category": object_entry(payload, "ref_obj2").get("category"),
    }


def rotate_vec(vec: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).apply(vec)


def move_forward(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = pos.copy()
    right = rotate_vec(np.array([1.0, 0.0, 0.0]), quat)
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(right, up)
    forward[2] = 0.0
    norm = np.linalg.norm(forward)
    if norm > 1e-9:
        pos += forward / norm * MOVE_STEP
    return pos, quat.copy()


def check_exist_in_frame(obj_names: list[str]) -> dict[str, bool]:
    try:
        raw_obs = og.sim._viewer_camera._annotators["seg_instance"].get_data()
        id_to_labels = raw_obs["info"]["idToLabels"]
        visible_str = " ".join(str(value) for value in id_to_labels.values())
        return {name: name in visible_str for name in obj_names}
    except Exception:
        return {name: False for name in obj_names}


def mock_navigate(robot, obj, floor_obj, max_attempts: int = MAX_NAV_ATTEMPTS) -> bool:
    obj_pos, _ = obj.get_position_orientation()
    obj_xy = np.asarray(obj_pos[:2], dtype=float)
    for _attempt in range(max_attempts):
        try:
            if sample_kinematics("onTop", robot, floor_obj):
                robot_pos, robot_ori = robot.get_position_orientation()
                og.sim.step()
                robot_xy = np.asarray(robot_pos[:2], dtype=float)
                distance = float(np.linalg.norm(robot_xy - obj_xy))
                if distance < 2.0:
                    direction = obj_xy - robot_xy
                    desired_angle = np.arctan2(direction[1], direction[0])
                    current_angle = float(quat2euler(robot_ori)[2])
                    angle_diff = abs(np.arctan2(np.sin(desired_angle - current_angle), np.cos(desired_angle - current_angle)))
                    if angle_diff < np.pi / 2:
                        robot.reset()
                        for _ in range(100):
                            og.sim.step()
                        return True
        except Exception:
            continue
    return False


def mock_grasp(robot, obj) -> tuple[bool, Any, Any]:
    global AG_JOINT_PRIM
    grasp_point = robot.get_eef_position(robot.default_arm)
    obj_pos, obj_ori = obj.get_position_orientation()
    distance = float(np.linalg.norm(np.asarray(grasp_point[:2], dtype=float) - np.asarray(obj_pos[:2], dtype=float)))
    if distance > 2.0:
        return False, None, None
    obj.visual_only = True
    obj.set_position_orientation(grasp_point, [0.0, 0.0, 0.0, 1.0])
    obj.keep_still()
    og.sim.step()
    joint_prim_path = f"{robot.eef_links[robot.default_arm].prim_path}/ag_constraint"
    AG_JOINT_PRIM = create_joint(
        prim_path=joint_prim_path,
        joint_type="FixedJoint",
        body0=robot.eef_links[robot.default_arm].prim_path,
        body1=obj.root_link.prim_path,
        enabled=True,
        exclude_from_articulation=True,
    )
    for _ in range(100):
        og.sim.step()
    return True, obj_pos, obj_ori


def mock_put_down(obj, obj_pos, obj_ori) -> bool:
    global AG_JOINT_PRIM
    if AG_JOINT_PRIM is not None:
        delete_or_deactivate_prim(str(AG_JOINT_PRIM.GetPrimPath()))
        AG_JOINT_PRIM = None
    obj.set_position_orientation(obj_pos, obj_ori)
    obj.visual_only = False
    obj.keep_still()
    for _ in range(100):
        og.sim.step()
    return True


def capture_robot_eye(robot, run_dir: Path, label: str) -> str | None:
    eye_sensor = None
    for key, sensor in getattr(robot, "_sensors", {}).items():
        if "eyes:Camera:0" in key:
            eye_sensor = sensor
            break
    if eye_sensor is None:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    image = eye_sensor.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    path = run_dir / f"robot_eye_{label}.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return str(path)


def execute_task_action(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    action: str,
    pos: np.ndarray,
    quat: np.ndarray,
    task_state: dict[str, Any] | None = None,
    step: int | None = None,
    step_image_dir: Path | None = None,
) -> dict[str, Any]:
    action_lower = normalize_text(action).lower()
    if not action_lower.startswith("pick_up_obj_near_"):
        return {"handled": False}

    ref1 = ref_label(payload, 1).lower()
    ref2 = ref_label(payload, 2).lower()
    if action_lower == f"pick_up_obj_near_{ref1}":
        pair_index = 1
    elif action_lower == f"pick_up_obj_near_{ref2}":
        pair_index = 2
    else:
        return {"handled": False}

    scene = env.scene
    target_name = f"task_obj{pair_index}"
    ref_name = f"ref_obj{pair_index}"
    target_obj = scene.object_registry("name", target_name)
    ref_obj = scene.object_registry("name", ref_name)
    visible = check_exist_in_frame([target_name, ref_name])
    if not visible.get(target_name, False) or not visible.get(ref_name, False):
        next_pos, next_quat = move_forward(pos, quat)
        return {
            "handled": True,
            "operation": "pickup_invalid_move_forward",
            "success": False,
            "visibility": visible,
            "position": next_pos.tolist(),
            "quaternion_xyzw": next_quat.tolist(),
        }

    robot = env.robots[0] if getattr(env, "robots", None) else None
    floor_name = normalize_text((task_state or {}).get("floor_name") or payload.get("floor_name") or payload.get("floor"))
    floor_obj = scene.object_registry("name", floor_name) if floor_name else None
    if robot is None or floor_obj is None or target_obj is None or ref_obj is None:
        return {
            "handled": True,
            "operation": "pickup",
            "success": False,
            "reason": "missing_robot_floor_or_object",
            "position": pos.tolist(),
            "quaternion_xyzw": quat.tolist(),
        }

    robot_pos_before, robot_ori_before = robot.get_position_orientation()
    robot_eye_path = None
    nav_ok = grasp_ok = False
    saved_obj_pos = saved_obj_ori = None
    try:
        nav_ok = mock_navigate(robot, target_obj, floor_obj)
        if nav_ok:
            grasp_ok, saved_obj_pos, saved_obj_ori = mock_grasp(robot, target_obj)
            if grasp_ok:
                label = f"step_{int(step or 0):03d}_{target_name}"
                robot_eye_path = capture_robot_eye(robot, step_image_dir or Path("."), label)
                mock_put_down(target_obj, saved_obj_pos, saved_obj_ori)
    finally:
        if grasp_ok and saved_obj_pos is not None and saved_obj_ori is not None and AG_JOINT_PRIM is not None:
            mock_put_down(target_obj, saved_obj_pos, saved_obj_ori)
        robot.set_position_orientation(robot_pos_before, robot_ori_before)
        robot.keep_still()
        for _ in range(30):
            og.sim.step()

    return {
        "handled": True,
        "operation": "pickup",
        "pair_index": pair_index,
        "object": target_name,
        "reference": ref_name,
        "success": bool(robot_eye_path),
        "navigation_success": nav_ok,
        "grasp_success": grasp_ok,
        "extra_image_paths": [robot_eye_path] if robot_eye_path else [],
        "position": pos.tolist(),
        "quaternion_xyzw": quat.tolist(),
    }
