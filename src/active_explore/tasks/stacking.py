from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import torch as th
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.usd_utils import create_joint, delete_or_deactivate_prim
from scipy.spatial.transform import Rotation


TASK_NAME = "stacking"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
STAGING_X_BASE = 150.0
STAGING_Y = 100.0
STAGING_Z = 100.0
STAGING_X_STRIDE = 5.0
SETTLE_STEPS = 60
PLACEMENT_RETRIES = 5
TILT_DOT_TOL = 0.9
XY_IOU_TOL = 0.1
AG_JOINT_PRIM = None

CAMERA_ACTIONS = {
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


def object_names(payload: dict[str, Any]) -> list[str]:
    return [normalize_text(item) for item in payload.get("object_names") or [] if normalize_text(item)]


def object_categories(payload: dict[str, Any]) -> list[str]:
    return [normalize_text(item) for item in payload.get("object_categories") or [] if normalize_text(item)]


def object_models(payload: dict[str, Any]) -> list[str]:
    return [normalize_text(item) for item in payload.get("object_models") or [] if normalize_text(item)]


def object_scales(payload: dict[str, Any]) -> list[Any]:
    return list(payload.get("object_scales") or [])


def name_to_category(payload: dict[str, Any]) -> dict[str, str]:
    return dict(zip(object_names(payload), object_categories(payload)))


def category_to_name(payload: dict[str, Any]) -> dict[str, str]:
    return {cat: name for name, cat in name_to_category(payload).items()}


def initial_pose_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("initial_poses"))


def initial_camera_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return keyed_list_to_map(payload.get("initial_cam_poses"))


def initial_exist_map(payload: dict[str, Any]) -> dict[str, bool]:
    return {key: bool(value.get("value")) for key, value in keyed_list_to_map(payload.get("initial_exist_flags")).items()}


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")) or "unknown_scene", normalize_text(payload.get("room")) or "unknown_room"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or source_path.stem


def all_initial_visible(payload: dict[str, Any], az_str: str) -> bool:
    exist = initial_exist_map(payload)
    return all(exist.get(f"exist_{name}_initial_side_{az_str}.png", False) for name in object_names(payload))


def get_initial_camera_pose(payload: dict[str, Any]) -> tuple[list[float], list[float], list[str], str]:
    poses = initial_camera_map(payload)
    names = object_names(payload)
    for az in ("az270", "az090"):
        key = f"initial_side_{az}.png"
        if key in poses and all_initial_visible(payload, az):
            lr_names = names[:] if az == "az270" else list(reversed(names))
            return poses[key]["position"], poses[key]["quaternion_xyzw"], lr_names, key
    key = "initial_side_az000.png"
    if key in poses:
        return poses[key]["position"], poses[key]["quaternion_xyzw"], names[:], key
    if poses:
        first_key = next(iter(poses))
        pose = poses[first_key]
        return pose["position"], pose["quaternion_xyzw"], names[:], first_key
    raise ValueError("Missing initial_cam_poses for stacking")


def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    fwd = np.array(target, dtype=float) - np.array(eye, dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    fwd /= norm
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    true_up /= np.linalg.norm(true_up)
    return Rotation.from_matrix(np.column_stack([right, true_up, -fwd])).as_quat()


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    names = object_names(payload)
    cats = object_categories(payload)
    models = object_models(payload)
    scales = object_scales(payload)
    poses = initial_pose_map(payload)
    if len(names) != 3 or len(cats) != 3 or len(models) != 3 or len(scales) != 3:
        return {"skip_reason": "expected_three_stack_objects"}
    missing_pose = [name for name in names if name not in poses]
    if missing_pose:
        return {"skip_reason": f"missing_initial_pose_{missing_pose[0]}"}
    if not normalize_text(payload.get("floor")):
        return {"skip_reason": "missing_floor"}
    gt_orders = get_gt_orders(payload)
    if not gt_orders:
        return {"skip_reason": "missing_stable_gt_orders"}
    return {"gt_orders": gt_orders, "current_stack": [], "held_obj_name": None}


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for index, (name, category, model, scale) in enumerate(
        zip(object_names(payload), object_categories(payload), object_models(payload), object_scales(payload))
    ):
        output.append(
            {
                "type": "DatasetObject",
                "name": name,
                "category": category,
                "model": model,
                "scale": scale,
                "position": [STAGING_X_BASE + index * STAGING_X_STRIDE, STAGING_Y, STAGING_Z],
                "orientation": SQUARE_ORI,
            }
        )
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eye, quat, lr_names, camera_key = get_initial_camera_pose(payload)
    lr_cats = [name_to_category(payload)[name] for name in lr_names]
    return (
        np.array(eye, dtype=float),
        np.array(quat, dtype=float),
        {"camera_key": camera_key, "lr_names": lr_names, "lr_categories": lr_cats},
    )


def find_eye_camera_key(robot) -> str:
    for key in getattr(robot, "_sensors", {}):
        if "eyes:Camera:0" in key:
            return key
    return ""


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = task_state if task_state is not None else {}
    robot = env.robots[0] if getattr(env, "robots", None) else None
    if robot is not None:
        eye_camera_key = find_eye_camera_key(robot)
        if eye_camera_key:
            position = robot._sensors[eye_camera_key].get_position()
            position[2] -= 0.3
            position[0] += 0.05
            position[1] += 0.2
            robot._sensors[eye_camera_key].set_position(position)
            for _ in range(100):
                og.sim.step()
            state["eye_camera_key"] = eye_camera_key

    floor_name = normalize_text(payload.get("floor"))
    floor_obj = env.scene.object_registry("name", floor_name)
    if floor_obj is None:
        raise ValueError(f"Stacking floor object not found: {floor_name}")
    state["floor_obj"] = floor_obj

    poses = initial_pose_map(payload)
    objs_by_name = {}
    for name in object_names(payload):
        obj = env.scene.object_registry("name", name)
        if obj is None:
            raise ValueError(f"Stacking object not found: {name}")
        pose = poses[name]
        obj.set_position_orientation(
            position=th.tensor(pose["position"], dtype=th.float32),
            orientation=th.tensor(pose["quaternion_xyzw"], dtype=th.float32),
        )
        obj.keep_still()
        objs_by_name[name] = obj
    for _ in range(30):
        og.sim.step()
    state["objs_by_name"] = objs_by_name

    all_positions = [obj.get_position_orientation()[0].cpu().numpy() for obj in objs_by_name.values()]
    cluster_centre = np.mean(all_positions, axis=0)
    look_target = np.array([cluster_centre[0], cluster_centre[1], 0.25])
    eye = np.array(camera_info["view"]["position"] if "view" in camera_info else get_initial_camera_pose(payload)[0], dtype=float)
    direction_xy = np.array([eye[0] - look_target[0], eye[1] - look_target[1], 0.0])
    norm = np.linalg.norm(direction_xy)
    if norm > 1e-9:
        eye = eye + direction_xy / norm * 0.3
    quat = look_at_quat(eye, look_target)
    return {"camera_override": {"position": eye.tolist(), "quaternion_xyzw": quat.tolist()}}


def get_aabb(obj):
    bmin, bmax = [x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x) for x in obj.aabb]
    return bmin, bmax


def tilt_check(obj) -> tuple[bool, float]:
    _, quat = obj.get_position_orientation()
    quat_np = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
    world_up = Rotation.from_quat(quat_np).apply(np.array([0.0, 0.0, 1.0]))
    dot = float(np.dot(world_up, np.array([0.0, 0.0, 1.0])))
    return dot >= TILT_DOT_TOL, dot


def xy_iou(bmin_a, bmax_a, bmin_b, bmax_b) -> float:
    ix_min = max(float(bmin_a[0]), float(bmin_b[0]))
    ix_max = min(float(bmax_a[0]), float(bmax_b[0]))
    iy_min = max(float(bmin_a[1]), float(bmin_b[1]))
    iy_max = min(float(bmax_a[1]), float(bmax_b[1]))
    inter = max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)
    area_a = max(0.0, float(bmax_a[0] - bmin_a[0])) * max(0.0, float(bmax_a[1] - bmin_a[1]))
    area_b = max(0.0, float(bmax_b[0] - bmin_b[0])) * max(0.0, float(bmax_b[1] - bmin_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def is_on_top(upper_obj, lower_obj) -> bool:
    upper_min, upper_max = get_aabb(upper_obj)
    lower_min, lower_max = get_aabb(lower_obj)
    z_ok = float(upper_min[2]) > float(lower_min[2])
    xy_ok = xy_iou(upper_min, upper_max, lower_min, lower_max) >= XY_IOU_TOL
    return z_ok and xy_ok


def check_stack_stable(objs: list) -> bool:
    for index in range(1, len(objs)):
        if not is_on_top(objs[index], objs[index - 1]):
            return False
    return all(tilt_check(obj)[0] for obj in objs)


def get_gt_orders(payload: dict[str, Any]) -> list[list[str]]:
    raw_gt = payload.get("_ground_truth")
    if isinstance(raw_gt, list) and raw_gt:
        output = []
        for order in raw_gt:
            if isinstance(order, list) and len(order) == 3:
                output.append([normalize_text(item) for item in order])
        if output:
            return output
    names = object_names(payload)
    cats = object_categories(payload)
    name_to_cat = dict(zip(names, cats))
    output = []
    for trial in payload.get("trials") or []:
        if trial.get("all_stable", False):
            output.append([name_to_cat[name] for name in trial.get("order", []) if name in name_to_cat])
    return [order for order in output if len(order) == 3]


def order_matches_gt(order_cats: list[str] | None, gt_orders: list[list[str]]) -> bool:
    if not order_cats:
        return False
    norm = [item.strip().lower() for item in order_cats]
    return any(norm == [item.strip().lower() for item in gt] for gt in gt_orders)


def parse_answer_order(answer: Any, obj_cats: list[str]) -> list[str] | None:
    text = normalize_text(answer).lower()
    if not text or text.startswith("unsure") or text in {"not sure", "unknown"}:
        return None
    for prefix in ("1)", "2)", "3)", "1.", "2.", "3.", "bottom:", "middle:", "top:"):
        text = text.replace(prefix, ",")
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    matched = []
    for part in parts:
        for cat in obj_cats:
            cat_norm = cat.lower().replace("_", " ")
            part_norm = part.lower().replace("_", " ")
            if cat_norm in part_norm or part_norm in cat_norm:
                matched.append(cat)
                break
    return matched if len(matched) == 3 and len(set(matched)) == 3 else None


def try_expand_gt(task_state: dict[str, Any], payload: dict[str, Any]) -> None:
    current_stack = task_state.get("current_stack") or []
    if len(current_stack) != 3:
        return
    objs_by_name = task_state.get("objs_by_name") or {}
    name_to_cat = name_to_category(payload)
    if not all(name in objs_by_name for name in current_stack):
        return
    if check_stack_stable([objs_by_name[name] for name in current_stack]):
        cats = [name_to_cat[name] for name in current_stack]
        gt_orders = task_state.setdefault("gt_orders", get_gt_orders(payload))
        if not order_matches_gt(cats, gt_orders):
            gt_orders.append(cats)


def category_display_to_name(payload: dict[str, Any]) -> dict[str, str]:
    output = {}
    for name, category in name_to_category(payload).items():
        output[category] = name
        output[display_category(category)] = name
    return output


def resolve_object_name(text: str, payload: dict[str, Any]) -> str | None:
    target = normalize_text(text).lower().replace("_", " ")
    for display, name in sorted(category_display_to_name(payload).items(), key=lambda item: -len(item[0])):
        display_norm = display.lower().replace("_", " ")
        if display_norm in target or target in display_norm or name.lower() in target:
            return name
    return None


def mock_grasp(robot, obj) -> bool:
    global AG_JOINT_PRIM
    grasp_point = robot.get_eef_position(robot.default_arm)
    obj.visual_only = True
    obj.set_position_orientation(grasp_point, SQUARE_ORI)
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
    for _ in range(10):
        og.sim.step()
    return True


def mock_release(obj) -> None:
    global AG_JOINT_PRIM
    if AG_JOINT_PRIM is not None:
        delete_or_deactivate_prim(str(AG_JOINT_PRIM.GetPrimPath()))
        AG_JOINT_PRIM = None
    obj.set_position_orientation([100.0, 100.0, 100.0], SQUARE_ORI)
    obj.visual_only = False
    obj.keep_still()
    for _ in range(10):
        og.sim.step()


def place_on_top(obj, target_obj) -> bool:
    mock_release(obj)
    ok = False
    for _attempt in range(PLACEMENT_RETRIES):
        try:
            ok = sample_kinematics("onTop", obj, target_obj, use_last_ditch_effort=True, use_trav_map=False)
            if ok:
                break
        except Exception:
            pass
    for _ in range(SETTLE_STEPS):
        og.sim.step()
    return bool(ok)


def capture_robot_eye(robot, eye_camera_key: str, path: Path) -> str | None:
    if not eye_camera_key or eye_camera_key not in getattr(robot, "_sensors", {}):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        og.sim.render()
    image = robot._sensors[eye_camera_key].get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return str(path)


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    lr_cats = (camera_info or {}).get("lr_categories") or object_categories(payload)
    obj_list = "; ".join(display_category(cat) for cat in lr_cats)
    actions = (
        "pick up <object> | place <object> on top of <object> | place <object> on the floor | "
        "move_forward | move_backward | move_left | move_right | move_up | move_down | "
        "turn_left | turn_right | turn_up | turn_down | stop"
    )
    return (
        f"You are a physical reasoning agent. Your goal is to find the stable stacking order, bottom to top, for these 3 objects: {obj_list}.\n\n"
        "You can pick up objects and place them on top of each other to test stability. After placing, the next image shows the result.\n\n"
        "Rules:\n"
        "- Always pick up an object before placing it on another object or on the floor.\n"
        "- A stack is stable if all objects are upright and each upper object rests on the one below.\n"
        f"- Output confidence >= {threshold:.2f} only if you have physically tested and confirmed a stable stack.\n"
        "- If unsure, keep exploring.\n"
        "- Only use move/turn actions if you cannot see the objects.\n"
        "- Use 'place <object> on the floor' to reset an object back to its original position.\n\n"
        'Always output valid JSON: {"action": "<action>", "reasoning": "<description>", '
        '"answer": "1) <bottom> 2) <middle> 3) <top> or unsure", "confidence": <float>}\n\n'
        f"Available actions: {actions}\n"
        f"Use only object names from this list: {[display_category(cat) for cat in object_categories(payload)]}"
    )


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> str:
    return (
        "Exploration budget is exhausted. Choose the most likely stable stacking order, bottom to top. "
        "Return JSON with answer formatted as '1) <bottom> 2) <middle> 3) <top>', confidence, and reasoning."
    )


def parse_model_output(parsed: dict[str, Any], payload: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    action = normalize_text(parsed.get("action")).lower()
    if action == "<end>":
        action = "stop"
    if not action:
        action = "move_forward"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    parsed_order = parse_answer_order(parsed.get("answer"), object_categories(payload or {})) if isinstance(payload, dict) else None
    return {
        **parsed,
        "action": action,
        "answer": normalize_text(parsed.get("answer")) or "unsure",
        "parsed_order": parsed_order,
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
    if normalize_text(parsed.get("action")) == "stop":
        return True, "model_stop"
    if parsed.get("parsed_order") and float(parsed.get("confidence", 0.0) or 0.0) >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        if normalize_text(item.get("answer")).lower() not in {"", "unsure", "not sure", "unknown"}:
            return normalize_text(item.get("answer")), int(item["step"])
    if history:
        return "unsure", int(history[-1]["step"])
    return "unsure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_text(answer).lower() in {"", "unsure", "not sure", "unknown"}


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
    state = task_state or {}
    action_lower = normalize_text(action).lower()
    if action_lower in CAMERA_ACTIONS:
        return {"handled": False}
    objs_by_name = state.get("objs_by_name") or {}
    robot = env.robots[0] if getattr(env, "robots", None) else None

    if action_lower.startswith("pick up "):
        obj_name = resolve_object_name(action_lower[len("pick up "):], payload)
        obj = objs_by_name.get(obj_name)
        success = bool(robot is not None and obj is not None and mock_grasp(robot, obj))
        if success:
            state["held_obj_name"] = obj_name
            if obj_name in (state.get("current_stack") or []):
                state["current_stack"] = []
        extra_paths = []
        eye_key = state.get("eye_camera_key")
        if success and robot is not None and eye_key:
            image_path = (step_image_dir or Path(".")) / f"robot_eye_step_{int(step or 0):03d}_{obj_name}.png"
            captured = capture_robot_eye(robot, eye_key, image_path)
            if captured:
                extra_paths.append(captured)
        return {
            "handled": True,
            "operation": "pick_up",
            "object": obj_name,
            "success": success,
            "extra_image_paths": extra_paths,
            "position": pos.tolist(),
            "quaternion_xyzw": quat.tolist(),
        }

    if "place" in action_lower and "on top of" in action_lower:
        obj_part, target_part = action_lower.split("on top of", 1)
        obj_name = resolve_object_name(obj_part.replace("place", "").strip(), payload)
        target_name = resolve_object_name(target_part.strip(), payload)
        obj = objs_by_name.get(obj_name)
        target = objs_by_name.get(target_name)
        success = bool(obj is not None and target is not None and place_on_top(obj, target))
        if success:
            state["held_obj_name"] = None
            current_stack = list(state.get("current_stack") or [])
            if not current_stack:
                current_stack = [target_name, obj_name]
            elif current_stack[-1] == target_name:
                current_stack.append(obj_name)
            else:
                current_stack = [target_name, obj_name]
            state["current_stack"] = current_stack
            if len(current_stack) == 3:
                try_expand_gt(state, payload)
        return {
            "handled": True,
            "operation": "place_on_top",
            "object": obj_name,
            "target": target_name,
            "success": success,
            "current_stack": state.get("current_stack") or [],
            "gt_orders": state.get("gt_orders") or get_gt_orders(payload),
            "position": pos.tolist(),
            "quaternion_xyzw": quat.tolist(),
        }

    if "place" in action_lower and "on the floor" in action_lower:
        obj_name = resolve_object_name(action_lower.replace("place", "").replace("on the floor", "").strip(), payload)
        obj = objs_by_name.get(obj_name)
        pose = initial_pose_map(payload).get(obj_name)
        other_poses = {}
        for other_name, other_obj in objs_by_name.items():
            if other_name != obj_name:
                p, q = other_obj.get_position_orientation()
                other_poses[other_name] = (p.cpu().numpy().copy(), q.cpu().numpy().copy())
        success = False
        if obj is not None and pose:
            mock_release(obj)
            obj.set_position_orientation(
                position=th.tensor(pose["position"], dtype=th.float32),
                orientation=th.tensor(pose["quaternion_xyzw"], dtype=th.float32),
            )
            obj.keep_still()
            for _ in range(SETTLE_STEPS):
                og.sim.step()
            for other_name, (other_pos, other_quat) in other_poses.items():
                other_obj = objs_by_name[other_name]
                other_obj.set_position_orientation(
                    position=th.tensor(other_pos, dtype=th.float32),
                    orientation=th.tensor(other_quat, dtype=th.float32),
                )
                other_obj.keep_still()
            for _ in range(10):
                og.sim.step()
            state["held_obj_name"] = None
            state["current_stack"] = []
            success = True
        return {
            "handled": True,
            "operation": "place_on_floor",
            "object": obj_name,
            "success": success,
            "position": pos.tolist(),
            "quaternion_xyzw": quat.tolist(),
        }

    return {"handled": False}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = task_state or {}
    gt_orders = state.get("gt_orders") or get_gt_orders(payload)
    parsed = parse_answer_order(final_answer.get("answer"), object_categories(payload))
    return {
        "answer": normalize_text(final_answer.get("answer")) or "unsure",
        "final_parsed": parsed,
        "ground_truth": gt_orders,
        "correct": order_matches_gt(parsed, gt_orders),
        "object_categories": object_categories(payload),
        "lr_order_shown": (camera_info or {}).get("lr_categories") or object_categories(payload),
        "current_stack": state.get("current_stack") or [],
    }
