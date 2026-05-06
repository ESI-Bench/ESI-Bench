#!/usr/bin/env python3
"""
batch_angle_confusion.py

Single-run batch worker for the angle-confusion setup.
This is the batch analogue of the original single-object demo, with:
  - batch CLI contract
  - sampled support mode (room-floor centre vs random-table centre)
  - richer metadata
  - exit-code contract

Exit codes:
  0 = success
  1 = partial failure (renders completed but visibility criteria not met)
  2 = hard setup failure
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch as th
import yaml
from scipy.spatial.transform import Rotation

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

RESOLUTION_DEFAULT = 1000
SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
SCENES_DIR = "scenes5"
SUPPORT_MARGIN = 0.02

DEFAULT_CATEGORY = "bookcase"
DEFAULT_MODEL = "bsvnni"
DEFAULT_ASSET_MANIFEST = "asset_manifest.json"
DEFAULT_NUM_AZIMUTHS = 12
DEFAULT_AZIMUTH_STEP_DEG = 30.0
DEFAULT_HEIGHTS = (0.05, 0.45)
DEFAULT_RADIUS_PAD = 0.55
DEFAULT_TOPDOWN_Z = 2.2




def load_asset_manifest(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def resolve_angle_asset(args, rng: random.Random):
    manual = args.category is not None and args.model is not None
    use_manifest = args.asset_manifest is not None and os.path.exists(args.asset_manifest)
    if manual:
        return {
            "category": args.category,
            "model": args.model,
            "object_id": f"{args.category}-{args.model}",
            "selection_mode": "manual",
            "asset_manifest": args.asset_manifest,
        }
    if not use_manifest:
        return {
            "category": DEFAULT_CATEGORY,
            "model": DEFAULT_MODEL,
            "object_id": f"{DEFAULT_CATEGORY}-{DEFAULT_MODEL}",
            "selection_mode": "hardcoded_default",
            "asset_manifest": None,
        }
    manifest = load_asset_manifest(args.asset_manifest)
    candidates = manifest.get("angle_confusion_candidates", [])
    if not candidates:
        raise RuntimeError(f"No angle_confusion_candidates in manifest: {args.asset_manifest}")
    chosen = rng.choice(candidates)
    return {
        **chosen,
        "selection_mode": "manifest_random",
        "asset_manifest": args.asset_manifest,
    }

def build_config(robot_type: str, scene_model: str, category: str, model: str):
    config_filename = os.path.join(og.example_config_path, f"{robot_type.lower()}_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"]["scene_model"] = scene_model
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["objects"] = [{
        "type": "DatasetObject",
        "name": "target_obj",
        "category": category,
        "model": model,
        "position": [150.0, 100.0, 100.0],
        "orientation": SQUARE_ORI,
        "scale": [1, 1, 1],
    }]
    config["robots"][0]["name"] = "demo"
    if "sensor_config" in config["robots"][0]:
        vision_cfg = config["robots"][0]["sensor_config"].get("VisionSensor", {})
        sensor_kwargs = vision_cfg.get("sensor_kwargs", {})
        sensor_kwargs["image_height"] = RESOLUTION_DEFAULT
        sensor_kwargs["image_width"] = RESOLUTION_DEFAULT
        vision_cfg["sensor_kwargs"] = sensor_kwargs
        config["robots"][0]["sensor_config"]["VisionSensor"] = vision_cfg
    return config


def step_n(n: int = 5):
    for _ in range(n):
        og.sim.step()


def aabb_minmax_np(obj):
    aabb = obj.aabb
    return aabb[0].cpu().numpy(), aabb[1].cpu().numpy()


def get_xy_half_diag(obj) -> float:
    bb_min, bb_max = aabb_minmax_np(obj)
    dx = abs(float(bb_max[0]) - float(bb_min[0])) / 2.0
    dy = abs(float(bb_max[1]) - float(bb_min[1])) / 2.0
    return float(np.sqrt(dx * dx + dy * dy))


def get_xy_half_extents(obj) -> Tuple[float, float]:
    bb_min, bb_max = aabb_minmax_np(obj)
    return abs(float(bb_max[0]) - float(bb_min[0])) / 2.0, abs(float(bb_max[1]) - float(bb_min[1])) / 2.0


def get_object_center_np(obj) -> np.ndarray:
    pos, _ = obj.get_position_orientation()
    return pos.cpu().numpy()


def get_scene_objects(scene):
    raw = getattr(scene, "objects", [])
    if isinstance(raw, dict):
        return list(raw.values())
    return list(raw)


def clamp_xy_inside_support(desired_xy: np.ndarray, obj, support_obj, margin: float = SUPPORT_MARGIN) -> np.ndarray:
    support_min, support_max = aabb_minmax_np(support_obj)
    obj_hx, obj_hy = get_xy_half_extents(obj)
    x = min(max(float(desired_xy[0]), float(support_min[0]) + obj_hx + margin), float(support_max[0]) - obj_hx - margin)
    y = min(max(float(desired_xy[1]), float(support_min[1]) + obj_hy + margin), float(support_max[1]) - obj_hy - margin)
    return np.array([x, y], dtype=float)


def get_support_center_xy(support_obj) -> np.ndarray:
    bmin, bmax = aabb_minmax_np(support_obj)
    return np.array([0.5 * (float(bmin[0]) + float(bmax[0])), 0.5 * (float(bmin[1]) + float(bmax[1]))], dtype=float)


def place_on_top_centered(obj, support_obj):
    ok = obj.states[object_states.OnTop].set_value(support_obj, True)
    if not ok:
        raise RuntimeError(f"Failed to place {obj.name} on top of {support_obj.name}")
    step_n(15)
    pos, _ = obj.get_position_orientation()
    z = float(pos.cpu().numpy()[2])
    center_xy = clamp_xy_inside_support(get_support_center_xy(support_obj), obj, support_obj)
    obj.set_position_orientation(
        position=th.tensor([center_xy[0], center_xy[1], z], dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    obj.keep_still()
    step_n(10)
    return center_xy


def object_xy_center_inside_floor(obj, floor_obj, slack: float = 0.05) -> bool:
    floor_min, floor_max = aabb_minmax_np(floor_obj)
    obj_center = get_object_center_np(obj)
    return (
        float(floor_min[0]) - slack <= float(obj_center[0]) <= float(floor_max[0]) + slack and
        float(floor_min[1]) - slack <= float(obj_center[1]) <= float(floor_max[1]) + slack
    )


def find_tables_for_floor(scene, floor_obj) -> List:
    tables = []
    for obj in get_scene_objects(scene):
        if obj.name == floor_obj.name:
            continue
        cat = str(getattr(obj, "category", "")).lower()
        if "table" not in cat:
            continue
        try:
            if object_xy_center_inside_floor(obj, floor_obj):
                tables.append(obj)
        except Exception:
            continue
    tables.sort(key=lambda o: o.name)
    return tables


def choose_support(scene, floor_obj, rng: random.Random, placement_mode: str = "sample"):
    tables = find_tables_for_floor(scene, floor_obj)
    modes = ["floor_center"]
    if tables:
        modes.append("table_center")
    if placement_mode == "sample":
        chosen_mode = rng.choice(modes)
    elif placement_mode in modes:
        chosen_mode = placement_mode
    elif placement_mode == "table_center" and not tables:
        print("[support] requested table_center but no room table found; falling back to floor_center")
        chosen_mode = "floor_center"
    else:
        raise RuntimeError(f"Unsupported placement_mode={placement_mode}")

    support_obj = floor_obj if chosen_mode == "floor_center" else rng.choice(tables)
    bmin, bmax = aabb_minmax_np(support_obj)
    return support_obj, {
        "placement_mode": chosen_mode,
        "support_type": "floor" if chosen_mode == "floor_center" else "table",
        "support_name": support_obj.name,
        "support_category": str(getattr(support_obj, "category", "")),
        "support_bbox_min": bmin.tolist(),
        "support_bbox_max": bmax.tolist(),
        "support_selection_policy": "room_floor_center" if chosen_mode == "floor_center" else "random_table_center",
        "num_candidate_tables_in_room": len(tables),
        "candidate_table_names": [t.name for t in tables],
    }


def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos, dtype=float) - np.array(eye_pos, dtype=float)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        return np.array([0, 0, 0, 1], dtype=float)
    forward /= norm
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0], dtype=float)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()


def capture_rgb(path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] saved -> {path}")


def set_camera_and_capture(eye: np.ndarray, look_target: np.ndarray, path: str) -> Dict:
    quat = look_at_quaternion(eye, look_target)
    og.sim._viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float32), th.tensor(quat, dtype=th.float32))
    capture_rgb(path)
    return {"position": eye.tolist(), "quaternion_xyzw": quat.tolist()}


def visibility_check(actual_names: Tuple[str, ...]) -> Dict[str, bool]:
    for _ in range(60):
        og.sim.step()
    raw_obs = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"].get("idToLabels", {})
    visible_str = " ".join(id_to_labels.values())
    result = {name: (name in visible_str) for name in actual_names}
    print(f"[seg] {result}")
    return result


def live_object_info(obj) -> Dict:
    pos, quat = obj.get_position_orientation()
    bmin, bmax = aabb_minmax_np(obj)
    return {
        "name": obj.name,
        "category": str(getattr(obj, "category", "")),
        "position": pos.cpu().numpy().tolist(),
        "quaternion_xyzw": quat.cpu().numpy().tolist(),
        "aabb_min": bmin.tolist(),
        "aabb_max": bmax.tolist(),
        "xy_half_diag": get_xy_half_diag(obj),
    }


def render_angle_confusion_views(obj, out_dir: str, obj_names: Tuple[str, ...], num_azimuths: int,
                                 azimuth_step_deg: float, heights: Tuple[float, ...],
                                 radius_pad: float, topdown_z: float):
    os.makedirs(out_dir, exist_ok=True)
    center = get_object_center_np(obj)
    radius = get_xy_half_diag(obj) + radius_pad

    metadata = {}
    visible_count = 0
    view_idx = 0
    for az_idx in range(num_azimuths):
        az_deg = az_idx * azimuth_step_deg
        az_rad = math.radians(az_deg)
        for h in heights:
            eye = np.array([
                center[0] + radius * math.cos(az_rad),
                center[1] + radius * math.sin(az_rad),
                center[2] + h,
            ], dtype=float)
            fname = f"view_{view_idx:02d}_az{int(round(az_deg)):03d}_h{h:.2f}.png"
            fpath = os.path.join(out_dir, fname)
            pose = set_camera_and_capture(eye, center, fpath)
            vis = visibility_check(obj_names)
            visible_count += int(vis[obj_names[0]])
            metadata[fname] = {**pose, "type": "ring", "azimuth_deg": az_deg, "height_offset": h, "visibility": vis}
            view_idx += 1

    top_eye = np.array([center[0], center[1], topdown_z], dtype=float)
    fname = f"view_{view_idx:02d}_topdown.png"
    fpath = os.path.join(out_dir, fname)
    pose = set_camera_and_capture(top_eye, center.copy(), fpath)
    vis = visibility_check(obj_names)
    visible_count += int(vis[obj_names[0]])
    metadata[fname] = {**pose, "type": "topdown", "visibility": vis}
    return metadata, visible_count


def main():
    parser = argparse.ArgumentParser(description="Batch angle-confusion worker")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--floor", required=True)
    parser.add_argument("--run_idx", type=int, required=True)
    parser.add_argument("--keys_json", type=str, default="keys.json")
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--output_root", type=str, default="renders_angle_confusion")

    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--asset_manifest", type=str, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--placement_mode", choices=["sample", "floor_center", "table_center"], default="sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_azimuths", type=int, default=DEFAULT_NUM_AZIMUTHS)
    parser.add_argument("--azimuth_step_deg", type=float, default=DEFAULT_AZIMUTH_STEP_DEG)
    parser.add_argument("--radius_pad", type=float, default=DEFAULT_RADIUS_PAD)
    parser.add_argument("--topdown_z", type=float, default=DEFAULT_TOPDOWN_Z)
    args = parser.parse_args()

    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else (abs(hash((args.scene, args.room, args.run_idx, "angle_confusion"))) % (2**31))
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    asset_choice = resolve_angle_asset(args, rng)
    print(f"[asset] mode={asset_choice['selection_mode']} object={asset_choice['object_id']}")
    config = build_config(robot_type=args.robot, scene_model=args.scene, category=asset_choice["category"], model=asset_choice["model"])
    env = og.Environment(configs=config)
    exit_code = 2

    try:
        scene = env.scene
        target_obj = scene.object_registry("name", "target_obj")
        floor_obj = scene.object_registry("name", args.floor)
        if floor_obj is None:
            raise RuntimeError(f"Could not resolve floor object: {args.floor}")

        for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
            og.sim._viewer_camera.add_modality(modality)
        step_n(50)

        support_obj, support_meta = choose_support(scene, floor_obj, rng, placement_mode=args.placement_mode)
        center_xy = place_on_top_centered(target_obj, support_obj)
        step_n(20)

        target_live = live_object_info(target_obj)
        views_meta, visible_count = render_angle_confusion_views(
            obj=target_obj,
            out_dir=run_dir,
            obj_names=(target_obj.name,),
            num_azimuths=args.num_azimuths,
            azimuth_step_deg=args.azimuth_step_deg,
            heights=tuple(DEFAULT_HEIGHTS),
            radius_pad=args.radius_pad,
            topdown_z=args.topdown_z,
        )

        metadata = {
            "scene": args.scene,
            "room": args.room,
            "run_idx": args.run_idx,
            "seed": seed,
            "floor_name": args.floor,
            "layout": "angle_confusion_batch",
            "task_type": "angle_confusion",
            "answer": "same_object_different_viewpoints",
            "asset_manifest": args.asset_manifest if (args.asset_manifest and os.path.exists(args.asset_manifest)) else None,
            **support_meta,
            "objects": {
                "target": {**target_live, "requested_category": asset_choice["category"], "model": asset_choice["model"], "centered_xy": center_xy.tolist(), "selected_asset": asset_choice},
                "support": live_object_info(support_obj),
            },
            "camera_layout": {
                "num_azimuths": args.num_azimuths,
                "azimuth_step_deg": args.azimuth_step_deg,
                "heights": list(DEFAULT_HEIGHTS),
                "radius_pad": args.radius_pad,
                "topdown_z": args.topdown_z,
            },
            "visibility_summary": {
                "target_visible_count": int(visible_count),
            },
            "views": views_meta,
        }

        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[done] wrote metadata -> {metadata_path}")

        success = visible_count >= 2
        exit_code = 0 if success else 1
        print("[success] batch angle-confusion run complete" if success else "[partial] run completed but visibility criteria were not met")

    except Exception as e:
        err_path = os.path.join(run_dir, "error.txt")
        with open(err_path, "w") as f:
            f.write(str(e) + "\n")
        print(f"[error] {e}")
        exit_code = 2
    finally:
        og.shutdown()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
