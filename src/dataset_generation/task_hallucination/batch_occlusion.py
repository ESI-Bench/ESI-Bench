#!/usr/bin/env python3
"""
batch_occlusion.py

Single-run batch worker for partial-occlusion examples.

Compared with the original occlusion demo:
  - batch CLI contract: one (scene, room, run_idx) per process
  - samples support placement mode between room-floor centre and random-table centre
  - stores richer run metadata / ground truth
  - returns batch-friendly exit codes

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

SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
IMG_WIDTH = 560
IMG_HEIGHT = 560

DEFAULT_TARGET_CATEGORY = "chalice"
DEFAULT_TARGET_MODEL = "sfkezf"
DEFAULT_OCCLUDER_CATEGORY = "blender"
DEFAULT_OCCLUDER_MODEL = "cwkvib"
DEFAULT_ASSET_MANIFEST = "asset_manifest.json"

DEFAULT_NUM_AZIMUTHS = 12
DEFAULT_AZIMUTH_STEP_DEG = 30.0
DEFAULT_HEIGHTS = (0.05, 0.45)
DEFAULT_RADIUS_PAD = 0.55
DEFAULT_TOPDOWN_Z = 1.8

DEFAULT_VIEW_SIDE = "W"
DEFAULT_OCCLUDER_CORNER = "NE"
DEFAULT_FRONT_OVERLAP = 0.0
DEFAULT_LATERAL_EXTRA = 0.0
DEFAULT_CHALLENGE_HEIGHT = 0.10
DEFAULT_CHALLENGE_RADIUS_PAD = 0.40

SUPPORT_MARGIN = 0.02
SCENES_DIR = "scenes5"

VALID_TABLE_SUPPORT_CATEGORIES = {
    "breakfast_table",
    "coffee_table",
    "commercial_kitchen_table",
    "conference_table",
    "console_table",
    "gaming_table",
    "garden_coffee_table",
    "lab_table",
    "pedestal_table",
    "ping_pong_table",
    "pool_table",
}


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def load_asset_manifest(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def sample_view_and_corner(rng: random.Random, view_side: str, occluder_corner: str):
    compat = {
        "E": ["NW", "SW"],
        "W": ["NE", "SE"],
        "N": ["SW", "SE"],
        "S": ["NW", "NE"],
    }
    if view_side == "sample":
        chosen_view = rng.choice(sorted(compat.keys()))
    else:
        chosen_view = view_side
    if chosen_view not in compat:
        raise ValueError(f"Unsupported view_side={chosen_view}")
    if occluder_corner == "sample":
        chosen_corner = rng.choice(compat[chosen_view])
    else:
        chosen_corner = occluder_corner
    if chosen_corner not in compat[chosen_view]:
        raise ValueError(f"Incompatible pair: view_side={chosen_view}, occluder_corner={chosen_corner}")
    return chosen_view, chosen_corner


def resolve_occlusion_assets(args, rng: random.Random):
    manual = all(v is not None for v in [args.target_category, args.target_model, args.occluder_category, args.occluder_model])
    use_manifest = args.asset_manifest is not None and os.path.exists(args.asset_manifest)
    if manual:
        return {
            "target": {"category": args.target_category, "model": args.target_model, "object_id": f"{args.target_category}-{args.target_model}"},
            "occluder": {"category": args.occluder_category, "model": args.occluder_model, "object_id": f"{args.occluder_category}-{args.occluder_model}"},
            "selection_mode": "manual",
            "asset_manifest": args.asset_manifest,
        }
    if not use_manifest:
        return {
            "target": {"category": DEFAULT_TARGET_CATEGORY, "model": DEFAULT_TARGET_MODEL, "object_id": f"{DEFAULT_TARGET_CATEGORY}-{DEFAULT_TARGET_MODEL}"},
            "occluder": {"category": DEFAULT_OCCLUDER_CATEGORY, "model": DEFAULT_OCCLUDER_MODEL, "object_id": f"{DEFAULT_OCCLUDER_CATEGORY}-{DEFAULT_OCCLUDER_MODEL}"},
            "selection_mode": "hardcoded_default",
            "asset_manifest": None,
        }
    manifest = load_asset_manifest(args.asset_manifest)
    targets = manifest.get("occlusion_target_candidates", [])
    compatible = manifest.get("occlusion_compatible_occluders", {})
    if not targets or not compatible:
        raise RuntimeError(f"Manifest missing occlusion candidate pools: {args.asset_manifest}")
    target = rng.choice(targets)
    occ_ids = compatible.get(target["object_id"], [])
    if not occ_ids:
        raise RuntimeError(f"No compatible occluders for target {target['object_id']}")
    all_assets = {a["object_id"]: a for a in manifest.get("occlusion_occluder_candidates", [])}
    occ_choices = [all_assets[oid] for oid in occ_ids if oid in all_assets]
    if not occ_choices:
        raise RuntimeError(f"Compatible occluders not found in occlusion_occluder_candidates for target {target['object_id']}")
    occluder = rng.choice(occ_choices)
    return {
        "target": target,
        "occluder": occluder,
        "num_compatible_occluders": len(occ_choices),
        "selection_mode": "manifest_random_compatible",
        "asset_manifest": args.asset_manifest,
    }

def build_config(
    scene_model: str,
    robot_type: str,
    target_category: str,
    target_model: str,
    occluder_category: str,
    occluder_model: str,
):
    config_filename = os.path.join(og.example_config_path, f"{robot_type.lower()}_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = scene_model
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "target_obj",
            "category": target_category,
            "model": target_model,
            "position": [150.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale": [1, 1, 1],
        },
        {
            "type": "DatasetObject",
            "name": "occluder_obj",
            "category": occluder_category,
            "model": occluder_model,
            "position": [160.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale": [1, 1, 1],
        },
    ]

    config["robots"][0]["name"] = "demo"
    config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = IMG_HEIGHT
    config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = IMG_WIDTH
    return config


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
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
    hx = abs(float(bb_max[0]) - float(bb_min[0])) / 2.0
    hy = abs(float(bb_max[1]) - float(bb_min[1])) / 2.0
    return hx, hy


def get_object_center_np(obj) -> np.ndarray:
    pos, _ = obj.get_position_orientation()
    return pos.cpu().numpy()


def get_scene_objects(scene):
    raw = getattr(scene, "objects", [])
    if isinstance(raw, dict):
        return list(raw.values())
    return list(raw)


def get_support_center_xy(support_obj) -> np.ndarray:
    bb_min, bb_max = aabb_minmax_np(support_obj)
    return np.array([
        0.5 * (float(bb_min[0]) + float(bb_max[0])),
        0.5 * (float(bb_min[1]) + float(bb_max[1])),
    ], dtype=float)


def clamp_xy_inside_support(desired_xy: np.ndarray, obj, support_obj, margin: float = SUPPORT_MARGIN) -> np.ndarray:
    support_min, support_max = aabb_minmax_np(support_obj)
    obj_hx, obj_hy = get_xy_half_extents(obj)
    x = min(max(float(desired_xy[0]), float(support_min[0]) + obj_hx + margin), float(support_max[0]) - obj_hx - margin)
    y = min(max(float(desired_xy[1]), float(support_min[1]) + obj_hy + margin), float(support_max[1]) - obj_hy - margin)
    return np.array([x, y], dtype=float)


def object_fits_on_support(obj, support_obj, margin: float = SUPPORT_MARGIN) -> bool:
    support_min, support_max = aabb_minmax_np(support_obj)
    obj_hx, obj_hy = get_xy_half_extents(obj)
    support_hx = abs(float(support_max[0]) - float(support_min[0])) / 2.0
    support_hy = abs(float(support_max[1]) - float(support_min[1])) / 2.0
    return (obj_hx + margin <= support_hx) and (obj_hy + margin <= support_hy)


def get_supported_pose_z_for_xy(obj, support_obj, z_epsilon: float = 0.005) -> float:
    pos, _ = obj.get_position_orientation()
    bmin, _ = aabb_minmax_np(obj)
    support_min, support_max = aabb_minmax_np(support_obj)
    bottom_clearance = float(pos.cpu().numpy()[2]) - float(bmin[2])
    return float(support_max[2]) + bottom_clearance + z_epsilon


def reposition_on_support(obj, xy: np.ndarray, support_obj, orientation_xyzw=SQUARE_ORI, z_epsilon: float = 0.005):
    z = get_supported_pose_z_for_xy(obj, support_obj, z_epsilon=z_epsilon)
    obj.set_position_orientation(
        position=th.tensor([float(xy[0]), float(xy[1]), z], dtype=th.float32),
        orientation=th.tensor(orientation_xyzw, dtype=th.float32),
    )
    obj.keep_still()
    step_n(10)


def place_on_top_centered_with_orientation(obj, support_obj, orientation_xyzw=SQUARE_ORI, margin: float = SUPPORT_MARGIN):
    ok = obj.states[object_states.OnTop].set_value(support_obj, True)
    if not ok:
        raise RuntimeError(f"Failed to place {obj.name} on top of {support_obj.name}")
    step_n(15)
    center_xy = clamp_xy_inside_support(get_support_center_xy(support_obj), obj, support_obj, margin=margin)
    reposition_on_support(obj, center_xy, support_obj, orientation_xyzw=orientation_xyzw)
    return center_xy


def reposition_xy_keep_z(obj, xy: np.ndarray, orientation_xyzw=SQUARE_ORI):
    pos, _ = obj.get_position_orientation()
    z = float(pos.cpu().numpy()[2])
    obj.set_position_orientation(
        position=th.tensor([float(xy[0]), float(xy[1]), z], dtype=th.float32),
        orientation=th.tensor(orientation_xyzw, dtype=th.float32),
    )
    obj.keep_still()
    step_n(10)


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


# -----------------------------------------------------------------------------
# Support selection
# -----------------------------------------------------------------------------
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
        if cat not in VALID_TABLE_SUPPORT_CATEGORIES:
            continue
        try:
            if object_xy_center_inside_floor(obj, floor_obj):
                tables.append(obj)
        except Exception:
            continue
    tables.sort(key=lambda o: o.name)
    return tables


def choose_support(scene, floor_obj, target_obj, occluder_obj, rng: random.Random, placement_mode: str = "sample"):
    tables = find_tables_for_floor(scene, floor_obj)
    fit_tables = [t for t in tables if object_fits_on_support(target_obj, t) and object_fits_on_support(occluder_obj, t)]

    modes = []
    if floor_obj is not None:
        modes.append("floor_center")
    if fit_tables:
        modes.append("table_center")
    if not modes:
        raise RuntimeError("No valid support modes available.")

    if placement_mode == "sample":
        chosen_mode = rng.choice(modes)
    elif placement_mode in modes:
        chosen_mode = placement_mode
    elif placement_mode == "table_center" and not fit_tables:
        print("[support] requested table_center but no fitting room table found; falling back to floor_center")
        chosen_mode = "floor_center"
    else:
        raise RuntimeError(f"Unsupported placement_mode={placement_mode} with modes={modes}")

    if chosen_mode == "floor_center":
        support_obj = floor_obj
        selection_detail = "room_floor_center"
    else:
        support_obj = rng.choice(fit_tables)
        selection_detail = "random_table_center_fit_filtered"

    support_min, support_max = aabb_minmax_np(support_obj)
    return support_obj, {
        "placement_mode": chosen_mode,
        "support_type": "floor" if chosen_mode == "floor_center" else "table",
        "support_name": support_obj.name,
        "support_category": str(getattr(support_obj, "category", "")),
        "support_bbox_min": support_min.tolist(),
        "support_bbox_max": support_max.tolist(),
        "support_selection_policy": selection_detail,
        "num_candidate_tables_in_room": len(tables),
        "num_fit_tables_in_room": len(fit_tables),
        "candidate_table_names": [t.name for t in tables],
        "fit_table_names": [t.name for t in fit_tables],
    }


# -----------------------------------------------------------------------------
# Occluder placement
# -----------------------------------------------------------------------------
def compute_occluder_xy_from_bbox_rule(
    target_obj,
    occluder_obj,
    view_side: str,
    occluder_corner: str,
    front_overlap: float,
    lateral_extra: float,
) -> Dict:
    target_center = get_object_center_np(target_obj)
    target_xy = target_center[:2]
    occ_hx, occ_hy = get_xy_half_extents(occluder_obj)
    tgt_hx, tgt_hy = get_xy_half_extents(target_obj)

    view_side = view_side.upper()
    occluder_corner = occluder_corner.upper()
    if view_side not in {"E", "W", "N", "S"}:
        raise ValueError(f"Unsupported view_side={view_side}")
    if occluder_corner not in {"NW", "NE", "SW", "SE"}:
        raise ValueError(f"Unsupported occluder_corner={occluder_corner}")

    # IMPORTANT:
    # We place the occluder just OUTSIDE the target's XY footprint using BOTH objects' half extents.
    # The previous version ignored the target half extents, which put the target center on the
    # occluder's corner / face and caused physical interpenetration.
    cx = float(target_xy[0])
    cy = float(target_xy[1])

    if view_side == "E":
        if "W" not in occluder_corner:
            raise ValueError("For view_side='E', use NW or SW")
        cx = float(target_xy[0]) + tgt_hx + occ_hx + front_overlap
        cy = float(target_xy[1]) + ((-(tgt_hy + occ_hy)) if "N" in occluder_corner else (tgt_hy + occ_hy)) + lateral_extra
    elif view_side == "W":
        if "E" not in occluder_corner:
            raise ValueError("For view_side='W', use NE or SE")
        cx = float(target_xy[0]) - tgt_hx - occ_hx - front_overlap
        cy = float(target_xy[1]) + ((-(tgt_hy + occ_hy)) if "N" in occluder_corner else (tgt_hy + occ_hy)) + lateral_extra
    elif view_side == "N":
        if "S" not in occluder_corner:
            raise ValueError("For view_side='N', use SW or SE")
        cy = float(target_xy[1]) + tgt_hy + occ_hy + front_overlap
        cx = float(target_xy[0]) + ((-(tgt_hx + occ_hx)) if "E" in occluder_corner else (tgt_hx + occ_hx)) + lateral_extra
    elif view_side == "S":
        if "N" not in occluder_corner:
            raise ValueError("For view_side='S', use NW or NE")
        cy = float(target_xy[1]) - tgt_hy - occ_hy - front_overlap
        cx = float(target_xy[0]) + ((-(tgt_hx + occ_hx)) if "E" in occluder_corner else (tgt_hx + occ_hx)) + lateral_extra

    desired_xy = np.array([cx, cy], dtype=float)
    return {
        "target_center_xy": target_xy.tolist(),
        "target_half_extents_xy": [float(tgt_hx), float(tgt_hy)],
        "occluder_half_extents_xy": [float(occ_hx), float(occ_hy)],
        "view_side": view_side,
        "occluder_corner": occluder_corner,
        "front_overlap": float(front_overlap),
        "lateral_extra": float(lateral_extra),
        "desired_occluder_xy_pre_clamp": desired_xy.tolist(),
    }


def challenge_camera_for_target(target_obj, view_side: str, height: float, radius_pad: float):
    center = get_object_center_np(target_obj)
    radius = get_xy_half_diag(target_obj) + radius_pad
    view_side = view_side.upper()
    if view_side == "E":
        eye = np.array([center[0] + radius, center[1], center[2] + height], dtype=float)
    elif view_side == "W":
        eye = np.array([center[0] - radius, center[1], center[2] + height], dtype=float)
    elif view_side == "N":
        eye = np.array([center[0], center[1] + radius, center[2] + height], dtype=float)
    elif view_side == "S":
        eye = np.array([center[0], center[1] - radius, center[2] + height], dtype=float)
    else:
        raise ValueError(f"Unsupported view_side={view_side}")
    return eye, center.copy()


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def render_ring_views(obj, out_dir: str, obj_names: Tuple[str, ...], num_azimuths: int, azimuth_step_deg: float,
                      heights: Tuple[float, ...], radius_pad: float, topdown_z: float):
    os.makedirs(out_dir, exist_ok=True)
    center = get_object_center_np(obj)
    radius = get_xy_half_diag(obj) + radius_pad

    metadata = {}
    aggregate = {name: 0 for name in obj_names}
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
            for k, v in vis.items():
                aggregate[k] += int(v)
            metadata[fname] = {**pose, "type": "ring", "azimuth_deg": az_deg, "height_offset": h, "visibility": vis}
            view_idx += 1

    top_eye = np.array([center[0], center[1], topdown_z], dtype=float)
    fname = f"view_{view_idx:02d}_topdown.png"
    fpath = os.path.join(out_dir, fname)
    pose = set_camera_and_capture(top_eye, center.copy(), fpath)
    vis = visibility_check(obj_names)
    for k, v in vis.items():
        aggregate[k] += int(v)
    metadata[fname] = {**pose, "type": "topdown", "visibility": vis}
    return metadata, aggregate


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch partial occlusion worker")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--floor", required=True)
    parser.add_argument("--run_idx", type=int, required=True)
    parser.add_argument("--keys_json", type=str, default="keys.json")
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--output_root", type=str, default="renders_occlusion")

    parser.add_argument("--target_category", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--occluder_category", type=str, default=None)
    parser.add_argument("--occluder_model", type=str, default=None)
    parser.add_argument("--asset_manifest", type=str, default=DEFAULT_ASSET_MANIFEST)

    parser.add_argument("--placement_mode", choices=["sample", "floor_center", "table_center"], default="sample")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--view_side", choices=["sample", "E", "W", "N", "S"], default="sample")
    parser.add_argument("--occluder_corner", choices=["sample", "NW", "NE", "SW", "SE"], default="sample")
    parser.add_argument("--front_overlap", type=float, default=DEFAULT_FRONT_OVERLAP)
    parser.add_argument("--lateral_extra", type=float, default=DEFAULT_LATERAL_EXTRA)
    parser.add_argument("--challenge_height", type=float, default=DEFAULT_CHALLENGE_HEIGHT)
    parser.add_argument("--challenge_radius_pad", type=float, default=DEFAULT_CHALLENGE_RADIUS_PAD)

    parser.add_argument("--num_azimuths", type=int, default=DEFAULT_NUM_AZIMUTHS)
    parser.add_argument("--azimuth_step_deg", type=float, default=DEFAULT_AZIMUTH_STEP_DEG)
    parser.add_argument("--radius_pad", type=float, default=DEFAULT_RADIUS_PAD)
    parser.add_argument("--topdown_z", type=float, default=DEFAULT_TOPDOWN_Z)
    args = parser.parse_args()

    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else (abs(hash((args.scene, args.room, args.run_idx, "occlusion"))) % (2**31))
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    asset_choice = resolve_occlusion_assets(args, rng)
    resolved_view_side, resolved_occluder_corner = sample_view_and_corner(rng, args.view_side, args.occluder_corner)
    print(f"[asset] mode={asset_choice['selection_mode']} target={asset_choice['target']['object_id']} occluder={asset_choice['occluder']['object_id']}")
    print(f"[occlusion_rule] view_side={resolved_view_side} occluder_corner={resolved_occluder_corner}")
    config = build_config(
        scene_model=args.scene,
        robot_type=args.robot,
        target_category=asset_choice["target"]["category"],
        target_model=asset_choice["target"]["model"],
        occluder_category=asset_choice["occluder"]["category"],
        occluder_model=asset_choice["occluder"]["model"],
    )

    env = og.Environment(configs=config)
    exit_code = 2
    metadata = None

    try:
        scene = env.scene
        target_obj = scene.object_registry("name", "target_obj")

        # Initially place the occluder somewhere far away so it doesn't interfere with support selection
        occluder_obj = scene.object_registry("name", "occluder_obj")
        occluder_obj.set_position_orientation(
            position=th.tensor([300.0, 300.0, 300.0], dtype=th.float32),
            orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        occluder_obj.keep_still()
        step_n(2)

        floor_obj = scene.object_registry("name", args.floor)
        if floor_obj is None:
            raise RuntimeError(f"Could not resolve floor object: {args.floor}")

        for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
            og.sim._viewer_camera.add_modality(modality)
        step_n(50)

        support_obj, support_meta = choose_support(scene, floor_obj, target_obj, occluder_obj, rng, placement_mode=args.placement_mode)
        print(f"[support] mode={support_meta['placement_mode']} support={support_obj.name} ({support_meta['support_category']})")

        if support_meta["support_type"] == "floor":
            target_center_xy = place_on_top_centered_with_orientation(target_obj, support_obj)
        else:
            ok = target_obj.states[object_states.OnTop].set_value(support_obj, True)
            if not ok:
                raise RuntimeError(f"Failed to place {target_obj.name} on top of {support_obj.name}")
            step_n(15)
            target_center_xy = get_object_center_np(target_obj)[:2]

        # Put occluder on support only long enough to get a valid supported Z
        ok = occluder_obj.states[object_states.OnTop].set_value(support_obj, True)
        if not ok:
            raise RuntimeError(f"Failed to place {occluder_obj.name} on top of {support_obj.name}")

        # Step as little as possible
        step_n(2)

        placement_info = compute_occluder_xy_from_bbox_rule(target_obj, occluder_obj, view_side=resolved_view_side, occluder_corner=resolved_occluder_corner, front_overlap=args.front_overlap, lateral_extra=args.lateral_extra)
        desired_xy = np.array(placement_info["desired_occluder_xy_pre_clamp"], dtype=float)
        clamped_xy = clamp_xy_inside_support(desired_xy, occluder_obj, support_obj)

        # Move to final XY while recomputing Z from the support top instead of reusing
        # the sampled Z at a different XY, which can cause penetration / physics explosions.
        reposition_on_support(occluder_obj, clamped_xy, support_obj, orientation_xyzw=SQUARE_ORI)

        placement_info["target_centered_xy"] = target_center_xy.tolist()
        placement_info["desired_occluder_xy_post_clamp"] = clamped_xy.tolist()
        placement_info["occluder_xy_was_clamped"] = bool(np.linalg.norm(clamped_xy - desired_xy) > 1e-6)

        challenge_eye, challenge_target = challenge_camera_for_target(
            target_obj,
            view_side=resolved_view_side,
            height=args.challenge_height,
            radius_pad=args.challenge_radius_pad,
        )
        challenge_path = os.path.join(run_dir, "view_challenge.png")
        challenge_pose = set_camera_and_capture(challenge_eye, challenge_target, challenge_path)
        challenge_vis = visibility_check((target_obj.name, occluder_obj.name))

        views_meta, agg_vis = render_ring_views(
            obj=target_obj,
            out_dir=run_dir,
            obj_names=(target_obj.name, occluder_obj.name),
            num_azimuths=args.num_azimuths,
            azimuth_step_deg=args.azimuth_step_deg,
            heights=tuple(DEFAULT_HEIGHTS),
            radius_pad=args.radius_pad,
            topdown_z=args.topdown_z,
        )

        target_live = live_object_info(target_obj)
        occluder_live = live_object_info(occluder_obj)
        support_live = live_object_info(support_obj)

        metadata = {
            "scene": args.scene,
            "room": args.room,
            "run_idx": args.run_idx,
            "seed": seed,
            "floor_name": args.floor,
            "layout": "partial_occlusion_bbox_batch",
            "task_type": "partial_occlusion",
            "answer": "target_partially_occluded",
            **support_meta,
            "objects": {
                "target": {
                    **target_live,
                    "model": asset_choice["target"]["model"],
                    "requested_category": asset_choice["target"]["category"],
                    "object_id": asset_choice["target"]["object_id"],
                },
                "occluder": {
                    **occluder_live,
                    "model": asset_choice["occluder"]["model"],
                    "requested_category": asset_choice["occluder"]["category"],
                    "object_id": asset_choice["occluder"]["object_id"],
},
                "support": support_live,
            },
            "bbox_rule": placement_info,
            "challenge_view": {
                **challenge_pose,
                "view_side": resolved_view_side,
                "occluder_corner": resolved_occluder_corner,
                "challenge_height": float(args.challenge_height),
                "challenge_radius_pad": float(args.challenge_radius_pad),
                "visibility": challenge_vis,
            },
            "camera_layout": {
                "num_azimuths": args.num_azimuths,
                "azimuth_step_deg": args.azimuth_step_deg,
                "heights": list(DEFAULT_HEIGHTS),
                "radius_pad": args.radius_pad,
                "topdown_z": args.topdown_z,
            },
            "visibility_summary": {
                "challenge_target_visible": bool(challenge_vis[target_obj.name]),
                "challenge_occluder_visible": bool(challenge_vis[occluder_obj.name]),
                "ring_target_visible_count": int(agg_vis[target_obj.name]),
                "ring_occluder_visible_count": int(agg_vis[occluder_obj.name]),
            },
            "views": views_meta,
        }

        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[done] wrote metadata -> {metadata_path}")

        success = (
            bool(challenge_vis[target_obj.name]) and
            bool(challenge_vis[occluder_obj.name]) and
            agg_vis[target_obj.name] >= 2
        )
        exit_code = 0 if success else 1
        print("[success] batch occlusion run complete" if success else "[partial] run completed but visibility criteria were not met")

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
