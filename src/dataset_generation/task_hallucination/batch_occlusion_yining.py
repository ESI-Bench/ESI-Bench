#!/usr/bin/env python3
"""
batch_occlusion.py

Single-run batch worker for partial-occlusion examples.

Camera layout:
  - View 0 (canonical hard view): camera is laterally offset 15 degrees from
    the perfectly collinear position (occluded -> occluder -> camera), at the
    same height as the target center, looking horizontally at the target.
  - Views 1-5: orbit every 60 degrees from view 0, same height, horizontal gaze.
  - View 6: top-down view.

Occluder sizing:
  The occluder is loaded and scaled so that each of its bbox axes equals
  the target's corresponding bbox axis * occluder_scale_factor (default 1.1).
  Scale is computed from object_inventory.json at runtime.

Flat exist flags written to metadata root:
  exist_target_obj_0 ... exist_target_obj_6
  exist_occluder_obj_0 ... exist_occluder_obj_6

Exit codes:
  0 = success
  1 = partial failure (renders completed but visibility criteria not met)
  2 = hard setup failure
"""

import argparse
import csv
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
DEFAULT_CATEGORY_CSV = "category_confusables.csv"

DEFAULT_NUM_AZIMUTHS = 6
DEFAULT_AZIMUTH_STEP_DEG = 60.0
DEFAULT_RADIUS_PAD = 0.55
DEFAULT_TOPDOWN_Z = 1.8
DEFAULT_LATERAL_OFFSET_DEG = 15.0
DEFAULT_OCCLUDER_SCALE_FACTOR = 1.1

DEFAULT_VIEW_SIDE = "W"
DEFAULT_OCCLUDER_CORNER = "NE"
DEFAULT_FRONT_OVERLAP = 0.0
DEFAULT_LATERAL_EXTRA = 0.0

SUPPORT_MARGIN = 0.02

INVENTORY_PATHS = [
    "bddl/bddl/generated_data/object_inventory.json",
    "bddl3/bddl/generated_data/object_inventory.json",
    os.path.join(os.path.dirname(__file__), "object_inventory.json"),
]

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
# Inventory helpers
# -----------------------------------------------------------------------------

def load_inventory(path: Optional[str] = None) -> Optional[dict]:
    search = ([path] if path else []) + INVENTORY_PATHS
    for p in search:
        if p and os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            print(f"[inventory] loaded from {p}")
            return data
    print("[inventory] WARNING: object_inventory.json not found — occluder will use scale [1,1,1]")
    return None


def get_bbox_from_inventory(inventory: Optional[dict], model_id: str) -> Optional[List[float]]:
    if inventory is None:
        return None
    bbox = inventory.get("bounding_box_sizes", {}).get(model_id)
    if bbox and len(bbox) >= 3:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2])]
    return None


def compute_occluder_scale(
    target_model: str,
    occluder_model: str,
    inventory: Optional[dict],
    scale_factor: float = 1.1,
) -> List[float]:
    """
    Compute per-axis scale for the occluder so that:
        occluder_rendered_bbox[i] = target_bbox[i] * scale_factor

    scale[i] = (target_bbox[i] * scale_factor) / occluder_inventory_bbox[i]

    Falls back to [1, 1, 1] if inventory or bbox is missing.
    """
    target_bbox = get_bbox_from_inventory(inventory, target_model)
    occluder_bbox = get_bbox_from_inventory(inventory, occluder_model)

    if target_bbox is None or occluder_bbox is None:
        print(f"[scale] missing bbox for target={target_model} or occluder={occluder_model} — using [1,1,1]")
        return [1.0, 1.0, 1.0]

    scale = [
        (target_bbox[i] * scale_factor) / occluder_bbox[i] if occluder_bbox[i] > 1e-6 else 1.0
        for i in range(3)
    ]
    print(f"[scale] target_bbox  = {[round(v, 4) for v in target_bbox]}")
    print(f"[scale] occluder_bbox = {[round(v, 4) for v in occluder_bbox]}")
    print(f"[scale] occluder_scale= {[round(v, 4) for v in scale]} (factor={scale_factor})")
    return scale


# -----------------------------------------------------------------------------
# Confusable map
# -----------------------------------------------------------------------------

def load_confusable_map(csv_path: str) -> Dict[str, Dict]:
    if not os.path.exists(csv_path):
        print(f"[confusable] CSV not found at {csv_path} — skipping")
        return {}

    def _norm(s):
        return str(s or "").strip().lower().replace(" ", "_")

    result = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = _norm(row.get("Category or Particle System", ""))
            if not cat:
                continue
            raw = row.get("confusable_with", "") or ""
            confusables = [_norm(x) for x in raw.split(",") if _norm(x) and _norm(x) != cat]
            seen, deduped = set(), []
            for c in confusables:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            result[cat] = {
                "confusable_with": deduped,
                "confusion_reason": (row.get("confusion_reason") or "").strip(),
                "reason": (row.get("reason") or "").strip(),
                "view_ambiguous": (row.get("view_ambiguous") or "").strip(),
                "occlusion_sensitive": (row.get("occlusion_sensitive") or "").strip(),
            }
    print(f"[confusable] loaded {len(result)} entries from {csv_path}")
    return result


# -----------------------------------------------------------------------------
# Asset manifest / config
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
    chosen_view = rng.choice(sorted(compat.keys())) if view_side == "sample" else view_side
    if chosen_view not in compat:
        raise ValueError(f"Unsupported view_side={chosen_view}")
    chosen_corner = rng.choice(compat[chosen_view]) if occluder_corner == "sample" else occluder_corner
    if chosen_corner not in compat[chosen_view]:
        raise ValueError(f"Incompatible pair: view_side={chosen_view}, occluder_corner={chosen_corner}")
    return chosen_view, chosen_corner


def resolve_occlusion_assets(args, rng: random.Random):
    manual = all(
        v is not None
        for v in [args.target_category, args.target_model, args.occluder_category, args.occluder_model]
    )
    use_manifest = args.asset_manifest is not None and os.path.exists(args.asset_manifest)

    if manual:
        return {
            "target": {
                "category": args.target_category,
                "model": args.target_model,
                "object_id": f"{args.target_category}-{args.target_model}",
            },
            "occluder": {
                "category": args.occluder_category,
                "model": args.occluder_model,
                "object_id": f"{args.occluder_category}-{args.occluder_model}",
            },
            "selection_mode": "manual",
            "asset_manifest": args.asset_manifest,
        }

    if not use_manifest:
        return {
            "target": {
                "category": DEFAULT_TARGET_CATEGORY,
                "model": DEFAULT_TARGET_MODEL,
                "object_id": f"{DEFAULT_TARGET_CATEGORY}-{DEFAULT_TARGET_MODEL}",
            },
            "occluder": {
                "category": DEFAULT_OCCLUDER_CATEGORY,
                "model": DEFAULT_OCCLUDER_MODEL,
                "object_id": f"{DEFAULT_OCCLUDER_CATEGORY}-{DEFAULT_OCCLUDER_MODEL}",
            },
            "selection_mode": "hardcoded_default",
            "asset_manifest": None,
        }

    manifest = load_asset_manifest(args.asset_manifest)
    targets = manifest.get("occlusion_target_candidates", [])
    compatible = manifest.get("occlusion_compatible_occluders", {})
    if not targets or not compatible:
        raise RuntimeError(f"Manifest missing occlusion candidate pools: {args.asset_manifest}")
    # 1. random category, 2. random model within that category, 3. random occluder
    all_cats = sorted({t["category"] for t in targets})
    chosen_cat = rng.choice(all_cats)
    cat_targets = [t for t in targets if t["category"] == chosen_cat]
    target = rng.choice(cat_targets)
    occ_ids = compatible.get(target["object_id"], [])
    if not occ_ids:
        raise RuntimeError(f"No compatible occluders for target {target['object_id']}")
    all_assets = {a["object_id"]: a for a in manifest.get("occlusion_occluder_candidates", [])}
    occ_choices = [all_assets[oid] for oid in occ_ids if oid in all_assets]
    if not occ_choices:
        raise RuntimeError(
            f"Compatible occluders not found in occlusion_occluder_candidates for {target['object_id']}"
        )
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
    room: str,
    robot_type: str,
    target_category: str,
    target_model: str,
    occluder_category: str,
    occluder_model: str,
    occluder_scale: List[float],
) -> Dict:
    config_filename = os.path.join(og.example_config_path, f"{robot_type.lower()}_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"]["scene_model"] = scene_model
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"] = [room]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "target_obj",
            "category": target_category,
            "model": target_model,
            "position": [150.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "type": "DatasetObject",
            "name": "occluder_obj",
            "category": occluder_category,
            "model": occluder_model,
            "position": [160.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale": occluder_scale,
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
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def get_support_center_xy(support_obj) -> np.ndarray:
    bb_min, bb_max = aabb_minmax_np(support_obj)
    return np.array([
        0.5 * (float(bb_min[0]) + float(bb_max[0])),
        0.5 * (float(bb_min[1]) + float(bb_max[1])),
    ], dtype=float)


def clamp_xy_inside_support(
    desired_xy: np.ndarray, obj, support_obj, margin: float = SUPPORT_MARGIN
) -> np.ndarray:
    support_min, support_max = aabb_minmax_np(support_obj)
    obj_hx, obj_hy = get_xy_half_extents(obj)
    x = min(
        max(float(desired_xy[0]), float(support_min[0]) + obj_hx + margin),
        float(support_max[0]) - obj_hx - margin,
    )
    y = min(
        max(float(desired_xy[1]), float(support_min[1]) + obj_hy + margin),
        float(support_max[1]) - obj_hy - margin,
    )
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
    _, support_max = aabb_minmax_np(support_obj)
    bottom_clearance = float(pos.cpu().numpy()[2]) - float(bmin[2])
    return float(support_max[2]) + bottom_clearance + z_epsilon


def reposition_on_support(
    obj, xy: np.ndarray, support_obj, orientation_xyzw=SQUARE_ORI, z_epsilon: float = 0.005
):
    z = get_supported_pose_z_for_xy(obj, support_obj, z_epsilon=z_epsilon)
    obj.set_position_orientation(
        position=th.tensor([float(xy[0]), float(xy[1]), z], dtype=th.float32),
        orientation=th.tensor(orientation_xyzw, dtype=th.float32),
    )
    obj.keep_still()
    step_n(10)


def place_on_top_centered_with_orientation(
    obj, support_obj, orientation_xyzw=SQUARE_ORI, margin: float = SUPPORT_MARGIN
):
    ok = obj.states[object_states.OnTop].set_value(support_obj, True)
    if not ok:
        raise RuntimeError(f"Failed to place {obj.name} on top of {support_obj.name}")
    step_n(15)
    center_xy = clamp_xy_inside_support(
        get_support_center_xy(support_obj), obj, support_obj, margin=margin
    )
    reposition_on_support(obj, center_xy, support_obj, orientation_xyzw=orientation_xyzw)
    return center_xy


def capture_rgb(path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] saved -> {path}")


def set_camera_pose(pos: np.ndarray, quat: np.ndarray):
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(pos, dtype=th.float32),
        th.tensor(quat, dtype=th.float32),
    )
    for _ in range(10):
        og.sim.render()


def visibility_check(actual_names: Tuple[str, ...]) -> Dict[str, bool]:
    for _ in range(60):
        og.sim.render()
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
        float(floor_min[0]) - slack <= float(obj_center[0]) <= float(floor_max[0]) + slack
        and float(floor_min[1]) - slack <= float(obj_center[1]) <= float(floor_max[1]) + slack
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


def choose_support(
    scene, floor_obj, target_obj, occluder_obj, rng: random.Random, placement_mode: str = "sample"
):
    tables = find_tables_for_floor(scene, floor_obj)
    fit_tables = [
        t for t in tables
        if object_fits_on_support(target_obj, t) and object_fits_on_support(occluder_obj, t)
    ]

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
        print("[support] requested table_center but no fitting table found; falling back to floor_center")
        chosen_mode = "floor_center"
    else:
        raise RuntimeError(f"Unsupported placement_mode={placement_mode} with modes={modes}")

    support_obj = floor_obj if chosen_mode == "floor_center" else rng.choice(fit_tables)
    selection_detail = "room_floor_center" if chosen_mode == "floor_center" else "random_table_center_fit_filtered"

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
    cx, cy = float(target_xy[0]), float(target_xy[1])

    if view_side == "E":
        cx = float(target_xy[0]) + tgt_hx + occ_hx + front_overlap
        cy = float(target_xy[1]) + (
            -(tgt_hy + occ_hy) if "N" in occluder_corner else (tgt_hy + occ_hy)
        ) + lateral_extra
    elif view_side == "W":
        cx = float(target_xy[0]) - tgt_hx - occ_hx - front_overlap
        cy = float(target_xy[1]) + (
            -(tgt_hy + occ_hy) if "N" in occluder_corner else (tgt_hy + occ_hy)
        ) + lateral_extra
    elif view_side == "N":
        cy = float(target_xy[1]) + tgt_hy + occ_hy + front_overlap
        cx = float(target_xy[0]) + (
            -(tgt_hx + occ_hx) if "E" in occluder_corner else (tgt_hx + occ_hx)
        ) + lateral_extra
    elif view_side == "S":
        cy = float(target_xy[1]) - tgt_hy - occ_hy - front_overlap
        cx = float(target_xy[0]) + (
            -(tgt_hx + occ_hx) if "E" in occluder_corner else (tgt_hx + occ_hx)
        ) + lateral_extra

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


# -----------------------------------------------------------------------------
# Camera helpers
# -----------------------------------------------------------------------------

def compute_view0_azimuth(target_obj, occluder_obj) -> float:
    target_xy = get_object_center_np(target_obj)[:2]
    occluder_xy = get_object_center_np(occluder_obj)[:2]
    direction = occluder_xy - target_xy
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
    return float(np.arctan2(direction[1], direction[0]))


def make_horizontal_quaternion(azimuth_rad: float) -> np.ndarray:
    forward = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()


def render_ring_views(
    target_obj,
    occluder_obj,
    out_dir: str,
    obj_names: Tuple[str, ...],
    num_azimuths: int,
    azimuth_step_deg: float,
    radius_pad: float,
    topdown_z: float,
    lateral_offset_deg: float,
) -> Tuple[Dict, Dict, Dict]:
    os.makedirs(out_dir, exist_ok=True)

    target_center = get_object_center_np(target_obj)
    radius = get_xy_half_diag(target_obj) + radius_pad
    collinear_az_rad = compute_view0_azimuth(target_obj, occluder_obj)
    az0_rad = collinear_az_rad + math.radians(lateral_offset_deg)
    cam_z = float(target_center[2])

    metadata: Dict = {}
    aggregate: Dict = {name: 0 for name in obj_names}
    exist_flags: Dict = {}

    for az_step in range(num_azimuths):
        az_rad = az0_rad + math.radians(az_step * azimuth_step_deg)
        az_deg = math.degrees(az_rad) % 360.0

        eye = np.array([
            float(target_center[0]) + radius * math.cos(az_rad),
            float(target_center[1]) + radius * math.sin(az_rad),
            cam_z,
        ], dtype=float)
        quat = make_horizontal_quaternion(az_rad + math.pi)

        fname = f"view_{az_step:02d}_az{int(round(az_deg)):03d}.png"
        set_camera_pose(eye, quat)
        capture_rgb(os.path.join(out_dir, fname))
        vis = visibility_check(obj_names)

        for k, v in vis.items():
            aggregate[k] += int(v)
            exist_flags[f"exist_{k}_{az_step}"] = bool(v)

        metadata[fname] = {
            "position": eye.tolist(),
            "quaternion_xyzw": quat.tolist(),
            "type": "ring_lateral_hard" if az_step == 0 else "ring",
            "azimuth_deg": az_deg,
            "collinear_az_deg": math.degrees(collinear_az_rad) % 360.0,
            "lateral_offset_deg": lateral_offset_deg if az_step == 0 else 0.0,
            "visibility": vis,
        }

    # Top-down view
    top_idx = num_azimuths
    top_eye = np.array([float(target_center[0]), float(target_center[1]), topdown_z], dtype=float)
    forward_td = np.array([0.0, 0.0, -1.0])
    up_td = np.array([1.0, 0.0, 0.0])
    right_td = np.cross(forward_td, up_td)
    right_td /= np.linalg.norm(right_td)
    true_up_td = np.cross(right_td, forward_td)
    true_up_td /= np.linalg.norm(true_up_td)
    top_quat = Rotation.from_matrix(np.column_stack([right_td, true_up_td, -forward_td])).as_quat()

    fname = f"view_{top_idx:02d}_topdown.png"
    set_camera_pose(top_eye, top_quat)
    capture_rgb(os.path.join(out_dir, fname))
    vis = visibility_check(obj_names)
    for k, v in vis.items():
        aggregate[k] += int(v)
        exist_flags[f"exist_{k}_{top_idx}"] = bool(v)
    metadata[fname] = {
        "position": top_eye.tolist(),
        "quaternion_xyzw": top_quat.tolist(),
        "type": "topdown",
        "visibility": vis,
    }

    return metadata, aggregate, exist_flags


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
    parser.add_argument("--output_root", type=str, default="renders_occlusion_v2")
    parser.add_argument("--target_category", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--occluder_category", type=str, default=None)
    parser.add_argument("--occluder_model", type=str, default=None)
    parser.add_argument("--asset_manifest", type=str, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--category_csv", type=str, default=DEFAULT_CATEGORY_CSV)
    parser.add_argument("--inventory", type=str, default=None,
                        help="Path to object_inventory.json (auto-discovered if not set)")
    parser.add_argument("--occluder_scale_factor", type=float, default=DEFAULT_OCCLUDER_SCALE_FACTOR,
                        help="Occluder rendered bbox = target bbox * this factor on each axis (default 1.1)")
    parser.add_argument("--placement_mode", choices=["sample", "floor_center", "table_center"], default="sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--view_side", choices=["sample", "E", "W", "N", "S"], default="sample")
    parser.add_argument("--occluder_corner", choices=["sample", "NW", "NE", "SW", "SE"], default="sample")
    parser.add_argument("--front_overlap", type=float, default=DEFAULT_FRONT_OVERLAP)
    parser.add_argument("--lateral_extra", type=float, default=DEFAULT_LATERAL_EXTRA)
    parser.add_argument("--num_azimuths", type=int, default=DEFAULT_NUM_AZIMUTHS)
    parser.add_argument("--azimuth_step_deg", type=float, default=DEFAULT_AZIMUTH_STEP_DEG)
    parser.add_argument("--radius_pad", type=float, default=DEFAULT_RADIUS_PAD)
    parser.add_argument("--topdown_z", type=float, default=DEFAULT_TOPDOWN_Z)
    parser.add_argument("--lateral_offset_deg", type=float, default=DEFAULT_LATERAL_OFFSET_DEG)
    args = parser.parse_args()

    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else (
        abs(hash((args.scene, args.room, args.run_idx, "occlusion"))) % (2 ** 31)
    )
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 32 - 1))

    # Load inventory for occluder scale computation
    inventory = load_inventory(args.inventory)
    confusable_map = load_confusable_map(args.category_csv)

    asset_choice = resolve_occlusion_assets(args, rng)
    resolved_view_side, resolved_occluder_corner = sample_view_and_corner(
        rng, args.view_side, args.occluder_corner
    )
    print(f"[asset] mode={asset_choice['selection_mode']} "
          f"target={asset_choice['target']['object_id']} "
          f"occluder={asset_choice['occluder']['object_id']}")
    print(f"[occlusion_rule] view_side={resolved_view_side} occluder_corner={resolved_occluder_corner}")

    # Compute occluder scale so it renders at target_bbox * scale_factor on each axis
    occluder_scale = compute_occluder_scale(
        target_model=asset_choice["target"]["model"],
        occluder_model=asset_choice["occluder"]["model"],
        inventory=inventory,
        scale_factor=args.occluder_scale_factor,
    )

    config = build_config(
        scene_model=args.scene,
        room=args.room,
        robot_type=args.robot,
        target_category=asset_choice["target"]["category"],
        target_model=asset_choice["target"]["model"],
        occluder_category=asset_choice["occluder"]["category"],
        occluder_model=asset_choice["occluder"]["model"],
        occluder_scale=occluder_scale,
    )

    env = og.Environment(configs=config)
    exit_code = 2

    try:
        scene = env.scene
        target_obj = scene.object_registry("name", "target_obj")
        occluder_obj = scene.object_registry("name", "occluder_obj")

        # Park occluder far away during support selection
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

        support_obj, support_meta = choose_support(
            scene, floor_obj, target_obj, occluder_obj, rng, placement_mode=args.placement_mode
        )
        print(f"[support] mode={support_meta['placement_mode']} "
              f"support={support_obj.name} ({support_meta['support_category']})")

        if support_meta["support_type"] == "floor":
            target_center_xy = place_on_top_centered_with_orientation(target_obj, support_obj)
        else:
            ok = target_obj.states[object_states.OnTop].set_value(support_obj, True)
            if not ok:
                raise RuntimeError(f"Failed to place {target_obj.name} on top of {support_obj.name}")
            step_n(15)
            target_center_xy = get_object_center_np(target_obj)[:2]

        ok = occluder_obj.states[object_states.OnTop].set_value(support_obj, True)
        if not ok:
            raise RuntimeError(f"Failed to place {occluder_obj.name} on top of {support_obj.name}")
        step_n(2)

        placement_info = compute_occluder_xy_from_bbox_rule(
            target_obj, occluder_obj,
            view_side=resolved_view_side,
            occluder_corner=resolved_occluder_corner,
            front_overlap=args.front_overlap,
            lateral_extra=args.lateral_extra,
        )
        desired_xy = np.array(placement_info["desired_occluder_xy_pre_clamp"], dtype=float)
        clamped_xy = clamp_xy_inside_support(desired_xy, occluder_obj, support_obj)
        reposition_on_support(occluder_obj, clamped_xy, support_obj, orientation_xyzw=SQUARE_ORI)

        placement_info["target_centered_xy"] = target_center_xy.tolist()
        placement_info["desired_occluder_xy_post_clamp"] = clamped_xy.tolist()
        placement_info["occluder_xy_was_clamped"] = bool(np.linalg.norm(clamped_xy - desired_xy) > 1e-6)

        views_meta, agg_vis, exist_flags = render_ring_views(
            target_obj=target_obj,
            occluder_obj=occluder_obj,
            out_dir=run_dir,
            obj_names=(target_obj.name, occluder_obj.name),
            num_azimuths=args.num_azimuths,
            azimuth_step_deg=args.azimuth_step_deg,
            radius_pad=args.radius_pad,
            topdown_z=args.topdown_z,
            lateral_offset_deg=args.lateral_offset_deg,
        )

        target_live = live_object_info(target_obj)
        occluder_live = live_object_info(occluder_obj)
        support_live = live_object_info(support_obj)

        hard_view_name = list(views_meta.keys())[0]
        hard_view_vis = views_meta[hard_view_name]["visibility"]

        target_cat_norm = str(asset_choice["target"]["category"]).strip().lower().replace(" ", "_")
        confusable_info = confusable_map.get(target_cat_norm, {})

        metadata = {
            "scene": args.scene,
            "room": args.room,
            "run_idx": args.run_idx,
            "seed": seed,
            "floor_name": args.floor,
            "layout": "partial_occlusion_bbox_batch",
            "task_type": "partial_occlusion",
            "answer": "target_partially_occluded",
            "hard_view": hard_view_name,
            "lateral_offset_deg": args.lateral_offset_deg,
            "occluder_scale_factor": args.occluder_scale_factor,
            "occluder_scale": occluder_scale,
            "target_confusable_with": confusable_info.get("confusable_with", []),
            "target_confusion_reason": confusable_info.get("confusion_reason", ""),
            "target_occlusion_reason": confusable_info.get("reason", ""),
            "target_view_ambiguous": confusable_info.get("view_ambiguous", ""),
            "target_occlusion_sensitive": confusable_info.get("occlusion_sensitive", ""),
            **support_meta,
            **exist_flags,
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
                    "applied_scale": occluder_scale,
                },
                "support": support_live,
            },
            "bbox_rule": placement_info,
            "camera_layout": {
                "num_azimuths": args.num_azimuths,
                "azimuth_step_deg": args.azimuth_step_deg,
                "lateral_offset_deg": args.lateral_offset_deg,
                "radius_pad": args.radius_pad,
                "topdown_z": args.topdown_z,
                "view_0": "hard view: collinear azimuth + lateral_offset_deg, horizontal gaze",
                "view_k": "view_0 azimuth + k * azimuth_step_deg, horizontal gaze",
            },
            "visibility_summary": {
                "hard_view_target_visible": bool(hard_view_vis.get(target_obj.name, False)),
                "hard_view_occluder_visible": bool(hard_view_vis.get(occluder_obj.name, False)),
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
            bool(hard_view_vis.get(target_obj.name, False))
            and bool(hard_view_vis.get(occluder_obj.name, False))
            and agg_vis[target_obj.name] >= 2
        )
        exit_code = 0 if success else 1
        print(
            "[success] batch occlusion run complete"
            if success
            else "[partial] run completed but visibility criteria were not met"
        )

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