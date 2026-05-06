"""
batch_containment_positive.py

"Positive" script: the small object is placed BEHIND the container (not inside).
The scene is a collinear arrangement:

    camera_0  ──── STANDOFF ────  container  ──── GAP ────  small_obj

All three share the same X coordinate.  From camera_0 the small object is
partially occluded by the container — the viewer cannot trivially tell whether
the object is inside or outside.

Camera layout (6 views, all at surface_top_z height, all looking at container centre):
  view 0 — front  (camera_0): the base axis, behind container in -Y direction
  view 1 — +60°  from view 0
  view 2 — +120°
  view 3 — +180° (back, directly behind small object looking through container)
  view 4 — +240°
  view 5 — +300°

Placement of small object:
  1. Drop small object on surface to get its settled Z and live half-extents.
  2. Compute target Y = container_far_face + GAP + small_half_y
     (container_far_face = container bbox max Y, i.e. the face away from camera_0).
  3. Check that the target XY is not occupied by any scene object bbox.
     If occupied, try +X and -X offsets up to MAX_LATERAL_TRIES times.
     If all blocked, fall back to placing directly at container_far_y (no gap).
  4. Teleport small object to chosen XY, keep_still, settle physics.

Saves metadata JSON matching the batch_containment.py format with:
  - exist_obj_container_k, exist_obj_small_k  for k in 0..5
  - camera_poses, object metadata, placement info
  - placement_behind = true  (ground truth: small obj is NOT inside container)

Exit codes:
  0 — success
  1 — partial (placed but not visible)
  2 — fatal error
"""

import os
import json
import yaml
import argparse
import random
import numpy as np
import torch as th
import cv2

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from scipy.spatial.transform import Rotation

gm.ENABLE_FLATCACHE      = False
gm.USE_GPU_DYNAMICS      = False
gm.ENABLE_OBJECT_STATES  = True
gm.ENABLE_TRANSITION_RULES = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCENES_DIR         = "scenes5"
CONTAINERS_JSON    = os.path.join(os.path.dirname(__file__), "containers.json")
SMALL_OBJECTS_JSON = os.path.join(os.path.dirname(__file__), "small_objects.json")
INVENTORY_PATHS    = [
    "bddl3/bddl/generated_data/object_inventory.json",
    os.path.join(os.path.dirname(__file__), "object_inventory.json"),
]

# ── Scale parameters ──────────────────────────────────────────────────────────
SMALL_OBJ_SCALE_MIN = 0.05
SMALL_OBJ_SCALE_MAX = 0.075
SMALL_OBJ_MAX_FRAC  = 0.5    # target <= this fraction of container min(dx, dy)

# ── Camera / placement ────────────────────────────────────────────────────────
CAM_STANDOFF      = 0.50     # metres from container centre to camera_0 (front)
CAM_RADIUS_PAD    = 0.25     # extra beyond container xy half-diagonal for orbit radius
NUM_CAMERAS       = 6
AZIMUTH_STEP_DEG  = 60.0     # 360 / 6
BEHIND_GAP        = 0.02     # metres between container far face and small obj near face
MAX_LATERAL_TRIES = 8        # how many ±X offsets to try if directly behind is blocked
LATERAL_STEP      = 0.05     # metres per lateral offset step

SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Inventory helpers  (identical to batch_containment.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_inventory() -> dict:
    for path in INVENTORY_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise RuntimeError("No object_inventory.json found.")


def get_model_for_category(category: str, seed: int, inventory: dict):
    providers = inventory.get("providers", inventory)
    matches = [k for k in providers if k.startswith(f"{category}-")]
    if not matches:
        return None
    rng = random.Random(seed)
    chosen = rng.choice(matches)
    model_id = chosen.split("-", 1)[1]
    print(f"  [{category}] {len(matches)} model(s), picked model_id={model_id}")
    return model_id


def get_bbox_size(model_id: str, inventory: dict):
    size = inventory.get("bounding_box_sizes", {}).get(model_id)
    if size is None:
        return None
    return np.array(size, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Scene-graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scene_dict(scene_name: str) -> dict:
    path = os.path.join(SCENES_DIR, f"{scene_name}_scene_dict.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene dict not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_room_objects(scene_dict: dict, room: str) -> dict:
    if room not in scene_dict:
        raise KeyError(f"Room '{room}' not in scene dict.")
    return scene_dict[room]


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers  (identical to reference scripts)
# ─────────────────────────────────────────────────────────────────────────────

def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos, dtype=float) - np.array(eye_pos, dtype=float)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        return np.array([0, 0, 0, 1], dtype=float)
    forward /= norm
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()


def _capture(path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] Saved -> {path}")


def _set_camera_and_capture(eye: np.ndarray, look_target: np.ndarray, path: str) -> dict:
    quat = look_at_quaternion(eye, look_target)
    og.sim._viewer_camera.set_position_orientation(eye, quat)
    _capture(path)
    return {"position": eye.tolist(), "quaternion_xyzw": quat.tolist()}


def _visibility_check(obj_names: tuple) -> dict:
    for _ in range(100):
        og.sim.step()
    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    visible_str  = " ".join(id_to_labels.values())
    result = {name: (name in visible_str) for name in obj_names}
    print(f"[seg] {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Surface selection: any object whose name contains "table", else floor
# ─────────────────────────────────────────────────────────────────────────────

def pick_surface(scene, floor_obj, rng: random.Random):
    tables = []
    for obj in scene.objects:
        if "table" in (getattr(obj, "name", "") or "").lower():
            try:
                _, bmax = [x.cpu().numpy() for x in obj.aabb]
                top_z = float(bmax[2])
                if top_z > 0.3:
                    tables.append((obj, top_z))
            except Exception:
                pass
    if tables:
        chosen, top_z = rng.choice(tables)
        print(f"[surface] Using table: {chosen.name}  top_z={top_z:.3f}")
        return chosen, top_z
    _, floor_bmax = [x.cpu().numpy() for x in floor_obj.aabb]
    top_z = float(floor_bmax[2])
    print(f"[surface] No table found — using floor  top_z={top_z:.3f}")
    return floor_obj, top_z


# ─────────────────────────────────────────────────────────────────────────────
# Vacancy check: does a proposed XY (with given half-extents) overlap any
# scene-graph object bbox or the container's live AABB?
# ─────────────────────────────────────────────────────────────────────────────

def _aabb_overlaps_xy(ax0, ax1, bx0, bx1, ay0, ay1, by0, by1) -> bool:
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _position_is_clear(cx: float, cy: float,
                        hx: float, hy: float,
                        room_objs: dict,
                        extra_bboxes: list,
                        skip_cats: set) -> bool:
    """
    Return True if a bbox centred at (cx, cy) with half-extents (hx, hy)
    does not overlap any object in room_objs (excluding skip_cats) or extra_bboxes.
    """
    SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                       "door", "curtain", "electric_switch"}
    all_skip = SKIP_STRUCTURAL | skip_cats

    px0, px1 = cx - hx, cx + hx
    py0, py1 = cy - hy, cy + hy

    for cat, bboxes in room_objs.items():
        if any(s in cat.lower() for s in all_skip):
            continue
        for (bmin, bmax) in bboxes:
            if _aabb_overlaps_xy(px0, px1, float(bmin[0]), float(bmax[0]),
                                  py0, py1, float(bmin[1]), float(bmax[1])):
                return False

    for (bmin, bmax) in extra_bboxes:
        if _aabb_overlaps_xy(px0, px1, float(bmin[0]), float(bmax[0]),
                              py0, py1, float(bmin[1]), float(bmax[1])):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",       type=str, required=True)
    parser.add_argument("--room",        type=str, required=True)
    parser.add_argument("--floor",       type=str, required=True)
    parser.add_argument("--run_idx",     type=int, default=0)
    parser.add_argument("--robot",       type=str, default="R1")
    parser.add_argument("--output_root", type=str, default="renders_containment_positive")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 1999
    rng     = random.Random(seed)
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"  Task: containment-positive (small obj BEHIND container)")
    print(f"{'='*70}\n")

    # ── Load static data ───────────────────────────────────────────────────────
    inventory = _load_inventory()

    with open(CONTAINERS_JSON)     as f: all_containers    = json.load(f)
    with open(SMALL_OBJECTS_JSON)  as f: all_small_objects = json.load(f)

    # ── Pick container ─────────────────────────────────────────────────────────
    container_entry = rng.choice(all_containers)
    container_cat   = container_entry["category"]
    container_model = container_entry["model"]
    print(f"[container] category={container_cat}  model={container_model}")

    container_bbox = get_bbox_size(container_model, inventory)
    if container_bbox is None:
        container_bbox = np.array([0.15, 0.15, 0.20])
        print(f"[container] bbox not in inventory — using default {container_bbox}")
    else:
        print(f"[container] inventory bbox={container_bbox.round(4)}")
    container_min_dim = float(np.min(container_bbox[:2]))   # min of dx, dy (XY opening)

    # ── Pick small object ──────────────────────────────────────────────────────
    small_cats_shuffled = all_small_objects[:]
    rng.shuffle(small_cats_shuffled)

    small_cat = small_model = None
    for cat in small_cats_shuffled:
        model = get_model_for_category(cat, seed + 1, inventory)
        if model is not None:
            small_cat, small_model = cat, model
            break

    if small_cat is None:
        print("[ERROR] No small object category found in inventory.")
        raise SystemExit(2)
    print(f"[small_obj] category={small_cat}  model={small_model}")

    # ── Compute scale ──────────────────────────────────────────────────────────
    small_bbox = get_bbox_size(small_model, inventory)
    if small_bbox is None:
        small_bbox = np.array([0.05, 0.05, 0.05])
        print(f"[small_obj] bbox not in inventory — using default {small_bbox}")
    else:
        print(f"[small_obj] inventory bbox={small_bbox.round(4)}")

    small_max_dim = float(np.max(small_bbox))
    raw_target    = rng.uniform(SMALL_OBJ_SCALE_MIN, SMALL_OBJ_SCALE_MAX)
    max_allowed   = SMALL_OBJ_MAX_FRAC * container_min_dim
    target_size   = min(raw_target, max_allowed)
    small_scale   = target_size / small_max_dim if small_max_dim > 1e-6 else 1.0
    print(f"[small_obj] container_min_dim={container_min_dim:.4f}  "
          f"target_size={target_size:.4f}  small_max_dim={small_max_dim:.4f}  "
          f"scale={small_scale:.4f}")

    # ── Build OmniGibson config ────────────────────────────────────────────────
    config_filename = os.path.join(og.example_config_path,
                                   f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"] = [
        {
            "type":        "DatasetObject",
            "name":        "obj_container",
            "category":    container_cat,
            "model":       container_model,
            "position":    [150.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale":       1.0,
        },
        {
            "type":        "DatasetObject",
            "name":        "obj_small",
            "category":    small_cat,
            "model":       small_model,
            "position":    [160.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale":       [small_scale, small_scale, small_scale],
        },
    ]

    env   = og.Environment(configs=config)
    scene = env.scene

    obj_container = scene.object_registry("name", "obj_container")
    obj_small     = scene.object_registry("name", "obj_small")
    floor_obj     = scene.object_registry("name", args.floor)

    if floor_obj is None:
        print(f"[ERROR] Floor '{args.floor}' not found.")
        raise SystemExit(2)

    scene_dict = load_scene_dict(args.scene)
    room_objs  = get_room_objects(scene_dict, args.room)

    # ── Pick surface ───────────────────────────────────────────────────────────
    surface_obj, surface_top_z = pick_surface(scene, floor_obj, rng)

    # ── Place container on surface ─────────────────────────────────────────────
    ok = obj_container.states[object_states.OnTop].set_value(surface_obj, True)
    print(f"[place] container OnTop({surface_obj.name}): {ok}")
    for _ in range(20):
        og.sim.step()

    _, surf_bmax  = [x.cpu().numpy() for x in surface_obj.aabb]
    surface_top_z = float(surf_bmax[2])

    cont_bmin, cont_bmax = [x.cpu().numpy() for x in obj_container.aabb]
    cont_centre   = (cont_bmin + cont_bmax) / 2.0
    cont_centre_z = float(cont_centre[2])
    cx = float(cont_centre[0])
    cy = float(cont_centre[1])

    # Container XY half-diagonal (for orbit radius)
    cont_hx = abs(float(cont_bmax[0]) - float(cont_bmin[0])) / 2.0
    cont_hy = abs(float(cont_bmax[1]) - float(cont_bmin[1])) / 2.0
    cont_half_diag = float(np.sqrt(cont_hx**2 + cont_hy**2))

    print(f"[container] centre=({cx:.3f}, {cy:.3f}, {cont_centre_z:.3f})  "
          f"half_diag={cont_half_diag:.3f}")

    # cam0_azimuth_rad is derived after final placement (see render section below)

    # Orbit radius: just beyond container bbox
    orbit_radius = cont_half_diag + CAM_RADIUS_PAD

    # ── Drop small object anywhere to get its settled Z and live half-extents ──
    obj_small.states[object_states.OnTop].set_value(surface_obj, True)
    for _ in range(20):
        og.sim.step()

    small_bmin, small_bmax = [x.cpu().numpy() for x in obj_small.aabb]
    small_hx  = abs(float(small_bmax[0]) - float(small_bmin[0])) / 2.0
    small_hy  = abs(float(small_bmax[1]) - float(small_bmin[1])) / 2.0
    small_z   = float(obj_small.get_position_orientation()[0].cpu().numpy()[2])
    print(f"[small_obj] live half_x={small_hx:.4f}  half_y={small_hy:.4f}  z={small_z:.4f}")

    # ── Find vacant position behind container on the -Y side ──────────────────
    # "Behind" from view 0 = the -Y face of the container (camera_0 is at +Y).
    # Layout:  small_obj  ←—0.01m gap—→  container  ←—orbit_radius—→  camera_0
    cont_near_y = float(cont_bmin[1])   # -Y face of container (the side facing small_obj)

    # Target small_obj centre Y: its +Y face is 0.01 m from container's -Y face.
    # small_obj centre = cont_near_y - BEHIND_GAP - small_hy
    TARGET_GAP = 0.01   # metres between container -Y face and small_obj +Y face

    cont_extra = [(cont_bmin, cont_bmax)]
    skip_cats  = {container_cat, small_cat}

    chosen_xy    = None
    lateral_signs = [0, 1, -1, 2, -2, 3, -3, 4, -4]

    for sign in lateral_signs[:MAX_LATERAL_TRIES + 1]:
        proposed_x = cx + sign * LATERAL_STEP
        proposed_y = cont_near_y - TARGET_GAP - small_hy

        if _position_is_clear(proposed_x, proposed_y, small_hx, small_hy,
                               room_objs, cont_extra, skip_cats):
            chosen_xy = np.array([proposed_x, proposed_y])
            print(f"[small_obj] vacant spot: x={proposed_x:.4f}  y={proposed_y:.4f}  "
                  f"lateral_offset={sign * LATERAL_STEP:.4f}m")
            break
        else:
            print(f"[small_obj] lateral offset {sign * LATERAL_STEP:.4f}m blocked — trying next")

    if chosen_xy is None:
        chosen_xy = np.array([cx, cont_near_y - TARGET_GAP - small_hy])
        print(f"[small_obj] WARNING: all blocked — forcing ({chosen_xy[0]:.4f}, {chosen_xy[1]:.4f})")

    # ── Teleport to rough position first, then use live AABBs to nail the gap ──
    obj_small.set_position_orientation(
        position=th.tensor([float(chosen_xy[0]), float(chosen_xy[1]), small_z],
                            dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    obj_small.keep_still()
    for _ in range(20):
        og.sim.step()

    # Re-read both live AABBs and correct the Y position so the gap is exactly 0.01 m.
    # small_obj +Y face (small_bmax[1]) should equal cont_bmin[1] - TARGET_GAP.
    cont_bmin_live, _     = [x.cpu().numpy() for x in obj_container.aabb]
    small_bmin_live, small_bmax_live = [x.cpu().numpy() for x in obj_small.aabb]

    current_small_pos = obj_small.get_position_orientation()[0].cpu().numpy()
    offset_y = (float(cont_bmin_live[1]) - TARGET_GAP) - float(small_bmax_live[1])
    corrected_y = float(current_small_pos[1]) + offset_y

    obj_small.set_position_orientation(
        position=th.tensor([float(current_small_pos[0]), corrected_y, float(current_small_pos[2])],
                            dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    obj_small.keep_still()
    for _ in range(10):
        og.sim.step()

    small_pos_final, _ = obj_small.get_position_orientation()
    _, small_bmax_final = [x.cpu().numpy() for x in obj_small.aabb]
    cont_bmin_final, _  = [x.cpu().numpy() for x in obj_container.aabb]
    actual_gap = float(cont_bmin_final[1]) - float(small_bmax_final[1])
    print(f"[small_obj] final pos={small_pos_final.cpu().numpy().round(4)}  "
          f"actual_gap={actual_gap:.4f}m  (target={TARGET_GAP}m)")

    # ── Add seg modalities ─────────────────────────────────────────────────────
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # ── Render 6 cameras ──────────────────────────────────────────────────────
    # Derive cam0_azimuth from actual final positions:
    # camera_0 sits on the same side as the small object (container→small_obj direction),
    # further out by orbit_radius, so the line is:
    #   camera_0  ──  small_obj  ──0.01m──  container
    # Views 1-5 step +60° each from there.
    # eye Z and look_target Z are both cam_z → perfectly horizontal.
    cam_z = float(small_pos_final.cpu().numpy()[2])

    small_xy = small_pos_final.cpu().numpy()[:2]
    cont_xy  = np.array([cx, cy])
    axis     = small_xy - cont_xy                      # vector from container to small obj
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis /= axis_norm
    else:
        axis = np.array([0.0, -1.0])                   # fallback: -Y

    # cam0 is on the OPPOSITE side from the small object:
    # small_obj  ──  container  ──  camera_0
    # So negate the container→small_obj axis.
    cam0_azimuth_rad = np.arctan2(-axis[1], -axis[0])

    look_target = np.array([cx, cy, cam_z])

    camera_poses = {}
    exist_flags  = {}
    obj_names    = ("obj_container", "obj_small")

    for view_idx in range(NUM_CAMERAS):
        azimuth_rad = cam0_azimuth_rad + np.deg2rad(view_idx * AZIMUTH_STEP_DEG)
        azimuth_deg = np.rad2deg(azimuth_rad) % 360

        # Camera sits at orbit_radius from container centre, at small object height
        eye = np.array([
            cx + orbit_radius * np.cos(azimuth_rad),
            cy + orbit_radius * np.sin(azimuth_rad),
            cam_z,
        ])

        fname = f"{view_idx}.png"
        fpath = os.path.join(run_dir, fname)
        print(f"\n[camera {view_idx}] azimuth={azimuth_deg:.1f}°  eye={eye.round(3)}  "
              f"look_at={look_target.round(3)}")

        pose = _set_camera_and_capture(eye, look_target, fpath)
        camera_poses[fname] = {**pose, "azimuth_deg": azimuth_deg,
                                "height": cam_z, "type": "side"}

        vis = _visibility_check(obj_names)
        for name in obj_names:
            exist_flags[f"exist_{name}_{view_idx}"] = vis[name]
        print(f"[camera {view_idx}] visibility: {vis}")

    # ── Metadata ───────────────────────────────────────────────────────────────
    def _obj_meta(obj, category, model, scale_val):
        pos, quat = obj.get_position_orientation()
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        return {
            "category":        category,
            "model":           model,
            "scale":           scale_val,
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
            "aabb_min":        bmin.tolist(),
            "aabb_max":        bmax.tolist(),
        }

    metadata = {
        "scene":              args.scene,
        "room":               args.room,
        "run_idx":            args.run_idx,
        "seed":               seed,
        "floor_name":         args.floor,
        "surface_name":       surface_obj.name,
        "surface_top_z":      surface_top_z,
        "layout":             "containment_positive",
        "container_cat":      container_cat,
        "container_model":    container_model,
        "container_bbox_inventory": container_bbox.tolist(),
        "container_min_dim":  container_min_dim,
        "small_obj_cat":      small_cat,
        "small_obj_model":    small_model,
        "small_obj_bbox_inventory": small_bbox.tolist(),
        "small_obj_max_dim":  float(small_max_dim),
        "small_obj_scale":    small_scale,
        "target_world_size":  float(target_size),
        "placement_behind":   True,
        "target_gap_m":       TARGET_GAP,
        "actual_gap_m":       actual_gap,
        "answer":             "obj_small is NOT inside obj_container",
        "collinear_axis":     "Y",
        "camera_0_azimuth_deg": float(np.rad2deg(cam0_azimuth_rad)),
        "camera_layout": {
            "num_views":       NUM_CAMERAS,
            "azimuth_step_deg": AZIMUTH_STEP_DEG,
            "orbit_radius":    orbit_radius,
            "height_z":        cam_z,
            "height_z_source": "small object settled Z (horizontal gaze at object height)",
            "view_0":          "behind small_obj: camera at +Y of container, looking -Y",
            "view_k":          "view_0 azimuth + k * 60°",
        },
        **exist_flags,
        "objects": {
            "obj_container": _obj_meta(obj_container, container_cat,
                                        container_model, 1.0),
            "obj_small":     _obj_meta(obj_small, small_cat,
                                        small_model, small_scale),
        },
        "camera_poses": camera_poses,
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    container_visible = any(exist_flags.get(f"exist_obj_container_{k}", False)
                             for k in range(NUM_CAMERAS))
    small_visible     = any(exist_flags.get(f"exist_obj_small_{k}", False)
                             for k in range(NUM_CAMERAS))
    success = container_visible and small_visible
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()