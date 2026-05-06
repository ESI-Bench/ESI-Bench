"""
batch_containment.py

Single-run script: loads ONE scene restricted to ONE room.
  1. Randomly picks a transparent container from containers.json and places
     it on a table (any object whose name contains "table") or floor fallback.
  2. Randomly picks a small object from small_objects.json, finds a model
     via object_inventory.json bounding_box_sizes, and scales it to fit.
  3. Attempts to place the small object inside the container using
     object_states.Inside.  Falls back to direct set_position_orientation
     if the state check fails.  Writes placement_success=false and exits
     with code 2 if both attempts fail.
  4. Renders exactly 2 views:
       view 0 — front: horizontal, at surface height, pulled back -Y
       view 1 — back:  180° from view 0, same height
  5. Saves metadata JSON with positions, camera poses, visibility flags,
     and placement_success.

Scale logic:
  - Container bbox from inventory = [dx, dy, dz] world size at scale=1.
    container_min_dim = min(dx, dy, dz)
  - Small object bbox from inventory = [dx, dy, dz] world size at scale=1.
    small_max_dim = max(dx, dy, dz)
  - target_size = min(uniform(0.02, 0.04), 0.5 * container_min_dim)
  - scale = target_size / small_max_dim   (so largest dim becomes target_size)

Exit codes:
  0 — success (placed and visible in at least one view)
  1 — partial (placed but not visible)
  2 — fatal error (floor/objects not found, or placement failed twice)
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
SMALL_OBJ_SCALE_MIN = 0.025   # uniform lower bound for target world size (m)
SMALL_OBJ_SCALE_MAX = 0.05   # uniform upper bound for target world size (m)
SMALL_OBJ_MAX_FRAC  = 0.5    # target must not exceed this fraction of container_min_dim

# ── Camera ────────────────────────────────────────────────────────────────────
NUM_CAMERAS      = 6
AZIMUTH_STEP_DEG = 60.0
CAM_RADIUS_PAD   = 0.25   # extra beyond container xy half-diagonal for orbit radius

SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Inventory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_inventory() -> dict:
    for path in INVENTORY_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise RuntimeError("No object_inventory.json found.")
    

def get_model_for_category(category: str, seed: int, inventory: dict):
    """Return model_id string or None if category not in inventory."""
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
    """
    Return np.array([dx, dy, dz]) from inventory bounding_box_sizes,
    or None if missing.  These are the actual world-space dimensions at scale=1.
    """
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
# Inside check
# ─────────────────────────────────────────────────────────────────────────────

def is_inside_bbox(small_obj, container_obj) -> bool:
    """
    Return True iff the small object's AABB is fully within the container's AABB.
    i.e. small_min >= container_min  AND  small_max <= container_max  (all axes).
    """
    s_min, s_max = [x.cpu().numpy() for x in small_obj.aabb]
    c_min, c_max = [x.cpu().numpy() for x in container_obj.aabb]
    result = bool(np.all(s_min >= c_min) and np.all(s_max <= c_max))
    print(f"[inside_check] small=[{s_min.round(3)}, {s_max.round(3)}]  "
          f"container=[{c_min.round(3)}, {c_max.round(3)}]  inside={result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Surface selection: any object whose name contains "table", else floor
# ─────────────────────────────────────────────────────────────────────────────

def pick_surface(scene, floor_obj, rng: random.Random):
    """
    Return (surface_obj, surface_top_z).
    Looks for any loaded object whose name contains "table".
    Falls back to floor_obj if none found.
    """
    tables = []
    for obj in scene.objects:
        if "table" in (getattr(obj, "name", "") or "").lower():
            try:
                _, bmax = [x.cpu().numpy() for x in obj.aabb]
                top_z = float(bmax[2])
                if top_z > 0.3:   # must be actually elevated
                    tables.append((obj, top_z))
            except Exception:
                pass

    if tables:
        chosen, top_z = rng.choice(tables)
        print(f"[surface] Using table: {chosen.name}  top_z={top_z:.3f}")
        return chosen, top_z

    # Fall back to floor
    _, floor_bmax = [x.cpu().numpy() for x in floor_obj.aabb]
    top_z = float(floor_bmax[2])
    print(f"[surface] No table found — using floor  top_z={top_z:.3f}")
    return floor_obj, top_z


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
    parser.add_argument("--output_root", type=str, default="renders_containment")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 999
    rng     = random.Random(seed)
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
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

    # bbox = [dx, dy, dz] at scale=1 — the actual world size
    container_bbox = get_bbox_size(container_model, inventory)
    if container_bbox is None:
        container_bbox = np.array([0.15, 0.15, 0.20])
        print(f"[container] bbox not in inventory — using default {container_bbox}")
    else:
        print(f"[container] inventory bbox={container_bbox.round(4)}")
    container_min_dim = float(np.min(container_bbox))   # smallest of dx/dy/dz

    # ── Pick small object ──────────────────────────────────────────────────────
    # Shuffle categories and try until we find one present in the inventory.
    small_cats_shuffled = all_small_objects[:]
    rng.shuffle(small_cats_shuffled)

    small_cat   = None
    small_model = None
    for cat in small_cats_shuffled:
        model = get_model_for_category(cat, seed + 1, inventory)
        if model is not None:
            small_cat   = cat
            small_model = model
            break

    if small_cat is None:
        print("[ERROR] No small object category found in inventory.")
        raise SystemExit(2)
    print(f"[small_obj] category={small_cat}  model={small_model}")

    # ── Compute scale ──────────────────────────────────────────────────────────
    # bbox = [dx, dy, dz] world size at scale=1
    small_bbox = get_bbox_size(small_model, inventory)
    if small_bbox is None:
        small_bbox = np.array([0.05, 0.05, 0.05])
        print(f"[small_obj] bbox not in inventory — using default {small_bbox}")
    else:
        print(f"[small_obj] inventory bbox={small_bbox.round(4)}")

    small_max_dim = float(np.max(small_bbox))   # largest of dx/dy/dz

    # target world size for the small object's largest dimension
    raw_target  = rng.uniform(SMALL_OBJ_SCALE_MIN, SMALL_OBJ_SCALE_MAX)
    max_allowed = SMALL_OBJ_MAX_FRAC * container_min_dim
    target_size = min(raw_target, max_allowed)

    # scale = how much to multiply the mesh so its largest dim == target_size
    small_scale = target_size / small_max_dim if small_max_dim > 1e-6 else 1.0
    print(f"[small_obj] container_min_dim={container_min_dim:.4f}  "
          f"target_size={target_size:.4f}  "
          f"small_max_dim={small_max_dim:.4f}  scale={small_scale:.4f}")

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

    # ── Pick surface ───────────────────────────────────────────────────────────
    surface_obj, surface_top_z = pick_surface(scene, floor_obj, rng)

    # ── Place container on surface ─────────────────────────────────────────────
    ok = obj_container.states[object_states.OnTop].set_value(surface_obj, True)
    print(f"[place] container OnTop({surface_obj.name}): {ok}")
    for _ in range(20):
        og.sim.step()

    # Re-read surface top Z from live AABB (more reliable after physics settle)
    _, surf_bmax  = [x.cpu().numpy() for x in surface_obj.aabb]
    surface_top_z = float(surf_bmax[2])

    cont_bmin, cont_bmax = [x.cpu().numpy() for x in obj_container.aabb]
    cont_centre   = (cont_bmin + cont_bmax) / 2.0
    cont_centre_z = float(cont_centre[2])
    print(f"[container] cont_centre={cont_centre.round(3)}")

    # ── Attempt 1: Inside state ────────────────────────────────────────────────
    placement_success = False
    attempt_log       = []

    print("[place] Attempt 1: object_states.Inside ...")
    inside_ok = obj_small.states[object_states.Inside].set_value(obj_container, True)
    for _ in range(30):
        og.sim.step()

    if inside_ok and is_inside_bbox(obj_small, obj_container):
        placement_success = True
        attempt_log.append({"attempt": 1, "method": "Inside_state", "success": True})
        print("[place] Attempt 1 SUCCESS")
    else:
        attempt_log.append({"attempt": 1, "method": "Inside_state",
                             "success": False, "inside_state_returned": inside_ok})
        print(f"[place] Attempt 1 FAILED (state_returned={inside_ok}) — trying manual")

        # ── Attempt 2: set_position_orientation to container interior ──────────
        print("[place] Attempt 2: set_position_orientation ...")
        cont_h   = float(cont_bmax[2] - cont_bmin[2])
        target_z = float(cont_bmin[2]) + 0.25 * cont_h   # 25% up from bottom

        obj_small.set_position_orientation(
            position=th.tensor([float(cont_centre[0]),
                                float(cont_centre[1]),
                                target_z], dtype=th.float32),
            orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        obj_small.keep_still()
        for _ in range(30):
            og.sim.step()

        if is_inside_bbox(obj_small, obj_container):
            placement_success = True
            attempt_log.append({"attempt": 2, "method": "set_position_orientation",
                                 "success": True})
            print("[place] Attempt 2 SUCCESS")
        else:
            attempt_log.append({"attempt": 2, "method": "set_position_orientation",
                                 "success": False})
            print("[place] Attempt 2 FAILED — writing failure JSON and exiting")
            meta_fail = {
                "scene":             args.scene,
                "room":              args.room,
                "run_idx":           args.run_idx,
                "seed":              seed,
                "floor_name":        args.floor,
                "container_cat":     container_cat,
                "container_model":   container_model,
                "small_obj_cat":     small_cat,
                "small_obj_model":   small_model,
                "small_obj_scale":   small_scale,
                "placement_success": False,
                "attempt_log":       attempt_log,
            }
            with open(os.path.join(run_dir, "metadata.json"), "w") as f:
                json.dump(meta_fail, f, indent=2)
            raise SystemExit(2)

    # ── Add seg modalities ─────────────────────────────────────────────────────
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # ── Camera orbit setup (identical to batch_containment_positive.py) ────────
    # Re-read container and small object positions after all settling.
    container_pos, _ = obj_container.get_position_orientation()
    cx = float(container_pos[0])
    cy = float(container_pos[1])

    cont_bmin_f, cont_bmax_f = [x.cpu().numpy() for x in obj_container.aabb]
    cont_hx       = abs(float(cont_bmax_f[0]) - float(cont_bmin_f[0])) / 2.0
    cont_hy       = abs(float(cont_bmax_f[1]) - float(cont_bmin_f[1])) / 2.0
    cont_half_diag = float(np.sqrt(cont_hx**2 + cont_hy**2))
    orbit_radius  = cont_half_diag + CAM_RADIUS_PAD

    small_pos_f, _ = obj_small.get_position_orientation()
    cam_z = float(small_pos_f.cpu().numpy()[2])   # eye and look_target share this Z

    # Small object is inside the container so positions nearly coincide —
    # no meaningful axis to derive. Use a fixed azimuth: camera sits at -Y.
    cam0_azimuth_rad = -np.pi / 2

    look_target = np.array([cx, cy, cam_z])

    # ── Render 6 views ─────────────────────────────────────────────────────────
    camera_poses = {}
    exist_flags  = {}
    obj_names    = ("obj_container", "obj_small")

    for view_idx in range(NUM_CAMERAS):
        azimuth_rad = cam0_azimuth_rad + np.deg2rad(view_idx * AZIMUTH_STEP_DEG)
        azimuth_deg = np.rad2deg(azimuth_rad) % 360

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
        "scene":             args.scene,
        "room":              args.room,
        "run_idx":           args.run_idx,
        "seed":              seed,
        "floor_name":        args.floor,
        "surface_name":      surface_obj.name,
        "surface_top_z":     surface_top_z,
        "layout":            "containment",
        "container_cat":     container_cat,
        "container_model":   container_model,
        "container_bbox_inventory": container_bbox.tolist(),
        "container_min_dim": container_min_dim,
        "small_obj_cat":     small_cat,
        "small_obj_model":   small_model,
        "small_obj_bbox_inventory": small_bbox.tolist(),
        "small_obj_max_dim": float(small_max_dim),
        "small_obj_scale":   small_scale,
        "target_world_size": float(target_size),
        "placement_success": placement_success,
        "attempt_log":       attempt_log,
        "answer":            "obj_small is inside obj_container",
        "camera_layout": {
            "num_views":        NUM_CAMERAS,
            "azimuth_step_deg": AZIMUTH_STEP_DEG,
            "orbit_radius":     orbit_radius,
            "height_z":         cam_z,
            "height_z_source":  "small object settled Z",
            "view_0":           "small_obj behind container from camera perspective",
            "view_k":           "view_0 azimuth + k * 60°",
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
    success = placement_success and container_visible and small_visible
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()