"""
batch_storage_multi.py

Multi-object storage script. DO NOT modify batch_storage.py.

3 vs 4 object mode is decided randomly (50/50) via rng.random() < 0.5.

  3-object:
    - containee → fit container
    - extra1 + extra2 → big container
      (scaled so pair fills big container in X, Y, Z)

  4-object (50% chance):
    - same as 3-object PLUS
    - extra3 → small container
      (scaled as small_cont_world / (ratio * extra3_bbox), ratio=uniform(1.1,1.2))

Layout:
  - The containee and all 3 containers are placed in a strict straight ROW
    (same logic as batch_storage.py: one-at-a-time measurement, furniture
    collision avoidance, exact bbox-to-bbox gaps).
  - extra1, extra2 (and extra3 if present) are each placed randomly but
    within EXTRA_NEAR_GAP metres bbox-to-bbox of their target container,
    clear of all other already-placed objects and room furniture.

6 orbital cameras (45 deg downward pitch, locked geometry) are rendered:
  - BEFORE any placements (orbital_before_*.png, saved to run_dir)
  - AFTER each place_into_container call (orbital_after_{label}_*.png)

All camera poses and exist flags are saved in metadata.json.
"""

import os
import json
import yaml
import argparse
import random
import traceback
import numpy as np
import torch as th
import cv2

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.usd_utils import create_joint, delete_or_deactivate_prim
from scipy.spatial.transform import Rotation

gm.ENABLE_FLATCACHE        = False
gm.USE_GPU_DYNAMICS        = False
gm.ENABLE_OBJECT_STATES    = True
gm.ENABLE_TRANSITION_RULES = False

SQUARE_ORI           = [0.0, 0.0, 0.0, 1.0]
TOP_DOWN_HEIGHT_PAD  = 1.2
CONTAINER_HEIGHT_PAD = 0.8
SCENES_DIR           = "scenes5"
NAV_DIST_THRESHOLD   = 2.0
MAX_NAV_ATTEMPTS     = 50

ROW_GAP              = 0.2     # bbox-to-bbox gap in the containee+containers row
EXTRA_NEAR_GAP       = 0.3     # bbox-to-bbox distance from extra to its target container
EXTRA_PLACE_MARGIN   = 0.05    # clearance margin when checking extra placement

ORBITAL_N_VIEWS      = 6
ORBITAL_RADIUS_MARGIN = 0.6

FIT_RATIO        = {"small": (0.9, 0.95), "fit": (1.3, 1.4), "big": (1.6, 1.75)}
EXTRA_PAIR_RATIO = (0.75, 0.8)
EXTRA3_FIT_RATIO = (1.3, 1.4)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_inventory(path: str) -> dict:
    fallbacks = [path,
                 "bddl3/bddl/generated_data/object_inventory.json",
                 os.path.join(os.path.dirname(__file__), "object_inventory.json")]
    for p in fallbacks:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise RuntimeError("object_inventory.json not found.")


def get_bbox_size(model_id: str, inventory: dict):
    size = inventory.get("bounding_box_sizes", {}).get(model_id)
    return np.array(size, dtype=float) if size is not None else None


def get_model_for_category(category: str, inventory: dict, rng: random.Random):
    providers = inventory.get("providers", inventory)
    clean = category.replace(" (fillable)", "").strip()
    matches = [k for k in providers if k.startswith(f"{clean}-")]
    if not matches:
        return None, None
    chosen = rng.choice(matches)
    model_id = chosen.split("-", 1)[1]
    bbox = get_bbox_size(model_id, inventory)
    return model_id, bbox


def get_model_id(category: str, instance_id: str, inventory: dict) -> str:
    providers = inventory.get("providers", inventory)
    clean = category.replace(" (fillable)", "").strip()
    if f"{clean}-{instance_id}" in providers:
        return instance_id
    matches = [k for k in providers if k.startswith(f"{clean}-")]
    if matches:
        model = random.choice(matches).split("-", 1)[1]
        print(f"[inventory] '{instance_id}' not found for '{clean}', using '{model}'")
        return model
    raise RuntimeError(f"No inventory entries for '{clean}'.")


def find_floor_name(scene_name: str, room_name: str, room_objects_path: str) -> str:
    with open(room_objects_path) as f:
        data = json.load(f)
    for o in data.get("scenes", data).get(scene_name, {}).get(room_name, []):
        if o.startswith("floors-"):
            return o.replace("-", "_") + "_0"
    return ""


def parse_location_room(location: list, scene_desp: dict) -> str:
    for loc in location:
        if loc.startswith("in-"):
            name = loc[3:]
            if name in scene_desp:
                return name
    return ""


def parse_task(task_data: dict, scene_desp: dict):
    scene_name = task_data.get("scene_name", "Rs_int")
    if scene_name.endswith("_scene_dict.json"):
        scene_name = scene_name.replace("_scene_dict.json", "")
    answer     = task_data["parsed_ans"]
    containee  = None
    containers = []
    for spec in answer.get("new objects", []):
        t = spec.get("type", "").lower()
        if t == "contents":
            containee = spec
        elif t == "container":
            containers.append(spec)
    if containee is None:
        raise ValueError("No type='contents' object in task JSON.")
    if not containers:
        raise ValueError("No type='container' objects in task JSON.")
    return scene_name, answer.get("Rooms", []), containee, containers


def look_at_quat(eye, target, up=np.array([0., 0., 1.])):
    fwd = np.array(target, float) - np.array(eye, float)
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        return np.array([0, 0, 0, 1], float)
    fwd /= n
    r = np.cross(fwd, up)
    if np.linalg.norm(r) < 1e-6:
        up = np.array([0, 1, 0])
        r  = np.cross(fwd, up)
    r /= np.linalg.norm(r)
    u  = np.cross(r, fwd)
    u /= np.linalg.norm(u)
    return Rotation.from_matrix(np.column_stack([r, u, -fwd])).as_quat()


def do_capture(path: str):
    for _ in range(10):
        og.sim.render()
    img = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[render] {path}")


def set_cam_capture(eye, target, path, up=np.array([0., 1., 0.])) -> dict:
    q = look_at_quat(eye, target, up=up)
    og.sim._viewer_camera.set_position_orientation(eye, q)
    do_capture(path)
    return {"position": eye.tolist(), "quaternion_xyzw": q.tolist()}


def seg_visibility(obj_names: tuple) -> dict:
    for _ in range(100):
        og.sim.step()
    raw    = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    labels = " ".join(raw["info"]["idToLabels"].values())
    result = {n: (n in labels) for n in obj_names}
    print(f"[seg] {result}")
    return result


def get_aabb(obj):
    bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
    return bmin, bmax


def is_inside_bbox(small_obj, container_obj) -> bool:
    s_min, s_max = get_aabb(small_obj)
    c_min, c_max = get_aabb(container_obj)
    TOL = 0.05
    xy = bool(s_min[0] >= c_min[0] - TOL and s_max[0] <= c_max[0] + TOL and
              s_min[1] >= c_min[1] - TOL and s_max[1] <= c_max[1] + TOL)
    z  = bool(s_min[2] < c_max[2] - 0.05)
    print(f"[inside_check] xy={xy}  z={z}  inside={xy and z}")
    return xy and z


def load_scene_dict(scene_name: str) -> dict:
    path = os.path.join(SCENES_DIR, f"{scene_name}_scene_dict.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene dict not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_room_object_names(scene_dict: dict, room: str) -> list:
    return list(scene_dict.get(room, {}).keys())


def pick_surface(scene, floor_obj, room_obj_names: list, rng: random.Random):
    tables = []
    for obj_name in room_obj_names:
        if "table" not in obj_name.lower():
            continue
        obj = scene.object_registry("name", obj_name)
        if obj is None:
            continue
        try:
            _, bmax = get_aabb(obj)
            top_z = float(bmax[2])
            if top_z > 0.3:
                tables.append((obj, top_z))
        except Exception:
            pass
    if tables:
        chosen, top_z = rng.choice(tables)
        print(f"[surface] Table: {chosen.name}  top_z={top_z:.3f}")
        return chosen, top_z
    _, fbmax = get_aabb(floor_obj)
    top_z = float(fbmax[2])
    print(f"[surface] Floor  top_z={top_z:.3f}")
    return floor_obj, top_z


def snap(obj) -> dict:
    pos, quat = obj.get_position_orientation()
    bmin, bmax = get_aabb(obj)
    return {
        "position":        pos.cpu().numpy().tolist(),
        "quaternion_xyzw": quat.cpu().numpy().tolist(),
        "aabb_min":        bmin.tolist(),
        "aabb_max":        bmax.tolist(),
    }


def find_eye_camera_key(robot) -> str:
    for key in robot._sensors:
        if "eyes:Camera:0" in key:
            return key
    return ""


def capture_robot_eye(robot, eye_camera_key: str, path: str):
    for _ in range(5):
        og.sim.render()
    img = robot._sensors[eye_camera_key].get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[robot_eye] {path}")


def get_robot_pose(robot) -> dict:
    pos, ori = robot.get_position_orientation()
    return {
        "robot_position":        pos.cpu().numpy().tolist(),
        "robot_quaternion_xyzw": ori.cpu().numpy().tolist(),
    }


def mock_navigate(robot, target_obj, floor_obj,
                  max_attempts=MAX_NAV_ATTEMPTS, dist_threshold=NAV_DIST_THRESHOLD) -> bool:
    target_pos, _ = target_obj.get_position_orientation()
    target_xy = np.array(target_pos.cpu().numpy()[:2])
    for attempt in range(max_attempts):
        if sample_kinematics("onTop", robot, floor_obj):
            robot_pos, _ = robot.get_position_orientation()
            og.sim.step()
            dist = float(np.linalg.norm(robot_pos.cpu().numpy()[:2] - target_xy))
            if dist <= dist_threshold:
                print(f"[nav] reached {target_obj.name} after {attempt+1} tries  dist={dist:.3f}m")
                robot.reset()
                for _ in range(50):
                    og.sim.step()
                return True
    print(f"[nav] FAILED to reach {target_obj.name}")
    return False


AG_JOINT_PRIM = None


def mock_grasp(robot, obj) -> bool:
    global AG_JOINT_PRIM
    grasp_point = robot.get_eef_position(robot.default_arm)
    obj.visual_only = True
    obj.set_position_orientation(grasp_point, [0., 0., 0., 1.])
    obj.keep_still()
    og.sim.step()
    joint_prim_path = f"{robot.eef_links[robot.default_arm].prim_path}/ag_constraint"
    AG_JOINT_PRIM = create_joint(
        prim_path=joint_prim_path, joint_type="FixedJoint",
        body0=robot.eef_links[robot.default_arm].prim_path,
        body1=obj.root_link.prim_path,
        enabled=True, exclude_from_articulation=True,
    )
    for _ in range(10):
        og.sim.step()
    print(f"[grasp] Grasped {obj.name}")
    return True


def mock_release(obj):
    global AG_JOINT_PRIM
    if AG_JOINT_PRIM is not None:
        delete_or_deactivate_prim(str(AG_JOINT_PRIM.GetPrimPath()))
        AG_JOINT_PRIM = None
    obj.set_position_orientation([100., 100., 100.], [0., 0., 0., 1.])
    obj.visual_only = False
    obj.keep_still()
    for _ in range(10):
        og.sim.step()
    print(f"[release] Released {obj.name}")


def pick_non_fillable_category(keys, exclude, inventory, rng):
    candidates = [k for k in keys if "(fillable)" not in k and k not in exclude]
    rng.shuffle(candidates)
    for cat in candidates:
        model_id, bbox = get_model_for_category(cat, inventory, rng)
        if model_id is not None and bbox is not None:
            return cat, model_id, bbox
    raise RuntimeError("No valid non-fillable category found.")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xy_half_extents(obj):
    bmin, bmax = get_aabb(obj)
    return (abs(float(bmax[0]) - float(bmin[0])) / 2.0,
            abs(float(bmax[1]) - float(bmin[1])) / 2.0)


def _aabb_overlap_xy(axmin, axmax, aymin, aymax,
                     bxmin, bxmax, bymin, bymax) -> bool:
    return not (axmax <= bxmin or axmin >= bxmax or
                aymax <= bymin or aymin >= bymax)


def _collect_furniture_aabbs(scene_dict_room: dict) -> list:
    SKIP = {"floors", "walls", "ceilings", "carpet", "window",
            "door", "curtain", "electric_switch"}
    result = []
    for cat, bboxes in scene_dict_room.items():
        if any(s in cat.lower() for s in SKIP):
            continue
        for (bmin, bmax) in bboxes:
            result.append((np.array(bmin, dtype=float)[:2].copy(),
                           np.array(bmax, dtype=float)[:2].copy()))
    print(f"[placement] furniture obstacles from scenes5: {len(result)} bboxes")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Row placement — containee + all 3 containers
# ─────────────────────────────────────────────────────────────────────────────

def _row_footprint_clear(anchor_xy, axis, half_along,
                         measured_hx, measured_hy, total_span,
                         furniture_aabbs, surface_aabb, margin=0.05) -> bool:
    row_start = anchor_xy - axis * (total_span / 2.0)
    cursor    = row_start + axis * half_along[0]
    centres   = [cursor.copy()]
    for i in range(1, len(half_along)):
        cursor = cursor + axis * half_along[i-1] + axis * ROW_GAP + axis * half_along[i]
        centres.append(cursor.copy())
    surf_bmin, surf_bmax = surface_aabb
    for i, c in enumerate(centres):
        cx, cy = float(c[0]), float(c[1])
        hx, hy = measured_hx[i], measured_hy[i]
        oxmin = cx - hx - margin;  oxmax = cx + hx + margin
        oymin = cy - hy - margin;  oymax = cy + hy + margin
        if (oxmin < float(surf_bmin[0]) or oxmax > float(surf_bmax[0]) or
                oymin < float(surf_bmin[1]) or oymax > float(surf_bmax[1])):
            return False
        for (f_bmin, f_bmax) in furniture_aabbs:
            if _aabb_overlap_xy(oxmin, oxmax, oymin, oymax,
                                float(f_bmin[0]), float(f_bmax[0]),
                                float(f_bmin[1]), float(f_bmax[1])):
                return False
    return True


def place_row(surface_obj, floor_obj, row_objs, robot,
              scene_dict_room, rng) -> dict:
    """
    Place row_objs in a straight row with ROW_GAP gaps, clear of furniture.
    Identical algorithm to batch_storage.py.
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    PARK_BASE    = [[200.0 + i * 10.0, 200.0, 100.0] for i in range(8)]
    MAX_ATTEMPTS = 200
    row_names    = [o.name for o in row_objs]

    # Park robot
    if robot is not None:
        robot.set_position_orientation(
            position=th.tensor([300.0, 300.0, 0.0], dtype=th.float32),
            orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        robot.keep_still()
        for _ in range(10):
            og.sim.step()

    # Park all row objects
    for i, obj in enumerate(row_objs):
        px, py, pz = PARK_BASE[i]
        obj.set_position_orientation(
            position=th.tensor([px, py, pz], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
    for _ in range(10):
        og.sim.step()

    # Measure each object one at a time
    surf_bmin, surf_bmax = get_aabb(surface_obj)
    anchor_x = float((surf_bmin[0] + surf_bmax[0]) / 2.0)
    anchor_y = float((surf_bmin[1] + surf_bmax[1]) / 2.0)
    measured_hx, measured_hy, settled_z = [], [], []

    for i, obj in enumerate(row_objs):
        obj.set_position_orientation(
            position=th.tensor([anchor_x, anchor_y, 5.0], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        for _ in range(5):
            og.sim.step()
        obj.states[object_states.OnTop].set_value(surface_obj, True)
        for _ in range(20):
            og.sim.step()
        pos, _ = obj.get_position_orientation()
        obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
        obj.keep_still()
        for _ in range(10):
            og.sim.step()
        hx, hy = _xy_half_extents(obj)
        z = float(obj.get_position_orientation()[0].cpu().numpy()[2])
        measured_hx.append(hx)
        measured_hy.append(hy)
        settled_z.append(z)
        print(f"[row] measured {obj.name}: hx={hx:.4f}  hy={hy:.4f}  z={z:.4f}")
        px, py, pz = PARK_BASE[i]
        obj.set_position_orientation(
            position=th.tensor([px, py, pz], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
        for _ in range(5):
            og.sim.step()

    furniture_aabbs = _collect_furniture_aabbs(scene_dict_room)
    surface_aabb    = (surf_bmin[:2].copy(), surf_bmax[:2].copy())
    sx_min, sx_max  = float(surf_bmin[0]), float(surf_bmax[0])
    sy_min, sy_max  = float(surf_bmin[1]), float(surf_bmax[1])

    best_angle  = rng.uniform(0.0, 2.0 * np.pi)
    best_anchor = np.array([anchor_x, anchor_y])
    found       = False

    for attempt in range(MAX_ATTEMPTS):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        axis  = np.array([np.cos(angle), np.sin(angle)])
        half_along = [abs(measured_hx[i] * axis[0]) + abs(measured_hy[i] * axis[1])
                      for i in range(len(row_objs))]
        total_span = sum(2 * h for h in half_along) + ROW_GAP * (len(row_objs) - 1)
        candidates = [np.array([anchor_x, anchor_y])]
        for _ in range(20):
            candidates.append(np.array([rng.uniform(sx_min, sx_max),
                                         rng.uniform(sy_min, sy_max)]))
        for cand in candidates:
            if _row_footprint_clear(cand, axis, half_along,
                                    measured_hx, measured_hy, total_span,
                                    furniture_aabbs, surface_aabb):
                best_angle, best_anchor, found = angle, cand, True
                print(f"[row] clear on attempt {attempt+1}  "
                      f"angle={np.degrees(angle):.1f}°")
                break
        if found:
            break

    if not found:
        print("[row] WARNING: no clear placement — using fallback")

    axis       = np.array([np.cos(best_angle), np.sin(best_angle)])
    half_along = [abs(measured_hx[i] * axis[0]) + abs(measured_hy[i] * axis[1])
                  for i in range(len(row_objs))]
    total_span = sum(2 * h for h in half_along) + ROW_GAP * (len(row_objs) - 1)
    row_start  = best_anchor - axis * (total_span / 2.0)
    cursor     = row_start + axis * half_along[0]
    centres    = [cursor.copy()]
    for i in range(1, len(row_objs)):
        cursor = cursor + axis * half_along[i-1] + axis * ROW_GAP + axis * half_along[i]
        centres.append(cursor.copy())

    for obj, centre, z in zip(row_objs, centres, settled_z):
        obj.set_position_orientation(
            position=th.tensor([float(centre[0]), float(centre[1]), z], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
        print(f"[row] placed {obj.name} → ({centre[0]:.3f}, {centre[1]:.3f}, {z:.3f})")
    for _ in range(30):
        og.sim.step()

    row_midpoint = (centres[0] + centres[-1]) / 2.0
    return {
        "row_axis":       axis.tolist(),
        "row_angle_deg":  float(np.degrees(best_angle)),
        "row_gap_m":      ROW_GAP,
        "row_order":      row_names,
        "row_centres":    [c.tolist() for c in centres],
        "row_half_along": half_along,
        "row_midpoint":   row_midpoint.tolist(),
        "total_span_m":   total_span,
        "anchor_xy":      best_anchor.tolist(),
        "collision_free": found,
        # also return furniture_aabbs and surface_aabb for extra placement
        "_furniture_aabbs": furniture_aabbs,
        "_surface_aabb":    surface_aabb,
        "_row_centres":     centres,
        "_settled_z":       settled_z,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extra object placement — near a target container, clear of everything
# ─────────────────────────────────────────────────────────────────────────────

def place_extra_near_container(extra_obj, target_container_obj,
                               surface_obj, floor_obj,
                               placed_aabbs: list,
                               furniture_aabbs: list,
                               surface_aabb: tuple,
                               rng: random.Random,
                               label: str) -> bool:
    """
    Place extra_obj so its bbox is EXTRA_NEAR_GAP metres (face-to-face) from
    target_container_obj, on one of its 4 cardinal sides (N/S/E/W).
    Checks clearance against placed_aabbs and furniture_aabbs.
    Appends the placed AABB to placed_aabbs on success.
    Returns True if a clear side was found, False if fallback was used.
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)

    # Park extra_obj, measure it alone
    extra_obj.set_position_orientation(
        position=th.tensor([250.0, 250.0, 100.0], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    extra_obj.keep_still()
    for _ in range(5):
        og.sim.step()

    surf_bmin, surf_bmax = get_aabb(surface_obj)
    anchor_x = float((surf_bmin[0] + surf_bmax[0]) / 2.0)
    anchor_y = float((surf_bmin[1] + surf_bmax[1]) / 2.0)

    extra_obj.set_position_orientation(
        position=th.tensor([anchor_x, anchor_y, 5.0], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    for _ in range(5):
        og.sim.step()
    extra_obj.states[object_states.OnTop].set_value(surface_obj, True)
    for _ in range(20):
        og.sim.step()
    pos, _ = extra_obj.get_position_orientation()
    extra_obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
    extra_obj.keep_still()
    for _ in range(10):
        og.sim.step()

    ehx, ehy = _xy_half_extents(extra_obj)
    ez = float(extra_obj.get_position_orientation()[0].cpu().numpy()[2])

    # Park back
    extra_obj.set_position_orientation(
        position=th.tensor([250.0, 250.0, 100.0], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    extra_obj.keep_still()
    for _ in range(5):
        og.sim.step()

    # Container face positions
    c_bmin, c_bmax = get_aabb(target_container_obj)
    cx_min, cx_max = float(c_bmin[0]), float(c_bmax[0])
    cy_min, cy_max = float(c_bmin[1]), float(c_bmax[1])
    c_cx = (cx_min + cx_max) / 2.0
    c_cy = (cy_min + cy_max) / 2.0

    candidates = {
        "N": np.array([c_cx,                           cy_max + EXTRA_NEAR_GAP + ehy]),
        "S": np.array([c_cx,                           cy_min - EXTRA_NEAR_GAP - ehy]),
        "E": np.array([cx_max + EXTRA_NEAR_GAP + ehx, c_cy]),
        "W": np.array([cx_min - EXTRA_NEAR_GAP - ehx, c_cy]),
    }

    surf_bmin_xy, surf_bmax_xy = surface_aabb

    def _clear(cx, cy):
        oxmin = cx - ehx - EXTRA_PLACE_MARGIN
        oxmax = cx + ehx + EXTRA_PLACE_MARGIN
        oymin = cy - ehy - EXTRA_PLACE_MARGIN
        oymax = cy + ehy + EXTRA_PLACE_MARGIN
        if (oxmin < float(surf_bmin_xy[0]) or oxmax > float(surf_bmax_xy[0]) or
                oymin < float(surf_bmin_xy[1]) or oymax > float(surf_bmax_xy[1])):
            return False
        for (pb, pm) in placed_aabbs:
            if _aabb_overlap_xy(oxmin, oxmax, oymin, oymax,
                                float(pb[0]), float(pm[0]),
                                float(pb[1]), float(pm[1])):
                return False
        for (fb, fm) in furniture_aabbs:
            if _aabb_overlap_xy(oxmin, oxmax, oymin, oymax,
                                float(fb[0]), float(fm[0]),
                                float(fb[1]), float(fm[1])):
                return False
        return True

    sides = list(candidates.keys())
    rng.shuffle(sides)
    chosen_xy   = None
    chosen_side = None
    for side in sides:
        cand = candidates[side]
        if _clear(float(cand[0]), float(cand[1])):
            chosen_xy   = cand
            chosen_side = side
            break

    clear_ok = chosen_xy is not None
    if not clear_ok:
        print(f"[extra_{label}] WARNING: all sides blocked — using N fallback")
        chosen_xy   = candidates["N"]
        chosen_side = "N"
    else:
        print(f"[extra_{label}] side={chosen_side}  "
              f"xy=({chosen_xy[0]:.3f}, {chosen_xy[1]:.3f})")

    extra_obj.set_position_orientation(
        position=th.tensor([float(chosen_xy[0]), float(chosen_xy[1]), ez],
                           dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    extra_obj.keep_still()
    for _ in range(20):
        og.sim.step()

    eb_min, eb_max = get_aabb(extra_obj)
    placed_aabbs.append((eb_min[:2].copy(), eb_max[:2].copy()))
    return clear_ok


# ─────────────────────────────────────────────────────────────────────────────
# Orbital views — locked geometry, same as batch_storage.py
# ─────────────────────────────────────────────────────────────────────────────

def render_orbital_views(run_dir: str,
                         all_objs_named: dict,
                         row_midpoint: np.ndarray,
                         n_views: int = ORBITAL_N_VIEWS,
                         image_prefix: str = "orbital",
                         radius: float = None,
                         cam_height: float = None,
                         centre_z: float = None) -> dict:
    short_names = tuple(all_objs_named.keys())
    og_names    = tuple(obj.name for obj in all_objs_named.values() if obj is not None)
    valid_shorts = tuple(sn for sn, obj in all_objs_named.items() if obj is not None)

    if radius is None or cam_height is None or centre_z is None:
        valid_objs = [o for o in all_objs_named.values() if o is not None]
        all_mins   = np.array([get_aabb(o)[0] for o in valid_objs])
        all_maxs   = np.array([get_aabb(o)[1] for o in valid_objs])
        scene_min  = all_mins.min(axis=0)
        scene_max  = all_maxs.max(axis=0)
        row_half   = float(np.sqrt((scene_max[0]-scene_min[0])**2 +
                                    (scene_max[1]-scene_min[1])**2)) / 2.0
        if radius    is None: radius    = row_half + ORBITAL_RADIUS_MARGIN
        if centre_z  is None: centre_z  = (float(scene_min[2]) + float(scene_max[2])) / 2.0
        if cam_height is None: cam_height = centre_z + radius

    look_target = np.array([float(row_midpoint[0]),
                             float(row_midpoint[1]),
                             centre_z])

    print(f"\n[orbital] prefix={image_prefix}  look={look_target.round(3)}  "
          f"r={radius:.3f}  h={cam_height:.3f}")

    results = {}
    for i in range(n_views):
        az_rad = 2.0 * np.pi * i / n_views
        az_deg = float(np.degrees(az_rad)) % 360.0
        eye    = np.array([
            float(row_midpoint[0]) + radius * np.cos(az_rad),
            float(row_midpoint[1]) + radius * np.sin(az_rad),
            cam_height,
        ])
        fname = f"{image_prefix}_{i}.png"
        fpath = os.path.join(run_dir, fname)
        pose  = set_cam_capture(eye, look_target, fpath, up=np.array([0., 0., 1.]))
        vis_by_og = seg_visibility(og_names)
        vis = {sn: vis_by_og[ogn]
               for sn, ogn in zip(valid_shorts, og_names)}
        results[str(i)] = {
            "image":           fname,
            "eye":             pose["position"],
            "quaternion_xyzw": pose["quaternion_xyzw"],
            "azimuth_deg":     az_deg,
            "cam_height":      cam_height,
            "radius":          radius,
            **{f"exist_{sn}": vis.get(sn, False) for sn in short_names},
        }

    return {"orbital_views": results,
            "radius": radius, "cam_height": cam_height, "centre_z": centre_z}


# ─────────────────────────────────────────────────────────────────────────────
# place_into_container — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def place_into_container(obj, container_obj, label, robot, eye_camera_key, run_dir) -> dict:
    img_pre_pickup = f"robot_eye_{label}_before_pickup.png"
    img_pre_place  = f"robot_eye_{label}_before_place.png"
    img_post_place = f"robot_eye_{label}_after_place.png"

    pose_obj_before  = snap(obj)
    pose_cont_before = snap(container_obj)
    robot_pose_before_pickup = get_robot_pose(robot) if robot else None
    if robot and eye_camera_key:
        capture_robot_eye(robot, eye_camera_key, os.path.join(run_dir, img_pre_pickup))

    grasp_ok = mock_grasp(robot, obj) if robot else False

    robot_pose_before_place = get_robot_pose(robot) if robot else None
    if robot and eye_camera_key:
        capture_robot_eye(robot, eye_camera_key, os.path.join(run_dir, img_pre_place))

    if robot and grasp_ok:
        mock_release(obj)

    method_used = "inside"
    ok = sample_kinematics("inside", obj, container_obj,
                           use_last_ditch_effort=True, use_trav_map=False)
    for _ in range(30):
        og.sim.step()
    bbox_ok = is_inside_bbox(obj, container_obj) if ok else False
    inside  = bbox_ok

    if not inside:
        print(f"[place {label}] inside failed — trying onTop")
        method_used = "onTop"
        ok = sample_kinematics("onTop", obj, container_obj,
                               use_last_ditch_effort=True, use_trav_map=False)
        for _ in range(30):
            og.sim.step()
        bbox_ok = is_inside_bbox(obj, container_obj) if ok else False
        inside  = bbox_ok

    pose_obj_after  = snap(obj)
    pose_cont_after = snap(container_obj)
    robot_pose_after_place = get_robot_pose(robot) if robot else None
    if robot and eye_camera_key:
        capture_robot_eye(robot, eye_camera_key, os.path.join(run_dir, img_post_place))

    obj_ext  = np.array(pose_obj_after["aabb_max"])  - np.array(pose_obj_after["aabb_min"])
    cont_ext = np.array(pose_cont_after["aabb_max"]) - np.array(pose_cont_after["aabb_min"])
    print(f"[place {label}] grasp={grasp_ok}  state={ok}  bbox={bbox_ok}  inside={inside}")

    return {
        "label":                    label,
        "grasp_success":            grasp_ok,
        "success":                  inside,
        "state_returned":           ok,
        "bbox_check":               bbox_ok,
        "method_used":              method_used,
        "object_pose_before":       pose_obj_before,
        "object_pose_after":        pose_obj_after,
        "container_pose_before":    pose_cont_before,
        "container_pose_after":     pose_cont_after,
        "robot_pose_before_pickup": robot_pose_before_pickup,
        "robot_pose_before_place":  robot_pose_before_place,
        "robot_pose_after_place":   robot_pose_after_place,
        "images": {
            "before_pickup": img_pre_pickup,
            "before_place":  img_pre_place,
            "after_place":   img_post_place,
        },
        "aabb": {
            "object":    {"min": pose_obj_after["aabb_min"],
                          "max": pose_obj_after["aabb_max"],
                          "extent": obj_ext.tolist()},
            "container": {"min": pose_cont_after["aabb_min"],
                          "max": pose_cont_after["aabb_max"],
                          "extent": cont_ext.tolist()},
            "aabb_difference_container_minus_object": (cont_ext - obj_ext).tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Global top-down — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def render_global_topdown(all_objs, run_dir, filename="global_topdown.png") -> dict:
    all_mins  = [get_aabb(o)[0] for o in all_objs]
    all_maxs  = [get_aabb(o)[1] for o in all_objs]
    scene_min = np.min(all_mins, axis=0)
    scene_max = np.max(all_maxs, axis=0)
    centre    = (scene_min + scene_max) / 2.0
    xy_diag   = float(np.sqrt((scene_max[0]-scene_min[0])**2 +
                               (scene_max[1]-scene_min[1])**2))
    top_z     = float(scene_max[2]) + max(TOP_DOWN_HEIGHT_PAD, xy_diag * 0.8)
    cam_eye   = np.array([centre[0], centre[1], top_z])
    cam_tgt   = np.array([centre[0], centre[1], centre[2]])
    cam_up    = np.array([0., 1., 0.])
    cam_pose  = set_cam_capture(cam_eye, cam_tgt,
                                os.path.join(run_dir, filename), up=cam_up)
    v_pos, v_ori = og.sim._viewer_camera.get_position_orientation()
    return {
        "image":           filename,
        "eye":             cam_eye.tolist(),
        "look_target":     cam_tgt.tolist(),
        "up_hint":         cam_up.tolist(),
        "position":        cam_pose["position"],
        "quaternion_xyzw": cam_pose["quaternion_xyzw"],
        "scene_aabb_min":  scene_min.tolist(),
        "scene_aabb_max":  scene_max.tolist(),
        "viewer_world_position":        v_pos.cpu().numpy().tolist(),
        "viewer_world_quaternion_xyzw": v_ori.cpu().numpy().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file",        type=str, required=True)
    parser.add_argument("--keys_json",        type=str, required=True)
    parser.add_argument("--object-inventory", type=str,
                        default="bddl3/bddl/generated_data/object_inventory.json")
    parser.add_argument("--room-objects",     type=str,
                        default="bddl3/bddl/generated_data/combined_room_object_list_future.json")
    parser.add_argument("--robot",            type=str, default="R1")
    parser.add_argument("--output_root",      type=str, default="renders_storage_multi")
    args = parser.parse_args()

    with open(args.task_file) as f:
        task_data = json.load(f)
    with open(args.keys_json) as f:
        all_keys = json.load(f)

    _basename  = os.path.basename(args.task_file)
    task_name  = _basename.replace(".json", "").strip("_").strip(".")
    scene_desp = task_data.get("scene_desp", {})
    scene_name, rooms, containee_spec, container_specs = parse_task(task_data, scene_desp)

    seed = hash(task_name) & 0xFFFFFFFF
    rng  = random.Random(seed)
    num_objects = 4 if rng.random() < 0.5 else 3

    print(f"\n{'='*70}")
    print(f"  Task        : {task_name}")
    print(f"  Scene       : {scene_name}  |  Rooms: {rooms}")
    print(f"  Containee   : {containee_spec.get('category')}")
    print(f"  Num objects : {num_objects}")
    print(f"{'='*70}\n")

    inventory = load_inventory(args.object_inventory)

    # ── Containee ─────────────────────────────────────────────────────────────
    containee_cat   = containee_spec["category"].replace(" (fillable)", "").strip()
    containee_inst  = containee_spec.get("instance", "")
    containee_model = get_model_id(containee_cat, containee_inst, inventory)
    containee_scale = 1.0
    containee_bbox  = get_bbox_size(containee_model, inventory)
    if containee_bbox is None:
        containee_bbox = np.array([0.1, 0.1, 0.1])

    # ── Containers ────────────────────────────────────────────────────────────
    resolved_containers = []
    for ci, spec in enumerate(container_specs):
        cat   = spec["category"].replace(" (fillable)", "").strip()
        inst  = spec.get("instance", "")
        model = get_model_id(cat, inst, inventory)
        fit   = spec.get("fit_check", "fit").lower()
        room  = parse_location_room(spec.get("location", []), scene_desp) or (rooms[0] if rooms else "")
        cont_bbox = get_bbox_size(model, inventory)
        if cont_bbox is None:
            cont_bbox = np.array([0.2, 0.2, 0.2])
        lo, hi = FIT_RATIO.get(fit, (1.2, 1.3))
        ratio  = rng.uniform(lo, hi)
        sc_xyz = (ratio * containee_bbox / cont_bbox).tolist()
        print(f"[container {ci}] {cat}/{model}  fit={fit}  ratio={ratio:.3f}")
        resolved_containers.append(dict(idx=ci, cat=cat, model=model, scale=sc_xyz,
                                        fit=fit, room=room,
                                        cont_bbox=cont_bbox.tolist(), ratio=ratio))

    large_cont = next((rc for rc in resolved_containers if rc["fit"] == "big"),
                      resolved_containers[0])
    lc_world = np.array(large_cont["cont_bbox"]) * np.array(large_cont["scale"])

    small_cont = None
    if num_objects == 4:
        small_cont = next((rc for rc in resolved_containers if rc["fit"] == "small"),
                          resolved_containers[0])

    # ── Extras ────────────────────────────────────────────────────────────────
    exclude_cats = {containee_cat} | {rc["cat"] for rc in resolved_containers}

    extra1_cat, extra1_model, extra1_bbox = pick_non_fillable_category(
        all_keys, exclude_cats, inventory, rng)
    exclude_cats.add(extra1_cat)

    extra2_cat, extra2_model, extra2_bbox = pick_non_fillable_category(
        all_keys, exclude_cats, inventory, rng)
    exclude_cats.add(extra2_cat)

    pair_ratio = rng.uniform(*EXTRA_PAIR_RATIO)
    tx_total   = pair_ratio * lc_world[0]
    ty         = pair_ratio * lc_world[1]
    tz         = pair_ratio * lc_world[2]

    def axis_scale(tx, ty, tz, bbox):
        return [float(tx / bbox[0]) if bbox[0] > 1e-6 else 1.0,
                float(ty / bbox[1]) if bbox[1] > 1e-6 else 1.0,
                float(tz / bbox[2]) if bbox[2] > 1e-6 else 1.0]

    extra1_scale = axis_scale(tx_total / 2, ty, tz / 2, extra1_bbox)
    extra2_scale = axis_scale(tx_total / 2, ty, tz / 2, extra2_bbox)

    extra3_cat = extra3_model = extra3_scale = extra3_bbox_inv = None
    if num_objects == 4 and small_cont is not None:
        extra3_cat, extra3_model, extra3_bbox_inv_arr = pick_non_fillable_category(
            all_keys, exclude_cats, inventory, rng)
        sc_world = np.array(small_cont["cont_bbox"]) * np.array(small_cont["scale"])
        e3_ratio = rng.uniform(*EXTRA3_FIT_RATIO)
        e3_bbox  = extra3_bbox_inv_arr if extra3_bbox_inv_arr is not None else np.array([0.1]*3)
        extra3_scale    = (sc_world / (e3_ratio * e3_bbox)).tolist()
        extra3_bbox_inv = e3_bbox.tolist()

    # ── Room / floor ──────────────────────────────────────────────────────────
    containee_room = (parse_location_room(containee_spec.get("location", []), scene_desp)
                      or (rooms[0] if rooms else ""))
    floor_name = find_floor_name(scene_name, containee_room, args.room_objects)
    if not floor_name:
        print(f"[ERROR] No floor for {scene_name}/{containee_room}")
        raise SystemExit(2)

    scene_dict     = load_scene_dict(scene_name)
    room_obj_names = get_room_object_names(scene_dict, containee_room)

    # ── OmniGibson config ─────────────────────────────────────────────────────
    objects_cfg = [
        {"type": "DatasetObject", "name": "obj_containee",
         "category": containee_cat, "model": containee_model,
         "position": [150., 100., 100.], "orientation": SQUARE_ORI,
         "scale": [containee_scale]*3},
        {"type": "DatasetObject", "name": "obj_extra1",
         "category": extra1_cat, "model": extra1_model,
         "position": [155., 100., 100.], "orientation": SQUARE_ORI,
         "scale": extra1_scale},
        {"type": "DatasetObject", "name": "obj_extra2",
         "category": extra2_cat, "model": extra2_model,
         "position": [160., 100., 100.], "orientation": SQUARE_ORI,
         "scale": extra2_scale},
    ]
    if num_objects == 4 and extra3_model:
        objects_cfg.append({
            "type": "DatasetObject", "name": "obj_extra3",
            "category": extra3_cat, "model": extra3_model,
            "position": [165., 100., 100.], "orientation": SQUARE_ORI,
            "scale": extra3_scale})
    for rc in resolved_containers:
        objects_cfg.append({
            "type": "DatasetObject", "name": f"obj_container_{rc['idx']}",
            "category": rc["cat"], "model": rc["model"],
            "position": [170. + rc["idx"]*5, 100., 100.],
            "orientation": SQUARE_ORI, "scale": rc["scale"]})

    cfg_file = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config   = yaml.safe_load(open(cfg_file))
    config["scene"]["scene_model"]                = scene_name
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [containee_room]
    config["objects"]                             = objects_cfg
    config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = 512
    config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"]  = 512

    env   = og.Environment(configs=config)
    scene = env.scene

    floor_obj     = scene.object_registry("name", floor_name)
    obj_containee = scene.object_registry("name", "obj_containee")
    obj_extra1    = scene.object_registry("name", "obj_extra1")
    obj_extra2    = scene.object_registry("name", "obj_extra2")
    obj_extra3    = scene.object_registry("name", "obj_extra3") if num_objects == 4 else None

    if floor_obj is None or obj_containee is None:
        print("[ERROR] Floor or containee not found.")
        raise SystemExit(2)

    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    robot = env.robots[0] if hasattr(env, "robots") and env.robots else None
    eye_camera_key = find_eye_camera_key(robot) if robot else ""
    if eye_camera_key:
        pos = robot._sensors[eye_camera_key].get_position()
        pos[2] -= 0.3; pos[0] += 0.05; pos[1] += 0.2
        robot._sensors[eye_camera_key].set_position(pos)
        for _ in range(100):
            og.sim.step()

    surface_obj, surface_top_z = pick_surface(scene, floor_obj, room_obj_names, rng)

    # ── Identify containers ───────────────────────────────────────────────────
    rc_fit   = next((rc for rc in resolved_containers if rc["fit"] == "fit"),   None)
    rc_big   = next((rc for rc in resolved_containers if rc["fit"] == "big"),   None)
    rc_small = next((rc for rc in resolved_containers if rc["fit"] == "small"), None)

    def get_cont(rc):
        return scene.object_registry("name", f"obj_container_{rc['idx']}") if rc else None

    obj_cont_fit   = get_cont(rc_fit)
    obj_cont_big   = get_cont(rc_big)
    obj_cont_small = get_cont(rc_small)

    # ── Place containee + 3 containers in row ─────────────────────────────────
    container_objs_ordered = [o for o in [obj_cont_fit, obj_cont_big, obj_cont_small]
                               if o is not None]
    row_objs = [obj_containee] + container_objs_ordered

    print(f"\n[row] Placing containee + {len(container_objs_ordered)} containers in row")
    row_meta = place_row(
        surface_obj, floor_obj, row_objs, robot,
        scene_dict.get(containee_room, {}), rng,
    )
    row_midpoint    = np.array(row_meta["row_midpoint"])
    furniture_aabbs = row_meta.pop("_furniture_aabbs")
    surface_aabb    = row_meta.pop("_surface_aabb")
    row_meta.pop("_row_centres", None)
    row_meta.pop("_settled_z",   None)

    # placed_aabbs: live AABBs of all row objects after placement
    placed_aabbs = []
    for obj in row_objs:
        bmin, bmax = get_aabb(obj)
        placed_aabbs.append((bmin[:2].copy(), bmax[:2].copy()))

    # ── Place extras near their target containers ─────────────────────────────
    extra1_near_cont = obj_cont_big if obj_cont_big is not None else container_objs_ordered[0]
    extra1_clear = place_extra_near_container(
        obj_extra1, extra1_near_cont,
        surface_obj, floor_obj,
        placed_aabbs, furniture_aabbs, surface_aabb, rng, "extra1",
    )

    extra2_near_cont = obj_cont_big if obj_cont_big is not None else container_objs_ordered[0]
    extra2_clear = place_extra_near_container(
        obj_extra2, extra2_near_cont,
        surface_obj, floor_obj,
        placed_aabbs, furniture_aabbs, surface_aabb, rng, "extra2",
    )

    extra3_clear = None
    extra3_near_cont = None
    if obj_extra3 is not None:
        extra3_near_cont = obj_cont_small if obj_cont_small is not None else container_objs_ordered[0]
        extra3_clear = place_extra_near_container(
            obj_extra3, extra3_near_cont,
            surface_obj, floor_obj,
            placed_aabbs, furniture_aabbs, surface_aabb, rng, "extra3",
        )

    # ── Build all_objs_named for orbital cameras ──────────────────────────────
    all_objs_named = {
        "obj_containee": obj_containee,
        "obj_extra1":    obj_extra1,
        "obj_extra2":    obj_extra2,
    }
    if obj_extra3:
        all_objs_named["obj_extra3"] = obj_extra3
    for rc in resolved_containers:
        n = f"obj_container_{rc['idx']}"
        all_objs_named[n] = scene.object_registry("name", n)

    all_obj_names  = tuple(all_objs_named.keys())
    all_scene_objs = [o for o in all_objs_named.values() if o is not None]

    # ── Output dir ────────────────────────────────────────────────────────────
    run_dir = os.path.join(args.output_root, task_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── Initial poses ─────────────────────────────────────────────────────────
    initial_poses = {n: snap(o) for n, o in all_objs_named.items() if o is not None}

    # ── Robot initial nav ─────────────────────────────────────────────────────
    if robot:
        nav_ok = mock_navigate(robot, obj_containee, floor_obj)
        print(f"[robot] Initial nav: {'OK' if nav_ok else 'FAILED'}")

    # ── Global top-down BEFORE ────────────────────────────────────────────────
    global_topdown_before = render_global_topdown(
        all_scene_objs, run_dir, "global_topdown_before.png")
    global_topdown_before["visibility"] = seg_visibility(all_obj_names)

    # ── Orbital views BEFORE all placements (locked geometry) ─────────────────
    orbital_before  = render_orbital_views(
        run_dir, all_objs_named, row_midpoint, image_prefix="orbital_before")
    orb_radius      = orbital_before["radius"]
    orb_cam_height  = orbital_before["cam_height"]
    orb_centre_z    = orbital_before["centre_z"]

    # ── place_into_container + orbital after each ─────────────────────────────
    def _place_and_orbit(obj, container_obj, label):
        print(f"\n[place] {obj.name} → {container_obj.name} ({label})")
        result = place_into_container(
            obj, container_obj, label, robot, eye_camera_key, run_dir)
        orb_after = render_orbital_views(
            run_dir, all_objs_named, row_midpoint,
            image_prefix=f"orbital_after_{label}",
            radius=orb_radius, cam_height=orb_cam_height, centre_z=orb_centre_z,
        )
        result["orbital_views_after"] = orb_after["orbital_views"]
        return result

    containee_result = None
    if obj_cont_fit:
        containee_result = _place_and_orbit(obj_containee, obj_cont_fit, "containee")

    extra1_result = None
    if obj_cont_big:
        extra1_result = _place_and_orbit(obj_extra1, obj_cont_big, "extra1")

    extra2_result = None
    if obj_cont_big:
        extra2_result = _place_and_orbit(obj_extra2, obj_cont_big, "extra2")

    extra3_result = None
    if num_objects == 4 and obj_extra3 and obj_cont_small:
        extra3_result = _place_and_orbit(obj_extra3, obj_cont_small, "extra3")

    for _ in range(20):
        og.sim.step()

    # ── Global top-down AFTER ─────────────────────────────────────────────────
    global_topdown_after = render_global_topdown(
        all_scene_objs, run_dir, "global_topdown_after.png")
    global_topdown_after["visibility"] = seg_visibility(all_obj_names)

    # ── Per-container top-down AFTER ──────────────────────────────────────────
    containers_to_render = [(rc_fit, obj_cont_fit, "fit"),
                            (rc_big, obj_cont_big, "big")]
    if num_objects == 4 and obj_extra3:
        containers_to_render.append((rc_small, obj_cont_small, "small"))

    container_cameras = {}
    for rc, obj_c, label in containers_to_render:
        if rc is None or obj_c is None:
            continue
        cb_min, cb_max = get_aabb(obj_c)
        centre   = (cb_min + cb_max) / 2.0
        top_z    = float(cb_max[2]) + CONTAINER_HEIGHT_PAD
        c_eye    = np.array([centre[0], centre[1], top_z])
        c_tgt    = np.array([centre[0], centre[1], centre[2]])
        img_path = os.path.join(run_dir, f"top_down_{label}.png")
        c_pose   = set_cam_capture(c_eye, c_tgt, img_path, up=np.array([0., 1., 0.]))
        v_pos, v_ori = og.sim._viewer_camera.get_position_orientation()
        container_cameras[label] = {
            "image":           f"top_down_{label}.png",
            "container_cat":   rc["cat"],
            "container_model": rc["model"],
            "eye":             c_eye.tolist(),
            "look_target":     c_tgt.tolist(),
            "position":        c_pose["position"],
            "quaternion_xyzw": c_pose["quaternion_xyzw"],
            "viewer_world_position":        v_pos.cpu().numpy().tolist(),
            "viewer_world_quaternion_xyzw": v_ori.cpu().numpy().tolist(),
        }

    vis = seg_visibility(all_obj_names)
    any_success = bool(containee_result and containee_result["success"])

    # ── Final snaps ───────────────────────────────────────────────────────────
    snap_containee  = snap(obj_containee)
    snap_extra1     = snap(obj_extra1)
    snap_extra2     = snap(obj_extra2)
    snap_extra3     = snap(obj_extra3) if obj_extra3 else None
    snap_cont_fit   = snap(obj_cont_fit)   if obj_cont_fit   else None
    snap_cont_big   = snap(obj_cont_big)   if obj_cont_big   else None
    snap_cont_small = snap(obj_cont_small) if obj_cont_small else None

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "task_name":     task_name,
        "scene":         scene_name,
        "room":          containee_room,
        "floor_name":    floor_name,
        "surface_name":  surface_obj.name,
        "surface_top_z": surface_top_z,
        "num_objects":   num_objects,

        "row_layout": {
            **row_meta,
            "initial_poses": initial_poses,
        },

        "extra_placement": {
            "extra1_near_container": extra1_near_cont.name,
            "extra1_clear":          extra1_clear,
            "extra2_near_container": extra2_near_cont.name,
            "extra2_clear":          extra2_clear,
            **({"extra3_near_container": extra3_near_cont.name,
                "extra3_clear":          extra3_clear}
               if obj_extra3 and extra3_near_cont else {}),
            "near_gap_m": EXTRA_NEAR_GAP,
        },

        "containee": {
            "cat":              containee_cat,
            "model":            containee_model,
            "scale":            [containee_scale]*3,
            "bbox_inventory":   containee_bbox.tolist(),
            "target_container": "fit",
            "placement":        containee_result,
            "final_pose":       snap_containee,
        },

        "containers": {
            rc["fit"]: {
                "idx":            rc["idx"],
                "cat":            rc["cat"],
                "model":          rc["model"],
                "scale":          rc["scale"],
                "bbox_inventory": rc["cont_bbox"],
                "scale_ratio":    rc["ratio"],
                "fit_check":      rc["fit"],
                "final_pose":     snap_c,
            }
            for rc, snap_c in [(rc_fit,   snap_cont_fit),
                               (rc_big,   snap_cont_big),
                               (rc_small, snap_cont_small)]
            if rc is not None
        },

        "extras": {
            "extra1": {
                "cat":              extra1_cat,
                "model":            extra1_model,
                "scale":            extra1_scale,
                "bbox_inventory":   extra1_bbox.tolist() if extra1_bbox is not None else None,
                "target_container": "big",
                "pair_ratio":       pair_ratio,
                "placement":        extra1_result,
                "final_pose":       snap_extra1,
            },
            "extra2": {
                "cat":              extra2_cat,
                "model":            extra2_model,
                "scale":            extra2_scale,
                "bbox_inventory":   extra2_bbox.tolist() if extra2_bbox is not None else None,
                "target_container": "big",
                "pair_ratio":       pair_ratio,
                "placement":        extra2_result,
                "final_pose":       snap_extra2,
            },
            **({"extra3": {
                "cat":              extra3_cat,
                "model":            extra3_model,
                "scale":            extra3_scale,
                "bbox_inventory":   extra3_bbox_inv,
                "target_container": "small",
                "placement":        extra3_result,
                "final_pose":       snap_extra3,
            }} if obj_extra3 else {}),
        },

        "initial_poses": initial_poses,

        "orbital_camera_geometry": {
            "radius":     orb_radius,
            "cam_height": orb_cam_height,
            "centre_z":   orb_centre_z,
            "n_views":    ORBITAL_N_VIEWS,
            "pitch_deg":  45.0,
        },
        "orbital_views_before": orbital_before["orbital_views"],

        "containee_placement_success": containee_result["success"]   if containee_result else False,
        "containee_inside_check":      containee_result["bbox_check"] if containee_result else False,
        "extra1_placement_success":    extra1_result["success"]       if extra1_result    else False,
        "extra1_inside_check":         extra1_result["bbox_check"]    if extra1_result    else False,
        "extra2_placement_success":    extra2_result["success"]       if extra2_result    else False,
        "extra2_inside_check":         extra2_result["bbox_check"]    if extra2_result    else False,
        "extra3_placement_success":    extra3_result["success"]       if extra3_result    else False,
        "extra3_inside_check":         extra3_result["bbox_check"]    if extra3_result    else False,

        "global_topdown_before": global_topdown_before,
        "global_topdown_after":  global_topdown_after,
        "container_cameras":     container_cameras,
        "visibility":            vis,
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] → {meta_path}")

    raise SystemExit(0 if any_success else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise SystemExit(2)