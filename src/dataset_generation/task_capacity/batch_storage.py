"""
batch_storage.py
Doc 11 verbatim. Only change vs doc 11:
  Placement uses sample_kinematics("onTop", ..., use_last_ditch_effort=True,
  use_trav_map=False) for ALL fit types (fit/big/small), up to
  SMALL_KINEMATICS_TRIES attempts, with is_inside_bbox check and
  park() reset between attempts. Mirrors executor mock_put_down exactly.

Additional changes vs doc 11:
  - All 4 objects are placed in a single ROW on the surface:
      containee | container_A | container_B | container_C
    with 0.2 m bbox-to-bbox gap between every consecutive pair.
    The 3 containers are in a random order. The row direction is random.
  - 6 orbital SIDE views are rendered BEFORE pick-place begins: cameras
    orbit the combined AABB centre of the whole row at a height slightly
    above the tallest object. Each camera looks at the row midpoint.
    Per-object seg_visibility (exist flags) is stored for all 4 objects
    in every frame.
"""
import os
import sys
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
from omnigibson.utils.transform_utils import quat2euler

gm.ENABLE_FLATCACHE        = False
gm.USE_GPU_DYNAMICS        = False
gm.ENABLE_OBJECT_STATES    = True
gm.ENABLE_TRANSITION_RULES = False

SQUARE_ORI             = [0.0, 0.0, 0.0, 1.0]
TOP_DOWN_HEIGHT_PAD    = 0.8
SMALL_KINEMATICS_TRIES = 10
SCENES_DIR             = "scenes5"
NAV_DIST_THRESHOLD     = 2.0
MAX_NAV_ATTEMPTS       = 50

# Row layout parameters
ROW_GAP                = 0.2    # bbox-to-bbox gap (m) between every consecutive pair in the row

# Orbital pre-placement side views
ORBITAL_N_VIEWS        = 6      # evenly spaced azimuths (60 deg apart)
ORBITAL_RADIUS_MARGIN  = 0.6    # extra metres beyond the row half-length
# Camera is positioned at horizontal distance=radius and height=radius above
# the look-at centre, giving exactly 45 deg downward pitch from every azimuth.


# ─────────────────────────────────────────────────────────────────────────────
# Inventory / task helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_inventory(path: str) -> dict:
    """Return the full inventory dict (contains both 'providers' and 'bounding_box_sizes')."""
    fallbacks = [
        path,
        "bddl3/bddl/generated_data/object_inventory.json",
        os.path.join(os.path.dirname(__file__), "object_inventory.json"),
    ]
    for p in fallbacks:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise RuntimeError("object_inventory.json not found.")


def get_bbox_size(model_id: str, inventory: dict):
    """
    Return np.array([dx, dy, dz]) from inventory bounding_box_sizes at scale=1,
    or None if missing. Mirrors batch_containment.py get_bbox_size.
    """
    size = inventory.get("bounding_box_sizes", {}).get(model_id)
    if size is None:
        return None
    return np.array(size, dtype=float)


def get_model_id(category: str, instance_id: str, inventory: dict) -> str:
    providers = inventory.get("providers", inventory)
    clean    = category.replace(" (fillable)", "").strip()
    full_key = f"{clean}-{instance_id}"
    if full_key in providers:
        return instance_id
    matches = [k for k in providers if k.startswith(f"{clean}-")]
    if matches:
        model = random.choice(matches).split("-", 1)[1]
        print(f"[inventory] '{instance_id}' not found for '{clean}', using random model '{model}'")
        return model
    raise RuntimeError(f"No inventory entries for category '{clean}'.")


def find_floor_name(scene_name: str, room_name: str, room_objects_path: str) -> str:
    with open(room_objects_path) as f:
        data = json.load(f)
    scenes = data.get("scenes", data)
    for o in scenes.get(scene_name, {}).get(room_name, []):
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


def uniform_scale(spec: dict) -> float:
    if spec.get("type", "").lower() == "contents":
        return 1.0
    s = spec.get("scale", [1.0, 1.0, 1.0])
    if isinstance(s, (int, float)):
        v = float(s)
    elif isinstance(s, list):
        if len(s) == 1:
            v = float(s[0])
        elif len(s) >= 3:
            v = float(sum(s[:3]) / 3.0)
        else:
            v = 1.0
    else:
        v = 1.0
    return max(0.1, min(v, 5.0))


def parse_task(task_data: dict, scene_desp: dict):
    scene_name = task_data.get("scene_name", "Rs_int")
    if scene_name.endswith("_scene_dict.json"):
        scene_name = scene_name.replace("_scene_dict.json", "")
    answer      = task_data["parsed_ans"]
    new_objects = answer.get("new objects", [])
    rooms       = answer.get("Rooms", [])
    containee   = None
    containers  = []
    for spec in new_objects:
        t = spec.get("type", "").lower()
        if t == "contents":
            containee = spec
        elif t == "container":
            containers.append(spec)
    if containee is None:
        raise ValueError("No object with type='contents' in task JSON.")
    if not containers:
        raise ValueError("No objects with type='container' in task JSON.")
    return scene_name, rooms, containee, containers


# ─────────────────────────────────────────────────────────────────────────────
# Camera / render helpers (unchanged except FOV setter added below)
# ─────────────────────────────────────────────────────────────────────────────

def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
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



def capture(path: str):
    for _ in range(10):
        og.sim.render()
    img = (og.sim._viewer_camera.get_obs()[0]["rgb"]
           .cpu().numpy()[:, :, :3].astype(np.uint8))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[render] {path}")


def set_cam_capture(eye, target, path, up=np.array([0.0, 1.0, 0.0])) -> dict:
    q = look_at_quat(eye, target, up=up)
    og.sim._viewer_camera.set_position_orientation(eye, q)
    capture(path)
    return {"position": eye.tolist(), "quaternion_xyzw": q.tolist()}


def seg_visibility(obj_names: tuple) -> dict:
    """Step 100 frames then check seg_instance for each object name."""
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
    xy_inside = bool(
        s_min[0] >= c_min[0] and s_max[0] <= c_max[0] and
        s_min[1] >= c_min[1] and s_max[1] <= c_max[1]
    )
    z_inside = bool(s_min[2] < c_max[2] - 0.05)
    result = xy_inside and z_inside
    print(f"[inside_check] small=[{s_min.round(3)}, {s_max.round(3)}]  "
          f"container=[{c_min.round(3)}, {c_max.round(3)}]  "
          f"xy={xy_inside}  z={z_inside}  inside={result}")
    return result


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
        print(f"[surface] Table from scene graph: {chosen.name}  top_z={top_z:.3f}")
        return chosen, top_z
    _, fbmax = get_aabb(floor_obj)
    top_z = float(fbmax[2])
    print(f"[surface] No table in room scene graph — using floor  top_z={top_z:.3f}")
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


def park(obj, x: float, y: float, steps: int = 5):
    obj.set_position_orientation(
        position=th.tensor([x, y, 100.0], dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    for _ in range(steps):
        og.sim.step()


def find_eye_camera_key(robot) -> str:
    for key in robot._sensors:
        if "eyes:Camera:0" in key:
            return key
    return ""


def capture_robot_eye(robot, eye_camera_key: str, path: str):
    for _ in range(5):
        og.sim.render()
    img = (robot._sensors[eye_camera_key].get_obs()[0]["rgb"]
           .cpu().numpy()[:, :, :3].astype(np.uint8))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[robot_eye] Saved -> {path}")


def mock_navigate(robot, target_obj, floor_obj,
                  max_attempts: int = MAX_NAV_ATTEMPTS,
                  dist_threshold: float = NAV_DIST_THRESHOLD) -> bool:
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
    print(f"[nav] FAILED to reach {target_obj.name} within {dist_threshold}m after {max_attempts} attempts")
    return False


AG_JOINT_PRIM = None


def mock_grasp(robot, obj) -> bool:
    global AG_JOINT_PRIM
    grasp_point = robot.get_eef_position(robot.default_arm)
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
    for _ in range(10):
        og.sim.step()
    print(f"[grasp] Grasped {obj.name}")
    return True


def mock_release(obj):
    global AG_JOINT_PRIM
    if AG_JOINT_PRIM is not None:
        delete_or_deactivate_prim(str(AG_JOINT_PRIM.GetPrimPath()))
        AG_JOINT_PRIM = None
    obj.set_position_orientation([100.0, 100.0, 100.0], [0.0, 0.0, 0.0, 1.0])
    obj.visual_only = False
    obj.keep_still()
    for _ in range(10):
        og.sim.step()
    print(f"[release] Released {obj.name}")


def get_robot_camera_pose(robot, eye_camera_key: str) -> dict:
    """Snapshot robot base position + orientation only."""
    robot_pos, robot_ori = robot.get_position_orientation()
    return {
        "robot_position":        robot_pos.cpu().numpy().tolist(),
        "robot_quaternion_xyzw": robot_ori.cpu().numpy().tolist(),
    }


def _fail_json(run_dir, ci, rc, containee_cat, task_name,
               scene_name, room, floor_name, reason):
    meta = {
        "task_name":         task_name,
        "scene":             scene_name,
        "room":              room,
        "floor_name":        floor_name,
        "container_idx":     ci,
        "container_cat":     rc["cat"],
        "container_model":   rc["model"],
        "containee_cat":     containee_cat,
        "fit_check":         rc["fit"],
        "placement_success": False,
        "inside_check":      False,
        "error":             reason,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Row layout placement
# ─────────────────────────────────────────────────────────────────────────────


def _xy_half_extents(obj):
    """Return (half_x, half_y) from the object's live AABB."""
    bmin, bmax = get_aabb(obj)
    return (abs(float(bmax[0]) - float(bmin[0])) / 2.0,
            abs(float(bmax[1]) - float(bmin[1])) / 2.0)


def _row_footprint_clear(anchor_xy: np.ndarray,
                         axis: np.ndarray,
                         half_along: list,
                         measured_hx: list,
                         measured_hy: list,
                         total_span: float,
                         furniture_aabbs: list,
                         surface_aabb: tuple,
                         margin: float = 0.05) -> bool:
    """
    Return True if the full row footprint (all 4 objects placed along `axis`
    centred on `anchor_xy`) does not overlap any furniture AABB in XY and
    stays within the surface AABB (with `margin` inset).

    The footprint of each object i is approximated as its AABB projected onto
    the world XY plane: a rectangle of half-extents (measured_hx[i], measured_hy[i])
    centred at its planned XY centre. We check axis-aligned bounding box overlap
    (AABB vs AABB in XY) for each object against every furniture AABB.

    furniture_aabbs: list of (bmin_xy, bmax_xy) numpy arrays, one per piece
        of furniture visible in the scene (excluding floor, walls, ceilings).
    surface_aabb: (bmin, bmax) of the surface object (table or floor) in XY.
    """
    # Recompute centres from anchor
    row_start = anchor_xy - axis * (total_span / 2.0)
    centres   = []
    cursor    = row_start + axis * half_along[0]
    centres.append(cursor.copy())
    for i in range(1, len(half_along)):
        cursor = cursor + axis * half_along[i-1] + axis * 0.2 + axis * half_along[i]
        centres.append(cursor.copy())

    surf_bmin, surf_bmax = surface_aabb

    for i, (cx, cy) in enumerate([(c[0], c[1]) for c in centres]):
        hx = measured_hx[i]
        hy = measured_hy[i]
        obj_xmin = cx - hx - margin
        obj_xmax = cx + hx + margin
        obj_ymin = cy - hy - margin
        obj_ymax = cy + hy + margin

        # Must stay within surface bounds
        if (obj_xmin < float(surf_bmin[0]) or obj_xmax > float(surf_bmax[0]) or
                obj_ymin < float(surf_bmin[1]) or obj_ymax > float(surf_bmax[1])):
            return False

        # Must not overlap any furniture
        for (f_bmin, f_bmax) in furniture_aabbs:
            fx_min, fx_max = float(f_bmin[0]), float(f_bmax[0])
            fy_min, fy_max = float(f_bmin[1]), float(f_bmax[1])
            overlap = not (obj_xmax <= fx_min or obj_xmin >= fx_max or
                           obj_ymax <= fy_min or obj_ymin >= fy_max)
            if overlap:
                return False
    return True


def _collect_furniture_aabbs(scene_dict_room: dict) -> list:
    """
    Return a list of (bmin_xy, bmax_xy) numpy arrays for every object in the
    room's scenes5 scene dict, excluding structural categories.

    scene_dict_room: the per-room sub-dict from scenes5, i.e.
        scene_dict[room]  — format: { category: [[bmin, bmax], ...], ... }
    """
    SKIP = {"floors", "walls", "ceilings", "carpet", "window",
            "door", "curtain", "electric_switch"}
    result = []
    for cat, bboxes in scene_dict_room.items():
        if any(s in cat.lower() for s in SKIP):
            continue
        for (bmin, bmax) in bboxes:
            bmin_arr = np.array(bmin, dtype=float)
            bmax_arr = np.array(bmax, dtype=float)
            result.append((bmin_arr[:2].copy(), bmax_arr[:2].copy()))
    print(f"[row] furniture obstacles from scenes5: {len(result)} bboxes")
    return result


def place_row(scene, surface_obj, floor_obj,
              containers_ordered: list,
              obj_containee,
              robot,
              scene_dict_room: dict,
              rng: random.Random) -> dict:
    """
    Place all 4 objects in a strict straight row on surface_obj:

        containee | container_A | container_B | container_C

    with ROW_GAP metres (bbox-to-bbox) between every consecutive pair,
    guaranteed clear of all room furniture.

    Algorithm:
      0. Park robot and all task objects in outer space.
      1. Measure each object individually (one at a time, all others parked):
         OnTop drop → read live bbox → park back. Gets exact hx, hy, Z.
      2. Collect live AABBs of all room furniture (non-task, non-structural).
      3. Try up to MAX_ROW_ATTEMPTS (angle, anchor) combinations:
           - Sample a random row axis angle.
           - Compute the full row footprint centred on the surface centre.
           - Check every object's AABB against every furniture AABB in XY
             and against the surface boundary.
           - Accept the first combination that is fully clear.
         If nothing clears, fall back to the surface centre with the last angle.
      4. Teleport all 4 objects to their final positions and settle physics.
    """
    SQUARE_ORI_T  = th.tensor(SQUARE_ORI, dtype=th.float32)
    PARK_POSITIONS = [[200.0 + i * 10.0, 200.0, 100.0] for i in range(5)]
    MAX_ROW_ATTEMPTS = 200

    row_objs  = [obj_containee] + containers_ordered
    row_names = ["obj_containee"] + [o.name for o in containers_ordered]

    # ── Step 0: park robot and all objects ───────────────────────────────────
    if robot is not None:
        robot.set_position_orientation(
            position=th.tensor([300.0, 300.0, 0.0], dtype=th.float32),
            orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        robot.keep_still()
        for _ in range(10):
            og.sim.step()
        print("[row] Robot parked at (300, 300, 0)")

    for i, obj in enumerate(row_objs):
        px, py, pz = PARK_POSITIONS[i]
        obj.set_position_orientation(
            position=th.tensor([px, py, pz], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
    for _ in range(10):
        og.sim.step()

    # ── Step 1: measure each object individually ──────────────────────────────
    surf_bmin, surf_bmax = get_aabb(surface_obj)
    anchor_x = float((surf_bmin[0] + surf_bmax[0]) / 2.0)
    anchor_y = float((surf_bmin[1] + surf_bmax[1]) / 2.0)

    measured_hx = []
    measured_hy = []
    settled_z   = []

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

        px, py, pz = PARK_POSITIONS[i]
        obj.set_position_orientation(
            position=th.tensor([px, py, pz], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
        for _ in range(5):
            og.sim.step()

    # ── Step 2: collect furniture obstacles from scenes5 ─────────────────────
    furniture_aabbs = _collect_furniture_aabbs(scene_dict_room)
    surface_aabb    = (surf_bmin[:2].copy(), surf_bmax[:2].copy())

    # Surface usable XY extent (for random anchor sampling)
    sx_min, sx_max = float(surf_bmin[0]), float(surf_bmax[0])
    sy_min, sy_max = float(surf_bmin[1]), float(surf_bmax[1])

    # ── Step 3: find a clear (angle, anchor) combination ─────────────────────
    best_angle  = rng.uniform(0.0, 2.0 * np.pi)
    best_anchor = np.array([anchor_x, anchor_y])
    found       = False

    for attempt in range(MAX_ROW_ATTEMPTS):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        axis  = np.array([np.cos(angle), np.sin(angle)])

        # Exact along-axis half-extents for this angle
        half_along = [
            abs(measured_hx[i] * axis[0]) + abs(measured_hy[i] * axis[1])
            for i in range(len(row_objs))
        ]
        total_span = (sum(2 * h for h in half_along) +
                      ROW_GAP * (len(row_objs) - 1))

        # Try a few random anchor positions for this angle
        # (first try the surface centre, then random samples)
        anchor_candidates = [np.array([anchor_x, anchor_y])]
        for _ in range(20):
            ax = rng.uniform(sx_min, sx_max)
            ay = rng.uniform(sy_min, sy_max)
            anchor_candidates.append(np.array([ax, ay]))

        for candidate in anchor_candidates:
            if _row_footprint_clear(candidate, axis, half_along,
                                    measured_hx, measured_hy,
                                    total_span, furniture_aabbs,
                                    surface_aabb):
                best_angle  = angle
                best_anchor = candidate
                found = True
                print(f"[row] clear placement found on attempt {attempt+1}  "
                      f"angle={np.degrees(angle):.1f}°  "
                      f"anchor=({candidate[0]:.3f}, {candidate[1]:.3f})")
                break
        if found:
            break

    if not found:
        print(f"[row] WARNING: no collision-free placement found after "
              f"{MAX_ROW_ATTEMPTS} attempts — using surface centre fallback")

    # Final axis and half-along from chosen angle
    axis  = np.array([np.cos(best_angle), np.sin(best_angle)])
    half_along = [
        abs(measured_hx[i] * axis[0]) + abs(measured_hy[i] * axis[1])
        for i in range(len(row_objs))
    ]
    total_span = (sum(2 * h for h in half_along) +
                  ROW_GAP * (len(row_objs) - 1))

    row_start = best_anchor - axis * (total_span / 2.0)
    centres   = []
    cursor    = row_start + axis * half_along[0]
    centres.append(cursor.copy())
    for i in range(1, len(row_objs)):
        cursor = (cursor + axis * half_along[i - 1]
                  + axis * ROW_GAP + axis * half_along[i])
        centres.append(cursor.copy())

    print(f"[row] axis angle={np.degrees(best_angle):.1f}°  "
          f"total_span={total_span:.3f}m")
    for name, c in zip(row_names, centres):
        print(f"[row]   {name}: centre=({c[0]:.3f}, {c[1]:.3f})")

    # ── Step 4: teleport all objects to final positions ───────────────────────
    for obj, centre, z in zip(row_objs, centres, settled_z):
        obj.set_position_orientation(
            position=th.tensor([float(centre[0]), float(centre[1]), z],
                               dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
        print(f"[row] placed {obj.name} → ({centre[0]:.3f}, {centre[1]:.3f}, {z:.3f})")

    for _ in range(30):
        og.sim.step()

    row_midpoint = (centres[0] + centres[-1]) / 2.0

    return {
        "row_axis":        axis.tolist(),
        "row_angle_deg":   float(np.degrees(best_angle)),
        "row_gap_m":       ROW_GAP,
        "row_order":       row_names,
        "row_centres":     [c.tolist() for c in centres],
        "row_half_along":  half_along,
        "row_midpoint":    row_midpoint.tolist(),
        "total_span_m":    total_span,
        "anchor_xy":       best_anchor.tolist(),
        "collision_free":  found,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orbital side views (before pick-place)
# ─────────────────────────────────────────────────────────────────────────────

def render_orbital_views(run_dir: str,
                         all_objs_named: dict,
                         row_midpoint: np.ndarray,
                         n_views: int = ORBITAL_N_VIEWS,
                         image_prefix: str = "orbital",
                         radius: float = None,
                         cam_height: float = None,
                         centre_z: float = None) -> dict:
    """
    Render `n_views` evenly-spaced views orbiting the row at 45 deg downward pitch.

    all_objs_named: dict of { short_name: og_object } for all 4 objects,
        in row order (containee first).
    row_midpoint: XY midpoint of the full row (look-at target XY).

    Camera geometry (identical for all azimuths):
      - look-at  = row_midpoint at the Z centre of the combined AABB
      - radius   = row XY half-length + ORBITAL_RADIUS_MARGIN
      - eye XY   = look-at XY + radius * [cos(az), sin(az)]
      - eye Z    = look-at Z  + radius
      → horizontal dist = radius, vertical offset = radius
      → downward pitch  = atan(radius / radius) = 45 deg exactly
      - up hint  = (0, 0, 1)

    Per-object seg_visibility is checked for all 4 objects in every frame.
    Returns orbital metadata dict.
    """
    short_names = tuple(all_objs_named.keys())
    og_names    = tuple(obj.name for obj in all_objs_named.values())

    # Combined AABB of all 4 objects — only used when geometry not pre-supplied
    if radius is None or cam_height is None or centre_z is None:
        all_mins  = np.array([get_aabb(o)[0] for o in all_objs_named.values()])
        all_maxs  = np.array([get_aabb(o)[1] for o in all_objs_named.values()])
        scene_min = all_mins.min(axis=0)
        scene_max = all_maxs.max(axis=0)

        row_half_len = float(np.sqrt((scene_max[0] - scene_min[0])**2 +
                                      (scene_max[1] - scene_min[1])**2)) / 2.0
        if radius is None:
            radius = row_half_len + ORBITAL_RADIUS_MARGIN
        if centre_z is None:
            centre_z = (float(scene_min[2]) + float(scene_max[2])) / 2.0
        if cam_height is None:
            cam_height = centre_z + radius

    look_target = np.array([float(row_midpoint[0]),
                             float(row_midpoint[1]),
                             centre_z])

    print(f"\n[orbital] look_target={look_target.round(3)}  "
          f"radius={radius:.3f}  cam_height={cam_height:.3f}  pitch=45deg")

    results = {}
    for i in range(n_views):
        az_rad = 2.0 * np.pi * i / n_views
        az_deg = float(np.degrees(az_rad)) % 360.0

        # Horizontal distance = radius, vertical height = centre_z + radius
        # → angle below horizontal = atan(radius / radius) = 45 deg
        eye = np.array([
            float(row_midpoint[0]) + radius * np.cos(az_rad),
            float(row_midpoint[1]) + radius * np.sin(az_rad),
            cam_height,
        ])

        fname = f"{image_prefix}_{i}.png"
        fpath = os.path.join(run_dir, fname)
        print(f"\n[orbital {i}] az={az_deg:.1f}°  eye={eye.round(3)}")

        pose = set_cam_capture(eye, look_target, fpath,
                               up=np.array([0.0, 0.0, 1.0]))

        # Visibility check: same pattern as batch_size.py
        vis_by_og_name = seg_visibility(og_names)
        vis = {sn: vis_by_og_name[ogn]
               for sn, ogn in zip(short_names, og_names)}
        print(f"[orbital {i}] visibility: {vis}")

        results[str(i)] = {
            "image":           fname,
            "eye":             pose["position"],
            "quaternion_xyzw": pose["quaternion_xyzw"],
            "azimuth_deg":     az_deg,
            "cam_height":      cam_height,
            "radius":          radius,
            **{f"exist_{sn}": vis[sn] for sn in short_names},
        }

    return {
        "orbital_views": results,
        "radius":        radius,
        "cam_height":    cam_height,
        "centre_z":      centre_z,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file",        type=str, required=True)
    parser.add_argument("--object-inventory", type=str,
                        default="bddl3/bddl/generated_data/object_inventory.json")
    parser.add_argument("--room-objects",     type=str,
                        default="bddl3/bddl/generated_data/combined_room_object_list_future.json")
    parser.add_argument("--robot",            type=str, default="R1")
    parser.add_argument("--output_root",      type=str, default="renders_storage")
    args = parser.parse_args()

    with open(args.task_file) as f:
        task_data = json.load(f)

    _basename  = os.path.basename(args.task_file)
    task_name  = _basename.replace(".json", "").strip("_").strip(".")
    scene_desp = task_data.get("scene_desp", {})
    scene_name, rooms, containee_spec, container_specs = parse_task(task_data, scene_desp)

    print(f"\n{'='*70}")
    print(f"  Task      : {task_name}")
    print(f"  Scene     : {scene_name}  |  Rooms: {rooms}")
    print(f"  Containee : {containee_spec.get('category')}  instance={containee_spec.get('instance')}")
    print(f"  Containers: {len(container_specs)}")
    print(f"{'='*70}\n")

    seed = hash(task_name) & 0xFFFFFFFF
    rng  = random.Random(seed)

    inventory = load_inventory(args.object_inventory)

    FIT_RATIO = {"small": (0.9, 0.95), "fit": (1.1, 1.2), "big": (1.4, 1.6)}

    containee_cat   = containee_spec["category"].replace(" (fillable)", "").strip()
    containee_inst  = containee_spec.get("instance", "")
    containee_model = get_model_id(containee_cat, containee_inst, inventory)
    containee_scale = 1.0

    containee_bbox = get_bbox_size(containee_model, inventory)
    if containee_bbox is None:
        containee_bbox = np.array([0.1, 0.1, 0.1])
        print(f"[containee] bbox not in inventory — using default {containee_bbox}")
    else:
        print(f"[containee] inventory bbox={containee_bbox.round(4)}")

    resolved = []
    for ci, spec in enumerate(container_specs):
        cat   = spec["category"].replace(" (fillable)", "").strip()
        inst  = spec.get("instance", "")
        model = get_model_id(cat, inst, inventory)
        fit   = spec.get("fit_check", "fit").lower()
        room  = (parse_location_room(spec.get("location", []), scene_desp)
                 or (rooms[0] if rooms else ""))
        cont_bbox = get_bbox_size(model, inventory)
        if cont_bbox is None:
            cont_bbox = np.array([0.2, 0.2, 0.2])
            print(f"[container {ci}] bbox not in inventory — using default {cont_bbox}")
        else:
            print(f"[container {ci}] inventory bbox={cont_bbox.round(4)}")
        lo, hi = FIT_RATIO.get(fit, (1.1, 1.2))
        ratio  = rng.uniform(lo, hi)
        sc_xyz = (ratio * containee_bbox / cont_bbox).tolist()
        print(f"[container {ci}] {cat}/{model}  fit={fit}  ratio={ratio:.3f}  "
              f"scale_xyz={[round(s,4) for s in sc_xyz]}")
        resolved.append(dict(idx=ci, cat=cat, model=model, scale=sc_xyz,
                             fit=fit, room=room,
                             cont_bbox=cont_bbox.tolist(),
                             ratio=ratio))

    containee_room = (
        parse_location_room(containee_spec.get("location", []), scene_desp)
        or (rooms[0] if rooms else "")
    )

    print(f"\n[room] Loading: {containee_room}")
    floor_name = find_floor_name(scene_name, containee_room, args.room_objects)
    if not floor_name:
        print(f"[ERROR] No floor for {scene_name}/{containee_room}")
        raise SystemExit(2)
    print(f"[floor] {floor_name}")

    scene_dict     = load_scene_dict(scene_name)
    room_obj_names = get_room_object_names(scene_dict, containee_room)
    print(f"[scene_graph] {len(room_obj_names)} objects in room '{containee_room}'")

    objects_cfg = [
        {
            "type":        "DatasetObject",
            "name":        "obj_containee",
            "category":    containee_cat,
            "model":       containee_model,
            "position":    [150.0, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale":       [containee_scale, containee_scale, containee_scale],
        }
    ]
    for rc in resolved:
        objects_cfg.append({
            "type":        "DatasetObject",
            "name":        f"obj_container_{rc['idx']}",
            "category":    rc["cat"],
            "model":       rc["model"],
            "position":    [150.0 + (rc["idx"] + 1) * 5, 100.0, 100.0],
            "orientation": SQUARE_ORI,
            "scale":       rc["scale"],
        })

    cfg_file = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(cfg_file))
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

    if floor_obj is None:
        print(f"[ERROR] Floor '{floor_name}' not found in scene.")
        raise SystemExit(2)
    if obj_containee is None:
        print(f"[ERROR] obj_containee not found in scene.")
        raise SystemExit(2)

    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    robot = env.robots[0] if hasattr(env, "robots") and env.robots else None
    eye_camera_key = find_eye_camera_key(robot) if robot is not None else ""
    if eye_camera_key:
        print(f"[robot] Eye camera key: {eye_camera_key}")
        position = robot._sensors[eye_camera_key].get_position()
        position[2] -= 0.3
        position[0] += 0.05
        position[1] += 0.2
        robot._sensors[eye_camera_key].set_position(position)
        for _ in range(100):
            og.sim.step()
    else:
        print("[robot] Eye camera not found — robot eye images will be skipped")

    # ── Pick surface ──────────────────────────────────────────────────────────
    surface_obj, surface_top_z = pick_surface(scene, floor_obj, room_obj_names, rng)

    # ── Row layout: randomise container order ─────────────────────────────────
    container_objs = [scene.object_registry("name", f"obj_container_{rc['idx']}")
                      for rc in resolved]
    container_order = list(range(len(container_objs)))
    rng.shuffle(container_order)
    containers_ordered = [container_objs[i] for i in container_order]

    print(f"\n[row] Placing row — container order: "
          f"{[container_objs[i].name for i in container_order]}")
    row_meta = place_row(
        scene, surface_obj, floor_obj,
        containers_ordered, obj_containee, robot,
        scene_dict.get(containee_room, {}), rng,
    )
    row_midpoint = np.array(row_meta["row_midpoint"])

    # ── Snapshot initial poses of all 4 objects ───────────────────────────────
    initial_poses = {
        "obj_containee": snap(obj_containee),
        **{f"obj_container_{rc['idx']}": snap(container_objs[rc['idx']])
           for rc in resolved},
    }

    # ── Robot: initial nav to containee ──────────────────────────────────────
    if robot is not None:
        nav_ok = mock_navigate(robot, obj_containee, floor_obj)
        if not nav_ok:
            print(f"[ERROR] Initial nav to containee failed — aborting.")
            raise SystemExit(2)
        print(f"[robot] Initial nav to containee: OK")

    # ── Orbital side views BEFORE pick-place ─────────────────────────────────
    # Build the shared output dir (task-level, not per-container)
    task_run_dir = os.path.join(args.output_root, task_name)
    os.makedirs(task_run_dir, exist_ok=True)

    # Mapping: short name → og object (for seg check)
    all_objs_named = {"obj_containee": obj_containee}
    for rc in resolved:
        all_objs_named[f"obj_container_{rc['idx']}"] = container_objs[rc['idx']]

    orbital_meta   = render_orbital_views(task_run_dir, all_objs_named, row_midpoint)
    orbital_radius    = orbital_meta["radius"]
    orbital_cam_height = orbital_meta["cam_height"]
    orbital_centre_z   = orbital_meta["centre_z"]
    print(f"[orbital] Done — {ORBITAL_N_VIEWS} side views saved to {task_run_dir}")

    # ── Per-container pick-place loop (unchanged) ─────────────────────────────
    any_success = False
    for rc in resolved:
        ci       = rc["idx"]
        fit_tag  = rc["fit"]
        obj_cont = container_objs[ci]
        run_dir  = os.path.join(args.output_root, task_name, f"{ci}_{fit_tag}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  Container [{ci}]: {rc['cat']}/{rc['model']}  fit={fit_tag}")
        print(f"{'─'*60}")

        if obj_cont is None:
            print(f"[ERROR] obj_container_{ci} not in scene.")
            _fail_json(run_dir, ci, rc, containee_cat, task_name,
                       scene_name, containee_room, floor_name, "container_not_registered")
            continue

        # ── Place containee on floor for "before" pose ────────────────────────
        ok2 = obj_containee.states[object_states.OnTop].set_value(floor_obj, True)
        print(f"[place] Containee OnTop(floor) before placement: {ok2}")
        for _ in range(20):
            og.sim.step()
        pos_e, _ = obj_containee.get_position_orientation()
        obj_containee.set_position_orientation(
            position=pos_e,
            orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        obj_containee.keep_still()
        for _ in range(10):
            og.sim.step()

        pose_container_before = snap(obj_cont)
        pose_containee_before = snap(obj_containee)

        # ── Robot nav to containee → grasp ────────────────────────────────────
        robot_nav_to_containee  = False
        robot_grasp_success     = False
        robot_pose_before_pickup = None
        robot_pose_before_place  = None
        robot_pose_after_place   = None

        if robot is not None:
            robot_nav_to_containee = mock_navigate(robot, obj_containee, floor_obj)
            robot_pose_before_pickup = get_robot_camera_pose(robot, eye_camera_key)
            if not robot_nav_to_containee:
                print(f"[ERROR] Nav to containee failed on container [{ci}] — aborting.")
                raise SystemExit(2)
            if eye_camera_key:
                capture_robot_eye(robot, eye_camera_key,
                                  os.path.join(run_dir, "robot_eye_before_pickup.png"))
            robot_grasp_success = mock_grasp(robot, obj_containee)
            if not robot_grasp_success:
                print(f"[ERROR] Grasp failed on container [{ci}] — aborting.")
                raise SystemExit(2)

        # ── Robot nav to container (holding containee) ────────────────────────
        robot_nav_to_container = False
        if robot is not None:
            robot_nav_to_container = mock_navigate(robot, obj_cont, floor_obj)
            robot_pose_before_place = get_robot_camera_pose(robot, eye_camera_key)
            if not robot_nav_to_container:
                print(f"[ERROR] Nav to container [{ci}] failed — aborting.")
                mock_release(obj_containee)
                raise SystemExit(2)
            if eye_camera_key:
                capture_robot_eye(robot, eye_camera_key,
                                  os.path.join(run_dir, "robot_eye_before_place.png"))
            mock_release(obj_containee)

        # ── Placement: sample_kinematics("onTop") for ALL fit types ──────────
        placement_success = False
        attempt_log       = []
        print(f"[place] {fit_tag} → sample_kinematics onTop × {SMALL_KINEMATICS_TRIES}")
        for attempt in range(1, SMALL_KINEMATICS_TRIES + 1):
            success = sample_kinematics(
                "onTop", obj_containee, obj_cont,
                use_last_ditch_effort=True, use_trav_map=False
            )
            for _ in range(30):
                og.sim.step()
            bbox_ok = is_inside_bbox(obj_containee, obj_cont) if success else False
            inside  = success and bbox_ok
            attempt_log.append({
                "attempt":        attempt,
                "method":         "sample_kinematics_onTop",
                "fit_tag":        fit_tag,
                "state_returned": success,
                "bbox_check":     bbox_ok,
                "success":        inside,
            })
            print(f"[place] attempt {attempt}: state={success}  bbox={bbox_ok}  inside={inside}")
            if inside:
                placement_success = True
                break
            park(obj_containee, 150.0, 100.0)

        # ── If placement failed, place containee onTop of container for render ─
        # Ensures the top-down image always shows both objects together.
        # Uses the same sample_kinematics("onTop") call as the rest of the script.
        if not placement_success:
            print("[place] placement failed — placing containee onTop of container for render")
            ok_fallback = sample_kinematics(
                "onTop", obj_containee, obj_cont,
                use_last_ditch_effort=True, use_trav_map=False
            )
            for _ in range(20):
                og.sim.step()
            pos_fb, _ = obj_containee.get_position_orientation()
            obj_containee.set_position_orientation(
                position=pos_fb,
                orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
            )
            obj_containee.keep_still()
            for _ in range(10):
                og.sim.step()
            print(f"[place] fallback onTop state={ok_fallback}")

        # ── Settle + after snapshots ──────────────────────────────────────────
        for _ in range(10):
            og.sim.step()
        pose_container_after = snap(obj_cont)
        pose_containee_after = snap(obj_containee)

        # ── Robot: eye after place ────────────────────────────────────────────
        if robot is not None:
            robot_pose_after_place = get_robot_camera_pose(robot, eye_camera_key)
            if eye_camera_key and robot_nav_to_container:
                capture_robot_eye(robot, eye_camera_key,
                                  os.path.join(run_dir, "robot_eye_after_place.png"))
        else:
            robot_pose_after_place = None

        cont_ext      = (np.array(pose_container_after["aabb_max"])
                         - np.array(pose_container_after["aabb_min"]))
        containee_ext = (np.array(pose_containee_after["aabb_max"])
                         - np.array(pose_containee_after["aabb_min"]))
        aabb_diff     = (cont_ext - containee_ext).tolist()

        # ── Top-down camera ───────────────────────────────────────────────────
        cb_min, cb_max = get_aabb(obj_cont)
        centre         = (cb_min + cb_max) / 2.0
        top_height     = float(cb_max[2]) + TOP_DOWN_HEIGHT_PAD
        cam_eye        = np.array([centre[0], centre[1], top_height])
        cam_target     = np.array([centre[0], centre[1], centre[2]])
        cam_up         = np.array([0.0, 1.0, 0.0])
        img_path = os.path.join(run_dir, "top_down.png")
        print(f"\n[camera] eye={cam_eye.round(3)}  target={cam_target.round(3)}")
        cam_pose = set_cam_capture(cam_eye, cam_target, img_path, up=cam_up)

        # ── Orbital side views AFTER placement ───────────────────────────────
        # Same geometry as the pre-placement views, saved into the per-container
        # run_dir with prefix "orbital_after_". The row_midpoint is recomputed
        # from the current live AABBs so it stays accurate even if objects shifted.
        post_orbital_meta = render_orbital_views(
            run_dir, all_objs_named, row_midpoint,
            image_prefix="orbital_after",
            radius=orbital_radius,
            cam_height=orbital_cam_height,
            centre_z=orbital_centre_z,
        )
        print(f"[orbital_after] Done — {ORBITAL_N_VIEWS} post-placement views "
              f"saved to {run_dir}")

        # ── Visibility ────────────────────────────────────────────────────────
        cont_name = f"obj_container_{ci}"
        vis = seg_visibility(("obj_containee", cont_name))
        containee_vis = vis.get("obj_containee", False)
        container_vis = vis.get(cont_name, False)
        print(f"[vis] containee={containee_vis}  container={container_vis}")

        if placement_success and containee_vis and container_vis:
            any_success = True

        # ── Metadata ──────────────────────────────────────────────────────────
        metadata = {
            "task_name":     task_name,
            "scene":         scene_name,
            "room":          containee_room,
            "floor_name":    floor_name,
            "surface_name":  surface_obj.name,
            "surface_top_z": surface_top_z,

            "container_idx":   ci,
            "container_cat":   rc["cat"],
            "container_model": rc["model"],
            "container_scale":        rc["scale"],
            "container_bbox_inventory": rc["cont_bbox"],
            "container_scale_ratio":    rc["ratio"],
            "containee_bbox_inventory": containee_bbox.tolist(),
            "containee_cat":   containee_cat,
            "containee_model": containee_model,
            "containee_scale": containee_scale,
            "fit_check":         fit_tag,
            "placement_success": placement_success,
            "inside_check":      placement_success,
            "attempt_log":       attempt_log,

            # Row layout info
            "row": {
                **row_meta,
                "initial_poses": initial_poses,
            },

            # Orbital views before placement (task-level dir)
            "orbital_views_dir":    task_run_dir,
            "orbital_views_before": orbital_meta["orbital_views"],

            # Orbital views after placement (per-container dir)
            "orbital_views_after":  post_orbital_meta["orbital_views"],

            "aabb": {
                "container": {
                    "min":    pose_container_after["aabb_min"],
                    "max":    pose_container_after["aabb_max"],
                    "extent": cont_ext.tolist(),
                },
                "containee": {
                    "min":    pose_containee_after["aabb_min"],
                    "max":    pose_containee_after["aabb_max"],
                    "extent": containee_ext.tolist(),
                },
                "aabb_difference_container_minus_containee": aabb_diff,
            },
            "poses": {
                "container_before": pose_container_before,
                "container_after":  pose_container_after,
                "containee_before": pose_containee_before,
                "containee_after":  pose_containee_after,
            },
            "camera": {
                "type":                       "top_down",
                "image":                      "top_down.png",
                "eye":                        cam_eye.tolist(),
                "look_target":                cam_target.tolist(),
                "up_hint":                    cam_up.tolist(),
                "position":                   cam_pose["position"],
                "quaternion_xyzw":            cam_pose["quaternion_xyzw"],
                "height_above_container_top": TOP_DOWN_HEIGHT_PAD,
            },
            "visibility": {
                "containee_in_frame": containee_vis,
                "container_in_frame": container_vis,
            },
            "robot": {
                "nav_to_containee_success": robot_nav_to_containee,
                "grasp_success":            robot_grasp_success,
                "nav_to_container_success": robot_nav_to_container,
                "nav_dist_threshold_m":     NAV_DIST_THRESHOLD,
                "robot_eye_before_pickup":  (
                    "robot_eye_before_pickup.png"
                    if (robot is not None and eye_camera_key and robot_nav_to_containee)
                    else None
                ),
                "robot_eye_before_place":   (
                    "robot_eye_before_place.png"
                    if (robot is not None and eye_camera_key and robot_nav_to_container)
                    else None
                ),
                "robot_eye_after_place":    (
                    "robot_eye_after_place.png"
                    if (robot is not None and eye_camera_key and robot_nav_to_container)
                    else None
                ),
                "pose_before_pickup": robot_pose_before_pickup,
                "pose_before_place":  robot_pose_before_place,
                "pose_after_place":   robot_pose_after_place,
            },
        }

        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[meta] → {meta_path}")

        # ── Park both before next container ───────────────────────────────────
        try:
            obj_containee.set_position_orientation(
                position=th.tensor([150.0, 100.0, 100.0], dtype=th.float32),
                orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
            )
            obj_containee.keep_still()
            obj_cont.set_position_orientation(
                position=th.tensor([150.0 + (ci + 1) * 5, 105.0, 100.0], dtype=th.float32),
                orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
            )
            obj_cont.keep_still()
            for _ in range(5):
                og.sim.step()
        except Exception as e:
            print(f"[park] Non-fatal error parking objects: {e}")

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