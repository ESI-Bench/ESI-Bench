"""
batch_stacking.py

ESI-Bench stacking dependency task.

GPT selects N_OBJECTS (2 or 3, default 3) household objects that are plausible
stacking candidates.  All objects are XY-scaled to a uniform footprint (the
largest natural XY extent among them), with Z scaled proportionally.

Every permutation of the objects is tried as a stacking order (bottom to top):
  2 objects -> 2 trials    (A-on-B,  B-on-A)
  3 objects -> 6 trials    (all orderings)

Per trial:
  1. Send all objects to outer-space staging positions.
  2. Place the bottom object on the floor via sample_kinematics("onTop", floor).
  3. (3-obj only) Place the middle object on top of the bottom.
  4. Place the top object on top of the previous layer.
  5. Settle physics (SETTLE_STEPS steps).
  6. Run is_on_top() for each stacked pair.
  7. Render N_AZIMUTHS side cameras + 1 top-down camera.
  8. Record pre-settle and post-settle AABB for every object.

Output:
  renders_stacking/<scene>/<room>/run_<NNNN>/
    <view_idx>.png
    metadata.json

Called once per (scene, room, run_idx) by batch_stacking.sh.
"""

import os
import sys
import json
import yaml
import argparse
import random
import itertools
import traceback
import numpy as np
import torch as th
import cv2

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from omnigibson.utils.object_state_utils import sample_kinematics
from scipy.spatial.transform import Rotation
from openai import OpenAI

# ── OmniGibson flags ──────────────────────────────────────────────────────────
gm.ENABLE_FLATCACHE        = False
gm.USE_GPU_DYNAMICS        = False
gm.ENABLE_OBJECT_STATES    = True
gm.ENABLE_TRANSITION_RULES = False

# ── OpenAI key ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Scene constants ────────────────────────────────────────────────────────────
SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
SCENES_DIR = "scenes5"

# ── Staging (outer space) ─────────────────────────────────────────────────────
STAGING_X_BASE   = 150.0
STAGING_Y        = 100.0
STAGING_Z        = 100.0
STAGING_X_STRIDE = 5.0

# ── Camera ────────────────────────────────────────────────────────────────────
N_AZIMUTHS   = 4
AZIMUTH_STEP = 360.0 / N_AZIMUTHS   # becomes 90.0
CAM_HEIGHT             = 0.60
CAM_TOPDOWN_Z          = 1.80
CAM_RADIUS_PAD         = 0.45
INITIAL_CAM_RADIUS_PAD = 0.45
INITIAL_CAM_HEIGHT     = 0.05
VIEWS_PER_TRIAL        = N_AZIMUTHS + 1

# ── Physics ───────────────────────────────────────────────────────────────────
SETTLE_STEPS      = 60
PLACEMENT_RETRIES = 5

# ── Stability thresholds ──────────────────────────────────────────────────────
Z_DROP_TOL   = 0.10
XY_DRIFT_TOL = 0.15

# ── Cluster placement ─────────────────────────────────────────────────────────
CLUSTER_EDGE_GAP = 0.08
WALL_MARGIN      = 0.10
CLUSTER_RETRIES  = 20

# ── Scene graph skip categories ───────────────────────────────────────────────
SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                   "door", "curtain", "electric_switch"}


# =============================================================================
# Scene graph helpers
# =============================================================================

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


def _aabb_overlaps_xy(ax_min, ax_max, bx_min, bx_max,
                      ay_min, ay_max, by_min, by_max) -> bool:
    return not (ax_max <= bx_min or bx_max <= ax_min or
                ay_max <= by_min or by_max <= ay_min)


def get_floor_bbox(room_objs: dict):
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        return (float(fb[0][0]), float(fb[1][0]),
                float(fb[0][1]), float(fb[1][1]))
    return -99.0, 99.0, -99.0, 99.0


def cluster_fits_in_room(obj_bboxes_xy: list, room_objs: dict) -> bool:
    """
    Check that every object bbox:
      (a) stays >= WALL_MARGIN inside the floor bbox
      (b) does not overlap any scene-graph object bbox (excluding structural cats)
    obj_bboxes_xy: list of (xmin, xmax, ymin, ymax)
    """
    fx_min, fx_max, fy_min, fy_max = get_floor_bbox(room_objs)

    for (xmin, xmax, ymin, ymax) in obj_bboxes_xy:
        if xmin < fx_min + WALL_MARGIN: return False
        if xmax > fx_max - WALL_MARGIN: return False
        if ymin < fy_min + WALL_MARGIN: return False
        if ymax > fy_max - WALL_MARGIN: return False

        for cat, bboxes in room_objs.items():
            if any(s in cat.lower() for s in SKIP_STRUCTURAL):
                continue
            for (bmin, bmax) in bboxes:
                if _aabb_overlaps_xy(xmin, xmax,
                                     float(bmin[0]), float(bmax[0]),
                                     ymin, ymax,
                                     float(bmin[1]), float(bmax[1])):
                    return False
    return True


# =============================================================================
# Low-level helpers
# =============================================================================

def get_aabb(obj):
    bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
    return bmin, bmax


def snap(obj) -> dict:
    pos, quat = obj.get_position_orientation()
    bmin, bmax = get_aabb(obj)
    return {
        "position":        pos.cpu().numpy().tolist(),
        "quaternion_xyzw": quat.cpu().numpy().tolist(),
        "aabb_min":        bmin.tolist(),
        "aabb_max":        bmax.tolist(),
    }


def aabb_to_dict(bmin, bmax) -> dict:
    centre = (bmin + bmax) / 2.0
    extent = bmax - bmin
    return {
        "min":    bmin.tolist(),
        "max":    bmax.tolist(),
        "centre": centre.tolist(),
        "extent": extent.tolist(),
    }


def aabb_half_diag_xy(bmin, bmax) -> float:
    ext = (bmax - bmin) / 2.0
    return float(np.sqrt(ext[0] ** 2 + ext[1] ** 2))


def look_at_quat(eye, target, up=np.array([0., 0., 1.])) -> np.ndarray:
    fwd = np.array(target, float) - np.array(eye, float)
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        return np.array([0., 0., 0., 1.])
    fwd /= n
    r = np.cross(fwd, up)
    if np.linalg.norm(r) < 1e-6:
        up = np.array([0., 1., 0.])
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


def set_cam_capture(eye: np.ndarray, target: np.ndarray, path: str,
                    up=np.array([0., 0., 1.])) -> dict:
    q = look_at_quat(eye, target, up=up)
    og.sim._viewer_camera.set_position_orientation(eye, q)
    do_capture(path)
    return {"position": eye.tolist(), "quaternion_xyzw": q.tolist()}


def send_to_staging(obj, slot: int):
    pos = th.tensor([
        STAGING_X_BASE + slot * STAGING_X_STRIDE,
        STAGING_Y,
        STAGING_Z,
    ], dtype=th.float32)
    obj.set_position_orientation(pos, th.tensor(SQUARE_ORI, dtype=th.float32))
    obj.keep_still()


def seg_visibility(obj_names: tuple) -> dict:
    for _ in range(100):
        og.sim.step()
    raw    = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    labels = " ".join(raw["info"]["idToLabels"].values())
    result = {name: (name in labels) for name in obj_names}
    print(f"[seg] {result}")
    return result


# =============================================================================
# Keys / inventory / GPT
# =============================================================================

def load_keys(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def sample_candidates(all_keys: list, seed: int, n: int = 200) -> list:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(n, len(all_keys)))


def get_model_for_category(category: str, inventory_path: str, seed: int) -> str:
    with open(inventory_path) as f:
        inventory = json.load(f)
    providers = inventory.get("providers", inventory)
    matches = [k for k in providers if k.startswith(f"{category}-")]
    if not matches:
        raise RuntimeError(
            f"Category '{category}' not found in inventory '{inventory_path}'.")
    rng = random.Random(seed)
    chosen   = rng.choice(matches)
    model_id = chosen.split("-", 1)[1]
    print(f"  [{category}] {len(matches)} model(s), picked model_id={model_id}")
    return model_id


def gpt_select_stackable(candidate_categories: list, n_objects: int) -> list:
    client = OpenAI(api_key=OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", ""))

    system_prompt = (
        "You are a robotics simulation assistant. "
        "Given a list of household object categories, select exactly "
        f"{n_objects} that are good stacking candidates: solid (not soft/flexible/liquid), "
        "flat-bottomed (stable base), table-top or hand-held size (not large furniture). "
        "They should be plausible to stack on top of one another. "
        "Good examples: book, box, can, mug, bowl, plate, container, tray, jar. "
        "Avoid: bags, pillows, clothing, large appliances, fragile thin objects. "
        f"Reply with ONLY a JSON array of exactly {n_objects} strings from the input list."
    )
    user_prompt = (
        f"Candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        f"Select {n_objects} good stacking objects."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=1.0,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    chosen = json.loads(raw)
    assert isinstance(chosen, list) and len(chosen) == n_objects, \
        f"GPT returned unexpected format: {raw}"
    for c in chosen:
        assert c in candidate_categories, f"GPT returned unknown category: {c}"

    print(f"[GPT] selected categories: {chosen}")
    return chosen


# =============================================================================
# XY scale unification
# =============================================================================

def compute_unified_scale(obj_categories: list, obj_models: list,
                          inventory_path: str) -> list:
    with open(inventory_path) as f:
        inventory = json.load(f)
    bbox_sizes = inventory.get("bounding_box_sizes", {})

    native_xy = []
    for cat, model in zip(obj_categories, obj_models):
        key  = f"{cat}-{model}"
        size = bbox_sizes.get(key)
        if size is not None:
            xy = max(float(size[0]), float(size[1]))
        else:
            print(f"  [scale] no inventory bbox for {key} — assuming 0.15 m")
            xy = 0.15
        native_xy.append(xy)

    target_xy = max(native_xy)
    print(f"[scale] inventory XY extents : {[f'{e:.3f}' for e in native_xy]}")
    print(f"[scale] unified target XY    : {target_xy:.3f} m")

    scales = []
    for xy in native_xy:
        ratio = target_xy / xy if xy > 1e-4 else 1.0
        scales.append([ratio, ratio, ratio])
        print(f"  ratio={ratio:.4f}")
    return scales


# =============================================================================
# Stability check
# =============================================================================

TILT_DOT_TOL = 0.9


def tilt_check(obj) -> tuple:
    _, quat = obj.get_position_orientation()
    quat_np  = quat.cpu().numpy()
    world_up = Rotation.from_quat(quat_np).apply(np.array([0., 0., 1.]))
    tilt_dot = float(np.dot(world_up, np.array([0., 0., 1.])))
    return tilt_dot >= TILT_DOT_TOL, tilt_dot


def xy_iou(bmin_a, bmax_a, bmin_b, bmax_b) -> float:
    ix_min = max(float(bmin_a[0]), float(bmin_b[0]))
    ix_max = min(float(bmax_a[0]), float(bmax_b[0]))
    iy_min = max(float(bmin_a[1]), float(bmin_b[1]))
    iy_max = min(float(bmax_a[1]), float(bmax_b[1]))

    inter_w = max(0.0, ix_max - ix_min)
    inter_h = max(0.0, iy_max - iy_min)
    inter   = inter_w * inter_h

    area_a = max(0.0, float(bmax_a[0]) - float(bmin_a[0])) * \
             max(0.0, float(bmax_a[1]) - float(bmin_a[1]))
    area_b = max(0.0, float(bmax_b[0]) - float(bmin_b[0])) * \
             max(0.0, float(bmax_b[1]) - float(bmin_b[1]))
    union  = area_a + area_b - inter

    return inter / union if union > 1e-8 else 0.0


XY_IOU_TOL = 0.1


def is_on_top(upper_obj, lower_obj) -> bool:
    u_bmin, u_bmax = get_aabb(upper_obj)
    l_bmin, l_bmax = get_aabb(lower_obj)

    z_ok  = float(u_bmin[2]) > float(l_bmin[2]) + 0.02
    iou   = xy_iou(u_bmin, u_bmax, l_bmin, l_bmax)
    xy_ok = iou >= XY_IOU_TOL

    print(f"  [is_on_top] {upper_obj.name} on {lower_obj.name}: "
          f"u_bmin_z={u_bmin[2]:.3f}  l_bmin_z={l_bmin[2]:.3f}  z_ok={z_ok}  "
          f"iou={iou:.3f}  xy_ok={xy_ok}")
    return z_ok and xy_ok


def is_stable(obj) -> tuple:
    ok, dot = tilt_check(obj)
    print(f"  [is_stable] {obj.name}: tilt_dot={dot:.3f}  stable={ok}")
    return ok, dot


# =============================================================================
# Camera rendering
# =============================================================================

def render_trial_views(run_dir: str, trial_idx: int,
                       perm_names: list,
                       stack_centre_xy: np.ndarray,
                       stack_half_diag: float,
                       stack_top_z: float) -> tuple:
    obj_names_tuple = tuple(perm_names)
    cx, cy  = float(stack_centre_xy[0]), float(stack_centre_xy[1])
    look_z  = max(stack_top_z / 2.0, CAM_HEIGHT * 0.5)
    target  = np.array([cx, cy, look_z])
    radius  = stack_half_diag + CAM_RADIUS_PAD
    poses   = {}
    exist_flags = {}

    order_str = "__".join(perm_names)
    prefix    = f"trial{trial_idx}_{order_str}"

    for step in range(N_AZIMUTHS):
        az_deg   = int(step * AZIMUTH_STEP)
        az_rad   = np.deg2rad(az_deg)
        eye = np.array([
            cx + radius * np.cos(az_rad),
            cy + radius * np.sin(az_rad),
            CAM_HEIGHT,
        ])
        fname = f"{prefix}_side_az{az_deg:03d}.png"
        print(f"  [cam trial{trial_idx} az={az_deg:3d}] eye={eye.round(3)}")
        pose = set_cam_capture(eye, target, os.path.join(run_dir, fname))
        poses[fname] = {**pose, "type": "side", "azimuth_deg": az_deg,
                        "look_target": target.tolist()}
        vis = seg_visibility(obj_names_tuple)
        for name in obj_names_tuple:
            exist_flags[f"exist_{name}_{fname}"] = vis[name]

    td_eye   = np.array([cx, cy, CAM_TOPDOWN_Z])
    td_up    = np.array([1., 0., 0.])
    fname_td = f"{prefix}_topdown.png"
    print(f"  [cam trial{trial_idx} topdown] eye={td_eye.round(3)}")
    pose_td = set_cam_capture(td_eye, np.array([cx, cy, 0.]),
                              os.path.join(run_dir, fname_td), up=td_up)
    poses[fname_td] = {**pose_td, "type": "top_down", "azimuth_deg": None}
    vis_td = seg_visibility(obj_names_tuple)
    for name in obj_names_tuple:
        exist_flags[f"exist_{name}_{fname_td}"] = vis_td[name]

    return poses, exist_flags


# =============================================================================
# Cluster placement + initial render
# =============================================================================

def place_cluster_on_floor(objs: list, floor_obj, natural_aabbs: dict,
                           room_objs: dict, seed: int) -> bool:
    """
    Place all objects on the floor as a tight row with CLUSTER_EDGE_GAP (0.08m)
    between adjacent bbox edges. Uses live AABBs for accurate spacing.

    After placing all objects, checks that every bbox stays inside the room
    (>= WALL_MARGIN from floor edges) and does not overlap any scene-graph object.
    Retries up to CLUSTER_RETRIES times if the check fails.

    Returns True if a valid placement was found, False otherwise.
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)

    for attempt in range(CLUSTER_RETRIES):
        print(f"  [cluster] placement attempt {attempt+1}/{CLUSTER_RETRIES}")

        # Re-stage all objects before each attempt
        for i, obj in enumerate(objs):
            send_to_staging(obj, i)
        for _ in range(10):
            og.sim.step()

        # Place first object on floor via sample_kinematics
        first = objs[0]
        placed = False
        for retry in range(PLACEMENT_RETRIES):
            try:
                ok = sample_kinematics(
                    "onTop", first, floor_obj,
                    use_last_ditch_effort=True, use_trav_map=False,
                )
                if ok:
                    placed = True
                    break
            except Exception as e:
                print(f"    [cluster] floor placement retry {retry}: {e}")

        if not placed:
            print("  [cluster] could not place first object — retrying")
            continue

        pos0, _ = first.get_position_orientation()
        first.set_position_orientation(pos0, SQUARE_ORI_T)
        first.keep_still()
        for _ in range(30):
            og.sim.step()

        # Read live AABB of first object
        bmin0, bmax0 = get_aabb(first)
        anchor_y        = float((bmin0[1] + bmax0[1]) / 2.0)
        anchor_z        = float(first.get_position_orientation()[0].cpu().numpy()[2])
        current_right_x = float(bmax0[0])

        print(f"  [cluster] {first.name}: bbox_x=[{bmin0[0]:.3f}, {bmax0[0]:.3f}]")

        all_bboxes_xy = [(float(bmin0[0]), float(bmax0[0]),
                          float(bmin0[1]), float(bmax0[1]))]

        # Place remaining objects face-to-face
        for obj in objs[1:]:
            nat_half_x = float(natural_aabbs[obj.name]["extent"][0]) / 2.0
            target_x   = current_right_x + CLUSTER_EDGE_GAP + nat_half_x

            obj.set_position_orientation(
                th.tensor([target_x, anchor_y, anchor_z + 0.5], dtype=th.float32),
                SQUARE_ORI_T,
            )
            obj.keep_still()
            for _ in range(5):
                og.sim.step()

            sample_kinematics(
                "onTop", obj, floor_obj,
                use_last_ditch_effort=True, use_trav_map=False,
            )
            pos_i, _ = obj.get_position_orientation()
            obj.set_position_orientation(pos_i, SQUARE_ORI_T)
            obj.keep_still()
            for _ in range(20):
                og.sim.step()

            # Read live AABB and correct to exact face-to-face gap
            bmin_i, bmax_i = get_aabb(obj)
            live_half_x   = float((bmax_i[0] - bmin_i[0]) / 2.0)
            live_centre_x = float((bmin_i[0] + bmax_i[0]) / 2.0)
            corrected_x   = current_right_x + CLUSTER_EDGE_GAP + live_half_x

            if abs(corrected_x - live_centre_x) > 1e-3:
                obj.set_position_orientation(
                    th.tensor([corrected_x, anchor_y, float(pos_i.cpu().numpy()[2])],
                              dtype=th.float32),
                    SQUARE_ORI_T,
                )
                obj.keep_still()
                for _ in range(10):
                    og.sim.step()
                bmin_i, bmax_i = get_aabb(obj)

            current_right_x = float(bmax_i[0])
            all_bboxes_xy.append((float(bmin_i[0]), float(bmax_i[0]),
                                  float(bmin_i[1]), float(bmax_i[1])))
            print(f"  [cluster] {obj.name}: bbox_x=[{bmin_i[0]:.3f}, {bmax_i[0]:.3f}]")

        # Validate: all objects inside room, no overlap with scene objects
        if cluster_fits_in_room(all_bboxes_xy, room_objs):
            print(f"  [cluster] valid placement found on attempt {attempt+1}")
            for _ in range(30):
                og.sim.step()
            return True
        else:
            print(f"  [cluster] out of room or overlapping scene objects — retrying")

    print(f"[cluster] WARNING: no valid placement after {CLUSTER_RETRIES} attempts — using last")
    for _ in range(30):
        og.sim.step()
    return False


def render_initial_views(run_dir: str, objs: list) -> tuple:
    """
    Render 4 side cameras + 1 top-down around the cluster before any stacking.
    Cameras use INITIAL_CAM_RADIUS_PAD (0.20m) and INITIAL_CAM_HEIGHT (1.20m),
    looking down at the cluster centre.
    """
    obj_names_tuple = tuple(o.name for o in objs)
    all_mins  = [get_aabb(o)[0] for o in objs]
    all_maxs  = [get_aabb(o)[1] for o in objs]
    scene_min = np.min(all_mins, axis=0)
    scene_max = np.max(all_maxs, axis=0)
    centre    = (scene_min + scene_max) / 2.0
    cx, cy    = float(centre[0]), float(centre[1])
    half_diag = float(np.sqrt(
        ((scene_max[0]-scene_min[0])/2)**2 +
        ((scene_max[1]-scene_min[1])/2)**2
    ))
    radius = half_diag + INITIAL_CAM_RADIUS_PAD
    look_z = float((scene_min[2] + scene_max[2]) / 2.0)
    target = np.array([cx, cy, look_z])

    poses       = {}
    exist_flags = {}

    for az_deg in [0, 90, 180, 270]:
        az_rad = np.deg2rad(az_deg)
        eye    = np.array([
            cx + radius * np.cos(az_rad),
            cy + radius * np.sin(az_rad),
            INITIAL_CAM_HEIGHT,
        ])
        fname = f"initial_side_az{az_deg:03d}.png"
        print(f"  [initial cam az={az_deg}] eye={eye.round(3)}  target={target.round(3)}")
        pose = set_cam_capture(eye, target, os.path.join(run_dir, fname))
        poses[fname] = {**pose, "type": "side", "azimuth_deg": az_deg}
        vis = seg_visibility(obj_names_tuple)
        for name in obj_names_tuple:
            exist_flags[f"exist_{name}_{fname}"] = vis[name]

    td_eye   = np.array([cx, cy, CAM_TOPDOWN_Z])
    td_up    = np.array([1., 0., 0.])
    fname_td = "initial_topdown.png"
    print(f"  [initial cam topdown] eye={td_eye.round(3)}")
    pose_td = set_cam_capture(td_eye, np.array([cx, cy, 0.]),
                              os.path.join(run_dir, fname_td), up=td_up)
    poses[fname_td] = {**pose_td, "type": "top_down", "azimuth_deg": None}
    vis_td = seg_visibility(obj_names_tuple)
    for name in obj_names_tuple:
        exist_flags[f"exist_{name}_{fname_td}"] = vis_td[name]

    return poses, exist_flags


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",            required=True)
    parser.add_argument("--room",             required=True)
    parser.add_argument("--floor",            required=True)
    parser.add_argument("--run_idx",          type=int, default=0)
    parser.add_argument("--keys_json",        default="keys.json")
    parser.add_argument("--object-inventory",
                        default="bddl3/bddl/generated_data/object_inventory.json")
    parser.add_argument("--robot",            default="R1")
    parser.add_argument("--output_root",      default="renders_stacking")
    parser.add_argument("--scenes_dir",       default="scenes5")
    parser.add_argument("--n_objects",        type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

    seed  = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF)
    n_obj = args.n_objects
    inventory_path = args.object_inventory

    run_dir = os.path.join(args.output_root,
                           args.scene, args.room, f"run_{args.run_idx:04d}")
    os.makedirs(run_dir, exist_ok=True)

    # Load scene graph for room bounds + overlap checks
    scene_dict = load_scene_dict(args.scene)
    room_objs  = get_room_objects(scene_dict, args.room)

    all_keys       = load_keys(args.keys_json)
    candidates     = sample_candidates(all_keys, seed=seed)
    obj_categories = gpt_select_stackable(candidates, n_objects=n_obj)

    obj_models = [
        get_model_for_category(cat, inventory_path, seed=seed + i)
        for i, cat in enumerate(obj_categories)
    ]
    obj_names  = [f"stack_obj{i}" for i in range(n_obj)]
    obj_scales = compute_unified_scale(obj_categories, obj_models, inventory_path)

    print(f"\n{'='*70}")
    print(f"  scene={args.scene}  room={args.room}  run={args.run_idx}")
    for name, cat, model, sc in zip(obj_names, obj_categories, obj_models, obj_scales):
        print(f"  {name}: {cat} / {model}  scale={[round(s,4) for s in sc]}")
    print(f"{'='*70}\n")

    cfg_file = os.path.join(og.example_config_path,
                            f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(cfg_file))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [args.room]

    config["objects"] = []
    for i, (cat, model, name, scale) in enumerate(
            zip(obj_categories, obj_models, obj_names, obj_scales)):
        config["objects"].append({
            "type":        "DatasetObject",
            "name":        name,
            "category":    cat,
            "model":       model,
            "scale":       scale,
            "position":    [STAGING_X_BASE + i * STAGING_X_STRIDE,
                            STAGING_Y, STAGING_Z],
            "orientation": SQUARE_ORI,
        })

    env   = og.Environment(configs=config)
    scene = env.scene

    objs = []
    for name in obj_names:
        o = scene.object_registry("name", name)
        if o is None:
            print(f"[ERROR] Object '{name}' not found in scene.")
            raise SystemExit(2)
        objs.append(o)

    floor_obj = scene.object_registry("name", args.floor)
    if floor_obj is None:
        print(f"[ERROR] Floor '{args.floor}' not found in scene.")
        raise SystemExit(2)

    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)

    for _ in range(30):
        og.sim.step()

    for i, obj in enumerate(objs):
        send_to_staging(obj, i)
    for _ in range(30):
        og.sim.step()

    natural_aabbs = {}
    for obj in objs:
        bmin, bmax = get_aabb(obj)
        natural_aabbs[obj.name] = aabb_to_dict(bmin, bmax)
    print("[natural AABBs]",
          {k: [f"{e:.3f}" for e in v["extent"]] for k, v in natural_aabbs.items()})

    print("\n[initial] Placing all objects in cluster for initial render ...")
    place_cluster_on_floor(objs, floor_obj, natural_aabbs, room_objs, seed)

    for _ in range(60):
        og.sim.step()

    initial_poses = {obj.name: snap(obj) for obj in objs}

    print("[initial] Rendering initial views ...")
    initial_cam_poses, initial_exist_flags = render_initial_views(run_dir, objs)
    print(f"[initial] {len(initial_cam_poses)} views saved.")

    perms = list(itertools.permutations(range(n_obj)))

    trials_meta = []
    any_stable  = False

    for trial_idx, perm in enumerate(perms):
        perm_names = [obj_names[i] for i in perm]
        perm_objs  = [objs[i]      for i in perm]

        print(f"\n{'='*60}")
        print(f"  Trial {trial_idx}: {' -> '.join(perm_names)}")
        print(f"{'='*60}")

        for i, obj in enumerate(objs):
            send_to_staging(obj, i)
        for _ in range(20):
            og.sim.step()

        bottom_obj = perm_objs[0]
        placed_ok  = False
        for attempt in range(PLACEMENT_RETRIES):
            try:
                ok = sample_kinematics(
                    "onTop", bottom_obj, floor_obj,
                    use_last_ditch_effort=True, use_trav_map=False,
                )
                if ok:
                    placed_ok = True
                    break
            except Exception as e:
                print(f"    [floor placement] attempt {attempt}: {e}")

        if not placed_ok:
            print(f"  [WARN] floor placement failed — skipping trial {trial_idx}")
            trials_meta.append({
                "trial_idx":   trial_idx,
                "order":       perm_names,
                "permutation": list(perm),
                "success":     False,
                "skip_reason": "floor_placement_failed",
            })
            continue

        pos, _ = bottom_obj.get_position_orientation()
        bottom_obj.set_position_orientation(
            pos, th.tensor(SQUARE_ORI, dtype=th.float32))
        bottom_obj.keep_still()
        for _ in range(SETTLE_STEPS):
            og.sim.step()

        b_bmin, b_bmax = get_aabb(bottom_obj)
        bottom_centre_xy = ((b_bmin + b_bmax) / 2.0)[:2].copy()
        print(f"  Bottom settled at centre={((b_bmin+b_bmax)/2.0).round(3)}")

        pre_aabbs = {bottom_obj.name: aabb_to_dict(b_bmin, b_bmax)}
        for obj in perm_objs[1:]:
            pre_aabbs[obj.name] = natural_aabbs[obj.name]

        stack_results   = []
        current_top_obj = bottom_obj
        all_placed      = True

        for upper_obj in perm_objs[1:]:
            print(f"  Stacking {upper_obj.name} on {current_top_obj.name} ...")

            ok_stack = False
            for attempt in range(PLACEMENT_RETRIES):
                try:
                    ok = sample_kinematics(
                        "onTop", upper_obj, current_top_obj,
                        use_last_ditch_effort=True, use_trav_map=False,
                    )
                    if ok:
                        ok_stack = True
                        break
                except Exception as e:
                    print(f"    [stack placement] attempt {attempt}: {e}")

            if not ok_stack:
                print(f"  [WARN] stack placement failed for {upper_obj.name}")
                stack_results.append({
                    "upper":        upper_obj.name,
                    "lower":        current_top_obj.name,
                    "placement_ok": False,
                })
                all_placed = False
                break

            pos_u, _ = upper_obj.get_position_orientation()
            upper_obj.set_position_orientation(
                pos_u, th.tensor(SQUARE_ORI, dtype=th.float32))
            upper_obj.keep_still()

            u_bmin_pre, u_bmax_pre = get_aabb(upper_obj)
            pre_aabbs[upper_obj.name] = aabb_to_dict(u_bmin_pre, u_bmax_pre)

            stack_results.append({
                "upper":        upper_obj.name,
                "lower":        current_top_obj.name,
                "placement_ok": True,
            })
            current_top_obj = upper_obj

        print(f"  Settling full stack ({SETTLE_STEPS} steps) ...")
        for _ in range(SETTLE_STEPS):
            og.sim.step()

        post_aabbs = {}
        for obj in perm_objs:
            bmin, bmax = get_aabb(obj)
            post_aabbs[obj.name] = aabb_to_dict(bmin, bmax)

        on_top_checks = []
        stable_checks = []
        if all_placed:
            for i in range(1, len(perm_objs)):
                on_top_checks.append(is_on_top(perm_objs[i], perm_objs[i-1]))
            for obj in perm_objs:
                stable_checks.append(is_stable(obj))

        all_on_top  = all(on_top_checks)
        all_upright = all(ok for ok, _ in stable_checks)
        all_stable  = all_placed and all_on_top and all_upright
        if all_stable:
            any_stable = True
        print(f"  -> all_stable={all_stable}  "
              f"on_top={on_top_checks}  "
              f"tilt_dots={[round(d,3) for _, d in stable_checks]}")

        stack_centre_xy = bottom_centre_xy
        stack_half_diag = max(
            aabb_half_diag_xy(
                np.array(post_aabbs[obj.name]["min"]),
                np.array(post_aabbs[obj.name]["max"]),
            )
            for obj in perm_objs
        )
        stack_top_z = max(post_aabbs[obj.name]["max"][2] for obj in perm_objs)

        print(f"  Rendering trial {trial_idx}: {' -> '.join(perm_names)} ...")
        cam_poses, exist_flags = render_trial_views(
            run_dir, trial_idx, perm_names,
            stack_centre_xy, stack_half_diag, stack_top_z,
        )

        trial_snaps = {obj.name: snap(obj) for obj in perm_objs}

        trials_meta.append({
            "trial_idx":     trial_idx,
            "order":         perm_names,
            "permutation":   list(perm),
            "success":       all_placed,
            "all_on_top":    all_on_top    if all_placed else False,
            "all_upright":   all_upright   if all_placed else False,
            "all_stable":    all_stable,
            "on_top_checks": on_top_checks,
            "tilt_dots":     {obj.name: round(d, 4)
                              for obj, (_, d) in zip(perm_objs, stable_checks)},
            "stack_results": stack_results,
            "pre_aabbs":     pre_aabbs,
            "post_aabbs":    post_aabbs,
            "snaps":         trial_snaps,
            "camera_poses":  cam_poses,
            "exist_flags":   exist_flags,
        })

    n_stable = sum(1 for t in trials_meta if t.get("all_stable", False))
    metadata = {
        "scene":               args.scene,
        "room":                args.room,
        "floor":               args.floor,
        "run_idx":             args.run_idx,
        "n_objects":           n_obj,
        "object_names":        obj_names,
        "object_categories":   obj_categories,
        "object_models":       obj_models,
        "object_scales":       obj_scales,
        "natural_aabbs":       natural_aabbs,
        "initial_poses":       initial_poses,
        "initial_cam_poses":   initial_cam_poses,
        "initial_exist_flags": initial_exist_flags,
        "n_trials":            len(trials_meta),
        "n_stable_trials":     n_stable,
        "views_per_trial":     VIEWS_PER_TRIAL,
        "total_views":         len(trials_meta) * VIEWS_PER_TRIAL,
        "trials":              trials_meta,
        "constants": {
            "N_AZIMUTHS":             N_AZIMUTHS,
            "CAM_HEIGHT":             CAM_HEIGHT,
            "CAM_TOPDOWN_Z":          CAM_TOPDOWN_Z,
            "CAM_RADIUS_PAD":         CAM_RADIUS_PAD,
            "INITIAL_CAM_RADIUS_PAD": INITIAL_CAM_RADIUS_PAD,
            "INITIAL_CAM_HEIGHT":     INITIAL_CAM_HEIGHT,
            "SETTLE_STEPS":           SETTLE_STEPS,
            "Z_DROP_TOL":             Z_DROP_TOL,
            "XY_DRIFT_TOL":           XY_DRIFT_TOL,
            "CLUSTER_EDGE_GAP":       CLUSTER_EDGE_GAP,
            "WALL_MARGIN":            WALL_MARGIN,
        },
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[meta] -> {meta_path}")
    print(f"  {n_stable}/{len(trials_meta)} trials produced a stable stack")

    og.clear()
    raise SystemExit(0 if any_stable else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise SystemExit(2)