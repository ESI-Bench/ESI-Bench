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

View numbering:
  trial t, views = t * VIEWS_PER_TRIAL + 0 ... + (VIEWS_PER_TRIAL-1)
  First N_AZIMUTHS views are side cameras; last view is top-down.

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
SQUARE_ORI       = [0.0, 0.0, 0.0, 1.0]

# ── Staging (outer space) ─────────────────────────────────────────────────────
STAGING_X_BASE   = 150.0
STAGING_Y        = 100.0
STAGING_Z        = 100.0
STAGING_X_STRIDE = 5.0

# ── Camera ────────────────────────────────────────────────────────────────────
N_AZIMUTHS      = 8
AZIMUTH_STEP    = 360.0 / N_AZIMUTHS
CAM_HEIGHT      = 0.60
CAM_TOPDOWN_Z   = 1.80
CAM_RADIUS_PAD  = 0.45
VIEWS_PER_TRIAL = N_AZIMUTHS + 1        # 8 side + 1 top-down

# ── Physics ────────────────────────────────────────────────────────────────────
SETTLE_STEPS      = 60
PLACEMENT_RETRIES = 5

# ── Stability thresholds ───────────────────────────────────────────────────────
Z_DROP_TOL   = 0.10
XY_DRIFT_TOL = 0.15


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
    """
    Step 100 frames, then check seg_instance annotator for each object name.
    Returns {name: bool} — mirrors batch_distance.py exactly.
    """
    for _ in range(100):
        og.sim.step()
    raw    = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    labels = " ".join(raw["info"]["idToLabels"].values())
    result = {name: (name in labels) for name in obj_names}
    print(f"[seg] {result}")
    return result


# =============================================================================
# Keys / inventory / GPT  (mirrors batch_distance.py)
# =============================================================================

def load_keys(path: str) -> list:
    """keys.json is a plain list of category name strings, e.g. ["mug", "bowl", ...]"""
    with open(path) as f:
        return json.load(f)


def sample_candidates(all_keys: list, seed: int, n: int = 200) -> list:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(n, len(all_keys)))


def get_model_for_category(category: str, inventory_path: str, seed: int) -> str:
    """
    Search object_inventory.json for keys matching 'category-*' and
    return a randomly chosen model ID (the part after the hyphen).
    Mirrors batch_distance.py's get_model_for_category exactly.
    """
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
    """
    Ask GPT-4o to pick n_objects categories that are solid, flat-bottomed,
    table-top sized, and plausibly stackable on each other.
    candidate_categories is a plain list of category name strings.
    Returns a list of exactly n_objects category name strings.
    """
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
    """
    Compute per-object scale factors so all objects share the same XY footprint,
    matched to the largest natural XY extent in the set.

    Uses bounding_box_sizes from object_inventory.json (native bbox in metres).
    Falls back to scale=1.0 for any object without an inventory entry.

    Returns a list of [sx, sy, sz] scale lists, one per object.
    Z is scaled by the same ratio as XY to preserve proportions.
    """
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

# Minimum dot product between object's world-up and [0,0,1] to be considered flat.
# dot=1.0 means perfectly upright; 0.9 ≈ within ~26° tilt; 0.95 ≈ within ~18°.
TILT_DOT_TOL = 0.9


def tilt_check(obj) -> tuple:
    """
    Return (tilt_ok, tilt_dot).
    Rotates local Z=[0,0,1] by the object's world quaternion and checks
    how close it is to world Z=[0,0,1].  dot=1.0 = perfectly upright.
    """
    _, quat = obj.get_position_orientation()
    quat_np  = quat.cpu().numpy()   # [qx, qy, qz, qw]
    world_up = Rotation.from_quat(quat_np).apply(np.array([0., 0., 1.]))
    tilt_dot = float(np.dot(world_up, np.array([0., 0., 1.])))
    return tilt_dot >= TILT_DOT_TOL, tilt_dot


def xy_iou(bmin_a, bmax_a, bmin_b, bmax_b) -> float:
    """Compute IoU of two axis-aligned bounding boxes projected onto the XY plane."""
    ix_min = max(float(bmin_a[0]), float(bmin_b[0]))
    ix_max = min(float(bmax_a[0]), float(bmax_b[0]))
    iy_min = max(float(bmin_a[1]), float(bmin_b[1]))
    iy_max = min(float(bmax_a[1]), float(bmax_b[1]))

    inter_w = max(0.0, ix_max - ix_min)
    inter_h = max(0.0, iy_max - iy_min)
    inter   = inter_w * inter_h

    area_a = max(0.0, float(bmax_a[0]) - float(bmin_a[0])) *              max(0.0, float(bmax_a[1]) - float(bmin_a[1]))
    area_b = max(0.0, float(bmax_b[0]) - float(bmin_b[0])) *              max(0.0, float(bmax_b[1]) - float(bmin_b[1]))
    union  = area_a + area_b - inter

    return inter / union if union > 1e-8 else 0.0


# Minimum XY IoU between upper and lower footprints to be considered on top.
XY_IOU_TOL = 0.1   # at least 10% overlap required


def is_on_top(upper_obj, lower_obj) -> bool:
    """
    Geometric check: is upper_obj resting on top of lower_obj?

    Two criteria:
      1. Z position: upper_obj AABB min Z > lower_obj AABB min Z + 0.02
      2. XY overlap: IoU of upper and lower XY footprints >= XY_IOU_TOL
                     (upper footprint must meaningfully overlap lower footprint)
    """
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
    """
    Stability check for a single object in the stack.
    Returns (stable, tilt_dot).

    Stable = not tilted: object's world-up axis is within TILT_DOT_TOL of [0,0,1].
    """
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
    """
    Render N_AZIMUTHS side cameras + 1 top-down camera for the settled stack.
    Runs seg_instance visibility check after each capture.

    File naming:
      trial<T>_<order>_side_az<AAA>.png
      trial<T>_<order>_topdown.png

    Returns (camera_poses dict, exist_flags dict).
      exist_flags keys: exist_<obj_name>_<fname>  e.g. exist_stack_obj0_trial0_..._side_az000.png
    """
    obj_names_tuple = tuple(perm_names)
    cx, cy  = float(stack_centre_xy[0]), float(stack_centre_xy[1])
    look_z  = max(stack_top_z / 2.0, CAM_HEIGHT * 0.5)
    target  = np.array([cx, cy, look_z])
    radius  = stack_half_diag + CAM_RADIUS_PAD
    poses   = {}
    exist_flags = {}

    order_str = "__".join(perm_names)
    prefix    = f"trial{trial_idx}_{order_str}"

    # Side cameras
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

    # Top-down camera
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

# Gap between adjacent object edges in the initial cluster (metres)
CLUSTER_GAP = 0.08


def place_cluster_on_floor(objs: list, floor_obj, natural_aabbs: dict):
    """
    Place all objects on the floor as a tight row.
    Spacing between consecutive objects = half_x(left) + half_x(right) + CLUSTER_GAP,
    so edges are just CLUSTER_GAP apart regardless of object size.
    Uses sample_kinematics for the first object to find a valid floor position,
    then teleports remaining objects to computed XY positions.
    """
    def half_x(obj):
        return float(natural_aabbs[obj.name]["extent"][0]) / 2.0

    # Place first object on the floor
    first = objs[0]
    placed = False
    for attempt in range(PLACEMENT_RETRIES):
        try:
            ok = sample_kinematics(
                "onTop", first, floor_obj,
                use_last_ditch_effort=True, use_trav_map=False,
            )
            if ok:
                placed = True
                break
        except Exception as e:
            print(f"    [cluster] floor placement attempt {attempt}: {e}")

    if not placed:
        print("[cluster] WARNING: could not place first object on floor")
        return

    pos0, _ = first.get_position_orientation()
    first.set_position_orientation(pos0, th.tensor(SQUARE_ORI, dtype=th.float32))
    first.keep_still()
    for _ in range(30):
        og.sim.step()

    anchor_pos = first.get_position_orientation()[0].cpu().numpy()

    # Place remaining objects: right edge of prev + gap + half_x of next
    current_right_x = float(anchor_pos[0]) + half_x(objs[0])
    for obj in objs[1:]:
        target_x = current_right_x + half_x(obj) + CLUSTER_GAP
        target_y = float(anchor_pos[1])
        obj.set_position_orientation(
            th.tensor([target_x, target_y, float(anchor_pos[2]) + 0.5], dtype=th.float32),
            th.tensor(SQUARE_ORI, dtype=th.float32),
        )
        obj.keep_still()
        for _ in range(5):
            og.sim.step()
        sample_kinematics(
            "onTop", obj, floor_obj,
            use_last_ditch_effort=True, use_trav_map=False,
        )
        pos_i, _ = obj.get_position_orientation()
        obj.set_position_orientation(pos_i, th.tensor(SQUARE_ORI, dtype=th.float32))
        obj.keep_still()
        for _ in range(20):
            og.sim.step()
        final_x = float(obj.get_position_orientation()[0].cpu().numpy()[0])
        current_right_x = final_x + half_x(obj)
        print(f"  [cluster] {obj.name} placed at {obj.get_position_orientation()[0].cpu().numpy().round(3)}")

    # Final settle
    for _ in range(30):
        og.sim.step()
    print(f"  [cluster] anchor={anchor_pos.round(3)}")


def render_initial_views(run_dir: str, objs: list) -> tuple:
    """
    Render 4 side cameras around the cluster + 1 top-down, before any stacking.
    Files: initial_side_az000.png, initial_side_az090.png, ..., initial_topdown.png
    Runs seg_instance visibility check after each capture.
    Returns (camera_poses dict, exist_flags dict).
    """
    obj_names_tuple = tuple(o.name for o in objs)
    all_mins = [get_aabb(o)[0] for o in objs]
    all_maxs = [get_aabb(o)[1] for o in objs]
    scene_min = np.min(all_mins, axis=0)
    scene_max = np.max(all_maxs, axis=0)
    centre    = (scene_min + scene_max) / 2.0
    cx, cy    = float(centre[0]), float(centre[1])
    half_diag = float(np.sqrt(
        ((scene_max[0]-scene_min[0])/2)**2 +
        ((scene_max[1]-scene_min[1])/2)**2
    ))
    radius   = half_diag + CAM_RADIUS_PAD
    look_z   = float((scene_min[2] + scene_max[2]) / 2.0)
    target   = np.array([cx, cy, look_z])
    poses       = {}
    exist_flags = {}

    # 4 side cameras at 0, 90, 180, 270 degrees
    for az_deg in [0, 90, 180, 270]:
        az_rad = np.deg2rad(az_deg)
        eye    = np.array([
            cx + radius * np.cos(az_rad),
            cy + radius * np.sin(az_rad),
            CAM_HEIGHT,
        ])
        fname = f"initial_side_az{az_deg:03d}.png"
        print(f"  [initial cam az={az_deg}] eye={eye.round(3)}")
        pose = set_cam_capture(eye, target, os.path.join(run_dir, fname))
        poses[fname] = {**pose, "type": "side", "azimuth_deg": az_deg}
        vis = seg_visibility(obj_names_tuple)
        for name in obj_names_tuple:
            exist_flags[f"exist_{name}_{fname}"] = vis[name]

    # Top-down
    td_eye = np.array([cx, cy, CAM_TOPDOWN_Z])
    td_up  = np.array([1., 0., 0.])
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

    # Seed mixes run_idx + scene + room so every (scene, room, run) combination
    # gets a unique candidate subset — GPT won't keep picking the same objects.
    # Mirrors the pattern in batch_distance.py.
    seed  = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF)
    n_obj = args.n_objects
    inventory_path = args.object_inventory

    # Output dir
    run_dir = os.path.join(args.output_root,
                           args.scene, args.room, f"run_{args.run_idx:04d}")
    os.makedirs(run_dir, exist_ok=True)

    # GPT object selection
    # keys.json = plain list of category strings like ["mug", "bowl", "book", ...]
    # sample_candidates draws a different 200-subset per seed, so different
    # (scene, room, run_idx) combinations see different candidate pools.
    all_keys         = load_keys(args.keys_json)
    candidates       = sample_candidates(all_keys, seed=seed)   # 200 categories
    obj_categories   = gpt_select_stackable(candidates, n_objects=n_obj)

    # Look up one model per chosen category from object_inventory.json
    obj_models = [
        get_model_for_category(cat, inventory_path, seed=seed + i)
        for i, cat in enumerate(obj_categories)
    ]
    obj_names = [f"stack_obj{i}" for i in range(n_obj)]

    # Compute unified XY scales from inventory BEFORE loading the environment
    obj_scales = compute_unified_scale(obj_categories, obj_models, inventory_path)

    print(f"\n{'='*70}")
    print(f"  scene={args.scene}  room={args.room}  run={args.run_idx}")
    for name, cat, model, sc in zip(obj_names, obj_categories, obj_models, obj_scales):
        print(f"  {name}: {cat} / {model}  scale={[round(s,4) for s in sc]}")
    print(f"{'='*70}\n")

    # OmniGibson config
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

    # Retrieve objects and floor
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

    # Add seg_instance modality (needed for visibility checks)
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)

    # Initial settle
    for _ in range(30):
        og.sim.step()

    # Re-stage to get clean AABBs after settling
    for i, obj in enumerate(objs):
        send_to_staging(obj, i)
    for _ in range(30):
        og.sim.step()

    # Record natural (post-scale, pre-trial) AABB
    natural_aabbs = {}
    for obj in objs:
        bmin, bmax = get_aabb(obj)
        natural_aabbs[obj.name] = aabb_to_dict(bmin, bmax)
    print("[natural AABBs]",
          {k: [f"{e:.3f}" for e in v["extent"]] for k, v in natural_aabbs.items()})

    # ── Initial cluster render (all objects on floor, grouped together) ──────
    print("\n[initial] Placing all objects in cluster for initial render ...")
    place_cluster_on_floor(objs, floor_obj, natural_aabbs)

    # Snap poses after cluster placement on floor (before any stacking trials)
    initial_poses = {obj.name: snap(obj) for obj in objs}

    print("[initial] Rendering initial views ...")
    initial_cam_poses, initial_exist_flags = render_initial_views(run_dir, objs)
    print(f"[initial] {len(initial_cam_poses)} views saved.")

    # All permutations: (i0, i1, i2) = bottom -> middle -> top
    perms = list(itertools.permutations(range(n_obj)))

    trials_meta = []
    any_stable  = False

    for trial_idx, perm in enumerate(perms):
        perm_names  = [obj_names[i] for i in perm]
        perm_objs   = [objs[i]      for i in perm]

        print(f"\n{'='*60}")
        print(f"  Trial {trial_idx}: {' -> '.join(perm_names)}")
        print(f"{'='*60}")

        # 1. Stage all objects
        for i, obj in enumerate(objs):
            send_to_staging(obj, i)
        for _ in range(20):
            og.sim.step()

        # 2. Place bottom object on the floor
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

        # Square up orientation
        pos, _ = bottom_obj.get_position_orientation()
        bottom_obj.set_position_orientation(
            pos, th.tensor(SQUARE_ORI, dtype=th.float32))
        bottom_obj.keep_still()
        for _ in range(SETTLE_STEPS):
            og.sim.step()

        b_bmin, b_bmax = get_aabb(bottom_obj)
        bottom_centre_xy = ((b_bmin + b_bmax) / 2.0)[:2].copy()
        print(f"  Bottom settled at centre={((b_bmin+b_bmax)/2.0).round(3)}")

        # pre_aabbs: bottom measured now; others from natural AABB (still in staging)
        pre_aabbs = {bottom_obj.name: aabb_to_dict(b_bmin, b_bmax)}
        for obj in perm_objs[1:]:
            pre_aabbs[obj.name] = natural_aabbs[obj.name]

        # 3. Stack remaining objects — place only, no stability checks yet
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

            # Square up orientation
            pos_u, _ = upper_obj.get_position_orientation()
            upper_obj.set_position_orientation(
                pos_u, th.tensor(SQUARE_ORI, dtype=th.float32))
            upper_obj.keep_still()

            # Record pre-settle AABB and XY for later drift check
            u_bmin_pre, u_bmax_pre = get_aabb(upper_obj)
            pre_aabbs[upper_obj.name] = aabb_to_dict(u_bmin_pre, u_bmax_pre)

            stack_results.append({
                "upper":        upper_obj.name,
                "lower":        current_top_obj.name,
                "placement_ok": True,
            })
            current_top_obj = upper_obj

        # 4. Settle the entire stack together once all objects are placed
        print(f"  Settling full stack ({SETTLE_STEPS} steps) ...")
        for _ in range(SETTLE_STEPS):
            og.sim.step()

        # Post-settle AABBs
        post_aabbs = {}
        for obj in perm_objs:
            bmin, bmax = get_aabb(obj)
            post_aabbs[obj.name] = aabb_to_dict(bmin, bmax)

        # Final check on the complete settled stack — all objects checked together:
        #   (a) is_on_top  — every consecutive pair: Z and XY IoU check
        #   (b) is_stable  — every object: not tilted
        on_top_checks  = []   # bool per consecutive pair
        stable_checks  = []   # (bool, tilt_dot) per object
        if all_placed:
            for i in range(1, len(perm_objs)):
                upper = perm_objs[i]
                lower = perm_objs[i - 1]
                on_top_checks.append(
                    is_on_top(upper, lower))

            for obj in perm_objs:
                stable_checks.append(is_stable(obj))

        all_on_top = all(on_top_checks)
        all_upright = all(ok for ok, _ in stable_checks)
        all_stable = all_placed and all_on_top and all_upright
        if all_stable:
            any_stable = True
        print(f"  -> all_stable={all_stable}  "
              f"on_top={on_top_checks}  "
              f"tilt_dots={[round(d,3) for _, d in stable_checks]}")

        # 5. Camera geometry
        stack_centre_xy = bottom_centre_xy
        stack_half_diag = max(
            aabb_half_diag_xy(
                np.array(post_aabbs[obj.name]["min"]),
                np.array(post_aabbs[obj.name]["max"]),
            )
            for obj in perm_objs
        )
        stack_top_z = max(post_aabbs[obj.name]["max"][2] for obj in perm_objs)

        # 6. Render
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

    # Write metadata JSON
    n_stable = sum(1 for t in trials_meta if t.get("all_stable", False))
    metadata = {
        "scene":             args.scene,
        "room":              args.room,
        "floor":             args.floor,
        "run_idx":           args.run_idx,
        "n_objects":         n_obj,
        "object_names":      obj_names,
        "object_categories": obj_categories,
        "object_models":     obj_models,
        "object_scales":     obj_scales,
        "natural_aabbs":     natural_aabbs,
        "initial_poses":     initial_poses,
        "initial_cam_poses": initial_cam_poses,
        "initial_exist_flags": initial_exist_flags,
        "n_trials":          len(trials_meta),
        "n_stable_trials":   n_stable,
        "views_per_trial":   VIEWS_PER_TRIAL,
        "total_views":       len(trials_meta) * VIEWS_PER_TRIAL,
        "trials":            trials_meta,
        "constants": {
            "N_AZIMUTHS":      N_AZIMUTHS,
            "CAM_HEIGHT":      CAM_HEIGHT,
            "CAM_TOPDOWN_Z":   CAM_TOPDOWN_Z,
            "CAM_RADIUS_PAD":  CAM_RADIUS_PAD,
            "SETTLE_STEPS":    SETTLE_STEPS,
            "Z_DROP_TOL":      Z_DROP_TOL,
            "XY_DRIFT_TOL":    XY_DRIFT_TOL,
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