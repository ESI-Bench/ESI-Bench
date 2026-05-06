"""
batch_dependency.py

ESI-Bench: Procedural Sequencing / Action Order Inference task.

Setup:
  - GPT-4o picks a placement hierarchy:
      2-object: obj_A (2 instances) → obj_fixed (1 instance)
                e.g. plate (large/small) → bowl
      3-object: obj_A (2 instances) → obj_fixed (1 instance) → obj_C (2 instances)
                e.g. plate (large/small) → bowl → food (small/large)

  - Scaling:
      obj_A_correct : clearly fits (scale ratio 1.3–1.5x obj_fixed XY)
      obj_A_wrong   : barely too small (scale ratio 0.88–0.93x obj_fixed XY)
      obj_C_correct : clearly fits inside obj_fixed (scale ratio 0.55–0.70x)
      obj_C_wrong   : barely too large (scale ratio 1.05–1.12x)

  - Layout: all objects placed in a row on floor/table, obj_fixed in center,
    candidates spread left/right. 4 orbital views rendered at each stage.

  - Trial sequence (3-object):
      0. Initial scene  → 4 orbital renders + exist check
      1. Place A_wrong  → 4 orbital + on_top check → FAIL → remove
      2. Place A_correct→ 4 orbital + on_top check → PASS
      3. Place C_wrong  → 4 orbital + inside check → FAIL → remove
      4. Place C_correct→ 4 orbital + inside check → PASS

  - QA (from initial orbital view 0):
      2-object: "Which [A_cat] should you place [under/on] the [fixed_cat]?
                 The one to the LEFT or RIGHT of the [fixed_cat]?"
      3-object: same + "Which [C_cat] should you place in the [fixed_cat]?"
      Answer: spatial relation string, e.g. "left" or "right"

  - Metadata: all orbital poses, exist flags, placement results, QA, GT action sequence

Usage:
  python batch_dependency.py \
    --scene Beechwood_0_int \
    --room  living_room_0 \
    --floor floors_yrqekq_0 \
    --run_idx 0 \
    --keys_json keys.json \
    --robot R1 \
    --output_root renders_dependency \
    --n_objects 3
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
from scipy.spatial.transform import Rotation
from openai import OpenAI

gm.ENABLE_FLATCACHE        = False
gm.USE_GPU_DYNAMICS        = False
gm.ENABLE_OBJECT_STATES    = True
gm.ENABLE_TRANSITION_RULES = False

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
SQUARE_ORI   = [0.0, 0.0, 0.0, 1.0]
SCENES_DIR   = "scenes5"

STAGING_X_BASE   = 150.0
STAGING_Y        = 100.0
STAGING_Z        = 100.0
STAGING_X_STRIDE = 5.0

# Scale ratios
A_CORRECT_RATIO   = (1.30, 1.50)   # clearly large enough to support obj_fixed
A_WRONG_RATIO_SM  = (0.88, 0.93)   # too small — clearly cannot support B
A_WRONG_RATIO_LG  = (1.25, 1.35)   # too large relative to A_correct (ratio of A_correct_scale)
C_CORRECT_RATIO   = (0.55, 0.70)   # clearly fits on B
C_WRONG_RATIO_LG  = (1.10, 1.30)   # too large — spills over B
C_WRONG_RATIO_SM  = (0.65, 0.75)   # too small relative to C_correct (ratio of C_correct_scale)

ROW_GAP          = 0.20           # bbox-to-bbox gap between objects in row
WALL_MARGIN      = 0.10

# Orbital camera
N_ORBITS         = 4
ORBIT_RADIUS_PAD = 0.50
ORBIT_HEIGHT_PAD = 0.60

# Placement
SETTLE_STEPS     = 60
PLACEMENT_RETRIES = 5

# Stability / fit thresholds
ON_TOP_Z_TOL     = 0.02
ON_TOP_IOU_TOL   = 0.05
INSIDE_TOL       = 0.05

SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                   "door", "curtain", "electric_switch"}


# =============================================================================
# Scene graph helpers (copied from batch_size.py)
# =============================================================================

def load_scene_dict(scene_name):
    path = os.path.join(SCENES_DIR, f"{scene_name}_scene_dict.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene dict not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_room_objects(scene_dict, room):
    return scene_dict.get(room, {})


def get_floor_bbox(room_objs):
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        return float(fb[0][0]), float(fb[1][0]), float(fb[0][1]), float(fb[1][1])
    return -99.0, 99.0, -99.0, 99.0


def aabb_overlaps_xy(ax0, ax1, bx0, bx1, ay0, ay1, by0, by1):
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


# =============================================================================
# Basic helpers
# =============================================================================

def get_scene_objects(scene):
    raw = getattr(scene, "objects", [])
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def get_aabb(obj):
    bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
    return bmin, bmax


def snap(obj):
    pos, quat = obj.get_position_orientation()
    bmin, bmax = get_aabb(obj)
    return {
        "position":        [float(x) for x in pos.cpu().numpy()],
        "quaternion_xyzw": [float(x) for x in quat.cpu().numpy()],
        "aabb_min":        [float(x) for x in bmin],
        "aabb_max":        [float(x) for x in bmax],
    }


def step_env(env, n=10):
    idle = th.zeros(env.robots[0].action_dim, dtype=th.float32)
    for _ in range(int(n)):
        env.step(idle)


def send_to_staging(obj, slot):
    obj.set_position_orientation(
        position=th.tensor([STAGING_X_BASE + slot * STAGING_X_STRIDE,
                             STAGING_Y, STAGING_Z], dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    obj.keep_still()


# =============================================================================
# Camera helpers (copied from batch_size.py / batch_stacking.py)
# =============================================================================

def look_at_quat(eye, target, up=np.array([0., 0., 1.])):
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


def do_capture(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(10):
        og.sim.render()
    img = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[render] {path}")


def set_cam_capture(eye, target, path, up=np.array([0., 0., 1.])):
    q = look_at_quat(eye, target, up=up)
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(eye, dtype=th.float32),
        th.tensor(q,   dtype=th.float32),
    )
    do_capture(path)
    return {"position": list(eye), "quaternion_xyzw": q.tolist()}


def seg_visibility(obj_names):
    for _ in range(100):
        og.sim.step()
    raw    = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    labels = " ".join(raw["info"]["idToLabels"].values())
    result = {n: bool(n in labels) for n in obj_names}
    print(f"[seg] {result}")
    return result


def render_orbital(run_dir, stage_label, all_objs_named,
                   centre_xy, radius, cam_height, centre_z,
                   view_offset=0):
    """
    Render N_ORBITS views around centre_xy at fixed geometry.
    Returns list of view dicts with pose + exist flags.
    """
    obj_keys  = tuple(all_objs_named.keys())
    obj_names = tuple(o.name for o in all_objs_named.values())
    look_tgt  = np.array([centre_xy[0], centre_xy[1], centre_z])
    views     = []

    for i in range(N_ORBITS):
        az_rad = 2.0 * np.pi * i / N_ORBITS
        az_deg = float(np.degrees(az_rad))
        eye    = np.array([
            centre_xy[0] + radius * np.cos(az_rad),
            centre_xy[1] + radius * np.sin(az_rad),
            cam_height,
        ])
        fname = f"{stage_label}_orb{i}.png"
        fpath = os.path.join(run_dir, fname)
        pose  = set_cam_capture(eye, look_tgt, fpath)
        vis   = seg_visibility(obj_names)
        exist = {f"exist_{k}": vis[n]
                 for k, n in zip(obj_keys, obj_names)}
        views.append({
            "view_idx":      view_offset + i,
            "stage":         stage_label,
            "azimuth_deg":   az_deg,
            "filename":      fname,
            "pose":          pose,
            **exist,
        })
        print(f"[orbital {stage_label} az={az_deg:.0f}] {exist}")

    return views


# =============================================================================
# Placement checks (copied from batch_stacking.py)
# =============================================================================

def xy_iou(bmin_a, bmax_a, bmin_b, bmax_b):
    ix0 = max(float(bmin_a[0]), float(bmin_b[0]))
    ix1 = min(float(bmax_a[0]), float(bmax_b[0]))
    iy0 = max(float(bmin_a[1]), float(bmin_b[1]))
    iy1 = min(float(bmax_a[1]), float(bmax_b[1]))
    inter = max(0., ix1-ix0) * max(0., iy1-iy0)
    aa = max(0., float(bmax_a[0])-float(bmin_a[0])) * max(0., float(bmax_a[1])-float(bmin_a[1]))
    ab = max(0., float(bmax_b[0])-float(bmin_b[0])) * max(0., float(bmax_b[1])-float(bmin_b[1]))
    union = aa + ab - inter
    return inter / union if union > 1e-8 else 0.0


def check_on_top(upper, lower):
    """
    Check if upper fits on top of lower by XY containment:
    upper's XY footprint must be within lower's XY footprint.
    """
    u_bmin, u_bmax = get_aabb(upper)
    l_bmin, l_bmax = get_aabb(lower)

    u_hx = abs(float(u_bmax[0]) - float(u_bmin[0]))
    u_hy = abs(float(u_bmax[1]) - float(u_bmin[1]))
    l_hx = abs(float(l_bmax[0]) - float(l_bmin[0]))
    l_hy = abs(float(l_bmax[1]) - float(l_bmin[1]))

    # upper fits on lower if upper's XY spans are both smaller than lower's XY spans
    xy_ok = bool(u_hx <= l_hx and u_hy <= l_hy)

    print(f"[on_top] {upper.name} on {lower.name}: xy_ok={xy_ok} "
          f"u_xy=({u_hx:.3f},{u_hy:.3f}) l_xy=({l_hx:.3f},{l_hy:.3f})")
    return xy_ok


# check_inside removed — all placements use on_top_of


# =============================================================================
# Inventory / GPT
# =============================================================================

def load_inventory():
    paths = [
        "bddl3/bddl/generated_data/object_inventory.json",
        os.path.join(os.path.dirname(__file__), "object_inventory.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise RuntimeError("object_inventory.json not found")


def get_model(category, inventory, seed):
    providers = inventory.get("providers", inventory)
    matches   = [k for k in providers if k.startswith(f"{category}-")]
    if not matches:
        raise RuntimeError(f"Category '{category}' not in inventory")
    rng    = random.Random(seed)
    chosen = rng.choice(matches)
    model  = chosen.split("-", 1)[1]
    bbox   = inventory.get("bounding_box_sizes", {}).get(model)
    print(f"  [{category}] picked model={model}")
    return model, (np.array(bbox, dtype=float) if bbox else None)


def load_keys(path):
    with open(path) as f:
        return json.load(f)


def sample_200(all_keys, seed):
    return random.Random(seed).sample(all_keys, min(200, len(all_keys)))


def gpt_pick_hierarchy(candidate_categories, n_objects):
    """
    Ask GPT-4o to pick a placement hierarchy.
    Returns dict with:
      n_objects=2: {obj_A_cat, obj_fixed_cat, relation_A: "on_top_of"}
      n_objects=3: {obj_A_cat, obj_fixed_cat, obj_C_cat,
                    relation_A: "on_top_of", relation_C: "inside"}
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    if n_objects == 2:
        system = (
            "You are a robotics simulation assistant.\n"
            "Pick a 2-object placement hierarchy. The SIZE relationship is critical:\n"
            "  obj_A is LARGER/WIDER than obj_fixed. obj_fixed sits ON TOP OF obj_A.\n"
            "  Think: obj_A is the BASE, obj_fixed is placed ON obj_A.\n"
            "  Example: plate (obj_A, large flat base) + bowl (obj_fixed, smaller, sits on plate).\n"
            "  Example: tray (obj_A) + cup (obj_fixed).\n"
            "obj_A_cat: FLAT WIDE BASE — plate, tray, cutting_board, baking_sheet, platter.\n"
            "obj_fixed_cat: SMALLER object that sits on obj_A — bowl, cup, mug, pot, jar, bottle.\n"
            "  obj_fixed MUST be smaller in XY footprint than obj_A.\n"
            "FORBIDDEN: bags, pillows, cloth, furniture, large boxes, suitcase.\n"
            "All must appear verbatim in the candidate list.\n"
            'Reply ONLY: {"obj_A_cat": "...", "obj_fixed_cat": "...", '
            '"relation_A": "on_top_of", '
            '"description": "[obj_fixed] sits on top of [obj_A]"}'
        )
    else:
        system = (
            "You are a robotics simulation assistant.\n"
            "Pick a 3-object placement hierarchy. SIZE relationships are critical:\n"
            "  obj_A is the LARGEST — flat wide BASE. obj_fixed sits ON obj_A.\n"
            "  obj_C is the SMALLEST — sits ON TOP OF obj_fixed.\n"
            "  Think: plate (biggest) → bowl (medium, on plate) → apple (small, on bowl).\n"
            "obj_A_cat: FLAT WIDE BASE, LARGEST of the three "
            "— plate, tray, cutting_board, baking_sheet, platter.\n"
            "obj_fixed_cat: MEDIUM object, smaller than obj_A but larger than obj_C, "
            "with a flat top surface — bowl, pot, box, basket, crate.\n"
            "obj_C_cat: SMALL object, smallest of the three, rests on obj_fixed "
            "— apple, orange, lemon, egg, small_ball, rock, small_bottle.\n"
            "SIZE ORDER MUST BE: obj_A (biggest XY) > obj_fixed (medium XY) > obj_C (smallest XY).\n"
            "FORBIDDEN: bags, pillows, cloth, furniture, large appliances, suitcase.\n"
            "All must appear verbatim in the candidate list.\n"
            'Reply ONLY: {"obj_A_cat": "...", "obj_fixed_cat": "...", "obj_C_cat": "...", '
            '"relation_A": "on_top_of", "relation_C": "on_top_of", '
            '"description": "[obj_fixed] on [obj_A], [obj_C] on [obj_fixed]"}'
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Candidates:\n{json.dumps(candidate_categories)}"},
        ],
        temperature=0.3,
        max_tokens=128,
    )
    raw = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    result = json.loads(raw)
    print(f"[GPT] hierarchy: {result.get('description','')}")

    # Validate
    for key in (["obj_A_cat", "obj_fixed_cat"] +
                (["obj_C_cat"] if n_objects == 3 else [])):
        if result[key] not in candidate_categories:
            raise ValueError(f"GPT returned '{result[key]}' not in candidates")

    return result


# =============================================================================
# Object building
# =============================================================================

def build_cfg(name, category, model, idx, scale):
    return {
        "type":        "DatasetObject",
        "name":        name,
        "category":    category,
        "model":       model,
        "position":    [STAGING_X_BASE + idx * STAGING_X_STRIDE, STAGING_Y, STAGING_Z],
        "orientation": SQUARE_ORI,
        "scale":       scale if isinstance(scale, list) else [scale, scale, scale],
    }


# =============================================================================
# Row placement (adapted from batch_stacking.py cluster logic + batch_size.py)
# =============================================================================

def measure_obj_on_surface(obj, surface_obj, env):
    """Drop obj onto surface, measure live AABB half-extents and Z. Park back."""
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    surf_bmin, surf_bmax = get_aabb(surface_obj)
    anchor_x = float((surf_bmin[0] + surf_bmax[0]) / 2.0)
    anchor_y = float((surf_bmin[1] + surf_bmax[1]) / 2.0)

    obj.set_position_orientation(
        th.tensor([anchor_x, anchor_y, 5.0], dtype=th.float32), SQUARE_ORI_T)
    step_env(env, 5)
    obj.states[object_states.OnTop].set_value(surface_obj, True)
    step_env(env, 20)
    pos, _ = obj.get_position_orientation()
    obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
    obj.keep_still()
    step_env(env, 10)

    bmin, bmax = get_aabb(obj)
    hx = abs(float(bmax[0]) - float(bmin[0])) / 2.0
    hy = abs(float(bmax[1]) - float(bmin[1])) / 2.0
    z  = float(obj.get_position_orientation()[0].cpu().numpy()[2])
    return hx, hy, z, bmin, bmax


def place_row_on_surface(objs, surface_obj, floor_obj, room_objs, env, rng):
    """
    Place all objs in a straight row on surface_obj with ROW_GAP gaps.
    obj_fixed (index 0 of objs) goes in the center.
    Returns (row_centres, settled_zs, row_midpoint, axis).
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)

    # Park all
    for i, obj in enumerate(objs):
        send_to_staging(obj, i)
    step_env(env, 10)

    # Measure each
    measured = []
    for obj in objs:
        hx, hy, z, bmin, bmax = measure_obj_on_surface(obj, surface_obj, env)
        measured.append({"hx": hx, "hy": hy, "z": z, "bmin": bmin, "bmax": bmax})
        send_to_staging(obj, objs.index(obj))
        step_env(env, 5)

    surf_bmin, surf_bmax = get_aabb(surface_obj)
    # Use top surface XY bounds — shrink slightly to keep objects on surface
    surf_margin = 0.05
    fx_min = float(surf_bmin[0]) + surf_margin
    fx_max = float(surf_bmax[0]) - surf_margin
    fy_min = float(surf_bmin[1]) + surf_margin
    fy_max = float(surf_bmax[1]) - surf_margin
    anchor_x = (fx_min + fx_max) / 2.0
    anchor_y = (fy_min + fy_max) / 2.0
    print(f"[row] surface bbox x=[{fx_min:.3f},{fx_max:.3f}] y=[{fy_min:.3f},{fy_max:.3f}]")

    # Collect furniture bboxes for collision check
    furniture_bboxes = []
    for cat, bboxes in room_objs.items():
        if any(s in cat.lower() for s in SKIP_STRUCTURAL):
            continue
        for (bmin, bmax) in bboxes:
            furniture_bboxes.append((
                np.array(bmin[:2], dtype=float),
                np.array(bmax[:2], dtype=float),
            ))

    # Try random row angles until clear
    best_angle = 0.0
    found = False
    for attempt in range(200):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        axis  = np.array([np.cos(angle), np.sin(angle)])
        # half-extents along row axis
        half_along = [abs(m["hx"] * axis[0]) + abs(m["hy"] * axis[1])
                      for m in measured]
        total_span = sum(2 * h for h in half_along) + ROW_GAP * (len(objs) - 1)

        # Try a few anchor candidates
        candidates = [np.array([anchor_x, anchor_y])]
        for _ in range(20):
            candidates.append(np.array([
                rng.uniform(fx_min + WALL_MARGIN, fx_max - WALL_MARGIN),
                rng.uniform(fy_min + WALL_MARGIN, fy_max - WALL_MARGIN),
            ]))

        for cand in candidates:
            row_start = cand - axis * (total_span / 2.0)
            cursor    = row_start.copy() + axis * half_along[0]
            centres   = [cursor.copy()]
            for i in range(1, len(objs)):
                cursor = cursor + axis * half_along[i-1] + axis * ROW_GAP + axis * half_along[i]
                centres.append(cursor.copy())

            ok = True
            for i, c in enumerate(centres):
                cx, cy = float(c[0]), float(c[1])
                hx, hy = measured[i]["hx"], measured[i]["hy"]
                if (cx - hx < fx_min + WALL_MARGIN or cx + hx > fx_max - WALL_MARGIN or
                        cy - hy < fy_min + WALL_MARGIN or cy + hy > fy_max - WALL_MARGIN):
                    ok = False
                    break
                for (fb, fm) in furniture_bboxes:
                    if aabb_overlaps_xy(cx-hx, cx+hx, float(fb[0]), float(fm[0]),
                                        cy-hy, cy+hy, float(fb[1]), float(fm[1])):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best_angle  = angle
                best_anchor = cand
                best_centres = centres
                found = True
                break
        if found:
            print(f"[row] clear on attempt {attempt+1}  angle={np.degrees(angle):.1f}°")
            break

    if not found:
        print("[row] WARNING: no clear placement — using centre fallback")
        axis       = np.array([1.0, 0.0])
        half_along = [abs(m["hx"]) for m in measured]
        total_span = sum(2*h for h in half_along) + ROW_GAP*(len(objs)-1)
        row_start  = np.array([anchor_x, anchor_y]) - axis * (total_span/2.0)
        cursor     = row_start + axis * half_along[0]
        best_centres = [cursor.copy()]
        for i in range(1, len(objs)):
            cursor = cursor + axis*half_along[i-1] + axis*ROW_GAP + axis*half_along[i]
            best_centres.append(cursor.copy())
        best_anchor = np.array([anchor_x, anchor_y])
        best_angle  = 0.0

    # Teleport each object to its row position
    settled_zs = []
    for obj, c, m in zip(objs, best_centres, measured):
        obj.set_position_orientation(
            th.tensor([float(c[0]), float(c[1]), m["z"]], dtype=th.float32),
            SQUARE_ORI_T,
        )
        obj.keep_still()
        settled_zs.append(m["z"])
        print(f"[row] placed {obj.name} → ({c[0]:.3f}, {c[1]:.3f}, {m['z']:.3f})")
    step_env(env, 30)

    row_midpoint = (best_centres[0] + best_centres[-1]) / 2.0
    return best_centres, settled_zs, row_midpoint, np.array([np.cos(best_angle), np.sin(best_angle)])


# =============================================================================
# Spatial relation computation
# =============================================================================

def spatial_relation_in_view(obj_pos_world, ref_pos_world, cam_eye, cam_target):
    """
    Given world positions of obj and ref, and camera eye/target,
    compute whether obj is LEFT/RIGHT/ABOVE/BELOW of ref in camera image space.
    Returns one of: "left", "right", "above", "below"
    """
    # Build camera basis vectors
    fwd = np.array(cam_target) - np.array(cam_eye)
    fwd /= np.linalg.norm(fwd)
    up  = np.array([0., 0., 1.])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    cam_up /= np.linalg.norm(cam_up)

    delta = np.array(obj_pos_world) - np.array(ref_pos_world)
    horiz = float(np.dot(delta, right))   # positive = right in image
    vert  = float(np.dot(delta, cam_up))  # positive = up in image

    if abs(horiz) >= abs(vert):
        return "right" if horiz > 0 else "left"
    else:
        return "above" if vert > 0 else "below"


# =============================================================================
# Place / remove helpers
# =============================================================================

def do_place_on_top(obj, target, env, retries=10):
    """Place obj on top of target, retrying up to 10 times. Returns True if check passes."""
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    for attempt in range(retries):
        try:
            ok = sample_kinematics("onTop", obj, target,
                                   use_last_ditch_effort=True, use_trav_map=False)
            if ok:
                pos, _ = obj.get_position_orientation()
                obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
                obj.keep_still()
                step_env(env, SETTLE_STEPS)
                result = check_on_top(obj, target)
                print(f"[place_on_top] attempt {attempt+1}: {obj.name} on {target.name}: {result}")
                if result:
                    return True
        except Exception as e:
            print(f"[place_on_top] attempt {attempt+1} failed: {e}")
    print(f"[place_on_top] all 10 attempts failed for {obj.name} on {target.name}")
    return False


def do_place_on_top_of(obj, target, env, retries=PLACEMENT_RETRIES):
    """Place obj on top of target. Alias used for C objects (same as A)."""
    return do_place_on_top(obj, target, env, retries=retries)


def reset_to_original(obj, original_pose, env):
    """Reset obj to its original row position and orientation."""
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    pos = original_pose["position"]
    obj.set_position_orientation(
        position=th.tensor(pos, dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    obj.keep_still()
    step_env(env, 15)
    print(f"[reset] {obj.name} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


def reset_all_except(active_objs, all_objs_named, original_poses, env):
    """
    Reset all objects that are NOT in active_objs back to their original positions.
    active_objs: set of obj names currently involved in a placement (don't touch these).
    """
    for name, obj in all_objs_named.items():
        if name in active_objs:
            continue
        if name not in original_poses:
            continue
        reset_to_original(obj, original_poses[name], env)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",       required=True)
    parser.add_argument("--room",        required=True)
    parser.add_argument("--floor",       required=True)
    parser.add_argument("--run_idx",     type=int, default=0)
    parser.add_argument("--keys_json",   default="keys.json")
    parser.add_argument("--robot",       default="R1")
    parser.add_argument("--output_root", default="renders_dependency")
    parser.add_argument("--n_objects",   type=int, default=0, choices=[0, 2, 3],
                        help="0=random (default), 2 or 3 to fix")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 999
    rng     = random.Random(seed)

    # Randomize n_objects if not specified
    if args.n_objects == 0:
        args.n_objects = rng.choice([2, 3])
        print(f"[n_objects] randomly selected: {args.n_objects}")
    run_dir = os.path.join(args.output_root, args.scene,
                           f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  scene={args.scene}  room={args.room}  run={args.run_idx}")
    print(f"  n_objects={args.n_objects}  output={run_dir}")
    print(f"{'='*70}\n")

    # ── Load scene graph ──────────────────────────────────────────────────────
    scene_dict = load_scene_dict(args.scene)
    room_objs  = get_room_objects(scene_dict, args.room)

    # ── GPT picks hierarchy ───────────────────────────────────────────────────
    inventory  = load_inventory()
    all_keys   = load_keys(args.keys_json)
    sampled    = sample_200(all_keys, seed=seed)
    hierarchy  = gpt_pick_hierarchy(sampled, args.n_objects)

    A_cat    = hierarchy["obj_A_cat"]
    fix_cat  = hierarchy["obj_fixed_cat"]
    C_cat    = hierarchy.get("obj_C_cat")        # None for 2-object
    rel_A    = hierarchy["relation_A"]           # "on_top_of"
    rel_C    = hierarchy.get("relation_C")       # "inside" or None

    # ── Pick models ───────────────────────────────────────────────────────────
    fix_model,  fix_bbox  = get_model(fix_cat,  inventory, seed)
    A_model,    A_bbox    = get_model(A_cat,    inventory, seed + 1)
    if C_cat:
        C_model, C_bbox = get_model(C_cat, inventory, seed + 2)
    else:
        C_model = C_bbox = None

    # ── Compute scales from inventory bboxes ──────────────────────────────────
    # All scales are relative to obj_fixed XY footprint
    if fix_bbox is None:
        fix_bbox = np.array([0.20, 0.20, 0.15])
    if A_bbox is None:
        A_bbox = np.array([0.20, 0.20, 0.05])
    if C_bbox is None and C_cat:
        C_bbox = np.array([0.08, 0.08, 0.08])

    # Use min(x,y) for all XY comparisons so both axes are guaranteed to satisfy the ratio
    fix_xy = min(float(fix_bbox[0]), float(fix_bbox[1]))
    fix_z  = float(fix_bbox[2])

    # obj_fixed: scale 0.5
    GLOBAL_SCALE = 0.5
    fix_scale = [GLOBAL_SCALE, GLOBAL_SCALE, GLOBAL_SCALE]

    # obj_A_correct: A's min(x,y) after scale >= fix_xy * ratio (both axes guaranteed larger)
    A_xy  = min(float(A_bbox[0]), float(A_bbox[1]))
    r_Ac  = rng.uniform(*A_CORRECT_RATIO)
    A_correct_scale = float(fix_xy * r_Ac / A_xy) * GLOBAL_SCALE

    # A_wrong: 50/50 — too small (ratio of fix_xy) or too large (ratio of A_correct_scale)
    if rng.random() < 0.5:
        r_Aw         = rng.uniform(*A_WRONG_RATIO_SM)
        A_wrong_scale = float(fix_xy * r_Aw / A_xy) * GLOBAL_SCALE
        A_wrong_mode  = "too_small"
    else:
        r_Aw         = rng.uniform(*A_WRONG_RATIO_LG)
        A_wrong_scale = float(A_correct_scale * r_Aw)
        A_wrong_mode  = "too_large"

    if C_cat:
        # C's min(x,y) after scale relative to fix_xy
        C_xy = min(float(C_bbox[0]), float(C_bbox[1]))
        r_Cc = rng.uniform(*C_CORRECT_RATIO)
        C_correct_scale = float(fix_xy * r_Cc / C_xy) * GLOBAL_SCALE

        # C_wrong: 50/50 — too large (ratio of fix_xy) or too small (ratio of C_correct_scale)
        if rng.random() < 0.5:
            r_Cw         = rng.uniform(*C_WRONG_RATIO_LG)
            C_wrong_scale = float(fix_xy * r_Cw / C_xy) * GLOBAL_SCALE
            C_wrong_mode  = "too_large"
        else:
            r_Cw         = rng.uniform(*C_WRONG_RATIO_SM)
            C_wrong_scale = float(C_correct_scale * r_Cw)
            C_wrong_mode  = "too_small"

    print(f"[scales] A_correct={A_correct_scale:.3f}  A_wrong={A_wrong_scale:.3f} ({A_wrong_mode})")
    if C_cat:
        print(f"[scales] C_correct={C_correct_scale:.3f}  C_wrong={C_wrong_scale:.3f} ({C_wrong_mode})")

    # ── Build OmniGibson config ───────────────────────────────────────────────
    obj_cfgs = [
        build_cfg("obj_fixed",     fix_cat,  fix_model,  0, fix_scale),
        build_cfg("obj_A_correct", A_cat,    A_model,    1, [A_correct_scale]*3),
        build_cfg("obj_A_wrong",   A_cat,    A_model,    2, [A_wrong_scale]*3),
    ]
    if C_cat:
        obj_cfgs += [
            build_cfg("obj_C_correct", C_cat, C_model, 3, [C_correct_scale]*3),
            build_cfg("obj_C_wrong",   C_cat, C_model, 4, [C_wrong_scale]*3),
        ]

    cfg_file = os.path.join(og.example_config_path,
                            f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(cfg_file))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"]                             = obj_cfgs

    env   = og.Environment(configs=config)
    scene = env.scene

    obj_fixed     = scene.object_registry("name", "obj_fixed")
    obj_A_correct = scene.object_registry("name", "obj_A_correct")
    obj_A_wrong   = scene.object_registry("name", "obj_A_wrong")
    obj_C_correct = scene.object_registry("name", "obj_C_correct") if C_cat else None
    obj_C_wrong   = scene.object_registry("name", "obj_C_wrong")   if C_cat else None
    floor_obj     = scene.object_registry("name", args.floor)

    if floor_obj is None or obj_fixed is None:
        print("[ERROR] floor or obj_fixed not found")
        raise SystemExit(2)

    # Add seg modalities
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    step_env(env, 100)

    # ── Always use floor as surface ───────────────────────────────────────────
    surface_obj = floor_obj
    print(f"[surface] using: {surface_obj.name}")

    # ── Build all_objs_named for seg checks ───────────────────────────────────
    all_objs_named = {"obj_fixed": obj_fixed,
                      "obj_A_correct": obj_A_correct,
                      "obj_A_wrong":   obj_A_wrong}
    if C_cat:
        all_objs_named["obj_C_correct"] = obj_C_correct
        all_objs_named["obj_C_wrong"]   = obj_C_wrong

    # ── Place all objects in a row ────────────────────────────────────────────
    # Row order: A_wrong, A_correct, fixed, C_correct, C_wrong (fixed in center)
    if C_cat:
        # A_correct/A_wrong on opposite sides of fixed; C_correct/C_wrong on opposite sides
        A_left, A_right = (obj_A_correct, obj_A_wrong) if rng.random() < 0.5 else (obj_A_wrong, obj_A_correct)
        C_left, C_right = (obj_C_correct, obj_C_wrong) if rng.random() < 0.5 else (obj_C_wrong, obj_C_correct)
        row_objs = [A_left, C_left, obj_fixed, C_right, A_right]
    else:
        if rng.random() < 0.5:
            row_objs = [obj_A_correct, obj_fixed, obj_A_wrong]
        else:
            row_objs = [obj_A_wrong, obj_fixed, obj_A_correct]

    print("\n[row] Placing all objects in row ...")
    row_centres, settled_zs, row_midpoint, row_axis = place_row_on_surface(
        row_objs, surface_obj, floor_obj, room_objs, env, rng
    )

    # ── Compute orbital geometry from all objects ─────────────────────────────
    all_mins = np.array([get_aabb(o)[0] for o in all_objs_named.values()])
    all_maxs = np.array([get_aabb(o)[1] for o in all_objs_named.values()])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    half_diag = float(np.sqrt(((scene_max[0]-scene_min[0])/2)**2 +
                               ((scene_max[1]-scene_min[1])/2)**2))
    orb_radius   = half_diag + ORBIT_RADIUS_PAD
    orb_centre_z = float((scene_min[2] + scene_max[2]) / 2.0)
    orb_height   = orb_centre_z + orb_radius * 0.6 + ORBIT_HEIGHT_PAD
    centre_xy    = row_midpoint[:2]

    # Save initial poses
    initial_poses = {n: snap(o) for n, o in all_objs_named.items()}

    # ── Determine spatial relations for QA from orbital view 0 ───────────────
    # Use the best orbital view where both candidates are distinguishable (left vs right).
    # Try all 4 orbital azimuths and pick the one where A_correct and A_wrong
    # are on OPPOSITE sides of obj_fixed in image space.

    fix_pos = obj_fixed.get_position_orientation()[0].cpu().numpy()
    Ac_pos  = obj_A_correct.get_position_orientation()[0].cpu().numpy()
    Aw_pos  = obj_A_wrong.get_position_orientation()[0].cpu().numpy()
    if C_cat:
        Cc_pos = obj_C_correct.get_position_orientation()[0].cpu().numpy()
        Cw_pos = obj_C_wrong.get_position_orientation()[0].cpu().numpy()

    # Build QA for ALL 4 orbital views — each view has its own spatial relations and question
    qa_per_view = []
    best_az_idx = None  # first view where A_correct and A_wrong are on opposite sides

    for az_idx in range(N_ORBITS):
        az_rad  = 2.0 * np.pi * az_idx / N_ORBITS
        cam_eye = np.array([
            centre_xy[0] + orb_radius * np.cos(az_rad),
            centre_xy[1] + orb_radius * np.sin(az_rad),
            orb_height,
        ])
        cam_tgt = np.array([centre_xy[0], centre_xy[1], orb_centre_z])

        rel_Ac = spatial_relation_in_view(Ac_pos, fix_pos, cam_eye, cam_tgt)
        rel_Aw = spatial_relation_in_view(Aw_pos, fix_pos, cam_eye, cam_tgt)
        opposite = (rel_Ac != rel_Aw)

        if opposite and best_az_idx is None:
            best_az_idx = az_idx

        if C_cat:
            rel_Cc = spatial_relation_in_view(Cc_pos, fix_pos, cam_eye, cam_tgt)
            rel_Cw = spatial_relation_in_view(Cw_pos, fix_pos, cam_eye, cam_tgt)
        else:
            rel_Cc = rel_Cw = None

        if args.n_objects == 2:
            view_qa = {
                "view_idx":   az_idx,
                "view_file":  f"stage0_initial_orb{az_idx}.png",
                "opposite_sides": opposite,
                "question": (f"Which {A_cat} should you place under the {fix_cat} "
                             f"as its base — the one to the {rel_Ac} "
                             f"or the one to the {rel_Aw} of the {fix_cat}?"),
                "answer":        rel_Ac,
                "choices":       [rel_Ac, rel_Aw],
                "rel_A_correct": rel_Ac,
                "rel_A_wrong":   rel_Aw,
            }
        else:
            view_qa = {
                "view_idx":   az_idx,
                "view_file":  f"stage0_initial_orb{az_idx}.png",
                "opposite_sides": opposite,
                "question": (f"Which {A_cat} should go under the {fix_cat} as its base, "
                             f"and which {C_cat} should go on top of the {fix_cat}? "
                             f"The {A_cat} to the {rel_Ac} or {rel_Aw} of the {fix_cat}? "
                             f"The {C_cat} to the {rel_Cc} or {rel_Cw} of the {fix_cat}?"),
                "answer_A":      rel_Ac,
                "answer_C":      rel_Cc,
                "choices_A":     [rel_Ac, rel_Aw],
                "choices_C":     [rel_Cc, rel_Cw],
                "rel_A_correct": rel_Ac,
                "rel_A_wrong":   rel_Aw,
                "rel_C_correct": rel_Cc,
                "rel_C_wrong":   rel_Cw,
            }

        qa_per_view.append(view_qa)
        print(f"[QA] orb{az_idx}: A_correct={rel_Ac}  A_wrong={rel_Aw}"
              + (f"  C_correct={rel_Cc}  C_wrong={rel_Cw}" if C_cat else "")
              + f"  opposite={opposite}")

    # qa = the first view where candidates are on opposite sides (best for evaluation)
    # fall back to orb0 if none found (should not happen with fixed-between layout)
    qa = qa_per_view[best_az_idx]
    print(f"[QA] primary view: orb{best_az_idx}")

    # ── Stage 0: initial orbital renders ─────────────────────────────────────
    print("\n[stage 0] Initial orbital renders ...")
    stage0_views = render_orbital(
        run_dir, "stage0_initial", all_objs_named,
        centre_xy, orb_radius, orb_height, orb_centre_z,
    )

    # original_poses captured after row placement — used to reset objects each step
    original_poses = {n: snap(o) for n, o in all_objs_named.items()}

    all_stages  = []
    gt_action_sequence = []
    success     = False
    obj_B       = obj_fixed   # B is the middle fixed reference object

    # ── Step 1: Place B on A_wrong ────────────────────────────────────────────
    # Expected FAIL: A_wrong is too small to support B
    print(f"\n[step 1] Place {fix_cat} (B) on {A_cat}_wrong ...")
    ok_Aw = do_place_on_top(obj_B, obj_A_wrong, env)
    stage1_views = render_orbital(run_dir, "stage1_B_on_A_wrong", all_objs_named,
                                  centre_xy, orb_radius, orb_height, orb_centre_z)
    gt_action_sequence.append({"action": f"place_B_on_{A_cat}_wrong",
                                "success": bool(ok_Aw), "expected_fail": True})
    all_stages.append({"step": 1, "label": "place_B_on_A_wrong",
                        "result": bool(ok_Aw), "views": stage1_views,
                        "snap_B": snap(obj_B), "snap_A_wrong": snap(obj_A_wrong),
                        "all_poses": {n: snap(o) for n, o in all_objs_named.items()}})

    # Reset B and A_wrong back to original row positions
    reset_to_original(obj_B,       original_poses["obj_fixed"],   env)
    reset_to_original(obj_A_wrong, original_poses["obj_A_wrong"], env)
    reset_all_except(set(), all_objs_named, original_poses, env)
    gt_action_sequence.append({"action": "remove_B_from_A_wrong"})

    # ── Step 2: Place B on A_correct ──────────────────────────────────────────
    # Expected PASS: A_correct is wide enough to support B
    print(f"\n[step 2] Place {fix_cat} (B) on {A_cat}_correct ...")
    ok_Ac = do_place_on_top(obj_B, obj_A_correct, env)
    stage2_views = render_orbital(run_dir, "stage2_B_on_A_correct", all_objs_named,
                                  centre_xy, orb_radius, orb_height, orb_centre_z)
    gt_action_sequence.append({"action": f"place_B_on_{A_cat}_correct",
                                "success": bool(ok_Ac), "expected_fail": False})
    all_stages.append({"step": 2, "label": "place_B_on_A_correct",
                        "result": bool(ok_Ac), "views": stage2_views,
                        "snap_B": snap(obj_B), "snap_A_correct": snap(obj_A_correct),
                        "all_poses": {n: snap(o) for n, o in all_objs_named.items()}})

    if not ok_Ac:
        print("[FAIL] B on A_correct failed — breaking")
        success = False
    elif C_cat:
        # B is now on A_correct — keep them in place

        # ── Step 3: Place C_wrong on B ────────────────────────────────────────
        # Expected FAIL: C_wrong is too large to sit stably on B
        print(f"\n[step 3] Place {C_cat}_wrong (C) on {fix_cat} (B) ...")
        ok_Cw = do_place_on_top(obj_C_wrong, obj_B, env)
        stage3_views = render_orbital(run_dir, "stage3_C_wrong_on_B", all_objs_named,
                                      centre_xy, orb_radius, orb_height, orb_centre_z)
        gt_action_sequence.append({"action": f"place_{C_cat}_wrong_on_B",
                                    "success": bool(ok_Cw), "expected_fail": True})
        all_stages.append({"step": 3, "label": "place_C_wrong_on_B",
                            "result": bool(ok_Cw), "views": stage3_views,
                            "snap_C_wrong": snap(obj_C_wrong), "snap_B": snap(obj_B),
                            "all_poses": {n: snap(o) for n, o in all_objs_named.items()}})

        # Reset C_wrong; A_correct and B stay as-is
        reset_to_original(obj_C_wrong, original_poses["obj_C_wrong"], env)
        reset_all_except({"obj_A_correct", "obj_fixed", "obj_C_correct", "obj_C_wrong"},
                         all_objs_named, original_poses, env)
        gt_action_sequence.append({"action": "remove_C_wrong_from_B"})

        # ── Step 4: Place C_correct on B ──────────────────────────────────────
        # Expected PASS: C_correct is small enough to sit stably on B
        print(f"\n[step 4] Place {C_cat}_correct (C) on {fix_cat} (B) ...")
        ok_Cc = do_place_on_top(obj_C_correct, obj_B, env)
        stage4_views = render_orbital(run_dir, "stage4_C_correct_on_B", all_objs_named,
                                      centre_xy, orb_radius, orb_height, orb_centre_z)
        gt_action_sequence.append({"action": f"place_{C_cat}_correct_on_B",
                                    "success": bool(ok_Cc), "expected_fail": False})
        all_stages.append({"step": 4, "label": "place_C_correct_on_B",
                            "result": bool(ok_Cc), "views": stage4_views,
                            "snap_C_correct": snap(obj_C_correct), "snap_B": snap(obj_B),
                            "all_poses": {n: snap(o) for n, o in all_objs_named.items()}})

        success = ok_Ac and ok_Cc
    else:
        success = ok_Ac

        # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        "scene":      args.scene,
        "room":       args.room,
        "run_idx":    args.run_idx,
        "seed":       seed,
        "task":       "action_order_inference",
        "n_objects":  args.n_objects,

        "hierarchy": {
            "obj_A_cat":   A_cat,
            "obj_fixed_cat": fix_cat,
            "obj_C_cat":   C_cat,
            "relation_A":  rel_A,
            "relation_C":  rel_C,
            "description": hierarchy.get("description", ""),
        },

        "objects": {
            "obj_fixed":     {"cat": fix_cat,  "model": fix_model,  "scale": fix_scale},
            "obj_A_correct": {"cat": A_cat,    "model": A_model,
                              "scale": [A_correct_scale]*3, "ratio": r_Ac},
            "obj_A_wrong":   {"cat": A_cat,    "model": A_model,
                              "scale": [A_wrong_scale]*3,   "ratio": r_Aw, "mode": A_wrong_mode},
            **({
                "obj_C_correct": {"cat": C_cat, "model": C_model,
                                  "scale": [C_correct_scale]*3, "ratio": r_Cc},
                "obj_C_wrong":   {"cat": C_cat, "model": C_model,
                                  "scale": [C_wrong_scale]*3,   "ratio": r_Cw, "mode": C_wrong_mode},
            } if C_cat else {}),
        },

        "row_layout": {
            "row_order":   [o.name for o in row_objs],
            "row_centres": [c.tolist() for c in row_centres],
            "row_midpoint": row_midpoint.tolist(),
            "axis":         row_axis.tolist(),
        },

        "orbital_geometry": {
            "centre_xy":  centre_xy.tolist(),
            "radius":     orb_radius,
            "height":     orb_height,
            "centre_z":   orb_centre_z,
            "n_views":    N_ORBITS,
        },

        "spatial_relations_per_view": [
            {
                "view_idx":      v["view_idx"],
                "view_file":     v["view_file"],
                "opposite_sides": v["opposite_sides"],
                "rel_A_correct": v["rel_A_correct"],
                "rel_A_wrong":   v["rel_A_wrong"],
                **({
                    "rel_C_correct": v["rel_C_correct"],
                    "rel_C_wrong":   v["rel_C_wrong"],
                } if C_cat else {}),
            }
            for v in qa_per_view
        ],
        "primary_qa_view_idx": best_az_idx,

        "initial_poses":       initial_poses,
        "stage0_initial_views": stage0_views,
        "stages":              all_stages,
        "gt_action_sequence":  gt_action_sequence,
        "qa":                  qa,
        "qa_per_view":         qa_per_view,
        "success":             success,
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[meta] saved -> {meta_path}")
    print(f"[done] success={success}")

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise SystemExit(2)