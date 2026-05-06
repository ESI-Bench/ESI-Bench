"""
batch_trajectory_observer.py

Multi-agent spatial observation task for ESI-BENCH.
One scene + one room per invocation.

Pipeline:
  1. Load scene with hidden objects already in config (GPT picks category first)
  2. Sample start/goal from traversable pixels of the trav map
  3. Plan shortest path, validate length, dry-run robot to confirm reachability
     If failed, resample start/goal and retry (same env, no reload)
  4. Place hidden objects along the confirmed path (surface or floor via OnTop)
  5. Final navigation walk with frame capture from three cameras:
       - Static observer: fixed at start XY, always looks at robot
       - Trajectory cam:  follows robot XY, forward-facing along heading
       - GT close-up:     when robot within PROXIMITY_THRESH of hidden object,
                          snap to look at it, save one frame per encounter
  6. Save metadata.json with ground truth, camera poses, per-step log

Usage:
  python batch_trajectory_observer.py \
    --scene Beechwood_0_int \
    --room living_room_0 \
    --floor floors_yrqekq_0 \
    --run_idx 0 \
    --keys_json keys.json \
    --robot R1 \
    --output_root renders_trajectory
"""

import os
import sys
import json
import math
import argparse
import random
import numpy as np
import torch as th
import cv2

import omnigibson as og
import omnigibson.object_states as object_states
import omnigibson.utils.transform_utils as T

from omnigibson.macros import gm
from openai import OpenAI
from scipy.spatial.transform import Rotation

# ── OmniGibson settings ───────────────────────────────────────────────────────
gm.ENABLE_FLATCACHE     = False
gm.USE_GPU_DYNAMICS     = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Task parameters ───────────────────────────────────────────────────────────
PROXIMITY_THRESH       = 1.0
NUM_OBJECTS_MIN        = 1
NUM_OBJECTS_MAX        = 3
SQUARE_ORI             = [0.0, 0.0, 0.0, 1.0]

# ── Navigation parameters ─────────────────────────────────────────────────────
MIN_PATH_WAYPOINTS     = 10
MAX_PAIR_ATTEMPTS      = 20
GOAL_REACHED_THRESH    = 0.8
MAX_STEPS_PER_WP       = 40
DIST_THRESH_WP         = 0.2
SAMPLES_PER_SEG        = 6
N_TRAV_CANDIDATES      = 200   # how many traversable positions to sample

# ── Camera parameters ─────────────────────────────────────────────────────────
OBSERVER_HEIGHT        = 5.0   # top-down, ceiling removed
TRAJ_CAM_HEIGHT        = 1.2
CLOSEUP_HEIGHT         = 1.0
CLOSEUP_STANDOFF       = 0.8
TRAJ_LOOKAHEAD         = 3
TRAJ_TURN_ALPHA        = 0.3

# ── Object placement parameters ───────────────────────────────────────────────
OBJ_MIN_PATH_DIST      = 0.2
OBJ_MAX_PATH_DIST      = 1.2
OBJ_MIN_SEPARATION     = 1.5
OBJ_PLACE_CANDIDATES   = 300

SURFACE_CATS = {
    "table", "shelf", "counter", "desk", "cabinet",
    "sofa", "bed", "chest", "dresser", "nightstand",
    "coffee_table", "side_table", "end_table",
}

# ─────────────────────────────────────────────────────────────────────────────
# Inlined helpers
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


def _get_idle_action(env):
    return th.zeros(env.robots[0].action_dim, dtype=th.float32)


def _wrap_to_pi(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def _yaw_from_quat(quat):
    if hasattr(quat, "cpu"):
        quat = quat.cpu()
    q = np.array(quat, dtype=float).reshape(-1)
    siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _replace_trav_map_with_variant(scene, basename="floor_trav_no_door"):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None:
        return
    map_dir = getattr(trav_map, "map_dir", None) or getattr(trav_map, "_map_dir", None)
    if map_dir is None:
        return
    variant_path = os.path.join(str(map_dir), f"{basename}.png")
    if not os.path.exists(variant_path):
        return
    import PIL.Image as Image
    img    = np.array(Image.open(variant_path).convert("L"))
    binary = th.from_numpy((img > 128).astype(np.uint8))
    if hasattr(trav_map, "floor_map") and trav_map.floor_map:
        trav_map.floor_map[0] = binary
    print(f"[trav_map] replaced with {basename}", flush=True)


_GLOBAL_ENV = None


def _step_env(env, n=5):
    idle = _get_idle_action(env)
    for _ in range(int(n)):
        env.step(idle)


def _step_simple(n=5):
    _step_env(_GLOBAL_ENV, n)


def _get_scene_objects(scene):
    raw = getattr(scene, "objects", [])
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def _capture(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(5):
        og.sim.render()
    img = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _look_at_quat(eye, look_at):
    eye     = np.array(eye, dtype=float).reshape(3)
    look_at = np.array(look_at, dtype=float).reshape(3)
    if np.linalg.norm(look_at - eye) < 1e-6:
        look_at = eye + np.array([1.0, 0.0, 0.0])
    quat = np.array(look_at_quaternion(eye, look_at), dtype=float).reshape(4)
    if not (np.all(np.isfinite(quat)) and np.linalg.norm(quat) > 1e-8):
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quat / np.linalg.norm(quat)


def _set_camera_and_capture(eye, look_at, path):
    eye_np = np.array(eye, dtype=float).reshape(3)
    quat   = _look_at_quat(eye_np, look_at)
    og.sim._viewer_camera.set_position_orientation(
        position=th.tensor(eye_np, dtype=th.float32),
        orientation=th.tensor(quat, dtype=th.float32),
    )
    _capture(path)
    return {"position": eye_np.tolist(), "quaternion_xyzw": quat.tolist()}


def _interpolate_path_xy(path_xy, samples_per_seg=6):
    pts = [np.array(p[:2], dtype=float) for p in path_xy]
    if len(pts) <= 1:
        return pts
    out = []
    n = max(1, int(samples_per_seg))
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for k in range(n):
            out.append(a * (1 - k / n) + b * (k / n))
    out.append(pts[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup
# ─────────────────────────────────────────────────────────────────────────────

def remove_ceiling(env):
    if og.sim.is_playing():
        og.sim.stop()
    scene  = env.scene
    stage  = og.sim.stage
    prefix = "/World/scene_0/ceilings_"
    for obj in _get_scene_objects(scene):
        pp   = (getattr(obj, "prim_path", None) or getattr(obj, "_prim_path", None) or "")
        name = str(getattr(obj, "name", "")).lower()
        if str(pp).startswith(prefix) or "ceilings_" in name:
            try:
                scene.remove_object(obj)
            except Exception:
                pass
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.startswith(prefix) and p.count("/") == 3:
            prim.SetActive(False)
    og.sim.play()


def remove_all_door_like_objects(scene, env):
    keywords = ("door", "gate", "hatch")
    targets = [
        obj for obj in _get_scene_objects(scene)
        if any(k in str(getattr(obj, "name", "")).lower() or
               k in str(getattr(obj, "category", "")).lower()
               for k in keywords)
    ]
    for obj in targets:
        try:
            scene.remove_object(obj)
        except Exception:
            pass
    _step_env(env, 3)
    print(f"[doors] removed {len(targets)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trav map candidate sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_traversable_positions(scene, floor_idx, floor_obj, rng, n=N_TRAV_CANDIDATES):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None:
        raise RuntimeError("No trav map found")
    fmap = trav_map.floor_map[floor_idx]
    if hasattr(fmap, "cpu"):
        fmap = fmap.cpu()
    fmap_np = np.array(fmap, dtype=np.uint8)
    trav_pixels = np.argwhere(fmap_np > 0)
    if len(trav_pixels) == 0:
        raise RuntimeError("No traversable pixels found")

    # Room bounds from floor object AABB
    bmin, bmax = [x.cpu().numpy() for x in floor_obj.aabb]
    fx_min, fx_max = float(bmin[0]), float(bmax[0])
    fy_min, fy_max = float(bmin[1]), float(bmax[1])

    positions = []
    attempts  = 0
    max_attempts = n * 50
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        row, col = trav_pixels[rng.randint(0, len(trav_pixels) - 1)]
        xy = trav_map.map_to_world(
            th.tensor([float(row), float(col)], dtype=th.float32)
        ).cpu().numpy()
        x, y = float(xy[0]), float(xy[1])
        if fx_min <= x <= fx_max and fy_min <= y <= fy_max:
            positions.append(np.array([x, y], dtype=float))

    if len(positions) == 0:
        raise RuntimeError("No traversable points found inside room AABB")
    print(f"[trav_sample] sampled {len(positions)} points inside room AABB", flush=True)
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Navigation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_shortest_path(scene, floor_idx, start_xy, goal_xy, robot=None):
    try:
        kwargs = dict(
            floor=int(floor_idx),
            source_world=th.tensor(start_xy, dtype=th.float32),
            target_world=th.tensor(goal_xy, dtype=th.float32),
            entire_path=True,
        )
        if robot is not None:
            kwargs["robot"] = robot
        path_world, _ = scene.get_shortest_path(**kwargs)
        return path_world
    except Exception:
        return None


def _estimate_robot_safe_z(robot, scene, floor_idx):
    robot_bbox_top_z  = float(robot.aabb[1].cpu().numpy()[2])  # max Z of robot bbox:
    robot_pos, _ = robot.get_position_orientation()
    floor_z      = float(scene.get_floor_height(int(floor_idx)))
    offset       = float(robot_pos[2]) - floor_z
    return float(floor_z + max(0.10, offset))


def _set_robot_pose(robot, xy, z, next_xy):
    yaw  = float(math.atan2(next_xy[1] - xy[1], next_xy[0] - xy[0]))
    quat = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
    robot.set_position_orientation(
        position=th.tensor([float(xy[0]), float(xy[1]), float(z)], dtype=th.float32),
        orientation=quat,
    )


def select_valid_start_goal(scene, robot, floor_idx, robot_start_z, floor_obj, rng):
    """
    Sample start/goal from traversable pixels, plan path, validate length.
    Returns dict with start_xy, goal_xy, path_xy (dense).
    Raises RuntimeError if no valid pair found after MAX_PAIR_ATTEMPTS.
    """
    candidates = _sample_traversable_positions(scene, floor_idx, floor_obj, rng)
    min_sep    = 2.0

    for attempt in range(MAX_PAIR_ATTEMPTS):
        rng.shuffle(candidates)
        start_xy = goal_xy = None
        for i, c in enumerate(candidates):
            for j, d in enumerate(candidates):
                if i != j and float(np.linalg.norm(c - d)) >= min_sep:
                    start_xy, goal_xy = c, d
                    break
            if start_xy is not None:
                break
        if start_xy is None:
            start_xy, goal_xy = candidates[0], candidates[1]

        path_world = _get_shortest_path(scene, floor_idx, start_xy, goal_xy, robot)
        if path_world is None or len(path_world) < 2:
            print(f"[pair {attempt+1}] no path", flush=True)
            continue

        dense = _interpolate_path_xy(path_world, samples_per_seg=SAMPLES_PER_SEG)
        if len(dense) < MIN_PATH_WAYPOINTS:
            print(f"[pair {attempt+1}] too short ({len(dense)} wps)", flush=True)
            continue

        print(f"[pair {attempt+1}] OK  wps={len(dense)}", flush=True)
        return {
            "start_xy":  start_xy,
            "goal_xy":   goal_xy,
            "path_world": [np.array(p[:2], dtype=float) for p in path_world],
            "path_xy":   dense,
        }

    raise RuntimeError(f"No valid start/goal after {MAX_PAIR_ATTEMPTS} attempts")


# ─────────────────────────────────────────────────────────────────────────────
# Navigation with per-step callback
# ─────────────────────────────────────────────────────────────────────────────

def navigate_with_callback(env, robot, path_xy, floor_z, robot_start_z, step_callback):
    idle      = _get_idle_action(env)
    base_idx  = robot.controller_action_idx["base"]
    half_idx  = max(1, len(path_xy) // 2)
    step_idx  = 0
    forced    = False
    failed_wp = None

    for wp_idx, waypoint in enumerate(path_xy):
        wp      = np.array(waypoint[:2], dtype=float)
        reached = False

        for _ in range(int(MAX_STEPS_PER_WP)):
            robot_pos, robot_quat = robot.get_position_orientation()
            cur_xy = np.array(robot_pos[:2], dtype=float)

            if step_callback is not None:
                step_callback(step_idx, cur_xy, robot_quat)
            step_idx += 1

            to_wp   = wp - cur_xy
            dist_wp = float(np.linalg.norm(to_wp))
            if dist_wp < DIST_THRESH_WP:
                reached = True
                break

            yaw     = _yaw_from_quat(robot_quat)
            c, s    = math.cos(yaw), math.sin(yaw)
            local_x = c * to_wp[0] + s * to_wp[1]
            local_y = -s * to_wp[0] + c * to_wp[1]
            tgt_yaw = math.atan2(to_wp[1], to_wp[0])
            yaw_err = _wrap_to_pi(tgt_yaw - yaw)

            action = idle.clone()
            action[base_idx] = th.tensor(
                [np.clip(local_x, -0.01, 0.01),
                 np.clip(local_y, -0.01, 0.01),
                 np.clip(yaw_err, -0.01, 0.01)],
                dtype=th.float32,
            )
            env.step(action)

        if reached:
            continue

        failed_wp = wp_idx
        if wp_idx >= half_idx:
            goal_xy = np.array(path_xy[-1][:2], dtype=float)
            robot.set_position_orientation(
                position=th.tensor(
                    [float(goal_xy[0]), float(goal_xy[1]), float(robot_start_z)],
                    dtype=th.float32,
                ),
                orientation=T.euler2quat(th.tensor([0.0, 0.0, 0.0], dtype=th.float32)),
            )
            forced = True
            _step_env(env, 2)
            break

    final_pos, final_quat = robot.get_position_orientation()
    final_xy   = np.array(final_pos[:2], dtype=float)
    goal_xy    = np.array(path_xy[-1][:2], dtype=float)
    final_dist = float(np.linalg.norm(final_xy - goal_xy))

    if step_callback is not None:
        step_callback(step_idx, final_xy, final_quat)

    return {
        "ok":             bool(final_dist < GOAL_REACHED_THRESH),
        "final_xy":       final_xy.tolist(),
        "goal_xy":        goal_xy.tolist(),
        "final_dist":     final_dist,
        "forced_to_goal": forced,
        "failed_wp_idx":  failed_wp,
        "total_steps":    step_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPT object selection
# ─────────────────────────────────────────────────────────────────────────────

def load_keys(path):
    with open(path) as f:
        return json.load(f)


def sample_200(all_keys, seed):
    rng = random.Random(seed)
    return rng.sample(all_keys, min(200, len(all_keys)))


def gpt_pick_object_category(candidate_categories):
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = (
        "Pick exactly ONE object category for objects hidden along a robot path.\n"
        "Requirements:\n"
        "  - Medium-sized, easy to spot\n"
        "  - NOT tiny (no: dice, coin, key, screw, pen)\n"
        "  - NOT huge furniture (no: sofa, wardrobe, bed, bathtub)\n"
        "  - Good examples: apple, bottle, mug, shoe, ball, vase, book, flower_pot\n"
        "  - Must appear verbatim in the provided list\n"
        'Reply ONLY: {"category": "..."}'
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Candidates:\n{json.dumps(candidate_categories)}"},
        ],
        temperature=0.2,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    cat = json.loads(raw)["category"]
    if cat not in candidate_categories:
        raise ValueError(f"GPT returned '{cat}' not in candidate list")
    print(f"[GPT] chosen category: {cat}", flush=True)
    return cat


def get_model_for_category(category, seed):
    inventory_paths = [
        "bddl3/bddl/generated_data/object_inventory.json",
        os.path.join(os.path.dirname(__file__), "object_inventory.json"),
    ]
    for path in inventory_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            inventory = json.load(f)
        providers = inventory.get("providers", inventory)
        matches   = [k for k in providers if k.startswith(f"{category}-")]
        if not matches:
            raise RuntimeError(f"Category '{category}' not in inventory")
        rng    = random.Random(seed)
        chosen = rng.choice(matches)
        model_id = chosen.split("-", 1)[1]
        print(f"  [{category}] picked model_id={model_id}", flush=True)
        return model_id
    raise RuntimeError("No object inventory file found")


def build_object_config(name, category, model, idx):
    return {
        "type":        "DatasetObject",
        "name":        name,
        "category":    category,
        "model":       model,
        "position":    [200.0 + idx * 10, 100.0, 100.0],
        "orientation": SQUARE_ORI,
        "scale":       [1.0, 1.0, 1.0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Object placement along path
# ─────────────────────────────────────────────────────────────────────────────

def _visibility_check(obj_names):
    """Check seg_instance for each object name. Returns dict {name: bool}."""
    for _ in range(10):
        og.sim.step()
    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    visible_str  = " ".join(id_to_labels.values())
    result = {name: (name in visible_str) for name in obj_names}
    return result


def _dist_to_path(xy, path_dense):
    xy = np.array(xy[:2], dtype=float)
    return min(float(np.linalg.norm(xy - np.array(p[:2], dtype=float))) for p in path_dense)


def _aabb_overlaps_xy(ax_min, ax_max, bx_min, bx_max, ay_min, ay_max, by_min, by_max):
    return not (ax_max <= bx_min or bx_max <= ax_min or ay_max <= by_min or by_max <= ay_min)


def place_objects_along_path(scene, obj_list, floor_obj, path_dense, robot, rng):
    """
    Place each hidden object beside the path (not on it), offset perpendicular
    by robot half-width + margin so robot doesn't collide with them.
    """
    SQUARE_ORI_T  = th.tensor(SQUARE_ORI, dtype=th.float32)
    OBJ_MARGIN    = 0.20
    WALL_MARGIN   = 0.10
    SKIP_CATS     = {"ceilings", "walls", "floors", "carpet", "window", "door", "curtain"}
    placed_info   = []
    placed_bboxes = []
    path_arr      = [np.array(p[:2], dtype=float) for p in path_dense]

    # Robot half-width from AABB — offset objects this far perpendicular to path
    robot_bmin, robot_bmax = [x.cpu().numpy() for x in robot.aabb]
    robot_half_x = abs(float(robot_bmax[0]) - float(robot_bmin[0])) / 2.0
    robot_half_y = abs(float(robot_bmax[1]) - float(robot_bmin[1])) / 2.0
    robot_half   = max(robot_half_x, robot_half_y)
    side_offset  = robot_half + 0.15  # extra margin beyond robot width
    print(f"[place] robot half-width={robot_half:.3f}m  side_offset={side_offset:.3f}m", flush=True)

    # Collect scene object bboxes for collision checking
    # Skip structural objects (floors, walls, ceilings) and hidden objects themselves
    # Also skip furniture that the path already navigates around
    hidden_names = {getattr(o, "name", "") for o in obj_list}
    scene_bboxes = []
    for obj in _get_scene_objects(scene):
        cat  = str(getattr(obj, "category", "")).lower()
        name = str(getattr(obj, "name", ""))
        if any(s in cat for s in SKIP_CATS):
            continue
        if name in hidden_names:
            continue
        # Skip large furniture that spans the whole room
        try:
            bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
            area = abs(float(bmax[0]) - float(bmin[0])) * abs(float(bmax[1]) - float(bmin[1]))
            if area > 2.0:  # skip objects with footprint > 2m^2
                continue
            scene_bboxes.append((bmin, bmax))
        except Exception:
            pass

    for obj in obj_list:
        obj_name = getattr(obj, "name", "?")

        # Step 1: drop onto floor to get floor Z and half-extents
        try:
            obj.states[object_states.OnTop].set_value(floor_obj, True)
        except Exception:
            pass
        _step_simple(15)
        pos, _ = obj.get_position_orientation()
        obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
        obj.keep_still()
        _step_simple(10)
        bmin_raw, bmax_raw = [x.cpu().numpy() for x in obj.aabb]
        hx   = abs(float(bmax_raw[0]) - float(bmin_raw[0])) / 2.0
        hy   = abs(float(bmax_raw[1]) - float(bmin_raw[1])) / 2.0
        drop_z = float(obj.get_position_orientation()[0].cpu().numpy()[2])

        # Step 2: pick waypoint evenly spaced along path, offset perpendicular
        placed   = False
        obj_idx  = obj_list.index(obj)
        n_objs   = len(obj_list)
        frac     = (obj_idx + 1) / (n_objs + 1)
        base_idx = int(frac * (len(path_arr) - 1))
        wp_indices = sorted(range(len(path_arr)), key=lambda i: abs(i - base_idx))
        # Try both left and right side of path
        sides = [1.0, -1.0]
        for wp_idx in wp_indices[:OBJ_PLACE_CANDIDATES]:
            wp = path_arr[wp_idx]
            # Compute path tangent direction at this waypoint
            next_idx = min(wp_idx + 1, len(path_arr) - 1)
            prev_idx = max(wp_idx - 1, 0)
            tangent  = path_arr[next_idx] - path_arr[prev_idx]
            t_norm   = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                tangent = np.array([1.0, 0.0])
            else:
                tangent = tangent / t_norm
            # Perpendicular to tangent
            perp = np.array([-tangent[1], tangent[0]])
            found_side = False
            for side in sides:
                cx = float(wp[0]) + side * perp[0] * side_offset
                cy = float(wp[1]) + side * perp[1] * side_offset

                # Inflate proposed bbox by margin
                pxmin = cx - hx - OBJ_MARGIN
                pxmax = cx + hx + OBJ_MARGIN
                pymin = cy - hy - OBJ_MARGIN
                pymax = cy + hy + OBJ_MARGIN

                # Check against scene objects
                collision = False
                for (bmin, bmax) in scene_bboxes:
                    if _aabb_overlaps_xy(pxmin, pxmax, float(bmin[0]), float(bmax[0]),
                                         pymin, pymax, float(bmin[1]), float(bmax[1])):
                        collision = True
                        break
                if collision:
                    continue

                # Check against already-placed objects
                for (bmin, bmax) in placed_bboxes:
                    if _aabb_overlaps_xy(pxmin, pxmax, float(bmin[0]), float(bmax[0]),
                                         pymin, pymax, float(bmin[1]), float(bmax[1])):
                        collision = True
                        break
                if collision:
                    continue

                # Teleport to chosen position and pin
                obj.set_position_orientation(
                    position=th.tensor([cx, cy, drop_z], dtype=th.float32),
                    orientation=SQUARE_ORI_T,
                )
                obj.keep_still()
                _step_simple(15)
                pos_after, _ = obj.get_position_orientation()
                obj.set_position_orientation(
                    position=th.tensor([cx, cy, float(pos_after[2])], dtype=th.float32),
                    orientation=SQUARE_ORI_T,
                )
                obj.keep_still()
                _step_simple(5)
                final_pos = obj.get_position_orientation()[0].cpu().numpy()
                bmin_f, bmax_f = [x.cpu().numpy() for x in obj.aabb]
                placed_bboxes.append((bmin_f, bmax_f))
                placed_info.append({
                    "name":         obj_name,
                    "position":     final_pos.tolist(),
                    "placed_on":    "floor",
                    "dist_to_path": float(_dist_to_path(final_pos[:2], path_arr)),
                })
                print(f"[place] {obj_name} at ({cx:.2f},{cy:.2f}) side={side:+.0f} "
                      f"dist_to_path={_dist_to_path([cx,cy], path_arr):.2f}m", flush=True)
                placed = True
                found_side = True
                break
            if found_side:
                break

        if not placed:
            print(f"[place] WARNING: could not place {obj_name}", flush=True)
            placed_info.append({"name": obj_name, "position": None, "placed_on": "failed"})

    return placed_info


# ─────────────────────────────────────────────────────────────────────────────
# QA generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_qa(category, true_count, num_objects_total, rng):
    all_counts  = list(range(num_objects_total + 1))
    distractors = [c for c in all_counts if c != true_count]
    rng.shuffle(distractors)
    chosen = []
    if true_count != 0:
        chosen.append(0)
    for d in distractors:
        if len(chosen) >= 3:
            break
        if d not in chosen:
            chosen.append(d)
    extra = 1
    while len(chosen) < 3:
        c = true_count + extra
        if c not in chosen and c != true_count:
            chosen.append(c)
        extra += 1
    options = sorted(set([true_count] + chosen[:3]))[:4]
    if true_count not in options:
        options[-1] = true_count
        options = sorted(options)
    labels       = ["A", "B", "C", "D"]
    choices      = {labels[i]: options[i] for i in range(len(options))}
    answer_label = labels[options.index(true_count)]
    return {
        "question":     (
            f"How many {category} objects did the robot pass within "
            f"{PROXIMITY_THRESH}m during its walk?"
        ),
        "choices":      choices,
        "answer_label": answer_label,
        "answer_count": true_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",       type=str, required=True)
    parser.add_argument("--room",        type=str, required=True)
    parser.add_argument("--floor",       type=str, required=True)
    parser.add_argument("--run_idx",     type=int, default=0)
    parser.add_argument("--keys_json",   type=str, default="keys.json")
    parser.add_argument("--robot",       type=str, default="R1")
    parser.add_argument("--output_root", type=str, default="renders_trajectory")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 777
    rng     = random.Random(seed)
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene={args.scene}  Room={args.room}  Run={args.run_idx}")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}\n")

    # ── GPT picks category before loading env ─────────────────────────────────
    print("[gpt] picking object category ...", flush=True)
    all_keys  = load_keys(args.keys_json)
    sampled   = sample_200(all_keys, seed=seed)
    category  = gpt_pick_object_category(sampled)
    n_objects = rng.randint(NUM_OBJECTS_MIN, NUM_OBJECTS_MAX)
    print(f"[gpt] category={category}  n_objects={n_objects}", flush=True)

    obj_configs = []
    obj_names   = []
    for i in range(n_objects):
        model = get_model_for_category(category, seed=seed + i + 1)
        name  = f"hidden_obj_{i}"
        obj_names.append(name)
        obj_configs.append(build_object_config(name, category, model, i))

    # ── Load env once with objects already in config ──────────────────────────
    import yaml
    config_filename = os.path.join(
        og.example_config_path, f"{args.robot.lower()}_primitives.yaml"
    )
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"]                             = obj_configs

    env = og.Environment(configs=config)
    _replace_trav_map_with_variant(env.scene, basename="floor_trav_no_door")

    global _GLOBAL_ENV
    _GLOBAL_ENV = env

    remove_ceiling(env)
    scene = env.scene
    robot = env.robots[0]
    remove_all_door_like_objects(scene=scene, env=env)
    _step_env(env, 10)

    floor_idx     = 0
    floor_z       = float(scene.get_floor_height(int(floor_idx)))
    robot_start_z = _estimate_robot_safe_z(robot, scene, floor_idx)
    robot_bbox_top_z  = float(robot.aabb[1].cpu().numpy()[2])  # max Z of robot bbox

    # Add seg modalities for visibility checks
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # ── Find floor object early (needed for trav map sampling) ──────────────────
    floor_obj = scene.object_registry("name", args.floor)
    if floor_obj is None:
        for obj in _get_scene_objects(scene):
            if "floor" in str(getattr(obj, "category", "")).lower():
                floor_obj = obj
                break

    # ── Find valid navigable start/goal by sampling trav map ──────────────────
    confirmed_pair = None
    for nav_attempt in range(MAX_PAIR_ATTEMPTS):
        try:
            pair = select_valid_start_goal(
                scene=scene, robot=robot,
                floor_idx=floor_idx, robot_start_z=robot_start_z,
                floor_obj=floor_obj, rng=rng,
            )
        except RuntimeError as e:
            print(f"[nav attempt {nav_attempt+1}] {e}", flush=True)
            break

        path_xy = pair["path_xy"]
        _set_robot_pose(robot, path_xy[0], robot_start_z,
                        path_xy[min(1, len(path_xy) - 1)])
        _step_env(env, 5)

        print(f"[nav attempt {nav_attempt+1}] dry-run ...", flush=True)
        nav_res = navigate_with_callback(
            env=env, robot=robot, path_xy=path_xy,
            floor_z=floor_z, robot_start_z=robot_start_z,
            step_callback=None,
        )
        print(f"[nav attempt {nav_attempt+1}] "
              f"final_dist={nav_res['final_dist']:.3f}m  ok={nav_res['ok']}", flush=True)

        if nav_res["ok"]:
            confirmed_pair = pair
            break
        print(f"[nav attempt {nav_attempt+1}] failed, resampling ...", flush=True)

    if confirmed_pair is None:
        print("[ERROR] Could not find navigable path. Exiting.", flush=True)
        sys.exit(2)

    path_xy = confirmed_pair["path_xy"]
    print(f"[nav] confirmed: {len(path_xy)} dense waypoints", flush=True)

    # ── Place hidden objects along the path ───────────────────────────────────
    hidden_objs = [o for o in
                   [scene.object_registry("name", n) for n in obj_names]
                   if o is not None]

    placed_info = place_objects_along_path(
        scene=scene, obj_list=hidden_objs,
        floor_obj=floor_obj, path_dense=path_xy, robot=robot, rng=rng,
    )

    obj_positions = {
        info["name"]: np.array(info["position"][:2], dtype=float)
        for info in placed_info if info.get("position") is not None
    }
    print(f"[place] placed {len(obj_positions)}/{n_objects} objects", flush=True)

    # ── Final navigation with frame capture ───────────────────────────────────
    _set_robot_pose(robot, path_xy[0], robot_start_z,
                    path_xy[min(1, len(path_xy) - 1)])
    _step_env(env, 5)

    # Observer cam: fixed at robot start position (same XY and Z as robot)
    start_xy_3d = np.array([path_xy[0][0], path_xy[0][1], float(robot_bbox_top_z)])
    traj_yaw    = float(math.atan2(
        path_xy[min(1, len(path_xy) - 1)][1] - path_xy[0][1],
        path_xy[min(1, len(path_xy) - 1)][0] - path_xy[0][0],
    ))

    step_log       = []
    observer_poses = {}
    traj_poses     = {}
    closeup_frames = {}
    passed_objects = {}
    closeup_active = set()
    wp_ptr         = [0]

    for subdir in ("observer", "trajectory", "closeup"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    robot_bbox_bot_z = float(robot.aabb[0].cpu().numpy()[2])  # min Z of robot bbox

    def step_callback(step_idx, cur_xy, robot_quat):
        robot_xy  = np.array(cur_xy[:2], dtype=float)
        robot_z   = float(robot_bbox_top_z)
        robot_eye = np.array([robot_xy[0], robot_xy[1], robot_z])
        robot_yaw = _yaw_from_quat(robot_quat)
        robot_look_z = float(robot_bbox_bot_z) + 0.05

        # Proximity check every step
        for obj_name, obj_xy in obj_positions.items():
            dist    = float(np.linalg.norm(robot_xy - obj_xy))
            in_zone = dist < PROXIMITY_THRESH
            if in_zone:
                if obj_name not in passed_objects:
                    passed_objects[obj_name] = step_idx
                    print(f"[proximity] {obj_name} at step {step_idx} dist={dist:.3f}m",
                          flush=True)
                # Record closeup pose when proximity first triggered
                # Actual render happens after nav walk (robot out of the way)
                if obj_name not in closeup_active:
                    closeup_active.add(obj_name)
                    obj_scene = scene.object_registry("name", obj_name)
                    try:
                        bmin, bmax = [x.cpu().numpy() for x in obj_scene.aabb]
                        look_target = ((bmin + bmax) / 2.0)
                    except Exception:
                        look_target = np.array([obj_xy[0], obj_xy[1], robot_z])
                    closeup_frames.setdefault(obj_name, []).append({
                        "step":        step_idx,
                        "eye":         robot_eye.tolist(),
                        "look_target": look_target.tolist(),
                        "dist":        dist,
                    })
                    print(f"[closeup] pose recorded for {obj_name} at step {step_idx}", flush=True)
            else:
                closeup_active.discard(obj_name)

        # Render observer + trajectory every 50 steps
        if step_idx % 50 == 0:
            fname = f"step_{step_idx:05d}.png"

            # Camera 1: observer — fixed at start XY+maxZ, always looks at robot
            obs_pose = _set_camera_and_capture(
                eye=start_xy_3d,
                look_at=np.array([robot_xy[0], robot_xy[1], robot_look_z]),
                path=os.path.join(run_dir, "observer", fname),
            )
            obs_vis = _visibility_check(list(obj_positions.keys()) + ["robot"])
            observer_poses[fname] = {**obs_pose, "visibility": obs_vis}

            # Camera 2: trajectory — robot XY+maxZ, robot yaw orientation
            traj_look = robot_eye + np.array([math.cos(robot_yaw) * 1.5,
                                              math.sin(robot_yaw) * 1.5, 0.0])
            traj_pose = _set_camera_and_capture(
                eye=robot_eye, look_at=traj_look,
                path=os.path.join(run_dir, "trajectory", fname),
            )
            traj_poses[fname] = traj_pose

        step_log.append({
            "step":            step_idx,
            "robot_xy":        robot_xy.tolist(),
            "robot_z":         float(robot_z),
            "robot_yaw_deg":   float(math.degrees(robot_yaw)),
            "robot_quat_xyzw": [float(x) for x in np.array(robot_quat.cpu() if hasattr(robot_quat, "cpu") else robot_quat).reshape(-1)],
        })

        if wp_ptr[0] < len(path_xy) - 1:
            wp_xy = np.array(path_xy[wp_ptr[0]][:2], dtype=float)
            if float(np.linalg.norm(robot_xy - wp_xy)) < DIST_THRESH_WP:
                wp_ptr[0] = min(wp_ptr[0] + 1, len(path_xy) - 1)

    final_nav = navigate_with_callback(
        env=env, robot=robot, path_xy=path_xy,
        floor_z=floor_z, robot_start_z=robot_start_z,
        step_callback=step_callback,
    )
    print(f"[nav] final_dist={final_nav['final_dist']:.3f}m  "
          f"ok={final_nav['ok']}  steps={final_nav['total_steps']}", flush=True)

    # ── Render closeups now that robot has finished (no robot blocking view) ───
    # Move robot out of the way first
    robot.set_position_orientation(
        position=th.tensor([500.0, 500.0, 0.0], dtype=th.float32),
        orientation=T.euler2quat(th.tensor([0.0, 0.0, 0.0], dtype=th.float32)),
    )
    _step_env(env, 5)

    for obj_name, frames in closeup_frames.items():
        for frame in frames:
            eye        = np.array(frame["eye"])
            look_target = np.array(frame["look_target"])
            cu_fname   = f"{obj_name}_step_{frame['step']:05d}.png"
            cu_pose    = _set_camera_and_capture(
                eye=eye, look_at=look_target,
                path=os.path.join(run_dir, "closeup", cu_fname),
            )
            cu_vis = _visibility_check(list(obj_positions.keys()))
            frame["filename"]   = cu_fname
            frame["pose"]       = cu_pose
            frame["visibility"] = cu_vis
            print(f"[closeup] rendered {cu_fname}  visibility={cu_vis}", flush=True)

    # ── Metadata + QA ─────────────────────────────────────────────────────────
    true_count = len(passed_objects)
    qa = generate_qa(category, true_count, n_objects, rng)

    print(f"\n[QA] {qa['question']}")
    for label, val in qa["choices"].items():
        print(f"  {label}: {val}{'  <-- ANSWER' if label == qa['answer_label'] else ''}")

    # Collect final object positions and aabbs
    objects_meta = {}
    for info in placed_info:
        name = info["name"]
        obj_scene = scene.object_registry("name", name)
        if obj_scene is not None:
            try:
                pos, quat = obj_scene.get_position_orientation()
                bmin, bmax = [x.cpu().numpy() for x in obj_scene.aabb]
                objects_meta[name] = {
                    "category":        category,
                    "position":        pos.cpu().numpy().tolist(),
                    "quaternion_xyzw": quat.cpu().numpy().tolist(),
                    "aabb_min":        bmin.tolist(),
                    "aabb_max":        bmax.tolist(),
                    "aabb_centre":     ((bmin + bmax) / 2.0).tolist(),
                    "placed_on":       info.get("placed_on"),
                    "dist_to_path":    info.get("dist_to_path"),
                }
            except Exception:
                objects_meta[name] = info
        else:
            objects_meta[name] = info

    metadata = {
        "scene":               args.scene,
        "room":                args.room,
        "run_idx":             args.run_idx,
        "seed":                seed,
        "task":                "trajectory_observer",

        # Path
        "start_xy":            confirmed_pair["start_xy"].tolist(),
        "goal_xy":             confirmed_pair["goal_xy"].tolist(),
        "path_waypoints":      [p.tolist() for p in path_xy],
        "num_dense_waypoints": len(path_xy),

        # Robot
        "robot_start_z":       float(robot_start_z),
        "robot_bbox_top_z":    float(robot_bbox_top_z),

        # Observer camera (fixed)
        "observer_camera": {
            "position":        start_xy_3d.tolist(),
            "description":     "fixed at initial robot XY + robot max Z, always looks at robot",
        },

        # Objects
        "object_category":     category,
        "num_objects":         n_objects,
        "placed_objects":      placed_info,
        "objects_meta":        objects_meta,

        # Ground truth
        "proximity_thresh":    PROXIMITY_THRESH,
        "passed_objects":      {k: int(v) for k, v in passed_objects.items()},
        "true_count":          true_count,
        "qa":                  qa,

        # Navigation result
        "navigation":          final_nav,

        "camera_params": {
            "observer_height":  OBSERVER_HEIGHT,
            "traj_cam_height":  TRAJ_CAM_HEIGHT,
            "closeup_height":   CLOSEUP_HEIGHT,
            "closeup_standoff": CLOSEUP_STANDOFF,
            "render_every_n_steps": 50,
        },

        # Per-step log (every step: robot pose)
        "step_log":            step_log,

        # Camera poses (every 50 steps)
        "observer_poses":      observer_poses,
        "traj_poses":          traj_poses,
        "closeup_frames":      closeup_frames,
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[meta] saved -> {meta_path}")
    print(f"[done] observer={len(observer_poses)}  traj={len(traj_poses)}  "
          f"closeups={sum(len(v) for v in closeup_frames.values())}")


if __name__ == "__main__":
    main()