"""
batch_touching.py

Single-run script: loads ONE scene restricted to ONE room, places 2 GPT-selected
medium-sized objects (toys, books — not furniture) so they are TRULY touching,
renders 18 camera views (every 20° around the contact centre), and saves a
metadata JSON.

Camera views (0–17):
  Camera k is placed at azimuth k*20° around the contact midpoint, at a fixed
  radius and height, always pointing at the contact centre.
  View 0 = 0° (along +X axis), incrementing counter-clockwise.

JSON flags per view:
  exist_obj1_k — obj1 visible in seg_instance
  exist_obj2_k — obj2 visible in seg_instance
  touching_k   — pixel-mask adjacency: masks touch at any point

Called once per (scene, room, run_idx) by batch_touching.sh.

Orientation policy:
  Both objects are explicitly set to identity orientation [0, 0, 0, 1] —
  i.e. axes perfectly aligned with the world frame (square/perpendicular).
  This ensures AABB extents are axis-aligned before placement math is applied.

Contact placement conditions (verified on final AABB):
  X: zero gap  — f2_xmin == f1_xmax  (faces meet, no gap, no overlap in X)
  Y: 0.05 m overlap — f2_ymin = f1_ymax - 0.05  (confirms true contact)
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
from openai import OpenAI
from scipy.spatial.transform import Rotation

# ── OmniGibson settings ───────────────────────────────────────────────────────
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

# ── OpenAI key ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Contact placement parameters ─────────────────────────────────────────────
# X: zero gap  — f2_xmin == f1_xmax exactly (faces meet, no gap)
# Y: 0.05 m overlap — f2_ymin = f1_ymax - 0.05 (confirms true contact)
Y_TOUCH_OVERLAP = 0.05   # how far obj2 overlaps obj1 in Y
GAP_DRIFT       = 0.03   # small random X drift per run for variety

# ── Axis-aligned (square) orientation — identity quaternion ──────────────────
# Both objects use this so their AABB is perfectly aligned with world X/Y/Z.
SQUARE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]   # [qx, qy, qz, qw]

# ── Camera ring parameters ────────────────────────────────────────────────────
NUM_VIEWS      = 24          # 360 / 20 = 18
VIEW_STEP_DEG  = 15.0
CAMERA_HEIGHT  = 0.40        # metres above the floor — raised so objects fill frame
CAMERA_RADIUS  = 0.60        # horizontal radius from contact midpoint
LOOK_HEIGHT    = 0.10        # Z of look-at target (mid-height of objects)


# ─────────────────────────────────────────────────────────────────────────────
# GPT + inventory helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_keys(keys_json_path: str) -> list[str]:
    with open(keys_json_path) as f:
        return json.load(f)


def sample_200(all_keys: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(200, len(all_keys)))


def gpt_pick_2_medium(candidate_categories: list[str]) -> list[str]:
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No OpenAI API key found.")
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a helpful assistant for a robotics simulation. "
        "Given a list of object category names from a household/indoor dataset, "
        "select exactly 2 categories whose objects are MEDIUM-to-small-SIZED — larger than "
        "hand-held items but NOT large furniture. Good examples: teddy_bear, book, "
        "basketball, backpack, box, pillow, laptop. The item should be at least of some height so we can see it from the ground (so no carpet or mat)."
        "Avoid tiny items (cups, pens) and large furniture (sofa, table, cabinet, microwave). "
        "Reply with ONLY a JSON array of exactly 2 strings from the input list, e.g.: "
        '[\"teddy_bear\", \"book\"]'
    )
    user_prompt = (
        f"Here are the candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        "Pick 2 that are medium-to-small-sized (toys, books, bags — not furniture, not tiny, and not huge)."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0.2,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    chosen = json.loads(raw)
    print(chosen)
    assert isinstance(chosen, list) and len(chosen) == 2
    for c in chosen:
        assert c in candidate_categories
    print(f"[GPT] Chose categories: {chosen}")
    return chosen


def get_model_for_category(category: str, seed: int) -> str:
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
        matches = [k for k in providers if k.startswith(f"{category}-")]
        if not matches:
            raise RuntimeError(f"Category '{category}' not found in inventory at '{path}'.")
        rng = random.Random(seed)
        chosen = rng.choice(matches)
        model_id = chosen.split("-", 1)[1]
        print(f"  [{category}] {len(matches)} model(s) available, picked model_id={model_id} (seed={seed})")
        return model_id
    raise RuntimeError("No object inventory file found.")


def build_object_config(name: str, category: str, model: str, idx: int) -> dict:
    """
    Build object config with identity (square/axis-aligned) orientation.
    Both objects are perpendicular to the world axes so AABB == true extents.
    """
    return {
        "type": "DatasetObject",
        "name": name,
        "category": category,
        "model": model,
        "position": [150.0 + idx * 10, 100.0, 100.0],
        "orientation": SQUARE_ORIENTATION,   # [0, 0, 0, 1] — axis-aligned
        "scale": [1, 1, 1]
        # "fixed_base": True,
        # "kinematic_only": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos) - np.array(eye_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()  # [qx, qy, qz, qw]


# ─────────────────────────────────────────────────────────────────────────────
# Capture + seg checks
# ─────────────────────────────────────────────────────────────────────────────

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


def _seg_and_touch_check(obj_names: tuple = ("obj1", "obj2")) -> tuple[bool, bool, bool, bool]:
    """
    Step 100 frames, then check seg_instance.

    Returns (exist_obj1: bool, exist_obj2: bool, both_exist: bool, touching: bool).
    exist_obj1 / exist_obj2 are True when that individual object is visible.
    touching is True when both masks are adjacent (pixel-level dilation overlap).
    """
    for _ in range(100):
        og.sim.step()

    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    instance_np  = raw_obs["data"].astype(np.int32)

    visible     = " ".join(id_to_labels.values())
    exist_obj1  = obj_names[0] in visible
    exist_obj2  = obj_names[1] in visible
    both_exist  = exist_obj1 and exist_obj2

    print(f"[seg] exist_{obj_names[0]}={exist_obj1}  exist_{obj_names[1]}={exist_obj2}  both={both_exist}")

    if not both_exist:
        return exist_obj1, exist_obj2, both_exist, False

    ids = {}
    for iid, label in id_to_labels.items():
        for name in obj_names:
            if name in label:
                ids[name] = int(iid)
                break

    if len(ids) < 2:
        print(f"[touch] Could not resolve IDs: {ids}")
        return exist_obj1, exist_obj2, True, False

    id_a = ids[obj_names[0]]
    id_b = ids[obj_names[1]]

    mask_a = (instance_np == id_a).astype(np.uint8)
    mask_b = (instance_np == id_b).astype(np.uint8)

    px_a = int(mask_a.sum())
    px_b = int(mask_b.sum())
    print(f"[touch] {obj_names[0]} pixels={px_a}  {obj_names[1]} pixels={px_b}")

    if px_a == 0 or px_b == 0:
        print(f"[touch] One mask is empty — cannot check adjacency")
        return exist_obj1, exist_obj2, True, False

    kernel    = np.ones((3, 3), dtype=np.uint8)
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)
    overlap   = bool(np.any((dilated_a > 0) & (mask_b > 0)))

    print(f"[touch] mask adjacency touching={overlap}")
    return exist_obj1, exist_obj2, True, overlap


# ─────────────────────────────────────────────────────────────────────────────
# Object placement — axis-aligned (square) orientation + AABB contact conditions
# ─────────────────────────────────────────────────────────────────────────────

def place_two_objects(env, scene, obj1, obj2, floor, seed: int) -> dict:
    """
    Step 1: Place BOTH objects on the floor via OnTop, then force both to
            square (identity) orientation and read their AABBs.
    Step 2: Reposition obj2 only, using obj1's AABB to satisfy:
              X: zero gap    — f2_xmin == f1_xmax  (no gap, no overlap in X)
              Y: 0.05 m overlap — f2_ymin = f1_ymax - 0.05
    obj1 is never moved after step 1.
    """
    SQUARE_ORI_TENSOR = th.tensor(SQUARE_ORIENTATION, dtype=th.float32)
    rng = random.Random(seed + 77)

    # ── Floor bbox for bounds checking ────────────────────────────────────────
    try:
        floor_bbox_min, floor_bbox_max = [x.cpu().numpy() for x in floor.aabb]
        print(f"[bbox] floor X=[{floor_bbox_min[0]:.3f}, {floor_bbox_max[0]:.3f}]"
              f"  Y=[{floor_bbox_min[1]:.3f}, {floor_bbox_max[1]:.3f}]")
    except Exception as e:
        print(f"[bbox] Could not get floor bbox: {e} — skipping check")
        floor_bbox_min = floor_bbox_max = None

    def within(pos_xy):
        if floor_bbox_min is None:
            return True
        return (floor_bbox_min[0] <= pos_xy[0] <= floor_bbox_max[0] and
                floor_bbox_min[1] <= pos_xy[1] <= floor_bbox_max[1])

    # ── Step 1: Place BOTH objects on floor, force square orientation ─────────
    for obj in (obj1, obj2):
        obj.states[object_states.OnTop].set_value(floor, True)
        for _ in range(10):
            og.sim.step()
        # Force axis-aligned orientation (physics may have rotated them)
        pos_settled, _ = obj.get_position_orientation()
        obj.set_position_orientation(position=pos_settled, orientation=SQUARE_ORI_TENSOR)
        obj.keep_still()
        for _ in range(5):
            og.sim.step()

    # Read AABB of both objects after orientation is locked
    f1_min, f1_max = [x.cpu().numpy() for x in obj1.aabb]
    f2_min, f2_max = [x.cpu().numpy() for x in obj2.aabb]
    f1_pos = obj1.get_position_orientation()[0].cpu().numpy()
    f2_pos = obj2.get_position_orientation()[0].cpu().numpy()

    f2_half_x = (f2_max[0] - f2_min[0]) / 2.0
    f2_half_y = (f2_max[1] - f2_min[1]) / 2.0
    f2_z      = f2_pos[2]   # keep obj2's settled floor height

    print(f"[obj1] pos={f1_pos.round(3)}  extents={(f1_max - f1_min).round(3)}")
    print(f"[obj2] pos={f2_pos.round(3)}  extents={(f2_max - f2_min).round(3)}  z={f2_z:.3f}")

    # ── Step 2: Reposition obj2 to satisfy contact conditions ────────────────
    drift_x = rng.uniform(-GAP_DRIFT, GAP_DRIFT)

    # X: zero gap — obj2_xmin == obj1_xmax  →  obj2_centre_x = f1_xmax + drift_x + f2_half_x
    target_x = f1_max[0] + f2_half_x
    # Y: 0.05 m overlap — obj2_ymin = obj1_ymax - 0.05  →  obj2_centre_y = (f1_ymax - 0.05) + f2_half_y
    target_y = (f1_max[1] - Y_TOUCH_OVERLAP) + f2_half_y - 0.05
    target   = np.array([target_x, target_y, f2_z])

    print(f"[place] obj2 target={target.round(3)}  drift_x={drift_x:+.3f}")
    print(f"[place] X: f2_xmin == f1_xmax (zero gap)  |  Y: {Y_TOUCH_OVERLAP:.2f} m overlap")

    if not within(f1_pos[:2]):
        raise RuntimeError(f"obj1 landed outside floor bbox: {f1_pos[:2]}")
    if not within(target[:2]):
        raise RuntimeError(f"obj2 target outside floor bbox: {target[:2]}")

    obj2.set_position_orientation(
        position=th.tensor(target, dtype=th.float32),
        orientation=SQUARE_ORI_TENSOR,
    )
    obj2.keep_still()
    for _ in range(30):
        og.sim.step()

    # ── Verify final AABB ─────────────────────────────────────────────────────
    f1_final_min, f1_final_max = [x.cpu().numpy() for x in obj1.aabb]
    f2_final_min, f2_final_max = [x.cpu().numpy() for x in obj2.aabb]

    gap = max(0, f2_final_min[0] - f1_final_max[0])   # target: 0.000

    target_x = f1_max[0] + f2_half_x - gap
    target_y = (f1_max[1] - Y_TOUCH_OVERLAP) + f2_half_y - 0.05
    target   = np.array([target_x, target_y, f2_z])
    obj2.set_position_orientation(
        position=th.tensor(target, dtype=th.float32),
        orientation=SQUARE_ORI_TENSOR,
    )
    obj2.keep_still()
    for _ in range(30):
        og.sim.step()

    f1_final_min, f1_final_max = [x.cpu().numpy() for x in obj1.aabb]
    f2_final_min, f2_final_max = [x.cpu().numpy() for x in obj2.aabb]
    x_gap = f2_final_min[0] - f1_final_max[0]
    y_gap = f2_final_min[1] - f1_final_max[1]   # target: -0.050

    print(f"[verify] x_gap={x_gap:.4f}  (target: 0.000)")
    print(f"[verify] y_gap={y_gap:.4f}  (target: -{Y_TOUCH_OVERLAP:.3f})")

    k = 0
    while x_gap < 0.05 and y_gap < 0.05 and k < 20:
        k += 1
        f1_pos = obj1.get_position_orientation()[0].cpu().numpy()
        f2_pos = obj2.get_position_orientation()[0].cpu().numpy()
        target_x = f1_max[0] + f2_half_x - gap - 0.01 * k
        target_y = (f1_max[1] - Y_TOUCH_OVERLAP) + f2_half_y - 0.05
        target   = np.array([target_x, target_y, f2_z])
        obj2.set_position_orientation(
            position=th.tensor(target, dtype=th.float32),
            orientation=SQUARE_ORI_TENSOR,
        )
        obj2.keep_still()
        for _ in range(30):
            og.sim.step()

        f1_final_min, f1_final_max = [x.cpu().numpy() for x in obj1.aabb]
        f2_final_min, f2_final_max = [x.cpu().numpy() for x in obj2.aabb]
        x_gap = f2_final_min[0] - f1_final_max[0]
        y_gap = f2_final_min[1] - f1_final_max[1]   # target: -0.050

        print(f"[verify] x_gap={x_gap:.4f}  (target: 0.000)")
        print(f"[verify] y_gap={y_gap:.4f}  (target: -{Y_TOUCH_OVERLAP:.3f})")

    
    obj1.set_position_orientation(
        position=th.tensor(f1_pos, dtype=th.float32),
        orientation=SQUARE_ORI_TENSOR,
    )
    obj1.keep_still()
    for _ in range(30):
            og.sim.step()

    obj2.set_position_orientation(
        position=th.tensor(f2_pos, dtype=th.float32),
        orientation=SQUARE_ORI_TENSOR,
    )
    obj2.keep_still()
    for _ in range(30):
            og.sim.step()

    f1_final_min, f1_final_max = [x.cpu().numpy() for x in obj1.aabb]
    f2_final_min, f2_final_max = [x.cpu().numpy() for x in obj2.aabb]
    x_gap = f2_final_min[0] - f1_final_max[0]
    y_gap = f2_final_min[1] - f1_final_max[1]   # target: -0.050

    print(f"[verify] x_gap={x_gap:.4f}  (target: 0.000)")
    print(f"[verify] y_gap={y_gap:.4f}  (target: -{Y_TOUCH_OVERLAP:.3f})")

    contact_mid = np.array([
        (f1_final_max[0] + f2_final_min[0]) / 2.0,
        (f1_final_max[1] + f2_final_min[1]) / 2.0,
        (f1_final_max[2] + f2_final_max[2]) / 2.0 * 0.5,
    ])

    return {
        "obj1_aabb":   {"min": f1_final_min.tolist(), "max": f1_final_max.tolist()},
        "obj2_aabb":   {"min": f2_final_min.tolist(), "max": f2_final_max.tolist()},
        "x_gap":       float(x_gap),
        "y_gap":       float(y_gap),
        "contact_mid": contact_mid.tolist(),
        "drift_x":     float(drift_x),
        "drift_y":     0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# render_and_save — 24 cameras every 15° around the contact centre
# ─────────────────────────────────────────────────────────────────────────────

def render_and_save(seed: int, output_dir: str,
                    placement: dict) -> tuple[dict, dict, dict, dict]:
    """
    Renders NUM_VIEWS (24) cameras placed uniformly every VIEW_STEP_DEG (15°)
    on a horizontal circle of radius CAMERA_RADIUS centred on the contact
    midpoint, at height CAMERA_HEIGHT, all looking at LOOK_HEIGHT on the
    contact midpoint.

    View k is at azimuth k * 15° (0° = +X axis, counter-clockwise).

    Returns (exist_obj1_flags, exist_obj2_flags, touching_flags, camera_poses).
    """
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    contact_mid = np.array(placement["contact_mid"])
    cx, cy      = contact_mid[0], contact_mid[1]
    look_target = np.array([cx, cy, LOOK_HEIGHT])

    os.makedirs(output_dir, exist_ok=True)
    camera_poses      = {}
    exist_obj1_flags  = {}
    exist_obj2_flags  = {}
    touching_flags    = {}

    for k in range(NUM_VIEWS):
        azimuth_deg = k * VIEW_STEP_DEG
        azimuth_rad = np.deg2rad(azimuth_deg)

        eye = np.array([
            cx + CAMERA_RADIUS * np.cos(azimuth_rad),
            cy + CAMERA_RADIUS * np.sin(azimuth_rad),
            CAMERA_HEIGHT,
        ])

        fname = f"{k}.png"
        fpath = os.path.join(output_dir, fname)

        print(f"\n[camera {k}] azimuth={azimuth_deg:.0f}°  eye={eye.round(3)}")
        pose = _set_camera_and_capture(eye, look_target, fpath)
        camera_poses[fname] = {**pose, "azimuth_deg": azimuth_deg}

        exist_obj1, exist_obj2, both_exist, touching = _seg_and_touch_check(("obj1", "obj2"))
        exist_obj1_flags[f"exist_obj1_{k}"] = exist_obj1
        exist_obj2_flags[f"exist_obj2_{k}"] = exist_obj2
        touching_flags[f"touching_{k}"]     = touching

    return exist_obj1_flags, exist_obj2_flags, touching_flags, camera_poses


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
    parser.add_argument("--output_root", type=str, default="renders_touching")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 100
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"  Orientation: SQUARE (axis-aligned) [0, 0, 0, 1] for both objects")
    print(f"  Contact: X no-gap  |  Y overlap={Y_TOUCH_OVERLAP:.3f} m")
    print(f"{'='*70}\n")

    all_keys          = load_keys(args.keys_json)
    sampled           = sample_200(all_keys, seed=seed)
    chosen_categories = gpt_pick_2_medium(sampled)
    models            = [get_model_for_category(cat, seed) for cat in chosen_categories]

    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"] = [
        build_object_config(f"obj{i+1}", cat, model, i)
        for i, (cat, model) in enumerate(zip(chosen_categories, models))
    ]

    env   = og.Environment(configs=config)
    scene = env.scene

    obj1  = scene.object_registry("name", "obj1")
    obj2  = scene.object_registry("name", "obj2")
    floor = scene.object_registry("name", args.floor)

    if floor is None:
        print(f"[ERROR] Floor '{args.floor}' not found — exiting with code 2")
        raise SystemExit(2)

    placement = place_two_objects(env, scene, obj1, obj2, floor, seed=seed)

    exist_obj1_flags, exist_obj2_flags, touching_flags, camera_poses = render_and_save(
        seed=seed, output_dir=run_dir, placement=placement)

    # ── Collect object metadata ───────────────────────────────────────────────
    objects_meta = {}
    for i, (obj, cat) in enumerate(zip([obj1, obj2], chosen_categories), start=1):
        pos, quat = obj.get_position_orientation()
        objects_meta[f"obj{i}"] = {
            "category":        cat,
            "model":           models[i - 1],
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    metadata = {
        "scene":       args.scene,
        "room":        args.room,
        "run_idx":     args.run_idx,
        "seed":        seed,
        "floor_name":  args.floor,
        "layout":      "touching_pair",
        "orientation": "square_axis_aligned",
        "x_gap":       placement["x_gap"],
        "y_gap":       placement["y_gap"],
        "drift_x":     placement["drift_x"],
        "drift_y":     placement["drift_y"],
        "obj1_aabb":   placement["obj1_aabb"],
        "obj2_aabb":   placement["obj2_aabb"],
        **exist_obj1_flags,   # exist_obj1_0 .. exist_obj1_23
        **exist_obj2_flags,   # exist_obj2_0 .. exist_obj2_23
        **touching_flags,     # touching_0  .. touching_23
        "objects":      objects_meta,
        "camera_poses": camera_poses,
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # Exit 0 if all views have both objects visible and touching
    all_exist = all(
        exist_obj1_flags[f"exist_obj1_{k}"] and exist_obj2_flags[f"exist_obj2_{k}"]
        for k in range(NUM_VIEWS)
    )
    all_touch = all(touching_flags[f"touching_{k}"] for k in range(NUM_VIEWS))
    raise SystemExit(0 if (all_exist and all_touch) else 1)


if __name__ == "__main__":
    main()