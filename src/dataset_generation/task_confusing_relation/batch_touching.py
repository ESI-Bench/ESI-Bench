"""
batch_touching.py

Single-run script: loads ONE scene restricted to ONE room, places 2 GPT-selected
medium-sized objects (toys, books — not furniture) so they are touching, renders
6 camera views, and saves a metadata JSON.

Views:
  0.png — X-axis view, slightly +Y offset  (sees contact zone from right side)
  1.png — X-axis view, slightly -Y offset  (sees contact zone from left side)
  2.png — Y-axis view, slightly +X offset  (sees contact zone from front)
  3.png — Y-axis view, slightly -X offset  (sees contact zone from back)
  4.png — separation view from +X (pulls far back, shows gap between objects)
  5.png — separation view from -X (pulls far back from other side)

JSON flags per view:
  exist_0 .. exist_5   — both objects visible in seg_instance
  touching_0 .. touching_5 — pixel-mask adjacency: masks touch at any point

Called once per (scene, room, run_idx) by batch_touching.sh.
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

# Contact placement parameters (matching your script's values ± 0.05 random drift)
X_GAP_BASE     =  0.01   # f2_xmin = f1_xmax + X_GAP_BASE  (nearly touching in X)
Y_OVERLAP_BASE = -0.01   # f2_ymin = f1_ymax + Y_OVERLAP_BASE (slightly overlapping → touching)
GAP_DRIFT      =  0.05   # uniform random drift applied to both


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
    print (chosen)
    assert isinstance(chosen, list) and len(chosen) == 2
    for c in chosen:
        assert c in candidate_categories
    print(f"[GPT] Chose categories: {chosen}")
    return chosen


def get_model_for_category(category: str, seed: int) -> str:
    """
    Look up all model IDs for a category from the inventory and pick one
    randomly using the seed, so different runs get different model variants.
    Raises RuntimeError if the category is not found.
    """
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
    return {
        "type": "DatasetObject",
        "name": name,
        "category": category,
        "model": model,
        "position": [150.0 + idx * 10, 100.0, 100.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1, 1, 1],
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


def _seg_and_touch_check(obj_names: tuple = ("obj1", "obj2")) -> tuple[bool, bool, bool]:
    """
    Step 100 frames, then check seg_instance.

    Returns (both_exist: bool, any_exist: bool, touching: bool).
      both_exist: True if BOTH objects are visible
      any_exist:  True if AT LEAST ONE object is visible
      touching:   pixel-mask adjacency — only meaningful when both_exist=True
    """
    for _ in range(100):
        og.sim.step()

    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    instance_np  = raw_obs["data"].astype(np.int32)

    visible    = " ".join(id_to_labels.values())
    found      = [n for n in obj_names if n in visible]
    both_exist = len(found) == 2
    any_exist  = len(found) > 0

    print(f"[seg] found={found}  both_exist={both_exist}  any_exist={any_exist}")

    if not both_exist:
        return both_exist, any_exist, False

    ids = {}
    for iid, label in id_to_labels.items():
        for name in obj_names:
            if name in label:
                ids[name] = int(iid)
                break

    if len(ids) < 2:
        print(f"[touch] Could not resolve IDs: {ids}")
        return True, True, False

    id_a = ids[obj_names[0]]
    id_b = ids[obj_names[1]]

    mask_a = (instance_np == id_a).astype(np.uint8)
    mask_b = (instance_np == id_b).astype(np.uint8)

    px_a = int(mask_a.sum())
    px_b = int(mask_b.sum())
    print(f"[touch] {obj_names[0]} pixels={px_a}  {obj_names[1]} pixels={px_b}")

    if px_a == 0 or px_b == 0:
        print(f"[touch] One mask is empty — cannot check adjacency")
        return True, True, False

    kernel    = np.ones((3, 3), dtype=np.uint8)
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)
    overlap   = bool(np.any((dilated_a > 0) & (mask_b > 0)))

    print(f"[touch] mask adjacency touching={overlap}")
    return True, True, overlap


# ─────────────────────────────────────────────────────────────────────────────
# Object placement
# ─────────────────────────────────────────────────────────────────────────────

def place_two_objects(env, scene, obj1, obj2, floor, seed: int) -> dict:
    """
    Place obj1 on floor via OnTop, then place obj2 so it touches obj1.

    Contact geometry (matching your script ± GAP_DRIFT):
      X: f2_xmin = f1_xmax + X_GAP_BASE + drift_x   (nearly touching in X)
      Y: f2_ymin = f1_ymax + Y_OVERLAP_BASE + drift_y (slightly overlapping in Y)
      Z: same floor height as obj2's landed Z

    Both objects are checked against the floor bbox. If either falls outside,
    obj1 is resampled via OnTop (up to MAX_ATTEMPTS). Raises RuntimeError if
    all attempts fail.

    Returns dict with AABB info for metadata.
    """
    MAX_ATTEMPTS = 3
    rng = random.Random(seed + 77)

    # ── Get floor bbox ────────────────────────────────────────────────────────
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

    # ── Measure obj2 once, park it far away ───────────────────────────────────
    obj2.states[object_states.OnTop].set_value(floor, True)
    for _ in range(10):
        og.sim.step()

    f2_min, f2_max = [x.cpu().numpy() for x in obj2.aabb]
    f2_pos, f2_ori = obj2.get_position_orientation()
    f2_pos = f2_pos.cpu().numpy()
    f2_half_x = (f2_max[0] - f2_min[0]) / 2.0
    f2_half_y = (f2_max[1] - f2_min[1]) / 2.0
    f2_z      = f2_pos[2]
    print(f"[obj2] extents={(f2_max - f2_min).round(3)}")

    obj2.set_position_orientation(
        position=th.tensor([500.0, 500.0, 100.0], dtype=th.float32),
        orientation=f2_ori,
    )
    obj2.keep_still()
    for _ in range(30):
        og.sim.step()

    # ── Resample loop: drop obj1, compute obj2 target, check floor bbox ───────
    drift_x = drift_y = 0.0
    target = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        drift_x = rng.uniform(-GAP_DRIFT, GAP_DRIFT)
        drift_y = rng.uniform(-GAP_DRIFT, GAP_DRIFT)

        obj1.states[object_states.OnTop].set_value(floor, True)
        for _ in range(10):
            og.sim.step()

        f1_min, f1_max = [x.cpu().numpy() for x in obj1.aabb]
        f1_pos, _ = obj1.get_position_orientation()
        f1_pos = f1_pos.cpu().numpy()
        print(f"[attempt {attempt}] obj1 pos={f1_pos.round(3)}"
              f"  extents={(f1_max - f1_min).round(3)}")

        target_x = f1_max[0] + X_GAP_BASE     + drift_x + f2_half_x
        target_y = f1_max[1] + Y_OVERLAP_BASE  + drift_y + f2_half_y
        target   = np.array([target_x, target_y, f2_z])

        obj1_ok = within(f1_pos[:2])
        obj2_ok = within(target[:2])

        if obj1_ok and obj2_ok:
            print(f"[bbox] Both within floor bbox ✓  (attempt {attempt})")
            break

        outside = ([] if obj1_ok else ["obj1"]) + ([] if obj2_ok else ["obj2"])
        print(f"[bbox] Attempt {attempt}: {outside} outside — resampling")
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed. Last offenders: {outside}")

    print(f"[place] obj2 target={target.round(3)}  drift_x={drift_x:+.3f}  drift_y={drift_y:+.3f}")

    # ── Teleport obj2 to touching position ────────────────────────────────────
    obj2.set_position_orientation(
        position=th.tensor(target, dtype=th.float32),
        orientation=f2_ori,
    )
    obj2.keep_still()
    for _ in range(30):
        og.sim.step()

    # ── Final AABB verification ───────────────────────────────────────────────
    f1_final_min, f1_final_max = [x.cpu().numpy() for x in obj1.aabb]
    f2_final_min, f2_final_max = [x.cpu().numpy() for x in obj2.aabb]
    x_gap = f2_final_min[0] - f1_final_max[0]
    y_gap = f2_final_min[1] - f1_final_max[1]
    print(f"[verify] x_gap={x_gap:.4f}  y_gap={y_gap:.4f}")

    # Gap midpoint in XY — used as look target for contact views
    contact_mid = np.array([
        (f1_final_max[0] + f2_final_min[0]) / 2.0,
        (f1_final_max[1] + f2_final_min[1]) / 2.0,
        (f1_final_max[2] + f2_final_max[2]) / 2.0 * 0.5,  # halfway up objects
    ])

    return {
        "obj1_aabb":  {"min": f1_final_min.tolist(), "max": f1_final_max.tolist()},
        "obj2_aabb":  {"min": f2_final_min.tolist(), "max": f2_final_max.tolist()},
        "x_gap":      float(x_gap),
        "y_gap":      float(y_gap),
        "contact_mid": contact_mid,
        "drift_x":    float(drift_x),
        "drift_y":    float(drift_y),
    }

# ─────────────────────────────────────────────────────────────────────────────
# render_and_save
# ─────────────────────────────────────────────────────────────────────────────
def render_and_save(seed: int, output_dir: str,
                    placement: dict) -> tuple[dict, dict, dict]:
    """
    Renders 6 base views, then for each base view sweeps left/right in 15-degree
    increments (orbiting around look_contact at the same radius) until the
    touching state flips. All swept frames are saved and flagged in metadata.

    Sweep filenames: {base_idx}_left15.png, {base_idx}_right15.png,
                     {base_idx}_left30.png, {base_idx}_right30.png, ...

    Returns (exist_flags, touching_flags, camera_poses).
    Keys in exist_flags / touching_flags follow the same naming as the filenames
    (without .png), e.g. exist_0, touching_2_left30, etc.
    """
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    f1_min = np.array(placement["obj1_aabb"]["min"])
    f1_max = np.array(placement["obj1_aabb"]["max"])
    f2_min = np.array(placement["obj2_aabb"]["min"])
    f2_max = np.array(placement["obj2_aabb"]["max"])

    cz     = 0.025
    look_z = 0.1

    obj_span_x    = f2_max[0] - f1_min[0]
    obj_span_y    = max(f1_max[1] - f1_min[1], f2_max[1] - f2_min[1])
    close_dist    = obj_span_x * 0.8 + 0.3
    y_side_offset = obj_span_y * 0.3 + 0.05
    far_dist      = obj_span_x * 1.5 + 0.5
    x_side_offset = obj_span_x * 0.15 + 0.05

    cx = (f1_max[0] + f2_min[0]) / 2.0
    cy = (f1_max[1] + f2_min[1]) / 2.0
    look_contact = np.array([cx, cy, look_z])

    os.makedirs(output_dir, exist_ok=True)
    camera_poses   = {}
    exist_flags    = {}
    touching_flags = {}

    # ── shoot: render one frame, run seg+touch check, store results ───────────
    def shoot(key: str, eye: np.ndarray, look_target: np.ndarray):
        fname = f"{key}.png"
        camera_poses[fname] = _set_camera_and_capture(
            eye, look_target, os.path.join(output_dir, fname))
        both_exist, any_exist, touching = _seg_and_touch_check(("obj1", "obj2"))
        exist_flags[f"exist_{key}"]     = both_exist
        touching_flags[f"touching_{key}"] = touching
        return both_exist, any_exist, touching

    # ── sweep_one_direction: step along one side until a stop condition ───────
    def sweep_one_direction(base_key: str, base_angle: float, radius: float,
                            sign: int, base_touching: bool):
        """
        Walk in 15° steps in one direction.

        Stop conditions:
          - both_exist=True  AND touching flips  → stop (found what we want)
          - any_exist=False (BOTH objects gone)   → stop
          - one object missing (any_exist=True, both_exist=False) → keep going

        For not-touching base:
          Phase 1: walk until touching found (or both gone).
          Phase 2: once touching found, keep going until not-touching or both gone.
        """
        MAX_STEPS = 24
        STEP_DEG  = 15.0
        side      = "left" if sign == 1 else "right"

        if base_touching:
            # Walk until touching flips to not-touching, or both objects gone
            for step in range(1, MAX_STEPS + 1):
                deg   = step * STEP_DEG
                key   = f"{base_key}_{side}{int(deg)}"
                angle = base_angle + sign * np.deg2rad(deg)
                eye   = np.array([cx + radius * np.cos(angle),
                                   cy + radius * np.sin(angle), cz])
                print(f"[sweep]   {key}  angle={np.rad2deg(angle):.1f}°")
                both_exist, any_exist, touching = shoot(key, eye, look_contact)
                if not any_exist:
                    print(f"[sweep]   {side} stopped — both objects gone at {key}")
                    return
                if both_exist and not touching:
                    print(f"[sweep]   {side} stopped — flipped to not-touching at {key}")
                    return
            print(f"[sweep]   {side} exhausted {MAX_STEPS} steps")

        else:
            # Phase 1: find touching
            touching_found_at = None
            for step in range(1, MAX_STEPS + 1):
                deg   = step * STEP_DEG
                key   = f"{base_key}_{side}{int(deg)}"
                angle = base_angle + sign * np.deg2rad(deg)
                eye   = np.array([cx + radius * np.cos(angle),
                                   cy + radius * np.sin(angle), cz])
                print(f"[sweep]   {key}  angle={np.rad2deg(angle):.1f}°")
                both_exist, any_exist, touching = shoot(key, eye, look_contact)
                if not any_exist:
                    print(f"[sweep]   {side} phase-1 stopped — both gone at {key}")
                    return
                if both_exist and touching:
                    print(f"[sweep]   {side} phase-1 found touching at {key} — phase 2")
                    touching_found_at = step
                    break
            if touching_found_at is None:
                print(f"[sweep]   {side} phase-1 exhausted without finding touching")
                return

            # Phase 2: keep going until not-touching or both gone
            for step in range(touching_found_at + 1, touching_found_at + MAX_STEPS + 1):
                deg   = step * STEP_DEG
                key   = f"{base_key}_{side}{int(deg)}"
                angle = base_angle + sign * np.deg2rad(deg)
                eye   = np.array([cx + radius * np.cos(angle),
                                   cy + radius * np.sin(angle), cz])
                print(f"[sweep]   {key}  angle={np.rad2deg(angle):.1f}°")
                both_exist, any_exist, touching = shoot(key, eye, look_contact)
                if not any_exist:
                    print(f"[sweep]   {side} phase-2 stopped — both gone at {key}")
                    return
                if both_exist and not touching:
                    print(f"[sweep]   {side} phase-2 stopped — flipped to not-touching at {key}")
                    return
            print(f"[sweep]   {side} phase-2 exhausted {MAX_STEPS} steps")

    # ── sweep_from: launch both directions from a base view ──────────────────
    def sweep_from(base_key: str, base_eye: np.ndarray, base_touching: bool):
        dx         = base_eye[0] - cx
        dy         = base_eye[1] - cy
        base_angle = np.arctan2(dy, dx)
        radius     = np.sqrt(dx**2 + dy**2)
        state_str  = "touching" if base_touching else "not-touching"
        print(f"[sweep] Base view {base_key} is {state_str} — sweeping both directions")
        sweep_one_direction(base_key, base_angle, radius, +1, base_touching)  # left
        sweep_one_direction(base_key, base_angle, radius, -1, base_touching)  # right

    # ── 6 base views ──────────────────────────────────────────────────────────
    base_views = [
        ("0", np.array([cx + close_dist,    cy + y_side_offset, cz])),
        ("1", np.array([cx + close_dist,    cy - y_side_offset, cz])),
        ("2", np.array([cx + x_side_offset, cy + close_dist,    cz])),
        ("3", np.array([cx - x_side_offset, cy + close_dist,    cz])),
        ("4", np.array([cx + far_dist,      cy,                  cz])),
        ("5", np.array([cx - far_dist,      cy,                  cz])),
    ]

    for key, eye in base_views:
        both_exist, any_exist, touching = shoot(key, eye, look_contact)
        if both_exist:
            sweep_from(key, eye, touching)

    return exist_flags, touching_flags, camera_poses

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

    seed = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF)
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"{'='*70}\n")

    all_keys = load_keys(args.keys_json)
    sampled  = sample_200(all_keys, seed=seed)
    chosen_categories = gpt_pick_2_medium(sampled)
    models = [get_model_for_category(cat, seed) for cat in chosen_categories]

    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"] = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"] = [args.room]
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

    exist_flags, touching_flags, camera_poses = render_and_save(
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
        "scene":      args.scene,
        "room":       args.room,
        "run_idx":    args.run_idx,
        "seed":       seed,
        "floor_name": args.floor,
        "layout":     "touching_pair",
        "x_gap":      placement["x_gap"],
        "y_gap":      placement["y_gap"],
        "drift_x":    placement["drift_x"],
        "drift_y":    placement["drift_y"],
        "obj1_aabb":  placement["obj1_aabb"],
        "obj2_aabb":  placement["obj2_aabb"],
        **exist_flags,      # exist_0 .. exist_5
        **touching_flags,   # touching_0 .. touching_5
        "objects":      objects_meta,
        "camera_poses": camera_poses,
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # Exit 0 if all 6 views exist + first 4 touching flags true
    all_exist = all(exist_flags[f"exist_{i}"]       for i in range(6))
    touch_ok  = all(touching_flags[f"touching_{i}"] for i in range(4))
    raise SystemExit(0 if (all_exist and touch_ok) else 1)


if __name__ == "__main__":
    main()