"""
place_objects.py

Single-run script: loads ONE scene restricted to ONE room, places 3 GPT-selected
small objects on the room floor, renders 5 camera views, and saves a metadata JSON.

Called once per (scene, room, run_idx) by run_experiments.sh.

Output layout:
    <output_root>/<scene_name>/<room_name>_<run_idx>/
        0.png
        0_final_1.png
        0_final_2.png
        0_middle_1.png
        0_middle_2.png
        metadata.json

Usage:
    python place_objects.py \
        --scene Merom_1_int \
        --room  living_room_0 \
        --floor floors_xzlkei_0 \
        --run_idx 0
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


# ─────────────────────────────────────────────────────────────────────────────
# GPT + inventory helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_keys(keys_json_path: str) -> list[str]:
    with open(keys_json_path) as f:
        return json.load(f)


def sample_200(all_keys: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(200, len(all_keys)))


def gpt_pick_3_small(candidate_categories: list[str]) -> list[str]:
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No OpenAI API key found.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a helpful assistant for a robotics simulation. "
        "Given a list of object category names from a household/indoor dataset, "
        "select exactly 3 categories whose objects have a SMALL physical footprint "
        "(small in X and Y dimensions, i.e. table-top / hand-held size). "
        "But it can be tall in Z. "
        "Avoid large furniture, vehicles, or anything that wouldn't fit on a table. "
        "Reply with ONLY a JSON array of exactly 3 strings from the input list, e.g.: "
        '[\"candle\", \"apple\", \"mug\"]'
    )
    user_prompt = (
        f"Here are the candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        "Pick 3 that are small in footprint (X/Y) and preferably tall."
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
    assert isinstance(chosen, list) and len(chosen) == 3
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

def get_floor_bbox(floor):
    try:
        return floor.aabb[0].cpu().numpy(), floor.aabb[1].cpu().numpy()
    except Exception as e:
        print(f"[bbox] Could not get floor bbox: {e}")
        return None, None


def within_floor_bbox(pos_xy, bbox_min, bbox_max):
    return (bbox_min[0] <= pos_xy[0] <= bbox_max[0] and
            bbox_min[1] <= pos_xy[1] <= bbox_max[1])


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
    return Rotation.from_matrix(rot_matrix).as_quat()   # [qx, qy, qz, qw]


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture(path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# render_and_save
# ─────────────────────────────────────────────────────────────────────────────

def render_and_save(objects, seed: int, output_dir: str = ".") -> tuple[bool, dict]:
    """
    Five camera views orbiting the centroid of all three objects.
    Angles clockwise from -Y (front) viewed top-down:
      <seed>.png          —   0°  front
      <seed>_final_1.png  —  90°  right (+X)
      <seed>_final_2.png  — 270°  left  (-X)
      <seed>_middle_1.png —  60°  right-mid
      <seed>_middle_2.png — 300°  left-mid

    After the front view, checks seg_instance: finds IDs for obj1/obj2/obj3
    in id_to_labels, then verifies those IDs appear in the rendered pixel tensor.

    Returns (success: bool, camera_poses: dict).
    """
    # ── Add segmentation modalities and warm up ───────────────────────────────
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # ── Compute centroid and orbit radius ─────────────────────────────────────
    positions = []
    for obj in objects:
        omin, omax = [x.cpu().numpy() for x in obj.aabb]
        positions.append(np.array([(omin[0] + omax[0]) / 2,
                                   (omin[1] + omax[1]) / 2]))
    positions = np.array(positions)
    cx = positions[:, 0].mean()
    cy = positions[:, 1].mean()
    cz = 0.025

    x_spread = positions[:, 0].max() - positions[:, 0].min()
    y_spread = positions[:, 1].max() - positions[:, 1].min()
    half_diag = np.sqrt((x_spread / 2) ** 2 + (y_spread / 2) ** 2)
    d = half_diag + 0.75

    look_target = np.array([cx, cy, 0.1])
    os.makedirs(output_dir, exist_ok=True)

    camera_poses = {}

    def _place_and_capture(angle_deg: float, name: str):
        a = np.radians(angle_deg)
        eye = np.array([cx + d * np.sin(a), cy - d * np.cos(a), cz])
        quat = look_at_quaternion(eye, look_target)
        og.sim._viewer_camera.set_position_orientation(eye, quat)
        _capture(os.path.join(output_dir, name))
        camera_poses[name] = {
            "position": eye.tolist(),
            "quaternion_xyzw": quat.tolist(),
            "angle_deg": angle_deg,
        }

    # ── Front view ────────────────────────────────────────────────────────────
    _place_and_capture(0, f"{seed}.png")

    for _ in range(100):
        og.sim.step()

    raw_obs = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    print(f"[seg] id_to_labels: {id_to_labels}")

    # Check "obj1", "obj2", "obj3" appear in any label value
    visible_names = " ".join(id_to_labels.values())
    missing = [name for name in ("obj1", "obj2", "obj3") if name not in visible_names]

    if missing:
        print(f"[seg] Front view missing {missing} — skipping seed {seed}")
        return False, camera_poses

    print("[seg] All 3 objects visible in front view ✓")

    # ── Remaining four views ──────────────────────────────────────────────────
    _place_and_capture( 90, f"{seed}_final_1.png")
    _place_and_capture(270, f"{seed}_final_2.png")
    _place_and_capture( 60, f"{seed}_middle_1.png")
    _place_and_capture(300, f"{seed}_middle_2.png")

    return True, camera_poses


# ─────────────────────────────────────────────────────────────────────────────
# Object placement
# ─────────────────────────────────────────────────────────────────────────────

def place_three_objects(env, scene, objects, floor, seed: int):
    MAX_ATTEMPTS = 3
    rng = random.Random(seed + 999)
    obj1, obj2, obj3 = objects

    floor_bbox_min, floor_bbox_max = get_floor_bbox(floor)
    if floor_bbox_min is not None:
        print(f"[bbox] X=[{floor_bbox_min[0]:.3f}, {floor_bbox_max[0]:.3f}]"
              f"  Y=[{floor_bbox_min[1]:.3f}, {floor_bbox_max[1]:.3f}]")
    else:
        print("[bbox] Warning: floor bbox unavailable — skipping check")

    # ── Measure obj2, park far away ───────────────────────────────────────────
    obj2.states[object_states.OnTop].set_value(floor, True)
    for _ in range(10): og.sim.step()
    f2_min, f2_max = [x.cpu().numpy() for x in obj2.aabb]
    f2_half_x = (f2_max[0] - f2_min[0]) / 2.0
    f2_z   = obj2.get_position_orientation()[0].cpu().numpy()[2]
    f2_ori = obj2.get_position_orientation()[1]
    print(f"[{obj2.name}] extents: {(f2_max - f2_min).round(3)}")
    obj2.set_position_orientation(position=th.tensor([500.0, 500.0, 100.0], dtype=th.float32), orientation=f2_ori)
    obj2.keep_still()
    for _ in range(30): og.sim.step()

    # ── Measure obj3, park far away ───────────────────────────────────────────
    obj3.states[object_states.OnTop].set_value(floor, True)
    for _ in range(10): og.sim.step()
    f3_min, f3_max = [x.cpu().numpy() for x in obj3.aabb]
    f3_half_x = (f3_max[0] - f3_min[0]) / 2.0
    f3_z   = obj3.get_position_orientation()[0].cpu().numpy()[2]
    f3_ori = obj3.get_position_orientation()[1]
    print(f"[{obj3.name}] extents: {(f3_max - f3_min).round(3)}")
    obj3.set_position_orientation(position=th.tensor([500.0, 510.0, 100.0], dtype=th.float32), orientation=f3_ori)
    obj3.keep_still()
    for _ in range(30): og.sim.step()

    # ── Y layout pattern ──────────────────────────────────────────────────────
    pattern = rng.randint(0, 2)
    sign    = rng.choice([-1, 1])
    if pattern == 0:
        dy2 = sign  * rng.uniform(0.125, 0.175)
        dy3 = sign  * rng.uniform(0.125, 0.175)
    elif pattern == 1:
        dy2 = -sign * rng.uniform(0.125, 0.175)
        dy3 = -sign * rng.uniform(0.125, 0.175)
    else:
        dy2 = rng.uniform(-0.05, 0.05)
        dy3 = sign  * rng.uniform(0.125, 0.175)
    print(f"  layout=pattern{pattern}  dy2={dy2:+.3f}  dy3={dy3:+.3f}")

    # ── Sample obj1, compute layout, check bbox ───────────────────────────────
    t2 = t3 = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        obj1.states[object_states.OnTop].set_value(floor, True)
        for _ in range(10): og.sim.step()

        f1_min, f1_max = [x.cpu().numpy() for x in obj1.aabb]
        f1_pos = obj1.get_position_orientation()[0].cpu().numpy()
        print(f"[attempt {attempt}] [{obj1.name}] pos={f1_pos.round(3)}"
              f"  extents={(f1_max - f1_min).round(3)}")
        y1 = f1_pos[1]

        gap2_x = rng.uniform(0.25, 0.30)
        t2 = np.array([f1_max[0] + gap2_x + f2_half_x, y1 + dy2, f2_z])
        gap3_x = rng.uniform(0.25, 0.30)
        t3 = np.array([f1_min[0] - gap3_x - f3_half_x, y1 + dy3, f3_z])

        if floor_bbox_min is None:
            print("[bbox] No floor bbox — accepting placement")
            break

        obj1_ok = within_floor_bbox(f1_pos[:2], floor_bbox_min, floor_bbox_max)
        obj2_ok = within_floor_bbox(t2[:2],     floor_bbox_min, floor_bbox_max)
        obj3_ok = within_floor_bbox(t3[:2],     floor_bbox_min, floor_bbox_max)

        if obj1_ok and obj2_ok and obj3_ok:
            print(f"[bbox] All 3 within floor bbox ✓  (attempt {attempt})")
            break

        outside = ([obj1.name] if not obj1_ok else []) + \
                  ([obj2.name] if not obj2_ok else []) + \
                  ([obj3.name] if not obj3_ok else [])
        print(f"[bbox] Attempt {attempt}: {outside} outside — resampling obj1")
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed. Last offenders: {outside}")

    # ── Teleport obj2 and obj3 ────────────────────────────────────────────────
    obj2.set_position_orientation(position=th.tensor(t2, dtype=th.float32), orientation=f2_ori)
    obj2.keep_still()
    print(f"[{obj2.name}] placed at {t2.round(3)}")
    for _ in range(30): og.sim.step()

    obj3.set_position_orientation(position=th.tensor(t3, dtype=th.float32), orientation=f3_ori)
    obj3.keep_still()
    print(f"[{obj3.name}] placed at {t3.round(3)}")
    for _ in range(30): og.sim.step()

    print("\n=== Final positions ===")
    for obj in objects:
        p = obj.get_position_orientation()[0].cpu().numpy()
        print(f"  {obj.name}: {p.round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main — single simulator call
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",       type=str, required=True,  help="Scene model name, e.g. Merom_1_int")
    parser.add_argument("--room",        type=str, required=True,  help="Room instance name, e.g. living_room_0")
    parser.add_argument("--floor",       type=str, required=True,  help="Floor object name in scene, e.g. floors_xzlkei_0")
    parser.add_argument("--run_idx",     type=int, default=0,      help="Run index (used as seed)")
    parser.add_argument("--keys_json",   type=str, default="keys.json")
    parser.add_argument("--robot",       type=str, default="R1")
    parser.add_argument("--output_root", type=str, default="renders")
    args = parser.parse_args()

    seed = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF)
    run_dir   = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"{'='*70}\n")

    # ── GPT object selection ──────────────────────────────────────────────────
    all_keys = load_keys(args.keys_json)
    sampled  = sample_200(all_keys, seed=seed)
    chosen_categories = gpt_pick_3_small(sampled)
    models = [get_model_for_category(cat, seed) for cat in chosen_categories]

    # ── Build env config ──────────────────────────────────────────────────────
    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"] = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"] = [args.room]
    config["objects"] = [
        build_object_config(f"obj{i+1}", cat, model, i)
        for i, (cat, model) in enumerate(zip(chosen_categories, models))
    ]

    # ── Launch environment (only once per process) ────────────────────────────
    env   = og.Environment(configs=config)
    scene = env.scene

    obj1  = scene.object_registry("name", "obj1")
    obj2  = scene.object_registry("name", "obj2")
    obj3  = scene.object_registry("name", "obj3")
    floor = scene.object_registry("name", args.floor)

    if floor is None:
        print(f"[ERROR] Floor '{args.floor}' not found in scene — exiting with code 2")
        raise SystemExit(2)

    objects = [obj1, obj2, obj3]

    # ── Place objects ─────────────────────────────────────────────────────────
    place_three_objects(env, scene, objects, floor, seed=seed)

    # ── Render + visibility check ─────────────────────────────────────────────
    success, camera_poses = render_and_save(objects, seed=seed, output_dir=run_dir)

    # ── Collect object metadata ───────────────────────────────────────────────
    objects_meta = {}
    for i, (obj, cat) in enumerate(zip(objects, chosen_categories), start=1):
        pos, quat = obj.get_position_orientation()
        objects_meta[f"obj{i}"] = {
            "category":        cat,
            "model":           models[i - 1],
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    # ── Save metadata JSON ────────────────────────────────────────────────────
    metadata = {
        "scene":        args.scene,
        "room":         args.room,
        "run_idx":      args.run_idx,
        "seed":         seed,
        "floor_name":   args.floor,
        "success":      success,
        "objects":      objects_meta,
        "camera_poses": camera_poses,
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # Exit code 0 = success, 1 = objects not visible in front view
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()