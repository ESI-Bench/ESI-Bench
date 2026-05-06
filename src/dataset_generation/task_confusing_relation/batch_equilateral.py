"""
batch_equilateral.py

Single-run script: loads ONE scene restricted to ONE room, places 3 GPT-selected
small objects on the room floor in an equilateral triangle, renders 5 camera
views, and saves a metadata JSON.

Views:
  0.png  — initial front view: camera at 0° orbit from centroid, z=0.025, looking at z=0.1
  1.png  — edge 1-2 observation: camera on outward normal of edge obj1-obj2
  2.png  — edge 2-3 observation: camera on outward normal of edge obj2-obj3
  3.png  — edge 1-3 observation: camera on outward normal of edge obj1-obj3
  4.png  — top-down view: camera 0.4 m directly above centroid, looking straight down

For edge views: camera sits on the outward normal of the edge at the same
distance from the edge midpoint for all 3 views (= side_length * 1.5),
so the two edge objects appear close and the apex appears far.

Called once per (scene, room, run_idx) by batch_equilateral.sh.
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

SIDE_LENGTH_MIN = 0.4
SIDE_LENGTH_MAX = 0.65

# Maximum ratio allowed between the largest and smallest object XY footprint.
# If max_xy_extent / min_xy_extent > this threshold, the object set is resampled.
XY_SIZE_RATIO_MAX = 2.0
XY_SIZE_RESAMPLE_ATTEMPTS = 5


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
        "But it should have some height in Z so we can see it from the floor. (so no mat)"
        "Avoid large furniture, vehicles, toys or anything that wouldn't fit on a table. "
        "Reply with ONLY a JSON array of exactly 3 strings from the input list, e.g.: "
        '[\"candle\", \"apple\", \"mug\"]'
    )
    user_prompt = (
        f"Here are the candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        "Pick 3 that are small in footprint (X/Y)."
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
# XY size check
# ─────────────────────────────────────────────────────────────────────────────

def get_xy_extent(obj) -> float:
    """Return the larger of the object's X and Y AABB extents (its XY footprint)."""
    bbox_min, bbox_max = [x.cpu().numpy() for x in obj.aabb]
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    return max(dx, dy)


def check_xy_size_ratio(objects: list, ratio_max: float = XY_SIZE_RATIO_MAX) -> bool:
    """
    Return True if all objects are within an acceptable XY size ratio.
    Specifically, passes when:
        max(xy_extent) / min(xy_extent) <= ratio_max

    Returns False (and logs the offending sizes) if any object is more than
    ratio_max times larger in XY than the smallest object.
    """
    extents = {obj.name: get_xy_extent(obj) for obj in objects}
    min_ext = min(extents.values())
    max_ext = max(extents.values())
    ratio   = max_ext / max(min_ext, 1e-6)

    print(f"[size_check] XY extents: { {k: round(v, 4) for k, v in extents.items()} }")
    print(f"[size_check] max/min ratio = {ratio:.3f}  (limit = {ratio_max})")

    if ratio > ratio_max:
        worst = max(extents, key=extents.get)
        print(f"[size_check] FAIL — '{worst}' ({extents[worst]:.4f} m) is >{ratio_max}x "
              f"the smallest ({min_ext:.4f} m). Will resample.")
        return False

    print("[size_check] PASS ✓")
    return True


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
    return Rotation.from_matrix(rot_matrix).as_quat()


def freeze_at(obj, x, y, z, ori):
    obj.set_position_orientation(
        position=th.tensor([x, y, z], dtype=th.float32),
        orientation=ori,
    )
    obj.keep_still()


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture(path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] Saved -> {path}")


def _set_camera_and_capture(eye: np.ndarray, look_target: np.ndarray, path: str):
    quat = look_at_quaternion(eye, look_target)
    og.sim._viewer_camera.set_position_orientation(eye, quat)
    _capture(path)
    return {
        "position": eye.tolist(),
        "quaternion_xyzw": quat.tolist(),
    }


def _check_visibility(obj_names: tuple = ("obj1", "obj2", "obj3")) -> bool:
    """Step 100 frames, then check seg_instance for the given object names."""
    for _ in range(100):
        og.sim.step()
    raw_obs = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    visible_names = " ".join(id_to_labels.values())
    missing = [n for n in obj_names if n not in visible_names]
    if missing:
        print(f"[seg] Missing {missing} in current view")
        return False
    print(f"[seg] All of {obj_names} visible ✓")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# render_and_save
# ─────────────────────────────────────────────────────────────────────────────

def render_and_save(objects, seed: int, output_dir: str,
                    p1_xy: np.ndarray, p2_xy: np.ndarray, p3_xy: np.ndarray,
                    centroid_xy: np.ndarray,
                    side_length: float) -> tuple[bool, bool, bool, bool, bool, dict]:
    """
    Camera views:
      0.png        — front view (0° orbit from centroid)
      0_1.png      — same base as 0, shifted ±15–60° randomly (left or right)
      0_2.png      — same base as 0, shifted ±15–60° randomly (opposite side)
      1.png        — edge obj1-obj2 outward normal
      1_1.png      — edge 1 shifted ±15–60°
      1_2.png      — edge 1 shifted ±15–60° (opposite side)
      2.png        — edge obj2-obj3 outward normal
      2_1.png      — edge 2 shifted ±15–60°
      2_2.png      — edge 2 shifted ±15–60° (opposite side)
      3.png        — edge obj1-obj3 outward normal
      3_1.png      — edge 3 shifted ±15–60°
      3_2.png      — edge 3 shifted ±15–60° (opposite side)
      4.png        — top-down view (no shifts)
    """
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    cz         = 0.025
    look_z     = 0.1
    edge_dist  = side_length * 1.5
    orbit_dist = side_length * 2.5

    os.makedirs(output_dir, exist_ok=True)
    camera_poses = {}
    cx, cy = centroid_xy

    rng = random.Random(seed + 77777)  # dedicated RNG for shift angles

    def random_shift_deg() -> float:
        """Return a random angle in [15, 60] degrees, sign chosen randomly."""
        mag = rng.uniform(30.0, 60.0)
        return mag * rng.choice([-1.0, 1.0])

    def orbit_around(pivot_xy: np.ndarray, eye_xy: np.ndarray,
                     delta_deg: float, z: float) -> np.ndarray:
        """
        Rotate eye_xy around pivot_xy by delta_deg degrees in the XY plane,
        keeping the same distance from pivot, returning a 3-D eye position.
        """
        dx, dy   = eye_xy[0] - pivot_xy[0], eye_xy[1] - pivot_xy[1]
        rad      = np.radians(delta_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        nx = cos_a * dx - sin_a * dy + pivot_xy[0]
        ny = sin_a * dx + cos_a * dy + pivot_xy[1]
        return np.array([nx, ny, z])

    def capture_shifted_pair(base_eye: np.ndarray, look_target: np.ndarray,
                             pivot_xy: np.ndarray, base_name: str):
        """
        Take two shifted shots from base_eye orbited around pivot_xy.
        Shift 1: random ±15–60°
        Shift 2: random ±15–60° on the opposite side (negated sign)
        Names: {base_name}_1.png, {base_name}_2.png
        Returns (success_1, success_2).
        """
        delta1 = random_shift_deg()
        delta2 = -delta1 + rng.uniform(-5.0, 5.0)  # roughly opposite with small jitter

        eye1 = orbit_around(pivot_xy, base_eye[:2], delta1, base_eye[2])
        camera_poses[f"{base_name}_1.png"] = _set_camera_and_capture(
            eye1, look_target, os.path.join(output_dir, f"{base_name}_1.png"))
        s1 = _check_visibility(("obj1", "obj2", "obj3"))
        print(f"[seg] success_{base_name}_1 = {s1}  (shift={delta1:+.1f}°)")

        eye2 = orbit_around(pivot_xy, base_eye[:2], delta2, base_eye[2])
        camera_poses[f"{base_name}_2.png"] = _set_camera_and_capture(
            eye2, look_target, os.path.join(output_dir, f"{base_name}_2.png"))
        s2 = _check_visibility(("obj1", "obj2", "obj3"))
        print(f"[seg] success_{base_name}_2 = {s2}  (shift={delta2:+.1f}°)")

        return s1, s2

    # ── Helper: outward normal camera for one edge ────────────────────────────
    def edge_camera(pa: np.ndarray, pb: np.ndarray, opposite: np.ndarray) -> np.ndarray:
        mid    = (pa + pb) / 2.0
        edge_v = pb - pa
        n1 = np.array([-edge_v[1],  edge_v[0]])
        n2 = np.array([ edge_v[1], -edge_v[0]])
        to_opposite = opposite - mid
        normal = n1 if np.dot(n1, to_opposite) < 0 else n2
        normal /= np.linalg.norm(normal)
        eye_xy = mid + normal * edge_dist
        return np.array([eye_xy[0], eye_xy[1], cz])

    # ── View 0: front ─────────────────────────────────────────────────────────
    a         = np.radians(0)
    eye_front = np.array([cx + orbit_dist * np.sin(a),
                          cy - orbit_dist * np.cos(a),
                          cz])
    look_front = np.array([cx, cy, look_z])
    camera_poses["0.png"] = _set_camera_and_capture(
        eye_front, look_front, os.path.join(output_dir, "0.png"))
    success_0 = _check_visibility(("obj1", "obj2", "obj3"))
    print(f"[seg] success_0 = {success_0}")

    s0_1, s0_2 = capture_shifted_pair(
        eye_front, look_front, pivot_xy=centroid_xy, base_name="0")

    # ── View 1: edge obj1-obj2, opposite = obj3 ───────────────────────────────
    eye1      = edge_camera(p1_xy, p2_xy, p3_xy)
    mid_12    = np.array([(p1_xy[0]+p2_xy[0])/2, (p1_xy[1]+p2_xy[1])/2, look_z])
    pivot_12  = (p1_xy + p2_xy) / 2.0
    camera_poses["1.png"] = _set_camera_and_capture(
        eye1, mid_12, os.path.join(output_dir, "1.png"))
    success_1 = _check_visibility(("obj1", "obj2", "obj3"))
    print(f"[seg] success_1 = {success_1}")

    s1_1, s1_2 = capture_shifted_pair(
        eye1, mid_12, pivot_xy=pivot_12, base_name="1")

    # ── View 2: edge obj2-obj3, opposite = obj1 ───────────────────────────────
    eye2      = edge_camera(p2_xy, p3_xy, p1_xy)
    mid_23    = np.array([(p2_xy[0]+p3_xy[0])/2, (p2_xy[1]+p3_xy[1])/2, look_z])
    pivot_23  = (p2_xy + p3_xy) / 2.0
    camera_poses["2.png"] = _set_camera_and_capture(
        eye2, mid_23, os.path.join(output_dir, "2.png"))
    success_2 = _check_visibility(("obj1", "obj2", "obj3"))
    print(f"[seg] success_2 = {success_2}")

    s2_1, s2_2 = capture_shifted_pair(
        eye2, mid_23, pivot_xy=pivot_23, base_name="2")

    # ── View 3: edge obj1-obj3, opposite = obj2 ───────────────────────────────
    eye3      = edge_camera(p1_xy, p3_xy, p2_xy)
    mid_13    = np.array([(p1_xy[0]+p3_xy[0])/2, (p1_xy[1]+p3_xy[1])/2, look_z])
    pivot_13  = (p1_xy + p3_xy) / 2.0
    camera_poses["3.png"] = _set_camera_and_capture(
        eye3, mid_13, os.path.join(output_dir, "3.png"))
    success_3 = _check_visibility(("obj1", "obj2", "obj3"))
    print(f"[seg] success_3 = {success_3}")

    s3_1, s3_2 = capture_shifted_pair(
        eye3, mid_13, pivot_xy=pivot_13, base_name="3")

    # ── View 4: top-down ──────────────────────────────────────────────────────
    eye_top  = np.array([cx, (p1_xy[1]+p2_xy[1]+p3_xy[1])/3, 1.5])
    look_top = np.array([cx, (p1_xy[1]+p2_xy[1]+p3_xy[1])/3, 0.0])
    quat_top = look_at_quaternion(eye_top, look_top, up=np.array([0.0, 1.0, 0.0]))
    og.sim._viewer_camera.set_position_orientation(eye_top, quat_top)
    _capture(os.path.join(output_dir, "4.png"))
    camera_poses["4.png"] = {
        "position": eye_top.tolist(),
        "quaternion_xyzw": quat_top.tolist(),
    }
    success_4 = _check_visibility(("obj1", "obj2", "obj3"))
    print(f"[seg] success_4 = {success_4}")

    all_success = all([
        success_0, s0_1, s0_2,
        success_1, s1_1, s1_2,
        success_2, s2_1, s2_2,
        success_3, s3_1, s3_2,
        success_4,
    ])

    return success_0, success_1, success_2, success_3, success_4, camera_poses

# ─────────────────────────────────────────────────────────────────────────────
# Triangle placement
# ─────────────────────────────────────────────────────────────────────────────

def place_equilateral_triangle(env, scene, objects, floor,
                               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Place three objects in an equilateral triangle on the floor.
    Side length drawn from [SIDE_LENGTH_MIN, SIDE_LENGTH_MAX] using seed.

    All three vertex positions (p1, p2, p3) are checked against the floor
    bbox. If any fall outside, obj1 is resampled via OnTop (up to MAX_ATTEMPTS).
    Raises RuntimeError if all attempts fail.

    Returns (p1_xy, p2_xy, p3_xy, centroid_xy, side_length).
    """
    MAX_ATTEMPTS = 3
    obj1, obj2, obj3 = objects

    rng = random.Random(seed)
    side_length = rng.uniform(SIDE_LENGTH_MIN, SIDE_LENGTH_MAX)

    h = max(side_length * np.sqrt(3) / 2.0, 0.2)

    print(f"[triangle] side_length={side_length:.4f} m  (seed={seed})")

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

    # ── Get obj2/obj3 Z by dropping them once, then park far away ─────────────
    obj2.states[object_states.OnTop].set_value(floor, True)
    obj3.states[object_states.OnTop].set_value(floor, True)
    for _ in range(20):
        og.sim.step()

    p2_raw, ori2 = obj2.get_position_orientation()
    z2 = float(p2_raw.cpu()[2])
    p3_raw, ori3 = obj3.get_position_orientation()
    z3 = float(p3_raw.cpu()[2])

    for obj, ori in [(obj2, ori2), (obj3, ori3)]:
        obj.set_position_orientation(
            position=th.tensor([500.0, 500.0, 100.0], dtype=th.float32),
            orientation=ori,
        )
        obj.keep_still()
    for _ in range(20):
        og.sim.step()

    # ── Resample loop: drop obj1, compute p2/p3, check all three in bbox ──────
    p1_xy = p2_xy = p3_xy = None
    ori1 = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        obj1.states[object_states.OnTop].set_value(floor, True)
        for _ in range(20):
            og.sim.step()

        p1_raw, ori1 = obj1.get_position_orientation()
        p1 = p1_raw.cpu().numpy()
        p1_xy = p1[:2]

        p2_xy = np.array([p1_xy[0] + side_length,       p1_xy[1]    ])
        p3_xy = np.array([p1_xy[0] + side_length / 2.0, p1_xy[1] + h])

        ok1 = within(p1_xy)
        ok2 = within(p2_xy)
        ok3 = within(p3_xy)

        if ok1 and ok2 and ok3:
            print(f"[bbox] All three vertices within floor bbox ✓  (attempt {attempt})")
            break

        outside = ([" obj1"] if not ok1 else []) + \
                  (["obj2"] if not ok2 else []) + \
                  (["obj3"] if not ok3 else [])
        print(f"[bbox] Attempt {attempt}: {outside} outside floor — resampling obj1")
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed. Outside: {outside}")

    print("=" * 60)
    print(f"[triangle] obj1 anchor : {p1_xy.round(3)}")
    print(f"[triangle] obj2 target : {p2_xy.round(3)}")
    print(f"[triangle] obj3 target : {p3_xy.round(3)}")
    print(f"[triangle] side length : {side_length:.4f} m")
    d12 = np.linalg.norm(p2_xy - p1_xy)
    d23 = np.linalg.norm(p3_xy - p2_xy)
    d31 = np.linalg.norm(p1_xy - p3_xy)
    print(f"[triangle] verify: |1-2|={d12:.4f}  |2-3|={d23:.4f}  |3-1|={d31:.4f}")
    print("=" * 60)

    # ── Freeze obj2 and obj3 in place ─────────────────────────────────────────
    freeze_at(obj2, p2_xy[0], p2_xy[1], z2, ori2)
    freeze_at(obj3, p3_xy[0], p3_xy[1], z3, ori3)
    for _ in range(30):
        og.sim.step()
        freeze_at(obj2, p2_xy[0], p2_xy[1], z2, ori2)
        freeze_at(obj3, p3_xy[0], p3_xy[1], z3, ori3)

    print("\n=== Final positions ===")
    for obj in objects:
        pos = obj.get_position_orientation()[0].cpu().numpy()
        print(f"  {obj.name}: {pos.round(3)}")

    centroid = np.array([
        (p1_xy[0] + p2_xy[0] + p3_xy[0]) / 3.0,
        (p1_xy[1] + p2_xy[1] + side_length * np.sqrt(3) / 2.0) / 3.0,
    ])
    return p1_xy, p2_xy, p3_xy, centroid, side_length


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
    parser.add_argument("--output_root", type=str, default="renders_equilateral")
    args = parser.parse_args()

    seed = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF)
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"{'='*70}\n")

    all_keys = load_keys(args.keys_json)

    # ── GPT selection + XY size check loop ───────────────────────────────────
    # Because OmniGibson cannot clear objects between runs in the same process,
    # we must do the size check *after* loading. If the ratio fails we rebuild
    # the env with a fresh GPT selection (different subseed per attempt).
    chosen_categories = None
    models            = None
    env               = None
    scene             = None
    objects           = None

    for size_attempt in range(1, XY_SIZE_RESAMPLE_ATTEMPTS + 1):
        # Vary the seed slightly each attempt so GPT sees a different candidate list
        attempt_seed = seed + size_attempt * 9973

        sampled           = sample_200(all_keys, seed=attempt_seed)
        chosen_categories = gpt_pick_3_small(sampled)
        models            = [get_model_for_category(cat, attempt_seed)
                             for cat in chosen_categories]

        config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
        config = yaml.safe_load(open(config_filename))
        config["scene"]["scene_model"]               = args.scene
        config["scene"]["not_load_object_categories"] = ["ceilings"]
        config["scene"]["load_room_instances"]        = [args.room]
        config["objects"] = [
            build_object_config(f"obj{i+1}", cat, model, i)
            for i, (cat, model) in enumerate(zip(chosen_categories, models))
        ]

        if env is not None:
            # OmniGibson does not support clearing objects; shut down and restart.
            print(f"[size_check] Tearing down environment for resample attempt {size_attempt}…")
            og.shutdown()

        print(f"\n[size_check] Loading environment (attempt {size_attempt}/{XY_SIZE_RESAMPLE_ATTEMPTS})…")
        env   = og.Environment(configs=config)
        scene = env.scene

        obj1  = scene.object_registry("name", "obj1")
        obj2  = scene.object_registry("name", "obj2")
        obj3  = scene.object_registry("name", "obj3")
        objects = [obj1, obj2, obj3]

        # Settle objects so AABB is stable before measuring
        for _ in range(30):
            og.sim.step()

        if check_xy_size_ratio(objects, ratio_max=XY_SIZE_RATIO_MAX):
            print(f"[size_check] Accepted object set on attempt {size_attempt}: {chosen_categories}")
            break

        if size_attempt == XY_SIZE_RESAMPLE_ATTEMPTS:
            print(f"[size_check] WARNING: Could not find a balanced set in "
                  f"{XY_SIZE_RESAMPLE_ATTEMPTS} attempts — proceeding with last selection.")
    # ── End size-check loop ───────────────────────────────────────────────────

    floor = scene.object_registry("name", args.floor)
    if floor is None:
        print(f"[ERROR] Floor '{args.floor}' not found — exiting with code 2")
        raise SystemExit(2)

    p1_xy, p2_xy, p3_xy, centroid_xy, side_length = \
        place_equilateral_triangle(env, scene, objects, floor, seed=seed)

    success_0, success_1, success_2, success_3, success_4, camera_poses = render_and_save(
        objects, seed=seed, output_dir=run_dir,
        p1_xy=p1_xy, p2_xy=p2_xy, p3_xy=p3_xy,
        centroid_xy=centroid_xy, side_length=side_length,
    )

    objects_meta = {}
    for i, (obj, cat) in enumerate(zip(objects, chosen_categories), start=1):
        pos, quat = obj.get_position_orientation()
        objects_meta[f"obj{i}"] = {
            "category":        cat,
            "model":           models[i - 1],
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    metadata = {
        "scene":         args.scene,
        "room":          args.room,
        "run_idx":       args.run_idx,
        "seed":          seed,
        "floor_name":    args.floor,
        "layout":        "equilateral_triangle",
        "side_length_m": side_length,
        "vertices_xy":   {"obj1": p1_xy.tolist(), "obj2": p2_xy.tolist(), "obj3": p3_xy.tolist()},
        "centroid_xy":   centroid_xy.tolist(),
        "success":       success_0,
        "success_1":     success_1,
        "success_2":     success_2,
        "success_3":     success_3,
        "success_4":     success_4,
        "objects":       objects_meta,
        "camera_poses":  camera_poses,
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # Exit 0 only if all 5 views succeeded
    raise SystemExit(0 if (success_0 and success_1 and success_2 and success_3 and success_4) else 1)


if __name__ == "__main__":
    main()