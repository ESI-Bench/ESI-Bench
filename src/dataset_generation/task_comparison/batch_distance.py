"""
batch_distance.py

Single-run script: loads ONE scene restricted to ONE room, picks a reference
object from the scene graph (something already in the room with spacious space
around it), selects 2 similar objects from GPT-sampled candidates, places them
at two distances from the reference object's bounding-box corners, renders
24 camera views (12 azimuths × 2 heights) + 1 top-down view, and saves a
metadata JSON.

Camera layout:
  - 12 azimuths: starting from the direction of obj_near, then every 30°
  - 2 heights per azimuth: z=0.05 (low, ground-level) and z=0.40 (elevated)
  - All cameras look at the reference object centre
  → 24 side views + 1 top-down view (view 24) = 25 renders total

Placement:
  - Reference object: already in the scene (its AABB is read directly)
  - All 4 corners of the reference AABB are evaluated for clearance (distance
    to the nearest other object's bbox centre in the room).
  - 2 corners are RANDOMLY SAMPLED from those with clearance > 0.5 m.
    If fewer than 2 qualify, fall back to the top-2 by clearance.
  - obj_near is placed at the first sampled corner at distance D_NEAR outward.
  - obj_far  is placed at the second sampled corner at distance D_NEAR+0.2 m.
  - Outward direction for each corner is the diagonal unit vector away from
    the bbox centre (e.g. top-right corner → +X+Y / √2).

JSON flags per view:
  exist_ref_k    — reference object visible
  exist_near_k   — obj_near visible
  exist_far_k    — obj_far  visible

Called once per (scene, room, run_idx) by a companion batch shell script.
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

# ── Placement parameters ──────────────────────────────────────────────────────
D_NEAR        = 0.20   # metres from ref corner to obj_near centre
D_FAR_EXTRA   = 0.25   # obj_far is this much further than obj_near
SQUARE_ORI    = [0.0, 0.0, 0.0, 1.0]   # identity quaternion — axis-aligned

# ── Camera ring parameters ────────────────────────────────────────────────────
NUM_AZIMUTHS    = 12
AZIMUTH_STEP    = 30.0     # degrees
CAM_HEIGHTS     = [0.05, 0.40]   # two Z heights per azimuth
CAMERA_OFFSET   = 0.45     # metres beyond the ref bbox edge (clearance from ref)
CAMERA_OFFSET2  = 0.20     # metres beyond obj_far's outer edge
CAM_TOPDOWN_Z   = 2.4     # fallback height of the top-down overview camera (view 24)
                           # (overridden at runtime to ceiling_z - 0.02 if ceiling bbox found)
# actual radius = max(ref_xy_half_diag + CAMERA_OFFSET,
#                     corner_dist + d_far + half_far + CAMERA_OFFSET2)
# → 12 × 2 = 24 side renders + 1 top-down = 25 renders total

# ── Scenes folder ─────────────────────────────────────────────────────────────
SCENES_DIR = "scenes5"


# ─────────────────────────────────────────────────────────────────────────────
# Scene-graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scene_dict(scene_name: str) -> dict:
    """Return the room→category→bbox-list dict for the given scene."""
    path = os.path.join(SCENES_DIR, f"{scene_name}_scene_dict.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene dict not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_room_objects(scene_dict: dict, room: str) -> dict:
    """Return category→bbox-list for the given room."""
    if room not in scene_dict:
        raise KeyError(f"Room '{room}' not in scene dict. Available: {list(scene_dict.keys())}")
    return scene_dict[room]


def bbox_footprint(bboxes: list) -> float:
    """Return the 2-D footprint area of the union of given bboxes."""
    total = 0.0
    for (bmin, bmax) in bboxes:
        dx = abs(bmax[0] - bmin[0])
        dy = abs(bmax[1] - bmin[1])
        total += dx * dy
    return total


def bbox_clearance(bboxes: list, all_bboxes_in_room: list) -> float:
    """
    Overall clearance: minimum over all 4 corners of the per-corner clearance.
    Used only for the GPT summary (higher = more spacious surroundings).
    """
    corner_clears = corner_clearances(bboxes, all_bboxes_in_room)
    return min(corner_clears.values()) if corner_clears else 999.0


# Named corners: tr=top-right, tl=top-left, br=bottom-right, bl=bottom-left
# in XY plane where X→right, Y→up (i.e. max-X/max-Y = top-right, etc.)
CORNER_NAMES = ["tr", "tl", "br", "bl"]

def corner_xy(bmin, bmax, corner: str) -> np.ndarray:
    """Return the 2-D XY position of the named corner of a bbox."""
    xmin, ymin = float(bmin[0]), float(bmin[1])
    xmax, ymax = float(bmax[0]), float(bmax[1])
    return {
        "tr": np.array([xmax, ymax]),
        "tl": np.array([xmin, ymax]),
        "br": np.array([xmax, ymin]),
        "bl": np.array([xmin, ymin]),
    }[corner]

def outward_dir(corner: str) -> np.ndarray:
    """Unit vector pointing diagonally outward from the named corner."""
    return {
        "tr": np.array([ 1,  1], dtype=float) / np.sqrt(2),
        "tl": np.array([-1,  1], dtype=float) / np.sqrt(2),
        "br": np.array([ 1, -1], dtype=float) / np.sqrt(2),
        "bl": np.array([-1, -1], dtype=float) / np.sqrt(2),
    }[corner]

def corner_clearances(bboxes: list, all_bboxes_in_room: list) -> dict:
    """
    For each of the 4 corners of the (first) bbox, compute the minimum
    distance from that corner to the nearest other object's bbox centre.

    Returns {"tr": float, "tl": float, "br": float, "bl": float}.
    Uses the first bbox in `bboxes` as the reference shape.
    """
    if not bboxes:
        return {c: 999.0 for c in CORNER_NAMES}

    bmin, bmax = bboxes[0]
    result = {}
    for cname in CORNER_NAMES:
        cxy = corner_xy(bmin, bmax, cname)
        min_dist = 999.0
        for (omin, omax) in all_bboxes_in_room:
            ocx = (float(omin[0]) + float(omax[0])) / 2
            ocy = (float(omin[1]) + float(omax[1])) / 2
            dist = float(np.linalg.norm(cxy - np.array([ocx, ocy])))
            if dist > 0.01:   # skip self
                min_dist = min(min_dist, dist)
        result[cname] = min_dist
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GPT helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_keys(keys_json_path: str) -> list:
    with open(keys_json_path) as f:
        return json.load(f)


def sample_200(all_keys: list, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(200, len(all_keys)))


def gpt_pick_reference_and_2_similar(
    room_categories: list,
    candidate_categories: list,
    room_object_summaries: dict,
) -> dict:
    """
    Ask GPT to:
      1. Pick ONE reference object category from room_categories that is
         close to the centre of the room and has spacious surroundings.
      2. Pick 2 similar SMALL object categories from candidate_categories.

    Returns {"reference": str, "obj_near_cat": str, "obj_far_cat": str}
    """
    api_key = OPENAI_API_KEY
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a spatial-reasoning assistant for a robotics simulation. "
        "You will be given:\n"
        "  (A) a list of object categories already present in a room, each with "
        "      a rough 2-D footprint and distance-to-room-centre summary,\n"
        "  (B) a list of candidate object categories that can be loaded.\n\n"
        "Your task:\n"
        "  1. From list (A), choose ONE 'reference' object that:\n"
        "       - is as CLOSE TO THE ROOM CENTRE as possible (small dist_to_centre)\n"
        "       - has relatively OPEN space around it (not tightly packed by others)\n"
        "       - is NOT a floor, ceiling, wall, carpet, door, or switch\n"
        "  2. From list (B), choose exactly 2 categories for comparison objects "
        "     that are similar in size. You should avoid large objects like furnitures.Avoid too small objects like dice or too thin objects like mat\n\n"
        "Reply with ONLY a JSON object with keys:\n"
        '  "reference"   : string (category from list A)\n'
        '  "obj_near_cat": string (category from list B)\n'
        '  "obj_far_cat" : string (category from list B)\n'
        "No extra text, no markdown fences."
    )

    summaries_str = "\n".join(
        f"  {cat}: footprint≈{info['footprint']:.2f} m², "
        f"clearance≈{info['clearance']:.2f} m, "
        f"dist_to_centre≈{info['dist_to_centre']:.2f} m"
        for cat, info in room_object_summaries.items()
    )

    user_prompt = (
        f"List (A) — room objects with spatial info:\n{summaries_str}\n\n"
        f"List (B) — candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        "Choose the reference object (closest to room centre, open space) "
        "and 2 similar SMALL comparison objects as instructed."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0.2,
        max_tokens=128,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    result = json.loads(raw)

    print(f"[GPT] reference={result['reference']}  "
          f"obj_near_cat={result['obj_near_cat']}  "
          f"obj_far_cat={result['obj_far_cat']}")
    return result


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
        print(f"  [{category}] {len(matches)} model(s), picked model_id={model_id}")
        return model_id
    raise RuntimeError("No object inventory file found.")


def build_object_config(name: str, category: str, model: str, idx: int) -> dict:
    return {
        "type": "DatasetObject",
        "name": name,
        "category": category,
        "model": model,
        "position": [150.0 + idx * 10, 100.0, 100.0],
        "orientation": SQUARE_ORI,
        "scale": [1, 1, 1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
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
    return Rotation.from_matrix(rot_matrix).as_quat()   # [qx, qy, qz, qw]


def get_ceiling_z(room_objs: dict) -> float:
    """
    Return the Z coordinate to use for the top-down camera: the minimum Z of
    the ceiling bbox minus 0.02 m (i.e. 0.02 m below the ceiling surface).

    The scene dict stores the ceiling under the key "ceilings".  If no ceiling
    bbox is found, falls back to CAM_TOPDOWN_Z.
    """
    ceiling_bboxes = room_objs.get("ceilings", [])
    if not ceiling_bboxes:
        print(f"[ceiling] No 'ceilings' entry found in room — "
              f"using fallback CAM_TOPDOWN_Z={CAM_TOPDOWN_Z:.2f}")
        return CAM_TOPDOWN_Z

    # The ceiling's min Z is its lower face (the surface we want to stay below).
    ceiling_min_z = min(float(bbox[0][2]) for bbox in ceiling_bboxes)
    cam_z = ceiling_min_z - 0.02
    print(f"[ceiling] ceiling bbox min Z={ceiling_min_z:.3f}  →  top-down camera Z={cam_z:.3f}")
    return cam_z


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


def _visibility_check(obj_names: tuple = ("obj_ref", "obj_near", "obj_far")) -> dict:
    """
    Step 100 frames, then check seg_instance for each object name.
    Returns dict {name: bool} indicating visibility.
    """
    for _ in range(100):
        og.sim.step()

    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    visible_str  = " ".join(id_to_labels.values())

    result = {name: (name in visible_str) for name in obj_names}
    print(f"[seg] {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Object placement
# ─────────────────────────────────────────────────────────────────────────────

def pick_two_best_corners(ref_bboxes: list, other_bboxes: list,
                          seed: int = 0, min_clearance: float = 0.5) -> tuple:
    """
    Compute per-corner clearance for the reference object, then randomly
    sample 2 corners from those whose clearance exceeds `min_clearance`.
    If fewer than 2 corners qualify, fall back to the top-2 by clearance.

    Returns (corner_name_1, xy_1, corner_name_2, xy_2, clearances_dict).
    """
    clears = corner_clearances(ref_bboxes, other_bboxes)
    print(f"[corners] clearances: { {k: round(v,3) for k,v in clears.items()} }")

    qualified = [c for c, v in clears.items() if v > min_clearance]

    rng = random.Random(seed)
    if len(qualified) >= 2:
        chosen = rng.sample(qualified, 2)
        print(f"[corners] {len(qualified)} corners with clearance>{min_clearance}m — "
              f"randomly sampled: {chosen[0]}({clears[chosen[0]]:.3f})  {chosen[1]}({clears[chosen[1]]:.3f})")
    else:
        # Fallback: top-2 by clearance
        sorted_corners = sorted(clears.items(), key=lambda kv: kv[1], reverse=True)
        chosen = [sorted_corners[0][0], sorted_corners[1][0]]
        print(f"[corners] fewer than 2 corners qualify (min_clearance={min_clearance}m) — "
              f"falling back to top-2: "
              f"{chosen[0]}({clears[chosen[0]]:.3f})  {chosen[1]}({clears[chosen[1]]:.3f})")

    bmin, bmax = ref_bboxes[0]
    c1_name, c2_name = chosen[0], chosen[1]
    c1_xy = corner_xy(bmin, bmax, c1_name)
    c2_xy = corner_xy(bmin, bmax, c2_name)

    return c1_name, c1_xy, c2_name, c2_xy, clears



def place_comparison_objects(scene, obj_near, obj_far, floor,
                             ref_aabb_min, ref_aabb_max,
                             other_bboxes: list, seed: int) -> dict:
    """
    Randomly sample 2 corners from those with clearance > 0.5 m (falling back
    to top-2 if fewer than 2 qualify).  Place obj_near at the first sampled
    corner (distance D_NEAR) and obj_far at the second (distance D_NEAR +
    D_FAR_EXTRA).

    Collision avoidance: after computing the initial placements, check whether
    either proposed XY position is closer to any other scene object than the
    corner's clearance value (which would indicate a collision).  If so,
    subtract 0.05 m from BOTH distances (preserving the D_FAR_EXTRA gap)
    and recheck — repeat until both positions are safely within their
    respective corner clearances, or a maximum of 40 iterations is reached.

    Both objects are first placed via OnTop (for correct floor Z), then
    teleported to the computed XY positions with identity orientation.

    Returns placement metadata dict.
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    rng = random.Random(seed + 42)

    ref_bboxes = [[ref_aabb_min.tolist(), ref_aabb_max.tolist()]]

    # ── Pick two corners (random sample from those with clearance > 0.5 m) ───
    c1_name, c1_xy, c2_name, c2_xy, corner_clears = pick_two_best_corners(
        ref_bboxes, other_bboxes, seed=seed)

    # Clearance at each chosen corner = the safe radius we must stay within
    clear_near = corner_clears[c1_name]
    clear_far  = corner_clears[c2_name]

    # Outward directions from each chosen corner
    dir_near = outward_dir(c1_name)
    dir_far  = outward_dir(c2_name)

    # Small random jitter so runs vary
    jitter = rng.uniform(-0.05, 0.05)

    d_near = D_NEAR + jitter
    d_far  = d_near + D_FAR_EXTRA

    # ── Place both objects on floor first (for correct Z + live AABB) ──────────
    for obj in (obj_near, obj_far):
        obj.states[object_states.OnTop].set_value(floor, True)
        for _ in range(10):
            og.sim.step()
        pos, _ = obj.get_position_orientation()
        obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
        obj.keep_still()
        for _ in range(5):
            og.sim.step()

    near_z = obj_near.get_position_orientation()[0].cpu().numpy()[2]
    far_z  = obj_far.get_position_orientation()[0].cpu().numpy()[2]

    # Read each object's XY half-extent from its live AABB (axis-aligned).
    # We use the diagonal half-extent (half of bbox diagonal in XY) as a
    # conservative radius: d + half_extent must not exceed corner clearance.
    def _xy_half_extent(obj) -> float:
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        dx = abs(bmax[0] - bmin[0]) / 2
        dy = abs(bmax[1] - bmin[1]) / 2
        return float(np.sqrt(dx**2 + dy**2))   # diagonal half-extent

    half_near = _xy_half_extent(obj_near)
    half_far  = _xy_half_extent(obj_far)
    print(f"[collision] half_extent  near={half_near:.3f}  far={half_far:.3f}")
    print(f"[collision] corner clear near={clear_near:.3f}  far={clear_far:.3f}")

    # ── Collision-avoidance: shrink both distances until safe ─────────────────
    # Condition: d_near + half_near < clear_near  AND  d_far + half_far < clear_far
    MAX_SHRINK_ITERS = 40
    SHRINK_STEP      = 0.05   # metres to subtract per iteration

    for shrink_iter in range(MAX_SHRINK_ITERS + 1):
        near_ok = (d_near + half_near < clear_near) or (d_near <= SHRINK_STEP)
        far_ok  = (d_far  + half_far  < clear_far)  or (d_far  <= SHRINK_STEP)

        print(f"[collision] iter={shrink_iter}  "
              f"d_near={d_near:.3f}  d_near+half={d_near+half_near:.3f}  clear_near={clear_near:.3f}  ok={near_ok} | "
              f"d_far={d_far:.3f}  d_far+half={d_far+half_far:.3f}  clear_far={clear_far:.3f}  ok={far_ok}")

        if near_ok and far_ok:
            break

        # Shrink both distances together to preserve the D_FAR_EXTRA gap
        d_near = max(d_near - SHRINK_STEP, SHRINK_STEP)
        d_far  = d_near + D_FAR_EXTRA

    else:
        print(f"[collision] WARNING: could not fully resolve collision after "
              f"{MAX_SHRINK_ITERS} iterations — using best available positions")

    near_xy = c1_xy + dir_near * d_near
    far_xy  = c2_xy + dir_far  * d_far
    print(f"[place] final d_near={d_near:.3f}  d_far={d_far:.3f}"
          f"  near_reach={d_near+half_near:.3f}  far_reach={d_far+half_far:.3f}")

    # ── Teleport to computed XY ───────────────────────────────────────────────
    obj_near.set_position_orientation(
        position=th.tensor([near_xy[0], near_xy[1], near_z], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    obj_near.keep_still()

    obj_far.set_position_orientation(
        position=th.tensor([far_xy[0], far_xy[1], far_z], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    obj_far.keep_still()

    for _ in range(30):
        og.sim.step()

    # ── Verify final positions ────────────────────────────────────────────────
    near_pos_final = obj_near.get_position_orientation()[0].cpu().numpy()
    far_pos_final  = obj_far.get_position_orientation()[0].cpu().numpy()

    ref_centre_xy = np.array([(ref_aabb_min[0] + ref_aabb_max[0]) / 2,
                               (ref_aabb_min[1] + ref_aabb_max[1]) / 2])

    corner_dist_near = float(np.linalg.norm(c1_xy - ref_centre_xy))
    corner_dist_far  = float(np.linalg.norm(c2_xy - ref_centre_xy))

    dist_near = float(np.linalg.norm(near_pos_final[:2] - ref_centre_xy))
    dist_far  = float(np.linalg.norm(far_pos_final[:2]  - ref_centre_xy))

    print(f"[place] obj_near pos={near_pos_final.round(3)}  dist_from_ref_centre={dist_near:.3f}")
    print(f"[place] obj_far  pos={far_pos_final.round(3)}   dist_from_ref_centre={dist_far:.3f}")
    print(f"[place] gap={dist_far - dist_near:.3f} m  (target ≈ {D_FAR_EXTRA:.2f} m)")

    return {
        "d_near":              float(d_near),
        "d_far":               float(d_far),
        "half_near":           half_near,
        "half_far":            half_far,
        "corner_dist_near":    corner_dist_near,
        "corner_dist_far":     corner_dist_far,
        "near_pos":            near_pos_final.tolist(),
        "far_pos":             far_pos_final.tolist(),
        "dist_near_centre":    dist_near,
        "dist_far_centre":     dist_far,
        "near_corner":         c1_name,
        "far_corner":          c2_name,
        "near_corner_xy":      c1_xy.tolist(),
        "far_corner_xy":       c2_xy.tolist(),
        "corner_clearances":   {k: round(v, 4) for k, v in corner_clears.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render 24 side views (12 azimuths × 2 heights) + 1 top-down view
# ─────────────────────────────────────────────────────────────────────────────

def render_and_save(output_dir: str, ref_centre: np.ndarray,
                    ref_aabb_min: np.ndarray, ref_aabb_max: np.ndarray,
                    near_pos: np.ndarray, placement: dict,
                    ceiling_z: float = CAM_TOPDOWN_Z) -> tuple:
    """
    12 azimuths starting from the direction of obj_near relative to ref centre,
    stepping +30° each time.  At each azimuth, render at z=0.05 and z=0.40.
    After the 24 side views, render view 24 as a top-down shot placed 0.02 m
    below the ceiling (ceiling_z = ceiling_bbox_min_z - 0.02), directly above
    the reference object centre, looking straight down.

    Camera radius (side views) is computed as:
        max(ref_xy_half_diag   + CAMERA_OFFSET,
            corner_dist_far + d_far + half_far + CAMERA_OFFSET2)
    so cameras are (a) outside the reference object's footprint and
    (b) beyond obj_far's outer edge, with two independent offsets.
    All side cameras look at the reference object's 3-D centre (mid-height).

    View numbering:
      view  0 = azimuth 0  z=0.05
      view  1 = azimuth 0  z=0.40
      ...
      view 22 = azimuth 330 z=0.05
      view 23 = azimuth 330 z=0.40
      view 24 = top-down   z=ceiling_z  (0.02 m below ceiling, looking down)

    Returns (exist_ref_flags, exist_near_flags, exist_far_flags, camera_poses).
    """
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # ── Camera radius: max of two independent lower bounds ───────────────────
    dx = abs(float(ref_aabb_max[0]) - float(ref_aabb_min[0])) / 2
    dy = abs(float(ref_aabb_max[1]) - float(ref_aabb_min[1])) / 2
    ref_xy_half_diag = float(np.sqrt(dx**2 + dy**2))

    # d_far is measured from the ref bbox CORNER, not the centre.
    # Full reach from ref centre = corner_dist_far + d_far + half_far.
    corner_dist_far = placement["corner_dist_far"]
    d_far           = placement["d_far"]
    half_far        = placement["half_far"]
    far_reach       = corner_dist_far + d_far + half_far

    r_from_ref  = ref_xy_half_diag + CAMERA_OFFSET    # clear of ref bbox
    r_from_far  = far_reach        + CAMERA_OFFSET2   # clear of obj_far
    camera_radius = max(r_from_ref, r_from_far)
    print(f"[camera] ref_half_diag={ref_xy_half_diag:.3f}+{CAMERA_OFFSET:.2f}={r_from_ref:.3f}  "
          f"far_reach={far_reach:.3f}+{CAMERA_OFFSET2:.2f}={r_from_far:.3f}  "
          f"final_radius={camera_radius:.3f}")

    # ── Look-at target: 3-D centre of reference bbox ─────────────────────────
    cx  = float(ref_centre[0])
    cy  = float(ref_centre[1])
    # vertical centre of the reference object (not floor level)
    look_z = (float(ref_aabb_min[2]) + float(ref_aabb_max[2])) / 2.0
    look_target = np.array([cx, cy, look_z])
    print(f"[camera] look_target={look_target.round(3)}")

    # Starting azimuth: direction from ref_centre to near_pos (projected to XY)
    delta = near_pos[:2] - ref_centre[:2]
    start_az_rad = np.arctan2(delta[1], delta[0])

    os.makedirs(output_dir, exist_ok=True)

    camera_poses     = {}
    exist_ref_flags  = {}
    exist_near_flags = {}
    exist_far_flags  = {}

    # ── Views 0–23: 12 azimuths × 2 heights ──────────────────────────────────
    view_idx = 0
    for az_step in range(NUM_AZIMUTHS):
        azimuth_rad = start_az_rad + np.deg2rad(az_step * AZIMUTH_STEP)
        azimuth_deg = np.rad2deg(azimuth_rad) % 360

        for z in CAM_HEIGHTS:
            eye = np.array([
                cx + camera_radius * np.cos(azimuth_rad),
                cy + camera_radius * np.sin(azimuth_rad),
                z,
            ])

            fname = f"{view_idx}.png"
            fpath = os.path.join(output_dir, fname)

            print(f"\n[camera {view_idx}] azimuth={azimuth_deg:.1f}°  z={z:.2f}  eye={eye.round(3)}")
            pose = _set_camera_and_capture(eye, look_target, fpath)
            camera_poses[fname] = {**pose, "azimuth_deg": azimuth_deg, "height": z}

            vis = _visibility_check(("obj_ref", "obj_near", "obj_far"))
            exist_ref_flags [f"exist_ref_{view_idx}"]  = vis["obj_ref"]
            exist_near_flags[f"exist_near_{view_idx}"] = vis["obj_near"]
            exist_far_flags [f"exist_far_{view_idx}"]  = vis["obj_far"]

            view_idx += 1

    # ── View 24: top-down overview ────────────────────────────────────────────
    # Camera is placed directly above the ref centre at ceiling_z (= ceiling
    # bbox min Z - 0.02 m) and looks straight down.  We tilt the up-vector to
    # +X so the image is consistently oriented (avoids gimbal-lock when
    # forward == -Z).
    topdown_eye = np.array([cx, cy, ceiling_z])
    topdown_target = np.array([cx, cy, 0.0])   # look at floor level below ref

    topdown_up  = np.array([1.0, 0.0, 0.0])    # +X as image "up"
    forward_td  = topdown_target - topdown_eye
    forward_td /= np.linalg.norm(forward_td)    # (0, 0, -1)
    right_td    = np.cross(forward_td, topdown_up)
    right_td   /= np.linalg.norm(right_td)
    true_up_td  = np.cross(right_td, forward_td)
    true_up_td /= np.linalg.norm(true_up_td)
    rot_td      = np.column_stack([right_td, true_up_td, -forward_td])
    topdown_quat = Rotation.from_matrix(rot_td).as_quat()

    fname_td = "24.png"
    fpath_td = os.path.join(output_dir, fname_td)
    print(f"\n[camera 24] top-down  eye={topdown_eye.round(3)}  z={ceiling_z:.3f} (0.02m below ceiling)")
    og.sim._viewer_camera.set_position_orientation(topdown_eye, topdown_quat)
    _capture(fpath_td)
    camera_poses[fname_td] = {
        "position":        topdown_eye.tolist(),
        "quaternion_xyzw": topdown_quat.tolist(),
        "azimuth_deg":     None,
        "height":          ceiling_z,
        "type":            "top_down",
    }

    vis_td = _visibility_check(("obj_ref", "obj_near", "obj_far"))
    exist_ref_flags ["exist_ref_24"]  = vis_td["obj_ref"]
    exist_near_flags["exist_near_24"] = vis_td["obj_near"]
    exist_far_flags ["exist_far_24"]  = vis_td["obj_far"]
    print(f"[camera 24] visibility: {vis_td}")

    return exist_ref_flags, exist_near_flags, exist_far_flags, camera_poses


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
    parser.add_argument("--output_root", type=str, default="renders_distance")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 200
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"  Task: distance comparison (near vs far from reference object)")
    print(f"{'='*70}\n")

    # ── Load scene graph ───────────────────────────────────────────────────────
    scene_dict  = load_scene_dict(args.scene)
    room_objs   = get_room_objects(scene_dict, args.room)

    # ── Resolve top-down camera Z from ceiling bbox ────────────────────────────
    ceiling_z = get_ceiling_z(room_objs)

    # Build spatial summaries for GPT
    all_room_bboxes = []
    for bboxes in room_objs.values():
        all_room_bboxes.extend(bboxes)

    SKIP_CATS = {"ceilings", "walls", "floors", "carpet", "window",
                 "door", "curtain", "electric_switch"}

    # Room centre = centre of the floor bbox in XY
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        room_centre_xy = np.array([
            (float(fb[0][0]) + float(fb[1][0])) / 2,
            (float(fb[0][1]) + float(fb[1][1])) / 2,
        ])
    else:
        # Fallback: centroid of all object centres
        all_centres = []
        for bboxes in room_objs.values():
            for (bmin, bmax) in bboxes:
                all_centres.append([(float(bmin[0]) + float(bmax[0])) / 2,
                                    (float(bmin[1]) + float(bmax[1])) / 2])
        room_centre_xy = np.mean(all_centres, axis=0) if all_centres else np.zeros(2)
    print(f"[room] centre_xy={room_centre_xy.round(3)}")

    room_summaries = {}
    for cat, bboxes in room_objs.items():
        if any(s in cat.lower() for s in SKIP_CATS):
            continue
        other_bboxes = [b for c, bb in room_objs.items() for b in bb if c != cat]
        # Distance from object centre to room centre
        obj_centre_xy = np.array([
            (float(bboxes[0][0][0]) + float(bboxes[0][1][0])) / 2,
            (float(bboxes[0][0][1]) + float(bboxes[0][1][1])) / 2,
        ])
        dist_to_centre = float(np.linalg.norm(obj_centre_xy - room_centre_xy))
        room_summaries[cat] = {
            "footprint":      bbox_footprint(bboxes),
            "clearance":      bbox_clearance(bboxes, other_bboxes),
            "dist_to_centre": dist_to_centre,
        }

    room_categories = list(room_summaries.keys())
    if not room_categories:
        print("[ERROR] No usable room categories found.")
        raise SystemExit(2)

    # ── Sample 200 candidate categories and ask GPT ────────────────────────────
    all_keys          = load_keys(args.keys_json)
    sampled           = sample_200(all_keys, seed=seed)
    gpt_result        = gpt_pick_reference_and_2_similar(
        room_categories, sampled, room_summaries)

    ref_category      = gpt_result["reference"]
    near_category     = gpt_result["obj_near_cat"]
    far_category      = gpt_result["obj_far_cat"]

    # ── Validate reference object ──────────────────────────────────────────────
    # Two conditions must both pass:
    #   (1) At least 2 of its 4 bbox corners are ≥ EDGE_MARGIN from every room edge.
    #   (2) Its bbox min Z is within FLOOR_Z_MARGIN of the floor's max Z,
    #       i.e. abs(bbox_min_z - floor_max_z) <= FLOOR_Z_MARGIN.
    #       This ensures the object is resting on (or very near) the floor surface
    #       and rejects elevated objects like shelves or things on top of furniture.
    # If either condition fails the GPT pick, we fall back to the next-best
    # candidate sorted by dist_to_centre.

    EDGE_MARGIN    = 0.5   # metres — minimum distance from any room edge
    FLOOR_Z_MARGIN = 0.2   # metres — max allowed |bbox_min_z − floor_max_z|

    # Derive floor surface Z from the scene-dict floor bbox max Z.
    if floor_bboxes:
        floor_z_max = float(floor_bboxes[0][1][2])
    else:
        floor_z_max = 0.0
    print(f"[ref] floor_z_max={floor_z_max:.3f}  "
          f"(ref bbox_min Z must be within ±{FLOOR_Z_MARGIN}m of this)")

    def _on_floor(cat: str) -> bool:
        """Return True if the category's bbox min Z is within FLOOR_Z_MARGIN of the floor max Z."""
        bboxes = room_objs.get(cat, [])
        if not bboxes:
            return True   # can't check → assume fine
        bbox_min_z = float(bboxes[0][0][2])
        return abs(bbox_min_z - floor_z_max) <= FLOOR_Z_MARGIN

    def _corners_clear_of_edges(cat: str) -> int:
        """Return how many of the 4 ref bbox corners are ≥ EDGE_MARGIN from every room edge."""
        bboxes = room_objs.get(cat, [])
        if not bboxes or not floor_bboxes:
            return 4   # can't check → assume fine
        bmin, bmax = bboxes[0]
        fb = floor_bboxes[0]
        fx_min, fy_min = float(fb[0][0]), float(fb[0][1])
        fx_max, fy_max = float(fb[1][0]), float(fb[1][1])
        count = 0
        for cname in CORNER_NAMES:
            cxy = corner_xy(bmin, bmax, cname)
            if (cxy[0] - fx_min >= EDGE_MARGIN and
                fx_max - cxy[0] >= EDGE_MARGIN and
                cxy[1] - fy_min >= EDGE_MARGIN and
                fy_max - cxy[1] >= EDGE_MARGIN):
                count += 1
        return count

    def _ref_qualifies(cat: str) -> bool:
        return _corners_clear_of_edges(cat) >= 2 and _on_floor(cat)

    if not _ref_qualifies(ref_category):
        # Log which check(s) failed for diagnostics
        edge_ok  = _corners_clear_of_edges(ref_category) >= 2
        floor_ok = _on_floor(ref_category)
        reasons  = []
        if not edge_ok:
            reasons.append(f"<2 corners with ≥{EDGE_MARGIN}m from room edges")
        if not floor_ok:
            bbox_min_z = float(room_objs[ref_category][0][0][2]) if room_objs.get(ref_category) else float("nan")
            reasons.append(
                f"bbox_min Z={bbox_min_z:.3f} not within ±{FLOOR_Z_MARGIN}m "
                f"of floor_z_max={floor_z_max:.3f}"
            )
        print(f"[ref] '{ref_category}' disqualified ({'; '.join(reasons)}) — searching for fallback")

        fallback = None
        for cat, _ in sorted(room_summaries.items(),
                              key=lambda kv: kv[1]["dist_to_centre"]):
            if cat == ref_category:
                continue
            if _ref_qualifies(cat):
                fallback = cat
                break

        if fallback is None:
            print("[ref] ERROR: no candidate passes edge-margin and floor-Z checks")
            raise SystemExit(2)
        print(f"[ref] Falling back to '{fallback}'")
        ref_category = fallback

    # ── Get AABB of reference from scene dict ─────────────────────────────────
    ref_bboxes   = room_objs[ref_category]          # list of [bmin, bmax]
    # Use the first instance (or union)
    ref_bbox     = ref_bboxes[0]
    ref_bbox_min = np.array(ref_bbox[0], dtype=float)
    ref_bbox_max = np.array(ref_bbox[1], dtype=float)
    ref_centre   = (ref_bbox_min + ref_bbox_max) / 2.0
    print(f"[ref] category={ref_category}  centre={ref_centre.round(3)}")
    print(f"[ref] bbox_min={ref_bbox_min.round(3)}  bbox_max={ref_bbox_max.round(3)}")

    # ── Load models for comparison objects ────────────────────────────────────
    near_model = get_model_for_category(near_category, seed)
    far_model  = get_model_for_category(far_category, seed + 1)

    # ── Find categories within 1 m of the reference object centre ───────────
    # These clutter the reference area and are excluded from loading.
    NEARBY_THRESHOLD = 1.0   # metres
    ref_centre_xy_2d = np.array([ref_centre[0], ref_centre[1]])
    nearby_cats = []
    for cat, bboxes in room_objs.items():
        if cat == ref_category:
            continue
        if "wall" in cat or "floor" in cat:
            continue
        for (bmin, bmax) in bboxes:
            obj_cx = (float(bmin[0]) + float(bmax[0])) / 2
            obj_cy = (float(bmin[1]) + float(bmax[1])) / 2
            dist = float(np.linalg.norm(np.array([obj_cx, obj_cy]) - ref_centre_xy_2d))
            if dist < NEARBY_THRESHOLD:
                nearby_cats.append(cat)
                break   # one instance close enough → exclude whole category
    print(f"[nearby] categories within {NEARBY_THRESHOLD}m of ref centre, will not load: {nearby_cats}")

    # ── Build OmniGibson config ───────────────────────────────────────────────
    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"] + nearby_cats
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"] = [
        build_object_config("obj_near", near_category, near_model, 0),
        build_object_config("obj_far",  far_category,  far_model,  1),
    ]

    env   = og.Environment(configs=config)
    scene = env.scene

    obj_near = scene.object_registry("name", "obj_near")
    obj_far  = scene.object_registry("name", "obj_far")
    floor    = scene.object_registry("name", args.floor)

    if floor is None:
        print(f"[ERROR] Floor '{args.floor}' not found — exiting with code 2")
        raise SystemExit(2)

    # ── Find the in-scene reference object by category ────────────────────────
    # We read its live AABB if it exists in the loaded scene; otherwise use
    # the scene-dict bbox (which already gives us ref_bbox_min/max above).
    obj_ref = None
    for obj in scene.objects:
        if hasattr(obj, "category") and obj.category == ref_category:
            obj_ref = obj
            break

    if obj_ref is not None:
        live_min, live_max = [x.cpu().numpy() for x in obj_ref.aabb]
        ref_bbox_min = live_min
        ref_bbox_max = live_max
        ref_centre   = (live_min + live_max) / 2.0
        print(f"[ref] Live AABB min={live_min.round(3)}  max={live_max.round(3)}")
    else:
        print(f"[ref] Object '{ref_category}' not in loaded scene — using scene-dict bbox")

    # ── Build list of all other room bboxes (excluding the reference category) ─
    other_bboxes = [b for cat, bbs in room_objs.items()
                    for b in bbs if cat != ref_category]

    # ── Place comparison objects ──────────────────────────────────────────────
    placement = place_comparison_objects(
        scene, obj_near, obj_far, floor,
        ref_bbox_min, ref_bbox_max,
        other_bboxes=other_bboxes, seed=seed)

    near_pos_np = np.array(placement["near_pos"])

    # ── Render 24 side views + 1 top-down view ────────────────────────────────
    exist_ref, exist_near, exist_far, camera_poses = render_and_save(
        output_dir=run_dir,
        ref_centre=ref_centre,
        ref_aabb_min=ref_bbox_min,
        ref_aabb_max=ref_bbox_max,
        near_pos=near_pos_np,
        placement=placement,
        ceiling_z=ceiling_z)

    # ── Collect object metadata ───────────────────────────────────────────────
    def obj_meta(obj, category, model):
        pos, quat = obj.get_position_orientation()
        return {
            "category":        category,
            "model":           model,
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    objects_meta = {
        "obj_near": obj_meta(obj_near, near_category, near_model),
        "obj_far":  obj_meta(obj_far,  far_category,  far_model),
    }
    if obj_ref is not None:
        pos_r, quat_r = obj_ref.get_position_orientation()
        objects_meta["obj_ref"] = {
            "category":        ref_category,
            "model":           "scene_native",
            "position":        pos_r.cpu().numpy().tolist(),
            "quaternion_xyzw": quat_r.cpu().numpy().tolist(),
        }

    metadata = {
        "scene":           args.scene,
        "room":            args.room,
        "run_idx":         args.run_idx,
        "seed":            seed,
        "floor_name":      args.floor,
        "layout":          "distance_comparison",
        "ref_category":    ref_category,
        "near_category":   near_category,
        "far_category":    far_category,
        "ref_bbox_min":    ref_bbox_min.tolist(),
        "ref_bbox_max":    ref_bbox_max.tolist(),
        "ref_centre":      ref_centre.tolist(),
        "d_near":          placement["d_near"],
        "d_far":           placement["d_far"],
        "near_corner":     placement["near_corner"],
        "far_corner":      placement["far_corner"],
        "near_corner_xy":  placement["near_corner_xy"],
        "far_corner_xy":   placement["far_corner_xy"],
        "corner_clearances": placement["corner_clearances"],
        "d_gap":           D_FAR_EXTRA,
        "dist_near_centre": placement["dist_near_centre"],
        "dist_far_centre":  placement["dist_far_centre"],
        "near_pos":        placement["near_pos"],
        "far_pos":         placement["far_pos"],
        "answer":          "obj_near",      # ground-truth: obj_near is closer
        "nearby_excluded": nearby_cats,
        "topdown_ceiling_z": ceiling_z,
        **exist_ref,    # exist_ref_0  .. exist_ref_24
        **exist_near,   # exist_near_0 .. exist_near_24
        **exist_far,    # exist_far_0  .. exist_far_24
        "objects":      objects_meta,
        "camera_poses": camera_poses,
        "camera_layout": {
            "num_azimuths":    NUM_AZIMUTHS,
            "azimuth_step":    AZIMUTH_STEP,
            "heights":         CAM_HEIGHTS,
            "camera_offset":   CAMERA_OFFSET,
            "camera_offset2":  CAMERA_OFFSET2,
            "topdown_z":       ceiling_z,
            "topdown_z_source": "ceiling_bbox_min_z - 0.02",
            "radius":          "max(ref_half_diag+OFFSET, corner_dist_far+d_far+half_far+OFFSET2)",
            "start_azimuth":   "direction from ref_centre to obj_near",
            "view_ordering":   "view 2k = azimuth k z=0.05, view 2k+1 = azimuth k z=0.40, view 24 = top-down",
            "corner_selection": "random sample from corners with clearance>0.5m (fallback: top-2)",
        },
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # Exit 0 if all views have the reference and at least one comparison visible
    all_ref_visible  = all(exist_ref [f"exist_ref_{k}"]  for k in range(25))
    any_near_visible = any(exist_near[f"exist_near_{k}"] for k in range(25))
    any_far_visible  = any(exist_far [f"exist_far_{k}"]  for k in range(25))
    success = all_ref_visible and any_near_visible and any_far_visible
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()