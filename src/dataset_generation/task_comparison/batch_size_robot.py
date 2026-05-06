"""
batch_size.py

Single-run script: loads ONE scene restricted to ONE room, uses GPT to select
a "task object" category (not tiny, not large — medium-sized household objects).
Two copies of the SAME model are loaded:
  - task_obj1: scale 1.0 (normal size)
  - task_obj2: scale uniform(1.2, 1.3) (larger version)

A pair of "reference objects" — chosen to be visually very different from each
other — are placed one per task object, within 0.1 m of its bbox boundary.

Camera layout (per task object):
  - 12 azimuths, starting from the direction of its reference object, every 30°
  - 1 height per azimuth: z = 0.40 (eye-level)
  - All cameras look at the task object centre
  → 12 side views + 1 top-down view (z=1.5) = 13 renders per task object
  → 26 renders total (views 0–12 for task_obj1, views 13–25 for task_obj2)

View numbering:
  views  0–11  : task_obj1, azimuths 0°–330° (step 30°)
  view  12     : task_obj1, top-down z=1.5
  views 13–24  : task_obj2, azimuths 0°–330°
  view  25     : task_obj2, top-down z=1.5
  views 26–34  : cross-views — behind task_obj1, looking at task_obj2
                 (3 standoffs × 3 tilts: centre/−15°/+15°)
  views 35–43  : cross-views — behind task_obj2, looking at task_obj1
                 (3 standoffs × 3 tilts: centre/−15°/+15°)

JSON flags per view:
  exist_task_obj1_k  — task_obj1 visible in view k
  exist_task_obj2_k  — task_obj2 visible in view k
  exist_ref_obj1_k   — ref_obj1  visible in view k
  exist_ref_obj2_k   — ref_obj2  visible in view k

Ground-truth answer: "task_obj2" is bigger (scale > 1.0).

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

# ── Scale parameters ──────────────────────────────────────────────────────────
SCALE_NORMAL_MIN = 1.2   # task_obj2 scale lower bound
SCALE_NORMAL_MAX = 1.3   # task_obj2 scale upper bound
SQUARE_ORI       = [0.0, 0.0, 0.0, 1.0]   # identity quaternion — axis-aligned

# ── Reference placement ───────────────────────────────────────────────────────
REF_PROXIMITY    = 0.05  # metres — reference object bbox must be within this of task obj bbox

# ── Camera ring parameters ────────────────────────────────────────────────────
NUM_AZIMUTHS     = 6
AZIMUTH_STEP     = 60.0      # degrees
CAM_HEIGHT       = 0.05      # single Z height for side views
CAM_RADIUS_PAD   = 0.15      # metres beyond task-obj bbox half-diagonal
CAM_TOPDOWN_Z    = 1.5       # fixed top-down camera height

# ── Scenes folder ─────────────────────────────────────────────────────────────
SCENES_DIR = "scenes5"

# ── Placement margins ─────────────────────────────────────────────────────────
WALL_MARGIN    = 0.10   # min gap between any placed object bbox and floor bbox edge
OBJ_MARGIN     = 0.20   # min gap between placed object bbox and scene-graph objects

# ── Cross-view stand-off distances (beyond near-object bbox face) ─────────────
CROSS_STANDOFFS = [0.15, 0.20, 0.25, 0.30]   # metres past the near-face

# ── Task-object minimum centre-to-centre separation ───────────────────────────
MIN_TASK_SEPARATION = 0.35   # metres


# ─────────────────────────────────────────────────────────────────────────────
# Scene-graph helpers  (unchanged from batch_distance.py)
# ─────────────────────────────────────────────────────────────────────────────

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


def bbox_footprint(bboxes: list) -> float:
    total = 0.0
    for (bmin, bmax) in bboxes:
        total += abs(bmax[0] - bmin[0]) * abs(bmax[1] - bmin[1])
    return total


CORNER_NAMES = ["tr", "tl", "br", "bl"]

def corner_xy(bmin, bmax, corner: str) -> np.ndarray:
    xmin, ymin = float(bmin[0]), float(bmin[1])
    xmax, ymax = float(bmax[0]), float(bmax[1])
    return {
        "tr": np.array([xmax, ymax]),
        "tl": np.array([xmin, ymax]),
        "br": np.array([xmax, ymin]),
        "bl": np.array([xmin, ymin]),
    }[corner]


def outward_dir(corner: str) -> np.ndarray:
    return {
        "tr": np.array([ 1,  1], dtype=float) / np.sqrt(2),
        "tl": np.array([-1,  1], dtype=float) / np.sqrt(2),
        "br": np.array([ 1, -1], dtype=float) / np.sqrt(2),
        "bl": np.array([-1, -1], dtype=float) / np.sqrt(2),
    }[corner]


def corner_clearances(bboxes: list, all_bboxes_in_room: list) -> dict:
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
            if dist > 0.01:
                min_dist = min(min_dist, dist)
        result[cname] = min_dist
    return result


def bbox_clearance(bboxes: list, all_bboxes_in_room: list) -> float:
    cc = corner_clearances(bboxes, all_bboxes_in_room)
    return min(cc.values()) if cc else 999.0


# ─────────────────────────────────────────────────────────────────────────────
# GPT helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_keys(keys_json_path: str) -> list:
    with open(keys_json_path) as f:
        return json.load(f)


def sample_200(all_keys: list, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(all_keys, min(200, len(all_keys)))


def gpt_pick_task_and_refs(
    candidate_categories: list,
) -> dict:
    """
    Ask GPT to pick THREE distinct categories from candidate_categories:
      1. task_cat  — medium-sized household object (both copies will be loaded,
                     one at scale 1.0, one at scale 1.2–1.3).
      2. ref1_cat  — medium-sized object, visually very different from ref2_cat,
                     placed near task_obj1.
      3. ref2_cat  — medium-sized object, visually very different from ref1_cat,
                     placed near task_obj2.

    All three must come from candidate_categories and must be distinct.
    Returns {"task_cat": str, "ref1_cat": str, "ref2_cat": str}
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a spatial-reasoning assistant for a robotics simulation.\n\n"
        "You will be given a list of loadable object categories.\n\n"
        "Choose exactly 3 DISTINCT categories from the list:\n"
        "  1. 'task_cat': a MEDIUM-SIZED household object that exists as two "
        "     copies — one normal-sized, one slightly larger.  "
        "     Must NOT be tiny (no: dice, coin, key, screw, pebble, pen, eraser) "
        "     and NOT large objects (no: furniture, sofa, wardrobe, bookcase, bed, "
        "     dining_table, bathtub, no suitcase, no instruments).  "
        "     Good examples: mug, bottle, shoe, helmet, flower_pot, vase, "
        "     plate, basketball, teddy_bear, guitar, suitcase.\n"
        "  2. 'ref1_cat': a SMALL-SIZED object placed as a visual reference "
        "     next to the first task copy.  \n"
        "  3. 'ref2_cat': a SMALL-SIZED object placed as a visual reference "
        "     next to the second task copy.  \n\n"
        "Additional constraints:\n"
        "  - ref1_cat and ref2_cat must be VISUALLY VERY DIFFERENT from each other "
        "    — contrasting shape, material, and function "
        "    (e.g. a round soft ball vs a teddy bear).\n"
        "  - All three values must be different from each other.\n"
        "  - All three must appear verbatim in the provided list.\n\n"
        "  - The ref cats you choose should be ideally smaller than the task cats-"
        "Reply with ONLY a JSON object:\n"
        '  {"task_cat": "...", "ref1_cat": "...", "ref2_cat": "..."}\n'
        "No markdown fences, no extra text."
    )

    user_prompt = (
        f"Candidate categories:\n{json.dumps(candidate_categories)}\n\n"
        "Return JSON with task_cat, ref1_cat, ref2_cat — all distinct, "
        "all from the list above."
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

    # Validate all three are distinct and present in candidates
    for key in ("task_cat", "ref1_cat", "ref2_cat"):
        if result[key] not in candidate_categories:
            raise ValueError(f"GPT returned '{result[key]}' for '{key}' "
                             f"which is not in the candidate list.")
    cats = [result["task_cat"], result["ref1_cat"], result["ref2_cat"]]
    if len(set(cats)) != 3:
        raise ValueError(f"GPT returned non-distinct categories: {cats}")

    print(f"[GPT] task_cat={result['task_cat']}  "
          f"ref1_cat={result['ref1_cat']}  ref2_cat={result['ref2_cat']}")
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


def build_object_config(name: str, category: str, model: str,
                        idx: int, scale: list) -> dict:
    return {
        "type": "DatasetObject",
        "name": name,
        "category": category,
        "model": model,
        "position": [150.0 + idx * 10, 100.0, 100.0],
        "orientation": SQUARE_ORI,
        "scale": scale,
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
    return Rotation.from_matrix(rot_matrix).as_quat()


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


def _visibility_check(obj_names: tuple) -> dict:
    """Step 100 frames then check seg_instance for each object name."""
    for _ in range(100):
        og.sim.step()
    raw_obs      = og.sim._viewer_camera._annotators["seg_instance"].get_data()
    id_to_labels = raw_obs["info"]["idToLabels"]
    visible_str  = " ".join(id_to_labels.values())
    result = {name: (name in visible_str) for name in obj_names}
    print(f"[seg] {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reference object placement
# ─────────────────────────────────────────────────────────────────────────────

def _xy_half_extent(obj) -> float:
    bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
    dx = abs(bmax[0] - bmin[0]) / 2
    dy = abs(bmax[1] - bmin[1]) / 2
    return float(np.sqrt(dx**2 + dy**2))


def _aabb_overlaps_xy(ax_min, ax_max, bx_min, bx_max,
                      ay_min, ay_max, by_min, by_max) -> bool:
    """Return True if two axis-aligned boxes overlap in XY (2-D check only)."""
    return not (ax_max <= bx_min or bx_max <= ax_min or
                ay_max <= by_min or by_max <= ay_min)


def _collision_free_sides(sides: dict, ref_half_x: float, ref_half_y: float,
                          room_objs: dict, extra_bboxes: list,
                          skip_cats: set, label: str) -> list:
    """
    For each candidate side and its proposed ref_obj centre XY, build the
    proposed ref AABB and check it for XY overlap against:

      (a) every bbox in room_objs (the scenes5 scene-graph dict,
          category → [[bmin, bmax], ...]) — skipping categories in skip_cats,
      (b) every bbox in extra_bboxes — live AABBs of freshly-loaded objects
          not present in the scene graph (e.g. task_obj1, task_obj2 after
          placement).  Each entry is (label_str, bmin_arr, bmax_arr).

    Returns the list of side names that are collision-free, in input order.
    Logs which objects/categories block each side.

    Proposed ref AABB at centre (cx, cy):
        [cx - ref_half_x, cx + ref_half_x] × [cy - ref_half_y, cy + ref_half_y]
    """
    SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                       "door", "curtain", "electric_switch"}
    all_skip = SKIP_STRUCTURAL | skip_cats

    free = []
    for side, centre_xy in sides.items():
        cx, cy = float(centre_xy[0]), float(centre_xy[1])
        p_xmin = cx - ref_half_x
        p_xmax = cx + ref_half_x
        p_ymin = cy - ref_half_y
        p_ymax = cy + ref_half_y

        collisions = []

        # (a) scene-graph bboxes
        for cat, bboxes in room_objs.items():
            if any(s in cat.lower() for s in all_skip):
                continue
            for (bmin, bmax) in bboxes:
                if _aabb_overlaps_xy(p_xmin, p_xmax,
                                     float(bmin[0]), float(bmax[0]),
                                     p_ymin, p_ymax,
                                     float(bmin[1]), float(bmax[1])):
                    collisions.append(cat)
                    break   # one instance of this category is enough

        # (b) extra live bboxes (task objects placed this session)
        for (name, bmin, bmax) in extra_bboxes:
            if _aabb_overlaps_xy(p_xmin, p_xmax,
                                 float(bmin[0]), float(bmax[0]),
                                 p_ymin, p_ymax,
                                 float(bmin[1]), float(bmax[1])):
                collisions.append(name)

        if collisions:
            print(f"[ref_{label}] side={side} BLOCKED by: {collisions}")
        else:
            print(f"[ref_{label}] side={side} clear")
            free.append(side)
    return free


def place_ref_near_task(scene, ref_obj, floor_obj,
                        task_obj, room_objs: dict, extra_bboxes: list,
                        seed: int, label: str,
                        forbidden_sides: set = None) -> dict:
    """
    Place `ref_obj` so that the gap between its bbox boundary and the task
    object's bbox boundary equals REF_PROXIMITY (0.05 m), measured face-to-face
    on the chosen side (N/S/E/W).

    All bbox measurements are taken from LIVE AABBs so that the gap
    calculation always uses real post-physics extents.

    forbidden_sides: set of side names (subset of {"N","S","E","W"}) that are
        unconditionally excluded before the collision check.  Used to prevent
        the reference object from being placed in the corridor between the two
        task objects (i.e. the side of task_obj1 that faces task_obj2, and
        vice versa).

    Steps:
      1. Drop ref_obj via OnTop to get a valid floor Z.
      2. Read ref_obj live AABB → XY half-extents.
      3. Read task_obj live AABB → task face coordinates.
      4. Compute the 4 candidate ref_obj centre positions (one per side) using:
             task_face + REF_PROXIMITY + ref_half  (N/E sides)
             task_face - REF_PROXIMITY - ref_half  (S/W sides)
      5. Remove forbidden_sides, then pre-collision-check remaining candidates.
         - Pick randomly from collision-free sides (preferred).
         - If all remaining sides collide, fall back to a random non-forbidden side.
      6. Teleport ref_obj to chosen XY, step physics.
      7. Re-read both live AABBs and compute/log actual bbox gap.

    Returns placement metadata dict including the verified actual_bbox_gap.
    """
    if forbidden_sides is None:
        forbidden_sides = set()
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    rng = random.Random(seed)

    # ── Floor bbox (for wall-margin check on ref obj) ─────────────────────────
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        fx_min = float(fb[0][0])
        fx_max = float(fb[1][0])
        fy_min = float(fb[0][1])
        fy_max = float(fb[1][1])
    else:
        fx_min, fx_max, fy_min, fy_max = -99.0, 99.0, -99.0, 99.0

    # ── Step 1: drop ref_obj onto floor to get a valid Z ─────────────────────
    ref_obj.states[object_states.OnTop].set_value(floor_obj, True)
    for _ in range(15):
        og.sim.step()
    pos, _ = ref_obj.get_position_orientation()
    ref_obj.set_position_orientation(position=pos, orientation=SQUARE_ORI_T)
    ref_obj.keep_still()
    for _ in range(10):
        og.sim.step()

    ref_z = ref_obj.get_position_orientation()[0].cpu().numpy()[2]

    # ── Step 2: read ref_obj live AABB for XY half-extents ───────────────────
    ref_bmin, ref_bmax = [x.cpu().numpy() for x in ref_obj.aabb]
    ref_half_x = abs(float(ref_bmax[0]) - float(ref_bmin[0])) / 2.0
    ref_half_y = abs(float(ref_bmax[1]) - float(ref_bmin[1])) / 2.0
    print(f"[ref_{label}] live ref half_x={ref_half_x:.4f}  half_y={ref_half_y:.4f}")

    # ── Step 3: read task_obj live AABB for face coordinates ─────────────────
    task_bmin, task_bmax = [x.cpu().numpy() for x in task_obj.aabb]
    task_cx = (float(task_bmin[0]) + float(task_bmax[0])) / 2.0
    task_cy = (float(task_bmin[1]) + float(task_bmax[1])) / 2.0
    task_face = {
        "N": float(task_bmax[1]),
        "S": float(task_bmin[1]),
        "E": float(task_bmax[0]),
        "W": float(task_bmin[0]),
    }
    print(f"[ref_{label}] live task bbox  min={task_bmin[:2].round(4)}  max={task_bmax[:2].round(4)}")

    # ── Step 4: candidate ref_obj centres (face-to-face gap = REF_PROXIMITY) ──
    sides = {
        "N": np.array([task_cx,                                      task_face["N"] + REF_PROXIMITY + ref_half_y]),
        "S": np.array([task_cx,                                      task_face["S"] - REF_PROXIMITY - ref_half_y]),
        "E": np.array([task_face["E"] + REF_PROXIMITY + ref_half_x, task_cy]),
        "W": np.array([task_face["W"] - REF_PROXIMITY - ref_half_x, task_cy]),
    }

    # ── Step 5: pre-collision check against scene graph + live task bboxes ─────
    # First remove sides that would place the ref obj outside the floor bbox
    # (respecting WALL_MARGIN, same constraint as task objects).
    def _ref_inside_floor(centre_xy, hx, hy):
        cx, cy = float(centre_xy[0]), float(centre_xy[1])
        return (cx - hx >= fx_min and
                cx + hx <= fx_max and
                cy - hy >= fy_min and
                cy + hy <= fy_max)

    sides_in_floor = {k: v for k, v in sides.items()
                      if _ref_inside_floor(v, ref_half_x, ref_half_y)}
    if len(sides_in_floor) < len(sides):
        excluded = set(sides) - set(sides_in_floor)
        print(f"[ref_{label}] sides excluded (outside floor bbox): {excluded}")
    if not sides_in_floor:
        sides_in_floor = sides   # safety fallback: skip floor check if all fail
        print(f"[ref_{label}] WARNING: all sides outside floor bbox — ignoring floor check")

    # Then remove any forbidden sides (the side facing the other task object).
    allowed_sides = {k: v for k, v in sides_in_floor.items() if k not in forbidden_sides}
    if forbidden_sides:
        print(f"[ref_{label}] forbidden sides (face toward other task obj): {forbidden_sides}")
    if not allowed_sides:
        allowed_sides = sides_in_floor   # safety: if forbidden_sides wiped everything

    # skip_cats: skip the ref obj's own category and the task obj's category
    # from scene-graph checks (they are either already excluded or irrelevant).
    skip_cats = {getattr(ref_obj, "category", ""), getattr(task_obj, "category", "")}
    free_sides = _collision_free_sides(allowed_sides, ref_half_x, ref_half_y,
                                       room_objs, extra_bboxes, skip_cats, label)

    # Shuffle the preferred order with the seeded rng for variety
    if free_sides:
        rng.shuffle(free_sides)
        chosen_side = free_sides[0]
        collision_fallback = False
        print(f"[ref_{label}] collision-free sides={free_sides}  → chose {chosen_side}")
    else:
        # All allowed sides blocked — pick randomly from allowed and warn
        chosen_side = rng.choice(list(allowed_sides.keys()))
        collision_fallback = True
        print(f"[ref_{label}] WARNING: all allowed sides blocked — falling back to {chosen_side}")

    ref_xy = sides[chosen_side]
    print(f"[ref_{label}] side={chosen_side}  target_xy={ref_xy.round(4)}  z={ref_z:.4f}")

    # ── Step 6: teleport ref_obj ──────────────────────────────────────────────
    ref_obj.set_position_orientation(
        position=th.tensor([ref_xy[0], ref_xy[1], ref_z], dtype=th.float32),
        orientation=SQUARE_ORI_T,
    )
    ref_obj.keep_still()
    for _ in range(30):
        og.sim.step()

    # ── Step 7: re-read BOTH live AABBs and compute actual bbox gap ───────────
    ref_bmin_f,  ref_bmax_f  = [x.cpu().numpy() for x in ref_obj.aabb]
    task_bmin_f, task_bmax_f = [x.cpu().numpy() for x in task_obj.aabb]

    ref_near_face = {
        "N": float(ref_bmin_f[1]),
        "S": float(ref_bmax_f[1]),
        "E": float(ref_bmin_f[0]),
        "W": float(ref_bmax_f[0]),
    }[chosen_side]
    task_near_face = {
        "N": float(task_bmax_f[1]),
        "S": float(task_bmin_f[1]),
        "E": float(task_bmax_f[0]),
        "W": float(task_bmin_f[0]),
    }[chosen_side]
    actual_gap = abs(ref_near_face - task_near_face)

    final_pos = ref_obj.get_position_orientation()[0].cpu().numpy()
    print(f"[ref_{label}] final_pos={final_pos.round(4)}")
    print(f"[ref_{label}] task_face={task_near_face:.4f}  ref_near_face={ref_near_face:.4f}  "
          f"actual_bbox_gap={actual_gap:.4f}m  (target={REF_PROXIMITY}m)  "
          f"collision_fallback={collision_fallback}")

    return {
        "side":              chosen_side,
        "position":          final_pos.tolist(),
        "actual_bbox_gap":   float(actual_gap),
        "target_bbox_gap":   REF_PROXIMITY,
        "collision_fallback": collision_fallback,
        "ref_half_x":        float(abs(float(ref_bmax_f[0]) - float(ref_bmin_f[0])) / 2.0),
        "ref_half_y":        float(abs(float(ref_bmax_f[1]) - float(ref_bmin_f[1])) / 2.0),
        "task_face_coord":   float(task_near_face),
        "ref_near_face":     float(ref_near_face),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task object placement
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_clear(cx: float, cy: float,
                     hx: float, hy: float,
                     floor_xmin: float, floor_xmax: float,
                     floor_ymin: float, floor_ymax: float,
                     room_objs: dict,
                     extra_bboxes: list,
                     skip_cats: set) -> bool:
    """
    Return True if placing an object with XY half-extents (hx, hy) at centre
    (cx, cy) satisfies:
      (a) its bbox stays >= WALL_MARGIN inside the floor bbox, and
      (b) its bbox is >= OBJ_MARGIN away from every scene-graph object bbox
          (excluding skip_cats) and every extra_bboxes entry.
    """
    SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                       "door", "curtain", "electric_switch"}
    all_skip = SKIP_STRUCTURAL | skip_cats

    # (a) wall margin check
    if (cx - hx) < floor_xmin + WALL_MARGIN:   return False
    if (cx + hx) > floor_xmax - WALL_MARGIN:   return False
    if (cy - hy) < floor_ymin + WALL_MARGIN:   return False
    if (cy + hy) > floor_ymax - WALL_MARGIN:   return False

    # Inflate the proposed bbox by OBJ_MARGIN for overlap tests
    pxmin = cx - hx - OBJ_MARGIN
    pxmax = cx + hx + OBJ_MARGIN
    pymin = cy - hy - OBJ_MARGIN
    pymax = cy + hy + OBJ_MARGIN

    # (b) scene-graph objects
    for cat, bboxes in room_objs.items():
        if any(s in cat.lower() for s in all_skip):
            continue
        for (bmin, bmax) in bboxes:
            if _aabb_overlaps_xy(pxmin, pxmax, float(bmin[0]), float(bmax[0]),
                                 pymin, pymax, float(bmin[1]), float(bmax[1])):
                return False

    # (c) freshly-placed objects (extra_bboxes)
    for (_, bmin, bmax) in extra_bboxes:
        if _aabb_overlaps_xy(pxmin, pxmax, float(bmin[0]), float(bmax[0]),
                             pymin, pymax, float(bmin[1]), float(bmax[1])):
            return False
    return True


def _find_clear_position(hx: float, hy: float,
                         floor_xmin: float, floor_xmax: float,
                         floor_ymin: float, floor_ymax: float,
                         room_objs: dict,
                         extra_bboxes: list,
                         skip_cats: set,
                         rng: random.Random,
                         n_candidates: int = 200,
                         label: str = "") -> np.ndarray:
    """
    Uniformly sample n_candidates random positions inside the floor bbox and
    return the first that passes _candidate_clear (wall margin + obj margin)
    and MIN_TASK_SEPARATION from extra_bboxes.  Falls back to room centre on failure.
    """
    x_lo = floor_xmin + hx + WALL_MARGIN
    x_hi = floor_xmax - hx - WALL_MARGIN
    y_lo = floor_ymin + hy + WALL_MARGIN
    y_hi = floor_ymax - hy - WALL_MARGIN

    if x_lo >= x_hi or y_lo >= y_hi:
        print(f"[place_{label}] WARNING: floor too small for object")
        return None

    for _ in range(n_candidates):
        cx = rng.uniform(x_lo, x_hi)
        cy = rng.uniform(y_lo, y_hi)

        if not _candidate_clear(cx, cy, hx, hy,
                                floor_xmin, floor_xmax, floor_ymin, floor_ymax,
                                room_objs, extra_bboxes, skip_cats):
            continue

        too_close = False
        for (_, bmin, bmax) in extra_bboxes:
            other_cx = (float(bmin[0]) + float(bmax[0])) / 2.0
            other_cy = (float(bmin[1]) + float(bmax[1])) / 2.0
            if np.hypot(cx - other_cx, cy - other_cy) < MIN_TASK_SEPARATION:
                too_close = True
                break
        if too_close:
            continue

        print(f"[place_{label}] clear candidate found at ({cx:.3f}, {cy:.3f})")
        return np.array([cx, cy])

    # fallback: failed
    print(f"[place_{label}] WARNING: no clear candidate in {n_candidates} samples")
    return None


def place_task_objects(scene, task_obj1, task_obj2, floor_obj,
                       room_objs: dict, seed: int) -> dict:
    """
    Place the two task objects on the floor.

    Strategy:
      1. Drop each obj via OnTop once to get floor Z and live half-extents.
      2. Sample both XY positions upfront using the scene graph only:
           - obj1: WALL_MARGIN from floor edges, OBJ_MARGIN from scene-graph objects
           - obj2: same, plus >= MIN_TASK_SEPARATION (0.35 m) centre-to-centre from obj1
      3. Teleport both objects to their sampled positions and settle physics.
    """
    SQUARE_ORI_T = th.tensor(SQUARE_ORI, dtype=th.float32)
    rng = random.Random(seed)

    SKIP_STRUCTURAL = {"ceilings", "walls", "floors", "carpet", "window",
                       "door", "curtain", "electric_switch"}

    # Floor bbox
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        fx_min = float(fb[0][0])
        fx_max = float(fb[1][0])
        fy_min = float(fb[0][1])
        fy_max = float(fb[1][1])
    else:
        fx_min, fx_max, fy_min, fy_max = -2.0, 2.0, -2.0, 2.0

    def _drop_and_get_half(obj):
        obj.states[object_states.OnTop].set_value(floor_obj, True)
        for _ in range(15):
            og.sim.step()
        pos, _ = obj.get_position_orientation()
        obj.set_position_orientation(position=pos, orientation=th.tensor(SQUARE_ORI, dtype=th.float32))
        obj.keep_still()
        for _ in range(10):
            og.sim.step()
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        hx = abs(float(bmax[0]) - float(bmin[0])) / 2.0
        hy = abs(float(bmax[1]) - float(bmin[1])) / 2.0
        z  = float(obj.get_position_orientation()[0].cpu().numpy()[2])
        return hx, hy, z

    def _teleport_and_settle(obj, xy, z):
        obj.set_position_orientation(
            position=th.tensor([xy[0], xy[1], z], dtype=th.float32),
            orientation=SQUARE_ORI_T,
        )
        obj.keep_still()
        for _ in range(20):
            og.sim.step()

    def _get_info(obj):
        pos  = obj.get_position_orientation()[0].cpu().numpy()
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        centre    = (bmin + bmax) / 2.0
        half_diag = float(np.sqrt(((bmax[0]-bmin[0])/2)**2 +
                                   ((bmax[1]-bmin[1])/2)**2))
        return pos, bmin, bmax, centre, half_diag

    # ── Step 1: get half-extents via physics drop ─────────────────────────────
    hx1, hy1, z1 = _drop_and_get_half(task_obj1)
    print(f"[task_obj1] live half_x={hx1:.4f}  half_y={hy1:.4f}  z={z1:.4f}")
    hx2, hy2, z2 = _drop_and_get_half(task_obj2)
    print(f"[task_obj2] live half_x={hx2:.4f}  half_y={hy2:.4f}  z={z2:.4f}")

    # ── Step 2: sample obj1 position ─────────────────────────────────────────
    # ── Step 2: sample both XY positions upfront ─────────────────────────────
    # Sample xy1, then xy2 >= MIN_TASK_SEPARATION from xy1.
    # If no valid xy2 exists for a given xy1, resample xy1 and try again.
    MAX_PAIR_ATTEMPTS = 50
    xy1, xy2 = None, None
    for attempt in range(MAX_PAIR_ATTEMPTS):
        xy1 = _find_clear_position(
            hx1, hy1, fx_min, fx_max, fy_min, fy_max,
            room_objs, extra_bboxes=[], skip_cats=SKIP_STRUCTURAL,
            rng=rng, label="task_obj1",
        )
        if xy1 is None:
            print(f"[task] pair attempt {attempt+1}: no valid xy1 — retrying")
            continue

        planned_bbox1 = [("task_obj1",
                          np.array([xy1[0]-hx1, xy1[1]-hy1, 0.0]),
                          np.array([xy1[0]+hx1, xy1[1]+hy1, 0.0]))]
        xy2 = _find_clear_position(
            hx2, hy2, fx_min, fx_max, fy_min, fy_max,
            room_objs, extra_bboxes=planned_bbox1, skip_cats=SKIP_STRUCTURAL,
            rng=rng, label="task_obj2",
        )
        if xy2 is None:
            print(f"[task] pair attempt {attempt+1}: no valid xy2 for xy1={xy1.round(3)} — resampling xy1")
            continue

        print(f"[task] pair found on attempt {attempt+1}: "
              f"xy1={xy1.round(3)}  xy2={xy2.round(3)}  "
              f"sep={np.hypot(xy1[0]-xy2[0], xy1[1]-xy2[1]):.3f}m")
        break
    else:
        raise RuntimeError(f"[task] failed to find valid xy1/xy2 pair after {MAX_PAIR_ATTEMPTS} attempts")

    # ── Step 4: teleport both and settle ──────────────────────────────────────
    _teleport_and_settle(task_obj1, xy1, z1)
    pos1, bmin1, bmax1, centre1, hd1 = _get_info(task_obj1)
    print(f"[task_obj1] settled at centre={centre1[:2].round(3)}")

    _teleport_and_settle(task_obj2, xy2, z2)
    pos2, bmin2, bmax2, centre2, hd2 = _get_info(task_obj2)
    print(f"[task_obj2] settled at centre={centre2[:2].round(3)}")

    sep = float(np.linalg.norm(centre1[:2] - centre2[:2]))
    print(f"[task] obj1 centre={centre1.round(3)}  half_diag={hd1:.3f}")
    print(f"[task] obj2 centre={centre2.round(3)}  half_diag={hd2:.3f}")
    print(f"[task] separation={sep:.3f} m")

    return {
        "task_obj1": {
            "position":   pos1.tolist(),
            "aabb_min":   bmin1.tolist(),
            "aabb_max":   bmax1.tolist(),
            "centre":     centre1.tolist(),
            "half_diag":  hd1,
        },
        "task_obj2": {
            "position":   pos2.tolist(),
            "aabb_min":   bmin2.tolist(),
            "aabb_max":   bmax2.tolist(),
            "centre":     centre2.tolist(),
            "half_diag":  hd2,
        },
        "separation": sep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render camera ring (12 side views + 1 top-down) for ONE task object
# ─────────────────────────────────────────────────────────────────────────────

def render_ring(output_dir: str, view_offset: int,
                task_centre: np.ndarray, task_half_diag: float,
                ref_centre_xy: np.ndarray,
                obj_key_to_name: dict) -> tuple:
    """
    Render 12 azimuth views (z=CAM_HEIGHT) + 1 top-down view for one task object.

    obj_key_to_name maps stable short keys → actual OmniGibson object names:
        {"task_obj1": "task_obj1", "task_obj2": "task_obj2",
         "ref_obj1": "<scene_native_name>", "ref_obj2": "<scene_native_name>"}

    Exist flags in the returned flat dict always use the SHORT keys:
        "exist_task_obj1_0", "exist_ref_obj1_0", ...
    covering all 4 objects × (12 side + 1 top-down) views in this ring.

    Views are numbered starting from `view_offset`.
    Returns (exist_flags_dict, camera_poses_dict).
    """
    obj_keys  = tuple(obj_key_to_name.keys())
    obj_names = tuple(obj_key_to_name.values())  # names queried in seg_instance

    # Starting azimuth toward the reference object
    delta = ref_centre_xy - task_centre[:2]
    start_az_rad = np.arctan2(delta[1], delta[0])

    camera_radius = task_half_diag + CAM_RADIUS_PAD
    cx, cy, cz = float(task_centre[0]), float(task_centre[1]), float(task_centre[2])
    look_target = np.array([cx, cy, cz])

    print(f"[camera] task_centre={look_target.round(3)}  radius={camera_radius:.3f}")

    camera_poses = {}
    exist_flags  = {}   # flat: "exist_{short_key}_{view_idx}" -> bool

    # ── Side views: 12 azimuths × 1 height ───────────────────────────────────
    for az_step in range(NUM_AZIMUTHS):
        azimuth_rad = start_az_rad + np.deg2rad(az_step * AZIMUTH_STEP)
        azimuth_deg = np.rad2deg(azimuth_rad) % 360
        view_idx    = view_offset + az_step

        eye = np.array([
            cx + camera_radius * np.cos(azimuth_rad),
            cy + camera_radius * np.sin(azimuth_rad),
            CAM_HEIGHT,
        ])

        fname = f"{view_idx}.png"
        fpath = os.path.join(output_dir, fname)
        print(f"\n[camera {view_idx}] azimuth={azimuth_deg:.1f}°  z={CAM_HEIGHT}  eye={eye.round(3)}")

        pose = _set_camera_and_capture(eye, look_target, fpath)
        camera_poses[fname] = {**pose, "azimuth_deg": azimuth_deg,
                               "height": CAM_HEIGHT, "type": "side"}

        # Seg check uses actual OmniGibson names; store under stable short keys
        vis_by_name = _visibility_check(obj_names)
        for key, name in obj_key_to_name.items():
            exist_flags[f"exist_{key}_{view_idx}"] = vis_by_name[name]
        print(f"[seg] { {k: exist_flags[f'exist_{k}_{view_idx}'] for k in obj_keys} }")

    # ── Top-down view ─────────────────────────────────────────────────────────
    topdown_idx = view_offset + NUM_AZIMUTHS   # view 12 or 25
    topdown_eye = np.array([cx, cy, CAM_TOPDOWN_Z])

    topdown_up   = np.array([1.0, 0.0, 0.0])
    forward_td   = np.array([0.0, 0.0, -1.0])
    right_td     = np.cross(forward_td, topdown_up)
    right_td    /= np.linalg.norm(right_td)
    true_up_td   = np.cross(right_td, forward_td)
    true_up_td  /= np.linalg.norm(true_up_td)
    rot_td       = np.column_stack([right_td, true_up_td, -forward_td])
    topdown_quat = Rotation.from_matrix(rot_td).as_quat()

    fname_td = f"{topdown_idx}.png"
    fpath_td = os.path.join(output_dir, fname_td)
    print(f"\n[camera {topdown_idx}] top-down  eye={topdown_eye.round(3)}")
    og.sim._viewer_camera.set_position_orientation(topdown_eye, topdown_quat)
    _capture(fpath_td)
    camera_poses[fname_td] = {
        "position":        topdown_eye.tolist(),
        "quaternion_xyzw": topdown_quat.tolist(),
        "azimuth_deg":     None,
        "height":          CAM_TOPDOWN_Z,
        "type":            "top_down",
    }
    vis_td_by_name = _visibility_check(obj_names)
    for key, name in obj_key_to_name.items():
        exist_flags[f"exist_{key}_{topdown_idx}"] = vis_td_by_name[name]
    print(f"[camera {topdown_idx}] visibility: "
          f"{ {k: exist_flags[f'exist_{k}_{topdown_idx}'] for k in obj_keys} }")

    return exist_flags, camera_poses



# ─────────────────────────────────────────────────────────────────────────────
# Cross-views: behind obj1 looking at obj2, and vice versa  (views 26–55)
# ─────────────────────────────────────────────────────────────────────────────

# Lateral tilt angles applied to each cross-view camera (degrees).
# For each (side, standoff) position the camera is rendered 3 times:
#   centre (0°), left (−15°), right (+15°) — all still looking at the same target.
CROSS_TILT_DEGS = [-8, 8, -15, 15]


def render_cross_views(output_dir: str,
                       centre1: np.ndarray, half_diag1: float,
                       centre2: np.ndarray, half_diag2: float,
                       obj_key_to_name: dict) -> tuple:
    """
    Render 30 cross-views (views 26–55) that show both task objects together.

    For each of the two "behind" sides (behind obj1 and behind obj2),
    5 stand-off distances × 3 lateral tilts = 15 views per side:

      views 26–40 — behind task_obj1 (axis centre2→centre1 extended):
        standoff 0.09 m: centre(26), left−15°(27), right+15°(28)
        standoff 0.15 m: centre(29), left−15°(30), right+15°(31)
        standoff 0.20 m: centre(32), left−15°(33), right+15°(34)
        standoff 0.25 m: centre(35), left−15°(36), right+15°(37)
        standoff 0.30 m: centre(38), left−15°(39), right+15°(40)

      views 41–55 — behind task_obj2 (axis centre1→centre2 extended):
        standoff 0.09 m: centre(41), left−15°(42), right+15°(43)
        standoff 0.15 m: centre(44), left−15°(45), right+15°(46)
        standoff 0.20 m: centre(47), left−15°(48), right+15°(49)
        standoff 0.25 m: centre(50), left−15°(51), right+15°(52)
        standoff 0.30 m: centre(53), left−15°(54), right+15°(55)

    The lateral tilt rotates the eye position in the XY plane around the
    look-at target by ±15°, keeping the eye-to-target distance constant and
    the look-at target unchanged.

    All cameras use CAM_HEIGHT (z=CAM_HEIGHT).
    Returns (exist_flags_dict, camera_poses_dict) for views 26–55.
    """
    obj_keys  = tuple(obj_key_to_name.keys())
    obj_names = tuple(obj_key_to_name.values())

    # Unit axis vectors between the two objects
    axis_2_to_1 = centre1[:2] - centre2[:2]
    dist_12 = float(np.linalg.norm(axis_2_to_1))
    if dist_12 < 1e-6:
        axis_2_to_1 = np.array([1.0, 0.0])
    else:
        axis_2_to_1 = axis_2_to_1 / dist_12
    axis_1_to_2 = -axis_2_to_1

    exist_flags  = {}
    camera_poses = {}

    # base_idx 26 → views 26–34  (behind obj1, looking at obj2)
    # base_idx 35 → views 35–43  (behind obj2, looking at obj1)
    sides = [
        # (base_idx, near_centre, near_half_diag, cam_axis, look_centre, side_label)
        (14, centre1, half_diag1, axis_2_to_1, centre2, "behind_obj1→obj2"),
        (30, centre2, half_diag2, axis_1_to_2, centre1, "behind_obj2→obj1"),
    ]

    for base_idx, near_centre, near_half_diag, cam_axis, look_centre, side_label in sides:
        view_idx = base_idx
        for standoff in CROSS_STANDOFFS:
            # Centre eye position for this standoff (before lateral tilt)
            eye_dist = near_half_diag + standoff
            eye_centre_xy = np.array([
                float(near_centre[0]) + cam_axis[0] * eye_dist,
                float(near_centre[1]) + cam_axis[1] * eye_dist,
            ])
            look_target = np.array([
                float(look_centre[0]),
                float(look_centre[1]),
                float(look_centre[2]),
            ])

            # Eye-to-target vector in XY, for computing orbit radius
            eye_to_target_xy = look_target[:2] - eye_centre_xy
            orbit_radius = float(np.linalg.norm(eye_to_target_xy))
            base_angle_rad = np.arctan2(eye_to_target_xy[1], eye_to_target_xy[0])

            for tilt_deg in CROSS_TILT_DEGS:
                tilt_rad   = np.deg2rad(tilt_deg)
                tilt_angle = base_angle_rad + tilt_rad

                # Rotate eye around look_target by tilt_deg in XY plane
                tilted_eye = np.array([
                    look_target[0] - orbit_radius * np.cos(tilt_angle),
                    look_target[1] - orbit_radius * np.sin(tilt_angle),
                    CAM_HEIGHT,
                ])

                tilt_label = (f"tilt{int(tilt_deg):+d}deg" if tilt_deg != 0.0
                              else "tilt+0deg")
                desc  = f"{view_idx}_{side_label}_standoff{standoff}m_{tilt_label}"
                fname = f"{view_idx}.png"
                fpath = os.path.join(output_dir, fname)

                print(f"\n[camera {view_idx}] cross-view {desc}  "
                      f"eye={tilted_eye.round(3)}  look_at={look_target.round(3)}")

                pose = _set_camera_and_capture(tilted_eye, look_target, fpath)
                camera_poses[fname] = {
                    **pose,
                    "azimuth_deg":  None,
                    "height":       CAM_HEIGHT,
                    "type":         "cross_view",
                    "description":  desc,
                    "standoff_m":   standoff,
                    "tilt_deg":     tilt_deg,
                }

                vis_by_name = _visibility_check(obj_names)
                for key, name in obj_key_to_name.items():
                    exist_flags[f"exist_{key}_{view_idx}"] = vis_by_name[name]
                print(f"[seg] { {k: exist_flags[f'exist_{k}_{view_idx}'] for k in obj_keys} }")

                view_idx += 1

    return exist_flags, camera_poses

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
    parser.add_argument("--output_root", type=str, default="renders_size")
    args = parser.parse_args()

    seed    = args.run_idx ^ (hash(args.scene + args.room) & 0xFFFFFFFF) + 300
    run_dir = os.path.join(args.output_root, args.scene, f"{args.room}_{args.run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Scene: {args.scene}  |  Room: {args.room}  |  Run: {args.run_idx}")
    print(f"  Floor: {args.floor}  |  Output: {run_dir}")
    print(f"  Task: size comparison (normal vs scaled-up copy of same object)")
    print(f"{'='*70}\n")

    # ── Load scene graph ───────────────────────────────────────────────────────
    scene_dict = load_scene_dict(args.scene)
    room_objs  = get_room_objects(scene_dict, args.room)

    SKIP_CATS = {"ceilings", "walls", "floors", "carpet", "window",
                 "door", "curtain", "electric_switch"}

    # Room centre from floor bbox
    floor_bboxes = room_objs.get("floors", room_objs.get("floor", []))
    if floor_bboxes:
        fb = floor_bboxes[0]
        room_centre_xy = np.array([
            (float(fb[0][0]) + float(fb[1][0])) / 2,
            (float(fb[0][1]) + float(fb[1][1])) / 2,
        ])
    else:
        all_centres = []
        for bboxes in room_objs.values():
            for (bmin, bmax) in bboxes:
                all_centres.append([(float(bmin[0]) + float(bmax[0])) / 2,
                                    (float(bmin[1]) + float(bmax[1])) / 2])
        room_centre_xy = np.mean(all_centres, axis=0) if all_centres else np.zeros(2)
    print(f"[room] centre_xy={room_centre_xy.round(3)}")

    # Build summaries for GPT (room objects as ref candidates)
    room_summaries = {}
    all_room_bboxes = [b for bbs in room_objs.values() for b in bbs]
    for cat, bboxes in room_objs.items():
        if any(s in cat.lower() for s in SKIP_CATS):
            continue
        other_bboxes = [b for c, bb in room_objs.items() for b in bb if c != cat]
        obj_centre_xy = np.array([
            (float(bboxes[0][0][0]) + float(bboxes[0][1][0])) / 2,
            (float(bboxes[0][0][1]) + float(bboxes[0][1][1])) / 2,
        ])
        room_summaries[cat] = {
            "footprint":      bbox_footprint(bboxes),
            "clearance":      bbox_clearance(bboxes, other_bboxes),
            "dist_to_centre": float(np.linalg.norm(obj_centre_xy - room_centre_xy)),
        }

    room_categories = list(room_summaries.keys())
    if not room_categories:
        print("[ERROR] No usable room categories found.")
        raise SystemExit(2)

    # ── Sample 200 candidate categories and ask GPT ────────────────────────────
    all_keys   = load_keys(args.keys_json)
    sampled    = sample_200(all_keys, seed=seed)
    gpt_result = gpt_pick_task_and_refs(sampled)

    task_category = gpt_result["task_cat"]
    ref1_category = gpt_result["ref1_cat"]
    ref2_category = gpt_result["ref2_cat"]

    # ── Sample scale for task_obj2 ─────────────────────────────────────────────
    rng_scale  = random.Random(seed + 7)
    scale2_val = rng_scale.uniform(SCALE_NORMAL_MIN, SCALE_NORMAL_MAX)
    scale2     = [scale2_val, scale2_val, scale2_val]
    scale1     = [1.0, 1.0, 1.0]
    print(f"[scale] task_obj1=1.0  task_obj2={scale2_val:.4f}")

    # ── Load models ────────────────────────────────────────────────────────────
    task_model = get_model_for_category(task_category, seed)
    ref1_model = get_model_for_category(ref1_category, seed + 2)
    ref2_model = get_model_for_category(ref2_category, seed + 3)
    # Both task copies share the same model; ref objects are independent.

    # ── Build OmniGibson config ────────────────────────────────────────────────
    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"]                = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    config["scene"]["load_room_instances"]        = [args.room]
    config["objects"] = [
        build_object_config("task_obj1", task_category, task_model, 0, scale1),
        build_object_config("task_obj2", task_category, task_model, 1, scale2),
        build_object_config("ref_obj1",  ref1_category, ref1_model,  2, [1.0, 1.0, 1.0]),
        build_object_config("ref_obj2",  ref2_category, ref2_model,  3, [1.0, 1.0, 1.0]),
    ]

    env   = og.Environment(configs=config)
    scene = env.scene

    task_obj1 = scene.object_registry("name", "task_obj1")
    task_obj2 = scene.object_registry("name", "task_obj2")
    floor_obj = scene.object_registry("name", args.floor)

    if floor_obj is None:
        print(f"[ERROR] Floor '{args.floor}' not found — exiting with code 2")
        raise SystemExit(2)

    # ── Look up reference objects by name (loaded from keys, not scene-native) ──
    ref_obj1 = scene.object_registry("name", "ref_obj1")
    ref_obj2 = scene.object_registry("name", "ref_obj2")

    if ref_obj1 is None or ref_obj2 is None:
        missing = ([ref1_category] if ref_obj1 is None else []) +                   ([ref2_category] if ref_obj2 is None else [])
        print(f"[ERROR] Reference object(s) failed to load: {missing}")
        raise SystemExit(2)

    print(f"[ref] ref_obj1 ({ref1_category}, model={ref1_model})")
    print(f"[ref] ref_obj2 ({ref2_category}, model={ref2_model})")

    # ── Place task objects ─────────────────────────────────────────────────────
    task_placement = place_task_objects(
        scene, task_obj1, task_obj2, floor_obj, room_objs, seed)

    task1_info   = task_placement["task_obj1"]
    task2_info   = task_placement["task_obj2"]
    task1_aabb_min = np.array(task1_info["aabb_min"])
    task1_aabb_max = np.array(task1_info["aabb_max"])
    task2_aabb_min = np.array(task2_info["aabb_min"])
    task2_aabb_max = np.array(task2_info["aabb_max"])
    centre1 = np.array(task1_info["centre"])
    centre2 = np.array(task2_info["centre"])

    # ── Place reference objects near their respective task objects ─────────────
    # extra_bboxes: live AABBs of the two task objects (freshly placed this
    # session — not present in the scenes5 scene graph).
    # Each entry: (descriptive_label, bmin_ndarray, bmax_ndarray).
    def _live_aabb(obj):
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        return bmin, bmax

    task1_bmin, task1_bmax = _live_aabb(task_obj1)
    task2_bmin, task2_bmax = _live_aabb(task_obj2)
    extra_bboxes_for_ref = [
        ("task_obj1", task1_bmin, task1_bmax),
        ("task_obj2", task2_bmin, task2_bmax),
    ]

    # Determine which side of task_obj1 faces task_obj2 → forbid that side
    # so the reference is never placed in the corridor between the two objects.
    def _facing_side(from_centre: np.ndarray, to_centre: np.ndarray) -> str:
        """Return the cardinal side of `from` that most faces `to`."""
        dx = float(to_centre[0]) - float(from_centre[0])
        dy = float(to_centre[1]) - float(from_centre[1])
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "N" if dy > 0 else "S"

    forbidden1 = {_facing_side(centre1, centre2)}
    forbidden2 = {_facing_side(centre2, centre1)}

    ref1_meta = place_ref_near_task(
        scene, ref_obj1, floor_obj,
        task_obj1, room_objs, extra_bboxes_for_ref,
        seed=seed + 10, label="1",
        forbidden_sides=forbidden1)

    # After placing ref_obj1, add its settled AABB to the extra list so that
    # ref_obj2's collision check also avoids ref_obj1.
    ref1_bmin, ref1_bmax = _live_aabb(ref_obj1)
    extra_bboxes_for_ref2 = extra_bboxes_for_ref + [("ref_obj1", ref1_bmin, ref1_bmax)]

    ref2_meta = place_ref_near_task(
        scene, ref_obj2, floor_obj,
        task_obj2, room_objs, extra_bboxes_for_ref2,
        seed=seed + 20, label="2",
        forbidden_sides=forbidden2)

    # ── Add segmentation modalities ────────────────────────────────────────────
    for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim._viewer_camera.add_modality(modality)
    for _ in range(100):
        og.sim.step()

    # Map stable short keys → actual scene object names for seg lookup.
    # Using short keys guarantees predictable flag names in the JSON regardless
    # of whatever scene-native name OmniGibson assigns to the reference objects.
    OBJ_KEY_TO_NAME = {
        "task_obj1": "task_obj1",
        "task_obj2": "task_obj2",
        "ref_obj1":  ref_obj1.name,
        "ref_obj2":  ref_obj2.name,
    }
    ALL_OBJ_KEYS  = tuple(OBJ_KEY_TO_NAME.keys())   # stable short keys for JSON
    ALL_OBJ_NAMES = tuple(OBJ_KEY_TO_NAME.values())  # actual names used by seg_instance

    os.makedirs(run_dir, exist_ok=True)

    # ── Re-read live AABBs after all placements are settled ────────────────────
    def _live_info(obj):
        bmin, bmax = [x.cpu().numpy() for x in obj.aabb]
        centre   = (bmin + bmax) / 2.0
        half_diag = float(np.sqrt(((bmax[0]-bmin[0])/2)**2 +
                                   ((bmax[1]-bmin[1])/2)**2))
        return bmin, bmax, centre, half_diag

    bmin1, bmax1, centre1, hd1 = _live_info(task_obj1)
    bmin2, bmax2, centre2, hd2 = _live_info(task_obj2)
    ref1_pos = ref_obj1.get_position_orientation()[0].cpu().numpy()
    ref2_pos = ref_obj2.get_position_orientation()[0].cpu().numpy()

    # ── Render views for task_obj1 (views 0–12) ────────────────────────────────
    print("\n" + "─"*50)
    print("  Rendering task_obj1 views (0–12)")
    print("─"*50)
    flags1, poses1 = render_ring(
        output_dir=run_dir,
        view_offset=0,
        task_centre=centre1,
        task_half_diag=hd1,
        ref_centre_xy=ref1_pos[:2],
        obj_key_to_name=OBJ_KEY_TO_NAME,
    )

    # ── Render views for task_obj2 (views 13–25) ──────────────────────────────
    print("\n" + "─"*50)
    print("  Rendering task_obj2 views (13–25)")
    print("─"*50)
    flags2, poses2 = render_ring(
        output_dir=run_dir,
        view_offset=7,
        task_centre=centre2,
        task_half_diag=hd2,
        ref_centre_xy=ref2_pos[:2],
        obj_key_to_name=OBJ_KEY_TO_NAME,
    )

    # ── Cross-views: view 26 (behind obj1→obj2) and view 27 (behind obj2→obj1) ──
    print("\n" + "─"*50)
    print("  Rendering cross-views (26–43)")
    print("─"*50)
    flags_cross, poses_cross = render_cross_views(
        output_dir=run_dir,
        centre1=centre1, half_diag1=hd1,
        centre2=centre2, half_diag2=hd2,
        obj_key_to_name=OBJ_KEY_TO_NAME,
    )

    # ── Merge visibility flags ─────────────────────────────────────────────────
    # flags1/flags2: all 4 objects × views 0–12 and 13–25.
    # flags_cross:   all 4 objects × views 26–55.
    # Together: all 4 objects × 44 views in one flat dict.
    all_exist = {**flags1, **flags2, **flags_cross}

    # ── Camera poses ───────────────────────────────────────────────────────────
    all_poses = {**poses1, **poses2, **poses_cross}

    # ── Object metadata ────────────────────────────────────────────────────────
    def obj_meta(obj, category, model, scale_val):
        pos, quat = obj.get_position_orientation()
        return {
            "category": category,
            "model":    model,
            "scale":    scale_val,
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    def ref_meta(obj, category, model):
        pos, quat = obj.get_position_orientation()
        return {
            "category": category,
            "model":    model,
            "scale":    1.0,
            "position":        pos.cpu().numpy().tolist(),
            "quaternion_xyzw": quat.cpu().numpy().tolist(),
        }

    metadata = {
        "scene":          args.scene,
        "room":           args.room,
        "run_idx":        args.run_idx,
        "seed":           seed,
        "floor_name":     args.floor,
        "layout":         "size_comparison",

        # Task object info
        "task_category":  task_category,
        "task_model":     task_model,
        "scale_obj1":     1.0,
        "scale_obj2":     scale2_val,

        # Reference object info
        "ref1_category":  ref1_category,
        "ref1_model":     ref1_model,
        "ref2_category":  ref2_category,
        "ref2_model":     ref2_model,

        # Ground truth
        "answer":         "task_obj2",   # task_obj2 is always bigger

        # Bounding boxes
        "task_obj1_aabb_min": bmin1.tolist(),
        "task_obj1_aabb_max": bmax1.tolist(),
        "task_obj1_centre":   centre1.tolist(),
        "task_obj2_aabb_min": bmin2.tolist(),
        "task_obj2_aabb_max": bmax2.tolist(),
        "task_obj2_centre":   centre2.tolist(),
        "separation":         task_placement["separation"],

        # Reference placement info
        "ref1_placement": ref1_meta,
        "ref2_placement": ref2_meta,

        # Visibility flags (exist_task_obj1_0 … exist_task_obj1_12, etc.)
        **all_exist,

        # Camera layout
        "camera_layout": {
            "num_azimuths":        NUM_AZIMUTHS,
            "azimuth_step_deg":    AZIMUTH_STEP,
            "side_height":         CAM_HEIGHT,
            "topdown_z":           CAM_TOPDOWN_Z,
            "radius_pad":          CAM_RADIUS_PAD,
            "start_azimuth":       "direction from task_obj centre to its ref object",
            "view_ordering": (
                "views 0–11: task_obj1 azimuths 0°–330°  "
                "view 12: task_obj1 top-down  "
                "views 13–24: task_obj2 azimuths 0°–330°  "
                "view 25: task_obj2 top-down  "
                "views 26–40: cross-views behind obj1 looking at obj2 "
                "(5 standoffs × 3 tilts: centre/−15°/+15°)  "
                "views 41–55: cross-views behind obj2 looking at obj1 "
                "(5 standoffs × 3 tilts: centre/−15°/+15°)"
            ),
            "cross_standoffs_m": CROSS_STANDOFFS,
            "cross_tilt_degs":   CROSS_TILT_DEGS,
        },

        "objects": {
            "task_obj1": obj_meta(task_obj1, task_category, task_model, 1.0),
            "task_obj2": obj_meta(task_obj2, task_category, task_model, scale2_val),
            "ref_obj1":  ref_meta(ref_obj1, ref1_category, ref1_model),
            "ref_obj2":  ref_meta(ref_obj2, ref2_category, ref2_model),
        },
        "camera_poses": all_poses,
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[meta] Saved -> {meta_path}")

    # ── Success check ─────────────────────────────────────────────────────────
    # Success if both task objects are visible in at least one of their own views.
    t1_visible = any(all_exist.get(f"exist_task_obj1_{k}", False) for k in range(56))
    t2_visible = any(all_exist.get(f"exist_task_obj2_{k}", False) for k in range(56))
    r1_visible = any(all_exist.get(f"exist_ref_obj1_{k}", False) for k in range(56))
    r2_visible = any(all_exist.get(f"exist_ref_obj2_{k}", False) for k in range(56))
    success = t1_visible and t2_visible
    print(f"\n[done] task_obj1={t1_visible}  task_obj2={t2_visible}  "
          f"ref_obj1={r1_visible}  ref_obj2={r2_visible}  success={success}")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()