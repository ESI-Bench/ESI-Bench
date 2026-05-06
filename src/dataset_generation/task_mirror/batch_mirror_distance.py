"""
Generate mirror-distance candidates from a runtime OmniGibson scene.

Pipeline:
1) Place one mirror and multiple real objects on the floor.
2) Compute mirror-image correspondences with a geometric mirror model.
3) Generate only mirror distance questions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import struct
import sys
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OG_ROOT = str(SCRIPT_DIR / "OmniGibson")
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

DEFAULT_OBJECT_INVENTORY = str(SCRIPT_DIR / "bddl3" / "bddl" / "generated_data" / "object_inventory.json")
DEFAULT_AVG_CATEGORY_SPECS = str(SCRIPT_DIR / "OmniGibson" / "omnigibson" / "configs" / "avg_category_specs.json")
DEFAULT_DATASET_OBJECTS_ROOT = SCRIPT_DIR / "datasets" / "behavior-1k-assets" / "objects"
DEFAULT_KEYS_JSON = str(SCRIPT_DIR / "keys.json")

DEFAULT_TARGET_RADIUS = 0.08
TARGET_CLEARANCE = 0.08
GRID_STEP = 0.35
CAMERA_HEIGHT = 0.8
CAMERA_FOV_DEG = 70.0
ROOM_REFERENCE_VIEW_COUNT = 4
GT_ORBIT_STEP_DEG = 5
MIRROR_AHEAD_DISTANCE = 2.0
MIRROR_TILT_MAX_DEG = 30.0
MIRROR_MIN_DEPTH = 0.55
MIRROR_MAX_DEPTH = 1.8
CORRESPONDENCE_MIRROR_WIDTH_SCALE = 1.5
MULTI_VIEW_ARC_MAX_DEG = 45
MULTI_VIEW_STEP_DEG = 5
GT_SIDE_OFFSET_M = 1.0

NON_PLACEABLE_CATEGORIES = {
    "background",
    "baseboard",
    "ceilings",
    "door",
    "floors",
    "roof",
    "stairs",
    "structural_element",
    "walls",
    "fixed_window",
    "openable_window",
    "window_blind",
    "room_light",
    "mirror",
    "standing_mirror",
    "makeup_mirror",
}

DEFAULT_OBJECT_CANDIDATE_FALLBACK = ("apple", "mug", "book", "bowl", "plate", "banana")
RENDER_PARK_X = 1000.0
RENDER_PARK_Y = 1000.0
RENDER_PARK_Z = 120.0
DEFAULT_TRAV_MAP_BASENAME = "floor_trav"
TASK_TYPES = ("mirror_distance",)


def _set_viewer_camera_fov(fov_deg: float = CAMERA_FOV_DEG) -> None:
    cam = og.sim.viewer_camera
    aperture_mm = float(cam.horizontal_aperture)
    target_fov_deg = float(fov_deg)
    focal_length_mm = aperture_mm / (2.0 * math.tan(math.radians(target_fov_deg) * 0.5))
    cam.focal_length = focal_length_mm
    print(
        f"[camera] horizontal FOV set to {target_fov_deg:.1f} deg "
        f"(aperture={aperture_mm:.3f} mm, focal_length={focal_length_mm:.3f} mm)",
        flush=True,
    )


@dataclass
class RuntimeObjectRecord:
    name: str
    category: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    in_rooms: tuple[str, ...]
    has_open_state: bool
    open_state: bool | None
    obj: object = field(repr=False)

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.bbox_min, self.bbox_max))

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.bbox_min, self.bbox_max))


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _humanize_object_label(value: str) -> str:
    return str(value).replace("_", " ")


def _collect_image_paths(payload) -> list[str]:
    image_paths: list[str] = []

    def _walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"image", "topdown_map", "image_path"} and isinstance(item, str):
                    image_paths.append(item)
                else:
                    _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    return image_paths


def _write_single_question_json(
    output_root: str,
    scene_metadata: dict,
    task_type: str,
    q_idx: int,
    entry: dict,
) -> str:
    task_dir = os.path.join(output_root, task_type)
    os.makedirs(task_dir, exist_ok=True)
    payload = {
        "scene": scene_metadata.get("scene"),
        "room": scene_metadata.get("room"),
        "floor_name": scene_metadata.get("floor_name"),
        "seed": scene_metadata.get("seed"),
        "camera_setup": scene_metadata.get("camera_setup"),
        "mirror_setup": scene_metadata.get("mirror_setup"),
        "task_type": task_type,
        "question_index": q_idx,
        "question_id": f"{task_type}/q_{q_idx:03d}",
        "question_data": entry,
    }
    payload["image_paths"] = _collect_image_paths(payload["question_data"])
    out_path = os.path.join(task_dir, f"q_{q_idx:03d}.json")
    _write_json(out_path, payload)
    return out_path


def _resolve_input_path(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    cwd_path = Path.cwd() / candidate
    if cwd_path.exists():
        return str(cwd_path)
    return str(SCRIPT_DIR / candidate)


def _normalize_bbox_xyxy(bbox_xyxy, name: str = "bbox", min_extent: float = 1e-3):
    arr = np.array(bbox_xyxy, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain 4 numbers: [xmin, ymin, xmax, ymax], got {bbox_xyxy}")
    xmin, ymin, xmax, ymax = [float(v) for v in arr.tolist()]
    if xmax - xmin < float(min_extent):
        pad = (float(min_extent) - (xmax - xmin)) * 0.5
        xmin -= pad
        xmax += pad
    if ymax - ymin < float(min_extent):
        pad = (float(min_extent) - (ymax - ymin)) * 0.5
        ymin -= pad
        ymax += pad
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def _room_bbox_center_xy(bbox_xyxy):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_mirror_distance] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _tensor_to_tuple3(tensor) -> tuple[float, float, float]:
    vals = tensor.cpu().tolist()
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def _distance_xy(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _norm_xy(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _point_inside_bbox_xy(point_xy, bbox_min, bbox_max, margin: float = 0.0) -> bool:
    return (
        bbox_min[0] - margin <= float(point_xy[0]) <= bbox_max[0] + margin
        and bbox_min[1] - margin <= float(point_xy[1]) <= bbox_max[1] + margin
    )


def _dot_xy(a, b) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _segment_bbox_overlap_xy(a_xy, b_xy, bbox_min, bbox_max, margin: float = 0.0):
    ax, ay = float(a_xy[0]), float(a_xy[1])
    bx, by = float(b_xy[0]), float(b_xy[1])
    dx = bx - ax
    dy = by - ay
    t0, t1 = 0.0, 1.0
    for origin, delta, lo, hi in (
        (ax, dx, float(bbox_min[0]) - margin, float(bbox_max[0]) + margin),
        (ay, dy, float(bbox_min[1]) - margin, float(bbox_max[1]) + margin),
    ):
        if abs(delta) < 1e-8:
            if origin < lo or origin > hi:
                return None
            continue
        inv = 1.0 / delta
        t_enter = (lo - origin) * inv
        t_exit = (hi - origin) * inv
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter
        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return None
    return (t0, t1)


def _segment_is_occluded_by_blockers(
    start_xy,
    start_z: float,
    end_xy,
    end_z: float,
    blockers: list[RuntimeObjectRecord],
    ignore_labels: set[str] | None = None,
    margin_xy: float = 0.02,
    margin_z: float = 0.02,
) -> bool:
    ignore_labels = ignore_labels or set()
    for blocker in blockers:
        if blocker.name in ignore_labels:
            continue
        overlap = _segment_bbox_overlap_xy(start_xy, end_xy, blocker.bbox_min, blocker.bbox_max, margin=margin_xy)
        if overlap is None:
            continue
        mid_t = min(max((float(overlap[0]) + float(overlap[1])) * 0.5, 0.0), 1.0)
        seg_z = float(start_z) + (float(end_z) - float(start_z)) * mid_t
        if float(blocker.bbox_min[2]) - margin_z <= seg_z <= float(blocker.bbox_max[2]) + margin_z:
            return True
    return False


def _trav_map_floor_image(scene, floor_idx: int = 0, scene_name: str | None = None, basename: str = DEFAULT_TRAV_MAP_BASENAME):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is not None and getattr(trav_map, "floor_map", None) is not None and 0 <= floor_idx < len(trav_map.floor_map):
        floor_img = trav_map.floor_map[floor_idx]
        if hasattr(floor_img, "detach"):
            floor_img = floor_img.detach()
        if getattr(floor_img, "device", None) is not None and floor_img.device.type != "cpu":
            floor_img = floor_img.cpu()
        return trav_map, np.array(floor_img)

    resolved_scene = scene_name or getattr(scene, "scene_model", None) or getattr(scene, "model", None)
    if not resolved_scene:
        return trav_map, None
    img_path = SCRIPT_DIR / "datasets" / "behavior-1k-assets" / "scenes" / str(resolved_scene) / "layout" / f"{basename}_{int(floor_idx)}.png"
    if not img_path.exists():
        return trav_map, None
    return trav_map, np.array(Image.open(img_path).convert("L"))


def _clip_map_rc(map_img: np.ndarray, rc) -> tuple[int, int]:
    row = int(np.clip(int(round(float(rc[0]))), 0, map_img.shape[0] - 1))
    col = int(np.clip(int(round(float(rc[1]))), 0, map_img.shape[1] - 1))
    return row, col


def _world_to_map_rc(trav_map, xy) -> tuple[int, int] | None:
    if trav_map is None:
        return None
    rc_arr = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    return int(round(float(rc_arr[0]))), int(round(float(rc_arr[1])))


def _segment_is_occluded_by_trav_map(
    start_xy,
    end_xy,
    trav_map,
    map_img: np.ndarray | None,
    endpoint_margin_px: int = 2,
) -> bool:
    if trav_map is None or map_img is None:
        return False
    start_rc = _world_to_map_rc(trav_map, start_xy)
    end_rc = _world_to_map_rc(trav_map, end_xy)
    if start_rc is None or end_rc is None:
        return False
    dr = float(end_rc[0] - start_rc[0])
    dc = float(end_rc[1] - start_rc[1])
    steps = max(int(math.ceil(max(abs(dr), abs(dc)) * 2.0)), 1)
    if steps <= endpoint_margin_px * 2:
        return False
    for idx in range(endpoint_margin_px, steps - endpoint_margin_px + 1):
        t = float(idx) / float(steps)
        rc = (start_rc[0] + dr * t, start_rc[1] + dc * t)
        row, col = _clip_map_rc(map_img, rc)
        if float(map_img[row, col]) <= 0.0:
            return True
    return False


def _get_scene_objects(scene):
    raw_objects = getattr(scene, "objects", [])
    if isinstance(raw_objects, dict):
        return list(raw_objects.values())
    return list(raw_objects)


def _should_keep_room_object(category: str, in_rooms: tuple[str, ...], room_name: str | None) -> bool:
    if room_name is None:
        return True
    if room_name in in_rooms:
        return True
    if category == "floors":
        return True
    if not in_rooms and category in {"walls", "ceilings", "door", "sliding_door"}:
        return True
    return False


def _collect_room_objects(scene, room_name: str | None) -> list[RuntimeObjectRecord]:
    robot_names = {robot.name for robot in getattr(scene, "robots", [])}
    room_objects = []
    for obj in _get_scene_objects(scene):
        if obj.name in robot_names:
            continue
        category = str(getattr(obj, "category", "object"))
        in_rooms = tuple(str(room) for room in (getattr(obj, "in_rooms", None) or []))
        if not _should_keep_room_object(category, in_rooms, room_name):
            continue
        try:
            bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
        except Exception as exc:
            _log_exception(f"Failed to read AABB for object {obj.name}", exc)
            continue
        has_open_state = False
        open_state = None
        states = getattr(obj, "states", None)
        if states and object_states.Open in states:
            has_open_state = True
            try:
                open_state = bool(states[object_states.Open].get_value())
            except Exception:
                open_state = None
        room_objects.append(
            RuntimeObjectRecord(
                name=obj.name,
                category=category,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                in_rooms=in_rooms,
                has_open_state=has_open_state,
                open_state=open_state,
                obj=obj,
            )
        )
    room_objects.sort(key=lambda record: (record.category, record.name))
    return room_objects


def _resolve_agent_position(env, agent_position):
    if agent_position is not None:
        return (float(agent_position[0]), float(agent_position[1]), float(agent_position[2]))
    robots = list(getattr(env, "robots", []))
    if robots:
        robot_pos, _ = robots[0].get_position_orientation()
        vals = robot_pos.cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    return (0.0, 0.0, 0.0)


def _floor_selection_sort_key(floor: RuntimeObjectRecord, agent_pos, room_name: str | None):
    room_match = 0 if room_name is not None and room_name in floor.in_rooms else 1
    xy_contains = 0 if (
        floor.bbox_min[0] <= agent_pos[0] <= floor.bbox_max[0]
        and floor.bbox_min[1] <= agent_pos[1] <= floor.bbox_max[1]
    ) else 1
    z_gap = abs(float(floor.bbox_max[2]) - float(agent_pos[2]))
    area = float(floor.extents[0]) * float(floor.extents[1])
    xy_gap = _distance_xy(floor.center, agent_pos)
    return (room_match, xy_contains, z_gap, xy_gap, -area, floor.name)


def _select_floor(room_objects: list[RuntimeObjectRecord], floor_name: str | None, agent_pos, room_name: str | None = None):
    floors = [obj for obj in room_objects if obj.category == "floors"]
    if floor_name is not None:
        for floor in floors:
            if floor.name == floor_name:
                return floor
        raise ValueError(f"Floor '{floor_name}' not found among loaded room objects.")
    if floors:
        floors.sort(key=lambda floor: _floor_selection_sort_key(floor, agent_pos, room_name))
        return floors[0]
    raise ValueError("No floor object found in loaded room.")


def _is_floor_blocker(record: RuntimeObjectRecord, floor_z: float) -> bool:
    category = record.category.lower()
    if category in NON_PLACEABLE_CATEGORIES:
        return False
    if "wall" in category or "door" in category or "window" in category:
        return False
    ext = record.extents
    if ext[0] * ext[1] < 0.03:
        return False
    if record.bbox_min[2] > floor_z + 0.75 and ext[0] * ext[1] < 0.25:
        return False
    return True


def _is_wall_occluder(record: RuntimeObjectRecord) -> bool:
    return "wall" in record.category.lower()


def _point_is_free(
    point_xy,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    clearance: float,
    ignore_labels: set[str] | None = None,
) -> bool:
    ignore_labels = ignore_labels or set()
    if not (
        floor_record.bbox_min[0] + clearance <= float(point_xy[0]) <= floor_record.bbox_max[0] - clearance
        and floor_record.bbox_min[1] + clearance <= float(point_xy[1]) <= floor_record.bbox_max[1] - clearance
    ):
        return False
    for blocker in blockers:
        if blocker.name in ignore_labels:
            continue
        if _point_inside_bbox_xy(point_xy, blocker.bbox_min, blocker.bbox_max, margin=clearance):
            return False
    return True


def _generate_free_positions(
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    agent_pos,
    count: int,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    clearance: float | None = None,
) -> list[list[float]]:
    clearance = target_radius + TARGET_CLEARANCE if clearance is None else clearance
    x_min = floor_record.bbox_min[0] + clearance
    x_max = floor_record.bbox_max[0] - clearance
    y_min = floor_record.bbox_min[1] + clearance
    y_max = floor_record.bbox_max[1] - clearance
    if x_min >= x_max or y_min >= y_max:
        return []

    candidates = []
    x = x_min
    while x <= x_max + 1e-6:
        y = y_min
        while y <= y_max + 1e-6:
            pos_xy = (float(x), float(y))
            dist_to_agent = _distance_xy(pos_xy, agent_pos)
            if dist_to_agent >= 0.6 and _point_is_free(pos_xy, floor_record, blockers, clearance=clearance):
                candidates.append((round(dist_to_agent, 6), [float(x), float(y), float(floor_record.bbox_max[2]) + target_radius]))
            y += GRID_STEP
        x += GRID_STEP
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return [pos for _, pos in candidates[:count]]


def _step_sim(steps: int = 10) -> None:
    for _ in range(max(steps, 0)):
        og.sim.step()


def _park_position(slot_idx: int) -> th.Tensor:
    return th.tensor(
        [RENDER_PARK_X + slot_idx * 2.0, RENDER_PARK_Y + slot_idx * 1.2, RENDER_PARK_Z],
        dtype=th.float32,
    )


def _park_object(obj, slot_idx: int) -> None:
    obj.set_position_orientation(
        position=_park_position(slot_idx),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    if hasattr(obj, "keep_still"):
        obj.keep_still()


def _set_object_pose(obj, position, orientation=None) -> None:
    orientation = [0.0, 0.0, 0.0, 1.0] if orientation is None else orientation
    obj.set_position_orientation(
        position=th.tensor([float(v) for v in position], dtype=th.float32),
        orientation=th.tensor([float(v) for v in orientation], dtype=th.float32),
    )
    if hasattr(obj, "keep_still"):
        obj.keep_still()


def _yaw_to_quaternion_xyzw(yaw: float) -> list[float]:
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _read_current_aabb(obj) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
    return bbox_min, bbox_max


def _safe_half_height(obj, default_half_height: float = 0.2) -> float:
    try:
        bbox_min, bbox_max = _read_current_aabb(obj)
        return max((bbox_max[2] - bbox_min[2]) / 2.0, 0.02)
    except Exception:
        return default_half_height


def _segmap_get(scene):
    return getattr(scene, "seg_map", None) or getattr(scene, "_seg_map", None)


def _segmap_room_bbox_from_maps(scene, room_ins_id):
    seg = _segmap_get(scene)
    if seg is None or not hasattr(seg, "room_ins_map"):
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}
    m = seg.room_ins_map
    if m is None:
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}
    if hasattr(m, "detach"):
        m = m.detach()
    if getattr(m, "device", None) is not None and m.device.type != "cpu":
        m = m.cpu()
    mask = m == int(room_ins_id)
    if not bool(mask.any().item()):
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}
    idx = mask.nonzero(as_tuple=False)
    rmin = int(idx[:, 0].min().item())
    rmax = int(idx[:, 0].max().item())
    cmin = int(idx[:, 1].min().item())
    cmax = int(idx[:, 1].max().item())
    corners_rc = th.tensor(
        [[float(rmin), float(cmin)], [float(rmin), float(cmax)], [float(rmax), float(cmin)], [float(rmax), float(cmax)]],
        dtype=th.float32,
    )
    corners_xy = seg.map_to_world(corners_rc).detach().cpu().numpy()
    xmin, ymin = corners_xy.min(axis=0)
    xmax, ymax = corners_xy.max(axis=0)
    return {"bbox_map_rc": (rmin, cmin, rmax, cmax), "bbox_world_xy": (float(xmin), float(ymin), float(xmax), float(ymax)), "pixel_count": int(idx.shape[0])}


def _resolve_room_bbox_world_xy(scene, room_name: str | None, floor_record: RuntimeObjectRecord):
    seg = _segmap_get(scene)
    candidate_room_names = []
    if room_name:
        candidate_room_names.append(str(room_name))
    candidate_room_names.extend(str(room) for room in (floor_record.in_rooms or []))
    deduped = []
    seen = set()
    for name in candidate_room_names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    if seg is not None and hasattr(seg, "room_ins_id_to_ins_name"):
        for room_id, room_instance in getattr(seg, "room_ins_id_to_ins_name", {}).items():
            room_instance = str(room_instance)
            if room_instance not in deduped:
                continue
            bbox_info = _segmap_room_bbox_from_maps(scene, int(room_id))
            if bbox_info.get("bbox_world_xy") is not None:
                return _normalize_bbox_xyxy(bbox_info["bbox_world_xy"]), room_instance
    floor_bbox = _normalize_bbox_xyxy((floor_record.bbox_min[0], floor_record.bbox_min[1], floor_record.bbox_max[0], floor_record.bbox_max[1]), name="floor_bbox")
    return floor_bbox, (deduped[0] if deduped else None)


def _load_available_categories(keys_json: str | None) -> set[str]:
    keys_path = _resolve_input_path(keys_json)
    if keys_path and os.path.exists(keys_path):
        raw = _read_json(keys_path)
        if isinstance(raw, list):
            return {str(item) for item in raw}
    if os.path.exists(DEFAULT_OBJECT_INVENTORY):
        inventory = _read_json(DEFAULT_OBJECT_INVENTORY)
        providers = inventory.get("providers", inventory)
        return {str(key).split("-", 1)[0] for key in providers}
    return set()


def _asset_exists_for_model(category: str, model: str) -> bool:
    asset_path = DEFAULT_DATASET_OBJECTS_ROOT / category / model / "usd" / f"{model}.encrypted.usd"
    return asset_path.exists()


def _load_category_models() -> dict[str, list[str]]:
    if not os.path.exists(DEFAULT_OBJECT_INVENTORY):
        return {}
    inventory = _read_json(DEFAULT_OBJECT_INVENTORY)
    providers = inventory.get("providers", inventory)
    category_models: dict[str, list[str]] = {}
    for provider_key in providers:
        category, _, model = str(provider_key).partition("-")
        if not category or not model:
            continue
        if not _asset_exists_for_model(category, model):
            continue
        category_models.setdefault(category, []).append(model)
    for category, models in category_models.items():
        category_models[category] = sorted(set(models))
    return category_models


def _load_avg_specs() -> dict:
    if os.path.exists(DEFAULT_AVG_CATEGORY_SPECS):
        return _read_json(DEFAULT_AVG_CATEGORY_SPECS)
    return {}


def _estimate_radius_from_volume(volume: float) -> float:
    if volume <= 0:
        return DEFAULT_TARGET_RADIUS
    return float(((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0))


def _build_placeable_category_pool(keys_json: str | None) -> list[dict]:
    available = _load_available_categories(keys_json)
    category_models = _load_category_models()
    specs = _load_avg_specs()
    categories = sorted(available) if available else sorted(category_models)
    pool = []
    for category in categories:
        if category in NON_PLACEABLE_CATEGORIES:
            continue
        models = category_models.get(category, [])
        if not models:
            continue
        volume = float(specs.get(category, {}).get("volume", 0.0) or 0.0)
        radius = _estimate_radius_from_volume(volume) if volume > 0 else DEFAULT_TARGET_RADIUS
        if not (0.02 <= radius <= 0.2):
            continue
        low = category.lower()
        if "wall" in low or "ceiling" in low or "floor" in low or "door" in low:
            continue
        pool.append({"category": category, "models": models, "estimated_radius": radius, "estimated_volume": volume})
    if pool:
        return pool

    fallback_pool = []
    for category in DEFAULT_OBJECT_CANDIDATE_FALLBACK:
        models = category_models.get(category, [])
        if models:
            fallback_pool.append({"category": category, "models": models, "estimated_radius": DEFAULT_TARGET_RADIUS, "estimated_volume": 0.0})
    return fallback_pool


def _reflect_xy(point_xy, mirror_point_xy, mirror_normal_xy):
    px, py = float(point_xy[0]), float(point_xy[1])
    mx, my = float(mirror_point_xy[0]), float(mirror_point_xy[1])
    nx, ny = float(mirror_normal_xy[0]), float(mirror_normal_xy[1])
    norm = math.hypot(nx, ny)
    if norm < 1e-8:
        raise ValueError("Mirror normal is near zero.")
    nx /= norm
    ny /= norm
    dx = px - mx
    dy = py - my
    signed_dist = dx * nx + dy * ny
    return [px - 2.0 * signed_dist * nx, py - 2.0 * signed_dist * ny]


def _rotate_xy(v_xy, angle_rad: float) -> list[float]:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return [float(v_xy[0]) * c - float(v_xy[1]) * s, float(v_xy[0]) * s + float(v_xy[1]) * c]


def _line_plane_intersection_xy(a_xy, b_xy, plane_point_xy, plane_normal_xy):
    ax, ay = float(a_xy[0]), float(a_xy[1])
    bx, by = float(b_xy[0]), float(b_xy[1])
    px, py = float(plane_point_xy[0]), float(plane_point_xy[1])
    nx, ny = float(plane_normal_xy[0]), float(plane_normal_xy[1])
    dx, dy = bx - ax, by - ay
    denom = dx * nx + dy * ny
    if abs(denom) < 1e-8:
        return None
    t = ((px - ax) * nx + (py - ay) * ny) / denom
    return [ax + t * dx, ay + t * dy], float(t)


def _angle_between_xy(a_xy, b_xy) -> float:
    na = _norm_xy(a_xy)
    nb = _norm_xy(b_xy)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    dot = max(-1.0, min(1.0, _dot_xy(a_xy, b_xy) / (na * nb)))
    return math.degrees(math.acos(dot))


def _is_point_in_camera_fov_xy(point_xy, camera_xy, camera_forward_xy, fov_deg: float) -> bool:
    ray = [float(point_xy[0]) - float(camera_xy[0]), float(point_xy[1]) - float(camera_xy[1])]
    if _dot_xy(ray, camera_forward_xy) <= 0.0:
        return False
    return _angle_between_xy(ray, camera_forward_xy) <= float(fov_deg) * 0.5

def _select_mirror_pose(
    free_positions,
    blockers,
    floor_record: RuntimeObjectRecord,
    anchor_xy,
    anchor_z: float,
    rng: random.Random,
):
    if not free_positions:
        raise RuntimeError("No free positions available to place the mirror.")
    anchor_xy = [float(anchor_xy[0]), float(anchor_xy[1])]
    floor_center = [float(floor_record.center[0]), float(floor_record.center[1])]
    candidates = []
    for pos in free_positions:
        xy = [float(pos[0]), float(pos[1])]
        if not _point_is_free(xy, floor_record, blockers, clearance=DEFAULT_TARGET_RADIUS + 0.06):
            continue
        if _segment_is_occluded_by_blockers(
            start_xy=anchor_xy,
            start_z=float(anchor_z),
            end_xy=xy,
            end_z=float(anchor_z),
            blockers=blockers,
        ):
            continue
        dist_to_anchor = _distance_xy(xy, anchor_xy)
        dist_to_center = _distance_xy(xy, floor_center)
        if dist_to_anchor < 0.45:
            continue
        score = abs(dist_to_anchor - MIRROR_AHEAD_DISTANCE) + 0.25 * dist_to_center
        candidates.append((score, xy))
    if not candidates:
        raise RuntimeError("Unable to place mirror before camera placement.")
    candidates.sort(key=lambda item: item[0])
    best_pool = candidates[: min(12, len(candidates))]
    mirror_xy = list(rng.choice(best_pool)[1])

    base_normal = [anchor_xy[0] - mirror_xy[0], anchor_xy[1] - mirror_xy[1]]
    norm = _norm_xy(base_normal)
    if norm < 1e-8:
        yaw = rng.uniform(-math.pi, math.pi)
        base_normal = [math.cos(yaw), math.sin(yaw)]
    else:
        base_normal = [base_normal[0] / norm, base_normal[1] / norm]
    delta = math.radians(rng.uniform(-MIRROR_TILT_MAX_DEG, MIRROR_TILT_MAX_DEG))
    mirror_normal = _rotate_xy(base_normal, delta)
    norm = _norm_xy(mirror_normal)
    mirror_normal = [mirror_normal[0] / norm, mirror_normal[1] / norm]
    mirror_tangent = [-mirror_normal[1], mirror_normal[0]]
    return mirror_xy, mirror_normal, mirror_tangent, math.degrees(delta)


def _select_camera_pose_for_mirror(
    free_positions,
    floor_record: RuntimeObjectRecord,
    blockers,
    wall_blockers,
    mirror_xy,
    mirror_normal,
    mirror_z: float,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
    preferred_xy=None,
):
    mirror_tangent = [-float(mirror_normal[1]), float(mirror_normal[0])]
    camera_z = float(floor_record.bbox_max[2]) + CAMERA_HEIGHT
    if preferred_xy is not None:
        camera_xy = [float(preferred_xy[0]), float(preferred_xy[1])]
        if _segment_is_occluded_by_blockers(
            start_xy=camera_xy,
            start_z=camera_z,
            end_xy=mirror_xy,
            end_z=float(mirror_z),
            blockers=blockers,
        ) or _segment_is_occluded_by_blockers(
            start_xy=camera_xy,
            start_z=camera_z,
            end_xy=mirror_xy,
            end_z=float(mirror_z),
            blockers=wall_blockers,
        ) or _segment_is_occluded_by_trav_map(
            start_xy=camera_xy,
            end_xy=mirror_xy,
            trav_map=trav_map,
            map_img=trav_map_img,
        ):
            raise RuntimeError("Preferred camera pose cannot see the mirror without scene-object occlusion.")
    else:
        best = None
        for pos in free_positions:
            xy = [float(pos[0]), float(pos[1])]
            rel = [xy[0] - float(mirror_xy[0]), xy[1] - float(mirror_xy[1])]
            ahead = _dot_xy(rel, mirror_normal)
            lateral = abs(_dot_xy(rel, mirror_tangent))
            if ahead <= 0.35:
                continue
            if _segment_is_occluded_by_blockers(
                start_xy=xy,
                start_z=camera_z,
                end_xy=mirror_xy,
                end_z=float(mirror_z),
                blockers=blockers,
            ) or _segment_is_occluded_by_blockers(
                start_xy=xy,
                start_z=camera_z,
                end_xy=mirror_xy,
                end_z=float(mirror_z),
                blockers=wall_blockers,
            ) or _segment_is_occluded_by_trav_map(
                start_xy=xy,
                end_xy=mirror_xy,
                trav_map=trav_map,
                map_img=trav_map_img,
            ):
                continue
            score = abs(ahead - MIRROR_AHEAD_DISTANCE) + lateral * 0.8
            if best is None or score < best[0]:
                best = (score, xy)
        if best is None:
            raise RuntimeError("Unable to place camera with a clear view of the mirror.")
        camera_xy = best[1]

    camera_forward = [float(mirror_xy[0]) - camera_xy[0], float(mirror_xy[1]) - camera_xy[1]]
    norm = _norm_xy(camera_forward)
    if norm < 1e-8:
        camera_forward = [-float(mirror_normal[0]), -float(mirror_normal[1])]
    else:
        camera_forward = [camera_forward[0] / norm, camera_forward[1] / norm]
    camera_pos = [float(camera_xy[0]), float(camera_xy[1]), float(floor_record.bbox_max[2]) + CAMERA_HEIGHT]
    camera_target = [float(mirror_xy[0]), float(mirror_xy[1]), float(camera_pos[2])]
    camera_quat = look_at_quaternion(camera_pos, camera_target)
    return camera_xy, camera_forward, camera_pos, camera_quat


def _is_reflection_visible(
    obj_xy,
    obj_z: float,
    camera_xy,
    camera_z: float,
    camera_forward,
    mirror_xy,
    mirror_z: float,
    mirror_normal,
    mirror_tangent,
    mirror_half_width: float,
    fov_deg: float,
    blockers,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
):
    # Object should be on the same side as camera relative to mirror plane.
    obj_side = _dot_xy([obj_xy[0] - mirror_xy[0], obj_xy[1] - mirror_xy[1]], mirror_normal)
    cam_side = _dot_xy([camera_xy[0] - mirror_xy[0], camera_xy[1] - mirror_xy[1]], mirror_normal)
    if obj_side <= 0.02 or cam_side <= 0.02:
        return None
    if not (MIRROR_MIN_DEPTH <= obj_side <= MIRROR_MAX_DEPTH):
        return None

    reflected = _reflect_xy(obj_xy, mirror_xy, mirror_normal)
    hit, t = _line_plane_intersection_xy(camera_xy, reflected, mirror_xy, mirror_normal) or (None, None)
    if hit is None or t is None or not (0.0 < t < 1.0):
        return None
    u = _dot_xy([hit[0] - mirror_xy[0], hit[1] - mirror_xy[1]], mirror_tangent)
    if abs(u) > mirror_half_width * 0.95:
        return None
    if not _is_point_in_camera_fov_xy(hit, camera_xy, camera_forward, fov_deg):
        return None
    if not _is_point_in_camera_fov_xy(mirror_xy, camera_xy, camera_forward, fov_deg):
        return None
    if _segment_is_occluded_by_blockers(
        start_xy=camera_xy,
        start_z=float(camera_z),
        end_xy=hit,
        end_z=float(mirror_z),
        blockers=blockers,
    ) or _segment_is_occluded_by_trav_map(
        start_xy=camera_xy,
        end_xy=hit,
        trav_map=trav_map,
        map_img=trav_map_img,
    ):
        return None
    if _segment_is_occluded_by_blockers(
        start_xy=hit,
        start_z=float(mirror_z),
        end_xy=obj_xy,
        end_z=float(obj_z),
        blockers=blockers,
    ) or _segment_is_occluded_by_trav_map(
        start_xy=hit,
        end_xy=obj_xy,
        trav_map=trav_map,
        map_img=trav_map_img,
    ):
        return None
    return {
        "mirror_hit_xy": [float(hit[0]), float(hit[1])],
        "mirror_u": float(u),
        "depth_from_mirror": float(obj_side),
        "reflected_xy": [float(reflected[0]), float(reflected[1])],
    }

def _collect_visible_slots(
    free_positions,
    mirror_xy,
    mirror_z: float,
    mirror_normal,
    mirror_tangent,
    mirror_half_width: float,
    camera_xy,
    camera_z: float,
    camera_forward,
    fov_deg: float,
    blockers,
    floor_z: float,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
):
    slots = []
    for pos in free_positions:
        xy = [float(pos[0]), float(pos[1])]
        if _distance_xy(xy, mirror_xy) < 0.35:
            continue
        vis = _is_reflection_visible(
            obj_xy=xy,
            obj_z=float(floor_z) + DEFAULT_TARGET_RADIUS,
            camera_xy=camera_xy,
            camera_z=float(camera_z),
            camera_forward=camera_forward,
            mirror_xy=mirror_xy,
            mirror_z=float(mirror_z),
            mirror_normal=mirror_normal,
            mirror_tangent=mirror_tangent,
            mirror_half_width=mirror_half_width,
            fov_deg=fov_deg,
            blockers=blockers,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )
        if vis is None:
            continue
        slots.append({"xy": xy, "visibility": vis})
    slots.sort(key=lambda s: (s["visibility"]["depth_from_mirror"], abs(s["visibility"]["mirror_u"])))
    return slots


def _fallback_visible_slots(
    mirror_xy,
    mirror_z: float,
    mirror_normal,
    mirror_tangent,
    floor_record: RuntimeObjectRecord,
    blockers,
    camera_xy,
    camera_z: float,
    camera_forward,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
):
    slots = []
    depths = [0.65, 0.9, 1.15, 1.4]
    laterals = [-0.45, 0.45, -0.2, 0.2, 0.0]
    for depth in depths:
        for lateral in laterals:
            xy = [
                float(mirror_xy[0]) + float(mirror_normal[0]) * depth + float(mirror_tangent[0]) * lateral,
                float(mirror_xy[1]) + float(mirror_normal[1]) * depth + float(mirror_tangent[1]) * lateral,
            ]
            in_floor = (
                floor_record.bbox_min[0] + 0.03 <= xy[0] <= floor_record.bbox_max[0] - 0.03
                and floor_record.bbox_min[1] + 0.03 <= xy[1] <= floor_record.bbox_max[1] - 0.03
            )
            if not in_floor:
                continue
            if not _point_is_free(xy, floor_record, blockers, clearance=0.04):
                # Relaxed fallback still allows slightly crowded placements.
                if not _point_is_free(xy, floor_record, blockers, clearance=0.0):
                    continue
            visibility = _is_reflection_visible(
                obj_xy=xy,
                obj_z=float(floor_record.bbox_max[2]) + DEFAULT_TARGET_RADIUS,
                camera_xy=camera_xy,
                camera_z=float(camera_z),
                camera_forward=camera_forward,
                mirror_xy=mirror_xy,
                mirror_z=float(mirror_z),
                mirror_normal=mirror_normal,
                mirror_tangent=mirror_tangent,
                mirror_half_width=10.0,
                fov_deg=max(CAMERA_FOV_DEG, 105.0),
                blockers=blockers,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )
            if visibility is None:
                continue
            slots.append({"xy": xy, "visibility": visibility})
    return slots


def _vec_sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _vec_cross(a, b):
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _vec_norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _vec_normalize(v):
    norm = _vec_norm(v)
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector.")
    return [float(x) / norm for x in v]


def _rotation_matrix_to_quaternion_xyzw(matrix):
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]


def look_at_quaternion(eye_pos, target_pos, up=(0.0, 0.0, 1.0)):
    forward = _vec_normalize(_vec_sub(target_pos, eye_pos))
    right = _vec_cross(forward, up)
    if _vec_norm(right) < 1e-6:
        right = _vec_cross(forward, (0.0, 1.0, 0.0))
    right = _vec_normalize(right)
    true_up = _vec_normalize(_vec_cross(right, forward))
    rot_matrix = [
        [right[0], true_up[0], -forward[0]],
        [right[1], true_up[1], -forward[1]],
        [right[2], true_up[2], -forward[2]],
    ]
    return _rotation_matrix_to_quaternion_xyzw(rot_matrix)


def _save_rgb_png(path: str, image) -> None:
    if isinstance(image, th.Tensor):
        image = image.detach().cpu().to(dtype=th.uint8)
        shape = tuple(image.shape)
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape HxWx3, got {shape}")
        height, width = shape[:2]
        raw = b"".join(b"\x00" + bytes(image[row].contiguous().view(-1).tolist()) for row in range(height))
    else:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape HxWx3, got {shape}")
        height, width = shape[:2]
        raw = b"".join(b"\x00" + bytes(image[row].reshape(-1).tolist()) for row in range(height))
    compressed = zlib.compress(raw, level=9)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png = [
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _chunk(b"IDAT", compressed),
        _chunk(b"IEND", b""),
    ]
    with open(path, "wb") as f:
        f.write(b"".join(png))


def _capture(path: str) -> None:
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].detach().cpu().to(dtype=th.uint8)[..., :3]
    _save_rgb_png(path, image)


def _save_topdown_map(
    path: str,
    floor_record: RuntimeObjectRecord,
    camera_xy,
    camera_forward,
    mirror_xy,
    mirror_tangent,
    mirror_half_width: float,
    placed_objects: list[dict],
    reflection_samples: list[dict],
) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        floor_x0, floor_x1 = floor_record.bbox_min[0], floor_record.bbox_max[0]
        floor_y0, floor_y1 = floor_record.bbox_min[1], floor_record.bbox_max[1]
        ax.plot([floor_x0, floor_x1, floor_x1, floor_x0, floor_x0], [floor_y0, floor_y0, floor_y1, floor_y1, floor_y0], "k-", lw=1.5, label="floor")

        cam_x, cam_y = float(camera_xy[0]), float(camera_xy[1])
        fwd_x, fwd_y = float(camera_forward[0]), float(camera_forward[1])
        ax.scatter([cam_x], [cam_y], c="tab:blue", s=80, marker="o", label="camera")
        ax.arrow(cam_x, cam_y, fwd_x * 0.4, fwd_y * 0.4, width=0.01, head_width=0.08, color="tab:blue")

        mx, my = float(mirror_xy[0]), float(mirror_xy[1])
        tx, ty = float(mirror_tangent[0]), float(mirror_tangent[1])
        p0 = [mx - tx * mirror_half_width, my - ty * mirror_half_width]
        p1 = [mx + tx * mirror_half_width, my + ty * mirror_half_width]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:red", lw=3.0, label="mirror")

        for item in placed_objects:
            px, py = float(item["real_position"][0]), float(item["real_position"][1])
            ax.scatter([px], [py], c="tab:green", s=50)
            ax.text(px + 0.03, py + 0.03, item["alias"], fontsize=9, color="tab:green")

        if reflection_samples:
            hits_x = [float(s["mirror_hit_xy"][0]) for s in reflection_samples]
            hits_y = [float(s["mirror_hit_xy"][1]) for s in reflection_samples]
            ax.scatter(hits_x, hits_y, c="tab:orange", s=22, marker="x", label="reflection_hits")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Mirror / Objects / Camera Top-Down")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return True
    except Exception as exc:
        _log_exception("save_topdown_map", exc)
        return False


def _gather_visible_slots_with_fallback(
    free_positions,
    mirror_xy,
    mirror_z: float,
    mirror_normal,
    mirror_tangent,
    mirror_half_width,
    camera_xy,
    camera_z: float,
    camera_forward,
    floor_record,
    blockers,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
):
    slots = _collect_visible_slots(
        free_positions=free_positions,
        mirror_xy=mirror_xy,
        mirror_z=mirror_z,
        mirror_normal=mirror_normal,
        mirror_tangent=mirror_tangent,
        mirror_half_width=mirror_half_width,
        camera_xy=camera_xy,
        camera_z=camera_z,
        camera_forward=camera_forward,
        fov_deg=CAMERA_FOV_DEG,
        blockers=blockers,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
        floor_z=float(floor_record.bbox_max[2]),
    )
    if len(slots) < 2:
        slots = _collect_visible_slots(
            free_positions=free_positions,
            mirror_xy=mirror_xy,
            mirror_z=mirror_z,
            mirror_normal=mirror_normal,
            mirror_tangent=mirror_tangent,
            mirror_half_width=mirror_half_width * 1.35,
            camera_xy=camera_xy,
            camera_z=camera_z,
            camera_forward=camera_forward,
            fov_deg=max(CAMERA_FOV_DEG, 105.0),
            blockers=blockers,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            floor_z=float(floor_record.bbox_max[2]),
        )
    if len(slots) < 2:
        slots = _fallback_visible_slots(
            mirror_xy=mirror_xy,
            mirror_z=mirror_z,
            mirror_normal=mirror_normal,
            mirror_tangent=mirror_tangent,
            floor_record=floor_record,
            blockers=blockers,
            camera_xy=camera_xy,
            camera_z=camera_z,
            camera_forward=camera_forward,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )
    return slots


def _sample_distinct_slots(slots, k: int, min_sep: float, rng: random.Random):
    shuffled = list(slots)
    rng.shuffle(shuffled)
    picked = []
    for slot in shuffled:
        xy = slot["xy"]
        if any(_distance_xy(xy, p["xy"]) < min_sep for p in picked):
            continue
        picked.append(slot)
        if len(picked) >= k:
            break
    return picked


def _distance3(a, b) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _sample_distinct_xy(points, k: int, min_sep: float, rng: random.Random):
    shuffled = list(points)
    rng.shuffle(shuffled)
    picked = []
    for xy in shuffled:
        if any(_distance_xy(xy, prev) < min_sep for prev in picked):
            continue
        picked.append(xy)
        if len(picked) >= k:
            break
    return picked


def _build_camera_pose(position, target):
    quat = look_at_quaternion(position, target)
    forward_xy = [float(target[0]) - float(position[0]), float(target[1]) - float(position[1])]
    norm = _norm_xy(forward_xy)
    if norm < 1e-8:
        forward_xy = [0.0, 1.0]
    else:
        forward_xy = [forward_xy[0] / norm, forward_xy[1] / norm]
    return {
        "position": [float(v) for v in position],
        "target": [float(v) for v in target],
        "forward_xy": [float(v) for v in forward_xy],
        "quaternion_xyzw": [float(v) for v in quat],
    }


def _build_render_view_groups(setup):
    mirror_center = [
        float(setup["mirror_pos_xy"][0]),
        float(setup["mirror_pos_xy"][1]),
        float(setup["camera_pos"][2]),
    ]
    observer_pos = [float(v) for v in setup["camera_pos"]]
    observer_rel_xy = [
        float(observer_pos[0]) - float(setup["mirror_pos_xy"][0]),
        float(observer_pos[1]) - float(setup["mirror_pos_xy"][1]),
    ]
    radius = max(_norm_xy(observer_rel_xy), 1e-6)
    base_angle = math.atan2(observer_rel_xy[1], observer_rel_xy[0])

    single_view = _build_camera_pose(observer_pos, mirror_center)

    multi_view = []
    for angle_deg in range(-MULTI_VIEW_ARC_MAX_DEG, MULTI_VIEW_ARC_MAX_DEG + 1, MULTI_VIEW_STEP_DEG):
        theta = base_angle + math.radians(float(angle_deg))
        position = [
            float(setup["mirror_pos_xy"][0]) + radius * math.cos(theta),
            float(setup["mirror_pos_xy"][1]) + radius * math.sin(theta),
            float(observer_pos[2]),
        ]
        pose = _build_camera_pose(position, mirror_center)
        pose["relative_angle_deg"] = float(angle_deg)
        multi_view.append(pose)

    observer_forward_xy = list(single_view["forward_xy"])
    perp_xy = [-float(observer_forward_xy[1]), float(observer_forward_xy[0])]
    gt_view = []
    for side_name, side_sign in (("left", -1.0), ("right", 1.0)):
        position = [
            float(observer_pos[0]) + side_sign * GT_SIDE_OFFSET_M * float(perp_xy[0]),
            float(observer_pos[1]) + side_sign * GT_SIDE_OFFSET_M * float(perp_xy[1]),
            float(observer_pos[2]),
        ]
        pose = _build_camera_pose(position, observer_pos)
        pose["side"] = side_name
        pose["offset_m"] = float(GT_SIDE_OFFSET_M)
        gt_view.append(pose)

    return {
        "single_view": single_view,
        "multi_view": multi_view,
        "gt_view": gt_view,
    }


def _generate_questions_with_per_question_placement(
    scene,
    rng: random.Random,
    mirror_obj,
    floor_record: RuntimeObjectRecord,
    free_positions,
    blockers,
    placeable_pool,
    max_q: int,
    render_root: str,
    enable_render: bool,
    trav_map=None,
    trav_map_img: np.ndarray | None = None,
    preferred_camera_xy=None,
    wall_blockers=None,
    task_types: set[str] | None = None,
    question_json_root: str | None = None,
    question_scene_metadata: dict | None = None,
    room_bbox_world_xy=None,
    room_instance_name: str | None = None,
):
    wall_blockers = [] if wall_blockers is None else list(wall_blockers)
    if task_types is None:
        task_types = set(TASK_TYPES)
    else:
        task_types = set(task_types)
    category_to_models = {entry["category"]: list(entry["models"]) for entry in placeable_pool}
    object_cache: dict[str, list] = {}
    all_render_objects = []

    def _acquire_object(category: str, used_names: set[str], preferred_model: str | None = None):
        models = category_to_models.get(category, [])
        if not models:
            raise RuntimeError(f"No models for category '{category}'")
        model = str(preferred_model) if preferred_model is not None else rng.choice(models)
        if model not in models:
            raise RuntimeError(f"Model '{model}' is not available for category '{category}'")
        key = f"{category}:{model}"
        for obj in object_cache.get(key, []):
            if obj.name not in used_names:
                used_names.add(obj.name)
                return obj, model
        obj = DatasetObject(
            name=f"render_item_{len(all_render_objects):03d}",
            category=category,
            model=model,
            visual_only=True,
        )
        scene.add_object(obj)
        try:
            obj.visual_only = True
        except Exception:
            pass
        object_cache.setdefault(key, []).append(obj)
        all_render_objects.append(obj)
        used_names.add(obj.name)
        _park_object(obj, len(all_render_objects))
        _step_sim(5)
        return obj, model

    def _park_all():
        for idx, obj in enumerate(all_render_objects):
            _park_object(obj, idx)
        if all_render_objects:
            _step_sim(5)

    def _reset_scene_for_question():
        _park_all()
        _park_object(mirror_obj, len(all_render_objects) + 50)
        _step_sim(5)
        anchor_xy = preferred_camera_xy
        if anchor_xy is None:
            anchor_xy = [float(floor_record.center[0]), float(floor_record.center[1])]
        mirror_pos_xy, mirror_normal, mirror_tangent, mirror_delta_deg = _select_mirror_pose(
            free_positions=free_positions,
            blockers=blockers,
            floor_record=floor_record,
            anchor_xy=anchor_xy,
            anchor_z=float(floor_record.bbox_max[2]) + CAMERA_HEIGHT,
            rng=rng,
        )
        mirror_yaw = math.atan2(mirror_normal[1], mirror_normal[0])
        mirror_quat = _yaw_to_quaternion_xyzw(mirror_yaw)
        mirror_half_height = _safe_half_height(mirror_obj, default_half_height=0.75)
        mirror_pos = [mirror_pos_xy[0], mirror_pos_xy[1], float(floor_record.bbox_max[2]) + mirror_half_height + 0.01]
        _set_object_pose(mirror_obj, mirror_pos, mirror_quat)
        _step_sim(20)
        mirror_bbox_min, mirror_bbox_max = _read_current_aabb(mirror_obj)
        mirror_half_width = max(0.2, min(1.2, max(mirror_bbox_max[0] - mirror_bbox_min[0], mirror_bbox_max[1] - mirror_bbox_min[1]) * 0.5))
        camera_xy, camera_forward, camera_pos, camera_quat = _select_camera_pose_for_mirror(
            free_positions=free_positions,
            floor_record=floor_record,
            blockers=blockers,
            wall_blockers=wall_blockers,
            mirror_xy=mirror_pos_xy,
            mirror_normal=mirror_normal,
            mirror_z=float(mirror_pos[2]),
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            preferred_xy=preferred_camera_xy,
        )
        og.sim._viewer_camera.set_position_orientation(
            th.tensor(camera_pos, dtype=th.float32),
            th.tensor(camera_quat, dtype=th.float32),
        )
        _step_sim(5)
        return {
            "mirror_pos_xy": mirror_pos_xy,
            "mirror_normal": mirror_normal,
            "mirror_tangent": mirror_tangent,
            "mirror_delta_deg": float(mirror_delta_deg),
            "mirror_quat": [float(v) for v in mirror_quat],
            "mirror_pos": [float(v) for v in mirror_pos],
            "mirror_half_width": float(mirror_half_width),
            "camera_xy": [float(camera_xy[0]), float(camera_xy[1])],
            "camera_forward": [float(camera_forward[0]), float(camera_forward[1])],
            "camera_pos": [float(v) for v in camera_pos],
            "camera_quat": [float(v) for v in camera_quat],
        }

    def _validate_slot_visibility(slot_xy, obj_pos, setup):
        return _is_reflection_visible(
            obj_xy=[float(slot_xy[0]), float(slot_xy[1])],
            obj_z=float(obj_pos[2]),
            camera_xy=setup["camera_xy"],
            camera_z=float(setup["camera_pos"][2]),
            camera_forward=setup["camera_forward"],
            mirror_xy=setup["mirror_pos_xy"],
            mirror_z=float(setup["mirror_pos"][2]),
            mirror_normal=setup["mirror_normal"],
            mirror_tangent=setup["mirror_tangent"],
            mirror_half_width=float(setup["mirror_half_width"]),
            fov_deg=CAMERA_FOV_DEG,
            blockers=blockers,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )

    def _validate_correspondence_slot_visibility(slot_xy, obj_pos, setup):
        return _is_reflection_visible(
            obj_xy=[float(slot_xy[0]), float(slot_xy[1])],
            obj_z=float(obj_pos[2]),
            camera_xy=setup["camera_xy"],
            camera_z=float(setup["camera_pos"][2]),
            camera_forward=setup["camera_forward"],
            mirror_xy=setup["mirror_pos_xy"],
            mirror_z=float(setup["mirror_pos"][2]),
            mirror_normal=setup["mirror_normal"],
            mirror_tangent=setup["mirror_tangent"],
            mirror_half_width=float(setup["mirror_half_width"]) * CORRESPONDENCE_MIRROR_WIDTH_SCALE,
            fov_deg=CAMERA_FOV_DEG,
            blockers=blockers,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )

    def _render_question(task_type: str, q_idx: int, placed, reflection_samples, setup):
        render = None
        if not enable_render:
            return render
        task_dir = os.path.join(render_root, task_type)
        os.makedirs(task_dir, exist_ok=True)
        img_path = os.path.join(task_dir, f"q_{q_idx:03d}.png")
        map_path = os.path.join(task_dir, f"q_{q_idx:03d}_map.png")
        og.sim._viewer_camera.set_position_orientation(
            th.tensor(setup["camera_pos"], dtype=th.float32),
            th.tensor(setup["camera_quat"], dtype=th.float32),
        )
        _step_sim(5)
        _capture(img_path)

        map_ok = _save_topdown_map(
            path=map_path,
            floor_record=floor_record,
            camera_xy=setup["camera_xy"],
            camera_forward=setup["camera_forward"],
            mirror_xy=setup["mirror_pos_xy"],
            mirror_tangent=setup["mirror_tangent"],
            mirror_half_width=setup["mirror_half_width"],
            placed_objects=placed,
            reflection_samples=reflection_samples,
        )
        render = {
            "image": img_path,
            "camera_pose": {"position": list(setup["camera_pos"]), "quaternion_xyzw": list(setup["camera_quat"])},
            "mirror_pose": {
                "position": list(setup["mirror_pos"]),
                "quaternion_xyzw": list(setup["mirror_quat"]),
                "normal_xy": list(setup["mirror_normal"]),
                "tangent_xy": list(setup["mirror_tangent"]),
            },
            "topdown_map": map_path if map_ok else None,
        }
        return render

    def _capture_camera_view(image_path: str, eye, target):
        quat = look_at_quaternion(eye, target)
        og.sim._viewer_camera.set_position_orientation(
            th.tensor([float(v) for v in eye], dtype=th.float32),
            th.tensor([float(v) for v in quat], dtype=th.float32),
        )
        _step_sim(5)
        _capture(image_path)
        return {
            "image_path": image_path,
            "camera_pose": {"position": [float(v) for v in eye], "quaternion_xyzw": [float(v) for v in quat]},
            "target": [float(v) for v in target],
        }

    def _orbit_eye_is_valid(eye_xy) -> bool:
        floor_bbox = (float(floor_record.bbox_min[0]), float(floor_record.bbox_min[1]), float(floor_record.bbox_max[0]), float(floor_record.bbox_max[1]))
        if not _point_inside_bbox_xy(eye_xy, floor_bbox[:2], floor_bbox[2:], margin=-0.05):
            return False
        if room_bbox_world_xy is not None and not _point_inside_bbox_xy(eye_xy, room_bbox_world_xy[:2], room_bbox_world_xy[2:], margin=-0.05):
            return False
        if trav_map is not None and trav_map_img is not None:
            rc = _world_to_map_rc(trav_map, eye_xy)
            if rc is None:
                return False
            row, col = _clip_map_rc(trav_map_img, rc)
            if float(trav_map_img[row, col]) <= 0.0:
                return False
        return True

    def _render_room_corner_views(task_type: str, q_idx: int, setup):
        if not enable_render:
            return None
        bbox = room_bbox_world_xy or (
            float(floor_record.bbox_min[0]),
            float(floor_record.bbox_min[1]),
            float(floor_record.bbox_max[0]),
            float(floor_record.bbox_max[1]),
        )
        xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox, name="room_bbox")
        center_xy = _room_bbox_center_xy((xmin, ymin, xmax, ymax))
        width = xmax - xmin
        height = ymax - ymin
        inset = min(max(0.35, min(width, height) * 0.08), max(min(width, height) * 0.4, 0.35))
        task_dir = os.path.join(render_root, task_type, f"q_{q_idx:03d}_room_views")
        os.makedirs(task_dir, exist_ok=True)
        views = []
        for view_id, xy in [
            ("corner_00", [xmin + inset, ymin + inset]),
            ("corner_01", [xmin + inset, ymax - inset]),
            ("corner_02", [xmax - inset, ymin + inset]),
            ("corner_03", [xmax - inset, ymax - inset]),
        ][:ROOM_REFERENCE_VIEW_COUNT]:
            eye = [float(np.clip(float(xy[0]), float(floor_record.bbox_min[0]) + 0.2, float(floor_record.bbox_max[0]) - 0.2)),
                   float(np.clip(float(xy[1]), float(floor_record.bbox_min[1]) + 0.2, float(floor_record.bbox_max[1]) - 0.2)),
                   float(floor_record.bbox_max[2]) + 1.35]
            if _segment_is_occluded_by_blockers(
                start_xy=eye[:2],
                start_z=float(eye[2]),
                end_xy=setup["mirror_pos_xy"],
                end_z=float(setup["mirror_pos"][2]),
                blockers=wall_blockers,
            ):
                continue
            target = [float(center_xy[0]), float(center_xy[1]), float(floor_record.bbox_max[2]) + 0.9]
            view = _capture_camera_view(os.path.join(task_dir, f"{view_id}.png"), eye=eye, target=target)
            view["view_id"] = view_id
            views.append(view)
        return {"room_instance": room_instance_name, "room_bbox_world_xy": [float(xmin), float(ymin), float(xmax), float(ymax)], "views": views}

    def _gt_orbit_angle_offsets_deg(answer_value: str) -> list[int]:
        ans = str(answer_value).lower()
        if ans == "middle":
            return list(range(-45, 46, GT_ORBIT_STEP_DEG))
        if ans == "left":
            return list(range(0, 91, GT_ORBIT_STEP_DEG))
        if ans == "right":
            return list(range(-90, 1, GT_ORBIT_STEP_DEG))
        return [0]

    def _render_gt_orbit_views(task_type: str, q_idx: int, setup, answer_value: str):
        if not enable_render:
            return None
        orbit_dir = [float(setup["camera_xy"][0]) - float(setup["mirror_pos_xy"][0]), float(setup["camera_xy"][1]) - float(setup["mirror_pos_xy"][1])]
        orbit_radius = max(_norm_xy(orbit_dir), 0.8)
        base_angle = math.atan2(float(orbit_dir[1]), float(orbit_dir[0]))
        task_dir = os.path.join(render_root, task_type, f"q_{q_idx:03d}_gt_orbit")
        os.makedirs(task_dir, exist_ok=True)
        views = []
        for offset_deg in _gt_orbit_angle_offsets_deg(answer_value):
            angle = base_angle + math.radians(float(offset_deg))
            eye = [float(setup["mirror_pos_xy"][0]) + orbit_radius * math.cos(angle),
                   float(setup["mirror_pos_xy"][1]) + orbit_radius * math.sin(angle),
                   float(setup["camera_pos"][2])]
            if not _orbit_eye_is_valid(eye[:2]):
                continue
            if _segment_is_occluded_by_blockers(
                start_xy=eye[:2],
                start_z=float(eye[2]),
                end_xy=setup["mirror_pos_xy"],
                end_z=float(setup["mirror_pos"][2]),
                blockers=wall_blockers,
            ):
                continue
            target = [float(setup["mirror_pos"][0]), float(setup["mirror_pos"][1]), float(setup["mirror_pos"][2])]
            view = _capture_camera_view(os.path.join(task_dir, f"orbit_{offset_deg:+03d}.png"), eye=eye, target=target)
            view["offset_deg"] = int(offset_deg)
            views.append(view)
        return {"anchor_answer_label": str(answer_value), "orbit_center": [float(v) for v in setup["mirror_pos"]], "orbit_radius_xy_m": float(orbit_radius), "views": views}

    def _render_context_without_mirror(task_type: str, q_idx: int, placed, setup):
        if not enable_render:
            return None
        context_view = _find_context_view_for_correspondence(placed, setup)
        if context_view is None:
            return None
        task_dir = os.path.join(render_root, task_type)
        os.makedirs(task_dir, exist_ok=True)
        img_path = os.path.join(task_dir, f"q_{q_idx:03d}_context.png")

        parked_mirror_pos = _park_position(len(all_render_objects) + 80).tolist()
        _park_object(mirror_obj, len(all_render_objects) + 80)
        _step_sim(5)
        og.sim._viewer_camera.set_position_orientation(
            th.tensor(context_view["eye"], dtype=th.float32),
            th.tensor(context_view["quat"], dtype=th.float32),
        )
        _step_sim(5)
        _capture(img_path)
        _set_object_pose(mirror_obj, setup["mirror_pos"], setup["mirror_quat"])
        _step_sim(5)
        return {
            "image": img_path,
            "camera_pose": {"position": context_view["eye"], "quaternion_xyzw": context_view["quat"]},
            "target": context_view["target"],
            "mirror_removed": True,
            "parked_mirror_position": parked_mirror_pos,
            "all_objects_visible": True,
        }

    def _scene_setup_payload(setup):
        render_views = _build_render_view_groups(setup)
        return {
            "camera": {
                "position": list(setup["camera_pos"]),
                "forward_xy": list(setup["camera_forward"]),
                "quaternion_xyzw": list(setup["camera_quat"]),
            },
            "render_views": render_views,
            "mirror": {
                "position": list(setup["mirror_pos"]),
                "quaternion_xyzw": list(setup["mirror_quat"]),
                "normal_xy": list(setup["mirror_normal"]),
                "tangent_xy": list(setup["mirror_tangent"]),
                "half_width_m": float(setup["mirror_half_width"]),
                "normal_yaw_offset_deg": float(setup["mirror_delta_deg"]),
            },
        }

    def _is_xy_in_floor(xy):
        return (
            float(floor_record.bbox_min[0]) + 0.03 <= float(xy[0]) <= float(floor_record.bbox_max[0]) - 0.03
            and float(floor_record.bbox_min[1]) + 0.03 <= float(xy[1]) <= float(floor_record.bbox_max[1]) - 0.03
        )

    def _try_place_correspondence_line(setup, visible_slot, target_rank: int):
        anchor_xy = [float(visible_slot["xy"][0]), float(visible_slot["xy"][1])]
        depth = float(
            _dot_xy(
                [anchor_xy[0] - setup["mirror_pos_xy"][0], anchor_xy[1] - setup["mirror_pos_xy"][1]],
                setup["mirror_normal"],
            )
        )
        anchor_u = float(
            _dot_xy(
                [anchor_xy[0] - setup["mirror_pos_xy"][0], anchor_xy[1] - setup["mirror_pos_xy"][1]],
                setup["mirror_tangent"],
            )
        )

        def _xy_from_u(u_val: float):
            return [
                float(setup["mirror_pos_xy"][0]) + float(setup["mirror_normal"][0]) * depth + float(setup["mirror_tangent"][0]) * u_val,
                float(setup["mirror_pos_xy"][1]) + float(setup["mirror_normal"][1]) * depth + float(setup["mirror_tangent"][1]) * u_val,
            ]

        u_offsets = [i * 0.12 for i in range(-16, 17)]
        samples = []
        for du in u_offsets:
            u_val = anchor_u + float(du)
            xy = _xy_from_u(u_val)
            if not _is_xy_in_floor(xy):
                continue
            if not _point_is_free(xy, floor_record, blockers, clearance=0.0):
                continue
            vis = _is_reflection_visible(
                obj_xy=xy,
                obj_z=float(floor_record.bbox_max[2]) + DEFAULT_TARGET_RADIUS,
                camera_xy=setup["camera_xy"],
                camera_z=float(setup["camera_pos"][2]),
                camera_forward=setup["camera_forward"],
                mirror_xy=setup["mirror_pos_xy"],
                mirror_z=float(setup["mirror_pos"][2]),
                mirror_normal=setup["mirror_normal"],
                mirror_tangent=setup["mirror_tangent"],
                mirror_half_width=float(setup["mirror_half_width"]) * CORRESPONDENCE_MIRROR_WIDTH_SCALE,
                fov_deg=CAMERA_FOV_DEG,
                blockers=blockers,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )
            samples.append({"u": u_val, "xy": xy, "visible": vis is not None})

        visible_samples = [s for s in samples if s["visible"]]
        hidden_samples = [s for s in samples if not s["visible"]]
        if not visible_samples or len(hidden_samples) < 2:
            debug_line = [s["xy"] for s in samples[:3]]
            return {"chosen_xys": None, "debug_line_xys": debug_line}

        min_visible_u = min(s["u"] for s in visible_samples)
        max_visible_u = max(s["u"] for s in visible_samples)
        left_hidden = [s for s in hidden_samples if s["u"] < min_visible_u - 0.06]
        right_hidden = [s for s in hidden_samples if s["u"] > max_visible_u + 0.06]
        visible_sorted = sorted(visible_samples, key=lambda s: s["u"])
        left_hidden = sorted(left_hidden, key=lambda s: s["u"])
        right_hidden = sorted(right_hidden, key=lambda s: s["u"])

        if target_rank == 0:
            if len(right_hidden) < 2:
                debug_line = [visible_sorted[0]["xy"]] + [s["xy"] for s in right_hidden[:2]]
                return {"chosen_xys": None, "debug_line_xys": debug_line}
            trio = [visible_sorted[0], right_hidden[0], right_hidden[-1]]
        elif target_rank == 1:
            if not left_hidden or not right_hidden:
                debug_line = [left_hidden[-1]["xy"]] if left_hidden else []
                debug_line += [visible_sorted[len(visible_sorted) // 2]["xy"]]
                debug_line += [right_hidden[0]["xy"]] if right_hidden else []
                return {"chosen_xys": None, "debug_line_xys": debug_line}
            trio = [left_hidden[-1], visible_sorted[len(visible_sorted) // 2], right_hidden[0]]
        else:
            if len(left_hidden) < 2:
                debug_line = [s["xy"] for s in left_hidden[-2:]] + [visible_sorted[-1]["xy"]]
                return {"chosen_xys": None, "debug_line_xys": debug_line}
            trio = [left_hidden[0], left_hidden[-1], visible_sorted[-1]]

        chosen_xys = [item["xy"] for item in sorted(trio, key=lambda s: s["u"])]
        if any(_distance_xy(chosen_xys[a], chosen_xys[b]) < 0.18 for a in range(3) for b in range(a + 1, 3)):
            return {"chosen_xys": None, "debug_line_xys": chosen_xys}
        return {"chosen_xys": chosen_xys, "debug_line_xys": chosen_xys}

    def _find_context_view_for_correspondence(placed, setup):
        center = [
            sum(float(item["real_position"][0]) for item in placed) / max(len(placed), 1),
            sum(float(item["real_position"][1]) for item in placed) / max(len(placed), 1),
            sum(float(item["real_position"][2]) for item in placed) / max(len(placed), 1),
        ]
        target = [float(center[0]), float(center[1]), float(center[2]) + 0.03]
        depth_candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
        height_offsets = [0.18, 0.22, 0.3, 0.38, 0.46]
        fov_deg = max(CAMERA_FOV_DEG, 110.0)
        for depth in depth_candidates:
            eye_xy = [
                float(center[0]) + float(setup["mirror_normal"][0]) * depth,
                float(center[1]) + float(setup["mirror_normal"][1]) * depth,
            ]
            if not _is_xy_in_floor(eye_xy):
                continue
            if not _point_is_free(eye_xy, floor_record, blockers, clearance=0.0):
                continue
            for height_offset in height_offsets:
                eye = [float(eye_xy[0]), float(eye_xy[1]), float(setup["camera_pos"][2]) + height_offset]
                quat = look_at_quaternion(eye, target)
                forward_xy = [float(target[0]) - float(eye[0]), float(target[1]) - float(eye[1])]
                norm = _norm_xy(forward_xy)
                if norm < 1e-8:
                    continue
                forward_xy = [forward_xy[0] / norm, forward_xy[1] / norm]
                all_visible = True
                for item in placed:
                    pos = item["real_position"]
                    obj_xy = [float(pos[0]), float(pos[1])]
                    if not _is_point_in_camera_fov_xy(obj_xy, eye_xy, forward_xy, fov_deg):
                        all_visible = False
                        break
                    if _segment_is_occluded_by_blockers(
                        start_xy=eye_xy,
                        start_z=float(eye[2]),
                        end_xy=obj_xy,
                        end_z=float(pos[2]),
                        blockers=blockers,
                    ) or _segment_is_occluded_by_trav_map(
                        start_xy=eye_xy,
                        end_xy=obj_xy,
                        trav_map=trav_map,
                        map_img=trav_map_img,
                    ):
                        all_visible = False
                        break
                if all_visible:
                    return {"eye": eye, "target": target, "quat": quat}
        return None

    def _assign_context_position_labels(placed, context_view):
        forward = _vec_normalize(_vec_sub(context_view["target"], context_view["eye"]))
        right = _vec_cross(forward, (0.0, 0.0, 1.0))
        if _vec_norm(right) < 1e-8:
            right = (1.0, 0.0, 0.0)
        right = _vec_normalize(right)
        ordering = []
        for idx, item in enumerate(placed):
            rel = _vec_sub(item["real_position"], context_view["eye"])
            horiz = sum(float(rel[i]) * float(right[i]) for i in range(3))
            ordering.append((horiz, idx))
        ordering.sort(key=lambda pair: pair[0])
        labels = ["left", "middle", "right"]
        assigned = [None] * len(placed)
        for label, (_, idx) in zip(labels, ordering):
            assigned[idx] = label
        return assigned

    questions = {task_type: [] for task_type in TASK_TYPES if task_type in task_types}
    correspondence_debug = {
        "attempted": 0,
        "success": 0,
        "fail_no_visible_slots": 0,
        "fail_no_line_layout": 0,
        "fail_visibility_constraint": 0,
        "fail_context_view": 0,
        "last_failure": None,
    }

    if "mirror_object_reality" in task_types:
        for i in range(max_q):
            setup = _reset_scene_for_question()
            slots = _gather_visible_slots_with_fallback(
                free_positions=free_positions,
                mirror_xy=setup["mirror_pos_xy"],
                mirror_z=setup["mirror_pos"][2],
                mirror_normal=setup["mirror_normal"],
                mirror_tangent=setup["mirror_tangent"],
                mirror_half_width=setup["mirror_half_width"],
                camera_xy=setup["camera_xy"],
                camera_z=setup["camera_pos"][2],
                camera_forward=setup["camera_forward"],
                floor_record=floor_record,
                blockers=blockers,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )
            chosen = _sample_distinct_slots(slots, k=1, min_sep=0.2, rng=rng)
            if not chosen:
                continue
            used_names = set()
            category = rng.choice(placeable_pool)["category"]
            obj, model = _acquire_object(category, used_names)
            slot = chosen[0]
            half_h = _safe_half_height(obj, default_half_height=0.06)
            pos = [float(slot["xy"][0]), float(slot["xy"][1]), float(floor_record.bbox_max[2]) + half_h + 0.01]
            _set_object_pose(obj, pos)
            _step_sim(8)
            visibility = _validate_slot_visibility(slot["xy"], pos, setup)
            if visibility is None:
                continue
            placed = [
                {
                    "name": obj.name,
                    "alias": "A",
                    "category": category,
                    "model": model,
                    "real_position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "mirror_position": [float(v) for v in visibility["reflected_xy"]] + [float(pos[2])],
                }
            ]
            ask_positive = (i % 2 == 0)
            if ask_positive:
                q = {
                    "task_type": "mirror_object_reality",
                    "question": f"Does the {category} seen in the mirror correspond to the real object {obj.name} in the physical scene?",
                    "answer": "Yes",
                    "answer_bool": True,
                    "placement": placed,
                    "evidence": {"mirror_label": f"{obj.name}_mirror", "real_object_name": obj.name},
                }
            else:
                fake_candidates = [e["category"] for e in placeable_pool if e["category"] != category]
                fake_cat = rng.choice(fake_candidates) if fake_candidates else "unknown_object"
                q = {
                    "task_type": "mirror_object_reality",
                    "question": f"Does a {fake_cat} seen in the mirror necessarily correspond to a real object in the physical scene?",
                    "answer": "No",
                    "answer_bool": False,
                    "placement": placed,
                    "evidence": {"queried_category": fake_cat, "real_categories_present": [category]},
                }
            q["scene_setup"] = _scene_setup_payload(setup)
            q["render"] = _render_question("mirror_object_reality", i, placed, [{"mirror_hit_xy": visibility["mirror_hit_xy"]}], setup)
            if q["render"] is not None:
                q["render"]["multi_image_input"] = _render_room_corner_views("mirror_object_reality", i, setup)
                q["render"]["gt_view_input"] = _render_gt_orbit_views("mirror_object_reality", i, setup, answer_value="middle")
            questions["mirror_object_reality"].append(q)
            if question_json_root is not None and question_scene_metadata is not None:
                _write_single_question_json(question_json_root, question_scene_metadata, "mirror_object_reality", i, q)

    if "mirror_distance" in task_types:
        for i in range(max_q):
            setup = _reset_scene_for_question()
            slots = _gather_visible_slots_with_fallback(
                free_positions=free_positions,
                mirror_xy=setup["mirror_pos_xy"],
                mirror_z=setup["mirror_pos"][2],
                mirror_normal=setup["mirror_normal"],
                mirror_tangent=setup["mirror_tangent"],
                mirror_half_width=setup["mirror_half_width"],
                camera_xy=setup["camera_xy"],
                camera_z=setup["camera_pos"][2],
                camera_forward=setup["camera_forward"],
                floor_record=floor_record,
                blockers=blockers,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )
            chosen = _sample_distinct_slots(slots, k=2, min_sep=0.26, rng=rng)
            if len(chosen) < 2:
                continue
            used_names = set()
            cats = rng.sample([e["category"] for e in placeable_pool], k=min(2, len(placeable_pool)))
            if len(cats) < 2:
                cats = [cats[0], cats[0]]
            placed = []
            placement_valid = True
            for j in range(2):
                obj, model = _acquire_object(cats[j], used_names)
                slot = chosen[j]
                half_h = _safe_half_height(obj, default_half_height=0.06)
                pos = [float(slot["xy"][0]), float(slot["xy"][1]), float(floor_record.bbox_max[2]) + half_h + 0.01]
                _set_object_pose(obj, pos)
                visibility = _validate_slot_visibility(slot["xy"], pos, setup)
                if visibility is None:
                    placement_valid = False
                    break
                placed.append(
                    {
                        "name": obj.name,
                        "alias": "AB"[j],
                        "category": cats[j],
                        "model": model,
                        "real_position": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "mirror_position": [float(v) for v in visibility["reflected_xy"]] + [float(pos[2])],
                        "real_u": float(_dot_xy([pos[0] - setup["mirror_pos_xy"][0], pos[1] - setup["mirror_pos_xy"][1]], setup["mirror_tangent"])),
                        "mirror_u": float(_dot_xy([visibility["reflected_xy"][0] - setup["mirror_pos_xy"][0], visibility["reflected_xy"][1] - setup["mirror_pos_xy"][1]], setup["mirror_tangent"])),
                        "mirror_hit_xy": visibility["mirror_hit_xy"],
                    }
                )
            _step_sim(8)
            if not placement_valid or len(placed) < 2:
                continue
            a, b = placed[0], placed[1]
            rel_mirror = "left" if float(a["mirror_u"]) < float(b["mirror_u"]) else "right"
            a_dist = _distance3(a["real_position"], setup["camera_pos"])
            b_dist = _distance3(b["real_position"], setup["camera_pos"])
            closer = "A" if a_dist <= b_dist else "B"
            a_label = _humanize_object_label(a.get("category", a["name"]))
            b_label = _humanize_object_label(b.get("category", b["name"]))
            q = {
                "task_type": "mirror_distance",
                "question": (
                    f"In the mirror, you can see two reflected objects: object A ({a_label}) "
                    f"and object B ({b_label}). In the real world, which object is closer "
                    f"to the observation position, A or B?"
                ),
                "answer": closer,
                "placement": placed,
                "evidence": {
                    "mirror_relation": rel_mirror,
                    "observation_position": list(setup["camera_pos"]),
                    "real_a_name": a["name"],
                    "real_b_name": b["name"],
                    "real_a_label": a_label,
                    "real_b_label": b_label,
                    "closer_real_name": a["name"] if closer == "A" else b["name"],
                    "closer_real_label": a_label if closer == "A" else b_label,
                    "closer_object": closer,
                    "a_distance_to_observer": float(a_dist),
                    "b_distance_to_observer": float(b_dist),
                    "a_real_u": float(a["real_u"]),
                    "b_real_u": float(b["real_u"]),
                    "a_mirror_u": float(a["mirror_u"]),
                    "b_mirror_u": float(b["mirror_u"]),
                },
            }
            q["scene_setup"] = _scene_setup_payload(setup)
            q["render"] = _render_question("mirror_distance", i, placed, [{"mirror_hit_xy": a["mirror_hit_xy"]}, {"mirror_hit_xy": b["mirror_hit_xy"]}], setup)
            if q["render"] is not None:
                q["render"]["multi_image_input"] = _render_room_corner_views("mirror_distance", i, setup)
                q["render"]["gt_view_input"] = _render_gt_orbit_views("mirror_distance", i, setup, answer_value="middle")
            questions["mirror_distance"].append(q)
            if question_json_root is not None and question_scene_metadata is not None:
                _write_single_question_json(question_json_root, question_scene_metadata, "mirror_distance", i, q)

    if "mirror_correspondence" in task_types:
        for i in range(max_q):
            case_success = False
            last_failure = None
            for retry_idx in range(10):
                setup = _reset_scene_for_question()
                slots = _gather_visible_slots_with_fallback(
                    free_positions=free_positions,
                    mirror_xy=setup["mirror_pos_xy"],
                    mirror_z=setup["mirror_pos"][2],
                    mirror_normal=setup["mirror_normal"],
                    mirror_tangent=setup["mirror_tangent"],
                    mirror_half_width=setup["mirror_half_width"],
                    camera_xy=setup["camera_xy"],
                    camera_z=setup["camera_pos"][2],
                    camera_forward=setup["camera_forward"],
                    floor_record=floor_record,
                    blockers=blockers,
                    trav_map=trav_map,
                    trav_map_img=trav_map_img,
                )
                if not slots:
                    correspondence_debug["fail_no_visible_slots"] += 1
                    last_failure = {
                        "reason": "no_visible_slots",
                        "retry_idx": retry_idx,
                    }
                    continue

                chosen_xys = None
                debug_line_xys = None
                category = None
                model = None
                target_rank = None
                attempt_configs = []
                slot_order = rng.sample(slots, k=len(slots))
                category_pool_order = rng.sample(placeable_pool, k=min(len(placeable_pool), 12))
                for candidate_rank in rng.sample([0, 1, 2], k=3):
                    for category_entry in category_pool_order:
                        models = list(category_entry["models"])
                        rng.shuffle(models)
                        for candidate_model in models[: min(3, len(models))]:
                            attempt_configs.append((candidate_rank, str(category_entry["category"]), str(candidate_model)))
                for candidate_rank, candidate_category, candidate_model in attempt_configs:
                    for visible_slot in slot_order:
                        candidate_line = _try_place_correspondence_line(setup, visible_slot, target_rank=candidate_rank)
                        if candidate_line is None:
                            continue
                        debug_line_xys = candidate_line.get("debug_line_xys")
                        candidate_xys = candidate_line.get("chosen_xys")
                        if candidate_xys is None:
                            continue
                        chosen_xys = candidate_xys
                        category = candidate_category
                        model = candidate_model
                        target_rank = candidate_rank
                        break
                    if chosen_xys is not None:
                        break
                if chosen_xys is None or category is None or model is None or target_rank is None:
                    correspondence_debug["fail_no_line_layout"] += 1
                    last_failure = {
                        "reason": "no_line_layout",
                        "retry_idx": retry_idx,
                        "category": category,
                        "model": model,
                        "target_rank": target_rank,
                        "line_xys": debug_line_xys,
                    }
                    continue

                used_names = set()
                placed = []
                placement_valid = True
                for j in range(3):
                    obj, model_name = _acquire_object(category, used_names, preferred_model=model)
                    slot_xy = chosen_xys[j]
                    half_h = _safe_half_height(obj, default_half_height=0.06)
                    pos = [float(slot_xy[0]), float(slot_xy[1]), float(floor_record.bbox_max[2]) + half_h + 0.01]
                    _set_object_pose(obj, pos)
                    visibility = _validate_correspondence_slot_visibility(slot_xy, pos, setup)
                    if j == target_rank and visibility is None:
                        placement_valid = False
                        break
                    if j != target_rank and visibility is not None:
                        placement_valid = False
                        break
                    placed.append(
                        {
                            "name": obj.name,
                            "alias": "ABC"[j],
                            "category": category,
                            "model": model_name,
                            "real_position": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "mirror_position": None if visibility is None else [float(v) for v in visibility["reflected_xy"]] + [float(pos[2])],
                            "mirror_hit_xy": None if visibility is None else visibility["mirror_hit_xy"],
                            "visible_in_mirror": bool(visibility is not None),
                        }
                    )
                _step_sim(8)
                if not placement_valid or len(placed) < 3:
                    correspondence_debug["fail_visibility_constraint"] += 1
                    last_failure = {
                        "reason": "visibility_constraint",
                        "retry_idx": retry_idx,
                        "category": category,
                        "model": model,
                        "target_rank": int(target_rank),
                        "line_xys": chosen_xys or debug_line_xys,
                    }
                    continue

                context_view = _find_context_view_for_correspondence(placed, setup)
                if context_view is None:
                    correspondence_debug["fail_context_view"] += 1
                    last_failure = {
                        "reason": "context_view",
                        "retry_idx": retry_idx,
                        "category": category,
                        "model": model,
                        "target_rank": int(target_rank),
                        "line_xys": chosen_xys or debug_line_xys,
                    }
                    continue

                position_labels = _assign_context_position_labels(placed, context_view)
                for item, label in zip(placed, position_labels):
                    item["position_label"] = label
                target = placed[target_rank]
                options = ["left", "middle", "right"]
                q = {
                    "task_type": "mirror_correspondence",
                    "question": "After checking the mirror image and the reference view without the mirror, which real object corresponds to the mirrored object: left, middle, or right?",
                    "options": options,
                    "answer": target["position_label"],
                    "placement": placed,
                    "evidence": {
                        "mirror_label": "target_mirror_object",
                        "real_object_name": target["name"],
                        "shared_category": category,
                        "shared_model": model,
                        "only_target_visible_in_mirror": True,
                        "target_position_label": target["position_label"],
                        "objects_aligned_parallel_to_mirror": True,
                    },
                }
                q["scene_setup"] = _scene_setup_payload(setup)
                q["render"] = _render_question("mirror_correspondence", i, placed, [{"mirror_hit_xy": target["mirror_hit_xy"]}], setup)
                if q["render"] is not None:
                    q["render"]["context_without_mirror"] = _render_context_without_mirror("mirror_correspondence", i, placed, setup)
                    q["render"]["multi_image_input"] = _render_room_corner_views("mirror_correspondence", i, setup)
                    q["render"]["gt_view_input"] = _render_gt_orbit_views("mirror_correspondence", i, setup, answer_value=target["position_label"])
                questions["mirror_correspondence"].append(q)
                if question_json_root is not None and question_scene_metadata is not None:
                    _write_single_question_json(question_json_root, question_scene_metadata, "mirror_correspondence", i, q)
                correspondence_debug["success"] += 1
                case_success = True
                break

            correspondence_debug["attempted"] += 1
            if not case_success:
                correspondence_debug["last_failure"] = last_failure

    _park_all()
    questions["metadata"] = {
        "left_right_axis": "mirror_tangent_axis",
        "per_question_object_replacement": True,
        "per_question_camera_reset": True,
        "per_question_mirror_reset": True,
        "camera_looks_at_mirror_center": True,
        "reflection_visibility_checked": True,
        "scene_object_occlusion_checked": True,
        "mirror_correspondence_uses_identical_objects": True,
        "mirror_correspondence_single_visible_target": True,
        "mirror_correspondence_context_render_without_mirror": True,
        "mirror_correspondence_aligned_parallel_to_mirror": True,
        "mirror_correspondence_context_view_visibility_checked": True,
        "mirror_correspondence_effective_mirror_width_scale": CORRESPONDENCE_MIRROR_WIDTH_SCALE,
        "enabled_task_types": sorted(task_types),
        "mirror_correspondence_debug": correspondence_debug if "mirror_correspondence" in task_types else None,
    }
    return questions


def _build_config(args):
    config_filename = os.path.join(og.example_config_path, f"{args.robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"] = args.scene
    # Load the full house so floors from other rooms remain available when the
    # target room is initialized.
    config["scene"].pop("load_room_instances", None)
    # Disable robot placement for this generation script.
    config["robots"] = []
    config["objects"] = []
    return config


def _summarize_room_objects(room_objects):
    counts = {}
    for obj in room_objects:
        counts[obj.category] = counts.get(obj.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate mirror distance candidates from runtime OmniGibson scene.")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name")
    parser.add_argument("--room", type=str, default=None, help="Optional room instance name")
    parser.add_argument("--floor", type=str, default=None, help="Optional floor object name")
    parser.add_argument("--agent_position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--keys_json", type=str, default=DEFAULT_KEYS_JSON)
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--max_questions_per_type", type=int, default=8)
    parser.add_argument("--output_root", type=str, default="renders_mirror")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument(
        "--disable_trav_map_occlusion_check",
        action="store_true",
        help="Disable the secondary 2D occlusion check based on scene traversability pixels.",
    )
    parser.add_argument(
        "--exit_on_finish",
        action="store_true",
        help="Exit immediately after generation. By default, keep simulator alive.",
    )
    args = parser.parse_args(argv)

    resolved_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    rng = random.Random(resolved_seed)

    run_dir = os.path.join(args.output_root, args.scene)
    os.makedirs(run_dir, exist_ok=True)
    question_json_root = os.path.join(run_dir, "mirror_question_jsons")
    render_root = os.path.join(run_dir, "mirror_renders")

    config = _build_config(args)
    try:
        env = og.Environment(configs=config)
        _set_viewer_camera_fov()
        scene = env.scene

        agent_pos = _resolve_agent_position(env, args.agent_position)
        room_objects = _collect_room_objects(scene, args.room)
        floor_record = _select_floor(room_objects, args.floor, agent_pos, room_name=args.room)
        if args.agent_position is None and not getattr(env, "robots", []):
            agent_pos = (
                float(floor_record.center[0]),
                float(floor_record.center[1]),
                float(floor_record.bbox_max[2]) + CAMERA_HEIGHT,
            )
        room_bbox_world_xy, resolved_room_instance = _resolve_room_bbox_world_xy(scene, args.room, floor_record)
        blockers = [obj for obj in room_objects if _is_floor_blocker(obj, floor_record.bbox_max[2])]
        wall_blockers = [obj for obj in room_objects if _is_wall_occluder(obj)]
        trav_map = None
        trav_map_img = None
        if not args.disable_trav_map_occlusion_check:
            trav_map, trav_map_img = _trav_map_floor_image(scene, floor_idx=0, scene_name=args.scene)
        free_positions = _generate_free_positions(
            floor_record=floor_record,
            blockers=blockers,
            agent_pos=agent_pos,
            count=220,
            target_radius=DEFAULT_TARGET_RADIUS,
        )
        if not free_positions:
            raise RuntimeError("No free floor positions found for mirror/object placement.")

        category_models = _load_category_models()
        placeable_pool = _build_placeable_category_pool(args.keys_json)
        if not placeable_pool:
            raise RuntimeError("No placeable categories found for object placement.")

        preferred_camera_xy = None
        if args.agent_position is not None:
            preferred_camera_xy = [float(args.agent_position[0]), float(args.agent_position[1])]

        mirror_spec = {"category": "mirror", "model": "tytkbq"}
        question_scene_metadata = {
            "scene": args.scene,
            "room": resolved_room_instance or args.room,
            "floor_name": floor_record.name,
            "seed": resolved_seed,
            "camera_setup": {
                "mode": "per_question_reset",
                "preferred_xy": None if preferred_camera_xy is None else [float(v) for v in preferred_camera_xy],
                "height_m": CAMERA_HEIGHT,
                "fov_deg": CAMERA_FOV_DEG,
                "look_target": "mirror_center",
                "render_view_groups": {
                    "single_view": "observer_view",
                    "multi_view": {
                        "type": "arc_around_mirror_center",
                        "angle_range_deg": [-MULTI_VIEW_ARC_MAX_DEG, MULTI_VIEW_ARC_MAX_DEG],
                        "step_deg": MULTI_VIEW_STEP_DEG,
                    },
                    "gt_view": {
                        "type": "observer_perpendicular_side_views",
                        "side_offset_m": GT_SIDE_OFFSET_M,
                        "look_target": "observer_position",
                    },
                },
                "trav_map_pixel_occlusion_check": bool(trav_map is not None and trav_map_img is not None),
            },
            "mirror_setup": {
                "name": "render_mirror_main",
                "category": "mirror",
                "model": "tytkbq",
                "mode": "per_question_reset",
                "camera_forward_distance_m": MIRROR_AHEAD_DISTANCE,
            },
        }
        mirror_obj = scene.object_registry("name", "render_mirror_main")
        if mirror_obj is None:
            mirror_obj = DatasetObject(
                name="render_mirror_main",
                category=mirror_spec["category"],
                model=mirror_spec["model"],
                visual_only=True,
            )
            scene.add_object(mirror_obj)
        try:
            mirror_obj.visual_only = True
        except Exception:
            pass
        _step_sim(20)

        _park_object(mirror_obj, 999)
        _step_sim(10)

        qa = _generate_questions_with_per_question_placement(
            scene=scene,
            rng=rng,
            mirror_obj=mirror_obj,
            floor_record=floor_record,
            free_positions=free_positions,
            blockers=blockers,
            wall_blockers=wall_blockers,
            placeable_pool=placeable_pool,
            max_q=max(1, args.max_questions_per_type),
            render_root=render_root,
            enable_render=not args.skip_render,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            preferred_camera_xy=preferred_camera_xy,
            task_types={"mirror_distance"},
            question_json_root=question_json_root,
            question_scene_metadata=question_scene_metadata,
            room_bbox_world_xy=room_bbox_world_xy,
            room_instance_name=resolved_room_instance,
        )
        summary = {key: len(value) for key, value in qa.items() if isinstance(value, list)}
        print(
            json.dumps(
                {
                    "scene": args.scene,
                    "room": args.room,
                    "seed": resolved_seed,
                    "question_summary": summary,
                    "question_json_root": question_json_root,
                    "render_root": None if args.skip_render else render_root,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        if not args.exit_on_finish:
            print("[batch_mirror_distance] Generation finished. Simulator is kept alive. Press Ctrl+C to exit.")
            try:
                while True:
                    og.sim.render()
                    _step_sim(1)
            except KeyboardInterrupt:
                print("[batch_mirror_distance] Exit requested by user (Ctrl+C).")
    except Exception as exc:
        _log_exception("main", exc)
        raise


if __name__ == "__main__":
    main()
