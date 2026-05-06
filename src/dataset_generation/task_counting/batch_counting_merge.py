"""
Generate counting candidates from a runtime OmniGibson scene.

The script follows the same overall style as the task_confusing_relation batch
scripts: build an OmniGibson config, launch one environment, query env.scene,
and export candidate metadata as JSON.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import random
import struct
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
OG_ROOT = str(SCRIPT_DIR / "OmniGibson")
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import ControllableObjectViewAPI, delete_or_deactivate_prim


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

HIDDEN_BOX_FIXED_ASSETS = (
    ("carton", "cdmmwy"),
    ("cedar_chest", "fwstpx"),
    ("cedar_chest", "gbdzls"),
)
HIDDEN_BOX_CONTAINER_COUNT = len(HIDDEN_BOX_FIXED_ASSETS)
HIDDEN_IN_BOX_QUESTION_COUNT = 2
BALL_RADIUS = 0.035
BALL_CLEARANCE = 0.06
HIDDEN_BOX_TARGET_Z_OFFSET_M = -0.10
GRID_STEP = 0.35
ROOM_BBOX_EXPANSION_RATIO = 0.35
ROOM_BBOX_EXPANSION_MIN = 0.75
MIN_BALL_DISTANCE_FROM_AGENT = 1.5
VIEWER_CAMERA_FOV_DEG = 100.0
VIEWER_FRAME_RENDER_STEPS = 10
VIEWER_FRAME_MAX_RETRIES = 3
VIEWER_FRAME_RETRY_SLEEP_SEC = 0.1
SEMANTIC_FAULT_CONFUSER_MIN_SEPARATION_M = 0.5
MIN_ROOM_BBOX_AREA_M2 = 8.0
MAX_ROOM_BBOX_AREA_M2 = 80.0
ATTEMPTED_ROOM_MARKER = "counting_room_attempted.json"
TIMING_LOG_ENABLED = os.environ.get("COUNTING_TIMING_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}
SIM_STEP_MINIMAL = 1
SIM_STEP_CAMERA_MODALITY = 3

WIDE_OCCLUDER_CATEGORIES = {
    "bar",
    "bookcase",
    "bottom_cabinet",
    "bottom_cabinet_no_top",
    "cedar_chest",
    "checkout_counter",
    "coffee_table",
    "commercial_kitchen_shelf",
    "conference_table",
    "console_table",
    "countertop",
    "desk",
    "display_case",
    "freezer",
    "fridge",
    "grocery_shelf",
    "lab_table",
    "storage_box",
    "nightstand",
    "oven",
    "pool_table",
    "reception_desk",
    "room_divider",
    "shelf",
    "snack_rack",
    "sofa",
    "stove",
    "wardrobe",
    "washer",
    "wine_fridge",
}


def _set_viewer_camera_fov(fov_deg: float = VIEWER_CAMERA_FOV_DEG) -> None:
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

NARROW_DIVIDER_CATEGORIES = {
    "armchair",
    "chair",
    "eames_chair",
    "garden_chair",
    "music_stool",
    "ottoman",
    "pillar",
    "shopping_cart",
    "straight_chair",
    "stool",
    "swivel_chair",
    "taboret",
    "turnstile",
}

CHAIR_CATEGORIES = {
    "armchair",
    "chair",
    "eames_chair",
    "garden_chair",
    "straight_chair",
    "swivel_chair",
}

CONTAINER_CATEGORIES = {
    "box",
    "cedar_chest",
    "hamper",
    "packing_box",
    "public_trash_can",
    "toy_box",
    "trash_can",
}

LIGHT_KEYWORDS = ("light", "lamp", "chandelier")
TARGET_CATEGORIES = [
    "storage_box",
    "blender",
    "cooler",
    "rice_cooker",
    "instant_pot",
]
NON_BLOCKING_CATEGORIES = {
    "background",
    "baseboard",
    "ceilings",
    "fire_alarm",
    "fire_sprinkler",
    "fixed_window",
    "floors",
    "mirror",
    "picture",
    "roof",
    "walls",
}

DEFAULT_AVG_CATEGORY_SPECS = str(SCRIPT_DIR / "OmniGibson" / "omnigibson" / "configs" / "avg_category_specs.json")
DEFAULT_OBJECT_INVENTORY = str(SCRIPT_DIR / "bddl3" / "bddl" / "generated_data" / "object_inventory.json")
DEFAULT_OBJECT_DATASET_ROOT = SCRIPT_DIR / "datasets" / "behavior-1k-assets" / "objects"
DEFAULT_KEYS_CLIP_TOP3 = str(SCRIPT_DIR / "keys_clip_top3.json")
MAX_RANDOM_BALL_COUNT = 6
RENDER_TARGET_PREFIX = "render_count_target_"
RENDER_CONFUSER_PREFIX = "render_confuser_"
RENDER_HIDDEN_BOX_PREFIX = "render_hidden_box_"
FAILED_RENDER_MODELS_BY_CATEGORY: dict[str, set[str]] = {}
HIDDEN_BOX_METADATA_CACHE: dict[tuple[str, str], dict] = {}
RENDER_PARK_X = 1000.0
RENDER_PARK_Y = 1000.0
RENDER_PARK_Z = 120.0
COUNT_TARGET_MIN_EDGE_M = 0.05
COUNT_TARGET_MAX_EDGE_M = 0.25
OBSERVATION_MERGED_MIN_SEPARATION_SCALE = 0.92
OBSERVATION_MERGED_CLUSTER_MARGIN_M = 0.10
COUNTING_TASK_TYPES = (
    "hidden_by_others",
    "observation_divided",
    "observation_merged",
    "light_change",
    "semantic_fault",
    "hidden_in_box",
)
VIEW_SPECS = [
    (0.0, "_front", "front"),
    (45.0, "_front_left", "front_left"),
    (90.0, "_left", "left"),
    (135.0, "_back_left", "back_left"),
    (180.0, "_back", "back"),
    (225.0, "_back_right", "back_right"),
    (270.0, "_right", "right"),
    (315.0, "_front_right", "front_right"),
]

DIRECT_PLACEMENT_MODE = False
RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT = True
AGENT_POSITION_CLEARANCE = 0.35
AGENT_SHORT_EDGE_OFFSET_RATIOS = (0.10, 0.16, 0.22)
AGENT_SHORT_EDGE_SCAN_STEP = 0.2
AGENT_CAMERA_HEIGHT_M = 1.25
AGENT_CAMERA_TARGET_HEIGHT_M = 0.9
AGENT_CAMERA_PITCH_DEG = 15.0
OBSERVATION_MERGED_CAMERA_PITCH_DEG = 45.0
PRIMARY_VIEW_CLEARANCE = 0.25
CLOSEUP_MIN_VISIBLE_PIXELS = 80
CLOSEUP_CAMERA_RADII_M = (0.42, 0.58, 0.78, 0.98)
CLOSEUP_SEARCH_CAMERA_HEIGHT_M = 0.12
CLOSEUP_ACCEPT_VISIBLE_PIXELS = 10
CLOSEUP_GOOD_ENOUGH_VISIBLE_PIXELS = max(1, CLOSEUP_ACCEPT_VISIBLE_PIXELS // 2)
CLOSEUP_AZIMUTH_DEG = (0.0, 35.0, 70.0, 110.0, 145.0, 180.0, 215.0, 250.0, 290.0, 325.0)
CLOSEUP_REVEAL_LINK_KEYWORDS = ("lid", "cover", "door", "top", "panel")
CONTAINER_LID_LINK_KEYWORDS = ("lid", "cover", "top", "up")
ROOM_VIEW_CAMERA_HEIGHT_M = 2.2
ROOM_VIEW_TARGET_HEIGHT_M = 1.05
HIDDEN_BOX_BOX_FRONT_CLEARANCE_M = 0.5
HIDDEN_BOX_BOX_CLOSEUP_FRONT_OFFSET_M = 0.6
HIDDEN_BOX_BOX_CLOSEUP_HEIGHT_OFFSET_M = 0.6
HIDDEN_BOX_BOX_CLOSEUP_PITCH_DEG = 45.0
HIDDEN_BOX_BOX_CLOSEUP_FOV_DEG = 55.0


def _ordered_task_types(task_types) -> tuple[str, ...]:
    ordered = []
    seen = set()
    for case_name in task_types:
        if case_name in seen:
            continue
        seen.add(case_name)
        ordered.append(case_name)
    if "hidden_in_box" in seen:
        ordered = [case_name for case_name in ordered if case_name != "hidden_in_box"] + ["hidden_in_box"]
    return tuple(ordered)


def _question_count_for_task(task_type: str, default_question_count: int) -> int:
    if task_type == "hidden_in_box":
        return int(HIDDEN_IN_BOX_QUESTION_COUNT)
    return int(default_question_count)


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
    def label(self) -> str:
        return self.name

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.bbox_min, self.bbox_max))

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.bbox_min, self.bbox_max))

    @property
    def footprint_area(self) -> float:
        ext = self.extents
        return float(max(ext[0], 0.0) * max(ext[1], 0.0))

    @property
    def bbox_world_xy(self) -> tuple[float, float, float, float]:
        return _normalize_bbox_xyxy((self.bbox_min[0], self.bbox_min[1], self.bbox_max[0], self.bbox_max[1]))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "in_rooms": list(self.in_rooms),
            "bbox": [list(self.bbox_min), list(self.bbox_max)],
            "has_open_state": self.has_open_state,
            "open_state": self.open_state,
        }


@dataclass
class WallRecord:
    name: str
    category: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_world_xy: tuple[float, float, float, float]
    is_structural_wall: bool


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _bbox_area_xyxy(bbox_xyxy) -> float:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return float(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))


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


def _entry_component_entries(entry: dict) -> list[dict]:
    component_entries = entry.get("component_entries")
    if isinstance(component_entries, list) and component_entries:
        return component_entries
    return [entry]


def _merged_ball_positions(entries: list[dict]) -> list[list[float]]:
    seen_ball_positions = set()
    ball_positions = []
    for entry in entries:
        for pos in entry.get("ball_positions", []):
            key = (round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4))
            if key in seen_ball_positions:
                continue
            seen_ball_positions.add(key)
            ball_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    return ball_positions


def _entry_ball_positions(entry: dict) -> list[list[float]]:
    return _merged_ball_positions(_entry_component_entries(entry))


def _serialize_live_scene_object(
    obj,
    *,
    role: str,
    requested_position=None,
    container_name: str | None = None,
    contains_ball: bool | None = None,
    source_entry_case: str | None = None,
    source_category: str | None = None,
    source_model: str | None = None,
    source_clip_score: float | None = None,
    source_sampling_source: str | None = None,
) -> dict:
    payload = {
        "role": str(role),
        "name": str(getattr(obj, "name", "")),
        "category": str(getattr(obj, "category", "object")),
        "model": getattr(obj, "model", None),
    }

    try:
        position, quaternion = obj.get_position_orientation()
        payload["position"] = [float(v) for v in position.detach().cpu().tolist()]
        payload["quaternion_xyzw"] = [float(v) for v in quaternion.detach().cpu().tolist()]
    except Exception:
        payload["position"] = None
        payload["quaternion_xyzw"] = None

    try:
        bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
        payload["bbox"] = [list(bbox_min), list(bbox_max)]
    except Exception:
        payload["bbox"] = None

    in_rooms = getattr(obj, "in_rooms", None) or []
    payload["in_rooms"] = [str(room) for room in in_rooms]

    states = getattr(obj, "states", None) or {}
    if object_states.Open in states:
        try:
            payload["open_state"] = bool(states[object_states.Open].get_value())
        except Exception:
            payload["open_state"] = None

    if requested_position is not None:
        payload["requested_position"] = [float(v) for v in requested_position]
    if container_name is not None:
        payload["container_name"] = str(container_name)
    if contains_ball is not None:
        payload["contains_ball"] = bool(contains_ball)
    if source_entry_case is not None:
        payload["source_entry_case"] = str(source_entry_case)
    if source_category is not None:
        payload["source_category"] = str(source_category)
    if source_model is not None:
        payload["source_model"] = str(source_model)
    if source_clip_score is not None:
        payload["source_clip_score"] = float(source_clip_score)
    if source_sampling_source is not None:
        payload["source_sampling_source"] = str(source_sampling_source)
    return payload


def _attach_resolved_target_metadata(entry: dict, placed_targets: list[dict]) -> None:
    count_object = dict(entry.get("count_object") or {})
    resolved_models = sorted(
        {
            str(target.get("model"))
            for target in placed_targets
            if target.get("model") not in {None, ""}
        }
    )
    if resolved_models:
        count_object["resolved_models"] = resolved_models
    if len(resolved_models) == 1:
        count_object["target_model"] = resolved_models[0]
    count_object["resolved_object_count"] = int(len(placed_targets))
    entry["count_object"] = count_object


def _collect_resolved_hidden_box_containers(scene, entries: list[dict]) -> list[dict]:
    containers: list[dict] = []
    seen_names: set[str] = set()
    for entry in entries:
        spec = entry.get("container_spec") or {}
        container_name = spec.get("name")
        if not container_name or container_name in seen_names:
            continue
        obj = scene.object_registry("name", container_name)
        if obj is None:
            continue
        seen_names.add(container_name)
        containers.append(
            _serialize_live_scene_object(
                obj,
                role="container",
                contains_ball=entry.get("contains_ball"),
                source_entry_case=entry.get("case"),
                source_category=spec.get("category"),
                source_model=spec.get("model"),
            )
        )
    containers.sort(key=lambda item: (item.get("name") or "", item.get("model") or ""))
    return containers


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
        "mirror_setup": None,
        "task_type": task_type,
        "question_index": q_idx,
        "question_id": f"{task_type}/q_{q_idx:03d}",
        "question_data": entry,
    }
    payload["image_paths"] = _collect_image_paths(payload["question_data"])
    out_path = os.path.join(task_dir, f"q_{q_idx:03d}.json")
    _write_json(out_path, payload)
    return out_path


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_counting] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _log_timing(event: str, /, **fields) -> None:
    if not TIMING_LOG_ENABLED:
        return
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    print("[timing] " + " ".join(parts), flush=True)


def _distance_xy(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _sub_xy(a, b) -> tuple[float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _add_xy(a, b) -> tuple[float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))


def _scale_xy(v, scale: float) -> tuple[float, float]:
    return (float(v[0]) * scale, float(v[1]) * scale)


def _norm_xy(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _normalize_bbox_xyxy(bbox_xyxy, name="bbox", min_extent: float = 1e-3):
    arr = np.array(bbox_xyxy, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain 4 numbers: [xmin, ymin, xmax, ymax], got {bbox_xyxy}")
    xmin, ymin, xmax, ymax = [float(v) for v in arr.tolist()]
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if abs(xmax - xmin) < float(min_extent):
        pad = max(float(min_extent) - abs(xmax - xmin), float(min_extent)) * 0.5
        xmin -= pad
        xmax += pad
    if abs(ymax - ymin) < float(min_extent):
        pad = max(float(min_extent) - abs(ymax - ymin), float(min_extent)) * 0.5
        ymin -= pad
        ymax += pad
    return xmin, ymin, xmax, ymax


def _expand_bbox_xyxy(bbox_xyxy, expansion_ratio: float, expansion_min: float):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    pad_x = max(float(expansion_min), width * float(expansion_ratio))
    pad_y = max(float(expansion_min), height * float(expansion_ratio))
    return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)


def _segment_bbox_overlap_xy(start_xy, end_xy, bbox_min, bbox_max, margin: float = 0.0):
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    dx = x1 - x0
    dy = y1 - y0
    xmin = float(bbox_min[0]) - float(margin)
    ymin = float(bbox_min[1]) - float(margin)
    xmax = float(bbox_max[0]) + float(margin)
    ymax = float(bbox_max[1]) + float(margin)
    t0, t1 = 0.0, 1.0
    for origin, delta, lo, hi in ((x0, dx, xmin, xmax), (y0, dy, ymin, ymax)):
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


def _bboxes_intersect_xy(bbox_a_xyxy, bbox_b_xyxy, eps: float = 1e-6) -> bool:
    axmin, aymin, axmax, aymax = _normalize_bbox_xyxy(bbox_a_xyxy)
    bxmin, bymin, bxmax, bymax = _normalize_bbox_xyxy(bbox_b_xyxy)
    return not (
        axmax < bxmin - float(eps)
        or bxmax < axmin - float(eps)
        or aymax < bymin - float(eps)
        or bymax < aymin - float(eps)
    )


def _expand_bbox_until_wall_touch(
    bbox_xyxy,
    wall_bboxes_xyxy: list[tuple[float, float, float, float]],
    expansion_ratio: float,
    expansion_min: float,
    step_ratio: float = 0.05,
    max_steps_per_direction: int = 256,
):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    target_pad_x = max(float(expansion_min), width * float(expansion_ratio))
    target_pad_y = max(float(expansion_min), height * float(expansion_ratio))
    step_x = max(1e-3, target_pad_x * float(step_ratio))
    step_y = max(1e-3, target_pad_y * float(step_ratio))

    remaining = {
        "left": target_pad_x,
        "right": target_pad_x,
        "down": target_pad_y,
        "up": target_pad_y,
    }
    finished = {direction: remaining[direction] <= 1e-8 for direction in remaining}

    for _ in range(int(max_steps_per_direction)):
        if all(finished.values()):
            break
        progressed = False
        for direction in ("left", "right", "down", "up"):
            if finished[direction]:
                continue

            delta = min(
                remaining[direction],
                step_x if direction in {"left", "right"} else step_y,
            )
            candidate = [xmin, ymin, xmax, ymax]
            if direction == "left":
                candidate[0] -= delta
            elif direction == "right":
                candidate[2] += delta
            elif direction == "down":
                candidate[1] -= delta
            else:
                candidate[3] += delta

            hit_wall = any(_bboxes_intersect_xy(candidate, wall_bbox) for wall_bbox in wall_bboxes_xyxy)
            if hit_wall:
                finished[direction] = True
                remaining[direction] = 0.0
                continue

            xmin, ymin, xmax, ymax = candidate
            remaining[direction] -= delta
            progressed = True
            if remaining[direction] <= 1e-8:
                finished[direction] = True

        if not progressed and not all(finished.values()):
            break

    return xmin, ymin, xmax, ymax


def _point_inside_bbox_xyxy(point_xy, bbox_xyxy, margin: float = 0.0) -> bool:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return (
        xmin - float(margin) <= float(point_xy[0]) <= xmax + float(margin)
        and ymin - float(margin) <= float(point_xy[1]) <= ymax + float(margin)
    )


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


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _tensor_to_tuple3(tensor) -> tuple[float, float, float]:
    vals = tensor.cpu().tolist()
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def _tensor_to_list(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _segment_intersects_bbox_xy(p0, p1, record: RuntimeObjectRecord) -> bool:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0

    def clip(p: float, q: float, t_min: float, t_max: float):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False, t_min, t_max
            return True, t_min, t_max
        r = q / p
        if p < 0.0:
            if r > t_max:
                return False, t_min, t_max
            if r > t_min:
                t_min = r
        else:
            if r < t_min:
                return False, t_min, t_max
            if r < t_max:
                t_max = r
        return True, t_min, t_max

    ok, t0, t1 = clip(-dx, x0 - record.bbox_min[0], t0, t1)
    if not ok:
        return False
    ok, t0, t1 = clip(dx, record.bbox_max[0] - x0, t0, t1)
    if not ok:
        return False
    ok, t0, t1 = clip(-dy, y0 - record.bbox_min[1], t0, t1)
    if not ok:
        return False
    ok, t0, t1 = clip(dy, record.bbox_max[1] - y0, t0, t1)
    return ok and t1 >= t0


def _point_inside_bbox_xy(point_xy, bbox_min, bbox_max, margin: float = 0.0) -> bool:
    return (
        bbox_min[0] - margin <= float(point_xy[0]) <= bbox_max[0] + margin
        and bbox_min[1] - margin <= float(point_xy[1]) <= bbox_max[1] + margin
    )


def _is_light(record: RuntimeObjectRecord) -> bool:
    category = record.category.lower()
    return any(keyword in category for keyword in LIGHT_KEYWORDS)


def _is_wide_occluder(record: RuntimeObjectRecord) -> bool:
    category = record.category.lower()
    if _is_light(record):
        return False
    if category in NON_BLOCKING_CATEGORIES or "wall" in category or "door" in category or "floor" in category:
        return False
    footprint = record.extents[:2]
    return (
        record.category in WIDE_OCCLUDER_CATEGORIES
        or max(float(footprint[0]), float(footprint[1])) >= 0.9
        or record.footprint_area >= 0.75
    )


def _is_hidden_anchor_candidate(
    record: RuntimeObjectRecord,
    room_bbox_xyxy=None,
    floor_record: RuntimeObjectRecord | None = None,
) -> bool:
    category = record.category.lower()
    if category == "floors":
        return False
    if _is_light(record):
        return False
    if category in NON_BLOCKING_CATEGORIES or "wall" in category or "door" in category or "floor" in category:
        return False
    if "window" in category or "picture" in category or "wall_mounted" in category:
        return False

    ext_x, ext_y, ext_z = [float(v) for v in record.extents]
    long_side = max(ext_x, ext_y)
    short_side = min(ext_x, ext_y)

    # Allow broadly any object that can plausibly occlude the ball,
    # but filter out tiny footprint objects.
    if ext_z < 0.22:
        return False
    if long_side < 0.22:
        return False
    if short_side < 0.08 and record.footprint_area < 0.03:
        return False
    if record.footprint_area < 0.035:
        return False

    return True


def _distance_point_to_bbox_xy(point_xy, bbox_min, bbox_max) -> float:
    px, py = float(point_xy[0]), float(point_xy[1])
    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])
    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    if dx <= 0.0 and dy <= 0.0:
        return 0.0
    return math.hypot(dx, dy)


def _is_narrow_divider(record: RuntimeObjectRecord) -> bool:
    category = record.category.lower()
    if _is_light(record):
        return False
    if category in NON_BLOCKING_CATEGORIES or "wall" in category or "door" in category or "floor" in category:
        return False
    footprint = record.extents[:2]
    long_side = max(float(footprint[0]), float(footprint[1]))
    short_side = min(float(footprint[0]), float(footprint[1]))
    return (
        record.category in NARROW_DIVIDER_CATEGORIES
        or (
            record.extents[2] >= 0.5
            and 0.08 <= short_side <= 0.8
            and long_side <= 1.8
            and record.footprint_area <= 0.8
        )
    )


def _is_chair(record: RuntimeObjectRecord) -> bool:
    category = record.category.lower()
    if _is_light(record):
        return False
    if category in NON_BLOCKING_CATEGORIES or "wall" in category or "door" in category or "floor" in category:
        return False
    return category in CHAIR_CATEGORIES or "chair" in category


def _is_container(record: RuntimeObjectRecord) -> bool:
    category = record.category.lower()
    if record.category in CONTAINER_CATEGORIES:
        return True
    if record.has_open_state:
        return True
    return any(token in category for token in ("box", "bin", "basket", "hamper", "chest"))


def _is_floor_blocker(record: RuntimeObjectRecord, floor_z: float) -> bool:
    category = record.category.lower()
    if category in NON_BLOCKING_CATEGORIES:
        return False
    if _is_light(record):
        return False
    if "window" in category or "picture" in category or "wall_mounted" in category:
        return False
    if record.bbox_min[2] > floor_z + 0.65 and record.footprint_area < 0.25:
        return False
    return True


def _support_distance_xy(record: RuntimeObjectRecord, direction_xy) -> float:
    norm = _norm_xy(direction_xy)
    if norm < 1e-8:
        return 0.0
    direction = (float(direction_xy[0]) / norm, float(direction_xy[1]) / norm)
    ext = record.extents
    half_ext = (ext[0] / 2.0, ext[1] / 2.0)
    return float(abs(direction[0]) * half_ext[0] + abs(direction[1]) * half_ext[1])


def _footprint_corners(record: RuntimeObjectRecord) -> list[tuple[float, float]]:
    return [
        (record.bbox_min[0], record.bbox_min[1]),
        (record.bbox_min[0], record.bbox_max[1]),
        (record.bbox_max[0], record.bbox_min[1]),
        (record.bbox_max[0], record.bbox_max[1]),
    ]


def _get_scene_objects(scene):
    raw_objects = getattr(scene, "objects", [])
    if isinstance(raw_objects, dict):
        return list(raw_objects.values())
    return list(raw_objects)


def _segmap_get(scene):
    return getattr(scene, "seg_map", None) or getattr(scene, "_seg_map", None)


def _segmap_room_bbox_from_maps(scene, room_ins_id):
    seg = _segmap_get(scene)
    if seg is None or not hasattr(seg, "room_ins_map"):
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}

    room_map = seg.room_ins_map
    if room_map is None:
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()

    mask = room_map == int(room_ins_id)
    if not bool(mask.any().item()):
        return {"bbox_map_rc": None, "bbox_world_xy": None, "pixel_count": 0}

    idx = mask.nonzero(as_tuple=False)
    rmin = int(idx[:, 0].min().item())
    rmax = int(idx[:, 0].max().item())
    cmin = int(idx[:, 1].min().item())
    cmax = int(idx[:, 1].max().item())
    bbox_map = (rmin, cmin, rmax, cmax)

    corners_rc = th.tensor(
        [
            [float(rmin), float(cmin)],
            [float(rmin), float(cmax)],
            [float(rmax), float(cmin)],
            [float(rmax), float(cmax)],
        ],
        dtype=th.float32,
    )
    corners_xy = seg.map_to_world(corners_rc).detach().cpu().numpy()
    xmin, ymin = corners_xy.min(axis=0)
    xmax, ymax = corners_xy.max(axis=0)
    return {
        "bbox_map_rc": bbox_map,
        "bbox_world_xy": (float(xmin), float(ymin), float(xmax), float(ymax)),
        "pixel_count": int(idx.shape[0]),
    }


def _collect_wall_records(scene) -> list[WallRecord]:
    robot_names = {robot.name for robot in getattr(scene, "robots", [])}
    wall_records = []
    for obj in _get_scene_objects(scene):
        if obj.name in robot_names:
            continue
        name = str(getattr(obj, "name", ""))
        category = str(getattr(obj, "category", "object"))
        low_name = name.lower()
        low_category = category.lower()
        if "wall" not in low_name:
            continue
        try:
            bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
        except Exception as exc:
            _log_exception(f"Failed to read wall AABB for object {name}", exc)
            continue
        bbox_world_xy = _normalize_bbox_xyxy((bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]))
        wall_records.append(
            WallRecord(
                name=name,
                category=category,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                bbox_world_xy=bbox_world_xy,
                is_structural_wall=("walls" in low_name or low_category == "walls" or low_category == "wall"),
            )
        )
    wall_records.sort(key=lambda record: (not record.is_structural_wall, record.category, record.name))
    return wall_records


def _resolve_room_bbox(scene, room_name: str, wall_bboxes_xyxy: list[tuple[float, float, float, float]]):
    seg = _segmap_get(scene)
    if seg is None or not hasattr(seg, "room_ins_id_to_ins_name"):
        return {
            "room_instance": room_name,
            "room_id": None,
            "bbox_map_rc": None,
            "bbox_world_xy": None,
            "expanded_bbox_world_xy": None,
            "pixel_count": 0,
            "source": "unavailable",
        }

    room_id = None
    for candidate_id, candidate_name in seg.room_ins_id_to_ins_name.items():
        if str(candidate_name) == str(room_name):
            room_id = int(candidate_id)
            break
    if room_id is None:
        return {
            "room_instance": room_name,
            "room_id": None,
            "bbox_map_rc": None,
            "bbox_world_xy": None,
            "expanded_bbox_world_xy": None,
            "pixel_count": 0,
            "source": "room_not_found",
        }

    bbox_info = _segmap_room_bbox_from_maps(scene, room_id)
    bbox_world_xy = bbox_info.get("bbox_world_xy")
    expanded_bbox_world_xy = None
    if bbox_world_xy is not None:
        if wall_bboxes_xyxy:
            expanded_bbox_world_xy = _expand_bbox_until_wall_touch(
                bbox_world_xy,
                wall_bboxes_xyxy=wall_bboxes_xyxy,
                expansion_ratio=ROOM_BBOX_EXPANSION_RATIO,
                expansion_min=ROOM_BBOX_EXPANSION_MIN,
            )
        else:
            expanded_bbox_world_xy = _expand_bbox_xyxy(
                bbox_world_xy,
                expansion_ratio=ROOM_BBOX_EXPANSION_RATIO,
                expansion_min=ROOM_BBOX_EXPANSION_MIN,
            )
    return {
        "room_instance": room_name,
        "room_id": room_id,
        "bbox_map_rc": bbox_info.get("bbox_map_rc"),
        "bbox_world_xy": _normalize_bbox_xyxy(bbox_world_xy) if bbox_world_xy is not None else None,
        "expanded_bbox_world_xy": _normalize_bbox_xyxy(expanded_bbox_world_xy) if expanded_bbox_world_xy is not None else None,
        "bbox_area_m2": _bbox_area_xyxy(bbox_world_xy) if bbox_world_xy is not None else None,
        "expanded_bbox_area_m2": _bbox_area_xyxy(expanded_bbox_world_xy) if expanded_bbox_world_xy is not None else None,
        "pixel_count": int(bbox_info.get("pixel_count", 0)),
        "source": "seg_map_expanded_with_walls" if wall_bboxes_xyxy else "seg_map_expanded",
    }


def _is_door_named_object(obj) -> bool:
    name = str(getattr(obj, "name", "")).lower()
    return "door" in name


def _collect_room_objects(scene, room_name: str) -> list[RuntimeObjectRecord]:
    robot_names = {robot.name for robot in getattr(scene, "robots", [])}
    room_objects = []
    for obj in _get_scene_objects(scene):
        if obj.name in robot_names:
            continue
        if _is_door_named_object(obj):
            continue
        category = str(getattr(obj, "category", "object"))
        in_rooms = tuple(str(room) for room in (getattr(obj, "in_rooms", None) or []))
        is_structure = category in {"floors", "walls", "ceilings", "door", "sliding_door"}
        if room_name not in in_rooms and not is_structure:
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
            except Exception as exc:
                _log_exception(f"Failed to read open state for object {obj.name}", exc)
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
    robots = list(getattr(env, "robots", []) or [])
    if robots:
        # We only read the existing robot pose as a reference and do not place the agent here.
        robot_pos, _ = robots[0].get_position_orientation()
        vals = robot_pos.cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    return None


def _select_floor(room_objects: list[RuntimeObjectRecord], floor_name: str | None, agent_pos):
    floors = [obj for obj in room_objects if obj.category == "floors"]
    if floor_name is not None:
        for floor in floors:
            if floor.name == floor_name:
                return floor
        raise ValueError(f"Floor '{floor_name}' not found among loaded room objects.")
    containing = [
        floor
        for floor in floors
        if floor.bbox_min[0] <= agent_pos[0] <= floor.bbox_max[0]
        and floor.bbox_min[1] <= agent_pos[1] <= floor.bbox_max[1]
    ]
    if containing:
        containing.sort(key=lambda floor: floor.footprint_area, reverse=True)
        return containing[0]
    if floors:
        floors.sort(key=lambda floor: (_distance_xy(floor.center, agent_pos), -floor.footprint_area))
        return floors[0]
    raise ValueError("No floor object found in loaded room.")


def _point_is_free(
    point_xy,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    clearance: float,
    room_bbox_xyxy=None,
    ignore_labels: set[str] | None = None,
    agent_pos=None,
    min_agent_distance: float = 0.0,
) -> bool:
    ignore_labels = ignore_labels or set()
    if agent_pos is not None and _distance_xy(point_xy, agent_pos[:2]) < float(min_agent_distance):
        return False
    if not (
        floor_record.bbox_min[0] + clearance <= float(point_xy[0]) <= floor_record.bbox_max[0] - clearance
        and floor_record.bbox_min[1] + clearance <= float(point_xy[1]) <= floor_record.bbox_max[1] - clearance
    ):
        return False
    if room_bbox_xyxy is not None and not _point_inside_bbox_xyxy(point_xy, room_bbox_xyxy, margin=-clearance):
        return False
    for blocker in blockers:
        if blocker.label in ignore_labels:
            continue
        if _point_inside_bbox_xy(point_xy, blocker.bbox_min, blocker.bbox_max, margin=clearance):
            return False
    return True


def _diagnose_point_availability(
    point_xy,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    clearance: float,
    room_bbox_xyxy=None,
    ignore_labels: set[str] | None = None,
    agent_pos=None,
    min_agent_distance: float = 0.0,
) -> dict:
    ignore_labels = ignore_labels or set()
    point_xy = (float(point_xy[0]), float(point_xy[1]))
    diag = {
        "point_xy": [point_xy[0], point_xy[1]],
        "clearance": float(clearance),
        "agent_distance_ok": True,
        "floor_ok": True,
        "room_bbox_ok": True,
        "blocking_objects": [],
        "ok": True,
        "failure_reason": None,
    }

    if agent_pos is not None and _distance_xy(point_xy, agent_pos[:2]) < float(min_agent_distance):
        diag["agent_distance_ok"] = False
        diag["ok"] = False
        diag["failure_reason"] = "too_close_to_agent"

    if not (
        floor_record.bbox_min[0] + clearance <= point_xy[0] <= floor_record.bbox_max[0] - clearance
        and floor_record.bbox_min[1] + clearance <= point_xy[1] <= floor_record.bbox_max[1] - clearance
    ):
        diag["floor_ok"] = False
        diag["ok"] = False
        diag["failure_reason"] = "outside_floor_with_clearance"

    if room_bbox_xyxy is not None and not _point_inside_bbox_xyxy(point_xy, room_bbox_xyxy, margin=-clearance):
        diag["room_bbox_ok"] = False
        diag["ok"] = False
        if diag["failure_reason"] is None:
            diag["failure_reason"] = "outside_room_bbox_with_clearance"

    for blocker in blockers:
        if blocker.label in ignore_labels:
            continue
        if _point_inside_bbox_xy(point_xy, blocker.bbox_min, blocker.bbox_max, margin=clearance):
            diag["blocking_objects"].append(
                {
                    "name": blocker.name,
                    "category": blocker.category,
                    "bbox_min": [float(v) for v in blocker.bbox_min],
                    "bbox_max": [float(v) for v in blocker.bbox_max],
                    "footprint_area": float(blocker.footprint_area),
                    "distance_to_point_xy": round(_distance_xy(point_xy, blocker.center[:2]), 4),
                }
            )

    if diag["blocking_objects"]:
        diag["ok"] = False
        if diag["failure_reason"] is None:
            diag["failure_reason"] = "intersects_blockers_with_clearance"
    return diag


def _describe_runtime_object(record: RuntimeObjectRecord, agent_pos=None) -> dict:
    payload = {
        "name": record.name,
        "category": record.category,
        "center": [float(v) for v in record.center],
        "bbox_min": [float(v) for v in record.bbox_min],
        "bbox_max": [float(v) for v in record.bbox_max],
        "extents": [float(v) for v in record.extents],
        "footprint_area": float(record.footprint_area),
        "has_open_state": bool(record.has_open_state),
        "open_state": record.open_state,
    }
    if agent_pos is not None:
        payload["distance_to_agent_xy"] = round(_distance_xy(record.center[:2], agent_pos[:2]), 4)
    return payload

def _generate_free_positions(
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    agent_pos,
    count: int,
    room_bbox_xyxy=None,
    clearance: float = BALL_RADIUS + BALL_CLEARANCE,
) -> list[list[float]]:
    x_min = floor_record.bbox_min[0] + clearance
    x_max = floor_record.bbox_max[0] - clearance
    y_min = floor_record.bbox_min[1] + clearance
    y_max = floor_record.bbox_max[1] - clearance
    if room_bbox_xyxy is not None:
        rxmin, rymin, rxmax, rymax = _normalize_bbox_xyxy(room_bbox_xyxy)
        x_min = max(x_min, rxmin + clearance)
        x_max = min(x_max, rxmax - clearance)
        y_min = max(y_min, rymin + clearance)
        y_max = min(y_max, rymax - clearance)
    if x_min >= x_max or y_min >= y_max:
        return []

    candidates = []
    x = x_min
    while x <= x_max + 1e-6:
        y = y_min
        while y <= y_max + 1e-6:
            pos_xy = (float(x), float(y))
            dist_to_agent = _distance_xy(pos_xy, agent_pos)
            if dist_to_agent >= MIN_BALL_DISTANCE_FROM_AGENT and _point_is_free(
                pos_xy,
                floor_record,
                blockers,
                clearance=clearance,
                room_bbox_xyxy=room_bbox_xyxy,
                agent_pos=agent_pos,
                min_agent_distance=MIN_BALL_DISTANCE_FROM_AGENT,
            ):
                candidates.append(
                    (
                        round(dist_to_agent, 6),
                        [float(x), float(y), floor_record.bbox_max[2] + BALL_RADIUS],
                    )
                )
            y += GRID_STEP
        x += GRID_STEP
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return [pos for _, pos in candidates[:count]]


def _find_neighbor_position(
    base_pos,
    floor_record,
    blockers,
    clearance: float,
    room_bbox_xyxy=None,
    agent_pos=None,
    min_agent_distance: float = 0.0,
) -> list[float] | None:
    offsets = [
        (SEMANTIC_FAULT_CONFUSER_MIN_SEPARATION_M, 0.0),
        (-SEMANTIC_FAULT_CONFUSER_MIN_SEPARATION_M, 0.0),
        (0.0, SEMANTIC_FAULT_CONFUSER_MIN_SEPARATION_M),
        (0.0, -SEMANTIC_FAULT_CONFUSER_MIN_SEPARATION_M),
        (0.36, 0.36),
        (-0.36, 0.36),
        (0.36, -0.36),
        (-0.36, -0.36),
        (0.6, 0.0),
        (-0.6, 0.0),
        (0.0, 0.6),
        (0.0, -0.6),
    ]
    for offset in offsets:
        neighbor_xy = (float(base_pos[0]) + offset[0], float(base_pos[1]) + offset[1])
        if _point_is_free(
            neighbor_xy,
            floor_record,
            blockers,
            clearance=clearance,
            room_bbox_xyxy=room_bbox_xyxy,
            agent_pos=agent_pos,
            min_agent_distance=min_agent_distance,
        ):
            return [float(neighbor_xy[0]), float(neighbor_xy[1]), float(base_pos[2])]
    return None


def _load_available_categories(keys_json: str | None) -> set[str]:
    if keys_json and os.path.exists(keys_json):
        raw = _read_json(keys_json)
        if isinstance(raw, list):
            return {str(item) for item in raw}
    if os.path.exists(DEFAULT_OBJECT_INVENTORY):
        inventory = _read_json(DEFAULT_OBJECT_INVENTORY)
        providers = inventory.get("providers", inventory)
        return {str(key).split("-", 1)[0] for key in providers}
    return set()


def _characteristic_size_from_volume(volume: float | int | None) -> float | None:
    if volume is None:
        return None
    volume = float(volume)
    if volume <= 0.0:
        return None
    return float(volume ** (1.0 / 3.0))


def _count_target_label(category: str) -> str:
    return str(category).replace("_", " ")


@functools.lru_cache(maxsize=1)
def _load_object_inventory_payload() -> dict:
    if os.path.exists(DEFAULT_OBJECT_INVENTORY):
        payload = _read_json(DEFAULT_OBJECT_INVENTORY)
        if isinstance(payload, dict):
            return payload
    return {}


@functools.lru_cache(maxsize=1)
def _bounding_box_sizes_by_model() -> dict[str, list[float]]:
    payload = _load_object_inventory_payload()
    raw = payload.get("bounding_box_sizes", {})
    if not isinstance(raw, dict):
        return {}
    normalized = {}
    for model, dims in raw.items():
        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            continue
        try:
            normalized[str(model)] = [float(dims[0]), float(dims[1]), float(dims[2])]
        except Exception:
            continue
    return normalized


def _pick_count_target_model_and_dims(category: str) -> tuple[str, list[float]] | None:
    bbox_sizes = _bounding_box_sizes_by_model()
    passing = []
    for model in _list_models_for_category(category):
        dims = bbox_sizes.get(str(model))
        if not dims:
            continue
        if max(dims) > COUNT_TARGET_MAX_EDGE_M:
            continue
        if all(edge < COUNT_TARGET_MIN_EDGE_M for edge in dims):
            continue
        passing.append((str(model), list(dims)))
    if not passing:
        return None
    passing.sort(key=lambda item: (max(item[1]), sum(item[1]), item[0]))
    return passing[0]


def _build_count_target_candidates(keys_json: str | None) -> list[dict]:
    available = _load_available_categories(keys_json)
    specs = _read_json(DEFAULT_AVG_CATEGORY_SPECS) if os.path.exists(DEFAULT_AVG_CATEGORY_SPECS) else {}
    candidates = []
    for category in sorted(available):
        model_dims = _pick_count_target_model_and_dims(category)
        if model_dims is None:
            continue
        model, dims = model_dims
        volume = specs.get(category, {}).get("volume") if isinstance(specs, dict) else None
        characteristic_size_m = max(float(edge) for edge in dims)
        candidates.append(
            {
                "category": category,
                "estimated_volume": float(volume) if volume is not None else None,
                "characteristic_size_m": characteristic_size_m,
                "bbox_size_m": [float(edge) for edge in dims],
                "representative_model": model,
                "display_name": _count_target_label(category),
            }
        )
    return candidates


def _sample_count_target(count_target_candidates: list[dict], rng: random.Random) -> dict:
    if not count_target_candidates:
        raise ValueError(
            "No count-target category found in keys.json with local models whose bbox edges all stay within 25cm and are not all below 5cm."
        )
    sampled = dict(rng.choice(count_target_candidates))
    sampled["sampling_source"] = "keys_json_filtered_by_local_model_bbox_edges"
    return sampled


def _sample_count_target_with_seed(count_target_candidates: list[dict], seed: int) -> dict:
    return _sample_count_target(count_target_candidates, random.Random(int(seed)))


def _scoped_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "parts": list(parts),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int(zlib.crc32(payload) & 0x7FFFFFFF)


def _count_target_seed_for_question(
    *,
    base_seed: int,
    scene_name: str,
    room_name: str,
    task_type: str,
    q_idx: int,
) -> int:
    # Mirrors the batch_occlusion.py style: derive a deterministic per-question
    # seed from semantic identifiers instead of reusing only the global seed.
    return _scoped_seed(base_seed, scene_name, room_name, task_type, int(q_idx), "count_target")


def _load_clip_top3_neighbors(path: str | None) -> dict[str, list[dict]]:
    if not path or not os.path.exists(path):
        return {}
    payload = _read_json(path)
    neighbors = payload.get("neighbors", payload)
    if not isinstance(neighbors, dict):
        return {}
    return neighbors


def _sample_semantic_confuser_candidates_for_target(
    target_category: str,
    clip_neighbors: dict[str, list[dict]],
    rng: random.Random,
    count_target_candidates: list[dict],
) -> list[dict]:
    neighbors = clip_neighbors.get(target_category, [])
    valid_neighbors = []
    for item in neighbors[:3]:
        category = str(item.get("item", ""))
        if not category or category == target_category:
            continue
        if not _list_models_for_category(category):
            continue
        valid_neighbors.append(
            {
                "category": category,
                "clip_score": float(item.get("score", 0.0)),
                "source": "keys_clip_top3",
            }
        )
    if valid_neighbors:
        rng.shuffle(valid_neighbors)
        return [dict(item) for item in valid_neighbors]

    fallback_candidates = [
        candidate
        for candidate in count_target_candidates
        if candidate["category"] != target_category
    ]
    if not fallback_candidates:
        return []
    rng.shuffle(fallback_candidates)
    fallback_neighbors = []
    for sampled in fallback_candidates[:3]:
        item = dict(sampled)
        item["source"] = "fallback_count_target_pool"
        fallback_neighbors.append(item)
    return fallback_neighbors


def _list_models_for_category(category: str) -> list[str]:
    category_dir = DEFAULT_OBJECT_DATASET_ROOT / str(category)
    if not category_dir.exists():
        return []
    return sorted(path.name for path in category_dir.iterdir() if path.is_dir())


def _get_model_for_category(category: str, seed: int) -> str:
    models = _list_models_for_category(category)
    if not models:
        raise FileNotFoundError(f"No local models found for category '{category}' under {DEFAULT_OBJECT_DATASET_ROOT}")
    rng = random.Random(f"{category}:{seed}")
    return models[rng.randrange(len(models))]


def _get_candidate_models_for_category(category: str, seed: int) -> list[str]:
    models = _list_models_for_category(category)
    if not models:
        raise FileNotFoundError(f"No local models found for category '{category}' under {DEFAULT_OBJECT_DATASET_ROOT}")
    failed_models = FAILED_RENDER_MODELS_BY_CATEGORY.get(category, set())
    candidates = [model for model in models if model not in failed_models]
    if not candidates:
        return []
    rng = random.Random(f"{category}:{seed}")
    rng.shuffle(candidates)
    return candidates


def _step_sim(steps: int = 10) -> None:
    for _ in range(max(steps, 0)):
        og.sim.step()


def _ensure_object_ready(obj, max_steps: int = 10) -> None:
    if obj is None:
        return
    if getattr(obj, "initialized", False):
        return
    if not og.sim.is_playing():
        return
    for _ in range(max(int(max_steps), 0)):
        if getattr(obj, "initialized", False):
            return
        og.sim.step()


def _configure_spawned_object_for_direct_placement(obj, force: bool = False) -> None:
    if (not DIRECT_PLACEMENT_MODE and not force) or obj is None:
        return
    try:
        obj.disable_gravity()
    except Exception:
        pass
    try:
        obj.visual_only = True
    except Exception:
        pass
    if hasattr(obj, "keep_still"):
        try:
            obj.keep_still()
        except Exception:
            pass


def _park_position(slot_idx: int) -> th.Tensor:
    return th.tensor(
        [RENDER_PARK_X + slot_idx * 2.5, RENDER_PARK_Y + slot_idx * 1.5, RENDER_PARK_Z],
        dtype=th.float32,
    )


def _park_object(obj, slot_idx: int, settle_steps: int = SIM_STEP_MINIMAL, max_retries: int = 4) -> bool:
    last_exc: Exception | None = None
    for attempt_idx in range(max(1, int(max_retries))):
        if settle_steps > 0:
            _step_sim(settle_steps)
        try:
            obj.set_position_orientation(
                position=_park_position(slot_idx),
                orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            )
            if hasattr(obj, "keep_still"):
                obj.keep_still()
            return True
        except Exception as exc:
            last_exc = exc
            _log_exception(
                f"Failed to park object {getattr(obj, 'name', '<unknown>')} "
                f"(attempt {attempt_idx + 1}/{max(1, int(max_retries))})",
                exc,
            )
            _step_sim(SIM_STEP_MINIMAL)
    return False


def _safe_remove_scene_object(scene, obj, reason: str | None = None) -> None:
    if obj is None:
        return
    obj_name = getattr(obj, "name", "<unknown>")

    # Failed render objects may never finish initialization, which makes the
    # normal scene.remove_object() path unsafe because it dumps simulator state.
    # We therefore aggressively clean up the init queue, registry, and prims.
    try:
        queued = getattr(og.sim, "_objects_to_initialize", None)
        if queued is not None:
            og.sim._objects_to_initialize = [item for item in queued if getattr(item, "name", None) != obj_name]
    except Exception as exc:
        _log_exception(f"Failed to prune init queue for object {obj_name}", exc)

    articulation_root_path = getattr(obj, "articulation_root_path", None)
    if articulation_root_path:
        try:
            ControllableObjectViewAPI.clear_object(articulation_root_path)
        except Exception:
            pass

    try:
        if scene.object_registry.object_is_registered(obj):
            scene.object_registry.remove(obj)
    except Exception as exc:
        prefix = reason or f"Failed to detach object {obj_name} from registry"
        _log_exception(prefix, exc)

    prim_paths = []
    for prim_path in (articulation_root_path, getattr(obj, "prim_path", None)):
        if prim_path and prim_path not in prim_paths:
            prim_paths.append(prim_path)

    try:
        obj.remove()
    except Exception:
        pass

    for prim_path in prim_paths:
        try:
            delete_or_deactivate_prim(prim_path)
        except Exception as exc:
            prefix = reason or f"Failed to delete prim for object {obj_name}"
            _log_exception(f"{prefix} at prim_path {prim_path}", exc)

    try:
        og.sim.update_handles()
    except Exception:
        pass


def _spawn_render_dataset_object(
    scene,
    category: str,
    seed: int,
    idx: int,
    name_prefix: str,
    fixed_model: str | None = None,
    force_direct_placement: bool = False,
):
    name = f"{name_prefix}{category}_{idx}"
    stale_obj = scene.object_registry("name", name)
    if stale_obj is not None and not getattr(stale_obj, "initialized", False):
        _safe_remove_scene_object(scene, stale_obj, reason=f"Removing stale uninitialized render object {name}")
        stale_obj = scene.object_registry("name", name)
    if stale_obj is not None:
        if getattr(stale_obj, "initialized", False):
            return stale_obj
        raise RuntimeError(f"Stale render object {name} could not be removed cleanly")

    if fixed_model is not None:
        candidate_models = [str(fixed_model)]
    else:
        candidate_models = _get_candidate_models_for_category(category, seed=seed + idx)
    if not candidate_models:
        failed_models = sorted(FAILED_RENDER_MODELS_BY_CATEGORY.get(category, set()))
        if failed_models:
            raise RuntimeError(
                f"All local render models for category '{category}' were previously marked unusable: {failed_models}"
            )
        raise RuntimeError(f"No usable render models remain for category '{category}'")

    last_exc: Exception | None = None
    for model in candidate_models:
        use_direct_placement = bool(DIRECT_PLACEMENT_MODE or force_direct_placement)
        obj = DatasetObject(
            name=name,
            category=category,
            model=model,
            fixed_base=use_direct_placement,
            visual_only=use_direct_placement,
            kinematic_only=None,
        )
        paused_for_direct_placement = bool(use_direct_placement and og.sim.is_playing())
        try:
            if paused_for_direct_placement:
                og.sim.stop()
            scene.add_object(obj)
            if not paused_for_direct_placement:
                _ensure_object_ready(obj)
            if paused_for_direct_placement:
                og.sim.play()
                _ensure_object_ready(obj, max_steps=20)
            _configure_spawned_object_for_direct_placement(obj, force=force_direct_placement)
            if not _park_object(obj, idx):
                raise RuntimeError(
                    f"Failed to park render object {name} for category={category} model={model}"
                )
            return obj
        except Exception as exc:
            if paused_for_direct_placement and not og.sim.is_playing():
                try:
                    og.sim.play()
                except Exception:
                    pass
            last_exc = exc
            _log_exception(
                f"Render object initialization failed for category={category} model={model} name={name}",
                exc,
            )
            _safe_remove_scene_object(
                scene,
                obj,
                reason=f"Cleaning up failed render object {name} (category={category}, model={model})",
            )
            if scene.object_registry("name", name) is not None:
                raise RuntimeError(
                    f"Failed render object {name} still exists in scene after cleanup; aborting retries for this slot"
                ) from exc
            if _should_blacklist_render_model(exc):
                FAILED_RENDER_MODELS_BY_CATEGORY.setdefault(category, set()).add(model)

    raise RuntimeError(
        f"Unable to create a stable render object for category={category}; "
        f"tried models={candidate_models}"
    ) from last_exc

def _flatten_object_pool_cache(cache: dict[str, list[object]]) -> list[object]:
    objects: list[object] = []
    for category in sorted(cache):
        objects.extend(cache[category])
    return objects


def _object_pool_cache_key(category: str, fixed_model: str | None = None) -> str:
    if fixed_model is None:
        return str(category)
    return f"{category}::{fixed_model}"


def _repark_object_caches(
    target_cache: dict[str, list[object]],
    confuser_cache: dict[str, list[object]],
    hidden_box_cache: dict[str, object] | None = None,
) -> None:
    slot_idx = 0
    for obj in _flatten_object_pool_cache(target_cache):
        _park_object(obj, slot_idx)
        slot_idx += 1
    for obj in _flatten_object_pool_cache(confuser_cache):
        _park_object(obj, slot_idx)
        slot_idx += 1
    for obj in (hidden_box_cache or {}).values():
        _park_object(obj, slot_idx)
        slot_idx += 1


def _ensure_render_dataset_object_pool(
    scene,
    cache: dict[str, list[object]],
    category: str,
    count: int,
    seed: int,
    name_prefix: str,
    fixed_model: str | None = None,
    force_direct_placement: bool = False,
) -> list[object]:
    start_time = time.perf_counter()
    count = max(0, min(int(count), MAX_RANDOM_BALL_COUNT))
    cache_key = _object_pool_cache_key(category, fixed_model)
    existing = cache.setdefault(cache_key, [])
    existing_before = len(existing)
    while len(existing) < count:
        idx = len(existing)
        obj = _spawn_render_dataset_object(
            scene=scene,
            category=category,
            seed=seed,
            idx=idx,
            name_prefix=name_prefix,
            fixed_model=fixed_model,
            force_direct_placement=force_direct_placement,
        )
        existing.append(obj)
    _step_sim(SIM_STEP_MINIMAL)
    _log_timing(
        "render_pool",
        category=category,
        fixed_model=fixed_model,
        requested_count=count,
        cached_before=existing_before,
        created_count=max(0, len(existing) - existing_before),
        returned_count=min(len(existing), count),
        name_prefix=name_prefix,
        direct=int(bool(force_direct_placement)),
        elapsed_s=time.perf_counter() - start_time,
    )
    return existing[:count]


def _should_blacklist_render_model(exc: Exception) -> bool:
    if isinstance(exc, (FileNotFoundError, ValueError, AttributeError)):
        return True
    message = str(exc).lower()
    return "nonetype" in message or "physics" in message or "max_shapes" in message or "count" in message


def _should_blacklist_hidden_box_model(exc: Exception) -> bool:
    if isinstance(exc, (FileNotFoundError, ValueError, AttributeError)):
        return True
    message = str(exc).lower()
    return "nonetype" in message or "physics" in message


def _hidden_box_metadata(category: str, model: str) -> dict:
    key = (str(category), str(model))
    cached = HIDDEN_BOX_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    metadata_path = DEFAULT_OBJECT_DATASET_ROOT / str(category) / str(model) / "misc" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Hidden-box metadata not found: {metadata_path}")
    metadata = _read_json(str(metadata_path))
    HIDDEN_BOX_METADATA_CACHE[key] = metadata
    return metadata


def _hidden_box_extents_from_metadata(category: str, model: str) -> tuple[float, float, float]:
    metadata = _hidden_box_metadata(category, model)
    bbox_size = metadata.get("bbox_size")
    arr = np.array(bbox_size, dtype=float).reshape(-1) if bbox_size is not None else np.zeros(3, dtype=float)
    if arr.size != 3 or np.any(arr <= 1e-4):
        raise ValueError(f"Invalid bbox_size in hidden-box metadata for category={category} model={model}: {bbox_size}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _spawn_hidden_box_container_at_pose(
    scene,
    cache: dict[str, object],
    entry_name: str,
    category: str,
    model: str,
    position,
    orientation,
    seed: int,
):
    cached_obj = cache.get(entry_name)
    if cached_obj is not None and getattr(cached_obj, "initialized", False):
        _set_object_pose(cached_obj, position, orientation=orientation, keep_still=True)
        _force_container_lid_visible(cached_obj)
        return cached_obj, getattr(cached_obj, "model", model)

    stale_obj = scene.object_registry("name", entry_name)
    if stale_obj is not None and not getattr(stale_obj, "initialized", False):
        _safe_remove_scene_object(scene, stale_obj, reason=f"Removing stale hidden-box container {entry_name}")
        stale_obj = scene.object_registry("name", entry_name)
    if stale_obj is not None and getattr(stale_obj, "initialized", False):
        cache[entry_name] = stale_obj
        _set_object_pose(stale_obj, position, orientation=orientation, keep_still=True)
        _force_container_lid_visible(stale_obj)
        return stale_obj, getattr(stale_obj, "model", model)
    if stale_obj is not None:
        raise RuntimeError(f"Stale hidden-box container {entry_name} could not be removed cleanly")

    fallback_models = _get_candidate_models_for_category(category, seed=seed)
    candidate_models = [model] + [candidate for candidate in fallback_models if candidate != model]
    last_exc: Exception | None = None

    for candidate_model in candidate_models:
        use_direct_placement = True
        obj = DatasetObject(
            name=entry_name,
            category=category,
            model=candidate_model,
            fixed_base=use_direct_placement,
            visual_only=use_direct_placement,
            kinematic_only=None,
        )
        paused_for_direct_placement = bool(use_direct_placement and og.sim.is_playing())
        try:
            if paused_for_direct_placement:
                og.sim.stop()
            scene.add_object(obj)
            if not paused_for_direct_placement:
                _ensure_object_ready(obj)
            _set_object_pose(obj, position, orientation=orientation, keep_still=True)
            if paused_for_direct_placement:
                og.sim.play()
                _ensure_object_ready(obj, max_steps=20)
            _configure_spawned_object_for_direct_placement(obj, force=True)
            cache[entry_name] = obj
            _force_container_lid_visible(obj)
            _step_sim(SIM_STEP_MINIMAL)
            return obj, candidate_model
        except Exception as exc:
            if paused_for_direct_placement and not og.sim.is_playing():
                try:
                    og.sim.play()
                except Exception:
                    pass
            last_exc = exc
            _log_exception(
                f"Hidden-box container initialization failed for category={category} "
                f"model={candidate_model} name={entry_name}",
                exc,
            )
            _safe_remove_scene_object(
                scene,
                obj,
                reason=(
                    f"Cleaning up failed hidden-box container {entry_name} "
                    f"(category={category}, model={candidate_model})"
                ),
            )
            if scene.object_registry("name", entry_name) is not None:
                raise RuntimeError(
                    f"Failed hidden-box container {entry_name} still exists in scene after cleanup; "
                    "aborting retries for this slot"
                ) from exc
            if _should_blacklist_hidden_box_model(exc):
                FAILED_RENDER_MODELS_BY_CATEGORY.setdefault(category, set()).add(candidate_model)

    raise RuntimeError(
        f"Unable to create a stable hidden-box container for category={category}; "
        f"tried models={candidate_models}"
    ) from last_exc


def _find_container_lid_links(container_obj) -> list[object]:
    links = list((getattr(container_obj, "links", {}) or {}).values())
    if not links:
        return []

    category = str(getattr(container_obj, "category", ""))
    model = str(getattr(container_obj, "model", ""))
    lid_names = set()
    try:
        metadata = _hidden_box_metadata(category, model)
        link_tags = metadata.get("link_tags") or {}
        for link_name in link_tags:
            if any(keyword in str(link_name).lower() for keyword in CONTAINER_LID_LINK_KEYWORDS):
                lid_names.add(str(link_name).lower())
    except Exception:
        pass

    named_links = []
    seen_ids = set()
    for link in links:
        link_name = str(getattr(link, "name", "")).lower()
        if any(keyword in link_name for keyword in CONTAINER_LID_LINK_KEYWORDS) or link_name in lid_names:
            named_links.append(link)
            seen_ids.add(id(link))
    if named_links:
        return named_links

    try:
        container_record = _runtime_record_from_obj(container_obj)
        center_z = float(container_record.center[2])
        width = max(float(container_record.extents[0]), 1e-4)
        depth = max(float(container_record.extents[1]), 1e-4)
        height = max(float(container_record.extents[2]), 1e-4)
    except Exception:
        return []

    heuristic = []
    for link in links:
        if id(link) in seen_ids:
            continue
        try:
            bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in link.aabb]
        except Exception:
            continue
        extent_x = float(bbox_max[0] - bbox_min[0])
        extent_y = float(bbox_max[1] - bbox_min[1])
        extent_z = float(bbox_max[2] - bbox_min[2])
        link_center_z = float((bbox_min[2] + bbox_max[2]) * 0.5)
        if link_center_z <= center_z + 0.1 * height:
            continue
        lateral_cover = max(extent_x * extent_y, extent_x * extent_z, extent_y * extent_z)
        if max(extent_x, extent_y) < 0.45 * min(width, depth):
            continue
        if extent_z > 0.45 * height:
            continue
        heuristic.append((link_center_z, lateral_cover, -extent_z, link))

    heuristic.sort(reverse=True)
    return [item[3] for item in heuristic[:1]]


def _force_container_lid_visible(container_obj) -> None:
    if container_obj is None:
        return
    for link in _find_container_lid_links(container_obj):
        try:
            link.visible = True
        except Exception as exc:
            _log_exception(
                f"Failed to force lid visible for {getattr(container_obj, 'name', '<unknown>')}",
                exc,
            )


def _set_object_pose(obj, position, orientation=None, keep_still: bool = True, force_direct_placement: bool = False) -> None:
    _ensure_object_ready(obj)
    orientation = [0.0, 0.0, 0.0, 1.0] if orientation is None else orientation
    obj.set_position_orientation(
        position=th.tensor([float(v) for v in position], dtype=th.float32),
        orientation=th.tensor([float(v) for v in orientation], dtype=th.float32),
    )
    _configure_spawned_object_for_direct_placement(obj, force=force_direct_placement)
    if keep_still and hasattr(obj, "keep_still"):
        obj.keep_still()


def _place_ball(ball, position, keep_still: bool = True, force_direct_placement: bool = False) -> None:
    _set_object_pose(ball, position, keep_still=keep_still, force_direct_placement=force_direct_placement)


def _place_confuser_on_floor(confuser, target_position, floor_record: RuntimeObjectRecord) -> None:
    try:
        bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in confuser.aabb]
        half_height = max(float(bbox_max[2] - bbox_min[2]) / 2.0, 0.02)
    except Exception as exc:
        _log_exception(f"Failed to read confuser AABB for {getattr(confuser, 'name', '<unknown>')}", exc)
        half_height = 0.12
    z = max(float(target_position[2]), float(floor_record.bbox_max[2]) + half_height + 0.01)
    _set_object_pose(
        confuser,
        [float(target_position[0]), float(target_position[1]), z],
        force_direct_placement=RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT,
    )


def _try_place_inside_container(ball, container_obj, fallback_position, desired_open: bool = False) -> bool | None:
    del desired_open
    target_position = [float(v) for v in fallback_position]
    try:
        container_record = _runtime_record_from_obj(container_obj)
        target_position = [float(v) for v in container_record.center]
    except Exception as exc:
        _log_exception(
            f"Failed to read container center for {getattr(container_obj, 'name', '<unknown>')}",
            exc,
        )
    _place_ball(ball, target_position, force_direct_placement=True)
    return None


def _restore_container_open_state(container_obj, original_open: bool | None) -> None:
    if original_open is None:
        return
    states = getattr(container_obj, "states", {})
    if object_states.Open not in states:
        return
    try:
        states[object_states.Open].set_value(bool(original_open))
        _step_sim(1)
    except Exception as exc:
        _log_exception(f"Failed to restore container state for {getattr(container_obj, 'name', '<unknown>')}", exc)
        pass


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


def _yaw_to_quaternion_xyzw(yaw_deg: float) -> list[float]:
    half_yaw = math.radians(float(yaw_deg)) * 0.5
    return [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]


def _quaternion_xyzw_to_front_xy(quat_xyzw) -> tuple[float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(-math.cos(yaw)), float(-math.sin(yaw))


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


def _trav_map_floor_image(scene, floor_idx: int = 0, scene_name: str | None = None, basename: str = "floor_trav"):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is not None and getattr(trav_map, "floor_map", None) is not None and 0 <= floor_idx < len(trav_map.floor_map):
        floor_img = trav_map.floor_map[floor_idx]
        if hasattr(floor_img, "detach"):
            floor_img = floor_img.detach()
        if getattr(floor_img, "device", None) is not None and floor_img.device.type != "cpu":
            floor_img = floor_img.cpu()
        return trav_map, Image.fromarray(floor_img.numpy()).convert("L")

    resolved_scene = scene_name or getattr(scene, "scene_model", None) or getattr(scene, "model", None)
    if not resolved_scene:
        return trav_map, None
    img_path = SCRIPT_DIR / "datasets" / "behavior-1k-assets" / "scenes" / str(resolved_scene) / "layout" / f"{basename}_{int(floor_idx)}.png"
    if not img_path.exists():
        return trav_map, None
    return trav_map, Image.open(img_path).convert("L")


def _infer_floor_idx(floor_record: RuntimeObjectRecord) -> int:
    suffix = str(floor_record.name).rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _world_to_map_rc(trav_map, xy) -> tuple[int, int] | None:
    if trav_map is None:
        return None
    try:
        rc_arr = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    except Exception as exc:
        _log_exception("world_to_map_rc", exc)
        return None
    return int(round(float(rc_arr[0]))), int(round(float(rc_arr[1])))


def _world_to_plot_xy(trav_map, xy) -> tuple[float, float]:
    rc = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    return float(rc[1]), float(rc[0])


def _clip_map_rc(map_img, rc) -> tuple[int, int]:
    if isinstance(map_img, Image.Image):
        row = min(max(int(round(float(rc[0]))), 0), map_img.height - 1)
        col = min(max(int(round(float(rc[1]))), 0), map_img.width - 1)
        return row, col
    row = int(np.clip(int(round(float(rc[0]))), 0, map_img.shape[0] - 1))
    col = int(np.clip(int(round(float(rc[1]))), 0, map_img.shape[1] - 1))
    return row, col


def _clip_rc(image: Image.Image, rc) -> tuple[int, int]:
    row = min(max(int(round(float(rc[0]))), 0), image.height - 1)
    col = min(max(int(round(float(rc[1]))), 0), image.width - 1)
    return row, col


def _map_pixel_value(map_img, row: int, col: int) -> float:
    if isinstance(map_img, Image.Image):
        pixel = map_img.getpixel((int(col), int(row)))
        if isinstance(pixel, tuple):
            return float(pixel[0])
        return float(pixel)
    return float(map_img[int(row), int(col)])


def _eye_xy_is_traversable(trav_map, map_img, eye_xy) -> bool:
    if trav_map is None or map_img is None:
        return True
    rc = _world_to_map_rc(trav_map, eye_xy)
    if rc is None:
        return False
    row, col = _clip_map_rc(map_img, rc)
    return _map_pixel_value(map_img, row, col) > 0.0


def _segment_is_occluded_by_trav_map(
    start_xy,
    end_xy,
    trav_map,
    map_img,
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
        if _map_pixel_value(map_img, row, col) <= 0.0:
            return True
    return False


def _draw_map_marker(draw: ImageDraw.ImageDraw, row: int, col: int, radius: int, fill, outline) -> None:
    draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=fill, outline=outline, width=max(1, radius // 3))


def _draw_square_marker(draw: ImageDraw.ImageDraw, row: int, col: int, half_size: int, fill) -> None:
    draw.rectangle((col - half_size, row - half_size, col + half_size, row + half_size), fill=fill)


def _disk_floor_trav_path(scene_name: str, floor_idx: int) -> Path:
    return SCRIPT_DIR / "datasets" / "behavior-1k-assets" / "scenes" / str(scene_name) / "layout" / f"floor_trav_{int(floor_idx)}.png"


def _fallback_world_to_map_rc(image: Image.Image, floor_record: RuntimeObjectRecord, xy) -> tuple[int, int] | None:
    x0, x1 = float(floor_record.bbox_min[0]), float(floor_record.bbox_max[0])
    y0, y1 = float(floor_record.bbox_min[1]), float(floor_record.bbox_max[1])
    if abs(x1 - x0) < 1e-8 or abs(y1 - y0) < 1e-8:
        return None
    col = (float(xy[0]) - x0) / (x1 - x0) * (image.width - 1)
    row = (1.0 - (float(xy[1]) - y0) / (y1 - y0)) * (image.height - 1)
    return int(round(row)), int(round(col))


def _center_from_bbox_json(payload: dict | None) -> list[float] | None:
    if not payload:
        return None
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 2:
        return None
    lo, hi = bbox
    if len(lo) != 3 or len(hi) != 3:
        return None
    return [float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), float((lo[2] + hi[2]) / 2.0)]


def _collect_selected_item_positions(entries: list[dict], extra_markers: list[dict] | None = None) -> list[dict]:
    markers: list[dict] = []
    seen = set()

    def _add_marker(kind: str, position) -> None:
        if position is None or len(position) < 2:
            return
        key = (kind, round(float(position[0]), 4), round(float(position[1]), 4), round(float(position[2]) if len(position) > 2 else 0.0, 4))
        if key in seen:
            return
        seen.add(key)
        markers.append({"kind": kind, "position": [float(position[0]), float(position[1]), float(position[2]) if len(position) > 2 else 0.0]})

    for entry in entries:
        for pos in entry.get("ball_positions", []):
            _add_marker("count_object", pos)
        _add_marker("anchor", _center_from_bbox_json(entry.get("anchor_object")))
        _add_marker("container", _center_from_bbox_json(entry.get("container_object")))
        confuser = entry.get("confuser_object") or {}
        _add_marker("confuser", confuser.get("position"))
        lighting = entry.get("lighting") or {}
        for light in lighting.get("target_lights", []):
            _add_marker("light", _center_from_bbox_json(light))
    for marker in extra_markers or []:
        if not isinstance(marker, dict):
            continue
        _add_marker(str(marker.get("kind", "anchor")), marker.get("position"))
    return markers


def _default_agent_position_from_floor(floor_record: RuntimeObjectRecord) -> tuple[float, float, float]:
    return (
        float((floor_record.bbox_min[0] + floor_record.bbox_max[0]) / 2.0),
        float((floor_record.bbox_min[1] + floor_record.bbox_max[1]) / 2.0),
        float(floor_record.bbox_max[2]) + AGENT_CAMERA_HEIGHT_M,
    )


def _sample_agent_position_near_short_edge(
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
) -> tuple[float, float, float]:
    bbox = room_bbox_xyxy
    if bbox is None:
        bbox = (
            float(floor_record.bbox_min[0]),
            float(floor_record.bbox_min[1]),
            float(floor_record.bbox_max[0]),
            float(floor_record.bbox_max[1]),
        )
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox, name="room_bbox")
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    center_xy = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
    target_z = float(floor_record.bbox_max[2]) + AGENT_CAMERA_TARGET_HEIGHT_M
    short_axis = "x" if width <= height else "y"
    candidates = []

    for ratio in AGENT_SHORT_EDGE_OFFSET_RATIOS:
        if short_axis == "x":
            x_options = [xmin + width * float(ratio), xmax - width * float(ratio)]
            y = ymin + AGENT_POSITION_CLEARANCE
            while y <= ymax - AGENT_POSITION_CLEARANCE + 1e-6:
                for x in x_options:
                    point_xy = (float(x), float(y))
                    if not _point_is_free(
                        point_xy,
                        floor_record,
                        blockers,
                        clearance=AGENT_POSITION_CLEARANCE,
                        room_bbox_xyxy=room_bbox_xyxy,
                    ):
                        continue
                    if not _eye_xy_is_traversable(trav_map, trav_map_img, point_xy):
                        continue
                    eye_z = float(floor_record.bbox_max[2]) + AGENT_CAMERA_HEIGHT_M
                    if _segment_is_occluded_by_blockers(
                        start_xy=point_xy,
                        start_z=eye_z,
                        end_xy=center_xy,
                        end_z=target_z,
                        blockers=blockers,
                    ):
                        continue
                    if _segment_is_occluded_by_trav_map(point_xy, center_xy, trav_map, trav_map_img):
                        continue
                    nearest_blocker = min(
                        (_distance_point_to_bbox_xy(point_xy, blocker.bbox_min, blocker.bbox_max) for blocker in blockers),
                        default=10.0,
                    )
                    center_dist = _distance_xy(point_xy, center_xy)
                    side_bias = min(abs(point_xy[0] - xmin), abs(xmax - point_xy[0]))
                    candidates.append((nearest_blocker, -center_dist, -side_bias, point_xy))
                y += AGENT_SHORT_EDGE_SCAN_STEP
        else:
            y_options = [ymin + height * float(ratio), ymax - height * float(ratio)]
            x = xmin + AGENT_POSITION_CLEARANCE
            while x <= xmax - AGENT_POSITION_CLEARANCE + 1e-6:
                for y in y_options:
                    point_xy = (float(x), float(y))
                    if not _point_is_free(
                        point_xy,
                        floor_record,
                        blockers,
                        clearance=AGENT_POSITION_CLEARANCE,
                        room_bbox_xyxy=room_bbox_xyxy,
                    ):
                        continue
                    if not _eye_xy_is_traversable(trav_map, trav_map_img, point_xy):
                        continue
                    eye_z = float(floor_record.bbox_max[2]) + AGENT_CAMERA_HEIGHT_M
                    if _segment_is_occluded_by_blockers(
                        start_xy=point_xy,
                        start_z=eye_z,
                        end_xy=center_xy,
                        end_z=target_z,
                        blockers=blockers,
                    ):
                        continue
                    if _segment_is_occluded_by_trav_map(point_xy, center_xy, trav_map, trav_map_img):
                        continue
                    nearest_blocker = min(
                        (_distance_point_to_bbox_xy(point_xy, blocker.bbox_min, blocker.bbox_max) for blocker in blockers),
                        default=10.0,
                    )
                    center_dist = _distance_xy(point_xy, center_xy)
                    side_bias = min(abs(point_xy[1] - ymin), abs(ymax - point_xy[1]))
                    candidates.append((nearest_blocker, -center_dist, -side_bias, point_xy))
                x += AGENT_SHORT_EDGE_SCAN_STEP

    if candidates:
        candidates.sort(reverse=True)
        best_xy = candidates[0][3]
        return (
            float(best_xy[0]),
            float(best_xy[1]),
            float(floor_record.bbox_max[2]) + AGENT_CAMERA_HEIGHT_M,
        )

    return _default_agent_position_from_floor(floor_record)


def _save_shared_topdown_map(
    scene,
    scene_name: str,
    floor_record: RuntimeObjectRecord,
    agent_pos,
    selected_positions: list[dict],
    output_path: str,
    room_bbox_info: dict | None = None,
) -> str | None:
    floor_idx = _infer_floor_idx(floor_record)
    trav_map = getattr(scene, "_trav_map", None)
    map_img = None
    if trav_map is not None and getattr(trav_map, "floor_map", None) is not None and 0 <= floor_idx < len(trav_map.floor_map):
        map_img = trav_map.floor_map[floor_idx]
        if hasattr(map_img, "detach"):
            map_img = map_img.detach()
        if getattr(map_img, "device", None) is not None and map_img.device.type != "cpu":
            map_img = map_img.cpu()
        map_img = map_img.numpy()
    if map_img is None:
        img_path = _disk_floor_trav_path(scene_name, floor_idx)
        if not img_path.exists():
            return None
        map_img = np.array(Image.open(img_path).convert("L"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(7.0, 7.0))
    plt.imshow(map_img, cmap="gray", vmin=0, vmax=255)

    if room_bbox_info:
        expanded_bbox_world_xy = room_bbox_info.get("expanded_bbox_world_xy")
        if trav_map is not None and expanded_bbox_world_xy is not None:
            xmin, ymin, xmax, ymax = expanded_bbox_world_xy
            p0 = _world_to_plot_xy(trav_map, (xmin, ymin))
            p1 = _world_to_plot_xy(trav_map, (xmax, ymax))
            left = min(p0[0], p1[0])
            top = min(p0[1], p1[1])
            width = max(1.0, abs(p1[0] - p0[0]))
            height = max(1.0, abs(p1[1] - p0[1]))
            plt.gca().add_patch(
                plt.Rectangle(
                    (left, top),
                    width,
                    height,
                    fill=False,
                    edgecolor="orange",
                    linewidth=1.8,
                    label="expanded bbox",
                )
            )

    if trav_map is not None:
        px, py = _world_to_plot_xy(trav_map, agent_pos[:2])
        plt.scatter([px], [py], c="dodgerblue", s=18, zorder=4)

        color_map = {
            "count_object": "red",
            "anchor": "orange",
            "container": "limegreen",
            "confuser": "magenta",
            "light": "gold",
        }
        marker_map = {
            "count_object": "s",
            "anchor": "o",
            "container": "o",
            "confuser": "o",
            "light": "o",
        }
        size_map = {
            "count_object": 35,
            "anchor": 20,
            "container": 20,
            "confuser": 20,
            "light": 20,
        }
        for item in selected_positions:
            position = item["position"]
            px, py = _world_to_plot_xy(trav_map, position[:2])
            plt.scatter(
                [px],
                [py],
                c=color_map.get(item["kind"], "tomato"),
                s=size_map.get(item["kind"], 20),
                marker=marker_map.get(item["kind"], "o"),
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
    else:
        canvas = Image.fromarray(map_img).convert("RGB")
        for item in selected_positions:
            rc = _fallback_world_to_map_rc(canvas, floor_record, item["position"][:2])
            if rc is None:
                continue
            row, col = _clip_rc(canvas, rc)
            if item["kind"] == "count_object":
                plt.scatter([col], [row], c="red", s=22, marker="s", zorder=4)
            else:
                plt.scatter([col], [row], c="orange", s=20, zorder=4)

    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def _capture(path: str) -> None:
    _, _, image = _get_viewer_frame()
    _save_rgb_png(path, image)


def _get_viewer_frame():
    last_exc = None
    for attempt in range(1, VIEWER_FRAME_MAX_RETRIES + 1):
        try:
            for _ in range(VIEWER_FRAME_RENDER_STEPS):
                og.sim.render()
            obs, info = og.sim._viewer_camera.get_obs()
            image = obs["rgb"].detach().cpu().to(dtype=th.uint8)[..., :3]
            return obs, info, image
        except Exception as exc:
            last_exc = exc
            if attempt >= VIEWER_FRAME_MAX_RETRIES:
                break
            print(
                f"[batch_counting] _get_viewer_frame attempt {attempt}/{VIEWER_FRAME_MAX_RETRIES} failed: "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            time.sleep(VIEWER_FRAME_RETRY_SLEEP_SEC)
    assert last_exc is not None
    raise last_exc


def _set_camera_pose(eye, target) -> tuple[list[float], list[float]]:
    eye = [float(v) for v in eye]
    quat = [float(v) for v in look_at_quaternion(eye, target)]
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(eye, dtype=th.float32),
        th.tensor(quat, dtype=th.float32),
    )
    return eye, quat


def _quat_xyzw_to_wxyz(quat_xyzw):
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _target_visibility_metrics(obs, info, target_name: str, target_aliases: list[str] | None = None) -> dict:
    seg = obs.get("seg_instance")
    seg_info = info.get("seg_instance") if isinstance(info, dict) else None
    if seg is None or not seg_info:
        return {
            "target_name": target_name,
            "visible_pixels": 0,
            "centered_score": 0.0,
            "bbox_xyxy": None,
            "is_visible": False,
        }

    def _candidate_names() -> list[str]:
        names = [str(target_name)]
        for alias in target_aliases or []:
            alias_str = str(alias).strip()
            if alias_str and alias_str not in names:
                names.append(alias_str)
        return names

    def _seg_name_matches(raw_name: object, candidates: list[str]) -> bool:
        seg_name = str(raw_name)
        seg_name_lower = seg_name.lower()
        basename = seg_name.rsplit("/", 1)[-1].lower()
        for candidate in candidates:
            cand = str(candidate).strip().lower()
            if not cand:
                continue
            if seg_name_lower == cand:
                return True
            if basename == cand:
                return True
            if seg_name_lower.endswith("/" + cand):
                return True
            if f"/{cand}/" in seg_name_lower:
                return True
        return False

    seg_np = seg.detach().cpu().numpy()
    candidate_names = _candidate_names()
    visible_ids = [int(seg_id) for seg_id, name in seg_info.items() if _seg_name_matches(name, candidate_names)]
    if not visible_ids:
        return {
            "target_name": target_name,
            "visible_pixels": 0,
            "centered_score": 0.0,
            "bbox_xyxy": None,
            "is_visible": False,
            "matched_aliases": candidate_names,
        }

    mask = np.isin(seg_np, np.array(visible_ids, dtype=seg_np.dtype))
    visible_pixels = int(mask.sum())
    if visible_pixels <= 0:
        return {
            "target_name": target_name,
            "visible_pixels": 0,
            "centered_score": 0.0,
            "bbox_xyxy": None,
            "is_visible": False,
            "matched_aliases": candidate_names,
        }

    rows, cols = np.where(mask)
    row_center = float(rows.mean())
    col_center = float(cols.mean())
    height, width = mask.shape[:2]
    center_dist = math.sqrt((row_center - (height - 1) * 0.5) ** 2 + (col_center - (width - 1) * 0.5) ** 2)
    max_center_dist = max(math.sqrt(((height - 1) * 0.5) ** 2 + ((width - 1) * 0.5) ** 2), 1e-6)
    centered_score = max(0.0, 1.0 - center_dist / max_center_dist)
    return {
        "target_name": target_name,
        "visible_pixels": visible_pixels,
        "centered_score": round(float(centered_score), 4),
        "bbox_xyxy": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
        "is_visible": bool(visible_pixels >= CLOSEUP_ACCEPT_VISIBLE_PIXELS),
        "matched_aliases": candidate_names,
    }


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


def _prepare_camera_render(agent_pos, floor_z: float, target_xy=None) -> tuple[list[float], list[float], float]:
    for modality in ("seg_semantic", "seg_instance", "seg_instance_id"):
        try:
            og.sim._viewer_camera.add_modality(modality)
        except Exception as exc:
            _log_exception(f"Failed to add camera modality {modality}", exc)
            pass
    _step_sim(SIM_STEP_CAMERA_MODALITY)
    eye = [
        float(agent_pos[0]),
        float(agent_pos[1]),
        max(float(agent_pos[2]), float(floor_z) + AGENT_CAMERA_HEIGHT_M),
    ]
    if target_xy is None:
        target_xy = [float(agent_pos[0]), float(agent_pos[1]) - 1.0]
    look_target = [
        float(target_xy[0]),
        float(target_xy[1]),
        float(floor_z) + AGENT_CAMERA_TARGET_HEIGHT_M,
    ]
    pitch_deg = math.degrees(
        math.atan2(max(0.0, float(eye[2]) - float(look_target[2])), max(1e-6, _distance_xy(eye[:2], look_target[:2])))
    )
    return eye, look_target, pitch_deg


def _camera_position_is_clear(
    eye_xy,
    eye_z: float,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    clearance: float = PRIMARY_VIEW_CLEARANCE,
) -> bool:
    if not (
        floor_record.bbox_min[0] + clearance <= float(eye_xy[0]) <= floor_record.bbox_max[0] - clearance
        and floor_record.bbox_min[1] + clearance <= float(eye_xy[1]) <= floor_record.bbox_max[1] - clearance
    ):
        return False
    if room_bbox_xyxy is not None and not _point_inside_bbox_xyxy(eye_xy, room_bbox_xyxy, margin=-clearance):
        return False
    for blocker in blockers:
        if (
            blocker.bbox_min[0] - 0.02 <= float(eye_xy[0]) <= blocker.bbox_max[0] + 0.02
            and blocker.bbox_min[1] - 0.02 <= float(eye_xy[1]) <= blocker.bbox_max[1] + 0.02
            and blocker.bbox_min[2] - 0.02 <= float(eye_z) <= blocker.bbox_max[2] + 0.02
        ):
            return False
    return True


def _sample_primary_view_pose(
    fallback_agent_pos,
    floor_record: RuntimeObjectRecord,
    room_center_xy,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
):
    eye_z = float(floor_record.bbox_max[2]) + AGENT_CAMERA_HEIGHT_M
    target_z = float(floor_record.bbox_max[2]) + AGENT_CAMERA_TARGET_HEIGHT_M

    def _build_pose(point_xy):
        eye = [float(point_xy[0]), float(point_xy[1]), eye_z]
        target = [float(room_center_xy[0]), float(room_center_xy[1]), target_z]
        pitch_deg = math.degrees(
            math.atan2(max(0.0, float(eye[2]) - float(target[2])), max(1e-6, _distance_xy(eye[:2], target[:2])))
        )
        return eye, target, pitch_deg

    fallback_xy = (float(fallback_agent_pos[0]), float(fallback_agent_pos[1]))
    if (
        _camera_position_is_clear(fallback_xy, eye_z, floor_record, blockers, room_bbox_xyxy=room_bbox_xyxy)
        and _eye_xy_is_traversable(trav_map, trav_map_img, fallback_xy)
        and not _segment_is_occluded_by_blockers(
            start_xy=fallback_xy,
            start_z=eye_z,
            end_xy=room_center_xy,
            end_z=target_z,
            blockers=blockers,
        )
        and not _segment_is_occluded_by_trav_map(fallback_xy, room_center_xy, trav_map, trav_map_img)
    ):
        return _build_pose(fallback_xy)

    sampled = _sample_agent_position_near_short_edge(
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    return _build_pose(sampled[:2])


def _refresh_camera_modalities_for_dynamic_objects(agent_pos, floor_z: float) -> None:
    # Re-adding modalities after dynamic object spawn / placement helps force
    # seg_instance / seg_semantic registries to include newly inserted targets.
    try:
        _prepare_camera_render(agent_pos, floor_z)
    except Exception as exc:
        _log_exception("Failed to refresh camera modalities for dynamic objects", exc)
    _step_sim(SIM_STEP_MINIMAL)


def _closeup_candidate_eyes(
    target_position,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    container_obj=None,
    target_obj=None,
):
    target_position = [float(v) for v in target_position]
    container_record = _runtime_record_from_obj(container_obj) if container_obj is not None else None
    target_record = _runtime_record_from_obj(target_obj) if target_obj is not None else None
    candidates = []
    floor_z = float(floor_record.bbox_max[2])
    eye_z_candidates = []

    def _append_eye_z(value: float) -> None:
        eye_z = max(float(value), floor_z + CLOSEUP_SEARCH_CAMERA_HEIGHT_M)
        rounded = round(eye_z, 4)
        if rounded not in eye_z_candidates:
            eye_z_candidates.append(rounded)

    _append_eye_z(float(target_position[2]) + 0.03)
    _append_eye_z(float(target_position[2]) + 0.10)
    _append_eye_z(float(target_position[2]) + 0.18)
    if target_record is not None:
        _append_eye_z(float(target_record.center[2]))
        _append_eye_z(float(target_record.center[2]) + 0.08)
        _append_eye_z(float(target_record.bbox_max[2]) + 0.04)
        _append_eye_z(float(target_record.bbox_max[2]) + 0.12)
    if container_record is not None:
        _append_eye_z(float(container_record.center[2]))
        _append_eye_z(float(container_record.bbox_max[2]) + 0.04)

    for radius in CLOSEUP_CAMERA_RADII_M:
        for eye_z in eye_z_candidates:
            for azimuth_deg in CLOSEUP_AZIMUTH_DEG:
                azimuth = math.radians(float(azimuth_deg))
                eye = [
                    float(target_position[0] + math.cos(azimuth) * radius),
                    float(target_position[1] + math.sin(azimuth) * radius),
                    float(eye_z),
                ]
                if not _camera_position_is_clear(
                    eye[:2],
                    eye[2],
                    floor_record,
                    blockers,
                    room_bbox_xyxy=room_bbox_xyxy,
                    clearance=0.02,
                ):
                    continue
                candidates.append(eye)

    if container_record is not None:
        center = list(container_record.center)
        for eye_z in eye_z_candidates:
            for radius in (0.0, 0.06, 0.12):
                for azimuth_deg in (0.0, 120.0, 240.0):
                    azimuth = math.radians(float(azimuth_deg))
                    eye = [
                        float(center[0] + math.cos(azimuth) * radius),
                        float(center[1] + math.sin(azimuth) * radius),
                        float(eye_z),
                    ]
                    if not _camera_position_is_clear(
                        eye[:2],
                        eye[2],
                        floor_record,
                        blockers,
                        room_bbox_xyxy=room_bbox_xyxy,
                        clearance=0.02,
                    ):
                        continue
                    candidates.insert(0, eye)

    return candidates


def _find_container_reveal_links(container_obj) -> list[object]:
    links = list((getattr(container_obj, "links", {}) or {}).values())
    if not links:
        return []
    container_category = str(getattr(container_obj, "category", "")).lower()

    def _link_record(link):
        try:
            bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in link.aabb]
            center_x = float((bbox_min[0] + bbox_max[0]) * 0.5)
            center_y = float((bbox_min[1] + bbox_max[1]) * 0.5)
            center_z = float((bbox_min[2] + bbox_max[2]) * 0.5)
            extent_x = float(bbox_max[0] - bbox_min[0])
            extent_y = float(bbox_max[1] - bbox_min[1])
            extent_z = float(bbox_max[2] - bbox_min[2])
            footprint = max(float(bbox_max[0] - bbox_min[0]), 0.0) * max(float(bbox_max[1] - bbox_min[1]), 0.0)
        except Exception:
            center_x = 0.0
            center_y = 0.0
            center_z = 0.0
            extent_x = 0.0
            extent_y = 0.0
            extent_z = 0.0
            footprint = 0.0
        return center_x, center_y, center_z, extent_x, extent_y, extent_z, footprint

    named = []
    for link in links:
        link_name = str(getattr(link, "name", "")).lower()
        if any(keyword in link_name for keyword in CLOSEUP_REVEAL_LINK_KEYWORDS):
            named.append(link)
    if named:
        return named

    container_record = _runtime_record_from_obj(container_obj)
    if "box" in container_category:
        container_pos, container_quat = container_obj.get_position_orientation()
        front_xy = _quaternion_xyzw_to_front_xy(container_quat.detach().cpu().tolist())
        box_candidates = []
        container_width = max(float(container_record.extents[0]), 1e-4)
        container_depth = max(float(container_record.extents[1]), 1e-4)
        container_height = max(float(container_record.extents[2]), 1e-4)
        for link in links:
            center_x, center_y, center_z, extent_x, extent_y, extent_z, footprint = _link_record(link)
            if max(extent_x, extent_y, extent_z) <= 1e-4:
                continue
            rel_x = float(center_x - container_pos[0].item())
            rel_y = float(center_y - container_pos[1].item())
            front_score = rel_x * front_xy[0] + rel_y * front_xy[1]
            front_thickness = abs(front_xy[0]) * extent_x + abs(front_xy[1]) * extent_y
            lateral_width = abs(front_xy[1]) * extent_x + abs(front_xy[0]) * extent_y
            if lateral_width < 0.35 * min(container_width, container_depth):
                continue
            if extent_z < 0.35 * container_height:
                continue
            thickness_ratio = front_thickness / max(min(container_width, container_depth), 1e-4)
            if thickness_ratio > 0.45:
                continue
            box_candidates.append((front_score, -thickness_ratio, lateral_width * extent_z, link))
        box_candidates.sort(key=lambda item: (-item[0], item[1], item[2]), reverse=True)
        if box_candidates:
            return [box_candidates[0][3]]

    container_center_z = float((container_record.bbox_min[2] + container_record.bbox_max[2]) * 0.5)
    heuristic = []
    for link in links:
        _, _, center_z, _, _, extent_z, footprint = _link_record(link)
        if center_z <= container_center_z:
            continue
        if extent_z <= 0.0:
            continue
        heuristic.append((footprint, -extent_z, link))
    heuristic.sort(reverse=True)
    return [item[2] for item in heuristic[:1]]


def _reveal_container_for_closeup(container_obj):
    if container_obj is None:
        return None
    token = {
        "container_obj": container_obj,
        "original_open": None,
        "hidden_links": [],
    }
    states = getattr(container_obj, "states", {}) or {}
    if object_states.Open in states:
        try:
            token["original_open"] = bool(states[object_states.Open].get_value())
            states[object_states.Open].set_value(True)
            _step_sim(SIM_STEP_MINIMAL)
        except Exception as exc:
            _log_exception(f"Failed to open container for closeup {getattr(container_obj, 'name', '<unknown>')}", exc)
    for link in _find_container_reveal_links(container_obj):
        try:
            token["hidden_links"].append((link, bool(link.visible)))
            link.visible = False
        except Exception as exc:
            _log_exception(f"Failed to hide reveal link on {getattr(container_obj, 'name', '<unknown>')}", exc)
    if token["hidden_links"]:
        _step_sim(SIM_STEP_MINIMAL)
    return token


def _restore_container_after_closeup(token) -> None:
    if not token:
        return
    for link, was_visible in reversed(token.get("hidden_links", [])):
        try:
            link.visible = bool(was_visible)
        except Exception as exc:
            _log_exception(f"Failed to restore reveal link visibility for {getattr(link, 'name', '<unknown>')}", exc)
    if token.get("hidden_links"):
        _step_sim(SIM_STEP_MINIMAL)
    container_obj = token.get("container_obj")
    original_open = token.get("original_open")
    if container_obj is not None and original_open is not None:
        _restore_container_open_state(container_obj, original_open)


def _open_all_closeup_containers(placements: list[dict], *, hidden_in_box_only: bool = False) -> list[dict]:
    tokens = []
    seen = set()
    for placement in placements:
        placement_case = placement.get("entry_case") or placement.get("source_entry_case")
        if hidden_in_box_only and placement_case != "hidden_in_box":
            continue
        container_obj = placement.get("container_obj")
        if container_obj is None:
            continue
        container_name = str(getattr(container_obj, "name", ""))
        if container_name in seen:
            continue
        seen.add(container_name)
        token = _reveal_container_for_closeup(container_obj)
        if token is not None:
            tokens.append(token)
    return tokens


def _hidden_box_fixed_closeup_pose(target_obj, container_obj) -> tuple[list[float], list[float]]:
    container_record = _runtime_record_from_obj(container_obj)
    container_center = [float(v) for v in container_record.center]
    _, container_quat = container_obj.get_position_orientation()
    container_quat = container_quat.detach().cpu().tolist()
    front_xy = _quaternion_xyzw_to_front_xy(container_quat)
    camera_dir_xy = [-float(front_xy[0]), -float(front_xy[1])]

    look_target = list(container_center)
    if target_obj is not None:
        try:
            target_record = _runtime_record_from_obj(target_obj)
            look_target = [
                float(target_record.center[0]),
                float(target_record.center[1]),
                float(target_record.bbox_min[2] + max(0.6 * target_record.extents[2], 0.02)),
            ]
        except Exception as exc:
            _log_exception(
                f"Failed to read hidden-box target bounds for {getattr(target_obj, 'name', '<unknown>')}",
                exc,
            )

    # Place the camera relative to the box front face rather than the box
    # center so the requested offset means a true clearance in front of the box.
    half_front_depth = 0.5 * (
        abs(camera_dir_xy[0]) * float(container_record.extents[0])
        + abs(camera_dir_xy[1]) * float(container_record.extents[1])
    )
    horizontal_distance = half_front_depth + float(HIDDEN_BOX_BOX_CLOSEUP_FRONT_OFFSET_M)
    eye_xy = [
        float(container_center[0] + camera_dir_xy[0] * horizontal_distance),
        float(container_center[1] + camera_dir_xy[1] * horizontal_distance),
    ]
    required_height_offset = _distance_xy(eye_xy, look_target[:2]) * math.tan(
        math.radians(float(HIDDEN_BOX_BOX_CLOSEUP_PITCH_DEG))
    )
    eye = [
        float(eye_xy[0]),
        float(eye_xy[1]),
        float(look_target[2] + max(float(HIDDEN_BOX_BOX_CLOSEUP_HEIGHT_OFFSET_M), required_height_offset)),
    ]
    return eye, look_target


def _capture_target_closeups(
    output_dir: str,
    placements: list[dict],
    agent_pos,
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
):
    os.makedirs(output_dir, exist_ok=True)
    closeups = []
    global_reveal_tokens = _open_all_closeup_containers(placements, hidden_in_box_only=True)
    try:
        for idx, placement in enumerate(placements):
            target_obj = placement.get("target_obj")
            target_name = str(getattr(target_obj, "name", placement.get("target_name", f"target_{idx:03d}")))
            target_position = placement.get("position")
            container_obj = placement.get("container_obj")
            placement_case = placement.get("entry_case") or placement.get("source_entry_case")
            is_hidden_box_closeup = placement_case == "hidden_in_box" and container_obj is not None
            reveal_token = None
            if container_obj is not None and placement_case != "hidden_in_box":
                reveal_token = _reveal_container_for_closeup(container_obj)
            try:
                best = None
                target = [
                    float(target_position[0]),
                    float(target_position[1]),
                    float(target_position[2]),
                ]
                if target_obj is not None:
                    try:
                        target_record = _runtime_record_from_obj(target_obj)
                        target = [
                            float(target_record.center[0]),
                            float(target_record.center[1]),
                            float(target_record.bbox_min[2] + max(0.6 * target_record.extents[2], 0.02)),
                        ]
                    except Exception as exc:
                        _log_exception(f"Failed to read target bounds for closeup {target_name}", exc)
                elif container_obj is not None:
                    container_record = _runtime_record_from_obj(container_obj)
                    target[2] = max(target[2], float(container_record.center[2]))
                target_aliases = []
                if target_obj is not None:
                    target_aliases.append(str(getattr(target_obj, "name", "")))
                    target_aliases.append(str(getattr(target_obj, "category", "")))
                if is_hidden_box_closeup:
                    fixed_eye, fixed_target = _hidden_box_fixed_closeup_pose(target_obj, container_obj)
                    candidate_eyes = [fixed_eye]
                    target = fixed_target
                else:
                    candidate_eyes = _closeup_candidate_eyes(
                        target_position=target,
                        floor_record=floor_record,
                        blockers=blockers,
                        room_bbox_xyxy=room_bbox_xyxy,
                        container_obj=container_obj,
                        target_obj=target_obj,
                    )
                for eye in candidate_eyes:
                    if not _camera_position_is_clear(
                        eye[:2],
                        eye[2],
                        floor_record,
                        blockers,
                        room_bbox_xyxy=room_bbox_xyxy,
                        clearance=0.02 if not is_hidden_box_closeup else 0.0,
                    ):
                        if not is_hidden_box_closeup:
                            continue
                    original_closeup_fov_deg = None
                    eye, quat = _set_camera_pose(eye, target)
                    try:
                        if is_hidden_box_closeup:
                            original_closeup_fov_deg = VIEWER_CAMERA_FOV_DEG
                            _set_viewer_camera_fov(HIDDEN_BOX_BOX_CLOSEUP_FOV_DEG)
                        obs, info, image = _get_viewer_frame()
                    except Exception as exc:
                        _log_exception(
                            f"Failed to capture closeup frame for target={target_name} eye={eye}",
                            exc,
                        )
                        continue
                    finally:
                        if original_closeup_fov_deg is not None:
                            _set_viewer_camera_fov(original_closeup_fov_deg)
                    metrics = _target_visibility_metrics(obs, info, target_name, target_aliases=target_aliases)
                    score = float(metrics["visible_pixels"]) + float(metrics["centered_score"]) * 500.0
                    candidate_result = {
                        "eye": eye,
                        "quat": quat,
                        "target": [float(v) for v in target],
                        "metrics": metrics,
                        "image": image,
                        "score": score,
                    }
                    if best is None or score > best["score"]:
                        best = candidate_result
                    if metrics["visible_pixels"] >= CLOSEUP_GOOD_ENOUGH_VISIBLE_PIXELS:
                        best = candidate_result
                        break
                    if is_hidden_box_closeup:
                        break

                if best is None:
                    raise RuntimeError(
                        f"Could not capture any closeup frame for counting target {target_name}"
                    )
                filename = f"target_closeup_{idx:03d}.png"
                image_path = os.path.join(output_dir, filename)
                _save_rgb_png(image_path, best["image"])
                closeups.append(
                    {
                        "image_path": image_path,
                        "target_name": target_name,
                        "target_position": [float(v) for v in target_position],
                        "camera_pose": {
                            "position": best["eye"],
                            "quaternion_xyzw": best["quat"],
                        },
                        "look_target": best["target"],
                        "fov_deg": (
                            float(HIDDEN_BOX_BOX_CLOSEUP_FOV_DEG)
                            if is_hidden_box_closeup
                            else float(VIEWER_CAMERA_FOV_DEG)
                        ),
                        "visibility": best["metrics"],
                        "best_effort_only": bool(not best["metrics"]["is_visible"]),
                        "container_name": getattr(container_obj, "name", None) if container_obj is not None else None,
                    }
                )
            finally:
                _restore_container_after_closeup(reveal_token)
    finally:
        for token in reversed(global_reveal_tokens):
            _restore_container_after_closeup(token)
    if len(closeups) != len(placements):
        raise RuntimeError(
            f"Target closeup count mismatch: expected {len(placements)} images, got {len(closeups)}"
        )
    return closeups


def render_and_save(
    image_prefix: str,
    output_dir: str,
    agent_pos,
    floor_record: RuntimeObjectRecord,
    room_center_xy,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
) -> dict:
    eye, look_target, pitch_deg = _sample_primary_view_pose(
        fallback_agent_pos=agent_pos,
        floor_record=floor_record,
        room_center_xy=room_center_xy,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    os.makedirs(output_dir, exist_ok=True)
    camera_poses = {}
    eye, quat = _set_camera_pose(eye, look_target)
    filename = f"{image_prefix}.png"
    _capture(os.path.join(output_dir, filename))
    camera_poses[filename] = {
        "position": eye,
        "quaternion_xyzw": quat,
        "angle_deg": None,
        "view_name": "room_center_single_view",
        "pitch_deg": pitch_deg,
        "look_target": look_target,
    }
    return camera_poses


def _sample_observation_merged_view_pose(
    selected_positions: list[dict],
    fallback_agent_pos,
    floor_record: RuntimeObjectRecord,
    room_center_xy,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
) -> tuple[list[float], list[float], float]:
    count_positions = [
        marker.get("position")
        for marker in selected_positions
        if isinstance(marker, dict) and marker.get("kind") == "count_object"
    ]
    count_positions = [
        [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else float(floor_record.bbox_max[2])]
        for pos in count_positions
        if isinstance(pos, (list, tuple)) and len(pos) >= 2
    ]
    if not count_positions:
        return _sample_primary_view_pose(
            fallback_agent_pos=fallback_agent_pos,
            floor_record=floor_record,
            room_center_xy=room_center_xy,
            blockers=blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )

    center_xy = [
        float(sum(pos[0] for pos in count_positions) / len(count_positions)),
        float(sum(pos[1] for pos in count_positions) / len(count_positions)),
    ]
    target_z = max(
        float(floor_record.bbox_max[2]) + 0.06,
        float(sum(pos[2] for pos in count_positions) / len(count_positions)),
    )
    look_target = [float(center_xy[0]), float(center_xy[1]), float(target_z)]

    max_cluster_radius = max((_distance_xy(pos[:2], center_xy) for pos in count_positions), default=0.0)
    horizontal_radius = min(2.4, max(1.0, max_cluster_radius + 0.9))
    pitch_deg = float(OBSERVATION_MERGED_CAMERA_PITCH_DEG)
    eye_z = float(look_target[2]) + horizontal_radius * math.tan(math.radians(pitch_deg))

    def _normalize_dir(vec_xy) -> tuple[float, float] | None:
        norm = _norm_xy(vec_xy)
        if norm < 1e-6:
            return None
        return (float(vec_xy[0]) / norm, float(vec_xy[1]) / norm)

    preferred_dirs: list[tuple[float, float]] = []
    for raw_dir in (
        _sub_xy(fallback_agent_pos[:2], center_xy),
        _sub_xy(room_center_xy, center_xy),
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    ):
        direction = _normalize_dir(raw_dir)
        if direction is None:
            continue
        rounded_key = (round(direction[0], 4), round(direction[1], 4))
        if rounded_key in {(round(item[0], 4), round(item[1], 4)) for item in preferred_dirs}:
            continue
        preferred_dirs.append(direction)

    fallback_eye = None
    for direction in preferred_dirs:
        eye = [
            float(center_xy[0] + direction[0] * horizontal_radius),
            float(center_xy[1] + direction[1] * horizontal_radius),
            float(eye_z),
        ]
        if fallback_eye is None:
            fallback_eye = list(eye)
        if not _camera_position_is_clear(
            eye[:2],
            eye[2],
            floor_record,
            blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            clearance=0.02,
        ):
            continue
        return eye, look_target, pitch_deg

    if fallback_eye is None:
        fallback_eye = [
            float(center_xy[0] - horizontal_radius / math.sqrt(2.0)),
            float(center_xy[1] - horizontal_radius / math.sqrt(2.0)),
            float(eye_z),
        ]
    return fallback_eye, look_target, pitch_deg


def _render_observation_merged_and_save(
    image_prefix: str,
    output_dir: str,
    selected_positions: list[dict],
    agent_pos,
    floor_record: RuntimeObjectRecord,
    room_center_xy,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
) -> dict:
    eye, look_target, pitch_deg = _sample_observation_merged_view_pose(
        selected_positions=selected_positions,
        fallback_agent_pos=agent_pos,
        floor_record=floor_record,
        room_center_xy=room_center_xy,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    os.makedirs(output_dir, exist_ok=True)
    camera_poses = {}
    eye, quat = _set_camera_pose(eye, look_target)
    filename = f"{image_prefix}.png"
    _capture(os.path.join(output_dir, filename))
    camera_poses[filename] = {
        "position": eye,
        "quaternion_xyzw": quat,
        "angle_deg": None,
        "view_name": "observation_merged_45deg_down_view",
        "pitch_deg": pitch_deg,
        "look_target": look_target,
    }
    return camera_poses


def _capture_camera_view(image_path: str, eye, target) -> dict:
    eye, quat = _set_camera_pose(eye, target)
    _capture(image_path)
    return {
        "image_path": image_path,
        "camera_pose": {
            "position": eye,
            "quaternion_xyzw": quat,
        },
        "target": [float(v) for v in target],
    }


def _capture_room_corner_views(
    output_dir: str,
    floor_record: RuntimeObjectRecord,
    room_bbox_info: dict,
    blockers: list[RuntimeObjectRecord],
    trav_map=None,
    trav_map_img=None,
) -> dict:
    bbox = room_bbox_info.get("expanded_bbox_world_xy")
    if bbox is None:
        bbox = (
            float(floor_record.bbox_min[0]),
            float(floor_record.bbox_min[1]),
            float(floor_record.bbox_max[0]),
            float(floor_record.bbox_max[1]),
        )
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox, name="room_bbox")
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    center_xy = [float((xmin + xmax) * 0.5), float((ymin + ymax) * 0.5)]
    inset = min(max(0.35, min(width, height) * 0.08), max(min(width, height) * 0.4, 0.35))
    floor_margin = 0.2
    corners = [
        ("corner_00", [xmin + inset, ymin + inset]),
        ("corner_01", [xmin + inset, ymax - inset]),
        ("corner_02", [xmax - inset, ymin + inset]),
        ("corner_03", [xmax - inset, ymax - inset]),
    ]
    os.makedirs(output_dir, exist_ok=True)
    views = []
    for view_id, xy in corners:
        eye_xy = [
            float(np.clip(float(xy[0]), float(floor_record.bbox_min[0]) + floor_margin, float(floor_record.bbox_max[0]) - floor_margin)),
            float(np.clip(float(xy[1]), float(floor_record.bbox_min[1]) + floor_margin, float(floor_record.bbox_max[1]) - floor_margin)),
        ]
        is_traversable = bool(_eye_xy_is_traversable(trav_map, trav_map_img, eye_xy))
        eye = [
            float(eye_xy[0]),
            float(eye_xy[1]),
            float(floor_record.bbox_max[2]) + float(ROOM_VIEW_CAMERA_HEIGHT_M),
        ]
        target = [
            float(center_xy[0]),
            float(center_xy[1]),
            float(floor_record.bbox_max[2]) + float(ROOM_VIEW_TARGET_HEIGHT_M),
        ]
        blocker_occluded = bool(
            _segment_is_occluded_by_blockers(
                start_xy=eye[:2],
                start_z=float(eye[2]),
                end_xy=target[:2],
                end_z=float(target[2]),
                blockers=blockers,
            )
        )
        trav_occluded = bool(_segment_is_occluded_by_trav_map(eye[:2], target[:2], trav_map, trav_map_img))
        view = _capture_camera_view(os.path.join(output_dir, f"{view_id}.png"), eye=eye, target=target)
        view["view_id"] = view_id
        view["is_traversable"] = is_traversable
        view["blocker_occluded"] = blocker_occluded
        view["trav_occluded"] = trav_occluded
        views.append(view)
    return {
        "room_instance": room_bbox_info.get("room_instance", None),
        "room_bbox_world_xy": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "views": views,
    }


def _set_light_scale(room_object_by_name: dict[str, RuntimeObjectRecord], light_names: list[str], scale: float):
    original = []
    for light_name in light_names:
        record = room_object_by_name.get(light_name)
        if record is None or not hasattr(record.obj, "intensity"):
            continue
        try:
            base_intensity = float(record.obj.intensity)
            original.append((record.obj, base_intensity))
            record.obj.intensity = base_intensity * float(scale)
        except Exception as exc:
            _log_exception(f"Failed to adjust light intensity for {light_name}", exc)
            continue
    if original:
        _step_sim(SIM_STEP_MINIMAL)
    return original


def _restore_light_scale(light_states) -> None:
    for obj, intensity in light_states:
        try:
            obj.intensity = intensity
        except Exception as exc:
            _log_exception(f"Failed to restore light intensity for {getattr(obj, 'name', '<unknown>')}", exc)
            continue
    if light_states:
        _step_sim(SIM_STEP_MINIMAL)


def _iter_stage_light_prims(root_prim):
    for child in root_prim.GetChildren():
        if "Light" in child.GetPrimTypeInfo().GetTypeName():
            yield child
        yield from _iter_stage_light_prims(child)


def _collect_stage_lights() -> list[dict]:
    world_prim = getattr(og.sim, "world_prim", None)
    if world_prim is None:
        return []

    lights = []
    for prim in _iter_stage_light_prims(world_prim):
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if not intensity_attr or not intensity_attr.IsValid():
            continue
        try:
            intensity = intensity_attr.Get()
        except Exception:
            intensity = None
        if intensity is None:
            continue
        type_name = str(prim.GetPrimTypeInfo().GetTypeName())
        prim_path = str(prim.GetPath())
        lights.append(
            {
                "prim": prim,
                "path": prim_path,
                "type_name": type_name,
                "is_dome": type_name == "DomeLight" or prim_path.lower().endswith("/skybox/base_link/light"),
                "intensity": float(intensity),
            }
        )
    return lights


def _apply_stage_light_scales(stage_light_scales: list[dict]) -> list[tuple[object, float]]:
    original = []
    for item in stage_light_scales:
        prim = item["prim"]
        scale = float(item["scale"])
        attr = prim.GetAttribute("inputs:intensity")
        if not attr or not attr.IsValid():
            continue
        try:
            base_intensity = attr.Get()
            if base_intensity is None:
                continue
            original.append((prim, float(base_intensity)))
            attr.Set(float(base_intensity) * scale)
        except Exception as exc:
            _log_exception(f"Failed to adjust stage light intensity for {item['path']}", exc)
            continue
    if original:
        _step_sim(SIM_STEP_MINIMAL)
    return original


def _restore_stage_light_scales(light_states: list[tuple[object, float]]) -> None:
    for prim, intensity in light_states:
        attr = prim.GetAttribute("inputs:intensity")
        if not attr or not attr.IsValid():
            continue
        try:
            attr.Set(float(intensity))
        except Exception as exc:
            _log_exception(f"Failed to restore stage light intensity for {prim.GetPath()}", exc)
            continue
    if light_states:
        _step_sim(SIM_STEP_MINIMAL)


def _sample_large_light_change(
    room_object_by_name: dict[str, RuntimeObjectRecord],
    light_names: list[str],
    rng: random.Random,
) -> tuple[list[str], list[float], list[dict], str]:
    valid_names = [
        name
        for name in light_names
        if room_object_by_name.get(name) is not None and hasattr(room_object_by_name[name].obj, "intensity")
    ]
    stage_lights = _collect_stage_lights()
    internal_stage_lights = [item for item in stage_lights if not item["is_dome"]]
    dome_stage_lights = [item for item in stage_lights if item["is_dome"]]

    if internal_stage_lights:
        changed_stage_lights = [
            {
                "prim": item["prim"],
                "path": item["path"],
                "type_name": item["type_name"],
                "scale": 0.0 if rng.random() < 0.7 else rng.uniform(0.01, 0.08),
            }
            for item in internal_stage_lights
        ]
        changed_object_lights = []
        changed_object_scales = []
        return changed_object_lights, changed_object_scales, changed_stage_lights, "disabled_or_severely_dimmed_internal_stage_lights"

    if stage_lights:
        changed_stage_lights = [
            {
                "prim": item["prim"],
                "path": item["path"],
                "type_name": item["type_name"],
                "scale": rng.uniform(0.01, 0.12) if item in dome_stage_lights else rng.uniform(0.0, 0.08),
            }
            for item in stage_lights
        ]
        return [], [], changed_stage_lights, "severely_dimmed_all_stage_lights"

    if not valid_names:
        return [], [], [], "no_adjustable_lights_found"

    shuffled = list(valid_names)
    rng.shuffle(shuffled)
    change_count = rng.randint(1, len(shuffled))
    selected = shuffled[:change_count]
    scales = []
    for _ in selected:
        scales.append(0.0 if rng.random() < 0.7 else rng.uniform(0.01, 0.08))
    return selected, scales, [], "disabled_or_severely_dimmed_object_lights"


def _apply_per_light_scales(
    room_object_by_name: dict[str, RuntimeObjectRecord],
    light_names: list[str],
    scales: list[float],
):
    original = []
    for light_name, scale in zip(light_names, scales):
        record = room_object_by_name.get(light_name)
        if record is None or not hasattr(record.obj, "intensity"):
            continue
        try:
            base_intensity = float(record.obj.intensity)
            original.append((record.obj, base_intensity))
            record.obj.intensity = base_intensity * float(scale)
        except Exception as exc:
            _log_exception(f"Failed to adjust light intensity for {light_name}", exc)
            continue
    if original:
        _step_sim(SIM_STEP_MINIMAL)
    return original


def _render_light_change_and_save(
    image_prefix: str,
    output_dir: str,
    agent_pos,
    floor_record: RuntimeObjectRecord,
    room_object_by_name: dict[str, RuntimeObjectRecord],
    light_names: list[str],
    rng: random.Random,
    room_center_xy,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy=None,
    trav_map=None,
    trav_map_img=None,
) -> tuple[dict, dict]:
    eye, look_target, pitch_deg = _sample_primary_view_pose(
        fallback_agent_pos=agent_pos,
        floor_record=floor_record,
        room_center_xy=room_center_xy,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    os.makedirs(output_dir, exist_ok=True)
    camera_poses = {}

    def _place_and_capture(suffix: str, view_name: str, lighting_phase: str) -> None:
        _, quat = _set_camera_pose(eye, look_target)
        filename = f"{image_prefix}{suffix}.png"
        _capture(os.path.join(output_dir, filename))
        camera_poses[filename] = {
            "position": eye,
            "quaternion_xyzw": quat,
            "angle_deg": None,
            "view_name": view_name,
            "pitch_deg": pitch_deg,
            "look_target": look_target,
            "lighting_phase": lighting_phase,
        }
    _place_and_capture("_normal", "room_center_single_view_normal", "normal")

    changed_light_names, changed_scales, changed_stage_lights, change_strategy = _sample_large_light_change(
        room_object_by_name, light_names, rng
    )
    changed_light_states = _apply_per_light_scales(room_object_by_name, changed_light_names, changed_scales)
    changed_stage_light_states = _apply_stage_light_scales(changed_stage_lights)
    try:
        _place_and_capture("_changed", "room_center_single_view_changed", "changed")
    finally:
        _restore_light_scale(changed_light_states)
        _restore_stage_light_scales(changed_stage_light_states)

    lighting_change = {
        "change_strategy": change_strategy,
        "normal_view_names": ["room_center_single_view_normal"],
        "changed_view_names": ["room_center_single_view_changed"],
        "changed_lights": changed_light_names,
        "changed_light_scales": [round(float(scale), 4) for scale in changed_scales],
        "changed_stage_lights": [
            {
                "path": item["path"],
                "type_name": item["type_name"],
                "scale": round(float(item["scale"]), 4),
            }
            for item in changed_stage_lights
        ],
    }
    return camera_poses, lighting_change


def _clear_render_objects(ball_pool: list, confuser_cache: dict[str, object], hidden_box_cache: dict[str, object] | None = None) -> None:
    target_cache = ball_pool if isinstance(ball_pool, dict) else {}
    conf_cache = confuser_cache if isinstance(confuser_cache, dict) else {}
    try:
        _repark_object_caches(target_cache, conf_cache, hidden_box_cache=hidden_box_cache)
        _step_sim(SIM_STEP_MINIMAL)
    except Exception as exc:
        _log_exception("clear_render_objects", exc)


def _prepare_hidden_in_box_entries(
    scene,
    scene_name: str,
    room_name: str,
    base_seed: int,
    entries: list[dict],
    hidden_box_cache: dict[str, object],
    room_objects: list[RuntimeObjectRecord],
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy,
    agent_pos,
    timing_scope: str = "unspecified",
) -> None:
    start_time = time.perf_counter()
    supports = _hidden_box_support_candidates(room_objects, float(floor_record.bbox_max[2]))
    wall_records = [record for record in room_objects if "wall" in str(record.category).lower()]
    occupied_bboxes: list[tuple[float, float, float, float]] = []
    spawned_container_count = 0
    finalize_record_s_total = 0.0
    finalize_ball_position_s_total = 0.0
    for entry in entries:
        container_start_time = time.perf_counter()
        spec = entry.get("container_spec") or {}
        name = spec.get("name")
        category = spec.get("category")
        model = spec.get("model")
        orientation = spec.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        if not (name and category and model):
            continue
        metadata_s = 0.0
        placement_resolve_s = 0.0
        orientation_s = 0.0
        spawn_s = 0.0
        post_spawn_s = 0.0
        record_s = 0.0
        placement_source = "missing"
        try:
            metadata_start = time.perf_counter()
            extents = _hidden_box_extents_from_metadata(category, model)
            metadata_s = time.perf_counter() - metadata_start
        except Exception as exc:
            _log_exception(
                f"Failed to read hidden-box metadata category={category} model={model} name={name}",
                exc,
            )
            continue

        placement = None
        placement_resolve_start = time.perf_counter()
        existing_placement = spec.get("placement")
        if isinstance(existing_placement, dict):
            existing_position = existing_placement.get("position")
            if isinstance(existing_position, (list, tuple)) and len(existing_position) >= 3:
                placement = {
                    key: value
                    for key, value in existing_placement.items()
                }
                placement["position"] = [float(existing_position[0]), float(existing_position[1]), float(existing_position[2])]
                placement_source = str(existing_placement.get("placement_type") or "existing_placement")

        if placement is None:
            container_payload = entry.get("container_object") or {}
            container_bbox = container_payload.get("bbox")
            if isinstance(container_bbox, list) and len(container_bbox) == 2:
                try:
                    bbox_min = np.array(container_bbox[0], dtype=float).reshape(-1)
                    bbox_max = np.array(container_bbox[1], dtype=float).reshape(-1)
                    if bbox_min.size >= 3 and bbox_max.size >= 3:
                        placement = {
                            "placement_type": "existing_container_bbox_center",
                            "position": [
                                float((bbox_min[0] + bbox_max[0]) * 0.5),
                                float((bbox_min[1] + bbox_max[1]) * 0.5),
                                float((bbox_min[2] + bbox_max[2]) * 0.5),
                            ],
                        }
                        placement_source = "existing_container_bbox_center"
                except Exception as exc:
                    _log_exception(f"Failed to reuse stored hidden-box bbox for {name}", exc)

        if placement is None:
            placement_rng = random.Random(
                _scoped_seed(base_seed, scene_name, room_name, name, "hidden_box_placement")
            )
            if _hidden_box_is_small_container(extents):
                for support_record in supports:
                    placement = _hidden_box_place_on_support(support_record, extents, occupied_bboxes, placement_rng)
                    if placement is not None:
                        placement_source = "support"
                        break
            if placement is None:
                placement = _hidden_box_place_on_floor(
                    floor_record,
                    blockers,
                    room_bbox_xyxy,
                    extents,
                    occupied_bboxes,
                    agent_pos,
                    placement_rng,
                )
                if placement is not None:
                    placement_source = str(placement.get("placement_type") or "floor")
        placement_resolve_s = time.perf_counter() - placement_resolve_start
        if placement is None:
            continue

        orientation_start = time.perf_counter()
        if "box" in str(category).lower():
            front_clearance = HIDDEN_BOX_BOX_FRONT_CLEARANCE_M
            front_x = float(placement["position"][0]) + float(extents[0]) * 0.5 + float(front_clearance)
            front_band = (
                float(placement["position"][0]) + float(extents[0]) * 0.5,
                float(placement["position"][1]) - float(extents[1]) * 0.5 - 0.02,
                front_x,
                float(placement["position"][1]) + float(extents[1]) * 0.5 + 0.02,
            )
            if any(_bboxes_intersect_xy(front_band, wall.bbox_world_xy) for wall in wall_records):
                orientation = _yaw_to_quaternion_xyzw(180.0)
            else:
                orientation = _yaw_to_quaternion_xyzw(0.0)
            spec["orientation"] = [float(v) for v in orientation]
        orientation_s = time.perf_counter() - orientation_start

        try:
            spawn_start = time.perf_counter()
            obj, resolved_model = _spawn_hidden_box_container_at_pose(
                scene=scene,
                cache=hidden_box_cache,
                entry_name=name,
                category=category,
                model=model,
                position=placement["position"],
                orientation=orientation,
                seed=_scoped_seed(base_seed, scene_name, room_name, name, "hidden_box_fallback_model"),
            )
            spawn_s = time.perf_counter() - spawn_start
        except Exception as exc:
            _log_exception(
                f"Failed to spawn hidden-box container category={category} model={model} name={name}",
                exc,
            )
            continue
        spec["model"] = resolved_model
        post_spawn_start = time.perf_counter()
        _close_container_if_possible(obj)
        _force_container_lid_visible(obj)
        _step_sim(SIM_STEP_MINIMAL)
        post_spawn_s = time.perf_counter() - post_spawn_start
        record_start = time.perf_counter()
        placed_record = _runtime_record_from_obj(obj)
        record_s = time.perf_counter() - record_start
        occupied_bboxes.append(placed_record.bbox_world_xy)
        spec["placement"] = placement
        spawned_container_count += 1
        _log_timing(
            "hidden_box_prepare_container",
            scene=scene_name,
            room=room_name,
            scope=timing_scope,
            container_name=name,
            category=category,
            model=resolved_model,
            placement_type=placement_source,
            metadata_s=metadata_s,
            placement_resolve_s=placement_resolve_s,
            orientation_s=orientation_s,
            spawn_s=spawn_s,
            post_spawn_s=post_spawn_s,
            record_s=record_s,
            total_s=time.perf_counter() - container_start_time,
        )
    _step_sim(SIM_STEP_MINIMAL)

    for entry in entries:
        spec = entry.get("container_spec") or {}
        name = spec.get("name")
        if not name:
            continue
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        finalize_record_start = time.perf_counter()
        record = _runtime_record_from_obj(obj)
        finalize_record_s_total += time.perf_counter() - finalize_record_start
        entry["container_object"] = record.to_json()
        spec["size_class"] = "small" if _hidden_box_is_small_container(record.extents) else "large"
        ball_position_start = time.perf_counter()
        if entry.get("contains_ball"):
            entry["ball_positions"] = [_hidden_box_candidate_ball_position(record.bbox_min, record.bbox_max)]
        else:
            entry["ball_positions"] = []
        finalize_ball_position_s_total += time.perf_counter() - ball_position_start
    _log_timing(
        "hidden_box_prepare",
        scene=scene_name,
        room=room_name,
        scope=timing_scope,
        entry_count=len(entries),
        support_count=len(supports),
        wall_count=len(wall_records),
        spawned_count=spawned_container_count,
        finalize_record_s=finalize_record_s_total,
        finalize_ball_position_s=finalize_ball_position_s_total,
        elapsed_s=time.perf_counter() - start_time,
    )


def _place_task_targets(
    scene,
    target_pool: list[object],
    entries: list[dict],
    *,
    scene_name: str | None = None,
    room_name: str | None = None,
    case_name: str | None = None,
    entry_idx: int | None = None,
) -> tuple[int, list[tuple[object, bool | None]], list[dict]]:
    start_time = time.perf_counter()
    placed_count = 0
    restore_states: list[tuple[object, bool | None]] = []
    placed_targets: list[dict] = []
    seen = set()
    container_lookup_s_total = 0.0
    placement_s_total = 0.0
    serialize_s_total = 0.0
    for entry in entries:
        for pos in entry.get("ball_positions", []):
            if placed_count >= len(target_pool):
                break
            target_start_time = time.perf_counter()
            key = (
                round(float(pos[0]), 4),
                round(float(pos[1]), 4),
                round(float(pos[2]), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            target_obj = target_pool[placed_count]
            container_payload = entry.get("container_object") or {}
            container_name = container_payload.get("name")
            container_state = entry.get("container_state") or {}
            desired_open = bool(container_state.get("open", False))
            container_obj = None
            lookup_s = 0.0
            placement_s = 0.0
            serialize_s = 0.0
            if container_name:
                lookup_start = time.perf_counter()
                container_obj = scene.object_registry("name", container_name)
                lookup_s = time.perf_counter() - lookup_start
                container_lookup_s_total += lookup_s
                if container_obj is not None:
                    placement_start = time.perf_counter()
                    original_open = _try_place_inside_container(
                        target_obj,
                        container_obj,
                        [float(pos[0]), float(pos[1]), float(pos[2])],
                        desired_open=desired_open,
                    )
                    placement_s = time.perf_counter() - placement_start
                    placement_s_total += placement_s
                    restore_states.append((container_obj, original_open))
                else:
                    placement_start = time.perf_counter()
                    _place_ball(
                        target_obj,
                        pos,
                        keep_still=True,
                        force_direct_placement=RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT or (entry.get("case") == "hidden_in_box"),
                    )
                    placement_s = time.perf_counter() - placement_start
                    placement_s_total += placement_s
            else:
                placement_start = time.perf_counter()
                _place_ball(
                    target_obj,
                    pos,
                    keep_still=True,
                    force_direct_placement=RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT or (entry.get("case") == "hidden_in_box"),
                )
                placement_s = time.perf_counter() - placement_start
                placement_s_total += placement_s
            if entry.get("case") == "hidden_in_box" and container_name:
                lookup_start = time.perf_counter()
                container_obj = scene.object_registry("name", container_name)
                second_lookup_s = time.perf_counter() - lookup_start
                lookup_s += second_lookup_s
                container_lookup_s_total += second_lookup_s
            serialize_start = time.perf_counter()
            placed_targets.append(
                _serialize_live_scene_object(
                    target_obj,
                    role="count_target",
                    requested_position=[float(pos[0]), float(pos[1]), float(pos[2])],
                    container_name=container_name,
                    source_entry_case=entry.get("case"),
                    source_category=(entry.get("count_object") or {}).get("category"),
                    source_sampling_source=(entry.get("count_object") or {}).get("sampling_source"),
                )
            )
            serialize_s = time.perf_counter() - serialize_start
            serialize_s_total += serialize_s
            placed_targets[-1]["target_obj"] = target_obj
            placed_targets[-1]["target_name"] = getattr(target_obj, "name", f"target_{placed_count:03d}")
            placed_targets[-1]["container_obj"] = container_obj
            placed_targets[-1]["entry_case"] = entry.get("case")
            if entry.get("case") == "hidden_in_box":
                _log_timing(
                    "hidden_box_place_target",
                    scene=scene_name,
                    room=room_name,
                    case=case_name,
                    q_idx=entry_idx,
                    target_name=placed_targets[-1]["target_name"],
                    target_category=(entry.get("count_object") or {}).get("category"),
                    container_name=container_name,
                    lookup_s=lookup_s,
                    placement_s=placement_s,
                    serialize_s=serialize_s,
                    total_s=time.perf_counter() - target_start_time,
                )
            placed_count += 1
    if placed_count:
        _step_sim(SIM_STEP_MINIMAL)
    _log_timing(
        "place_targets",
        entry_count=len(entries),
        target_pool_size=len(target_pool),
        unique_positions=len(seen),
        placed_count=placed_count,
        restore_state_count=len(restore_states),
        container_lookup_s=container_lookup_s_total,
        placement_s=placement_s_total,
        serialize_s=serialize_s_total,
        elapsed_s=time.perf_counter() - start_time,
    )
    return placed_count, restore_states, placed_targets


def _place_semantic_confusers(
    scene,
    scene_name: str,
    room_name: str,
    base_seed: int,
    case_name: str,
    confuser_cache: dict[str, list[object]],
    entries: list[dict],
    floor_record: RuntimeObjectRecord,
    rng: random.Random,
) -> tuple[int, list[dict]]:
    start_time = time.perf_counter()
    placed_count = 0
    placed_confusers: list[dict] = []
    category_counts: dict[str, int] = {}
    for entry in entries:
        confuser_payload = entry.get("confuser_object") or {}
        position = confuser_payload.get("position")
        category = confuser_payload.get("category")
        if position is None or not category:
            continue
        category_idx = category_counts.get(category, 0)
        try:
            confuser_seed = _scoped_seed(
                base_seed,
                scene_name,
                room_name,
                case_name,
                category,
                category_idx,
                "confuser_pool",
            )
            confuser_pool = _ensure_render_dataset_object_pool(
                scene,
                cache=confuser_cache,
                category=category,
                count=category_idx + 1,
                seed=confuser_seed,
                name_prefix=RENDER_CONFUSER_PREFIX,
                force_direct_placement=RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT,
            )
        except Exception as exc:
            _log_exception(f"Failed to build semantic confuser pool for category={category}", exc)
            continue
        if len(confuser_pool) <= category_idx:
            print(
                f"[render] semantic_fault confuser skipped category={category} "
                f"requested_index={category_idx}",
                flush=True,
            )
            continue
        try:
            _place_confuser_on_floor(confuser_pool[category_idx], position, floor_record)
        except Exception as exc:
            _log_exception(f"Failed to place semantic confuser category={category}", exc)
            continue
        placed_confusers.append(
            _serialize_live_scene_object(
                confuser_pool[category_idx],
                role="confuser",
                requested_position=[float(position[0]), float(position[1]), float(position[2])],
                source_entry_case=entry.get("case"),
                source_category=category,
                source_clip_score=confuser_payload.get("clip_score"),
                source_sampling_source=confuser_payload.get("source"),
            )
        )
        placed_confusers[-1]["confuser_pool_seed"] = int(confuser_seed)
        category_counts[category] = category_idx + 1
        placed_count += 1
    if placed_count:
        _step_sim(SIM_STEP_MINIMAL)
    _log_timing(
        "semantic_confusers",
        scene=scene_name,
        room=room_name,
        entry_count=len(entries),
        placed_count=placed_count,
        unique_categories=len(category_counts),
        elapsed_s=time.perf_counter() - start_time,
    )
    return placed_count, placed_confusers


def _render_all_candidates(
    scene,
    scene_name: str,
    room_name: str,
    base_seed: int,
    floor_record: RuntimeObjectRecord,
    room_objects: list[RuntimeObjectRecord],
    blockers: list[RuntimeObjectRecord],
    room_object_by_name: dict[str, RuntimeObjectRecord],
    room_bbox_info: dict,
    cases: dict[str, list[dict]],
    render_root: str,
    agent_pos,
    rng: random.Random,
    hidden_box_cache: dict[str, object] | None = None,
    extra_markers_by_case: dict[str, list[dict]] | None = None,
    trav_map=None,
    trav_map_img=None,
) -> dict:
    render_all_start = time.perf_counter()
    os.makedirs(render_root, exist_ok=True)
    target_object_cache: dict[str, list[object]] = {}
    confuser_cache: dict[str, list[object]] = {}
    render_summary = {}
    room_center_xy = None
    if room_bbox_info.get("expanded_bbox_world_xy") is not None:
        xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(room_bbox_info["expanded_bbox_world_xy"], name="room_bbox")
        room_center_xy = [float((xmin + xmax) * 0.5), float((ymin + ymax) * 0.5)]
    else:
        room_center_xy = [float(floor_record.center[0]), float(floor_record.center[1])]
    case_render_order = _ordered_task_types(
        [case_name for case_name in COUNTING_TASK_TYPES if case_name != "hidden_in_box"] + ["hidden_in_box"]
    )
    for case_name in case_render_order:
        case_start_time = time.perf_counter()
        entries = cases.get(case_name, [])
        try:
            _clear_render_objects(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
            hidden_box_render_entries = []
            if case_name == "hidden_in_box" and entries:
                hidden_box_render_entries = []
                for entry in entries:
                    hidden_box_render_entries.extend(_entry_component_entries(entry))
                _prepare_hidden_in_box_entries(
                    scene,
                    scene_name,
                    room_name,
                    base_seed,
                    hidden_box_render_entries,
                    hidden_box_cache or {},
                    room_objects=room_objects,
                    floor_record=floor_record,
                    blockers=blockers,
                    room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                    agent_pos=agent_pos,
                    timing_scope=f"{case_name}:case_setup",
                )
            render_summary[case_name] = 0
            for entry_idx, entry in enumerate(entries):
                entry_start_time = time.perf_counter()
                entry_dir = os.path.join(render_root, case_name, f"q_{entry_idx:03d}")
                image_prefix = f"q_{entry_idx:03d}"
                render_entries = _entry_component_entries(entry)
                projected_positions = []
                container_restore_states: list[tuple[object, bool | None]] = []
                count_object = dict(entry.get("count_object") or {})
                target_category = count_object.get("category")
                target_model = count_object.get("target_model")
                target_pool = []
                target_pool_seed = None
                hidden_box_prepare_s = 0.0
                target_pool_s = 0.0
                confuser_s = 0.0
                target_place_s = 0.0
                image_render_s = 0.0
                closeup_s = 0.0
                topdown_s = 0.0
                room_views_s = 0.0
                success = False
                try:
                    _clear_render_objects(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
                    if case_name == "hidden_in_box" and render_entries:
                        hidden_box_prepare_start = time.perf_counter()
                        _prepare_hidden_in_box_entries(
                            scene,
                            scene_name,
                            room_name,
                            base_seed,
                            render_entries,
                            hidden_box_cache or {},
                            room_objects=room_objects,
                            floor_record=floor_record,
                            blockers=blockers,
                            room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                            agent_pos=agent_pos,
                            timing_scope=f"{case_name}:entry_{entry_idx:03d}",
                        )
                        hidden_box_prepare_s = time.perf_counter() - hidden_box_prepare_start
                    if target_category:
                        try:
                            target_pool_start = time.perf_counter()
                            if not target_model:
                                candidate_models = _get_candidate_models_for_category(
                                    target_category,
                                    seed=_scoped_seed(
                                        base_seed,
                                        scene_name,
                                        room_name,
                                        case_name,
                                        entry_idx,
                                        target_category,
                                        "target_model",
                                    ),
                                )
                                if not candidate_models:
                                    raise RuntimeError(f"No usable render models remain for category '{target_category}'")
                                target_model = candidate_models[0]
                                count_object["target_model"] = target_model
                                entry["count_object"] = count_object
                            target_pool_seed = _scoped_seed(
                                base_seed,
                                scene_name,
                                room_name,
                                case_name,
                                target_category,
                                target_model,
                                "target_pool",
                            )
                            target_pool = _ensure_render_dataset_object_pool(
                                scene,
                                cache=target_object_cache,
                                category=target_category,
                                count=_count_unique_ball_positions(render_entries),
                                seed=target_pool_seed,
                                name_prefix=RENDER_TARGET_PREFIX,
                                fixed_model=target_model,
                                force_direct_placement=RENDER_OBJECTS_FORCE_DIRECT_PLACEMENT or (case_name == "hidden_in_box"),
                            )
                            target_pool_s = time.perf_counter() - target_pool_start
                        except Exception as exc:
                            _log_exception(f"Failed to build target render pool for category={target_category}", exc)
                            target_pool = []
                    projected_positions = _collect_selected_item_positions(
                        render_entries,
                        extra_markers=(extra_markers_by_case or {}).get(case_name),
                    )
                    confuser_count = 0
                    resolved_confusers: list[dict] = []
                    if case_name == "semantic_fault":
                        _repark_object_caches(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
                        confuser_start = time.perf_counter()
                        confuser_count, resolved_confusers = _place_semantic_confusers(
                            scene,
                            scene_name,
                            room_name,
                            base_seed,
                            case_name,
                            confuser_cache,
                            render_entries,
                            floor_record,
                            rng,
                        )
                        confuser_s = time.perf_counter() - confuser_start
                    target_place_start = time.perf_counter()
                    placed_ball_count, container_restore_states, placed_targets = _place_task_targets(
                        scene,
                        target_pool,
                        render_entries,
                        scene_name=scene_name,
                        room_name=room_name,
                        case_name=case_name,
                        entry_idx=entry_idx,
                    )
                    target_place_s = time.perf_counter() - target_place_start
                    resolved_targets = [
                        {
                            key: value
                            for key, value in placed_target.items()
                            if key not in {"target_obj", "container_obj"}
                        }
                        for placed_target in placed_targets
                    ]
                    resolved_containers = _collect_resolved_hidden_box_containers(scene, render_entries)
                    _attach_resolved_target_metadata(entry, resolved_targets)
                    if placed_ball_count > 0:
                        _refresh_camera_modalities_for_dynamic_objects(agent_pos, floor_record.bbox_max[2])
                    lighting_change = None
                    if case_name == "light_change":
                        light_name_set = set()
                        light_names = []
                        for render_entry in render_entries:
                            for item in (render_entry.get("lighting", {}) or {}).get("target_lights", []):
                                light_name = item.get("name")
                                if not light_name or light_name in light_name_set:
                                    continue
                                light_name_set.add(light_name)
                                light_names.append(light_name)
                        image_render_start = time.perf_counter()
                        camera_poses, lighting_change = _render_light_change_and_save(
                            image_prefix=image_prefix,
                            output_dir=entry_dir,
                            agent_pos=agent_pos,
                            floor_record=floor_record,
                            room_object_by_name=room_object_by_name,
                            light_names=light_names,
                            rng=rng,
                            room_center_xy=room_center_xy,
                            blockers=blockers,
                            room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                            trav_map=trav_map,
                            trav_map_img=trav_map_img,
                        )
                        image_render_s = time.perf_counter() - image_render_start
                    elif case_name == "observation_merged":
                        image_render_start = time.perf_counter()
                        camera_poses = _render_observation_merged_and_save(
                            image_prefix=image_prefix,
                            output_dir=entry_dir,
                            selected_positions=projected_positions,
                            agent_pos=agent_pos,
                            floor_record=floor_record,
                            room_center_xy=room_center_xy,
                            blockers=blockers,
                            room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                            trav_map=trav_map,
                            trav_map_img=trav_map_img,
                        )
                        image_render_s = time.perf_counter() - image_render_start
                    else:
                        image_render_start = time.perf_counter()
                        camera_poses = render_and_save(
                            image_prefix=image_prefix,
                            output_dir=entry_dir,
                            agent_pos=agent_pos,
                            floor_record=floor_record,
                            room_center_xy=room_center_xy,
                            blockers=blockers,
                            room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                            trav_map=trav_map,
                            trav_map_img=trav_map_img,
                        )
                        image_render_s = time.perf_counter() - image_render_start
                    image_paths = [os.path.join(entry_dir, filename) for filename in camera_poses]
                    closeup_start = time.perf_counter()
                    target_closeups = _capture_target_closeups(
                        output_dir=os.path.join(entry_dir, "target_closeups"),
                        placements=placed_targets,
                        agent_pos=agent_pos,
                        floor_record=floor_record,
                        blockers=blockers,
                        room_bbox_xyxy=room_bbox_info.get("expanded_bbox_world_xy"),
                    )
                    closeup_s = time.perf_counter() - closeup_start
                    topdown_start = time.perf_counter()
                    topdown_map_path = _save_shared_topdown_map(
                        scene=scene,
                        scene_name=scene_name,
                        floor_record=floor_record,
                        agent_pos=agent_pos,
                        selected_positions=projected_positions,
                        output_path=os.path.join(entry_dir, f"{image_prefix}_topdown_map.png"),
                        room_bbox_info=room_bbox_info,
                    )
                    topdown_s = time.perf_counter() - topdown_start
                    room_views_start = time.perf_counter()
                    room_corner_views = _capture_room_corner_views(
                        output_dir=os.path.join(entry_dir, "room_views"),
                        floor_record=floor_record,
                        room_bbox_info=room_bbox_info,
                        blockers=blockers,
                        trav_map=trav_map,
                        trav_map_img=trav_map_img,
                    )
                    room_views_s = time.perf_counter() - room_views_start
                    entry_render = {
                        "success": True,
                        "output_dir": entry_dir,
                        "images": image_paths,
                        "camera_poses": camera_poses,
                        "topdown_map": topdown_map_path,
                        "room_bbox": room_bbox_info,
                        "shared_across_candidates": False,
                        "shared_across_task": False,
                        "question_index": int(entry_idx),
                        "projected_item_count": len(projected_positions),
                        "placed_target_count": placed_ball_count,
                        "placed_confuser_count": confuser_count,
                        "target_category": target_category,
                        "target_model": target_model,
                        "target_pool_seed": target_pool_seed,
                        "multi_image_input": room_corner_views,
                        "target_closeups": target_closeups,
                        "resolved_objects": {
                            "targets": resolved_targets,
                            "confusers": resolved_confusers,
                            "containers": resolved_containers,
                        },
                    }
                    if lighting_change is not None:
                        entry_render["lighting_change"] = lighting_change
                    entry["resolved_objects"] = copy.deepcopy(entry_render["resolved_objects"])
                    entry["render"] = entry_render
                    render_summary[case_name] += 1
                    success = True
                except Exception as exc:
                    entry["render"] = {
                        "success": False,
                        "output_dir": entry_dir,
                        "images": [],
                        "camera_poses": {},
                        "topdown_map": None,
                        "room_bbox": room_bbox_info,
                        "shared_across_candidates": False,
                        "shared_across_task": False,
                        "question_index": int(entry_idx),
                        "projected_item_count": len(projected_positions),
                        "placed_target_count": 0,
                        "placed_confuser_count": 0,
                        "target_category": target_category,
                        "target_model": target_model,
                        "target_pool_seed": target_pool_seed,
                        "multi_image_input": None,
                        "target_closeups": [],
                        "resolved_objects": {
                            "targets": [],
                            "confusers": [],
                            "containers": [],
                        },
                    }
                    entry["resolved_objects"] = copy.deepcopy(entry["render"]["resolved_objects"])
                    _log_exception(f"render candidate failed: {case_name}/q_{entry_idx:03d}", exc)
                finally:
                    entry_total_s = time.perf_counter() - entry_start_time
                    _log_timing(
                        "render_entry",
                        scene=scene_name,
                        room=room_name,
                        case=case_name,
                        q_idx=entry_idx,
                        success=int(success),
                        render_entry_count=len(render_entries),
                        projected_item_count=len(projected_positions),
                        target_category=target_category,
                        target_model=target_model,
                        target_pool_size=len(target_pool),
                        placed_target_count=entry.get("render", {}).get("placed_target_count"),
                        placed_confuser_count=entry.get("render", {}).get("placed_confuser_count"),
                        hidden_box_prepare_s=hidden_box_prepare_s,
                        target_pool_s=target_pool_s,
                        confuser_s=confuser_s,
                        target_place_s=target_place_s,
                        image_render_s=image_render_s,
                        closeup_s=closeup_s,
                        topdown_s=topdown_s,
                        room_views_s=room_views_s,
                        total_s=entry_total_s,
                    )
                    if case_name == "hidden_in_box":
                        _log_timing(
                            "hidden_box_render_breakdown",
                            scene=scene_name,
                            room=room_name,
                            q_idx=entry_idx,
                            hidden_box_prepare_s=hidden_box_prepare_s,
                            target_pool_s=target_pool_s,
                            target_place_s=target_place_s,
                            image_render_s=image_render_s,
                            closeup_s=closeup_s,
                            topdown_s=topdown_s,
                            room_views_s=room_views_s,
                            residual_s=max(
                                0.0,
                                entry_total_s
                                - (
                                    hidden_box_prepare_s
                                    + target_pool_s
                                    + target_place_s
                                    + image_render_s
                                    + closeup_s
                                    + topdown_s
                                    + room_views_s
                                ),
                            ),
                            total_s=entry_total_s,
                        )
                    for container_obj, original_open in reversed(container_restore_states):
                        _restore_container_open_state(container_obj, original_open)
                    _clear_render_objects(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
        except Exception as exc:
            render_summary[case_name] = 0
            _log_exception(f"render case failed: {case_name}", exc)
        finally:
            _log_timing(
                "render_case",
                scene=scene_name,
                room=room_name,
                case=case_name,
                entry_count=len(entries),
                success_count=render_summary.get(case_name, 0),
                total_s=time.perf_counter() - case_start_time,
            )
            _clear_render_objects(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
    _clear_render_objects(target_object_cache, confuser_cache, hidden_box_cache=hidden_box_cache)
    _log_timing(
        "render_all_candidates",
        scene=scene_name,
        room=room_name,
        case_count=len(cases),
        total_s=time.perf_counter() - render_all_start,
    )
    return render_summary


def _counting_answer_for_case(case_name: str, entry: dict) -> str:
    render_info = entry.get("render") or {}
    placed_target_count = render_info.get("placed_target_count")
    if placed_target_count is not None:
        return str(int(placed_target_count))
    return str(len(_entry_ball_positions(entry)))


def _build_counting_render(entry: dict) -> dict:
    render_info = entry.get("render") or {}
    images = list(render_info.get("images") or [])
    primary_image = images[0] if images else None
    context_image = images[1] if len(images) > 1 else primary_image
    render_payload = {
        "image": primary_image,
        "topdown_map": render_info.get("topdown_map"),
        "context_without_mirror": {
            "image": context_image,
            "kind": "agent_centered_alternate_view",
        },
        "all_images": images,
        "camera_poses": render_info.get("camera_poses", {}),
        "render_output_dir": render_info.get("output_dir"),
        "room_bbox": render_info.get("room_bbox"),
        "multi_image_input": render_info.get("multi_image_input"),
        "target_closeups": render_info.get("target_closeups", []),
        "resolved_objects": render_info.get(
            "resolved_objects",
            {"targets": [], "confusers": [], "containers": []},
        ),
        "target_category": render_info.get("target_category"),
        "target_model": render_info.get("target_model"),
        "target_pool_seed": render_info.get("target_pool_seed"),
        "success": bool(render_info.get("success")),
    }
    if len(images) > 2:
        render_payload["extra_images"] = images[2:]
    return render_payload


def _aggregate_task_entries(entries: list[dict], source_indices: list[int] | None = None) -> dict:
    if not entries:
        return {
            "ball_positions": [],
            "object_positions": [],
            "count_object": {},
            "case_metadata": {"source_candidates": 0, "source_entries": []},
            "component_entries": [],
            "render": {},
        }

    component_entries = [copy.deepcopy(entry) for entry in entries]
    ball_positions = _merged_ball_positions(component_entries)

    source_entries = []
    for idx, entry in enumerate(component_entries):
        source_entries.append(
            {
                "source_index": int(source_indices[idx] if source_indices is not None and idx < len(source_indices) else idx),
                "ball_positions": entry.get("ball_positions", []),
                "metadata": {
                    key: value
                    for key, value in entry.items()
                    if key not in {"case", "ball_positions", "render", "component_entries"}
                },
            }
        )

    return {
        "case": entries[0].get("case"),
        "ball_positions": ball_positions,
        "object_positions": list(ball_positions),
        "count_object": dict(entries[0].get("count_object") or {}),
        "case_metadata": {
            "source_candidates": len(entries),
            "source_entries": source_entries,
        },
        "component_entries": component_entries,
        "render": {},
    }


def _sample_multi_entry_questions(
    raw_entries: list[dict],
    question_count: int,
    rng: random.Random,
    min_objects_per_question: int = 3,
    max_objects_per_question: int = MAX_RANDOM_BALL_COUNT,
) -> list[dict]:
    if not raw_entries:
        return []
    available_count = len(raw_entries)
    if available_count < 2:
        return []

    upper = min(int(max_objects_per_question), available_count)
    lower = min(int(min_objects_per_question), upper)
    if upper < 2:
        return []
    if lower < 2:
        lower = 2

    questions = []
    for _ in range(max(0, int(question_count))):
        object_count = rng.randint(lower, upper)
        selected_indices = sorted(rng.sample(range(available_count), k=object_count))
        selected_entries = [raw_entries[idx] for idx in selected_indices]
        questions.append(_aggregate_task_entries(selected_entries, source_indices=selected_indices))
    return questions


def _count_unique_ball_positions(entries: list[dict]) -> int:
    seen_ball_positions = set()
    for entry in entries:
        for pos in entry.get("ball_positions", []):
            key = (round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4))
            seen_ball_positions.add(key)
    return len(seen_ball_positions)


def _limit_entries_to_ball_count(entries: list[dict], target_ball_count: int) -> list[dict]:
    if target_ball_count <= 0:
        return []

    remaining = int(target_ball_count)
    limited_entries = []
    for entry in entries:
        if remaining <= 0:
            break
        ball_positions = list(entry.get("ball_positions", []))
        if not ball_positions:
            continue
        selected_positions = ball_positions[:remaining]
        limited_entry = dict(entry)
        limited_entry["ball_positions"] = [
            [float(pos[0]), float(pos[1]), float(pos[2])] for pos in selected_positions
        ]
        limited_entries.append(limited_entry)
        remaining -= len(selected_positions)
    return limited_entries


def _sample_task_ball_counts(cases: dict[str, list[dict]], rng: random.Random) -> dict[str, int]:
    counts = {}
    for case_name in COUNTING_TASK_TYPES:
        available_ball_count = _count_unique_ball_positions(cases.get(case_name, []))
        if available_ball_count <= 0:
            counts[case_name] = 0
            continue
        lower = min(3, available_ball_count)
        upper = min(MAX_RANDOM_BALL_COUNT, available_ball_count)
        counts[case_name] = rng.randint(lower, upper)
    return counts


def _build_counting_options(answer: int, seed: int) -> list[str]:
    upper = max(3, int(answer) + 1)
    full_options = [str(idx) for idx in range(upper + 1)]
    if len(full_options) <= 4:
        return full_options

    answer_str = str(int(answer))
    distractors = [option for option in full_options if option != answer_str]
    rng = random.Random(int(seed))
    rng.shuffle(distractors)
    selected = [answer_str, *distractors[:3]]
    selected.sort(key=lambda item: int(item))
    return selected


def _build_counting_question(case_name: str, entry: dict, candidate_index: int = 0) -> dict:
    answer = int(_counting_answer_for_case(case_name, entry))
    count_object = dict(entry.get("count_object") or {})
    target_category = count_object.get("category", "object")
    ball_positions = _entry_ball_positions(entry)
    option_seed = _scoped_seed(
        answer,
        case_name,
        int(candidate_index),
        target_category,
        len(ball_positions),
        "counting_options",
    )
    return {
        "task_type": case_name,
        "question": f'Considering both images, how many objects of category "{target_category}" are present in this scene?',
        "options": _build_counting_options(answer, option_seed),
        "answer": str(answer),
        "case": case_name,
        "candidate_index": int(candidate_index),
        "count_target": target_category,
        "count_object": count_object,
        "count_unit": "objects",
        "ball_positions": ball_positions,
        "object_positions": entry.get("object_positions", ball_positions),
        "case_metadata": entry.get("case_metadata", {}),
        "render": _build_counting_render(entry),
    }


def _export_single_question_jsons(cases: dict[str, list[dict]], output_root: str, scene_metadata: dict) -> dict:
    os.makedirs(output_root, exist_ok=True)
    written: dict[str, list[str]] = {}
    for task_type in COUNTING_TASK_TYPES:
        entries = cases.get(task_type, [])
        written[task_type] = []
        for q_idx, entry in enumerate(entries):
            question_entry = _build_counting_question(task_type, entry, candidate_index=q_idx)
            out_path = _write_single_question_json(
                output_root=output_root,
                scene_metadata=scene_metadata,
                task_type=task_type,
                q_idx=q_idx,
                entry=question_entry,
            )
            written[task_type].append(out_path)
    return {
        "enabled": True,
        "question_json_root": output_root,
        "counts": {task_type: len(paths) for task_type, paths in written.items()},
        "paths": written,
    }


def _attach_count_object(entries: list[dict], count_object: dict) -> list[dict]:
    attached = []
    for entry in entries:
        updated = dict(entry)
        updated["count_object"] = dict(count_object)
        attached.append(updated)
    return attached


def _generate_hidden_by_others(room_objects, floor_record, blockers, agent_pos, room_bbox_xyxy, max_per_case):
    candidates = []
    anchors = [
        obj
        for obj in room_objects
        if _is_hidden_anchor_candidate(obj, room_bbox_xyxy=room_bbox_xyxy, floor_record=floor_record)
    ]
    anchors.sort(key=lambda rec: (_distance_xy(rec.center, agent_pos), -rec.footprint_area, rec.name))
    diagnostics = []

    for anchor in anchors:
        center_xy = anchor.center[:2]
        direction = _sub_xy(center_xy, agent_pos)
        dist_to_center = _norm_xy(direction)
        diag = {
            "anchor": _describe_runtime_object(anchor, agent_pos=agent_pos),
            "center_xy": [float(center_xy[0]), float(center_xy[1])],
            "agent_xy": [float(agent_pos[0]), float(agent_pos[1])],
            "distance_to_center_xy": round(float(dist_to_center), 4),
            "status": "rejected",
            "reasons": [],
        }
        if dist_to_center < 0.25:
            diag["reasons"].append("anchor_too_close_to_agent_center_dist_lt_0.25")
            diagnostics.append(diag)
            continue
        unit_dir = (direction[0] / dist_to_center, direction[1] / dist_to_center)
        diag["unit_dir"] = [round(float(unit_dir[0]), 4), round(float(unit_dir[1]), 4)]
        behind_offset = _support_distance_xy(anchor, unit_dir) + BALL_RADIUS + 0.03
        perp_dir = (-unit_dir[1], unit_dir[0])
        offset_candidates = [
            (0.0, 0.0),
            (0.06, 0.0),
            (0.12, 0.0),
            (-0.06, 0.0),
            (-0.12, 0.0),
            (0.0, 0.06),
            (0.0, 0.12),
        ]
        diag["behind_offset"] = round(float(behind_offset), 4)
        diag["candidate_trials"] = []
        selected_candidate_xy = None
        selected_point_diag = None

        for lateral_offset, extra_behind in offset_candidates:
            candidate_xy = _add_xy(
                _add_xy(center_xy, _scale_xy(unit_dir, behind_offset + extra_behind)),
                _scale_xy(perp_dir, lateral_offset),
            )
            trial = {
                "candidate_xy": [round(float(candidate_xy[0]), 4), round(float(candidate_xy[1]), 4)],
                "lateral_offset": round(float(lateral_offset), 4),
                "extra_behind": round(float(extra_behind), 4),
            }
            point_diag = _diagnose_point_availability(
                candidate_xy,
                floor_record,
                blockers,
                clearance=BALL_RADIUS + 0.005,
                room_bbox_xyxy=room_bbox_xyxy,
                ignore_labels={anchor.label},
                agent_pos=agent_pos,
                min_agent_distance=MIN_BALL_DISTANCE_FROM_AGENT,
            )
            trial["candidate_point_check"] = point_diag
            if not point_diag["ok"]:
                trial["status"] = "rejected"
                trial["reason"] = point_diag["failure_reason"]
                diag["candidate_trials"].append(trial)
                continue

            segment_ok = _segment_intersects_bbox_xy(agent_pos[:2], candidate_xy, anchor)
            trial["segment_intersects_anchor_bbox_xy"] = bool(segment_ok)
            if not segment_ok:
                trial["status"] = "rejected"
                trial["reason"] = "agent_to_candidate_segment_does_not_intersect_anchor_bbox"
                diag["candidate_trials"].append(trial)
                continue

            trial["status"] = "selected"
            diag["candidate_trials"].append(trial)
            selected_candidate_xy = candidate_xy
            selected_point_diag = point_diag
            break

        if selected_candidate_xy is None:
            diag["reasons"].append("no_valid_candidate_xy_found")
            diagnostics.append(diag)
            continue

        diag["candidate_xy"] = [round(float(selected_candidate_xy[0]), 4), round(float(selected_candidate_xy[1]), 4)]
        diag["candidate_point_check"] = selected_point_diag
        diag["segment_intersects_anchor_bbox_xy"] = True
        diag["status"] = "selected"
        candidates.append(
            {
                "case": "hidden_by_others",
                "anchor_object": anchor.to_json(),
                "ball_positions": [
                    [
                        float(selected_candidate_xy[0]),
                        float(selected_candidate_xy[1]),
                        floor_record.bbox_max[2] + BALL_RADIUS,
                    ]
                ],
                "visibility": {
                    "occluded_from_agent": True,
                    "occlusion_test": "agent_to_ball_segment_intersects_anchor_bbox_xy",
                },
            }
        )
        diagnostics.append(diag)
        if len(candidates) >= max_per_case:
            break
    return candidates


def _generate_observation_divided(room_objects, floor_record, blockers, agent_pos, room_bbox_xyxy, max_per_case):
    candidates = []
    chairs = [obj for obj in room_objects if _is_chair(obj)]
    chairs.sort(key=lambda rec: (_distance_xy(rec.center, agent_pos), rec.name))

    for chair in chairs:
        center_xy = chair.center[:2]
        candidate_pos = [
            float(center_xy[0]),
            float(center_xy[1]),
            float(floor_record.bbox_max[2] + BALL_RADIUS),
        ]
        candidates.append(
            {
                "case": "observation_divided",
                "anchor_object": chair.to_json(),
                "ball_positions": [candidate_pos],
                "visibility": {
                    "placement": "ball_center_aligned_with_chair_bbox_center_xy",
                    "support": "placed_on_floor_below_chair",
                },
            }
        )
        if len(candidates) >= max_per_case:
            break
    return candidates


def _generate_semantic_fault(
    free_positions,
    floor_record,
    blockers,
    room_bbox_xyxy,
    confuser_candidates: list[dict],
    max_per_case,
    agent_pos,
    rng: random.Random,
):
    candidates = []
    for idx in range(min(max_per_case, len(free_positions))):
        if not confuser_candidates:
            break
        base_pos = free_positions[idx]
        neighbor_pos = _find_neighbor_position(
            base_pos,
            floor_record,
            blockers,
            clearance=BALL_RADIUS + 0.04,
            room_bbox_xyxy=room_bbox_xyxy,
            agent_pos=agent_pos,
            min_agent_distance=MIN_BALL_DISTANCE_FROM_AGENT,
        )
        if neighbor_pos is None:
            continue
        confuser = dict(rng.choice(confuser_candidates))
        candidates.append(
            {
                "case": "semantic_fault",
                "ball_positions": [[float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]],
                "confuser_object": {
                    "category": confuser["category"],
                    "clip_score": confuser.get("clip_score"),
                    "source": confuser.get("source"),
                    "position": neighbor_pos,
                },
            }
        )
    return candidates


def _runtime_record_from_obj(obj) -> RuntimeObjectRecord:
    bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
    states = getattr(obj, "states", {}) or {}
    open_state = None
    has_open_state = object_states.Open in states
    if has_open_state:
        try:
            open_state = bool(states[object_states.Open].get_value())
        except Exception:
            open_state = None
    return RuntimeObjectRecord(
        name=str(getattr(obj, "name", "object")),
        category=str(getattr(obj, "category", "object")),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        in_rooms=tuple(str(room) for room in (getattr(obj, "in_rooms", None) or [])),
        has_open_state=has_open_state,
        open_state=open_state,
        obj=obj,
    )


def _hidden_box_is_small_container(extents: tuple[float, float, float]) -> bool:
    width = float(extents[0])
    depth = float(extents[1])
    height = float(extents[2])
    return max(width, depth) <= 0.42 and height <= 0.58 and width * depth <= 0.16


def _hidden_box_support_candidates(room_objects: list[RuntimeObjectRecord], floor_z: float) -> list[RuntimeObjectRecord]:
    allowed_support_categories = {
        "breakfast_table",
        "coffee_table",
        "console_table",
        "conference_table",
        "gaming_table",
        "pedestal_table",
        "table",
    }
    supports = []
    for record in room_objects:
        category = record.category.lower()
        if category not in allowed_support_categories:
            continue
        if category in NON_BLOCKING_CATEGORIES:
            continue
        if _is_light(record) or "wall" in category or "door" in category or "window" in category:
            continue
        if record.has_open_state:
            continue
        if record.bbox_max[2] <= floor_z + 0.18:
            continue
        if record.footprint_area < 0.18:
            continue
        if min(record.extents[0], record.extents[1]) < 0.35:
            continue
        supports.append(record)
    supports.sort(key=lambda rec: (-rec.bbox_max[2], -rec.footprint_area, rec.name))
    return supports


def _bbox_with_margin_xy(center_xy, extents_xy, margin: float = 0.0) -> tuple[float, float, float, float]:
    half_x = float(extents_xy[0]) / 2.0 + float(margin)
    half_y = float(extents_xy[1]) / 2.0 + float(margin)
    return (
        float(center_xy[0]) - half_x,
        float(center_xy[1]) - half_y,
        float(center_xy[0]) + half_x,
        float(center_xy[1]) + half_y,
    )


def _hidden_box_candidate_ball_position(bbox_min, bbox_max) -> list[float]:
    center_x = float((bbox_min[0] + bbox_max[0]) / 2.0)
    center_y = float((bbox_min[1] + bbox_max[1]) / 2.0)
    height = max(float(bbox_max[2] - bbox_min[2]), BALL_RADIUS * 2.0)
    z = float(bbox_min[2]) + 0.68 * height
    z += HIDDEN_BOX_TARGET_Z_OFFSET_M
    z = max(z, float(bbox_min[2]) + BALL_RADIUS + 0.01)
    z = min(z, float(bbox_max[2]) - BALL_RADIUS - 0.01)
    if z <= float(bbox_min[2]) + BALL_RADIUS:
        z = float((bbox_min[2] + bbox_max[2]) / 2.0)
    return [center_x, center_y, z]


def _hidden_box_support_layouts(
    support_record: RuntimeObjectRecord,
    extents: tuple[float, float, float],
    rng: random.Random,
) -> list[tuple[float, float]]:
    margin_x = float(extents[0]) / 2.0 + 0.03
    margin_y = float(extents[1]) / 2.0 + 0.03
    xmin = float(support_record.bbox_min[0]) + margin_x
    xmax = float(support_record.bbox_max[0]) - margin_x
    ymin = float(support_record.bbox_min[1]) + margin_y
    ymax = float(support_record.bbox_max[1]) - margin_y
    if xmin > xmax or ymin > ymax:
        return []
    xmid = float((xmin + xmax) / 2.0)
    ymid = float((ymin + ymax) / 2.0)
    candidates = [
        (xmid, ymid),
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymid),
        (xmax, ymid),
        (xmid, ymin),
        (xmid, ymax),
    ]
    unique_candidates = list(dict.fromkeys((round(x, 4), round(y, 4)) for x, y in candidates))
    if not unique_candidates:
        return []
    center_candidate = unique_candidates[0]
    remaining = unique_candidates[1:]
    rng.shuffle(remaining)
    ordered = [center_candidate] + remaining
    return [(float(x), float(y)) for x, y in ordered]


def _hidden_box_place_on_support(
    support_record: RuntimeObjectRecord,
    extents: tuple[float, float, float],
    occupied_bboxes: list[tuple[float, float, float, float]],
    rng: random.Random,
) -> dict | None:
    for center_xy in _hidden_box_support_layouts(support_record, extents, rng):
        bbox_xy = _bbox_with_margin_xy(center_xy, extents[:2], margin=0.01)
        if any(_bboxes_intersect_xy(bbox_xy, other_bbox) for other_bbox in occupied_bboxes):
            continue
        position = [
            float(center_xy[0]),
            float(center_xy[1]),
            float(support_record.bbox_max[2]) + float(extents[2]) / 2.0 + 0.015,
        ]
        return {
            "placement_type": "on_top",
            "support_name": support_record.name,
            "support_category": support_record.category,
            "position": [float(v) for v in position],
        }
    return None


def _hidden_box_floor_positions(
    floor_record: RuntimeObjectRecord,
    room_bbox_xyxy,
    extents: tuple[float, float, float],
    agent_pos,
    rng: random.Random,
) -> list[tuple[float, float]]:
    margin_x = float(extents[0]) / 2.0 + 0.04
    margin_y = float(extents[1]) / 2.0 + 0.04
    x_min = float(floor_record.bbox_min[0]) + margin_x
    x_max = float(floor_record.bbox_max[0]) - margin_x
    y_min = float(floor_record.bbox_min[1]) + margin_y
    y_max = float(floor_record.bbox_max[1]) - margin_y
    if room_bbox_xyxy is not None:
        rxmin, rymin, rxmax, rymax = _normalize_bbox_xyxy(room_bbox_xyxy)
        x_min = max(x_min, float(rxmin) + margin_x)
        x_max = min(x_max, float(rxmax) - margin_x)
        y_min = max(y_min, float(rymin) + margin_y)
        y_max = min(y_max, float(rymax) - margin_y)
    if x_min > x_max or y_min > y_max:
        return []

    candidates = []
    x = x_min
    while x <= x_max + 1e-6:
        y = y_min
        while y <= y_max + 1e-6:
            if _distance_xy((x, y), agent_pos[:2]) >= MIN_BALL_DISTANCE_FROM_AGENT:
                candidates.append((float(x), float(y)))
            y += GRID_STEP
        x += GRID_STEP
    rng.shuffle(candidates)
    return candidates


def _hidden_box_place_on_floor(
    floor_record: RuntimeObjectRecord,
    blockers: list[RuntimeObjectRecord],
    room_bbox_xyxy,
    extents: tuple[float, float, float],
    occupied_bboxes: list[tuple[float, float, float, float]],
    agent_pos,
    rng: random.Random,
) -> dict | None:
    for center_xy in _hidden_box_floor_positions(floor_record, room_bbox_xyxy, extents, agent_pos, rng):
        bbox_xy = _bbox_with_margin_xy(center_xy, extents[:2], margin=0.02)
        if any(_bboxes_intersect_xy(bbox_xy, blocker.bbox_world_xy) for blocker in blockers):
            continue
        if any(_bboxes_intersect_xy(bbox_xy, other_bbox) for other_bbox in occupied_bboxes):
            continue
        position = [
            float(center_xy[0]),
            float(center_xy[1]),
            float(floor_record.bbox_max[2]) + float(extents[2]) / 2.0 + 0.015,
        ]
        return {
            "placement_type": "floor",
            "position": [float(v) for v in position],
        }
    return None


def _close_container_if_possible(obj) -> None:
    states = getattr(obj, "states", {}) or {}
    if object_states.Open not in states:
        return
    try:
        states[object_states.Open].set_value(False)
        _step_sim(SIM_STEP_MINIMAL)
    except Exception as exc:
        _log_exception(f"Failed to close hidden-box container {getattr(obj, 'name', '<unknown>')}", exc)


def _generate_hidden_in_box(
    target_ball_count: int,
    rng: random.Random,
):
    selected_assets = list(HIDDEN_BOX_FIXED_ASSETS)
    placed_entries = []

    for slot_idx, (category, model) in enumerate(selected_assets):
        entry_name = f"{RENDER_HIDDEN_BOX_PREFIX}{slot_idx}_{category}"
        placed_entries.append(
            {
                "case": "hidden_in_box",
                "contains_ball": False,
                "container_object": None,
                "container_state": {"open": False},
                "container_spec": {
                    "name": entry_name,
                    "category": category,
                    "model": model,
                    "placement": None,
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "size_class": "unknown",
                },
                "ball_positions": [],
            }
        )

    if not placed_entries:
        return []

    target_ball_count = max(0, min(int(target_ball_count), len(placed_entries)))
    ball_indices = set(rng.sample(range(len(placed_entries)), k=target_ball_count))
    for idx, entry in enumerate(placed_entries):
        if idx not in ball_indices:
            continue
        entry["contains_ball"] = True
    return placed_entries


def _generate_observation_merged(
    free_positions,
    floor_record,
    blockers,
    room_bbox_xyxy,
    target_ball_count,
    rng,
    target_size_m: float | None = None,
):
    if target_ball_count <= 0:
        return []

    object_span = float(target_size_m) if target_size_m is not None else float(BALL_RADIUS * 2.0)
    object_span = min(max(object_span, float(BALL_RADIUS * 2.0)), 0.28)
    placement_clearance = max(BALL_RADIUS + 0.01, object_span * 0.42)
    min_pair_dist = max(BALL_RADIUS * 2.0 + 0.01, object_span * OBSERVATION_MERGED_MIN_SEPARATION_SCALE)
    cluster_radius = max(
        0.35,
        min(
            1.25,
            min_pair_dist * (0.65 + 0.36 * max(1, int(target_ball_count) - 1)) + OBSERVATION_MERGED_CLUSTER_MARGIN_M,
        ),
    )
    candidate_radii = (
        0.0,
        min_pair_dist * 0.95,
        min_pair_dist * 1.25,
        min_pair_dist * 1.55,
        min_pair_dist * 1.9,
    )

    shuffled_bases = list(free_positions)
    rng.shuffle(shuffled_bases)

    for base_pos in shuffled_bases:
        base_xy = (float(base_pos[0]), float(base_pos[1]))
        base_z = float(base_pos[2])
        if not _point_is_free(
            base_xy,
            floor_record,
            blockers,
            clearance=placement_clearance,
            room_bbox_xyxy=room_bbox_xyxy,
        ):
            continue

        for _ in range(64):
            cluster_xy: list[tuple[float, float]] = [base_xy]
            candidate_positions: list[tuple[float, float]] = []

            for radius in candidate_radii:
                if radius <= 1e-6:
                    candidate_positions.append(base_xy)
                    continue
                sample_count = max(10, int(math.ceil(2.0 * math.pi * radius / max(min_pair_dist * 0.85, 0.08))))
                angle_offset = rng.uniform(0.0, 2.0 * math.pi)
                for sample_idx in range(sample_count):
                    angle = angle_offset + (2.0 * math.pi * sample_idx / sample_count)
                    candidate_positions.append(
                        (
                            float(base_xy[0] + math.cos(angle) * radius),
                            float(base_xy[1] + math.sin(angle) * radius),
                        )
                    )

            rng.shuffle(candidate_positions)

            valid = True
            for candidate_xy in candidate_positions:
                if len(cluster_xy) >= int(target_ball_count):
                    break
                if _distance_xy(candidate_xy, base_xy) > cluster_radius + 1e-6:
                    continue
                if not _point_is_free(
                    candidate_xy,
                    floor_record,
                    blockers,
                    clearance=placement_clearance,
                    room_bbox_xyxy=room_bbox_xyxy,
                ):
                    continue
                if any(_distance_xy(candidate_xy, existing_xy) < min_pair_dist for existing_xy in cluster_xy):
                    continue
                cluster_xy.append(candidate_xy)

            if len(cluster_xy) == int(target_ball_count):
                cluster = [[float(x), float(y), float(base_z)] for (x, y) in cluster_xy]
                return [
                    {
                        "case": "observation_merged",
                        "ball_positions": cluster,
                        "merge_pattern": "planar_size_aware_cluster",
                        "layout_constraints": {
                            "target_size_m": round(float(object_span), 4),
                            "min_pair_dist_m": round(float(min_pair_dist), 4),
                            "cluster_radius_m": round(float(cluster_radius), 4),
                            "same_plane": True,
                        },
                    }
                ]
    return []


def _generate_light_change(free_positions, room_objects, max_per_case):
    lights = [obj for obj in room_objects if _is_light(obj)]
    if not lights:
        return []
    lights_json = [light.to_json() for light in sorted(lights, key=lambda rec: rec.name)]
    candidates = []
    for idx in range(min(max_per_case, len(free_positions))):
        candidates.append(
            {
                "case": "light_change",
                "ball_positions": [free_positions[idx]],
                "lighting": {
                    "target_lights": lights_json,
                    "render_behavior": "first_four_views_normal_then_random_large_light_change_for_last_four_views",
                },
            }
        )
    return candidates


def _summarize_room_objects(room_objects):
    counts = {}
    for obj in room_objects:
        counts[obj.category] = counts.get(obj.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _build_config(scene_name: str, robot: str, load_full_scene: bool, room_names: list[str] | tuple[str, ...]):
    config_filename = os.path.join(og.example_config_path, f"{robot.lower()}_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"] = scene_name
    if load_full_scene:
        config["scene"].pop("load_room_instances", None)
        config["scene"].pop("load_room_types", None)
        config["scene"]["not_load_object_categories"] = []
    else:
        ordered_rooms = []
        seen_rooms = set()
        for room_name in room_names:
            if room_name in seen_rooms:
                continue
            seen_rooms.add(room_name)
            ordered_rooms.append(room_name)
        config["scene"]["load_room_instances"] = ordered_rooms
    config["robots"] = []
    config["objects"] = []
    return config


def _room_run_paths(output_root: str, scene_name: str, room_name: str) -> dict[str, str]:
    run_dir = os.path.join(output_root, scene_name, room_name)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "output_json": os.path.join(run_dir, "counting_candidates.json"),
        "render_root": os.path.join(run_dir, "candidate_renders"),
        "question_json_root": os.path.join(run_dir, "counting_question_jsons"),
        "attempted_marker_path": os.path.join(run_dir, ATTEMPTED_ROOM_MARKER),
    }


def _resolve_requested_rooms(args) -> list[tuple[str, str | None]]:
    if args.rooms:
        rooms = list(args.rooms)
        floors = list(args.floors or [])
        if floors and len(floors) != len(rooms):
            raise ValueError("--floors must have the same length as --rooms when provided.")
        if not floors:
            floors = [None] * len(rooms)
        return list(zip(rooms, floors))
    return [(args.room, args.floor)]


def _process_room_in_loaded_scene(
    *,
    env,
    scene,
    scene_name: str,
    room_name: str,
    floor_name: str | None,
    output_root: str,
    seed: int,
    questions_per_task: int,
    max_per_case: int,
    skip_render: bool,
    selected_task_types: tuple[str, ...],
    count_target_candidates,
    clip_neighbors,
    wall_records,
    structural_wall_bboxes,
    agent_position,
) -> dict:
    room_start_time = time.perf_counter()
    _log_timing(
        "room_start",
        scene=scene_name,
        room=room_name,
        floor=floor_name,
        skip_render=int(bool(skip_render)),
        task_types=",".join(selected_task_types),
    )
    room_seed = _scoped_seed(int(seed), scene_name, room_name, "room_rng")
    rng = random.Random(room_seed)
    paths = _room_run_paths(output_root, scene_name, room_name)
    output_json = paths["output_json"]
    render_root = paths["render_root"]
    question_json_root = paths["question_json_root"]
    attempted_marker_path = paths["attempted_marker_path"]

    room_objects = _collect_room_objects(scene, room_name)
    floors = [obj for obj in room_objects if obj.category == "floors"]
    if not floors:
        raise ValueError(f"No floor object found in loaded room '{room_name}'.")
    if floor_name is not None:
        matching_floors = [floor for floor in floors if floor.name == floor_name]
        if not matching_floors:
            raise ValueError(f"Floor '{floor_name}' not found among loaded room objects for room '{room_name}'.")
        provisional_floor = matching_floors[0]
    else:
        provisional_floor = max(floors, key=lambda floor: floor.footprint_area)
    room_bbox_info = _resolve_room_bbox(scene, room_name, structural_wall_bboxes)
    room_bbox_area_m2 = room_bbox_info.get("bbox_area_m2")
    if room_bbox_area_m2 is not None and (
        float(room_bbox_area_m2) < MIN_ROOM_BBOX_AREA_M2 or float(room_bbox_area_m2) > MAX_ROOM_BBOX_AREA_M2
    ):
        skip_reason = (
            "room_bbox_area_below_threshold"
            if float(room_bbox_area_m2) < MIN_ROOM_BBOX_AREA_M2
            else "room_bbox_area_above_threshold"
        )
        skip_payload = {
            "scene": scene_name,
            "room": room_name,
            "seed": int(seed),
            "room_seed": int(room_seed),
            "status": "skipped",
            "skip_reason": skip_reason,
            "min_room_bbox_area_m2": float(MIN_ROOM_BBOX_AREA_M2),
            "max_room_bbox_area_m2": float(MAX_ROOM_BBOX_AREA_M2),
            "room_bbox_area_m2": float(room_bbox_area_m2),
            "room_bbox": room_bbox_info,
        }
        _write_json(output_json, skip_payload)
        _write_json(attempted_marker_path, skip_payload)
        _log_timing(
            "room_skip",
            scene=scene_name,
            room=room_name,
            skip_reason=skip_reason,
            room_bbox_area_m2=float(room_bbox_area_m2),
            total_s=time.perf_counter() - room_start_time,
        )
        return {
            "output_json": output_json,
            "attempted_marker": attempted_marker_path,
            "room": room_name,
            "status": "skipped",
            "skip_reason": skip_reason,
            "min_room_bbox_area_m2": float(MIN_ROOM_BBOX_AREA_M2),
            "max_room_bbox_area_m2": float(MAX_ROOM_BBOX_AREA_M2),
            "room_bbox_area_m2": float(room_bbox_area_m2),
        }

    room_bbox_xyxy = room_bbox_info.get("expanded_bbox_world_xy")
    blockers = [obj for obj in room_objects if _is_floor_blocker(obj, provisional_floor.bbox_max[2])]
    floor_idx = _infer_floor_idx(provisional_floor)
    trav_map, trav_map_img = _trav_map_floor_image(scene, floor_idx=floor_idx, scene_name=scene_name)
    agent_pos = _resolve_agent_position(env, agent_position)
    if agent_pos is None:
        agent_pos = _sample_agent_position_near_short_edge(
            floor_record=provisional_floor,
            blockers=blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )
    floor_record = _select_floor(room_objects, floor_name, agent_pos)
    blockers = [obj for obj in room_objects if _is_floor_blocker(obj, floor_record.bbox_max[2])]
    task_count_objects = {
        case_name: [
            _sample_count_target_with_seed(
                count_target_candidates,
                _count_target_seed_for_question(
                    base_seed=int(seed),
                    scene_name=scene_name,
                    room_name=room_name,
                    task_type=case_name,
                    q_idx=q_idx,
                ),
            )
            for q_idx in range(
                int(
                    _question_count_for_task(
                        case_name,
                        max(0, int(questions_per_task)),
                    )
                )
            )
        ]
        for case_name in selected_task_types
    }
    semantic_fault_confuser_candidates = []
    if "semantic_fault" in selected_task_types and task_count_objects.get("semantic_fault"):
        semantic_fault_confuser_candidates = _sample_semantic_confuser_candidates_for_target(
            task_count_objects["semantic_fault"][0]["category"],
            clip_neighbors,
            rng,
            count_target_candidates,
        )
    hidden_box_cache: dict[str, object] = {}
    free_positions = _generate_free_positions(
        floor_record=floor_record,
        blockers=blockers,
        agent_pos=agent_pos,
        count=max(max_per_case * 4, 24),
        room_bbox_xyxy=room_bbox_xyxy,
    )
    hidden_anchor_candidates = [
        obj
        for obj in room_objects
        if _is_hidden_anchor_candidate(obj, room_bbox_xyxy=room_bbox_xyxy, floor_record=floor_record)
    ]

    raw_cases = {
        "hidden_by_others": _generate_hidden_by_others(
            room_objects, floor_record, blockers, agent_pos, room_bbox_xyxy, max_per_case
        ),
        "observation_divided": _generate_observation_divided(
            room_objects, floor_record, blockers, agent_pos, room_bbox_xyxy, max_per_case
        ),
        "semantic_fault": _generate_semantic_fault(
            free_positions,
            floor_record,
            blockers,
            room_bbox_xyxy,
            semantic_fault_confuser_candidates,
            max_per_case,
            agent_pos,
            rng,
        ),
        "hidden_in_box": [],
        "observation_merged": [],
        "light_change": _generate_light_change(free_positions, room_objects, max_per_case),
    }

    cases: dict[str, list[dict]] = {}
    task_ball_counts: dict[str, list[int]] = {}
    default_question_count = max(0, int(questions_per_task))
    question_count_by_task = {
        case_name: _question_count_for_task(case_name, default_question_count)
        for case_name in selected_task_types
    }
    for case_name in selected_task_types:
        question_count = int(question_count_by_task.get(case_name, default_question_count))
        if case_name == "observation_merged":
            case_questions = []
            for q_idx in range(question_count):
                count_object = dict(task_count_objects[case_name][q_idx])
                if not free_positions:
                    break
                merged_upper = min(MAX_RANDOM_BALL_COUNT, len(free_positions))
                merged_lower = min(3, merged_upper)
                if merged_upper <= 0:
                    break
                target_ball_count = rng.randint(merged_lower, merged_upper)
                generated = _generate_observation_merged(
                    free_positions,
                    floor_record,
                    blockers,
                    room_bbox_xyxy,
                    target_ball_count,
                    rng,
                    target_size_m=count_object.get("characteristic_size_m"),
                )
                if not generated:
                    continue
                case_questions.append(_aggregate_task_entries(_attach_count_object(generated, count_object)))
            cases[case_name] = case_questions
            task_ball_counts[case_name] = [len(_entry_ball_positions(entry)) for entry in case_questions]
            continue

        if case_name == "hidden_in_box":
            case_questions = []
            for q_idx in range(question_count):
                count_object = dict(task_count_objects[case_name][q_idx])
                target_ball_count = rng.randint(3, HIDDEN_BOX_CONTAINER_COUNT)
                generated = _generate_hidden_in_box(
                    target_ball_count=target_ball_count,
                    rng=rng,
                )
                if not generated:
                    continue
                case_questions.append(_aggregate_task_entries(_attach_count_object(generated, count_object)))
            cases[case_name] = case_questions
            task_ball_counts[case_name] = [len(_entry_ball_positions(entry)) for entry in case_questions]
            continue

        case_questions = _sample_multi_entry_questions(
            raw_entries=raw_cases.get(case_name, []),
            question_count=question_count,
            rng=rng,
        )
        case_questions = [
            _attach_count_object([entry], dict(task_count_objects[case_name][q_idx]))[0]
            for q_idx, entry in enumerate(case_questions)
        ]
        cases[case_name] = case_questions
        task_ball_counts[case_name] = [len(_entry_ball_positions(entry)) for entry in case_questions]

    metadata = {
        "scene": scene_name,
        "room": room_name,
        "floor_name": floor_record.name,
        "seed": int(seed),
        "room_seed": int(room_seed),
        "agent": {"position": [float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2])]},
        "count_object_candidates": count_target_candidates,
        "count_object_by_task": {
            case_name: count_objects[0] if count_objects else {}
            for case_name, count_objects in task_count_objects.items()
        },
        "count_object_by_task_and_question": task_count_objects,
        "room_object_count": len(room_objects),
        "room_category_counts": _summarize_room_objects(room_objects),
        "room_objects": [obj.to_json() for obj in room_objects],
        "room_bbox": room_bbox_info,
        "wall_count": len(wall_records),
        "structural_wall_count": len(structural_wall_bboxes),
        "semantic_fault_confuser_candidates": semantic_fault_confuser_candidates,
        "question_count_per_task": int(default_question_count),
        "question_count_by_task": question_count_by_task,
        "task_ball_counts": task_ball_counts,
        "cases": cases,
        "notes": [
            "Primary source is the runtime OmniGibson scene loaded via og.Environment.",
            (
                f"Most tasks export up to {int(default_question_count)} independent counting questions per room; "
                f"hidden_in_box exports up to {int(HIDDEN_IN_BOX_QUESTION_COUNT)}."
            ),
            "Each exported question aggregates multiple count targets into one self-contained render folder.",
            "Each counting question re-samples its own count target category from keys.json using local model bbox sizes: any edge above 25cm is rejected, and objects with all three edges below 5cm are rejected.",
            "Room-level randomness uses a derived seed that mixes base seed, scene, and room.",
            "Count-target category sampling uses a derived per-question seed that mixes base seed, scene, room, task type, and question index.",
            "Semantic fault samples a confuser independently for each instance from up to three keys_clip_top3 neighbors when available.",
            "Each counting question samples its own target object count independently from 3 to 6, capped by feasible placements.",
        ],
    }

    question_scene_metadata = {
        "scene": scene_name,
        "room": room_name,
        "floor_name": floor_record.name,
        "seed": int(seed),
        "room_seed": int(room_seed),
        "camera_setup": {
            "mode": "single_agent_view_toward_room_center",
            "primary_image": "room_center_single_view",
            "context_image": "room_center_single_view",
            "views_per_candidate": 1,
            "topdown_map": "floor_trav_with_count_object_annotations",
            "agent_position_policy": "sample_free_point_near_short_edge_of_room_bbox",
            "pitch_deg": AGENT_CAMERA_PITCH_DEG,
        },
    }
    render_summary = None
    if not skip_render:
        render_extra_markers = {
            "hidden_by_others": [
                {"kind": "anchor", "position": list(anchor.center)}
                for anchor in hidden_anchor_candidates
            ]
        }
        try:
            render_start_time = time.perf_counter()
            render_summary = _render_all_candidates(
                scene=scene,
                scene_name=scene_name,
                room_name=room_name,
                base_seed=int(seed),
                floor_record=floor_record,
                room_objects=room_objects,
                blockers=blockers,
                room_object_by_name={obj.name: obj for obj in room_objects},
                room_bbox_info=room_bbox_info,
                cases=cases,
                render_root=render_root,
                agent_pos=agent_pos,
                rng=rng,
                hidden_box_cache=hidden_box_cache,
                extra_markers_by_case=render_extra_markers,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )
            metadata["render_root"] = render_root
            metadata["render_summary"] = render_summary
            _log_timing(
                "room_render",
                scene=scene_name,
                room=room_name,
                total_s=time.perf_counter() - render_start_time,
            )
        except Exception as exc:
            _log_exception("render_all_candidates", exc)
            metadata["render_root"] = render_root
            metadata["render_summary"] = {}
            metadata["render_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
    question_json_summary = _export_single_question_jsons(
        cases=cases,
        output_root=question_json_root,
        scene_metadata=question_scene_metadata,
    )
    metadata["question_json_root"] = question_json_root
    metadata["question_json_summary"] = question_json_summary
    _write_json(output_json, metadata)

    summary = {case_name: len(entries) for case_name, entries in cases.items()}
    _log_timing(
        "room_done",
        scene=scene_name,
        room=room_name,
        room_object_count=len(room_objects),
        selected_task_count=len(selected_task_types),
        rendered_case_count=0 if render_summary is None else len(render_summary),
        question_count=sum(summary.values()),
        total_s=time.perf_counter() - room_start_time,
    )
    return {
        "output_json": output_json,
        "room": room_name,
        "selected_task_types": list(selected_task_types),
        "summary": summary,
        "render_root": None if skip_render else render_root,
        "render_summary": render_summary,
        "question_json_root": question_json_root,
        "question_json_summary": question_json_summary["counts"],
    }


def main():
    main_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate counting candidate metadata from runtime OmniGibson scene.")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name, e.g. grocery_store_cafe")
    parser.add_argument("--room", type=str, default="living_room_0", help="Room instance name, e.g. grocery_store_0")
    parser.add_argument("--rooms", nargs="+", help="Optional batch of room instance names to process in one scene load.")
    parser.add_argument("--floor", type=str, default=None, help="Optional floor object name")
    parser.add_argument("--floors", nargs="*", help="Optional floors matching --rooms one-to-one.")
    parser.add_argument("--agent_position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--keys_json", type=str, default="keys.json")
    parser.add_argument("--keys_clip_top3_json", type=str, default=DEFAULT_KEYS_CLIP_TOP3)
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--max_per_case", type=int, default=8)
    parser.add_argument("--questions_per_task", type=int, default=3)
    parser.add_argument("--output_root", type=str, default="renders_counting_new")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--task_types",
        nargs="+",
        choices=COUNTING_TASK_TYPES,
        default=list(COUNTING_TASK_TYPES),
        help="Optional subset of counting task types to generate.",
    )
    parser.add_argument(
        "--hidden_in_box_only",
        action="store_true",
        help="Shortcut for `--task_types hidden_in_box` while keeping the same execution path.",
    )
    parser.add_argument("--skip_render", action="store_true", help="Only export metadata JSON, do not render candidate images.")
    parser.add_argument(
        "--load_full_scene",
        action="store_true",
        help="Load the full scene instead of restricting OmniGibson to the target room instance.",
    )
    parser.add_argument(
        "--disable_runtime_physics",
        action="store_true",
        help="Spawn counting objects in direct-placement mode without gravity or collision response.",
    )
    args = parser.parse_args()
    if args.hidden_in_box_only:
        args.task_types = ["hidden_in_box"]
    requested_rooms = _resolve_requested_rooms(args)
    selected_task_types = _ordered_task_types(args.task_types)

    global DIRECT_PLACEMENT_MODE
    DIRECT_PLACEMENT_MODE = bool(args.disable_runtime_physics)

    config = _build_config(
        scene_name=args.scene,
        robot=args.robot,
        load_full_scene=args.load_full_scene,
        room_names=[room_name for room_name, _ in requested_rooms],
    )
    try:
        env_start_time = time.perf_counter()
        env = og.Environment(configs=config)
        _set_viewer_camera_fov()
        scene = env.scene
        wall_records = _collect_wall_records(scene)
        structural_wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]
        count_target_candidates = _build_count_target_candidates(args.keys_json)
        clip_neighbors = _load_clip_top3_neighbors(args.keys_clip_top3_json)
        _log_timing(
            "scene_setup",
            scene=args.scene,
            requested_room_count=len(requested_rooms),
            load_full_scene=int(bool(args.load_full_scene)),
            total_s=time.perf_counter() - env_start_time,
        )

        room_results = []
        room_failures = []
        for room_name, floor_name in requested_rooms:
            try:
                room_results.append(
                    _process_room_in_loaded_scene(
                        env=env,
                        scene=scene,
                        scene_name=args.scene,
                        room_name=room_name,
                        floor_name=floor_name,
                        output_root=args.output_root,
                        seed=int(args.seed),
                        questions_per_task=int(args.questions_per_task),
                        max_per_case=int(args.max_per_case),
                        skip_render=bool(args.skip_render),
                        selected_task_types=selected_task_types,
                        count_target_candidates=count_target_candidates,
                        clip_neighbors=clip_neighbors,
                        wall_records=wall_records,
                        structural_wall_bboxes=structural_wall_bboxes,
                        agent_position=args.agent_position,
                    )
                )
            except Exception as exc:
                _log_exception(f"room batch failed: {args.scene}/{room_name}", exc)
                room_failures.append(
                    {
                        "room": room_name,
                        "floor": floor_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                )
                _log_timing(
                    "room_failed",
                    scene=args.scene,
                    room=room_name,
                    floor=floor_name,
                    error_type=exc.__class__.__name__,
                    total_s=time.perf_counter() - main_start_time,
                )

        _log_timing(
            "scene_done",
            scene=args.scene,
            processed_room_count=len(room_results),
            failed_room_count=len(room_failures),
            total_s=time.perf_counter() - main_start_time,
        )
        print(
            json.dumps(
                {
                    "scene": args.scene,
                    "requested_rooms": [room_name for room_name, _ in requested_rooms],
                    "processed_room_count": len(room_results),
                    "failed_room_count": len(room_failures),
                    "rooms": room_results,
                    "failures": room_failures,
                },
                indent=2,
            )
        )
        if room_failures:
            raise SystemExit(1)
    except Exception as exc:
        _log_exception("main", exc)
        raise


if __name__ == "__main__":
    main()
