"""
Generate cognitive-map region QA candidates from a runtime OmniGibson scene.

Compared with batch_cognitivemap_connect.py, this script focuses on
object-to-region relations:
1. Whether an object belongs to a region
2. Whether two objects are in the same region
3. Which region an object is closer to

Because room bboxes can be noisy, we first extract room regions from the
segmentation map, expand their 2D world-space bbox, and use the expanded bbox
as the primary region proxy for object-region assignment.
"""

from __future__ import annotations

import argparse
import cv2
import json
import math
import os
import random
import shutil
import struct
import sys
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
OG_ROOT = str(SCRIPT_DIR / "OmniGibson")
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
from omnigibson.macros import gm


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

DEFAULT_OUTPUT_NAME = "cognitivemap_region_candidates.json"
DEFAULT_TRAV_MAP_BASENAME = "floor_trav_no_door"
DEFAULT_POINT_CANDIDATES = 7
DEFAULT_BELONG_CASE_LIMIT = 6
DEFAULT_SAME_REGION_LIMIT = 6
DEFAULT_CLOSER_REGION_LIMIT = 6
DEFAULT_EXPANSION_RATIO = 0.35
DEFAULT_EXPANSION_MIN = 0.75
DEFAULT_CLOSER_MARGIN = 0.75
DEFAULT_ROOM_VIEW_COUNT = 4
DEFAULT_OBJECT_VIEW_COUNT = 1
VIEWER_CAMERA_FOV_DEG = 100.0
SEMANTIC_VISIBLE_MIN_PIXELS = 100
SEMANTIC_VIEW_YAW_COUNT = 8
ROOM_VIEW_CAMERA_HEIGHT_M = 1.35
ROOM_VIEW_TARGET_HEIGHT_M = 1.0
OBJECT_CLOSEUP_DISTANCE_M = 0.8
OBJECT_CLOSEUP_DOWNWARD_ANGLE_DEG = 45.0

TASK_FAMILY_BY_TYPE = {
    "object_in_region": "Regional Boundry",
    "objects_same_region": "Regional Boundry",
    "object_closer_region": "Regional Boundry",
}

NON_QUERY_OBJECT_CATEGORIES = {
    "agent",
    "background",
    "baseboard",
    "carpet",
    "ceilings",
    "door",
    "fire_alarm",
    "fire_sprinkler",
    "fixed_window",
    "floors",
    "mirror",
    "picture",
    "roof",
    "sliding_door",
    "walls",
}


def _set_viewer_camera_fov(fov_deg: float = VIEWER_CAMERA_FOV_DEG) -> None:
    cam = og.sim.viewer_camera
    aperture_mm = float(cam.horizontal_aperture)
    target_fov_deg = float(fov_deg)
    focal_length_mm = aperture_mm / (2.0 * math.tan(math.radians(target_fov_deg) * 0.5))
    cam.focal_length = focal_length_mm
    _debug(
        f"viewer camera horizontal FOV set to {target_fov_deg:.1f} deg "
        f"(aperture={aperture_mm:.3f} mm, focal_length={focal_length_mm:.3f} mm)"
    )


def _make_case_id(case_type: str, *parts: str) -> str:
    cleaned = [case_type] + [str(part).replace(" ", "_") for part in parts]
    return "__".join(cleaned)


@dataclass
class RegionRecord:
    room_id: int
    room_instance: str
    room_type: str
    bbox_map_rc: tuple[int, int, int, int] | None
    bbox_world_xy: tuple[float, float, float, float] | None
    expanded_bbox_world_xy: tuple[float, float, float, float] | None
    center_xy: tuple[float, float] | None
    candidate_points_xy: list[list[float]]
    pixel_count: int

    def to_json(self) -> dict:
        return {
            "room_id": self.room_id,
            "room_instance": self.room_instance,
            "room_type": self.room_type,
            "bbox_map_rc": list(self.bbox_map_rc) if self.bbox_map_rc is not None else None,
            "bbox_world_xy": list(self.bbox_world_xy) if self.bbox_world_xy is not None else None,
            "expanded_bbox_world_xy": list(self.expanded_bbox_world_xy) if self.expanded_bbox_world_xy is not None else None,
            "center_xy": list(self.center_xy) if self.center_xy is not None else None,
            "candidate_points_xy": self.candidate_points_xy,
            "pixel_count": self.pixel_count,
        }


@dataclass
class ObjectRecord:
    name: str
    category: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    aabb_center: tuple[float, float, float]
    aabb_extent: tuple[float, float, float]
    center_xy: tuple[float, float]
    in_rooms: tuple[str, ...]
    assigned_region: str | None
    assigned_room_type: str | None
    assignment_source: str | None
    containment_regions: tuple[str, ...]
    obj: object

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "bbox": [list(self.bbox_min), list(self.bbox_max)],
            "aabb_center": list(self.aabb_center),
            "aabb_extent": list(self.aabb_extent),
            "center_xy": list(self.center_xy),
            "in_rooms": list(self.in_rooms),
            "assigned_region": self.assigned_region,
            "assigned_room_type": self.assigned_room_type,
            "assignment_source": self.assignment_source,
            "containment_regions": list(self.containment_regions),
        }


@dataclass
class WallRecord:
    name: str
    category: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_world_xy: tuple[float, float, float, float]
    is_structural_wall: bool

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "bbox": [list(self.bbox_min), list(self.bbox_max)],
            "bbox_world_xy": list(self.bbox_world_xy),
            "is_structural_wall": self.is_structural_wall,
        }


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def _task_family_for_type(task_type: str) -> str:
    return TASK_FAMILY_BY_TYPE.get(task_type, task_type)


def _copy_asset(src_path: str, output_dir: str, copied_paths: dict[tuple[str, str], str]) -> str:
    cache_key = (src_path, output_dir)
    cached = copied_paths.get(cache_key)
    if cached is not None:
        return cached

    os.makedirs(output_dir, exist_ok=True)
    dst_path = os.path.join(output_dir, os.path.basename(src_path))
    if not os.path.abspath(src_path) == os.path.abspath(dst_path):
        shutil.copy2(src_path, dst_path)
    copied_paths[cache_key] = dst_path
    return dst_path


def _asset_target_dir(candidate_dir: str, map_dir: str, key_path: tuple[str, ...], asset_key: str) -> str:
    if asset_key == "topdown_map":
        return map_dir
    path_keys = set(key_path)
    if "room_views" in path_keys and not path_keys.intersection({"gt_open_view", "gt_region_views", "object_views"}):
        return os.path.join(candidate_dir, "room_views")
    if path_keys.intersection({"object_views", "gt_region_views", "gt_open_view", "gt_views"}):
        return os.path.join(candidate_dir, "gt_views")
    return candidate_dir


def _relocate_entry_assets(
    payload,
    candidate_dir: str,
    map_dir: str,
    copied_paths: dict[tuple[str, str], str],
    key_path: tuple[str, ...] = (),
):
    if isinstance(payload, dict):
        relocated = {}
        for key, value in payload.items():
            if key in {"image", "topdown_map", "image_path"} and isinstance(value, str) and os.path.isfile(value):
                target_dir = _asset_target_dir(candidate_dir, map_dir, key_path, key)
                relocated[key] = _copy_asset(value, target_dir, copied_paths)
            else:
                relocated[key] = _relocate_entry_assets(
                    value,
                    candidate_dir,
                    map_dir,
                    copied_paths,
                    key_path + (str(key),),
                )
        return relocated
    if isinstance(payload, list):
        return [
            _relocate_entry_assets(item, candidate_dir, map_dir, copied_paths, key_path)
            for item in payload
        ]
    return payload


def _write_single_question_json(
    output_root: str,
    candidate_root: str,
    map_root: str,
    scene_metadata: dict,
    task_type: str,
    q_idx: int,
    entry: dict,
) -> str:
    task_family = _task_family_for_type(task_type)
    task_dir = os.path.join(output_root, task_family)
    os.makedirs(task_dir, exist_ok=True)
    question_stem = f"q_{q_idx:03d}"
    copied_paths: dict[tuple[str, str], str] = {}
    entry = _relocate_entry_assets(
        payload=entry,
        candidate_dir=os.path.join(candidate_root, task_family, question_stem),
        map_dir=os.path.join(map_root, task_family, question_stem),
        copied_paths=copied_paths,
    )
    payload = {
        "scene": scene_metadata.get("scene"),
        "room": scene_metadata.get("room"),
        "floor_name": scene_metadata.get("floor_name"),
        "seed": scene_metadata.get("seed"),
        "trav_map_basename": scene_metadata.get("trav_map_basename"),
        "task_family": task_family,
        "task_type": task_type,
        "question_index": q_idx,
        "question_id": f"{task_family}/{question_stem}",
        "question_data": entry,
    }
    payload["image_paths"] = _collect_image_paths(payload["question_data"])
    out_path = os.path.join(task_dir, f"{question_stem}.json")
    _write_json(out_path, payload)
    return out_path


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_cognitivemap_region] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _debug(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[batch_cognitivemap_region {now}] {message}", flush=True)


def _get_scene_objects(scene) -> list[object]:
    objects = getattr(scene, "objects", None)
    if objects is None:
        return []
    return list(objects)


def _is_door_named_object(obj) -> bool:
    name = str(getattr(obj, "name", "")).lower()
    return "door" in name


def _remove_named_doors(scene) -> dict:
    targets = []
    for obj in _get_scene_objects(scene):
        if _is_door_named_object(obj):
            targets.append(obj)

    removed = []
    failed = []
    for obj in targets:
        obj_name = str(getattr(obj, "name", "unknown"))
        try:
            scene.remove_object(obj)
            removed.append(obj_name)
        except Exception:
            failed.append(obj_name)

    if targets:
        _step_sim(3)
    summary = {
        "name_match": "contains 'door'",
        "target_total": len(targets),
        "removed": removed,
        "failed": failed,
    }
    _debug(
        f"door removal summary: match=contains 'door' target_total={len(targets)} "
        f"removed={len(removed)} failed={len(failed)}"
    )
    return summary


def _tensor_to_list(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _tensor_to_tuple3(value) -> tuple[float, float, float]:
    vals = _tensor_to_list(value)
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def _step_sim(steps: int = 10) -> None:
    for _ in range(int(steps)):
        og.sim.step()


def _vec_sub(a, b):
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def _vec_norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _vec_normalize(v):
    norm = _vec_norm(v)
    if norm < 1e-8:
        return [0.0, 0.0, 1.0]
    return [float(x) / norm for x in v]


def _vec_cross(a, b):
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


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


def _ensure_viewer_modalities() -> None:
    for modality in ("seg_semantic", "seg_instance", "seg_instance_id"):
        try:
            og.sim._viewer_camera.add_modality(modality)
        except Exception:
            pass
    _step_sim(3)


def _get_viewer_frame():
    _ensure_viewer_modalities()
    for _ in range(10):
        og.sim.render()
    obs, info = og.sim._viewer_camera.get_obs()
    return obs, info


def _replace_trav_map_with_variant(scene, basename: str = DEFAULT_TRAV_MAP_BASENAME) -> bool:
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        _debug("trav-map replace skipped: scene._trav_map unavailable")
        return False

    scene_dir = getattr(scene, "scene_dir", None)
    if not scene_dir:
        _debug("trav-map replace skipped: scene.scene_dir unavailable")
        return False

    maps_path = os.path.join(scene_dir, "layout")
    if not os.path.isdir(maps_path):
        _debug(f"trav-map replace skipped: maps path missing: {maps_path}")
        return False

    if trav_map.map_size is None and getattr(trav_map, "trav_map_original_size", None) is not None:
        trav_map.map_size = int(trav_map.trav_map_original_size * trav_map.map_default_resolution / trav_map.map_resolution)
    map_size = int(trav_map.map_size) if trav_map.map_size is not None else None
    if map_size is None:
        _debug("trav-map replace skipped: map_size unavailable")
        return False

    loaded = 0
    for floor in range(len(trav_map.floor_map)):
        candidates = [f"{basename}_{floor}.png"]
        if floor == 0:
            candidates.append(f"{basename}.png")
        if "{}" in basename:
            candidates.insert(0, basename.format(floor))

        src_img = None
        src_path = None
        for fname in candidates:
            fpath = os.path.join(maps_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                src_img = img
                src_path = fpath
                break

        if src_img is None:
            continue

        resized = cv2.resize(src_img, (map_size, map_size))
        trav_tensor = th.tensor(resized)
        trav_tensor[trav_tensor < 255] = 0
        trav_map.floor_map[floor] = trav_tensor
        loaded += 1
        _debug(f"trav-map floor={floor}: loaded no-door variant from {src_path}")

    return loaded > 0


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


def _room_bbox_center_xy(bbox_xyxy):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)


def _distance_xy(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _point_inside_bbox_xy(point_xy, bbox_xyxy, margin: float = 0.0) -> bool:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return (
        xmin - float(margin) <= float(point_xy[0]) <= xmax + float(margin)
        and ymin - float(margin) <= float(point_xy[1]) <= ymax + float(margin)
    )


def _distance_point_to_bbox_xy(point_xy, bbox_xyxy) -> float:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    px = float(point_xy[0])
    py = float(point_xy[1])
    dx = 0.0 if xmin <= px <= xmax else min(abs(px - xmin), abs(px - xmax))
    dy = 0.0 if ymin <= py <= ymax else min(abs(py - ymin), abs(py - ymax))
    return math.hypot(dx, dy)


def _build_config(args):
    config_filename = os.path.join(OG_ROOT, "omnigibson", "configs", "r1_primitives.yaml")
    config = yaml.safe_load(open(config_filename))
    config["scene"]["scene_model"] = args.scene
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "door", "sliding_door"]
    if args.room is not None:
        config["scene"]["load_room_instances"] = [args.room]
    else:
        config["scene"].pop("load_room_instances", None)
    config["robots"] = []
    config["objects"] = []
    return config


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


def _query_room_instance_by_point(seg, xy) -> str | None:
    if seg is None:
        return None
    try:
        result = seg.get_room_instance_by_point(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32))
    except Exception:
        return None
    if result is None:
        return None
    return str(result)


def _build_region_records(
    scene,
    seed: int,
    point_candidates: int,
    expansion_ratio: float,
    expansion_min: float,
    wall_bboxes_xyxy: list[tuple[float, float, float, float]],
) -> list[RegionRecord]:
    seg = _segmap_get(scene)
    if seg is None or not hasattr(seg, "room_ins_id_to_ins_name"):
        raise RuntimeError("scene.seg_map.room_ins_id_to_ins_name not found")

    room_records = []
    rng = random.Random(seed)
    room_ids = sorted(int(room_id) for room_id in seg.room_ins_id_to_ins_name.keys())
    room_map = seg.room_ins_map
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()
    room_map_np = np.array(room_map, dtype=np.int64)

    _debug(f"building region records for {len(room_ids)} room instances")
    for room_id in room_ids:
        room_instance = str(seg.room_ins_id_to_ins_name[room_id])
        room_type = room_instance.rsplit("_", 1)[0] if "_" in room_instance else room_instance
        bbox_info = _segmap_room_bbox_from_maps(scene, room_id)
        bbox_world_xy = bbox_info["bbox_world_xy"]
        expanded_bbox_world_xy = None
        center_xy = None
        if bbox_world_xy is not None:
            if wall_bboxes_xyxy:
                expanded_bbox_world_xy = _expand_bbox_until_wall_touch(
                    bbox_world_xy,
                    wall_bboxes_xyxy=wall_bboxes_xyxy,
                    expansion_ratio=expansion_ratio,
                    expansion_min=expansion_min,
                )
            else:
                expanded_bbox_world_xy = _expand_bbox_xyxy(
                    bbox_world_xy,
                    expansion_ratio=expansion_ratio,
                    expansion_min=expansion_min,
                )
            center_xy = tuple(float(x) for x in _room_bbox_center_xy(bbox_world_xy).tolist())

        pixels = np.argwhere(room_map_np == int(room_id))
        if pixels.shape[0] == 0:
            continue

        candidate_indices = [0, pixels.shape[0] // 2, pixels.shape[0] - 1]
        if point_candidates > 3:
            extra = list(range(pixels.shape[0]))
            rng.shuffle(extra)
            candidate_indices.extend(extra[: max(0, point_candidates - len(candidate_indices))])

        candidate_points = []
        seen_rc = set()
        for idx in candidate_indices:
            idx = int(np.clip(idx, 0, pixels.shape[0] - 1))
            rc = (int(pixels[idx, 0]), int(pixels[idx, 1]))
            if rc in seen_rc:
                continue
            seen_rc.add(rc)
            xy = seg.map_to_world(th.tensor([float(rc[0]), float(rc[1])], dtype=th.float32))
            candidate_points.append([float(xy[0].item()), float(xy[1].item())])

        if center_xy is not None:
            candidate_points.insert(0, [float(center_xy[0]), float(center_xy[1])])

        deduped_points = []
        seen_points = set()
        for xy in candidate_points:
            key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
            if key in seen_points:
                continue
            seen_points.add(key)
            deduped_points.append([float(xy[0]), float(xy[1])])

        room_records.append(
            RegionRecord(
                room_id=room_id,
                room_instance=room_instance,
                room_type=room_type,
                bbox_map_rc=bbox_info["bbox_map_rc"],
                bbox_world_xy=bbox_world_xy,
                expanded_bbox_world_xy=expanded_bbox_world_xy,
                center_xy=center_xy,
                candidate_points_xy=deduped_points,
                pixel_count=int(bbox_info["pixel_count"]),
            )
        )

    room_records.sort(key=lambda record: (record.room_type, record.room_instance))
    _debug(f"finished region extraction: {len(room_records)} valid regions")
    return room_records


def _trav_map_floor_image(scene, floor_idx: int = 0):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        return None, None
    if floor_idx < 0 or floor_idx >= len(trav_map.floor_map):
        return trav_map, None
    map_img = trav_map.floor_map[floor_idx].cpu().numpy()
    return trav_map, map_img


def _clip_map_rc(map_img: np.ndarray, rc) -> tuple[int, int]:
    row = int(np.clip(round(float(rc[0])), 0, map_img.shape[0] - 1))
    col = int(np.clip(round(float(rc[1])), 0, map_img.shape[1] - 1))
    return row, col


def _region_center_xy(region: RegionRecord) -> tuple[float, float] | None:
    if region.center_xy is not None:
        return (float(region.center_xy[0]), float(region.center_xy[1]))
    if region.candidate_points_xy:
        point = region.candidate_points_xy[0]
        return (float(point[0]), float(point[1]))
    return None


def _get_scene_objects(scene):
    raw_objects = getattr(scene, "objects", [])
    if isinstance(raw_objects, dict):
        return list(raw_objects.values())
    return list(raw_objects)


def _assign_object_region(center_xy, in_rooms, seg, regions: list[RegionRecord]) -> tuple[str | None, str | None, tuple[str, ...]]:
    containing = []
    for region in regions:
        if region.expanded_bbox_world_xy is None:
            continue
        if _point_inside_bbox_xy(center_xy, region.expanded_bbox_world_xy):
            containing.append(region)

    if containing:
        containing.sort(
            key=lambda region: (
                _distance_xy(center_xy, region.center_xy if region.center_xy is not None else center_xy),
                region.room_instance,
            )
        )
        return containing[0].room_instance, "expanded_bbox", tuple(region.room_instance for region in containing)

    queried_room = _query_room_instance_by_point(seg, center_xy)
    if queried_room is not None:
        return queried_room, "seg_query", tuple()

    for room_name in in_rooms:
        if room_name:
            return room_name, "in_rooms", tuple()

    return None, None, tuple()


def _collect_wall_records(scene) -> list[WallRecord]:
    wall_records = []

    for obj in _get_scene_objects(scene):
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
    _debug(
        f"collected {len(wall_records)} wall-related objects, "
        f"structural_walls={sum(1 for record in wall_records if record.is_structural_wall)}"
    )
    return wall_records


def _collect_scene_objects(scene, regions_by_name: dict[str, RegionRecord]) -> list[ObjectRecord]:
    seg = _segmap_get(scene)
    regions = [regions_by_name[name] for name in sorted(regions_by_name)]
    object_records = []

    for obj in _get_scene_objects(scene):
        category = str(getattr(obj, "category", "object"))
        if category.lower() in NON_QUERY_OBJECT_CATEGORIES:
            continue
        if "floor" in category.lower() or "wall" in category.lower() or "ceiling" in category.lower():
            continue

        try:
            bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
            aabb_center = _tensor_to_tuple3(obj.aabb_center)
            aabb_extent = _tensor_to_tuple3(obj.aabb_extent)
        except Exception as exc:
            _log_exception(f"Failed to read AABB for object {obj.name}", exc)
            continue

        center_xy = (float(aabb_center[0]), float(aabb_center[1]))
        in_rooms = tuple(str(room) for room in (getattr(obj, "in_rooms", None) or []))
        assigned_region, assignment_source, containment_regions = _assign_object_region(
            center_xy=center_xy,
            in_rooms=in_rooms,
            seg=seg,
            regions=regions,
        )
        if assigned_region not in regions_by_name:
            continue
        if max(float(aabb_extent[0]), float(aabb_extent[1])) > 6.0:
            continue

        region = regions_by_name[assigned_region]
        object_records.append(
            ObjectRecord(
                name=str(obj.name),
                category=category,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                aabb_center=aabb_center,
                aabb_extent=aabb_extent,
                center_xy=center_xy,
                in_rooms=in_rooms,
                assigned_region=assigned_region,
                assigned_room_type=region.room_type,
                assignment_source=assignment_source,
                containment_regions=containment_regions,
                obj=obj,
            )
        )

    object_records.sort(key=lambda record: (record.assigned_region or "", record.category, record.name))
    _debug(f"collected {len(object_records)} scene objects with assigned regions")
    return object_records


def _world_to_plot_xy(trav_map, xy) -> tuple[float, float]:
    rc = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).cpu().numpy()
    return float(rc[1]), float(rc[0])


def _world_to_map_rc(trav_map, xy, map_img: np.ndarray) -> np.ndarray:
    rc = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    row, col = _clip_map_rc(map_img, rc)
    return np.array([row, col], dtype=np.int32)


def _map_rc_to_world_xy(trav_map, rc) -> list[float]:
    xy = trav_map.map_to_world(th.tensor([float(rc[0]), float(rc[1])], dtype=th.float32))
    return [float(xy[0].item()), float(xy[1].item())]


def _room_masks(scene, floor_idx: int, region: RegionRecord):
    seg = _segmap_get(scene)
    trav_map, map_img = _trav_map_floor_image(scene, floor_idx=floor_idx)
    if seg is None or not hasattr(seg, "room_ins_map") or trav_map is None or map_img is None:
        return seg, trav_map, map_img, None, None
    room_map = seg.room_ins_map
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()
    room_mask = np.array(room_map, dtype=np.int32) == int(region.room_id)
    if room_mask.shape != map_img.shape:
        room_mask = cv2.resize(
            room_mask.astype(np.uint8),
            (map_img.shape[1], map_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    free_mask = np.array(map_img > 0, dtype=bool)
    room_free_mask = room_mask & free_mask
    return seg, trav_map, map_img, room_mask, room_free_mask


def _nearest_free_rc(preferred_rc: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray | None:
    candidate_rcs = np.argwhere(candidate_mask)
    if len(candidate_rcs) == 0:
        return None
    deltas = candidate_rcs.astype(np.float32) - preferred_rc.astype(np.float32)
    best_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    return candidate_rcs[best_idx].astype(np.int32)


def _select_room_render_xy(scene, floor_idx: int, region: RegionRecord) -> tuple[list[float], dict]:
    seg = _segmap_get(scene)
    trav_map, map_img = _trav_map_floor_image(scene, floor_idx=floor_idx)
    fallback_xy = _region_center_xy(region)
    if fallback_xy is None and region.candidate_points_xy:
        fallback_xy = (float(region.candidate_points_xy[0][0]), float(region.candidate_points_xy[0][1]))
    if fallback_xy is None:
        fallback_xy = (0.0, 0.0)

    debug = {
        "room_instance": region.room_instance,
        "selected_by": "fallback",
        "clearance_px": None,
        "center_distance_px": None,
    }
    if seg is None or not hasattr(seg, "room_ins_map") or trav_map is None or map_img is None:
        return [float(fallback_xy[0]), float(fallback_xy[1])], debug

    room_map = seg.room_ins_map
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()
    room_mask = np.array(room_map, dtype=np.int32) == int(region.room_id)
    if room_mask.size == 0 or not room_mask.any():
        return [float(fallback_xy[0]), float(fallback_xy[1])], debug

    if room_mask.shape != map_img.shape:
        room_mask = cv2.resize(
            room_mask.astype(np.uint8),
            (map_img.shape[1], map_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    free_mask = np.array(map_img > 0, dtype=np.uint8)
    room_free_mask = room_mask & (free_mask > 0)
    if not room_free_mask.any():
        return [float(fallback_xy[0]), float(fallback_xy[1])], debug

    clearance_map = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
    candidate_rcs = np.argwhere(room_free_mask)
    if len(candidate_rcs) == 0:
        return [float(fallback_xy[0]), float(fallback_xy[1])], debug

    center_rc = trav_map.world_to_map(th.tensor([float(fallback_xy[0]), float(fallback_xy[1])], dtype=th.float32))
    center_rc = center_rc.detach().cpu().numpy()
    center_rc = np.array(_clip_map_rc(map_img, center_rc), dtype=np.float32)

    best_score = None
    best_rc = None
    best_clearance = None
    best_center_dist = None
    for rc in candidate_rcs:
        rc_arr = np.array([float(rc[0]), float(rc[1])], dtype=np.float32)
        clearance = float(clearance_map[int(rc[0]), int(rc[1])])
        center_dist = float(np.linalg.norm(rc_arr - center_rc))
        score = clearance - 0.15 * center_dist
        if best_score is None or score > best_score:
            best_score = score
            best_rc = rc_arr
            best_clearance = clearance
            best_center_dist = center_dist

    if best_rc is None:
        return [float(fallback_xy[0]), float(fallback_xy[1])], debug

    xy = trav_map.map_to_world(th.tensor([float(best_rc[0]), float(best_rc[1])], dtype=th.float32))
    debug.update(
        {
            "selected_by": "clearance_and_center",
            "clearance_px": float(best_clearance) if best_clearance is not None else None,
            "center_distance_px": float(best_center_dist) if best_center_dist is not None else None,
            "selected_map_rc": [int(best_rc[0]), int(best_rc[1])],
        }
    )
    return [float(xy[0].item()), float(xy[1].item())], debug


def _set_viewer_pose(position_xyz, yaw_rad: float, eye_height: float = 1.35) -> dict:
    eye = [float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2] + eye_height)]
    target = [
        float(eye[0] + math.cos(float(yaw_rad))),
        float(eye[1] + math.sin(float(yaw_rad))),
        float(eye[2]),
    ]
    quat = look_at_quaternion(eye, target)
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(eye, dtype=th.float32),
        th.tensor(quat, dtype=th.float32),
    )
    _step_sim(5)
    return {"position": eye, "quaternion_xyzw": quat, "yaw_rad": float(yaw_rad)}


def _set_camera_pose(eye, target) -> dict:
    eye = [float(v) for v in eye]
    target = [float(v) for v in target]
    quat = look_at_quaternion(eye, target)
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(eye, dtype=th.float32),
        th.tensor(quat, dtype=th.float32),
    )
    _step_sim(5)
    return {"position": eye, "quaternion_xyzw": quat, "target": target}


def _capture_camera_view(image_path: str, eye, target) -> dict:
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    camera_pose = _set_camera_pose(eye, target)
    _capture(image_path)
    return {
        "image_path": image_path,
        "camera_pose": camera_pose,
    }


def _capture_room_reference_views(region: RegionRecord, floor_z: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    bbox = region.expanded_bbox_world_xy or region.bbox_world_xy
    center_xy = _region_center_xy(region)
    if bbox is None or center_xy is None:
        render_xy = [0.0, 0.0]
        eye = [0.0, 0.0, float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
        target = [0.0, 1.0, float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M]
        view = _capture_camera_view(os.path.join(output_dir, "view_00.png"), eye=eye, target=target)
        return {
            "success": True,
            "room_instance": region.room_instance,
            "render_xy": render_xy,
            "views": {"view_00": view},
        }

    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox)
    width = max(1e-3, xmax - xmin)
    height = max(1e-3, ymax - ymin)
    inset = min(max(0.35, min(width, height) * 0.08), max(min(width, height) * 0.35, 0.35))
    corners = [
        ("view_00", [xmin + inset, ymin + inset]),
        ("view_01", [xmin + inset, ymax - inset]),
        ("view_02", [xmax - inset, ymin + inset]),
        ("view_03", [xmax - inset, ymax - inset]),
    ]
    views = {}
    render_points = []
    for view_key, xy in corners:
        eye = [float(xy[0]), float(xy[1]), float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
        target = [float(center_xy[0]), float(center_xy[1]), float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M]
        view = _capture_camera_view(os.path.join(output_dir, f"{view_key}.png"), eye=eye, target=target)
        view["view_role"] = view_key
        view["corner_xy"] = [float(xy[0]), float(xy[1])]
        views[view_key] = view
        render_points.append([float(xy[0]), float(xy[1])])
    return {
        "success": True,
        "room_instance": region.room_instance,
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "corner_render_xy": render_points,
        "views": views,
    }


def _seg_name_matches(raw_name: object, candidate: str) -> bool:
    seg_name = str(raw_name)
    seg_name_lower = seg_name.lower()
    cand = str(candidate).strip().lower()
    if not cand:
        return False
    basename = seg_name.rsplit("/", 1)[-1].lower()
    return seg_name_lower == cand or basename == cand or seg_name_lower.endswith("/" + cand) or f"/{cand}/" in seg_name_lower


def _semantic_pixel_counts(obs, info) -> dict[str, int]:
    seg = obs.get("seg_semantic")
    seg_info = info.get("seg_semantic") if isinstance(info, dict) else None
    if seg is None or not seg_info:
        return {}
    seg_np = seg.detach().cpu().numpy()
    counts: dict[str, int] = {}
    for seg_id, raw_name in seg_info.items():
        category = str(raw_name).strip()
        if not category:
            continue
        pixels = int((seg_np == int(seg_id)).sum())
        if pixels <= 0:
            continue
        counts[category] = counts.get(category, 0) + pixels
    return counts


def _instance_pixel_counts(obs, info) -> dict[str, int]:
    seg = obs.get("seg_instance")
    seg_info = info.get("seg_instance") if isinstance(info, dict) else None
    if seg is None or not seg_info:
        return {}
    seg_np = seg.detach().cpu().numpy()
    counts: dict[str, int] = {}
    for seg_id, raw_name in seg_info.items():
        name = str(raw_name).strip()
        if not name:
            continue
        pixels = int((seg_np == int(seg_id)).sum())
        if pixels <= 0:
            continue
        counts[name] = counts.get(name, 0) + pixels
    return counts


def _visible_unique_objects_from_frame(obs, info, objects_by_name: dict[str, ObjectRecord]) -> list[dict]:
    semantic_counts = _semantic_pixel_counts(obs, info)
    instance_counts = _instance_pixel_counts(obs, info)
    objects_by_category: dict[str, list[tuple[int, ObjectRecord]]] = {}
    for seg_name, pixels in instance_counts.items():
        matched_obj = None
        for obj_name, obj_record in objects_by_name.items():
            if _seg_name_matches(seg_name, obj_name):
                matched_obj = obj_record
                break
        if matched_obj is None:
            continue
        objects_by_category.setdefault(matched_obj.category, []).append((int(pixels), matched_obj))

    visible = []
    for category, entries in objects_by_category.items():
        semantic_pixels = 0
        for key, pixels in semantic_counts.items():
            if str(key).strip().lower() == str(category).strip().lower():
                semantic_pixels += int(pixels)
        if semantic_pixels < SEMANTIC_VISIBLE_MIN_PIXELS:
            continue
        # Keep only categories that correspond to a single visible instance so
        # category-based wording stays unambiguous.
        unique_instance_names = {entry[1].name for entry in entries}
        if len(unique_instance_names) != 1:
            continue
        entries.sort(key=lambda item: (-item[0], item[1].name))
        instance_pixels, obj_record = entries[0]
        if int(instance_pixels) < SEMANTIC_VISIBLE_MIN_PIXELS:
            continue
        visible.append(
            {
                "object_name": obj_record.name,
                "object_category": obj_record.category,
                "semantic_pixels": int(semantic_pixels),
                "instance_pixels": int(instance_pixels),
                "assigned_region": obj_record.assigned_region,
            }
        )
    visible.sort(key=lambda item: (-item["semantic_pixels"], -item["instance_pixels"], item["object_category"], item["object_name"]))
    return visible


def _build_region_semantic_view_catalog(
    scene,
    floor_idx: int,
    floor_z: float,
    regions: list[RegionRecord],
    objects_by_name: dict[str, ObjectRecord],
) -> tuple[list[dict], dict[str, dict]]:
    catalog = []
    best_by_region: dict[str, dict] = {}
    _ensure_viewer_modalities()
    for region in regions:
        render_xy, selection_debug = _select_room_render_xy(scene, floor_idx, region)
        base_xyz = [float(render_xy[0]), float(render_xy[1]), float(floor_z)]
        for idx in range(SEMANTIC_VIEW_YAW_COUNT):
            yaw = 2.0 * math.pi * float(idx) / float(SEMANTIC_VIEW_YAW_COUNT)
            pose = _set_viewer_pose(base_xyz, yaw_rad=yaw, eye_height=ROOM_VIEW_CAMERA_HEIGHT_M)
            obs, info = _get_viewer_frame()
            visible_objects = _visible_unique_objects_from_frame(obs, info, objects_by_name=objects_by_name)
            score = (
                len(visible_objects),
                sum(int(item["semantic_pixels"]) for item in visible_objects),
                sum(int(item["instance_pixels"]) for item in visible_objects),
            )
            view_record = {
                "view_id": f"{region.room_instance}__semantic_{idx:02d}",
                "room_instance": region.room_instance,
                "render_xy": [float(render_xy[0]), float(render_xy[1])],
                "camera_pose": pose,
                "yaw_index": int(idx),
                "selection_debug": selection_debug,
                "visible_objects": visible_objects,
                "score": score,
            }
            catalog.append(view_record)
            best = best_by_region.get(region.room_instance)
            if best is None or tuple(score) > tuple(best["score"]):
                best_by_region[region.room_instance] = view_record
    return catalog, best_by_region


def _capture_saved_view_from_pose(output_path: str, camera_pose: dict) -> dict:
    target = camera_pose.get("target")
    if target is not None:
        return _capture_camera_view(output_path, eye=camera_pose["position"], target=target)
    yaw = float(camera_pose.get("yaw_rad", 0.0))
    eye = [float(v) for v in camera_pose["position"]]
    target = [
        float(eye[0] + math.cos(yaw)),
        float(eye[1] + math.sin(yaw)),
        float(eye[2]),
    ]
    return _capture_camera_view(output_path, eye=eye, target=target)


def _render_object_views(obj_record: ObjectRecord, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    target = [
        float(obj_record.aabb_center[0]),
        float(obj_record.aabb_center[1]),
        float(obj_record.aabb_center[2]),
    ]
    total_distance = float(OBJECT_CLOSEUP_DISTANCE_M)
    horizontal = total_distance * math.cos(math.radians(OBJECT_CLOSEUP_DOWNWARD_ANGLE_DEG))
    vertical = total_distance * math.sin(math.radians(OBJECT_CLOSEUP_DOWNWARD_ANGLE_DEG))
    azimuth_rad = math.radians(45.0)
    eye = [
        float(target[0] + horizontal * math.cos(azimuth_rad)),
        float(target[1] + horizontal * math.sin(azimuth_rad)),
        float(target[2] + vertical),
    ]
    view = _capture_camera_view(os.path.join(output_dir, f"{obj_record.category}__closeup.png"), eye=eye, target=target)
    view["distance_m"] = float(OBJECT_CLOSEUP_DISTANCE_M)
    view["downward_angle_deg"] = float(OBJECT_CLOSEUP_DOWNWARD_ANGLE_DEG)
    return {
        "success": True,
        "object_name": obj_record.name,
        "object_category": obj_record.category,
        "views": {"view_00": view},
    }


def _attach_render_photos(
    scene,
    floor_idx: int,
    floor_z: float,
    regions: list[RegionRecord],
    cases: dict[str, list[dict]],
    objects_by_name: dict[str, ObjectRecord],
    room_photo_root: str,
    object_photo_root: str,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    os.makedirs(room_photo_root, exist_ok=True)
    os.makedirs(object_photo_root, exist_ok=True)

    region_catalog, best_region_views = _build_region_semantic_view_catalog(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions=regions,
        objects_by_name=objects_by_name,
    )
    room_view_records = {}
    gt_region_views = {}
    for index, region in enumerate(regions, start=1):
        _debug(f"region render capture: room {index}/{len(regions)} {region.room_instance}")
        room_dir = os.path.join(room_photo_root, region.room_instance)
        room_view_records[region.room_instance] = _capture_room_reference_views(region, floor_z=floor_z, output_dir=room_dir)
        best_view = best_region_views.get(region.room_instance)
        if best_view is not None:
            gt_region_views[region.room_instance] = _capture_saved_view_from_pose(
                os.path.join(room_dir, f"{region.room_instance}__open_view.png"),
                camera_pose=best_view["camera_pose"],
            )
            gt_region_views[region.room_instance]["visible_objects"] = list(best_view.get("visible_objects", []))
            room_view_records[region.room_instance]["gt_open_view"] = gt_region_views[region.room_instance]

    required_object_names = sorted(
        {
            str(obj_name)
            for case_group in cases.values()
            for case in case_group
            for obj_name in case.get("required_objects", [])
            if obj_name in objects_by_name
        }
    )
    object_view_records = {}
    for index, obj_name in enumerate(required_object_names, start=1):
        _debug(f"object gt capture: object {index}/{len(required_object_names)} {obj_name}")
        obj_record = objects_by_name[obj_name]
        obj_dir = os.path.join(object_photo_root, obj_name)
        object_view_records[obj_name] = _render_object_views(obj_record=obj_record, output_dir=obj_dir)

    overview_cache: dict[str, dict] = {}
    for case_group in cases.values():
        for case in case_group:
            source_view = case.get("source_view") or {}
            view_id = str(source_view.get("view_id", ""))
            if view_id and view_id not in overview_cache:
                overview_cache[view_id] = _capture_saved_view_from_pose(
                    os.path.join(object_photo_root, f"{view_id}.png"),
                    camera_pose=source_view["camera_pose"],
                )
                overview_cache[view_id]["visible_objects"] = list(source_view.get("visible_objects", []))
            case["overview_view"] = overview_cache.get(view_id)
            case["room_views"] = {
                region_name: room_view_records[region_name]
                for region_name in case.get("required_regions", [])
                if region_name in room_view_records
            }
            case["object_views"] = {
                obj_name: object_view_records[obj_name]
                for obj_name in case.get("required_objects", [])
                if obj_name in object_view_records
            }
            case["gt_region_views"] = {
                region_name: gt_region_views[region_name]
                for region_name in case.get("required_regions", [])
                if region_name in gt_region_views
            }

    summary = {
        "enabled": True,
        "room_photo_root": room_photo_root,
        "object_photo_root": object_photo_root,
        "semantic_overview_candidates": len(region_catalog),
        "rooms_photographed": len(room_view_records),
        "open_region_views": len(gt_region_views),
        "objects_requested": len(required_object_names),
        "objects_photographed": sum(1 for record in object_view_records.values() if record.get("success")),
    }
    return summary, room_view_records, object_view_records


def _render_region_bbox_debug(scene, regions_by_name: dict[str, RegionRecord], objects: list[ObjectRecord], debug_root: str) -> dict:
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        return {"enabled": False, "reason": "scene._trav_map unavailable"}

    os.makedirs(debug_root, exist_ok=True)
    map_img = trav_map.floor_map[0].cpu().numpy()
    object_groups: dict[str, list[ObjectRecord]] = {}
    for obj in objects:
        if obj.assigned_region is None:
            continue
        object_groups.setdefault(obj.assigned_region, []).append(obj)

    overview_path = os.path.join(debug_root, "region_bbox_overview.png")
    fig = plt.figure(figsize=(8.0, 8.0))
    plt.imshow(map_img, cmap="gray", vmin=0, vmax=255)
    for region in regions_by_name.values():
        if region.bbox_map_rc is not None:
            rmin, cmin, rmax, cmax = region.bbox_map_rc
            width = max(1.0, float(cmax - cmin))
            height = max(1.0, float(rmax - rmin))
            plt.gca().add_patch(plt.Rectangle((cmin, rmin), width, height, fill=False, edgecolor="white", linewidth=1.2))
        if region.expanded_bbox_world_xy is not None:
            xmin, ymin, xmax, ymax = region.expanded_bbox_world_xy
            p0 = _world_to_plot_xy(trav_map, (xmin, ymin))
            p1 = _world_to_plot_xy(trav_map, (xmax, ymax))
            left = min(p0[0], p1[0])
            top = min(p0[1], p1[1])
            width = max(1.0, abs(p1[0] - p0[0]))
            height = max(1.0, abs(p1[1] - p0[1]))
            plt.gca().add_patch(plt.Rectangle((left, top), width, height, fill=False, edgecolor="orange", linewidth=1.5))
        if region.center_xy is not None:
            px, py = _world_to_plot_xy(trav_map, region.center_xy)
            plt.scatter([px], [py], c="cyan", s=10)
            plt.text(px + 1.0, py + 1.0, region.room_instance, fontsize=6, color="cyan")
    plt.title("Room bbox expansion overview", fontsize=10)
    plt.tight_layout()
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    saved = [overview_path]
    for region_name, region in regions_by_name.items():
        fig = plt.figure(figsize=(7.0, 7.0))
        plt.imshow(map_img, cmap="gray", vmin=0, vmax=255)
        if region.bbox_map_rc is not None:
            rmin, cmin, rmax, cmax = region.bbox_map_rc
            width = max(1.0, float(cmax - cmin))
            height = max(1.0, float(rmax - rmin))
            plt.gca().add_patch(plt.Rectangle((cmin, rmin), width, height, fill=False, edgecolor="white", linewidth=1.5, label="original"))
        if region.expanded_bbox_world_xy is not None:
            xmin, ymin, xmax, ymax = region.expanded_bbox_world_xy
            p0 = _world_to_plot_xy(trav_map, (xmin, ymin))
            p1 = _world_to_plot_xy(trav_map, (xmax, ymax))
            left = min(p0[0], p1[0])
            top = min(p0[1], p1[1])
            width = max(1.0, abs(p1[0] - p0[0]))
            height = max(1.0, abs(p1[1] - p0[1]))
            plt.gca().add_patch(plt.Rectangle((left, top), width, height, fill=False, edgecolor="orange", linewidth=1.8, label="expanded"))
        for obj in object_groups.get(region_name, []):
            px, py = _world_to_plot_xy(trav_map, obj.center_xy)
            plt.scatter([px], [py], c="lime", s=20)
            plt.text(px + 0.8, py + 0.8, obj.category, fontsize=5, color="lime")
        plt.title(f"{region_name} bbox expansion debug", fontsize=10)
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(debug_root, f"{region_name}.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved.append(out_path)

    return {
        "enabled": True,
        "debug_root": debug_root,
        "saved_images": saved,
    }


def _render_case_visualizations(scene, regions_by_name: dict[str, RegionRecord], objects_by_name: dict[str, ObjectRecord], cases: dict[str, list[dict]], viz_root: str) -> dict:
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        return {"enabled": False, "reason": "scene._trav_map unavailable"}

    os.makedirs(viz_root, exist_ok=True)
    map_img = trav_map.floor_map[0].cpu().numpy()
    total_cases = sum(len(group) for group in cases.values())
    case_counter = 0

    for case_group in cases.values():
        for case in case_group:
            case_counter += 1
            if case_counter % 10 == 0 or case_counter == 1:
                _debug(f"visualization progress: case {case_counter}/{total_cases} {case['case_id']}")

            fig = plt.figure(figsize=(7.0, 7.0))
            plt.imshow(map_img, cmap="gray", vmin=0, vmax=255)

            highlighted_regions = set(case.get("required_regions", []))
            for region_name, region in regions_by_name.items():
                color = "deepskyblue" if region_name in highlighted_regions else "white"
                if region.bbox_map_rc is not None:
                    rmin, cmin, rmax, cmax = region.bbox_map_rc
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (cmin, rmin),
                            max(1.0, float(cmax - cmin)),
                            max(1.0, float(rmax - rmin)),
                            fill=False,
                            edgecolor=color,
                            linewidth=1.1,
                        )
                    )
                if region.expanded_bbox_world_xy is not None and region_name in highlighted_regions:
                    xmin, ymin, xmax, ymax = region.expanded_bbox_world_xy
                    p0 = _world_to_plot_xy(trav_map, (xmin, ymin))
                    p1 = _world_to_plot_xy(trav_map, (xmax, ymax))
                    left = min(p0[0], p1[0])
                    top = min(p0[1], p1[1])
                    width = max(1.0, abs(p1[0] - p0[0]))
                    height = max(1.0, abs(p1[1] - p0[1]))
                    plt.gca().add_patch(
                        plt.Rectangle((left, top), width, height, fill=False, edgecolor="orange", linewidth=1.8)
                    )
                if region.center_xy is not None and region_name in highlighted_regions:
                    px, py = _world_to_plot_xy(trav_map, region.center_xy)
                    plt.scatter([px], [py], c="orange", s=18)
                    plt.text(px + 1.0, py + 1.0, region_name, fontsize=7, color="orange")

            for obj_name in case.get("required_objects", []):
                obj = objects_by_name.get(obj_name)
                if obj is None:
                    continue
                px, py = _world_to_plot_xy(trav_map, obj.center_xy)
                plt.scatter([px], [py], c="lime", s=28, zorder=4)
                plt.text(px + 1.0, py + 1.0, obj.category, fontsize=7, color="lime")

            plt.title(f"{case['case_type']}\n{case['question']}", fontsize=10)
            plt.tight_layout()
            out_path = os.path.join(viz_root, f"{case['case_id']}.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            case["visualization"] = {"map_path": out_path}

    return {"enabled": True, "viz_root": viz_root}


def _generate_belong_region_cases(
    objects: list[ObjectRecord],
    regions_by_name: dict[str, RegionRecord],
    max_cases: int,
    rng: random.Random,
    semantic_views: list[dict],
) -> list[dict]:
    positives = []
    negatives = []
    region_names = sorted(regions_by_name)
    objects_by_name = {obj.name: obj for obj in objects}
    seen_positive = set()
    seen_negative = set()

    for view in semantic_views:
        for visible in view.get("visible_objects", []):
            obj = objects_by_name.get(visible["object_name"])
            if obj is None or obj.assigned_region is None:
                continue

            positive_key = (obj.category, obj.assigned_region)
            if positive_key not in seen_positive:
                positives.append(
                    {
                        "case_id": _make_case_id("object_in_region", obj.category, obj.assigned_region),
                        "case_type": "object_in_region",
                        "question": f"Does {obj.category} belong to {obj.assigned_region}?",
                        "answer": "Yes",
                        "answer_bool": True,
                        "required_objects": [obj.name],
                        "required_regions": [obj.assigned_region],
                        "object_name": obj.category,
                        "object_instance_name": obj.name,
                        "object_category": obj.category,
                        "region_name": obj.assigned_region,
                        "region_type": obj.assigned_room_type,
                        "source_view": view,
                        "verification": {
                            "assignment_source": obj.assignment_source,
                            "containment_regions": list(obj.containment_regions),
                            "semantic_visible_pixels": int(visible.get("semantic_pixels", 0)),
                            "instance_visible_pixels": int(visible.get("instance_pixels", 0)),
                        },
                    }
                )
                seen_positive.add(positive_key)

            other_regions = [name for name in region_names if name != obj.assigned_region]
            if not other_regions:
                continue
            neg_region = rng.choice(other_regions)
            negative_key = (obj.category, neg_region, obj.assigned_region)
            if negative_key in seen_negative:
                continue
            negatives.append(
                {
                    "case_id": _make_case_id("object_in_region", obj.category, neg_region),
                    "case_type": "object_in_region",
                    "question": f"Does {obj.category} belong to {neg_region}?",
                    "answer": "No",
                    "answer_bool": False,
                    "required_objects": [obj.name],
                    "required_regions": [neg_region, obj.assigned_region],
                    "object_name": obj.category,
                    "object_instance_name": obj.name,
                    "object_category": obj.category,
                    "region_name": neg_region,
                    "region_type": regions_by_name[neg_region].room_type,
                    "source_view": view,
                    "verification": {
                        "assigned_region": obj.assigned_region,
                        "assignment_source": obj.assignment_source,
                        "containment_regions": list(obj.containment_regions),
                        "semantic_visible_pixels": int(visible.get("semantic_pixels", 0)),
                        "instance_visible_pixels": int(visible.get("instance_pixels", 0)),
                    },
                }
            )
            seen_negative.add(negative_key)

    rng.shuffle(positives)
    rng.shuffle(negatives)
    mixed = []
    while positives or negatives:
        if positives:
            mixed.append(positives.pop())
        if negatives:
            mixed.append(negatives.pop())
        if len(mixed) >= max_cases:
            break
    return mixed[:max_cases]


def _generate_same_region_cases(
    objects: list[ObjectRecord],
    max_cases: int,
    rng: random.Random,
    semantic_views: list[dict],
) -> list[dict]:
    positives = []
    negatives = []
    objects_by_name = {obj.name: obj for obj in objects}
    seen_pairs = set()

    for view in semantic_views:
        visible_entries = list(view.get("visible_objects", []))
        for idx in range(len(visible_entries)):
            for jdx in range(idx + 1, len(visible_entries)):
                obj_a = objects_by_name.get(visible_entries[idx]["object_name"])
                obj_b = objects_by_name.get(visible_entries[jdx]["object_name"])
                if obj_a is None or obj_b is None:
                    continue
                if obj_a.category == obj_b.category:
                    continue
                pair_key = tuple(sorted((obj_a.category, obj_b.category)))
                if pair_key in seen_pairs:
                    continue
                same_region = obj_a.assigned_region == obj_b.assigned_region
                payload = {
                    "case_id": _make_case_id("objects_same_region", obj_a.category, obj_b.category),
                    "case_type": "objects_same_region",
                    "question": f"Are {obj_a.category} and {obj_b.category} in the same region?",
                    "answer": "Yes" if same_region else "No",
                    "answer_bool": same_region,
                    "required_objects": [obj_a.name, obj_b.name],
                    "required_regions": sorted({obj_a.assigned_region, obj_b.assigned_region}),
                    "object_names": [obj_a.category, obj_b.category],
                    "object_instance_names": [obj_a.name, obj_b.name],
                    "source_view": view,
                    "verification": {
                        "object_a_region": obj_a.assigned_region,
                        "object_b_region": obj_b.assigned_region,
                    },
                }
                if same_region:
                    positives.append(payload)
                else:
                    negatives.append(payload)
                seen_pairs.add(pair_key)

    rng.shuffle(positives)
    rng.shuffle(negatives)
    mixed = []
    while positives or negatives:
        if positives:
            mixed.append(positives.pop())
        if negatives:
            mixed.append(negatives.pop())
        if len(mixed) >= max_cases:
            break
    return mixed[:max_cases]


def _generate_closer_region_cases(
    objects: list[ObjectRecord],
    regions_by_name: dict[str, RegionRecord],
    max_cases: int,
    rng: random.Random,
    closer_margin: float,
    semantic_views: list[dict],
) -> list[dict]:
    candidates = []
    objects_by_name = {obj.name: obj for obj in objects}
    seen_categories = set()

    for view in semantic_views:
        for visible in view.get("visible_objects", []):
            obj = objects_by_name.get(visible["object_name"])
            if obj is None or obj.category in seen_categories:
                continue
            scored = []
            for region_name, region in regions_by_name.items():
                if region.expanded_bbox_world_xy is None:
                    continue
                bbox_distance = _distance_point_to_bbox_xy(obj.center_xy, region.expanded_bbox_world_xy)
                center_distance = _distance_xy(obj.center_xy, region.center_xy if region.center_xy is not None else obj.center_xy)
                scored.append((bbox_distance, center_distance, region_name))
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            if len(scored) < 2:
                continue

            best = scored[0]
            rival = None
            for item in scored[1:]:
                if item[2] == best[2]:
                    continue
                if item[0] - best[0] >= float(closer_margin) or item[1] - best[1] >= float(closer_margin):
                    rival = item
                    break
            if rival is None:
                rival = scored[1]
            if rival[2] == best[2]:
                continue

            options = [best[2], rival[2]]
            rng.shuffle(options)
            answer_region = best[2]
            candidates.append(
                {
                    "case_id": _make_case_id("object_closer_region", obj.category, options[0], options[1]),
                    "case_type": "object_closer_region",
                    "question": f"Is {obj.category} closer to {options[0]} or {options[1]}?",
                    "answer": answer_region,
                    "answer_region": answer_region,
                    "required_objects": [obj.name],
                    "required_regions": options,
                    "object_name": obj.category,
                    "object_instance_name": obj.name,
                    "source_view": view,
                    "verification": {
                        "distance_to_regions": {
                            best[2]: {
                                "bbox_distance": float(best[0]),
                                "center_distance": float(best[1]),
                            },
                            rival[2]: {
                                "bbox_distance": float(rival[0]),
                                "center_distance": float(rival[1]),
                            },
                        },
                        "metric": "distance from object XY center to expanded region bbox; center distance used as tiebreaker",
                        "semantic_visible_pixels": int(visible.get("semantic_pixels", 0)),
                        "instance_visible_pixels": int(visible.get("instance_pixels", 0)),
                    },
                }
            )
            seen_categories.add(obj.category)

    rng.shuffle(candidates)
    return candidates[:max_cases]


def _region_options_for_case(case: dict) -> list[str]:
    if case["case_type"] in {"object_in_region", "objects_same_region"}:
        return ["Yes", "No"]
    if case["case_type"] == "object_closer_region":
        return list(case.get("required_regions", []))
    return []


def _first_object_view_image(case: dict) -> str | None:
    overview_image = (case.get("overview_view") or {}).get("image_path")
    if overview_image:
        return overview_image
    object_views = case.get("object_views") or {}
    for obj_name in case.get("required_objects", []):
        render_info = object_views.get(obj_name) or {}
        views = render_info.get("views") or {}
        for view_key in sorted(views):
            image_path = (views.get(view_key) or {}).get("image_path")
            if image_path:
                return image_path
    return None


def _first_room_view_image(case: dict) -> str | None:
    room_views = case.get("room_views") or {}
    for region_name in case.get("required_regions", []):
        render_info = room_views.get(region_name) or {}
        views = render_info.get("views") or {}
        for view_key in sorted(views):
            image_path = (views.get(view_key) or {}).get("image_path")
            if image_path:
                return image_path
    return None


def _build_single_question_entry(case: dict) -> dict:
    task_type = str(case["case_type"])
    entry = dict(case)
    map_path = (case.get("visualization") or {}).get("map_path")
    reference_image = _first_object_view_image(case)
    render = {
        "image": reference_image,
        "kind": "semantic_overview_reference",
        "topdown_map": map_path,
        "overview_view": case.get("overview_view"),
        "room_views": case.get("room_views"),
        "object_views": case.get("object_views"),
        "gt_region_views": case.get("gt_region_views"),
    }
    entry.update(
        {
            "task_type": task_type,
            "options": _region_options_for_case(case),
            "render": render,
        }
    )
    return entry


def _export_single_question_jsons(
    cases: dict[str, list[dict]],
    output_root: str,
    candidate_root: str,
    map_root: str,
    scene_metadata: dict,
) -> dict:
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(candidate_root, exist_ok=True)
    os.makedirs(map_root, exist_ok=True)
    by_family = {}
    for task_type, entries in cases.items():
        family = _task_family_for_type(task_type)
        family_entries = by_family.setdefault(family, [])
        for case in entries:
            family_entries.append((task_type, _build_single_question_entry(case)))

    written = {}
    by_task_type = {}
    for task_family, entries in by_family.items():
        written_paths = []
        for q_idx, (task_type, entry) in enumerate(entries):
            out_path = _write_single_question_json(
                output_root=output_root,
                candidate_root=candidate_root,
                map_root=map_root,
                scene_metadata=scene_metadata,
                task_type=task_type,
                q_idx=q_idx,
                entry=entry,
            )
            written_paths.append(out_path)
            by_task_type[task_type] = by_task_type.get(task_type, 0) + 1
        written[task_family] = written_paths
    return {
        "enabled": True,
        "question_json_root": output_root,
        "candidate_render_root": candidate_root,
        "map_visualization_root": map_root,
        "counts": {task_family: len(paths) for task_family, paths in written.items()},
        "counts_by_task_type": by_task_type,
        "paths": written,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate region QA metadata from runtime OmniGibson scene.")
    parser.add_argument("--scene", default="house_double_floor_upper", help="Scene model name")
    parser.add_argument("--room", type=str, default=None, help="Optional room instance name")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point_candidates", type=int, default=DEFAULT_POINT_CANDIDATES)
    parser.add_argument("--trav_map_basename", type=str, default=DEFAULT_TRAV_MAP_BASENAME)
    parser.add_argument("--output_root", type=str, default="renders_region")
    parser.add_argument("--max_belong_cases", type=int, default=DEFAULT_BELONG_CASE_LIMIT)
    parser.add_argument("--max_same_region_cases", type=int, default=DEFAULT_SAME_REGION_LIMIT)
    parser.add_argument("--max_closer_region_cases", type=int, default=DEFAULT_CLOSER_REGION_LIMIT)
    parser.add_argument("--region_expansion_ratio", type=float, default=DEFAULT_EXPANSION_RATIO)
    parser.add_argument("--region_expansion_min", type=float, default=DEFAULT_EXPANSION_MIN)
    parser.add_argument("--closer_margin", type=float, default=DEFAULT_CLOSER_MARGIN)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    room_dirname = args.room if args.room is not None else "full_scene"
    run_dir = os.path.join(args.output_root, args.scene, room_dirname)
    os.makedirs(run_dir, exist_ok=True)
    output_json = os.path.join(run_dir, DEFAULT_OUTPUT_NAME)
    staging_root = os.path.join(run_dir, "_staging")
    room_photo_root = os.path.join(staging_root, "region_room_views")
    object_photo_root = os.path.join(staging_root, "region_object_views")
    question_json_root = os.path.join(run_dir, "cognitivemap_question_jsons")
    candidate_render_root = os.path.join(run_dir, "candidate_renders")
    map_visualization_root = os.path.join(run_dir, "map_visualization")

    config = _build_config(args)
    try:
        _debug(f"starting environment creation for scene={args.scene} room={args.room}")
        env = og.Environment(configs=config)
        _set_viewer_camera_fov()
        scene = env.scene
        # No explicit agent placement is performed in this script.
        removed_doors = _remove_named_doors(scene)
        replaced_trav_map = _replace_trav_map_with_variant(scene, basename=str(args.trav_map_basename))
        _debug(f"trav-map replacement status: {replaced_trav_map}")
        floor_idx = 0
        floor_z = float(scene.get_floor_height(int(floor_idx)))
        wall_records = _collect_wall_records(scene)
        structural_wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]

        regions = _build_region_records(
            scene=scene,
            seed=args.seed,
            point_candidates=args.point_candidates,
            expansion_ratio=args.region_expansion_ratio,
            expansion_min=args.region_expansion_min,
            wall_bboxes_xyxy=structural_wall_bboxes,
        )
        if len(regions) < 2:
            raise RuntimeError("Not enough regions found to generate region questions.")

        regions_by_name = {region.room_instance: region for region in regions}
        objects = _collect_scene_objects(scene=scene, regions_by_name=regions_by_name)
        if len(objects) < 2:
            raise RuntimeError("Not enough assigned objects found to generate region questions.")
        objects_by_name = {obj.name: obj for obj in objects}
        semantic_views, _ = _build_region_semantic_view_catalog(
            scene=scene,
            floor_idx=floor_idx,
            floor_z=floor_z,
            regions=regions,
            objects_by_name=objects_by_name,
        )

        cases = {
            "object_in_region": _generate_belong_region_cases(
                objects=objects,
                regions_by_name=regions_by_name,
                max_cases=args.max_belong_cases,
                rng=rng,
                semantic_views=semantic_views,
            ),
            "objects_same_region": _generate_same_region_cases(
                objects=objects,
                max_cases=args.max_same_region_cases,
                rng=rng,
                semantic_views=semantic_views,
            ),
            "object_closer_region": _generate_closer_region_cases(
                objects=objects,
                regions_by_name=regions_by_name,
                max_cases=args.max_closer_region_cases,
                rng=rng,
                closer_margin=args.closer_margin,
                semantic_views=semantic_views,
            ),
        }
        _debug(
            "case generation finished: "
            f"object_in_region={len(cases['object_in_region'])} "
            f"objects_same_region={len(cases['objects_same_region'])} "
            f"object_closer_region={len(cases['object_closer_region'])}"
        )
        visualization_summary = _render_case_visualizations(
            scene=scene,
            regions_by_name=regions_by_name,
            objects_by_name=objects_by_name,
            cases=cases,
            viz_root=map_visualization_root,
        )

        photo_summary, room_view_records, object_view_records = _attach_render_photos(
            scene=scene,
            floor_idx=floor_idx,
            floor_z=floor_z,
            regions=regions,
            cases=cases,
            objects_by_name=objects_by_name,
            room_photo_root=room_photo_root,
            object_photo_root=object_photo_root,
        )
        question_json_summary = _export_single_question_jsons(
            cases=cases,
            output_root=question_json_root,
            candidate_root=candidate_render_root,
            map_root=map_visualization_root,
            scene_metadata={
                "scene": args.scene,
                "room": args.room,
                "floor_name": None,
                "seed": args.seed,
                "trav_map_basename": args.trav_map_basename,
            },
        )
        _debug(f"single-question json summary: {question_json_summary['counts']}")

        metadata = {
            "scene": args.scene,
            "room": args.room,
            "seed": args.seed,
            "trav_map_basename": args.trav_map_basename,
            "trav_map_replaced": bool(replaced_trav_map),
            "removed_doors": removed_doors,
            "region_expansion_ratio": float(args.region_expansion_ratio),
            "region_expansion_min": float(args.region_expansion_min),
            "closer_margin": float(args.closer_margin),
            "region_count": len(regions),
            "object_count": len(objects),
            "wall_related_object_count": len(wall_records),
            "structural_wall_count": len(structural_wall_bboxes),
            "regions": [region.to_json() for region in regions],
            "wall_related_objects": [wall.to_json() for wall in wall_records],
            "objects": [obj.to_json() for obj in objects],
            "room_views": room_view_records,
            "object_views": object_view_records,
            "cases": cases,
            "visualization": visualization_summary,
            "render_photo_summary": photo_summary,
            "question_json_summary": question_json_summary,
            "notes": [
                "Room regions are extracted from scene.seg_map.room_ins_map.",
                "Each room bbox is expanded in world XY before object-region assignment because raw room bbox can be inaccurate.",
                "Wall-related scene objects are collected first by matching 'wall' in object names.",
                "Structural wall AABBs are used as stop boundaries during bbox expansion.",
                "Each expansion direction stops as soon as the candidate bbox touches a structural wall in that direction.",
                "Object-region assignment uses expanded room bbox first, then seg_map point query, then obj.in_rooms as fallback.",
                "Closer-region questions use distance from object XY center to expanded room bbox; region-center distance is used as a tiebreaker.",
                "Questions are sampled from category-unique objects visible in semantic overview views, requiring at least 100 visible pixels per category.",
                "Question wording uses object category names rather than instance ids, and same-region pairs never reuse the same category twice.",
                "Each case includes one semantic-overview reference image and one top-down map visualization.",
                "Each room stores four corner room views plus one wide open region view.",
                "Only objects referenced by generated questions are rendered, and each such object stores one 0.8m closeup view from a 45-degree downward angle.",
            ],
        }
        _write_json(output_json, metadata)

        print(
            json.dumps(
                {
                    "output_json": output_json,
                    "scene": args.scene,
                    "room": args.room,
                    "seed": args.seed,
                    "region_count": len(regions),
                    "object_count": len(objects),
                    "wall_related_object_count": len(wall_records),
                    "structural_wall_count": len(structural_wall_bboxes),
                    "summary": {case_name: len(entries) for case_name, entries in cases.items()},
                    "room_photo_root": photo_summary.get("room_photo_root"),
                    "object_photo_root": photo_summary.get("object_photo_root"),
                    "question_json_root": question_json_summary["question_json_root"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        _log_exception("main", exc)
        raise
    finally:
        if os.path.isdir(staging_root):
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    main()
