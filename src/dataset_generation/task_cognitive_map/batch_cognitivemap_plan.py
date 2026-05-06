"""
Generate navigation-planning QA candidates from a runtime OmniGibson scene.

Compared with batch_cognitivemap_connect.py, this script focuses on
room-to-room navigation plans:
1. Extract room regions from the segmentation map
2. Expand room bboxes because raw room bbox can be inaccurate
3. Replace the traversability map with the no-door variant
4. Find far and connected room pairs using scene shortest paths
5. Generate two planning question styles:
   - action-choice navigation plans
   - region-choice exploration plans
6. Save eight room-level reference views per room plus one initial view per question
"""

from __future__ import annotations

import argparse
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

import cv2
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

DEFAULT_OUTPUT_NAME = "cognitivemap_plan_candidates.json"
DEFAULT_TRAV_MAP_BASENAME = "floor_trav_no_door"
DEFAULT_POINT_CANDIDATES = 7
DEFAULT_EXPANSION_RATIO = 0.35
DEFAULT_EXPANSION_MIN = 0.75
DEFAULT_CASE_LIMIT = 6
DEFAULT_PLAN_REGION_LIMIT = 8
DEFAULT_OUTPUT_ROOT = str(SCRIPT_DIR / "renders_plan")
DEFAULT_ROOM_VIEW_COUNT = 8
VIEWER_CAMERA_FOV_DEG = 100.0

PLAN_REGION_LIMIT = DEFAULT_PLAN_REGION_LIMIT

TASK_FAMILY_BY_TYPE = {
    "navigation_actions": "Long-Horizon Navigation",
    "navigation_regions": "Long-Horizon Navigation",
}


class PlanTaskSkipped(RuntimeError):
    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = str(reason)
        self.details = details or {}


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


def _asset_target_dir(candidate_dir: str, key_path: tuple[str, ...]) -> str:
    path_keys = set(key_path)
    if "room_views" in path_keys or "source_room_views" in path_keys:
        return os.path.join(candidate_dir, "room_views")
    if path_keys.intersection({"path_views", "gt_views"}):
        return os.path.join(candidate_dir, "gt_views")
    return candidate_dir


def _relocate_entry_assets(
    payload,
    candidate_dir: str,
    copied_paths: dict[tuple[str, str], str],
    key_path: tuple[str, ...] = (),
):
    if isinstance(payload, dict):
        relocated = {}
        for key, value in payload.items():
            if key in {"image", "image_path"} and isinstance(value, str) and os.path.isfile(value):
                relocated[key] = _copy_asset(value, _asset_target_dir(candidate_dir, key_path), copied_paths)
            else:
                relocated[key] = _relocate_entry_assets(
                    value,
                    candidate_dir,
                    copied_paths,
                    key_path + (str(key),),
                )
        return relocated
    if isinstance(payload, list):
        return [_relocate_entry_assets(item, candidate_dir, copied_paths, key_path) for item in payload]
    return payload


def _write_single_question_json(
    output_root: str,
    candidate_root: str,
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


def _write_skip_metadata(
    output_json: str,
    scene_metadata: dict,
    reason: str,
    details: dict | None = None,
) -> dict:
    payload = {
        "scene": scene_metadata.get("scene"),
        "room": scene_metadata.get("room"),
        "seed": scene_metadata.get("seed"),
        "trav_map_basename": scene_metadata.get("trav_map_basename"),
        "skipped": True,
        "skip_reason": str(reason),
        "skip_details": details or {},
        "question_json_summary": {
            "enabled": False,
            "question_json_root": scene_metadata.get("question_json_root"),
            "counts": {},
            "paths": {},
        },
    }
    _write_json(output_json, payload)
    return payload


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_cognitivemap_plan] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _debug(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[batch_cognitivemap_plan {now}] {message}", flush=True)


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

        src_path = None
        src_img = None
        for fname in candidates:
            fpath = os.path.join(maps_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                src_path = fpath
                src_img = img
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


def _trav_map_floor_image(scene, floor_idx: int):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        return None, None
    if floor_idx < 0 or floor_idx >= len(trav_map.floor_map):
        return trav_map, None
    floor_img = trav_map.floor_map[floor_idx]
    if hasattr(floor_img, "detach"):
        floor_img = floor_img.detach()
    if getattr(floor_img, "device", None) is not None and floor_img.device.type != "cpu":
        floor_img = floor_img.cpu()
    return trav_map, np.array(floor_img)


def _clip_map_rc(map_img: np.ndarray, rc) -> tuple[int, int]:
    row = int(np.clip(int(round(float(rc[0]))), 0, map_img.shape[0] - 1))
    col = int(np.clip(int(round(float(rc[1]))), 0, map_img.shape[1] - 1))
    return row, col


def _is_free_rc(map_img: np.ndarray, rc) -> bool:
    row, col = _clip_map_rc(map_img, rc)
    return bool(map_img[row, col] > 0)


def _nearest_rc_from_candidates(target_rc, candidate_rcs: np.ndarray) -> tuple[int, int] | None:
    if candidate_rcs is None or len(candidate_rcs) == 0:
        return None
    deltas = candidate_rcs.astype(np.float32) - np.array([float(target_rc[0]), float(target_rc[1])], dtype=np.float32)
    dists = np.sum(deltas * deltas, axis=1)
    best_idx = int(np.argmin(dists))
    best = candidate_rcs[best_idx]
    return int(best[0]), int(best[1])


def _map_rc_to_world_xy(trav_map, rc) -> list[float]:
    xy = trav_map.map_to_world(th.tensor([float(rc[0]), float(rc[1])], dtype=th.float32))
    return [float(xy[0].item()), float(xy[1].item())]


def _adjust_xy_to_nearest_free_point(scene, floor_idx: int, xy, room_pixels_rc: np.ndarray | None = None) -> list[float]:
    trav_map, map_img = _trav_map_floor_image(scene, floor_idx=floor_idx)
    if trav_map is None or map_img is None:
        return [float(xy[0]), float(xy[1])]

    rc_arr = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    original_rc = _clip_map_rc(map_img, rc_arr)
    if _is_free_rc(map_img, original_rc):
        return [float(xy[0]), float(xy[1])]

    free_room_rcs = None
    if room_pixels_rc is not None and len(room_pixels_rc) > 0:
        room_pixels_rc = np.asarray(room_pixels_rc, dtype=np.int64)
        valid_mask = (
            (room_pixels_rc[:, 0] >= 0)
            & (room_pixels_rc[:, 0] < map_img.shape[0])
            & (room_pixels_rc[:, 1] >= 0)
            & (room_pixels_rc[:, 1] < map_img.shape[1])
        )
        room_pixels_rc = room_pixels_rc[valid_mask]
        if len(room_pixels_rc) > 0:
            free_mask = map_img[room_pixels_rc[:, 0], room_pixels_rc[:, 1]] > 0
            free_room_rcs = room_pixels_rc[free_mask]

    best_rc = _nearest_rc_from_candidates(original_rc, free_room_rcs)
    if best_rc is not None:
        return _map_rc_to_world_xy(trav_map, best_rc)

    all_free_rcs = np.argwhere(map_img > 0)
    best_rc = _nearest_rc_from_candidates(original_rc, all_free_rcs)
    if best_rc is not None:
        return _map_rc_to_world_xy(trav_map, best_rc)

    return [float(xy[0]), float(xy[1])]


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


def _bbox_gap_xy(a, b) -> float:
    ax0, ay0, ax1, ay1 = _normalize_bbox_xyxy(a)
    bx0, by0, bx1, by1 = _normalize_bbox_xyxy(b)
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _point_inside_bbox_xy(point_xy, bbox_xyxy, margin: float = 0.0) -> bool:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return (
        xmin - float(margin) <= float(point_xy[0]) <= xmax + float(margin)
        and ymin - float(margin) <= float(point_xy[1]) <= ymax + float(margin)
    )


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


def _collect_wall_records(scene) -> list[WallRecord]:
    wall_records = []

    for obj in getattr(scene, "objects", []):
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
        pixels = np.argwhere(room_map_np == int(room_id))
        if pixels.shape[0] == 0:
            continue

        bbox_info = _segmap_room_bbox_from_maps(scene, room_id)
        bbox_world_xy = bbox_info["bbox_world_xy"]
        expanded_bbox_world_xy = None
        center_xy = None
        candidate_points = []
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
            raw_center_xy = _room_bbox_center_xy(bbox_world_xy).tolist()
            adjusted_center_xy = _adjust_xy_to_nearest_free_point(
                scene=scene,
                floor_idx=0,
                xy=raw_center_xy,
                room_pixels_rc=pixels,
            )
            center_xy = tuple(float(x) for x in adjusted_center_xy)
            candidate_points = [[float(center_xy[0]), float(center_xy[1])]]

        room_records.append(
            RegionRecord(
                room_id=room_id,
                room_instance=room_instance,
                room_type=room_type,
                bbox_map_rc=bbox_info["bbox_map_rc"],
                bbox_world_xy=bbox_world_xy,
                expanded_bbox_world_xy=expanded_bbox_world_xy,
                center_xy=center_xy,
                candidate_points_xy=candidate_points,
                pixel_count=int(bbox_info["pixel_count"]),
            )
        )

    room_records.sort(key=lambda record: (record.room_type, record.room_instance))
    _debug(f"finished region extraction: {len(room_records)} valid regions")
    return room_records


def _region_center_xy(region: RegionRecord) -> tuple[float, float] | None:
    if region.center_xy is not None:
        return (float(region.center_xy[0]), float(region.center_xy[1]))
    if region.candidate_points_xy:
        point = region.candidate_points_xy[0]
        return (float(point[0]), float(point[1]))
    return None


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


def _assign_region_by_xy(xy, regions_by_name: dict[str, RegionRecord], seg=None) -> str | None:
    containing = []
    for region in regions_by_name.values():
        bbox = region.expanded_bbox_world_xy or region.bbox_world_xy
        if bbox is None:
            continue
        if _point_inside_bbox_xy(xy, bbox):
            center = _region_center_xy(region) or xy
            containing.append((_distance_xy(xy, center), region.room_instance))
    if containing:
        containing.sort()
        return containing[0][1]

    nearest = []
    for region in regions_by_name.values():
        bbox = region.expanded_bbox_world_xy or region.bbox_world_xy
        center = _region_center_xy(region)
        if bbox is None or center is None:
            continue
        nearest.append((_bbox_gap_xy((xy[0], xy[1], xy[0], xy[1]), bbox), _distance_xy(xy, center), region.room_instance))
    if nearest:
        nearest.sort()
        if nearest[0][0] <= 1.25:
            return nearest[0][2]

    if seg is not None:
        return _query_room_instance_by_point(seg, xy)
    return None


def _get_shortest_path(scene, floor_idx: int, start_xy, goal_xy, ignore_agent_footprint: bool = True):
    try:
        kwargs = {
            "floor": int(floor_idx),
            "source_world": th.tensor([float(start_xy[0]), float(start_xy[1])], dtype=th.float32),
            "target_world": th.tensor([float(goal_xy[0]), float(goal_xy[1])], dtype=th.float32),
            "entire_path": True,
        }
        path_world, distance = scene.get_shortest_path(**kwargs)
        return path_world, distance
    except Exception:
        return None, None


def _path_to_region_sequence(scene, regions_by_name: dict[str, RegionRecord], path_world) -> list[str]:
    if path_world is None:
        return []
    seg = _segmap_get(scene)
    sequence = []
    last = None
    for waypoint in path_world:
        xy = _tensor_to_list(waypoint)[:2]
        room_instance = _assign_region_by_xy(xy, regions_by_name=regions_by_name, seg=seg)
        if room_instance is None or room_instance == last:
            continue
        sequence.append(room_instance)
        last = room_instance
    return sequence


def _find_best_region_path(scene, floor_idx: int, src: RegionRecord, dst: RegionRecord, regions_by_name: dict[str, RegionRecord]):
    best = None
    for src_xy in src.candidate_points_xy:
        for dst_xy in dst.candidate_points_xy:
            path_world, distance = _get_shortest_path(
                scene,
                floor_idx,
                src_xy,
                dst_xy,
                ignore_agent_footprint=True,
            )
            if path_world is None or len(path_world) == 0:
                continue

            region_sequence = _path_to_region_sequence(scene, regions_by_name, path_world)
            if not region_sequence:
                continue
            if region_sequence[0] != src.room_instance or region_sequence[-1] != dst.room_instance:
                continue

            score = (
                float(distance) if distance is not None else float(len(path_world)),
                len(region_sequence),
                len(path_world),
            )
            if best is None or score > best["score"]:
                best = {
                    "start_xy": [float(src_xy[0]), float(src_xy[1])],
                    "goal_xy": [float(dst_xy[0]), float(dst_xy[1])],
                    "path_world": [[float(p[0]), float(p[1])] for p in _tensor_to_list(path_world)],
                    "path_distance": float(distance) if distance is not None else float(len(path_world)),
                    "region_sequence": region_sequence,
                    "score": score,
                }
    return best


def _build_expanded_connectivity_graph(regions: list[RegionRecord], gap_threshold: float = 1.0) -> dict[str, list[str]]:
    graph = {region.room_instance: set() for region in regions}
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a = regions[i]
            b = regions[j]
            bbox_a = a.expanded_bbox_world_xy or a.bbox_world_xy
            bbox_b = b.expanded_bbox_world_xy or b.bbox_world_xy
            if bbox_a is None or bbox_b is None:
                continue
            if _bbox_gap_xy(bbox_a, bbox_b) <= float(gap_threshold):
                graph[a.room_instance].add(b.room_instance)
                graph[b.room_instance].add(a.room_instance)
    return {name: sorted(neighbors) for name, neighbors in graph.items()}


def _yaw_to_direction_label(delta_rad: float) -> str:
    diff = ((float(delta_rad) + math.pi) % (2.0 * math.pi)) - math.pi
    if abs(diff) <= math.pi / 4.0:
        return "turn left" if diff >= 0.0 else "turn right"
    if diff > 0.0:
        return "turn left"
    return "turn back"


def _compute_heading_angles(path_xy: list[list[float]]) -> list[float]:
    headings = []
    for i in range(len(path_xy) - 1):
        dx = float(path_xy[i + 1][0]) - float(path_xy[i][0])
        dy = float(path_xy[i + 1][1]) - float(path_xy[i][1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        headings.append(float(math.atan2(dy, dx)))
    return headings


def _build_action_plan(case_id: str, pair_info: dict) -> dict:
    region_sequence = pair_info["region_sequence"]

    steps = []
    step_id = 1
    previous_heading = float(pair_info.get("initial_yaw", 0.0))
    for idx, region_name in enumerate(region_sequence[1:], start=1):
        if idx >= 2:
            prev_region = region_sequence[idx - 1]
            prev_center = pair_info["region_centers_xy"].get(prev_region)
            next_center = pair_info["region_centers_xy"].get(region_name)
            if prev_center is not None and next_center is not None:
                current_heading = float(math.atan2(next_center[1] - prev_center[1], next_center[0] - prev_center[0]))
                turn_answer = _yaw_to_direction_label(current_heading - previous_heading)
                steps.append(
                    {
                        "step_id": step_id,
                        "step_type": "turn_choice",
                        "text": "Choose the next turn direction.",
                        "placeholder": "[please fill in]",
                        "choices": ["turn back", "turn left", "turn right"],
                        "answer": turn_answer,
                        "target_region": region_name,
                    }
                )
                step_id += 1
                previous_heading = current_heading

        steps.append(
            {
                "step_id": step_id,
                "step_type": "go_to_region",
                "text": f"Go to the {region_name.replace('_', ' ')}.",
                "target_region": region_name,
            }
        )
        step_id += 1

    prompt_lines = [
        f"You are beginning in the {region_sequence[0].replace('_', ' ')} standing at the point shown in the image.",
        f"You want to navigate to the {region_sequence[-1].replace('_', ' ')}.",
        "You will perform the following actions.",
    ]
    if any(step["step_type"] == "turn_choice" for step in steps):
        prompt_lines.append("For each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'")
    for step in steps:
        prompt_lines.append(f"{step['step_id']}. {step.get('placeholder', '') + ' ' if step.get('placeholder') else ''}{step['text']}".strip())
    prompt_lines.append("You have reached the final destination.")

    return {
        "case_id": _make_case_id("navigation_actions", case_id),
        "question_type": "navigation_actions",
        "prompt": " ".join(prompt_lines),
        "steps": steps,
    }


def _build_region_plan(case_id: str, pair_info: dict) -> dict:
    region_sequence = pair_info["region_sequence"]
    steps = []
    for idx, region_name in enumerate(region_sequence[1:], start=1):
        options = []
        if idx < len(region_sequence):
            options.append(region_name)
        for candidate in region_sequence:
            if candidate != region_name and candidate not in options:
                options.append(candidate)
            if len(options) >= 3:
                break

        steps.append(
            {
                "step_id": idx,
                "step_type": "choose_region",
                "text": "Choose the next region to explore.",
                "placeholder": "[please fill in]",
                "options": options[:3],
                "answer": region_name,
            }
        )

    prompt_lines = [
        f"You are beginning in the {region_sequence[0].replace('_', ' ')}.",
        f"You want to navigate to the {region_sequence[-1].replace('_', ' ')}.",
        "At each step, choose the next region you should explore.",
    ]
    for step in steps:
        prompt_lines.append(f"{step['step_id']}. {step['text']} {step['placeholder']}")
    prompt_lines.append("You have reached the final destination.")

    return {
        "case_id": _make_case_id("navigation_regions", case_id),
        "question_type": "navigation_regions",
        "prompt": " ".join(prompt_lines),
        "steps": steps,
    }


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


def _capture_room_reference_views(region: RegionRecord, render_xy, floor_z: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    base_xyz = [float(render_xy[0]), float(render_xy[1]), float(floor_z)]
    views = {}
    for idx in range(DEFAULT_ROOM_VIEW_COUNT):
        yaw = 2.0 * math.pi * float(idx) / float(DEFAULT_ROOM_VIEW_COUNT)
        pose = _set_viewer_pose(base_xyz, yaw_rad=yaw)
        out_path = os.path.join(output_dir, f"{region.room_instance}__view_{idx:02d}.png")
        _capture(out_path)
        views[f"view_{idx:02d}"] = {
            "image_path": out_path,
            "camera_pose": pose,
        }
    return {
        "success": True,
        "room_instance": region.room_instance,
        "render_xy": [float(render_xy[0]), float(render_xy[1])],
        "views": views,
    }


def _initial_view_yaw(case: dict, render_xy) -> float:
    anchor = np.array([float(render_xy[0]), float(render_xy[1])], dtype=float)
    for waypoint in case.get("path_world", [])[1:]:
        target = np.array([float(waypoint[0]), float(waypoint[1])], dtype=float)
        if np.linalg.norm(target - anchor) > 0.5:
            return float(math.atan2(target[1] - anchor[1], target[0] - anchor[0]))
    goal_xy = case.get("goal_xy")
    if goal_xy is not None:
        return float(math.atan2(float(goal_xy[1]) - anchor[1], float(goal_xy[0]) - anchor[0]))
    return float(case.get("initial_yaw", 0.0))


def _capture_question_initial_view(case: dict, render_xy, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    floor_z = float(case["floor_z"])
    yaw = _initial_view_yaw(case, render_xy)
    base_xyz = [float(render_xy[0]), float(render_xy[1]), floor_z]
    pose = _set_viewer_pose(base_xyz, yaw_rad=yaw)
    out_path = os.path.join(output_dir, f"{case['case_id']}__initial.png")
    _capture(out_path)
    return {
        "success": True,
        "image_path": out_path,
        "camera_pose": pose,
        "render_xy": [float(render_xy[0]), float(render_xy[1])],
        "source_room": case["source_region"],
    }


def _generate_navigation_cases(
    scene,
    floor_idx: int,
    floor_z: float,
    regions: list[RegionRecord],
    max_cases: int,
    rng: random.Random,
    max_pair_evaluations: int | None = None,
) -> list[dict]:
    regions_by_name = {region.room_instance: region for region in regions}
    scored_pairs = []
    _debug(f"evaluating reachable room pairs from {len(regions)} regions")

    evaluated_pairs = 0
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if max_pair_evaluations is not None and evaluated_pairs >= int(max_pair_evaluations):
                break
            evaluated_pairs += 1
            src = regions[i]
            dst = regions[j]
            best_path = _find_best_region_path(scene, floor_idx, src, dst, regions_by_name)
            if best_path is None:
                continue
            euclidean = None
            src_center = _region_center_xy(src)
            dst_center = _region_center_xy(dst)
            if src_center is not None and dst_center is not None:
                euclidean = _distance_xy(src_center, dst_center)
            scored_pairs.append(
                {
                    "source_region": src.room_instance,
                    "target_region": dst.room_instance,
                    "source_room_type": src.room_type,
                    "target_room_type": dst.room_type,
                    "start_xy": best_path["start_xy"],
                    "goal_xy": best_path["goal_xy"],
                    "path_world": best_path["path_world"],
                    "path_distance": best_path["path_distance"],
                    "region_sequence": best_path["region_sequence"],
                    "euclidean_distance": euclidean,
                    "floor_idx": int(floor_idx),
                    "floor_z": float(floor_z),
                    "score": (
                        float(best_path["path_distance"]),
                        len(best_path["region_sequence"]),
                        0.0 if euclidean is None else float(euclidean),
                    ),
                }
            )
        if max_pair_evaluations is not None and evaluated_pairs >= int(max_pair_evaluations):
            break

    scored_pairs.sort(key=lambda item: item["score"], reverse=True)
    _debug(
        f"reachable far pairs found: {len(scored_pairs)} "
        f"evaluated_pairs={evaluated_pairs} "
        f"truncated={bool(max_pair_evaluations is not None and evaluated_pairs >= int(max_pair_evaluations))}"
    )

    selected = []
    used_pairs = set()
    for pair in scored_pairs:
        key = (pair["source_region"], pair["target_region"])
        if key in used_pairs:
            continue
        used_pairs.add(key)
        pair["case_id"] = _make_case_id("navigation_pair", pair["source_region"], pair["target_region"])
        headings = _compute_heading_angles(pair["path_world"])
        pair["initial_yaw"] = float(headings[0]) if headings else 0.0
        pair["region_centers_xy"] = {
            name: [float(center[0]), float(center[1])]
            for name, center in (
                (name, _region_center_xy(regions_by_name[name]))
                for name in pair["region_sequence"]
            )
            if center is not None
        }
        pair["action_question"] = _build_action_plan(pair["case_id"], pair)
        pair["region_question"] = _build_region_plan(pair["case_id"], pair)
        selected.append(pair)
        if len(selected) >= int(max_cases):
            break

    rng.shuffle(selected)
    selected.sort(key=lambda item: item["path_distance"], reverse=True)
    _debug(f"selected navigation cases: {len(selected)}")
    return selected


def _build_plan_render(case: dict) -> dict:
    initial_view = case.get("initial_view") or {}
    return {
        "image": initial_view.get("image_path"),
        "kind": "start_room_longest_ray_view",
        "source_room": case.get("source_region"),
        "initial_view": initial_view,
        "room_views": case.get("room_views"),
        "path_views": case.get("path_views"),
    }


def _steps_to_answer_string(steps: list[dict]) -> str:
    answers = [str(step.get("answer")) for step in steps if step.get("answer") is not None]
    return " | ".join(answers)


def _answerable_steps(steps: list[dict]) -> list[dict]:
    return [step for step in steps if step.get("answer") is not None]


def _build_plan_question_entry(case: dict, question: dict) -> dict:
    answerable_steps = _answerable_steps(question.get("steps", []))
    return {
        "task_type": str(question["question_type"]),
        "question": question["prompt"],
        "options": [list(step.get("choices", step.get("options", []))) for step in answerable_steps],
        "answer": _steps_to_answer_string(answerable_steps),
        "structured_answer": [step.get("answer") for step in answerable_steps],
        "case_id": case["case_id"],
        "question_case_id": question["case_id"],
        "source_region": case["source_region"],
        "target_region": case["target_region"],
        "source_room_type": case["source_room_type"],
        "target_room_type": case["target_room_type"],
        "path_distance": case["path_distance"],
        "region_sequence": case["region_sequence"],
        "path_world": case["path_world"],
        "start_xy": case["start_xy"],
        "goal_xy": case["goal_xy"],
        "floor_idx": case["floor_idx"],
        "floor_z": case["floor_z"],
        "initial_yaw": case["initial_yaw"],
        "steps": question.get("steps", []),
        "room_views": case.get("room_views", {}),
        "initial_view": case.get("initial_view", {}),
        "path_views": case.get("path_views", {}),
        "source_room_views": case.get("source_room_views", {}),
        "render": _build_plan_render(case),
        "question_payload": question,
    }


def _export_single_question_jsons(
    cases: list[dict],
    output_root: str,
    candidate_root: str,
    scene_metadata: dict,
) -> dict:
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(candidate_root, exist_ok=True)
    by_task_type = {
        "navigation_actions": [],
        "navigation_regions": [],
    }
    for case in cases:
        action_entry = _build_plan_question_entry(case, case["action_question"])
        if action_entry["structured_answer"]:
            by_task_type["navigation_actions"].append(action_entry)
        by_task_type["navigation_regions"].append(_build_plan_question_entry(case, case["region_question"]))

    by_family = {}
    for task_type, entries in by_task_type.items():
        family = _task_family_for_type(task_type)
        family_entries = by_family.setdefault(family, [])
        for entry in entries:
            family_entries.append((task_type, entry))

    written = {}
    counts_by_task_type = {}
    for task_family, entries in by_family.items():
        written_paths = []
        for q_idx, (task_type, entry) in enumerate(entries):
            out_path = _write_single_question_json(
                output_root=output_root,
                candidate_root=candidate_root,
                scene_metadata=scene_metadata,
                task_type=task_type,
                q_idx=q_idx,
                entry=entry,
            )
            written_paths.append(out_path)
            counts_by_task_type[task_type] = counts_by_task_type.get(task_type, 0) + 1
        written[task_family] = written_paths

    return {
        "enabled": True,
        "question_json_root": output_root,
        "candidate_render_root": candidate_root,
        "counts": {task_family: len(paths) for task_family, paths in written.items()},
        "counts_by_task_type": counts_by_task_type,
        "paths": written,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate navigation-planning metadata from runtime OmniGibson scene.")
    parser.add_argument("--scene", default="house_double_floor_upper", help="Scene model name, e.g. house_single_floor")
    parser.add_argument("--room", type=str, default=None, help="Optional room instance name; loads the full scene when omitted")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_cases", type=int, default=DEFAULT_CASE_LIMIT)
    parser.add_argument("--point_candidates", type=int, default=DEFAULT_POINT_CANDIDATES)
    parser.add_argument("--trav_map_basename", type=str, default=DEFAULT_TRAV_MAP_BASENAME)
    parser.add_argument("--region_expansion_ratio", type=float, default=DEFAULT_EXPANSION_RATIO)
    parser.add_argument("--region_expansion_min", type=float, default=DEFAULT_EXPANSION_MIN)
    parser.add_argument(
        "--plan_region_limit",
        type=int,
        default=DEFAULT_PLAN_REGION_LIMIT,
        help="Skip the planning task when the extracted room count is greater than this; <=0 disables the limit.",
    )
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    global PLAN_REGION_LIMIT
    PLAN_REGION_LIMIT = int(args.plan_region_limit)

    rng = random.Random(args.seed)
    room_dirname = args.room if args.room is not None else "full_scene"
    run_dir = os.path.join(args.output_root, args.scene, room_dirname)
    staging_root = os.path.join(run_dir, "_staging")
    room_view_root = os.path.join(staging_root, "plan_room_views")
    question_view_root = os.path.join(staging_root, "plan_initial_views")
    output_json = os.path.join(run_dir, DEFAULT_OUTPUT_NAME)
    question_json_root = os.path.join(run_dir, "cognitivemap_question_jsons")
    candidate_render_root = os.path.join(run_dir, "candidate_renders")
    os.makedirs(run_dir, exist_ok=True)

    config = _build_config(args)
    env = None
    try:
        _debug(f"starting environment creation for scene={args.scene} room={args.room}")
        env = og.Environment(configs=config)
        _set_viewer_camera_fov()
        scene = env.scene
        floor_idx = 0
        floor_z = float(scene.get_floor_height(int(floor_idx)))
        _debug("environment created")
        removed_doors = _remove_named_doors(scene)

        replaced_trav_map = _replace_trav_map_with_variant(scene, basename=str(args.trav_map_basename))
        _debug(f"trav-map replacement status: {replaced_trav_map}")
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
            raise RuntimeError("Not enough regions found to generate navigation planning questions.")
        if PLAN_REGION_LIMIT > 0 and len(regions) > PLAN_REGION_LIMIT:
            raise PlanTaskSkipped(
                "too_many_rooms",
                details={
                    "region_count": len(regions),
                    "plan_region_limit": PLAN_REGION_LIMIT,
                },
            )

        regions_by_name = {region.room_instance: region for region in regions}
        expanded_connectivity_graph = _build_expanded_connectivity_graph(regions)
        _debug("starting far connected-pair selection")
        cases = _generate_navigation_cases(
            scene=scene,
            floor_idx=floor_idx,
            floor_z=floor_z,
            regions=regions,
            max_cases=args.max_cases,
            rng=rng,
        )
        if not cases:
            raise RuntimeError("No reachable room pairs found on the no-door traversability map.")

        _debug("capturing room reference views and question initial views")
        room_view_records = {}
        for region in regions:
            render_xy, render_debug = _select_room_render_xy(scene, floor_idx, region)
            room_dir = os.path.join(room_view_root, region.room_instance)
            room_views = _capture_room_reference_views(region, render_xy, floor_z, room_dir)
            room_views["selection_debug"] = render_debug
            room_view_records[region.room_instance] = room_views

        for case in cases:
            source_room_views = room_view_records.get(case["source_region"])
            render_xy = (source_room_views or {}).get("render_xy", case["start_xy"])
            case["initial_view"] = _capture_question_initial_view(
                case,
                render_xy=render_xy,
                output_dir=question_view_root,
            )
            case["source_room_views"] = source_room_views
        question_json_summary = _export_single_question_jsons(
            cases=cases,
            output_root=question_json_root,
            candidate_root=candidate_render_root,
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
            "region_expansion_ratio": args.region_expansion_ratio,
            "region_expansion_min": args.region_expansion_min,
            "wall_related_object_count": len(wall_records),
            "structural_wall_count": len(structural_wall_bboxes),
            "region_count": len(regions),
            "regions": [region.to_json() for region in regions],
            "wall_related_objects": [wall.to_json() for wall in wall_records],
            "room_views": room_view_records,
            "expanded_connectivity_graph": expanded_connectivity_graph,
            "cases": cases,
            "question_json_summary": question_json_summary,
            "notes": [
                "Room bbox is expanded in world XY before region assignment because raw bbox can be inaccurate.",
                "Wall-related scene objects are collected first by matching 'wall' in object names.",
                "Structural wall AABBs are used as stop boundaries during bbox expansion.",
                "Each expansion direction stops as soon as the candidate bbox touches a structural wall in that direction.",
                "Navigation planning uses the no-door traversability map variant when available.",
                "Shortest-path queries ignore agent footprint so narrow but navigable routes are not over-pruned.",
                "Connected room pairs are selected from reachable shortest paths and then ranked by path distance.",
                "Each selected case includes an action-choice plan and a region-choice exploration plan.",
                "Each room stores eight reference views captured from a center-near and open vantage point.",
                "Each selected case stores one initial first-person image rendered from the source room.",
            ],
        }
        _debug(f"writing metadata json to {output_json}")
        _write_json(output_json, metadata)
        _debug("metadata json written successfully")

        print(
            json.dumps(
                {
                    "output_json": output_json,
                    "scene": args.scene,
                    "room": args.room,
                    "region_count": len(regions),
                    "case_count": len(cases),
                    "room_view_root": room_view_root,
                    "question_view_root": question_view_root,
                    "question_json_root": question_json_summary["question_json_root"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except PlanTaskSkipped as exc:
        _debug(f"plan task skipped: reason={exc.reason} details={json.dumps(exc.details, ensure_ascii=False)}")
        skip_payload = _write_skip_metadata(
            output_json=output_json,
            scene_metadata={
                "scene": args.scene,
                "room": args.room,
                "seed": args.seed,
                "trav_map_basename": args.trav_map_basename,
                "question_json_root": question_json_root,
            },
            reason=exc.reason,
            details=exc.details,
        )
        print(
            json.dumps(
                {
                    "output_json": output_json,
                    "scene": args.scene,
                    "room": args.room,
                    "seed": args.seed,
                    "skipped": True,
                    "skip_reason": skip_payload["skip_reason"],
                    "skip_details": skip_payload["skip_details"],
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
        if env is not None:
            og.clear()


if __name__ == "__main__":
    main()
