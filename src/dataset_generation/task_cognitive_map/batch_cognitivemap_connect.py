"""
Generate cognitive-map connectivity QA candidates from a runtime OmniGibson scene.

The script follows the same high-level structure as batch_counting.py:
build a config, launch one environment, query env.scene, and export
candidate metadata as JSON.
"""

from __future__ import annotations

import argparse
import cv2
import json
import math
import os
import random
import shutil
import signal
import struct
import sys
import time
import traceback
import zlib
from collections import deque
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

DEFAULT_PAIR_LIMIT = 8
DEFAULT_TRIPLE_LIMIT = 8
DEFAULT_POINT_CANDIDATES = 7
DEFAULT_DIAGNOSTIC_PAIR_LIMIT = 64
DEFAULT_CONNECT_REGION_LIMIT = 8
DEFAULT_SHORTEST_PATH_TIMEOUT_S = 60.0
DEFAULT_OUTPUT_NAME = "cognitivemap_connect_candidates.json"
DEFAULT_TRAV_MAP_BASENAME = "floor_trav_no_door"
DEFAULT_ROOM_VIEW_COUNT = 8
VIEWER_CAMERA_FOV_DEG = 100.0

CONNECT_REGION_LIMIT = DEFAULT_CONNECT_REGION_LIMIT
SHORTEST_PATH_TIMEOUT_S = DEFAULT_SHORTEST_PATH_TIMEOUT_S

TASK_FAMILY_BY_TYPE = {
    "pair_connectivity": "Topological Connectivity",
    "shortest_path_via_region": "Traversable Passage",
}


class ConnectTaskSkipped(RuntimeError):
    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = str(reason)
        self.details = details or {}


class _ShortestPathTimeout(RuntimeError):
    pass


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
            "center_xy": list(self.center_xy) if self.center_xy is not None else None,
            "candidate_points_xy": self.candidate_points_xy,
            "pixel_count": self.pixel_count,
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
    if "room_views" in path_keys and "gt_open_view" not in path_keys:
        return os.path.join(candidate_dir, "room_views")
    if path_keys.intersection({"path_views", "gt_views", "gt_open_view"}):
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
    print(f"[batch_cognitivemap_connect] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _debug(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[batch_cognitivemap_connect {now}] {message}", flush=True)


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
    n_floors = len(trav_map.floor_map)
    _debug(f"attempting trav-map replacement with basename={basename} across {n_floors} floor(s)")
    for floor in range(n_floors):
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
            _debug(f"trav-map floor={floor}: no variant found, keeping default map")
            continue

        resized = cv2.resize(src_img, (map_size, map_size))
        trav_tensor = th.tensor(resized)
        trav_tensor[trav_tensor < 255] = 0
        trav_map.floor_map[floor] = trav_tensor
        loaded += 1
        _debug(f"trav-map floor={floor}: loaded no-door variant from {src_path}")

    if loaded == 0:
        _debug(f"trav-map replace failed: no variant loaded for basename={basename}")
        return False

    _debug(f"trav-map replacement complete: {loaded}/{n_floors} floor map(s) replaced")
    return True


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


def _room_bbox_center_xy(bbox_xyxy):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)


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


def _get_shortest_path(scene, floor_idx: int, start_xy, goal_xy):
    timeout_s = float(SHORTEST_PATH_TIMEOUT_S)
    timer_enabled = timeout_s > 0.0 and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    previous_handler = None
    try:
        if timer_enabled:
            def _handle_timeout(_signum, _frame):
                raise _ShortestPathTimeout()

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        kwargs = {
            "floor": int(floor_idx),
            "source_world": th.tensor([float(start_xy[0]), float(start_xy[1])], dtype=th.float32),
            "target_world": th.tensor([float(goal_xy[0]), float(goal_xy[1])], dtype=th.float32),
            "entire_path": True,
        }
        path_world, distance = scene.get_shortest_path(**kwargs)
        return path_world, distance
    except _ShortestPathTimeout as exc:
        raise ConnectTaskSkipped(
            "shortest_path_timeout",
            details={
                "timeout_seconds": timeout_s,
                "floor_idx": int(floor_idx),
                "start_xy": [float(start_xy[0]), float(start_xy[1])],
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            },
        ) from exc
    except Exception:
        return None, None
    finally:
        if timer_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)


def _path_world_to_xy_list(path_world) -> list[list[float]]:
    if path_world is None:
        return []
    xy_list = []
    for waypoint in path_world:
        vals = _tensor_to_list(waypoint)
        if len(vals) >= 2:
            xy_list.append([float(vals[0]), float(vals[1])])
    return xy_list


def _path_to_room_sequence(seg, path_world) -> list[str]:
    if path_world is None:
        return []
    sequence = []
    last = None
    for waypoint in path_world:
        xy = _tensor_to_list(waypoint)[:2]
        room_instance = _query_room_instance_by_point(seg, xy)
        if room_instance is None or room_instance == last:
            continue
        sequence.append(room_instance)
        last = room_instance
    return sequence


def _trav_map_floor_image(scene, floor_idx: int = 0):
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


def _adjust_xy_to_nearest_free_point(scene, floor_idx: int, xy, room_pixels_rc: np.ndarray | None = None) -> tuple[list[float], dict]:
    trav_map, map_img = _trav_map_floor_image(scene, floor_idx=floor_idx)
    debug = {
        "original_xy": [float(xy[0]), float(xy[1])],
        "adjusted": False,
        "reason": "trav_map_unavailable",
        "original_map_rc": None,
        "adjusted_map_rc": None,
    }
    if trav_map is None or map_img is None:
        return [float(xy[0]), float(xy[1])], debug

    rc_arr = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    original_rc = _clip_map_rc(map_img, rc_arr)
    debug["original_map_rc"] = [int(original_rc[0]), int(original_rc[1])]

    if _is_free_rc(map_img, original_rc):
        debug["reason"] = "already_free"
        debug["adjusted_map_rc"] = [int(original_rc[0]), int(original_rc[1])]
        return [float(xy[0]), float(xy[1])], debug

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
        adjusted_xy = _map_rc_to_world_xy(trav_map, best_rc)
        debug["adjusted"] = True
        debug["reason"] = "moved_to_nearest_free_in_room"
        debug["adjusted_map_rc"] = [int(best_rc[0]), int(best_rc[1])]
        debug["adjusted_xy"] = adjusted_xy
        return adjusted_xy, debug

    all_free_rcs = np.argwhere(map_img > 0)
    best_rc = _nearest_rc_from_candidates(original_rc, all_free_rcs)
    if best_rc is not None:
        adjusted_xy = _map_rc_to_world_xy(trav_map, best_rc)
        debug["adjusted"] = True
        debug["reason"] = "moved_to_nearest_global_free"
        debug["adjusted_map_rc"] = [int(best_rc[0]), int(best_rc[1])]
        debug["adjusted_xy"] = adjusted_xy
        return adjusted_xy, debug

    debug["reason"] = "no_free_cell_found"
    return [float(xy[0]), float(xy[1])], debug


def _summarize_adjustment_debug(adjustment_logs: list[dict]) -> dict:
    summary = {
        "total_points": len(adjustment_logs),
        "adjusted_points": 0,
        "already_free_points": 0,
        "moved_to_nearest_free_in_room": 0,
        "moved_to_nearest_global_free": 0,
        "trav_map_unavailable": 0,
        "no_free_cell_found": 0,
    }
    for entry in adjustment_logs:
        reason = entry.get("reason")
        if entry.get("adjusted"):
            summary["adjusted_points"] += 1
        if reason in summary:
            summary[reason] += 1
    return summary


def _build_region_records(scene, seed: int, point_candidates: int) -> tuple[list[RegionRecord], dict]:
    seg = _segmap_get(scene)
    if seg is None or not hasattr(seg, "room_ins_id_to_ins_name"):
        raise RuntimeError("scene.seg_map.room_ins_id_to_ins_name not found")

    room_records = []
    adjustment_logs = []
    room_ids = sorted(int(room_id) for room_id in seg.room_ins_id_to_ins_name.keys())
    _debug(f"building region records for {len(room_ids)} room instances")
    room_map = seg.room_ins_map
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()
    room_map_np = np.array(room_map, dtype=np.int64)

    for room_id in room_ids:
        if len(room_records) > 0 and len(room_records) % 25 == 0:
            _debug(f"processed {len(room_records)} region records")
        room_instance = str(seg.room_ins_id_to_ins_name[room_id])
        room_type = room_instance.rsplit("_", 1)[0] if "_" in room_instance else room_instance
        pixels = np.argwhere(room_map_np == int(room_id))
        if pixels.shape[0] == 0:
            continue

        bbox_info = _segmap_room_bbox_from_maps(scene, room_id)
        bbox_world_xy = bbox_info["bbox_world_xy"]
        center_xy = None
        candidate_points = []
        if bbox_world_xy is not None:
            raw_center_xy = _room_bbox_center_xy(bbox_world_xy).tolist()
            adjusted_center_xy, adjust_info = _adjust_xy_to_nearest_free_point(
                scene=scene,
                floor_idx=0,
                xy=raw_center_xy,
                room_pixels_rc=pixels,
            )
            adjust_info.update({"room_instance": room_instance, "point_type": "center"})
            adjustment_logs.append(adjust_info)
            center_xy = tuple(float(x) for x in adjusted_center_xy)
            candidate_points = [[float(center_xy[0]), float(center_xy[1])]]

        room_records.append(
            RegionRecord(
                room_id=room_id,
                room_instance=room_instance,
                room_type=room_type,
                bbox_map_rc=bbox_info["bbox_map_rc"],
                bbox_world_xy=bbox_world_xy,
                center_xy=center_xy,
                candidate_points_xy=candidate_points,
                pixel_count=int(bbox_info["pixel_count"]),
            )
        )

    room_records.sort(key=lambda record: (record.room_type, record.room_instance))
    _debug(f"finished region extraction: {len(room_records)} valid regions")
    adjustment_summary = _summarize_adjustment_debug(adjustment_logs)
    _debug(f"point adjustment summary: {json.dumps(adjustment_summary, ensure_ascii=False)}")
    return room_records, {"logs": adjustment_logs, "summary": adjustment_summary}


def _build_adjacency_graph(scene, regions: list[RegionRecord]) -> dict[str, list[str]]:
    seg = _segmap_get(scene)
    _debug(f"building adjacency graph from seg map for {len(regions)} regions")
    room_map = seg.room_ins_map
    if hasattr(room_map, "detach"):
        room_map = room_map.detach()
    if getattr(room_map, "device", None) is not None and room_map.device.type != "cpu":
        room_map = room_map.cpu()
    room_map_np = np.array(room_map, dtype=np.int64)

    valid_ids = {region.room_id for region in regions}
    id_to_name = {region.room_id: region.room_instance for region in regions}
    adjacency: dict[str, set[str]] = {region.room_instance: set() for region in regions}

    def add_edge(a: int, b: int):
        if a == b or a not in valid_ids or b not in valid_ids:
            return
        name_a = id_to_name[a]
        name_b = id_to_name[b]
        adjacency[name_a].add(name_b)
        adjacency[name_b].add(name_a)

    if room_map_np.shape[0] > 1:
        upper = room_map_np[:-1, :]
        lower = room_map_np[1:, :]
        diff = (upper != lower) & (upper > 0) & (lower > 0)
        rows, cols = np.where(diff)
        for row, col in zip(rows.tolist(), cols.tolist()):
            add_edge(int(upper[row, col]), int(lower[row, col]))
    if room_map_np.shape[1] > 1:
        left = room_map_np[:, :-1]
        right = room_map_np[:, 1:]
        diff = (left != right) & (left > 0) & (right > 0)
        rows, cols = np.where(diff)
        for row, col in zip(rows.tolist(), cols.tolist()):
            add_edge(int(left[row, col]), int(right[row, col]))

    graph = {name: sorted(neighbors) for name, neighbors in adjacency.items()}
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    _debug(f"adjacency graph ready: {len(graph)} nodes, {edge_count} undirected edges")
    return graph


def _shortest_room_path(
    scene,
    floor_idx: int,
    regions_by_name: dict[str, RegionRecord],
    start: str,
    goal: str,
    blocked: set[str] | None = None,
    cache: dict | None = None,
) -> list[str] | None:
    info = _find_best_path_info(
        scene=scene,
        floor_idx=floor_idx,
        regions_by_name=regions_by_name,
        start=start,
        goal=goal,
        blocked=blocked,
        cache=cache,
    )
    return None if info is None else info["room_sequence"]


def _find_best_path_info(
    scene,
    floor_idx: int,
    regions_by_name: dict[str, RegionRecord],
    start: str,
    goal: str,
    blocked: set[str] | None = None,
    cache: dict | None = None,
) -> dict | None:
    blocked = set(blocked or set())
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return [start]
    if start not in regions_by_name or goal not in regions_by_name:
        return None

    cache_key = None
    if cache is not None:
        cache_key = ("path_info", int(floor_idx), str(start), str(goal), tuple(sorted(blocked)))
        if cache_key in cache:
            return cache[cache_key]

    seg = _segmap_get(scene)
    src = regions_by_name[start]
    dst = regions_by_name[goal]
    best_path = None
    best_score = None

    for src_xy in src.candidate_points_xy:
        for dst_xy in dst.candidate_points_xy:
            path_world, distance = _get_shortest_path(scene, floor_idx, src_xy, dst_xy)
            if path_world is None or len(path_world) == 0:
                continue
            room_sequence = _path_to_room_sequence(seg, path_world)
            if not room_sequence:
                continue
            if room_sequence[0] != start or room_sequence[-1] != goal:
                continue
            if blocked.intersection(room_sequence):
                continue

            score = (
                float(distance) if distance is not None else float(len(path_world)),
                len(room_sequence),
                len(path_world),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_path = {
                    "source_region": start,
                    "target_region": goal,
                    "blocked_regions": sorted(blocked),
                    "source_xy": [float(src_xy[0]), float(src_xy[1])],
                    "target_xy": [float(dst_xy[0]), float(dst_xy[1])],
                    "room_sequence": list(room_sequence),
                    "path_world_xy": _path_world_to_xy_list(path_world),
                    "path_length_steps": int(len(path_world)),
                    "path_distance": None if distance is None else float(distance),
                }

    if cache is not None and cache_key is not None:
        cache[cache_key] = best_path
    return best_path


def _all_reachable_nodes(graph: dict[str, list[str]], start: str, blocked: set[str] | None = None) -> set[str]:
    blocked = blocked or set()
    if start in blocked:
        return set()
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return visited


def _graph_shortest_path(graph: dict[str, list[str]], start: str, goal: str, blocked: set[str] | None = None) -> list[str] | None:
    blocked = set(blocked or set())
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt in blocked or nxt in visited:
                continue
            next_path = path + [nxt]
            if nxt == goal:
                return next_path
            visited.add(nxt)
            queue.append((nxt, next_path))
    return None


def _compute_articulation_like_rooms(graph: dict[str, list[str]]) -> list[str]:
    names = sorted(graph)
    articulation = []
    for blocked in names:
        remaining = [name for name in names if name != blocked]
        if len(remaining) <= 1:
            continue
        start = remaining[0]
        reached = _all_reachable_nodes(graph, start, blocked={blocked})
        if any(name not in reached for name in remaining):
            articulation.append(blocked)
    return articulation


def _log_connectivity_diagnostics(
    graph: dict[str, list[str]],
    scene,
    floor_idx: int,
    regions_by_name: dict[str, RegionRecord],
    path_cache: dict | None = None,
    max_pairs: int | None = None,
) -> None:
    names = sorted(regions_by_name)
    _debug(f"adjacency graph detail: {json.dumps(graph, ensure_ascii=False)}")
    articulation = _compute_articulation_like_rooms(graph)
    _debug(f"articulation-like rooms in adjacency graph: {articulation}")

    total_pairs = len(names) * (len(names) - 1) // 2
    if max_pairs is not None and int(max_pairs) > 0:
        effective_total = min(int(max_pairs), total_pairs)
    else:
        effective_total = total_pairs
    progress_interval = max(1, min(10, effective_total // 10 if effective_total > 0 else 1))
    _debug(
        f"connectivity diagnostics pair budget: total_pairs={total_pairs} "
        f"effective_pairs={effective_total}"
    )

    direct_pairs = 0
    indirect_pairs = 0
    unreachable_pairs = 0
    pair_lines = []
    checked_pairs = 0
    truncated = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if checked_pairs >= effective_total:
                truncated = effective_total < total_pairs
                break
            checked_pairs += 1
            if checked_pairs == 1 or checked_pairs % progress_interval == 0 or checked_pairs == effective_total:
                _debug(f"connectivity diagnostics progress: {checked_pairs}/{effective_total}")
            src_name, dst_name = names[i], names[j]
            graph_path = _graph_shortest_path(graph, src_name, dst_name)
            scene_path = _shortest_room_path(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                start=src_name,
                goal=dst_name,
                cache=path_cache,
            )
            if scene_path is None:
                unreachable_pairs += 1
            elif len(scene_path) <= 2:
                direct_pairs += 1
            else:
                indirect_pairs += 1
            pair_lines.append(
                {
                    "pair": [src_name, dst_name],
                    "adjacent_in_graph": bool(dst_name in graph.get(src_name, [])),
                    "graph_path": graph_path,
                    "scene_path": scene_path,
                }
            )
        if checked_pairs >= effective_total:
            break
    _debug(
        "pair path diagnostics: "
        f"checked={checked_pairs} "
        f"truncated={truncated} "
        f"direct_or_one_hop={direct_pairs} "
        f"indirect={indirect_pairs} "
        f"unreachable={unreachable_pairs}"
    )
    for entry in pair_lines:
        _debug(
            "pair detail: "
            f"{entry['pair'][0]} -> {entry['pair'][1]} "
            f"adjacent={entry['adjacent_in_graph']} "
            f"graph_path={entry['graph_path']} "
            f"scene_path={entry['scene_path']}"
        )


def _find_path_evidence(scene, floor_idx: int, src: RegionRecord, dst: RegionRecord):
    seg = _segmap_get(scene)
    attempts = []
    for src_xy in src.candidate_points_xy:
        for dst_xy in dst.candidate_points_xy:
            path_world, distance = _get_shortest_path(scene, floor_idx, src_xy, dst_xy)
            reachable = path_world is not None and len(path_world) > 0
            room_sequence = _path_to_room_sequence(seg, path_world) if reachable else []
            attempt = {
                "source_xy": [float(src_xy[0]), float(src_xy[1])],
                "target_xy": [float(dst_xy[0]), float(dst_xy[1])],
                "reachable": bool(reachable),
                "path_length_steps": int(len(path_world)) if reachable else 0,
                "path_distance": None if distance is None else float(distance),
                "room_sequence": room_sequence,
                "path_world_xy": _path_world_to_xy_list(path_world) if reachable else [],
            }
            attempts.append(attempt)
            if reachable:
                return attempt, attempts
    return None, attempts


def _world_to_plot_xy(trav_map, xy) -> tuple[float, float]:
    rc = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).cpu().numpy()
    return float(rc[1]), float(rc[0])


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


def _attach_room_photos(
    scene,
    floor_idx: int,
    floor_z: float,
    cases: dict[str, list[dict]],
    regions: list[RegionRecord],
    photo_root: str,
) -> tuple[dict, dict[str, dict]]:
    os.makedirs(photo_root, exist_ok=True)
    summary = {
        "enabled": True,
        "photo_root": photo_root,
        "rooms_requested": len(regions),
        "rooms_photographed": 0,
    }
    room_view_records = {}
    _debug(f"capturing {DEFAULT_ROOM_VIEW_COUNT} room views for {len(regions)} rooms into {photo_root}")

    for index, region in enumerate(regions, start=1):
        _debug(f"room view capture: room {index}/{len(regions)} {region.room_instance}")
        render_xy, render_debug = _select_room_render_xy(scene, floor_idx, region)
        room_dir = os.path.join(photo_root, region.room_instance)
        room_views = _capture_room_reference_views(region, render_xy, floor_z, room_dir)
        room_views["selection_debug"] = render_debug
        room_view_records[region.room_instance] = room_views
        if room_views.get("success"):
            summary["rooms_photographed"] += 1

    for case_group in cases.values():
        for case in case_group:
            case["room_views"] = {
                region_name: room_view_records[region_name]
                for region_name in case.get("required_regions", [])
                if region_name in room_view_records
            }

    _debug(
        "room view capture complete: "
        f"rooms_photographed={summary['rooms_photographed']} "
        f"rooms_requested={summary['rooms_requested']}"
    )
    return summary, room_view_records


def _region_path_to_xy_path(region_path: list[str] | None, regions_by_name: dict[str, RegionRecord]) -> list[tuple[float, float]]:
    if not region_path:
        return []
    xy_path = []
    for name in region_path:
        center_xy = _region_center_xy(regions_by_name[name])
        if center_xy is not None:
            xy_path.append(center_xy)
    return xy_path


def _draw_xy_polyline(trav_map, xy_path, color: str, label: str, linewidth: float = 2.5, linestyle: str = "-"):
    if len(xy_path) < 2:
        return
    plot_pts = np.array([_world_to_plot_xy(trav_map, xy) for xy in xy_path], dtype=float)
    plt.plot(plot_pts[:, 0], plot_pts[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _draw_path_info(trav_map, path_info: dict | None, color: str, label: str, linewidth: float = 2.5, linestyle: str = "-"):
    if not path_info:
        return
    path_world_xy = path_info.get("path_world_xy") or []
    if len(path_world_xy) >= 2:
        _draw_xy_polyline(trav_map, path_world_xy, color=color, label=label, linewidth=linewidth, linestyle=linestyle)


def _draw_attempt_points(trav_map, attempts: list[dict], max_points: int = 24):
    for idx, attempt in enumerate((attempts or [])[:max_points]):
        src_xy = attempt.get("source_xy")
        dst_xy = attempt.get("target_xy")
        if src_xy:
            plot_x, plot_y = _world_to_plot_xy(trav_map, src_xy)
            plt.scatter([plot_x], [plot_y], c="lime", s=18, marker="o", alpha=0.45, zorder=4)
        if dst_xy:
            plot_x, plot_y = _world_to_plot_xy(trav_map, dst_xy)
            plt.scatter([plot_x], [plot_y], c="red", s=18, marker="x", alpha=0.45, zorder=4)
        path_world_xy = attempt.get("path_world_xy") or []
        if len(path_world_xy) >= 2:
            color = "gold" if attempt.get("reachable") else "magenta"
            label = "attempt_path" if idx == 0 else "_nolegend_"
            _draw_xy_polyline(trav_map, path_world_xy, color=color, label=label, linewidth=1.1, linestyle=":")


def _render_case_visualizations(scene, regions_by_name: dict[str, RegionRecord], cases: dict[str, list[dict]], viz_root: str):
    trav_map = getattr(scene, "_trav_map", None)
    if trav_map is None or getattr(trav_map, "floor_map", None) is None:
        return {"enabled": False, "reason": "scene._trav_map unavailable"}

    floor_idx = 0
    if floor_idx < 0 or floor_idx >= len(trav_map.floor_map):
        return {"enabled": False, "reason": f"invalid floor_idx={floor_idx}"}

    os.makedirs(viz_root, exist_ok=True)
    map_img = trav_map.floor_map[floor_idx].cpu().numpy()
    total_cases = sum(len(case_group) for case_group in cases.values())
    _debug(f"rendering case visualizations for {total_cases} cases into {viz_root}")

    case_counter = 0
    for case_group in cases.values():
        for case in case_group:
            case_counter += 1
            if case_counter % 10 == 0 or case_counter == 1:
                _debug(f"visualization progress: case {case_counter}/{total_cases} {case['case_id']}")
            fig = plt.figure(figsize=(7.0, 7.0))
            plt.imshow(map_img, cmap="gray", vmin=0, vmax=255)

            highlighted = set(case.get("required_regions", []))
            blocked_name = case.get("via_region")
            for region_name, region in regions_by_name.items():
                if region.bbox_map_rc is not None:
                    rmin, cmin, rmax, cmax = region.bbox_map_rc
                    width = max(1.0, float(cmax - cmin))
                    height = max(1.0, float(rmax - rmin))
                    edge_color = "deepskyblue" if region_name in highlighted else "white"
                    if region_name == blocked_name:
                        edge_color = "orange"
                    rect = plt.Rectangle((cmin, rmin), width, height, fill=False, edgecolor=edge_color, linewidth=1.5)
                    plt.gca().add_patch(rect)
                center_xy = _region_center_xy(region)
                if center_xy is None:
                    continue
                plot_x, plot_y = _world_to_plot_xy(trav_map, center_xy)
                point_color = "deepskyblue" if region_name in highlighted else "white"
                if region_name == case.get("source_region"):
                    point_color = "lime"
                elif region_name == case.get("target_region"):
                    point_color = "red"
                elif region_name == blocked_name:
                    point_color = "orange"
                plt.scatter([plot_x], [plot_y], c=point_color, s=28, zorder=3)
                plt.text(plot_x + 2.0, plot_y + 2.0, region_name, fontsize=7, color=point_color)

            if case["case_type"] == "pair_connectivity":
                successful_path = case.get("verification", {}).get("successful_path")
                if successful_path and successful_path.get("reachable"):
                    src_xy = successful_path["source_xy"]
                    dst_xy = successful_path["target_xy"]
                    src_plot = _world_to_plot_xy(trav_map, src_xy)
                    dst_plot = _world_to_plot_xy(trav_map, dst_xy)
                    plt.scatter([src_plot[0]], [src_plot[1]], c="lime", s=48, marker="o", zorder=4)
                    plt.scatter([dst_plot[0]], [dst_plot[1]], c="red", s=52, marker="x", zorder=4)
                _draw_path_info(
                    trav_map,
                    successful_path,
                    color="yellow",
                    label="reachable_path",
                    linewidth=2.8,
                )
                if not (successful_path and (successful_path.get("path_world_xy") or [])):
                    path_regions = successful_path.get("room_sequence") if successful_path else case.get("graph_shortest_region_path")
                    _draw_xy_polyline(
                        trav_map,
                        _region_path_to_xy_path(path_regions, regions_by_name),
                        color="yellow",
                        label="reachable_path",
                        linewidth=2.8,
                    )
            else:
                base_path_info = case.get("verification", {}).get("shortest_path_info") or case.get("verification", {}).get("base_path_info")
                _draw_path_info(
                    trav_map,
                    base_path_info,
                    color="yellow",
                    label="shortest_path",
                    linewidth=2.8,
                )
                if not (base_path_info and (base_path_info.get("path_world_xy") or [])):
                    base_region_path = case.get("verification", {}).get("shortest_region_path") or case.get("verification", {}).get("base_region_path")
                    _draw_xy_polyline(
                        trav_map,
                        _region_path_to_xy_path(base_region_path, regions_by_name),
                        color="yellow",
                        label="shortest_path",
                        linewidth=2.8,
                    )

            answer_text = "yes" if case.get("answer_bool") else "no"
            plt.title(f"{case['case_type']} | {answer_text}\n{case['question']}", fontsize=10)
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()

            out_path = os.path.join(viz_root, f"{case['case_id']}.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            case["visualization"] = {"map_path": out_path}

    _debug("case visualization complete")
    return {"enabled": True, "viz_root": viz_root}


def _generate_pair_connectivity_cases(
    scene,
    floor_idx: int,
    regions_by_name: dict[str, RegionRecord],
    max_cases: int,
    rng: random.Random,
    path_cache: dict | None = None,
    diagnostics: list[dict] | None = None,
    max_pair_evaluations: int | None = None,
) -> list[dict]:
    names = sorted(regions_by_name)
    candidates = []
    evaluated = 0
    _debug(f"generating pair_connectivity cases from {len(names)} regions")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if max_pair_evaluations is not None and evaluated >= int(max_pair_evaluations):
                rng.shuffle(candidates)
                _debug(
                    f"pair_connectivity complete: evaluated={evaluated} kept={len(candidates)} "
                    f"returning={min(len(candidates), max_cases)} truncated=True"
                )
                return candidates[:max_cases]
            evaluated += 1
            if evaluated % 20 == 0:
                _debug(f"pair_connectivity progress: evaluated={evaluated} kept={len(candidates)}")
            src_name, dst_name = names[i], names[j]
            graph_path = _shortest_room_path(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                start=src_name,
                goal=dst_name,
                cache=path_cache,
            )
            evidence, attempts = _find_path_evidence(scene, floor_idx, regions_by_name[src_name], regions_by_name[dst_name])
            answer = bool(graph_path is not None)
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "diag_type": "pair_connectivity_analysis",
                        "source_region": src_name,
                        "target_region": dst_name,
                        "required_regions": [src_name, dst_name],
                        "reason": "scene shortest path found" if evidence is not None else "scene shortest path not found",
                        "answer_bool": answer,
                        "graph_shortest_region_path": graph_path,
                        "shortest_path_info": evidence,
                        "shortest_path_attempts": attempts,
                    }
                )
            if answer != bool(evidence is not None):
                continue
            candidates.append(
                {
                    "case_id": _make_case_id("pair_connectivity", src_name, dst_name),
                    "case_type": "pair_connectivity",
                    "question": f"Are {src_name} and {dst_name} connected?",
                    "answer": "Yes" if answer else "No",
                    "answer_bool": answer,
                    "source_region": src_name,
                    "target_region": dst_name,
                    "required_regions": [src_name, dst_name],
                    "source_room_type": regions_by_name[src_name].room_type,
                    "target_room_type": regions_by_name[dst_name].room_type,
                    "graph_shortest_region_path": graph_path,
                    "directly_adjacent": bool(graph_path is not None and len(graph_path) == 2),
                    "verification": {
                        "method": "scene.get_shortest_path + room_sequence_projection",
                        "graph_reachable": answer,
                        "shortest_path_attempts": attempts,
                        "successful_path": evidence,
                    },
                }
            )
    rng.shuffle(candidates)
    _debug(
        f"pair_connectivity complete: evaluated={evaluated} kept={len(candidates)} "
        f"returning={min(len(candidates), max_cases)} truncated=False"
    )
    return candidates[:max_cases]


def _generate_shortest_path_via_cases(
    scene,
    floor_idx: int,
    regions_by_name: dict[str, RegionRecord],
    max_cases: int,
    rng: random.Random,
    path_cache: dict | None = None,
    diagnostics: list[dict] | None = None,
    max_pair_evaluations: int | None = None,
) -> list[dict]:
    names = sorted(regions_by_name)
    candidates = []
    evaluated = 0
    _debug(f"generating shortest_path_via_region cases from {len(names)} regions")
    for src_name in names:
        for dst_name in names:
            if src_name >= dst_name:
                continue
            if max_pair_evaluations is not None and evaluated >= int(max_pair_evaluations):
                break
            evaluated += 1
            if evaluated % 20 == 0:
                _debug(f"shortest_path_via_region progress: evaluated_pairs={evaluated} kept={len(candidates)}")
            shortest_path_info = _find_best_path_info(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                start=src_name,
                goal=dst_name,
                cache=path_cache,
            )
            shortest_path = None if shortest_path_info is None else shortest_path_info["room_sequence"]
            if shortest_path is None:
                evidence, attempts = _find_path_evidence(scene, floor_idx, regions_by_name[src_name], regions_by_name[dst_name])
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "diag_type": "shortest_path_via_region_rejected_no_path",
                            "source_region": src_name,
                            "target_region": dst_name,
                            "required_regions": [src_name, dst_name],
                            "reason": "no valid shortest path survived room-sequence validation for this room pair",
                            "shortest_path_info": evidence,
                            "shortest_path_attempts": attempts,
                        }
                    )
                continue
            for via_name in names:
                if via_name in {src_name, dst_name}:
                    continue
                answer = via_name in shortest_path[1:-1]
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "diag_type": "shortest_path_via_region_analysis",
                            "source_region": src_name,
                            "target_region": dst_name,
                            "via_region": via_name,
                            "shortest_path_info": shortest_path_info,
                            "answer_bool": answer,
                            "reason": "region appears on the sampled shortest room sequence" if answer else "region does not appear on the sampled shortest room sequence",
                            "required_regions": [src_name, dst_name, via_name],
                        }
                    )
                candidates.append(
                    {
                        "case_id": _make_case_id("shortest_path_via_region", src_name, dst_name, via_name),
                        "case_type": "shortest_path_via_region",
                        "question": f"On the shortest path from {src_name} to {dst_name}, do you need to pass through {via_name}?",
                        "answer": "Yes" if answer else "No",
                        "answer_bool": answer,
                        "source_region": src_name,
                        "target_region": dst_name,
                        "via_region": via_name,
                        "required_regions": [src_name, dst_name, via_name],
                        "source_room_type": regions_by_name[src_name].room_type,
                        "target_room_type": regions_by_name[dst_name].room_type,
                        "via_room_type": regions_by_name[via_name].room_type,
                        "verification": {
                            "method": "scene.get_shortest_path + room_sequence_projection",
                            "shortest_region_path": shortest_path,
                            "shortest_path_info": shortest_path_info,
                            "via_region_on_shortest_path": answer,
                        },
                    }
                )
        if max_pair_evaluations is not None and evaluated >= int(max_pair_evaluations):
            break
    positives = [case for case in candidates if case["answer_bool"]]
    negatives = [case for case in candidates if not case["answer_bool"]]
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
    _debug(
        "shortest_path_via_region complete: "
        f"evaluated_pairs={evaluated} candidates={len(candidates)} "
        f"positives={len([case for case in candidates if case['answer_bool']])} "
        f"negatives={len([case for case in candidates if not case['answer_bool']])} "
        f"returning={len(mixed[:max_cases])} "
        f"truncated={bool(max_pair_evaluations is not None and evaluated >= int(max_pair_evaluations))}"
    )
    return mixed[:max_cases]


def _connect_options_for_case(case: dict) -> list[str]:
    return ["Yes", "No"]


def _first_room_view_image(case: dict) -> str | None:
    initial_image = (case.get("initial_view") or {}).get("image_path")
    if initial_image:
        return initial_image
    room_views = case.get("room_views") or {}
    for region_name in case.get("required_regions", []):
        render_info = room_views.get(region_name) or {}
        views = render_info.get("views") or {}
        ordered_keys = sorted(views.keys())
        for view_key in ordered_keys:
            image_path = (views.get(view_key) or {}).get("image_path")
            if image_path:
                return image_path
    return None


def _build_single_question_entry(case: dict) -> dict:
    task_type = str(case["case_type"])
    entry = dict(case)
    map_path = (case.get("visualization") or {}).get("map_path")
    reference_image = _first_room_view_image(case)
    render = {
        "image": reference_image,
        "kind": "start_room_longest_ray_view",
        "topdown_map": map_path,
        "initial_view": case.get("initial_view"),
        "room_views": case.get("room_views"),
        "path_views": case.get("path_views"),
    }
    entry.update(
        {
            "task_type": task_type,
            "options": _connect_options_for_case(case),
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
    parser = argparse.ArgumentParser(description="Generate connectivity QA metadata from runtime OmniGibson scene.")
    parser.add_argument("--scene", default="house_double_floor_upper", help="Scene model name, e.g. house_single_floor")
    parser.add_argument("--room", type=str, default=None, help="Optional room instance name; loads the full scene when omitted")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_pair_cases", type=int, default=DEFAULT_PAIR_LIMIT)
    parser.add_argument("--max_triple_cases", type=int, default=DEFAULT_TRIPLE_LIMIT)
    parser.add_argument("--point_candidates", type=int, default=DEFAULT_POINT_CANDIDATES)
    parser.add_argument(
        "--max_diagnostic_pairs",
        type=int,
        default=DEFAULT_DIAGNOSTIC_PAIR_LIMIT,
        help="Maximum number of room-pair shortest-path diagnostics to evaluate; <=0 means no limit.",
    )
    parser.add_argument(
        "--connect_region_limit",
        type=int,
        default=DEFAULT_CONNECT_REGION_LIMIT,
        help="Skip the connect task when the extracted room count is greater than this; <=0 disables the limit.",
    )
    parser.add_argument(
        "--shortest_path_timeout_s",
        type=float,
        default=DEFAULT_SHORTEST_PATH_TIMEOUT_S,
        help="Skip the connect task if any shortest-path query takes longer than this many seconds; <=0 disables the timeout.",
    )
    parser.add_argument("--trav_map_basename", type=str, default=DEFAULT_TRAV_MAP_BASENAME)
    parser.add_argument("--output_root", type=str, default="renders_connect")
    args = parser.parse_args()

    global CONNECT_REGION_LIMIT, SHORTEST_PATH_TIMEOUT_S
    CONNECT_REGION_LIMIT = int(args.connect_region_limit)
    SHORTEST_PATH_TIMEOUT_S = float(args.shortest_path_timeout_s)

    rng = random.Random(args.seed)
    room_dirname = args.room if args.room is not None else "full_scene"
    run_dir = os.path.join(args.output_root, args.scene, room_dirname)
    os.makedirs(run_dir, exist_ok=True)
    output_json = os.path.join(run_dir, DEFAULT_OUTPUT_NAME)
    staging_root = os.path.join(run_dir, "_staging")
    photo_root = os.path.join(staging_root, "connect_room_photos")
    question_json_root = os.path.join(run_dir, "cognitivemap_question_jsons")
    candidate_render_root = os.path.join(run_dir, "candidate_renders")
    map_visualization_root = os.path.join(run_dir, "map_visualization")

    config = _build_config(args)
    try:
        _debug(f"starting environment creation for scene={args.scene} room={args.room}")
        env = og.Environment(configs=config)
        _debug("environment created")
        _set_viewer_camera_fov()
        scene = env.scene
        _debug("scene handle acquired")
        removed_doors = _remove_named_doors(scene)
        replaced_trav_map = _replace_trav_map_with_variant(scene, basename=str(args.trav_map_basename))
        _debug(f"trav-map replacement status: {replaced_trav_map}")
        floor_z = float(scene.get_floor_height(0))
        floor_idx = 0

        _debug("starting region extraction")
        regions, point_adjustment_debug = _build_region_records(
            scene=scene,
            seed=args.seed,
            point_candidates=args.point_candidates,
        )
        if len(regions) < 2:
            raise RuntimeError("Not enough regions found to generate connectivity questions.")
        if CONNECT_REGION_LIMIT > 0 and len(regions) > CONNECT_REGION_LIMIT:
            raise ConnectTaskSkipped(
                "too_many_rooms",
                details={
                    "region_count": len(regions),
                    "connect_region_limit": CONNECT_REGION_LIMIT,
                },
            )

        regions_by_name = {region.room_instance: region for region in regions}
        _debug("starting adjacency graph build")
        adjacency_graph = _build_adjacency_graph(scene=scene, regions=regions)
        path_cache = {}
        _debug("computing connected components from adjacency graph")
        connected_components = []
        seen = set()
        for name in sorted(regions_by_name):
            if name in seen:
                continue
            comp = sorted(_all_reachable_nodes(adjacency_graph, name))
            seen.update(comp)
            connected_components.append(comp)
        _debug(f"connected components ready: {len(connected_components)}")
        _debug("starting connectivity diagnostics")
        _log_connectivity_diagnostics(
            graph=adjacency_graph,
            scene=scene,
            floor_idx=floor_idx,
            regions_by_name=regions_by_name,
            path_cache=path_cache,
            max_pairs=args.max_diagnostic_pairs,
        )

        _debug("starting case generation")
        pair_connectivity_diagnostics = []
        shortest_path_via_diagnostics = []
        cases = {
            "pair_connectivity": _generate_pair_connectivity_cases(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                max_cases=args.max_pair_cases,
                rng=rng,
                path_cache=path_cache,
                diagnostics=pair_connectivity_diagnostics,
            ),
            "shortest_path_via_region": _generate_shortest_path_via_cases(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                max_cases=args.max_triple_cases,
                rng=rng,
                path_cache=path_cache,
                diagnostics=shortest_path_via_diagnostics,
            ),
        }
        _debug(
            "case generation finished: "
            f"pair={len(cases['pair_connectivity'])} "
            f"shortest_path_via={len(cases['shortest_path_via_region'])}"
        )
        visualization_summary = _render_case_visualizations(
            scene=scene,
            regions_by_name=regions_by_name,
            cases=cases,
            viz_root=map_visualization_root,
        )
        _debug("starting room photo capture")
        photo_summary, room_view_records = _attach_room_photos(
            scene=scene,
            floor_idx=floor_idx,
            floor_z=floor_z,
            cases=cases,
            regions=regions,
            photo_root=photo_root,
        )
        _debug(f"room photo summary: {photo_summary}")
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
            "region_count": len(regions),
            "regions": [region.to_json() for region in regions],
            "point_adjustment_debug": point_adjustment_debug,
            "adjacency_graph": adjacency_graph,
            "connected_components": connected_components,
            "cases": cases,
            "room_views": room_view_records,
            "visualization": visualization_summary,
            "generation_diagnostics": {
                "pair_connectivity": pair_connectivity_diagnostics,
                "shortest_path_via_region": shortest_path_via_diagnostics,
            },
            "room_photo_summary": photo_summary,
            "question_json_summary": question_json_summary,
            "notes": [
                "Room-to-room shortest paths are derived from scene.get_shortest_path between sampled points.",
                "Returned world paths are projected back to room sequences via the scene segmentation map.",
                "Shortest-path-via-region answers are determined by whether the projected shortest room sequence includes the queried via-region.",
                "Region adjacency is still exported from room instance segmentation boundaries for reference.",
                "Each case includes one final reference image and one top-down map visualization.",
                "Each room is rendered once using a shared render point and exported with eight evenly spaced view images.",
            ],
        }
        _debug(f"writing metadata json to {output_json}")
        _write_json(output_json, metadata)
        _debug("metadata json written successfully")

        summary = {case_name: len(entries) for case_name, entries in cases.items()}
        print(
            json.dumps(
                {
                    "output_json": output_json,
                    "scene": args.scene,
                    "room": args.room,
                    "seed": args.seed,
                    "region_count": len(regions),
                    "summary": summary,
                    "photo_root": photo_summary.get("photo_root"),
                    "question_json_root": question_json_summary["question_json_root"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except ConnectTaskSkipped as exc:
        _debug(f"connect task skipped: reason={exc.reason} details={json.dumps(exc.details, ensure_ascii=False)}")
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


if __name__ == "__main__":
    main()
