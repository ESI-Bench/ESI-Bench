from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch as th

SCRIPT_DIR = Path(__file__).resolve().parent
OG_ROOT = str(SCRIPT_DIR / "OmniGibson")
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.constants import PrimType

import batch_mirror_distance as scene_utils


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

TASK_TYPE = "cover_small_item_cloth"
ATTEMPTED_ROOM_MARKER = "cover_small_item_room_attempted.json"
SKIPPED_ROOM_MARKER = "cover_small_item_room_skipped.json"
MIN_ROOM_AREA_M2 = 6.0
DEFAULT_OUTPUT_ROOT = "renders_cover_small_item"
DEFAULT_QUESTION_COUNT = 5
DEFAULT_RUNS_PER_ROOM = 5
DEFAULT_ITEM_FREE_RADIUS_M = 0.30
DEFAULT_ITEM_DROP_HEIGHT_M = 0.15
DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M = 0.10
DEFAULT_CAMERA_DISTANCE_M = 0.90
DEFAULT_CAMERA_HEIGHT_OFFSET_M = 0.55
DEFAULT_MAIN_VIEW_MIN_DISTANCE_M = 0.45
DEFAULT_MAIN_VIEW_MAX_DISTANCE_M = 0.68
DEFAULT_MAIN_VIEW_MIN_HEIGHT_OFFSET_M = 0.24
DEFAULT_MAIN_VIEW_MAX_HEIGHT_OFFSET_M = 0.40
DEFAULT_ROOM_VIEW_DISTANCE_M = 1.80
DEFAULT_ROOM_VIEW_HEIGHT_OFFSET_M = 1.05
DEFAULT_CAMERA_FOV_DEG = 70.0
DEFAULT_ROOM_VIEW_COUNT = 3
DEFAULT_SCENE_WARMUP_STEPS = 60
DEFAULT_ITEM_ADD_STEPS = 40
DEFAULT_SETTLE_STEPS = 240
DEFAULT_POST_ITEM_FREEZE_STEPS = 10
DEFAULT_CLOTH_ADD_STEPS = 60
DEFAULT_CLOTH_SETTLE_STEPS = 300
DEFAULT_CLOTH_MASS_KG = 1.0
DEFAULT_CLOTH_DOWNWARD_SPEED_MPS = 1.75
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
DEFAULT_SMALL_ITEM_JSON = str(SCRIPT_DIR / "inference" / "small_portable_item_candidates_5to15cm.json")
DEFAULT_CLOTH_JSON = str(SCRIPT_DIR / "inference" / "cover_small_item_cloth_usable.json")
RENDER_OBJECT_PREFIX = "cover_small_item_render_"
ERROR_LOG_BASENAME = "cover_small_item_errors.jsonl"
VIEWER_FRAME_RENDER_STEPS = 12
VIEWER_FRAME_MAX_RETRIES = 3
VIEWER_FRAME_RETRY_SLEEP_SEC = 0.15
FAST_SCENE_WARMUP_STEPS = 20
FAST_ITEM_ADD_STEPS = 12
FAST_ITEM_SETTLE_STEPS = 30
FAST_POST_ITEM_FREEZE_STEPS = 4
FAST_CLOTH_ADD_STEPS = 16
FAST_CLOTH_SETTLE_STEPS = 40
FAST_CAPTURE_RENDER_STEPS = 4

RUNTIME_VIEWER_FRAME_RENDER_STEPS = VIEWER_FRAME_RENDER_STEPS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cloth-cover small-item questions with one full scene load.")
    parser.add_argument("--scene", type=str, required=True, help="Scene model name")
    parser.add_argument("--room", action="append", default=None, help="Room instance name. Repeat for multiple rooms.")
    parser.add_argument("--floor", action="append", default=None, help="Floor object name paired with --room.")
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--small_item_json", type=str, default=DEFAULT_SMALL_ITEM_JSON)
    parser.add_argument("--cloth_json", type=str, default=DEFAULT_CLOTH_JSON)
    parser.add_argument("--question_count", type=int, default=DEFAULT_QUESTION_COUNT)
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--disable_trav_map_check", action="store_true")
    parser.add_argument("--fast_mode", action="store_true", default=True)
    parser.add_argument("--scene_warmup_steps", type=int, default=None)
    parser.add_argument("--item_add_steps", type=int, default=None)
    parser.add_argument("--item_settle_steps", type=int, default=None)
    parser.add_argument("--post_item_freeze_steps", type=int, default=None)
    parser.add_argument("--cloth_add_steps", type=int, default=None)
    parser.add_argument("--cloth_settle_steps", type=int, default=None)
    parser.add_argument("--capture_render_steps", type=int, default=None)
    parser.add_argument("--exit_on_finish", action="store_true")
    return parser


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_cover_small_item_merge] {context}: {exc.__class__.__name__}: {exc}", flush=True)
    traceback.print_exc()


def _serialize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if hasattr(value, "detach"):
        try:
            tensor = value.detach()
            if getattr(tensor, "device", None) is not None and tensor.device.type != "cpu":
                tensor = tensor.cpu()
            return {
                "type": type(value).__name__,
                "shape": list(getattr(tensor, "shape", [])),
                "dtype": str(getattr(tensor, "dtype", None)),
                "device": str(getattr(value, "device", None)),
            }
        except Exception:
            pass
    if hasattr(value, "shape"):
        try:
            return {
                "type": type(value).__name__,
                "shape": list(value.shape),
                "dtype": str(getattr(value, "dtype", None)),
            }
        except Exception:
            pass
    return repr(value)


def _to_float_list(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(v) for v in value]


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - float(start_time)) * 1000.0, 3)


def _resolve_runtime_steps(args) -> argparse.Namespace:
    def choose(explicit, normal, fast):
        if explicit is not None:
            return max(0, int(explicit))
        return int(fast if args.fast_mode else normal)

    args.scene_warmup_steps = choose(args.scene_warmup_steps, DEFAULT_SCENE_WARMUP_STEPS, FAST_SCENE_WARMUP_STEPS)
    args.item_add_steps = choose(args.item_add_steps, DEFAULT_ITEM_ADD_STEPS, FAST_ITEM_ADD_STEPS)
    args.item_settle_steps = choose(args.item_settle_steps, DEFAULT_SETTLE_STEPS, FAST_ITEM_SETTLE_STEPS)
    args.post_item_freeze_steps = choose(
        args.post_item_freeze_steps, DEFAULT_POST_ITEM_FREEZE_STEPS, FAST_POST_ITEM_FREEZE_STEPS
    )
    args.cloth_add_steps = choose(args.cloth_add_steps, DEFAULT_CLOTH_ADD_STEPS, FAST_CLOTH_ADD_STEPS)
    args.cloth_settle_steps = choose(args.cloth_settle_steps, DEFAULT_CLOTH_SETTLE_STEPS, FAST_CLOTH_SETTLE_STEPS)
    args.capture_render_steps = choose(args.capture_render_steps, VIEWER_FRAME_RENDER_STEPS, FAST_CAPTURE_RENDER_STEPS)
    return args


def _normalize_room_floor_args(args) -> list[tuple[str | None, str | None]]:
    rooms = [] if args.room is None else list(args.room)
    floors = [] if args.floor is None else list(args.floor)

    if not rooms:
        rooms = [None]
    if not floors:
        floors = [None] * len(rooms)
    if len(floors) == 1 and len(rooms) > 1:
        floors = floors * len(rooms)
    if len(rooms) != len(floors):
        raise ValueError(f"--room count ({len(rooms)}) must match --floor count ({len(floors)}).")
    return list(zip(rooms, floors))


def _build_config(scene_name: str, robot: str, room_name: str | None = None):
    args = type("Args", (), {"scene": scene_name, "robot": robot})()
    config = scene_utils._build_config(args)
    if room_name is None:
        config["scene"].pop("load_room_instances", None)
        config["scene"].pop("load_room_types", None)
    else:
        config["scene"]["load_room_instances"] = [str(room_name)]
    return config


def _stable_seed(scene_name: str, room_name: str | None, run_idx: int) -> int:
    key = f"{scene_name}::{room_name or '__scene__'}::{int(run_idx)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _question_seed(base_seed: int, q_idx: int) -> int:
    key = f"{int(base_seed)}::q::{int(q_idx)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _load_small_item_catalog(path: str) -> dict[str, list[str]]:
    raw = _read_json(path)
    categories = raw.get("categories", raw)
    catalog = {}
    for category, models in categories.items():
        if not isinstance(models, list):
            continue
        cleaned = sorted({str(model) for model in models})
        if cleaned:
            catalog[str(category)] = cleaned
    if not catalog:
        raise RuntimeError(f"No usable small-item categories found in {path}")
    return dict(sorted(catalog.items(), key=lambda item: item[0]))


def _load_cloth_catalog(path: str) -> list[dict]:
    raw = _read_json(path)
    accepted = raw.get("accepted", raw)
    catalog = []
    for entry in accepted:
        if str(entry.get("group", "")).lower() == "too_large_for_small_items":
            continue
        category = str(entry.get("category", "")).strip()
        model = str(entry.get("model", "")).strip()
        if not category or not model:
            continue
        catalog.append(
            {
                "category": category,
                "model": model,
                "mass_kg": float(entry.get("mass_kg", DEFAULT_CLOTH_MASS_KG) or DEFAULT_CLOTH_MASS_KG),
                "group": str(entry.get("group", "")),
            }
        )
    if not catalog:
        raise RuntimeError(f"No usable cloth assets found in {path}")
    catalog.sort(key=lambda item: (item["category"], item["model"]))
    return catalog


def _room_output_paths(output_root: str, scene_name: str, room_name: str | None, run_idx: int) -> tuple[str, str, str]:
    room_key = "scene_wide" if room_name is None else str(room_name)
    room_root = os.path.join(output_root, scene_name, room_key)
    question_json_root = os.path.join(room_root, "cover_small_item_question_jsons")
    render_root = os.path.join(room_root, "cover_small_item_renders")
    return room_root, question_json_root, render_root


def _error_log_path(room_run_root: str, scene_name: str) -> str:
    os.makedirs(room_run_root, exist_ok=True)
    return os.path.join(room_run_root, ERROR_LOG_BASENAME)


def _append_error_log(log_path: str | None, payload: dict) -> None:
    if log_path is None:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": dt.datetime.now().isoformat(),
        **payload,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _describe_camera_obs(cam: VisionSensor | None = None) -> dict:
    info = {}
    try:
        if cam is None:
            obs = og.sim._viewer_camera.get_obs()[0]
        else:
            obs = cam.get_obs()[0]
    except Exception as exc:
        return {"get_obs_error": f"{exc.__class__.__name__}: {exc}"}

    info["obs_type"] = type(obs).__name__
    if isinstance(obs, dict):
        info["obs_keys"] = sorted(str(k) for k in obs.keys())
        if "rgb" in obs:
            info["rgb"] = _serialize_value(obs["rgb"])
        for extra_key in ("depth", "seg_semantic", "seg_instance"):
            if extra_key in obs:
                info[extra_key] = _serialize_value(obs[extra_key])
    else:
        info["obs_repr"] = repr(obs)
    return info


def _camera_pose_snapshot(cam: VisionSensor | None = None) -> dict:
    try:
        if cam is None:
            pos, quat = og.sim._viewer_camera.get_position_orientation()
        else:
            pos, quat = cam.get_position_orientation()
        return {
            "position": _to_float_list(pos),
            "quaternion_xyzw": _to_float_list(quat),
        }
    except Exception as exc:
        return {"camera_pose_error": f"{exc.__class__.__name__}: {exc}"}


def _configure_sim_for_cloth_drop() -> None:
    try:
        og.sim.stop()
    except Exception:
        pass
    try:
        og.sim.set_simulation_dt(
            physics_dt=1.0 / 240.0,
            rendering_dt=1.0 / 60.0,
            sim_step_dt=1.0 / 60.0,
        )
    except Exception:
        pass
    try:
        og.sim.play()
    except Exception:
        pass


def _pause_simulation_for_capture() -> None:
    _render_only(2)


def _resume_simulation_after_capture() -> None:
    _render_only(1)


def _render_only(frames: int = 5) -> None:
    for _ in range(max(int(frames), 0)):
        try:
            og.sim.render()
        except Exception:
            break


def _warmup_render_pipeline(steps: int = 4, renders: int = 4) -> None:
    for _ in range(max(int(steps), 0)):
        try:
            og.sim.step()
        except Exception:
            break
    _render_only(renders)


def _create_capture_camera(width: int = DEFAULT_CAPTURE_WIDTH, height: int = DEFAULT_CAPTURE_HEIGHT) -> VisionSensor:
    cam = getattr(og.sim, "_viewer_camera", None) or getattr(og.sim, "viewer_camera", None)
    if cam is None:
        raise RuntimeError("Viewer camera is unavailable; cannot capture RGB frames.")
    try:
        if not cam.initialized:
            cam.initialize()
    except Exception:
        pass
    try:
        cam.image_height = int(height)
    except Exception:
        pass
    try:
        cam.image_width = int(width)
    except Exception:
        pass
    try:
        cam.initialize_sensors(names=["rgb"])
    except Exception:
        pass
    try:
        cam.clipping_range = th.tensor([0.001, 1000.0], dtype=th.float32)
    except Exception:
        pass
    _warmup_render_pipeline(steps=2, renders=4)
    return cam


def _set_capture_camera_pose(cam: VisionSensor, eye, quat) -> None:
    if getattr(cam, "_prim", None) is None:
        raise RuntimeError("Capture camera prim was not loaded into the stage before configuration.")
    cam.set_position_orientation(
        position=th.tensor([float(v) for v in eye], dtype=th.float32),
        orientation=th.tensor([float(v) for v in quat], dtype=th.float32),
    )
    try:
        cam.clipping_range = th.tensor([0.01, 100.0], dtype=th.float32)
    except Exception:
        pass
    _warmup_render_pipeline(steps=4, renders=4)


def _capture(cam: VisionSensor, path: str, *, log_path: str | None = None, context: dict | None = None) -> None:
    context = {} if context is None else dict(context)
    last_exc = None
    try:
        for attempt in range(1, VIEWER_FRAME_MAX_RETRIES + 1):
            _warmup_render_pipeline(steps=2, renders=RUNTIME_VIEWER_FRAME_RENDER_STEPS)
            obs, info = cam.get_obs()
            if not isinstance(obs, dict):
                last_exc = RuntimeError(f"get_obs() returned non-dict obs: {type(obs)}")
            elif "rgb" not in obs:
                last_exc = RuntimeError(f"RGB not found in obs. keys={list(obs.keys())}, info={info}")
            else:
                frame = obs["rgb"]
                if hasattr(frame, "detach"):
                    frame = frame.detach()
                if hasattr(frame, "cpu"):
                    frame = frame.cpu()
                image = np.array(frame)
                if image.ndim == 3 and image.shape[2] in (3, 4) and image.shape[0] > 0 and image.shape[1] > 0:
                    image = image[..., :3]
                    if np.issubdtype(image.dtype, np.floating):
                        if image.max() <= 1.0:
                            image = image * 255.0
                        image = np.clip(image, 0, 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                    scene_utils._save_rgb_png(path, image)
                    return
                last_exc = ValueError(f"Capture camera returned invalid rgb shape {getattr(image, 'shape', None)}")
            if last_exc is None:
                last_exc = RuntimeError("Unknown capture failure")
            if attempt < VIEWER_FRAME_MAX_RETRIES:
                time.sleep(VIEWER_FRAME_RETRY_SLEEP_SEC)
        assert last_exc is not None
        raise last_exc
    except Exception as exc:
        _append_error_log(
            log_path,
            {
                "event": "capture_failed",
                "image_path": path,
                "context": _serialize_value(context),
                "camera_pose": _camera_pose_snapshot(cam),
                "camera_obs": _describe_camera_obs(cam),
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _look_at_quaternion(position, target):
    eye = th.tensor([float(v) for v in position], dtype=th.float32)
    target_t = th.tensor([float(v) for v in target], dtype=th.float32)

    backward = eye - target_t
    backward = backward / th.norm(backward)

    world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
    right = th.linalg.cross(world_up, backward)
    if float(th.norm(right)) < 1e-6:
        world_up = th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
        right = th.linalg.cross(world_up, backward)

    right = right / th.norm(right)
    up = th.linalg.cross(backward, right)
    up = up / th.norm(up)

    rot = th.stack([right, up, backward], dim=1)
    quat = T.mat2quat(rot).to(dtype=th.float32)
    return _to_float_list(quat)


def _camera_eye_for_azimuth(
    target_xyz,
    azimuth_deg: float,
    *,
    distance_m: float = DEFAULT_CAMERA_DISTANCE_M,
    height_offset_m: float = DEFAULT_CAMERA_HEIGHT_OFFSET_M,
):
    theta = math.radians(float(azimuth_deg))
    return [
        float(target_xyz[0]) + float(distance_m) * math.cos(theta),
        float(target_xyz[1]) + float(distance_m) * math.sin(theta),
        float(target_xyz[2]) + float(height_offset_m),
    ]


def _bbox_center_and_extents(bbox_min, bbox_max) -> tuple[list[float], list[float]]:
    bbox_min = [float(v) for v in bbox_min]
    bbox_max = [float(v) for v in bbox_max]
    center = [float((lo + hi) * 0.5) for lo, hi in zip(bbox_min, bbox_max)]
    extents = [float(hi - lo) for lo, hi in zip(bbox_min, bbox_max)]
    return center, extents


def _main_view_camera_params(footprint_xy_m: float) -> tuple[float, float]:
    footprint_xy_m = max(float(footprint_xy_m), 0.05)
    normalized = min(max((footprint_xy_m - 0.10) / 0.30, 0.0), 1.0)
    distance_m = DEFAULT_MAIN_VIEW_MIN_DISTANCE_M + (
        DEFAULT_MAIN_VIEW_MAX_DISTANCE_M - DEFAULT_MAIN_VIEW_MIN_DISTANCE_M
    ) * normalized
    height_offset_m = DEFAULT_MAIN_VIEW_MIN_HEIGHT_OFFSET_M + (
        DEFAULT_MAIN_VIEW_MAX_HEIGHT_OFFSET_M - DEFAULT_MAIN_VIEW_MIN_HEIGHT_OFFSET_M
    ) * normalized
    return float(distance_m), float(height_offset_m)


def _sample_free_position(
    *,
    rng: random.Random,
    floor_record,
    blockers,
    preferred_center_xy,
    room_bbox_world_xy,
    trav_map,
    trav_map_img,
    clearance_m: float,
):
    candidates: list[list[float]] = []

    if trav_map is not None and trav_map_img is not None:
        map_img = np.array(trav_map_img)
        if map_img.ndim == 2 and map_img.size > 0:
            radius_px = max(1, int(np.ceil(float(clearance_m) / float(trav_map.map_resolution))))
            free_rc = np.argwhere(map_img > 0)
            for row, col in free_rc.tolist():
                r0 = max(0, int(row) - radius_px)
                r1 = min(map_img.shape[0], int(row) + radius_px + 1)
                c0 = max(0, int(col) - radius_px)
                c1 = min(map_img.shape[1], int(col) + radius_px + 1)
                patch = map_img[r0:r1, c0:c1]
                if patch.shape[0] < 2 * radius_px + 1 or patch.shape[1] < 2 * radius_px + 1:
                    continue
                if float(np.min(patch)) <= 0.0:
                    continue
                xy_world = trav_map.map_to_world(th.tensor([row, col], dtype=th.float32)).detach().cpu().numpy()
                x = float(xy_world[0])
                y = float(xy_world[1])
                if not (
                    float(floor_record.bbox_min[0]) + clearance_m <= x <= float(floor_record.bbox_max[0]) - clearance_m
                    and float(floor_record.bbox_min[1]) + clearance_m <= y <= float(floor_record.bbox_max[1]) - clearance_m
                ):
                    continue
                if room_bbox_world_xy is not None:
                    xmin, ymin, xmax, ymax = [float(v) for v in room_bbox_world_xy]
                    if not (xmin + clearance_m <= x <= xmax - clearance_m and ymin + clearance_m <= y <= ymax - clearance_m):
                        continue
                if not scene_utils._point_is_free((x, y), floor_record, blockers, clearance=clearance_m):
                    continue
                candidates.append([x, y, float(floor_record.bbox_max[2])])

    if not candidates:
        fallback_positions = scene_utils._generate_free_positions(
            floor_record=floor_record,
            blockers=blockers,
            agent_pos=(float(floor_record.center[0]), float(floor_record.center[1]), float(floor_record.center[2])),
            count=1000,
            target_radius=0.05,
            clearance=clearance_m,
        )
        for pos in fallback_positions:
            if room_bbox_world_xy is not None:
                xmin, ymin, xmax, ymax = [float(v) for v in room_bbox_world_xy]
                if not (xmin + clearance_m <= float(pos[0]) <= xmax - clearance_m and ymin + clearance_m <= float(pos[1]) <= ymax - clearance_m):
                    continue
            candidates.append([float(pos[0]), float(pos[1]), float(floor_record.bbox_max[2])])

    if not candidates:
        raise RuntimeError("No free placement position found for the small item.")

    ranked_candidates = []
    for pos in candidates:
        point_xy = (float(pos[0]), float(pos[1]))
        center_dist = scene_utils._distance_xy(point_xy, preferred_center_xy)
        nearest_blocker_dist = min((_distance_xy_to_record(point_xy, record) for record in blockers), default=999.0)
        ranked_candidates.append(
            (
                float(center_dist),
                -float(nearest_blocker_dist),
                round(float(pos[0]), 4),
                round(float(pos[1]), 4),
                list(pos),
            )
        )

    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    top_k = min(max(24, len(ranked_candidates) // 12), len(ranked_candidates))
    return list(rng.choice([item[-1] for item in ranked_candidates[:top_k]]))


def _remove_object(scene, obj, *, settle: bool = True) -> None:
    if obj is None:
        return
    try:
        scene.remove_object(obj)
    except Exception:
        pass
    if settle:
        scene_utils._step_sim(10)
    else:
        _render_only(5)


def _sample_target_and_distractors(rng: random.Random, item_catalog: dict[str, list[str]]) -> dict:
    categories = sorted(item_catalog)
    if len(categories) < 4:
        raise RuntimeError("Need at least 4 small-item categories to build one target and 3 distractors.")
    target_category = rng.choice(categories)
    target_model = rng.choice(item_catalog[target_category])
    distractor_categories = [category for category in categories if category != target_category]
    rng.shuffle(distractor_categories)
    distractors = []
    for category in distractor_categories[:3]:
        distractors.append(
            {
                "category": category,
                "model": rng.choice(item_catalog[category]),
            }
        )
    return {
        "target": {
            "category": target_category,
            "model": target_model,
        },
        "distractors": distractors,
    }


def _build_multiple_choice_options(rng: random.Random, target_category: str, distractors: list[dict]) -> dict:
    option_values = [str(target_category)] + [str(item["category"]) for item in distractors]
    rng.shuffle(option_values)
    options = []
    answer_key = None
    for idx, value in enumerate(option_values):
        option_id = chr(ord("A") + idx)
        options.append({"option_id": option_id, "text": value.replace("_", " ")})
        if value == target_category:
            answer_key = option_id
    if answer_key is None:
        raise RuntimeError("Failed to build multiple-choice options with a valid answer.")
    return {
        "question": "What object is under this cloth?",
        "options": options,
        "answer_option_id": answer_key,
        "answer_text": str(target_category).replace("_", " "),
    }


def _sample_cloth_asset(rng: random.Random, cloth_catalog: list[dict]) -> dict:
    recommended = [entry for entry in cloth_catalog if str(entry.get("group", "")).lower() == "recommended"]
    pool = recommended if recommended else cloth_catalog
    return dict(rng.choice(pool))


def _question_json_path(question_json_root: str, task_type: str, q_idx: int) -> str:
    task_dir = os.path.join(question_json_root, task_type)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, f"q_{q_idx:03d}.json")


def _question_render_dir(render_root: str, task_type: str, q_idx: int) -> str:
    out_dir = os.path.join(render_root, task_type, f"q_{q_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_room_attempted_marker(room_run_root: str, scene_name: str, payload: dict) -> None:
    os.makedirs(room_run_root, exist_ok=True)
    _write_json(os.path.join(room_run_root, ATTEMPTED_ROOM_MARKER), payload)


def _write_room_skipped_marker(room_run_root: str, scene_name: str, payload: dict) -> None:
    os.makedirs(room_run_root, exist_ok=True)
    _write_json(os.path.join(room_run_root, SKIPPED_ROOM_MARKER), payload)


def _set_velocity_zero(obj) -> None:
    try:
        obj.root_link.set_linear_velocity(th.tensor([0.0, 0.0, 0.0], dtype=th.float32))
    except Exception:
        pass
    try:
        obj.root_link.set_angular_velocity(th.tensor([0.0, 0.0, 0.0], dtype=th.float32))
    except Exception:
        pass


def _safe_get_pose(obj) -> dict | None:
    try:
        pos, quat = obj.get_position_orientation()
        return {
            "position": _to_float_list(pos),
            "quaternion_xyzw": _to_float_list(quat),
        }
    except Exception:
        return None


def _serialize_runtime_record(record) -> dict:
    return {
        "name": str(record.name),
        "category": str(record.category),
        "bbox_min": [float(v) for v in record.bbox_min],
        "bbox_max": [float(v) for v in record.bbox_max],
        "center": [float(v) for v in record.center],
        "extents": [float(v) for v in record.extents],
        "in_rooms": [str(v) for v in (record.in_rooms or ())],
        "has_open_state": bool(record.has_open_state),
        "open_state": record.open_state,
    }


def _point_inside_record_xy(point_xy, record, margin: float = 0.0) -> bool:
    return (
        float(record.bbox_min[0]) - margin <= float(point_xy[0]) <= float(record.bbox_max[0]) + margin
        and float(record.bbox_min[1]) - margin <= float(point_xy[1]) <= float(record.bbox_max[1]) + margin
    )


def _distance_xy_to_record(point_xy, record) -> float:
    x = float(point_xy[0])
    y = float(point_xy[1])
    dx = max(float(record.bbox_min[0]) - x, 0.0, x - float(record.bbox_max[0]))
    dy = max(float(record.bbox_min[1]) - y, 0.0, y - float(record.bbox_max[1]))
    return math.sqrt(dx * dx + dy * dy)


def _nearest_records(point_xy, records, limit: int = 8) -> list[dict]:
    enriched = []
    for record in records:
        enriched.append(
            {
                **_serialize_runtime_record(record),
                "distance_xy_to_point": float(_distance_xy_to_record(point_xy, record)),
                "contains_point_xy": bool(_point_inside_record_xy(point_xy, record)),
            }
        )
    enriched.sort(key=lambda item: (item["distance_xy_to_point"], item["category"], item["name"]))
    return enriched[: max(int(limit), 0)]


def _snapshot_object_state(obj) -> dict:
    payload = {
        "pose": _safe_get_pose(obj),
    }
    try:
        bbox_min, bbox_max = scene_utils._read_current_aabb(obj)
        payload["bbox"] = {
            "min": [float(v) for v in bbox_min],
            "max": [float(v) for v in bbox_max],
            "center": [float((lo + hi) * 0.5) for lo, hi in zip(bbox_min, bbox_max)],
        }
    except Exception as exc:
        payload["bbox_error"] = f"{exc.__class__.__name__}: {exc}"
    try:
        payload["linear_velocity"] = _to_float_list(obj.root_link.get_linear_velocity())
    except Exception:
        pass
    try:
        payload["angular_velocity"] = _to_float_list(obj.root_link.get_angular_velocity())
    except Exception:
        pass
    try:
        payload["mass_kg"] = float(obj.root_link.mass)
    except Exception:
        pass
    return payload


def _reset_cloth_to_best_configuration(cloth_obj) -> str | None:
    try:
        available = list(cloth_obj.root_link.get_available_configurations())
    except Exception:
        available = []

    preferred = "settled" if "settled" in available else ("default" if "default" in available else None)
    if preferred is None:
        return None

    try:
        cloth_obj.root_link.reset_points_to_configuration(preferred)
        return preferred
    except Exception:
        return None


def _generate_single_question(
    *,
    args,
    scene,
    capture_cam: VisionSensor,
    floor_record,
    room_name: str | None,
    room_bbox_world_xy,
    resolved_room_instance,
    blockers,
    room_objects,
    trav_map,
    trav_map_img,
    item_catalog: dict[str, list[str]],
    cloth_catalog: list[dict],
    question_json_root: str,
    render_root: str,
    seed: int,
    q_idx: int,
    error_log_path: str | None,
):
    question_start_time = time.perf_counter()
    rng = random.Random(seed)
    sampled = _sample_target_and_distractors(rng, item_catalog)
    cloth_spec = _sample_cloth_asset(rng, cloth_catalog)
    qa_payload = _build_multiple_choice_options(rng, sampled["target"]["category"], sampled["distractors"])

    target_spec = sampled["target"]
    item_name = f"{RENDER_OBJECT_PREFIX}item_{seed:010d}"
    cloth_name = f"{RENDER_OBJECT_PREFIX}cloth_{seed:010d}"

    item_obj = None
    cloth_obj = None
    debug_payload = {
        "scene": args.scene,
        "room": room_name,
        "run_idx": int(args.run_idx),
        "question_index": int(q_idx),
        "seed": int(seed),
        "timing_ms": {},
    }
    try:
        stage_start = time.perf_counter()
        scene_utils._set_viewer_camera_fov(DEFAULT_CAMERA_FOV_DEG)
        scene_utils._step_sim(args.scene_warmup_steps)
        debug_payload["timing_ms"]["scene_warmup"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        if room_bbox_world_xy is not None:
            preferred_center_xy = scene_utils._room_bbox_center_xy(room_bbox_world_xy)
        else:
            preferred_center_xy = th.tensor(
                [float(floor_record.center[0]), float(floor_record.center[1])],
                dtype=th.float32,
            )

        item_xy = _sample_free_position(
            rng=rng,
            floor_record=floor_record,
            blockers=blockers,
            preferred_center_xy=preferred_center_xy,
            room_bbox_world_xy=room_bbox_world_xy,
            trav_map=None if args.disable_trav_map_check else trav_map,
            trav_map_img=None if args.disable_trav_map_check else trav_map_img,
            clearance_m=DEFAULT_ITEM_FREE_RADIUS_M,
        )
        debug_payload["timing_ms"]["sample_free_position"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        item_obj = DatasetObject(
            name=item_name,
            category=target_spec["category"],
            model=target_spec["model"],
        )
        scene.add_object(item_obj)
        scene_utils._step_sim(args.item_add_steps)
        debug_payload["timing_ms"]["item_add_and_init"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        item_obj.set_position_orientation(
            position=th.tensor(
                [
                    float(item_xy[0]),
                    float(item_xy[1]),
                    float(floor_record.bbox_max[2]) + DEFAULT_ITEM_DROP_HEIGHT_M,
                ],
                dtype=th.float32,
            ),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        scene_utils._step_sim(args.item_settle_steps)
        _set_velocity_zero(item_obj)
        scene_utils._step_sim(args.post_item_freeze_steps)
        debug_payload["timing_ms"]["item_drop_and_settle"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        item_bbox_min, item_bbox_max = scene_utils._read_current_aabb(item_obj)
        item_center = [
            float((item_bbox_min[0] + item_bbox_max[0]) * 0.5),
            float((item_bbox_min[1] + item_bbox_max[1]) * 0.5),
            float((item_bbox_min[2] + item_bbox_max[2]) * 0.5),
        ]
        item_top_z = float(item_bbox_max[2])
        debug_payload["timing_ms"]["item_bbox_read"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        cloth_obj = DatasetObject(
            name=cloth_name,
            category=cloth_spec["category"],
            model=cloth_spec["model"],
            prim_type=PrimType.CLOTH,
            abilities={"cloth": {}},
            load_config={"default_configuration": "settled"},
        )
        scene.add_object(cloth_obj)
        scene_utils._step_sim(args.cloth_add_steps)
        debug_payload["timing_ms"]["cloth_add_and_init"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        try:
            cloth_available_configurations = list(cloth_obj.root_link.get_available_configurations())
        except Exception:
            cloth_available_configurations = []
        try:
            cloth_default_mass_kg = float(cloth_obj.root_link.mass)
        except Exception:
            cloth_default_mass_kg = None
        cloth_configuration_used = _reset_cloth_to_best_configuration(cloth_obj)
        try:
            cloth_obj.root_link.mass = float(DEFAULT_CLOTH_MASS_KG)
        except Exception:
            pass
        debug_payload["timing_ms"]["cloth_configuration"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        cloth_drop_pos = [
            float(item_center[0]),
            float(item_center[1]),
            float(item_top_z) + DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M,
        ]
        cloth_obj.set_position_orientation(
            position=th.tensor(cloth_drop_pos, dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        try:
            cloth_obj.root_link.set_linear_velocity(
                th.tensor([0.0, 0.0, -float(DEFAULT_CLOTH_DOWNWARD_SPEED_MPS)], dtype=th.float32)
            )
        except Exception:
            _set_velocity_zero(cloth_obj)
        scene_utils._step_sim(args.cloth_settle_steps)
        debug_payload["timing_ms"]["cloth_drop_and_settle"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        debug_payload["item_after_settle"] = _snapshot_object_state(item_obj)
        debug_payload["cloth_after_settle"] = _snapshot_object_state(cloth_obj)
        debug_payload["cloth_loading"] = {
            "available_configurations": [str(v) for v in cloth_available_configurations],
            "configuration_used": cloth_configuration_used,
            "default_mass_kg": cloth_default_mass_kg,
            "override_mass_kg": float(DEFAULT_CLOTH_MASS_KG),
            "initial_downward_speed_mps": float(DEFAULT_CLOTH_DOWNWARD_SPEED_MPS),
        }
        debug_payload["timing_ms"]["snapshot_after_settle"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        item_pos_after_cover, item_quat_after_cover = item_obj.get_position_orientation()
        item_pos_after_cover = item_pos_after_cover.detach().cpu().tolist()
        item_quat_after_cover = item_quat_after_cover.detach().cpu().tolist()
        item_bbox_min_after_cover, item_bbox_max_after_cover = scene_utils._read_current_aabb(item_obj)
        item_center_after_cover, item_extents_after_cover = _bbox_center_and_extents(
            item_bbox_min_after_cover, item_bbox_max_after_cover
        )
        cloth_bbox_min_after_settle = debug_payload["cloth_after_settle"]["bbox"]["min"]
        cloth_bbox_max_after_settle = debug_payload["cloth_after_settle"]["bbox"]["max"]
        cloth_center_after_settle, cloth_extents_after_settle = _bbox_center_and_extents(
            cloth_bbox_min_after_settle, cloth_bbox_max_after_settle
        )
        cover_target = [
            float(cloth_center_after_settle[0]),
            float(cloth_center_after_settle[1]),
            float(max(item_center_after_cover[2], cloth_center_after_settle[2])),
        ]
        cover_footprint_xy_m = float(max(cloth_extents_after_settle[0], cloth_extents_after_settle[1]))
        main_distance_m, main_height_offset_m = _main_view_camera_params(cover_footprint_xy_m)
        debug_payload["timing_ms"]["post_settle_geometry"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        render_dir = _question_render_dir(render_root, TASK_TYPE, q_idx)
        room_view_dir = os.path.join(render_dir, "room_view")
        gt_view_dir = os.path.join(render_dir, "gt_view")
        os.makedirs(room_view_dir, exist_ok=True)
        os.makedirs(gt_view_dir, exist_ok=True)
        debug_payload["timing_ms"]["prepare_render_dirs"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        main_azimuth_deg = -90.0
        room_view_azimuths = [0.0, 90.0, 180.0]
        main_eye = _camera_eye_for_azimuth(
            cover_target,
            main_azimuth_deg,
            distance_m=main_distance_m,
            height_offset_m=main_height_offset_m,
        )
        target = [float(v) for v in cover_target]
        blockers_containing_target = [record for record in blockers if _point_inside_record_xy(target[:2], record)]
        debug_payload["placement_diagnostics"] = {
            "item_xy_sampled": [float(item_xy[0]), float(item_xy[1])],
            "preferred_center_xy": _to_float_list(preferred_center_xy),
            "target_xyz": [float(v) for v in target],
            "item_center_after_cover_xyz": [float(v) for v in item_center_after_cover],
            "item_extents_after_cover_xyz": [float(v) for v in item_extents_after_cover],
            "cloth_center_after_settle_xyz": [float(v) for v in cloth_center_after_settle],
            "cloth_extents_after_settle_xyz": [float(v) for v in cloth_extents_after_settle],
            "cover_footprint_xy_m": float(cover_footprint_xy_m),
            "main_distance_m": float(main_distance_m),
            "main_height_offset_m": float(main_height_offset_m),
            "main_eye_xyz": [float(v) for v in main_eye],
            "blockers_containing_target_xy": [_serialize_runtime_record(record) for record in blockers_containing_target[:12]],
            "nearest_blockers_to_target": _nearest_records(target[:2], blockers, limit=12),
            "nearest_room_objects_to_target": _nearest_records(target[:2], room_objects, limit=12),
            "scene_clothes_dryers": [
                _serialize_runtime_record(record)
                for record in room_objects
                if "clothes_dryer" in str(record.category).lower() or "clothes_dryer" in str(record.name).lower()
            ],
        }
        debug_payload["timing_ms"]["camera_and_placement_diagnostics"] = _elapsed_ms(stage_start)

        render_payload = None
        if not args.skip_render:
            capture_total_start = time.perf_counter()
            _pause_simulation_for_capture()
            try:
                stage_start = time.perf_counter()
                main_quat = _look_at_quaternion(main_eye, target)
                _set_capture_camera_pose(capture_cam, main_eye, main_quat)
                _capture(
                    capture_cam,
                    os.path.join(render_dir, "main_view.png"),
                    log_path=error_log_path,
                    context={
                        "scene": args.scene,
                        "room": room_name,
                        "run_idx": int(args.run_idx),
                        "seed": int(seed),
                        "capture_camera": "viewer_camera",
                        "view_group": "main_view",
                        "task_type": TASK_TYPE,
                        "requested_eye": [float(v) for v in main_eye],
                        "requested_target": [float(v) for v in target],
                        "requested_quaternion_xyzw": [float(v) for v in main_quat],
                        "freeze_sim": True,
                    },
                )
                debug_payload["timing_ms"]["capture_main_view"] = _elapsed_ms(stage_start)
                render_payload = {
                    "main_view": {
                        "image_path": os.path.join(render_dir, "main_view.png"),
                        "camera_pose": {
                            "position": [float(v) for v in main_eye],
                            "quaternion_xyzw": [float(v) for v in main_quat],
                        },
                        "target": [float(v) for v in target],
                    },
                    "room_view": {
                        "views": []
                    },
                }
                room_view_timings = []
                for view_idx, azimuth_deg in enumerate(room_view_azimuths):
                    view_start = time.perf_counter()
                    room_eye = _camera_eye_for_azimuth(
                        item_center_after_cover,
                        azimuth_deg,
                        distance_m=DEFAULT_ROOM_VIEW_DISTANCE_M,
                        height_offset_m=DEFAULT_ROOM_VIEW_HEIGHT_OFFSET_M,
                    )
                    room_quat = _look_at_quaternion(room_eye, target)
                    room_image_path = os.path.join(room_view_dir, f"view_{view_idx:02d}.png")
                    _set_capture_camera_pose(capture_cam, room_eye, room_quat)
                    _capture(
                        capture_cam,
                        room_image_path,
                        log_path=error_log_path,
                        context={
                            "scene": args.scene,
                            "room": room_name,
                            "run_idx": int(args.run_idx),
                            "seed": int(seed),
                            "capture_camera": "viewer_camera",
                            "view_group": "room_view",
                            "view_index": int(view_idx),
                            "azimuth_deg": float(azimuth_deg),
                            "task_type": TASK_TYPE,
                            "requested_eye": [float(v) for v in room_eye],
                            "requested_target": [float(v) for v in target],
                            "requested_quaternion_xyzw": [float(v) for v in room_quat],
                            "freeze_sim": True,
                        },
                    )
                    render_payload["room_view"]["views"].append(
                        {
                            "image_path": room_image_path,
                            "camera_pose": {
                                "position": [float(v) for v in room_eye],
                                "quaternion_xyzw": [float(v) for v in room_quat],
                            },
                            "target": [float(v) for v in target],
                            "azimuth_deg": float(azimuth_deg),
                        }
                    )
                    room_view_timings.append(
                        {
                            "view_index": int(view_idx),
                            "azimuth_deg": float(azimuth_deg),
                            "elapsed_ms": _elapsed_ms(view_start),
                        }
                    )
                debug_payload["timing_ms"]["capture_room_views_total"] = round(
                    sum(entry["elapsed_ms"] for entry in room_view_timings), 3
                )
                debug_payload["timing_breakdown_room_views"] = room_view_timings

                stage_start = time.perf_counter()
                _remove_object(scene, cloth_obj, settle=False)
                cloth_obj = None
                debug_payload["timing_ms"]["remove_cloth_for_gt"] = _elapsed_ms(stage_start)

                stage_start = time.perf_counter()
                item_obj.set_position_orientation(
                    position=th.tensor([float(v) for v in item_pos_after_cover], dtype=th.float32),
                    orientation=th.tensor([float(v) for v in item_quat_after_cover], dtype=th.float32),
                )
                if hasattr(item_obj, "keep_still"):
                    item_obj.keep_still()
                _set_velocity_zero(item_obj)
                _render_only(5)
                debug_payload["timing_ms"]["restore_item_for_gt"] = _elapsed_ms(stage_start)

                stage_start = time.perf_counter()
                gt_quat = _look_at_quaternion(main_eye, target)
                gt_image_path = os.path.join(gt_view_dir, "main_view_without_cloth.png")
                _set_capture_camera_pose(capture_cam, main_eye, gt_quat)
                _capture(
                    capture_cam,
                    gt_image_path,
                    log_path=error_log_path,
                    context={
                        "scene": args.scene,
                        "room": room_name,
                        "run_idx": int(args.run_idx),
                        "seed": int(seed),
                        "capture_camera": "viewer_camera",
                        "view_group": "gt_view",
                        "task_type": TASK_TYPE,
                        "cloth_removed": True,
                        "requested_eye": [float(v) for v in main_eye],
                        "requested_target": [float(v) for v in target],
                        "requested_quaternion_xyzw": [float(v) for v in gt_quat],
                        "freeze_sim": True,
                    },
                )
                debug_payload["timing_ms"]["capture_gt_view"] = _elapsed_ms(stage_start)
                render_payload["gt_view"] = {
                    "main_view_without_cloth": {
                        "image_path": gt_image_path,
                        "camera_pose": {
                            "position": [float(v) for v in main_eye],
                            "quaternion_xyzw": [float(v) for v in gt_quat],
                        },
                        "target": [float(v) for v in target],
                    },
                    "cloth_removed": True,
                }
            finally:
                _resume_simulation_after_capture()
                debug_payload["timing_ms"]["capture_pipeline_total"] = _elapsed_ms(capture_total_start)
        else:
            stage_start = time.perf_counter()
            _remove_object(scene, cloth_obj)
            cloth_obj = None
            debug_payload["timing_ms"]["remove_cloth_skip_render"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        debug_payload["item_after_capture"] = _snapshot_object_state(item_obj)
        debug_payload["cloth_after_capture"] = None if cloth_obj is None else _snapshot_object_state(cloth_obj)
        debug_payload["timing_ms"]["snapshot_after_capture"] = _elapsed_ms(stage_start)

        question_payload = {
            "scene": args.scene,
            "room": room_name,
            "resolved_room": resolved_room_instance or room_name,
            "floor_name": floor_record.name,
            "task_type": TASK_TYPE,
            "question_index": q_idx,
            "question_id": f"{TASK_TYPE}/q_{q_idx:03d}",
            "run_idx": int(args.run_idx),
            "seed": int(seed),
            "small_item": {
                **target_spec,
                "placement_xy": [round(float(item_xy[0]), 4), round(float(item_xy[1]), 4)],
                "free_radius_m": float(DEFAULT_ITEM_FREE_RADIUS_M),
                "pose_after_cover": {
                    "position": [round(float(v), 4) for v in item_pos_after_cover],
                    "quaternion_xyzw": [round(float(v), 6) for v in item_quat_after_cover],
                },
                "bbox_after_cover": {
                    "min": [round(float(v), 4) for v in item_bbox_min_after_cover],
                    "max": [round(float(v), 4) for v in item_bbox_max_after_cover],
                },
            },
            "distractors": sampled["distractors"],
            "qa": qa_payload,
            "cloth": {
                "category": cloth_spec["category"],
                "model": cloth_spec["model"],
                "group": cloth_spec.get("group"),
                "configuration_used": cloth_configuration_used,
                "mass_kg": float(DEFAULT_CLOTH_MASS_KG),
                "initial_downward_speed_mps": float(DEFAULT_CLOTH_DOWNWARD_SPEED_MPS),
                "drop_position": [round(float(v), 4) for v in cloth_drop_pos],
                "drop_clearance_above_item_m": float(DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M),
            },
            "camera_setup": {
                "fov_deg": float(DEFAULT_CAMERA_FOV_DEG),
                "distance_to_item_m": float(main_distance_m),
                "height_offset_m": float(main_height_offset_m),
                "target_xyz": [round(float(v), 4) for v in target],
                "cloth_footprint_xy_m": round(float(cover_footprint_xy_m), 4),
                "main_view_azimuth_deg": float(main_azimuth_deg),
                "room_view_azimuths_deg": [float(v) for v in room_view_azimuths],
                "runtime_steps": {
                    "scene_warmup_steps": int(args.scene_warmup_steps),
                    "item_add_steps": int(args.item_add_steps),
                    "item_settle_steps": int(args.item_settle_steps),
                    "post_item_freeze_steps": int(args.post_item_freeze_steps),
                    "cloth_add_steps": int(args.cloth_add_steps),
                    "cloth_settle_steps": int(args.cloth_settle_steps),
                    "capture_render_steps": int(args.capture_render_steps),
                },
            },
            "render": render_payload,
        }
        out_path = _question_json_path(question_json_root, TASK_TYPE, q_idx)
        stage_start = time.perf_counter()
        _write_json(out_path, question_payload)
        debug_payload["timing_ms"]["write_question_json"] = _elapsed_ms(stage_start)
        debug_path = os.path.join(render_dir, "debug_diagnostics.json")
        debug_payload["question_json_path"] = out_path
        debug_payload["render_dir"] = render_dir
        debug_payload["render_payload"] = render_payload
        debug_payload["timing_ms"]["question_total"] = _elapsed_ms(question_start_time)
        debug_payload["timing_summary"] = {
            "question_total_ms": float(debug_payload["timing_ms"]["question_total"]),
            "slowest_stage": max(
                debug_payload["timing_ms"].items(),
                key=lambda item: float(item[1]),
            )[0],
        }
        stage_start = time.perf_counter()
        _write_json(debug_path, debug_payload)
        debug_payload["timing_ms"]["write_debug_json"] = _elapsed_ms(stage_start)
        debug_payload["timing_ms"]["question_total"] = _elapsed_ms(question_start_time)
        debug_payload["timing_summary"] = {
            "question_total_ms": float(debug_payload["timing_ms"]["question_total"]),
            "slowest_stage": max(
                debug_payload["timing_ms"].items(),
                key=lambda item: float(item[1]),
            )[0],
        }
        _write_json(debug_path, debug_payload)
        _append_error_log(
            error_log_path,
            {
                "event": "question_debug",
                "scene": args.scene,
                "room": room_name,
                "run_idx": int(args.run_idx),
                "question_index": int(q_idx),
                "seed": int(seed),
                "render_dir": render_dir,
                "debug_path": debug_path,
                "target_xyz": [float(v) for v in target],
                "main_eye_xyz": [float(v) for v in main_eye],
                "blockers_containing_target_xy_count": len(blockers_containing_target),
                "timing_ms": debug_payload["timing_ms"],
                "nearest_blockers_to_target": debug_payload["placement_diagnostics"]["nearest_blockers_to_target"][:5],
                "scene_clothes_dryers": debug_payload["placement_diagnostics"]["scene_clothes_dryers"][:5],
            },
        )
        timing_summary = ", ".join(
            f"{name}={float(value):.1f}ms" for name, value in sorted(debug_payload["timing_ms"].items(), key=lambda item: -float(item[1]))[:8]
        )
        print(
            f"[timing] scene={args.scene} room={room_name} q={q_idx:03d} total={float(debug_payload['timing_ms']['question_total']):.1f}ms {timing_summary}",
            flush=True,
        )
        return {
            "status": "ok",
            "question_count": 1,
            "question_index": int(q_idx),
            "question_json_path": out_path,
            "render_root": None if args.skip_render else render_dir,
        }
    finally:
        _remove_object(scene, cloth_obj)
        _remove_object(scene, item_obj)


def _generate_room_questions(
    *,
    args,
    scene,
    capture_cam: VisionSensor,
    floor_record,
    room_name: str | None,
    room_bbox_world_xy,
    resolved_room_instance,
    blockers,
    room_objects,
    trav_map,
    trav_map_img,
    item_catalog: dict[str, list[str]],
    cloth_catalog: list[dict],
    question_json_root: str,
    render_root: str,
    seed: int,
    error_log_path: str | None,
):
    if int(args.question_count) <= 0:
        raise ValueError(f"question_count must be positive, got {args.question_count}")

    question_results = []
    question_failures = []
    for q_idx in range(int(args.question_count)):
        question_seed = _question_seed(seed, q_idx)
        try:
            question_results.append(
                _generate_single_question(
                    args=args,
                    scene=scene,
                    capture_cam=capture_cam,
                    floor_record=floor_record,
                    room_name=room_name,
                    room_bbox_world_xy=room_bbox_world_xy,
                    resolved_room_instance=resolved_room_instance,
                    blockers=blockers,
                    room_objects=room_objects,
                    trav_map=trav_map,
                    trav_map_img=trav_map_img,
                    item_catalog=item_catalog,
                    cloth_catalog=cloth_catalog,
                    question_json_root=question_json_root,
                    render_root=render_root,
                    seed=question_seed,
                    q_idx=q_idx,
                    error_log_path=error_log_path,
                )
            )
        except Exception as exc:
            _log_exception(f"room={room_name} question={q_idx}", exc)
            failure_payload = {
                "event": "question_failed",
                "scene": args.scene,
                "room": room_name,
                "resolved_room": resolved_room_instance,
                "floor": floor_record.name,
                "run_idx": int(args.run_idx),
                "question_index": int(q_idx),
                "seed": int(question_seed),
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            _append_error_log(error_log_path, failure_payload)
            question_failures.append(failure_payload)
            continue

    status = "ok"
    if question_results and question_failures:
        status = "partial"
    elif question_failures and not question_results:
        status = "error"

    return {
        "status": status,
        "question_count": len(question_results),
        "question_json_paths": [entry["question_json_path"] for entry in question_results],
        "render_roots": [entry["render_root"] for entry in question_results if entry["render_root"] is not None],
        "failed_question_count": len(question_failures),
        "failed_questions": [
            {
                "question_index": entry["question_index"],
                "seed": entry["seed"],
                "error": entry["error"],
            }
            for entry in question_failures
        ],
    }


def _process_room(
    *,
    args,
    env,
    scene,
    capture_cam: VisionSensor,
    room_name: str | None,
    floor_name: str | None,
    item_catalog: dict[str, list[str]],
    cloth_catalog: list[dict],
):
    room_run_root, question_json_root, render_root = _room_output_paths(
        output_root=args.output_root,
        scene_name=args.scene,
        room_name=room_name,
        run_idx=args.run_idx,
    )
    os.makedirs(room_run_root, exist_ok=True)
    error_log_path = _error_log_path(room_run_root, args.scene)

    resolved_seed = _stable_seed(args.scene, room_name, args.run_idx)
    try:
        room_objects = scene_utils._collect_room_objects(scene, room_name)
        floor_record = scene_utils._select_floor(room_objects, floor_name, agent_pos=(0.0, 0.0, 0.0), room_name=room_name)
        room_area_m2 = float(floor_record.extents[0]) * float(floor_record.extents[1])
        if room_area_m2 < MIN_ROOM_AREA_M2:
            payload = {
                "scene": args.scene,
                "room": room_name,
                "floor": floor_name,
                "seed": resolved_seed,
                "skipped": True,
                "skip_reason": f"room area below {MIN_ROOM_AREA_M2:.1f} m^2",
                "room_area_m2": room_area_m2,
            }
            _write_room_skipped_marker(room_run_root, args.scene, payload)
            _write_room_attempted_marker(room_run_root, args.scene, {**payload, "attempted": True, "status": "skipped"})
            return {
                "room": room_name,
                "floor": floor_name,
                "seed": resolved_seed,
                "status": "skipped",
                "question_count": 0,
            }

        room_bbox_world_xy, resolved_room_instance = scene_utils._resolve_room_bbox_world_xy(scene, room_name, floor_record)
        blockers = [obj for obj in room_objects if scene_utils._is_floor_blocker(obj, floor_record.bbox_max[2])]
        trav_map, trav_map_img = scene_utils._trav_map_floor_image(scene, floor_idx=0, scene_name=args.scene)
        result = _generate_room_questions(
            args=args,
            scene=scene,
            capture_cam=capture_cam,
            floor_record=floor_record,
            room_name=room_name,
            room_bbox_world_xy=room_bbox_world_xy,
            resolved_room_instance=resolved_room_instance,
            blockers=blockers,
            room_objects=room_objects,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            item_catalog=item_catalog,
            cloth_catalog=cloth_catalog,
            question_json_root=question_json_root,
            render_root=render_root,
            seed=resolved_seed,
            error_log_path=error_log_path,
        )
        marker_payload = {
            "scene": args.scene,
            "room": room_name,
            "floor": floor_name,
            "seed": resolved_seed,
            "status": result["status"],
            "attempted": True,
            "question_count": int(result["question_count"]),
            "failed_question_count": int(result.get("failed_question_count", 0)),
            "failed_questions": result.get("failed_questions", []),
            "question_json_paths": result["question_json_paths"],
            "render_roots": result["render_roots"],
            "error_log_path": error_log_path,
        }
        _write_room_attempted_marker(room_run_root, args.scene, marker_payload)
        return {
            "room": room_name,
            "floor": floor_name,
            "seed": resolved_seed,
            **result,
        }
    except Exception as exc:
        _log_exception(f"room={room_name} floor={floor_name}", exc)
        _append_error_log(
            error_log_path,
            {
                "event": "room_failed",
                "scene": args.scene,
                "room": room_name,
                "floor": floor_name,
                "run_idx": int(args.run_idx),
                "seed": int(resolved_seed),
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        marker_payload = {
            "scene": args.scene,
            "room": room_name,
            "floor": floor_name,
            "seed": resolved_seed,
            "status": "error",
            "attempted": True,
            "question_count": 0,
            "error": f"{exc.__class__.__name__}: {exc}",
            "error_log_path": error_log_path,
        }
        _write_room_attempted_marker(room_run_root, args.scene, marker_payload)
        return {
            "room": room_name,
            "floor": floor_name,
            "seed": resolved_seed,
            "status": "error",
            "question_count": 0,
            "error": marker_payload["error"],
        }


def _process_room_in_fresh_env(
    *,
    args,
    room_name: str | None,
    floor_name: str | None,
    item_catalog: dict[str, list[str]],
    cloth_catalog: list[dict],
):
    env = None
    try:
        config = _build_config(args.scene, args.robot, room_name=room_name)
        env = og.Environment(configs=config)
        scene = env.scene
        _configure_sim_for_cloth_drop()
        _pause_simulation_for_capture()
        capture_cam = _create_capture_camera(width=DEFAULT_CAPTURE_WIDTH, height=DEFAULT_CAPTURE_HEIGHT)
        _resume_simulation_after_capture()
        return _process_room(
            args=args,
            env=env,
            scene=scene,
            capture_cam=capture_cam,
            room_name=room_name,
            floor_name=floor_name,
            item_catalog=item_catalog,
            cloth_catalog=cloth_catalog,
        )
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            og.clear()
        except Exception:
            pass


def main(argv=None):
    global RUNTIME_VIEWER_FRAME_RENDER_STEPS

    args = _build_parser().parse_args(argv)
    args = _resolve_runtime_steps(args)
    RUNTIME_VIEWER_FRAME_RENDER_STEPS = int(args.capture_render_steps)
    room_specs = _normalize_room_floor_args(args)
    item_catalog = _load_small_item_catalog(args.small_item_json)
    cloth_catalog = _load_cloth_catalog(args.cloth_json)
    room_results = []
    for room_name, floor_name in room_specs:
        print(
            f"[batch_cover_small_item_merge] scene={args.scene} room={room_name} floor={floor_name} run_idx={args.run_idx}",
            flush=True,
        )
        room_results.append(
            _process_room_in_fresh_env(
                args=args,
                room_name=room_name,
                floor_name=floor_name,
                item_catalog=item_catalog,
                cloth_catalog=cloth_catalog,
            )
        )

    summary = {
        "scene": args.scene,
        "run_idx": int(args.run_idx),
        "question_count": int(args.question_count),
        "rooms": room_results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.exit_on_finish:
        print("[batch_cover_small_item_merge] Generation finished. Simulator is kept alive. Press Ctrl+C to exit.", flush=True)
        try:
            while True:
                og.sim.render()
                scene_utils._step_sim(1)
        except KeyboardInterrupt:
            print("[batch_cover_small_item_merge] Exit requested by user (Ctrl+C).", flush=True)


if __name__ == "__main__":
    main()
