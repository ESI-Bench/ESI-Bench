"""
Generate all cognitivemap question families from one OmniGibson scene load.

This wrapper keeps the original connect / region / plan generation logic and
output formats, but shares a single environment creation and common scene
preprocessing pass so each (scene, room) only needs to be loaded once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch as th

SCRIPT_DIR = Path(__file__).resolve().parent
OG_ROOT = str(SCRIPT_DIR / "OmniGibson")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og

import batch_cognitivemap_connect as cm_connect
import batch_cognitivemap_plan as cm_plan
import batch_cognitivemap_region as cm_region


DEFAULT_OUTPUT_ROOT = str(SCRIPT_DIR / "renders_cognitivemap_batch")
TASK_TYPES = (
    "pair_connectivity",
    "shortest_path_via_region",
    "object_in_region",
    "objects_same_region",
    "object_closer_region",
    "navigation_actions",
    "navigation_regions",
)

VIEWER_CAMERA_FOV_DEG = 100.0
ROOM_CORNER_VIEW_COUNT = 4
ROOM_VIEW_CAMERA_HEIGHT_M = 1.35
ROOM_VIEW_TARGET_HEIGHT_M = 1.0
PATH_VIEW_SAMPLE_SPACING_M = 1.0
LONGEST_RAY_ANGLE_COUNT = 360
LONGEST_RAY_STEP_PX = 1.0


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_cognitivemap_merge] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _log_timing(label: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    print(f"[batch_cognitivemap_merge][timing] {label}: {elapsed:.3f}s", flush=True)
    return elapsed


def _quadratic_pair_budget(region_limit: int | None) -> int | None:
    if region_limit is None:
        return None
    limit = int(region_limit)
    if limit <= 0:
        return None
    return max(1, limit * limit)


def _set_viewer_camera_fov(fov_deg: float = VIEWER_CAMERA_FOV_DEG) -> None:
    cam = og.sim.viewer_camera
    aperture_mm = float(cam.horizontal_aperture)
    target_fov_deg = float(fov_deg)
    focal_length_mm = aperture_mm / (2.0 * math.tan(math.radians(target_fov_deg) * 0.5))
    cam.focal_length = focal_length_mm
    print(
        f"[batch_cognitivemap_merge] viewer camera horizontal FOV set to {target_fov_deg:.1f} deg "
        f"(aperture={aperture_mm:.3f} mm, focal_length={focal_length_mm:.3f} mm)",
        flush=True,
    )


def _room_dirname(room: str | None) -> str:
    return room if room is not None else "full_scene"


def _build_scene_metadata(args) -> dict:
    return {
        "scene": args.scene,
        "room": args.room,
        "floor_name": None,
        "seed": args.seed,
        "trav_map_basename": args.trav_map_basename,
    }


def _region_center_xy(region) -> tuple[float, float] | None:
    center_xy = getattr(region, "center_xy", None)
    if center_xy is not None:
        return (float(center_xy[0]), float(center_xy[1]))
    candidate_points = list(getattr(region, "candidate_points_xy", []) or [])
    if candidate_points:
        point = candidate_points[0]
        return (float(point[0]), float(point[1]))
    bbox = getattr(region, "expanded_bbox_world_xy", None) or getattr(region, "bbox_world_xy", None)
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)


def _region_bbox_xyxy(region) -> tuple[float, float, float, float] | None:
    bbox = getattr(region, "expanded_bbox_world_xy", None) or getattr(region, "bbox_world_xy", None)
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def _set_camera_pose(eye, target) -> dict:
    eye = [float(v) for v in eye]
    target = [float(v) for v in target]
    quat = cm_connect.look_at_quaternion(eye, target)
    og.sim._viewer_camera.set_position_orientation(
        th.tensor(eye, dtype=th.float32),
        th.tensor(quat, dtype=th.float32),
    )
    cm_connect._step_sim(5)
    return {
        "position": eye,
        "quaternion_xyzw": [float(v) for v in quat],
        "target": target,
    }


def _capture_view(image_path: str, eye, target) -> dict:
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    pose = _set_camera_pose(eye, target)
    cm_connect._capture(image_path)
    return {
        "image_path": image_path,
        "camera_pose": pose,
    }


def _room_mask_and_trav_map(scene, floor_idx: int, region):
    seg = cm_plan._segmap_get(scene)
    trav_map, map_img = cm_plan._trav_map_floor_image(scene, floor_idx=floor_idx)
    if seg is None or trav_map is None or map_img is None or not hasattr(seg, "room_ins_map"):
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


def _world_to_map_rc(trav_map, xy, map_img) -> np.ndarray:
    rc = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).detach().cpu().numpy()
    row = int(np.clip(round(float(rc[0])), 0, map_img.shape[0] - 1))
    col = int(np.clip(round(float(rc[1])), 0, map_img.shape[1] - 1))
    return np.array([row, col], dtype=np.int32)


def _map_rc_to_world_xy(trav_map, rc) -> list[float]:
    xy = trav_map.map_to_world(th.tensor([float(rc[0]), float(rc[1])], dtype=th.float32))
    return [float(xy[0].item()), float(xy[1].item())]


def _nearest_free_rc(preferred_rc: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray | None:
    candidate_rcs = np.argwhere(candidate_mask)
    if len(candidate_rcs) == 0:
        return None
    deltas = candidate_rcs.astype(np.float32) - preferred_rc.astype(np.float32)
    best_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    return candidate_rcs[best_idx].astype(np.int32)


def _select_center_near_free_xy(scene, floor_idx: int, region) -> tuple[list[float], dict]:
    render_selector = cm_plan._select_room_render_xy if hasattr(region, "expanded_bbox_world_xy") else cm_connect._select_room_render_xy
    fallback_xy, debug = render_selector(scene, floor_idx, region)
    _, trav_map, map_img, _, room_free_mask = _room_mask_and_trav_map(scene, floor_idx, region)
    if trav_map is None or map_img is None or room_free_mask is None or not room_free_mask.any():
        return [float(fallback_xy[0]), float(fallback_xy[1])], dict(debug)
    preferred_rc = _world_to_map_rc(trav_map, fallback_xy, map_img)
    best_rc = _nearest_free_rc(preferred_rc, room_free_mask)
    if best_rc is None:
        return [float(fallback_xy[0]), float(fallback_xy[1])], dict(debug)
    xy = _map_rc_to_world_xy(trav_map, best_rc)
    adjusted_debug = dict(debug)
    adjusted_debug["selected_map_rc"] = [int(best_rc[0]), int(best_rc[1])]
    adjusted_debug["selected_by"] = "center_near_free_xy"
    return xy, adjusted_debug


def _raycast_longest_direction(scene, floor_idx: int, region, start_xy) -> dict:
    _, trav_map, map_img, _, room_free_mask = _room_mask_and_trav_map(scene, floor_idx, region)
    if trav_map is None or map_img is None:
        return {
            "success": False,
            "reason": "trav_map_unavailable",
            "start_xy": [float(start_xy[0]), float(start_xy[1])],
        }
    start_rc = _world_to_map_rc(trav_map, start_xy, map_img)
    if room_free_mask is not None and room_free_mask.any():
        best_start_rc = _nearest_free_rc(start_rc, room_free_mask)
        if best_start_rc is not None:
            start_rc = best_start_rc
            start_xy = _map_rc_to_world_xy(trav_map, start_rc)
    best = None
    for angle_idx in range(LONGEST_RAY_ANGLE_COUNT):
        angle_rad = 2.0 * math.pi * float(angle_idx) / float(LONGEST_RAY_ANGLE_COUNT)
        dir_row = math.sin(angle_rad)
        dir_col = math.cos(angle_rad)
        endpoint_rc = start_rc.astype(np.float32)
        steps = 0
        while True:
            next_rc = start_rc.astype(np.float32) + np.array(
                [dir_row * (steps + 1) * LONGEST_RAY_STEP_PX, dir_col * (steps + 1) * LONGEST_RAY_STEP_PX],
                dtype=np.float32,
            )
            row = int(round(float(next_rc[0])))
            col = int(round(float(next_rc[1])))
            if row < 0 or row >= map_img.shape[0] or col < 0 or col >= map_img.shape[1]:
                break
            if int(map_img[row, col]) <= 0:
                break
            endpoint_rc = np.array([row, col], dtype=np.float32)
            steps += 1
        if steps <= 0:
            continue
        score = float(np.linalg.norm(endpoint_rc - start_rc.astype(np.float32)))
        if best is None or score > best["ray_length_px"]:
            best = {
                "angle_deg": round(math.degrees(angle_rad), 2),
                "angle_rad": float(angle_rad),
                "ray_length_px": score,
                "start_map_rc": [int(start_rc[0]), int(start_rc[1])],
                "end_map_rc": [int(round(float(endpoint_rc[0]))), int(round(float(endpoint_rc[1])))],
                "start_xy": [float(start_xy[0]), float(start_xy[1])],
                "target_xy": _map_rc_to_world_xy(trav_map, endpoint_rc),
            }
    if best is None:
        center_xy = _region_center_xy(region) or tuple(start_xy)
        target_xy = [float(center_xy[0]), float(center_xy[1])]
        if abs(target_xy[0] - start_xy[0]) < 1e-4 and abs(target_xy[1] - start_xy[1]) < 1e-4:
            target_xy = [float(start_xy[0] + 1.0), float(start_xy[1])]
        return {
            "success": True,
            "fallback": True,
            "angle_deg": 0.0,
            "angle_rad": 0.0,
            "ray_length_px": 0.0,
            "start_xy": [float(start_xy[0]), float(start_xy[1])],
            "target_xy": target_xy,
        }
    best["success"] = True
    return best


def _capture_longest_ray_view(scene, floor_idx: int, floor_z: float, region, output_path: str) -> dict:
    start_xy, selection_debug = _select_center_near_free_xy(scene, floor_idx, region)
    ray_info = _raycast_longest_direction(scene, floor_idx, region, start_xy)
    target_xy = ray_info.get("target_xy") or [float(start_xy[0] + 1.0), float(start_xy[1])]
    eye = [float(start_xy[0]), float(start_xy[1]), float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
    target = [float(target_xy[0]), float(target_xy[1]), float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M]
    view = _capture_view(output_path, eye=eye, target=target)
    view.update(
        {
            "success": True,
            "room_instance": region.room_instance,
            "render_xy": [float(start_xy[0]), float(start_xy[1])],
            "selection_debug": selection_debug,
            "ray_debug": ray_info,
        }
    )
    return view


def _capture_room_corner_views(scene, floor_idx: int, floor_z: float, region, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    bbox = _region_bbox_xyxy(region)
    center_xy = _region_center_xy(region)
    _, trav_map, map_img, _, room_free_mask = _room_mask_and_trav_map(scene, floor_idx, region)
    if bbox is None or center_xy is None or trav_map is None or map_img is None or room_free_mask is None:
        fallback_xy, selection_debug = _select_center_near_free_xy(scene, floor_idx, region)
        eye = [float(fallback_xy[0]), float(fallback_xy[1]), float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
        target = [float(center_xy[0] if center_xy else fallback_xy[0]), float(center_xy[1] if center_xy else fallback_xy[1]), float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M]
        single_view = _capture_view(os.path.join(output_dir, "view_00.png"), eye=eye, target=target)
        return {
            "success": True,
            "room_instance": region.room_instance,
            "render_xy": [float(fallback_xy[0]), float(fallback_xy[1])],
            "selection_debug": selection_debug,
            "views": {"view_00": single_view},
        }
    xmin, ymin, xmax, ymax = bbox
    width = max(1e-3, xmax - xmin)
    height = max(1e-3, ymax - ymin)
    inset = min(max(0.35, min(width, height) * 0.08), max(min(width, height) * 0.35, 0.35))
    preferred_corners = [
        np.array([xmin + inset, ymin + inset], dtype=np.float32),
        np.array([xmin + inset, ymax - inset], dtype=np.float32),
        np.array([xmax - inset, ymin + inset], dtype=np.float32),
        np.array([xmax - inset, ymax - inset], dtype=np.float32),
    ]
    views = {}
    corner_records = []
    for idx, preferred_xy in enumerate(preferred_corners):
        preferred_rc = _world_to_map_rc(trav_map, preferred_xy, map_img)
        best_rc = _nearest_free_rc(preferred_rc, room_free_mask)
        if best_rc is None:
            best_rc = _world_to_map_rc(trav_map, center_xy, map_img)
        eye_xy = _map_rc_to_world_xy(trav_map, best_rc)
        eye = [float(eye_xy[0]), float(eye_xy[1]), float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
        target = [float(center_xy[0]), float(center_xy[1]), float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M]
        view_id = f"view_{idx:02d}"
        view = _capture_view(os.path.join(output_dir, f"{view_id}.png"), eye=eye, target=target)
        view["preferred_corner_xy"] = [float(preferred_xy[0]), float(preferred_xy[1])]
        view["view_role"] = f"corner_{idx:02d}"
        views[view_id] = view
        corner_records.append([float(eye_xy[0]), float(eye_xy[1])])
    return {
        "success": True,
        "room_instance": region.room_instance,
        "room_bbox_world_xy": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "corner_render_xy": corner_records,
        "views": views,
    }


def _polyline_length(path_xy: list[list[float]]) -> float:
    total = 0.0
    for idx in range(1, len(path_xy)):
        total += math.dist(path_xy[idx - 1], path_xy[idx])
    return float(total)


def _interpolate_path_point(path_xy: list[list[float]], distance_along_m: float) -> tuple[list[float], float]:
    if len(path_xy) == 1:
        return [float(path_xy[0][0]), float(path_xy[0][1])], 0.0
    remaining = float(distance_along_m)
    for idx in range(1, len(path_xy)):
        p0 = path_xy[idx - 1]
        p1 = path_xy[idx]
        seg_len = math.dist(p0, p1)
        if seg_len <= 1e-6:
            continue
        if remaining <= seg_len:
            ratio = remaining / seg_len
            point = [
                float(p0[0] + (p1[0] - p0[0]) * ratio),
                float(p0[1] + (p1[1] - p0[1]) * ratio),
            ]
            yaw = float(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0])))
            return point, yaw
        remaining -= seg_len
    p0 = path_xy[-2]
    p1 = path_xy[-1]
    yaw = float(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0])))
    return [float(path_xy[-1][0]), float(path_xy[-1][1])], yaw


def _sample_path_with_spacing(path_xy: list[list[float]], spacing_m: float = PATH_VIEW_SAMPLE_SPACING_M) -> list[tuple[list[float], float, float]]:
    if not path_xy:
        return []
    total_len = _polyline_length(path_xy)
    distances = [0.0]
    cursor = float(spacing_m)
    while cursor < total_len:
        distances.append(cursor)
        cursor += float(spacing_m)
    if total_len > 1e-6 and (not distances or abs(distances[-1] - total_len) > 1e-6):
        distances.append(total_len)
    samples = []
    for dist_m in distances:
        point_xy, yaw = _interpolate_path_point(path_xy, dist_m)
        samples.append((point_xy, yaw, float(dist_m)))
    return samples


def _capture_path_views(path_xy: list[list[float]], floor_z: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    views = []
    for idx, (point_xy, yaw, dist_m) in enumerate(_sample_path_with_spacing(path_xy)):
        eye = [float(point_xy[0]), float(point_xy[1]), float(floor_z) + ROOM_VIEW_CAMERA_HEIGHT_M]
        target = [
            float(point_xy[0] + math.cos(float(yaw))),
            float(point_xy[1] + math.sin(float(yaw))),
            float(floor_z) + ROOM_VIEW_TARGET_HEIGHT_M,
        ]
        view = _capture_view(os.path.join(output_dir, f"path_view_{idx:03d}.png"), eye=eye, target=target)
        view["sample_index"] = int(idx)
        view["distance_along_path_m"] = round(float(dist_m), 3)
        view["yaw_rad"] = float(yaw)
        views.append(view)
    return {
        "success": True,
        "sample_spacing_m": float(PATH_VIEW_SAMPLE_SPACING_M),
        "path_length_m": round(_polyline_length(path_xy), 3),
        "sample_count": len(views),
        "views": views,
    }


def _select_proxy_target_region(start_region: str, target_region: str, graph: dict[str, list[str]], regions_by_name: dict[str, object]) -> str | None:
    reachable = cm_connect._all_reachable_nodes(graph, start_region)
    reachable.discard(start_region)
    target_center = _region_center_xy(regions_by_name[target_region])
    if not reachable or target_center is None:
        return None
    scored = []
    for name in reachable:
        center = _region_center_xy(regions_by_name[name])
        if center is None:
            continue
        scored.append((math.dist(center, target_center), name))
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def _build_connect_case_path_visual(scene, floor_idx: int, floor_z: float, case: dict, regions_by_name: dict[str, object], adjacency_graph: dict[str, list[str]], path_cache: dict, output_dir: str) -> dict:
    verification = case.get("verification") or {}
    if case.get("case_type") == "shortest_path_via_region":
        base_path = verification.get("shortest_path_info")
    else:
        base_path = verification.get("successful_path")
    selected_target_region = case.get("target_region")
    selection_reason = "direct_path"
    if not base_path or not (base_path.get("path_world_xy") or []):
        proxy_region = _select_proxy_target_region(case["source_region"], case["target_region"], adjacency_graph, regions_by_name)
        if proxy_region:
            base_path = cm_connect._find_best_path_info(
                scene=scene,
                floor_idx=floor_idx,
                regions_by_name=regions_by_name,
                start=case["source_region"],
                goal=proxy_region,
                cache=path_cache,
            )
            selected_target_region = proxy_region
            selection_reason = "nearest_connected_room_to_target"
    if not base_path or not (base_path.get("path_world_xy") or []):
        return {
            "success": False,
            "reason": "no_path_found_for_visualization",
            "selected_target_region": selected_target_region,
            "selection_reason": selection_reason,
        }
    path_views = _capture_path_views(base_path["path_world_xy"], floor_z=floor_z, output_dir=output_dir)
    path_views["selected_target_region"] = selected_target_region
    path_views["selection_reason"] = selection_reason
    path_views["room_sequence"] = list(base_path.get("room_sequence") or [])
    path_views["path_world_xy"] = list(base_path.get("path_world_xy") or [])
    return path_views


def _capture_case_room_views(scene, floor_idx: int, floor_z: float, regions_by_name: dict[str, object], room_names: list[str], output_root: str) -> dict[str, dict]:
    captured = {}
    for room_name in room_names:
        region = regions_by_name.get(room_name)
        if region is None:
            continue
        captured[room_name] = _capture_room_corner_views(
            scene=scene,
            floor_idx=floor_idx,
            floor_z=floor_z,
            region=region,
            output_dir=os.path.join(output_root, room_name),
        )
    return captured


def _run_connect_task(
    args,
    scene,
    run_dir: str,
    question_json_root: str,
    candidate_render_root: str,
    map_visualization_root: str,
    staging_root: str,
    shared_removed_doors: dict,
    shared_replaced_trav_map: bool,
) -> dict:
    task_start = time.perf_counter()
    output_json = os.path.join(run_dir, cm_connect.DEFAULT_OUTPUT_NAME)
    photo_root = os.path.join(staging_root, "connect_room_photos")
    initial_view_root = os.path.join(staging_root, "connect_initial_views")
    path_view_root = os.path.join(staging_root, "connect_path_views")
    rng = random.Random(args.seed)

    cm_connect.CONNECT_REGION_LIMIT = int(args.connect_region_limit)
    cm_connect.SHORTEST_PATH_TIMEOUT_S = float(args.shortest_path_timeout_s)

    cm_connect._debug("merge task start: connect")
    floor_z = float(scene.get_floor_height(0))
    floor_idx = 0

    t0 = time.perf_counter()
    regions, point_adjustment_debug = cm_connect._build_region_records(
        scene=scene,
        seed=args.seed,
        point_candidates=args.point_candidates,
    )
    _log_timing("connect.build_region_records", t0)
    if len(regions) < 2:
        raise RuntimeError("Not enough regions found to generate connectivity questions.")
    room_pair_budget = _quadratic_pair_budget(cm_connect.CONNECT_REGION_LIMIT)
    if room_pair_budget is not None:
        total_region_pairs = len(regions) * (len(regions) - 1) // 2
        effective_pair_budget = min(total_region_pairs, room_pair_budget)
        print(
            "[batch_cognitivemap_merge] connect pair-evaluation budget: "
            f"regions={len(regions)} "
            f"anchor_limit={cm_connect.CONNECT_REGION_LIMIT} "
            f"pair_budget={room_pair_budget} "
            f"effective_pair_budget={effective_pair_budget}",
            flush=True,
        )

    regions_by_name = {region.room_instance: region for region in regions}
    t0 = time.perf_counter()
    adjacency_graph = cm_connect._build_adjacency_graph(scene=scene, regions=regions)
    _log_timing("connect.build_adjacency_graph", t0)
    path_cache = {}
    t0 = time.perf_counter()
    connected_components = []
    seen = set()
    for name in sorted(regions_by_name):
        if name in seen:
            continue
        comp = sorted(cm_connect._all_reachable_nodes(adjacency_graph, name))
        seen.update(comp)
        connected_components.append(comp)
    _log_timing("connect.connected_components", t0)

    t0 = time.perf_counter()
    cm_connect._log_connectivity_diagnostics(
        graph=adjacency_graph,
        scene=scene,
        floor_idx=floor_idx,
        regions_by_name=regions_by_name,
        path_cache=path_cache,
        max_pairs=min(args.max_diagnostic_pairs, room_pair_budget) if room_pair_budget is not None else args.max_diagnostic_pairs,
    )
    _log_timing("connect.connectivity_diagnostics", t0)

    pair_connectivity_diagnostics = []
    shortest_path_via_diagnostics = []
    t0 = time.perf_counter()
    cases = {
        "pair_connectivity": cm_connect._generate_pair_connectivity_cases(
            scene=scene,
            floor_idx=floor_idx,
            regions_by_name=regions_by_name,
            max_cases=args.max_pair_cases,
            rng=rng,
            path_cache=path_cache,
            diagnostics=pair_connectivity_diagnostics,
            max_pair_evaluations=room_pair_budget,
        ),
        "shortest_path_via_region": cm_connect._generate_shortest_path_via_cases(
            scene=scene,
            floor_idx=floor_idx,
            regions_by_name=regions_by_name,
            max_cases=args.max_triple_cases,
            rng=rng,
            path_cache=path_cache,
            diagnostics=shortest_path_via_diagnostics,
            max_pair_evaluations=room_pair_budget,
        ),
    }
    _log_timing("connect.generate_cases", t0)

    t0 = time.perf_counter()
    visualization_summary = cm_connect._render_case_visualizations(
        scene=scene,
        regions_by_name=regions_by_name,
        cases=cases,
        viz_root=map_visualization_root,
    )
    _log_timing("connect.render_case_visualizations", t0)
    t0 = time.perf_counter()
    room_view_records = _capture_case_room_views(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions_by_name=regions_by_name,
        room_names=list(regions_by_name.keys()),
        output_root=photo_root,
    )
    _log_timing("connect.capture_case_room_views", t0)
    photo_summary = {
        "enabled": True,
        "photo_root": photo_root,
        "rooms_requested": len(regions_by_name),
        "rooms_photographed": len(room_view_records),
        "view_type": "four_room_corners",
    }
    t0 = time.perf_counter()
    for task_type, task_cases in cases.items():
        for case in task_cases:
            required_rooms = []
            for key in ("source_region", "target_region", "via_region"):
                value = case.get(key)
                if isinstance(value, str) and value and value not in required_rooms:
                    required_rooms.append(value)
            case["required_regions"] = required_rooms
            start_region = regions_by_name.get(case.get("source_region"))
            if start_region is not None:
                case["initial_view"] = _capture_longest_ray_view(
                    scene=scene,
                    floor_idx=floor_idx,
                    floor_z=floor_z,
                    region=start_region,
                    output_path=os.path.join(initial_view_root, f"{case['case_id']}__initial.png"),
                )
            case["room_views"] = {
                room_name: room_view_records[room_name]
                for room_name in required_rooms
                if room_name in room_view_records
            }
            case["path_views"] = _build_connect_case_path_visual(
                scene=scene,
                floor_idx=floor_idx,
                floor_z=floor_z,
                case=case,
                regions_by_name=regions_by_name,
                adjacency_graph=adjacency_graph,
                path_cache=path_cache,
                output_dir=os.path.join(path_view_root, case["case_id"]),
            )
    _log_timing("connect.attach_case_views", t0)
    t0 = time.perf_counter()
    question_json_summary = cm_connect._export_single_question_jsons(
        cases=cases,
        output_root=question_json_root,
        candidate_root=candidate_render_root,
        map_root=map_visualization_root,
        scene_metadata=_build_scene_metadata(args),
    )
    _log_timing("connect.export_single_question_jsons", t0)

    metadata = {
        "scene": args.scene,
        "room": args.room,
        "seed": args.seed,
        "trav_map_basename": args.trav_map_basename,
        "trav_map_replaced": bool(shared_replaced_trav_map),
        "removed_doors": shared_removed_doors,
        "region_count": len(regions),
        "room_pair_evaluation_budget": room_pair_budget,
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
            "Each case includes a top-down map visualization when traversability map data is available.",
            "Each referenced room stores four corner views looking toward the room center.",
            "Each case stores one start-room view captured from a center-near free point facing the longest traversable ray direction.",
            "Each case stores path-following images sampled every 1 meter along a shortest path; disconnected room pairs fall back to the start-room path toward the reachable room closest to the target room.",
        ],
    }
    cm_connect._write_json(output_json, metadata)
    _log_timing("connect.total", task_start)
    return {
        "output_json": output_json,
        "question_json_summary": question_json_summary,
        "case_counts": {key: len(value) for key, value in cases.items()},
    }


def _run_region_task(
    args,
    scene,
    run_dir: str,
    question_json_root: str,
    candidate_render_root: str,
    map_visualization_root: str,
    staging_root: str,
    shared_removed_doors: dict,
    shared_replaced_trav_map: bool,
) -> dict:
    task_start = time.perf_counter()
    output_json = os.path.join(run_dir, cm_region.DEFAULT_OUTPUT_NAME)
    room_photo_root = os.path.join(staging_root, "region_room_views")
    object_photo_root = os.path.join(staging_root, "region_object_views")
    rng = random.Random(args.seed)

    cm_region._debug("merge task start: region")
    floor_idx = 0
    floor_z = float(scene.get_floor_height(int(floor_idx)))
    wall_records = cm_region._collect_wall_records(scene)
    structural_wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]

    t0 = time.perf_counter()
    regions = cm_region._build_region_records(
        scene=scene,
        seed=args.seed,
        point_candidates=args.point_candidates,
        expansion_ratio=args.region_expansion_ratio,
        expansion_min=args.region_expansion_min,
        wall_bboxes_xyxy=structural_wall_bboxes,
    )
    _log_timing("region.build_region_records", t0)
    if len(regions) < 2:
        raise RuntimeError("Not enough regions found to generate region questions.")

    regions_by_name = {region.room_instance: region for region in regions}
    t0 = time.perf_counter()
    objects = cm_region._collect_scene_objects(scene=scene, regions_by_name=regions_by_name)
    _log_timing("region.collect_scene_objects", t0)
    if len(objects) < 2:
        raise RuntimeError("Not enough assigned objects found to generate region questions.")
    objects_by_name = {obj.name: obj for obj in objects}
    t0 = time.perf_counter()
    semantic_views, _ = cm_region._build_region_semantic_view_catalog(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions=regions,
        objects_by_name=objects_by_name,
    )
    _log_timing("region.build_semantic_view_catalog", t0)

    t0 = time.perf_counter()
    cases = {
        "object_in_region": cm_region._generate_belong_region_cases(
            objects=objects,
            regions_by_name=regions_by_name,
            max_cases=args.max_belong_cases,
            rng=rng,
            semantic_views=semantic_views,
        ),
        "objects_same_region": cm_region._generate_same_region_cases(
            objects=objects,
            max_cases=args.max_same_region_cases,
            rng=rng,
            semantic_views=semantic_views,
        ),
        "object_closer_region": cm_region._generate_closer_region_cases(
            objects=objects,
            regions_by_name=regions_by_name,
            max_cases=args.max_closer_region_cases,
            rng=rng,
            closer_margin=args.closer_margin,
            semantic_views=semantic_views,
        ),
    }
    _log_timing("region.generate_cases", t0)

    t0 = time.perf_counter()
    visualization_summary = cm_region._render_case_visualizations(
        scene=scene,
        regions_by_name=regions_by_name,
        objects_by_name=objects_by_name,
        cases=cases,
        viz_root=map_visualization_root,
    )
    _log_timing("region.render_case_visualizations", t0)
    t0 = time.perf_counter()
    photo_summary, room_view_records, object_view_records = cm_region._attach_render_photos(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions=regions,
        cases=cases,
        objects_by_name=objects_by_name,
        room_photo_root=room_photo_root,
        object_photo_root=object_photo_root,
    )
    _log_timing("region.attach_render_photos", t0)
    t0 = time.perf_counter()
    question_json_summary = cm_region._export_single_question_jsons(
        cases=cases,
        output_root=question_json_root,
        candidate_root=candidate_render_root,
        map_root=map_visualization_root,
        scene_metadata=_build_scene_metadata(args),
    )
    _log_timing("region.export_single_question_jsons", t0)

    metadata = {
        "scene": args.scene,
        "room": args.room,
        "seed": args.seed,
        "trav_map_basename": args.trav_map_basename,
        "trav_map_replaced": bool(shared_replaced_trav_map),
        "removed_doors": shared_removed_doors,
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
            "Each referenced room stores four corner room views and one wide open region view.",
            "Each referenced object stores one 0.8m closeup view captured from a 45-degree downward angle.",
            "Each case stores one semantic-overview reference image and one top-down map visualization.",
        ],
    }
    cm_region._write_json(output_json, metadata)
    _log_timing("region.total", task_start)
    return {
        "output_json": output_json,
        "question_json_summary": question_json_summary,
        "case_counts": {key: len(value) for key, value in cases.items()},
    }


def _run_plan_task(
    args,
    scene,
    run_dir: str,
    question_json_root: str,
    candidate_render_root: str,
    staging_root: str,
    shared_removed_doors: dict,
    shared_replaced_trav_map: bool,
) -> dict:
    task_start = time.perf_counter()
    output_json = os.path.join(run_dir, cm_plan.DEFAULT_OUTPUT_NAME)
    room_view_root = os.path.join(staging_root, "plan_room_views")
    question_view_root = os.path.join(staging_root, "plan_initial_views")
    path_view_root = os.path.join(staging_root, "plan_path_views")
    floor_idx = 0
    floor_z = float(scene.get_floor_height(int(floor_idx)))

    cm_plan.PLAN_REGION_LIMIT = int(args.plan_region_limit)

    cm_plan._debug("merge task start: plan")
    wall_records = cm_plan._collect_wall_records(scene)
    structural_wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]

    t0 = time.perf_counter()
    regions = cm_plan._build_region_records(
        scene=scene,
        seed=args.seed,
        point_candidates=args.point_candidates,
        expansion_ratio=args.region_expansion_ratio,
        expansion_min=args.region_expansion_min,
        wall_bboxes_xyxy=structural_wall_bboxes,
    )
    _log_timing("plan.build_region_records", t0)
    if len(regions) < 2:
        raise RuntimeError("Not enough regions found to generate navigation planning questions.")
    room_pair_budget = _quadratic_pair_budget(cm_plan.PLAN_REGION_LIMIT)
    if room_pair_budget is not None:
        total_region_pairs = len(regions) * (len(regions) - 1) // 2
        effective_pair_budget = min(total_region_pairs, room_pair_budget)
        print(
            "[batch_cognitivemap_merge] plan pair-evaluation budget: "
            f"regions={len(regions)} "
            f"anchor_limit={cm_plan.PLAN_REGION_LIMIT} "
            f"pair_budget={room_pair_budget} "
            f"effective_pair_budget={effective_pair_budget}",
            flush=True,
        )

    regions_by_name = {region.room_instance: region for region in regions}
    t0 = time.perf_counter()
    expanded_connectivity_graph = cm_plan._build_expanded_connectivity_graph(regions)
    _log_timing("plan.build_expanded_connectivity_graph", t0)
    t0 = time.perf_counter()
    cases = cm_plan._generate_navigation_cases(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions=regions,
        max_cases=args.max_plan_cases,
        rng=random.Random(args.seed),
        max_pair_evaluations=room_pair_budget,
    )
    _log_timing("plan.generate_navigation_cases", t0)
    if not cases:
        raise RuntimeError("No reachable room pairs found on the no-door traversability map.")

    t0 = time.perf_counter()
    room_view_records = _capture_case_room_views(
        scene=scene,
        floor_idx=floor_idx,
        floor_z=floor_z,
        regions_by_name=regions_by_name,
        room_names=list(regions_by_name.keys()),
        output_root=room_view_root,
    )
    _log_timing("plan.capture_case_room_views", t0)

    t0 = time.perf_counter()
    for case in cases:
        mentioned_rooms = []
        for room_name in list(case.get("region_sequence") or []) + [case.get("source_region"), case.get("target_region")]:
            if isinstance(room_name, str) and room_name and room_name not in mentioned_rooms:
                mentioned_rooms.append(room_name)
        case["room_views"] = {
            room_name: room_view_records[room_name]
            for room_name in mentioned_rooms
            if room_name in room_view_records
        }
        source_region = regions_by_name.get(case["source_region"])
        if source_region is not None:
            case["initial_view"] = _capture_longest_ray_view(
                scene=scene,
                floor_idx=floor_idx,
                floor_z=floor_z,
                region=source_region,
                output_path=os.path.join(question_view_root, f"{case['case_id']}__initial.png"),
            )
        case["path_views"] = _capture_path_views(
            case.get("path_world") or [],
            floor_z=floor_z,
            output_dir=os.path.join(path_view_root, case["case_id"]),
        )
        case["source_room_views"] = room_view_records.get(case["source_region"])
    _log_timing("plan.attach_case_views", t0)

    t0 = time.perf_counter()
    question_json_summary = cm_plan._export_single_question_jsons(
        cases=cases,
        output_root=question_json_root,
        candidate_root=candidate_render_root,
        scene_metadata=_build_scene_metadata(args),
    )
    _log_timing("plan.export_single_question_jsons", t0)

    metadata = {
        "scene": args.scene,
        "room": args.room,
        "seed": args.seed,
        "trav_map_basename": args.trav_map_basename,
        "trav_map_replaced": bool(shared_replaced_trav_map),
        "removed_doors": shared_removed_doors,
        "region_expansion_ratio": args.region_expansion_ratio,
        "region_expansion_min": args.region_expansion_min,
        "wall_related_object_count": len(wall_records),
        "structural_wall_count": len(structural_wall_bboxes),
        "region_count": len(regions),
        "room_pair_evaluation_budget": room_pair_budget,
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
            "Each referenced room stores four corner views looking toward the room center.",
            "Each selected case stores one source-room image rendered from a center-near free point facing the longest traversable ray direction.",
            "Each selected case stores path-following images sampled every 1 meter along the computed shortest path.",
        ],
    }
    cm_plan._write_json(output_json, metadata)
    _log_timing("plan.total", task_start)
    return {
        "output_json": output_json,
        "question_json_summary": question_json_summary,
        "case_count": len(cases),
    }


def main(argv=None):
    main_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate all cognitivemap metadata from one runtime OmniGibson scene.")
    parser.add_argument("--scene", default="house_double_floor_upper", help="Scene model name")
    parser.add_argument("--room", type=str, default=None, help="Optional room instance name")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point_candidates", type=int, default=cm_plan.DEFAULT_POINT_CANDIDATES)
    parser.add_argument("--trav_map_basename", type=str, default=cm_plan.DEFAULT_TRAV_MAP_BASENAME)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_pair_cases", type=int, default=cm_connect.DEFAULT_PAIR_LIMIT)
    parser.add_argument("--max_triple_cases", type=int, default=cm_connect.DEFAULT_TRIPLE_LIMIT)
    parser.add_argument(
        "--max_diagnostic_pairs",
        type=int,
        default=cm_connect.DEFAULT_DIAGNOSTIC_PAIR_LIMIT,
        help="Maximum number of connect room-pair diagnostics to evaluate; <=0 means no limit.",
    )
    parser.add_argument(
        "--connect_region_limit",
        type=int,
        default=cm_connect.DEFAULT_CONNECT_REGION_LIMIT,
        help="Skip the connect task when the extracted room count is greater than this; <=0 disables the limit.",
    )
    parser.add_argument(
        "--shortest_path_timeout_s",
        type=float,
        default=cm_connect.DEFAULT_SHORTEST_PATH_TIMEOUT_S,
        help="Skip the connect task if any shortest-path query takes longer than this many seconds; <=0 disables the timeout.",
    )
    parser.add_argument("--max_belong_cases", type=int, default=cm_region.DEFAULT_BELONG_CASE_LIMIT)
    parser.add_argument("--max_same_region_cases", type=int, default=cm_region.DEFAULT_SAME_REGION_LIMIT)
    parser.add_argument("--max_closer_region_cases", type=int, default=cm_region.DEFAULT_CLOSER_REGION_LIMIT)
    parser.add_argument("--max_plan_cases", "--max_cases", dest="max_plan_cases", type=int, default=cm_plan.DEFAULT_CASE_LIMIT)
    parser.add_argument("--region_expansion_ratio", type=float, default=cm_plan.DEFAULT_EXPANSION_RATIO)
    parser.add_argument("--region_expansion_min", type=float, default=cm_plan.DEFAULT_EXPANSION_MIN)
    parser.add_argument(
        "--plan_region_limit",
        type=int,
        default=cm_plan.DEFAULT_PLAN_REGION_LIMIT,
        help="Skip the planning task when the extracted room count is greater than this; <=0 disables the limit.",
    )
    parser.add_argument("--closer_margin", type=float, default=cm_region.DEFAULT_CLOSER_MARGIN)
    args = parser.parse_args(argv)

    room_dirname = _room_dirname(args.room)
    run_dir = os.path.join(args.output_root, args.scene, room_dirname)
    question_json_root = os.path.join(run_dir, "cognitivemap_question_jsons")
    candidate_render_root = os.path.join(run_dir, "candidate_renders")
    map_visualization_root = os.path.join(run_dir, "map_visualization")
    staging_root = os.path.join(run_dir, "_staging")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(question_json_root, exist_ok=True)
    os.makedirs(candidate_render_root, exist_ok=True)
    os.makedirs(map_visualization_root, exist_ok=True)

    config = cm_plan._build_config(args)
    env = None
    try:
        print(
            f"[batch_cognitivemap_merge] starting environment creation for scene={args.scene} "
            f"room={args.room}",
            flush=True,
        )
        t0 = time.perf_counter()
        env = og.Environment(configs=config)
        _log_timing("main.environment_creation", t0)
        t0 = time.perf_counter()
        _set_viewer_camera_fov()
        scene = env.scene
        print("[batch_cognitivemap_merge] environment created", flush=True)
        _log_timing("main.viewer_setup_and_scene_handle", t0)

        t0 = time.perf_counter()
        shared_removed_doors = cm_plan._remove_named_doors(scene)
        shared_replaced_trav_map = cm_plan._replace_trav_map_with_variant(scene, basename=str(args.trav_map_basename))
        print(f"[batch_cognitivemap_merge] trav-map replacement status: {shared_replaced_trav_map}", flush=True)
        _log_timing("main.scene_preprocessing", t0)

        try:
            t0 = time.perf_counter()
            connect_summary = _run_connect_task(
                args=args,
                scene=scene,
                run_dir=run_dir,
                question_json_root=question_json_root,
                candidate_render_root=candidate_render_root,
                map_visualization_root=map_visualization_root,
                staging_root=staging_root,
                shared_removed_doors=shared_removed_doors,
                shared_replaced_trav_map=shared_replaced_trav_map,
            )
            _log_timing("main.connect_task", t0)
        except cm_connect.ConnectTaskSkipped as exc:
            connect_output_json = os.path.join(run_dir, cm_connect.DEFAULT_OUTPUT_NAME)
            cm_connect._write_skip_metadata(
                output_json=connect_output_json,
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
                f"[batch_cognitivemap_merge] connect task skipped: reason={exc.reason} "
                f"details={json.dumps(exc.details, ensure_ascii=False)}",
                flush=True,
            )
            connect_summary = {
                "output_json": connect_output_json,
                "skipped": True,
                "skip_reason": exc.reason,
                "skip_details": exc.details,
                "question_json_summary": {
                    "enabled": False,
                    "question_json_root": question_json_root,
                    "counts": {},
                    "paths": {},
                },
            }
        t0 = time.perf_counter()
        region_summary = _run_region_task(
            args=args,
            scene=scene,
            run_dir=run_dir,
            question_json_root=question_json_root,
            candidate_render_root=candidate_render_root,
            map_visualization_root=map_visualization_root,
            staging_root=staging_root,
            shared_removed_doors=shared_removed_doors,
            shared_replaced_trav_map=shared_replaced_trav_map,
        )
        _log_timing("main.region_task", t0)
        try:
            t0 = time.perf_counter()
            plan_summary = _run_plan_task(
                args=args,
                scene=scene,
                run_dir=run_dir,
                question_json_root=question_json_root,
                candidate_render_root=candidate_render_root,
                staging_root=staging_root,
                shared_removed_doors=shared_removed_doors,
                shared_replaced_trav_map=shared_replaced_trav_map,
            )
            _log_timing("main.plan_task", t0)
        except cm_plan.PlanTaskSkipped as exc:
            plan_output_json = os.path.join(run_dir, cm_plan.DEFAULT_OUTPUT_NAME)
            cm_plan._write_skip_metadata(
                output_json=plan_output_json,
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
                f"[batch_cognitivemap_merge] plan task skipped: reason={exc.reason} "
                f"details={json.dumps(exc.details, ensure_ascii=False)}",
                flush=True,
            )
            plan_summary = {
                "output_json": plan_output_json,
                "skipped": True,
                "skip_reason": exc.reason,
                "skip_details": exc.details,
                "question_json_summary": {
                    "enabled": False,
                    "question_json_root": question_json_root,
                    "counts": {},
                    "paths": {},
                },
            }

        summary = {
            "scene": args.scene,
            "room": args.room,
            "seed": args.seed,
            "run_dir": run_dir,
            "question_json_root": question_json_root,
            "candidate_render_root": candidate_render_root,
            "map_visualization_root": map_visualization_root,
            "task_types": list(TASK_TYPES),
            "connect": connect_summary,
            "region": region_summary,
            "plan": plan_summary,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        _log_timing("main.total", main_start)
    except Exception as exc:
        _log_exception("main", exc)
        raise
    finally:
        if os.path.isdir(staging_root):
            shutil.rmtree(staging_root, ignore_errors=True)
        if env is not None:
            try:
                og.clear()
            except Exception:
                pass


if __name__ == "__main__":
    main()
