"""
Generate box-content change reasoning questions from a runtime OmniGibson scene.

The script reuses the hidden-in-box container placement and close-up rendering
flow from batch_counting_merge.py, but exports two explicit phases:

Phase 1: box contents are visible.
Phase 2: box contents may change or stay the same.

Question families:
1. change_detection
2. change_identification
3. current_state_reasoning
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import batch_counting_merge as bcm

TASK_TYPES = (
    "change_detection",
    "change_identification",
    "current_state_reasoning",
)

CHANGE_TYPES = ("replace", "remove", "add", "no_change")
DEFAULT_OUTPUT_ROOT = "renders_unobserved_changes"
DEFAULT_QUESTIONS_PER_TASK = 2
MAX_BOX_COUNT = min(3, len(bcm.HIDDEN_BOX_FIXED_ASSETS))
MIN_ROOM_BBOX_AREA_M2 = 6.0
CONTAINER_NAME_PREFIX = "render_unobserved_box_"
CONTENT_NAME_PREFIX = "render_unobserved_content_"
CURRENT_STATE_BOX_SPECS = (
    {"category": "cedar_chest", "model": "fwstpx", "color_label": "brown"},
    {"category": "cedar_chest", "model": "gbdzls", "color_label": "red"},
    {"category": "carton", "model": "cdmmwy", "color_label": "yellow"},
)
PRIMARY_VIEW_CANDIDATE_RATIOS = (0.2, 0.35, 0.5, 0.65, 0.8)
PRIMARY_VIEW_VISIBLE_PIXEL_THRESHOLD = 24


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _humanize(value: str | None) -> str:
    if value is None:
        return "empty"
    return str(value).replace("_", " ")


def _collect_image_paths(payload) -> list[str]:
    image_paths: list[str] = []

    def _walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"image", "image_path", "topdown_map"} and isinstance(item, str):
                    image_paths.append(item)
                else:
                    _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    seen = set()
    ordered = []
    for path in image_paths:
        norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def _write_single_question_json(
    *,
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
        "task_type": task_type,
        "question_index": int(q_idx),
        "question_id": f"{task_type}/q_{q_idx:03d}",
        "question_data": entry,
    }
    payload["image_paths"] = _collect_image_paths(payload["question_data"])
    out_path = os.path.join(task_dir, f"q_{q_idx:03d}.json")
    _write_json(out_path, payload)
    return out_path


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_unobserved_changes] {context}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _room_run_paths(output_root: str, scene_name: str, room_name: str) -> dict[str, str]:
    run_dir = os.path.join(output_root, scene_name, room_name)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "metadata_json": os.path.join(run_dir, "unobserved_changes_candidates.json"),
        "render_root": os.path.join(run_dir, "candidate_renders"),
        "question_json_root": os.path.join(run_dir, "question_jsons"),
    }


def _room_center_xy(room_bbox_info: dict, floor_record) -> list[float]:
    bbox = room_bbox_info.get("expanded_bbox_world_xy") or room_bbox_info.get("bbox_world_xy")
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        return [float((xmin + xmax) * 0.5), float((ymin + ymax) * 0.5)]
    return [float(floor_record.center[0]), float(floor_record.center[1])]


def _position_label_for_index(idx: int, total: int) -> str:
    if total <= 1:
        return "the box"
    if total == 2:
        return "left" if idx == 0 else "right"
    labels = ("left", "middle", "right")
    return labels[min(idx, len(labels) - 1)]


def _sample_content_candidate(
    *,
    count_target_candidates: list[dict],
    base_seed: int,
    scene_name: str,
    room_name: str,
    task_type: str,
    q_idx: int,
    slot_key: str,
    exclude_categories: set[str] | None = None,
) -> dict:
    exclude_categories = exclude_categories or set()
    for offset in range(max(8, len(count_target_candidates) * 2)):
        seed = bcm._scoped_seed(base_seed, scene_name, room_name, task_type, q_idx, slot_key, offset)
        sampled = bcm._sample_count_target_with_seed(count_target_candidates, seed)
        if sampled["category"] in exclude_categories:
            continue
        return dict(sampled)
    for candidate in count_target_candidates:
        if candidate["category"] not in exclude_categories:
            return dict(candidate)
    raise ValueError("Unable to sample a valid content candidate.")


def _empty_content() -> dict | None:
    return None


def _content_to_payload(content: dict | None) -> dict | None:
    if content is None:
        return None
    payload = {
        "category": content.get("category"),
        "display_name": content.get("display_name") or _humanize(content.get("category")),
        "model": content.get("representative_model"),
        "bbox_size_m": content.get("bbox_size_m"),
        "sampling_source": content.get("sampling_source"),
    }
    return payload


def _sample_box_count(task_type: str, rng: random.Random) -> int:
    if task_type == "current_state_reasoning":
        return min(3, MAX_BOX_COUNT)
    if task_type in {"change_detection", "change_identification"}:
        return 1
    raise ValueError(f"Unsupported task type for box count sampling: {task_type}")


def _build_change_detection_question(states: list[dict]) -> tuple[str, list[str], str]:
    any_changed = any(item["change_type"] != "no_change" for item in states)
    question = "Comparing Phase 1 and Phase 2, did the contents of any box change?"
    return question, ["yes", "no"], "yes" if any_changed else "no"


def _build_change_identification_question(states: list[dict], target_box_idx: int) -> tuple[str, list[str], str]:
    state = states[target_box_idx]
    label = state["position_label"]
    if label == "the box":
        question = "What happened to the contents of the box from Phase 1 to Phase 2?"
    else:
        question = f"What happened to the contents of the {label} box from Phase 1 to Phase 2?"
    return question, list(CHANGE_TYPES), str(state["change_type"])


def _build_current_state_reasoning_question(states: list[dict]) -> tuple[str, list[str], str]:
    options = [f"{item['position_label']} box" for item in states]
    changed = [item for item in states if item["change_type"] != "no_change"]
    if len(changed) != 1:
        raise ValueError("current_state_reasoning requires exactly one changed box.")
    question = "Which box changed from Phase 1 to Phase 2: the brown box, the red box, or the yellow box?"
    return question, options, f"{changed[0]['position_label']} box"


def _sample_question_states(
    *,
    task_type: str,
    q_idx: int,
    rng: random.Random,
    count_target_candidates: list[dict],
    base_seed: int,
    scene_name: str,
    room_name: str,
) -> tuple[list[dict], dict]:
    box_count = _sample_box_count(task_type, rng)
    states: list[dict] = []
    used_categories: set[str] = set()

    def _new_content(slot_key: str, extra_exclude: set[str] | None = None) -> dict:
        exclude = set(used_categories)
        if extra_exclude:
            exclude.update(extra_exclude)
        sampled = _sample_content_candidate(
            count_target_candidates=count_target_candidates,
            base_seed=base_seed,
            scene_name=scene_name,
            room_name=room_name,
            task_type=task_type,
            q_idx=q_idx,
            slot_key=slot_key,
            exclude_categories=exclude,
        )
        used_categories.add(sampled["category"])
        return sampled

    if task_type == "change_detection":
        should_change = bool(rng.randint(0, 1))
        changed_indices = set()
        if should_change:
            changed_count = rng.randint(1, box_count)
            changed_indices = set(rng.sample(range(box_count), k=changed_count))
        for box_idx in range(box_count):
            if box_idx in changed_indices:
                change_type = rng.choice(("replace", "remove", "add"))
            else:
                change_type = "no_change"
            if change_type == "add":
                phase1_content = _empty_content()
                phase2_content = _new_content(f"box{box_idx}_phase2_add")
            elif change_type == "remove":
                phase1_content = _new_content(f"box{box_idx}_phase1_remove")
                phase2_content = _empty_content()
            elif change_type == "replace":
                phase1_content = _new_content(f"box{box_idx}_phase1_replace")
                phase2_content = _new_content(
                    f"box{box_idx}_phase2_replace",
                    extra_exclude={phase1_content["category"]},
                )
            else:
                if rng.random() < 0.25:
                    phase1_content = _empty_content()
                    phase2_content = _empty_content()
                else:
                    phase1_content = _new_content(f"box{box_idx}_phase1_same")
                    phase2_content = dict(phase1_content)
            states.append(
                {
                    "box_index": box_idx,
                    "phase1_content": phase1_content,
                    "phase2_content": phase2_content,
                    "change_type": change_type,
                }
            )

    elif task_type == "change_identification":
        target_box_idx = rng.randrange(box_count)
        target_change_type = rng.choice(CHANGE_TYPES)
        for box_idx in range(box_count):
            change_type = target_change_type if box_idx == target_box_idx else "no_change"
            if change_type == "add":
                phase1_content = _empty_content()
                phase2_content = _new_content(f"box{box_idx}_phase2_add")
            elif change_type == "remove":
                phase1_content = _new_content(f"box{box_idx}_phase1_remove")
                phase2_content = _empty_content()
            elif change_type == "replace":
                phase1_content = _new_content(f"box{box_idx}_phase1_replace")
                phase2_content = _new_content(
                    f"box{box_idx}_phase2_replace",
                    extra_exclude={phase1_content["category"]},
                )
            elif rng.random() < 0.3:
                phase1_content = _empty_content()
                phase2_content = _empty_content()
            else:
                phase1_content = _new_content(f"box{box_idx}_phase1_same")
                phase2_content = dict(phase1_content)
            states.append(
                {
                    "box_index": box_idx,
                    "phase1_content": phase1_content,
                    "phase2_content": phase2_content,
                    "change_type": change_type,
                }
            )
        metadata = {"target_box_index": int(target_box_idx)}
        return states, metadata

    elif task_type == "current_state_reasoning":
        changed_box_idx = rng.randrange(box_count)
        changed_type = rng.choice(("replace", "remove", "add"))
        for box_idx in range(box_count):
            change_type = changed_type if box_idx == changed_box_idx else "no_change"
            if change_type == "add":
                phase1_content = _empty_content()
                phase2_content = _new_content(f"box{box_idx}_phase2_add")
            elif change_type == "remove":
                phase1_content = _new_content(f"box{box_idx}_phase1_remove")
                phase2_content = _empty_content()
            elif change_type == "replace":
                phase1_content = _new_content(f"box{box_idx}_phase1_replace")
                phase2_content = _new_content(
                    f"box{box_idx}_phase2_replace",
                    extra_exclude={phase1_content["category"]},
                )
            else:
                if rng.random() < 0.25:
                    phase1_content = _empty_content()
                    phase2_content = _empty_content()
                else:
                    phase1_content = _new_content(f"box{box_idx}_phase1_same")
                    phase2_content = dict(phase1_content)
            states.append(
                {
                    "box_index": box_idx,
                    "phase1_content": phase1_content,
                    "phase2_content": phase2_content,
                    "change_type": change_type,
                }
            )
        metadata = {"changed_box_index": int(changed_box_idx)}
        return states, metadata

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return states, {}


def _build_box_entries(
    *,
    scene,
    task_type: str,
    scene_name: str,
    room_name: str,
    base_seed: int,
    box_count: int,
    room_objects,
    floor_record,
    blockers,
    room_bbox_xyxy,
    agent_pos,
) -> tuple[list[dict], dict[str, object]]:
    hidden_box_cache: dict[str, object] = {}
    entries = []
    if task_type == "current_state_reasoning":
        selected_assets = [
            (item["category"], item["model"], item["color_label"])
            for item in CURRENT_STATE_BOX_SPECS[:box_count]
        ]
    else:
        selected_assets = [
            (category, model, None)
            for category, model in bcm.HIDDEN_BOX_FIXED_ASSETS[:box_count]
        ]

    for slot_idx, (category, model, color_label) in enumerate(selected_assets):
        name = f"{CONTAINER_NAME_PREFIX}{slot_idx}_{category}"
        entries.append(
            {
                "case": "hidden_in_box",
                "contains_ball": False,
                "container_object": None,
                "container_state": {"open": False},
                "container_spec": {
                    "name": name,
                    "category": category,
                    "model": model,
                    "color_label": color_label,
                    "placement": None,
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "size_class": "unknown",
                },
                "ball_positions": [],
            }
        )

    bcm._prepare_hidden_in_box_entries(
        scene=scene,
        scene_name=scene_name,
        room_name=room_name,
        base_seed=base_seed,
        entries=entries,
        hidden_box_cache=hidden_box_cache,
        room_objects=room_objects,
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        agent_pos=agent_pos,
    )
    live_entries = [entry for entry in entries if (entry.get("container_object") or {}).get("name")]
    if len(live_entries) != box_count:
        raise RuntimeError(f"Expected {box_count} boxes, but only placed {len(live_entries)}.")
    if task_type != "current_state_reasoning":
        live_entries.sort(
            key=lambda item: float(
                (
                    bcm._center_from_bbox_json(item.get("container_object")) or [0.0, 0.0, 0.0]
                )[0]
            )
        )
    for idx, entry in enumerate(live_entries):
        entry["box_index"] = idx
        color_label = (entry.get("container_spec") or {}).get("color_label")
        if task_type == "current_state_reasoning" and color_label:
            entry["position_label"] = str(color_label)
        else:
            entry["position_label"] = _position_label_for_index(idx, len(live_entries))
    return live_entries, hidden_box_cache


def _spawn_content_for_phase(
    *,
    scene,
    entries: list[dict],
    states: list[dict],
    phase_key: str,
    phase_seed: int,
) -> tuple[list[dict], list[object]]:
    placements: list[dict] = []
    spawned_objects: list[object] = []

    for state, entry in zip(states, entries):
        container_name = (entry.get("container_object") or {}).get("name")
        if not container_name:
            continue
        container_obj = scene.object_registry("name", container_name)
        if container_obj is None:
            continue
        content = state.get(phase_key)
        container_record = bcm._runtime_record_from_obj(container_obj)
        target_position = bcm._hidden_box_candidate_ball_position(container_record.bbox_min, container_record.bbox_max)
        placement = {
            "entry_case": "hidden_in_box",
            "container_obj": container_obj,
            "container_name": container_name,
            "position": [float(v) for v in target_position],
            "box_index": int(state["box_index"]),
            "position_label": str(state["position_label"]),
        }
        if content is not None:
            model = content.get("representative_model")
            obj = bcm._spawn_render_dataset_object(
                scene=scene,
                category=str(content["category"]),
                seed=int(phase_seed),
                idx=int(state["box_index"]),
                name_prefix=f"{CONTENT_NAME_PREFIX}{phase_key}_",
                fixed_model=str(model) if model else None,
                force_direct_placement=bool(bcm.DIRECT_PLACEMENT_MODE),
            )
            bcm._try_place_inside_container(
                obj,
                container_obj,
                target_position,
                desired_open=False,
            )
            bcm._step_sim(12)
            placement["target_obj"] = obj
            placement["target_name"] = getattr(obj, "name", f"{phase_key}_{state['box_index']:03d}")
            spawned_objects.append(obj)
        else:
            placement["target_obj"] = None
            placement["target_name"] = f"empty_box_{state['box_index']:03d}"
        placements.append(placement)
    bcm._step_sim(10)
    return placements, spawned_objects


def _cleanup_scene_objects(scene, objects: list[object]) -> None:
    for obj in reversed(list(objects)):
        try:
            bcm._safe_remove_scene_object(scene, obj, reason=f"cleanup {getattr(obj, 'name', '<unknown>')}")
        except Exception as exc:
            _log_exception("cleanup_scene_objects", exc)
    if objects:
        bcm._step_sim(10)


def _resolve_primary_view_bbox(floor_record, room_bbox_xyxy) -> tuple[float, float, float, float]:
    if room_bbox_xyxy is None:
        return (
            float(floor_record.bbox_min[0]),
            float(floor_record.bbox_min[1]),
            float(floor_record.bbox_max[0]),
            float(floor_record.bbox_max[1]),
        )
    xmin, ymin, xmax, ymax = room_bbox_xyxy
    return (
        float(min(xmin, xmax)),
        float(min(ymin, ymax)),
        float(max(xmin, xmax)),
        float(max(ymin, ymax)),
    )


def _collect_primary_view_targets(
    placements: list[dict],
    floor_record,
    room_center_xy,
) -> tuple[list[dict], list[float]]:
    targets: list[dict] = []
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    for placement in placements:
        container_obj = placement.get("container_obj")
        if container_obj is None:
            continue
        try:
            container_record = bcm._runtime_record_from_obj(container_obj)
        except Exception as exc:
            _log_exception(
                f"primary_view target bounds failed for {getattr(container_obj, 'name', '<unknown>')}",
                exc,
            )
            continue

        center = [float(v) for v in container_record.center]
        target_z = max(
            float(container_record.bbox_min[2]) + min(0.12, 0.5 * float(container_record.extents[2])),
            float(floor_record.bbox_max[2]) + 0.08,
        )
        targets.append(
            {
                "name": placement.get("container_name") or getattr(container_obj, "name", ""),
                "aliases": [placement.get("container_name"), getattr(container_obj, "name", "")],
                "center": center,
                "target_z": float(target_z),
            }
        )
        sum_x += center[0]
        sum_y += center[1]
        sum_z += float(target_z)

    if not targets:
        return [], [
            float(room_center_xy[0]),
            float(room_center_xy[1]),
            float(floor_record.bbox_max[2]) + float(bcm.AGENT_CAMERA_TARGET_HEIGHT_M),
        ]

    look_target = [
        float(sum_x / len(targets)),
        float(sum_y / len(targets)),
        float(sum_z / len(targets)),
    ]
    return targets, look_target


def _build_primary_view_candidate_points(
    floor_record,
    blockers,
    room_bbox_xyxy,
    trav_map,
    trav_map_img,
    fallback_agent_pos,
) -> list[tuple[str, list[float]]]:
    xmin, ymin, xmax, ymax = _resolve_primary_view_bbox(floor_record, room_bbox_xyxy)
    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    inset = min(max(0.45, min(width, height) * 0.1), max(min(width, height) * 0.35, 0.45))
    floor_margin = 0.2
    eye_z = float(floor_record.bbox_max[2]) + float(bcm.AGENT_CAMERA_HEIGHT_M)

    candidates: list[tuple[str, list[float]]] = [
        ("fallback", [float(fallback_agent_pos[0]), float(fallback_agent_pos[1])]),
    ]
    for ratio in PRIMARY_VIEW_CANDIDATE_RATIOS:
        x = float(xmin + width * ratio)
        y = float(ymin + height * ratio)
        candidates.extend(
            [
                (f"west_{ratio:.2f}", [xmin + inset, y]),
                (f"east_{ratio:.2f}", [xmax - inset, y]),
                (f"south_{ratio:.2f}", [x, ymin + inset]),
                (f"north_{ratio:.2f}", [x, ymax - inset]),
            ]
        )

    unique_candidates: list[tuple[str, list[float]]] = []
    seen_keys: set[tuple[float, float]] = set()
    for label, xy in candidates:
        clipped_xy = [
            float(min(max(float(xy[0]), float(floor_record.bbox_min[0]) + floor_margin), float(floor_record.bbox_max[0]) - floor_margin)),
            float(min(max(float(xy[1]), float(floor_record.bbox_min[1]) + floor_margin), float(floor_record.bbox_max[1]) - floor_margin)),
        ]
        key = (round(clipped_xy[0], 3), round(clipped_xy[1], 3))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if not bcm._camera_position_is_clear(
            clipped_xy,
            eye_z,
            floor_record,
            blockers,
            room_bbox_xyxy=room_bbox_xyxy,
        ):
            continue
        if not bcm._eye_xy_is_traversable(trav_map, trav_map_img, clipped_xy):
            continue
        unique_candidates.append((label, clipped_xy))

    return unique_candidates


def _score_primary_view_candidate(
    eye_xy: list[float],
    look_target: list[float],
    floor_record,
    target_specs: list[dict],
) -> tuple[tuple[float, ...], dict]:
    eye = [
        float(eye_xy[0]),
        float(eye_xy[1]),
        float(floor_record.bbox_max[2]) + float(bcm.AGENT_CAMERA_HEIGHT_M),
    ]
    eye, quat = bcm._set_camera_pose(eye, look_target)
    obs, info, image = bcm._get_viewer_frame()

    visible_count = 0
    visible_pixel_sum = 0
    centered_sum = 0.0
    per_target_metrics: list[dict] = []
    for target_spec in target_specs:
        metrics = bcm._target_visibility_metrics(
            obs,
            info,
            target_spec["name"],
            target_aliases=target_spec.get("aliases"),
        )
        pixels = int(metrics.get("visible_pixels") or 0)
        if pixels >= PRIMARY_VIEW_VISIBLE_PIXEL_THRESHOLD:
            visible_count += 1
            visible_pixel_sum += pixels
            centered_sum += float(metrics.get("centered_score") or 0.0)
        per_target_metrics.append(metrics)

    score = (
        float(visible_count),
        float(visible_pixel_sum),
        float(centered_sum),
    )
    return score, {
        "image": image,
        "eye": eye,
        "quat": quat,
        "visible_count": visible_count,
        "visible_pixel_sum": visible_pixel_sum,
        "centered_sum": round(centered_sum, 4),
        "target_metrics": per_target_metrics,
    }


def _capture_primary_view(
    *,
    output_dir: str,
    image_name: str,
    placements: list[dict],
    agent_pos,
    floor_record,
    room_center_xy,
    blockers,
    room_bbox_xyxy,
    trav_map,
    trav_map_img,
) -> dict:
    target_specs, look_target = _collect_primary_view_targets(
        placements=placements,
        floor_record=floor_record,
        room_center_xy=room_center_xy,
    )
    candidate_points = _build_primary_view_candidate_points(
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
        fallback_agent_pos=agent_pos,
    )

    best_result = None
    best_score = (-1.0, -1.0, -1.0)
    best_label = "fallback"
    for label, eye_xy in candidate_points:
        score, result = _score_primary_view_candidate(
            eye_xy=eye_xy,
            look_target=look_target,
            floor_record=floor_record,
            target_specs=target_specs,
        )
        if score > best_score:
            best_score = score
            best_result = result
            best_label = label

    if best_result is None:
        camera_poses = bcm.render_and_save(
            image_prefix=image_name,
            output_dir=output_dir,
            agent_pos=agent_pos,
            floor_record=floor_record,
            room_center_xy=room_center_xy,
            blockers=blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )
        primary_view = camera_poses.get(f"{image_name}.png", {})
        primary_view["selection_mode"] = "room_center_fallback"
        primary_view["visible_box_count"] = 0
        primary_view["visible_pixel_sum"] = 0
        return camera_poses

    image_path = os.path.join(output_dir, f"{image_name}.png")
    bcm._save_rgb_png(image_path, best_result["image"])
    horizontal_distance = max(1e-6, bcm._distance_xy(best_result["eye"][:2], look_target[:2]))
    pitch_deg = math.degrees(
        math.atan2(
            max(0.0, float(best_result["eye"][2]) - float(look_target[2])),
            horizontal_distance,
        )
    )
    return {
        f"{image_name}.png": {
            "position": best_result["eye"],
            "quaternion_xyzw": best_result["quat"],
            "angle_deg": None,
            "view_name": "maximize_box_visibility",
            "pitch_deg": pitch_deg,
            "look_target": [float(v) for v in look_target],
            "selection_mode": "maximize_visible_boxes",
            "candidate_label": best_label,
            "visible_box_count": int(best_result["visible_count"]),
            "visible_pixel_sum": int(best_result["visible_pixel_sum"]),
            "target_metrics": best_result["target_metrics"],
        }
    }


def _capture_gt_view(
    *,
    output_dir: str,
    image_name: str,
    placements: list[dict],
    floor_record,
    blockers,
    room_bbox_xyxy,
    trav_map,
    trav_map_img,
    room_center_xy,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    reveal_tokens = bcm._open_all_closeup_containers(placements, hidden_in_box_only=True)
    return {
        "image_path": None,
        "output_dir": output_dir,
        "camera_pose": None,
        "look_target": None,
        "fov_deg": float(bcm.VIEWER_CAMERA_FOV_DEG),
        "success": False,
    } if not placements else _capture_gt_view_impl(
        output_dir=output_dir,
        image_name=image_name,
        placements=placements,
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        reveal_tokens=reveal_tokens,
    )


def _build_gt_pose(
    *,
    placement: dict,
    floor_record,
    blockers,
    room_bbox_xyxy,
) -> tuple[list[float], list[float], str] | None:
    container_obj = placement.get("container_obj")
    if container_obj is None:
        return None
    container_record = bcm._runtime_record_from_obj(container_obj)
    _, container_quat = container_obj.get_position_orientation()
    front_xy = bcm._quaternion_xyzw_to_front_xy(container_quat.detach().cpu().tolist())
    front_vec = [float(front_xy[0]), float(front_xy[1])]
    front_norm = math.hypot(front_vec[0], front_vec[1])
    if front_norm <= 1e-6:
        front_vec = [1.0, 0.0]
    else:
        front_vec = [front_vec[0] / front_norm, front_vec[1] / front_norm]
    container_center = [float(v) for v in container_record.center]
    target_obj = placement.get("target_obj")
    look_target = list(container_center)
    if target_obj is not None:
        try:
            target_record = bcm._runtime_record_from_obj(target_obj)
            look_target = [
                float(target_record.center[0]),
                float(target_record.center[1]),
                float(target_record.bbox_min[2] + max(0.6 * target_record.extents[2], 0.02)),
            ]
        except Exception as exc:
            _log_exception(f"gt_view target bounds failed for {getattr(target_obj, 'name', '<unknown>')}", exc)
    else:
        look_target[2] = max(float(container_record.bbox_min[2]) + 0.08, float(container_center[2]))
    candidate_dirs = (
        ("front", [-front_vec[0], -front_vec[1]]),
        ("back", [front_vec[0], front_vec[1]]),
        ("left", [front_vec[1], -front_vec[0]]),
        ("right", [-front_vec[1], front_vec[0]]),
    )
    for direction_name, camera_dir_xy in candidate_dirs:
        half_depth = 0.5 * (
            abs(camera_dir_xy[0]) * float(container_record.extents[0])
            + abs(camera_dir_xy[1]) * float(container_record.extents[1])
        )
        eye_xy = [
            float(container_center[0] + camera_dir_xy[0] * (half_depth + 0.5)),
            float(container_center[1] + camera_dir_xy[1] * (half_depth + 0.5)),
        ]
        horizontal_distance = max(1e-6, bcm._distance_xy(eye_xy, look_target[:2]))
        eye = [
            float(eye_xy[0]),
            float(eye_xy[1]),
            float(look_target[2] + horizontal_distance),
        ]
        if bcm._camera_position_is_clear(
            eye[:2],
            eye[2],
            floor_record,
            blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            clearance=0.0,
        ):
            return eye, look_target, direction_name
    print(
        "[batch_unobserved_changes] gt_view pose failed for "
        f"{placement.get('container_name') or getattr(container_obj, 'name', '<unknown>')} "
        "(tried: front, back, left, right)",
        flush=True,
    )
    return None


def _capture_gt_view_impl(
    *,
    output_dir: str,
    image_name: str,
    placements: list[dict],
    floor_record,
    blockers,
    room_bbox_xyxy,
    reveal_tokens: list[dict],
) -> dict:
    try:
        ordered_placements = sorted(
            [placement for placement in placements if placement.get("container_obj") is not None],
            key=lambda item: (
                int(item.get("box_index", 0)),
                str(getattr(item.get("container_obj"), "name", "")),
            ),
        )
        if not ordered_placements:
            return {
                "output_dir": output_dir,
                "fov_deg": float(bcm.VIEWER_CAMERA_FOV_DEG),
                "images": [],
                "success": False,
            }

        gt_fov_deg = float(55.0)
        bcm._set_viewer_camera_fov(gt_fov_deg)
        images = []
        for idx, placement in enumerate(ordered_placements):
            pose = _build_gt_pose(
                placement=placement,
                floor_record=floor_record,
                blockers=blockers,
                room_bbox_xyxy=room_bbox_xyxy,
            )
            if pose is None:
                continue
            eye, target, direction_name = pose
            camera_eye, camera_quat = bcm._set_camera_pose(eye, target)
            _, _, image = bcm._get_viewer_frame()
            box_index = int(placement.get("box_index", idx))
            position_label = str(placement.get("position_label", f"box_{box_index}"))
            image_path = os.path.join(output_dir, f"{image_name}_box_{box_index:02d}.png")
            bcm._save_rgb_png(image_path, image)
            print(
                "[batch_unobserved_changes] gt_view pose selected "
                f"for box {box_index}: {direction_name}",
                flush=True,
            )
            images.append(
                {
                    "image_path": image_path,
                    "box_index": box_index,
                    "position_label": position_label,
                    "container_name": placement.get("container_name") or getattr(placement.get("container_obj"), "name", None),
                    "view_direction": direction_name,
                    "camera_pose": {
                        "position": camera_eye,
                        "quaternion_xyzw": camera_quat,
                    },
                    "look_target": [float(v) for v in target],
                    "pitch_deg": 45.0,
                    "requested_clearance_m": 0.5,
                    "fov_deg": gt_fov_deg,
                }
            )
        return {
            "output_dir": output_dir,
            "images": images,
            "fov_deg": gt_fov_deg,
            "success": bool(images),
        }
    finally:
        try:
            bcm._set_viewer_camera_fov()
        except Exception:
            pass
        for token in reversed(reveal_tokens):
            bcm._restore_container_after_closeup(token)


def _render_phase(
    *,
    output_dir: str,
    image_name: str,
    placements: list[dict],
    agent_pos,
    floor_record,
    room_center_xy,
    blockers,
    room_bbox_xyxy,
    room_bbox_info,
    trav_map,
    trav_map_img,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    image_poses = _capture_primary_view(
        output_dir=output_dir,
        image_name=image_name,
        placements=placements,
        agent_pos=agent_pos,
        floor_record=floor_record,
        room_center_xy=room_center_xy,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    image_path = os.path.join(output_dir, f"{image_name}.png")
    room_view_dir = os.path.join(output_dir, "room_view", image_name)
    gt_view_dir = os.path.join(output_dir, "gt_view")
    room_views = bcm._capture_room_corner_views(
        output_dir=room_view_dir,
        floor_record=floor_record,
        room_bbox_info=room_bbox_info,
        blockers=blockers,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
    )
    gt_view = _capture_gt_view(
        output_dir=gt_view_dir,
        image_name=image_name,
        placements=placements,
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        trav_map=trav_map,
        trav_map_img=trav_map_img,
        room_center_xy=room_center_xy,
    )
    return {
        "image_path": image_path,
        "camera_poses": image_poses,
        "room_view": room_views,
        "gt_view": gt_view,
        "output_dir": output_dir,
        "success": True,
    }


def _build_question_entry(
    *,
    task_type: str,
    q_idx: int,
    states: list[dict],
    question_meta: dict,
    render_payload: dict | None,
) -> dict:
    for idx, state in enumerate(states):
        if not state.get("position_label"):
            state["position_label"] = _position_label_for_index(idx, len(states))

    if task_type == "change_detection":
        question, options, answer = _build_change_detection_question(states)
    elif task_type == "change_identification":
        question, options, answer = _build_change_identification_question(states, int(question_meta["target_box_index"]))
    elif task_type == "current_state_reasoning":
        question, options, answer = _build_current_state_reasoning_question(states)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    boxes_payload = []
    for state in states:
        boxes_payload.append(
            {
                "box_index": int(state["box_index"]),
                "position_label": str(state["position_label"]),
                "change_type": str(state["change_type"]),
                "phase1_content": _content_to_payload(state.get("phase1_content")),
                "phase2_content": _content_to_payload(state.get("phase2_content")),
                "container": {
                    "name": state.get("container_name"),
                    "category": state.get("container_category"),
                    "model": state.get("container_model"),
                    "placement": state.get("container_placement"),
                    "bbox": state.get("container_bbox"),
                },
            }
        )

    return {
        "task_type": task_type,
        "question": question,
        "options": options,
        "answer": answer,
        "candidate_index": int(q_idx),
        "box_count": len(states),
        "boxes": boxes_payload,
        "question_metadata": question_meta,
        "phase_description": {
            "phase_1": "Initial observation: box contents are clearly visible.",
            "phase_2": "Later observation: box contents may have changed, been removed, been added, or stayed the same.",
        },
        "render": render_payload or {},
    }


def _generate_question_for_task(
    *,
    scene,
    scene_name: str,
    room_name: str,
    q_idx: int,
    task_type: str,
    base_seed: int,
    count_target_candidates: list[dict],
    room_objects,
    floor_record,
    blockers,
    room_bbox_xyxy,
    room_bbox_info,
    agent_pos,
    trav_map,
    trav_map_img,
    render_root: str,
    skip_render: bool,
) -> dict:
    question_seed = bcm._scoped_seed(base_seed, scene_name, room_name, task_type, q_idx)
    rng = random.Random(question_seed)
    states, question_meta = _sample_question_states(
        task_type=task_type,
        q_idx=q_idx,
        rng=rng,
        count_target_candidates=count_target_candidates,
        base_seed=base_seed,
        scene_name=scene_name,
        room_name=room_name,
    )

    box_entries, hidden_box_cache = _build_box_entries(
        scene=scene,
        task_type=task_type,
        scene_name=scene_name,
        room_name=room_name,
        base_seed=question_seed,
        box_count=len(states),
        room_objects=room_objects,
        floor_record=floor_record,
        blockers=blockers,
        room_bbox_xyxy=room_bbox_xyxy,
        agent_pos=agent_pos,
    )

    phase1_objects: list[object] = []
    phase2_objects: list[object] = []
    render_payload = None
    room_center_xy = _room_center_xy(room_bbox_info, floor_record)
    try:
        enriched_states = []
        for state, entry in zip(states, box_entries):
            container_spec = dict(entry.get("container_spec") or {})
            container_object = dict(entry.get("container_object") or {})
            enriched = dict(state)
            enriched["box_index"] = int(entry["box_index"])
            enriched["position_label"] = str(entry["position_label"])
            enriched["container_name"] = container_spec.get("name") or container_object.get("name")
            enriched["container_category"] = container_spec.get("category")
            enriched["container_model"] = container_spec.get("model")
            enriched["container_placement"] = copy.deepcopy(container_spec.get("placement"))
            enriched["container_bbox"] = copy.deepcopy(container_object.get("bbox"))
            enriched_states.append(enriched)
        states = enriched_states

        if not skip_render:
            phase1_seed = bcm._scoped_seed(question_seed, "phase1")
            phase1_placements, phase1_objects = _spawn_content_for_phase(
                scene=scene,
                entries=box_entries,
                states=states,
                phase_key="phase1_content",
                phase_seed=phase1_seed,
            )
            phase1_render = _render_phase(
                output_dir=os.path.join(render_root, task_type, f"q_{q_idx:03d}"),
                image_name="image1",
                placements=phase1_placements,
                agent_pos=agent_pos,
                floor_record=floor_record,
                room_center_xy=room_center_xy,
                blockers=blockers,
                room_bbox_xyxy=room_bbox_xyxy,
                room_bbox_info=room_bbox_info,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )

            _cleanup_scene_objects(scene, phase1_objects)
            phase1_objects = []

            phase2_seed = bcm._scoped_seed(question_seed, "phase2")
            phase2_placements, phase2_objects = _spawn_content_for_phase(
                scene=scene,
                entries=box_entries,
                states=states,
                phase_key="phase2_content",
                phase_seed=phase2_seed,
            )
            phase2_render = _render_phase(
                output_dir=os.path.join(render_root, task_type, f"q_{q_idx:03d}"),
                image_name="image2",
                placements=phase2_placements,
                agent_pos=agent_pos,
                floor_record=floor_record,
                room_center_xy=room_center_xy,
                blockers=blockers,
                room_bbox_xyxy=room_bbox_xyxy,
                room_bbox_info=room_bbox_info,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
            )

            render_payload = {
                "multi_image_input": True,
                "image1": phase1_render,
                "image2": phase2_render,
                "room_view": {
                    "image1_dir": os.path.join(render_root, task_type, f"q_{q_idx:03d}", "room_view", "image1"),
                    "image2_dir": os.path.join(render_root, task_type, f"q_{q_idx:03d}", "room_view", "image2"),
                    "image1": phase1_render.get("room_view"),
                    "image2": phase2_render.get("room_view"),
                },
                "gt_view": {
                    "dir": os.path.join(render_root, task_type, f"q_{q_idx:03d}", "gt_view"),
                    "image1": phase1_render.get("gt_view"),
                    "image2": phase2_render.get("gt_view"),
                },
                "room_bbox": room_bbox_info,
                "success": True,
            }
    finally:
        _cleanup_scene_objects(scene, phase1_objects)
        _cleanup_scene_objects(scene, phase2_objects)
        _cleanup_scene_objects(scene, list(hidden_box_cache.values()))

    return _build_question_entry(
        task_type=task_type,
        q_idx=q_idx,
        states=states,
        question_meta=question_meta,
        render_payload=render_payload,
    )


def _process_room(
    *,
    env,
    scene,
    scene_name: str,
    room_name: str,
    floor_name: str | None,
    output_root: str,
    seed: int,
    questions_per_task: int,
    task_types: tuple[str, ...],
    count_target_candidates: list[dict],
    structural_wall_bboxes,
    agent_position,
    skip_render: bool,
) -> dict:
    room_seed = bcm._scoped_seed(seed, scene_name, room_name, "unobserved_changes_room")
    paths = _room_run_paths(output_root, scene_name, room_name)
    room_objects = bcm._collect_room_objects(scene, room_name)
    if not room_objects:
        raise ValueError(f"No objects found in room '{room_name}'.")

    floors = [obj for obj in room_objects if obj.category == "floors"]
    if not floors:
        raise ValueError(f"No floor object found in room '{room_name}'.")

    provisional_floor = None
    if floor_name is not None:
        for floor in floors:
            if floor.name == floor_name:
                provisional_floor = floor
                break
        if provisional_floor is None:
            raise ValueError(f"Floor '{floor_name}' not found in room '{room_name}'.")
    else:
        provisional_floor = max(floors, key=lambda floor: floor.footprint_area)

    room_bbox_info = bcm._resolve_room_bbox(scene, room_name, structural_wall_bboxes)
    room_bbox_xyxy = room_bbox_info.get("expanded_bbox_world_xy")
    room_bbox_area_m2 = room_bbox_info.get("bbox_area_m2")
    if room_bbox_area_m2 is not None and float(room_bbox_area_m2) < MIN_ROOM_BBOX_AREA_M2:
        raise ValueError(
            f"Room '{room_name}' bbox area {float(room_bbox_area_m2):.2f} m^2 is below "
            f"minimum {MIN_ROOM_BBOX_AREA_M2:.2f} m^2."
        )

    blockers = [obj for obj in room_objects if bcm._is_floor_blocker(obj, provisional_floor.bbox_max[2])]
    floor_idx = bcm._infer_floor_idx(provisional_floor)
    trav_map, trav_map_img = bcm._trav_map_floor_image(scene, floor_idx=floor_idx, scene_name=scene_name)
    agent_pos = bcm._resolve_agent_position(env, agent_position)
    if agent_pos is None:
        agent_pos = bcm._sample_agent_position_near_short_edge(
            floor_record=provisional_floor,
            blockers=blockers,
            room_bbox_xyxy=room_bbox_xyxy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
        )
    floor_record = bcm._select_floor(room_objects, floor_name, agent_pos)
    blockers = [obj for obj in room_objects if bcm._is_floor_blocker(obj, floor_record.bbox_max[2])]

    scene_metadata = {
        "scene": scene_name,
        "room": room_name,
        "floor_name": floor_record.name,
        "seed": int(seed),
        "camera_setup": {
            "mode": "two_phase_primary_plus_room_and_gt_views",
            "primary_images": "image1.png + image2.png",
            "room_view_images": "room_view/image1/* + room_view/image2/*",
            "gt_view_images": "gt_view/image1_gt.png + gt_view/image2_gt.png",
            "agent_position_policy": "search_traversable_edge_positions_and_maximize_visible_boxes",
            "phases": 2,
        },
    }

    questions_by_task: dict[str, list[dict]] = {task_type: [] for task_type in task_types}
    question_paths: dict[str, list[str]] = {task_type: [] for task_type in task_types}
    for task_type in task_types:
        for q_idx in range(max(0, int(questions_per_task))):
            question_entry = _generate_question_for_task(
                scene=scene,
                scene_name=scene_name,
                room_name=room_name,
                q_idx=q_idx,
                task_type=task_type,
                base_seed=room_seed,
                count_target_candidates=count_target_candidates,
                room_objects=room_objects,
                floor_record=floor_record,
                blockers=blockers,
                room_bbox_xyxy=room_bbox_xyxy,
                room_bbox_info=room_bbox_info,
                agent_pos=agent_pos,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
                render_root=paths["render_root"],
                skip_render=skip_render,
            )
            questions_by_task[task_type].append(question_entry)
            question_paths[task_type].append(
                _write_single_question_json(
                    output_root=paths["question_json_root"],
                    scene_metadata=scene_metadata,
                    task_type=task_type,
                    q_idx=q_idx,
                    entry=question_entry,
                )
            )

    metadata = {
        "scene": scene_name,
        "room": room_name,
        "floor_name": floor_record.name,
        "seed": int(seed),
        "room_seed": int(room_seed),
        "task_types": list(task_types),
        "question_counts": {task_type: len(items) for task_type, items in questions_by_task.items()},
        "question_json_root": paths["question_json_root"],
        "question_json_paths": question_paths,
        "render_root": None if skip_render else paths["render_root"],
        "room_bbox": room_bbox_info,
        "agent_position": [float(v) for v in agent_pos],
        "room_object_summary": bcm._summarize_room_objects(room_objects),
    }
    _write_json(paths["metadata_json"], metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two-phase unobserved box-content change questions.")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name.")
    parser.add_argument("--room", default="living_room_0", help="Room instance name.")
    parser.add_argument("--floor", type=str, default=None, help="Optional floor object name.")
    parser.add_argument("--agent_position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--keys_json", type=str, default="keys.json", help="Path to keys.json.")
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--questions_per_task", type=int, default=DEFAULT_QUESTIONS_PER_TASK)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--task_types",
        nargs="+",
        choices=TASK_TYPES,
        default=list(TASK_TYPES),
        help="Optional subset of tasks to export.",
    )
    parser.add_argument("--skip_render", action="store_true", help="Only export question JSON without rendering images.")
    parser.add_argument(
        "--load_full_scene",
        action="store_true",
        help="Load the full scene instead of restricting to the requested room.",
    )
    parser.add_argument(
        "--disable_runtime_physics",
        action="store_true",
        help="Use direct placement mode for spawned objects.",
    )
    args = parser.parse_args()

    bcm.DIRECT_PLACEMENT_MODE = bool(args.disable_runtime_physics)

    output_root = str(Path(args.output_root) if Path(args.output_root).is_absolute() else SCRIPT_DIR / args.output_root)
    count_target_candidates = bcm._build_count_target_candidates(args.keys_json)
    config = bcm._build_config(
        scene_name=args.scene,
        robot=args.robot,
        load_full_scene=bool(args.load_full_scene),
        room_names=[args.room],
    )

    env = None
    try:
        env = bcm.og.Environment(configs=config)
        bcm._set_viewer_camera_fov()
        scene = env.scene
        wall_records = bcm._collect_wall_records(scene)
        structural_wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]
        summary = _process_room(
            env=env,
            scene=scene,
            scene_name=args.scene,
            room_name=args.room,
            floor_name=args.floor,
            output_root=output_root,
            seed=int(args.seed),
            questions_per_task=int(args.questions_per_task),
            task_types=tuple(args.task_types),
            count_target_candidates=count_target_candidates,
            structural_wall_bboxes=structural_wall_bboxes,
            agent_position=args.agent_position,
            skip_render=bool(args.skip_render),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if env is not None:
            try:
                bcm.og.clear()
            except Exception:
                pass


if __name__ == "__main__":
    main()
