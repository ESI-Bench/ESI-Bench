from __future__ import annotations

"""Passive-GT reconstruction for Temporal Understanding / Agent Observation.

The producer saved the observer camera pose plus robot base XY and orientation
for every GT frame, but it did not save a simulator state at those frames.
In particular, per-frame robot base Z, joint state, and hidden-object state are
absent.
Consequently this module deliberately labels its output as an approximate
semantic snapshot reconstruction; it must not be presented as an exact
saved-state or full-physics replay.

OmniGibson is imported only inside :func:`regenerate_agent_observation_sample`
so the audit and planning helpers remain pure and cheap to unit-test.
"""

import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifest import (
    AVAILABLE_VIEW_STATUSES,
    SCHEMA_VERSION,
    atomic_write_json,
    image_is_materializable,
    inspect_image,
    sha256_file,
)
from .pipeline import load_question, sample_directory
from .views import ViewSpec, extract_view_specs


PROVENANCE = "approximate_snapshot_reconstruction"
FIDELITY = "exact_saved_camera_and_robot_base_xy_orientation_with_intended_object_snapshots"
SQUARE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]
STRICT_STATE_GAPS = (
    "robot_base_z_per_frame",
    "robot_joint_positions_per_frame",
    "robot_joint_velocities_per_frame",
    "hidden_object_pose_and_velocity_per_frame",
    "native_scene_dynamic_state_per_frame",
    "simulator_checkpoint_or_executed_action_trace",
)
SYNC_POSITION_TOLERANCE_M = 0.002
SYNC_ORIENTATION_TOLERANCE_RAD = 0.01
SYNC_AABB_ORIGIN_MARGIN_M = 0.25
ROBOT_SYNC_POSITION_TOLERANCE_M = 0.0005
ROBOT_SYNC_ORIENTATION_TOLERANCE_RAD = 0.001


class AgentReplayMetadataError(ValueError):
    """The saved Agent Observation metadata is internally incomplete."""


class StrictAgentReplayUnavailable(RuntimeError):
    """Raised when a caller requests a strict replay from insufficient state."""


class AgentTransformSyncError(RuntimeError):
    """A teleported entity's live physics/render transform did not synchronize."""


@dataclass(frozen=True)
class AgentFramePlan:
    filename: str
    step: int
    camera_position: tuple[float, float, float]
    camera_quaternion_xyzw: tuple[float, float, float, float]
    robot_position: tuple[float, float, float]
    robot_quaternion_xyzw: tuple[float, float, float, float]
    saved_visibility: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentObjectPlan:
    name: str
    category: str
    model: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    final_position: tuple[float, float, float] | None
    final_drift_m: float | None

    def to_env_config(self) -> dict[str, Any]:
        return {
            "type": "DatasetObject",
            "name": self.name,
            "category": self.category,
            "model": self.model,
            "position": list(self.position),
            "orientation": list(self.quaternion_xyzw),
            "scale": [1.0, 1.0, 1.0],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _authoritative_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Prefer the exact producer JSON joined by ``pipeline.load_question``."""
    auxiliary = payload.get("_gt_auxiliary")
    return auxiliary if isinstance(auxiliary, dict) else payload


def _keyed_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if isinstance(item, dict)}
    if isinstance(value, list):
        return {
            str(item["_key"]): item
            for item in value
            if isinstance(item, dict) and item.get("_key") not in (None, "")
        }
    return {}


def _bool_map(value: Any) -> dict[str, bool]:
    output: dict[str, bool] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            output[str(key)] = bool(item.get("value")) if isinstance(item, dict) and "value" in item else bool(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("_key") not in (None, ""):
                output[str(item["_key"])] = bool(item.get("value"))
    return output


def _key_set(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value}
    if isinstance(value, list):
        return {
            str(item["_key"])
            for item in value
            if isinstance(item, dict) and item.get("_key") not in (None, "")
        }
    return set()


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AgentReplayMetadataError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise AgentReplayMetadataError(f"{label} contains a non-finite value")
    return result


def _live_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    """Convert a runtime tensor/array or a plain sequence into finite floats."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return _vector(value, length, label)


def validate_live_transform(
    *,
    entity_name: str,
    target_position: Any,
    target_quaternion_xyzw: Any,
    live_position: Any,
    live_quaternion_xyzw: Any,
    live_aabb_min: Any,
    live_aabb_max: Any,
    prior_position: Any | None = None,
    prior_aabb_min: Any | None = None,
    prior_aabb_max: Any | None = None,
    position_tolerance_m: float = SYNC_POSITION_TOLERANCE_M,
    orientation_tolerance_rad: float = SYNC_ORIENTATION_TOLERANCE_RAD,
    aabb_origin_margin_m: float = SYNC_AABB_ORIGIN_MARGIN_M,
) -> dict[str, Any]:
    """Validate a synchronized pose and AABB without depending on OmniGibson.

    Quaternion comparison is sign-invariant because ``q`` and ``-q`` encode
    the same rotation.  Requiring the target origin to lie within an expanded
    live AABB catches the common stale-transform failure where pose getters
    update but render bounds remain at the pre-teleport location.
    """
    target_pos = _live_vector(target_position, 3, f"{entity_name}.target_position")
    live_pos = _live_vector(live_position, 3, f"{entity_name}.live_position")
    target_quat = _live_vector(target_quaternion_xyzw, 4, f"{entity_name}.target_quaternion")
    live_quat = _live_vector(live_quaternion_xyzw, 4, f"{entity_name}.live_quaternion")
    aabb_min = _live_vector(live_aabb_min, 3, f"{entity_name}.aabb_min")
    aabb_max = _live_vector(live_aabb_max, 3, f"{entity_name}.aabb_max")

    position_error = math.dist(target_pos, live_pos)
    target_norm = math.sqrt(sum(value * value for value in target_quat))
    live_norm = math.sqrt(sum(value * value for value in live_quat))
    if target_norm <= 1e-12 or live_norm <= 1e-12:
        raise AgentTransformSyncError(f"{entity_name}: zero-norm quaternion in synchronized transform")
    dot = abs(sum(a * b for a, b in zip(target_quat, live_quat)) / (target_norm * live_norm))
    orientation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    if any(lower > upper for lower, upper in zip(aabb_min, aabb_max)):
        raise AgentTransformSyncError(f"{entity_name}: live AABB has min > max")
    aabb_extent = tuple(upper - lower for lower, upper in zip(aabb_min, aabb_max))
    origin_in_aabb = all(
        lower - float(aabb_origin_margin_m) <= value <= upper + float(aabb_origin_margin_m)
        for value, lower, upper in zip(target_pos, aabb_min, aabb_max)
    )
    aabb_max_delta_from_prior = None
    target_translation_from_prior = None
    if prior_position is not None and prior_aabb_min is not None and prior_aabb_max is not None:
        prior_pos = _live_vector(prior_position, 3, f"{entity_name}.prior_position")
        prior_min = _live_vector(prior_aabb_min, 3, f"{entity_name}.prior_aabb_min")
        prior_max = _live_vector(prior_aabb_max, 3, f"{entity_name}.prior_aabb_max")
        target_translation_from_prior = math.dist(target_pos, prior_pos)
        aabb_max_delta_from_prior = max(
            abs(current - previous)
            for current, previous in zip((*aabb_min, *aabb_max), (*prior_min, *prior_max))
        )

    failures = []
    if position_error > float(position_tolerance_m):
        failures.append(f"position error {position_error:.6f} m")
    if orientation_error > float(orientation_tolerance_rad):
        failures.append(f"orientation error {orientation_error:.6f} rad")
    if max(aabb_extent) <= 1e-9:
        failures.append("live AABB is degenerate")
    if not origin_in_aabb:
        failures.append("target origin is outside the synchronized live AABB")
    if (
        target_translation_from_prior is not None
        and target_translation_from_prior > float(position_tolerance_m)
        and aabb_max_delta_from_prior is not None
        and aabb_max_delta_from_prior <= 1e-7
    ):
        failures.append("live AABB did not refresh after a non-trivial teleport")
    if failures:
        raise AgentTransformSyncError(f"{entity_name}: " + "; ".join(failures))
    return {
        "position": list(live_pos),
        "quaternion_xyzw": list(live_quat),
        "aabb_min": list(aabb_min),
        "aabb_max": list(aabb_max),
        "aabb_extent": list(aabb_extent),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
        "target_origin_in_expanded_aabb": origin_in_aabb,
        "target_translation_from_prior_m": target_translation_from_prior,
        "aabb_max_delta_from_prior_m": aabb_max_delta_from_prior,
    }


def observer_step(filename: str) -> int:
    match = re.fullmatch(r"step_(\d+)\.(?:png|jpg|jpeg|webp)", Path(str(filename)).name, flags=re.IGNORECASE)
    if match is None:
        raise AgentReplayMetadataError(f"Unrecognized observer filename: {filename!r}")
    return int(match.group(1))


def build_agent_frame_plan(payload: dict[str, Any]) -> list[AgentFramePlan]:
    """Join every saved observer camera to the same-step saved robot base pose."""
    source = _authoritative_payload(payload)
    observer_poses = _keyed_map(source.get("observer_poses"))
    if not observer_poses:
        raise AgentReplayMetadataError("observer_poses is empty")

    logs: dict[int, dict[str, Any]] = {}
    for item in source.get("step_log") or []:
        if not isinstance(item, dict) or item.get("step") is None:
            continue
        step = int(item["step"])
        if step in logs:
            raise AgentReplayMetadataError(f"step_log contains duplicate step {step}")
        logs[step] = item
    if not logs:
        raise AgentReplayMetadataError("step_log is empty")

    robot_start_z = float(source.get("robot_start_z"))
    if not math.isfinite(robot_start_z):
        raise AgentReplayMetadataError("robot_start_z is not finite")

    output: list[AgentFramePlan] = []
    seen_steps: set[int] = set()
    for filename, camera in sorted(observer_poses.items(), key=lambda pair: observer_step(pair[0])):
        step = observer_step(filename)
        if step in seen_steps:
            raise AgentReplayMetadataError(f"observer_poses contains duplicate step {step}")
        seen_steps.add(step)
        log = logs.get(step)
        if log is None:
            raise AgentReplayMetadataError(f"observer frame {filename!r} has no matching step_log entry")
        robot_xy = _vector(log.get("robot_xy"), 2, f"step_log[{step}].robot_xy")
        output.append(
            AgentFramePlan(
                filename=Path(filename).name,
                step=step,
                camera_position=_vector(camera.get("position"), 3, f"observer_poses[{filename}].position"),
                camera_quaternion_xyzw=_vector(
                    camera.get("quaternion_xyzw"), 4, f"observer_poses[{filename}].quaternion_xyzw"
                ),
                robot_position=(robot_xy[0], robot_xy[1], robot_start_z),
                robot_quaternion_xyzw=_vector(
                    log.get("robot_quat_xyzw"), 4, f"step_log[{step}].robot_quat_xyzw"
                ),
                saved_visibility=_bool_map(camera.get("visibility")),
            )
        )
    return output


def build_agent_object_plan(payload: dict[str, Any]) -> list[AgentObjectPlan]:
    """Build the intended, count-consistent hidden-object state.

    ``placed_objects`` is the exact position table used by the producer's
    proximity ground truth.  ``objects_meta`` was collected only after the
    navigation and can contain physics drift, so its final pose must not be
    mistaken for a per-frame GT pose.  It is used here only for immutable asset
    identity and for measuring that drift.
    """
    source = _authoritative_payload(payload)
    # The exact producer-side auxiliary JSON retains ``placed_objects`` but an
    # older exporter omitted the model field from its ``objects_meta``.  The
    # cleaned benchmark row restores that immutable model identity.  Merge the
    # two exact-name tables instead of discarding either source.
    source_meta = _keyed_map(source.get("objects_meta"))
    clean_meta = _keyed_map(payload.get("objects_meta"))
    objects_meta = {
        name: {**source_meta.get(name, {}), **clean_meta.get(name, {})}
        for name in set(source_meta) | set(clean_meta)
    }
    placed_objects = {
        str(item.get("name")): item
        for item in source.get("placed_objects") or []
        if isinstance(item, dict) and item.get("name") not in (None, "")
    }
    category_default = str(source.get("object_category") or "")
    output: list[AgentObjectPlan] = []
    for name, placed in placed_objects.items():
        # Failed placements remained outside the scene and were excluded from
        # the producer's proximity table.  Omitting them is semantically
        # equivalent and avoids importing a meaningless remote/fallen pose.
        if not placed.get("position") or str(placed.get("placed_on") or "").lower() == "failed":
            continue
        meta = objects_meta.get(name) or {}
        model = str(meta.get("model") or "")
        category = str(meta.get("category") or category_default)
        if not model or not category:
            raise AgentReplayMetadataError(f"Placed object {name!r} is missing category/model identity")
        intended = _vector(placed.get("position"), 3, f"placed_objects[{name}].position")
        final_position = None
        final_drift = None
        if meta.get("position"):
            final_position = _vector(meta.get("position"), 3, f"objects_meta[{name}].position")
            final_drift = math.dist(intended, final_position)
        output.append(
            AgentObjectPlan(
                name=name,
                category=category,
                model=model,
                position=intended,
                # The placement producer explicitly used SQUARE_ORI here but
                # did not save the post-settle per-frame quaternion.
                quaternion_xyzw=tuple(SQUARE_ORIENTATION),
                final_position=final_position,
                final_drift_m=final_drift,
            )
        )
    return output


def validate_agent_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the producer's proximity labels from its saved intent table."""
    source = _authoritative_payload(payload)
    logs = [item for item in source.get("step_log") or [] if isinstance(item, dict) and item.get("robot_xy")]
    if not logs:
        raise AgentReplayMetadataError("Cannot validate semantics without robot_xy step_log entries")
    robot_xy = [_vector(item["robot_xy"], 2, "step_log.robot_xy") for item in logs]
    threshold = float(source.get("proximity_thresh", 1.0))
    intended_positions = {
        str(item.get("name")): _vector(item["position"], 3, f"placed_objects[{item.get('name')}].position")
        for item in source.get("placed_objects") or []
        if isinstance(item, dict)
        and item.get("name") not in (None, "")
        and item.get("position")
        and str(item.get("placed_on") or "").lower() != "failed"
    }
    minimum_distances = {
        name: min(math.dist(position[:2], xy) for xy in robot_xy)
        for name, position in intended_positions.items()
    }
    recomputed = {name for name, distance in minimum_distances.items() if distance < threshold}
    saved = _key_set(source.get("passed_objects"))
    if recomputed != saved:
        raise AgentReplayMetadataError(
            "Saved passed_objects disagrees with step_log + placed_objects: "
            f"saved={sorted(saved)}, recomputed={sorted(recomputed)}"
        )
    true_count = int(source.get("true_count"))
    if true_count != len(saved):
        raise AgentReplayMetadataError(f"true_count={true_count} but passed_objects has {len(saved)} entries")
    return {
        "validated": True,
        "proximity_threshold_m": threshold,
        "true_count": true_count,
        "passed_objects": sorted(saved),
        "minimum_distances_m": minimum_distances,
    }


def audit_agent_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an evidence-based strictness audit without importing OmniGibson."""
    source = _authoritative_payload(payload)
    try:
        frames = build_agent_frame_plan(payload)
        objects = build_agent_object_plan(payload)
        semantics = validate_agent_semantics(payload)
        error = None
    except (AgentReplayMetadataError, TypeError, ValueError) as exc:
        frames, objects, semantics = [], [], {"validated": False}
        error = f"{type(exc).__name__}: {exc}"
    drifts = [item.final_drift_m for item in objects if item.final_drift_m is not None]
    final_pose_count_matches_gt = None
    if error is None and objects:
        source_logs = [
            _vector(item["robot_xy"], 2, "step_log.robot_xy")
            for item in source.get("step_log") or []
            if isinstance(item, dict) and item.get("robot_xy")
        ]
        threshold = float(source.get("proximity_thresh", 1.0))
        final_passed = {
            item.name
            for item in objects
            if item.final_position is not None
            and min(math.dist(item.final_position[:2], xy) for xy in source_logs) < threshold
        }
        final_pose_count_matches_gt = final_passed == _key_set(source.get("passed_objects"))
    failed_placements = sum(
        1
        for item in source.get("placed_objects") or []
        if isinstance(item, dict)
        and (not item.get("position") or str(item.get("placed_on") or "").lower() == "failed")
    )
    return {
        "strictly_reconstructable": False,
        "best_available_reconstructable": error is None,
        "provenance": PROVENANCE,
        "fidelity": FIDELITY,
        "frame_count": len(frames),
        "intended_object_count": len(objects),
        "failed_placement_count": failed_placements,
        "exact_components": [
            "observer_camera_pose_per_frame",
            "robot_base_xy_and_orientation_per_frame",
            "hidden_object_asset_identity",
            "ground_truth_proximity_count",
        ],
        "derived_components": ["robot_base_z_from_saved_initial_robot_start_z"],
        "missing_strict_state": list(STRICT_STATE_GAPS),
        "object_pose_policy": "placed_objects intended positions; objects_meta final poses are not per-frame state",
        "objects_with_final_drift_over_1cm": sum(value > 0.01 for value in drifts),
        "objects_with_final_drift_over_1m": sum(value > 1.0 for value in drifts),
        "maximum_final_drift_m": max(drifts, default=0.0),
        "final_objects_meta_preserves_ground_truth_passed_set": final_pose_count_matches_gt,
        "semantic_validation": semantics,
        "error": error,
    }


def _load_or_initialize_record(
    row: dict[str, Any],
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    sample_dir: Path,
    specs: list[ViewSpec],
) -> dict[str, Any]:
    manifest_path = sample_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        if isinstance(record, dict) and record.get("views"):
            return record
    try:
        source_label = str(source_json.relative_to(dataset_root))
    except ValueError:
        source_label = str(source_json)
    views = []
    for spec in specs:
        view = spec.to_dict()
        view.update({"output_path": str((sample_dir / spec.view_id).relative_to(output_root)), "status": "planned"})
        views.append(view)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": str(row.get("id") or ""),
        "big_task": row.get("big_task"),
        "small_task": row.get("small_task"),
        "runner_task": row.get("runner_task"),
        "scene": row.get("scene"),
        "room": row.get("room"),
        "source_json": source_label,
        "sample_directory": str(sample_dir.relative_to(output_root)),
        "status": "planned",
        "issues": [],
        "views": views,
    }


def _build_exact_generator_config(og, yaml, source: dict[str, Any], objects: list[AgentObjectPlan]) -> dict[str, Any]:
    config_path = Path(og.example_config_path) / "r1_primitives.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    scene = config.setdefault("scene", {})
    scene["scene_model"] = source["scene"]
    scene["not_load_object_categories"] = ["ceilings", "carpet"]
    scene["load_room_instances"] = [source["room"]]
    config["objects"] = [item.to_env_config() for item in objects]
    return config


def _scene_objects(scene) -> list[Any]:
    objects = getattr(scene, "objects", [])
    return list(objects.values()) if isinstance(objects, dict) else list(objects)


def _remove_ceiling_like_original(env, og) -> list[str]:
    """Mirror the producer's redundant object- and stage-level ceiling removal."""
    removed: list[str] = []
    was_playing = bool(og.sim.is_playing())
    if was_playing:
        og.sim.stop()
    prefix = "/World/scene_0/ceilings_"
    for obj in _scene_objects(env.scene):
        name = str(getattr(obj, "name", ""))
        prim_path = str(getattr(obj, "prim_path", None) or getattr(obj, "_prim_path", None) or "")
        if not (prim_path.startswith(prefix) or "ceilings_" in name.lower()):
            continue
        try:
            env.scene.remove_object(obj)
            removed.append(name)
        except Exception:
            pass
    for prim in og.sim.stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith(prefix) and path.count("/") == 3:
            prim.SetActive(False)
    if was_playing:
        og.sim.play()
    return removed


def _remove_door_like_objects(env) -> list[str]:
    removed: list[str] = []
    for obj in _scene_objects(env.scene):
        name = str(getattr(obj, "name", ""))
        category = str(getattr(obj, "category", ""))
        if not any(word in name.lower() or word in category.lower() for word in ("door", "gate", "hatch")):
            continue
        try:
            env.scene.remove_object(obj)
            removed.append(name)
        except Exception:
            pass
    return removed


def _read_entity_live_transform(entity: Any, entity_name: str) -> dict[str, tuple[float, ...]]:
    position, quaternion = entity.get_position_orientation()
    aabb_min, aabb_max = entity.aabb
    return {
        "position": _live_vector(position, 3, f"{entity_name}.position"),
        "quaternion_xyzw": _live_vector(quaternion, 4, f"{entity_name}.quaternion_xyzw"),
        "aabb_min": _live_vector(aabb_min, 3, f"{entity_name}.aabb_min"),
        "aabb_max": _live_vector(aabb_max, 3, f"{entity_name}.aabb_max"),
    }


def _synchronize_entities(
    entries: list[tuple[str, Any, tuple[float, ...], tuple[float, ...]]],
    *,
    og: Any,
    th: Any,
    make_visual_only: bool,
    max_attempts: int = 2,
    position_tolerance_m: float = SYNC_POSITION_TOLERANCE_M,
    orientation_tolerance_rad: float = SYNC_ORIENTATION_TOLERANCE_RAD,
) -> dict[str, dict[str, Any]]:
    """Teleport, step once to sync render state, then fail closed on mismatch."""
    if not entries:
        return {}
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if make_visual_only:
        for name, entity, _position, _quaternion in entries:
            try:
                entity.visual_only = True
            except Exception as exc:
                raise AgentTransformSyncError(f"{name}: could not enable visual_only: {exc}") from exc
            if not bool(getattr(entity, "visual_only", False)):
                raise AgentTransformSyncError(f"{name}: visual_only did not become true")

    # Keep the pre-teleport bounds across retries.  A pose getter may update
    # while a stale renderer AABB remains byte-for-byte at this old location.
    prior_by_name = {
        name: _read_entity_live_transform(entity, name)
        for name, entity, _position, _quaternion in entries
    }

    failures: dict[str, str] = {}
    for attempt in range(1, max_attempts + 1):
        for _name, entity, position, quaternion in entries:
            entity.set_position_orientation(
                position=th.tensor(position, dtype=th.float32),
                orientation=th.tensor(quaternion, dtype=th.float32),
            )
            try:
                entity.keep_still()
            except Exception:
                pass
        # A render-only loop may retain the previous USD/render transform.  One
        # simulator step is the minimum synchronization barrier observed to
        # refresh both pose-backed rendering and AABBs.
        og.sim.step()

        reports: dict[str, dict[str, Any]] = {}
        failures = {}
        for name, entity, position, quaternion in entries:
            live = _read_entity_live_transform(entity, name)
            try:
                prior = prior_by_name[name]
                report = validate_live_transform(
                    entity_name=name,
                    target_position=position,
                    target_quaternion_xyzw=quaternion,
                    live_position=live["position"],
                    live_quaternion_xyzw=live["quaternion_xyzw"],
                    live_aabb_min=live["aabb_min"],
                    live_aabb_max=live["aabb_max"],
                    prior_position=prior["position"],
                    prior_aabb_min=prior["aabb_min"],
                    prior_aabb_max=prior["aabb_max"],
                    position_tolerance_m=position_tolerance_m,
                    orientation_tolerance_rad=orientation_tolerance_rad,
                )
            except AgentTransformSyncError as exc:
                failures[name] = str(exc)
                continue
            report.update(
                {
                    "visual_only": bool(getattr(entity, "visual_only", False)),
                    "sync_attempt": attempt,
                    "sim_steps_for_sync": attempt,
                }
            )
            reports[name] = report
        if not failures:
            return reports
    raise AgentTransformSyncError(
        "Entity transform synchronization failed after "
        f"{max_attempts} attempts: {json.dumps(failures, sort_keys=True)}"
    )


def _restore_intended_objects(
    env: Any,
    objects: list[AgentObjectPlan],
    th: Any,
    og: Any,
) -> dict[str, dict[str, Any]]:
    entries = []
    for item in objects:
        obj = env.scene.object_registry("name", item.name)
        if obj is None:
            raise RuntimeError(f"Agent reconstruction object {item.name!r} did not load")
        entries.append((item.name, obj, item.position, item.quaternion_xyzw))
    return _synchronize_entities(entries, og=og, th=th, make_visual_only=True)


def _synchronize_robot_frame(robot: Any, frame: AgentFramePlan, th: Any, og: Any) -> dict[str, Any]:
    reports = _synchronize_entities(
        [("robot", robot, frame.robot_position, frame.robot_quaternion_xyzw)],
        og=og,
        th=th,
        # The reconstruction is a sequence of independent visual snapshots.
        # Freezing the robot prevents gravity/controller state from moving the
        # exact saved XY/orientation during the required synchronization step.
        make_visual_only=True,
        position_tolerance_m=ROBOT_SYNC_POSITION_TOLERANCE_M,
        orientation_tolerance_rad=ROBOT_SYNC_ORIENTATION_TOLERANCE_RAD,
    )
    return reports["robot"]


def regenerate_agent_observation_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    allow_approximate: bool = True,
) -> dict[str, Any]:
    """Render one sample from exact cameras/base XY-orientations and intended objects.

    This is the best metadata-supported reconstruction, not a strict replay.
    Pass ``allow_approximate=False`` to turn that distinction into a hard
    failure.  The function is intentionally standalone so the main dispatcher
    can opt into it without changing this module.
    """
    source_json = Path(source_json)
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    row, payload = load_question(source_json)
    if row.get("small_task") != "Agent Observation":
        raise ValueError("regenerate_agent_observation_sample received a non-agent sample")
    audit = audit_agent_replay_payload(payload)
    if not audit["best_available_reconstructable"]:
        raise AgentReplayMetadataError(str(audit.get("error") or "Agent metadata audit failed"))
    if not allow_approximate:
        raise StrictAgentReplayUnavailable(
            "Agent Observation lacks per-frame robot Z, articulated, object, and world state; "
            "only approximate semantic snapshot reconstruction is possible"
        )

    source = _authoritative_payload(payload)
    frames = build_agent_frame_plan(payload)
    objects = build_agent_object_plan(payload)
    semantics = validate_agent_semantics(payload)
    specs = extract_view_specs(payload)
    if len(specs) != len(frames):
        raise AgentReplayMetadataError(f"View selector produced {len(specs)} views for {len(frames)} saved frames")
    plans_by_name = {item.filename: item for item in frames}
    specs_by_name = {Path(item.original_image_path).name: item for item in specs}
    if set(plans_by_name) != set(specs_by_name):
        raise AgentReplayMetadataError("Selected Agent views do not match observer_poses filenames")

    sample_dir = sample_directory(row, source_json, dataset_root, output_root)
    record = _load_or_initialize_record(row, source_json, dataset_root, output_root, sample_dir, specs)
    destinations = {spec.view_id: sample_dir / spec.view_id for spec in specs}
    pending_specs = [
        spec
        for spec in specs
        if overwrite
        or not destinations[spec.view_id].is_file()
        or not image_is_materializable(inspect_image(destinations[spec.view_id]))
    ]
    if not pending_specs:
        return {
            "source_json": str(source_json),
            "rendered": 0,
            "failed": 0,
            "skipped": "already_complete",
            "provenance": PROVENANCE,
        }

    # Keep the runtime import local: audit/tests never initialize Kit or touch a
    # GPU.  Match the renderer's constrained startup environment.
    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    fs_watcher_arg = "--/app/extensions/fsWatcherEnabled=false"
    if fs_watcher_arg not in sys.argv:
        sys.argv.append(fs_watcher_arg)
    from .replay import _ensure_active_imports, capture_image, configure_viewer_resolution

    og, th, yaml, cv2, _module_for_payload = _ensure_active_imports()
    config = _build_exact_generator_config(og, yaml, source, objects)
    config = configure_viewer_resolution(config)
    work_dir = sample_dir / f".agent_regeneration_{os.getpid()}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = og.Environment(configs=config)
        try:
            og.sim._viewer_camera.add_modality("rgb")
        except Exception:
            pass
        camera = og.sim._viewer_camera
        removed_ceilings = _remove_ceiling_like_original(env, og)
        removed_doors = _remove_door_like_objects(env)

        # The original navigation continuously sent zero commands to every
        # non-base controller.  This warm-up makes the unsaved articulation as
        # close as metadata permits; it does not make it an exact saved state.
        idle = th.zeros(env.robots[0].action_dim, dtype=th.float32)
        for _ in range(100):
            env.step(idle)
        restored_objects = _restore_intended_objects(env, objects, th, og)

        robot_sync_by_view_id: dict[str, dict[str, Any]] = {}
        for spec in pending_specs:
            frame = plans_by_name[Path(spec.original_image_path).name]
            robot = env.robots[0]
            robot_sync_by_view_id[spec.view_id] = _synchronize_robot_frame(robot, frame, th, og)
            camera.set_position_orientation(
                position=th.tensor(frame.camera_position, dtype=th.float32),
                orientation=th.tensor(frame.camera_quaternion_xyzw, dtype=th.float32),
            )
            # The producer rendered five times before reading the RGB buffer.
            capture_image(og, cv2, work_dir / spec.view_id, render_steps=5)

        views_by_id = {str(view.get("view_id")): view for view in record.get("views") or []}
        rendered = 0
        warnings = 0
        for spec in pending_specs:
            source_image = work_dir / spec.view_id
            quality = inspect_image(source_image)
            if not image_is_materializable(quality):
                raise RuntimeError(f"Agent frame is not materializable: {source_image}: {quality}")
            destination = destinations[spec.view_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_image, destination)
            view = views_by_id.get(spec.view_id)
            if view is None:
                view = spec.to_dict()
                record.setdefault("views", []).append(view)
            frame = plans_by_name[Path(spec.original_image_path).name]
            view.update(
                {
                    "output_path": str(destination.relative_to(output_root)),
                    "status": "rendered_pose" if quality.get("valid") else "rendered_pose_warning",
                    "provenance": PROVENANCE,
                    "pose_source": (
                        "exact /observer_poses camera + same-step /step_log robot XY/orientation; "
                        "robot Z derived from /robot_start_z"
                    ),
                    "sha256": sha256_file(destination),
                    "quality": quality,
                    "width": quality.get("width"),
                    "height": quality.get("height"),
                    "render_details": {
                        "environment": "metadata_semantic_snapshot_reconstruction",
                        "fidelity": FIDELITY,
                        "strictly_reconstructable": False,
                        "missing_strict_state": list(STRICT_STATE_GAPS),
                        "frame_step": frame.step,
                        "object_pose_source": "/placed_objects (producer proximity-GT intent)",
                        "final_objects_meta_not_used_as_frame_pose": True,
                        "restored_objects": sorted(restored_objects),
                        "restored_object_live_state": restored_objects,
                        "robot_live_state": robot_sync_by_view_id[spec.view_id],
                        "sim_step_transform_sync_validated": True,
                        "removed_ceiling_like_objects": removed_ceilings,
                        "removed_door_like_objects": removed_doors,
                        "resolution": [1280, 720],
                        "semantic_count_validated": True,
                    },
                }
            )
            rendered += 1
            warnings += int(not quality.get("valid"))
        issue_codes = {str(item.get("code")) for item in record.get("issues") or [] if isinstance(item, dict)}
        if "agent_dynamic_state_unavailable" not in issue_codes:
            record.setdefault("issues", []).append(
                {
                    "code": "agent_dynamic_state_unavailable",
                    "message": (
                        "Saved cameras and robot-base XY/orientations are exact, but per-frame robot Z/joints, "
                        "object dynamics, and world state were not saved; output is a semantic snapshot "
                        "reconstruction."
                    ),
                }
            )
        statuses = [view.get("status") for view in record.get("views") or []]
        record["status"] = (
            "complete" if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses) else "partial"
        )
        result = {
            "provenance": PROVENANCE,
            "strictly_reconstructable": False,
            "frame_count": len(frames),
            "semantic_validation": semantics,
            "audit": audit,
            "runtime_sync_validation": {
                "hidden_objects": restored_objects,
                "robot_frames_validated": len(robot_sync_by_view_id),
                "robot_max_position_error_m": max(
                    (item["position_error_m"] for item in robot_sync_by_view_id.values()),
                    default=0.0,
                ),
                "robot_max_orientation_error_rad": max(
                    (item["orientation_error_rad"] for item in robot_sync_by_view_id.values()),
                    default=0.0,
                ),
            },
        }
        atomic_write_json(sample_dir / "regeneration_result.json", result)
        atomic_write_json(sample_dir / "manifest.json", record)
        return {
            "source_json": str(source_json),
            "rendered": rendered,
            "warnings": warnings,
            "failed": 0,
            "provenance": PROVENANCE,
            "strictly_reconstructable": False,
            "semantic_count_validated": True,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            if getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass
