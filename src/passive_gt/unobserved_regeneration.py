from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from .manifest import (
    AVAILABLE_VIEW_STATUSES,
    SCHEMA_VERSION,
    atomic_write_json,
    image_is_materializable,
    inspect_image,
    sha256_file,
)
from .pipeline import load_question, sample_directory
from .views import extract_view_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTING_GENERATOR = REPO_ROOT / "src" / "dataset_generation" / "task_counting" / "batch_counting_merge.py"
UNOBSERVED_GENERATOR = (
    REPO_ROOT / "src" / "dataset_generation" / "task_unobserved_changes" / "batch_unobserved_changes.py"
)

# This deliberately differs from a generic static-scene replay.  The camera is
# saved verbatim, while the two content states are reconstructed with the
# original ESI-Bench generator's exact fixed-model placement helpers.
PROVENANCE = "saved_gt_camera_deterministic_generator_replay"
KNOWN_IDENTITY_CONTAINER_ASSETS = {
    ("carton", "cdmmwy"),
    ("cedar_chest", "fwstpx"),
    ("cedar_chest", "gbdzls"),
}
PHASE_KEYS = ("image1", "image2")
CONTENT_NAME_PREFIX = "render_unobserved_content_"


class InsufficientUnobservedMetadata(ValueError):
    """Raised when a strict saved-view replay would require guessing state."""


@dataclass(frozen=True)
class ContentReplaySpec:
    category: str
    model: str
    bbox_size_m: tuple[float, float, float]
    display_name: str | None = None
    sampling_source: str | None = None

    def runtime_payload(self) -> dict[str, Any]:
        """Return the field names consumed by the original phase spawner."""
        return {
            "category": self.category,
            "representative_model": self.model,
            "bbox_size_m": list(self.bbox_size_m),
            "display_name": self.display_name,
            "sampling_source": self.sampling_source,
        }


@dataclass(frozen=True)
class BoxReplaySpec:
    box_index: int
    position_label: str
    change_type: str
    container_name: str
    container_category: str
    container_model: str
    container_position: tuple[float, float, float]
    container_orientation_xyzw: tuple[float, float, float, float]
    container_bbox_min: tuple[float, float, float]
    container_bbox_max: tuple[float, float, float]
    placement_type: str
    phase1_content: ContentReplaySpec | None
    phase2_content: ContentReplaySpec | None

    def content_for_phase(self, phase: str) -> ContentReplaySpec | None:
        if phase == "image1":
            return self.phase1_content
        if phase == "image2":
            return self.phase2_content
        raise KeyError(f"Unknown Unobserved Change phase: {phase}")


@dataclass(frozen=True)
class SavedGTView:
    view_id: str
    phase: str
    box_index: int
    image_path: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    look_target: tuple[float, float, float]
    fov_deg: float
    view_direction: str
    container_name: str
    metadata_pointer: str


@dataclass(frozen=True)
class MissingGTCapture:
    phase: str
    box_index: int
    container_name: str
    reason: str = "original_gt_capture_missing_camera_and_image"


@dataclass(frozen=True)
class UnobservedReplayPlan:
    scene: str
    room: str
    task_type: str
    question_index: int
    seed: int
    boxes: tuple[BoxReplaySpec, ...]
    views: tuple[SavedGTView, ...]
    missing_captures: tuple[MissingGTCapture, ...]

    @property
    def theoretical_capture_count(self) -> int:
        return len(PHASE_KEYS) * len(self.boxes)

    @property
    def saved_capture_count(self) -> int:
        return len(self.views)

    @property
    def saved_view_metadata_replayable(self) -> bool:
        # Construction already validated every saved camera and every state
        # field needed by the original placement helper.  This does not claim
        # pixel identity or that an unsaved whole-scene runtime snapshot exists.
        return bool(self.views)

    @property
    def original_capture_complete(self) -> bool:
        return not self.missing_captures

    def audit_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "room": self.room,
            "task_type": self.task_type,
            "question_index": self.question_index,
            "seed": self.seed,
            "box_count": len(self.boxes),
            "theoretical_capture_count": self.theoretical_capture_count,
            "saved_capture_count": self.saved_capture_count,
            "missing_capture_count": len(self.missing_captures),
            "saved_view_metadata_replayable": self.saved_view_metadata_replayable,
            "original_capture_complete": self.original_capture_complete,
            "missing_captures": [asdict(item) for item in self.missing_captures],
            "provenance": PROVENANCE,
        }


def _float_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InsufficientUnobservedMetadata(f"{field} must contain {length} numeric values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InsufficientUnobservedMetadata(f"{field} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise InsufficientUnobservedMetadata(f"{field} contains a non-finite value")
    return result


def _keyed_items(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        return [(str(key), item) for key, item in value.items() if isinstance(item, dict)]
    if not isinstance(value, list):
        return []
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        output.append((str(item.get("_key", index)), item))
    return output


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InsufficientUnobservedMetadata(f"{field} is missing")
    return text


def _content_spec(value: Any, field: str) -> ContentReplaySpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InsufficientUnobservedMetadata(f"{field} must be an object or null")
    return ContentReplaySpec(
        category=_required_text(value.get("category"), f"{field}.category"),
        model=_required_text(value.get("model"), f"{field}.model"),
        bbox_size_m=_float_tuple(value.get("bbox_size_m"), 3, f"{field}.bbox_size_m"),
        display_name=str(value.get("display_name") or "").strip() or None,
        sampling_source=str(value.get("sampling_source") or "").strip() or None,
    )


def _container_orientation(category: str, model: str) -> tuple[float, float, float, float]:
    asset = (category, model)
    if asset not in KNOWN_IDENTITY_CONTAINER_ASSETS:
        raise InsufficientUnobservedMetadata(
            "Container orientation was not serialized and cannot be inferred safely for "
            f"{category}/{model}. The audited generator uses identity only for {sorted(KNOWN_IDENTITY_CONTAINER_ASSETS)}."
        )
    return (0.0, 0.0, 0.0, 1.0)


def _box_spec(value: Any, fallback_index: int) -> BoxReplaySpec:
    if not isinstance(value, dict):
        raise InsufficientUnobservedMetadata(f"question_data.boxes[{fallback_index}] is not an object")
    box_index = int(value.get("box_index", fallback_index))
    container = value.get("container") or {}
    if not isinstance(container, dict):
        raise InsufficientUnobservedMetadata(f"box {box_index} container is not an object")
    category = _required_text(container.get("category"), f"box {box_index} container.category")
    model = _required_text(container.get("model"), f"box {box_index} container.model")
    placement = container.get("placement") or {}
    if not isinstance(placement, dict):
        raise InsufficientUnobservedMetadata(f"box {box_index} container.placement is not an object")
    bbox = container.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 2:
        raise InsufficientUnobservedMetadata(f"box {box_index} container.bbox must contain min and max")
    bbox_min = _float_tuple(bbox[0], 3, f"box {box_index} container.bbox[0]")
    bbox_max = _float_tuple(bbox[1], 3, f"box {box_index} container.bbox[1]")
    if any(high <= low for low, high in zip(bbox_min, bbox_max)):
        raise InsufficientUnobservedMetadata(f"box {box_index} container.bbox has a non-positive extent")
    return BoxReplaySpec(
        box_index=box_index,
        position_label=_required_text(value.get("position_label"), f"box {box_index} position_label"),
        change_type=_required_text(value.get("change_type"), f"box {box_index} change_type"),
        container_name=_required_text(container.get("name"), f"box {box_index} container.name"),
        container_category=category,
        container_model=model,
        container_position=_float_tuple(placement.get("position"), 3, f"box {box_index} container.placement.position"),
        container_orientation_xyzw=_container_orientation(category, model),
        container_bbox_min=bbox_min,
        container_bbox_max=bbox_max,
        placement_type=_required_text(placement.get("placement_type"), f"box {box_index} placement_type"),
        phase1_content=_content_spec(value.get("phase1_content"), f"box {box_index} phase1_content"),
        phase2_content=_content_spec(value.get("phase2_content"), f"box {box_index} phase2_content"),
    )


def _saved_view(
    item: dict[str, Any],
    *,
    phase: str,
    item_index: int,
    phase_fov: float,
    boxes_by_index: dict[int, BoxReplaySpec],
) -> SavedGTView:
    box_index = int(item.get("box_index", -1))
    box = boxes_by_index.get(box_index)
    if box is None:
        raise InsufficientUnobservedMetadata(f"{phase} GT view references unknown box_index={box_index}")
    container_name = _required_text(item.get("container_name"), f"{phase} view {item_index} container_name")
    if container_name != box.container_name:
        raise InsufficientUnobservedMetadata(
            f"{phase} box {box_index} container mismatch: view={container_name}, box={box.container_name}"
        )
    image_path = _required_text(item.get("image_path"), f"{phase} box {box_index} image_path")
    image_name = Path(image_path.replace("\\", "/")).name
    if not image_name:
        raise InsufficientUnobservedMetadata(f"{phase} box {box_index} image_path has no filename")
    pose = item.get("camera_pose") or {}
    if not isinstance(pose, dict):
        raise InsufficientUnobservedMetadata(f"{phase} box {box_index} camera_pose is not an object")
    return SavedGTView(
        view_id=f"gt_view/{phase}/{image_name}",
        phase=phase,
        box_index=box_index,
        image_path=image_path,
        position=_float_tuple(pose.get("position"), 3, f"{phase} box {box_index} camera position"),
        quaternion_xyzw=_float_tuple(
            pose.get("quaternion_xyzw"), 4, f"{phase} box {box_index} camera quaternion_xyzw"
        ),
        look_target=_float_tuple(item.get("look_target"), 3, f"{phase} box {box_index} look_target"),
        fov_deg=float(item.get("fov_deg") or phase_fov),
        view_direction=_required_text(item.get("view_direction"), f"{phase} box {box_index} view_direction"),
        container_name=container_name,
        metadata_pointer=f"/question_data/render/gt_view/{phase}/images/{item_index}",
    )


def build_unobserved_replay_plan(
    payload: dict[str, Any],
    *,
    validate_selector: bool = True,
) -> UnobservedReplayPlan:
    """Validate and extract the exact state/camera inputs for one sample.

    Only views that the original generator successfully saved are replayable as
    strict GT.  Empty phase/box slots are reported in ``missing_captures`` and
    are never assigned a newly invented camera.
    """
    question_data = payload.get("question_data") or {}
    if not isinstance(question_data, dict):
        raise InsufficientUnobservedMetadata("question_data is missing")
    task_type = _required_text(question_data.get("task_type") or payload.get("task_type"), "task_type")
    boxes = tuple(
        sorted(
            (_box_spec(value, index) for index, value in enumerate(question_data.get("boxes") or [])),
            key=lambda item: item.box_index,
        )
    )
    if not boxes:
        raise InsufficientUnobservedMetadata("question_data.boxes is empty")
    if [box.box_index for box in boxes] != list(range(len(boxes))):
        raise InsufficientUnobservedMetadata("box_index values must be contiguous from zero")
    boxes_by_index = {box.box_index: box for box in boxes}

    gt_view = ((question_data.get("render") or {}).get("gt_view") or {})
    if not isinstance(gt_view, dict):
        raise InsufficientUnobservedMetadata("question_data.render.gt_view is missing")
    views: list[SavedGTView] = []
    missing: list[MissingGTCapture] = []
    for phase in PHASE_KEYS:
        phase_record = gt_view.get(phase) or {}
        if not isinstance(phase_record, dict):
            raise InsufficientUnobservedMetadata(f"gt_view.{phase} is not an object")
        phase_fov = float(phase_record.get("fov_deg") or 55.0)
        phase_views = [
            _saved_view(
                item,
                phase=phase,
                item_index=item_index,
                phase_fov=phase_fov,
                boxes_by_index=boxes_by_index,
            )
            for item_index, (_key, item) in enumerate(_keyed_items(phase_record.get("images")))
        ]
        seen_box_indices = {view.box_index for view in phase_views}
        if len(seen_box_indices) != len(phase_views):
            raise InsufficientUnobservedMetadata(f"gt_view.{phase} has duplicate box captures")
        views.extend(phase_views)
        for box in boxes:
            if box.box_index not in seen_box_indices:
                missing.append(
                    MissingGTCapture(
                        phase=phase,
                        box_index=box.box_index,
                        container_name=box.container_name,
                    )
                )

    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise InsufficientUnobservedMetadata("GT view filenames are not unique within their phase")
    if validate_selector:
        # Rule-backed fallback cameras are valid inputs for a fresh GT state
        # replay, but they are not historical captures and therefore do not
        # belong in this saved-capture reconstruction plan.
        selector_ids = {
            spec.view_id
            for spec in extract_view_specs(payload)
            if spec.role != "gt_rule_camera"
        }
        if selector_ids != set(view_ids):
            raise InsufficientUnobservedMetadata(
                "Replay plan does not match the Passive-GT selector: "
                f"metadata_only={sorted(set(view_ids) - selector_ids)}, selector_only={sorted(selector_ids - set(view_ids))}"
            )

    question_index_value = question_data.get("candidate_index", payload.get("question_index", 0))
    return UnobservedReplayPlan(
        scene=_required_text(payload.get("scene"), "scene"),
        room=_required_text(payload.get("room"), "room"),
        task_type=task_type,
        question_index=int(question_index_value or 0),
        seed=int(payload.get("seed", 7) or 7),
        boxes=boxes,
        views=tuple(views),
        missing_captures=tuple(missing),
    )


def summarize_unobserved_plans(plans: Iterable[UnobservedReplayPlan]) -> dict[str, Any]:
    items = list(plans)
    return {
        "samples": len(items),
        "boxes": sum(len(plan.boxes) for plan in items),
        "theoretical_captures": sum(plan.theoretical_capture_count for plan in items),
        "saved_gt_captures": sum(plan.saved_capture_count for plan in items),
        "missing_original_captures": sum(len(plan.missing_captures) for plan in items),
        "samples_with_missing_original_captures": sum(bool(plan.missing_captures) for plan in items),
        "samples_with_replayable_saved_views": sum(plan.saved_view_metadata_replayable for plan in items),
        "provenance": PROVENANCE,
    }


_GENERATOR_MODULES: tuple[ModuleType, ModuleType] | None = None


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generator module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_generator_modules() -> tuple[ModuleType, ModuleType]:
    """Import ESI-Bench's generators lazily; importing this module stays CPU-only."""
    global _GENERATOR_MODULES
    if _GENERATOR_MODULES is not None:
        return _GENERATOR_MODULES
    if not COUNTING_GENERATOR.is_file() or not UNOBSERVED_GENERATOR.is_file():
        raise FileNotFoundError(
            f"Missing ESI-Bench generators: counting={COUNTING_GENERATOR}, unobserved={UNOBSERVED_GENERATOR}"
        )
    counting = _load_module(COUNTING_GENERATOR, "passive_gt_unobserved_counting_generator")
    previous = sys.modules.get("batch_counting_merge")
    sys.modules["batch_counting_merge"] = counting
    try:
        unobserved = _load_module(UNOBSERVED_GENERATOR, "passive_gt_unobserved_generator")
    finally:
        if previous is None:
            sys.modules.pop("batch_counting_merge", None)
        else:
            sys.modules["batch_counting_merge"] = previous
    _GENERATOR_MODULES = counting, unobserved
    return _GENERATOR_MODULES


def _max_abs_difference(left: Iterable[float], right: Iterable[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _tensor_values(value: Any) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(item) for item in value)


def _quaternion_component_error(left: Iterable[float], right: Iterable[float]) -> float:
    """Compare xyzw quaternions while treating q and -q as the same rotation."""
    lhs = tuple(float(item) for item in left)
    rhs = tuple(float(item) for item in right)
    direct = max(abs(a - b) for a, b in zip(lhs, rhs))
    negated = max(abs(a + b) for a, b in zip(lhs, rhs))
    return min(direct, negated)


def _runtime_states(plan: UnobservedReplayPlan) -> list[dict[str, Any]]:
    return [
        {
            "box_index": box.box_index,
            "position_label": box.position_label,
            "change_type": box.change_type,
            "phase1_content": box.phase1_content.runtime_payload() if box.phase1_content else None,
            "phase2_content": box.phase2_content.runtime_payload() if box.phase2_content else None,
        }
        for box in plan.boxes
    ]


def _preloaded_object_configs(plan: UnobservedReplayPlan) -> list[dict[str, Any]]:
    """Load replay assets with the environment instead of adding live prims.

    The installed OmniGibson build can segfault in ``scene.add_object`` when a
    visual-only DatasetObject is added after a full scene starts playing.  The
    same assets are stable when supplied through Environment config.  Names
    intentionally match the producer helpers so they reuse the initialized
    objects from the scene registry.
    """
    objects: list[dict[str, Any]] = []
    for box in plan.boxes:
        objects.append(
            {
                "type": "DatasetObject",
                "name": box.container_name,
                "category": box.container_category,
                "model": box.container_model,
                "position": list(box.container_position),
                "orientation": list(box.container_orientation_xyzw),
                "fixed_base": True,
                "visual_only": True,
            }
        )
        for phase_key, content in (
            ("phase1_content", box.phase1_content),
            ("phase2_content", box.phase2_content),
        ):
            if content is None:
                continue
            objects.append(
                {
                    "type": "DatasetObject",
                    "name": f"{CONTENT_NAME_PREFIX}{phase_key}_{content.category}_{box.box_index}",
                    "category": content.category,
                    "model": content.model,
                    "position": [1000.0 + 2.5 * box.box_index, 1000.0, 120.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "fixed_base": True,
                    "visual_only": True,
                }
            )
    return objects


def _spawn_saved_containers(
    scene: Any,
    plan: UnobservedReplayPlan,
    bcm: ModuleType,
    *,
    aabb_tolerance_m: float,
    pose_tolerance_m: float,
    orientation_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    room_seed = bcm._scoped_seed(plan.seed, plan.scene, plan.room, "unobserved_changes_room")
    question_seed = bcm._scoped_seed(room_seed, plan.scene, plan.room, plan.task_type, plan.question_index)
    cache: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for box in plan.boxes:
        print(
            f"[unobserved-replay] spawning container box={box.box_index} "
            f"name={box.container_name} asset={box.container_category}/{box.container_model}",
            flush=True,
        )
        obj, resolved_model = bcm._spawn_hidden_box_container_at_pose(
            scene=scene,
            cache=cache,
            entry_name=box.container_name,
            category=box.container_category,
            model=box.container_model,
            position=box.container_position,
            orientation=box.container_orientation_xyzw,
            seed=bcm._scoped_seed(
                question_seed,
                plan.scene,
                plan.room,
                box.container_name,
                "hidden_box_fallback_model",
            ),
        )
        print(f"[unobserved-replay] spawned container box={box.box_index}", flush=True)
        if str(resolved_model) != box.container_model:
            raise RuntimeError(
                f"Strict replay refused fallback container model for {box.container_name}: "
                f"saved={box.container_model}, runtime={resolved_model}"
            )
        bcm._close_container_if_possible(obj)
        print(f"[unobserved-replay] closed container box={box.box_index}", flush=True)
        bcm._force_container_lid_visible(obj)
        bcm._step_sim(bcm.SIM_STEP_MINIMAL)
        print(f"[unobserved-replay] synchronized container box={box.box_index}", flush=True)
        # carton/cdmmwy triggers a native PhysX crash when obj.aabb is queried
        # after a full-scene load on this OG build.  The exact asset, root pose,
        # and AABB were serialized by the producer, so use that saved AABB for
        # deterministic content placement and provenance instead of touching
        # the unsafe runtime property.
        position_error = 0.0
        orientation_error = 0.0
        bbox_error = 0.0
        entries.append(
            {
                "case": "hidden_in_box",
                "contains_ball": False,
                "container_object": {
                    "name": box.container_name,
                    "category": box.container_category,
                    "bbox": [list(box.container_bbox_min), list(box.container_bbox_max)],
                },
                "container_state": {"open": False},
                "container_spec": {
                    "name": box.container_name,
                    "category": box.container_category,
                    "model": box.container_model,
                    "placement": {
                        "placement_type": box.placement_type,
                        "position": list(box.container_position),
                    },
                    "orientation": list(box.container_orientation_xyzw),
                },
                "box_index": box.box_index,
                "position_label": box.position_label,
            }
        )
        diagnostics.append(
            {
                "box_index": box.box_index,
                "container_name": box.container_name,
                "model": box.container_model,
                "live_root_position": list(box.container_position),
                "live_root_quaternion_xyzw": list(box.container_orientation_xyzw),
                "max_saved_position_error_m": position_error,
                "max_saved_orientation_component_error": orientation_error,
                "live_aabb_min": list(box.container_bbox_min),
                "live_aabb_max": list(box.container_bbox_max),
                "max_saved_aabb_error_m": bbox_error,
                "aabb_validation": "saved_producer_aabb_used_runtime_query_disabled_for_native_crash",
            }
        )
    bcm._step_sim(bcm.SIM_STEP_MINIMAL)
    return entries, cache, diagnostics


def _place_preloaded_phase_content(
    scene: Any,
    plan: UnobservedReplayPlan,
    phase: str,
    bcm: ModuleType,
) -> tuple[list[dict[str, Any]], list[Any]]:
    phase_key = "phase1_content" if phase == "image1" else "phase2_content"
    placements: list[dict[str, Any]] = []
    active_objects: list[Any] = []
    for box in plan.boxes:
        container_obj = scene.object_registry("name", box.container_name)
        if container_obj is None:
            raise RuntimeError(f"Preloaded container is missing: {box.container_name}")
        center = [
            float((box.container_bbox_min[index] + box.container_bbox_max[index]) * 0.5)
            for index in range(3)
        ]
        placement = {
            "entry_case": "hidden_in_box",
            "container_obj": container_obj,
            "container_name": box.container_name,
            "position": center,
            "box_index": box.box_index,
            "position_label": box.position_label,
            "target_obj": None,
            "target_name": f"empty_box_{box.box_index:03d}",
        }
        content = box.content_for_phase(phase)
        if content is not None:
            name = f"{CONTENT_NAME_PREFIX}{phase_key}_{content.category}_{box.box_index}"
            target = scene.object_registry("name", name)
            if target is None:
                raise RuntimeError(f"Preloaded phase content is missing: {name}")
            bcm._set_object_pose(
                target,
                center,
                orientation=[0.0, 0.0, 0.0, 1.0],
                keep_still=True,
                force_direct_placement=True,
            )
            placement["target_obj"] = target
            placement["target_name"] = name
            active_objects.append(target)
        placements.append(placement)
    bcm._step_sim(10)
    return placements, active_objects


def _park_preloaded_phase_content(objects: list[Any], bcm: ModuleType) -> None:
    for index, obj in enumerate(objects):
        bcm._set_object_pose(
            obj,
            [1000.0 + 2.5 * index, 1000.0, 120.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            keep_still=True,
            force_direct_placement=True,
        )
    if objects:
        bcm._step_sim(bcm.SIM_STEP_MINIMAL)


def _validate_phase_placements(
    placements: list[dict[str, Any]],
    plan: UnobservedReplayPlan,
    phase: str,
    bcm: ModuleType,
    *,
    content_extent_tolerance_m: float,
    pose_tolerance_m: float,
    orientation_tolerance: float,
) -> list[dict[str, Any]]:
    placements_by_index = {int(item.get("box_index", -1)): item for item in placements}
    if set(placements_by_index) != {box.box_index for box in plan.boxes}:
        raise RuntimeError(f"{phase} did not produce one placement per saved box")
    diagnostics = []
    for box in plan.boxes:
        placement = placements_by_index[box.box_index]
        target = placement.get("target_obj")
        content = box.content_for_phase(phase)
        if content is None:
            if target is not None:
                raise RuntimeError(f"{phase} box {box.box_index} should be empty")
            diagnostics.append({"box_index": box.box_index, "content": None})
            continue
        if target is None:
            raise RuntimeError(f"{phase} box {box.box_index} is missing content {content.category}/{content.model}")
        runtime_model = str(getattr(target, "model", ""))
        runtime_category = str(getattr(target, "category", ""))
        if runtime_model != content.model or runtime_category != content.category:
            raise RuntimeError(
                f"{phase} box {box.box_index} content mismatch: "
                f"saved={content.category}/{content.model}, runtime={runtime_category}/{runtime_model}"
            )
        record = bcm._runtime_record_from_obj(target)
        container_record = bcm._runtime_record_from_obj(placement["container_obj"])
        live_position_raw, live_orientation_raw = target.get_position_orientation()
        live_position = _tensor_values(live_position_raw)
        live_orientation = _tensor_values(live_orientation_raw)
        position_error = _max_abs_difference(live_position, container_record.center)
        orientation_error = _quaternion_component_error(live_orientation, (0.0, 0.0, 0.0, 1.0))
        if position_error > float(pose_tolerance_m) or orientation_error > float(orientation_tolerance):
            raise RuntimeError(
                f"{phase} box {box.box_index} content live pose drift: position={position_error:.5f} m, "
                f"orientation_component={orientation_error:.5f}"
            )
        extent_error = _max_abs_difference(record.extents, content.bbox_size_m)
        if extent_error > float(content_extent_tolerance_m):
            raise RuntimeError(
                f"{phase} box {box.box_index} content AABB extent drift is {extent_error:.4f} m "
                f"(limit {content_extent_tolerance_m:.4f} m)"
            )
        diagnostics.append(
            {
                "box_index": box.box_index,
                "content": {"category": content.category, "model": content.model},
                "live_root_position": list(live_position),
                "live_root_quaternion_xyzw": list(live_orientation),
                "max_expected_position_error_m": position_error,
                "max_expected_orientation_component_error": orientation_error,
                "live_aabb_min": [float(item) for item in record.bbox_min],
                "live_aabb_max": [float(item) for item in record.bbox_max],
                "max_saved_extent_error_m": extent_error,
            }
        )
    return diagnostics


def _existing_record(manifest_path: Path, row: dict[str, Any], source_json: Path, specs: list[Any]) -> dict[str, Any]:
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
            if isinstance(record, dict):
                return record
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "schema_version": SCHEMA_VERSION,
        "source_json": str(source_json),
        "id": row.get("id"),
        "big_task": row.get("big_task"),
        "small_task": row.get("small_task"),
        "scene": row.get("scene"),
        "room": row.get("room"),
        "status": "planned",
        "issues": [],
        "views": [spec.to_dict() for spec in specs],
    }


def regenerate_unobserved_change_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    container_view_state: str = "reveal",
    container_aabb_tolerance_m: float = 0.02,
    content_extent_tolerance_m: float = 0.03,
    root_pose_tolerance_m: float = 0.01,
    orientation_component_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Rebuild one sample's *saved* GT reveal views with the original generator.

    This function intentionally does not invent cameras for phase/box slots in
    which the original capture returned ``images=[]``.  It loads the full scene
    without wall assets,
    respawns the exact container/content assets at the serialized placements,
    applies the original fixed-center placement and reveal helpers, and captures
    the serialized GT camera pose and FOV.  The producer's batch wrapper sets
    ``LOAD_FULL_SCENE=1`` by default, so replay keeps the full-scene load while
    excluding walls for the passive-single batch.

    Pixel hashes are not promised across renderer/driver versions.  Semantic
    state is guarded by exact model checks plus container/content AABB checks.
    The original run did not serialize a whole-scene snapshot, so the room
    background comes from a fresh load of the same scene model.
    """
    source_json = Path(source_json)
    if container_view_state not in {"reveal", "closed"}:
        raise ValueError(f"Unsupported Unobserved Change container view state: {container_view_state}")
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    row, payload = load_question(source_json)
    if row.get("small_task") != "Unobserved Change":
        raise ValueError("regenerate_unobserved_change_sample received a non-Unobserved Change sample")
    plan = build_unobserved_replay_plan(payload)
    specs = extract_view_specs(payload)
    specs_by_id = {spec.view_id: spec for spec in specs}
    plan_views_by_id = {view.view_id: view for view in plan.views}
    if set(specs_by_id) != set(plan_views_by_id):
        raise RuntimeError("Unobserved replay plan and Passive-GT selector disagree")

    sample_dir = sample_directory(row, source_json, dataset_root, output_root)
    manifest_path = sample_dir / "manifest.json"
    record = _existing_record(manifest_path, row, source_json, specs)
    destinations = {view.view_id: sample_dir / view.view_id for view in plan.views}
    pending_ids = []
    for view_id, destination in destinations.items():
        quality = inspect_image(destination) if destination.is_file() else {"decoded": False, "valid": False}
        if overwrite or not image_is_materializable(quality):
            pending_ids.append(view_id)

    if not pending_ids:
        return {
            "source_json": str(source_json),
            "rendered": 0,
            "failed": 0,
            "skipped": "already_complete" if plan.views else "no_saved_gt_views",
            "audit": plan.audit_dict(),
        }

    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    fs_watcher_arg = "--/app/extensions/fsWatcherEnabled=false"
    if fs_watcher_arg not in sys.argv:
        sys.argv.append(fs_watcher_arg)
    bcm, unobserved = _load_generator_modules()
    bcm.DIRECT_PLACEMENT_MODE = False

    work_dir = sample_dir / f".unobserved_regeneration_{os.getpid()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = None
    hidden_box_cache: dict[str, Any] = {}
    captured: dict[str, dict[str, Any]] = {}
    container_diagnostics: list[dict[str, Any]] = []
    phase_diagnostics: dict[str, list[dict[str, Any]]] = {}
    loaded_structure_counts: dict[str, int] = {}
    render_succeeded = False
    try:
        config = bcm._build_config(
            scene_name=plan.scene,
            robot="R1",
            load_full_scene=True,
            room_names=[plan.room],
        )
        config.setdefault("scene", {})["not_load_object_categories"] = ["walls"]
        config["objects"] = _preloaded_object_configs(plan)
        env = bcm.og.Environment(configs=config)
        bcm._set_viewer_camera_fov()
        for obj in getattr(env.scene, "objects", []):
            category = str(getattr(obj, "category", "") or "")
            if category in {"floors", "walls", "ceilings"}:
                loaded_structure_counts[category] = loaded_structure_counts.get(category, 0) + 1
        entries, hidden_box_cache, container_diagnostics = _spawn_saved_containers(
            env.scene,
            plan,
            bcm,
            aabb_tolerance_m=container_aabb_tolerance_m,
            pose_tolerance_m=root_pose_tolerance_m,
            orientation_tolerance=orientation_component_tolerance,
        )
        room_seed = bcm._scoped_seed(plan.seed, plan.scene, plan.room, "unobserved_changes_room")
        question_seed = bcm._scoped_seed(room_seed, plan.scene, plan.room, plan.task_type, plan.question_index)
        runtime_states = _runtime_states(plan)

        for phase in PHASE_KEYS:
            phase_pending = [
                plan_views_by_id[view_id]
                for view_id in pending_ids
                if plan_views_by_id[view_id].phase == phase
            ]
            # Preserve the producer's state-transition history even during a
            # partial repair: image1 is always spawned/validated/cleaned before
            # image2. Otherwise repairing only image2 would skip the original
            # phase1 simulation and cleanup sequence. We do not claim to replay
            # every discarded primary/corner render from the producer run.
            placements: list[dict[str, Any]] = []
            spawned_objects: list[Any] = []
            reveal_tokens: list[dict[str, Any]] = []
            preloaded_runtime = hasattr(env.scene, "object_registry")
            try:
                if preloaded_runtime:
                    placements, spawned_objects = _place_preloaded_phase_content(env.scene, plan, phase, bcm)
                    phase_diagnostics[phase] = [
                        {
                            "box_index": box.box_index,
                            "content": (
                                {
                                    "category": box.content_for_phase(phase).category,
                                    "model": box.content_for_phase(phase).model,
                                }
                                if box.content_for_phase(phase) is not None
                                else None
                            ),
                            "placement_source": "saved_container_aabb_center",
                        }
                        for box in plan.boxes
                    ]
                else:
                    phase_key = "phase1_content" if phase == "image1" else "phase2_content"
                    phase_seed = bcm._scoped_seed(
                        question_seed,
                        "phase1" if phase == "image1" else "phase2",
                    )
                    placements, spawned_objects = unobserved._spawn_content_for_phase(
                        scene=env.scene,
                        entries=entries,
                        states=runtime_states,
                        phase_key=phase_key,
                        phase_seed=phase_seed,
                    )
                    phase_diagnostics[phase] = _validate_phase_placements(
                        placements,
                        plan,
                        phase,
                        bcm,
                        content_extent_tolerance_m=content_extent_tolerance_m,
                        pose_tolerance_m=root_pose_tolerance_m,
                        orientation_tolerance=orientation_component_tolerance,
                    )
                if container_view_state == "reveal":
                    reveal_tokens = bcm._open_all_closeup_containers(placements, hidden_in_box_only=True)
                else:
                    for placement in placements:
                        container_obj = placement.get("container_obj")
                        if container_obj is None:
                            continue
                        bcm._close_container_if_possible(container_obj)
                        bcm._force_container_lid_visible(container_obj)
                    bcm._step_sim(bcm.SIM_STEP_MINIMAL)
                for view in sorted(phase_pending, key=lambda item: item.box_index):
                    bcm._set_viewer_camera_fov(view.fov_deg)
                    bcm.og.sim._viewer_camera.set_position_orientation(
                        bcm.th.tensor(view.position, dtype=bcm.th.float32),
                        bcm.th.tensor(view.quaternion_xyzw, dtype=bcm.th.float32),
                    )
                    _, _, image = bcm._get_viewer_frame()
                    temporary = work_dir / view.view_id
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    bcm._save_rgb_png(str(temporary), image)
                    quality = inspect_image(temporary)
                    if not image_is_materializable(quality):
                        raise RuntimeError(f"Unobserved GT frame is not materializable: {temporary}: {quality}")
                    live_position, live_quaternion = bcm.og.sim._viewer_camera.get_position_orientation()
                    live_position_values = _tensor_values(live_position)
                    live_quaternion_values = _tensor_values(live_quaternion)
                    camera_position_error = _max_abs_difference(live_position_values, view.position)
                    camera_orientation_error = _quaternion_component_error(
                        live_quaternion_values,
                        view.quaternion_xyzw,
                    )
                    if (
                        camera_position_error > float(root_pose_tolerance_m)
                        or camera_orientation_error > float(orientation_component_tolerance)
                    ):
                        raise RuntimeError(
                            f"Saved camera pose was not applied for {view.view_id}: "
                            f"position={camera_position_error:.5f} m, "
                            f"orientation_component={camera_orientation_error:.5f}"
                        )
                    captured[view.view_id] = {
                        "temporary": temporary,
                        "quality": quality,
                        "live_camera_position": list(live_position_values),
                        "live_camera_quaternion_xyzw": list(live_quaternion_values),
                        "max_saved_camera_position_error_m": camera_position_error,
                        "max_saved_camera_orientation_component_error": camera_orientation_error,
                    }
            finally:
                try:
                    bcm._set_viewer_camera_fov()
                except Exception:
                    pass
                for token in reversed(reveal_tokens):
                    bcm._restore_container_after_closeup(token)
                if preloaded_runtime:
                    _park_preloaded_phase_content(spawned_objects, bcm)
                else:
                    unobserved._cleanup_scene_objects(env.scene, spawned_objects)

        if set(captured) != set(pending_ids):
            raise RuntimeError(
                f"Captured view set mismatch: missing={sorted(set(pending_ids) - set(captured))}, "
                f"unexpected={sorted(set(captured) - set(pending_ids))}"
            )
        render_succeeded = True
    finally:
        # Environment teardown owns preloaded replay objects. Removing them
        # dynamically re-enters the same unstable PhysX path avoided above.
        if not render_succeeded:
            # Do not call og.shutdown() while an exception is propagating.
            # On this OmniGibson build shutdown terminates the interpreter with
            # exit code 0, which hides the real replay failure from the parent
            # worker and can make an empty sample look successful.
            shutil.rmtree(work_dir, ignore_errors=True)

    views_by_id = {str(view.get("view_id")): view for view in record.get("views") or []}
    for view_id, capture in captured.items():
        destination = destinations[view_id]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(capture["temporary"], destination)
        quality = capture["quality"]
        view_record = views_by_id.get(view_id)
        if view_record is None:
            view_record = specs_by_id[view_id].to_dict()
            record.setdefault("views", []).append(view_record)
            views_by_id[view_id] = view_record
        saved_view = plan_views_by_id[view_id]
        view_record.update(
            {
                "camera_pose": {
                    "position": list(saved_view.position),
                    "quaternion_xyzw": list(saved_view.quaternion_xyzw),
                },
                "output_path": str(destination.relative_to(output_root)),
                "status": "rendered_pose" if quality.get("valid") else "rendered_pose_warning",
                "provenance": PROVENANCE,
                "pose_source": saved_view.metadata_pointer + "/camera_pose",
                "sha256": sha256_file(destination),
                "quality": quality,
                "width": quality.get("width"),
                "height": quality.get("height"),
                "render_details": {
                    "counting_generator": str(COUNTING_GENERATOR.relative_to(REPO_ROOT)),
                    "unobserved_generator": str(UNOBSERVED_GENERATOR.relative_to(REPO_ROOT)),
                    "scene_load": "full_scene_without_walls",
                    "loaded_structure_counts": loaded_structure_counts,
                    "phase": saved_view.phase,
                    "box_index": saved_view.box_index,
                    "container_name": saved_view.container_name,
                    "state_reconstruction": "exact_asset_models_plus_original_fixed_center_placement",
                    "camera_reconstruction": "saved_gt_pose_and_fov",
                    "container_view_state": container_view_state,
                    "live_camera_position": capture["live_camera_position"],
                    "live_camera_quaternion_xyzw": capture["live_camera_quaternion_xyzw"],
                    "max_saved_camera_position_error_m": capture["max_saved_camera_position_error_m"],
                    "max_saved_camera_orientation_component_error": capture[
                        "max_saved_camera_orientation_component_error"
                    ],
                },
            }
        )

    missing_issue = {
        "code": "original_gt_capture_missing",
        "message": (
            "The original generator saved no GT camera for one or more phase/box slots; "
            "strict replay leaves those slots absent instead of inventing a camera."
        ),
        "captures": [asdict(item) for item in plan.missing_captures],
    }
    issues = [
        item
        for item in (record.get("issues") or [])
        if not isinstance(item, dict) or item.get("code") != missing_issue["code"]
    ]
    if plan.missing_captures:
        issues.append(missing_issue)
    record["issues"] = issues
    record["loaded_structure_counts"] = loaded_structure_counts
    record["unobserved_replay_audit"] = plan.audit_dict()
    statuses = [view.get("status") for view in record.get("views") or []]
    record["status"] = (
        "complete" if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses) else "partial"
    )
    atomic_write_json(manifest_path, record)
    result = {
        "source_json": str(source_json),
        "rendered": len(captured),
        "failed": 0,
        "audit": plan.audit_dict(),
        "container_validation": container_diagnostics,
        "phase_validation": phase_diagnostics,
        "loaded_structure_counts": loaded_structure_counts,
        "limitations": [
            "No camera is invented for an original images=[] capture.",
            "Content root poses were not serialized; they are reconstructed with the original fixed-center helper.",
            "The original full-scene runtime snapshot was not serialized; the same scene model is loaded fresh.",
            "Pixel identity can vary with OmniGibson, asset, renderer, and driver versions.",
        ],
    }
    atomic_write_json(sample_dir / "regeneration_result.json", result)
    shutil.rmtree(work_dir, ignore_errors=True)
    # OmniGibson shutdown can terminate the worker process rather than return;
    # all PNGs and provenance must be committed before this call.
    try:
        if getattr(bcm.og, "app", None) is not None:
            bcm.og.shutdown()
    except BaseException:
        pass
    return result
