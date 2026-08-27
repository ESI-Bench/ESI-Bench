from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ViewSpec:
    """One Passive-GT image and the camera/state metadata needed to reproduce it."""

    view_id: str
    group: str
    role: str
    metadata_pointer: str
    original_image_path: str
    position: tuple[float, float, float] | None
    quaternion_xyzw: tuple[float, float, float, float] | None
    fov_deg: float | None = None
    phase: str | None = None
    state_key: str | None = None
    render_mode: str = "static"
    pose_source: str = "saved"
    baseline_selected: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_pose"] = (
            {
                "position": list(self.position),
                "quaternion_xyzw": list(self.quaternion_xyzw),
            }
            if self.position is not None and self.quaternion_xyzw is not None
            else None
        )
        del payload["position"]
        del payload["quaternion_xyzw"]
        return payload


def _task_key(payload: dict[str, Any]) -> tuple[str, str]:
    big = str(payload.get("_hf_big_task") or payload.get("big_task") or "").strip()
    small = str(payload.get("_hf_small_task") or payload.get("small_task") or "").strip()
    return big, small


def _keyed_items(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        return [(str(key), item) for key, item in value.items() if isinstance(item, dict)]
    if isinstance(value, list):
        output: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            key = item.get("_key")
            output.append((str(key) if key not in (None, "") else str(index), item))
        return output
    return []


def _keyed_map(value: Any) -> dict[str, dict[str, Any]]:
    return {key: item for key, item in _keyed_items(value)}


def _keyed_bool_map(value: Any) -> dict[str, bool]:
    output: dict[str, bool] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict) and "value" in item:
                output[str(key)] = bool(item.get("value"))
            else:
                output[str(key)] = bool(item)
        return output
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            key = item.get("_key")
            if key not in (None, ""):
                output[str(key)] = bool(item.get("value"))
    return output


def _nested_bool(value: Any, key: str, default: bool = False) -> bool:
    if isinstance(value, dict):
        item = value.get(key)
        if isinstance(item, dict) and "value" in item:
            return bool(item.get("value"))
        return bool(item) if item is not None else default
    return _keyed_bool_map(value).get(key, default)


def _image_name(item: dict[str, Any], fallback: str) -> str:
    for key in ("image_path", "filename", "image", "view_file", "rgb"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    key = item.get("_key")
    if isinstance(key, str) and Path(key).suffix.lower() in IMAGE_SUFFIXES:
        return key
    return fallback


def _pose(item: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pose = item
    for key in ("camera_pose", "pose"):
        nested = item.get(key)
        if isinstance(nested, dict):
            pose = nested
            break
    position = pose.get("position") or pose.get("eye")
    quaternion = pose.get("quaternion_xyzw") or pose.get("orientation")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("camera position/eye must contain three values")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("camera quaternion_xyzw must contain four values")
    return tuple(float(x) for x in position), tuple(float(x) for x in quaternion)


def _safe_component(value: str) -> str:
    text = str(value or "view").replace("\\", "/").split("/")[-1]
    text = "_".join(text.strip().split())
    return text or "view"


def _view(
    item: dict[str, Any],
    *,
    group: str,
    role: str,
    pointer: str,
    fallback_name: str,
    fov_deg: float | None = None,
    phase: str | None = None,
    state_key: str | None = None,
    render_mode: str = "static",
    pose_source: str = "saved",
    baseline_selected: bool = True,
) -> ViewSpec:
    image_path = _image_name(item, fallback_name)
    position, quaternion = _pose(item)
    item_fov = item.get("fov_deg")
    effective_fov = float(item_fov) if item_fov is not None else fov_deg
    name = _safe_component(image_path)
    return ViewSpec(
        view_id=f"{group}/{name}",
        group=group,
        role=role,
        metadata_pointer=pointer,
        original_image_path=image_path,
        position=position,
        quaternion_xyzw=quaternion,
        fov_deg=effective_fov,
        phase=phase,
        state_key=state_key,
        render_mode=render_mode,
        pose_source=pose_source,
        baseline_selected=baseline_selected,
    )


def _views_from_items(
    value: Any,
    *,
    pointer: str,
    group: str,
    role: str = "gt",
    fov_deg: float | None = None,
    phase: str | None = None,
    state_key: str | None = None,
    render_mode: str = "static",
) -> list[ViewSpec]:
    output = []
    for index, (key, item) in enumerate(_keyed_items(value)):
        fallback_name = _safe_component(key)
        if Path(fallback_name).suffix.lower() not in IMAGE_SUFFIXES:
            fallback_name = f"{fallback_name}.png"
        output.append(
            _view(
                item,
                group=group,
                role=role,
                pointer=f"{pointer}/{index}",
                fallback_name=fallback_name,
                fov_deg=fov_deg,
                phase=phase,
                state_key=state_key,
                render_mode=render_mode,
            )
        )
    return output


def _action(payload: dict[str, Any]) -> list[ViewSpec]:
    """Use verified outcome stages; stage0 is the passive-single observation."""
    auxiliary = payload.get("_gt_auxiliary") or {}
    output: list[ViewSpec] = []
    for stage_index, (_stage_key, stage) in enumerate(_keyed_items(auxiliary.get("stages"))):
        label = str(stage.get("label") or f"stage_{stage_index + 1}")
        for view_index, (_view_key, item) in enumerate(_keyed_items(stage.get("views"))):
            visibility = [bool(value) for key, value in item.items() if str(key).startswith("exist_")]
            if visibility and not all(visibility):
                continue
            output.append(
                _view(
                    item,
                    group=f"stages/{label}",
                    role="gt_outcome",
                    pointer=f"/_gt_auxiliary/stages/{stage_index}/views/{view_index}",
                    fallback_name=f"{label}_orb{view_index}.png",
                    state_key=f"action_stage:{stage_index}",
                    render_mode="action_snapshot",
                    pose_source="BEHAVIOR-ESI/output_metas/action/stages.views",
                )
            )
    return output


def _cognitive(payload: dict[str, Any], include_regions: bool, include_objects: bool) -> list[ViewSpec]:
    qd = payload.get("question_data") or {}
    output: list[ViewSpec] = []
    path_views = (qd.get("path_views") or {}).get("views")
    if path_views:
        output.extend(
            _views_from_items(
                path_views,
                pointer="/question_data/path_views/views",
                group="path_views",
                fov_deg=100.0,
                render_mode="static_scene",
            )
        )
    if include_regions:
        output.extend(
            _views_from_items(
                qd.get("gt_region_views"),
                pointer="/question_data/gt_region_views",
                group="region_views",
                fov_deg=100.0,
                render_mode="static_scene",
            )
        )
    if include_objects:
        for object_index, (object_name, record) in enumerate(_keyed_items(qd.get("object_views"))):
            output.extend(
                _views_from_items(
                    record.get("views"),
                    pointer=f"/question_data/object_views/{object_index}/views",
                    group=f"object_views/{_safe_component(object_name)}",
                    fov_deg=100.0,
                    render_mode="static_scene",
                )
            )
    return output


def _enumerative(payload: dict[str, Any]) -> list[ViewSpec]:
    render = ((payload.get("question_data") or {}).get("render") or {})
    small_task = str(payload.get("_hf_small_task") or "")
    fov_deg = 55.0 if small_task == "Structural Enclosure" else 100.0
    approved = []
    for key, item in _keyed_items(render.get("target_closeups")):
        visibility = item.get("visibility") or {}
        if isinstance(visibility, dict) and visibility.get("is_visible") is False:
            continue
        if item.get("best_effort_only") is True:
            continue
        approved.append((key, item))
    output = _views_from_items(
        {key: item for key, item in approved},
        pointer="/question_data/render/target_closeups",
        group="target_closeups",
        fov_deg=fov_deg,
        render_mode="static_objects",
    )
    if output:
        return output

    # A failed legacy visibility check does not erase the generator-selected
    # camera. Reuse one target candidate while replaying the task's GT object
    # state. For a legitimate zero-target question there is no closeup, so the
    # saved room-centre context camera is the GT evidence-of-absence view.
    target_candidates = _keyed_items(render.get("target_closeups"))
    if target_candidates:
        key, item = target_candidates[0]
        return [
            _view(
                item,
                group="target_closeups",
                role="gt_rule_camera",
                pointer="/question_data/render/target_closeups/0",
                fallback_name=f"{_safe_component(key)}.png",
                fov_deg=fov_deg,
                render_mode="static_objects",
                pose_source="saved_target_camera_for_gt_visibility_rule_replay",
            )
        ]
    primary_cameras = _keyed_items(render.get("camera_poses"))
    if primary_cameras:
        key, item = primary_cameras[0]
        return [
            _view(
                item,
                group="gt_context",
                role="gt_rule_camera",
                pointer="/question_data/render/camera_poses/0",
                fallback_name=f"{_safe_component(key)}.png",
                fov_deg=fov_deg,
                render_mode="static_objects",
                pose_source="saved_primary_camera_for_zero_target_gt_context",
            )
        ]
    return []


def _top_level(payload: dict[str, Any], field: str, group: str, *, mode: str = "static_objects") -> list[ViewSpec]:
    return _views_from_items(payload.get(field), pointer=f"/{field}", group=group, render_mode=mode)


def _indexed_bools(value: Any) -> dict[tuple[int, ...], bool]:
    output: dict[tuple[int, ...], bool] = {}
    if not isinstance(value, list):
        return output
    for item in value:
        if not isinstance(item, dict):
            continue
        indices = item.get("_indices") or []
        if indices:
            try:
                output[tuple(int(index) for index in indices)] = bool(item.get("value"))
            except (TypeError, ValueError):
                pass
    return output


def _view_number(key: str, item: dict[str, Any]) -> int | None:
    candidate = str(item.get("_key") or key)
    match = re.match(r"^(\d+)", Path(candidate).stem)
    return int(match.group(1)) if match else None


def _approved_top_level(
    payload: dict[str, Any],
    *,
    predicate: Callable[[str, dict[str, Any]], bool],
    group: str = "gt_views",
) -> list[ViewSpec]:
    approved = {key: item for key, item in _keyed_items(payload.get("camera_poses")) if predicate(key, item)}
    return _views_from_items(
        approved,
        pointer="/camera_poses",
        group=group,
        role="gt_view",
        render_mode="static_objects",
    )


def _dimensional_size(payload: dict[str, Any]) -> list[ViewSpec]:
    task_visible = _indexed_bools(payload.get("exist_task_obj"))
    ref_visible = _indexed_bools(payload.get("exist_ref_obj"))

    def approved(key: str, item: dict[str, Any]) -> bool:
        view = _view_number(key, item)
        if view is None:
            return False
        return any(task_visible.get((pair, view), False) and ref_visible.get((pair, view), False) for pair in (1, 2))

    return _approved_top_level(payload, predicate=approved)


def _spatial_distance(payload: dict[str, Any]) -> list[ViewSpec]:
    near = _indexed_bools(payload.get("exist_near"))
    far = _indexed_bools(payload.get("exist_far"))

    def approved(key: str, item: dict[str, Any]) -> bool:
        view = _view_number(key, item)
        return view is not None and near.get((view,), False) and far.get((view,), False)

    return _approved_top_level(payload, predicate=approved)


def _material_transparency(payload: dict[str, Any]) -> list[ViewSpec]:
    container = _indexed_bools(payload.get("exist_obj_container"))

    def approved(key: str, item: dict[str, Any]) -> bool:
        view = _view_number(key, item)
        return view is not None and container.get((view,), False)

    output = _approved_top_level(payload, predicate=approved)
    if output:
        return output
    candidates = _keyed_items(payload.get("camera_poses"))
    if not candidates:
        return []
    key, item = candidates[0]
    return [
        _view(
            item,
            group="gt_views",
            role="gt_rule_camera",
            pointer="/camera_poses/0",
            fallback_name=f"{_safe_component(key)}.png",
            render_mode="static_objects",
            pose_source="saved_candidate_camera_for_transparency_gt_state_replay",
        )
    ]


def _partial_observation(payload: dict[str, Any], *, require_occluder: bool) -> list[ViewSpec]:
    approved: dict[str, dict[str, Any]] = {}
    for key, item in _keyed_items(payload.get("views")):
        visibility = item.get("visibility") or {}
        if not _nested_bool(visibility, "target_obj"):
            continue
        if require_occluder and not _nested_bool(visibility, "occluder_obj"):
            continue
        approved[key] = item
    output = _views_from_items(
        approved,
        pointer="/views",
        group="gt_visible_views",
        role="gt_view",
        render_mode="static_objects",
    )
    if output:
        return output
    candidates = _keyed_items(payload.get("views"))
    if not candidates:
        return []
    # Prefer a camera that already sees the target. The GT replay rule is
    # responsible for restoring the missing occluder/visibility state.
    key, item = next(
        (
            (candidate_key, candidate)
            for candidate_key, candidate in candidates
            if _nested_bool(candidate.get("visibility") or {}, "target_obj")
        ),
        candidates[0],
    )
    return [
        _view(
            item,
            group="gt_visible_views",
            role="gt_rule_camera",
            pointer=f"/views/{key}",
            fallback_name=f"{_safe_component(key)}.png",
            render_mode="static_objects",
            pose_source="saved_candidate_camera_for_occlusion_gt_state_replay",
        )
    ]


def _inclined_plane(payload: dict[str, Any]) -> list[ViewSpec]:
    # camera_poses_floor are the Phase-1 / initial observation (passive
    # single), not GT.  The outcome evidence is the saved slope sequence.
    output: list[ViewSpec] = []
    pose_by_view: dict[int, dict[str, Any]] = {}
    for _key, item in _keyed_items(payload.get("camera_poses")):
        try:
            pose_by_view.setdefault(int(item.get("view_idx")), item)
        except (TypeError, ValueError):
            continue
    for index, (key, flag) in enumerate(_keyed_items(payload.get("exist_step_view"))):
        if flag.get("value") is False or not key.startswith("exist_step") or "_view" not in key:
            continue
        step_text, view_text = key.removeprefix("exist_step").split("_view", 1)
        try:
            step = int(step_text)
            view_index = int(view_text)
        except ValueError:
            continue
        pose = pose_by_view.get(view_index)
        if pose is None:
            continue
        item = dict(pose)
        item["filename"] = f"step_{step:04d}_view_{view_index}.png"
        output.append(
            _view(
                item,
                group="slope_steps",
                role="gt_dynamics",
                pointer=f"/exist_step_view/{index}",
                fallback_name=item["filename"],
                phase=f"step_{step:04d}",
                state_key=f"step:{step}",
                render_mode="dynamic_replay",
                pose_source="saved_fixed_view_by_view_idx",
            )
        )
    return output


def _stacking(payload: dict[str, Any]) -> list[ViewSpec]:
    output: list[ViewSpec] = []
    initial_cameras = _keyed_items(payload.get("initial_cam_poses"))
    visibility = {
        str(key): bool(item.get("value")) if isinstance(item, dict) else bool(item)
        for key, item in _keyed_items(payload.get("initial_exist_flags"))
    }
    object_names = [str(value) for value in payload.get("object_names") or []]
    selected_initial = None
    for key, item in initial_cameras:
        filename = str(item.get("_key") or key)
        if all(visibility.get(f"exist_{name}_{filename}", False) for name in object_names):
            selected_initial = (key, item)
            break
    if selected_initial is None and initial_cameras:
        selected_initial = initial_cameras[0]
    if selected_initial is not None:
        key, item = selected_initial
        output.append(
            _view(
                item,
                group="interaction_initial",
                role="interaction_initial",
                pointer="/initial_cam_poses/0",
                fallback_name=str(item.get("_key") or key or "stack_initial.png"),
                state_key="stack_initial",
                render_mode="stack_snapshot",
                pose_source="saved_initial_stacking_camera",
                baseline_selected=False,
            )
        )
    for trial_index, (_key, trial) in enumerate(_keyed_items(payload.get("trials"))):
        if trial.get("all_stable") is not True:
            continue
        output.extend(
            _views_from_items(
                trial.get("camera_poses"),
                pointer=f"/trials/{trial_index}/camera_poses",
                group=f"trials/trial_{trial_index:02d}",
                role="gt_outcome",
                state_key=f"trial:{trial_index}",
                render_mode="stack_snapshot",
            )
        )
    return output


def _deformable(payload: dict[str, Any]) -> list[ViewSpec]:
    entry = ((((payload.get("render") or {}).get("gt_view") or {}).get("main_view_without_cloth")) or {})
    if not entry:
        return []
    return [
        _view(
            entry,
            group="gt_view",
            role="gt_reveal",
            pointer="/render/gt_view/main_view_without_cloth",
            fallback_name="main_view_without_cloth.png",
            fov_deg=float(entry.get("fov_deg") or 70.0),
            state_key="cloth_removed",
            render_mode="special_state",
        )
    ]


def _placeholder_view(
    *,
    view_id: str,
    group: str,
    role: str,
    pointer: str,
    image_path: str,
    phase: str,
    state_key: str,
    pose_source: str,
) -> ViewSpec:
    """Represent a required GT frame whose live pose/state must be regenerated."""
    return ViewSpec(
        view_id=view_id,
        group=group,
        role=role,
        metadata_pointer=pointer,
        original_image_path=image_path,
        position=None,
        quaternion_xyzw=None,
        phase=phase,
        state_key=state_key,
        render_mode="full_regeneration",
        pose_source=pose_source,
    )


def _liquid(_payload: dict[str, Any]) -> list[ViewSpec]:
    # batch_pour.py defines a four-frame oracle trajectory.  json_clean stores
    # the measured capacities but neither the live AABBs used for the camera nor
    # the particle state, so keep all four required frames visible in the plan.
    phases = (
        ("before_fill_left", "00_before_fill_left.png"),
        ("after_fill_left", "01_after_fill_left.png"),
        ("after_fill_right", "02_after_fill_right.png"),
        ("final", "03_final.png"),
    )
    return [
        _placeholder_view(
            view_id=f"pour_trajectory/{filename}",
            group="pour_trajectory",
            role="gt_dynamics",
            pointer=f"/regeneration_contract/{phase}",
            image_path=filename,
            phase=phase,
            state_key=f"pour:{phase}",
            pose_source="live_AABB_in_BEHAVIOR-ESI/tasks/task_capacity/batch_pour.py",
        )
        for phase, filename in phases
    ]


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _rigid_answer_labels(payload: dict[str, Any]) -> set[str]:
    value = payload.get("_ground_truth")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    labels: set[str] = set()
    if isinstance(value, dict):
        labels.update(_normalize_label(key) for key in value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                key = item.get("_key") or item.get("object") or item.get("name")
                if key not in (None, ""):
                    labels.add(_normalize_label(key))
    return labels


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-10:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _matrix_quaternion_xyzw(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    """Equivalent to scipy Rotation.from_matrix(matrix).as_quat()."""
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    return (qx, qy, qz, qw)


def _look_at_quaternion(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    forward = _unit(tuple(target[index] - eye[index] for index in range(3)))  # type: ignore[arg-type]
    right_raw = _cross(forward, up)
    if math.sqrt(sum(value * value for value in right_raw)) < 1e-6:
        right_raw = _cross(forward, (0.0, 1.0, 0.0))
    right = _unit(right_raw)
    true_up = _unit(_cross(right, forward))
    # np.column_stack([right, true_up, -forward]) from batch_storage.py.
    matrix = (
        (right[0], true_up[0], -forward[0]),
        (right[1], true_up[1], -forward[1]),
        (right[2], true_up[2], -forward[2]),
    )
    return _matrix_quaternion_xyzw(matrix)


def _aabb_camera(
    pose: dict[str, Any], *, topdown: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    minimum = pose.get("aabb_min") or ((pose.get("aabb") or {}).get("min") if isinstance(pose.get("aabb"), dict) else None)
    maximum = pose.get("aabb_max") or ((pose.get("aabb") or {}).get("max") if isinstance(pose.get("aabb"), dict) else None)
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        return None
    if len(minimum) != 3 or len(maximum) != 3:
        return None
    centre = tuple((float(minimum[index]) + float(maximum[index])) / 2.0 for index in range(3))
    if topdown:
        eye = (centre[0], centre[1], float(maximum[2]) + 0.8)
        up = (0.0, 1.0, 0.0)
    else:
        radius = max(float(maximum[0]) - float(minimum[0]), float(maximum[1]) - float(minimum[1])) / 2.0 + 0.4
        angle = math.radians(45.0)
        eye = (
            centre[0] + radius * math.cos(angle),
            centre[1] + radius * math.sin(angle),
            float(maximum[2]) + 0.5,
        )
        up = (0.0, 0.0, 1.0)
    return eye, _look_at_quaternion(eye, centre, up)


def _derived_rigid_view(
    *,
    camera: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None,
    view_id: str,
    group: str,
    role: str,
    pointer: str,
    image_path: str,
    state_key: str,
    render_mode: str,
) -> ViewSpec:
    if camera is None:
        return _placeholder_view(
            view_id=view_id,
            group=group,
            role=role,
            pointer=pointer,
            image_path=image_path,
            phase="post_placement" if state_key.startswith("placement:") else "initial_inspection",
            state_key=state_key,
            pose_source="live_AABB_required_by_BEHAVIOR-1K/batch_storage.py",
        )
    return ViewSpec(
        view_id=view_id,
        group=group,
        role=role,
        metadata_pointer=pointer,
        original_image_path=image_path,
        position=camera[0],
        quaternion_xyzw=camera[1],
        phase="post_placement" if state_key.startswith("placement:") else "initial_inspection",
        state_key=state_key,
        render_mode=render_mode,
        pose_source="derived_exactly_from_saved_AABB_using_BEHAVIOR-1K/batch_storage.py",
    )


def _rigid(payload: dict[str, Any]) -> list[ViewSpec]:
    # BEHAVIOR-1K/batch_storage.py explicitly marks these producer frames as GT:
    # queried-object + candidate-container inspection closeups, followed by two
    # closeups after every correct placement.  Ordinary before/after orbits and
    # wrong trials are not Ground-Truth Passive trajectory views.
    output: list[ViewSpec] = []
    queried_labels = _rigid_answer_labels(payload)
    queried: list[tuple[str, dict[str, Any], str]] = []
    containee = payload.get("containee")
    if isinstance(containee, dict):
        # Every benchmark question includes the primary containee.  Retaining
        # this fallback also makes rows robust to older answer serialization.
        queried.append(("containee", containee, "/containee"))
    for index, (label, item) in enumerate(_keyed_items(payload.get("extras"))):
        if _normalize_label(item.get("cat")) in queried_labels:
            queried.append((_safe_component(label), item, f"/extras/{index}"))

    initial = _keyed_map(payload.get("initial_poses"))
    for label, _item, pointer in queried:
        pose = initial.get(f"obj_{label}") or {}
        filename = f"gt_closeup_{label}.png"
        output.append(
            _derived_rigid_view(
                camera=_aabb_camera(pose, topdown=False),
                view_id=f"initial_inspection/objects/{filename}",
                group="initial_inspection/objects",
                role="gt_inspection",
                pointer=f"/initial_poses/{label}",
                image_path=filename,
                state_key="rigid_initial",
                render_mode="rigid_initial_snapshot",
            )
        )
    for index, (fit_label, container) in enumerate(_keyed_items(payload.get("containers"))):
        object_index = container.get("idx")
        if object_index is None:
            object_index = {"small": 0, "fit": 1, "big": 2}.get(str(container.get("fit_check") or fit_label))
        pose = initial.get(f"obj_container_{object_index}") or {}
        safe_label = _safe_component(str(container.get("fit_check") or fit_label))
        filename = f"gt_closeup_container_{safe_label}.png"
        output.append(
            _derived_rigid_view(
                camera=_aabb_camera(pose, topdown=False),
                view_id=f"initial_inspection/containers/{filename}",
                group="initial_inspection/containers",
                role="gt_inspection",
                pointer=f"/containers/{index}",
                image_path=filename,
                state_key="rigid_initial",
                render_mode="rigid_initial_snapshot",
            )
        )

    for label, item, pointer in queried:
        placement = item.get("placement") or {}
        state_key = f"placement:{label}"
        object_name = f"gt_closeup_placed_{label}.png"
        output.append(
            _derived_rigid_view(
                # Four benchmark rows record a failed fit attempt, but still
                # save the actual after-state root pose and AABB.  Reconstruct
                # that saved outcome instead of inventing a new successful
                # placement; the sample manifest records the failed check.
                camera=_aabb_camera(placement.get("object_pose_after") or {}, topdown=False),
                view_id=f"correct_placements/{label}/{object_name}",
                group=f"correct_placements/{label}",
                role="gt_outcome_object",
                pointer=f"{pointer}/placement/object_pose_after",
                image_path=object_name,
                state_key=state_key,
                render_mode="placement_snapshot",
            )
        )
        container_name = f"container_final_{label}.png"
        output.append(
            _derived_rigid_view(
                camera=_aabb_camera(placement.get("container_pose_after") or {}, topdown=True),
                view_id=f"correct_placements/{label}/{container_name}",
                group=f"correct_placements/{label}",
                role="gt_outcome_container",
                pointer=f"{pointer}/placement/container_pose_after",
                image_path=container_name,
                state_key=state_key,
                render_mode="placement_snapshot",
            )
        )
    return output


def _geometric(payload: dict[str, Any]) -> list[ViewSpec]:
    # Official GT consumer uses the three edge-normal views plus top-down view.
    approved = {
        key: item
        for key, item in _keyed_items(payload.get("camera_poses"))
        if Path(str(item.get("_key") or key)).name in {"1.png", "2.png", "3.png", "4.png"}
    }
    return _views_from_items(
        approved,
        pointer="/camera_poses",
        group="gt_configuration_views",
        role="gt_view",
        render_mode="static_objects",
    )


def _linear(payload: dict[str, Any]) -> list[ViewSpec]:
    # The unsuffixed front frame is the initial observation.  The four named
    # side/middle frames form the oracle multi-view evidence.
    approved = {
        key: item
        for key, item in _keyed_items(payload.get("camera_poses"))
        if re.search(r"_(?:final|middle)_[12]\.png$", str(item.get("_key") or key))
    }
    return _views_from_items(
        approved,
        pointer="/camera_poses",
        group="gt_alignment_views",
        role="gt_view",
        render_mode="static_objects",
    )


def _physical_contact(payload: dict[str, Any]) -> list[ViewSpec]:
    def flags(*fields: str) -> dict[str, bool]:
        output: dict[str, bool] = {}
        for field in fields:
            for key, item in _keyed_items(payload.get(field)):
                output[str(item.get("_key") or key)] = bool(item.get("value"))
        return output

    new_existence = flags("exist_obj")
    old_existence = flags("exist", "exist_left", "exist_right")
    touching = flags("touching", "touching_left", "touching_right")
    answer = str(payload.get("_ground_truth") or "").strip().lower() in {"yes", "true", "1"}
    approved: list[tuple[str, dict[str, Any]]] = []
    for key, item in _keyed_items(payload.get("camera_poses")):
        token = Path(str(item.get("_key") or key)).stem
        if new_existence:
            visible = new_existence.get(f"exist_obj1_{token}", False) and new_existence.get(f"exist_obj2_{token}", False)
        else:
            visible = old_existence.get(f"exist_{token}", False)
        if not visible or touching.get(f"touching_{token}", not answer) != answer:
            continue
        approved.append((key, item))

    output: list[ViewSpec] = []
    for approved_index, (key, item) in enumerate(approved):
        output.append(
            _view(
                item,
                group="gt_contact_views",
                role="gt_view",
                pointer=f"/camera_poses/{key}",
                fallback_name=f"{_safe_component(key)}.png",
                render_mode="static_objects",
                baseline_selected=approved_index < 5,
            )
        )
    if output:
        return output
    candidates = _keyed_items(payload.get("camera_poses"))
    if not candidates:
        return []

    def candidate_visible(candidate_key: str, candidate: dict[str, Any]) -> bool:
        token = Path(str(candidate.get("_key") or candidate_key)).stem
        if new_existence:
            return new_existence.get(f"exist_obj1_{token}", False) and new_existence.get(
                f"exist_obj2_{token}", False
            )
        return old_existence.get(f"exist_{token}", False)

    key, item = next(
        ((candidate_key, candidate) for candidate_key, candidate in candidates if candidate_visible(candidate_key, candidate)),
        candidates[0],
    )
    return [
        _view(
            item,
            group="gt_contact_views",
            role="gt_rule_camera",
            pointer=f"/camera_poses/{key}",
            fallback_name=f"{_safe_component(key)}.png",
            render_mode="static_objects",
            pose_source="saved_candidate_camera_for_contact_gt_state_replay",
        )
    ]


def _mirror(payload: dict[str, Any]) -> list[ViewSpec]:
    render = ((payload.get("question_data") or {}).get("render") or {})
    gt_input = render.get("gt_view_input") or {}
    small_task = str(payload.get("_hf_small_task") or "")
    default_fov = 70.0 if small_task == "Spatial Relations" else 100.0
    fov = gt_input.get("fov_deg") or render.get("fov_deg") or default_fov
    return _views_from_items(
        gt_input.get("views"),
        pointer="/question_data/render/gt_view_input/views",
        group="gt_orbit",
        fov_deg=float(fov),
        render_mode="static_objects",
    )


def _unobserved(payload: dict[str, Any]) -> list[ViewSpec]:
    render = ((payload.get("question_data") or {}).get("render") or {})
    gt_view = render.get("gt_view") or {}
    phase_inputs = _keyed_map(render.get("image"))
    output: list[ViewSpec] = []
    for phase in ("image1", "image2"):
        phase_record = gt_view.get(phase) or {}
        phase_views = _views_from_items(
            phase_record.get("images"),
            pointer=f"/question_data/render/gt_view/{phase}/images",
            group=f"gt_view/{phase}",
            role="gt_reveal",
            fov_deg=float(phase_record.get("fov_deg") or 55.0),
            phase=phase,
            state_key=phase,
            render_mode="state_snapshot",
        )
        if phase_views:
            output.extend(phase_views)
            continue
        # The failed reveal capture still records the phase's selected primary
        # camera. Reuse it after the GT hidden-state rule is replayed.
        phase_input = phase_inputs.get(phase) or {}
        candidates = _keyed_items(phase_input.get("camera_poses"))
        if candidates:
            key, item = candidates[0]
            output.append(
                _view(
                    item,
                    group=f"gt_view/{phase}",
                    role="gt_rule_camera",
                    pointer=f"/question_data/render/image/{phase}/camera_poses/0",
                    fallback_name=f"{phase}_gt_rule.png",
                    fov_deg=float(phase_record.get("fov_deg") or 55.0),
                    phase=phase,
                    state_key=phase,
                    render_mode="state_snapshot",
                    pose_source="saved_phase_camera_for_unobserved_gt_state_replay",
                )
            )
    return output


def _agent(payload: dict[str, Any]) -> list[ViewSpec]:
    return _views_from_items(
        payload.get("observer_poses"),
        pointer="/observer_poses",
        group="trajectory",
        role="gt_trajectory",
        render_mode="dynamic_replay",
    )


Selector = Callable[[dict[str, Any]], list[ViewSpec]]


SELECTORS: dict[tuple[str, str], Selector] = {
    ("Action Sequencing", "Action Order Inference"): _action,
    ("Cognitive Mapping", "Connectivity"): lambda p: _cognitive(p, False, False),
    ("Cognitive Mapping", "Long-Term Navigation"): lambda p: _cognitive(p, False, False),
    ("Cognitive Mapping", "Regional Boundary"): lambda p: _cognitive(p, True, True),
    ("Cognitive Mapping", "Traversable Passage"): lambda p: _cognitive(p, True, False),
    ("Enumerative Perception", "Category Ambiguity"): _enumerative,
    ("Enumerative Perception", "Counting w Occlusion"): _enumerative,
    ("Enumerative Perception", "Illumination Variability"): _enumerative,
    ("Enumerative Perception", "Merged Observation"): _enumerative,
    ("Enumerative Perception", "Spatial Segmentation"): _enumerative,
    ("Enumerative Perception", "Structural Enclosure"): _enumerative,
    ("Metric Comparison", "Dimensional Size"): _dimensional_size,
    ("Metric Comparison", "Spatial Distance"): _spatial_distance,
    ("Perceptual Grounding", "Material Transparency"): _material_transparency,
    ("Perceptual Grounding", "Partial Occlusion"): lambda p: _partial_observation(p, require_occluder=True),
    ("Perceptual Grounding", "View Hallucination"): lambda p: _partial_observation(p, require_occluder=False),
    ("Physical Dynamics", "Inclined Plane"): _inclined_plane,
    ("Physical Dynamics", "Stacking & Stability"): _stacking,
    ("Physical Structure", "Deformable"): _deformable,
    ("Physical Structure", "Liquid Volume"): _liquid,
    ("Physical Structure", "Rigid Containment"): _rigid,
    ("Spatial Relations", "Geometric Configuration"): _geometric,
    ("Spatial Relations", "Linear Alignment"): _linear,
    ("Spatial Relations", "Physical Contact"): _physical_contact,
    ("Specular Reflection", "Correspondence"): _mirror,
    ("Specular Reflection", "Reflection Authoring"): _mirror,
    ("Specular Reflection", "Spatial Relations"): _mirror,
    ("Temporal Understanding", "Agent Observation"): _agent,
    ("Temporal Understanding", "Unobserved Change"): _unobserved,
}


def _deduplicate(specs: Iterable[ViewSpec]) -> list[ViewSpec]:
    output: list[ViewSpec] = []
    seen: set[str] = set()
    for spec in specs:
        candidate = spec
        suffix = 2
        while candidate.view_id in seen:
            path = Path(spec.view_id)
            candidate = ViewSpec(
                **{
                    **asdict(spec),
                    "view_id": str(path.with_name(f"{path.stem}__{suffix}{path.suffix}")),
                }
            )
            suffix += 1
        seen.add(candidate.view_id)
        output.append(candidate)
    return output


def extract_view_specs(payload: dict[str, Any]) -> list[ViewSpec]:
    key = _task_key(payload)
    selector = SELECTORS.get(key)
    if selector is None:
        raise KeyError(f"No Passive-GT selector registered for {key!r}")
    return _deduplicate(selector(payload))
