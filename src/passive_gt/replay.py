from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .manifest import (
    AVAILABLE_VIEW_STATUSES,
    SCHEMA_VERSION,
    atomic_write_json,
    image_is_materializable,
    inspect_image,
    sha256_file,
)
from .pipeline import load_question, sample_directory, sample_issues
from .views import ViewSpec, extract_view_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_EXPLORE_ROOT = REPO_ROOT / "src" / "active_explore"
SUPPORTED_RENDER_MODES = {
    "static_scene",
    "static_objects",
    "action_snapshot",
    "stack_snapshot",
    "rigid_initial_snapshot",
    "placement_snapshot",
    "special_state",
}


def _ensure_active_imports():
    # Match the active-explore launcher and avoid two machine-specific Isaac
    # startup failures.  Emitter APIs can abort under the installed Kit build,
    # while extension hot-reload consumes tens of thousands of inotify watches
    # even though a batch renderer never reloads extensions.
    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    # Isaac's LLVM / shader startup can otherwise exhaust this 16-GB host when
    # each numerical backend creates its default worker pool.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "2")
    fs_watcher_arg = "--/app/extensions/fsWatcherEnabled=false"
    if fs_watcher_arg not in sys.argv:
        sys.argv.append(fs_watcher_arg)
    if str(ACTIVE_EXPLORE_ROOT) not in sys.path:
        sys.path.insert(0, str(ACTIVE_EXPLORE_ROOT))
    import cv2
    import omnigibson as og
    import torch as th
    import yaml
    from omnigibson.macros import gm
    from tasks._registry import module_for_payload

    gm.ENABLE_FLATCACHE = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    return og, th, yaml, cv2, module_for_payload


def call_with_supported_args(func, *args, **kwargs):
    parameters = inspect.signature(func).parameters
    return func(*args, **{key: value for key, value in kwargs.items() if key in parameters})


def build_env_config(og, yaml, task_module, scene_name: str, room_name: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    hook = getattr(task_module, "build_env_config", None)
    full_scene = bool(getattr(task_module, "FULL_SCENE", False))
    if hook is not None:
        return call_with_supported_args(hook, scene_name, room_name, "R1", objects, full_scene=full_scene)
    config_path = Path(og.example_config_path) / "r1_primitives.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    scene = config.setdefault("scene", {})
    scene["scene_model"] = scene_name
    scene["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    if room_name and not full_scene:
        scene["load_room_instances"] = [room_name]
    else:
        scene.pop("load_room_instances", None)
        scene.pop("load_room_types", None)
    config["objects"] = objects
    return config


def configure_viewer_resolution(
    config: dict[str, Any], *, width: int = 1280, height: int = 720
) -> dict[str, Any]:
    """Create the viewer render product at its final size.

    Reassigning ``VisionSensor.image_width`` / ``image_height`` after a large
    scene has loaded destroys and recreates the render product once per
    assignment.  That transient duplication can exhaust a 12-GB GPU even when
    the requested resolution is already the default.
    """
    render = config.setdefault("render", {})
    render["viewer_width"] = int(width)
    render["viewer_height"] = int(height)
    return config


def set_camera(og, position: tuple[float, ...], quaternion: tuple[float, ...]) -> None:
    og.sim._viewer_camera.set_position_orientation(position=np.array(position), orientation=np.array(quaternion))
    for _ in range(10):
        og.sim.render()


def capture_image(og, cv2, path: Path, *, render_steps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(max(int(render_steps), 1)):
        og.sim.render()
    rgb = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        if not cv2.imwrite(tmp_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"OpenCV failed to encode {path}")
        # Replacing instead of truncating is important: reused assets may be
        # hardlinks, and overwriting one must never mutate the source dataset.
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def set_camera_fov(og, fov_deg: float | None) -> None:
    if fov_deg is None:
        return
    camera = og.sim._viewer_camera
    aperture = float(camera.horizontal_aperture)
    camera.focal_length = aperture / (2.0 * math.tan(math.radians(float(fov_deg)) * 0.5))


def prepare_passive_config(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Match the original generator's scene/robot policy for static replay."""
    scene = config.setdefault("scene", {})
    big = str(payload.get("_hf_big_task") or "")
    small = str(payload.get("_hf_small_task") or "")
    if big in {"Cognitive Mapping", "Enumerative Perception", "Specular Reflection"}:
        config["robots"] = []
    if big == "Enumerative Perception":
        # The producer loads the target room's native clutter, then adds the
        # resolved generated objects.  The active runner's floor-only shortcut
        # is useful for exploration stability but is not faithful GT replay.
        scene["load_object_categories"] = None
        scene["not_load_object_categories"] = None
        scene.pop("load_room_instances", None)
        scene.pop("load_room_types", None)
    if big == "Specular Reflection":
        # Mirror GT was generated in the full house; walls are essential to
        # reflection/occlusion context and must not inherit the generic omit list.
        scene["load_object_categories"] = None
        scene["not_load_object_categories"] = None
        scene.pop("load_room_instances", None)
        scene.pop("load_room_types", None)
    if small in {"Partial Occlusion", "View Hallucination"}:
        # Both grounding generators used a full scene and generated their own
        # target/occluder objects on top of it.
        scene.pop("load_room_instances", None)
        scene.pop("load_room_types", None)
    if small == "Spatial Distance":
        excluded = list((payload.get("_gt_auxiliary") or {}).get("nearby_excluded") or [])
        scene["not_load_object_categories"] = ["ceilings", "carpet", "walls", *excluded]
    if small == "Geometric Configuration":
        scene["not_load_object_categories"] = ["ceilings"]
    if small in {"Connectivity", "Long-Term Navigation", "Regional Boundary", "Traversable Passage"}:
        scene["not_load_object_categories"] = ["ceilings", "carpet", "door", "sliding_door"]
        scene.pop("load_room_instances", None)
        scene.pop("load_room_types", None)
    return config


def _keyed_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if isinstance(item, dict)}
    if isinstance(value, list):
        return {
            str(item.get("_key")): item
            for item in value
            if isinstance(item, dict) and item.get("_key") not in (None, "")
        }
    return {}


def _scale_list(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [1.0, 1.0, 1.0]


def build_passive_objects(task_module, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Add outcome-only objects omitted by active-explore initial-state modules."""
    objects = list(task_module.build_env_objects(payload))
    small_task = str(payload.get("_hf_small_task") or "")
    if small_task == "Deformable":
        # Passive-GT is explicitly the reveal frame *without* cloth.  Building
        # and dropping the deformable cloth first is both expensive and less
        # faithful than loading the saved small-item pose directly.
        seed = int(payload.get("seed", 0) or 0)
        item = payload.get("small_item") or {}
        pose = item.get("pose_after_cover") or {}
        if not item.get("category") or not item.get("model") or not pose.get("position"):
            raise ValueError("Deformable GT reveal is missing small-item asset or saved pose")
        return [
            {
                "type": "DatasetObject",
                "name": f"cover_small_item_render_item_{seed:010d}",
                "category": item["category"],
                "model": item["model"],
                "position": pose["position"],
                "orientation": pose.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0],
                "visual_only": True,
            }
        ]
    if small_task != "Rigid Containment":
        return objects
    existing_names = {str(item.get("name")) for item in objects}
    initial = _keyed_map(payload.get("initial_poses"))
    for label, item in _keyed_map(payload.get("extras")).items():
        name = f"obj_{label}"
        if name in existing_names:
            continue
        pose = initial.get(name) or {}
        objects.append(
            {
                "type": "DatasetObject",
                "name": name,
                "category": item["cat"],
                "model": item["model"],
                "scale": _scale_list(item.get("scale")),
                "position": pose.get("position") or [190.0 + len(objects) * 5.0, 100.0, 100.0],
                "orientation": pose.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0],
            }
        )
    return objects


def _set_object_pose(scene, name: str, pose: dict[str, Any], th) -> bool:
    obj = scene.object_registry("name", name)
    if obj is None or not pose.get("position") or not pose.get("quaternion_xyzw"):
        return False
    obj.set_position_orientation(
        position=th.tensor(pose["position"], dtype=th.float32),
        orientation=th.tensor(pose["quaternion_xyzw"], dtype=th.float32),
    )
    try:
        obj.keep_still()
    except Exception:
        pass
    return True


def restore_configured_object_poses(scene, objects: list[dict[str, Any]], th) -> list[str]:
    """Undo initialization/postprocess drift before rendering saved GT cameras."""
    restored = []
    for item in objects:
        name = str(item.get("name") or "")
        position = item.get("position")
        orientation = item.get("orientation") or item.get("quaternion_xyzw")
        if not name or not position or not orientation:
            continue
        obj = scene.object_registry("name", name)
        if obj is not None:
            # Passive-GT metadata describes a saved state.  Freezing generated
            # objects prevents the synchronization step below from changing
            # that state while still updating USD render transforms / AABBs.
            try:
                obj.visual_only = True
            except Exception:
                pass
        if _set_object_pose(
            scene,
            name,
            {"position": position, "quaternion_xyzw": orientation},
            th,
        ):
            restored.append(name)
    return restored


def apply_stack_snapshot(env, payload: dict[str, Any], state_key: str, th) -> dict[str, Any]:
    if state_key == "stack_initial":
        restored = []
        for name, pose in _keyed_map(payload.get("initial_poses")).items():
            obj = env.scene.object_registry("name", name)
            if obj is None:
                continue
            try:
                obj.visual_only = True
            except Exception:
                pass
            if _set_object_pose(env.scene, name, pose, th):
                restored.append(name)
        if not restored:
            raise ValueError("Stacking initial state has no restorable object poses")
        return {
            "state_adapter": "stack_initial_snapshot",
            "restored_objects": restored,
        }
    try:
        trial_index = int(state_key.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid stacking state key: {state_key}") from exc
    trials = [item for _key, item in _keyed_map(payload.get("trials")).items()]
    # Cleaned lists retain their order even when their _key is a trial number.
    if isinstance(payload.get("trials"), list):
        trials = [item for item in payload["trials"] if isinstance(item, dict)]
    if trial_index >= len(trials):
        raise IndexError(f"Stacking trial {trial_index} is missing")
    snaps = _keyed_map(trials[trial_index].get("snaps"))
    restored = []
    for name, pose in snaps.items():
        obj = env.scene.object_registry("name", name)
        if obj is None:
            continue
        # These are outcome snapshots, not a physics continuation.  Freezing
        # the rigid bodies lets one simulator step synchronize their render
        # transforms / AABBs without letting the saved stack fall or drift.
        try:
            obj.visual_only = True
        except Exception:
            pass
        if _set_object_pose(env.scene, name, pose, th):
            restored.append(name)
    if not restored:
        raise ValueError(f"Stacking trial {trial_index} has no restorable snaps")
    import omnigibson as og

    og.sim.step()
    live_objects: dict[str, Any] = {}
    for name in restored:
        obj = env.scene.object_registry("name", name)
        position, quaternion = obj.get_position_orientation()
        aabb_min, aabb_max = obj.aabb
        live_position = position.detach().cpu().numpy()
        saved_position = np.asarray(snaps[name]["position"], dtype=float)
        pose_error = float(np.linalg.norm(live_position - saved_position))
        if pose_error > 0.01:
            raise RuntimeError(
                f"Stacking object {name!r} did not restore to its GT snapshot "
                f"(position error {pose_error:.4f} m)"
            )
        saved_aabb_min = np.asarray(snaps[name].get("aabb_min") or [], dtype=float)
        saved_aabb_max = np.asarray(snaps[name].get("aabb_max") or [], dtype=float)
        live_aabb_min = aabb_min.detach().cpu().numpy()
        live_aabb_max = aabb_max.detach().cpu().numpy()
        aabb_error = None
        if saved_aabb_min.shape == (3,) and saved_aabb_max.shape == (3,):
            aabb_error = float(
                max(
                    np.max(np.abs(live_aabb_min - saved_aabb_min)),
                    np.max(np.abs(live_aabb_max - saved_aabb_max)),
                )
            )
            if aabb_error > 0.02:
                raise RuntimeError(
                    f"Stacking object {name!r} rendered AABB differs from its GT snapshot "
                    f"by {aabb_error:.4f} m"
                )
        live_objects[name] = {
            "position": live_position.tolist(),
            "quaternion_xyzw": quaternion.detach().cpu().numpy().tolist(),
            "aabb_min": live_aabb_min.tolist(),
            "aabb_max": live_aabb_max.tolist(),
            "saved_position_error_m": pose_error,
            "saved_aabb_max_error_m": aabb_error,
        }
    floor = env.scene.object_registry("name", str(payload.get("floor") or ""))
    floor_aabb = None
    if floor is not None:
        floor_min, floor_max = floor.aabb
        floor_aabb = {
            "min": floor_min.detach().cpu().numpy().tolist(),
            "max": floor_max.detach().cpu().numpy().tolist(),
        }
    return {
        "state_adapter": "stack_snapshot",
        "trial_index": trial_index,
        "restored_objects": restored,
        "live_objects": live_objects,
        "floor_aabb": floor_aabb,
    }


def apply_action_snapshot(env, payload: dict[str, Any], state_key: str, th) -> dict[str, Any]:
    try:
        stage_index = int(state_key.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid action stage state key: {state_key}") from exc
    auxiliary = payload.get("_gt_auxiliary") or {}
    stages = [item for item in auxiliary.get("stages") or [] if isinstance(item, dict)]
    if stage_index >= len(stages):
        raise IndexError(f"Action stage {stage_index} is missing from auxiliary metadata")
    poses = _keyed_map(stages[stage_index].get("all_poses"))
    restored = [name for name, pose in poses.items() if _set_object_pose(env.scene, name, pose, th)]
    if not restored:
        raise ValueError(f"Action stage {stage_index} has no restorable object poses")
    return {
        "state_adapter": "action_snapshot",
        "stage_index": stage_index,
        "stage_label": stages[stage_index].get("label"),
        "restored_objects": restored,
        "auxiliary_path": payload.get("_gt_auxiliary_path"),
    }


def apply_rigid_initial_snapshot(env, payload: dict[str, Any], th) -> dict[str, Any]:
    restored = [
        name
        for name, pose in _keyed_map(payload.get("initial_poses")).items()
        if _set_object_pose(env.scene, name, pose, th)
    ]
    if not restored:
        raise ValueError("Rigid sample has no restorable initial poses")
    return {"state_adapter": "rigid_initial_snapshot", "restored_objects": restored}


def apply_deformable_gt_state(env, payload: dict[str, Any], th) -> dict[str, Any]:
    seed = int(payload.get("seed", 0) or 0)
    cloth_name = f"cover_small_item_render_cloth_{seed:010d}"
    item_name = f"cover_small_item_render_item_{seed:010d}"
    cloth = env.scene.object_registry("name", cloth_name)
    removed = False
    if cloth is not None:
        try:
            env.scene.remove_object(cloth)
            removed = True
        except Exception:
            try:
                cloth.visible = False
                removed = True
            except Exception:
                pass
    pose = ((payload.get("small_item") or {}).get("pose_after_cover") or {})
    restored = _set_object_pose(env.scene, item_name, pose, th)
    if not restored:
        raise ValueError("Deformable GT item pose could not be restored")
    return {
        "state_adapter": "cloth_removed",
        "cloth_name": cloth_name,
        "cloth_removed_or_hidden": removed,
        "item_name": item_name,
        "item_pose_source": "/small_item/pose_after_cover",
    }


def _rigid_container_name(payload: dict[str, Any], fit_label: str) -> str | None:
    for _key, item in _keyed_map(payload.get("containers")).items():
        label = str(item.get("fit_check") or item.get("_key") or "")
        if label != fit_label:
            continue
        idx = item.get("idx")
        if idx is None:
            idx = {"small": 0, "fit": 1, "big": 2}.get(label)
        return f"obj_container_{idx}" if idx is not None else None
    return None


def apply_placement_snapshot(env, payload: dict[str, Any], state_key: str, th) -> dict[str, Any]:
    label = state_key.split(":", 1)[-1]
    ordered_items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload.get("containee"), dict):
        ordered_items.append(("containee", payload["containee"]))
    ordered_items.extend(_keyed_map(payload.get("extras")).items())
    item = next((record for item_label, record in ordered_items if item_label == label), {})
    object_name = f"obj_{label}"
    placement = item.get("placement") or {}
    if not placement:
        raise ValueError(f"Rigid placement state {label!r} is missing")

    restored = []
    for name, pose in _keyed_map(payload.get("initial_poses")).items():
        if _set_object_pose(env.scene, name, pose, th):
            restored.append(name)
    cumulative_labels = []
    target_container = None
    for item_label, ordered_item in ordered_items:
        ordered_placement = ordered_item.get("placement") or {}
        cumulative_object = f"obj_{item_label}"
        if not _set_object_pose(env.scene, cumulative_object, ordered_placement.get("object_pose_after") or {}, th):
            raise ValueError(f"Rigid outcome object {cumulative_object!r} is unavailable")
        cumulative_container = _rigid_container_name(payload, str(ordered_item.get("target_container") or ""))
        if cumulative_container and ordered_placement.get("container_pose_after"):
            _set_object_pose(env.scene, cumulative_container, ordered_placement["container_pose_after"], th)
        cumulative_labels.append(item_label)
        if item_label == label:
            target_container = cumulative_container
            break
    if label not in cumulative_labels:
        raise ValueError(f"Rigid placement label {label!r} is unavailable")

    robot_pose = placement.get("robot_pose_after_place") or {}
    if getattr(env, "robots", None) and robot_pose.get("robot_position") and robot_pose.get("robot_quaternion_xyzw"):
        try:
            env.robots[0].set_position_orientation(
                position=th.tensor(robot_pose["robot_position"], dtype=th.float32),
                orientation=th.tensor(robot_pose["robot_quaternion_xyzw"], dtype=th.float32),
            )
        except Exception:
            pass
    return {
        "state_adapter": "placement_snapshot",
        "placement_label": label,
        "outcome_object": object_name,
        "target_container": target_container,
        "cumulative_placements": cumulative_labels,
        "restored_initial_objects": restored,
    }


def apply_view_state(env, payload: dict[str, Any], spec: ViewSpec, th) -> dict[str, Any]:
    if spec.render_mode == "action_snapshot":
        return apply_action_snapshot(env, payload, str(spec.state_key or ""), th)
    if spec.render_mode == "stack_snapshot":
        return apply_stack_snapshot(env, payload, str(spec.state_key or ""), th)
    if spec.render_mode == "rigid_initial_snapshot":
        return apply_rigid_initial_snapshot(env, payload, th)
    if spec.render_mode == "placement_snapshot":
        return apply_placement_snapshot(env, payload, str(spec.state_key or ""), th)
    if spec.render_mode == "special_state":
        return apply_deformable_gt_state(env, payload, th)
    return {"state_adapter": "static"}


def _load_record(output_root: Path, sample_dir: Path) -> dict[str, Any]:
    path = sample_dir / "manifest.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return {"views": [], "status": "planned"}


def _initialize_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    sample_dir: Path,
    specs: list[ViewSpec],
) -> dict[str, Any]:
    views = []
    for spec in specs:
        view = spec.to_dict()
        view["output_path"] = str((sample_dir / spec.view_id).relative_to(output_root))
        view["status"] = "planned"
        views.append(view)
    try:
        source_label = str(source_json.relative_to(dataset_root))
    except ValueError:
        source_label = str(source_json)
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
        "issues": sample_issues(payload, specs),
        "views": views,
    }


def _update_view_record(record: dict[str, Any], spec: ViewSpec, updates: dict[str, Any]) -> None:
    for view in record.get("views") or []:
        if view.get("view_id") == spec.view_id:
            view.update(updates)
            return


def render_static_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    only_view_ids: set[str] | None = None,
) -> dict[str, Any]:
    row, payload = load_question(source_json)
    all_specs = extract_view_specs(payload)
    specs = [
        spec
        for spec in all_specs
        if spec.render_mode in SUPPORTED_RENDER_MODES and (not only_view_ids or spec.view_id in only_view_ids)
    ]
    sample_dir = sample_directory(row, source_json, dataset_root, output_root)
    record = _load_record(output_root, sample_dir)
    if not record.get("views"):
        record = _initialize_record(row, payload, source_json, dataset_root, output_root, sample_dir, all_specs)
    if not specs:
        return {"source_json": str(source_json), "rendered": 0, "failed": 0, "skipped": "no_supported_views"}

    pending: list[ViewSpec] = []
    for spec in specs:
        destination = sample_dir / spec.view_id
        if destination.is_file() and not overwrite:
            if image_is_materializable(inspect_image(destination)):
                continue
        pending.append(spec)
    if not pending:
        return {"source_json": str(source_json), "rendered": 0, "failed": 0, "skipped": "already_complete"}

    print(json.dumps({"stage": "import_runtime", "source_json": str(source_json)}), flush=True)
    og, th, yaml, cv2, module_for_payload = _ensure_active_imports()
    module_name = module_for_payload(payload)
    if not module_name:
        raise ValueError(f"No active task module for {row.get('big_task')}/{row.get('small_task')}")
    task_module = importlib.import_module(module_name)
    # Passive-GT static/snapshot replay never consumes thermal, fire, steam,
    # or transition-rule states.  Leaving them enabled can make unrelated room
    # furniture create emitter meshes during scene import; some legacy assets
    # have non-orthogonal emitter transforms and abort the entire worker before
    # any GT camera is captured.
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False
    if str(payload.get("_hf_small_task") or "") == "Deformable":
        # The active task module enables GPU dynamics for cloth simulation at
        # import time.  Passive-GT reveal reconstruction deliberately has no
        # cloth, so restore the lightweight rigid/static setting before env
        # creation.
        gm.USE_GPU_DYNAMICS = False
    scene_name, room_name = task_module.scene_room(payload)
    config_proxy = SimpleNamespace(json_root=dataset_root, robot="R1")
    preprocess = getattr(task_module, "preprocess", None)
    if str(payload.get("_hf_small_task") or "") == "Rigid Containment":
        # The active runner intentionally skips variant rows, but Passive GT
        # includes their saved after-placement snapshots.
        preprocess = None
    task_state = call_with_supported_args(preprocess, payload, source_json=source_json, config=config_proxy) if preprocess else {}
    task_state = task_state if isinstance(task_state, dict) else {}
    if task_state.get("skip_reason"):
        raise RuntimeError(f"Task preprocess skipped sample: {task_state['skip_reason']}")
    first = pending[0]
    if first.position is None or first.quaternion_xyzw is None:
        raise ValueError(f"Supported render view {first.view_id!r} has no camera pose")
    camera_info = {
        "camera_pose": {
            "position": list(first.position),
            "quaternion_xyzw": list(first.quaternion_xyzw),
        },
        "selection": "passive_gt_saved_pose",
    }
    objects = build_passive_objects(task_module, payload)
    env_config = build_env_config(og, yaml, task_module, scene_name, room_name, objects)
    if str(payload.get("_hf_small_task") or "") == "Deformable":
        # The active cloth task intentionally constructs its dynamic objects in
        # postprocess and therefore clears config["objects"].  Passive reveal
        # replay skips that postprocess and supplies the saved item directly.
        env_config["objects"] = objects
    env_config = prepare_passive_config(env_config, payload)
    env_config = configure_viewer_resolution(env_config)

    env = None
    rendered = 0
    warnings = 0
    failed = 0
    failures: list[dict[str, str]] = []
    run_completed = False
    try:
        print(json.dumps({"stage": "load_environment", "scene": scene_name, "room": room_name}), flush=True)
        env = og.Environment(configs=env_config)
        try:
            og.sim._viewer_camera.add_modality("rgb")
        except Exception:
            pass
        camera = og.sim._viewer_camera
        settle_steps = 0 if getattr(task_module, "SKIP_INITIAL_SETTLE", False) else int(getattr(task_module, "INITIAL_SETTLE_STEPS", 30))
        for _ in range(settle_steps):
            og.sim.step()
        postprocess = getattr(task_module, "postprocess_env", None)
        if str(payload.get("_hf_small_task") or "") == "Deformable":
            # GT is the producer's cloth-removed reveal; no cloth dynamics are
            # part of the state to reconstruct.
            postprocess = None
        env_postprocess = (
            call_with_supported_args(postprocess, env, payload, camera_info, task_state=task_state)
            if postprocess
            else {}
        )
        env_postprocess = env_postprocess if isinstance(env_postprocess, dict) else {}
        restored_base_objects = restore_configured_object_poses(env.scene, objects, th)
        # set_position_orientation updates the reported root pose immediately,
        # but OmniGibson does not publish the corresponding render transform and
        # AABB until a simulator step.  Without this, a valid PNG can silently
        # show the object's previous location.
        if restored_base_objects:
            og.sim.step()
        print(json.dumps({"stage": "capture_views", "count": len(pending)}), flush=True)

        active_state_key: tuple[str, str | None] | None = None
        state_details: dict[str, Any] = {}
        for spec in pending:
            destination = sample_dir / spec.view_id
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                state_key = (spec.render_mode, spec.state_key)
                if state_key != active_state_key:
                    state_details = apply_view_state(env, payload, spec, th)
                    active_state_key = state_key
                    # State adapters teleport frozen objects to a saved
                    # trajectory/snapshot pose; synchronize that pose before
                    # capturing.  Stack snapshots also self-validate their live
                    # AABBs, so the extra step is intentionally harmless.
                    if state_details.get("state_adapter") != "static":
                        og.sim.step()
                if spec.position is None or spec.quaternion_xyzw is None:
                    raise ValueError("Camera pose requires full task regeneration")
                set_camera_fov(og, spec.fov_deg)
                set_camera(og, spec.position, spec.quaternion_xyzw)
                capture_image(og, cv2, destination)
                quality = inspect_image(destination)
                if not image_is_materializable(quality):
                    raise RuntimeError(f"Rendered image could not be decoded: {quality.get('quality_reasons')}")
                status = "rendered_pose" if quality.get("valid") else "rendered_pose_warning"
                warnings += int(not quality.get("valid"))
                if spec.render_mode in {"action_snapshot", "stack_snapshot"}:
                    provenance = "approximate_snapshot_reconstruction"
                    fidelity = "semantic_gt_state_not_pixel_exact"
                elif spec.render_mode in {"rigid_initial_snapshot", "placement_snapshot"}:
                    provenance = "derived_gt_camera_saved_state_replay"
                    fidelity = "derived_camera_and_saved_object_state"
                else:
                    provenance = "saved_state_replay"
                    fidelity = "saved_camera_and_restored_state"
                details = {
                    "environment": "metadata_replay",
                    "module": module_name,
                    "fov_deg": spec.fov_deg,
                    "resolution": [1280, 720],
                    "robot_free": not bool(env_config.get("robots")),
                    "env_postprocess": env_postprocess,
                    "restored_base_objects": restored_base_objects,
                    "view_state": state_details,
                    "fidelity": fidelity,
                }
                _update_view_record(
                    record,
                    spec,
                    {
                        "status": status,
                        "error": None,
                        "provenance": provenance,
                        "sha256": sha256_file(destination),
                        "quality": quality,
                        "width": quality.get("width"),
                        "height": quality.get("height"),
                        "render_details": details,
                    },
                )
                rendered += 1
            except Exception as exc:
                failed += 1
                failure = {"view_id": spec.view_id, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                _update_view_record(record, spec, {"status": "render_failed", **failure})
        statuses = [view.get("status") for view in record.get("views") or []]
        if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses):
            record["status"] = "complete"
        elif rendered:
            record["status"] = "partial"
        elif failed:
            record["status"] = "failed"
        else:
            record["status"] = "planned"
        atomic_write_json(sample_dir / "manifest.json", record)
        result = {
            "source_json": str(source_json),
            "rendered": rendered,
            "warnings": warnings,
            "failed": failed,
            "failures": failures,
        }
        run_completed = True
        return result
    finally:
        if run_completed:
            try:
                cleanup = getattr(task_module, "cleanup_runtime", None)
                if cleanup:
                    call_with_supported_args(cleanup, env, payload, task_state=task_state)
            except Exception:
                pass
            try:
                if getattr(og, "app", None) is not None:
                    og.shutdown()
            except BaseException:
                pass
