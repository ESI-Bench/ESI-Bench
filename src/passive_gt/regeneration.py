from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .manifest import (
    AVAILABLE_VIEW_STATUSES,
    atomic_write_json,
    image_is_materializable,
    inspect_image,
    sha256_file,
)
from .pipeline import load_question, sample_directory
from .views import extract_view_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
POUR_SCRIPT = REPO_ROOT / "src" / "dataset_generation" / "task_capacity" / "batch_pour.py"
CAPACITY_JSON = POUR_SCRIPT.with_name("fillable_capacity.json")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generator module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capacity_entry(instances: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    matches = [
        entry
        for entry in instances
        if entry.get("category") == item.get("category") and entry.get("model") == item.get("model")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one fillable-capacity entry for {item.get('category')}/{item.get('model')}, got {len(matches)}"
        )
    return matches[0]


def _tensor_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _load_sample_manifest(path: Path, specs) -> dict[str, Any]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        if isinstance(record, dict):
            return record
    return {"status": "planned", "issues": [], "views": [spec.to_dict() for spec in specs]}


def regenerate_liquid_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the original ESI-Bench four-stage pour using the JSON's exact pair."""
    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    fs_watcher_arg = "--/app/extensions/fsWatcherEnabled=false"
    if fs_watcher_arg not in sys.argv:
        sys.argv.append(fs_watcher_arg)
    row, payload = load_question(source_json)
    if row.get("small_task") != "Liquid Volume":
        raise ValueError("regenerate_liquid_sample received a non-liquid sample")
    specs = extract_view_specs(payload)
    sample_dir = sample_directory(row, source_json, dataset_root, output_root)
    manifest_path = sample_dir / "manifest.json"
    record = _load_sample_manifest(manifest_path, specs)
    destinations = {spec.original_image_path: sample_dir / spec.view_id for spec in specs}
    if not overwrite and destinations and all(
        path.is_file() and image_is_materializable(inspect_image(path)) for path in destinations.values()
    ):
        return {"source_json": str(source_json), "rendered": 0, "failed": 0, "skipped": "already_complete"}

    with CAPACITY_JSON.open("r", encoding="utf-8") as stream:
        instances = (json.load(stream) or {}).get("instances") or []
    obj1 = payload.get("obj1") or {}
    obj2 = payload.get("obj2") or {}
    entry1 = _capacity_entry(instances, obj1)
    entry2 = _capacity_entry(instances, obj2)

    work_dir = sample_dir / f".liquid_regeneration_{os.getpid()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    module = _load_module(POUR_SCRIPT, "passive_gt_batch_pour")

    def installed_asset_scale(category: str, model: str) -> float:
        usd_path = Path(module.DatasetObject.get_usd_path(category=category, model=model))
        encrypted_candidates = (
            Path(f"{usd_path}.encrypted"),
            Path(str(usd_path).replace(".usd", ".encrypted.usd")),
        )
        encrypted = next((path for path in encrypted_candidates if path.is_file()), None)
        if encrypted is not None:
            context = module.decrypted(str(encrypted))
        elif usd_path.is_file():
            context = nullcontext(str(usd_path))
        else:
            raise FileNotFoundError(f"No installed USD asset for {category}/{model}: {usd_path}")
        with context as available_path:
            stage = module.lazy.pxr.Usd.Stage.Open(available_path)
            prim = stage.GetDefaultPrim()
            bounding_box = module.th.tensor(prim.GetAttribute("ig:nativeBB").Get())
        scale = module.MAX_BBOX / module.th.max(bounding_box)
        return min(float(scale), 1.0)

    # The installed assets use '<model>.usdz.encrypted'; the historical script
    # assumes '<model>.encrypted.usdz'.  Keep the generator unchanged and adapt
    # only this invocation to the actual local asset naming convention.
    module.get_scale = installed_asset_scale
    camera_poses: list[dict[str, list[float]]] = []
    original_set_side_camera = module.set_side_camera

    def recording_set_side_camera(left, right):
        original_set_side_camera(left, right)
        position, quaternion = module.og.sim._viewer_camera.get_position_orientation()
        camera_poses.append(
            {
                "position": _tensor_list(position),
                "quaternion_xyzw": _tensor_list(quaternion),
            }
        )

    module.set_side_camera = recording_set_side_camera
    run_idx = int(payload.get("run_idx", 0) or 0)
    try:
        # Match batch_pour.main(): launch Kit before get_scale opens encrypted
        # USDZ assets through pxr.  Opening those stages pre-launch can crash
        # the subsequent GPU-dynamics application startup.
        if module.og.sim:
            module.og.clear()
        else:
            module.og.launch()
        if module.og.sim.is_playing():
            module.og.sim.stop()
        module.process(entry1, entry2, str(work_dir), run_idx)
        result_path = work_dir / "result.json"
        with result_path.open("r", encoding="utf-8") as stream:
            physics_result = json.load(stream)
        measured_diff = int(physics_result.get("capacity_diff_measured", 0))
        expected = str(row.get("answer") or payload.get("gt_larger") or "").strip().lower()
        actual = "left" if measured_diff > 0 else ("right" if measured_diff < 0 else "tie")
        if expected in {"left", "right"} and actual != expected:
            raise RuntimeError(f"Regenerated capacity ordering changed: expected={expected}, actual={actual}")
        if len(camera_poses) != 3:
            raise RuntimeError(f"Expected three live AABB camera updates, got {len(camera_poses)}")
        pose_by_image = {
            "00_before_fill_left.png": camera_poses[0],
            "01_after_fill_left.png": camera_poses[1],
            "02_after_fill_right.png": camera_poses[1],
            "03_final.png": camera_poses[2],
        }

        views_by_id = {str(view.get("view_id")): view for view in record.get("views") or []}
        rendered = 0
        for spec in specs:
            source_image = work_dir / spec.original_image_path
            quality = inspect_image(source_image)
            if not image_is_materializable(quality):
                raise RuntimeError(f"Liquid frame is not materializable: {source_image}: {quality}")
            destination = destinations[spec.original_image_path]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_image, destination)
            view = views_by_id.get(spec.view_id)
            if view is None:
                view = spec.to_dict()
                record.setdefault("views", []).append(view)
            view.update(
                {
                    "camera_pose": pose_by_image[spec.original_image_path],
                    "output_path": str(destination.relative_to(output_root)),
                    "status": "rendered_pose" if quality.get("valid") else "rendered_pose_warning",
                    "provenance": "full_physics_regeneration",
                    "pose_source": "live_AABB_during_ESI-Bench_batch_pour.process",
                    "sha256": sha256_file(destination),
                    "quality": quality,
                    "width": quality.get("width"),
                    "height": quality.get("height"),
                    "render_details": {
                        "generator": str(POUR_SCRIPT.relative_to(REPO_ROOT)),
                        "run_idx": run_idx,
                        "expected_larger": expected,
                        "regenerated_larger": actual,
                        "capacity_diff_measured": measured_diff,
                    },
                }
            )
            rendered += 1
        result_destination = sample_dir / "regeneration_result.json"
        os.replace(result_path, result_destination)
        statuses = [view.get("status") for view in record.get("views") or []]
        record["status"] = (
            "complete" if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses) else "partial"
        )
        atomic_write_json(manifest_path, record)
        return {
            "source_json": str(source_json),
            "rendered": rendered,
            "failed": 0,
            "capacity_order_validated": True,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            if getattr(module.og, "app", None) is not None:
                module.og.shutdown()
        except BaseException:
            pass


def regenerate_inclined_plane_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay the saved slope parameters and capture all 30 x 4 GT frames."""
    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    fs_watcher_arg = "--/app/extensions/fsWatcherEnabled=false"
    if fs_watcher_arg not in sys.argv:
        sys.argv.append(fs_watcher_arg)
    row, payload = load_question(source_json)
    if row.get("small_task") != "Inclined Plane":
        raise ValueError("regenerate_inclined_plane_sample received a non-slope sample")
    specs = extract_view_specs(payload)
    sample_dir = sample_directory(row, source_json, dataset_root, output_root)
    manifest_path = sample_dir / "manifest.json"
    record = _load_sample_manifest(manifest_path, specs)
    destinations = {spec.view_id: sample_dir / spec.view_id for spec in specs}
    if not overwrite and destinations and all(
        path.is_file() and image_is_materializable(inspect_image(path)) for path in destinations.values()
    ):
        return {"source_json": str(source_json), "rendered": 0, "failed": 0, "skipped": "already_complete"}

    from .replay import _ensure_active_imports, capture_image

    og, th, _yaml, cv2, _module_for_payload = _ensure_active_imports()
    import importlib
    import numpy as np

    task_module = importlib.import_module("tasks.physical_dynamics.inclined_plane")
    room = str(payload.get("room") or "")
    room_type = "_".join(room.split("_")[:-1]) if room and room[-1].isdigit() else room
    config = {
        "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 240.0},
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": payload["scene"],
            "load_room_types": [room_type],
        },
        "robots": [],
        "objects": task_module.build_env_objects(payload),
    }
    work_dir = sample_dir / f".slope_regeneration_{os.getpid()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = og.Environment(configs=config)
        try:
            og.sim._viewer_camera.add_modality("rgb")
        except Exception:
            pass
        slope = env.scene.object_registry("name", "slope")
        task_obj = env.scene.object_registry("name", "task_obj")
        if slope is None or task_obj is None:
            raise RuntimeError("Slope environment is missing slope or task_obj")
        for _ in range(10):
            og.sim.step()
        slope.set_position_orientation(
            position=th.tensor(payload["slope"]["position"], dtype=th.float32),
            orientation=th.tensor(payload["slope_quaternion"], dtype=th.float32),
        )
        slope.keep_still()
        for _ in range(10):
            og.sim.step()
        task_module.apply_friction(slope, payload.get("static_friction", 0.5), payload.get("dynamic_friction", 0.5))
        task_module.apply_friction(task_obj, payload.get("static_friction", 0.5), payload.get("dynamic_friction", 0.5))
        slope_min, slope_max = [value.detach().cpu().numpy() for value in slope.aabb]
        slope_mid = (slope_min + slope_max) / 2.0
        task_obj.set_position_orientation(
            position=th.tensor([slope_mid[0], slope_mid[1], slope_mid[2] + 0.05], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        task_obj.keep_still()
        for _ in range(5):
            og.sim.step()
        pos_init = task_obj.get_position_orientation()[0].detach().cpu().numpy().copy()

        specs_by_step: dict[int, list[Any]] = {}
        for spec in specs:
            try:
                step = int(str(spec.state_key).split(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid slope state key {spec.state_key!r}") from exc
            specs_by_step.setdefault(step, []).append(spec)

        # The first RGB observation after an OmniGibson environment starts can
        # still contain the launch / scene-loading camera even after one
        # render.  Warm the exact first GT camera before advancing the saved
        # physics trajectory so step_0001_view_0 is not a stale frame.
        first_step_specs = specs_by_step.get(1) or []
        if first_step_specs:
            first_spec = first_step_specs[0]
            if first_spec.position is None or first_spec.quaternion_xyzw is None:
                raise ValueError(f"Slope view {first_spec.view_id} has no camera pose")
            og.sim._viewer_camera.set_position_orientation(
                position=np.asarray(first_spec.position, dtype=float),
                orientation=np.asarray(first_spec.quaternion_xyzw, dtype=float),
            )
            for _ in range(10):
                og.sim.render()
            og.sim._viewer_camera.get_obs()

        rendered = 0
        for step in range(1, 31):
            og.sim.step()
            for spec in specs_by_step.get(step, []):
                if spec.position is None or spec.quaternion_xyzw is None:
                    raise ValueError(f"Slope view {spec.view_id} has no camera pose")
                og.sim._viewer_camera.set_position_orientation(
                    position=np.asarray(spec.position, dtype=float),
                    orientation=np.asarray(spec.quaternion_xyzw, dtype=float),
                )
                capture_image(og, cv2, work_dir / spec.view_id, render_steps=1)
                rendered += 1
        pos_final = task_obj.get_position_orientation()[0].detach().cpu().numpy().copy()
        obj_min_final = task_obj.aabb[0].detach().cpu().numpy()
        slide_dist = float(np.linalg.norm(pos_final[:2] - pos_init[:2]))
        slid = slide_dist > 0.03
        floor = env.scene.object_registry("name", str(payload.get("floor_name") or ""))
        if floor is None:
            raise RuntimeError(f"Slope floor {payload.get('floor_name')!r} was not loaded")
        floor_z = float(floor.aabb[1][2].detach().cpu().item())
        fallen = bool(obj_min_final[2] < floor_z + 0.05)
        expected_slid = bool(payload.get("ground_truth_slid"))
        expected_fallen = bool(payload.get("ground_truth_fallen"))
        if (slid, fallen) != (expected_slid, expected_fallen):
            raise RuntimeError(
                "Slope replay changed the saved outcome: "
                f"expected slid/fallen={(expected_slid, expected_fallen)}, got {(slid, fallen)}"
            )

        views_by_id = {str(view.get("view_id")): view for view in record.get("views") or []}
        for spec in specs:
            source_image = work_dir / spec.view_id
            quality = inspect_image(source_image)
            if not image_is_materializable(quality):
                raise RuntimeError(f"Slope frame is not materializable: {source_image}: {quality}")
            destination = destinations[spec.view_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_image, destination)
            view = views_by_id.get(spec.view_id)
            if view is None:
                view = spec.to_dict()
                record.setdefault("views", []).append(view)
            view.update(
                {
                    "output_path": str(destination.relative_to(output_root)),
                    "status": "rendered_pose" if quality.get("valid") else "rendered_pose_warning",
                    "provenance": "full_physics_regeneration",
                    "sha256": sha256_file(destination),
                    "quality": quality,
                    "width": quality.get("width"),
                    "height": quality.get("height"),
                    "render_details": {
                        "generator": "metadata-adapted src/dataset_generation/task_physics/batch_slope.py",
                        "physics_timestep": 1.0 / 240.0,
                        "outcome_validated": True,
                        "slide_dist_m": slide_dist,
                        "slid": slid,
                        "fallen": fallen,
                    },
                }
            )
        result = {
            "pos_init": [float(value) for value in pos_init],
            "pos_final": [float(value) for value in pos_final],
            "slide_dist_m": slide_dist,
            "slid": slid,
            "fallen": fallen,
            "expected_slid": expected_slid,
            "expected_fallen": expected_fallen,
        }
        atomic_write_json(sample_dir / "regeneration_result.json", result)
        statuses = [view.get("status") for view in record.get("views") or []]
        record["status"] = (
            "complete" if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses) else "partial"
        )
        atomic_write_json(manifest_path, record)
        return {"source_json": str(source_json), "rendered": rendered, "failed": 0, "outcome_validated": True}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            if getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass


def render_or_regenerate_sample(
    source_json: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    only_view_ids: set[str] | None = None,
    unobserved_container_state: str = "reveal",
) -> dict[str, Any]:
    row, _payload = load_question(source_json)
    if row.get("small_task") == "Liquid Volume":
        if only_view_ids:
            raise ValueError("Liquid physics regeneration is an atomic four-frame trajectory")
        return regenerate_liquid_sample(source_json, dataset_root, output_root, overwrite=overwrite)
    if row.get("small_task") == "Inclined Plane":
        if only_view_ids:
            raise ValueError("Inclined-plane physics regeneration is an atomic 30-step trajectory")
        return regenerate_inclined_plane_sample(source_json, dataset_root, output_root, overwrite=overwrite)
    if row.get("small_task") == "Agent Observation":
        if only_view_ids:
            raise ValueError("Agent Observation reconstruction is an atomic saved observer sequence")
        from .agent_regeneration import regenerate_agent_observation_sample

        return regenerate_agent_observation_sample(
            source_json,
            dataset_root,
            output_root,
            overwrite=overwrite,
            allow_approximate=True,
        )
    if row.get("small_task") == "Unobserved Change":
        if only_view_ids:
            raise ValueError("Unobserved Change reconstruction preserves the producer's two-phase sequence")
        from .unobserved_regeneration import regenerate_unobserved_change_sample

        return regenerate_unobserved_change_sample(
            source_json,
            dataset_root,
            output_root,
            overwrite=overwrite,
            container_view_state=unobserved_container_state,
        )
    from .replay import render_static_sample

    return render_static_sample(
        source_json,
        dataset_root,
        output_root,
        overwrite=overwrite,
        only_view_ids=only_view_ids,
    )
