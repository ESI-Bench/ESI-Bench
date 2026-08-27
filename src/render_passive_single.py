#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "json_clean"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "passive_single_preview"
DEFAULT_ALL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "passive_single"
DEFAULT_EXISTING_RESULTS_ROOT = REPO_ROOT / "outputs" / "results_small_tasks"
DEFAULT_ACTION_SWEEP_ROOT = REPO_ROOT / "outputs" / "task_action_sweep"


def cleanup_dead_rstring_shm() -> list[str]:
    """Remove only Carbonite shared-memory files owned by dead PIDs."""
    removed = []
    shm = Path("/dev/shm")
    for pattern in ("carb-RStringInternals-*", "sem.carb-RStringInternals-*"):
        for path in shm.glob(pattern):
            try:
                pid = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            try:
                os.kill(pid, 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                pass
    return removed


def safe_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or fallback


def load_hf_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        row = json.load(stream)
    if not isinstance(row, dict) or "metadata_json" not in row:
        raise ValueError(f"Not a json_clean question row: {path}")
    return row


def decode_json_column(value: Any) -> Any:
    """Decode JSON-encoded dataset columns while preserving plain strings."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"" or stripped[-1] not in "]}\"":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def question_answer_fields(row: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a stable QA schema shared by every passive-single sidecar."""
    payload = payload or {}
    payload_qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}
    question = row.get("question") or payload_qa.get("question") or payload.get("question")
    answer_options = decode_json_column(row.get("options_json"))
    if answer_options is None:
        answer_options = payload_qa.get("options") or payload.get("options")
    right_answer = decode_json_column(row.get("answer"))
    if right_answer is None:
        right_answer = payload_qa.get("answer_text") or payload.get("answer_text")

    option_id = payload_qa.get("answer_option_id") or payload.get("answer_option_id")
    if option_id is None and isinstance(answer_options, list) and isinstance(right_answer, str):
        for option in answer_options:
            if isinstance(option, dict) and option.get("text") == right_answer:
                option_id = option.get("option_id")
                break

    return {
        "question": question,
        "answer_options": answer_options,
        "answer": right_answer,
        "right_answer": right_answer,
        "right_answer_option_id": option_id,
        "answer_type": row.get("answer_type"),
    }


def iter_questions(dataset_root: Path):
    for path in sorted(dataset_root.rglob("*.json")):
        try:
            row = load_hf_row(path)
        except Exception:
            continue
        yield path, row


def output_image_path(output_root: Path, row: dict[str, Any], source_json: Path) -> Path:
    return (
        output_root
        / safe_component(row.get("big_task"))
        / safe_component(row.get("small_task"))
        / safe_component(row.get("scene"))
        / safe_component(row.get("room"))
        / f"{safe_component(row.get('id'), 'no_id')}_{safe_component(source_json.stem)}.png"
    )


def inspect_preview(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        extrema = rgb.getextrema()
        ranges = [int(high) - int(low) for low, high in extrema]
        stddev = [float(value) for value in stat.stddev]
        mean = [float(value) for value in stat.mean]
    reasons = []
    if rgb.width < 64 or rgb.height < 64:
        reasons.append("image_too_small")
    if max(ranges) == 0:
        reasons.append("constant_image")
    elif max(stddev) < 15.0:
        reasons.append("very_low_information")
    return {
        "decoded": True,
        "width": int(rgb.width),
        "height": int(rgb.height),
        "size_bytes": path.stat().st_size,
        "channel_mean": [round(value, 4) for value in mean],
        "channel_stddev": [round(value, 4) for value in stddev],
        "channel_extrema": [[int(low), int(high)] for low, high in extrema],
        "quality_reasons": reasons,
        "usable_preview": not reasons,
    }


def _existing_first_frame(result_path: Path, result: dict[str, Any]) -> Path | None:
    history = result.get("history") or []
    first_image = None
    if history and isinstance(history[0], dict):
        first_image = history[0].get("image")
    if not first_image:
        first_image = "step_001.png"
    step_dir = result.get("step_image_dir")
    if not step_dir:
        return None
    candidate = Path(step_dir) / str(first_image)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve() if candidate.is_file() else None


def _workspace_path(value: Any, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _action_sweep_candidates(action_sweep_root: Path):
    """Yield exact initial frames saved by the task-action validation sweep."""
    if not action_sweep_root.is_dir():
        return
    for summary_path in sorted(action_sweep_root.rglob("summary.json")):
        try:
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            source_json = _workspace_path(summary.get("source_json"), summary_path.parent)
            initial_frame = summary_path.parent / "frame_000_initial.png"
            if source_json is None or not source_json.is_file() or not initial_frame.is_file():
                continue
            yield summary_path, source_json, initial_frame.resolve()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue


def _contact_sheet(entries: list[dict[str, Any]], destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns = 4
    thumb_width, thumb_height = 448, 252
    label_height = 72
    margin = 16
    cell_width = thumb_width + margin * 2
    cell_height = thumb_height + label_height + margin * 2
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        x = column * cell_width + margin
        y = row * cell_height + margin
        with Image.open(REPO_ROOT / entry["output_image"]) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
            tile = Image.new("RGB", (thumb_width, thumb_height), (30, 30, 30))
            tile.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_width - 1, y + thumb_height - 1), outline=(80, 80, 80), width=1)
        if not entry["quality"]["usable_preview"]:
            draw.rectangle((x, y, x + 112, y + 28), fill=(170, 25, 25))
            draw.text((x + 8, y + 4), "WARNING", fill=(255, 255, 255), font=small_font)
        draw.text((x, y + thumb_height + 8), f"{index + 1:02d}. {entry['small_task']}", fill=(15, 15, 15), font=font)
        draw.text(
            (x, y + thumb_height + 34),
            f"{entry['scene']} / {entry['room']}",
            fill=(70, 70, 70),
            font=small_font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def collect_existing_preview(args: argparse.Namespace) -> int:
    results_root = args.results_root.resolve()
    action_sweep_root = args.action_sweep_root.resolve()
    output_root = args.output_root.resolve()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []

    for result_path in sorted(results_root.rglob("*_answer.json")):
        try:
            with result_path.open("r", encoding="utf-8") as stream:
                result = json.load(stream)
            source_json = Path(result["source_json"])
            if not source_json.is_absolute():
                source_json = REPO_ROOT / source_json
            row = load_hf_row(source_json)
            first_frame = _existing_first_frame(result_path, result)
            if first_frame is None:
                raise FileNotFoundError("missing first-frame image")
            quality = inspect_preview(first_frame)
            key = (str(row.get("big_task")), str(row.get("small_task")))
            grouped[key].append(
                {
                    "result_path": result_path,
                    "result": result,
                    "source_json": source_json,
                    "row": row,
                    "first_frame": first_frame,
                    "quality": quality,
                }
            )
        except Exception as exc:
            errors.append({"result": str(result_path), "error": f"{type(exc).__name__}: {exc}"})

    for summary_path, source_json, first_frame in _action_sweep_candidates(action_sweep_root):
        try:
            row = load_hf_row(source_json)
            quality = inspect_preview(first_frame)
            key = (str(row.get("big_task")), str(row.get("small_task")))
            grouped[key].append(
                {
                    "result_path": summary_path,
                    "result": {},
                    "source_json": source_json,
                    "row": row,
                    "first_frame": first_frame,
                    "quality": quality,
                    "source_kind": "existing_task_action_sweep_initial_frame",
                    "fidelity": "exact_same_question_initial_camera_frame",
                }
            )
        except Exception as exc:
            errors.append({"result": str(summary_path), "error": f"{type(exc).__name__}: {exc}"})

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        candidates.sort(
            key=lambda item: (
                bool(item["quality"]["usable_preview"]),
                min(item["quality"]["channel_stddev"]),
                item["quality"]["size_bytes"],
                str(item["source_json"]),
            ),
            reverse=True,
        )
        item = candidates[0]
        row = item["row"]
        destination = output_image_path(output_root, row, item["source_json"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not destination.is_file():
            shutil.copy2(item["first_frame"], destination)
        quality = inspect_preview(destination)
        entry = {
            "status": "collected" if quality["usable_preview"] else "collected_warning",
            "big_task": key[0],
            "small_task": key[1],
            "id": row.get("id"),
            "runner_task": row.get("runner_task"),
            "scene": row.get("scene"),
            "room": row.get("room"),
            "source_json": str(item["source_json"].relative_to(REPO_ROOT)),
            "source_result": str(item["result_path"].relative_to(REPO_ROOT)),
            "source_first_frame": str(item["first_frame"].relative_to(REPO_ROOT)),
            "output_image": str(destination.relative_to(REPO_ROOT)),
            "source_kind": item.get("source_kind", "existing_active_runner_step_001"),
            "fidelity": item.get("fidelity", "exact_active_runner_first_frame"),
            "quality": quality,
            "available_candidates": len(candidates),
        }
        metadata_path = destination.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        selected.append(entry)

    manifest = {
        "summary": {
            "expected_tasks": 29,
            "selected_tasks": len(selected),
            "usable": sum(item["quality"]["usable_preview"] for item in selected),
            "warnings": sum(not item["quality"]["usable_preview"] for item in selected),
            "result_files_scanned": sum(len(items) for items in grouped.values()),
            "scan_errors": len(errors),
        },
        "selection_rule": "Best exact initial frame per big_task/small_task from active-runner or action-sweep outputs, ranked by usability and image information.",
        "tasks": selected,
        "errors": errors,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "preview_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    contact_sheet = output_root / "contact_sheet.png"
    _contact_sheet(selected, contact_sheet)
    print(
        json.dumps(
            {
                **manifest["summary"],
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "contact_sheet": str(contact_sheet.relative_to(REPO_ROOT)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if len(selected) == 29 and not manifest["summary"]["warnings"] else 1


def _compose_unobserved_phases(phase_root: Path, sample_dir: Path, destination: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    manifest_path = sample_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    views_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    for view in manifest.get("views") or []:
        phase = str(view.get("phase") or view.get("state_key") or "")
        if phase not in {"image1", "image2"} or not view.get("output_path"):
            continue
        details = view.get("render_details") or {}
        box_index = int(details.get("box_index", 0) or 0)
        image_path = phase_root / str(view["output_path"])
        if image_path.is_file():
            views_by_slot[(phase, box_index)] = {**view, "resolved_path": image_path}

    audit = manifest.get("unobserved_replay_audit") or {}
    box_count = int(audit.get("box_count") or 1)
    canvas_width, canvas_height = 1280, 720
    header_height = 48
    cell_width = canvas_width // 2
    cell_height = max(1, (canvas_height - header_height) // box_count)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (36, 36, 36))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (phase, title) in enumerate((("image1", "Phase 1"), ("image2", "Phase 2"))):
        draw.rectangle(
            (column * cell_width, 0, (column + 1) * cell_width, header_height),
            fill=(20, 20, 20),
        )
        draw.text((column * cell_width + 16, 17), title, fill=(255, 255, 255), font=font)
        for box_index in range(box_count):
            x0 = column * cell_width
            y0 = header_height + box_index * cell_height
            x1 = x0 + cell_width
            y1 = canvas_height if box_index == box_count - 1 else y0 + cell_height
            slot = views_by_slot.get((phase, box_index))
            if slot is None:
                draw.rectangle((x0, y0, x1, y1), fill=(75, 75, 75))
                draw.text(
                    (x0 + 16, y0 + 16),
                    f"Box {box_index}: missing original GT capture",
                    fill=(255, 220, 150),
                    font=font,
                )
                continue
            with Image.open(slot["resolved_path"]) as source:
                fitted = ImageOps.fit(
                    source.convert("RGB"),
                    (cell_width, y1 - y0),
                    method=Image.Resampling.LANCZOS,
                )
            canvas.paste(fitted, (x0, y0))
            if box_count > 1:
                draw.rectangle((x0 + 8, y0 + 8, x0 + 70, y0 + 28), fill=(0, 0, 0))
                draw.text((x0 + 14, y0 + 12), f"Box {box_index}", fill=(255, 255, 255), font=font)
    draw.line((cell_width, 0, cell_width, canvas_height), fill=(255, 255, 255), width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {
        "phase_count": 2,
        "box_count": box_count,
        "phase_views": {
            f"{phase}/box_{box_index}": str(value["resolved_path"].relative_to(REPO_ROOT))
            for (phase, box_index), value in sorted(views_by_slot.items())
        },
        "phase_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "replay_status": manifest.get("status"),
        "replay_issues": manifest.get("issues") or [],
        "loaded_structure_counts": manifest.get("loaded_structure_counts") or {},
    }


def _render_unobserved_single(
    source_json: Path,
    output_root: Path,
    row: dict[str, Any],
    payload: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    from passive_gt.pipeline import sample_directory

    destination = output_image_path(output_root, row, source_json)
    metadata_path = destination.with_suffix(".json")
    if destination.is_file() and metadata_path.is_file() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    dataset_root = DEFAULT_DATASET_ROOT.resolve()
    phase_root = (output_root / "_unobserved_phase_replay").resolve()
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "render_passive_gt.py"),
        "_render-one",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(phase_root),
        "--source-json",
        str(source_json),
        "--unobserved-container-state",
        "closed",
    ]
    if overwrite:
        command.append("--overwrite")
    worker_env = os.environ.copy()
    worker_env.setdefault("OMNIGIBSON_HEADLESS", "True")
    worker_env.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
    worker_env.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=worker_env,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    sample_dir = sample_directory(row, source_json, dataset_root, phase_root)
    manifest_path = sample_dir / "manifest.json"
    if completed.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(
            "Unobserved two-phase replay failed: "
            f"returncode={completed.returncode}, stdout={completed.stdout[-2000:]}, "
            f"stderr={completed.stderr[-4000:]}"
        )
    phase_info = _compose_unobserved_phases(phase_root, sample_dir, destination)
    quality = inspect_preview(destination)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    phase_camera_poses = [
        {
            "phase": view.get("phase"),
            "box_index": (view.get("render_details") or {}).get("box_index"),
            "camera_pose": view.get("camera_pose"),
        }
        for view in (manifest.get("views") or [])
        if view.get("camera_pose")
    ]
    result = {
        "status": "rendered" if quality["usable_preview"] else "rendered_warning",
        "source_json": str(source_json.relative_to(REPO_ROOT)),
        "id": row.get("id"),
        "big_task": row.get("big_task"),
        "small_task": row.get("small_task"),
        "runner_task": row.get("runner_task"),
        "scene": row.get("scene"),
        "room": row.get("room"),
        "task_module": "tasks.temporal_understanding.unobserved_change",
        "output_image": str(destination.relative_to(REPO_ROOT)),
        "camera_pose": phase_camera_poses[-1]["camera_pose"] if phase_camera_poses else None,
        "camera_info": {"phase_camera_poses": phase_camera_poses},
        "env_postprocess": {
            **phase_info,
            "two_phase_state_reconstruction": True,
            "walls_requested": False,
            "scene_load": "full_scene_without_walls",
            "container_view_state": "closed",
        },
        "quality": quality,
        **question_answer_fields(row, payload),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def render_one(source_json: Path, output_root: Path, robot: str, overwrite: bool) -> dict[str, Any]:
    row = load_hf_row(source_json)
    if row.get("small_task") == "Unobserved Change":
        src_root = REPO_ROOT / "src"
        sys.path.insert(0, str(src_root))
        metadata = row.get("metadata_json")
        payload = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
        payload["scene"] = row.get("scene") or payload.get("scene")
        payload["room"] = row.get("room") or payload.get("room")
        return _render_unobserved_single(source_json, output_root, row, payload, overwrite)

    active_root = REPO_ROOT / "src" / "active_explore"
    src_root = REPO_ROOT / "src"
    sys.path.insert(0, str(active_root))
    sys.path.insert(0, str(src_root))

    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")

    import numpy as np
    import omnigibson as og

    import active_explore.pipeline as pipeline

    payload = pipeline.load_question_json(source_json)
    task_module = pipeline.load_task_module_for_payload(str(row.get("small_task") or ""), payload)
    destination = output_image_path(output_root, row, source_json)
    metadata_path = destination.with_suffix(".json")
    if destination.is_file() and metadata_path.is_file() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    config = SimpleNamespace(json_root=None, robot=robot)
    task_state = pipeline.task_preprocess(task_module, payload, source_json, config)
    if task_state.get("skip_reason"):
        raise RuntimeError(f"Task preprocess skipped sample: {task_state['skip_reason']}")

    scene_name, room_name = task_module.scene_room(payload)
    pos, quat, camera_info = task_module.initial_camera(payload)
    objects = task_module.build_env_objects(payload)
    env = None
    env_postprocess: dict[str, Any] = {}
    try:
        full_scene = bool(getattr(task_module, "FULL_SCENE", False))
        env_config = pipeline.task_build_env_config(
            task_module,
            scene_name,
            room_name,
            robot,
            objects,
            full_scene=full_scene,
            payload=payload,
        )
        env = og.Environment(configs=env_config)
        # RGB is required by every passive-single render. Some tasks disable
        # optional viewer-camera modalities because segmentation annotators are
        # incompatible with their simulation setup; that must not disable RGB.
        try:
            og.sim._viewer_camera.add_modality("rgb")
        except Exception:
            pass
        for _ in range(pipeline.task_initial_settle_steps(task_module)):
            og.sim.step()
        env_postprocess = pipeline.task_postprocess_env(task_module, env, payload, camera_info, task_state)
        camera_override = env_postprocess.get("camera_override") if isinstance(env_postprocess, dict) else None
        if isinstance(camera_override, dict) and camera_override.get("position") and camera_override.get("quaternion_xyzw"):
            pos = np.asarray(camera_override["position"], dtype=float)
            quat = np.asarray(camera_override["quaternion_xyzw"], dtype=float)
            camera_info = {**camera_info, "postprocess_camera_override": camera_override}

        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = pipeline.task_capture_image(
            task_module,
            env,
            payload,
            camera_info,
            np.asarray(pos, dtype=float),
            np.asarray(quat, dtype=float),
            destination,
            task_state,
        )
        rendered_path = Path(rendered_path)
        if rendered_path.resolve() != destination.resolve():
            destination.write_bytes(rendered_path.read_bytes())
        quality = inspect_preview(destination)
        result = {
            "status": "rendered" if quality["usable_preview"] else "rendered_warning",
            "source_json": str(source_json.relative_to(REPO_ROOT)),
            "id": row.get("id"),
            "big_task": row.get("big_task"),
            "small_task": row.get("small_task"),
            "runner_task": row.get("runner_task"),
            "scene": row.get("scene"),
            "room": row.get("room"),
            "task_module": task_module.__name__,
            "output_image": str(destination.relative_to(REPO_ROOT)),
            "camera_pose": {
                "position": [float(value) for value in pos],
                "quaternion_xyzw": [float(value) for value in quat],
            },
            "camera_info": camera_info,
            "env_postprocess": env_postprocess,
            "quality": quality,
            **question_answer_fields(row, payload),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        handled_cleanup = False
        try:
            handled_cleanup = pipeline.task_cleanup_runtime(task_module, env, payload, task_state)
        except Exception:
            handled_cleanup = False
        try:
            if not active_exception and not handled_cleanup and getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass


def preview_candidates(dataset_root: Path) -> dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, row in iter_questions(dataset_root):
        key = (str(row.get("big_task")), str(row.get("small_task")))
        grouped[key].append((path, row))
    for key, items in grouped.items():
        # Rigid active-single historically skips variant rows. Prefer the base
        # row for the preview while keeping variants available as later retries.
        if key[1] == "Rigid Containment":
            items.sort(key=lambda item: ("__" in item[0].stem, str(item[0])))
        else:
            items.sort(key=lambda item: str(item[0]))
    return dict(grouped)


def parse_last_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _worker_env(headless: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "True" if headless else "False")
    env.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
    env.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("PXR_WORK_THREAD_LIMIT", "2")
    return env


def _run_worker(
    source_json: Path,
    output_root: Path,
    conda_env: str,
    robot: str,
    headless: bool,
    overwrite: bool,
    worker_timeout: int,
) -> dict[str, Any]:
    removed_shm = cleanup_dead_rstring_shm()
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", str(Path(__file__).resolve()), "one",
        "--source-json", str(source_json),
        "--output-root", str(output_root),
        "--robot", robot,
    ]
    if overwrite:
        cmd.append("--overwrite")
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=_worker_env(headless),
            text=True,
            capture_output=True,
            timeout=worker_timeout,
        )
        parsed = parse_last_json(completed.stdout)
        recovered_from_metadata = False
        if completed.returncode == 0 and parsed is None:
            try:
                row = load_hf_row(source_json)
                metadata_path = output_image_path(output_root, row, source_json).with_suffix(".json")
                image_path = output_image_path(output_root, row, source_json)
                if metadata_path.is_file() and image_path.is_file():
                    with metadata_path.open("r", encoding="utf-8") as stream:
                        parsed = json.load(stream)
                    recovered_from_metadata = True
            except (OSError, ValueError, json.JSONDecodeError):
                parsed = None
        return {
            "returncode": completed.returncode,
            "removed_stale_shm": removed_shm,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
            "recovered_from_metadata": recovered_from_metadata,
            "result": parsed if completed.returncode == 0 and parsed else None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "error": "worker_timeout",
            "removed_stale_shm": removed_shm,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "result": None,
        }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def backfill_question_answers(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    updated = 0
    unchanged = 0
    errors: list[dict[str, str]] = []
    required = {
        "question",
        "answer_options",
        "answer",
        "right_answer",
        "right_answer_option_id",
        "answer_type",
    }
    for metadata_path in sorted(output_root.rglob("*.json")):
        if metadata_path.name in {"all_summary.json", "smoke_summary.json", "preview_manifest.json"}:
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict) or not metadata.get("source_json") or not metadata.get("output_image"):
                unchanged += 1
                continue
            source_json = Path(str(metadata["source_json"]))
            if not source_json.is_absolute():
                source_json = REPO_ROOT / source_json
            row = load_hf_row(source_json)
            payload = decode_json_column(row.get("metadata_json"))
            payload = payload if isinstance(payload, dict) else {}
            qa_fields = question_answer_fields(row, payload)
            if required.issubset(metadata) and all(metadata.get(key) == value for key, value in qa_fields.items()):
                unchanged += 1
                continue
            metadata.update(qa_fields)
            _write_json_atomic(metadata_path, metadata)
            updated += 1
        except Exception as exc:
            errors.append({"metadata": str(metadata_path), "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "output_root": str(output_root),
        "updated": updated,
        "unchanged": unchanged,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 1 if errors else 0


def run_all(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    questions = list(iter_questions(dataset_root))
    start = max(0, int(args.start_index))
    stop = len(questions) if args.limit is None else min(len(questions), start + max(0, int(args.limit)))
    selected = questions[start:stop]
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = safe_component(args.run_name, "all")
    manifest_path = output_root / f"{run_name}_manifest.jsonl"
    summary_path = output_root / f"{run_name}_summary.json"
    session = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "dataset_total": len(questions),
        "start_index": start,
        "selected_total": len(selected),
        "processed": 0,
        "rendered": 0,
        "warnings": 0,
        "failed": 0,
        "skipped_existing": 0,
        "current_source_json": None,
        "complete": False,
    }
    _write_json_atomic(summary_path, session)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for offset, (source_json, row) in enumerate(selected):
            absolute_index = start + offset
            destination = output_image_path(output_root, row, source_json)
            metadata_path = destination.with_suffix(".json")
            source_label = str(source_json.relative_to(REPO_ROOT))
            session["current_source_json"] = source_label
            _write_json_atomic(summary_path, session)
            print(
                json.dumps(
                    {
                        "event": "sample_start",
                        "index": absolute_index + 1,
                        "dataset_total": len(questions),
                        "session_index": offset + 1,
                        "session_total": len(selected),
                        "source_json": source_label,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            result = None
            attempts = []
            skipped_existing = False
            if destination.is_file() and metadata_path.is_file() and not args.overwrite:
                try:
                    with metadata_path.open("r", encoding="utf-8") as stream:
                        result = json.load(stream)
                    skipped_existing = True
                except (OSError, json.JSONDecodeError):
                    result = None

            if result is None:
                for retry in range(1, max(1, int(args.max_retries)) + 1):
                    attempt = _run_worker(
                        source_json=source_json,
                        output_root=output_root,
                        conda_env=args.conda_env,
                        robot=args.robot,
                        headless=args.headless,
                        overwrite=args.overwrite,
                        worker_timeout=args.worker_timeout,
                    )
                    attempt["retry"] = retry
                    attempts.append(attempt)
                    if attempt.get("result") is not None:
                        result = attempt["result"]
                        break

            status = str((result or {}).get("status") or "failed")
            if skipped_existing:
                session["skipped_existing"] += 1
            if status == "rendered":
                session["rendered"] += 1
            elif status == "rendered_warning":
                session["warnings"] += 1
            else:
                session["failed"] += 1
            session["processed"] += 1

            record = {
                "index": absolute_index + 1,
                "source_json": source_label,
                "big_task": row.get("big_task"),
                "small_task": row.get("small_task"),
                "scene": row.get("scene"),
                "room": row.get("room"),
                "status": status,
                "skipped_existing": skipped_existing,
                "result": result,
                "attempts": attempts,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            _write_json_atomic(summary_path, session)
            print(json.dumps({"event": "sample_done", **record}, ensure_ascii=False), flush=True)

    session["current_source_json"] = None
    session["complete"] = True
    _write_json_atomic(summary_path, session)
    print(json.dumps({"event": "all_done", **session}, ensure_ascii=False), flush=True)
    return 1 if session["failed"] else 0


def run_preview(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    grouped = preview_candidates(dataset_root)
    manifest_path = output_root / "preview_manifest.json"
    results: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for task_index, key in enumerate(sorted(grouped), start=1):
        candidates = grouped[key][: max(1, args.candidates_per_task)]
        attempts = []
        selected_result = None
        print(json.dumps({"event": "task_start", "index": task_index, "total": len(grouped), "task": "/".join(key)}, ensure_ascii=False), flush=True)
        for candidate_index, (source_json, row) in enumerate(candidates, start=1):
            for retry in range(1, max(1, args.max_retries) + 1):
                removed_shm = cleanup_dead_rstring_shm()
                cmd = [
                    "conda", "run", "--no-capture-output", "-n", args.conda_env,
                    "python", str(Path(__file__).resolve()), "one",
                    "--source-json", str(source_json),
                    "--output-root", str(output_root),
                    "--robot", args.robot,
                ]
                if args.overwrite:
                    cmd.append("--overwrite")
                env = os.environ.copy()
                env.setdefault("OMNIGIBSON_HEADLESS", "True" if args.headless else "False")
                env.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
                env.setdefault("OG_DISABLE_EMITTER_APIS", "1")
                env.setdefault("PYTHONFAULTHANDLER", "1")
                env.setdefault("MALLOC_ARENA_MAX", "2")
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("OPENBLAS_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                env.setdefault("NUMEXPR_NUM_THREADS", "1")
                env.setdefault("PXR_WORK_THREAD_LIMIT", "2")
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=args.worker_timeout,
                    )
                    attempt = {
                        "candidate": candidate_index,
                        "retry": retry,
                        "source_json": str(source_json.relative_to(REPO_ROOT)),
                        "returncode": completed.returncode,
                        "removed_stale_shm": removed_shm,
                        "stdout_tail": completed.stdout[-2000:],
                        "stderr_tail": completed.stderr[-4000:],
                    }
                    parsed = parse_last_json(completed.stdout)
                    if completed.returncode == 0 and parsed:
                        attempt["result"] = parsed
                        if parsed.get("status") == "rendered":
                            selected_result = parsed
                except subprocess.TimeoutExpired as exc:
                    attempt = {
                        "candidate": candidate_index,
                        "retry": retry,
                        "source_json": str(source_json.relative_to(REPO_ROOT)),
                        "returncode": -1,
                        "error": "worker_timeout",
                        "removed_stale_shm": removed_shm,
                        "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                        "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    }
                attempts.append(attempt)
                print(json.dumps({"event": "attempt_done", "task": "/".join(key), **attempt}, ensure_ascii=False), flush=True)
                if selected_result is not None or attempt.get("result"):
                    break
            if selected_result is not None:
                break

        if selected_result is None:
            warning_result = next((item.get("result") for item in attempts if item.get("result")), None)
            selected_result = warning_result
        task_result = {
            "big_task": key[0],
            "small_task": key[1],
            "status": selected_result.get("status") if selected_result else "failed",
            "selected": selected_result,
            "attempts": attempts,
        }
        results.append(task_result)
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump({"tasks": results}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"event": "task_done", "task": "/".join(key), "status": task_result["status"]}, ensure_ascii=False), flush=True)

    summary = {
        "tasks": len(results),
        "rendered": sum(item["status"] == "rendered" for item in results),
        "warnings": sum(item["status"] == "rendered_warning" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if summary["failed"] else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the initial passive-single observation without calling a model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("one")
    one.add_argument("--source-json", type=Path, required=True)
    one.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    one.add_argument("--robot", default="R1")
    one.add_argument("--overwrite", action="store_true")

    preview = subparsers.add_parser("preview")
    preview.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    preview.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    preview.add_argument("--conda-env", default="behavior")
    preview.add_argument("--robot", default="R1")
    preview.add_argument("--candidates-per-task", type=int, default=3)
    preview.add_argument("--max-retries", type=int, default=3)
    preview.add_argument("--worker-timeout", type=int, default=900)
    preview.add_argument("--headless", action="store_true")
    preview.add_argument("--overwrite", action="store_true")

    render_all = subparsers.add_parser("all", help="Render every json_clean passive-single sample with resume support.")
    render_all.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    render_all.add_argument("--output-root", type=Path, default=DEFAULT_ALL_OUTPUT_ROOT)
    render_all.add_argument("--conda-env", default="behavior")
    render_all.add_argument("--robot", default="R1")
    render_all.add_argument("--max-retries", type=int, default=2)
    render_all.add_argument("--worker-timeout", type=int, default=900)
    render_all.add_argument("--start-index", type=int, default=0)
    render_all.add_argument("--limit", type=int)
    render_all.add_argument("--run-name", default="all")
    render_all.add_argument("--headless", action="store_true")
    render_all.add_argument("--overwrite", action="store_true")

    backfill = subparsers.add_parser("backfill-qa", help="Add standardized question and answer fields to rendered sidecars.")
    backfill.add_argument("--output-root", type=Path, default=DEFAULT_ALL_OUTPUT_ROOT)

    existing = subparsers.add_parser("existing-preview", help="Collect existing active-runner step_001 frames for all small tasks.")
    existing.add_argument("--results-root", type=Path, default=DEFAULT_EXISTING_RESULTS_ROOT)
    existing.add_argument("--action-sweep-root", type=Path, default=DEFAULT_ACTION_SWEEP_ROOT)
    existing.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preview":
        return run_preview(args)
    if args.command == "all":
        return run_all(args)
    if args.command == "backfill-qa":
        return backfill_question_answers(args)
    if args.command == "existing-preview":
        return collect_existing_preview(args)
    try:
        result = render_one(args.source_json.resolve(), args.output_root.resolve(), args.robot, args.overwrite)
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
