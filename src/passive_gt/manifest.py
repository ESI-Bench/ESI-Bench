from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
AVAILABLE_VIEW_STATUSES = {
    "already_present",
    "already_present_warning",
    "copied_existing",
    "copied_existing_warning",
    "rendered_pose",
    "rendered_pose_warning",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def inspect_image(path: Path) -> dict[str, Any]:
    """Decode an image and reject blank renderer outputs before reusing them."""
    result: dict[str, Any] = {
        "decoded": False,
        "valid": False,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            stat = ImageStat.Stat(rgb)
            ranges = [int(high) - int(low) for low, high in extrema]
            result.update(
                {
                    "decoded": True,
                    "width": int(rgb.width),
                    "height": int(rgb.height),
                    "channel_extrema": [[int(low), int(high)] for low, high in extrema],
                    "channel_stddev": [round(float(value), 4) for value in stat.stddev],
                }
            )
            reasons: list[str] = []
            if rgb.width < 64 or rgb.height < 64:
                reasons.append("image_too_small")
            if max(ranges) == 0:
                reasons.append("constant_image")
            elif max(ranges) <= 6 and result["size_bytes"] < 10_000:
                reasons.append("near_constant_image")
            result["quality_reasons"] = reasons
            result["valid"] = not reasons
            return result
    except Exception as exc:
        result["quality_reasons"] = ["decode_failed"]
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
        return result


def image_is_materializable(quality: dict[str, Any]) -> bool:
    """Return whether an image can be kept, even if it has a quality warning.

    Saved GT poses occasionally point into a wall or another object.  Those
    captures are low-information but still authoritative dataset artifacts;
    only undecodable or thumbnail-sized files should be rejected outright.
    """
    return bool(
        quality.get("valid")
        or (
            quality.get("decoded")
            and int(quality.get("width") or 0) >= 64
            and int(quality.get("height") or 0) >= 64
        )
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    expected_views = 0
    baseline_selected_views = 0
    missing_camera_pose_views = 0
    for record in records:
        task = f"{record['big_task']}/{record['small_task']}"
        task_counts[task]["samples"] += 1
        task_counts[task][f"sample_status:{record.get('status', 'unknown')}"] += 1
        views = record.get("views") or []
        expected_views += len(views)
        task_counts[task]["views_expected"] += len(views)
        for view in views:
            status = str(view.get("status") or "planned")
            mode = str(view.get("render_mode") or "unknown")
            status_counts[status] += 1
            mode_counts[mode] += 1
            task_counts[task][f"view_status:{status}"] += 1
            task_counts[task][f"render_mode:{mode}"] += 1
            provenance = str(view.get("provenance") or "")
            if provenance:
                provenance_counts[provenance] += 1
                task_counts[task][f"provenance:{provenance}"] += 1
            if view.get("baseline_selected", True):
                baseline_selected_views += 1
                task_counts[task]["baseline_selected_views"] += 1
            if view.get("camera_pose") is None:
                missing_camera_pose_views += 1
                task_counts[task]["missing_camera_pose_views"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "samples": len(records),
        "views_expected": expected_views,
        "baseline_selected_views": baseline_selected_views,
        "missing_camera_pose_views": missing_camera_pose_views,
        "view_status": dict(sorted(status_counts.items())),
        "render_modes": dict(sorted(mode_counts.items())),
        "view_provenance": dict(sorted(provenance_counts.items())),
        "tasks": {key: dict(sorted(value.items())) for key, value in sorted(task_counts.items())},
    }
