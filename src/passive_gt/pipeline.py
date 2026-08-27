from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .manifest import (
    AVAILABLE_VIEW_STATUSES,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    image_is_materializable,
    inspect_image,
    sha256_file,
    summarize,
)
from .views import ViewSpec, extract_view_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "json_clean"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "passive_gt"
DEFAULT_AUXILIARY_ROOT = REPO_ROOT.parent / "BEHAVIOR-ESI" / "output_metas" / "output_metas"
DEFAULT_EXISTING_ROOTS = (
    REPO_ROOT.parent / "BEHAVIOR-1K" / "renders_occlusion_v2",
    REPO_ROOT.parent / "BEHAVIOR-1K" / "renders_angle_confusion_v2",
    REPO_ROOT.parent / "BEHAVIOR-1K" / "renders_line",
    REPO_ROOT.parent / "BEHAVIOR-1K" / "renders_line_positive",
    REPO_ROOT.parent / "ESI-Bench-ljg" / "example_for_each_task",
    REPO_ROOT.parent / "ESI-Bench-ljg" / "json-tmp" / "qualitative_example_render",
)

RENDERED_PROVENANCES = {
    "rendered_pose",  # schema-v1 compatibility
    "saved_state_replay",
    "derived_gt_camera_saved_state_replay",
    "approximate_snapshot_reconstruction",
    "full_physics_regeneration",
    "saved_gt_camera_deterministic_generator_replay",
}

COPIED_PROVENANCES = {
    "copied_existing",
    "copied_legacy_curated",
}


EXAMPLE_TASK_ALIASES: dict[str, set[str]] = {
    "Connectivity": {"Topological Connectivity"},
    "Long-Term Navigation": {"Long-Horizon Navigation"},
    "Regional Boundary": {"Regional Boundry"},
    "Traversable Passage": {"Traversable Passage"},
    "Category Ambiguity": {"semantic_fault"},
    "Counting w Occlusion": {"hidden_by_others"},
    "Illumination Variability": {"light_change"},
    "Merged Observation": {"observation_merged"},
    "Spatial Segmentation": {"observation_divided"},
    "Structural Enclosure": {"hidden_in_box"},
    "Deformable": {"cover_small_item_cloth"},
    "Correspondence": {"mirror_correspondence"},
    "Reflection Authoring": {"mirror_object_reality"},
    "Spatial Relations": {"mirror_distance"},
    "Unobserved Change": {"change_detection", "change_identification", "current_state_reasoning"},
}


QUALITATIVE_SAMPLE_DIRS: dict[tuple[str, str, str, str], str] = {
    ("Connectivity", "restaurant_asian", "full_scene", "q_004"): "cognitivemap/restaurant_asian/full_scene/q_004",
    ("Long-Term Navigation", "Beechwood_0_garden", "full_scene", "q_011"): "cognitivemap/q_011",
    ("Counting w Occlusion", "grocery_store_cafe", "dining_room_0", "q_001"): "counting/q_001",
    ("Deformable", "Beechwood_0_int", "kitchen_0", "q_001"): "deformable/q_001",
    ("Correspondence", "Beechwood_0_garden", "living_room_0", "q_006"): "mirror",
    ("Unobserved Change", "grocery_store_cafe", "dining_room_0", "q_000"): "unobserved_changes/q_000",
}


def safe_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(". ")
    return text or fallback


def load_question(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        row = json.load(stream)
    if not isinstance(row, dict) or "metadata_json" not in row:
        raise ValueError(f"Not an HF-style json_clean row: {path}")
    metadata = row.get("metadata_json")
    payload = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
    payload["scene"] = row.get("scene") or payload.get("scene") or "unknown_scene"
    payload["room"] = row.get("room") or payload.get("room") or "unknown_room"
    payload["_hf_id"] = row.get("id")
    payload["_hf_big_task"] = row.get("big_task")
    payload["_hf_small_task"] = row.get("small_task")
    payload["_hf_runner_task"] = row.get("runner_task")
    # json_clean intentionally stores benchmark metadata, but a few generators
    # kept replay-only state (notably Action Sequencing stage snapshots) only in
    # BEHAVIOR-ESI/output_metas.  Enrich in memory by exact runner + basename;
    # never mutate json_clean and never fall back to an ambiguous stem match.
    payload["_ground_truth"] = row.get("answer")
    payload["_question"] = row.get("question")
    payload["_options"] = row.get("options")
    runner = safe_component(row.get("runner_task"), "")
    auxiliary_path = DEFAULT_AUXILIARY_ROOT / runner / path.name if runner else None
    if auxiliary_path is not None and auxiliary_path.is_file():
        try:
            with auxiliary_path.open("r", encoding="utf-8") as stream:
                auxiliary = json.load(stream)
        except (OSError, json.JSONDecodeError):
            auxiliary = None
        if isinstance(auxiliary, dict):
            payload["_gt_auxiliary"] = auxiliary
            payload["_gt_auxiliary_path"] = str(auxiliary_path)
    return row, payload


def iter_question_files(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.rglob("*.json")):
        if len(path.relative_to(dataset_root).parts) < 3:
            continue
        try:
            with path.open("r", encoding="utf-8") as stream:
                head = json.load(stream)
        except Exception:
            continue
        if isinstance(head, dict) and "metadata_json" in head:
            yield path


def sample_directory(row: dict[str, Any], source: Path, dataset_root: Path, output_root: Path) -> Path:
    big = safe_component(row.get("big_task"))
    small = safe_component(row.get("small_task"))
    scene = safe_component(row.get("scene"))
    room = safe_component(row.get("room"))
    row_id = safe_component(row.get("id"), "no_id")
    stem = safe_component(source.stem)
    return output_root / big / small / scene / room / f"{row_id}_{stem}"


def sample_issues(payload: dict[str, Any], specs: list[ViewSpec]) -> list[dict[str, str]]:
    small = str(payload.get("_hf_small_task") or "")
    issues: list[dict[str, str]] = []
    if small == "Liquid Volume":
        issues.append({
            "code": "full_physics_regeneration_required",
            "message": "The four-frame GT pour trajectory needs live AABB cameras and particle-state replay; json_clean has no saved camera/particle state.",
        })
    elif small == "Rigid Containment":
        rigid_items = []
        if isinstance(payload.get("containee"), dict):
            rigid_items.append(payload["containee"])
        extras = payload.get("extras") or []
        rigid_items.extend(extras.values() if isinstance(extras, dict) else extras)
        if any(
            isinstance(item, dict)
            and item.get("placement")
            and not (
                item["placement"].get("success") is True
                and item["placement"].get("bbox_check") is True
            )
            for item in rigid_items
        ):
            issues.append({
                "code": "saved_failed_placement_outcome",
                "message": (
                    "The original fit attempt failed its inside-bbox check. "
                    "Passive-GT replays that saved failed after-state and does not invent a successful placement."
                ),
            })
    elif small == "Unobserved Change":
        gt = (((payload.get("question_data") or {}).get("render") or {}).get("gt_view") or {})
        for phase in ("image1", "image2"):
            entry = gt.get(phase) or {}
            if not entry.get("images"):
                issues.append({"code": f"missing_{phase}_gt", "message": f"The original {phase} GT capture failed or is empty."})
    elif small == "Structural Enclosure" and not specs:
        issues.append({"code": "no_target_closeup_required", "message": "Ground-truth count is zero, so this record legitimately has no target closeup."})
    elif small in {"Partial Occlusion", "Physical Contact"} and not specs:
        issues.append({
            "code": "no_approved_gt_view",
            "message": "The saved generator run has no view satisfying the official visibility/state predicate.",
        })
    if small == "Action Order Inference":
        if payload.get("_gt_auxiliary"):
            issues.append({
                "code": "external_stage_snapshot",
                "message": "Stage cameras and all-object outcome poses are joined by exact filename from BEHAVIOR-ESI/output_metas/action.",
            })
        else:
            issues.append({
                "code": "missing_external_stage_snapshot",
                "message": "Action Passive-GT requires the matching BEHAVIOR-ESI/output_metas/action record.",
            })
    if small == "Inclined Plane":
        issues.append({"code": "physics_replay_required", "message": "Per-step object orientations were not saved; the 30-step sequence requires a new physics replay."})
    if small == "Stacking & Stability":
        issues.append({
            "code": "snapshot_replay_approximation",
            "message": "Stable-trial poses are saved, but some objects drifted during visibility rendering; static snapshot replay is not pixel-exact.",
        })
    return issues


class ExistingImageResolver:
    """Resolve only exact task layouts or a unique, context-rich example match."""

    def __init__(self, roots: Iterable[Path]):
        self.roots = tuple(path.resolve() for path in roots if path.exists())
        self.by_basename: dict[str, list[Path]] = defaultdict(list)
        for root in self.roots:
            if root.name not in {"example_for_each_task", "qualitative_example_render"}:
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    self.by_basename[path.name].append(path.resolve())

    def _root_named(self, name: str) -> Path | None:
        return next((root for root in self.roots if root.name == name), None)

    def _known_layout(self, row: dict[str, Any], payload: dict[str, Any], spec: ViewSpec) -> Path | None:
        small = row.get("small_task")
        scene = str(row.get("scene"))
        room = str(row.get("room"))
        run_idx = payload.get("run_idx")
        image_name = Path(spec.original_image_path).name
        if run_idx is None:
            return None
        if small == "Partial Occlusion":
            root = self._root_named("renders_occlusion_v2")
            candidate = root / scene / f"{room}_{run_idx}" / image_name if root else None
            return candidate if candidate and candidate.is_file() else None
        if small == "View Hallucination":
            root = self._root_named("renders_angle_confusion_v2")
            candidate = root / scene / f"{room}_{run_idx}" / image_name if root else None
            return candidate if candidate and candidate.is_file() else None
        if small == "Linear Alignment":
            candidates = []
            for root_name in ("renders_line", "renders_line_positive"):
                root = self._root_named(root_name)
                candidate = root / scene / f"{room}_{run_idx}" / image_name if root else None
                if candidate and candidate.is_file():
                    candidates.append(candidate)
            return candidates[0] if len(candidates) == 1 else None
        return None

    def _qualitative_match(
        self, row: dict[str, Any], payload: dict[str, Any], spec: ViewSpec, source: Path
    ) -> Path | None:
        root = self._root_named("qualitative_example_render")
        if root is None:
            return None
        qid = str(payload.get("question_id") or "").replace("\\", "/").split("/")[-1]
        if not qid:
            match = re.search(r"q_\d+", source.stem)
            qid = match.group(0) if match else ""
        key = (
            str(row.get("small_task") or ""),
            str(row.get("scene") or ""),
            str(row.get("room") or ""),
            qid,
        )
        relative = QUALITATIVE_SAMPLE_DIRS.get(key)
        if relative is None:
            return None
        base = root / relative
        matches = [path for path in base.rglob(Path(spec.original_image_path).name) if path.is_file()]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _suffix_score(original: str, candidate: Path) -> int:
        original_parts = [part for part in original.replace("\\", "/").split("/") if part]
        candidate_parts = list(candidate.parts)
        score = 0
        for left, right in zip(reversed(original_parts), reversed(candidate_parts)):
            if left != right:
                break
            score += 4
        return score

    def _example_match(self, row: dict[str, Any], payload: dict[str, Any], spec: ViewSpec, source: Path) -> Path | None:
        candidates = self.by_basename.get(Path(spec.original_image_path).name, [])
        if not candidates:
            return None
        aliases = EXAMPLE_TASK_ALIASES.get(str(row.get("small_task")))
        task_type = str(payload.get("task_type") or "")
        if aliases and task_type in aliases:
            aliases = {task_type}
        scene = str(row.get("scene") or "")
        room = str(row.get("room") or "")
        qid = str(payload.get("question_id") or "").replace("\\", "/").split("/")[-1]
        stem = source.stem
        sample_tokens = {token for token in (qid, stem) if token}
        sample_tokens.update(re.findall(r"q_\d+", stem))
        scored: list[tuple[int, Path]] = []
        for candidate in candidates:
            parts = set(candidate.parts)
            if aliases and not (aliases & parts):
                continue
            if scene and scene not in parts:
                continue
            if room and room not in parts and not any(part.startswith(f"{room}_") for part in parts):
                continue
            if sample_tokens and not any(
                part == token or part.startswith(f"{token}_")
                for token in sample_tokens
                for part in parts
            ):
                continue
            score = self._suffix_score(spec.original_image_path, candidate)
            score += 8 if scene in parts else 0
            score += 6 if room in parts else 0
            score += 5 if qid and qid in parts else 0
            score += 4 if stem in parts else 0
            scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    def resolve(self, row: dict[str, Any], payload: dict[str, Any], spec: ViewSpec, source: Path) -> Path | None:
        raw = Path(spec.original_image_path).expanduser()
        if raw.is_absolute() and raw.is_file():
            return raw.resolve()
        known = self._known_layout(row, payload, spec)
        if known is not None:
            return known.resolve()
        qualitative = self._qualitative_match(row, payload, spec, source)
        if qualitative is not None:
            return qualitative.resolve()
        return self._example_match(row, payload, spec, source)


def _view_output_path(sample_dir: Path, spec: ViewSpec) -> Path:
    relative = Path(spec.view_id)
    safe_parts = [safe_component(part) for part in relative.parts]
    return sample_dir.joinpath(*safe_parts)


def _existing_output_metadata(path: Path, cached: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = cached or {}
    cached_quality = cached.get("quality") if isinstance(cached.get("quality"), dict) else None
    if (
        cached_quality
        and "decoded" in cached_quality
        and cached.get("sha256")
        and cached_quality.get("size_bytes") == path.stat().st_size
    ):
        quality = cached_quality
        digest = cached["sha256"]
    else:
        quality = inspect_image(path)
        digest = sha256_file(path)
    if quality.get("valid"):
        status = "already_present"
    elif image_is_materializable(quality):
        status = "already_present_warning"
    else:
        status = "invalid_output"
    return {
        "status": status,
        "output_path": str(path),
        "sha256": digest,
        "quality": quality,
        "width": quality.get("width"),
        "height": quality.get("height"),
    }


def build_records(
    dataset_root: Path,
    output_root: Path,
    *,
    big_task: str | None = None,
    small_task: str | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in iter_question_files(dataset_root):
        row, payload = load_question(source)
        if big_task and row.get("big_task") != big_task:
            continue
        if small_task and row.get("small_task") != small_task:
            continue
        specs = extract_view_specs(payload)
        sample_dir = sample_directory(row, source, dataset_root, output_root)
        previous_views: dict[str, dict[str, Any]] = {}
        previous_manifest = sample_dir / "manifest.json"
        if previous_manifest.is_file():
            try:
                with previous_manifest.open("r", encoding="utf-8") as stream:
                    previous = json.load(stream)
                previous_views = {
                    str(item.get("view_id")): item
                    for item in (previous.get("views") or [])
                    if isinstance(item, dict) and item.get("view_id")
                }
            except Exception:
                previous_views = {}
        views = []
        for spec in specs:
            view = spec.to_dict()
            output_path = _view_output_path(sample_dir, spec)
            view["output_path"] = str(output_path.relative_to(output_root))
            view["status"] = "planned"
            if output_path.is_file():
                old = previous_views.get(spec.view_id) or {}
                view.update(_existing_output_metadata(output_path, old))
                view["output_path"] = str(output_path.relative_to(output_root))
                if view["status"] in {"already_present", "already_present_warning"} and old.get("provenance"):
                    for key in ("provenance", "source_path", "render_details"):
                        if old.get(key) is not None:
                            view[key] = old[key]
                    if old.get("provenance") in RENDERED_PROVENANCES:
                        for key in ("camera_pose", "pose_source"):
                            if old.get(key) is not None:
                                view[key] = old[key]
                    warning = view["status"].endswith("_warning")
                    if old.get("provenance") in COPIED_PROVENANCES:
                        view["status"] = "copied_existing_warning" if warning else "copied_existing"
                    elif old.get("provenance") in RENDERED_PROVENANCES:
                        view["status"] = "rendered_pose_warning" if warning else "rendered_pose"
            views.append(view)
        issues = sample_issues(payload, specs)
        status = "complete" if views and all(view["status"] in AVAILABLE_VIEW_STATUSES for view in views) else "planned"
        if not views:
            status = "no_views" if any(issue["code"] == "no_target_closeup_required" for issue in issues) else "blocked"
        record = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": str(row.get("id") or ""),
            "big_task": row.get("big_task"),
            "small_task": row.get("small_task"),
            "runner_task": row.get("runner_task"),
            "scene": row.get("scene"),
            "room": row.get("room"),
            "source_json": str(source.relative_to(dataset_root)),
            "sample_directory": str(sample_dir.relative_to(output_root)),
            "status": status,
            "issues": issues,
            "views": views,
        }
        records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def materialize_records(
    records: list[dict[str, Any]],
    dataset_root: Path,
    output_root: Path,
    resolver: ExistingImageResolver,
    *,
    link_mode: str = "hardlink",
) -> list[dict[str, Any]]:
    for record in records:
        source = dataset_root / record["source_json"]
        row, payload = load_question(source)
        specs = {spec.view_id: spec for spec in extract_view_specs(payload)}
        for view in record.get("views") or []:
            destination = output_root / view["output_path"]
            spec = specs.get(view["view_id"])
            if spec is None:
                view["status"] = "invalid_manifest"
                view["error"] = "View selector no longer produced this view_id"
                continue
            if destination.is_file():
                if view.get("provenance") == "copied_legacy_curated":
                    view.update(_existing_output_metadata(destination, view))
                    if image_is_materializable(view.get("quality") or {}):
                        view["status"] = (
                            "copied_existing" if (view.get("quality") or {}).get("valid") else "copied_existing_warning"
                        )
                        view["provenance"] = "copied_legacy_curated"
                        view["output_path"] = str(destination.relative_to(output_root))
                        continue
                    destination.unlink()
                elif view.get("provenance") == "copied_existing":
                    resolved = resolver.resolve(row, payload, spec, source)
                    old_source = Path(str(view.get("source_path") or "")).resolve()
                    if resolved is None or resolved.resolve() != old_source:
                        destination.unlink()
                    else:
                        view.update(_existing_output_metadata(destination, view))
                        if not image_is_materializable(view.get("quality") or {}):
                            destination.unlink()
                        else:
                            view["status"] = (
                                "copied_existing" if (view.get("quality") or {}).get("valid") else "copied_existing_warning"
                            )
                            view["provenance"] = "copied_existing"
                            view["source_path"] = str(resolved)
                            view["output_path"] = str(destination.relative_to(output_root))
                            continue
                elif view.get("provenance") in RENDERED_PROVENANCES:
                    view.update(_existing_output_metadata(destination, view))
                    if image_is_materializable(view.get("quality") or {}):
                        view["status"] = (
                            "rendered_pose" if (view.get("quality") or {}).get("valid") else "rendered_pose_warning"
                        )
                        # Preserve the more precise replay provenance.
                        view["output_path"] = str(destination.relative_to(output_root))
                        continue
                else:
                    view.update(_existing_output_metadata(destination, view))
                    view["output_path"] = str(destination.relative_to(output_root))
                    if image_is_materializable(view.get("quality") or {}):
                        continue
            existing = resolver.resolve(row, payload, spec, source)
            if existing is None:
                view["status"] = "missing_source"
                continue
            quality = inspect_image(existing)
            if not image_is_materializable(quality):
                view.update(
                    {
                        "status": "invalid_existing",
                        "source_path": str(existing),
                        "quality": quality,
                    }
                )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            try:
                if link_mode == "hardlink":
                    os.link(existing, destination)
                elif link_mode == "symlink":
                    destination.symlink_to(existing)
                else:
                    shutil.copy2(existing, destination)
            except OSError:
                if link_mode == "hardlink":
                    shutil.copy2(existing, destination)
                else:
                    raise
            view.update(
                {
                    "status": "copied_existing" if quality.get("valid") else "copied_existing_warning",
                    "provenance": "copied_existing",
                    "source_path": str(existing),
                    "sha256": sha256_file(destination),
                    "quality": quality,
                    "width": quality.get("width"),
                    "height": quality.get("height"),
                }
            )
        statuses = [view.get("status") for view in record.get("views") or []]
        if statuses and all(status in AVAILABLE_VIEW_STATUSES for status in statuses):
            record["status"] = "complete"
        elif any(status in AVAILABLE_VIEW_STATUSES for status in statuses):
            record["status"] = "partial"
        elif statuses:
            record["status"] = "missing"
        sample_manifest = output_root / record["sample_directory"] / "manifest.json"
        atomic_write_json(sample_manifest, record)
    return records


def write_root_manifests(output_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_root / "manifest.jsonl", records)
    summary = summarize(records)
    atomic_write_json(output_root / "summary.json", summary)
    return summary
