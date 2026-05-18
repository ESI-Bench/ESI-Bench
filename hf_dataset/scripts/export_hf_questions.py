#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNNER_TASK_BY_SMALL = {
    "Action Order Inference": "action",
    "Connectivity": "cognitivemap",
    "Long-Term Navigation": "cognitivemap",
    "Regional Boundary": "cognitivemap",
    "Traversable Passage": "cognitivemap",
    "Category Ambiguity": "counting",
    "Counting w Occlusion": "counting",
    "Illumination Variability": "counting",
    "Merged Observation": "counting",
    "Spatial Segmentation": "counting",
    "Structural Enclosure": "counting",
    "Dimensional Size": "size",
    "Spatial Distance": "distance",
    "Material Transparency": "transparent",
    "Partial Occlusion": "occlusion",
    "View Hallucination": "angle_confusion",
    "Inclined Plane": "slope",
    "Stacking & Stability": "stacking",
    "Deformable": "deformable",
    "Liquid Volume": "pour",
    "Rigid Containment": "storage",
    "Geometric Configuration": "triangle",
    "Linear Alignment": "line",
    "Physical Contact": "touching",
    "Correspondence": "mirror",
    "Reflection Authoring": "mirror",
    "Spatial Relations": "mirror",
    "Agent Observation": "multiagent",
    "Unobserved Change": "unobserved_changes",
}

LIGHT_METADATA_KEYS = {
    "task",
    "task_type",
    "task_family",
    "task_category",
    "layout",
    "run_idx",
    "seed",
    "floor_name",
    "floor",
    "question_index",
    "question_id",
    "n_objects",
    "object_category",
    "obj_category",
    "container_cat",
    "small_obj_cat",
    "slope_angle_deg",
    "static_friction",
    "dynamic_friction",
    "ground_truth_slid",
    "ground_truth_fallen",
    "true_count",
    "proximity_thresh",
}


def text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_answer(raw: Any) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip()
    return raw


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def answer_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, str):
        return answer.strip() or None
    if isinstance(answer, (int, float, bool)):
        return str(answer)
    return json_text(answer)


def answer_type(answer: Any, options: list[Any]) -> str:
    if isinstance(answer, bool):
        return "boolean"
    if isinstance(answer, int):
        return "count"
    if isinstance(answer, list):
        return "sequence"
    if options:
        return "choice"
    if isinstance(answer, str) and answer.strip().isdigit():
        return "count"
    return "freeform"


def collect_image_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for key in ("image_path", "rgb", "view_file", "reference_image_path"):
                add(value.get(key))

    add(payload.get("image_paths"))
    add(payload.get("reference_image_paths"))
    add(payload.get("qa", {}).get("view_file") if isinstance(payload.get("qa"), dict) else None)

    qd = payload.get("question_data") if isinstance(payload.get("question_data"), dict) else {}
    render = qd.get("render") if isinstance(qd.get("render"), dict) else {}
    add(render.get("image_paths"))
    add(render.get("primary_image"))
    add(render.get("target_closeups"))
    add(qd.get("initial_view"))
    add(qd.get("overview_view"))
    add(qd.get("path_views"))

    render_top = payload.get("render") if isinstance(payload.get("render"), dict) else {}
    add(render_top.get("main_view"))
    add(render_top.get("room_view"))
    add(render_top.get("gt_view"))

    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def extract_question_answer(payload: dict[str, Any]) -> tuple[str | None, Any, list[Any]]:
    qd = payload.get("question_data") if isinstance(payload.get("question_data"), dict) else {}
    qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}

    question = (
        text(payload.get("_question"))
        or text(qa.get("question"))
        or text(qd.get("question"))
        or text((payload.get("_entry") or {}).get("question") if isinstance(payload.get("_entry"), dict) else None)
    )

    options: list[Any] = []
    for candidate in (qd.get("options"), qa.get("options"), qa.get("choices")):
        if isinstance(candidate, list):
            options = candidate
            break

    answer = None
    if qa.get("answer_A") is not None or qa.get("answer_C") is not None:
        answer = {
            key: qa[key]
            for key in ("answer_A", "answer_C")
            if qa.get(key) is not None
        }
    for candidate in (
        payload.get("_ground_truth"),
        qd.get("answer"),
        qa.get("answer"),
        qa.get("answer_text"),
        qa.get("answer_count"),
        qa.get("answer_label"),
        payload.get("answer"),
        (payload.get("_entry") or {}).get("ground_truth") if isinstance(payload.get("_entry"), dict) else None,
    ):
        if answer is not None:
            break
        if candidate is not None:
            answer = candidate
            break

    return question, normalize_answer(answer), options


def compact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: payload[key] for key in LIGHT_METADATA_KEYS if key in payload}
    qd = payload.get("question_data") if isinstance(payload.get("question_data"), dict) else None
    qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else None
    if qd is not None:
        metadata["question_data"] = {
            key: qd[key]
            for key in ("task_type", "case", "case_id", "case_type", "count_target", "count_unit")
            if key in qd
        }
    if qa is not None:
        metadata["qa"] = {
            key: qa[key]
            for key in ("answer_A", "answer_C", "answer_option_id", "answer_text", "answer_count", "answer_label")
            if key in qa
        }
    if payload.get("_missing") is not None:
        metadata["_missing"] = payload.get("_missing")
    if isinstance(payload.get("_entry"), dict):
        metadata["_entry"] = payload["_entry"]
    return metadata


def record_for_file(path: Path, json_root: Path, row_id: str) -> dict[str, Any]:
    rel = path.relative_to(json_root)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected question path: {rel}")

    big_task, small_task, scene_from_path, room_from_path = parts[:4]
    payload = json.loads(path.read_text(encoding="utf-8"))
    question, answer, options = extract_question_answer(payload)
    scene = text(payload.get("scene")) or scene_from_path
    room = text(payload.get("room")) or room_from_path

    return {
        "id": row_id,
        "big_task": big_task,
        "small_task": small_task,
        "runner_task": RUNNER_TASK_BY_SMALL.get(small_task),
        "scene": scene,
        "room": room,
        "question": question,
        "answer": answer_text(answer),
        "answer_type": answer_type(answer, options),
        "options_json": json_text(options),
        "image_paths_json": json_text(collect_image_paths(payload)),
        "metadata_json": json_text(compact_metadata(payload)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a fixed-column HF/Croissant-friendly questions table.")
    parser.add_argument("--json-root", type=Path, default=Path("dataset/json"))
    parser.add_argument("--output", type=Path, default=Path("data/questions.jsonl"))
    args = parser.parse_args()

    json_root = args.json_root
    question_files = sorted(p for p in json_root.rglob("*.json") if p.parent != json_root)
    if not question_files:
        raise SystemExit(f"No question JSON files found under {json_root}")

    id_width = max(4, len(str(len(question_files))))
    records = [
        record_for_file(path, json_root, f"{index:0{id_width}d}")
        for index, path in enumerate(question_files, start=1)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
