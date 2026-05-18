#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict


REPO_ROOT = Path(__file__).resolve().parents[2]

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

ROW_KEYS = (
    "id",
    "big_task",
    "small_task",
    "runner_task",
    "scene",
    "room",
    "question",
    "answer",
    "answer_type",
    "options_json",
    "image_paths_json",
    "metadata_json",
)

YES_NO_PREFIXES = (
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "can ",
    "will ",
    "would ",
    "should ",
    "comparing ",
)


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


def answer_type(answer: Any, options: Any) -> str:
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

    def clean_path(value: str) -> str:
        return Path(value).name if "/" in value or "\\" in value else value

    def add(value: Any) -> None:
        if isinstance(value, str) and value:
            paths.append(clean_path(value))
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


def non_empty(value: Any) -> bool:
    return value not in (None, [], {}, "")


def unique_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        key = json_text(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def display_text(value: Any) -> str:
    return text(value).replace("_", " ") if text(value) else ""


def keyed_list_to_map(items: Any) -> dict[str, Any]:
    if isinstance(items, dict):
        return {str(key): value for key, value in items.items()}
    output: dict[str, Any] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and text(item.get("_key")):
                output[str(item["_key"])] = item
    return output


def object_category(payload: dict[str, Any], key: str) -> str | None:
    item = keyed_list_to_map(payload.get("objects")).get(key)
    if isinstance(item, dict):
        return display_text(item.get("category") or item.get("cat"))
    return None


def split_or_choices(raw: str) -> list[str]:
    raw = raw.strip().strip("?").strip()
    raw = re.sub(r"\s+", " ", raw)
    parts = re.split(r"\s+or\s+", raw)
    return [part.strip(" .,:;") for part in parts if part.strip(" .,:;")]


def infer_options_from_question(payload: dict[str, Any], question: str | None, answer: Any) -> Any:
    if not question:
        return []
    q = question.strip()
    q_lower = q.lower()
    answer_lower = normalize_answer(answer).lower() if isinstance(normalize_answer(answer), str) else ""

    if answer_lower in {"yes", "no"} and q_lower.startswith(YES_NO_PREFIXES):
        return ["yes", "no"]
    if answer_lower in {"yes", "no"} and re.search(r"\b(is|are|do|does|did|can|will|would|should)\b", q_lower):
        return ["yes", "no"]

    if "equilateral triangle" in q_lower and "isosceles triangle" in q_lower and "random triangle" in q_lower:
        return ["equilateral", "isosceles", "random"]

    if "answer 0, 1, 2, or 3" in q_lower:
        return ["0", "1", "2", "3"]

    match = re.search(r"larger:\s*the one near the (.*?),\s*or the one near the (.*?)\?", q, flags=re.I)
    if match:
        return [f"near the {match.group(1).strip()}", f"near the {match.group(2).strip()}"]

    match = re.search(r"closer to .*?:\s*the (.*?)\s+or\s+the (.*?)\?", q, flags=re.I)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    match = re.search(r"Which object is this:\s*(.*?)\?", q, flags=re.I)
    if match:
        return split_or_choices(match.group(1))

    match = re.search(r"What is the occluded object:\s*(.*?)\?", q, flags=re.I)
    if match:
        return split_or_choices(match.group(1))

    match = re.search(r"larger volume:\s*the (.*?)\s+or\s+the (.*?)\?", q, flags=re.I)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    if payload.get("left_category") and payload.get("right_category"):
        return [display_text(payload.get("left_category")), display_text(payload.get("right_category"))]

    match = re.search(r"best fit inside:\s*the (.*?)\?", q, flags=re.I)
    if match:
        return [item.removeprefix("the ").strip() for item in split_or_choices(match.group(1))]
    match = re.search(r"Containers:\s*(.*)$", q, flags=re.I | re.S)
    if match:
        return [item.strip() for item in match.group(1).replace("\n", ",").split(",") if item.strip()]

    match = re.search(r"How should you stack (.*?) to maximize stability\?", q, flags=re.I)
    if match:
        return [item.strip() for item in match.group(1).split(",") if item.strip()]
    if isinstance(payload.get("object_categories"), list):
        return payload["object_categories"]

    if "which object is closer" in q_lower and " a or b" in q_lower:
        return ["A", "B"]
    if "left, middle, or right" in q_lower:
        return ["left", "middle", "right"]

    return []


def extract_options(payload: dict[str, Any], question: str | None, answer: Any) -> Any:
    qd = payload.get("question_data") if isinstance(payload.get("question_data"), dict) else {}
    qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}

    for candidate in (qd.get("options"), qa.get("options"), qa.get("choices")):
        if non_empty(candidate):
            return candidate

    choices_by_part = {
        suffix: qa.get(f"choices_{suffix}")
        for suffix in ("A", "C")
        if non_empty(qa.get(f"choices_{suffix}"))
    }
    if choices_by_part:
        return choices_by_part

    options = infer_options_from_question(payload, question, answer)
    if non_empty(options):
        return options

    if payload.get("near_category") and payload.get("far_category"):
        return [display_text(payload.get("near_category")), display_text(payload.get("far_category"))]

    size_options = [object_category(payload, "ref_obj1"), object_category(payload, "ref_obj2")]
    if all(size_options):
        return [f"near the {size_options[0]}", f"near the {size_options[1]}"]

    distance_options = [object_category(payload, "obj_near"), object_category(payload, "obj_far")]
    if all(distance_options):
        return unique_list(distance_options)

    return []


def extract_question_answer(payload: dict[str, Any]) -> tuple[str | None, Any, Any]:
    qd = payload.get("question_data") if isinstance(payload.get("question_data"), dict) else {}
    qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}

    question = (
        text(payload.get("_question"))
        or text(qa.get("question"))
        or text(qd.get("question"))
        or text((payload.get("_entry") or {}).get("question") if isinstance(payload.get("_entry"), dict) else None)
    )

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

    options = extract_options(payload, question, answer)
    return question, normalize_answer(answer), options


def clean_path_string(value: str, key_hint: str = "") -> str:
    if not value:
        return value
    lowered_key = key_hint.lower()
    looks_path_key = any(token in lowered_key for token in ("path", "file", "image", "rgb", "view"))
    looks_absolute = value.startswith("/")
    if looks_path_key or looks_absolute:
        if "/" in value or "\\" in value:
            return Path(value).name
    return value


def prune_empty(value: Any, key_hint: str = "") -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            cleaned = prune_empty(item, str(key))
            if cleaned not in (None, [], {}, ""):
                output[key] = cleaned
        return output
    if isinstance(value, list):
        output = [prune_empty(item, key_hint) for item in value]
        return [item for item in output if item not in (None, [], {}, "")]
    if isinstance(value, str):
        return clean_path_string(value, key_hint)
    return value


def metadata_payload(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    metadata = prune_empty(payload)
    if not isinstance(metadata, dict):
        return {}
    for key in ("scene", "room"):
        metadata.pop(key, None)
    if text(metadata.get("_question")) == row.get("question"):
        metadata.pop("_question", None)
    if answer_text(metadata.get("_ground_truth")) == row.get("answer"):
        metadata.pop("_ground_truth", None)
    if answer_text(metadata.get("answer")) == row.get("answer"):
        metadata.pop("answer", None)
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

    row = {
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
        "metadata_json": "",
    }
    row["metadata_json"] = json_text(metadata_payload(payload, row))
    return {key: row.get(key) for key in ROW_KEYS}


def write_clean_summaries(clean_root: Path, records: list[dict[str, Any]], sources: list[Path], json_root: Path) -> None:
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for record, source in zip(records, sources, strict=True):
        grouped[str(record["big_task"])].append((record, source))

    for big_task, items in sorted(grouped.items()):
        per_type = Counter(str(record["small_task"]) for record, _source in items)
        json_paths = [
            str(source.relative_to(json_root))
            for _record, source in sorted(items, key=lambda item: str(item[1].relative_to(json_root)))
        ]
        summary = {
            "root_dir": ".",
            "count": len(items),
            "sampling_seed": 7,
            "per_type_target": max(per_type.values()) if per_type else 0,
            "per_type_targets": dict(sorted(per_type.items())),
            "total_target": len(items),
            "available_json_count": len(items),
            "valid_json_count": len(items),
            "filtered_out_missing_images_count": 0,
            "available_per_task_type": dict(sorted(per_type.items())),
            "sampled_per_task_type": dict(sorted(per_type.items())),
            "json_paths": json_paths,
        }
        summary_path = clean_root / f"{big_task}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a fixed-column HF/Croissant-friendly questions table.")
    parser.add_argument("--json-root", type=Path, default=REPO_ROOT / "dataset/json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "hf_dataset/data/questions.jsonl")
    parser.add_argument("--clean-root", type=Path, default=REPO_ROOT / "dataset/json_clean")
    parser.add_argument("--include-invalid", action="store_true", help="Keep source JSONs that have no question and no answer.")
    args = parser.parse_args()

    json_root = args.json_root
    question_files = sorted(p for p in json_root.rglob("*.json") if p.parent != json_root)
    if not question_files:
        raise SystemExit(f"No question JSON files found under {json_root}")

    id_width = max(4, len(str(len(question_files))))
    records: list[dict[str, Any]] = []
    record_sources: list[Path] = []
    skipped_invalid: list[Path] = []
    for path in question_files:
        record = record_for_file(path, json_root, "0" * id_width)
        if not args.include_invalid and not record.get("question") and not record.get("answer"):
            skipped_invalid.append(path)
            continue
        records.append(record)
        record_sources.append(path)

    id_width = max(4, len(str(len(records))))
    for index, record in enumerate(records, start=1):
        record["id"] = f"{index:0{id_width}d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    if args.clean_root:
        if args.clean_root.exists():
            shutil.rmtree(args.clean_root)
        for record, source in zip(records, record_sources, strict=True):
            target = args.clean_root / source.relative_to(json_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_clean_summaries(args.clean_root, records, record_sources, json_root)
        print(f"Wrote {len(records)} clean JSON files to {args.clean_root}")

    print(f"Wrote {len(records)} records to {args.output}")
    if skipped_invalid:
        print(f"Skipped {len(skipped_invalid)} invalid source JSON files without question/answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
