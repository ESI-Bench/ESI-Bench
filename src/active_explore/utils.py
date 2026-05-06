from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def resolve_path(raw_path: str | None, source_json: Path, data_root: Path | None = None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    roots = [source_json.parent, Path.cwd()]
    if data_root is not None:
        roots.append(data_root)
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def option_lines(options: object) -> list[str]:
    if not isinstance(options, list):
        return []
    lines = []
    for option in options:
        if not isinstance(option, dict):
            continue
        option_id = normalize_text(option.get("option_id"))
        text = normalize_text(option.get("text"))
        if option_id and text:
            lines.append(f"{option_id}. {text}")
        elif text:
            lines.append(text)
    return lines


def normalize_option_answer(value: object) -> str:
    return normalize_text(value).lower()


def normalize_prediction_to_option(payload: dict[str, Any], predicted_answer: object) -> dict[str, Any]:
    normalized_prediction = normalize_option_answer(predicted_answer)
    options = payload.get("qa", {}).get("options", [])
    if not isinstance(options, list):
        options = []

    for option in options:
        if not isinstance(option, dict):
            continue
        option_id = normalize_text(option.get("option_id"))
        text = normalize_text(option.get("text"))
        candidates = {
            normalize_option_answer(option_id),
            normalize_option_answer(text),
            normalize_option_answer(f"{option_id}. {text}"),
        }
        if normalized_prediction in candidates:
            return {"matched": True, "answer_option_id": option_id, "answer_text": text}

    return {"matched": False, "answer_option_id": None, "answer_text": None}


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text or ""):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_model_output(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw_text = "\n".join(lines[1:-1]).strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].lstrip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
        return {"raw_response": raw_text}
    except json.JSONDecodeError:
        extracted = extract_json_object(raw_text)
        if extracted is not None:
            return extracted
        return {"raw_response": raw_text}


def normalize_parsed_response(parsed_response: object) -> dict[str, Any] | None:
    if isinstance(parsed_response, dict):
        return parsed_response
    if isinstance(parsed_response, str):
        return extract_json_object(parsed_response) or parse_model_output(parsed_response)
    return None


def normalize_options(options: object) -> object:
    if not isinstance(options, list):
        return None
    if not options:
        return []
    if all(isinstance(item, str) for item in options):
        return options
    if all(isinstance(item, list) for item in options):
        return options
    return options


def normalize_answer_sequence(value: object) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(item).lower() for item in value if normalize_text(item)]

    text = normalize_text(value)
    if not text:
        return []
    if "|" in text:
        return [part.strip().lower() for part in text.split("|") if part.strip()]
    if "\n" in text:
        return [part.strip().lower() for part in text.splitlines() if part.strip()]
    return [text.lower()]


def normalize_answer_for_eval(answer: object, options: object) -> object:
    normalized_options = normalize_options(options)
    if isinstance(normalized_options, list) and normalized_options and all(isinstance(item, list) for item in normalized_options):
        return normalize_answer_sequence(answer)
    return normalize_text(answer).lower()


def compute_exact_match(prediction: object, ground_truth: object, options: object) -> bool | None:
    pred = normalize_answer_for_eval(prediction, options)
    gt = normalize_answer_for_eval(ground_truth, options)
    if pred == "" or pred == [] or gt == "" or gt == []:
        return None
    return pred == gt


ERROR_TEXT_MARKERS = (
    "traceback",
    "exception",
    "error",
    "quota",
    "rate limit",
    "internal server error",
    "api key",
    "timeout",
    "unavailable",
)


def text_looks_like_error(value: object) -> bool:
    text = normalize_text(value).casefold()
    if not text:
        return False
    return any(marker in text for marker in ERROR_TEXT_MARKERS)


def prediction_has_error(prediction: object) -> bool:
    if isinstance(prediction, dict):
        for key in ("error", "errors", "exception"):
            if normalize_text(prediction.get(key)):
                return True
        answer = normalize_text(prediction.get("answer"))
        if answer and text_looks_like_error(answer):
            return True
        reasoning = normalize_text(prediction.get("reasoning"))
        if reasoning and text_looks_like_error(reasoning):
            return True
        raw_response = prediction.get("raw_response")
        if text_looks_like_error(raw_response):
            return True
    if isinstance(prediction, str):
        return text_looks_like_error(prediction)
    return False
