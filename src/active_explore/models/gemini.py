from __future__ import annotations

import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
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


def extract_partial_json_fields(raw_text: str) -> dict[str, Any] | None:
    """Recover simple scalar fields when a model response is cut off mid-JSON."""
    raw = raw_text or ""
    recovered: dict[str, Any] = {}
    for key in ("action", "answer", "reasoning"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)', raw, flags=re.DOTALL)
        if match:
            recovered[key] = match.group(1).strip()
    confidence = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if confidence:
        try:
            recovered["confidence"] = float(confidence.group(1))
        except ValueError:
            pass
    return recovered or None


def normalize_parsed_response(parsed_response: object) -> dict[str, Any] | None:
    if isinstance(parsed_response, dict):
        return parsed_response
    if isinstance(parsed_response, str):
        return extract_json_object(parsed_response)
    return None


def parse_json_response(raw_text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (raw_text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = extract_json_object(raw)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict):
        return parsed
    partial = extract_partial_json_fields(raw)
    if partial is not None:
        output = dict(fallback or {})
        output.update(partial)
        return output
    output = dict(fallback or {})
    output.setdefault("raw_response", raw)
    output.setdefault("error", "JSON parse failed")
    return output


def guess_mime_type(image_path: str | Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def image_part(image_path: str | Path) -> types.Part:
    path = Path(image_path)
    with path.open("rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=guess_mime_type(path))


class GeminiModel:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass api_key.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def _convert_contents(self, contents: list[Any]) -> list[Any]:
        converted = []
        for item in contents:
            if isinstance(item, Path):
                converted.append(image_part(item))
            else:
                converted.append(item)
        return converted

    def generate_json(
        self,
        contents: list[Any],
        system_instruction: str,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        fallback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, str | None]:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

        converted_contents = self._convert_contents(contents)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=converted_contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            if "Budget 0 is invalid" not in str(exc) or "thinking_config" not in config_kwargs:
                raise
            config_kwargs.pop("thinking_config", None)
            response = self.client.models.generate_content(
                model=self.model,
                contents=converted_contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        finish_reason = None
        raw_text = (response.text or "").strip()
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            finish_reason = str(getattr(candidate, "finish_reason", None))
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            candidate_text = "".join(part.text for part in parts if getattr(part, "text", None)).strip()
            if candidate_text:
                raw_text = candidate_text

        parsed = normalize_parsed_response(getattr(response, "parsed", None))
        if parsed is None:
            parsed = parse_json_response(raw_text, fallback=fallback)
        return parsed, raw_text, finish_reason
