from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


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
    output = dict(fallback or {})
    output.setdefault("raw_response", raw)
    output.setdefault("error", "JSON parse failed")
    return output


def guess_mime_type(image_path: str | Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def image_url_part(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{guess_mime_type(path)};base64,{encoded}"},
    }


class GPTModel:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5"):
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def _convert_contents(self, contents: list[Any]) -> list[dict[str, Any]]:
        converted = []
        for item in contents:
            if isinstance(item, Path):
                converted.append(image_url_part(item))
            else:
                converted.append({"type": "text", "text": str(item)})
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
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": self._convert_contents(contents)},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        # Some reasoning models reject custom sampling values; retry without them below.
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        raw_text = (choice.message.content or "").strip()
        finish_reason = str(getattr(choice, "finish_reason", None))
        return parse_json_response(raw_text, fallback=fallback), raw_text, finish_reason
