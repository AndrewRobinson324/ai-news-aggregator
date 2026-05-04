"""Structured JSON completions via OpenAI, Gemini, or Ollama (OpenAI-compatible API)."""

import json
import time
from typing import TypeVar, get_args, get_origin

from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError
from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_GEMINI_RETRY_CODES = frozenset({429, 500, 503})
_GEMINI_MAX_ATTEMPTS = 5


def structured_openai(
    client: OpenAI,
    model: str,
    instructions: str,
    user_prompt: str,
    temperature: float,
    schema: type[T],
) -> T | None:
    try:
        response = client.responses.parse(
            model=model,
            instructions=instructions,
            temperature=temperature,
            input=user_prompt,
            text_format=schema,
        )
        return response.output_parsed
    except Exception as e:
        print(f"OpenAI structured parse error: {e}")
        return None


def structured_gemini(
    client: genai.Client,
    model: str,
    instructions: str,
    user_prompt: str,
    temperature: float,
    schema: type[T],
) -> T | None:
    for attempt in range(_GEMINI_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                return None
            return schema.model_validate_json(text)
        except GeminiAPIError as e:
            if e.code not in _GEMINI_RETRY_CODES or attempt >= _GEMINI_MAX_ATTEMPTS - 1:
                print(f"Gemini structured parse error: {e}")
                return None
            delay = min(2.0**attempt, 30.0)
            print(f"Gemini transient error ({e.code}); retrying in {delay:.0f}s…")
            time.sleep(delay)
        except Exception as e:
            print(f"Gemini structured parse error: {e}")
            return None


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:])
        if "```" in text:
            text = text.rsplit("```", 1)[0].strip()
    return text.strip()


def _ollama_flat_string_fields_only(schema: type[BaseModel]) -> bool:
    """Every top-level field is str (``str | None`` allowed)."""
    for finfo in schema.model_fields.values():
        ann = finfo.annotation
        origin = get_origin(ann)
        if origin is not None and get_args(ann) and type(None) in get_args(ann):
            rest = [a for a in get_args(ann) if a is not type(None)]
            if len(rest) != 1:
                return False
            ann = rest[0]
        if ann is not str:
            return False
    return True


def _ollama_json_suffix(schema: type[BaseModel]) -> str:
    """Build user suffix: small models get value examples (avoids llama echoing JSON Schema)."""
    if _ollama_flat_string_fields_only(schema):
        example = {name: f"<your {name} text>" for name in schema.model_fields}
        return (
            "Respond with ONE JSON object only (no markdown).\n"
            "Fill these keys with real sentences — output DATA only, not JSON Schema "
            '(no "properties", "type", "$schema", or nested schema objects).\n'
            "Example shape (replace angle-bracket placeholders with real strings):\n"
            + json.dumps(example, indent=2, ensure_ascii=False)
        )
    doc = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with one JSON object only (no markdown). "
        "Use concrete values for fields (titles, summaries, scores), not a schema definition:\n"
        + doc
    )


def _looks_like_json_schema_echo(d: dict) -> bool:
    """Detect models that echoed JSON Schema instead of an instance."""
    if "greeting" in d and isinstance(d.get("greeting"), str):
        return False
    if "title" in d and isinstance(d.get("title"), str) and "summary" in d:
        return False
    if d.get("type") == "object" and "properties" in d:
        return True
    if "$schema" in d or "$defs" in d:
        return True
    return False


def _parse_json_lenient(raw: str) -> dict | list | None:
    text = _strip_json_fence(raw)
    if not text:
        return None
    try:
        out = json.loads(text)
        return out if isinstance(out, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            out = json.loads(text[start : end + 1])
            return out if isinstance(out, (dict, list)) else None
        except json.JSONDecodeError:
            return None
    return None


def structured_ollama_chat(
    client: OpenAI,
    model: str,
    instructions: str,
    user_prompt: str,
    temperature: float,
    schema: type[T],
    *,
    max_tokens: int | None = None,
) -> T | None:
    """Call Ollama via OpenAI-compatible `/v1/chat/completions`; tolerant of missing json mode."""
    suffix = _ollama_json_suffix(schema)
    augmented = f"{user_prompt}\n\n{suffix}"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": augmented},
    ]

    for use_json_object in (True, False):
        kwargs: dict = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if use_json_object:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            err_txt = str(e).lower()
            if use_json_object and any(
                x in err_txt for x in ("response_format", "json_object", "400", "unsupported", "invalid")
            ):
                print(f"Ollama: json_object rejected ({e}); retrying without response_format…")
                continue
            print(f"Ollama API error: {e}")
            return None

        raw = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json_lenient(raw)
        if parsed is None:
            if use_json_object:
                print("Ollama: empty or non-JSON reply with json_object; retrying without response_format…")
                continue
            print("Ollama structured parse error: could not extract JSON from response")
            return None

        if isinstance(parsed, list):
            if use_json_object:
                print("Ollama: expected object, got array; retrying without response_format…")
                continue
            print("Ollama structured parse error: expected JSON object, got array")
            return None

        if isinstance(parsed, dict) and _looks_like_json_schema_echo(parsed):
            if use_json_object:
                print("Ollama: model echoed JSON Schema; retrying without response_format…")
                continue
            print("Ollama: model echoed JSON Schema instead of data")
            return None

        try:
            return schema.model_validate(parsed)
        except ValidationError as e:
            if use_json_object:
                print(f"Ollama: schema mismatch with json_object ({e}); retrying without response_format…")
                continue
            print(f"Ollama structured parse error: {e}")
            return None

    return None
