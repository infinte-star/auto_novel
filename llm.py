from __future__ import annotations

import json
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from config import Paths, log, normalize_text

if TYPE_CHECKING:
    from openai import OpenAI

class LLMClientPool:
    def __init__(self, clients: list[OpenAI], primary_count: int | None = None) -> None:
        if not clients:
            raise ValueError("LLMClientPool requires at least one client")
        self.clients = clients
        self.primary_count = len(clients) if primary_count is None else min(max(primary_count, 0), len(clients))
        if self.primary_count == 0:
            self.primary_count = len(clients)
        self.lock = threading.Lock()
        self.next_index = 0

    def create_completion(self, **kwargs: Any) -> Any:
        attempts = self._attempt_order()
        first_error: Exception | None = None
        for index in attempts:
            client = self.clients[index]
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                if not self._should_try_next_client(exc):
                    raise
        if first_error is not None:
            raise first_error
        raise RuntimeError("LLMClientPool has no clients to try")

    def _attempt_order(self) -> list[int]:
        with self.lock:
            start = self.next_index % self.primary_count
            self.next_index += 1
        primary = [(start + offset) % self.primary_count for offset in range(self.primary_count)]
        fallback = list(range(self.primary_count, len(self.clients)))
        return primary + fallback

    @staticmethod
    def _should_try_next_client(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return int(status_code) in {401, 408, 409, 429, 500, 502, 503, 504}
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}

def _repair_truncated_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    for end in range(len(s), max(len(s) - 5000, 0), -100):
        candidate = s[:end].rstrip(", \t\n\r:")
        candidate = re.sub(r',\s*"[^"]*"?\s*:?\s*[^,{}\[\]]*$', "", candidate).rstrip(", \t\n\r:")
        stack: list[str] = []
        in_str = False
        esc = False
        broken = False
        for c in candidate:
            if esc:
                esc = False
                continue
            if in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    broken = True
                    break
                stack.pop()
        if broken or in_str:
            continue
        closer = "".join("}" if o == "{" else "]" for o in reversed(stack))
        repaired = candidate + closer
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            continue
    return None

def safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = normalize_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_json(cleaned)
    if repaired:
        return json.loads(repaired)
    raise json.JSONDecodeError(f"Could not recover JSON. Preview: {cleaned[:300]!r}", cleaned, 0)

JSON_REPAIR_SYSTEM = """You repair malformed JSON from an LLM response.
Return valid JSON only. Do not add explanations. Preserve the intended fields and values."""

JSON_OUTPUT_CONTRACT = """Output contract:
- Return exactly one valid JSON object and nothing else.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Do not use markdown headings, bullet lists, code fences, explanations, or prefaces.
- Use double quotes for every key and string value.
- Escape quotes inside string values.
- Do not use trailing commas, comments, NaN, Infinity, or Python-style booleans.
- Keep the schema keys exactly as requested; do not translate key names.
- If uncertain, still return the requested schema with conservative values and short Chinese strings."""

def json_prompt(user: str) -> str:
    return user.rstrip() + "\n\n## Mandatory JSON Output Contract\n" + JSON_OUTPUT_CONTRACT

def emergency_truncate(user_text: str, max_chars: int) -> str:
    if len(user_text) <= max_chars:
        return user_text
    sections = re.split(r"(?=^## )", user_text, flags=re.MULTILINE)
    priority_keywords = ["Creative Brief", "Current State", "Selected Plan", "Arbitration"]
    high = []
    medium = []
    low = []
    for section in sections:
        if any(kw in section[:80] for kw in priority_keywords):
            high.append(section)
        elif any(kw in section[:80] for kw in ["Characters", "Bible", "Volume Plan", "Threads"]):
            medium.append(section)
        else:
            low.append(section)
    result = "".join(high)
    for section in medium:
        if len(result) + len(section) < max_chars * 0.85:
            result += section
        else:
            remaining = int(max_chars * 0.85) - len(result)
            if remaining > 500:
                result += section[:remaining] + "\n...[truncated]"
            break
    for section in low:
        if len(result) + len(section) < max_chars:
            result += section
        else:
            remaining = max_chars - len(result)
            if remaining > 500:
                result += section[:remaining] + "\n...[truncated]"
            break
    return result

def call_llm(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    system: str,
    user: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool | None = None,
) -> str:
    api = config["api"]
    context_window = int(api.get("context_window", 1000000))
    max_input_chars = int(context_window * 1.8)
    total_chars = len(system) + len(user)
    if total_chars > max_input_chars:
        user = emergency_truncate(user, max_input_chars - len(system) - 1000)
    wants_json = json_mode if json_mode is not None else "Mandatory JSON Output Contract" in user
    use_response_format = wants_json and bool(api.get("json_response_format", True))
    for attempt in range(6):
        started = time.perf_counter()
        try:
            request = {
                "model": api["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens or int(api["max_tokens"]),
                "temperature": float(api["temperature"]) if temperature is None else temperature,
            }
            if use_response_format:
                request["response_format"] = {"type": "json_object"}
            stream = bool(api.get("stream", False))
            if stream:
                request["stream"] = True
            if hasattr(client, "create_completion"):
                resp = client.create_completion(**request)
            else:
                resp = client.chat.completions.create(**request)
            if stream:
                parts: list[str] = []
                reasoning_parts: list[str] = []
                chunk_count = 0
                finish_reason = None
                for chunk in resp:
                    chunk_count += 1
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = getattr(choice, "finish_reason", finish_reason)
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece:
                        parts.append(piece)
                    reasoning_piece = getattr(delta, "reasoning_content", None)
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                content = "".join(parts)
                elapsed = time.perf_counter() - started
                if not content.strip() and reasoning_parts:
                    log(
                        paths,
                        "LLM content empty but reasoning_content present, using reasoning fallback "
                        f"attempt={attempt + 1}/6 chunks={chunk_count} finish={finish_reason} "
                        f"reasoning_chars={sum(len(p) for p in reasoning_parts)} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
                    content = "".join(reasoning_parts)
                elif not content.strip():
                    log(
                        paths,
                        "LLM returned empty streamed response "
                        f"attempt={attempt + 1}/6 chunks={chunk_count} finish={finish_reason} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
            else:
                content = resp.choices[0].message.content or ""
                elapsed = time.perf_counter() - started
            if not content.strip():
                wait = min(60, 2**attempt)
                log(
                    paths,
                    f"LLM returned empty response attempt={attempt + 1}/6 wait={wait}s "
                    f"stream={stream} elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                )
                time.sleep(wait)
                continue
            return content
        except Exception as exc:
            if use_response_format and _looks_like_response_format_error(exc):
                use_response_format = False
                log(paths, f"JSON response_format unsupported, retrying without it: {exc}")
                continue
            wait = min(60, 2**attempt)
            elapsed = time.perf_counter() - started
            log(paths, f"LLM call failed attempt={attempt + 1}/6 wait={wait}s elapsed={elapsed:.1f}s error={exc}")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after 6 attempts")

def _looks_like_response_format_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return (
        status_code in {400, 404, 422}
        and "response_format" in text
        or "response_format" in text
        and any(word in text for word in ["unsupported", "not support", "unknown", "invalid", "extra"])
    )

def load_json_with_repair(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    raw: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return safe_json_loads(raw)
    except json.JSONDecodeError as exc:
        log(paths, f"JSON parse failed, attempting repair: {exc}")
    if not raw.strip():
        if fallback is not None:
            return fallback
        raise json.JSONDecodeError("Empty JSON response", raw, 0)
    repair_prompt = f"""Repair this malformed JSON into one valid JSON object.

## Malformed JSON
{raw[:20000]}"""
    try:
        repaired = call_llm(
            client,
            paths,
            config,
            JSON_REPAIR_SYSTEM,
            json_prompt(repair_prompt),
            max_tokens=8000,
            temperature=0,
        )
        return safe_json_loads(repaired)
    except Exception as exc:
        log(paths, f"JSON repair failed: {exc}")
        if fallback is not None:
            return fallback
        raise
