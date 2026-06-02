from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from config import Paths, log, normalize_text

if TYPE_CHECKING:
    from openai import OpenAI


_STREAM_END = object()


def _drain_stream_to_queue(resp: Any, q: "queue.Queue[Any]") -> None:
    """Consume `resp` iterator in a daemon thread, pushing chunks to `q`.

    Pushes raw chunks for normal items, an Exception instance on error, and
    a sentinel `_STREAM_END` when the iterator completes. The main thread
    can then use `q.get(timeout=...)` to enforce an idle deadline that
    actually fires when the upstream stalls (the bare `for chunk in resp`
    loop blocks indefinitely on slow chunked-encoding peers, defeating any
    elapsed-time check that only runs when a new chunk arrives).
    """
    try:
        for chunk in resp:
            q.put(chunk)
    except BaseException as exc:  # noqa: BLE001 - propagate to consumer
        q.put(exc)
    finally:
        q.put(_STREAM_END)

class LLMClientPool:
    def __init__(
        self,
        clients: list[OpenAI],
        primary_count: int | None = None,
        endpoints: list[tuple[str, str]] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("LLMClientPool requires at least one client")
        self.clients = clients
        self.primary_count = len(clients) if primary_count is None else min(max(primary_count, 0), len(clients))
        if self.primary_count == 0:
            self.primary_count = len(clients)
        self.lock = threading.Lock()
        self.next_index = 0
        self.dead: set[int] = set()
        self.log_fn = log_fn
        if endpoints is not None and len(endpoints) == len(clients):
            self.endpoint_labels = [f"{base_url} ...{key[-4:]}" for base_url, key in endpoints]
        else:
            self.endpoint_labels = [f"client[{i}]" for i in range(len(clients))]

    def _emit_log(self, msg: str) -> None:
        if self.log_fn is not None:
            try:
                self.log_fn(msg)
                return
            except Exception:
                pass
        print(msg, file=sys.stderr)

    def _mark_dead(self, idx: int, exc: Exception) -> None:
        status_code = getattr(exc, "status_code", None)
        with self.lock:
            if idx in self.dead:
                return
            self.dead.add(idx)
            alive_primary = sum(1 for i in range(self.primary_count) if i not in self.dead)
            fallback_total = len(self.clients) - self.primary_count
            alive_fallback = sum(
                1 for i in range(self.primary_count, len(self.clients)) if i not in self.dead
            )
            label = self.endpoint_labels[idx]
        self._emit_log(
            f"API key marked invalid endpoint={label} status={status_code} "
            f"alive={alive_primary}/{self.primary_count} primary, "
            f"{alive_fallback}/{fallback_total} fallback"
        )

    def create_completion(self, **kwargs: Any) -> Any:
        with self.lock:
            if len(self.dead) >= len(self.clients):
                raise RuntimeError("All API keys marked invalid; rotate keys in config.yaml")
        attempts = self._attempt_order()
        first_error: Exception | None = None
        for index in attempts:
            client = self.clients[index]
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) in {401, 403}:
                    self._mark_dead(index, exc)
                    continue
                if first_error is None:
                    first_error = exc
                if not self._should_try_next_client(exc):
                    raise
        if first_error is not None:
            raise first_error
        raise RuntimeError("All API keys marked invalid; rotate keys in config.yaml")

    def _attempt_order(self) -> list[int]:
        with self.lock:
            dead_snapshot = set(self.dead)
            primary_alive = [i for i in range(self.primary_count) if i not in dead_snapshot]
            if primary_alive:
                start = self.next_index % len(primary_alive)
                self.next_index += 1
            else:
                start = 0
        if primary_alive:
            primary = [primary_alive[(start + offset) % len(primary_alive)] for offset in range(len(primary_alive))]
        else:
            primary = []
        fallback = [i for i in range(self.primary_count, len(self.clients)) if i not in dead_snapshot]
        return primary + fallback

    @staticmethod
    def _should_try_next_client(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return int(status_code) in {401, 403, 408, 409, 429, 500, 502, 503, 504}
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}

REFUSAL_PATTERNS = (
    "request was rejected because it was considered high risk",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot help with",
    "i can't help with",
    "i cannot assist",
    "i can't assist",
    "i cannot generate",
    "i can't generate",
    "i'm unable to",
    "i am unable to",
    "i cannot create",
    "i can't create",
    "violates our content policy",
    "against my guidelines",
    "against the content policy",
    "我无法帮助",
    "我无法生成",
    "我不能生成",
    "无法满足该请求",
    "内容政策",
)

def _looks_like_refusal(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if len(stripped) > 600:
        return False
    lowered = stripped.lower()
    return any(pat in lowered for pat in REFUSAL_PATTERNS)

def _raw_starts_with_refusal(text: str) -> bool:
    head = text.strip()[:600].lower()
    if not head:
        return False
    return any(pat in head for pat in REFUSAL_PATTERNS)

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
    cacheable_prefix: str | None = None,
) -> str:
    api = config["api"]
    context_window = int(api.get("context_window", 1000000))
    max_input_chars = int(context_window * 1.8)
    # Prepend cacheable_prefix verbatim at the very top of user message so that
    # repeated invocations across chapters share an identical prefix and the
    # provider's prefix cache can hit. The prefix should contain ONLY content
    # that does not change call-to-call within a window (creative brief, bible,
    # characters, voice anchors).
    if cacheable_prefix:
        user = cacheable_prefix.rstrip() + "\n\n" + user
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
                stream_max = int(api.get("stream_timeout", 600))
                idle_startup = int(api.get("stream_idle_startup", api.get("stream_idle_timeout", 90)))
                idle_steady = int(api.get("stream_idle_steady", 15))
                startup_grace_secs = int(api.get("stream_startup_grace_secs", 30))
                # Phases:
                #   - TTFB (chunk_count == 0): no idle check; bounded by httpx
                #     read_timeout. Some providers take 60-300s to send the
                #     first SSE event while the model "thinks".
                #   - Pre-content (chunks arriving but no content yet):
                #     idle_startup applies (loose, allows reasoning gaps).
                #   - Steady (content has begun streaming): idle_steady
                #     applies (tight, catches mid-stream stalls fast).
                stream_timeout_reason: str | None = None
                first_content_at: float | None = None
                # Run the SSE iterator in a daemon thread so the main loop's
                # idle/total timeouts always fire, even when the upstream
                # connection is hanging mid-chunk (e.g. proxy buffering, slow
                # peer that never closes). queue.get(timeout=) is the only
                # reliable hard timeout for chunked iteration.
                chunk_q: "queue.Queue[Any]" = queue.Queue()
                reader = threading.Thread(
                    target=_drain_stream_to_queue,
                    args=(resp, chunk_q),
                    daemon=True,
                )
                reader.start()
                while True:
                    now = time.perf_counter()
                    if now - started > stream_max:
                        stream_timeout_reason = f"stream exceeded {stream_max}s total limit"
                        break
                    in_steady = first_content_at is not None
                    idle_max = idle_steady if in_steady else idle_startup
                    # Cap the per-get wait so we re-check stream_max regularly.
                    remaining_total = max(1.0, stream_max - (now - started))
                    wait_secs = min(float(idle_max), remaining_total)
                    try:
                        item = chunk_q.get(timeout=wait_secs)
                    except queue.Empty:
                        stream_timeout_reason = (
                            f"stream idle exceeded {idle_max}s "
                            f"(phase={'steady' if in_steady else 'startup'}); chunks={chunk_count}"
                        )
                        break
                    if item is _STREAM_END:
                        # Iterator exhausted cleanly.
                        break
                    if isinstance(item, BaseException):
                        stream_timeout_reason = f"stream connection dropped: {item}"
                        break
                    chunk = item
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
                        if first_content_at is None:
                            first_content_at = time.perf_counter()
                    reasoning_piece = getattr(delta, "reasoning_content", None)
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                if stream_timeout_reason is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                content = "".join(parts)
                elapsed = time.perf_counter() - started
                min_salvage_chars = int(api.get("stream_salvage_min_chars", 800))
                if stream_timeout_reason and len(content.strip()) >= min_salvage_chars:
                    log(
                        paths,
                        f"Stream cut off but salvaging partial content "
                        f"attempt={attempt + 1}/6 chunks={chunk_count} "
                        f"content_chars={len(content)} elapsed={elapsed:.1f}s reason={stream_timeout_reason}",
                    )
                    stream_timeout_reason = None
                if stream_timeout_reason:
                    raise TimeoutError(stream_timeout_reason)
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
            if _looks_like_refusal(content):
                wait = min(60, 2**attempt)
                log(
                    paths,
                    f"LLM returned refusal-like response attempt={attempt + 1}/6 wait={wait}s "
                    f"len={len(content.strip())} preview={content.strip()[:120]!r}",
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
    if _raw_starts_with_refusal(raw):
        log(paths, f"JSON repair skipped: provider refusal detected. Preview: {raw.strip()[:200]!r}")
        if fallback is not None:
            return fallback
        raise json.JSONDecodeError("Provider refusal, not malformed JSON", raw, 0)
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
