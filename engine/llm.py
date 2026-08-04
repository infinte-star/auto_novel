from __future__ import annotations

import json
import queue
import random
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from engine.config import Paths, log, normalize_text

if TYPE_CHECKING:
    from openai import OpenAI


_STREAM_END = object()

# Per-base_url throttle state. Each endpoint has its own lock + timestamp so
# parallel calls to different endpoints (or even to the same endpoint with
# different keys) don't serialize each other unnecessarily.
_ENDPOINT_THROTTLE_LOCKS: dict[str, threading.Lock] = {}
_ENDPOINT_LAST_STARTED_AT: dict[str, float] = {}
_ENDPOINT_THROTTLE_META_LOCK = threading.Lock()  # guards the two dicts above


def _get_endpoint_throttle_state(base_url: str) -> tuple[threading.Lock, str]:
    """Return (lock, key) for the given base_url, creating if needed."""
    with _ENDPOINT_THROTTLE_META_LOCK:
        if base_url not in _ENDPOINT_THROTTLE_LOCKS:
            _ENDPOINT_THROTTLE_LOCKS[base_url] = threading.Lock()
            _ENDPOINT_LAST_STARTED_AT[base_url] = 0.0
        return _ENDPOINT_THROTTLE_LOCKS[base_url], base_url

# Prompt roles are the single source of truth for both model routing and prompt
# policy.  A new call type must be registered here once; routing and instruction
# enhancement then cannot drift apart.
PLAN_TAGS = frozenset({
    "plan_candidate",
    "plan_screen",
    "plan_arbitrate",
    "arc_plan",
    "arc_card_repair",
    "structural_diagnose",
    "bootstrap",
    "bootstrap_bible",
    "bootstrap_characters",
    "bootstrap_voice",
    "bootstrap_volume_plan",
    "bootstrap_frame",
    "bootstrap_voice_repair",
    "bootstrap_voices",
    "replan",
    "creative_boost",
    "hook_package",
    "package",
    "trial_route",
    "screenplay_plan",
})

WRITE_TAGS = frozenset({
    "write",
    "beat_repair",
    "revise",
    "em_dash_fix",
    "revise_hook",
    "refine_rewrite",
    "fix_expand",
    "fix_dialogue",
    "fix_hook",
    "fix_ccc",
    "fix_ccc_hook",
    "trial_write",
    "screenplay_write",
    "screenplay_revise",
    "synopsis",
})

EXTRACT_TAGS = frozenset({
    "extract",
    "memory_compress",
    "json_repair",
    # v2's bounded fallback: the writer was asked for prose AND a state delta in
    # one call and returned only prose. Deriving the delta from a finished
    # chapter is the same structured-extraction job `extract` does, so it routes
    # to the same cheap model — the point of the fallback is that it costs a
    # fraction of the write it is repairing, not another write.
    "delta_backfill",
    "contract",
    "screenplay_extract",
})

REVIEW_TAGS = frozenset({
    "review",
    "cold_reader",
    "stage_review",
    "pack_review",
    "macro_progress",
    "plan_review_fused",
    "plan_review_axis",
    "refine_diagnose",
    # v2 call (3): the cite-or-drop canon check. It is a judging call, so it
    # routes to the review model like every other judge — and, like them, must
    # not be the model that wrote the chapter it is judging.
    "canon_check",
    "hook_package_score",
    "trial_review",
    "screenplay_review",
    "anchor_judge",
})

# Prefixes cover intentionally parameterised tags such as
# `bootstrap_volume_detail_v2` and future focused repair/judge variants.
_ROLE_TAG_PREFIXES: tuple[tuple[str, str], ...] = (
    ("plan_review_", "review"),
    ("bootstrap_", "planning"),
    ("arc_", "planning"),
    ("plan_", "planning"),
    ("fix_", "writing"),
    ("anchor_", "review"),
)


def prompt_role_for_tag(tag: str) -> str:
    """Return the call's semantic role, or ``""`` for primary-model fallback."""
    normalized = str(tag or "").strip().lower()
    for role, tags in (
        ("planning", PLAN_TAGS),
        ("writing", WRITE_TAGS),
        ("extraction", EXTRACT_TAGS),
        ("review", REVIEW_TAGS),
    ):
        if normalized in tags:
            return role
    for prefix, role in _ROLE_TAG_PREFIXES:
        if normalized.startswith(prefix):
            return role
    return ""


# Ordered list of (role, pool_attr, api_attr, model_key) for model routing.
_ROLE_ROUTING = [
    ("planning", "planning_pool", "planning_api", "planning_model"),
    ("writing", "writing_pool", "writing_api", "writing_model"),
    ("extraction", "extraction_pool", "extraction_api", "extraction_model"),
    ("review", "review_pool", "review_api", "review_model"),
]

# Lightweight observability sink. call_llm appends one JSON line per finished
# call to logs/llm_calls.jsonl. We use a plain file (not the SQLite store) so
# the recorder is reachable from every call site without threading a `conn`
# through, and so background worker threads never contend store._DB_LOCK just
# to log a metric. `novel.py stats` aggregates this file. Disable via
# api.metrics_enabled: false. The lock only serializes the local append.
_METRICS_LOCK = threading.Lock()


def _record_llm_call(
    paths: Paths,
    api: dict[str, Any],
    *,
    tag: str,
    model: str,
    stream: bool,
    json_mode: bool,
    attempt: int,
    prompt_chars: int,
    output_chars: int,
    elapsed: float,
    salvaged: bool,
    ok: bool,
    error: str = "",
    reasoning_chars: int = 0,
) -> None:
    if not bool(api.get("metrics_enabled", True)):
        return
    record = {
        "ts": time.time(),
        "tag": tag,
        "model": model,
        "stream": stream,
        "json_mode": json_mode,
        # attempt is 0-based internally; persist 1-based attempt count.
        "attempts": attempt + 1,
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
        "elapsed": round(elapsed, 3),
        "salvaged": salvaged,
        "ok": ok,
    }
    if reasoning_chars:
        record["reasoning_chars"] = reasoning_chars
    if error:
        record["error"] = error[:200]
    try:
        path = paths.logs_dir / "llm_calls.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with _METRICS_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate log when it exceeds 50 MB so it doesn't grow unboundedly.
            try:
                if path.exists() and path.stat().st_size > 50 * 1024 * 1024:
                    rotated = path.with_suffix(".jsonl.1")
                    if rotated.exists():
                        rotated.unlink()
                    path.rename(rotated)
            except OSError:
                pass
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Observability must never break generation.
        pass



def _configured_min_request_interval(api: dict[str, Any]) -> float:
    explicit = api.get("min_request_interval_secs")
    if explicit is not None and str(explicit).strip() != "":
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            return 0.0
    rpm = api.get("max_rpm")
    if rpm is None or str(rpm).strip() == "":
        return 0.0
    try:
        rpm_value = float(rpm)
    except (TypeError, ValueError):
        return 0.0
    if rpm_value <= 0:
        return 0.0
    return 60.0 / rpm_value


def _throttle_request_start(paths: Paths, api: dict[str, Any]) -> None:
    interval = _configured_min_request_interval(api)
    if interval <= 0:
        return
    base_url = str(api.get("base_url", "default"))
    lock, key = _get_endpoint_throttle_state(base_url)
    with lock:
        now = time.perf_counter()
        wait = (_ENDPOINT_LAST_STARTED_AT[key] + interval) - now
        if wait > 0:
            log(paths, f"LLM throttle sleeping {wait:.1f}s min_interval={interval:.1f}s endpoint={base_url[:40]}")
            time.sleep(wait)
        _ENDPOINT_LAST_STARTED_AT[key] = time.perf_counter()


def _effective_max_tokens(api: dict[str, Any], requested: int | None) -> int:
    value = requested or int(api["max_tokens"])
    cap = api.get("max_output_tokens_cap")
    if cap is None or str(cap).strip() == "":
        cap = api.get("max_tokens_cap")
    if cap is None or str(cap).strip() == "":
        return int(value)
    try:
        cap_value = int(cap)
    except (TypeError, ValueError):
        return int(value)
    if cap_value <= 0:
        return int(value)
    return min(int(value), cap_value)


def _retry_after_secs(exc: Exception) -> float | None:
    """Extract a Retry-After hint (seconds) from an HTTP 429/503 error, if present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _backoff_wait(attempt: int, exc: Exception | None = None) -> float:
    """Exponential backoff with full jitter, longer for rate limits.

    Without jitter, parallel candidate calls that all hit 429 at the same instant
    retry in lockstep and re-trigger the limit (thundering herd). Full jitter
    (random in [0, cap]) decorrelates them. Honor server Retry-After when given.
    """
    status_code = getattr(exc, "status_code", None) if exc is not None else None
    is_rate_limit = status_code is not None and int(status_code) == 429
    # Cloudflare 504/502/520-524 gateway errors set Retry-After: 120, but those
    # are origin-overload/timeout (transient), not a real rate-limit quota — the
    # next attempt usually succeeds within a few seconds. Honoring the 120s hint
    # there just burns 2 minutes per retry (observed: 3×120s on a single
    # bootstrap). Cap gateway-error backoff to a short jittered window instead.
    is_gateway_error = status_code is not None and int(status_code) in {502, 503, 504, 520, 521, 522, 523, 524}
    if exc is not None and not is_gateway_error:
        hinted = _retry_after_secs(exc)
        if hinted is not None:
            # Respect the server hint but add small jitter and a sane ceiling.
            return min(120.0, hinted + random.uniform(0, 2.0))
    if is_rate_limit:
        # Base grows 4,8,16,... capped at 90s for 429 so we actually back off.
        cap = min(90.0, 4.0 * (2 ** attempt))
    elif is_gateway_error:
        # Short backoff for transient gateway timeouts: 4,8,16,32 capped at 32s.
        cap = min(32.0, 4.0 * (2 ** attempt))
    else:
        cap = min(60.0, 2.0 * (2 ** attempt))
    # Full jitter: random point in [cap/2, cap] keeps a floor while decorrelating.
    return random.uniform(cap / 2.0, cap)



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
        endpoint_models: list[str | None] | None = None,
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
        # Per-endpoint model override (None => use the request's model verbatim).
        # Lets a fallback endpoint that speaks a different model name (e.g. a
        # mimo endpoint behind a claude-opus-4-8 primary) receive its own model.
        if endpoint_models is not None and len(endpoint_models) == len(clients):
            self.endpoint_models = list(endpoint_models)
        else:
            self.endpoint_models = [None] * len(clients)
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
        try:
            print(msg, file=sys.stderr)
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace"), file=sys.stderr)

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
            # Per-endpoint model override: a fallback endpoint may speak a
            # different model name than the primary. Apply it per-attempt so
            # rotation/fallback always sends the model this endpoint accepts.
            call_kwargs = kwargs
            override_model = self.endpoint_models[index]
            if override_model and kwargs.get("model") != override_model:
                call_kwargs = dict(kwargs)
                call_kwargs["model"] = override_model
            try:
                return client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) in {401, 403}:
                    if "remote api access" in str(exc).lower():
                        raise
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
            # Standard retryable/failover codes plus the Cloudflare origin-error
            # family (520-527, 530). Reseller gateways sit behind Cloudflare and
            # emit 524 (origin read timeout) etc. when their upstream stalls;
            # treating these as failover-able lets the pool fall through to a
            # healthy fallback endpoint instead of exhausting retries on the dead
            # primary (which never marks dead, so retries would loop forever).
            return int(status_code) in {
                401, 403, 408, 409, 429, 500, 502, 503, 504,
                520, 521, 522, 523, 524, 525, 526, 527, 530,
            }
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def build_client(config: dict[str, Any], paths: Paths) -> Any:
    """THE client/pool constructor for every entry point except `pipeline.main`.

    `trial`, `screenplay` and `package` each grew their own byte-identical copy of
    this (screenplay's even said "copied shape from trial.py"), so a timeout or
    header fix landed in one and not the others. `pipeline.main` keeps its own
    version on purpose: it is a superset that also resolves per-endpoint models via
    `configured_api_endpoints_with_models`, which the standalone tools don't use.

    Returns a bare `OpenAI` when there is exactly one endpoint — a pool of one adds
    rotation bookkeeping that can never fire.
    """
    from openai import OpenAI
    import httpx

    from engine.config import configured_api_endpoints

    api_endpoints, primary_endpoint_count = configured_api_endpoints(config)
    if not api_endpoints:
        raise RuntimeError(
            "Missing API key: set api.api_key, api.api_keys, or api.api_key_groups in config.yaml")
    connect_timeout = int(config["api"].get("client_connect_timeout", 15))
    client_read_timeout = int(config["api"].get("client_read_timeout", 180))
    httpx_timeout = httpx.Timeout(
        connect=connect_timeout,
        read=client_read_timeout,
        write=connect_timeout,
        pool=connect_timeout,
    )
    default_headers = {}
    user_agent = str(config["api"].get("user_agent", "")).strip()
    if user_agent:
        default_headers["User-Agent"] = user_agent
    clients = [
        OpenAI(base_url=base_url, api_key=api_key, timeout=httpx_timeout,
               default_headers=default_headers or None)
        for base_url, api_key in api_endpoints
    ]
    if len(clients) == 1:
        return clients[0]
    return LLMClientPool(clients, primary_endpoint_count, endpoints=api_endpoints,
                         log_fn=lambda msg: log(paths, msg))


def _resolve_thinking_param(
    api_section: dict[str, Any],
    *,
    mode_key: str = "thinking_mode",
    disabled_key: str = "thinking_disabled",
    budget_key: str = "thinking_budget_tokens",
    default_disabled: bool = True,
) -> dict[str, Any] | None:
    """Build the ``extra_body["thinking"]`` dict from config.

    Supports three modes (set via *mode_key*, e.g. ``api.thinking_mode``):
      - ``disabled`` — send ``{"type": "disabled"}`` (suppresses reasoning).
      - ``auto``     — omit the param entirely (provider decides).
      - ``enabled``  — send ``{"type": "enabled", "budget_tokens": N}``.

    Falls back to the legacy boolean *disabled_key* when *mode_key* is absent:
      - ``thinking_disabled: true``  → disabled
      - ``thinking_disabled: false`` → auto

    Returns the dict to put in ``extra_body["thinking"]``, or ``None`` to omit.
    """
    raw_mode = str(api_section.get(mode_key, "") or "").strip().lower()
    if raw_mode == "disabled":
        return {"type": "disabled"}
    if raw_mode == "auto":
        return None
    if raw_mode == "enabled":
        budget = api_section.get(budget_key)
        if budget is not None and str(budget).strip():
            try:
                budget_int = int(budget)
            except (TypeError, ValueError):
                budget_int = 0
            if budget_int > 0:
                return {"type": "enabled", "budget_tokens": budget_int}
        return {"type": "enabled"}

    # Legacy fallback: thinking_disabled boolean.
    disabled = api_section.get(disabled_key, default_disabled)
    if isinstance(disabled, str):
        disabled = disabled.strip().lower() not in {"false", "0", "no", "off", ""}
    return {"type": "disabled"} if disabled else None


def _resolve_reasoning_effort(api_section: dict[str, Any], *, key: str = "reasoning_effort") -> str | None:
    """Read the OpenAI-style ``reasoning_effort`` knob from config, or None to omit.

    Separate from ``thinking`` (Anthropic/豆包 style): some gateways only honour
    one of the two. Live evidence — littlesheep's gemini-2.5-pro reasons for well
    past nginx's ~70s proxy timeout and 504s before the first byte; neither
    ``thinking:{"type":"disabled"}`` nor omitting the param helps, only
    ``reasoning_effort: none`` does.

    Unknown values are passed through (providers keep adding tiers); empty/absent
    returns None so the param is never sent by default.
    """
    raw = str(api_section.get(key, "") or "").strip().lower()
    return raw or None


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

def _escape_inner_string_quotes_unchecked(text: str) -> str:
    """Core of the inner-quote escaper, WITHOUT the final parse gate.

    Walks the text tracking string state; a `"` inside a string is a genuine
    closing delimiter only when the next non-whitespace char is JSON structure
    (`,` `:` `}` `]`) or end-of-text, otherwise it is content and is escaped to
    `\\"`. Used both directly (last-resort, combined with truncation repair) and
    by `_escape_inner_string_quotes`, which adds a parse check.
    """
    out: list[str] = []
    n = len(text)
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if esc:
            out.append(c)
            esc = False
            continue
        if c == "\\":
            out.append(c)
            esc = True
            continue
        if c == '"':
            if not in_str:
                in_str = True
                out.append(c)
                continue
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            nxt = text[j] if j < n else ""
            if nxt in ",:}]" or nxt == "":
                in_str = False
                out.append(c)
            else:
                out.append('\\"')
            continue
        out.append(c)
    return "".join(out)


def _escape_inner_string_quotes(text: str) -> str | None:
    """Repair JSON whose string VALUES contain unescaped double quotes.

    The model frequently emits values like `"state": "...以"团建素描顾问"名义..."`
    — raw double quotes (often the CJK-content kind, sometimes ASCII) sitting
    inside a string value. Standard json.loads aborts at the first such quote,
    and neither the `\\{.*\\}` slice nor `_repair_truncated_json` can recover it
    because both assume well-formed string boundaries.

    Returns the repaired string if it then parses, else None. This is a
    last-resort fixer, tried after the cheaper paths in safe_json_loads.
    """
    if '"' not in text:
        return None
    repaired = _escape_inner_string_quotes_unchecked(text)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
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
    # Unescaped quotes inside string values (model emits 以"x"名义 verbatim).
    # Try this before the truncation repair, since a body full of inner quotes
    # makes _repair_truncated_json's brace-balancing read structure wrong.
    quote_fixed = _escape_inner_string_quotes(cleaned)
    if quote_fixed is not None:
        return json.loads(quote_fixed)
    repaired = _repair_truncated_json(cleaned)
    if repaired:
        return json.loads(repaired)
    # Last resort: a stream that is BOTH truncated AND carries inner quotes.
    # Escape the inner quotes first (without the parse gate), then balance braces.
    half_fixed = _escape_inner_string_quotes_unchecked(cleaned)
    repaired2 = _repair_truncated_json(half_fixed)
    if repaired2:
        try:
            return json.loads(repaired2)
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError(f"Could not recover JSON. Preview: {cleaned[:300]!r}", cleaned, 0)

JSON_REPAIR_SYSTEM = """你负责修复 LLM 返回的格式错误的 JSON。
只输出合法 JSON，不要添加任何解释。保留原本的字段与取值。"""

JSON_OUTPUT_CONTRACT = """输出约定：
- 只输出一个合法 JSON 对象；首尾必须是 `{` 和 `}`，不要代码围栏、解释或前言。
- 严格保留 schema 的英文键名和层级；字符串与键名用双引号，内部引号须转义。
- 使用标准 JSON 类型；禁止注释、尾随逗号、NaN、Infinity 和 Python 布尔值。
- 信息不足时保留全部必需字段，填保守且类型正确的值，不要猜测事实。"""

GLOBAL_PROMPT_HYGIENE = """## 全局提示词纪律（公共执行协议）
- 只完成当前任务并遵守其输出协议；除非任务要求，不输出推理、解释、前言、道歉或元评论。
- 冲突优先级：任务输出协议 > 用户明示硬约束 > 已确认事实（canon）> 当前输入材料 > 风格偏好；低优先级内容不得覆盖高优先级内容。
- 提交前静默检查格式、字段与硬约束，只输出最终结果。"""

JSON_PROMPT_HYGIENE = """## JSON 任务额外纪律（JSON 协议）
只输出一个合法 JSON 对象；保留 schema 键名、层级和字段类型，信息不足时不编造事实。"""

PLAN_PROMPT_HYGIENE = """## 规划/仲裁任务额外纪律（规划职责）
- 把目标与约束转成可执行的因果链：行动或选择 → 阻力 → 后果；不用“加强冲突”等抽象占位语。
- 只在输入允许的范围内设计，不改写 canon；每个新增设定都要服务当前任务。"""

WRITE_PROMPT_HYGIENE = """## 写作职责
- 把任务要求落到页面上的动作、对话、选择与后果，保持 canon、人物动机、视角和叙事声音一致。
- 修订只改任务指定范围；不遗漏既有事实，不擅自新增人物、能力、因果或章末状态。"""

REVIEW_PROMPT_HYGIENE = """## 评审职责
- 只按本任务量表和可见文本判断，不替作者补意图；结论须有文本证据，不因单项优点掩盖缺陷。
- 分数、胜负与建议必须相互一致；建议指出缺什么、在何处、怎样改，不用空泛评价。"""

MEMORY_PROMPT_HYGIENE = """## 抽取职责
- 只记录输入可证实的事实与状态变化；不得把例子、猜测、建议或缺省字段变成事实。
- 保留专名、数值、模态、因果和状态，去重后按 schema 输出；不确定则留空或标为未知。"""


def json_prompt(user: str) -> str:
    body = user.rstrip()
    if "## 强制 JSON 输出格式" in body:
        return body
    return body + "\n\n## 强制 JSON 输出格式\n" + JSON_OUTPUT_CONTRACT


def _enhance_system_prompt(
    system: str,
    config: dict[str, Any],
    *,
    tag: str,
    wants_json: bool,
    has_json_contract: bool = False,
) -> str:
    api = config.get("api", {})
    novel = config.get("novel", {})
    if not bool(api.get("prompt_enhancement_enabled", novel.get("prompt_enhancement_enabled", True))):
        return system
    if "## 全局提示词纪律（适用于本次调用）" in system:
        return system

    role = prompt_role_for_tag(tag)
    blocks = [GLOBAL_PROMPT_HYGIENE]
    # `json_prompt()` already carries the fuller protocol in the user message.
    # Repeating it in the system message wastes tokens and can create two subtly
    # different sources of truth.  `json_mode=True` callers without that marker
    # still receive this compact safety net.
    if wants_json and not has_json_contract:
        blocks.append(JSON_PROMPT_HYGIENE)
    if role == "planning":
        blocks.append(PLAN_PROMPT_HYGIENE)
    elif role == "writing":
        blocks.append(WRITE_PROMPT_HYGIENE)
    elif role == "review":
        blocks.append(REVIEW_PROMPT_HYGIENE)
    elif role == "extraction":
        blocks.append(MEMORY_PROMPT_HYGIENE)
    # Shared policy goes first; the task-specific contract stays closest to the
    # model's response and therefore wins the recency competition inside the
    # system message.  This also makes the documented L0 -> L1 -> L2 layering
    # match the bytes actually sent to providers.
    return "\n\n".join(blocks + [system.rstrip()])

def emergency_truncate(user_text: str, max_chars: int) -> str:
    if len(user_text) <= max_chars:
        return user_text
    sections = re.split(r"(?=^## )", user_text, flags=re.MULTILINE)
    priority_keywords = ["创作纲要", "当前状态", "选定大纲", "仲裁约束"]
    high = []
    medium = []
    low = []
    for section in sections:
        if any(kw in section[:80] for kw in priority_keywords):
            high.append(section)
        elif any(kw in section[:80] for kw in ["人物", "世界设定", "卷纲", "伏线"]):
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
    tag: str = "",
) -> str:
    api = config["api"]
    # Role-based routing: probe the client for role-specific pools attached by
    # pipeline.py. Check each role in priority order (planning > writing >
    # extraction > review). First match wins. When no role pool matches, fall
    # through to the primary model. Attribute probing keeps call_llm's
    # signature unchanged across its ~25 call sites.
    call_client = client
    model_name = api["model"]
    _active_role_prefix: str = ""
    prompt_role = prompt_role_for_tag(tag)
    for _role, _pool_attr, _api_attr, _model_key in _ROLE_ROUTING:
        _pool = getattr(client, _pool_attr, None)
        _rapi = getattr(client, _api_attr, None)
        if _pool and _rapi and prompt_role == _role:
            call_client = _pool
            model_name = str(_rapi[_model_key])
            _active_role_prefix = _model_key.replace("_model", "_")
            break
    use_reviewer = _active_role_prefix == "review_"
    context_window = int(api.get("context_window", 1000000))
    max_input_chars = int(context_window * 1.8)
    # Prepend cacheable_prefix verbatim at the very top of user message so that
    # repeated invocations across chapters share an identical prefix and the
    # provider's prefix cache can hit. The prefix should contain ONLY content
    # that does not change call-to-call within a window (creative brief, bible,
    # characters, voice anchors).
    if cacheable_prefix:
        user = cacheable_prefix.rstrip() + "\n\n" + user
    has_json_contract = "## 强制 JSON 输出格式" in user
    wants_json = json_mode if json_mode is not None else has_json_contract
    system = _enhance_system_prompt(
        system, config, tag=tag, wants_json=wants_json,
        has_json_contract=has_json_contract,
    )
    total_chars = len(system) + len(user)
    if total_chars > max_input_chars:
        truncated_to = max_input_chars - len(system) - 1000
        log(paths, f"[WARN] emergency_truncate fired: prompt {total_chars} chars > max {max_input_chars}; truncating user to {truncated_to} chars (tag={tag})")
        user = emergency_truncate(user, truncated_to)
    use_response_format = wants_json and bool(api.get("json_response_format", True))
    max_attempts = int(api.get("max_attempts", 6))
    salvaged_any = False
    mt_escalate: int | None = None  # adaptive max_tokens bump after finish=length empty responses
    for attempt in range(max_attempts):
        started = time.perf_counter()
        reasoning_total = 0
        finish_reason = None
        try:
            request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": _effective_max_tokens(api, mt_escalate or max_tokens),
                "temperature": float(api["temperature"]) if temperature is None else temperature,
            }
            if _active_role_prefix:
                _role_temp_key = f"{_active_role_prefix}temperature"
                _role_temp = str(api.get(_role_temp_key, "")).strip()
                if _role_temp:
                    try:
                        request["temperature"] = float(_role_temp)
                    except (TypeError, ValueError):
                        pass
            extra_body: dict[str, Any] = {}
            if not _active_role_prefix and api.get("group"):
                extra_body["group"] = str(api.get("group"))
            if _active_role_prefix:
                thinking_param = _resolve_thinking_param(
                    api,
                    mode_key=f"{_active_role_prefix}thinking_mode",
                    disabled_key=f"{_active_role_prefix}thinking_disabled",
                    budget_key=f"{_active_role_prefix}thinking_budget_tokens",
                    default_disabled=(_active_role_prefix != "review_"),
                )
                effort = _resolve_reasoning_effort(
                    api, key=f"{_active_role_prefix}reasoning_effort"
                )
            else:
                thinking_param = _resolve_thinking_param(api)
                effort = _resolve_reasoning_effort(api)
            if thinking_param is not None:
                extra_body["thinking"] = thinking_param
            if effort is not None:
                # Goes through extra_body (not a named kwarg) so it works on SDK
                # versions that predate the param; extra_body is merged into the
                # request body at top level.
                extra_body["reasoning_effort"] = effort
            # Reasoning-model max_tokens floor. Reasoning models spend part of the
            # token budget on a hidden chain of thought BEFORE any answer content,
            # so a tiny max_tokens (e.g. a 2000-token structural_diagnose) is
            # entirely consumed by reasoning and returns empty content with
            # finish_reason=length — retrying the same cap fails identically.
            # Applied UNCONDITIONALLY, NOT gated on our thinking flag: gateway
            # proxies (e.g. ldapi) routinely reason even when we send
            # thinking:{type:disabled}, so keying off our config misses them (live
            # evidence: a disabled-thinking call still streamed 2002 reasoning
            # chunks and returned empty). max_tokens is an UPPER bound, so this is
            # cost-neutral for genuinely non-reasoning models (they still stop when
            # done) and it only ever raises, never lowers — deliberate length caps
            # (writer/refine at >=12000) sit above the floor and are untouched.
            _rfloor = int(api.get("reasoning_min_max_tokens", 8000) or 0)
            if _rfloor > 0 and int(request["max_tokens"]) < _rfloor:
                request["max_tokens"] = _effective_max_tokens(api, _rfloor)
            _top_p = api.get("top_p")
            if _top_p is not None and str(_top_p).strip() and float(_top_p) != 1.0:
                request["top_p"] = float(_top_p)
            _freq = api.get("frequency_penalty")
            if _freq is not None and str(_freq).strip() and float(_freq) != 0:
                request["frequency_penalty"] = float(_freq)
            _pres = api.get("presence_penalty")
            if _pres is not None and str(_pres).strip() and float(_pres) != 0:
                request["presence_penalty"] = float(_pres)
            if extra_body:
                request["extra_body"] = extra_body
            if use_response_format:
                request["response_format"] = {"type": "json_object"}
            stream = bool(api.get("stream", False))
            if stream:
                request["stream"] = True
            _throttle_request_start(paths, api)
            if hasattr(call_client, "create_completion"):
                resp = call_client.create_completion(**request)
            else:
                resp = call_client.chat.completions.create(**request)
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
                reasoning_total = sum(len(p) for p in reasoning_parts)
                elapsed = time.perf_counter() - started
                min_salvage_chars = int(api.get("stream_salvage_min_chars", 800))
                if stream_timeout_reason and len(content.strip()) >= min_salvage_chars:
                    log(
                        paths,
                        f"Stream cut off but salvaging partial content "
                        f"attempt={attempt + 1}/{max_attempts} chunks={chunk_count} "
                        f"content_chars={len(content)} elapsed={elapsed:.1f}s reason={stream_timeout_reason}",
                    )
                    stream_timeout_reason = None
                    salvaged_any = True
                if stream_timeout_reason:
                    raise TimeoutError(stream_timeout_reason)
                if not content.strip() and reasoning_parts:
                    log(
                        paths,
                        "LLM content empty but reasoning_content present, using reasoning fallback "
                        f"attempt={attempt + 1}/{max_attempts} chunks={chunk_count} finish={finish_reason} "
                        f"reasoning_chars={sum(len(p) for p in reasoning_parts)} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
                    content = "".join(reasoning_parts)
                elif not content.strip():
                    log(
                        paths,
                        "LLM returned empty streamed response "
                        f"attempt={attempt + 1}/{max_attempts} chunks={chunk_count} finish={finish_reason} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
            else:
                msg = resp.choices[0].message
                content = msg.content or ""
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                reasoning = getattr(msg, "reasoning_content", None) or ""
                reasoning_total = len(reasoning)
                elapsed = time.perf_counter() - started
                if not content.strip() and reasoning.strip():
                    log(
                        paths,
                        "LLM content empty but reasoning_content present (non-stream), using reasoning fallback "
                        f"attempt={attempt + 1}/{max_attempts} "
                        f"reasoning_chars={len(reasoning)} elapsed={elapsed:.1f}s",
                    )
                    content = reasoning
            if not content.strip():
                wait = _backoff_wait(attempt)
                # finish=length + empty content means the model (a reasoner behind
                # this proxy) burned the whole budget before emitting any answer.
                # Retrying the identical cap is guaranteed to fail the same way —
                # widen max_tokens for the next attempt (builds on the value
                # actually used this attempt, so it compounds past the B-floor).
                if str(finish_reason) == "length":
                    _base = int(request["max_tokens"])
                    _factor = float(api.get("length_empty_retry_factor", 2.0) or 2.0)
                    _cap = int(api.get("length_empty_retry_cap", 32000) or 0)
                    _bumped = int(_base * _factor)
                    if _cap > 0:
                        _bumped = min(_bumped, _cap)
                    if _bumped > _base:
                        mt_escalate = _bumped
                        log(
                            paths,
                            f"finish=length with empty content; escalating max_tokens "
                            f"{_base}->{_bumped} for next attempt (reasoning likely consumed the budget)",
                        )
                log(
                    paths,
                    f"LLM returned empty response attempt={attempt + 1}/{max_attempts} wait={wait:.1f}s "
                    f"stream={stream} elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                )
                time.sleep(wait)
                continue
            if _looks_like_refusal(content):
                wait = _backoff_wait(attempt)
                log(
                    paths,
                    f"LLM returned refusal-like response attempt={attempt + 1}/{max_attempts} wait={wait:.1f}s "
                    f"len={len(content.strip())} preview={content.strip()[:120]!r}",
                )
                time.sleep(wait)
                continue
            _record_llm_call(
                paths, api,
                tag=tag, model=request["model"], stream=stream, json_mode=wants_json,
                attempt=attempt, prompt_chars=total_chars, output_chars=len(content),
                elapsed=elapsed, salvaged=salvaged_any, ok=True,
                reasoning_chars=reasoning_total,
            )
            return content
        except Exception as exc:
            if use_response_format and _looks_like_response_format_error(exc):
                use_response_format = False
                log(paths, f"JSON response_format unsupported, retrying without it: {exc}")
                continue
            if _looks_like_nonretryable_block(exc):
                elapsed = time.perf_counter() - started
                log(paths, f"LLM call blocked by provider; not retrying elapsed={elapsed:.1f}s error={exc}")
                _record_llm_call(
                    paths, api,
                    tag=tag, model=model_name, stream=bool(api.get("stream", False)),
                    json_mode=wants_json, attempt=attempt, prompt_chars=total_chars,
                    output_chars=0, elapsed=elapsed, salvaged=salvaged_any, ok=False,
                    error=str(exc),
                )
                raise RuntimeError(f"LLM provider blocked the request: {exc}") from exc
            wait = _backoff_wait(attempt, exc)
            elapsed = time.perf_counter() - started
            log(paths, f"LLM call failed attempt={attempt + 1}/{max_attempts} wait={wait:.1f}s elapsed={elapsed:.1f}s error={exc}")
            time.sleep(wait)
    _record_llm_call(
        paths, api,
        tag=tag, model=model_name, stream=bool(api.get("stream", False)),
        json_mode=wants_json, attempt=max_attempts - 1, prompt_chars=total_chars,
        output_chars=0, elapsed=0.0, salvaged=salvaged_any, ok=False,
        error=f"failed after {max_attempts} attempts",
    )
    raise RuntimeError(f"LLM call failed after {max_attempts} attempts")

def _looks_like_response_format_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return (
        status_code in {400, 404, 422}
        and "response_format" in text
        or "response_format" in text
        and any(word in text for word in ["unsupported", "not support", "unknown", "invalid", "extra"])
    )

def _looks_like_nonretryable_block(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status_code == 401 and "remote api access" in text:
        return False
    return (
        status_code in {401, 403}
        or "request was blocked" in text
        or "permissiondenied" in text
        or "permission denied" in text
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
    repair_prompt = f"""将下面这段格式错误的 JSON 修复为一个合法的 JSON 对象。

## 格式错误的 JSON
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
            tag="json_repair",
        )
        return safe_json_loads(repaired)
    except Exception as exc:
        log(paths, f"JSON repair failed: {exc}")
        if fallback is not None:
            return fallback
        raise
