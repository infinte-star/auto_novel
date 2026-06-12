from __future__ import annotations

import json
import queue
import random
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from config import Paths, log, normalize_text

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

# Tags whose LLM calls should be routed to the separate reviewer model
# (main writer = primary model, reviewer = GPT) when a reviewer pool is
# configured and attached to the client. Every other tag stays on the primary
# model. See call_llm's reviewer-routing block and pipeline.py's reviewer pool
# construction. When no reviewer pool is attached, this set is inert.
REVIEW_TAGS = frozenset({
    "review",
    "cold_reader",
    "stage_review",
    "pack_review",
    "macro_progress",
    "plan_review_fused",
    "plan_review_axis",
})

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
- 只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
- 第一个非空白字符必须是 `{`，最后一个非空白字符必须是 `}`。
- 不要使用 markdown 标题、项目符号、代码围栏、解释或开场白。
- 每一个键名和字符串值都用英文双引号包裹。
- 转义字符串值内部的引号。
- 不要使用末尾逗号、注释、NaN、Infinity，或 Python 风格的布尔值。
- 严格保留所要求的 schema 键名，键名一律用英文原样输出，不得翻译。
- 若不确定，仍要返回所要求的 schema，取保守值，字符串用简短中文。"""

GLOBAL_PROMPT_HYGIENE = """## 全局提示词纪律（适用于本次调用）
- 严格服从本任务要求的输出格式；除非任务明确要求，不要输出思考过程、解释、道歉、前言或元评论。
- 优先使用上文给出的事实、约束、schema、章节状态；遇到冲突时，以更具体、更近期、更硬性的约束为准。
- 输出必须具体可执行：用人物行动、场景变化、因果桥梁、资源代价、证据或字段值回答问题，避免空泛口号。
- 不把缺陷留给下游修订；在本次输出内先完成自检，再给最终结果。"""

JSON_PROMPT_HYGIENE = """## JSON 任务额外纪律
- 只生成一个 JSON 对象，不要在 JSON 前后添加任何文本。
- 保留 schema 中要求的键名和层级；缺失信息时填保守、简短、可解析的值，不要删键。
- 数字字段输出数字，布尔字段输出 true/false，数组字段输出数组；不要输出 NaN、Infinity、注释或尾随逗号。"""

PLAN_PROMPT_HYGIENE = """## 规划/仲裁任务额外纪律
- 大纲必须能直接驱动首稿写作：每个 beat 都要有可见行动、阻力、信息增量或资源代价。
- 不接受“加强冲突”“提升节奏”这类抽象修正；必须给出具体场景任务、人物选择和章末问题。
- 选择或合并方案时，优先解决近期低分原因、重复场景骨架、沉默伏线和兑现拖欠。"""

WRITE_PROMPT_HYGIENE = """## 写作/修订任务额外纪律
- 首稿就按终审标准执行：剧情推进、主角能动性、压力、兑现、新鲜度、文笔和连续性都必须同时过线。
- 所有大纲节拍必须落到页面上的动作、对话、后果或细节；不要用总结性旁白替代戏剧化呈现。
- 修订时优先修结构、因果、节奏和钩子，再润色措辞；不得引入新的事实矛盾。"""

REVIEW_PROMPT_HYGIENE = """## 审校/评分任务额外纪律
- 分维度独立判断，不让单一优点掩盖追读、兑现、新鲜度、文笔或连续性的短板。
- 所有扣分、风险和修改建议都要可执行；指出页面上缺什么、应补在哪里、下一次如何避免。
- 不要分数通胀；若关键节拍缺失、兑现空洞、重复或连续性风险明显，总分必须受限。"""

MEMORY_PROMPT_HYGIENE = """## 记忆/抽取/压缩任务额外纪律
- 保留稳定事实、人物目标、资源状态、因果链接、伏线状态和不可违背约束；删掉套话和风格性复述。
- 新增事实必须来自输入文本或明确任务，不要编造未出现的人物、地点、物品或因果。
- 输出要便于后续生成直接引用：短句、具体、去重、按状态变化组织。"""


def json_prompt(user: str) -> str:
    return user.rstrip() + "\n\n## 强制 JSON 输出格式\n" + JSON_OUTPUT_CONTRACT


def _enhance_system_prompt(system: str, config: dict[str, Any], *, tag: str, wants_json: bool) -> str:
    api = config.get("api", {})
    novel = config.get("novel", {})
    if not bool(api.get("prompt_enhancement_enabled", novel.get("prompt_enhancement_enabled", True))):
        return system
    if "## 全局提示词纪律（适用于本次调用）" in system:
        return system

    tag_l = (tag or "").lower()
    blocks = [GLOBAL_PROMPT_HYGIENE]
    if wants_json:
        blocks.append(JSON_PROMPT_HYGIENE)
    if tag_l.startswith("plan_") or tag_l in {"replan", "macro_progress"}:
        blocks.append(PLAN_PROMPT_HYGIENE)
    if tag_l in {"write", "revise"} or "write" in tag_l or "revise" in tag_l:
        blocks.append(WRITE_PROMPT_HYGIENE)
    if "review" in tag_l or tag_l in {"cold_reader", "stage_review", "pack_review"}:
        blocks.append(REVIEW_PROMPT_HYGIENE)
    if tag_l in {
        "bootstrap",
        "creative_boost",
        "memory_compress",
        "extract",
        "state_update",
        "state_sections",
        "voice_anchor",
        "voices_table",
    }:
        blocks.append(MEMORY_PROMPT_HYGIENE)
    return system.rstrip() + "\n\n" + "\n\n".join(blocks)

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
    # Reviewer routing: when a reviewer pool is attached to the client (see
    # pipeline.py) and this call's tag is a scoring/review tag, send it to the
    # separate reviewer model+endpoint instead of the primary model. Attribute
    # probing keeps call_llm's signature unchanged across its ~25 call sites.
    # `review_api` is the same config["api"] dict, read for the review_* keys.
    review_pool = getattr(client, "review_pool", None)
    review_api = getattr(client, "review_api", None)
    use_reviewer = bool(review_pool) and bool(review_api) and tag in REVIEW_TAGS
    call_client = review_pool if use_reviewer else client
    model_name = str(review_api["review_model"]) if use_reviewer else api["model"]
    context_window = int(api.get("context_window", 1000000))
    max_input_chars = int(context_window * 1.8)
    # Prepend cacheable_prefix verbatim at the very top of user message so that
    # repeated invocations across chapters share an identical prefix and the
    # provider's prefix cache can hit. The prefix should contain ONLY content
    # that does not change call-to-call within a window (creative brief, bible,
    # characters, voice anchors).
    if cacheable_prefix:
        user = cacheable_prefix.rstrip() + "\n\n" + user
    wants_json = json_mode if json_mode is not None else "强制 JSON 输出格式" in user
    system = _enhance_system_prompt(system, config, tag=tag, wants_json=wants_json)
    total_chars = len(system) + len(user)
    if total_chars > max_input_chars:
        truncated_to = max_input_chars - len(system) - 1000
        log(paths, f"[WARN] emergency_truncate fired: prompt {total_chars} chars > max {max_input_chars}; truncating user to {truncated_to} chars (tag={tag})")
        user = emergency_truncate(user, truncated_to)
    use_response_format = wants_json and bool(api.get("json_response_format", True))
    max_attempts = int(api.get("max_attempts", 6))
    salvaged_any = False
    for attempt in range(max_attempts):
        started = time.perf_counter()
        try:
            request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": _effective_max_tokens(api, max_tokens),
                "temperature": float(api["temperature"]) if temperature is None else temperature,
            }
            if use_reviewer and str(review_api.get("review_temperature", "")).strip() != "":
                try:
                    request["temperature"] = float(review_api["review_temperature"])
                except (TypeError, ValueError):
                    pass
            extra_body: dict[str, Any] = {}
            if not use_reviewer and api.get("group"):
                extra_body["group"] = str(api.get("group"))
            # Some providers (e.g. mimo-v2.5-pro) default to an unbounded
            # "thinking" / reasoning phase. On large write prompts this can emit
            # tens of thousands of reasoning_content chars and NEVER produce
            # `content`, so the stream runs until stream_timeout (1800s) and the
            # whole pipeline stalls retrying forever. Disabling thinking makes
            # content stream within seconds. `api.thinking_disabled` (default
            # true) sends the documented `extra_body={"thinking": {"type":
            # "disabled"}}`; set false to restore provider-default reasoning.
            # Reviewer (GPT) path reads review_thinking_disabled (default False),
            # since GPT reasoning models reject the Anthropic-style thinking flag.
            if use_reviewer:
                thinking_disabled = review_api.get("review_thinking_disabled", False)
            else:
                thinking_disabled = api.get("thinking_disabled", True)
            if isinstance(thinking_disabled, str):
                thinking_disabled = thinking_disabled.strip().lower() not in {"false", "0", "no", "off", ""}
            if thinking_disabled:
                extra_body["thinking"] = {"type": "disabled"}
            if api.get("top_p") is not None:
                request["top_p"] = float(api.get("top_p"))
            if api.get("frequency_penalty") is not None:
                request["frequency_penalty"] = float(api.get("frequency_penalty"))
            if api.get("presence_penalty") is not None:
                request["presence_penalty"] = float(api.get("presence_penalty"))
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
                content = resp.choices[0].message.content or ""
                elapsed = time.perf_counter() - started
            if not content.strip():
                wait = _backoff_wait(attempt)
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
