"""v3 write module — merged from v2/write.py + writing.py.

Self-contained prose doctrine, one-call write+delta, and persistence.  All genre
profiles, shared constants, prompt builders, and chapter-save helpers that v2/write
previously imported from the top-level ``writing`` module are inlined here so the
engine package has no upward dependency on legacy modules.

The one-call design is unchanged from v2 (REDESIGN_V2 §3.3): the write call
returns prose first, a sentinel line, then a bare JSON ``ChapterDelta``.  Four
load-bearing decisions carry over:

1. The prose doctrine (``GENRE_PROFILES``, ``ANTI_FRAGMENT_BAN``, aesthetic
   presets, ``_build_write_system``) is v1's, unchanged — rewriting style teaching
   would confound any A/B with a prompt-library variable.
2. The v1 output section is REPLACED, not appended to — two contradictory
   instructions would silently revert to v1's cost profile.
3. Prose first, JSON last; a parse failure loses the delta, never the prose.
4. The acceptance checklist names the literal fragments the gate greps for.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from engine.checkpoint import load_checkpoint
from engine.config import (
    Paths,
    append_text,
    chapter_path,
    count_chars,
    log,
    normalize_chapter,
    read_text,
    safe_score,
    write_text,
)
from engine.llm import call_llm, load_json_with_repair, safe_json_loads
from engine.quality import plan_score
from engine.store import db_event, db_lock, store_causal_links, upsert_reader_promise
from engine.types import ChapterDelta, StoryState

if TYPE_CHECKING:
    from openai import OpenAI


class WriteError(RuntimeError):
    """The write call produced nothing usable as a chapter.

    Raised rather than returned so a refusal cannot be mistaken for a short
    chapter. `writing.save_chapter` refuses below 500 chars anyway; catching it
    here means `run.py` sees it as a retryable draft failure instead of a crash
    three steps later.
    """


# ---------------------------------------------------------------------------
# The two-part response format
# ---------------------------------------------------------------------------

# What the model is told to emit between the prose and the JSON. Chosen to be
# something no Chinese web-novel chapter would contain by accident: '=' runs are
# not Chinese punctuation, and the phrase is a machine label, not narration.
DELTA_SENTINEL = "===状态增量==="

# Tolerant on the way in, strict on the way out — the standard asymmetry. Accepts
# a different number of '=', surrounding spaces, the English spelling, and the
# list/quote/heading prefixes models sprinkle on lines they think are headers.
_SENTINEL_RE = re.compile(
    r"(?:\A|\n)[ \t>*#\-]*=+\s*(?:状态增量|状態增量|STATE[ _]?DELTA|DELTA)\s*=+[ \t]*(?=\n|\Z)"
)

# The five keys `ChapterDelta` reads. Used to tell a real delta from a JSON
# object that merely happens to be at the end of the response.
DELTA_KEYS: frozenset[str] = frozenset(
    ("events", "entities", "threads", "protagonist_state",
     "next_12_directions", "next_directions"))

# The schema shown to the model. Deliberately v1's `EXTRACT_SYSTEM` spellings
# (`summary`, `state_patch`, thread `status` vocabulary) because
# `canon.apply_delta` hands the result straight to `writing.update_structured_state`
# — the persistence layer is v1's and so the key names must be.
#
# What is NOT here is the point: v1 also asks for `metrics`, `memory_updates`,
# `causal_links`, `relationship_changes`, `info_revelations`,
# `dialogue_fingerprints` and `title`. v2 drops them. `memory_updates` in
# particular is what appends to bible/characters every chapter — dropping it is
# what keeps `canon`'s stable prefix byte-identical for the whole book, which is
# the prompt-cache saving the redesign is built on.
DELTA_SCHEMA = """{
  "events": [{"type":"plot|world|character|thread|item|battle|relationship","summary":"本章真正发生的一件事，一句话","effects":[]}],
  "entities": [{"entity_type":"character|force|place|item|rule","name":"名字","state_patch":{"改变的字段":"新值"}}],
  "threads": [{"id":"稳定id","description":"这条线索是什么","status":"open|advanced|recovered|dropped","thread_type":"plot|reader_promise|character_arc|world_rule|relationship","introduced_chapter":1,"due_chapter":20,"priority":5}],
  "protagonist_state": {"目标":"...","资源":"...","恐惧":"...","秘密":"...","持续压力":"...","待决断":"..."},
  "next_12_directions": ["后续章节必须发生的具体事，每条一句，10-12 条"]
}"""

V2_OUTPUT_SECTION = """## 输出要求（两段式，先后顺序不可颠倒）

### 第一段：章节正文
- 约{chapter_words}个中文字符。
- 第一行固定格式：第{chapter_num}章 {title}
- 执行本章卡片与全部约束条件；卡片里的【转折】与【收尾钩子】必须在页面上实演出来，不能只被提及。
- 正文里严禁出现"写前自我审查"、"Pre-writing Self-Review"、"分析"、"reasoning"、`<analysis>`、`<thinking>`、代码围栏、清单或任何解释性文字。

### 第二段：状态增量
正文写完之后另起一行，先原样照抄这一行分隔符（前后不加任何字符）：
{sentinel}
然后输出**恰好一个** JSON 对象，不要代码围栏、不要注释、不要任何多余文字：
{schema}

两段式的规则：
- 分隔符之前的一切都算章节正文；之后的一切都算状态数据。
- **必须先把正文写完整、把结尾钩子写足，再写 JSON**；不要为了赶去写 JSON 而草草收尾。
- JSON 只记录本章真正产生的持久变化；某一项本章没有变化就给空数组 []。
- 第二段不计入正文字数。"""


def v2_output_section(chapter_words: int, chapter_num: int, title: str) -> str:
    return V2_OUTPUT_SECTION.format(
        chapter_words=chapter_words, chapter_num=chapter_num, title=title,
        sentinel=DELTA_SENTINEL, schema=DELTA_SCHEMA)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def clean_title(card: dict[str, Any] | None, chapter_num: int) -> str:
    """The card's title with any 「第N章」 prefix stripped — same rule as v1.

    The output section formats the first line as 「第{n}章 {title}」, so a title
    that already carries the prefix doubles it.
    """
    raw = str((card or {}).get("title") or "").strip()
    raw = re.sub(r"^\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*章\s*[:：、\-—\s]*", "", raw)
    return raw.strip() or f"第{chapter_num}章"


def build_system(config: dict[str, Any], chapter_num: int, title: str) -> str:
    """The writer system prompt with ONE section swapped. See docstring rule 2."""
    novel = config["novel"]
    preset = str(novel.get("style_preset", "history"))
    chapter_words = int(novel.get("chapter_words", 4000) or 4000)
    system = _build_write_system(
        preset,
        chapter_words=chapter_words,
        chapter_num=chapter_num,
        title=title,
        aesthetic=AESTHETIC_PRESETS.get(preset, AESTHETIC_HISTORY),
    )
    v1_section = _OUTPUT_SECTION.format(
        chapter_words=chapter_words, chapter_num=chapter_num, title=title)
    found = system.count(v1_section)
    if found != 1:
        raise WriteError(
            "v2 cannot assemble a writer prompt: `_OUTPUT_SECTION` appears "
            f"{found} times in the assembled system prompt, expected exactly 1. "
            "v2 must REPLACE that section (it forbids the JSON half of the "
            "response); appending an override would ship two contradictory "
            "instructions and silently cost an extraction call per chapter. "
            "Fix `v2/write.build_system` to match the new assembly.")
    system = system.replace(
        v1_section, v2_output_section(chapter_words, chapter_num, title))
    try:
        swa = sensitive_word_avoidance_block(config)
    except Exception:
        swa = ""
    return system + ("\n\n" + swa if swa else "")


# ---------------------------------------------------------------------------
# User prompt — pure builders
# ---------------------------------------------------------------------------

_FIELD_LABELS = {
    "where": "地点", "turn": "转折", "exit_hook": "收尾钩子",
    "payoff": "本章兑现", "who": "在场人物", "beats": "节拍",
}


def contract_checklist(card: dict[str, Any] | None,
                       config: dict[str, Any] | None = None) -> str:
    """The CCC contract, quoted to the writer in the gate's own words.

    Every anchor named here comes from `accept._anchors`, which is
    `quality._beat_anchor_fragments` — the exact fragments the acceptance gate
    will grep for. This is the one prompt block whose wording is derived from a
    measurement rather than chosen, and `tests/test_v2_write.py` asserts the
    derivation so the two cannot drift apart.

    52% of archived chapters clear the proxy contract (`tools/ccr_baseline.py`).
    If v2 moves that number, this block is the lever that moved it.
    """
    if not isinstance(card, dict) or not card:
        return ""
    from engine.types import DEFAULT_TAIL_CHARS
    from engine.quality import _beat_anchor_fragments

    cfg = (config or {}).get("novel", {}) if config else {}
    tail = int(cfg.get("ccc_tail_chars", DEFAULT_TAIL_CHARS) or
               DEFAULT_TAIL_CHARS)

    lines: list[str] = []

    def anchored(field: str, target: str, note: str = "") -> None:
        anchors = _beat_anchor_fragments(target)
        quoted = "".join(f"「{a}」" for a in anchors[:4])
        hint = f"　验收会在正文里搜这些词：{quoted}" if quoted else \
            "　（这一条写得太抽象，验收无法判定——请自己把它落成具体的人/物/动作）"
        lines.append(f"- **{_FIELD_LABELS.get(field, field)}**：{target}{note}\n{hint}")

    for field in ("where", "turn"):
        target = str(card.get(field) or "").strip()
        if target:
            anchored(field, target)

    hook = str(card.get("exit_hook") or "").strip()
    if hook:
        anchored("exit_hook", hook, f"（必须落在全章最后 {tail} 字之内，不能提前用掉）")

    names = [str(n).strip() for n in (card.get("who") or []) if str(n).strip()]
    if names:
        lines.append("- **在场人物**：" + "、".join(names) +
                     "\n　每个名字都必须在正文里原样出现。")

    payoff = str(card.get("payoff") or "").strip()
    if payoff:
        anchored("payoff", payoff, "（计入兑现率）")

    forbid = [str(f).strip() for f in (card.get("forbid") or []) if str(f).strip()]
    if forbid:
        lines.append("- **本章禁令（出现即判违约）**：" +
                     "、".join(f"「{f}」" for f in forbid) +
                     "\n　不许原样使用，也不许换个说法沿用同一个动作落点与句式。")

    if not lines:
        return ""
    return ("## ⚠ 本章契约自查表（交稿前逐条搜一遍正文，确认每条都真的在页面上）\n"
            "下面每一条都会被**确定性校验**逐字检查，没落到页面上就判本章未兑现、直接返工：\n"
            + "\n".join(lines))


def thread_ledger(conn: Any, chapter_num: int, limit: int = 40) -> str:
    """The open-thread id list, for the delta's `threads` field.

    `canon`'s threads projection deliberately carries no ids — it is story
    context, and an id is not something the writer should be reading as story.
    But the delta writes to the same table, so the id vocabulary has to be in
    front of it or every chapter mints a new id for a thread that already exists
    and the ledger explodes. That is v1's 「线索 id 复用铁律」, restated here
    because in v2 the writer IS the extractor.
    """
    try:
        import engine.store as store

        rows = store.get_open_threads(conn, chapter_num, limit=limit)
    except Exception:
        return ""
    items = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        tid = str(r.get("id") or "").strip()
        desc = str(r.get("description") or "").strip()
        if not tid or not desc:
            continue
        due = r.get("due_chapter")
        items.append(f"- `{tid}` [{r.get('status', 'open')}"
                     + (f"|第{due}章前" if due else "") + f"] {desc[:60]}")
    if not items:
        return ""
    return ("## 线索台账（写 JSON 的 threads 字段时用）\n"
            "本章推进或兑现的若是下列已有线索，**必须原样复用它的 id**；"
            "已收束的把 status 写成 recovered 或 dropped；只有全新伏笔才新建 id。\n"
            + "\n".join(items))


def negative_block(preflight: dict[str, Any] | None) -> str:
    """The pre-write avoid list, from `_preflight_negative_list`.

    The *measurement* is v1's and is inlined — it reads the same gate rejects and
    the same `logs/book_fossils.json` cache. Only the rendering is v2's, and it is
    shorter on purpose: v1's version escalates through three warning tiers and
    repeats itself, which is prompt weight spent on emphasis rather than on
    information. The hard fossils get their own tail anchor anyway
    (`fossil_tail_anchor`), which is the position that measurably works.
    """
    if not isinstance(preflight, dict):
        return ""
    items = list(preflight.get("items") or [])
    fossils = list(preflight.get("fossils") or [])
    warnings = list(preflight.get("style_warnings") or [])
    if not (items or fossils or warnings):
        return ""
    parts = ["## 本章避雷（来自前几章真实触发的质量门禁）"]
    for it in items[:8]:
        parts.append(f"- {it}")
    if fossils:
        parts.append("**已复读成口癖的句子，严禁复现原句或结构相似的变体：**")
        parts.extend(f"  • 「{f}」" for f in fossils[:20])
    if warnings:
        parts.append("**近期文风问题：**")
        parts.extend(f"  • {w}" for w in warnings[:4])
    return "\n".join(parts)


def length_block(config: dict[str, Any]) -> str:
    novel = config["novel"]
    target = int(novel.get("chapter_words", 4000) or 4000)
    lo = int(novel.get("chapter_min_chars", 2500) or 2500)
    hi = int(novel.get("chapter_max_chars", 7000) or 7000)
    dlg_pct = int(float(novel.get("dialogue_char_ratio_target", 0.20)) * 100)
    return (f"## 本章字数与对话区间（硬性约束）\n"
            f"正文目标约 {target} 字，必须落在 {lo}-{hi} 字之间。"
            f"低于下限会被判过短、高于上限会被判超长，两种都要返工。\n"
            f"对话（引号内台词）占正文篇幅不低于 {dlg_pct}%——"
            f"关键信息通过人物对白推进，不要通篇叙述。\n"
            f"（第二段的状态增量 JSON 不计入字数。）")


def constraints_block(constraints: Iterable[str]) -> str:
    items = [str(c).strip() for c in (constraints or []) if str(c).strip()]
    if not items:
        return ""
    return ("## 规划阶段带过来的硬性要求（与通用准则冲突时以这些为准）\n"
            + "\n".join(f"- {c}" for c in items[:10]))


# The very last thing the writer reads. Measured need: on this book's writer
# (deepseek-v4-pro) the two-part obligation stated in the system prompt was
# honoured in 1 of 2 chapters — Ch2 came back 5,015 chars of pure prose with no
# sentinel at all, `ok=True`, nothing truncated. That is the same weak
# instruction-following the repo already answers with tail anchors four times
# over (ability capsule, recovery directive, scene-entry salience, hard
# fossils), and the same remedy applies: restate the obligation where recency
# is strongest. Kept to two lines on purpose — a long tail dilutes the position
# it exploits, which is exactly why `FOSSIL_TAIL_ANCHOR_MAX` exists.
DELTA_TAIL_ANCHOR = """

## ⚠ 回复格式最后确认（本条最后读，最优先遵守）
正文收尾之后**必须**另起一行原样写下 `{sentinel}`，再输出那一个 JSON 对象。
只交正文、不交 JSON，本章的状态增量就要另花一次调用重新提取。"""


def delta_tail_anchor() -> str:
    return DELTA_TAIL_ANCHOR.format(sentinel=DELTA_SENTINEL)


def build_user(
    state: StoryState,
    card: dict[str, Any] | None,
    chapter_num: int,
    title: str,
    config: dict[str, Any],
    *,
    exemplars: str = "",
    negative: str = "",
    threads: str = "",
    constraints: Iterable[str] = (),
    tail_anchor: str = "",
    capsule: str = "",
) -> str:
    """Assemble the volatile half of the write prompt. Pure, so it is testable.

    Ordering is the design: the story context first, the obligations last. The
    three tail anchors keep v1's proven order — contract checklist, then hard
    fossils, then the ability boundary as the very last thing read (LESSONS: the
    capsule was added *because* the same rule mid-prompt was breached in 5 of 6
    chapters).
    """
    parts = [
        state.volatile_block(),
        exemplars,
        negative,
        length_block(config),
        constraints_block(constraints),
        threads,
        f"## 请求\n写第 {chapter_num} 章：{title}。"
        f"按上面的卡片执行，两段式输出（正文 → {DELTA_SENTINEL} → JSON）。",
        contract_checklist(card, config),
    ]
    user = "\n\n".join(p.strip() for p in parts if p and p.strip())
    if tail_anchor:
        user += tail_anchor
    if capsule:
        user += ("\n\n## ⚠ 写作前最后确认：能力边界（最高优先级，违反即作废重写）\n"
                 "动笔前再次确认——主角与关键人物**只能**使用下列白名单能力，"
                 "且严格限定在其标注的**模态**内推进剧情：\n" + capsule)
    # After the capsule, because this one is about the SHAPE of the reply rather
    # than its content: the capsule can be breached and still leave a chapter to
    # repair, whereas a reply with no second part leaves nothing to read the
    # book's state from.
    return user + delta_tail_anchor()


# ---------------------------------------------------------------------------
# Response splitting — pure
# ---------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out)
        out = re.sub(r"\s*```\s*$", "", out)
    return out.strip()


def _looks_like_delta(data: Any) -> bool:
    return isinstance(data, dict) and bool(DELTA_KEYS & set(data))


def split_response(raw: str) -> tuple[str, str]:
    """(prose, delta_text). Never raises; never returns prose it is unsure about.

    The LAST sentinel wins. If the model somehow writes the sentinel inside the
    prose, taking the last one keeps the chapter whole and risks only the delta —
    the correct direction to fail in, since the delta is re-derivable and the
    chapter is not.

    With no sentinel at all, a trailing JSON object is accepted only when it
    actually carries delta keys. Without that guard the fallback would happily
    slice a chapter in half at a piece of dialogue containing a brace.
    """
    text = str(raw or "")
    matches = list(_SENTINEL_RE.finditer(text))
    if matches:
        m = matches[-1]
        return text[:m.start()], _strip_fence(text[m.end():])

    for m in reversed(list(re.finditer(r"\n\s*(?:```[a-zA-Z]*\s*\n)?\s*\{", text))):
        tail = _strip_fence(text[m.start():])
        try:
            data = safe_json_loads(tail)
        except Exception:
            continue
        if _looks_like_delta(data):
            return text[:m.start()], tail
    return text, ""


def parse_delta(
    raw: str,
    *,
    client: Any = None,
    paths: Paths | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[str, ChapterDelta, str]:
    """Split the response and read the delta. Returns (prose, delta, status).

    `status` is one of `ok` / `repaired` / `missing` / `unparsed`, and `run.py`
    logs it per chapter: a v2 arm whose deltas mostly come back `unparsed` has
    silently reverted to v1's cost profile with worse persistence, and that must
    be visible in the A/B rather than inferred from a continuity bug later.

    The repair path costs one cheap call and only runs when a client is supplied.
    It is bounded by construction — one attempt, no loop — because a delta is
    worth one call and not two.
    """
    prose, tail = split_response(raw)
    if not tail.strip():
        return prose, ChapterDelta(), "missing"
    try:
        data = safe_json_loads(tail)
        status = "ok"
    except Exception:
        data, status = None, "unparsed"
        if client is not None and paths is not None and config is not None:
            try:
                repaired = load_json_with_repair(client, paths, config, tail,
                                                 fallback={})
                if _looks_like_delta(repaired):
                    data, status = repaired, "repaired"
            except Exception:
                pass
    if not _looks_like_delta(data):
        return prose, ChapterDelta(), \
            "unparsed" if status != "ok" else "missing"
    return prose, ChapterDelta.from_payload(data), status


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WriteResult:
    text: str
    delta: ChapterDelta
    delta_status: str
    title: str
    prompt_chars: int = 0
    raw_chars: int = 0

    @property
    def delta_ok(self) -> bool:
        return self.delta_status in {"ok", "repaired"}


def _exemplars(paths: Paths, conn: Any, config: dict[str, Any],
               plan: dict[str, Any], chapter_num: int) -> str:
    try:
        from engine.retrieval import exemplar_block

        return exemplar_block(paths, conn, config, plan, chapter_num)
    except Exception:
        return ""


def _preflight(paths: Paths, conn: Any, config: dict[str, Any],
               chapter_num: int) -> dict[str, Any] | None:
    if not bool(config["novel"].get("preflight_constraints_enabled", True)):
        return None
    try:
        return _preflight_negative_list(
            paths, conn, config, chapter_num,
            lookback=int(config["novel"].get("preflight_constraints_lookback", 5)))
    except Exception:
        return None


def _capsule(paths: Paths, config: dict[str, Any]) -> str:
    try:
        from engine.bootstrap import contract_capsule

        return contract_capsule(paths, config)
    except Exception:
        return ""


# `_chapter_write_max_tokens` sizes the response for prose ALONE —
# `chapter_max_chars * ratio + margin`. Under v2 the same response also has to
# carry the delta, and a delta that runs out of tokens is indistinguishable from
# a model that ignored the instruction: `delta_status` reads `unparsed`, run.py
# falls back to an extraction call, and v2 quietly pays v1's price. So the cap
# gets explicit room for it. Sized from `DELTA_SCHEMA`: five fields, a handful of
# rows each, CJK at roughly one token per character.
DELTA_TOKEN_HEADROOM = 1200


def max_tokens(config: dict[str, Any]) -> int | None:
    base = _chapter_write_max_tokens(config)
    return None if base is None else base + DELTA_TOKEN_HEADROOM


def write_chapter(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    card: dict[str, Any],
    state: StoryState,
    *,
    plan: dict[str, Any] | None = None,
    constraints: Sequence[str] = (),
    temperature: float | None = None,
    call: Callable[..., str] | None = None,
) -> WriteResult:
    """ONE call: the chapter and its state delta.

    `state.stable_prefix()` rides as `cacheable_prefix`, the same bytes the arc
    call used, so the write calls of every chapter in an arc share a warm prefix
    with the plan that produced them.

    Raises `WriteError` on a response too short to be a chapter. That is a
    provider refusal or a truncated stream, not a bad draft, and retrying the
    call is the only sensible response — which is `run.py`'s decision, so it gets
    an exception rather than a 200-char "chapter".
    """
    call = call or call_llm
    title = clean_title(card, chapter_num)
    system = build_system(config, chapter_num, title)
    plan = plan if isinstance(plan, dict) else {}

    preflight = _preflight(paths, conn, config, chapter_num)
    user = build_user(
        state, card, chapter_num, title, config,
        exemplars=_exemplars(paths, conn, config, plan, chapter_num),
        negative=negative_block(preflight),
        threads=thread_ledger(conn, chapter_num),
        constraints=constraints,
        tail_anchor=fossil_tail_anchor(preflight, config),
        capsule=_capsule(paths, config),
    )

    temp = float(config["api"]["temperature"]) if temperature is None else temperature
    prefix = state.stable_prefix()
    log(paths, f"v2.write Ch{chapter_num} temp={temp:.2f} "
               f"system={len(system)} user={len(user)} prefix={len(prefix)}")
    raw = call(client, paths, config, system, user, temperature=temp,
               cacheable_prefix=prefix, max_tokens=max_tokens(config), tag="write")

    prose, delta, status = parse_delta(raw, client=client, paths=paths, config=config)
    text = normalize_chapter(prose)
    if len(text.strip()) < 500:
        raise WriteError(
            f"Ch{chapter_num}: write call returned {len(text.strip())} chars of prose "
            f"(raw {len(raw or '')}). Likely a provider refusal or truncated stream. "
            f"Preview: {text[:200]!r}")
    log(paths, f"v2.write Ch{chapter_num} -> {len(text)} chars, delta={status} "
               f"(events={len(delta.events)} entities={len(delta.entities)} "
               f"threads={len(delta.threads)})")
    return WriteResult(text=text, delta=delta, delta_status=status, title=title,
                       prompt_chars=len(system) + len(user) + len(prefix),
                       raw_chars=len(raw or ""))


BACKFILL_SYSTEM = """你是小说状态提取器。读完给定的章节正文，输出该章产生的持久状态变化。
只输出**一个** JSON 对象，不要代码围栏、不要解释、不要任何多余文字。
只记录正文里真正发生的变化；某一项没有变化就给空数组 []。不要编造正文没写的内容。"""


def backfill_delta(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    *,
    call: Callable[..., str] | None = None,
) -> tuple[ChapterDelta, str]:
    """Derive the delta from finished prose, in ONE cheap call.

    The write call is supposed to return prose and delta together, and when it
    does this never runs. It exists because compliance is a property of the
    MODEL, not of the design: the same prompt got `delta=ok` from gemini-2.5-pro
    twice and from deepseek-v4-pro once in two tries, and deepseek is what the
    A/B arm writes with.

    The alternative — commit the chapter with no delta — is not the cheap option
    it looks like. `canon.load` projects facts / threads / recent from exactly
    what the delta writes, so a book that keeps missing it goes blind one
    chapter at a time, and the A/B would be measuring a v2 that cannot see its
    own story rather than the v2 being proposed.

    So it is bought, and it is bought VISIBLY: `tag="delta_backfill"` routes to
    the extraction model and lands its own row in `llm_calls.jsonl`, which is
    the file `compare._llm_totals` reads. The headline calls/chapter number
    therefore includes every one of these. Bounded to a single attempt — a delta
    is worth one call, and it was already worth one call when the writer skipped
    it.
    """
    call = call or call_llm
    body = str(text or "").strip()
    if not body:
        return ChapterDelta(), "missing"
    user = (f"章节正文（第 {chapter_num} 章）：\n\n{body}\n\n"
            f"按下列结构输出这一章的状态增量：\n{DELTA_SCHEMA}")
    try:
        raw = call(client, paths, config, BACKFILL_SYSTEM, user,
                   temperature=0.2, json_mode=True, tag="delta_backfill")
    except Exception as exc:
        log(paths, f"v2.delta_backfill Ch{chapter_num} call failed: {exc}")
        return ChapterDelta(), "missing"
    try:
        data = safe_json_loads(_strip_fence(str(raw or "")))
    except Exception:
        data = None
    if not _looks_like_delta(data):
        log(paths, f"v2.delta_backfill Ch{chapter_num} returned no usable delta")
        return ChapterDelta(), "missing"
    delta = ChapterDelta.from_payload(data)
    log(paths, f"v2.delta_backfill Ch{chapter_num} -> events={len(delta.events)} "
               f"entities={len(delta.entities)} threads={len(delta.threads)}")
    return delta, "backfilled"


__all__ = [
    "WriteError", "WriteResult", "DELTA_SENTINEL", "DELTA_SCHEMA",
    "V2_OUTPUT_SECTION", "v2_output_section", "build_system", "build_user",
    "clean_title", "contract_checklist", "thread_ledger", "negative_block",
    "length_block", "constraints_block", "split_response", "parse_delta",
    "write_chapter", "max_tokens", "DELTA_TOKEN_HEADROOM", "DELTA_KEYS",
    "DELTA_TAIL_ANCHOR", "delta_tail_anchor", "backfill_delta", "BACKFILL_SYSTEM",
    # Inlined from writing.py:
    "GENRE_PROFILES", "AESTHETIC_PRESETS", "AESTHETIC_HISTORY",
    "ANTI_FRAGMENT_BAN", "ANTI_PITFALL_BLOCK",
    "SENSITIVE_WORD_AVOIDANCE_BLOCK",
    "FOSSIL_TAIL_ANCHOR_MAX",
    "save_chapter", "update_structured_state", "update_state_file",
    "sensitive_word_avoidance_block", "fossil_tail_anchor",
    "_build_write_system", "_OUTPUT_SECTION",
    "_preflight_negative_list",
    "_chapter_write_max_tokens", "_hook_directives_block",
]


# ===========================================================================
# Below: constants and functions inlined from writing.py.
# Only active code used by v2/v3 write paths is kept; v1-only dead code
# (write_chapter, _write_chapter_by_scenes, _split_beats_into_scenes,
# extract_events, _attempt_write, _finalize_chapter) is deliberately excluded.
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared anti-style-collapse ban (no literal braces — may be .format()'d)
# ---------------------------------------------------------------------------

ANTI_FRAGMENT_BAN = """## 文风塌缩禁令（最高优先级）
- 禁止"句子——状态——状态"式破折号短句链；破折号全章不超过3个，且只用于正常插入语，不得用来粘连碎片。
- 每段至少含2个有主谓宾的完整句子，禁止整段单词短句堆叠或无标点的舞台提示式断行。
- 句子长短交替，避免连续3句字数相近的短句。
- 以完整成句的小说叙事和有潜台词的对话为主，绝不能退化成电报体/碎片体。"""

ANTI_PITFALL_BLOCK = """## 网文避雷+去AI味铁律（读者弃书的确定性触发器）
### 人设铁律
- 主角必须有独立判断力和行动力：不当圣母（无底线原谅敌人）、不当舔狗（无脑付出无回报）、不降智（行为逻辑配得上已建立的人设）。
- 反派/对手必须有自己的行事逻辑和动机：禁止降智送人头，禁止无理由针对主角，反派的失败必须合理而非智商下线。
- 角色性格转变必须有铺垫：禁止性格突变（谨慎到冲动）而无触发事件，变化须由情节驱动、有过渡。
- 每个出场角色须有区分度：不同角色的说话方式、用词习惯、思维模式必须不同，禁止千人一面。
### 情节铁律
- 信息/设定前后一致：本章出现的任何事实不得与前文已确立的设定矛盾。
- 伏笔必须有回收计划：埋下的悬念必须在可预见章节内给出回应，禁止无限挖坑。
- 禁止无效注水：每个场景必须推进剧情、揭示人物或制造张力，删掉不影响故事的段落就不该存在。
- 压抑须有节奏：虐主后必须在合理章数内给予释放/反击，不得连续3章以上纯压抑无爽点。
### 去AI味铁律
- 禁止情感贴标签：不写"他感到震惊"，而是写震惊时的具体身体反应和行为；不写"她很悲伤"，而是写她做了什么。
- 禁止比喻堆砌：每千字比喻不超过3个，且必须新鲜贴切，禁用"时间仿佛静止""心如刀绞""像是被抽空了"等陈腐比喻。
- 禁止程度副词撑场面：删掉"非常""极其""十分""无比"，用具体细节替代模糊的程度修饰。
- 禁止总结式叙述：不写"就这样一切都变了""命运的齿轮开始转动"，让读者从情节中自行感受。
- 段落结构要多变：不得每段都是"描写到心理到对话"的固定模板，段落长短、开头方式、叙述角度须变化。
- 禁止AI套话：不用"心中一沉/瞳孔一缩/嘴角微微上扬/缓缓开口/深吸一口气/一时间/此刻"等万能表达，换用只属于当前角色和场景的具体反应。
- 每个角色有语言指纹：不同角色的遣词造句、句式长短、口头禅必须有辨识度。
### 排版铁律
- 禁止碎段：每段至少3-5句、60字以上（纯对话句除外）；禁止单句成段，禁止2000字写出80+行。段落应有起承转，不是每句话换一行。
- 禁止问答式对话推进：角色对话不得连续3轮以上一问一答；对话间必须夹入动作、神态、心理或环境描写，打破审讯/采访节奏。
### 否定对仗禁令
- 禁止"没有X，也没有Y""不是X，也不是Y"式对仗否定句——这是最明显的AI写作指纹。要表达缺失，直接一句说完，不要用两个否定分句对称排列。
### 内容密度铁律
- 禁止用大段环境描写/华丽辞藻/大量比喻修辞来充字数：环境描写每场景不超过2句，且必须服务于情绪或信息推进，纯装饰性描写全删。
- 禁止整章都是行为流水账：每个场景至少有一段角色内心的犹豫/权衡/情绪波动（通过动作或内心活动体现），让读者能代入角色处境。"""

_SHOW_DONT_TELL_BLOCK = """## 展示而非告知 + 收尾技法（正向硬约束，最高优先级；与"去AI味铁律"的禁令互补——那是禁什么，这是怎么写好）
- 情绪一律"演"出来：落到可观察的动作、对白、生理反应、对环境的处理上，让读者自行体会，绝不用旁白替读者下结论。
  · 告知✗：她很愤怒。 展示✓：她把搪瓷缸子往桌上一顿，热水溅到手背也没缩手。
  · 告知✗：他松了口气。 展示✓：他后背抵上门板，慢慢滑坐下去，低低笑了一声。
- 强动词优先：一个精准有力的动词，胜过"程度副词+泛动词"——不写"非常用力地攥着"，写"攥得指节发白"。
- 段落收尾落在动作/神态/一句话上：每个自然段尽量以人物的动作、神态或对白收束，少用抒情、总结、内心独白、展望未来来收尾——这是AI腔与注水的重灾区。
- 章末金句钩：全章最后一句写成一句短而狠、可截图传播的金句（情绪爆发／认知颠覆／悬念／后果预告），独立成段收束；不用"欲知后事如何""一切才刚刚开始"这类套话，不用抒情或总结式收尾。
### ⚠ 防过度护栏（"展示"用过头=崩章，务必守住）
- **展示 ≠ 堆砌描写**：展示要精准、一两笔到位，不为"演"而注水；同一情绪点一次给足即可，宁可少写一笔也不铺陈。
- **不牺牲对白**：对白仍是推进主力（承担传信息/制造冲突/暴露性格），别把该说的话都改写成大段动作或环境描写。
- **payoff 优先于描写**：首要任务是把 plan 的 payoff / 关键 beat 全部写到位并被读者看见；描写服务于推进剧情，绝不能因为忙着"展示"而漏掉当章该兑现的 beat。
- **守字数上限**：本章须落在给定字数区间内，若逼近上限，先砍描写与铺陈、绝不砍情节与 payoff。"""

# ---------------------------------------------------------------------------
# Aesthetic presets (no literal braces; selected per genre)
# ---------------------------------------------------------------------------

AESTHETIC_COMMON = """## 审美与品味（在不牺牲可读性与节奏的前提下，写出质感）
- 克制与留白：最有力的情绪点用最省的笔墨，给读者回味空间；不把话说尽，不替读者下结论。
- 意象与潜台词：善用一两个贯穿场景的核心意象（物、光、声）折射心境；对白藏机锋，重要的事让人物用言外之意带出。
- 节奏即呼吸：长短句、缓急段交替形成韵律；高潮处可短促有力，铺陈处可舒展从容，避免通篇一个语速。
- 叙事腔调统一：维持与 voice 宪章一致的稳定声音；动词精准有力。
- 反套路表达：同一情绪/场景换一种写法呈现，避免与近章雷同的措辞与比喻；让句子有"只此一处"的辨识度。"""

AESTHETIC_HISTORY = AESTHETIC_COMMON + """
## 历史题材额外品味
- 语言有古意而不晦涩：用词、称谓、礼仪贴合时代，却让今人读得顺；避免半文不白的夹生腔。
- 以小见大：用一处器物、一道公文、一个仪轨折射时代重量，胜过空泛的"历史厚重感"宣称。
- 克制抒情，重白描与场面调度；权力与人心的张力靠细节与潜台词流露，不靠作者旁白点评。"""

AESTHETIC_SHUANG = AESTHETIC_COMMON + """
## 爽文题材额外品味
- 爽而不俗：爽点靠铺垫、反差与对手的真实反应挣来，落到精准的动作与神态上，而非形容词喊"太爽了"。
- 装而不油：主角的从容与机锋点到为止，留白比说满更有格调；忌油腻炫耀式独白。
- 反差与节奏感本身就是审美：压抑—释放的曲线干净利落，比堆砌爽点更耐读。"""

AESTHETIC_SYSTEM_STREAM = AESTHETIC_COMMON + """
## 系统流题材额外品味
- 面板服务情绪：数值与提示音要落在恰当的戏剧时刻，制造"叮"的爽感与期待，而非冷冰冰的数据罗列。
- 成长有质感：升级不止是数字变大，要让读者感到主角心境、处境、视野的真实跃迁。
- 系统语气有辨识度：系统提示的腔调（冷峻/毒舌/中二）保持统一，成为本书的趣味点而非噪音。"""

AESTHETIC_URBAN_ABILITY = AESTHETIC_COMMON + """
## 都市异能/重生题材额外品味
- 都市质感真实：行业、阶层、人情世故的细节要可信，让爽点扎根在真实生活的土壤里。
- 打脸要漂亮：反转靠信息差与铺垫的精巧，落点干脆，余味留给对手的反应与旁观者的神情，不靠主角自吹。
- 异能/先知写得有想象力：能力的呈现要有画面与代价，避免沦为万能开关。"""

AESTHETIC_ROMANCE_FEMALE = AESTHETIC_COMMON + """
## 女频言情题材额外品味
- 情绪靠细节传递：心动与受伤写在视线、距离、停顿、指尖的颤动里，绝不用旁白直接宣告"她很感动"。
- 暧昧的张力来自克制：欲言又止、若即若离比直白告白更撩人；甜与虐都要有余韵。
- 对白有化学反应：男女主的机锋、试探、心照不宣构成独特的二人语感，配角也各有声口。"""

AESTHETIC_WANZU_XUANHUAN = AESTHETIC_COMMON + """
## 现代玄幻/万族题材额外品味
- 斗法有美感：招式、气机、环境互动写出画面与节奏感，像一场编排过的武戏，而非招式名的罗列。
- 境界有意境：突破与顿悟落到具体的体感与心境变化上，避免空喊"实力大增"。
- 宏大不空洞：万族、天骄、大势的格局靠具体的人、物、场面撑起，杜绝纯设定名词的堆砌。"""

AESTHETIC_SUSPENSE = AESTHETIC_COMMON + """
## 悬疑/心理惊悚题材额外品味
- 限制视角是第一纪律：只写视角人物当下能看见、听见、想到的，绝不偷偷塞入全知者才知道的真相或他人内心；信息差就是悬念的引擎。
- 感官克制制造不安：恐惧与诡异写在反常的细节里（多出一只杯子、湿了一半的脚印、本该有却没有的声音），而非靠"恐怖""毛骨悚然"这类形容词喊出来。
- 公平线索：所有指向真相的线索必须公平地呈现在读者眼前（哪怕被淹没在干扰项里），结局的反转要让读者回头能找到伏笔，杜绝凭空掉落的关键信息。
- 留白与未言之事：最吓人的是没写出来的东西；关键时刻收笔、让画面停在悬而未决处，比写满更有压迫感。
- 不可靠与氛围：可善用视角人物的偏见、记忆缺口、自我说服制造叙事张力，但作者不得借此对读者撒谎（隐瞒可以，捏造不行）。"""

AESTHETIC_PRESETS = {
    "history": AESTHETIC_HISTORY,
    "xuanhuan_shuang": AESTHETIC_SHUANG,
    "system_stream": AESTHETIC_SYSTEM_STREAM,
    "urban_ability": AESTHETIC_URBAN_ABILITY,
    "romance_female": AESTHETIC_ROMANCE_FEMALE,
    "wanzu_xuanhuan": AESTHETIC_WANZU_XUANHUAN,
    "suspense": AESTHETIC_SUSPENSE,
}

# ---------------------------------------------------------------------------
# Writer system prompt: shared-base + genre-delta architecture
# ---------------------------------------------------------------------------

_SELF_REVIEW_PREAMBLE = """## 写前自我审查（内部执行，严禁输出）
以下步骤只用于动笔前校准；如果接口没有隐藏 reasoning_content，就在心里完成，不要写出任何标题、列表、分析、解释或 XML/Markdown 元信息。
1. 识别本章三项最高风险："""

_OUTPUT_SECTION = """## 输出要求
- 约{chapter_words}个中文字符。
- 第一行固定格式：第{chapter_num}章 {title}
- 执行选定的plan及所有约束条件。
- plan中的高风险beats必须直接在页面上演出。
- 只输出章节正文。第一行必须直接是章节标题，严禁输出"写前自我审查"、"Pre-writing Self-Review"、"分析"、"reasoning"、`<analysis>`、`<thinking>`、代码围栏、JSON、清单或任何解释。"""

_SENSORY_DIALOGUE_DEFAULT = """## 感官与对话
- 每个场景至少2种感官锚点；用具体细节代替抽象描述。
- 对话占全章25-45%，反映人物身份与心理，关键对话含潜台词。
- 破折号（——）全章最多3个（不是每千字3个，是全章总共≤3个）——超限的破折号在发布前会被机器全部替换成逗号，与其被机器改不如自己写对。需要停顿用逗号，需要解释用括号或另起短句。
- 连续超过2个自然段的纯叙述/描写必须被对话、电话、录音回放或角色自言自语打断——手机端连续3段无引号就是灰色文字墙，读者直接划走。开篇前3段内必须出现第一句对话（哪怕是自言自语或电话铃声）。"""

_TIME_MARKER_BAN_DEFAULT = """## 时间标记禁令
- 严禁以时间副词切换场景；时间流逝靠情节动作体现，每章最多2个时间词且与具体行为绑定。"""


GENRE_PROFILES: dict[str, dict[str, str]] = {
    "history": {
        "role": "擅长中国历史题材的长篇网文作家，风格厚重克制，兼具网文可读性",
        "self_review": (
            "   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？\n"
            "   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n"
            "   - 空洞兑现风险：主角在哪里可能轻松获胜却没有代价？\n"
            "2. 针对每项风险，写一条具体规避承诺（如“用茶寮而非文渊阁”、“在页面上呈现户部程序”、“让主角失去一张底牌”）。\n"
            "3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。\n"
            "4. 完成1-3后才开始正式写作。"
        ),
        "core_discipline": "",
        "structure_template": (
            "## 结构模板\n"
            "- 开场钩（200-400字）：紧接上章末尾，建立本章核心问题或悬念，禁止用时间词作开场\n"
            "- 场景一（1000-1500字）：主要冲突场景，含具体动作、对话与环境描写\n"
            "- 场景二（800-1200字）：转折或揭示场景，推进plan中的关键 beats\n"
            "- 场景三（600-1000字）：决定或代价场景，呈现选择后果\n"
            "- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾"
        ),
        "sensory_dialogue": (
            "## 感官纪律\n"
            "- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）\n"
            "- 用具体细节代替抽象描述（“墨迹未干的公文” 而非 “重要文件”）\n"
            "- 季节、天气、光线作为情绪衬托，不作章节进度的计时器\n\n"
            "## 对话比例\n"
            "- 对话占全章25-45%，避免连续500字以上无对白的段落\n"
            "- 每个角色的语气、用词必须反映其身份、立场和当下心理"
        ),
        "time_marker_ban": (
            "## 时间标记禁令（核心问题）\n"
            "- 严禁以“翌日清晨”“这天晚上”“次日黄昏”“午后”“深夜”等时间副词切换场景或开启段落\n"
            "- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记\n"
            "- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定（如“赶在衙门散班前”而非单纯“傍晚”）"
        ),
        "genre_bans": (
            "## 禁止模式\n"
            "- 禁止“他突然意识到/恍然大悟”式的廉价顿悟\n"
            "- 角色内心独白每次50-200字：关键抑择时刻必须有心理描写（不少于50字的犹豫/权衡/恐惧），但不超过200字避免意识流\n"
            "- 禁止同一章内出现超过3次相同的动作描写（如“皱眉”“沉默”“点头”）\n"
            "- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话"
        ),
        "extras": (
            "## 人物塑造要求\n"
            "- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或反派\n"
            "- 主角的成长必须来自挛败、情报或他人的推演，不能突然“顿悟”\n"
            "- 对话必须含潜台词和话术攻防，不能只喊口号和表态\n"
            "- 官场人物的措辞必须符合其政治处境（得势者与失势者说话方式不同）\n\n"
            "## 情节逻辑要求\n"
            "- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果\n"
            "- 如有伏笔，必须在本章或后续章节可查的文本中有对应的“收线”\n"
            "- 不得出现“某人神秘地笑了”类的模糊悬念代替真实信息\n\n"
            "## 本章必须满足的质量硬指标（写完前自检）\n"
            "- 显性代价：主角本章至少有一次**可见的资源/政治/情感代价**（失去一张底牌、得罪一方势力、付出信任或人情），不得轻松全胜。\n"
            "- 对白潜台词：本章至少有一处关键对话含**话术攻防/言外之意**（如表面奏对、暗里递价；正例：“臣不敢妄言”实指“陈下先表态臣才敢接”），不得只喊口号表态。\n"
            "- 差异化：禁止复用最近3章已用过的开场方式与章末钩子手法；若雷同，必须换一种结构（场景驱动↔反转↔压迫-兑现等）。"
        ),
    },
    "xuanhuan_shuang": {
        "role": '擅长穿越爽文的中文网文作家，节奏明快、爽点密集、画面感强、读者代入感极强',
        "self_review": (
            '   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n'
            '   - 无脑碾压风险：主角在哪里可能毫无铺垫地轻松获胜、缺乏代价或对手反应？\n'
            '2. 针对每项风险，写一条具体规避承诺（如“用一次失败的试探换信任”、“让赵高当场反将一军”、“现代知识落到一个具体器物/制度细节上”）。\n'
            '3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。\n'
            '4. 明确本章的“爽点高潮”是哪一段（兑现/打脸/翻盘/掌权之一），它如何被前文铺垫和压迫衬托。\n'
            '5. 完成1-4后才开始正式写作。'
        ),
        "core_discipline": (
            '## 爽点纪律（本类型核心）\n'
            '- 本章必须有**至少1个明确的爽点高潮**：兑现、打脸、翻盘、识破阴谋或掌权之一，且落到具体动作与对手反应上。\n'
            '- 压迫—兑现节奏要紧：铺垫不拖沓，先制造压迫/轻视/危机，再在高潮处一举兑现，让读者有“出了一口气”的快感。\n'
            '- 主角靠“现代灵魂的先知与见识”做出超越时代的判断，但每次施展**必须有铺垫与代价**（被猜忌、暴露底牌、消耗人情），不得无脑全知全能。\n'
            '- 章末必须留一个让读者想立刻看下一章的强钩子。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章末尾，立刻抛出本章核心冲突或压迫，禁止用时间词作开场\n'
            '- 场景一（1000-1500字）：主要冲突/压迫场景，含具体动作、对话与环境描写\n'
            '- 场景二（800-1200字）：转折或主角施展见识的场景，推进plan关键beats，埋下爽点引信\n'
            '- 场景三（600-1000字）：爽点兑现/打脸/翻盘场景，呈现选择的后果与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官纪律\n'
            '- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）\n'
            '- 用具体细节代替抽象描述（“竹简上未干的朱批” 而非 “重要文书”）\n'
            '- 季节、天气、光线作为情绪衬托，不作章节进度的计时器\n'
            '\n'
            '## 对话比例\n'
            '- 对话占全章25-45%，避免连续500字以上无对白的段落\n'
            '- 每个角色的语气、用词必须反映其身份、立场和当下心理\n'
            '- 关键对话需含潜台词与话术攻防，不只喊口号'
        ),
        "time_marker_ban": (
            '## 时间标记禁令（核心问题）\n'
            '- 严禁以“翌日清晨”“这天晚上”“次日黄昏”“午后”“深夜”等时间副词切换场景或开启段落\n'
            '- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记\n'
            '- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定'
        ),
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止“他突然意识到/恍然大悟”式的廉价顿悟\n'
            '- 角色内心独白每次50-200字：关键抉择时刻必须有心理描写（不少于50字的犹豫/权衡/恐惧），但不超过200字避免意识流\n'
            '- 禁止同一章内出现超过3次相同的动作描写\n'
            '- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话\n'
            '- 禁止主角无铺垫、无代价地碾压全场（爽要爽得有逻辑）'
        ),
        "extras": (
            '## 人物塑造要求\n'
            '- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或当反派\n'
            '- 主角（现代灵魂）的判断与成长必须来自现代见识、情报或挫败的推演，不能突然“顿悟”\n'
            '- 对手要聪明、有手段、有反应，不能是任主角宰割的纸片人\n'
            '- 秦制背景下的措辞与礼仪需大体得体，不出现现代名词穿帮\n'
            '\n'
            '## 情节逻辑要求\n'
            '- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果\n'
            '- 如有伏笔，必须在本章或后续章节可查的文本中有对应的“收线”\n'
            '- 不得出现“某人神秘地笑了”类的模糊悬念代替真实信息'
        ),
    },
    "system_stream": {
        "role": '擅长系统流网文的中文作家，面板感强、成长节奏明快、数值反馈清晰、读者代入与成就感强烈',
        "self_review": (
            '   - 重复风险：本章是否在重复上一次的“刷面板/做任务”套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n'
            '   - 无脑刷级风险：主角是否毫无代价、毫无策略地靠系统碾压？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“系统反馈高潮”是哪一段（升级/解锁/任务结算/奖励兑现之一），以及它如何被前文需求与压迫衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 系统流核心纪律（本类型核心）\n'
            '- 本章至少有一次**可见的系统反馈**：面板属性变动、任务发布/结算、技能解锁、奖励到账之一，落到具体数值或具体能力上，让读者有“看得见的成长”。\n'
            '- 系统不是万能：每次使用系统能力都要有**代价、冷却、前置条件或风险**，禁止无脑刷级、无脑碾压。\n'
            '- 成长有节奏：升级/变强必须解决一个具体困境，并立刻引出更高一级的新困境，不堆砌纯数值。\n'
            '- 面板信息要服务剧情张力，不做无意义的数据罗列。\n'
            '- **每章必兑现（爽点直给）**：本章必须有一次明确的爽点落地——打脸看衰者/碾压对手/奖励爆发/身份反差之一，且在本章内当场兑现，不得把爽点全部后置到“铺垫完成之后”。压迫与兑现五五开，不要写成只压不爽的慢热悬疑。\n'
            '- **少埋多兑（连贯性）**：本章必须先关闭至少 1 条上一章遗留的悬念/小钩子，才能开启新的悬念；禁止只开不收、让悬念与伏笔无限堆积。每章净新增悬念不超过 1 个。\n'
            '- **黄金三章特例**：若本章是前 3 章，必须有一个独立、当章即兑现的爽点高潮（打脸/碾压/第一桶金/能力首秀），让读者立刻看到本书的爽点形态，不得只做设定铺陈或氛围悬疑。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出本章核心困境或一个待结算的系统任务\n'
            '- 场景一（1000-1500字）：困境/压迫场景，主角在约束下挣扎，凸显系统能力的必要性与代价\n'
            '- 场景二（800-1200字）：主角运用系统/策略破局的关键场景，埋下结算引信\n'
            '- 场景三（600-1000字）：系统反馈兑现（升级/解锁/奖励）与其代价、后果\n'
            '- 结尾钩（200-400字）：制造下章悬念或抛出新任务，不用总结式收尾'
        ),
        "sensory_dialogue": "",
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止“他突然意识到/恍然大悟”式廉价顿悟。\n'
            '- 禁止系统无代价地解决主线危机；禁止纯数值罗列代替剧情。\n'
            '- 禁止开场连续两段是环境描写或纯面板描述。'
        ),
        "extras": "",
    },
    "urban_ability": {
        "role": '擅长都市异能/重生题材的中文网文作家，节奏明快、代入感强、打脸爽点密集、画面感强',
        "self_review": (
            '   - 重复风险：本章是否复制了上一次的打脸/装逼套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成空泛概括？\n'
            '   - 无脑碾压风险：打脸是否毫无铺垫、对手是否降智送人头？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“爽点高潮”是哪一段（打脸/反转/资源碾压/身份揭示之一），它如何被前文轻视/压迫衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 都市异能核心纪律（本类型核心）\n'
            '- 本章至少有一个明确爽点：打脸、扮猪吃虎后的反转、资源/实力碾压、身份揭示之一，落到具体动作与对手反应上。\n'
            '- 代入感优先：主角的优势（重生先知/异能/资源）要让读者有“我也想这样”的爽，但优势必须有边界与代价。\n'
            '- 打脸要有铺垫：先有轻视/压迫/挑衅，再有反击，对手要聪明、有反应，禁止降智捧哏式纸片人。\n'
            '- 现代都市细节要真实（行业、阶层、人情世故），不出戏。\n'
            '- 信息差变现：本章主角必须至少有一个"利用预知/异能信息做出常人无法做出的判断"的时刻，'
            '且该判断带来可见的收益或避免可见的损失。这个时刻必须有**外部可见反馈**——'
            '对方的表情变化（愣住/变脸/后退一步）、语气转变（从轻蔑到警惕）、'
            '或行为改变（放下手中的东西/拿起电话），'
            '不能只在主角内心完成"我赢了"的确认。'
            '独处章节中，用电话对话、监控画面、或事后他人反应来体现信息差收益。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，立刻抛出冲突、轻视或挑衅\n'
            '- 场景一（1000-1500字）：压迫/被轻视场景，铺垫反击的合理性\n'
            '- 场景二（800-1200字）：主角施展优势的关键场景，埋下打脸引信\n'
            '- 场景三（600-1000字）：打脸/反转/碾压兑现，呈现对手反应与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾\n'
            '- 结尾钩的三选一（必须用其中之一收束全章）：\n'
            '  ①反转弹出：最后一句揭示一个颠覆读者预期的新信息（新身份/新威胁/新发现）\n'
            '  ②危机迫近：最后一句让主角面临一个迫在眉睫的、下一章必须解决的具体威胁\n'
            '  ③悬念提问：最后一句抛出一个读者无法自行回答的具体问题\n'
            '  禁止以内心感悟、环境描写、情绪沉淀、或总结式评论收尾。最后一段必须短（≤100字），独立成段。'
        ),
        "sensory_dialogue": (
            '## 感官与对话\n'
            '- 每个场景至少2种感官锚点；用具体细节代替抽象描述。\n'
            '- 对话占全章25-45%，反映人物身份与心理，关键对话含潜台词。\n'
            '- 破折号（——）全章最多3个（总共≤3个，不是每千字）——超限会被机器替换成逗号。需要停顿用逗号，需要解释用括号或另起短句。\n'
            '- 连续超过2个自然段的纯叙述/描写必须被对话/电话/录音回放/自言自语打断。开篇前3段内必须出现第一句对话。\n'
            '- 内心独白每千字不超过200字——能力副作用（记忆丧失、感知异常）优先用外部行为暴露'
            '（叫错名字、忘记约定、重复已做过的事），而非用内心复盘告诉读者"我忘了"。'
        ),
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止廉价顿悟、解释性叙述代替戏剧化呈现。\n'
            '- 禁止对手降智送人头；禁止无铺垫无代价的碾压。\n'
            '- 禁止开场连续两段是环境描写。\n'
            '- 章尾禁止："他望着窗外陷入沉思"式收尾；"一切才刚刚开始"式套话；最后两段都是叙述/描写无对话无动作。\n'
            '- 道具过载禁令：笔记本/录音笔/手机等核心道具每章各最多出现3次'
            '（含翻开、查看、录音等动作），'
            '且每次使用必须揭示一条新信息或新线索，'
            '禁止重复"翻开→发现字迹→不记得写过"的同一循环。'
            '相同的震惊/困惑反应不能在两章内重复——'
            '第二次看到自己不记得的字迹时，角色的反应必须进化'
            '（从震惊→接受→利用→质疑→反抗）。'
        ),
        "extras": "",
    },
    "romance_female": {
        "role": '擅长女频言情/宠文的中文网文作家，情绪张力细腻、关系推进有节奏、甜虐拿捏精准、代入感强',
        "self_review": (
            '   - 重复风险：本章的情绪节拍是否在重复（又一次误会、又一次心动）？\n'
            '   - 浅层执行风险：plan中哪个情感beat最可能变成”作者旁白告知”而非”通过细节让读者感受到”？\n'
            '   - 情绪悬浮风险：人物情绪是否缺乏具体事件支撑、显得无病呻吟？\n'
            '   - 外部见证风险：本章打脸/翻盘/逆袭是否只在女主内心完成？有没有旁观者的可见反应？\n'
            '   - 情绪公式风险：是否又用了”吃菜→感动→流泪”？是否又让角色嘴里直说主题？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的”情绪高潮”是哪一段（心动/误会/和解/吃醋/守护之一），它如何被前文关系状态衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 女频言情核心纪律（本类型核心）\n'
            '- 本章必须有**明确的关系推进或情绪高潮**：拉近、误会、和解、吃醋、双向奔赴、守护之一，落到具体的眼神/动作/对白细节上。\n'
            '- 情绪要有事件支撑：每一次心动/受伤都来自具体的行为或话语，不靠旁白直接告知”她很感动”。\n'
            '- 甜虐配比：依据 plan 的节奏，把甜点与虐点落到位，避免通篇平淡或通篇狗血。\n'
            '- 男女主对手戏要有潜台词与张力；配角不做纯工具人，要有自己的立场。\n'
            '- **打脸/翻盘必须有外部见证**：对手的表情变化（愣住/变脸/后退）、围观者的反应（倒吸气/窃窃私语/弹幕炸裂）、'
            '或第三方的态度转变——不能只在女主内心完成”我赢了”。数字跃迁（粉丝/销售/票数）给具体数字。\n'
            '- **女主主导关键决策**：打脸靠她自己拿证据、靠她的手艺和判断，男主可助攻不可代打；'
            '不要让女主站在旁边看男主替她解决问题。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章的关系状态，抛出本章情绪悬念\n'
            '- 场景一（1000-1500字）：主要关系场景，含具体互动与潜台词\n'
            '- 场景二（800-1200字）：情绪转折场景（误会加深/心防松动），推进关系弧\n'
            '- 场景三（600-1000字）：情绪高潮兑现（心动/和解/守护），呈现人物内心变化与代价\n'
            '- 结尾钩（200-400字）：留一个让读者揪心或期待的情绪悬念，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官与对话\n'
            '- 用细腻的感官与肢体语言传递情绪（视线、距离、停顿、指尖），不直接说教情绪。\n'
            '- 对话占全章25-45%，反映关系亲疏与人物性格，关键对话含言外之意。\n'
            '- **心动破防**：写生理反应+嘴硬掩饰（耳朵红却假装看手机、眼眶热赶紧低头干活、声音拐弯假装没事），'
            '不写"她脸红了""她心跳加速"这种直给——让读者比角色先懂她心动。\n'
            '- **弹幕/群众反应**：写得像真弹幕（有错字、有跑题、有复读、有杠精、有沉默潜水党突然冒一句），'
            '不要写成整齐划一的应援团或作者传声筒。弹幕是情绪出口，不是评论文章。'
        ),
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止用旁白直接宣告情绪代替细节呈现。\n'
            '- 禁止无事件支撑的情绪悬浮、无逻辑的狗血反转。\n'
            '- 禁止配角沦为纯工具人；禁止开场连续两段是环境描写。\n'
            '- **禁止"吃了她做的菜感动到流泪"公式**：食物可以触动情绪，但不是每个吃她菜的人都要哭。'
            '用不同的情绪催化方式（沉默、行为改变、态度软化、主动帮忙），全书流泪反应最多两次。\n'
            '- **禁止spreadsheet式数据罗列**：不要逐条列举数字/天数/百分比（"第一家41天，第二家48天，第三家..."），'
            '概括+举一个最有冲击力的具体例子就够了。\n'
            '- **禁止角色明说主题**：不要让角色嘴里说出"这就是XX的意义""XX不是YY，是ZZ"式的总结金句——'
            '主题靠事件和沉默传递，让读者自己品出来。\n'
            '- **禁止roll-call式群戏收束**：不要让所有角色在同一段里轮流各说一句话/做一个动作（"A说...B笑...C举杯..."），'
            '这是剧本大纲不是小说。群戏聚焦2-3人的具体互动，其余人用环境动作带过。\n'
            '- **禁止全知式策略分析**：角色不准像读过剧本一样精准预测对手的多步策略（"她一定会先联系XX，然后利用YY..."），'
            '改为片段线索+直觉推断+行动试探，允许判断不完全正确。'
        ),
        "extras": "",
    },
    "wanzu_xuanhuan": {
        "role": '擅长现代玄幻/万族争锋题材的中文网文作家，境界体系清晰、斗法画面感强、天骄争锋热血、爽点密集',
        "self_review": (
            '   - 重复风险：本章的斗法/突破是否在重复套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成空泛概括？\n'
            '   - 无铺垫破局风险：主角是否靠没讲清规则的力量强行翻盘？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“爽点高潮”是哪一段（境界突破/斗法胜出/天骄争锋/夺宝之一），它如何被前文压迫与差距衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 万族玄幻核心纪律（本类型核心）\n'
            '- 境界/战力体系必须清晰、可被读者预期：每次斗法的胜负要能用已铺垫的规则解释，不靠突兀的金手指强行破局（Sanderson 第一定律：能力解题的合理度正比于规则被讲清的程度）。\n'
            '- 主角的力量有边界与代价（第二定律：限制比能力更出戏），突破/施展强法必须付出消耗或风险。\n'
            '- 本章至少一个热血爽点：境界突破、斗法胜出、天骄过招、夺宝/夺机缘之一，落到具体招式与对手反应上。\n'
            '- 斗法要有画面感：招式、气机、环境破坏、攻防转换写到位，不只报结果。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出本章的对峙、差距或威胁\n'
            '- 场景一（1000-1500字）：压迫/差距场景，凸显主角面临的实力鸿沟\n'
            '- 场景二（800-1200字）：主角借规则/底牌/机缘缩小差距的关键场景，埋下翻盘引信\n'
            '- 场景三（600-1000字）：斗法/突破爽点兑现，呈现招式攻防、对手反应与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念（更强的敌人/新的机缘），不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官与对话\n'
            '- 每个场景至少2种感官锚点；斗法用具体招式与气机描写代替抽象形容。\n'
            '- 对话占全章25-45%，反映人物境界与心性，关键对话含机锋。'
        ),
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止用没铺垫的力量强行破局；禁止对手降智。\n'
            '- 禁止廉价顿悟、解释性叙述代替战斗呈现。\n'
            '- 禁止开场连续两段是环境描写。'
        ),
        "extras": "",
    },
    "suspense": {
        "role": '擅长悬疑/心理惊悚题材的中文作家，擅长用限制视角、感官克制与公平线索营造步步紧逼的不安与反转快感',
        "self_review": (
            '   - 视角越界风险：哪一处最可能不小心写出视角人物当下不可能知道的信息（他人内心、未到场之事、最终真相）？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“在场景里一点点逼出的悬念”？\n'
            '   - 线索失衡风险：本章揭示是否凭空掉落、缺少前文铺垫？或者悬念只开不收、无限堆积？\n'
            '2. 针对每项风险，写一条具体规避承诺（如“反派动机只透过一件物证暗示，不进其内心”、“把怀疑写成视角人物对一个反常细节的反复打量”、“本章收掉上一章关于X的疑点”）。\n'
            '3. 明确本章埋下/兑现的线索：列出本章新增的1条悬念、以及兑现或推进的至少1条旧线索（公平地放在读者眼前）。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 悬疑核心纪律（本类型核心）\n'
            '- **限制视角**：严格贴住视角人物，只写其当下能感知、能推断的；不得插入全知旁白或他人内心。信息差由此而生，是本类型最大的张力来源。\n'
            '- **感官克制**：恐惧/诡异藏在反常的具体细节里（多出的物件、错位的声音、本该在却消失的东西），用白描呈现，禁止“恐怖”“惊悚”“不寒而栗”式贴标签。\n'
            '- **公平线索**：每一个指向真相的关键线索都要公平地出现在读者眼前（可被干扰项掩盖），保证反转揭晓时读者回看能找到伏笔；禁止凭空掉落的关键信息。\n'
            '- **少埋多兑（连贯性）**：本章至少推进或收束1条已有悬念，才能开启新悬念；每章净新增悬念不超过1个，杜绝疑点无限堆积。\n'
            '- **留白**：关键的惊悚/揭示时刻可在临界处收笔，把最可怕的东西留给读者想象，未言之事比写满更有压迫感。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出一个反常细节或新的疑点，立刻制造不安，禁止用时间词作开场\n'
            '- 场景一（1000-1500字）：调查/对峙/独处场景，视角人物在信息不全中试探，逐步累积怀疑与压迫\n'
            '- 场景二（800-1200字）：转折或局部揭示场景，推进plan关键beats，给出一条公平线索（或一个误导）\n'
            '- 场景三（600-1000字）：代价/反转/更深疑点场景，视角人物做出选择并承受后果\n'
            '- 结尾钩（200-400字）：以一个悬而未决的细节或反转收束本章悬念，制造追读冲动，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官纪律\n'
            '- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉），优先用“反常的感官”承载诡异。\n'
            '- 用具体细节代替抽象描述（“门缝下渗出的、温的水” 而非 “诡异的气氛”）。\n'
            '- 光线、声音、温度作为情绪与悬念的放大器，不作章节进度的计时器。\n'
            '\n'
            '## 对话比例\n'
            '- 对话占全章25-45%，避免连续500字以上无对白的段落。\n'
            '- 关键对话要有试探、隐瞒与潜台词；人物可能说谎或避重就轻，但作者不得借叙述向读者捏造事实。'
        ),
        "time_marker_ban": (
            '## 时间标记禁令\n'
            '- 严禁以“翌日清晨”“这天晚上”“午后”“深夜”等时间副词切换场景或开启段落。\n'
            '- 时间流逝靠情节动作与因果链体现，每章最多2个时间词且与具体行为绑定。'
        ),
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止视角越界（写出视角人物不可能知道的真相、他人内心、未到场之事）。\n'
            '- 禁止“他突然意识到/恍然大悟”式廉价顿悟，揭示要靠线索推导。\n'
            '- 禁止靠凭空掉落的关键信息推动反转。\n'
            '- 禁止悬念只开不收、疑点无限堆积。\n'
            '- 禁止开场连续两段是环境描写，第一段内必须有人物动作、感知或对话。'
        ),
        "extras": (
            '## 情节逻辑要求\n'
            '- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果。\n'
            '- 所有伏笔在本章或后续可查文本中“收线”；揭示必须能在前文找到依据。\n'
            '\n'
            '## 章末禁令（防零增量总结收尾·硬约束）\n'
            '- **严禁用“总结段”收尾**：章末绝不能把正文已经给出的推理、线索、结论再复述一遍（哪怕换了措辞）。读者刚读过，再列一遍=零信息增量=拖慢节奏。\n'
            '- 章末必须是【前进的钩子】：抛出一个【新】疑问、一个【新】动作、一个【新】威胁，或一处让人物（与读者）措手不及的反常细节，把张力推向下一章。\n'
            '- 检验标准：把本章结尾段单独拎出来，它必须包含正文里【没出现过】的信息或动作；若它只是“由此可见…”“综上…”“他明白了…”式的归纳，立即重写。\n'
            '\n'
            '## 推理呈现纪律（提升 prose 质感与可信度）\n'
            '- 推理过程要“演”不要“报”：让视角人物通过【具体的物理动作】触碰证据（用指腹蹭过断口、把两张纸并排对光、数粉尘的层次），在动作中带出推断，而非大段内心独白罗列逻辑。\n'
            '- 一次只递进一步：每揭示一个推断，先给读者看到那个【可感知的物证细节】，再给人物的一句反应或一个动作，避免把三四个结论挤在一段里连珠炮式抛出。\n'
            '- 关键转折点的“对峙/试探”场景，多用对白与停顿（沉默、回避、答非所问）承载信息差，少用叙述代述。'
        ),
    },
}

def _build_write_system(
    preset: str,
    chapter_words: int,
    chapter_num: int,
    title: str,
    aesthetic: str,
) -> str:
    """Assemble writer system prompt: shared base + genre delta.

    用轻量 XML 分区（<角色>/<写作纪律>/<结构与输出>）分隔"你是谁/怎么写/输出什么"三块——
    研究表明分隔标签能提升长上下文里的指令跟随（对弱跟随写手尤甚）。区内仍用 ## 小节。
    """
    gp = GENRE_PROFILES.get(preset, GENRE_PROFILES["history"])
    role_part = f'你是一位{gp["role"]}。\n用中文写作本章。'
    # 写作纪律区：自检 + 核心纪律 + 感官/对话 + 时间标记 + 禁止项 + 塌缩禁令 + 去AI味 +
    # 展示而非告知(正向) + 审美。show-don't-tell 紧邻去AI味禁令，形成"禁什么+怎么写好"闭环。
    discipline_parts = [
        _SELF_REVIEW_PREAMBLE + "\n" + gp["self_review"],
        gp["core_discipline"],
        gp.get("sensory_dialogue") or _SENSORY_DIALOGUE_DEFAULT,
        gp.get("time_marker_ban") or _TIME_MARKER_BAN_DEFAULT,
        gp["genre_bans"],
        gp.get("extras", ""),
        ANTI_FRAGMENT_BAN,
        ANTI_PITFALL_BLOCK,
        _SHOW_DONT_TELL_BLOCK,
        aesthetic,
    ]
    # 结构与输出区放最后（输出格式=最强 recency 锚：第一行标题/字数/只输出正文）。
    structure_parts = [
        gp["structure_template"],
        _OUTPUT_SECTION.format(
            chapter_words=chapter_words, chapter_num=chapter_num, title=title,
        ),
    ]
    discipline = "\n\n".join(p for p in discipline_parts if p)
    structure = "\n\n".join(p for p in structure_parts if p)
    return (
        f"<角色>\n{role_part}\n</角色>\n\n"
        f"<写作纪律>\n{discipline}\n</写作纪律>\n\n"
        f"<结构与输出>\n{structure}\n</结构与输出>"
    )

def _hook_directives_block(pkg: dict) -> str:
    """Render the 吸量包's hook_directives as an opening-writer prompt block.

    build_hook_package (package.py) 早在 bootstrap 就产出"书名/简介向读者承诺了
    哪些爽点、开篇必须兑现哪个"的落地指令，但历史上只写进 hook_package.md 从未
    注入写手 prompt（P3 断链修复）。上限 5 条 / ~600 字；缺失/畸形返回 ""。
    """
    if not isinstance(pkg, dict):
        return ""
    directives = pkg.get("hook_directives")
    if not isinstance(directives, list):
        return ""
    lines: list[str] = []
    used = 0
    for d in directives[:5]:
        d = str(d).strip()
        if not d:
            continue
        if used + len(d) > 600:
            break
        lines.append(f"- {d}")
        used += len(d)
    if not lines:
        return ""
    return (
        "## 开篇吸量指令（书名/简介已向读者承诺的爽点，前三章必须兑现）\n"
        + "\n".join(lines)
    )

SENSITIVE_WORD_AVOIDANCE_BLOCK = """## 内容分级与呈现方式（平台合规·硬性执行，最高优先级之一）
本作发布渠道带内容审核，正文过于露骨会被拦截而无法过审。写作时用**克制、含蓄、侧写、留白**的笔法处理黑暗内容——**只改呈现方式，绝不删弱情节、冲突、悬念与压迫感**。
- 暴力与伤亡：不做血腥的身体损伤特写，改写旁观者的反应、环境的变化、声音与温度的骤变、事后的痕迹与静默；用"倒下／不再动弹／再没起来／满地狼藉"这类结果性、暗示性的表达带过。
- 死亡与恐怖：不铺陈遗骸、腐坏、解剖的直观细节，改用气氛、光影、空气的凝滞、人物的战栗与心理惊惧来营造恐怖；场所与状态用偏侧写的说法（如"冷藏区／后室／失去体温的人"）。
- "吞噬/变强"设定：把核心能力写成对**能量、气息、本源、光**的汲取与消化，聚焦力量流动、身体的变化感与代价，而不是进食血肉脏器的生理过程。
- 涉性/低俗：点到为止，以情绪与张力替代露骨描写。涉政/违禁：不涉及真实政治人物、敏感时政、违禁品制法。
- 核心原则：黑暗、压迫、恐怖靠**氛围、心理、后果与感官暗示**营造，而非露骨的生理名词堆砌。宁可更克制、更留白，也不要触发审核。"""

def sensitive_word_avoidance_block(config: dict[str, Any]) -> str:
    """Content-register directive steering the model to render dark content (violence,
    death, horror, the 吞噬 power) obliquely — aftermath, sensory/psychological
    suggestion, energy-absorption framing — so a content-moderation gateway does not
    reject the chapter (sensitive_words_detected). Gated by novel.sensitive_word_avoidance.

    NOTE: deliberately category-based and positive-framed. It does NOT list explicit
    trigger nouns: echoing raw banned words into the prompt primes the model to emit
    them (observed: adding a word-list made generations fail FASTER), so we name
    categories and prescribe the oblique technique instead. Returns "" when off.
    """
    if not bool(config["novel"].get("sensitive_word_avoidance", False)):
        return ""
    return SENSITIVE_WORD_AVOIDANCE_BLOCK


# ---------------------------------------------------------------------------
# Functions inlined from writing.py
# ---------------------------------------------------------------------------

def _preflight_negative_list(
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    lookback: int = 5,
) -> dict[str, Any]:
    """Build a pre-write negative list from recent failure modes.

    Collects gate_rejects (cross-chapter fossils, adjacent repetition),
    style collapse flags (em-dash density, fragment lines), and concrete
    fossil clauses from the last N chapters to front-load avoidance directives
    BEFORE the first draft is generated, rather than discovering them only
    after a low review score.

    Returns {"items": [...], "fossils": [...], "style_warnings": [...],
             "hard_fossils": [(phrase, frac), ...]}

    `hard_fossils` is the book-cumulative subset that carries an outright ban, kept
    separate so `write_chapter` can restate it as a tail anchor. Same cache read,
    one source of truth.
    """
    if chapter_num <= 1:
        return {"items": [], "fossils": [], "style_warnings": [], "hard_fossils": []}

    items: list[str] = []
    fossils: set[str] = set()
    hard: list[tuple[str, float]] = []
    style_warnings: list[str] = []
    seen_gates: set[str] = set()

    if chapter_num >= 2:
        try:
            prev_text = read_text(chapter_path(paths, chapter_num - 1))
            prev_lines = [l.strip() for l in prev_text.split("\n") if l.strip()]
            content_lines = [l for l in prev_lines
                             if not l.startswith("第") or "章" not in l[:6]]
            if content_lines:
                prev_opener = content_lines[0][:50]
                style_warnings.append(
                    f"上一章开头是「{prev_opener}」"
                    "——本章必须用完全不同"
                    "的场景/动作/感官切入，"
                    "禁止复刻上章的首句句式"
                    "。即使故事机制需要重复"
                    "（如每天凌晨亮屏），"
                    "也要从不同角色视角、"
                    "不同身体状态、或不同"
                    "空间细节开始。"
                )
        except Exception:
            pass

    start = max(1, chapter_num - lookback)
    for ch in range(start, chapter_num):
        # Check final_review for gate_rejects
        for key in ("final_review.json", "review_round1.json", "review_round0.json"):
            data = load_checkpoint(paths, ch, key)
            if not isinstance(data, dict):
                continue

            gate_rejects = data.get("gate_rejects", [])
            if isinstance(gate_rejects, list):
                for gr in gate_rejects:
                    if not isinstance(gr, dict):
                        continue
                    gate = str(gr.get("gate", "")).strip()
                    if not gate or gate in seen_gates:
                        continue
                    seen_gates.add(gate)

                    evidence = gr.get("evidence", {})
                    if gate == "cross_chapter_repetition":
                        examples = evidence.get("examples", [])
                        if isinstance(examples, list):
                            for ex in examples[:4]:
                                clause = str(ex).strip()
                                if clause and len(clause) >= 6:
                                    fossils.add(clause)
                        items.append(
                            f"近期检测到跨章节化石句（逐字复读）；本章严禁再现以下措辞或结构相似的表达。"
                        )
                    elif gate == "adjacent_repetition":
                        metrics = evidence.get("metrics", {})
                        overlap = metrics.get("clause_overlap")
                        if overlap:
                            items.append(
                                f"Ch{ch} 大量逐字复述前章内容（overlap={overlap:.2f}）；"
                                "本章必须从新事件开始，前章场景只许一笔带过。"
                            )

            # Collect style flags
            flags = data.get("style_flags", [])
            if isinstance(flags, list):
                for flag in flags[:3]:
                    flag_text = str(flag).strip()
                    if flag_text and flag_text not in style_warnings:
                        style_warnings.append(flag_text)

            if gate_rejects or flags:
                break

    # Advisory gate flags from prior reviews — concrete patterns (AI cliches,
    # paragraph shape issues, texture problems) that the advisory gates detected.
    # These are zero-LLM checks whose results were persisted in acceptance_report;
    # reading them here gives the writer a 5-chapter lookback of specific avoidance
    # targets beyond the two gate_rejects names above.
    _ADVISORY_FLAG_KEYS = (
        "ai_flavor_health", "paragraph_shape_health",
        "intra_chapter_repetition", "prose_texture",
        "chapter_ending_strength",
    )
    for ch in range(max(1, chapter_num - lookback), chapter_num):
        for key in ("final_review.json", "review_round0.json"):
            data = load_checkpoint(paths, ch, key)
            if not isinstance(data, dict):
                continue
            for gate_key in _ADVISORY_FLAG_KEYS:
                gate_data = data.get(gate_key)
                if not isinstance(gate_data, dict):
                    continue
                for flag in (gate_data.get("flags") or [])[:3]:
                    flag_text = str(flag).strip()
                    if flag_text and flag_text not in style_warnings:
                        style_warnings.append(flag_text)
            break

    # Book-wide fossils: persistent avoid-list mined across the WHOLE book by
    # review.book_wide_fossils (cached every book_fossil_every chapters). Unlike
    # the lookback fossils above, these reflect chronic habit-stiffening over the
    # entire book, so they must be injected on EVERY chapter, not just after a
    # recent gate-reject. Includes severity (frac/chapter_count) so the writer
    # knows which fossils are most critical to avoid.
    if bool(config["novel"].get("book_fossil_enabled", True)):
        try:
            cache = paths.logs_dir / "book_fossils.json"
            if cache.exists():
                bf = json.loads(read_text(cache))
                bf_fossils = bf.get("fossils") or []
                bf_phrases = bf.get("phrases") or []
                hard_fossils = [f for f in bf_fossils if isinstance(f, dict) and f.get("frac", 0) >= 0.20]
                soft_fossils = [f for f in bf_fossils if isinstance(f, dict) and 0 < f.get("frac", 0) < 0.20]
                for f in hard_fossils[:6]:
                    ph = str(f.get("phrase", "")).strip()
                    if ph:
                        fossils.add(ph)
                        hard.append((ph, float(f.get("frac", 0) or 0)))
                        items.append(
                            "『%s』已出现在 %d 章 (%.0f%%)——硬化石，本章正文禁止出现。"
                            % (ph, f.get("chapter_count", 0), f.get("frac", 0) * 100)
                        )
                for f in soft_fossils[:8]:
                    ph = str(f.get("phrase", "")).strip()
                    if ph:
                        fossils.add(ph)
                for ph in bf_phrases[:12]:
                    ph = str(ph).strip()
                    if ph:
                        fossils.add(ph)
                if hard_fossils or bf_phrases:
                    items.append(
                        "全书高频僵化短语（机械口癖）已累积，本章起必须主动换用不同的"
                        "动作落点、感官通道与句式，严禁继续复刻下列微动作片段。"
                    )
        except Exception:
            pass

    # Cross-chapter emotional formula detection: scan recent chapters for
    # overused patterns and warn the writer to vary the emotional catalyst.
    import re as _re
    _cry_pattern = _re.compile(r'[泪哭]|眼眶.{0,4}[红湿热]|[抹擦].*泪')
    _food_cry_window = 10
    if chapter_num >= 4:
        _cry_near_food = 0
        for ch in range(max(1, chapter_num - _food_cry_window), chapter_num):
            try:
                _ct = read_text(chapter_path(paths, ch))
            except Exception:
                continue
            for _para in _ct.split("\n"):
                if _cry_pattern.search(_para) and _re.search(r'[吃尝端锅碗筷菜饭汤面]', _para):
                    _cry_near_food += 1
        if _cry_near_food >= 2:
            style_warnings.append(
                f"近{_food_cry_window}章内已出现{_cry_near_food}次'食物触发流泪/感动'场景——"
                "本章如有情绪转折，必须换用不同催化方式（沉默、行为改变、态度软化、主动帮忙），"
                "不要再写吃了她做的菜就流泪/红了眼眶。"
            )
    # "不是X，是Y" pattern density check
    _negcorr = _re.compile(r'不是.{1,15}[，,].{0,2}是')
    if chapter_num >= 6:
        _nc_count = 0
        _nc_chapters = 0
        for ch in range(max(1, chapter_num - 5), chapter_num):
            try:
                _ct = read_text(chapter_path(paths, ch))
            except Exception:
                continue
            _hits = len(_negcorr.findall(_ct))
            if _hits >= 2:
                _nc_count += _hits
                _nc_chapters += 1
        if _nc_chapters >= 3:
            style_warnings.append(
                "近5章中%d章频繁使用「不是X，是Y」否定-修正句式（共%d次）——"
                "本章严格限制此句式最多1次，换用直述、对比动作、或留白。"
                % (_nc_chapters, _nc_count)
            )

    genre_fatigue = config["novel"].get("fatigue_words", [])
    if isinstance(genre_fatigue, str):
        genre_fatigue = [w.strip() for w in genre_fatigue.split(",") if w.strip()]
    for word in genre_fatigue[:12]:
        w = str(word).strip()
        if w and w not in fossils:
            fossils.add(w)
            items.append(f"体裁疲劳词「{w}」——尽量避免或限制使用。")

    # Chapter-length variance check: when recent chapters swing wildly (CV > 0.30),
    # inject a style_warning so the writer gets a length-consistency nudge.
    if chapter_num >= 6:
        recent_lens: list[int] = []
        for ch in range(max(1, chapter_num - 5), chapter_num):
            cp = chapter_path(paths, ch)
            try:
                recent_lens.append(len(read_text(cp)))
            except Exception:
                pass
        if len(recent_lens) >= 3:
            mean_len = sum(recent_lens) / len(recent_lens)
            if mean_len > 0:
                std_len = (sum((x - mean_len) ** 2 for x in recent_lens) / len(recent_lens)) ** 0.5
                cv = std_len / mean_len
                if cv > 0.30:
                    target = int(config["novel"].get("chapter_words", 4000) or 4000)
                    style_warnings.append(
                        f"最近几章长度波动过大（最短{min(recent_lens)}字/最长{max(recent_lens)}字，CV={cv:.2f}），"
                        f"本章请控制在{max(target - 500, 1500)}-{target + 500}字之间。"
                    )

    return {
        "items": items[:10],
        "fossils": sorted(fossils)[:20],
        "style_warnings": style_warnings[:4],
        "hard_fossils": sorted(hard, key=lambda t: -t[1]),
    }


FOSSIL_TAIL_ANCHOR_MAX = 5


def fossil_tail_anchor(preflight_neg: dict[str, Any] | None,
                       config: dict[str, Any]) -> str:
    """Restate the HARD book-wide fossil ban as the writer prompt's tail anchor.

    Measured, and the measurement is what justifies the duplication: of the 20
    first-draft `gate_rejects` misses left in the library after the latching-gate
    fixes, 12 are ONE entrenched bank phrase per book, and a fresh 1..N-1
    avoid-list scan puts that phrase at **rank 0** of the mid-prompt list in every
    case. So the writer had the ban in front of it and used the phrase anyway --
    the same weak-instruction-following failure as the ability whitelist and the
    degradation-recovery directive, and the same fix: position, not wording.

    Hard-only and capped at FOSSIL_TAIL_ANCHOR_MAX on purpose; a long tail list
    dilutes the very position this block exists to exploit. `fix.rotate_fossils`
    still cleans up what slips through -- this is the half that keeps the phrase
    out of the saved text in the first place.
    """
    if not bool(config["novel"].get("fossil_tail_anchor_enabled", True)):
        return ""
    hard = (preflight_neg or {}).get("hard_fossils") or []
    if not hard:
        return ""
    lines = "\n".join(
        f"{i}. 「{ph}」（已出现在全书 {frac * 100:.0f}% 的章节）"
        for i, (ph, frac) in enumerate(hard[:FOSSIL_TAIL_ANCHOR_MAX], 1)
    )
    return (
        "\n\n## ⚠ 写作前最后确认：僵化短语绝对禁令（最高优先级，出现即作废重写）\n"
        "下列短语已在全书反复复读成机械口癖，本章正文**一次都不许出现**，"
        "同义改写也不许沿用同一个动作落点与句式——换感官通道、换身体部位、换句式结构：\n"
        f"{lines}\n"
        "交稿前用这几个短语在正文里逐个搜一遍，确认为零。"
    )


def _chapter_write_max_tokens(config: dict[str, Any]) -> int | None:
    """Generation-time length cap for chapter writing.

    The write call otherwise inherits the global api.max_tokens (often 64k+), so
    a chapter can balloon far past the target band. This bounds the writer's
    output by chapter_max_chars (which the genre profile sets per题材), sized with
    enough headroom that a complete in-band chapter is never truncated
    mid-sentence — it kills runaway over-length without cutting normal chapters.
    Returns None (no cap → global default) when disabled.

    Tune with `write_token_char_ratio` (lower = tighter, but risks truncation)
    and `write_token_margin`; or pin an absolute `write_max_tokens`.
    """
    nv = config.get("novel", {})
    if not bool(nv.get("chapter_length_cap_enabled", True)):
        return None
    try:
        explicit = int(nv.get("write_max_tokens", 0) or 0)
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return explicit
    try:
        cmax = int(nv.get("chapter_max_chars", 7000))
    except (TypeError, ValueError):
        cmax = 3600
    ratio = float(nv.get("write_token_char_ratio", 1.15))
    margin = int(nv.get("write_token_margin", 300))
    return max(int(cmax * ratio) + margin, 1200)


ABSTRACT_BEAT_MARKERS = (
    "推导出",
    "意识到",
    "想通",
    "完成",
    "还原",
    "引导",
    "心算",
    "反应过来",
    "发现",
    "确认",
    "证明",
    "判断",
    "说服",
    "揭示",
)


CONCRETE_BEAT_MARKERS = (
    "把",
    "按",
    "压",
    "递",
    "翻",
    "拿",
    "写",
    "锁",
    "摁",
    "贴",
    "拆",
    "划",
    "量",
    "照",
    "指",
    "举",
    "撕",
    "收",
    "扣",
    "放",
    "打开",
    "合上",
    "签",
    "盖",
    "查",
    "对照",
    "并排",
)


def _beat_needs_concretization(beat: str) -> bool:
    """Heuristic: abstract realization verbs need an object/action anchor."""
    text = str(beat or "").strip()
    if not text:
        return False
    has_abstract = any(marker in text for marker in ABSTRACT_BEAT_MARKERS)
    has_concrete = any(marker in text for marker in CONCRETE_BEAT_MARKERS)
    return has_abstract and not has_concrete


def _first_draft_execution_ledger(config: dict[str, Any], plan: dict[str, Any]) -> str:
    """Return a compact beat-to-page execution checklist for the writer prompt."""
    novel_cfg = config.get("novel", {}) if isinstance(config, dict) else {}
    if not bool(novel_cfg.get("first_draft_execution_ledger", True)):
        return ""
    beats = plan.get("beats") if isinstance(plan, dict) else None
    if not isinstance(beats, list):
        return ""
    beat_list = [str(b).strip() for b in beats if str(b).strip()]
    if not beat_list:
        return ""

    chapter_words = int(novel_cfg.get("chapter_words", 4000) or 4000)
    per_beat = max(260, int(chapter_words / max(1, len(beat_list)) * 0.75))
    lines = [
        "### 首稿页面执行账本（内部执行，不要输出账本）",
        "- 写作前先把每个 beat 映射成：上一拍后果 -> 角色当下目标 -> 阻力/对手动作 -> 可见动作或有攻防的对话 -> 新信息/代价/局势变化。",
        "- 每个 beat 至少占一个有场面功能的自然段或对话回合；禁止把两个以上关键 beat 压缩成一句总结。",
        f"- 节奏预算：本章 {len(beat_list)} 个 beat，平均每个关键 beat 约 {per_beat}-{per_beat + 220} 字；第一个 beat 必须在前 1/3 之前进入冲突或行动。",
        "- 转场只写因果，不写流水账时间标签；下一场必须由上一场的后果推出来。",
        "- 【细节保真·最高优先级】beat 里写明的每一个具体动作（谁的手做了什么）、具体物件、具体数字、具体动机，都是本章验收项，必须在正文里把该动作/物件本身实演出来；"
        "严禁用它的“结果”或“声音”替代动作本身（例如 beat 写“她另一只手在药箱搭扣上摸了一下”，正文只写“搭扣发出一声轻响”即判不合格——必须写出“摸”这个动作和沈澜看到的手），"
        "严禁用“一笔带过/读了也读不出/总结一句”抹掉 beat 里要求的内心挣扎或动机铺垫。删一个具体细节就少一分。",
    ]
    # Per-beat enumeration intentionally removed: each beat (with its concrete
    # acceptance details) is re-stated once in the tail-of-prompt 验收清单 built
    # by write_chapter, where recency makes it actually bind. Duplicating the
    # list here diluted that anchor and roughly doubled the beat token cost.
    return "\n".join(lines) + "\n"


def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    if len(chapter.strip()) < 500:
        raise RuntimeError(
            f"Refusing to save Ch{chapter_num}: only {len(chapter.strip())} chars "
            f"(likely provider refusal or empty response). Preview: {chapter[:200]!r}"
        )
    write_text(chapter_path(paths, chapter_num), chapter)
    append_text(paths.book, "\n\n" + chapter)
    # Incrementally index the saved chapter for retrieval (RAG). Best-effort.
    try:
        from engine.retrieval import index_chapter

        index_chapter(paths, chapter_num, chapter)
    except Exception:
        pass
    append_text(
        paths.logs_dir / "reviews.jsonl",
        json.dumps(
            {
                "chapter": chapter_num,
                "score": review.get("score"),
                "readthrough_score": review.get("readthrough_score"),
                "hook_score": review.get("hook_score"),
                "payoff_score": review.get("payoff_score"),
                "novelty_score": review.get("novelty_score"),
                "prose_score": review.get("prose_score"),
                "continuity_score": review.get("continuity_score"),
                "accepted": review.get("accepted"),
                "problems": review.get("problems", []),
                "continuity_risks": review.get("continuity_risks", []),
                "plan_title": plan.get("title"),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )


def append_memory(path: Path, chapter_num: int, items: list[Any]) -> None:
    if not items:
        return
    existing = read_text(path)
    section_header = f"## Ch{chapter_num}"
    if section_header in existing:
        return
    existing_bullets = set()
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            existing_bullets.add(stripped[2:].strip())
    fresh = []
    for item in items:
        text = str(item).strip()
        if not text or text in existing_bullets:
            continue
        fresh.append(text)
        existing_bullets.add(text)
    if not fresh:
        return
    append_text(path, f"\n\n{section_header}\n" + "\n".join(f"- {t}" for t in fresh) + "\n")


def update_structured_state(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    review: dict[str, Any],
    decision: dict[str, Any],
    plan: dict[str, Any] | None = None,
) -> None:
    db_event(conn, chapter_num, "chapter_extraction", extraction)

    for event in extraction.get("events", []):
        db_event(conn, chapter_num, "story_event", event)

    for entity in extraction.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type", "unknown"))
        name = str(entity.get("name", "unknown"))
        with db_lock():
            old = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
        state = json.loads(old["state_json"]) if old else {}
        patch = entity.get("state_patch") or {}
        if isinstance(patch, dict):
            state.update(patch)
        else:
            state["note"] = str(patch)
        with db_lock():
            conn.execute(
                """
                INSERT INTO entities(entity_type, name, state_json, updated_chapter)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_type, name)
                DO UPDATE SET state_json=excluded.state_json, updated_chapter=excluded.updated_chapter
                """,
                (entity_type, name, json.dumps(state, ensure_ascii=False), chapter_num),
            )

    def _as_chnum(v: Any) -> int | None:
        # chapter-number columns must bind as int/None; LLM may emit a dict/list/str.
        if isinstance(v, bool) or v is None:
            return None
        if isinstance(v, int):
            return v
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return None

    for thread in extraction.get("threads", []):
        if not isinstance(thread, dict):
            continue
        thread_id = str(thread.get("id") or f"ch{chapter_num}-{abs(hash(json.dumps(thread, ensure_ascii=False))) % 100000}")
        if str(thread.get("thread_type", "plot")) == "reader_promise":
            promise = dict(thread)
            promise["id"] = thread_id
            promise.setdefault("opened_chapter", thread.get("introduced_chapter", chapter_num))
            upsert_reader_promise(conn, chapter_num, promise)
        _payload = thread.get("payload")
        _depends = str(thread.get("depends_on", "") or "").strip()
        _priority = int(thread.get("priority", 5) or 5)
        _half_life = int(thread.get("half_life", 0) or 0)
        with db_lock():
            conn.execute(
                    """
                    INSERT INTO open_threads(id, description, status, thread_type, introduced_chapter, due_chapter, updated_chapter, payload_json, depends_on, priority, half_life)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id)
                    DO UPDATE SET description=excluded.description, status=excluded.status,
                                  thread_type=excluded.thread_type,
                                  due_chapter=excluded.due_chapter, updated_chapter=excluded.updated_chapter,
                                  payload_json=excluded.payload_json,
                                  depends_on=excluded.depends_on, priority=excluded.priority, half_life=excluded.half_life
                    """,
                    (
                        thread_id,
                        str(thread.get("description", "")),
                        str(thread.get("status", "open")),
                        str(thread.get("thread_type", "plot")),
                        _as_chnum(thread.get("introduced_chapter")),
                        _as_chnum(thread.get("due_chapter")),
                        chapter_num,
                        json.dumps(_payload if isinstance(_payload, (dict, list)) else {}, ensure_ascii=False),
                        _depends,
                        _priority,
                        _half_life,
                    ),
                )

    metrics = extraction.get("metrics") or {}
    # payoff_type / conflict_type are the arbiter's deliberate, bandit-varied plan
    # intent. Prefer them over the extraction model's re-classification of the
    # written prose: a cheap extraction model (deepseek-flash) collapses every
    # chapter to payoff_type='reveal' regardless of the (varied) plan, which then
    # fires false payoff-monotony penalties downstream (yeban_guize Ch5/7). Fall
    # back to the extraction value only when the plan didn't declare one.
    _plan = plan or {}
    _plan_payoff_type = str(_plan.get("payoff_type") or "").strip() or None
    _plan_conflict_type = str(_plan.get("conflict_type") or "").strip() or None
    _sh = review.get("style_health") or {}
    _sh_metrics = _sh.get("metrics") or {}
    _af = review.get("ai_flavor_health") or {}
    _af_metrics = _af.get("metrics") or {}
    metrics_row = {
        "chapter": chapter_num,
        "title": extraction.get("title"),
        "score": safe_score(review.get("score", 0)),
        "readthrough_score": safe_score(review.get("readthrough_score", 0)),
        "hook_score": safe_score(review.get("hook_score", review.get("hook_strength", 0))),
        "payoff_score": safe_score(review.get("payoff_score", 0)),
        "novelty_score": safe_score(review.get("novelty_score", 0)),
        "prose_score": safe_score(review.get("prose_score", review.get("aesthetic_score", 0))),
        "continuity_score": safe_score(review.get("continuity_score", 0)),
        "plan_score": plan_score(decision),
        "payoff_type": _plan_payoff_type or metrics.get("payoff_type"),
        "conflict_type": _plan_conflict_type or metrics.get("conflict_type"),
        "tension": metrics.get("tension"),
        "novelty": metrics.get("novelty"),
        "hook_strength": metrics.get("hook_strength"),
        "emotional_tone": metrics.get("emotional_tone"),
        "accepted": 1 if review.get("accepted") else 0,
        "em_dash_per_kchar": _sh_metrics.get("em_dash_per_kchar"),
        "style_penalty": _sh.get("penalty"),
        "emotional_impact": safe_score(review.get("emotional_impact", 0)),
        # 反过度书写锚点指标（趋势项/回放/退化诊断读取）。
        "avg_sentence_chars": _sh_metrics.get("avg_sentence_chars"),
        "dialogue_char_ratio": _sh_metrics.get("dialogue_char_ratio"),
        "tech_per_kchar": _sh_metrics.get("tech_per_kchar"),
        "genre_score": (review.get("genre_adherence") or {}).get("genre_score"),
        # AI味确定性检测指标（_prewrite_quality_contract AI味预算读取）。
        "ai_cliche_per_kchar": _af_metrics.get("ai_cliche_per_kchar"),
        "metaphor_per_kchar": _af_metrics.get("metaphor_per_kchar"),
        "tell_not_show_per_kchar": _af_metrics.get("tell_not_show_per_kchar"),
        "adverb_per_kchar": _af_metrics.get("adverb_per_kchar"),
        "ai_flavor_penalty": _af.get("penalty"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with db_lock():
        conn.execute(
            """
            INSERT INTO chapter_metrics(
                chapter, title, score, readthrough_score, hook_score, payoff_score,
                novelty_score, prose_score, continuity_score, plan_score, payoff_type, conflict_type, tension,
                novelty, hook_strength, emotional_tone, accepted, em_dash_per_kchar, style_penalty,
                emotional_impact, avg_sentence_chars, dialogue_char_ratio, tech_per_kchar, genre_score,
                ai_cliche_per_kchar, metaphor_per_kchar, tell_not_show_per_kchar, adverb_per_kchar, ai_flavor_penalty,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
                title=excluded.title,
                score=COALESCE(NULLIF(excluded.score, 0), score),
                readthrough_score=excluded.readthrough_score, hook_score=excluded.hook_score,
                payoff_score=excluded.payoff_score, novelty_score=excluded.novelty_score,
                prose_score=excluded.prose_score, continuity_score=excluded.continuity_score,
                plan_score=excluded.plan_score,
                payoff_type=excluded.payoff_type, conflict_type=excluded.conflict_type,
                tension=excluded.tension, novelty=excluded.novelty, hook_strength=excluded.hook_strength,
                emotional_tone=excluded.emotional_tone, accepted=excluded.accepted,
                em_dash_per_kchar=excluded.em_dash_per_kchar, style_penalty=excluded.style_penalty,
                emotional_impact=excluded.emotional_impact,
                avg_sentence_chars=excluded.avg_sentence_chars,
                dialogue_char_ratio=excluded.dialogue_char_ratio,
                tech_per_kchar=excluded.tech_per_kchar,
                genre_score=excluded.genre_score,
                ai_cliche_per_kchar=excluded.ai_cliche_per_kchar,
                metaphor_per_kchar=excluded.metaphor_per_kchar,
                tell_not_show_per_kchar=excluded.tell_not_show_per_kchar,
                adverb_per_kchar=excluded.adverb_per_kchar,
                ai_flavor_penalty=excluded.ai_flavor_penalty
            """,
            (
                metrics_row["chapter"],
                metrics_row["title"],
                metrics_row["score"],
                metrics_row["readthrough_score"],
                metrics_row["hook_score"],
                metrics_row["payoff_score"],
                metrics_row["novelty_score"],
                metrics_row["prose_score"],
                metrics_row["continuity_score"],
                metrics_row["plan_score"],
                metrics_row["payoff_type"],
                metrics_row["conflict_type"],
                metrics_row["tension"],
                metrics_row["novelty"],
                metrics_row["hook_strength"],
                metrics_row["emotional_tone"],
                metrics_row["accepted"],
                metrics_row["em_dash_per_kchar"],
                metrics_row["style_penalty"],
                metrics_row["emotional_impact"],
                metrics_row["avg_sentence_chars"],
                metrics_row["dialogue_char_ratio"],
                metrics_row["tech_per_kchar"],
                metrics_row["genre_score"],
                metrics_row["ai_cliche_per_kchar"],
                metrics_row["metaphor_per_kchar"],
                metrics_row["tell_not_show_per_kchar"],
                metrics_row["adverb_per_kchar"],
                metrics_row["ai_flavor_penalty"],
                metrics_row["created_at"],
            ),
        )
        conn.commit()

    updates = extraction.get("memory_updates") or {}
    # LLM extraction JSON: memory_updates may come back malformed (a bare string
    # instead of a dict, or a per-key value that isn't a list). Guard so finalize
    # can't crash here — a crash leaves chapter_completed.json unwritten and wedges
    # resume in an endless "Resuming partially indexed Ch{n}" loop.
    if not isinstance(updates, dict):
        updates = {}
    def _as_list(v: Any) -> list[Any]:
        return v if isinstance(v, list) else []
    append_memory(paths.bible, chapter_num, _as_list(updates.get("bible")))
    append_memory(paths.characters, chapter_num, _as_list(updates.get("characters")))
    append_memory(paths.timeline, chapter_num, _as_list(updates.get("timeline")))
    append_memory(paths.threads, chapter_num, _as_list(updates.get("threads")))

    _cl = extraction.get("causal_links")
    store_causal_links(conn, chapter_num, _cl if isinstance(_cl, list) else [])

    # Relationship changes extracted from this chapter
    try:
        from engine.store import upsert_relationship
        for rc in extraction.get("relationship_changes", []):
            if not isinstance(rc, dict):
                continue
            ca = str(rc.get("char_a", "")).strip()
            cb = str(rc.get("char_b", "")).strip()
            if not ca or not cb:
                continue
            delta = float(rc.get("intensity_delta", 0) or 0)
            upsert_relationship(
                conn, chapter_num, ca, cb,
                stage=str(rc.get("new_stage", "")),
                event_desc=str(rc.get("event", ""))[:120],
            )
    except Exception:
        pass

    # Info revelation tracking
    try:
        from engine.store import upsert_info_revelation
        for ir in extraction.get("info_revelations", []):
            if not isinstance(ir, dict):
                continue
            upsert_info_revelation(conn, chapter_num, ir)
    except Exception:
        pass

    # Dialogue fingerprint persistence
    try:
        fingerprints = extraction.get("dialogue_fingerprints", [])
        if fingerprints and isinstance(fingerprints, list):
            fp_path = paths.memory_dir / "dialogue_fingerprints.json"
            existing_fp: dict[str, Any] = {}
            if fp_path.exists():
                try:
                    existing_fp = json.loads(fp_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            changed = False
            for fp in fingerprints:
                if not isinstance(fp, dict):
                    continue
                name = str(fp.get("character", "")).strip()
                style = str(fp.get("speaking_style", "")).strip()
                if name and style:
                    existing_fp[name] = {"style": style, "updated_chapter": chapter_num}
                    changed = True
            if changed:
                fp_path.write_text(json.dumps(existing_fp, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _render_state_md_template(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    protagonist_state: str,
    next_directions: list[str],
) -> str:
    """Compose the new state.md deterministically.

    The structure follows what readers expect: progress meta, recent chapter
    summaries (5), key entity states, active threads (open), and the LLM-only
    sections (protagonist_state, next_12_directions).
    """
    from engine.store import recent_metrics

    total_chars = count_chars(paths.book)
    metrics = recent_metrics(conn, 5)
    threads_text = read_text(paths.threads).strip()

    # Last 5 chapter title+key payoff
    summary_lines: list[str] = []
    for m in metrics:
        ch = m.get("chapter")
        title = m.get("title") or ""
        score = m.get("score")
        tone = m.get("emotional_tone") or ""
        payoff = m.get("payoff_type") or ""
        summary_lines.append(f"- Ch{ch} 「{title}」 score={score} payoff={payoff} tone={tone}")

    # Pull events from this chapter's extraction
    this_chapter_events: list[str] = []
    for ev in extraction.get("events", [])[:8]:
        s = str(ev.get("summary", "")).strip()
        if s:
            this_chapter_events.append(f"- {s[:200]}")

    next_dir_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(next_directions[:12]))

    parts: list[str] = [
        f"# 第{chapter_num}章后状态快照",
        f"\n## 进度\n- 总字数：{total_chars}\n- 最新章节：Ch{chapter_num} 「{extraction.get('title', '')}」",
        "\n## 近期章节（最新在前）\n" + ("\n".join(summary_lines) if summary_lines else "_(无)_"),
        "\n## 最新章节关键事件\n" + ("\n".join(this_chapter_events) if this_chapter_events else "_(无)_"),
        "\n## 主角状态\n" + (protagonist_state.strip() or "_(空)_"),
        "\n## 接下来12章方向\n" + (next_dir_lines or "_(无)_"),
        "\n## 活跃伏线\n" + (threads_text[:4000] if threads_text else "_(无)_"),
    ]
    return "\n".join(parts) + "\n"


def update_state_file(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    extraction: dict[str, Any],
) -> None:
    """Render state.md deterministically from the extraction.

    The two dynamic sections (protagonist_state / next_12_directions) ride in the
    extraction JSON itself — extract_events is the single per-chapter state LLM
    call. No LLM here.
    """
    if paths.state.exists():
        shutil.copy2(paths.state, paths.state.with_suffix(".md.bak"))

    protagonist_state = str(extraction.get("protagonist_state", "")).strip()
    raw_dirs = extraction.get("next_12_directions") or []
    next_directions = [str(d).strip() for d in raw_dirs if str(d).strip()] if isinstance(raw_dirs, list) else []
    new_state = _render_state_md_template(
        paths, conn, chapter_num, extraction, protagonist_state, next_directions
    )
    write_text(paths.state, new_state)
