"""v2 writing: one call returns the chapter AND the state change it caused.

REDESIGN_V2 §3.3. v1 spends two calls per chapter here — `write_chapter` for the
prose and `extract_events` for the persistence delta — and the second one re-reads
the chapter it just wrote in order to answer questions the writer already knew the
answer to. v2 asks for both in one response: prose first, a sentinel line, then a
bare JSON object carrying exactly `canon.ChapterDelta`'s five fields.

Four decisions in here are load-bearing, and each of them was a fork in the road:

**1. The prose doctrine is v1's, unchanged.** `GENRE_PROFILES`, the shared
constants, `ANTI_FRAGMENT_BAN`, the aesthetic presets — all of it comes from
`writing._build_write_system`. Rewriting style teaching would make the A/B a test
of two prompt libraries instead of a test of two architectures, and the prose half
of v1 is not what the redesign claims is broken.

**2. The v1 output section is REPLACED, not appended to.** It ends with 「只输出章
节正文…严禁输出…JSON」, which is the exact opposite of what v2 asks for. Appending
an override would leave two contradictory instructions in one prompt for a writer
already documented as a weak instruction-follower (LESSONS: the ability capsule,
the recovery directive, the fossil tail anchor are all position hacks for this same
model). It would fail *quietly* — the model would obey the older, more emphatic
rule, v2 would parse no delta, and `run.py` would spend an extraction call per
chapter forever, corrupting the very cost number the A/B exists to measure. So the
swap is an exact-string replacement that RAISES when the string is not found, and
`tests/test_v2_write.py` pins the contract.

**3. Prose first, JSON last, and a parse failure loses the delta — never the
prose.** Prose first because the exit hook is the single most important sentence in
a web-novel chapter and it must be the last prose the model writes, not something
it hurries past on the way to a JSON object. And because a 5000-char chapter inside
a JSON string is a 5000-char escaping problem: one unescaped quote and the whole
chapter is unrecoverable. Split first, normalize the prose, parse the tail; if the
tail is junk, the chapter still stands and `run.py` decides what a missing delta is
worth.

**4. The acceptance checklist names the literal fragments the gate greps for.**
`accept.contract_fulfilment` is a string matcher over `quality._beat_anchor_fragments`.
Telling the writer 「把转折写出来」 and then grading it on whether 「钥匙」 appears is
grading a different question than the one asked. So the checklist is generated from
the same `_anchors` call the gate uses — one ruler, quoted to both sides.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Iterable, Sequence

import writing
from config import Paths, log, normalize_chapter
from llm import call_llm, load_json_with_repair, safe_json_loads
from v2 import accept, canon


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

# The five keys `canon.ChapterDelta` reads. Used to tell a real delta from a JSON
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
    """v1's writer system prompt with ONE section swapped. See docstring rule 2."""
    novel = config["novel"]
    preset = str(novel.get("style_preset", "history"))
    chapter_words = int(novel.get("chapter_words", 4000) or 4000)
    system = writing._build_write_system(
        preset,
        chapter_words=chapter_words,
        chapter_num=chapter_num,
        title=title,
        aesthetic=writing.AESTHETIC_PRESETS.get(preset, writing.AESTHETIC_HISTORY),
    )
    v1_section = writing._OUTPUT_SECTION.format(
        chapter_words=chapter_words, chapter_num=chapter_num, title=title)
    found = system.count(v1_section)
    if found != 1:
        raise WriteError(
            "v2 cannot assemble a writer prompt: `writing._OUTPUT_SECTION` appears "
            f"{found} times in the assembled system prompt, expected exactly 1. "
            "v2 must REPLACE that section (it forbids the JSON half of the "
            "response); appending an override would ship two contradictory "
            "instructions and silently cost an extraction call per chapter. "
            "Fix `v2/write.build_system` to match the new assembly.")
    system = system.replace(
        v1_section, v2_output_section(chapter_words, chapter_num, title))
    try:
        swa = writing.sensitive_word_avoidance_block(config)
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
    cfg = (config or {}).get("novel", {}) if config else {}
    tail = int(cfg.get("ccc_tail_chars", accept.DEFAULT_TAIL_CHARS) or
               accept.DEFAULT_TAIL_CHARS)

    lines: list[str] = []

    def anchored(field: str, target: str, note: str = "") -> None:
        anchors = accept._anchors(target)
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
        import store

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
    """The pre-write avoid list, from `writing._preflight_negative_list`.

    The *measurement* is v1's and is imported, not re-derived — it reads the same
    gate rejects and the same `logs/book_fossils.json` cache. Only the rendering
    is v2's, and it is shorter on purpose: v1's version escalates through three
    warning tiers and repeats itself, which is prompt weight spent on emphasis
    rather than on information. The hard fossils get their own tail anchor
    anyway (`writing.fossil_tail_anchor`), which is the position that measurably
    works.
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
    lo = int(novel.get("chapter_min_chars", 2800) or 2800)
    hi = int(novel.get("chapter_max_chars", 7000) or 7000)
    return (f"## 本章字数区间（硬性约束）\n"
            f"正文目标约 {target} 字，必须落在 {lo}-{hi} 字之间。"
            f"低于下限会被判过短、高于上限会被判超长，两种都要返工。"
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
    state: canon.StoryState,
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
) -> tuple[str, canon.ChapterDelta, str]:
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
        return prose, canon.ChapterDelta(), "missing"
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
        return prose, canon.ChapterDelta(), \
            "unparsed" if status != "ok" else "missing"
    return prose, canon.ChapterDelta.from_payload(data), status


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WriteResult:
    text: str
    delta: canon.ChapterDelta
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
        from retrieval import exemplar_block

        return exemplar_block(paths, conn, config, plan, chapter_num)
    except Exception:
        return ""


def _preflight(paths: Paths, conn: Any, config: dict[str, Any],
               chapter_num: int) -> dict[str, Any] | None:
    if not bool(config["novel"].get("preflight_constraints_enabled", True)):
        return None
    try:
        return writing._preflight_negative_list(
            paths, conn, config, chapter_num,
            lookback=int(config["novel"].get("preflight_constraints_lookback", 5)))
    except Exception:
        return None


def _capsule(paths: Paths, config: dict[str, Any]) -> str:
    try:
        from memory import contract_capsule

        return contract_capsule(paths, config)
    except Exception:
        return ""


# `writing._chapter_write_max_tokens` sizes the response for prose ALONE —
# `chapter_max_chars * ratio + margin`. Under v2 the same response also has to
# carry the delta, and a delta that runs out of tokens is indistinguishable from
# a model that ignored the instruction: `delta_status` reads `unparsed`, run.py
# falls back to an extraction call, and v2 quietly pays v1's price. So the cap
# gets explicit room for it. Sized from `DELTA_SCHEMA`: five fields, a handful of
# rows each, CJK at roughly one token per character.
DELTA_TOKEN_HEADROOM = 1200


def max_tokens(config: dict[str, Any]) -> int | None:
    base = writing._chapter_write_max_tokens(config)
    return None if base is None else base + DELTA_TOKEN_HEADROOM


def write_chapter(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    card: dict[str, Any],
    state: canon.StoryState,
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
        tail_anchor=writing.fossil_tail_anchor(preflight, config),
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
) -> tuple[canon.ChapterDelta, str]:
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
        return canon.ChapterDelta(), "missing"
    user = (f"章节正文（第 {chapter_num} 章）：\n\n{body}\n\n"
            f"按下列结构输出这一章的状态增量：\n{DELTA_SCHEMA}")
    try:
        raw = call(client, paths, config, BACKFILL_SYSTEM, user,
                   temperature=0.2, json_mode=True, tag="delta_backfill")
    except Exception as exc:
        log(paths, f"v2.delta_backfill Ch{chapter_num} call failed: {exc}")
        return canon.ChapterDelta(), "missing"
    try:
        data = safe_json_loads(_strip_fence(str(raw or "")))
    except Exception:
        data = None
    if not _looks_like_delta(data):
        log(paths, f"v2.delta_backfill Ch{chapter_num} returned no usable delta")
        return canon.ChapterDelta(), "missing"
    delta = canon.ChapterDelta.from_payload(data)
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
]
