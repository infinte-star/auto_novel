"""The decidable acceptance set, contract fulfilment (CCC), and cite-or-drop.

REDESIGN_V2 §3.4 ①②. Three rules govern this module:

**1. The acceptance set is defined by the ruler, not by taste.**  A gate belongs
here exactly when `quality.hard_block_reasons` — the release rule that
`tools/fpy_prime.py` replays to settle every engine A/B — can read its output and
declare the draft a write-off.  Choosing the members any other way would let v2
"win" by not measuring something v1 measures.  `ACCEPTANCE_GATES` is checked
against that criterion by `tests/test_v2_accept.py`, and every excluded blocking
gate must give its reason in `NOT_IN_ACCEPTANCE`.

**2. Every member is decidable with zero LLM calls, and every member is
actionable.**  `quality.REGISTRY.may_block()` enforces the second half: a
book-cumulative quantity may advise but never reject, because no rewrite of the
current chapter can lower it (LESSONS §13; the three latching gates cost 4.0pt of
library FPY').

**3. The output is a v1-schema review payload.**  `acceptance_report` returns the
same keys `review.py` writes — `gate_rejects`, `style_health`, `length_band`,
`opening_hook_gate`, `adjacent_repetition`, `contract_violations` — so both
engines are settled by the same `fpy_prime` invocation with no tool changes at
all.  v2-only findings ride along under `contract_fulfilment` / `citations`,
which the ruler ignores; they turn into `gate_rejects` entries explicitly, where
the ruler can see them.

The two v2-native checks:

  `contract_fulfilment(card, text)` — CCC.  Did the prose actually stage what the
  ChapterCard promised?  Zero LLM: the card's fields are concrete by
  construction (`arc.ARC_SYSTEM` rule 2 forbids abstract intent), so their
  anchors either appear on the page or they do not.  This is the measured
  successor to `quality.beat_coverage`, widened from `beats` to the whole card.

  `citation_check(claims, text)` — cite-or-drop.  A review finding that cannot
  point at a substring of the chapter it is judging is discarded, not weighed.
  This is the only defence against a reviewer inventing a violation, and unlike a
  confidence score it is decidable.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import arc
import quality
import store
from quality import REGISTRY, hard_block_reasons

# ---------------------------------------------------------------------------
# The acceptance set
# ---------------------------------------------------------------------------

# Gates whose verdict `quality.hard_block_reasons` reads. Membership is a fact
# about the ruler, not a preference -- see rule 1 in the module docstring.
ACCEPTANCE_GATES: tuple[str, ...] = (
    "style_health",               # -> style_collapse(penalty=...)
    "cross_chapter_repetition",   # -> gate_rejects
    "book_wide_fossils",          # -> gate_rejects
    "descriptor_frequency",       # -> gate_rejects
    "genre_adherence",            # -> gate_rejects (only if genre_drift_reject_enabled)
    "adjacent_repetition",        # -> adjacent_repeat_block
    "length_band_check",          # -> length_band_block
    "opening_hook_gate",          # -> opening_gate_block
)

# The two v2-native members, which have no v1 gate function behind them.
NATIVE_CHECKS: tuple[str, ...] = ("contract_fulfilment", "citation_check")

# Blocking-capable gates deliberately left out, each with the reason. A new
# `may_block` gate must be classified into ACCEPTANCE_GATES or into this dict --
# the test suite fails on an unclassified one, so the set cannot silently drift.
NOT_IN_ACCEPTANCE: dict[str, str] = {
    "beat_coverage":
        "superseded by contract_fulfilment, which asks the same question "
        "(did the prose stage what was promised) against the whole card rather "
        "than the `beats` list alone. Running both would double-charge.",
    "dialogue_health":
        "the ruler never reads it: its verdict is a penalty, and review.py "
        "never turns it into a gate_reject. Stays an L1 repair trigger.",
    "intra_chapter_repetition":
        "same -- penalty only, no path into hard_block_reasons. Advisory: it "
        "declared repair=L1 for months with no fix.ACTION_BY_GATE entry, and at "
        "1 firing in 638 archived chapters a fixer is not worth writing.",
    "hook_tail_repetition":
        "penalty only, and its thresholds have never been validated against a "
        "live BLOCKING distribution. Giving an unmeasured threshold blocking "
        "power is the dead-key defect. Advisory, same false-repair-layer story.",
    "scene_similarity": "card-phase; see CARD_GATES.",
    "narrative_pattern_repetition": "card-phase; see CARD_GATES.",
    "plan_visual_payoff_check": "card-phase; see CARD_GATES.",
    "plan_executability_gate": "card-phase; see CARD_GATES.",
}

# Gates that judge the CARD, before a word is written. Blocking here is nearly
# free -- the cost is one card repair, not a discarded chapter -- which is why
# `scope="card"` may block even though `scope="book"` may not.
CARD_GATES: tuple[str, ...] = (
    "scene_similarity",
    "narrative_pattern_repetition",
    "plan_visual_payoff_check",
    "plan_executability_gate",
)


# ---------------------------------------------------------------------------
# CCC -- contract fulfilment
# ---------------------------------------------------------------------------

# Card fields whose absence means the chapter did not write the card at all.
# Deliberately SHORT. Every hard field must satisfy two conditions: the card
# states it as a concrete thing (so anchors exist), and a chapter that missed it
# can fix it by mentioning the thing. `payoff` and `beats` are excluded on
# purpose -- they are the fields most likely to be phrased as intent, and a
# blocking gate built on an unanchorable target is the latching defect again.
HARD_FIELDS: tuple[str, ...] = ("where", "turn", "exit_hook")

# How much of the chapter's tail counts as "the ending" for `exit_hook`.
DEFAULT_TAIL_CHARS = 400

# A forbid entry is breached on a verbatim hit of the whole entry, or of an
# anchor at least this long. Four, not three, because anchors are produced by
# splitting on particles: 「用月光渲染悲伤」 yields 月光 / 渲染 / 悲伤, and charging a
# violation because the chapter contains 月光 would be a false reject -- which
# costs a rework, whereas a missed forbid costs one advisory line. Forbid
# entries are drawn from the used-element ledger (`arc.ARC_SYSTEM` rule 7), i.e.
# they are real phrases the book already used, so verbatim is the common case.
FORBID_MIN_ANCHOR = 4

# Markers that turn a `forbid` entry into an OBLIGATION rather than a ban, and the
# markers that keep it a ban despite them. Both lists are needed: 「必须避免器械报告
# 体」 contains an obligation marker and is still a ban, so obligation alone cannot
# decide it. A prohibition marker anywhere in the entry wins.
OBLIGATION_MARKERS: tuple[str, ...] = ("必须", "务必", "应当", "应该", "需要", "要求")
PROHIBITION_MARKERS: tuple[str, ...] = (
    "禁止", "严禁", "不要", "不得", "不可", "不能", "不准", "避免", "别", "勿", "忌",
)


def _is_misfiled_requirement(entry: str) -> bool:
    """True when this `forbid` entry is actually a REQUIREMENT, not a ban.

    Measured on ts_v2arm Ch220. The arc planner filed a whole obligation into
    `forbid` — 「第219-220章终章必须落在汤记后门初雪夜…两人并肩坐在后门台阶上…与第1章
    形成首尾呼应」 — and `_anchors` lifted the bare location 「后门台阶上」 out of it.
    The chapter obeyed the requirement, sat them on the back-door steps, and was
    charged with a violation for it. Nothing that chapter can write turns the gate
    green: writing the scene breaches `forbid`, omitting it breaches the finale
    obligation the same sentence states. That is the latching defect CLAUDE.md
    forbids, and it is the fourth measured instance of the class.

    This is the pure case that `_required_text` cannot catch: there, the phrase was
    ALSO in `payoff`, so the conflict was visible by comparing two fields. Here the
    requirement exists only inside the misfiled entry, so the entry has to be read
    on its own terms.

    Deliberately a marker test rather than anything cleverer. It is decidable from
    the entry alone, and the failure directions are asymmetric: mistaking a ban for
    a requirement costs one line of missing advice, while mistaking a requirement
    for a ban costs a guaranteed first-pass failure on every chapter the card
    covers.
    """
    e = str(entry or "")
    if any(m in e for m in PROHIBITION_MARKERS):
        return False
    return any(m in e for m in OBLIGATION_MARKERS)


def _body(text: str) -> str:
    return quality._strip_title_line(str(text or ""))


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _anchors(target: str) -> list[str]:
    """The distinctive fragments a card field promises.

    Reuses `quality._beat_anchor_fragments` rather than re-deriving one: two
    anchor extractors would mean CCC and `beat_coverage` disagree about what a
    plan promised, and the CCR baseline in `tools/ccr_baseline.py` would not be
    comparable to the archived `beat_coverage` results.
    """
    return quality._beat_anchor_fragments(str(target or ""))


# Card fields that state what the chapter MUST contain. Everything on the card
# except `forbid` itself -- named positively rather than by exclusion so a new
# card field is opt-in and cannot silently start waiving bans.
REQUIRED_FIELDS: tuple[str, ...] = (
    "where", "who", "turn", "payoff", "exit_hook", "beats", "goal", "conflict",
    "title", "opening_type",
)


def _required_text(card: dict[str, Any] | None) -> str:
    """Everything this card demands, as one blob to test forbid phrases against.

    A card that requires a phrase and bans it is unresolvable, and the gate that
    reports it is the latching defect CLAUDE.md forbids: writing the phrase fails
    `forbid`, omitting it fails `payoff`, and every forced replan is a guaranteed
    first-pass failure. Measured on ts_v2arm Ch219 -- the arc planner filed a
    whole requirement into `forbid`
    (「第219-220章终章必须…陆时砚说'那我负责每晚关灯'…」), `_anchors` lifted
    「负责每晚关灯」 out of it, and the same card's `payoff` required that line
    verbatim. CCR was 1.0 and the chapter was rejected anyway; two of those in a
    row tripped the run breaker.

    The guard lives here rather than in `validate_card` on purpose. Dropping such
    an entry at card time would need to guess which half of a 「必须」 sentence is
    the ban, whereas "does this card require the phrase" is decidable from the
    card alone -- so the gate stays non-latching no matter how badly a card is
    filed.
    """
    if not isinstance(card, dict):
        return ""
    parts: list[str] = []
    for field in REQUIRED_FIELDS:
        value = card.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if isinstance(v, (str, int, float)))
    return "\n".join(parts)


def _hit(target: str, text: str, grams: set[str]) -> tuple[bool, list[str], list[str]]:
    """(hit, anchors, matched). No anchors -> unjudgeable, reported as hit=False
    with an empty anchor list so the caller can drop it from the denominator."""
    anchors = _anchors(target)
    if not anchors:
        return False, [], []
    matched = [a for a in anchors if quality._fragment_hit(a, text, grams)]
    return bool(matched), anchors, matched


def contract_fulfilment(
    card: dict[str, Any] | None,
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Did the prose stage what the ChapterCard promised? Zero LLM.

    Returns::

        {
          "enabled": bool,
          "ccr": float,          # fulfilled / judgeable   (1.0 when nothing judgeable)
          "judgeable": int, "fulfilled": int, "unjudgeable": int,
          "items": [{"field","target","verdict","anchors","matched","hard"}],
          "missing": [...],      # items with verdict "miss" -- the repair list
          "violations": [...],   # forbid entries the prose used anyway
          "forbid_conflicts": [...],  # forbid entries this card ALSO requires (waived)
          "hard_misses": [...],  # subset of missing in HARD_FIELDS
          "passed": bool,        # no hard miss and no violation
          "directives": [...],
        }

    Conservative in the same way `beat_coverage` is: a field with no extractable
    anchors is *unjudgeable*, not a miss. It leaves the CCR denominator entirely
    rather than being scored as a failure -- an abstract card field is a card
    defect, caught by `arc.validate_card`, and charging the prose for it would
    make CCC unactionable.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    enabled = bool(cfg.get("ccc_enabled", True))
    out: dict[str, Any] = {
        "enabled": enabled, "ccr": 1.0, "judgeable": 0, "fulfilled": 0,
        "unjudgeable": 0, "items": [], "missing": [], "violations": [],
        "forbid_conflicts": [],
        "hard_misses": [], "passed": True, "directives": [],
    }
    if not enabled or not isinstance(card, dict) or not card:
        return out
    body = _body(text)
    if len(body) < 500:
        # Below `save_chapter`'s floor this is a refusal, not a chapter.
        return out
    grams = _bigrams(body)
    tail = body[-int(cfg.get("ccc_tail_chars", DEFAULT_TAIL_CHARS)):]
    tail_grams = _bigrams(tail)
    hard_fields = set(HARD_FIELDS)

    def add(field: str, target: str, hit: bool, anchors: list[str], matched: list[str]):
        verdict = "hit" if hit else ("miss" if anchors else "unjudgeable")
        item = {"field": field, "target": target, "verdict": verdict,
                "anchors": anchors, "matched": matched,
                "hard": field in hard_fields}
        out["items"].append(item)
        if verdict == "unjudgeable":
            out["unjudgeable"] += 1
            return
        out["judgeable"] += 1
        if hit:
            out["fulfilled"] += 1
        else:
            out["missing"].append(item)
            if item["hard"]:
                out["hard_misses"].append(item)

    # where / turn / payoff -- anywhere in the body.
    for field in ("where", "turn", "payoff"):
        target = str(card.get(field) or "").strip()
        if target:
            add(field, target, *_hit(target, body, grams))

    # who -- each name separately; names are short proper nouns, so a plain
    # substring is the right test and anchor-splitting would shred them.
    for name in (card.get("who") or []):
        name = str(name or "").strip()
        if len(name) >= 2:
            got = name in body
            add("who", name, got, [name], [name] if got else [])

    # exit_hook -- must land in the TAIL. A hook mentioned in paragraph two and
    # then dropped is exactly the failure v1 spent a `revise_hook_only` call on.
    # That call went with v1 and has no v2 successor, so this tail check is now
    # the only thing standing between a dropped hook and a shipped chapter.
    hook = str(card.get("exit_hook") or "").strip()
    if hook:
        add("exit_hook", hook, *_hit(hook, tail, tail_grams))

    # beats -- advisory, one item each (this is `beat_coverage`'s question).
    for beat in (card.get("beats") or []):
        beat = str(beat or "").strip()
        if beat:
            add("beats", beat, *_hit(beat, body, grams))

    # forbid -- a breach needs a VERBATIM hit on a distinctive anchor. Bigram
    # tolerance is right for "did you write this" and wrong for "did you avoid
    # this": a near-miss must not be charged as a violation.
    #
    # A phrase this card REQUIRES can never be a breach, no matter what `forbid`
    # says. See `_required_text`: the arc planner files whole 「…必须…」 sentences
    # into `forbid`, and `_anchors` happily lifts the requirement's own quoted
    # line out of one.
    required = _required_text(card)
    for entry in (card.get("forbid") or []):
        entry = str(entry or "").strip()
        if not entry:
            continue

        # An entry that is itself an obligation is not a ban at all, so it is not
        # even mined for anchors -- see `_is_misfiled_requirement`.
        if _is_misfiled_requirement(entry):
            out["forbid_conflicts"].append(
                {"field": "forbid", "target": entry, "phrase": "",
                 "why": "requirement_misfiled_as_ban"})
            continue

        def charge(phrase: str) -> bool:
            """True once the violation is settled (charged or waived)."""
            if phrase in required:
                out["forbid_conflicts"].append(
                    {"field": "forbid", "target": entry, "phrase": phrase,
                     "why": "card_requires_the_phrase_it_bans"})
                return True
            out["violations"].append(
                {"field": "forbid", "target": entry, "phrase": phrase})
            return True

        if len(entry) >= FORBID_MIN_ANCHOR and entry in body:
            charge(entry)
            continue
        for anchor in _anchors(entry):
            if len(anchor) >= FORBID_MIN_ANCHOR and anchor in body:
                charge(anchor)
                break

    if out["judgeable"]:
        out["ccr"] = out["fulfilled"] / out["judgeable"]
    out["passed"] = not out["hard_misses"] and not out["violations"]

    for item in out["hard_misses"]:
        out["directives"].append(
            f"本章卡片承诺的【{item['field']}】没有落到页面上："
            f"{item['target']}。必须补写具体动作+具体物，"
            f"至少让「{item['anchors'][0]}」真实出现。")
    for v in out["violations"]:
        out["directives"].append(
            f"本章违反卡片禁令【{v['target']}】：正文里出现了「{v['phrase']}」，必须换掉。")
    return out


# ---------------------------------------------------------------------------
# cite-or-drop
# ---------------------------------------------------------------------------

# Fields a reviewer might put its evidence under, in preference order.
QUOTE_KEYS = ("quote", "evidence", "excerpt", "原文", "证据", "引用", "locator")

# Punctuation and whitespace are the only things a quote may differ by. Anything
# else and the reviewer is paraphrasing, which is what cite-or-drop exists to
# catch.
_PUNCT_RE = re.compile(r"[\s“”\"'‘’「」『』（）()【】\[\]，,。.！!？?；;：:、…—\-~·]+")


def _normalize_quote(s: str) -> str:
    return _PUNCT_RE.sub("", str(s or ""))


def citation_check(
    claims: Iterable[dict[str, Any]] | None,
    text: str,
    min_quote_chars: int = 4,
) -> dict[str, Any]:
    """Drop every claim that cannot point at a substring of the chapter.

    A review finding is kept only when its quote survives punctuation/whitespace
    normalization and still appears, IN ORDER, in the normalized chapter. There
    is deliberately no fuzzy fallback here: bigram tolerance is correct for "did
    the writer stage this" (a near-miss still staged it) and wrong for "is this
    finding real" (a near-miss is a fabrication with good luck).

    A claim with no quote field at all is dropped as `uncited`. That is the whole
    point -- an assertion the reviewer would not back with the text is exactly
    the assertion that used to force a replan on nothing.

    Returns ``{"kept", "dropped", "total", "drop_rate"}``; each dropped entry
    carries a ``_drop_reason``.
    """
    claims = [c for c in (claims or []) if isinstance(c, dict)]
    body = _normalize_quote(_body(text))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for claim in claims:
        raw = ""
        for key in QUOTE_KEYS:
            if str(claim.get(key) or "").strip():
                raw = str(claim[key]).strip()
                break
        norm = _normalize_quote(raw)
        if not raw:
            dropped.append({**claim, "_drop_reason": "uncited"})
        elif len(norm) < min_quote_chars:
            dropped.append({**claim, "_drop_reason": "quote_too_short"})
        elif norm not in body:
            dropped.append({**claim, "_drop_reason": "quote_not_in_chapter"})
        else:
            kept.append(claim)
    total = len(claims)
    return {"kept": kept, "dropped": dropped, "total": total,
            "drop_rate": (len(dropped) / total) if total else 0.0}


# ---------------------------------------------------------------------------
# The canon check -- the one low-reasoning call that produces cite-or-drop claims
# ---------------------------------------------------------------------------

CANON_CHECK_SYSTEM = """你是长篇连载小说的「一致性核对员」。你不打分，不评价文笔，不提改进建议。

你只回答两个问题：
1. 本章有没有和既定事实（人物状态、已发生事件、未结线索）**直接矛盾**的地方？
2. 有没有**该收未收**——已到期的线索在本章本该兑现却没有兑现？

铁律（违反即整条结论作废）：
- **每一条结论都必须附一段 `quote`，逐字抄自本章正文**（可含标点，但字词必须一模一样，
  不得改写、不得概括、不得跨段拼接）。抄不出原文的结论，说明它不成立——**你自己删掉它**。
- 找不到问题是正常且常见的结果。宁可返回空列表，也不要为了交差编一条。
- 「写得不够好」「节奏偏慢」「人物动机可以更强」都**不是**本任务的输出，一律不写。

严重度：
- `hard`：与既定事实直接冲突（人死而复生、道具凭空出现、时间线倒错、能力越界）。
- `soft`：可疑但可自圆其说，或只是线索逾期。

修法（决定谁来改，不由你执行）：
- `this_chapter`：改一两句话即可消除的局部矛盾。
- `next_card`：结构性问题（该收未收、支线积压），**必须留给下一章的卡片**，
  本章不重写。

只输出 JSON，不要任何其它文字：
{"findings": [{"kind": "contradiction|overdue", "severity": "hard|soft",
  "detail": "一句话说明", "quote": "逐字抄自本章的原文片段",
  "target": "this_chapter|next_card"}]}"""

_CANON_TAG = "canon_check"


def canon_check(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    state: Any = None,
    *,
    card: dict[str, Any] | None = None,
    call: Any = None,
) -> list[dict[str, Any]]:
    """Call (3) of REDESIGN_V2 §3.2: contradictions and overdue threads, cited.

    Returns the RAW claims. Filtering is `citation_check`'s job and happens
    inside `acceptance_report`, so the drop rate is recorded rather than hidden:
    a model that fabricates findings shows up as a rising `citations.drop_rate`
    instead of as unexplained rework.

    Never raises. A canon check that dies is a check that found nothing, which is
    the same thing the cite-or-drop rule already says about an unbacked finding.
    """
    if not bool(config["novel"].get("canon_check_enabled", True)):
        return []
    body = str(text or "").strip()
    if len(body) < 500:
        return []
    from llm import call_llm, load_json_with_repair, safe_json_loads

    call = call or call_llm
    context = ""
    if state is not None:
        try:
            context = state.volatile_block()
        except Exception:
            context = ""
    prefix = ""
    if state is not None:
        try:
            prefix = state.stable_prefix()
        except Exception:
            prefix = ""
    user = "\n\n".join(p for p in (
        context,
        f"## 待核对的正文（第{chapter_num}章）\n{body}",
        "请按系统提示输出 JSON。没有问题就输出 {\"findings\": []}。",
    ) if p)
    try:
        raw = call(client, paths, config, CANON_CHECK_SYSTEM, user,
                   temperature=0.2, cacheable_prefix=prefix or None,
                   json_mode=True, tag=_CANON_TAG)
    except Exception:
        return []
    try:
        data = safe_json_loads(raw)
    except Exception:
        try:
            data = load_json_with_repair(client, paths, config, raw, fallback={})
        except Exception:
            data = {}
    findings = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(findings, list):
        return []
    return [f for f in findings if isinstance(f, dict)]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _em_history(
    conn: Any,
    chapter_num: int,
    config: dict[str, Any],
) -> list[float] | None:
    """The prior chapters' em-dash density, oldest-first, or None.

    `style_health`'s TREND term is a function of (this chapter's density, the
    recent mean) and is SILENT without a baseline. v2 called `style_health` with
    no history, so the term never fired here — the engine ran a strictly smaller
    gate than the one `quality.py` documents, and the difference is the case the
    static tier cannot see: a climb that is still below `style_em_dash_per_kchar_warn`.

    Measured on the 63 archived v2 round-0 drafts (2026-07-28): the term fires 4
    times, at em 3.3 / 4.67 / 5.32 / 5.85 rising off means of 1.5–3.1, and
    **crosses `style_penalty_block` zero times**. So wiring it cannot cost FPY′ —
    what it buys is 4 early-warning directives into the next chapter's writer
    prompt, one chapter before the static tier at 6.0 would have spoken.

    Two traps, both of which produced a wrong number first:
    * **The measurement must run on round-0 drafts, not on `chapters/*.md`.** On
      repaired final prose the same replay fires 18 times and every one is noise —
      L0 em-dash reduction has already pulled the drafts down to a ~3.0/k ceiling,
      so the surviving series oscillates 0.0↔2.8 and any normal chapter reads as a
      "2x rise" over a mean that em-free chapters dragged toward zero.
    * **Rows at or after `chapter_num` must be dropped.** On a resumed or rescued
      chapter this table already holds THIS chapter's own row from the previous
      attempt, and leaving it in compares the chapter against itself. That is why
      the query asks for one more row than the window.

    `tech_history` is deliberately NOT supplied even though the data is there
    (`chapter_metrics.tech_per_kchar`, populated on every row): `quality.py:643`
    is `_ = tech_history` under a note that the trend logic is deferred because
    the static conjunction already caught the collapse stretches in calibration
    replay. Passing it would read as a wired capability while changing no verdict.
    v1 passed it and got the same nothing.
    """
    # The no-conn case is checked explicitly rather than left to the `except`
    # below, which it would also reach: offline callers (tools, tests) passing no
    # conn are the NORMAL path, and routing them through an error handler makes
    # that handler look load-bearing for something it is not.
    if conn is None:
        return None
    window = max(int(config.get("novel", {}).get(
        "style_em_dash_trend_window", 5)), 1)
    try:
        rows = store.recent_metrics(conn, window + 1)
    except Exception:
        # A read failure degrades to "no baseline" rather than aborting the
        # chapter — this term is advisory and has never blocked (see above). The
        # column is guaranteed present by `store.init_db`'s migration, so the
        # realistic trigger is a conn that is not one.
        return None
    seq = sorted(
        (int(r["chapter"]), float(r["em_dash_per_kchar"]))
        for r in rows
        if r.get("chapter") is not None
        and isinstance(r.get("em_dash_per_kchar"), (int, float))
        and int(r["chapter"]) < chapter_num
    )
    return [v for _, v in seq[-window:]] or None


def acceptance_report(
    chapter_num: int,
    text: str,
    card: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    prior_texts: list[str] | None = None,
    prior_texts_long: list[str] | None = None,
    prev_text: str = "",
    book_texts: dict[int, str] | None = None,
    book_scans: Sequence[str] | None = None,
    recent_genre_scores: list[float] | None = None,
    recent_payoff_types: list[str] | None = None,
    conn: Any = None,
    fossil_whitelist: set[str] | None = None,
    canon_claims: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the acceptance set and emit a **v1-schema review payload**.

    Same keys `review.py` writes, so `quality.hard_block_reasons`,
    `tools/fpy_prime.py`, and `novel.py stats` all read a v2 chapter without
    knowing it is one. `score` is deliberately ABSENT: v2 has no self-score, and
    an absent key is honest where a fabricated 8.0 would be laundered into every
    downstream average.

    *canon_claims* are the cite-or-drop candidates from the canon check (the one
    low-reasoning LLM call). Uncited ones never reach the payload.

    *recent_payoff_types* and *conn* feed the gates that read the book's recent
    history rather than this chapter's text. Both are optional and both default to
    skipping their gate: an advisory that cannot be computed must stay ABSENT from
    the payload, not appear as a clean result. `payoff_beat_density` with no
    history would report a payoff drought of zero and read as healthy. *conn* also
    supplies `style_health`'s em-dash trend baseline (`_em_history`) — without it
    that gate runs with its trend term permanently silent, which is a smaller gate
    than the one `quality.py` documents rather than a passing one.

    *book_scans* names which book-level gates may run this chapter; `None` runs
    every one whose config switch is on. It exists because `review.py` runs the
    two whole-book scans on a cadence (`book_fossil_every`,
    `descriptor_freq_every`) rather than every chapter, and FPY' judges both arms
    by whatever their round-0 payload happens to contain. A v2 that scanned every
    chapter would find fossils v1 was never asked about and report itself as the
    worse engine for looking harder. `run.py:book_scan_gates` reproduces v1's
    cadence exactly; a caller with no cadence to match passes `None`.
    """
    cfg = config.get("novel", {})
    body = str(text or "")
    report: dict[str, Any] = {
        "engine": "v2",
        "chapter": chapter_num,
        "accepted": True,
        "gate_rejects": [],
        "problems": [],
        "writer_directives_for_next_chapter": [],
    }

    def enabled(gate: str) -> bool:
        return REGISTRY.is_enabled(gate, config)

    def scanning(gate: str) -> bool:
        return enabled(gate) and (book_scans is None or gate in book_scans)

    # --- style_health -> style_collapse -----------------------------------
    if enabled("style_health"):
        report["style_health"] = quality.style_health(
            body, config, em_history=_em_history(conn, chapter_num, config))

    # --- length / opening: the ruler reads `.block` -----------------------
    # `length_band_check` stores under a key that is NOT the gate name, and its
    # config switch only controls the PENALTY -- the gate always runs.
    report["length_band"] = quality.length_band_check(body, config)
    if enabled("opening_hook_gate"):
        report["opening_hook_gate"] = quality.opening_hook_gate(body, chapter_num, config)

    # --- adjacent repetition -> adjacent_repeat_block ---------------------
    if prev_text and enabled("adjacent_repetition"):
        ar = quality.adjacent_repetition(body, prev_text, config)
        report["adjacent_repetition"] = ar
        if str(ar.get("level", "")) == "block":
            report["gate_rejects"].append({"gate": "adjacent_repetition",
                                           "level": "block"})

    # --- cross-chapter repetition -> gate_rejects -------------------------
    if enabled("cross_chapter_repetition"):
        cr = quality.cross_chapter_repetition(
            body, prior_texts or [], config,
            prior_texts_long=list(prior_texts_long) if prior_texts_long else None)
        report["cross_chapter_repetition"] = cr
        if str(cr.get("level", "")) == "reject":
            report["gate_rejects"].append({
                "gate": "cross_chapter_repetition", "level": "reject",
                "phrases": [str(p) for p in (cr.get("repeated") or [])][:8]})

    # --- book-wide fossils -> gate_rejects (hard only) --------------------
    if book_texts and scanning("book_wide_fossils"):
        bf = quality.book_wide_fossils(book_texts, config,
                                       whitelist=fossil_whitelist,
                                       current_chapter=chapter_num)
        report["book_fossils"] = bf
        # BOTH of v1's rejects, not just the interesting one. `review.py` raises a
        # `book_wide_fossils` reject on distinct-phrase count as well as the
        # `book_wide_fossils_ratio` one on a single saturating phrase; a v2 that
        # emitted only the latter would collect a free pass on the condition v1
        # blocks for, and FPY' would read the missing reject as v2 doing better.
        phrases = [str(p) for p in (bf.get("phrases") or [])]
        struct_count = int(cfg.get("book_fossil_struct_count", 10))
        if len(phrases) >= struct_count:
            report["gate_rejects"].append({
                "gate": "book_wide_fossils", "level": "reject",
                "count": len(phrases), "phrases": phrases[:8]})
        hard = bf.get("hard_fossils") or []
        if hard:
            report["gate_rejects"].append({
                "gate": "book_wide_fossils_ratio", "level": "reject",
                "phrases": [str(f.get("phrase")) for f in hard]})

    # --- descriptor frequency -> gate_rejects -----------------------------
    if book_texts and scanning("descriptor_frequency"):
        df = quality.descriptor_frequency(book_texts, config)
        report["descriptor_frequency"] = df
        if str(df.get("level", "")) == "reject":
            report["gate_rejects"].append({"gate": "descriptor_frequency",
                                           "level": "reject"})

    # --- genre adherence -> gate_rejects ----------------------------------
    if enabled("genre_adherence"):
        ga = quality.genre_adherence(body, recent_genre_scores or [], config)
        report["genre_adherence"] = ga
        if str(ga.get("level", "")) == "reject":
            report["gate_rejects"].append({"gate": "genre_adherence",
                                           "level": "reject"})

    # --- CCC: the v2-native member ----------------------------------------
    ccc = contract_fulfilment(card, body, config)
    report["contract_fulfilment"] = ccc
    if ccc["enabled"] and not ccc["passed"]:
        # Surfaced as a gate_reject so the ONE ruler sees it. A v2-only key the
        # ruler ignores would mean the two arms are scored differently.
        report["gate_rejects"].append({
            "gate": "contract_fulfilment", "level": "reject",
            "phrases": [i["target"] for i in ccc["hard_misses"]][:4],
            "violations": [v["phrase"] for v in ccc["violations"]][:4]})
    if ccc.get("forbid_conflicts"):
        # A waived ban is a CARD defect, not a prose one, so it must not reach
        # `writer_directives_for_next_chapter` -- no chapter can repair it. It is
        # promoted to a top-level key instead of being dropped silently, because a
        # planner that keeps filing requirements into `forbid` is worth seeing.
        report["card_defects"] = [
            (f"卡片自相矛盾：`forbid` 里这条其实是硬性要求而不是禁令"
             f"（{c['target'][:60]}…），整条豁免"
             if c.get("why") == "requirement_misfiled_as_ban" else
             f"卡片自相矛盾：`forbid` 禁了本卡片自己要求的「{c['phrase']}」"
             f"（出自 {c['target'][:40]}…），已豁免")
            for c in ccc["forbid_conflicts"]]

    # --- advisory-only gates: directives, never a reject -------------------
    # These nine were orphaned by the v1 deletion -- `review.py` called them by
    # name and nothing replaced it, so `quality.py` registered them and the live
    # engine never reached them. Each verdict here is a MEASUREMENT
    # (`tools/orphan_gates.py`, 638 archived chapters + v2's 30), recorded in the
    # gate's own `proof=` string; two more (`flat_chapter_streak`,
    # `emotional_cadence`) were deleted at the same pass rather than wired.
    #
    # Advisory means exactly one thing: the result contributes `directives` and
    # NOTHING ELSE. None of them appends to `gate_rejects`, so `hard_block_reasons`
    # cannot move and the FPY' ruler reads the same number before and after this
    # wiring -- which is what makes the change safe to ship without an A/B.
    # `REGISTRY.may_block()` refuses any of them blocking power structurally.
    #
    # The cost of a reachable advisory gate is prompt bytes, and those only
    # materialize when it fires. That is why a 0%-firing gate
    # (`paragraph_shape_health`, `hook_tail_repetition`, `shareable_line` on v2)
    # is worth wiring anyway: it is a free regression tripwire.
    if enabled("ai_flavor_health"):
        report["ai_flavor_health"] = quality.ai_flavor_health(body, config)
    if enabled("paragraph_shape_health"):
        report["paragraph_shape_health"] = quality.paragraph_shape_health(body, config)
    if enabled("prose_texture"):
        report["prose_texture"] = quality.prose_texture(body, config)
    if enabled("shareable_line"):
        report["shareable_line"] = quality.shareable_line(body, config)
    if enabled("intra_chapter_repetition"):
        report["intra_chapter_repetition"] = quality.intra_chapter_repetition(body, config)
    if prior_texts and enabled("hook_tail_repetition"):
        report["hook_tail_repetition"] = quality.hook_tail_repetition(
            body, list(prior_texts), config)
    if enabled("payoff_beat_density"):
        # `recent_payoff_types` is NEWEST-FIRST, the order `store.recent_metrics`
        # returns and the order the gate's drought loop is written against. Fed
        # ascending it counts the streak forward from chapter 1 and reports 0 for
        # every book (the bug `tools/orphan_gates.py` hit first).
        report["payoff_beat_density"] = quality.payoff_beat_density(
            body, list(recent_payoff_types or []), config)
    if enabled("information_density"):
        # The card projected through the plan schema, because that is the shape the
        # gate reads. v2 has no `review["beats_audit"]`, and the gate no longer
        # asks for one -- see its docstring on the two signals that had no producer.
        plan = arc.card_to_plan(card)[0] if card else None
        report["information_density"] = quality.information_density(
            body, plan, None, config)
    if conn is not None and enabled("long_span_fatigue"):
        # scope="book": the loudest of the nine at 40% on v2, and structurally
        # incapable of rejecting -- no rewrite of THIS chapter can lower a quantity
        # accumulated over the last N finished ones.
        report["long_span_fatigue"] = quality.long_span_fatigue(
            conn, chapter_num, config)

    # --- directives -------------------------------------------------------
    wd = report["writer_directives_for_next_chapter"]
    for key in ("style_health", "length_band", "opening_hook_gate",
                "adjacent_repetition", "cross_chapter_repetition",
                "book_fossils", "descriptor_frequency", "genre_adherence",
                "contract_fulfilment",
                # the nine wired advisories
                "ai_flavor_health", "paragraph_shape_health", "prose_texture",
                "shareable_line", "intra_chapter_repetition",
                "hook_tail_repetition", "payoff_beat_density",
                "information_density", "long_span_fatigue"):
        for d in (report.get(key) or {}).get("directives", []):
            if d not in wd:
                wd.append(d)

    return fold_citations(report, body, canon_claims, config)


_DETERMINISTIC_PREFIX = "DETERMINISTIC: "


def fold_citations(
    report: dict[str, Any],
    text: str,
    canon_claims: Iterable[dict[str, Any]] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Attach cite-or-drop findings to a report and re-derive its verdict.

    Separated from `acceptance_report` because the canon check runs LAST in the
    decision table -- after L0/L1, so the one LLM judgement is spent on the text
    that will actually ship -- while the FIRST-DRAFT report that FPY' replays was
    archived several steps earlier. Re-folding the same claims into that earlier
    payload is what keeps the two arms judged by one definition of "first pass":
    v1's round-0 review carries its reviewer's contract violations, so v2's must
    carry its canon check's.

    The re-fold is not a copy of the verdict, it is a recomputation: the quotes
    are re-matched against *this* text, so a finding whose evidence exists only
    in the repaired prose is dropped from the first draft's report rather than
    charged to it. Mutates and returns `report`.
    """
    body = str(text or "")
    cite = citation_check(canon_claims, body)
    report["citations"] = {"total": cite["total"], "kept": len(cite["kept"]),
                           "dropped": len(cite["dropped"]),
                           "drop_rate": cite["drop_rate"],
                           "dropped_detail": cite["dropped"][:8]}
    hard_cited = [c for c in cite["kept"]
                  if str(c.get("severity", "")).lower() == "hard"]
    if hard_cited:
        report["contract_violations"] = hard_cited
    else:
        report.pop("contract_violations", None)

    reasons = hard_block_reasons(report, config)
    report["block_reasons"] = reasons
    report["accepted"] = not reasons
    problems = [p for p in (report.get("problems") or [])
                if not str(p).startswith(_DETERMINISTIC_PREFIX)]
    if reasons:
        problems.append(_DETERMINISTIC_PREFIX + "; ".join(reasons))
    report["problems"] = problems
    return report


def block_reasons(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """The one ruler, re-exported so `v2/run.py` never grows a second copy."""
    return hard_block_reasons(report, config)


__all__ = [
    "ACCEPTANCE_GATES", "NATIVE_CHECKS", "NOT_IN_ACCEPTANCE", "CARD_GATES",
    "HARD_FIELDS", "contract_fulfilment", "citation_check", "canon_check",
    "CANON_CHECK_SYSTEM", "fold_citations", "acceptance_report", "block_reasons",
]
