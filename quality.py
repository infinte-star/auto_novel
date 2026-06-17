"""Non-LLM, rule-based quality signals.

These run as cheap, deterministic checks that do NOT depend on the model's own
self-assessment (which is prone to inflation and to ratifying a degenerating
prose style as "the book's voice"). They provide an objective anchor that the
review/revise loop can react to.

The flagship signal is `style_health`, which catches "style collapse" — the
failure mode where chapters drift into telegraphic fragments glued together by
em-dashes (`句子——状态——状态——状态——`), one-clause lines, and数值化 stage
directions instead of human-readable prose. Self-review never catches this
because the model's own voice has drifted with the prose.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# Sentence-ending punctuation for Chinese prose.
_SENTENCE_ENDERS = "。！？…"
_EM_DASH = "——"

# Process-level cache: maps a fast text fingerprint -> normalized clause set.
# This avoids re-parsing the same prior chapter texts on every review call.
# Each entry is small (~1-3 KB), so 500 entries ≈ 1-2 MB max.
_CLAUSE_SET_CACHE: dict[str, frozenset[str]] = {}
_CLAUSE_CACHE_MAX = 500


def _get_cached_clause_set(text: str) -> frozenset[str]:
    """Return the normalized clause set for `text`, using a process-level cache."""
    # Use first 200 + last 100 chars as the fingerprint key — fast and collision-resistant enough.
    key = hashlib.md5((text[:200] + text[-100:]).encode("utf-8", errors="replace")).hexdigest()
    if key not in _CLAUSE_SET_CACHE:
        if len(_CLAUSE_SET_CACHE) >= _CLAUSE_CACHE_MAX:
            # Evict oldest half when full (simple FIFO via dict insertion order).
            evict = list(_CLAUSE_SET_CACHE.keys())[: _CLAUSE_CACHE_MAX // 2]
            for k in evict:
                del _CLAUSE_SET_CACHE[k]
        _CLAUSE_SET_CACHE[key] = frozenset(_normalize_clause(c) for c in _clause_segments(text))
    return _CLAUSE_SET_CACHE[key]


def _strip_title_line(text: str) -> str:
    """Drop the leading `第N章 标题` line so it doesn't skew line stats."""
    lines = text.lstrip().splitlines()
    if lines and re.match(r"^#?\s*第.{1,8}章", lines[0].strip()):
        return "\n".join(lines[1:])
    return text


def style_health(
    text: str,
    config: dict[str, Any] | None = None,
    em_history: list[float] | None = None,
) -> dict[str, Any]:
    """Compute deterministic prose-health metrics + a penalty + directives.

    Returns:
      {
        "metrics": {...},        # raw measurements for logging
        "penalty": float,        # >=0, to SUBTRACT from the LLM review score
        "flags": [str],          # human-readable problem tags
        "directives": [str],     # imperative fixes injected into the writer prompt
      }

    Thresholds are configurable under config["novel"] with sane defaults; the
    function is safe to call with config=None.

    `em_history` is the em-dash-per-kchar sequence of the most recent prior
    chapters (oldest→newest). When supplied, a TREND term fires: if this
    chapter's em density rises sharply versus the recent mean — even while still
    below the absolute warn threshold — it is penalized and a directive is
    emitted. This is the cure for slow style collapse (em creeping 0.94→4.15
    monotonically with the static threshold never tripping).
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    body = _strip_title_line(text)
    n = len(body)
    metrics: dict[str, Any] = {}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    if n < 200:
        return {"metrics": {"chars": n}, "penalty": 0.0, "flags": [], "directives": []}

    # --- 1. Em-dash density (the dominant collapse signature) --------------
    em_dashes = body.count(_EM_DASH)
    em_per_kchar = em_dashes / (n / 1000.0)
    metrics["em_dash_count"] = em_dashes
    metrics["em_dash_per_kchar"] = round(em_per_kchar, 2)
    em_warn = float(cfg.get("style_em_dash_per_kchar_warn", 6.0))
    em_bad = float(cfg.get("style_em_dash_per_kchar_bad", 12.0))
    if em_per_kchar >= em_bad:
        penalty += 2.0
        flags.append(f"em_dash_overload({em_per_kchar:.1f}/k≥{em_bad})")
        directives.append(
            "严重文体问题：上一章破折号（——）密度过高，整章读起来像电报/碎句堆叠。"
            "本章必须用完整的主谓宾句子叙事，破折号每千字不超过 3 个。"
        )
    elif em_per_kchar >= em_warn:
        penalty += 1.0
        flags.append(f"em_dash_high({em_per_kchar:.1f}/k≥{em_warn})")
        directives.append(
            "上一章破折号偏多，本章请减少破折号，改用完整句子与正常标点叙事。"
        )
    else:
        # --- 1b. TREND term: slow drift below the absolute warn threshold ---
        # Slow style collapse never trips the static threshold (em can creep
        # 0.94→4.15 monotonically while always < 6.0). Catch it by comparing
        # against the recent-chapter mean: a sharp rise is itself a problem.
        hist = [
            float(h) for h in (em_history or [])
            if isinstance(h, (int, float)) and h >= 0
        ]
        # Need at least 2 prior points for a meaningful baseline.
        if len(hist) >= 2:
            base = sum(hist) / len(hist)
            metrics["em_dash_recent_mean"] = round(base, 2)
            # Absolute rise (per-kchar) and multiplicative rise vs the baseline.
            rise_abs = float(cfg.get("style_em_dash_trend_rise", 1.0))
            rise_mult = float(cfg.get("style_em_dash_trend_mult", 1.8))
            # A tiny baseline (≈0) makes the multiplicative test trivially true,
            # so require the absolute delta too. Only fire when the chapter is
            # also above a small floor so we don't punish 0.1→0.3 noise.
            floor = float(cfg.get("style_em_dash_trend_floor", 1.5))
            delta = em_per_kchar - base
            if (
                em_per_kchar >= floor
                and delta >= rise_abs
                and em_per_kchar >= base * rise_mult
            ):
                penalty += 1.0
                flags.append(
                    f"em_dash_trend_rise({em_per_kchar:.1f}/k vs mean {base:.1f}/k)"
                )
                directives.append(
                    f"文体趋势预警：破折号密度从近几章均值 {base:.1f}/千字升到 "
                    f"{em_per_kchar:.1f}/千字，正在向碎句化滑坡（即使尚未触顶阈值）。"
                    "本章必须主动收敛破折号，回到完整句叙事。"
                )

    # --- 2. Average sentence length (collapse → very short sentences) ------
    # Split on sentence enders; measure mean length of non-empty segments.
    segments = [s for s in re.split(f"[{_SENTENCE_ENDERS}\n]", body) if s.strip()]
    if segments:
        avg_seg = sum(len(s.strip()) for s in segments) / len(segments)
        metrics["avg_sentence_chars"] = round(avg_seg, 1)
        min_avg = float(cfg.get("style_min_avg_sentence_chars", 12.0))
        if avg_seg < min_avg:
            penalty += 1.0
            flags.append(f"sentences_too_short(avg={avg_seg:.1f}<{min_avg})")
            # Bidirectional convergence: when em-dash density was just suppressed,
            # prose tends to overshoot into staccato single-clause lines (observed
            # v5 Ch4: em 0.3/k but avg sentence 11.5 chars). Em-suppression alone
            # is not "healthy" — pair it with an explicit "write fuller compound
            # sentences" directive so the writer doesn't trade one collapse mode
            # (em-fragments) for another (telegraphic shorts).
            em_low = em_per_kchar < float(cfg.get("style_em_dash_per_kchar_warn", 6.0))
            if em_low:
                directives.append(
                    f"上一章破折号已收敛，但平均句长仅 {avg_seg:.0f} 字、滑向了另一种碎句化（短促单句堆叠）。"
                    "本章请用带从句/状语的完整复合长句承载叙事与心理，"
                    "在不重新堆破折号的前提下把平均句长拉回 14 字以上。"
                )
            else:
                directives.append(
                    f"上一章平均句长仅 {avg_seg:.0f} 字，过于碎片化。本章请写完整、连贯的句子，"
                    "避免把一句话拆成多个单词短句。"
                )

    # --- 3. Fragment-line ratio (lines that are tiny standalone clauses) ---
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        # A "fragment line" is a short line that is NOT dialogue (no quote marks)
        # and does NOT end with sentence punctuation — i.e. a dangling clause.
        frag = 0
        for ln in lines:
            if len(ln) >= 25:
                continue
            if any(q in ln for q in "“”\"「」"):
                continue
            if ln and ln[-1] in _SENTENCE_ENDERS:
                continue
            if ln in ("---", "***"):
                continue
            frag += 1
        frag_ratio = frag / len(lines)
        metrics["fragment_line_ratio"] = round(frag_ratio, 2)
        frag_max = float(cfg.get("style_fragment_line_ratio_max", 0.35))
        if frag_ratio >= frag_max:
            penalty += 1.0
            flags.append(f"fragment_lines({frag_ratio:.0%}≥{frag_max:.0%})")
            directives.append(
                "上一章存在过多无标点的短促断行，像舞台提示而非小说。"
                "本章每个自然段须是连贯成句的叙事。"
            )

    # --- 4. Dialogue presence (collapse often drops real dialogue) --------
    # Prefer paired CJK quotes (the prose convention here): count matched
    # “…”/「…」 pairs directly. Only fall back to estimating from ASCII " pairs
    # when no CJK quotes are present, since ASCII straight quotes are ambiguous
    # (a chapter may use them for emphasis, not dialogue) and dividing the raw
    # count by 2 systematically over/under-counts.
    cjk_pairs = min(body.count("“"), body.count("”")) + min(body.count("「"), body.count("」"))
    if cjk_pairs > 0:
        quote_pairs = cjk_pairs
    else:
        quote_pairs = body.count('"') // 2
    metrics["dialogue_markers"] = quote_pairs
    # Only flag if the chapter is long enough that some dialogue is expected.
    if n > 2000 and quote_pairs < 3:
        penalty += 0.5
        flags.append("almost_no_dialogue")
        directives.append("上一章几乎没有对话，本章请加入有潜台词的人物对白。")

    penalty = round(min(penalty, float(cfg.get("style_penalty_cap", 4.0))), 2)
    metrics["penalty"] = penalty
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:4],
    }


# ---------------------------------------------------------------------------
# Raw-text similarity: catch adjacent-chapter duplicate generation.
# ---------------------------------------------------------------------------

def _text_bigrams(text: str) -> set[str]:
    """Character-bigram set over the substantive (CJK/alnum) content of a text."""
    cleaned = re.sub(r"[^一-鿿A-Za-z0-9]", "", text or "")
    if len(cleaned) < 2:
        return set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity of two prose blocks over their character bigrams.

    Used to catch the "adjacent chapters are near-verbatim duplicates" failure
    mode (observed in refine output: Ch5≈Ch6, Ch7≈Ch8) where the same scene is
    emitted twice. ~0.0 = unrelated, ~1.0 = (near-)identical.
    """
    return _jaccard(_text_bigrams(a), _text_bigrams(b))


# ---------------------------------------------------------------------------
# Adjacent-chapter repetition gate (O1): the deadliest observed failure mode is
# a chapter that re-narrates the previous chapter's ending scene near-verbatim
# (suspense_v11 Ch3 clause-overlap 0.73, Ch8 0.33; suspense_v8 Ch6 0.81 — all
# force-accepted at 3.5-5.5 while healthy chapters sit at 0.00-0.07). The LLM
# reviewer scores each chapter in isolation, so it rated an identical hook 9/10.
# This is the deterministic gate: measured against the previous chapter's text,
# fed into both the draft loop (regenerate) and review (cap + reject).
# ---------------------------------------------------------------------------

def adjacent_repetition(
    text: str,
    prev_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure how much of `text` re-narrates `prev_text`.

    Returns {"metrics", "level" (ok/warn/block), "penalty", "flags",
    "directives", "examples"}. Calibrated on real books:
      healthy adjacent chapters: clause_overlap 0.00-0.07, bigram_sim ~0.1-0.2
      duplicated chapters:       clause_overlap 0.33-0.81, bigram_sim 0.42-0.84
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "level": "ok", "penalty": 0.0,
        "flags": [], "directives": [], "examples": [],
    }
    if not bool(cfg.get("adjacent_repeat_enabled", True)) or not text or not prev_text:
        return result
    sim = text_similarity(text, prev_text)
    prev_set = _get_cached_clause_set(prev_text)
    cur_clauses = _clause_segments(text)
    hits = [c for c in cur_clauses if _normalize_clause(c) in prev_set]
    ratio = (len(hits) / len(cur_clauses)) if cur_clauses else 0.0
    result["metrics"] = {
        "bigram_sim": round(sim, 3),
        "clause_overlap": round(ratio, 3),
        "clause_hits": len(hits),
    }
    warn = float(cfg.get("adjacent_repeat_clause_warn", 0.10))
    block = float(cfg.get("adjacent_repeat_clause_block", 0.30))
    bigram_block = float(cfg.get("adjacent_repeat_bigram_block", 0.50))
    # Longest verbatim clauses make the most actionable avoid-list.
    result["examples"] = sorted(set(hits), key=len, reverse=True)[:5]
    if ratio >= block or sim >= bigram_block:
        result["level"] = "block"
        result["penalty"] = float(cfg.get("adjacent_repeat_block_penalty", 3.0))
        result["flags"].append(f"adjacent_duplicate(clause={ratio:.2f},bigram={sim:.2f})")
        result["directives"].append(
            f"本章有 {ratio:.0%} 的句子逐字复述上一章内容，属于复读废稿。"
            "必须从上一章结尾之后的【新】事件写起：上一章已发生的场景、对话、推理结论只许一笔带过引用，"
            "严禁重演。以下句子严禁再次出现：" + "；".join(f"“{c}”" for c in result["examples"][:3])
        )
    elif ratio >= warn:
        result["level"] = "warn"
        result["penalty"] = float(cfg.get("adjacent_repeat_warn_penalty", 1.0))
        result["flags"].append(f"adjacent_overlap(clause={ratio:.2f})")
        result["directives"].append(
            f"本章约 {ratio:.0%} 的句子与上一章重复，有原地复读倾向。"
            "请删去对上一章场景的复述，把篇幅用在新事件与新信息上。"
        )
    return result


def hook_tail_repetition(
    text: str,
    prev_texts: list[str] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a chapter-end hook recycled from recent chapters' endings.

    Recurring debt across books: "章末钩子与上章完全相同，锐利度被严重稀释"
    (the LLM reviewer still rated such hooks 9/10 because it never sees the
    previous endings side by side). Compares the clause set of this chapter's
    final ~300 chars against the final ~800 chars of each recent chapter.
    Returns {"repeat": bool, "repeated_clauses", "ratio"}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {"repeat": False, "repeated_clauses": [], "ratio": 0.0}
    if not text or not prev_texts:
        return result
    tail_chars = int(cfg.get("hook_repeat_tail_chars", 300))
    cur = [c for c in _clause_segments(text[-tail_chars:]) if len(c) >= 8]
    if not cur:
        return result
    repeated: set[str] = set()
    for pt in prev_texts:
        prev_tail_set = {_normalize_clause(c) for c in _clause_segments(pt[-max(tail_chars * 2, 600):])}
        for c in cur:
            if _normalize_clause(c) in prev_tail_set:
                repeated.add(c)
    ratio = len(repeated) / len(cur)
    result["repeated_clauses"] = sorted(repeated, key=len, reverse=True)[:4]
    result["ratio"] = round(ratio, 3)
    min_clauses = int(cfg.get("hook_repeat_min_clauses", 2))
    min_ratio = float(cfg.get("hook_repeat_min_ratio", 0.25))
    result["repeat"] = len(repeated) >= min_clauses or ratio >= min_ratio
    return result


# ---------------------------------------------------------------------------
# Intra-chapter self-repetition: catch a chapter that re-states its own content.
# ---------------------------------------------------------------------------

# Observed failure (suspense_10ch Ch7, mimo): a chapter ends with a "summary
# paragraph" that re-states reasoning/conclusions already delivered in the body
# (章末总结段与正文推理段高度重复, 信息量零增量). The LLM reviewer often rates
# this fine because each paragraph reads well in isolation. This deterministic
# check measures how much the chapter's TAIL re-states its own EARLIER content.

def intra_chapter_repetition(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure how much the chapter's ending re-states its own earlier content.

    Splits the chapter into a head (everything but the last `tail_chars`) and a
    tail. Counts how many distinctive tail clauses already appeared (verbatim or
    near-verbatim) in the head. A high ratio means the ending is a zero-增量
    summary recap rather than a forward-moving hook.

    Returns {"metrics", "level" (ok/warn/block), "penalty", "flags",
    "directives", "examples"}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "level": "ok", "penalty": 0.0,
        "flags": [], "directives": [], "examples": [],
    }
    if not bool(cfg.get("intra_repeat_enabled", True)) or not text:
        return result
    body = _strip_title_line(text)
    if len(body) < int(cfg.get("intra_repeat_min_chars", 1500)):
        return result
    tail_chars = int(cfg.get("intra_repeat_tail_chars", 600))
    if len(body) <= tail_chars + 200:
        return result
    head = body[:-tail_chars]
    tail = body[-tail_chars:]
    head_set = {_normalize_clause(c) for c in _clause_segments(head)}
    tail_clauses = _clause_segments(tail)
    if not tail_clauses:
        return result
    hits = [c for c in tail_clauses if _normalize_clause(c) in head_set]
    ratio = len(hits) / len(tail_clauses)
    result["metrics"] = {
        "tail_recap_ratio": round(ratio, 3),
        "tail_clauses": len(tail_clauses),
        "recap_hits": len(hits),
    }
    result["examples"] = sorted(set(hits), key=len, reverse=True)[:4]
    warn = float(cfg.get("intra_repeat_warn", 0.25))
    block = float(cfg.get("intra_repeat_block", 0.45))
    if ratio >= block:
        result["level"] = "block"
        result["penalty"] = float(cfg.get("intra_repeat_block_penalty", 2.0))
        result["flags"].append(f"intra_chapter_recap(ratio={ratio:.2f})")
        result["directives"].append(
            f"本章结尾有 {ratio:.0%} 的句子在复述正文已给出的推理/结论（零增量总结段）。"
            "章末必须是【前进的钩子】——抛出新疑问、新动作、新危机，而不是把已讲过的线索再列一遍。"
            "删去总结复述，让结尾推动剧情往下走。以下复述句严禁出现：" +
            "；".join(f"“{c}”" for c in result["examples"][:3])
        )
    elif ratio >= warn:
        result["level"] = "warn"
        result["penalty"] = float(cfg.get("intra_repeat_warn_penalty", 0.8))
        result["flags"].append(f"intra_chapter_recap(ratio={ratio:.2f})")
        result["directives"].append(
            f"本章结尾约 {ratio:.0%} 在复述正文已有信息，有总结收尾倾向。"
            "请把结尾改成推动剧情的钩子，而非已知信息的回顾。"
        )
    return result


# ---------------------------------------------------------------------------
# Cross-chapter repetition: catch sentence/metaphor "fossils" reused verbatim.
# ---------------------------------------------------------------------------

# A reused signature phrase ("像一颗心脏在缓慢地跳动", "不是暂时的，是永久的")
# becomes a tic when it recurs across chapters. Self-review never flags it
# because the drifted voice treats it as motif. This deterministic check counts
# how often this chapter's distinctive clauses already appeared in prior prose.

def _clause_segments(text: str, min_len: int = 6, max_len: int = 40) -> list[str]:
    """Split prose into clause-sized segments suitable for repeat detection."""
    body = _strip_title_line(text)
    # Strip quotes/markup so a recurring narration clause is comparable.
    raw = re.split(r"[，。！？…；\n“”\"「」]", body)
    out: list[str] = []
    for s in raw:
        s = re.sub(r"\s+", "", s.strip())
        if min_len <= len(s) <= max_len:
            out.append(s)
    return out


def _normalize_clause(s: str) -> str:
    """Collapse digits so 'every 3 seconds' / 'every 7 seconds' match as one tic."""
    return re.sub(r"[0-9一二三四五六七八九十两零]+", "#", s)


def cross_chapter_repetition(
    text: str,
    prior_texts: list[str] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect signature clauses in `text` that recur in earlier chapters.

    Returns {"metrics", "penalty", "flags", "directives", "repeats"}. `repeats`
    lists the offending clauses (with prior occurrence counts) for logging and
    for folding into the writer's avoid-list directive.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    enabled = bool(cfg.get("style_cross_repeat_enabled", True))
    result: dict[str, Any] = {
        "metrics": {}, "penalty": 0.0, "flags": [], "directives": [], "repeats": [],
        "level": "pass",
    }
    if not enabled or not prior_texts:
        return result

    # Build a frequency map of normalized clauses across prior chapters.
    prior_counts: dict[str, int] = {}
    for pt in prior_texts:
        for c in _get_cached_clause_set(pt):
            prior_counts[c] = prior_counts.get(c, 0) + 1

    cur_clauses = _clause_segments(text)
    cur_norm_seen: set[str] = set()
    repeats: list[tuple[str, int]] = []
    for c in cur_clauses:
        nc = _normalize_clause(c)
        if nc in cur_norm_seen:
            continue
        cur_norm_seen.add(nc)
        prior = prior_counts.get(nc, 0)
        if prior >= 1 and len(c) >= int(cfg.get("style_cross_repeat_min_len", 7)):
            repeats.append((c, prior))

    # Penalize by how many earlier chapters already used the clause.
    fossil_threshold = int(cfg.get("style_cross_repeat_chapters", 2))
    fossils = [(c, p) for c, p in repeats if p >= fossil_threshold]
    repeats.sort(key=lambda x: -x[1])
    result["repeats"] = [{"clause": c, "prior_chapters": p} for c, p in repeats[:12]]
    result["metrics"]["cross_repeat_count"] = len(repeats)
    result["metrics"]["cross_repeat_fossils"] = len(fossils)

    if fossils:
        # Each entrenched fossil adds penalty, capped.
        pen = min(2.0, 0.5 * len(fossils))
        result["penalty"] = round(pen, 2)
        result["flags"].append(f"cross_chapter_fossils({len(fossils)})")
        examples = "、".join(f"“{c}”(已出现{p}章)" for c, p in fossils[:4])
        result["directives"].append(
            "文体复读预警：以下标志性句子/比喻在前面多章反复出现，已成为口癖，"
            f"本章必须改写或避免：{examples}。同一意象请换新的具体写法。"
        )
        result["level"] = "advise"
        # Escalation: a chapter carrying MANY entrenched fossils is not a tic to
        # warn about — it is style collapse already in motion (suspense_v11 ran
        # 6 consecutive chapters at fossils 9-25 with only advisory directives,
        # and the prose never recovered). Past this count the verdict becomes
        # "reject": the pipeline must regenerate, not annotate.
        reject_count = int(cfg.get("style_cross_repeat_reject_count", 8))
        if len(fossils) >= reject_count:
            result["level"] = "reject"
            result["flags"].append(f"cross_chapter_fossil_collapse({len(fossils)})")
    elif len(repeats) >= int(cfg.get("style_cross_repeat_warn_count", 4)):
        result["penalty"] = 0.5
        result["flags"].append(f"cross_chapter_repeats({len(repeats)})")
        result["directives"].append(
            "本章有多处句子与前文几乎雷同，存在复读倾向，请用不同措辞重写这些重复表达。"
        )
        result["level"] = "advise"
    return result


# ---------------------------------------------------------------------------
# Beat-coverage gate: deterministic "did the prose actually stage each beat?".
#
# The single biggest first-pass score sink (v13 Ch10: the plan's core payoff
# beat — 安瓿碎裂方向矛盾 — never appeared in the prose AT ALL, despite three
# layers of prompt emphasis; the LLM reviewer then charged -1.0 per absent
# beat). An absent beat is detectable with plain substring/bigram matching:
# if the beat promises a concrete object ("安瓿"), the chapter must at least
# MENTION it. This gate runs at the writer layer (before any LLM review) so a
# vanished beat costs one cheap targeted repair call instead of a full
# review→revise→replan cycle.
#
# Design bias: CONSERVATIVE. A false "miss" wastes a repair call and may
# splice awkward prose; a false "pass" just falls through to the existing LLM
# beats_audit (current behaviour). So anchors are only the beat's distinctive
# content fragments, matching accepts loose rewording via bigram coverage,
# and beats with no extractable anchors auto-pass.
# ---------------------------------------------------------------------------

# Tokens that never carry beat-specific content: particles, copulas, pronouns,
# numerals/classifiers, and the abstract realization verbs whose objects (not
# the verbs themselves) are what must appear on the page. Multi-char tokens
# must come before their prefixes in the regex alternation (sorted by length).
_BEAT_STOP_TOKENS = (
    "意识到", "注意到", "反应过来",
    "发现", "看到", "看见", "听到", "听见", "想到", "想起", "认出", "确认",
    "开始", "决定", "进行", "出现", "通过", "利用", "试图", "准备", "继续",
    "随后", "然后", "同时", "必须", "可以", "已经", "没有", "不再", "再次",
    "终于", "突然", "悄悄", "暗中", "立刻", "马上",
    "他们", "她们", "我们", "你们",
    "一个", "一种", "一次", "一道", "一张", "一份", "一句", "一段",
    "的", "地", "得", "了", "着", "过", "是", "在", "把", "将", "被",
    "对", "向", "从", "给", "让", "使", "和", "与", "或", "及", "并",
    "而", "但", "又", "也", "都", "就", "才", "再", "很", "更", "最",
    "他", "她", "它", "我", "你", "这", "那", "其", "某", "并且", "因为",
    "所以", "如果", "虽然", "于是",
)

# Generic fragments that survive splitting but identify nothing specific.
_BEAT_GENERIC_FRAGMENTS = frozenset({
    "时候", "东西", "事情", "地方", "样子", "一下", "起来", "出来", "下来",
    "过来", "之后", "之前", "面前", "身上", "心里", "眼前", "此刻", "现在",
    "可能", "似乎", "仿佛", "其中", "之间", "内心", "情绪", "感觉", "目光",
    "动作", "反应", "结果", "过程", "方式", "问题",
})

_BEAT_SPLIT_RE = re.compile(
    "(?:" + "|".join(re.escape(t) for t in sorted(_BEAT_STOP_TOKENS, key=len, reverse=True)) + ")"
    "|[^一-鿿A-Za-z0-9]+"
)


def _beat_anchor_fragments(beat: str, max_anchors: int = 6) -> list[str]:
    """Extract the distinctive content fragments a beat promises.

    Splits the beat on particles/common verbs/punctuation and keeps 2-8 char
    CJK fragments that aren't generic filler. Longer fragments are preferred
    (more distinctive). Returns [] for fully abstract beats — those cannot be
    judged deterministically and auto-pass.
    """
    text = str(beat or "").strip()
    if not text:
        return []
    fragments: list[str] = []
    seen: set[str] = set()
    for frag in _BEAT_SPLIT_RE.split(text):
        frag = (frag or "").strip()
        if not (2 <= len(frag) <= 8):
            continue
        if not re.search(r"[一-鿿]", frag):
            continue
        if frag in _BEAT_GENERIC_FRAGMENTS or frag in seen:
            continue
        seen.add(frag)
        fragments.append(frag)
    fragments.sort(key=len, reverse=True)
    return fragments[:max_anchors]


def _fragment_hit(fragment: str, chapter_text: str, chapter_bigrams: set[str], min_bigram_cov: float = 0.7) -> bool:
    """True when the chapter plausibly realizes this anchor fragment.

    Exact substring first; for fragments >=3 chars, fall back to bigram
    coverage so loose rewording ("安瓿碎裂方向" vs "安瓿的碎裂方向") still
    counts. A chapter that never mentions the object at all fails both.
    """
    if fragment in chapter_text:
        return True
    if len(fragment) < 3 or not chapter_bigrams:
        return False
    grams = {fragment[i: i + 2] for i in range(len(fragment) - 1)}
    if not grams:
        return False
    return sum(1 for g in grams if g in chapter_bigrams) / len(grams) >= min_bigram_cov


def beat_coverage(
    chapter_text: str,
    plan: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic check that each plan beat's concrete anchors appear in prose.

    Returns:
      {
        "enabled": bool,
        "passed": bool,          # every anchored beat hit >=1 anchor AND
                                 # overall anchor hit-rate >= beat_coverage_min
        "coverage": float,       # matched anchors / total anchors (1.0 if none)
        "beats": [{"beat","anchors","missing","hit"}],
        "missing_beats": [...],  # beats with ZERO anchor hits (the repair list)
        "missing_anchors": [...] # flat list of all unmatched anchors
      }

    Conservative: beats with no extractable anchors auto-pass; matching accepts
    bigram-level rewording. A "pass" here is necessary, not sufficient — the
    LLM beats_audit still judges whether a mentioned beat was truly DRAMATIZED.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    enabled = bool(cfg.get("beat_coverage_enabled", True))
    result: dict[str, Any] = {
        "enabled": enabled, "passed": True, "coverage": 1.0,
        "beats": [], "missing_beats": [], "missing_anchors": [],
    }
    beats = plan.get("beats") if isinstance(plan, dict) else None
    if not enabled or not isinstance(beats, list) or not beats:
        return result
    body = _strip_title_line(str(chapter_text or ""))
    if len(body) < 500:
        # Too short to judge (provider refusal guard elsewhere refuses <500 anyway).
        return result

    min_cov = float(cfg.get("beat_coverage_min", 0.6))
    frag_bigram_cov = float(cfg.get("beat_coverage_fragment_bigram", 0.7))
    chapter_bigrams = _text_bigrams(body)

    total_anchors = 0
    total_hits = 0
    all_anchored_beats_hit = True
    for raw_beat in beats[:12]:
        beat = str(raw_beat or "").strip()
        if not beat:
            continue
        anchors = _beat_anchor_fragments(beat)
        hits = [a for a in anchors if _fragment_hit(a, body, chapter_bigrams, frag_bigram_cov)]
        missing = [a for a in anchors if a not in hits]
        beat_hit = (not anchors) or bool(hits)
        result["beats"].append({
            "beat": beat[:160], "anchors": anchors, "missing": missing, "hit": beat_hit,
        })
        total_anchors += len(anchors)
        total_hits += len(hits)
        result["missing_anchors"].extend(missing)
        if not beat_hit:
            all_anchored_beats_hit = False
            result["missing_beats"].append(beat[:200])

    coverage = (total_hits / total_anchors) if total_anchors else 1.0
    result["coverage"] = round(coverage, 3)
    result["passed"] = all_anchored_beats_hit and coverage >= min_cov
    return result


# ---------------------------------------------------------------------------
# Scene-skeleton dedupe: stop the engine from infinitely slicing one scene.
# ---------------------------------------------------------------------------

def _plan_skeleton_tokens(plan: dict[str, Any]) -> set[str]:
    """Character bigram set over a plan's concrete scene-defining fields."""
    parts: list[str] = []
    for key in ("conflict", "payoff", "pressure", "goal"):
        v = plan.get(key)
        if v:
            parts.append(str(v))
    beats = plan.get("beats")
    if isinstance(beats, list):
        parts.extend(str(b) for b in beats[:8])
    text = re.sub(r"[^一-鿿A-Za-z0-9]", "", " ".join(parts))
    if len(text) < 2:
        return set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def scene_similarity(plan: dict[str, Any], recent_plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Max Jaccard similarity of this plan's skeleton vs each recent plan.

    Returns {"max_sim": float, "most_similar_to": idx_or_None}. Used to detect
    the "endless slicing of the same micro-scene" failure mode at the planning
    stage, before any prose is written.
    """
    cur = _plan_skeleton_tokens(plan)
    best = 0.0
    best_i: int | None = None
    for i, rp in enumerate(recent_plans):
        if not isinstance(rp, dict):
            continue
        sim = _jaccard(cur, _plan_skeleton_tokens(rp))
        if sim > best:
            best = sim
            best_i = i
    return {"max_sim": round(best, 3), "most_similar_to": best_i}


# ---------------------------------------------------------------------------
# Plan executability gate: mechanize the arbiter's own stated "abstract-intent
# hard cap". The arbiter prompt repeats (3x, as prose) that a payoff/climax beat
# whose verb is "推导出/意识到/想通/完成/还原/引导/心算" with no concrete action +
# concrete object + visible result must score <=7.0 — but nothing ever enforced
# it. History (plan 8.0 -> draft 5-6) shows the LLM honour-system ignores it.
# This converts that rule into a deterministic check on the FINAL merged_plan.
# ---------------------------------------------------------------------------

# Verbs that signal a payoff stranded at "abstract realization" with no shootable
# action — the documented #1 cause of plan->draft score collapse.
_ABSTRACT_PAYOFF_VERBS = re.compile(
    r"(推导出|推理出|意识到|想通|想明白|明白了|反应过来|回过神|领悟|顿悟|"
    r"完成闭合|完成推演|还原(?:了)?真相|理清|厘清|心算|在心中|暗自推断|得出结论)"
)
# Concrete physical-action signals: a character operating a concrete object with a
# reader-visible result. If any of these co-occur with the abstract verb, the beat
# is doing real staging and is NOT blocked.
_CONCRETE_ACTION_SIG = re.compile(
    r"(把|将|抓住|按住?|压住?|划|举起?|摔|扔|递|撕|拼|对齐|并排|画(?:出|了)?|拍|掀|拽|"
    r"翻开|摊开|指着|塞进|拔出|插入|拧|敲|砸|拖|拎|捡起|铺开|贴在|钉在|挂在)"
)


def plan_executability_gate(plan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic check that the plan's payoff/climax is a shootable action.

    Returns {"blocked": bool, "evidence": str}. Blocked when the core payoff (and
    final beats) read as abstract realization with NO concrete physical action —
    exactly the failure the arbiter is told to cap at 7.0 but never mechanically
    enforces. Gated by `plan_executability_gate_enabled` (default true).
    """
    if not bool(config["novel"].get("plan_executability_gate_enabled", True)):
        return {"blocked": False, "evidence": ""}
    beats = plan.get("beats")
    tail_beats = [str(b) for b in beats[-3:]] if isinstance(beats, list) else []
    core = str(plan.get("payoff", "")) + " " + " ".join(tail_beats)
    if not core.strip():
        return {"blocked": False, "evidence": ""}
    if _ABSTRACT_PAYOFF_VERBS.search(core) and not _CONCRETE_ACTION_SIG.search(core):
        ev = (plan.get("payoff") or (tail_beats[-1] if tail_beats else ""))
        return {"blocked": True, "evidence": str(ev)[:160]}
    return {"blocked": False, "evidence": ""}


# ---------------------------------------------------------------------------
# Narrative-pattern dedupe: catch the "same procedural skeleton, different
# wording" failure that字面 Jaccard (scene_similarity) is blind to.
#
# scene_similarity matches on concrete tokens (新故事 vs 换水位 share almost no
# bigrams → max_sim low → passes), but the *abstract action flow* can be
# identical: 进入封闭空间 → 现场取证 → 数据比对 → 得出结论, chapter after chapter.
# That is exactly what dragged suspense_10ch Ch3(8.0)→Ch8(6.5): reviewers flagged
# "同一套流程骨架，只是把取证对象替换" as reader_fatigue, but no deterministic gate
# caught it. This classifies each plan into an ordered sequence of abstract
# "moves" and measures how identical that move-sequence is to recent chapters.
# ---------------------------------------------------------------------------

# Abstract narrative "moves". Each move maps to trigger lexemes that may appear
# anywhere in the plan's goal/conflict/payoff/beats free text. Order is detected
# from first-occurrence position in the concatenated beats, so two chapters that
# run the same moves in the same order score as duplicates regardless of wording.
_NARRATIVE_MOVES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("enter_space", (
        "进入", "走进", "来到", "抵达", "推开门", "打开门", "下到", "爬进",
        "钻进", "返回", "回到", "赶到", "开车到", "停在", "进了",
    )),
    ("collect_evidence", (
        "取证", "勘查", "勘察", "查看", "检查", "翻找", "搜查", "采集",
        "拍照", "记录", "测量", "提取", "采样", "调取", "翻出", "找到",
        "发现", "翻看", "查阅", "调档", "调取记录", "调日志",
    )),
    ("compare_data", (
        "比对", "对照", "核对", "对比", "比照", "印证", "吻合", "一致",
        "不一致", "对上", "对不上", "校验", "复核", "交叉", "比一比",
    )),
    ("deduce_conclusion", (
        "推断", "推理", "推导", "得出", "结论", "断定", "判定", "认定",
        "意识到", "明白", "想通", "反推", "证明", "说明", "确认", "看穿",
    )),
    ("confront_person", (
        "对峙", "质问", "逼问", "追问", "摊牌", "对质", "找上", "盘问",
        "拦住", "堵住", "面对面", "约见", "见面", "谈判",
    )),
    ("new_threat", (
        "威胁", "跟踪", "尾随", "被盯", "危险", "袭击", "警告", "恐吓",
        "逃", "追", "险些", "差点", "失踪", "失联", "出事", "意外",
    )),
    ("reveal_twist", (
        "反转", "翻转", "颠覆", "竟然", "原来", "真相", "其实", "并非",
        "另有", "嫁祸", "栽赃", "误导", "假象", "骗局",
    )),
)


def _narrative_pattern_sequence(plan: dict[str, Any]) -> list[str]:
    """Detect the ordered sequence of abstract narrative moves in a plan.

    Builds one position-tagged text from beats (ordered) plus the free-text
    plan fields, finds the first character offset at which each move's lexemes
    appear, and returns the moves sorted by that offset — i.e. the chapter's
    abstract "shape" (enter → collect → compare → deduce …) independent of the
    concrete subject matter.
    """
    beats = plan.get("beats")
    ordered_parts: list[str] = []
    if isinstance(beats, list):
        ordered_parts.extend(str(b) for b in beats[:12])
    # Append free-text fields after beats so a move mentioned only in
    # conflict/payoff still registers, but ordering is driven by the beats.
    for key in ("goal", "conflict", "pressure", "payoff", "hook"):
        v = plan.get(key)
        if v:
            ordered_parts.append(str(v))
    text = "\n".join(ordered_parts)
    if not text.strip():
        return []
    first_pos: dict[str, int] = {}
    for move, lexemes in _NARRATIVE_MOVES:
        best = -1
        for lex in lexemes:
            idx = text.find(lex)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
        if best != -1:
            first_pos[move] = best
    return [m for m, _ in sorted(first_pos.items(), key=lambda kv: kv[1])]


def _sequence_similarity(a: list[str], b: list[str]) -> float:
    """Similarity of two ordered move-sequences.

    Blends set overlap (which moves appear) with ordered-bigram overlap (the
    flow), so "enter→collect→compare→deduce" twice scores ~1.0 while a plan that
    swaps in confront/threat/reveal moves scores low even if it still collects
    evidence somewhere.
    """
    if not a or not b:
        return 0.0
    set_sim = _jaccard(set(a), set(b))
    bigrams_a = {(a[i], a[i + 1]) for i in range(len(a) - 1)}
    bigrams_b = {(b[i], b[i + 1]) for i in range(len(b) - 1)}
    if bigrams_a or bigrams_b:
        order_sim = (
            len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
            if (bigrams_a | bigrams_b) else 0.0
        )
    else:
        # Single-move sequences: ordering carries no information, lean on set_sim.
        order_sim = set_sim
    # Weight set-overlap higher than exact order: real monotony (suspense_10ch
    # Ch5-Ch7) reused the SAME moves (set jaccard 0.67-0.83) merely reshuffled,
    # so an even split would let "same moves, different order" slip under warn.
    # Reusing the move *vocabulary* is itself the fatigue; order is secondary.
    return 0.7 * set_sim + 0.3 * order_sim


def narrative_pattern_repetition(
    plan: dict[str, Any],
    recent_plans: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a plan that reruns the same abstract narrative flow as recent ones.

    Unlike ``scene_similarity`` (字面 token Jaccard), this compares the ORDERED
    sequence of abstract moves (enter_space → collect_evidence → compare_data →
    deduce_conclusion → …). It is the gate against "同一套流程骨架，只换取证对象"
    monotony — the documented cause of the Ch3→Ch8 score decline in suspense_10ch.

    Returns {"metrics", "level" (ok/warn/block), "max_sim", "most_similar_to",
    "sequence", "consecutive", "penalty", "flags", "directives"}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "level": "ok", "max_sim": 0.0, "most_similar_to": None,
        "sequence": [], "consecutive": 0,
        "penalty": 0.0, "flags": [], "directives": [],
    }
    if not bool(cfg.get("narrative_pattern_enabled", True)):
        return result
    cur = _narrative_pattern_sequence(plan)
    result["sequence"] = cur
    # A sequence too short to be a recognisable "flow" carries no signal.
    if len(cur) < int(cfg.get("narrative_pattern_min_moves", 3)):
        return result
    sims: list[float] = []
    best = 0.0
    best_i: int | None = None
    warn = float(cfg.get("narrative_pattern_sim_warn", 0.7))
    for i, rp in enumerate(recent_plans):
        if not isinstance(rp, dict):
            sims.append(0.0)
            continue
        sim = _sequence_similarity(cur, _narrative_pattern_sequence(rp))
        sims.append(sim)
        if sim > best:
            best = sim
            best_i = i
    # Consecutive run of recent chapters (newest-first ordering expected) that
    # share this flow ≥ warn — a single dup is tolerable, a *streak* is the
    # fatigue signal.
    consecutive = 0
    for s in sims:
        if s >= warn:
            consecutive += 1
        else:
            break
    result["metrics"] = {
        "max_sim": round(best, 3),
        "consecutive_similar": consecutive,
        "compared": len(sims),
    }
    result["max_sim"] = round(best, 3)
    result["most_similar_to"] = best_i
    result["consecutive"] = consecutive

    block_streak = int(cfg.get("narrative_pattern_block_streak", 2))
    block_sim = float(cfg.get("narrative_pattern_sim_block", 0.85))
    seq_label = "→".join(cur)
    if consecutive >= block_streak or best >= block_sim:
        result["level"] = "block"
        result["penalty"] = float(cfg.get("narrative_pattern_block_penalty", 1.5))
        result["flags"].append(
            f"narrative_pattern_repeat(streak={consecutive},max_sim={best:.2f})"
        )
        result["directives"].append(
            f"本章叙事流程骨架（{seq_label}）与近 {consecutive or 1} 章高度雷同，"
            "属于'同一套流程换个取证对象'的审美疲劳模式。必须改变章节的叙事形状："
            "例如把'静态取证→比对→推理'换成由人物对峙/外部威胁/时间压力驱动的场景，"
            "或调整信息揭示顺序（先抛结论再倒查、让对手先行动），不得再走一遍线性取证流程。"
        )
    elif best >= warn:
        result["level"] = "warn"
        result["penalty"] = float(cfg.get("narrative_pattern_warn_penalty", 0.6))
        result["flags"].append(f"narrative_pattern_repeat(max_sim={best:.2f})")
        result["directives"].append(
            f"本章叙事流程（{seq_label}）与近期相似度偏高，有流程化倾向。"
            "请让本章的推进方式与上一章不同——换一种场景驱动力或信息揭示顺序，避免连续线性取证。"
        )
    return result


# ---------------------------------------------------------------------------
# Visual contradiction payoff gate: keep mystery reveals concrete.
# ---------------------------------------------------------------------------

_ABSTRACT_DEDUCTION_TERMS = (
    "光源方向", "光源角度", "阴影方向", "反射路径", "几何关系", "角度计算",
    "比例关系", "透视关系", "逻辑推导", "推理出", "反推出", "说明存在",
    "不一致", "不合理", "异常", "矛盾",
)

_VISUAL_CONTRADICTION_PATTERNS = (
    ("presence_absence", ("有", "没有", "不见", "消失", "多出", "少了", "缺失", "出现")),
    ("left_right", ("左", "右", "反", "正", "镜像", "左右颠倒")),
    ("before_after", ("先", "后", "原本", "现在", "死前", "死后", "临终", "现实")),
    ("state_change", ("干", "湿", "新", "旧", "亮", "暗", "完整", "破裂", "裂纹", "血迹")),
    ("body_object", ("手表", "戒指", "钥匙", "纽扣", "袖口", "鞋印", "压痕", "伤口", "表带", "链节")),
    ("reflection_shadow", ("镜中", "倒影", "镜面", "影子", "反光", "投影")),
)

_CONCRETE_VISUAL_NOUNS = (
    "手", "手腕", "脸", "眼", "衣", "袖", "鞋", "门", "窗", "镜", "表", "戒指",
    "钥匙", "血", "水", "泥", "灰", "照片", "相机", "灯", "火", "绳", "锁",
)


def plan_visual_payoff_check(plan: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect abstract mystery payoffs before prose generation.

    Mystery chapters work best when the reveal lands as a visible contradiction
    the reader can inspect: "镜中有表 / 尸体现实无表", "左手在画面里举起 /
    现实垂落", "照片里反光在右 / 现场光源在左".  Plans that lean only on
    abstract deductions ("阴影方向不对", "光源角度矛盾") tend to produce
    low-payoff chapters even if the logic is sound. This deterministic gate does
    not judge truth; it checks whether the plan gives the writer a concrete
    visual task instead of an abstract reasoning slogan.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    fields: list[str] = []
    for key in ("goal", "conflict", "payoff", "pressure", "hook", "info_source", "risk"):
        v = plan.get(key)
        if v:
            fields.append(str(v))
    beats = plan.get("beats")
    if isinstance(beats, list):
        fields.extend(str(b) for b in beats[:12])
    text = "\n".join(fields)
    if not text.strip():
        return {
            "score": 0.0,
            "flags": ["empty_plan"],
            "directives": ["大纲缺少可检查文本，必须补齐 goal/conflict/payoff/beats。"],
            "template_hits": [],
            "abstract_hits": [],
            "concrete_hits": [],
            "blocked": True,
        }

    abstract_hits = [term for term in _ABSTRACT_DEDUCTION_TERMS if term in text]
    template_hits: list[str] = []
    for name, terms in _VISUAL_CONTRADICTION_PATTERNS:
        count = sum(1 for t in terms if t in text)
        if count >= 2 or (name in {"body_object", "reflection_shadow"} and count >= 1):
            template_hits.append(name)
    concrete_hits = [term for term in _CONCRETE_VISUAL_NOUNS if term in text]
    has_payoff = bool(str(plan.get("payoff") or "").strip())
    revealish = str(plan.get("payoff_type") or "").strip() in {"reveal", "reversal", "emotional", "strategic_setup", ""}

    score = 5.0
    score += min(3.0, len(set(template_hits)) * 0.75)
    score += min(1.5, len(set(concrete_hits)) * 0.15)
    if abstract_hits and len(set(template_hits)) < 2:
        score -= min(2.5, 0.7 * len(set(abstract_hits)))
    if not has_payoff:
        score -= 1.5
    score = max(1.0, min(10.0, round(score, 1)))

    min_score = float(cfg.get("visual_payoff_min_score", 7.0))
    blocked = bool(cfg.get("visual_payoff_blocks_plan", True)) and revealish and score < min_score
    flags: list[str] = []
    directives: list[str] = []
    if abstract_hits and len(set(template_hits)) < 2:
        flags.append("abstract_visual_payoff")
        directives.append(
            "核心推理爽点过抽象：不要只写'光源/阴影/角度不对'。必须改成读者一眼能懂的视觉矛盾，"
            "例如：画面里有某物而现实没有、镜中左右相反、死前姿态与尸体现状不一致、照片/倒影与现场状态冲突。"
        )
    if len(set(concrete_hits)) < 4:
        flags.append("not_enough_physical_anchors")
        directives.append(
            "本章 payoff 至少绑定 2 个可触摸/可观察物件或身体状态，如手腕压痕、表带链节、血迹方向、钥匙齿痕、照片反光。"
        )
    if not has_payoff:
        flags.append("missing_payoff")
        directives.append("大纲必须明确写出本章读者获得什么兑现，而不是只推进调查或铺设疑问。")

    return {
        "score": score,
        "flags": flags,
        "directives": directives[:4],
        "template_hits": sorted(set(template_hits)),
        "abstract_hits": abstract_hits[:8],
        "concrete_hits": concrete_hits[:12],
        "blocked": blocked,
    }
