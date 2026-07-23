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
import json
import re
from datetime import datetime
from typing import Any

# Sentence-ending punctuation for Chinese prose.
_SENTENCE_ENDERS = "。！？…"
_EM_DASH = "——"

# 书面腔连接词/虚词——下沉语体的反模式（低门槛口语体应改用"但是/所以/结果"等）。
_BOOKISH_CONNECTIVES = re.compile(
    r"然而|虽然|尽管|诸如|之于|继而|倘若|纵使|抑或|从而|遂|故而|"
    r"与此同时|不仅如此|更兼|愈发|颇为|不啻|乃是|实乃|须知"
)

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


_PREFIX_SET_CACHE: dict[str, frozenset[str]] = {}


def _get_cached_prefix_set(text: str, prefix_len: int = 8) -> frozenset[str]:
    """Return normalized clause-prefix set for template-fossil detection."""
    key = hashlib.md5(
        (text[:200] + text[-100:] + str(prefix_len)).encode("utf-8", errors="replace")
    ).hexdigest()
    if key not in _PREFIX_SET_CACHE:
        if len(_PREFIX_SET_CACHE) >= _CLAUSE_CACHE_MAX:
            evict = list(_PREFIX_SET_CACHE.keys())[: _CLAUSE_CACHE_MAX // 2]
            for k in evict:
                del _PREFIX_SET_CACHE[k]
        prefixes: set[str] = set()
        for c in _clause_segments(text, min_len=prefix_len + 2):
            nc = _normalize_clause(c)
            prefixes.add(nc[:prefix_len])
        _PREFIX_SET_CACHE[key] = frozenset(prefixes)
    return _PREFIX_SET_CACHE[key]


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
    tech_history: list[float] | None = None,
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

    `tech_history` is the tech-jargon-per-kchar sequence of prior chapters
    (oldest→newest), reserved for a trend term on the OPPOSITE collapse mode
    (overwriting / instrument-report register). Accepted from day one so call
    sites plumb it once; the static conjunction check below is the active
    detector.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    body = _strip_title_line(text)
    n = len(body)
    metrics: dict[str, Any] = {}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    # Register split (Gap-3): 免费流（番茄/七猫）是下沉口语体，要短句、低阅读门槛，
    # 平均句长阈值更宽，避免把"健康的下沉短句文"误判为碎句塌缩并反向扣分。
    # 反碎句塌缩的破折号密度/断行/对话检查不随之放宽——只解耦"书面腔长句"与"碎句塌缩"两个目标。
    _preset = str(cfg.get("platform_preset", "")).strip().lower()
    _low_barrier = (
        _preset in {"fanqie_free", "qimao_free"}
        or bool(cfg.get("style_low_barrier_register", False))
    )

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

    # --- 1b. TREND term: rising em-dash density is itself a collapse signal ----
    # Two failure modes this catches:
    #  (a) Slow drift BELOW the absolute warn threshold (em creeps 0.94→4.15
    #      monotonically while always < 6.0) — never trips the static tier.
    #  (b) A sustained climb ABOVE warn (6.6→7.8→8.8) — the static tier flat-lines
    #      at +1.0 and the acceleration is lost exactly when it matters most. This
    #      check runs REGARDLESS of the static tier (it used to be the static
    #      `else`, so it died once em crossed warn), so a rising-while-already-high
    #      chapter compounds static(+1.0) + trend(+1.0) = block. Observed
    #      gudai50_v2 Ch20-24: em 6.6→8.8 stuck at a flat +1.0 for 5 chapters.
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
            # Graduated penalty: scale by how far above the baseline the
            # chapter sits, instead of a flat +1.0 that blocks marginal cases.
            ratio = em_per_kchar / base if base > 0 else 3.0
            if ratio >= 3.0:
                trend_penalty = 1.0
            elif ratio >= 2.5:
                trend_penalty = 0.8
            elif ratio >= 2.0:
                trend_penalty = 0.5
            else:
                trend_penalty = 0.3
            penalty += trend_penalty
            flags.append(
                f"em_dash_trend_rise({em_per_kchar:.1f}/k vs mean {base:.1f}/k, ratio={ratio:.1f}x, pen={trend_penalty:.1f})"
            )
            # Avoid a near-duplicate directive when the static tier already told
            # the writer to cut em-dashes; the trend flag still surfaces for logs.
            if em_per_kchar < em_warn:
                directives.append(
                    f"文体趋势预警：破折号密度从近几章均值 {base:.1f}/千字升到 "
                    f"{em_per_kchar:.1f}/千字，正在向碎句化滑坡（即使尚未触顶阈值）。"
                    "本章必须主动收敛破折号，回到完整句叙事。"
                )
            else:
                directives.append(
                    f"破折号密度仍在上升（{base:.1f}→{em_per_kchar:.1f}/千字）且已超阈值，"
                    "本章必须显著回收破折号，否则判定为文体塌缩。"
                )

        # Sustained-collapse escalation: once the collapse has run long enough that
        # the recent MEAN is itself above warn, the multiplicative trend test goes
        # quiet (each step is < rise_mult× a now-high baseline) — the boiling-frog
        # gap. A plateau where BOTH the current chapter and its recent mean sit
        # above warn is not noise, it is the new (collapsed) normal, so add the
        # escalation that the trend term can no longer supply. This is what turns a
        # sustained 6.6→8.8 stretch into a block instead of a flat +1.0 forever.
        if (
            bool(cfg.get("style_em_dash_sustained_block", True))
            and em_per_kchar >= em_warn
            and base >= em_warn
        ):
            penalty += 1.0
            if not any(f.startswith("em_dash_sustained") for f in flags):
                flags.append(
                    f"em_dash_sustained({em_per_kchar:.1f}/k, mean {base:.1f}/k≥{em_warn})"
                )

    # --- 2. Average sentence length (collapse → very short sentences) ------
    # Split on sentence enders; measure mean length of non-empty segments.
    segments = [s for s in re.split(f"[{_SENTENCE_ENDERS}\n]", body) if s.strip()]
    if segments:
        avg_seg = sum(len(s.strip()) for s in segments) / len(segments)
        metrics["avg_sentence_chars"] = round(avg_seg, 1)
        # 免费流用更宽的下限（默认 9），起点/付费仍用 12；解决"下沉短句被罚"的冲突。
        if _low_barrier:
            min_avg = float(cfg.get("style_min_avg_sentence_chars_free", 9.0))
        else:
            min_avg = float(cfg.get("style_min_avg_sentence_chars", 12.0))
        if avg_seg < min_avg:
            penalty += 1.0
            flags.append(f"sentences_too_short(avg={avg_seg:.1f}<{min_avg})")
            # Bidirectional convergence: when em-dash density was just suppressed,
            # prose tends to overshoot into staccato single-clause lines (observed
            # v5 Ch4: em 0.3/k but avg sentence 11.5 chars). Em-suppression alone
            # is not "healthy" — pair it with an explicit "write fuller compound
            # sentences" directive so the writer doesn't trade one collapse mode
            # (em-fragments) for another (telegraphic shorts). For免费流 the target
            # is lower (口语成句即可)，避免反向逼出不合下沉调性的书面腔长句。
            em_low = em_per_kchar < float(cfg.get("style_em_dash_per_kchar_warn", 6.0))
            pull_target = 11 if _low_barrier else 14
            if em_low:
                if _low_barrier:
                    directives.append(
                        f"上一章破折号已收敛，但平均句长仅 {avg_seg:.0f} 字、滑向碎句化（单词短句堆叠）。"
                        f"本章在保持大白话、低阅读门槛的前提下用通顺成句叙事，把平均句长拉回 {pull_target} 字以上，"
                        "可以短但要成句，不要把一句话拆成多个无谓断句。"
                    )
                else:
                    directives.append(
                        f"上一章破折号已收敛，但平均句长仅 {avg_seg:.0f} 字、滑向了另一种碎句化（短促单句堆叠）。"
                        "本章请用带从句/状语的完整复合长句承载叙事与心理，"
                        f"在不重新堆破折号的前提下把平均句长拉回 {pull_target} 字以上。"
                    )
            else:
                directives.append(
                    f"上一章平均句长仅 {avg_seg:.0f} 字，过于碎片化。本章请写"
                    + ("通顺成句的口语叙事（可短但要成句），" if _low_barrier else "完整、连贯的句子，")
                    + "避免把一句话拆成多个单词短句。"
                )

        # --- 2b. 句长上限带（过度书写塌缩 = 碎句塌缩的镜像） -----------------
        # v12 huangliang Ch60-100：正文塌缩为"伪技术过度书写体"——超长句一逗到底、
        # 通篇说明书腔，而 LLM 自评反而打到 9.7。上面的下限只防碎句化；这里补上限。
        # 阈值题材分档（历史/悬疑容忍更长的书面句），见 config._genre_profile。
        max_avg = float(cfg.get("style_max_avg_sentence_chars", 42.0))
        bad_mult = float(cfg.get("style_max_avg_sentence_bad_mult", 1.3))
        if avg_seg > max_avg * bad_mult:
            penalty += 2.0
            flags.append(
                f"sentences_overlong_severe(avg={avg_seg:.1f}>{max_avg * bad_mult:.0f})"
            )
            directives.append(
                f"严重文体问题：上一章平均句长高达 {avg_seg:.0f} 字，超长句一逗到底，"
                "读起来像说明书。本章把复合长句拆成主谓宾清晰的短句，"
                "恢复正常句号节奏，长短句交替，让读者能喘气。"
            )
        elif avg_seg > max_avg:
            penalty += 1.0
            flags.append(f"sentences_too_long(avg={avg_seg:.1f}>{max_avg})")
            directives.append(
                f"上一章平均句长 {avg_seg:.0f} 字、偏向过度书写。本章拆分冗长复合句，"
                "多用句号收束，长短句交替，避免一逗到底。"
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

    # --- 4b. 对话字符占比下限（过度书写塌缩的第二症状：整章没人说话） --------
    # 存在性检查（<3 对引号）抓不住"有零星引号但通篇是叙述/说明"的章。
    # 这里量化：引号内字符 ÷ 正文字符。阈值题材分档（悬疑/历史容忍低对话）。
    dlg_chars = sum(len(m) for m in re.findall(r"“[^“”]{1,300}”", body))
    dlg_chars += sum(len(m) for m in re.findall(r"「[^「」]{1,300}」", body))
    dialogue_ratio = dlg_chars / n
    metrics["dialogue_char_ratio"] = round(dialogue_ratio, 3)
    ratio_min = float(cfg.get("style_dialogue_ratio_min", 0.04))
    # Only flag if the chapter is long enough that some dialogue is expected.
    if n > 2000 and ratio_min > 0 and dialogue_ratio < ratio_min:
        penalty += 1.0
        flags.append(f"dialogue_starved({dialogue_ratio:.1%}<{ratio_min:.0%})")
        directives.append(
            f"上一章对话占比仅 {dialogue_ratio:.0%}，几乎全是叙述。本章至少 {ratio_min:.0%} 的篇幅"
            "用有潜台词的人物对白推进情节，让信息从人物嘴里说出来而不是叙述灌输。"
        )
    elif n > 2000 and quote_pairs < 3:
        # 存在性检查是占比检查的真子集，仅在占比检查未触发/被禁用时兜底，不叠加。
        penalty += 0.5
        flags.append("almost_no_dialogue")
        directives.append("上一章几乎没有对话，本章请加入有潜台词的人物对白。")

    # --- 6. 伪技术腔（过度书写塌缩的标志性症状：像仪器报告，没人说话） --------
    # v12 huangliang Ch50-100 实测：塌缩章 = 技术黑话密度(频率/脉冲/共振/毫米…)
    # ≥12/k 且对话占比 <2%；而黑话高但对话充足的书（数据面板类爽文）读感正常。
    # 离线校准结论：单看数字密度不可分（健康悬疑 8-16/k > 塌缩章 2-5/k），
    # 必须用 [黑话高 × 对话枯竭] 的合取才是"仪器报告体"的确定性指纹。
    if bool(cfg.get("style_pseudo_precision_enabled", True)) and n >= 500:
        kchars = n / 1000.0
        tech_per_kchar = len(_PSEUDO_TECH_TERMS.findall(body)) / max(kchars, 0.1)
        metrics["tech_per_kchar"] = round(tech_per_kchar, 2)
        pp_warn = float(cfg.get("style_tech_jargon_per_kchar_warn", 8.0))
        pp_bad = float(cfg.get("style_tech_jargon_per_kchar_bad", 12.0))
        pp_dlg_max = float(cfg.get("style_tech_jargon_dialogue_max", 0.06))
        _pp_directive = (
            "严重文体问题：上一章堆砌技术名词与伪精确测量值（频率/脉冲/共振/零点X毫米），"
            "且几乎没有人物对话，读起来像仪器报告而不是小说。本章停止一切技术腔描写，"
            "把信息放进动作、对白和情绪里，让人物开口说话。"
        )
        if tech_per_kchar >= pp_bad and dialogue_ratio < pp_dlg_max:
            penalty += 2.0
            flags.append(
                f"pseudo_tech_collapse({tech_per_kchar:.1f}/k≥{pp_bad},dlg={dialogue_ratio:.1%})"
            )
            directives.append(_pp_directive)
        elif tech_per_kchar >= pp_warn and dialogue_ratio < pp_dlg_max:
            penalty += 1.0
            flags.append(
                f"pseudo_tech_high({tech_per_kchar:.1f}/k≥{pp_warn},dlg={dialogue_ratio:.1%})"
            )
            directives.append(_pp_directive)
        elif tech_per_kchar >= pp_bad:
            # 黑话高但对话充足：不罚分（数据面板类爽文的合法形态），只提醒收敛。
            directives.append(
                f"上一章技术名词密度偏高（{tech_per_kchar:.1f}/千字）。对话充足所以暂不扣分，"
                "但请注意用感官与比喻替代部分技术描述，防止滑向仪器报告腔。"
            )
        # tech_history 趋势项预留：静态合取已在校准回放中抓住塌缩段，趋势逻辑缓做。
        _ = tech_history

    # --- 5. 下沉语体校准（仅 low_barrier 模式）：罚书面腔，奖大白话 ----------
    # 番茄下沉读者要低阅读门槛口语体。这里在免费流/显式下沉模式下：
    #  (a) 书面腔连接词密度过高 → 小额扣分 + 改口语指令；
    #  (b) prose 已是健康大白话（句长在带内、有对话、破折号低、书面腔少）→ 发正向 directive 巩固。
    if _low_barrier and n >= 500:
        bookish = len(_BOOKISH_CONNECTIVES.findall(body))
        bookish_per_kchar = bookish / (n / 1000.0)
        metrics["bookish_per_kchar"] = round(bookish_per_kchar, 2)
        bookish_warn = float(cfg.get("style_bookish_per_kchar_warn", 2.0))
        if bookish_per_kchar >= bookish_warn:
            penalty += float(cfg.get("style_bookish_penalty", 0.5))
            flags.append(f"bookish_register({bookish_per_kchar:.1f}/k≥{bookish_warn})")
            directives.append(
                "下沉语体校准：上一章书面腔偏重（然而/虽然/尽管/诸如…密度过高）。"
                "本章改用大白话口语：用「但是/所以/结果/可是」等口语连接，去掉文绉绉的虚词，"
                "靠对话和具体动作推进，降低阅读门槛。"
            )
        else:
            avg_ok = metrics.get("avg_sentence_chars", 0) and not any(
                f.startswith("sentences_too_short") for f in flags)
            if (
                penalty == 0.0
                and avg_ok
                and quote_pairs >= 3
                and em_per_kchar < em_warn
            ):
                directives.append(
                    "下沉语体执行良好：大白话短句成句、对话充足、无碎句堆叠。本章保持这一调性，"
                    "继续低门槛口语体，每章给到具体爽点与章末钩子。"
                )

    penalty = round(min(penalty, float(cfg.get("style_penalty_cap", 4.0))), 2)
    metrics["penalty"] = penalty
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:4],
    }


# ---------------------------------------------------------------------------
# AI 味确定性检测 (anti-AI-flavor gate)
# ---------------------------------------------------------------------------
# AI 生成小说的"AI味"——读者一眼能辨的机械痕迹——不属于文风塌缩，而是一类
# 独立的质量退化：套话密集、比喻堆砌、情感贴标签、程度副词撑场面、总结式叙述、
# 段落结构千篇一律。这些症状的 LLM 自评完全失效（模型写出来的东西，模型自己
# 觉得"挺好的"），必须用确定性检测+前移惩罚来治。

_AI_CLICHE_PATTERNS = re.compile(
    # --- 情绪/微表情套话 ---
    r"心中一沉|心头一震|心中涌起|心底涌起|心中升起|心头涌上|"
    r"眼中闪过一丝|眼底闪过|目光闪烁|目光一凝|目光微凝|瞳孔一缩|瞳孔微缩|"
    r"嘴角微微上扬|嘴角勾起一抹|嘴角划过一丝|嘴角不自觉|嘴角微扬|"
    r"眉头微皱|眉头一皱|眉头紧锁|眉头微蹙|"
    r"倒吸一口凉气|倒吸一口冷气|浑身一颤|浑身一震|"
    r"心如刀绞|如释重负|心中五味杂陈|百感交集|"
    r"一股暖流|一股寒意|一阵恶寒|"
    # --- 动作套话 ---
    r"缓缓开口|缓缓说道|缓缓站起|缓缓走|缓缓闭上|缓缓睁开|"
    r"微微颔首|微微点头|轻轻点头|轻轻摇头|"
    r"目光如炬|目光灼灼|目光深邃|"
    r"身形一闪|身形一顿|脚步一顿|"
    r"负手而立|双拳紧握|双手紧握|攥紧了拳头|"
    r"深吸一口气|深深吸了一口气|长舒一口气|"
    # --- 叙述腔套话 ---
    r"一时间|此刻|这一刻|那一刻|一瞬间|刹那间|霎时间|"
    r"毫无疑问|不言而喻|众所周知|不出所料|"
    r"显然|事实上|实际上|说实话|不得不说|"
    r"仿佛|恍若|犹如|宛如|好似|一如"
)

_STALE_METAPHORS = re.compile(
    r"时间仿佛静止|时间好像静止|时间似乎凝固|"
    r"心如刀绞|心如刀割|"
    r"美得像画|美如画卷|"
    r"如同一记重锤|像一记重锤|仿佛重锤|"
    r"像是被抽空了|仿佛被抽空|"
    r"仿佛被钉在原地|像是被钉在|如同钉在|"
    r"如潮水般涌来|像潮水一样|如潮水般|"
    r"打翻了五味瓶|五味杂陈|"
    r"像是被泼了一盆冷水|如同一盆冷水|"
    r"仿佛过了一个世纪|像过了一个世纪|"
    r"命运的齿轮|历史的车轮|时代的洪流|"
    r"像是做了一场梦|如同一场梦"
)

_SIMILE_PATTERNS = re.compile(
    r"仿佛|犹如|宛如|恍若|好似|好像|一如|如同|像是|似乎|"
    r"般地|一般地|似的"
)

_TELL_NOT_SHOW = re.compile(
    r"[他她][感觉到了?|感到了?|知道|明白|清楚|意识到|觉得|心想|暗想|内心深处]"
    r".{0,6}"
    r"[震惊|愤怒|悲伤|恐惧|绝望|兴奋|激动|紧张|不安|焦虑|"
    r"开心|高兴|难过|痛苦|愤恨|沮丧|失落|孤独|恐慌|惊恐|"
    r"害怕|担忧|忧虑|欣慰|释然|无奈|茫然|困惑|惊讶|诧异]"
)

_DEGREE_ADVERBS = re.compile(
    r"非常|极其|十分|无比|格外|异常|万分|分外|"
    r"极为|极度|无限|莫大|至极|之极"
)

_NEGATIVE_PAIR = re.compile(
    r"没有.{1,15}[，,].{0,4}也没有|"
    r"不是.{1,15}[，,].{0,4}(?:也不是|更不是)|"
    r"不曾.{1,10}[，,].{0,4}也不曾|"
    r"无.{1,10}[，,].{0,4}(?:也无|亦无)|"
    r"并非.{1,10}[，,].{0,4}(?:也并非|更非)|"
    r"既不.{1,10}[，,].{0,4}也不"
)

_SUMMARY_NARRATION = re.compile(
    r"就这样|一切才刚刚开始|这只是.{0,2}开始|从此以后|自此|"
    r"一切都变了|一切都不同了|一切都已经|"
    r"命运的齿轮.{0,4}转动|故事远没有结束|新的篇章|"
    r"序幕才刚刚拉开|帷幕.{0,4}拉开|画上了句号|"
    r"一切尘埃落定|一个新的时代|历史的转折点|"
    r"冥冥之中|或许这就是|也许这就是|所谓的命运"
)


def ai_flavor_health(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic AI-flavor detection: clichés / metaphor spam / tell-not-show /
    degree-adverb inflation / summary narration / paragraph monotony.

    Returns the same {metrics, penalty, flags, directives} shape as style_health.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("ai_flavor_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    n = len(body)
    if n < 200:
        return {"metrics": {"chars": n}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {"chars": n}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0
    kchars = n / 1000.0

    # --- 1. AI cliché density ---
    cliche_matches = _AI_CLICHE_PATTERNS.findall(body)
    cliche_per_kchar = round(len(cliche_matches) / max(kchars, 0.1), 2)
    metrics["ai_cliche_count"] = len(cliche_matches)
    metrics["ai_cliche_per_kchar"] = cliche_per_kchar
    cliche_warn = float(cfg.get("ai_cliche_per_kchar_warn", 4.0))
    cliche_bad = float(cfg.get("ai_cliche_per_kchar_bad", 8.0))
    if cliche_per_kchar >= cliche_bad:
        penalty += 1.5
        flags.append(f"ai_cliche_overload({cliche_per_kchar:.1f}/k>={cliche_bad})")
        top_cliches = _top_n_matches(cliche_matches, 5)
        directives.append(
            "严重AI味：套话密度过高（%.1f/千字），整章读起来像AI生成模板。"
            "禁止使用以下表达及其变体：%s。"
            "用具体的、只属于本场景的身体反应/动作/环境变化替代。"
            % (cliche_per_kchar, "、".join(top_cliches))
        )
    elif cliche_per_kchar >= cliche_warn:
        penalty += 0.5
        top_cliches = _top_n_matches(cliche_matches, 4)
        directives.append(
            "AI味偏重：套话密度 %.1f/千字。减少以下表达：%s。"
            "换用新鲜的、贴合当前情境的具体描写。"
            % (cliche_per_kchar, "、".join(top_cliches))
        )

    # --- 2. Metaphor overload + stale metaphors ---
    simile_hits = _SIMILE_PATTERNS.findall(body)
    metaphor_per_kchar = round(len(simile_hits) / max(kchars, 0.1), 2)
    metrics["metaphor_per_kchar"] = metaphor_per_kchar
    stale_hits = _STALE_METAPHORS.findall(body)
    metrics["stale_metaphor_count"] = len(stale_hits)
    metaphor_warn = float(cfg.get("metaphor_per_kchar_warn", 5.0))
    if metaphor_per_kchar >= metaphor_warn:
        penalty += 0.5
        flags.append(f"metaphor_overload({metaphor_per_kchar:.1f}/k>={metaphor_warn})")
        directives.append(
            "比喻过载（%.1f/千字）：每千字比喻控制在3个以内，"
            "每个比喻必须新鲜准确且服务于情节，宁可朴素直白也不堆砌。"
            % metaphor_per_kchar
        )
    if len(stale_hits) >= 2:
        penalty += 0.5
        stale_examples = list(dict.fromkeys(stale_hits))[:4]
        flags.append(f"stale_metaphors({len(stale_hits)})")
        directives.append(
            "陈腐比喻 %d 处：%s。这些比喻已被用滥，禁止再用。"
            "用只属于本场景的新鲜意象替代。"
            % (len(stale_hits), "、".join("「%s」" % s for s in stale_examples))
        )

    # --- 3. Tell-not-show (emotion labeling) ---
    tns_hits = _TELL_NOT_SHOW.findall(body)
    tns_per_kchar = round(len(tns_hits) / max(kchars, 0.1), 2)
    metrics["tell_not_show_count"] = len(tns_hits)
    metrics["tell_not_show_per_kchar"] = tns_per_kchar
    tns_warn = float(cfg.get("tell_not_show_per_kchar_warn", 3.0))
    if tns_per_kchar >= tns_warn:
        penalty += 0.5
        flags.append(f"tell_not_show({tns_per_kchar:.1f}/k>={tns_warn})")
        directives.append(
            '情感贴标签（%.1f/千字）：不要写"他感到震惊/她觉得悲伤"，'
            '改为展示：震惊时手中的东西掉了、悲伤时沉默地做了某个动作。'
            '情绪必须通过行为、对话、生理反应间接呈现。'
            % tns_per_kchar
        )

    # --- 4. Degree-adverb inflation ---
    adv_hits = _DEGREE_ADVERBS.findall(body)
    adv_per_kchar = round(len(adv_hits) / max(kchars, 0.1), 2)
    metrics["adverb_count"] = len(adv_hits)
    metrics["adverb_per_kchar"] = adv_per_kchar
    adv_warn = float(cfg.get("adverb_inflation_per_kchar_warn", 4.0))
    if adv_per_kchar >= adv_warn:
        penalty += 0.5
        flags.append(f"adverb_inflation({adv_per_kchar:.1f}/k>={adv_warn})")
        directives.append(
            '程度副词泛滥（%.1f/千字）：删掉"非常/极其/十分/无比"等词，'
            '用精准的动词和具体的细节替代模糊的程度修饰。'
            % adv_per_kchar
        )

    # --- 5. Summary narration ---
    summary_hits = _SUMMARY_NARRATION.findall(body)
    metrics["summary_narration_count"] = len(summary_hits)
    if len(summary_hits) >= 2:
        penalty += 0.5
        examples = list(dict.fromkeys(summary_hits))[:3]
        flags.append(f"summary_narration({len(summary_hits)})")
        directives.append(
            "总结式叙述 %d 处：%s。删掉这类上帝视角的总结句，"
            "让读者从情节和角色行为中自行感受。"
            % (len(summary_hits), "、".join("「%s」" % s for s in examples))
        )

    # --- 6. Paragraph-start monotony ---
    paragraphs = [p.strip() for p in body.split("\n") if len(p.strip()) >= 8]
    if len(paragraphs) >= 6:
        starts = [p[:4] for p in paragraphs]
        from collections import Counter
        start_counts = Counter(starts)
        most_common_count = start_counts.most_common(1)[0][1] if start_counts else 0
        repeat_ratio = most_common_count / len(paragraphs)
        metrics["paragraph_start_repeat_ratio"] = round(repeat_ratio, 2)
        para_warn = float(cfg.get("paragraph_start_repeat_warn", 0.30))
        if repeat_ratio >= para_warn:
            dominant_start = start_counts.most_common(1)[0][0]
            penalty += 0.5
            flags.append(f"paragraph_monotony({repeat_ratio:.0%}>={para_warn:.0%})")
            directives.append(
                "段落开头单一：%.0f%%的段落以「%s」开头。"
                "变化段落的起始方式：对话、动作、环境、心理交替开篇。"
                % (repeat_ratio * 100, dominant_start)
            )

    # --- 7. Negative-pair constructions ("没有X，也没有Y") ---
    neg_hits = _NEGATIVE_PAIR.findall(body)
    neg_per_kchar = round(len(neg_hits) / max(kchars, 0.1), 2)
    metrics["negative_pair_count"] = len(neg_hits)
    metrics["negative_pair_per_kchar"] = neg_per_kchar
    neg_warn = float(cfg.get("negative_pair_per_kchar_warn", 2.0))
    neg_bad = float(cfg.get("negative_pair_per_kchar_bad", 4.0))
    if neg_per_kchar >= neg_bad:
        penalty += 1.0
        flags.append(f"negative_pair_overload({neg_per_kchar:.1f}/k>={neg_bad})")
        directives.append(
            "否定对仗句式泛滥（%.1f/千字）：「没有X，也没有Y」「不是X，也不是Y」"
            "是最明显的AI写作指纹。删去后半句或改写为正面描述。" % neg_per_kchar
        )
    elif neg_per_kchar >= neg_warn:
        penalty += 0.5
        flags.append(f"negative_pair({neg_per_kchar:.1f}/k>={neg_warn})")
        directives.append(
            "否定对仗偏多（%.1f/千字）：减少「没有X也没有Y」式句式，"
            "直接一句说完即可，不要对称排列两个否定分句。" % neg_per_kchar
        )

    cap = float(cfg.get("ai_flavor_penalty_cap", 3.0))
    penalty = round(min(penalty, cap), 2)
    metrics["penalty"] = penalty
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:6],
    }


def _top_n_matches(matches: list[str], n: int) -> list[str]:
    """Return the top-N most frequent matches, deduplicated."""
    from collections import Counter
    counts = Counter(matches)
    return [item for item, _ in counts.most_common(n)]


# ---------------------------------------------------------------------------
# 段落形态 + 模糊词密度检测 (paragraph shape + hedge word density)
# ---------------------------------------------------------------------------
_HEDGE_WORDS = re.compile(
    r"似乎|好像|仿佛|大概|或许|也许|可能|某种|某个|在某种程度上|一定程度|有所|不由得|不禁|不知为何"
)


def paragraph_shape_health(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic paragraph-uniformity and hedge-word density check.

    Returns the standard {metrics, penalty, flags, directives} shape.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("paragraph_shape_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    n = len(body)
    if n < 300:
        return {"metrics": {"chars": n}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {"chars": n}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    # --- 1. Paragraph length uniformity (coefficient of variation) ---
    paragraphs = [p for p in body.split("\n") if len(p.strip()) >= 20]
    if len(paragraphs) >= 5:
        lengths = [len(p.strip()) for p in paragraphs]
        mean_len = sum(lengths) / len(lengths)
        if mean_len > 0:
            variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
            std_len = variance ** 0.5
            cv = round(std_len / mean_len, 3)
            metrics["paragraph_count"] = len(paragraphs)
            metrics["paragraph_length_mean"] = round(mean_len, 1)
            metrics["paragraph_length_cv"] = cv
            cv_min = float(cfg.get("paragraph_cv_min", 0.15))
            if cv < 0.10:
                penalty += 1.0
                flags.append(f"paragraph_uniform_severe(cv={cv:.2f}<0.10)")
                directives.append(
                    "段落长度高度整齐（变异系数 %.2f），像 AI 流水线产出。"
                    "大幅增加段落长短交错——用1-2句短段制造节奏冲击，用长段深入细节。" % cv
                )
            elif cv < cv_min:
                penalty += 0.5
                flags.append(f"paragraph_uniform(cv={cv:.2f}<{cv_min})")
                directives.append(
                    "段落长度偏于整齐（变异系数 %.2f），增加段落的长短交错——"
                    "短段制造节奏感，长段深入细节。" % cv
                )

    # --- 2. Short paragraph detection (avg paragraph length) ---
    all_paras = [p for p in body.split("\n") if len(p.strip()) >= 8]
    if len(all_paras) >= 5:
        all_lens = [len(p.strip()) for p in all_paras]
        avg_para = sum(all_lens) / len(all_lens)
        short_count = sum(1 for l in all_lens if l < 30)
        metrics["avg_paragraph_chars"] = round(avg_para, 1)
        metrics["short_paragraph_ratio"] = round(short_count / len(all_lens), 2)
        severe_threshold = float(cfg.get("short_paragraph_severe", 30))
        warn_threshold = float(cfg.get("short_paragraph_warn", 50))
        if avg_para < severe_threshold:
            penalty += 1.5
            flags.append(f"short_paragraph_severe(avg={avg_para:.0f}<{severe_threshold:.0f})")
            directives.append(
                "AI碎段病严重：段均仅 %.0f 字，几乎每句话单独一行，观感极差。"
                "每段至少3-5句/60字以上（对话除外），把碎片合并成有起承转的段落。" % avg_para
            )
        elif avg_para < warn_threshold:
            penalty += 0.5
            flags.append(f"short_paragraph_warn(avg={avg_para:.0f}<{warn_threshold:.0f})")
            directives.append(
                "段落偏短（均 %.0f 字/段），合并相邻短句为完整段落，"
                "让叙事有呼吸感而非碎片堆砌。" % avg_para
            )

    # --- 3. Hedge word density ---
    kchars = n / 1000.0
    hedge_matches = _HEDGE_WORDS.findall(body)
    hedge_per_kchar = round(len(hedge_matches) / max(kchars, 0.1), 2)
    metrics["hedge_count"] = len(hedge_matches)
    metrics["hedge_per_kchar"] = hedge_per_kchar
    hedge_warn = float(cfg.get("hedge_per_kchar_warn", 5.0))
    hedge_bad = float(cfg.get("hedge_per_kchar_bad", 10.0))
    if hedge_per_kchar >= hedge_bad:
        penalty += 1.0
        flags.append(f"hedge_overload({hedge_per_kchar:.1f}/k>={hedge_bad})")
        top_hedges = _top_n_matches(hedge_matches, 3)
        directives.append(
            "模糊词密度过高（%.1f/千字），文风犹疑无力。"
            "删掉或替换：%s。用确定性描写替代模棱两可的叙述。"
            % (hedge_per_kchar, "、".join("「%s」" % h for h in top_hedges))
        )
    elif hedge_per_kchar >= hedge_warn:
        penalty += 0.5
        top_hedges = _top_n_matches(hedge_matches, 3)
        directives.append(
            "模糊词偏多（%.1f/千字）。减少：%s。换用确切的动作和事实。"
            % (hedge_per_kchar, "、".join("「%s」" % h for h in top_hedges))
        )

    cap = float(cfg.get("paragraph_shape_penalty_cap", 3.0))
    penalty = round(min(penalty, cap), 2)
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:4],
    }


# ---------------------------------------------------------------------------
# Q&A 乒乓对话检测 (dialogue ping-pong)
# ---------------------------------------------------------------------------
_DIALOGUE_RE = re.compile(r'“([^”]+)”')


def dialogue_pingpong(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect excessive question-answer ping-pong dialogue patterns.

    Counts consecutive dialogue turns where a question mark line is immediately
    followed by a non-question answer. High ratios indicate interview/interrogation
    style dialogue that feels mechanical.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("dialogue_pingpong_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    turns = _DIALOGUE_RE.findall(body)
    result: dict[str, Any] = {
        "metrics": {"dialogue_turns": len(turns)},
        "penalty": 0.0, "flags": [], "directives": [],
    }
    if len(turns) < 4:
        return result

    qa_pairs = 0
    for i in range(len(turns) - 1):
        if turns[i].rstrip().endswith("？") and not turns[i + 1].rstrip().endswith("？"):
            qa_pairs += 1
    qa_ratio = round(qa_pairs / max(len(turns) - 1, 1), 2)
    result["metrics"]["qa_pairs"] = qa_pairs
    result["metrics"]["qa_ratio"] = qa_ratio

    warn = float(cfg.get("dialogue_pingpong_warn", 0.50))
    bad = float(cfg.get("dialogue_pingpong_bad", 0.65))
    if qa_ratio >= bad:
        result["penalty"] = 1.0
        result["flags"].append(f"dialogue_pingpong_severe(qa={qa_ratio:.0%}>={bad:.0%})")
        result["directives"].append(
            "Q&A乒乓对话严重（%.0f%%为一问一答），读者会觉得像审讯。"
            "改为：多人交叉发言、用动作/心理/环境打断对话节奏、让角色主动说而非被问。" % (qa_ratio * 100)
        )
    elif qa_ratio >= warn:
        result["penalty"] = 0.5
        result["flags"].append(f"dialogue_pingpong(qa={qa_ratio:.0%}>={warn:.0%})")
        result["directives"].append(
            "Q&A对话偏多（%.0f%%），在对话间插入动作、神态、心理描写，"
            "打破采访式节奏。" % (qa_ratio * 100)
        )
    return result


# ---------------------------------------------------------------------------
# 章尾总结检测 (chapter ending summary quality)
# ---------------------------------------------------------------------------
_ENDING_SUMMARY_MARKERS = re.compile(
    r"他知道|她知道|他明白|她明白|他清楚|她清楚|"
    r"他意识到|她意识到|他理解|她理解|"
    r"这一切|而这一切|至此|至少.{0,6}知道|"
    r"心中.{0,4}清楚|心中.{0,4}明白|心里.{0,4}清楚|"
    r"一切都已|一切似乎|一切终于"
)


def chapter_ending_quality(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect dry summary-style chapter endings (complements intra_chapter_repetition).

    Scans the last 400 chars for summary markers and reflective-narration patterns
    that make the ending feel like an essay conclusion rather than forward momentum.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("chapter_ending_quality_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    result: dict[str, Any] = {
        "metrics": {}, "penalty": 0.0, "flags": [], "directives": [],
    }
    if len(body) < 800:
        return result

    tail_chars = int(cfg.get("ending_quality_tail_chars", 400))
    tail = body[-tail_chars:]
    markers = _ENDING_SUMMARY_MARKERS.findall(tail)
    marker_count = len(markers)
    result["metrics"]["ending_summary_markers"] = marker_count
    result["metrics"]["ending_tail_chars"] = len(tail)

    has_dialogue = "“" in tail
    tail_sentences = [s.strip() for s in re.split(r'[。！？\n]', tail) if len(s.strip()) >= 4]
    pronoun_start = sum(1 for s in tail_sentences if s and s[0] in "他她它")
    result["metrics"]["ending_pronoun_start_ratio"] = round(
        pronoun_start / max(len(tail_sentences), 1), 2
    )

    warn_threshold = int(cfg.get("chapter_ending_summary_warn", 3))
    bad_threshold = int(cfg.get("chapter_ending_summary_bad", 5))

    if marker_count >= bad_threshold or (marker_count >= warn_threshold and not has_dialogue):
        result["penalty"] = 1.0
        result["flags"].append(f"ending_summary_severe(markers={marker_count})")
        examples = list(dict.fromkeys(markers))[:3]
        result["directives"].append(
            "章末总结病：最后%d字有%d处总结性叙述（%s），像散文收尾。"
            "章末必须是前进的动作/对话/悬念，不是回顾式的'他知道/她明白'。"
            % (tail_chars, marker_count, "、".join("「%s」" % m for m in examples))
        )
    elif marker_count >= warn_threshold:
        result["penalty"] = 0.5
        result["flags"].append(f"ending_summary(markers={marker_count})")
        result["directives"].append(
            "章末总结倾向（%d处标志词）：减少'他知道/她明白/这一切'式收束，"
            "用动作或对话驱动结尾。" % marker_count
        )
    return result


# ---------------------------------------------------------------------------
# 黄金三句开篇闸门 (opening golden-three-sentences gate)
# ---------------------------------------------------------------------------
# 番茄 "3 秒定生死"：开篇必须把读者丢进"正在发生的危机"（动作/对话/具体冲突），
# 而不是景物/天气/时段/世界观铺垫。LLM 自评对文学性氛围开场打分偏高、抓不到这个
# 病灶，所以用确定性检测【反模式（开局铺垫）】——比正向检测"危机"更可靠。
_OPENING_BACKGROUND_MARKERS = re.compile(
    r"清晨|拂晓|黎明|黄昏|傍晚|日暮|夜色|夜幕|月光|月色|星空|阳光|晨光|天空|天色|"
    r"空气里?|微风|秋风|春风|寒风|细雨|小雨|大雨|雪花|薄雾|云雾|"
    r"很久很久|很久以前|从前|相传|传说|据说|某年|那一年|多年[前后]|纪元|"
    r"世界上|这片大陆|这个世界|大陆上|王朝|帝国"
)
_OPENING_ACTION_MARKERS = re.compile(
    r"喊|叫|吼|骂|嚷|扑|抓|拽|拖|拎|踹|踢|砸|摔|撞|冲|逃|跪|爬|血|刀|枪|剑|拳|"
    r"死|杀|抢|甩|揪|按|掐|捂|嘶|惨|救命|住手|滚|不许|危险|来不及|完了|糟了|"
    r"最后通牒|滚出去|放开|别动|站住"
)
_OPENING_DIALOGUE_OPEN = ("“", "「", "『", '"')
# 题材化"合格开场"标记：悬疑可用"线索/现场/异常"开场，言情可用"关系/情绪"开场——
# 这些都不是景物铺垫，不应被危机模式的反模式检测误伤。
_OPENING_CLUE_MARKERS = re.compile(
    r"尸|血|死|失踪|消失|案|线索|现场|诡异|规则|不对劲|反常|不合理|证据|凶|报警|"
    r"遗体|尖叫|惨叫|警察|命案|遇害|诅咒|怪|异常|消息|遗书|遗言|失联|藏|秘密"
)
_OPENING_RELATIONSHIP_MARKERS = re.compile(
    r"爱|恨|吻|拥抱|分手|离婚|结婚|前任|未婚|心动|嫉妒|背叛|表白|暧昧|情敌|"
    r"相亲|订婚|喜欢|讨厌|他和她|她和他|怀孕|追求|纠缠|旧情|重逢"
)


def opening_hook_gate(
    text: str,
    chapter_num: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic 黄金三句 opening gate for the first `opening_chapters` chapters.

    Penalizes the background-dump anti-pattern (opener is scenery / weather /
    time-of-day / world-setting exposition with no in-progress action or
    dialogue). Conservative: needs >=2 corroborating signals before it flags, so
    a legitimately tense narrative opening is not punished. Returns
    {penalty, flags, directives, block}; `block` is only set when
    `opening_golden_gate_block` is enabled.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {"penalty": 0.0, "flags": [], "directives": [], "block": False}
    if not bool(cfg.get("opening_golden_gate_enabled", True)):
        return result
    opening_chapters = int(cfg.get("opening_chapters", 3))
    if chapter_num <= 0 or chapter_num > opening_chapters:
        return result
    body = _strip_title_line(text or "").lstrip()
    if len(body) < 200:
        return result

    first_para = body.split("\n", 1)[0].strip()
    segs = [s for s in re.split(f"[{_SENTENCE_ENDERS}]", body) if s.strip()]
    first_sentence = (segs[0] if segs else body)[:120].strip()
    head = body[:200]  # opening window for dialogue/action detection

    has_dialogue_head = any(q in head for q in _OPENING_DIALOGUE_OPEN)
    has_action_head = bool(_OPENING_ACTION_MARKERS.search(head))
    bg_in_first = bool(_OPENING_BACKGROUND_MARKERS.search(first_sentence))

    # Genre-aware notion of a valid opening (opening_gate_mode set by the genre
    # detection profile): 爽文=crisis(动作/对话), 悬疑=clue(线索/现场/异常),
    # 言情=relationship(关系/情绪), 历史/中性=balanced(更宽松，只罚最严重纯景物).
    mode = str(cfg.get("opening_gate_mode", "crisis")).strip().lower()
    valid_extra = False
    if mode == "clue":
        valid_extra = bool(_OPENING_CLUE_MARKERS.search(head))
    elif mode == "relationship":
        valid_extra = bool(_OPENING_RELATIONSHIP_MARKERS.search(head))
    has_valid_open = has_dialogue_head or has_action_head or valid_extra

    signals: list[str] = []
    # Signal 1: first sentence is scenery/time/setting exposition.
    if bg_in_first and not has_valid_open:
        signals.append("opening_first_sentence_background")
    # Signal 2: a long, static, descriptive first sentence (no valid opening hook).
    if len(first_sentence) >= 50 and not has_valid_open:
        signals.append("opening_first_sentence_long_static")
    # Signal 3: the whole opening window has no genre-valid opening at all.
    if not has_valid_open:
        signals.append("opening_no_hook")

    # balanced (历史/中性) only flags the most egregious case (all signals);
    # crisis/clue/relationship flag at >=2 corroborating signals.
    need = 3 if mode == "balanced" else 2
    if len(signals) >= need:
        result["penalty"] = round(float(cfg.get("opening_golden_gate_penalty", 1.5)), 2)
        result["flags"].extend(signals)
        result["flags"].append(f"opening_mode:{mode}")
        _open_directive = {
            "clue": (
                "开篇硬约束（悬疑·钩子开场）：本章开头是景物/天气铺垫，而非一个具体的"
                "反常/线索/现场。请重写开头——第一句就把读者丢进一个不合理的具体细节、"
                "一具尸体、一条诡异规则或一个待解的疑点，章末留未解信息钩。"
            ),
            "relationship": (
                "开篇硬约束（言情·关系开场）：本章开头是景物/铺垫，缺少人物关系张力。"
                "请重写开头——第一句就给出一段关系冲突/情绪对峙/暧昧张力（具体的人在当下"
                "发生关系性的事），章末留情感悬念钩。"
            ),
            "balanced": (
                "开篇问题：本章以大段纯景物/设定开场，读者抓不到本章要发生什么。"
                "请把一个具体的人物动作、冲突或悬念前置到开头，景物服务于事件而非独立成段。"
            ),
        }.get(mode, (
            "开篇硬约束（黄金三句·番茄3秒定生死）：本章开头不是「正在发生的危机」，"
            "而是景物/天气/时段/设定铺垫。请重写开头——"
            "句1=直接抛出正在发生的冲突/动作/对话（具体、有人物在当下做事），禁止天气/景物/时间/世界观铺垫；"
            "句2=主角的核心反差（弱外表强承诺或反常行为）；"
            "句3=可截图金句钩子（情绪爆发/认知颠覆/后果预告，独立成段）。金手指/主角卖点在前 1/4 内亮相。"
        ))
        result["directives"].append(_open_directive)
        result["block"] = bool(cfg.get("opening_golden_gate_block", False))

    # NOTE: "人名≤5" stays as soft guidance in OPENING_RULES_BLOCK (writing.py).
    # A deterministic name count here proved unreliable (common surname chars
    # collide with ordinary words like 顾客/方向/林…), so it's intentionally omitted.
    return result


# ---------------------------------------------------------------------------
# 章节长度带 (chapter-length band): 番茄短章高频钩子 = 2.5-3k 字/章。
# ---------------------------------------------------------------------------


def length_band_check(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic chapter-length band check.

    Always emits a next-chapter directive when out of band (preserves the prior
    advisory behavior). Adds a SCORE PENALTY only when
    `length_band_penalty_enabled` is on (so existing novels with the flag unset
    keep directive-only behavior). Over-length penalty scales with the overshoot;
    `length_band_block` can escalate a gross overshoot to a hard block.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    cmin = int(cfg.get("chapter_min_chars", 2500))
    cmax = int(cfg.get("chapter_max_chars", 7000))
    clen = len((text or "").strip())
    result: dict[str, Any] = {
        "penalty": 0.0, "flags": [], "directives": [], "block": False, "chars": clen,
    }
    if clen == 0:
        return result
    penalty_on = bool(cfg.get("length_band_penalty_enabled", False))
    if clen < cmin:
        result["flags"].append(f"chapter_too_short({clen})")
        result["directives"].append(
            f"上一章仅 {clen} 字，偏短（目标区间 {cmin}-{cmax}）。本章请把关键场景与对白写足、"
            f"补足必要过程，达到目标字数区间，不要草草收尾。"
        )
        if penalty_on and clen < cmin * 0.75:
            result["penalty"] = 0.5
    elif clen > cmax:
        result["flags"].append(f"chapter_too_long({clen})")
        result["directives"].append(
            f"上一章 {clen} 字，超出目标区间（{cmin}-{cmax}，番茄短章高频钩子）。"
            f"本章压缩冗余的技术性/描写性堆砌，聚焦推进剧情与爽点，章长控制在目标区间内。"
        )
        if penalty_on:
            over = clen / max(cmax, 1)
            result["penalty"] = round(min(2.0, (over - 1.0) * 2.0), 2)
            if over >= 1.5 and bool(cfg.get("length_band_block", False)):
                result["block"] = True
    return result


# ---------------------------------------------------------------------------
# 连续平路闸门 (consecutive-flat-chapter gate): 番茄追读率要求情绪高峰间隔 ≤2 章。
# ---------------------------------------------------------------------------
_FLAT_STRONG_PAYOFF_TYPES = {
    "reveal", "reversal", "court_breakthrough", "military_victory",
    "policy_payoff", "personnel_payoff", "institutional_fix", "payoff",
}


def flat_chapter_streak(
    recent_rows: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count consecutive recent 'flat' chapters and penalize a too-long plateau.

    A chapter is 'flat' when it has neither a strong payoff_type NOR a meaningful
    emotional peak (emotional_impact below `flat_impact_floor`). `recent_rows` is
    the newest-first chapter_metrics list. Complements payoff_beat_density by
    counting an unbroken run of low-energy chapters (a chapter can lack a "strong
    payoff type" yet still be a high-emotion peak — that breaks the streak).
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {"streak": 0, "penalty": 0.0, "flags": [], "directives": []}
    if not bool(cfg.get("flat_streak_gate_enabled", True)):
        return result
    impact_floor = float(cfg.get("flat_impact_floor", 5.0))
    streak = 0
    for r in (recent_rows or []):  # newest-first
        ptype = str(r.get("payoff_type", "")).strip()
        try:
            impact = float(r.get("emotional_impact", 0) or 0)
        except (TypeError, ValueError):
            impact = 0.0
        is_flat = (ptype not in _FLAT_STRONG_PAYOFF_TYPES) and impact < impact_floor
        if is_flat:
            streak += 1
        else:
            break
    result["streak"] = streak
    max_flat = int(cfg.get("flat_chapters_max_consecutive", 3))
    if streak >= max_flat:
        result["flags"].append(f"flat_streak({streak})")
        result["penalty"] = round(float(cfg.get("flat_streak_penalty", 1.0)), 2)
        result["directives"].append(
            f"已连续 {streak} 章「平路」（无强爽点且情绪冲击偏低）。番茄追读率要求情绪高峰间隔 ≤2 章——"
            "本章必须给出一个明确的中爽点/情绪高峰（打脸/反转/揭晓/能力兑现/情感爆发），"
            "落到具体可见的当众场面或对手的可见崩溃上，不要再写过渡铺垫。"
        )
    return result


# ---------------------------------------------------------------------------
# Programmatic em-dash density reduction (Layer 3).
# Deterministic, no-LLM.  Replaces excess em-dashes with comma/period by
# pattern confidence.  Dialogue interruptions inside quotes are preserved.
# ---------------------------------------------------------------------------

_QUOTE_CHARS = set(chr(0x201c) + chr(0x201d) + chr(0x300c) + chr(0x300d) + chr(0x300e) + chr(0x300f))


def reduce_em_dash_density(
    text: str,
    config: dict[str, Any] | None = None,
    target_per_kchar: float | None = None,
) -> str:
    """Replace excess ``——`` with punctuation until density <= *target_per_kchar*.

    Replacement order (highest confidence first):
    1. Chained fragments  ``A——B——C``  →  ``A，B，C``
    2. Mid-sentence appositive (no adjacent quotes)  ``A——B``  →  ``A，B``
    Dialogue interruptions (em-dash near quote marks) are never touched.
    """
    cfg = (config or {}).get("novel", config or {})
    target = target_per_kchar or float(cfg.get("em_dash_reduce_target_per_kchar", 3.0))
    if not text or _EM_DASH not in text:
        return text

    def _density(t: str) -> float:
        return t.count(_EM_DASH) / (len(t) / 1000) if len(t) > 0 else 0.0

    if _density(text) <= target:
        return text

    lines = text.split("\n")
    # Build a list of (line_idx, col, confidence) for every em-dash occurrence.
    # confidence: 2 = chained fragment, 1 = mid-sentence appositive
    sites: list[tuple[int, int, int]] = []
    for li, line in enumerate(lines):
        col = 0
        while True:
            pos = line.find(_EM_DASH, col)
            if pos < 0:
                break
            # Skip if near quote marks (dialogue interruption).
            window = line[max(0, pos - 2) : pos + 4]
            if any(q in window for q in _QUOTE_CHARS):
                col = pos + 2
                continue
            # Check for chained pattern: another em-dash within 30 chars.
            next_em = line.find(_EM_DASH, pos + 2)
            if 0 < next_em - pos <= 30:
                sites.append((li, pos, 2))
            else:
                sites.append((li, pos, 1))
            col = pos + 2

    # Sort by confidence desc, then line order — replace highest-confidence first.
    sites.sort(key=lambda s: (-s[2], s[0], s[1]))

    result_lines = list(lines)
    for li, col, _conf in sites:
        line = result_lines[li]
        # Re-locate the em-dash (positions may shift after earlier replacements).
        pos = line.find(_EM_DASH, max(0, col - 10))
        if pos < 0:
            pos = line.find(_EM_DASH)
        if pos < 0:
            continue
        # Skip if it's now near quotes (could happen after prior replacements
        # exposed a quote boundary).
        window = line[max(0, pos - 2) : pos + 4]
        if any(q in window for q in _QUOTE_CHARS):
            continue
        # Replace with comma.
        result_lines[li] = line[:pos] + "，" + line[pos + 2:]
        # Re-check density after each replacement.
        candidate = "\n".join(result_lines)
        if _density(candidate) <= target:
            break

    return "\n".join(result_lines)


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
    prior_texts_long: list[str] | None = None,
) -> dict[str, Any]:
    """Detect signature clauses in `text` that recur in earlier chapters.

    `prior_texts_long`: extended lookback (default 20ch) for template-prefix
    matching only. Exact clause matching still uses the short `prior_texts`.
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

    # --- Template-prefix fossil detection ---
    prefix_len = int(cfg.get("template_fossil_prefix_len", 8))
    prefix_threshold = int(cfg.get("template_fossil_prefix_chapters", 3))
    long_texts = prior_texts_long if prior_texts_long else prior_texts
    template_fossils: list[tuple[str, int]] = []
    if prefix_len > 0 and prefix_threshold > 0 and long_texts:
        prior_prefix_counts: dict[str, int] = {}
        for pt in long_texts:
            for pfx in _get_cached_prefix_set(pt, prefix_len):
                prior_prefix_counts[pfx] = prior_prefix_counts.get(pfx, 0) + 1

        cur_prefix_seen: set[str] = set()
        for c in cur_clauses:
            nc = _normalize_clause(c)
            if len(nc) < prefix_len + 2:
                continue
            pfx = nc[:prefix_len]
            if pfx in cur_prefix_seen:
                continue
            cur_prefix_seen.add(pfx)
            prior_pfx = prior_prefix_counts.get(pfx, 0)
            if prior_pfx >= prefix_threshold:
                already_exact = any(
                    _normalize_clause(r[0])[:prefix_len] == pfx for r in repeats
                )
                if not already_exact:
                    template_fossils.append((c, prior_pfx))

    result["metrics"]["template_fossils"] = len(template_fossils)

    # Penalize by how many earlier chapters already used the clause.
    fossil_threshold = int(cfg.get("style_cross_repeat_chapters", 2))
    fossils = [(c, p) for c, p in repeats if p >= fossil_threshold]
    all_fossils = fossils + template_fossils
    repeats.sort(key=lambda x: -x[1])
    result["repeats"] = [{"clause": c, "prior_chapters": p} for c, p in repeats[:12]]
    if template_fossils:
        result["template_fossils"] = [
            {"clause": c, "prior_chapters": p} for c, p in template_fossils[:6]
        ]
    result["metrics"]["cross_repeat_count"] = len(repeats)
    result["metrics"]["cross_repeat_fossils"] = len(fossils)

    if all_fossils:
        pen = min(2.0, 0.5 * len(all_fossils))
        result["penalty"] = round(pen, 2)
        result["flags"].append(f"cross_chapter_fossils({len(fossils)})")
        if template_fossils:
            result["flags"].append(f"template_fossils({len(template_fossils)})")
        examples = "、".join(
            f"“{c}”(已出现{p}章)" for c, p in all_fossils[:4]
        )
        result["directives"].append(
            "文体复读预警：以下标志性句子/比喻在前面多章反复出现，已成为口癖，"
            f"本章必须改写或避免：{examples}。同一意象请换新的具体写法。"
        )
        result["level"] = "advise"
        reject_count = int(cfg.get("style_cross_repeat_reject_count", 8))
        if len(all_fossils) >= reject_count:
            result["level"] = "reject"
            result["flags"].append(f"cross_chapter_fossil_collapse({len(all_fossils)})")
    elif len(repeats) >= int(cfg.get("style_cross_repeat_warn_count", 4)):
        result["penalty"] = 0.5
        result["flags"].append(f"cross_chapter_repeats({len(repeats)})")
        result["directives"].append(
            "本章有多处句子与前文几乎雷同，存在复读倾向，请用不同措辞重写这些重复表达。"
        )
        result["level"] = "advise"
    return result


def _overlaps_kept(phrase: str, kept: list[str], min_shared: int = 4) -> bool:
    """True if `phrase` shares a contiguous run of >= min_shared chars with any
    already-kept phrase. Used to collapse shifted n-gram windows
    ('陆知白用左手' / '知白用左手从') into a single representative fossil."""
    subs = {phrase[i:i + min_shared] for i in range(len(phrase) - min_shared + 1)}
    for k in kept:
        for s in subs:
            if s in k:
                return True
    return False


def book_wide_fossils(
    texts_by_chapter: dict[int, str],
    config: dict[str, Any] | None = None,
    whitelist: set[str] | None = None,
) -> dict[str, Any]:
    """Detect micro-phrase tics recurring across a large fraction of the WHOLE
    book — the slow habit-stiffening that `cross_chapter_repetition` (6-chapter
    sliding window, min_len 7) structurally misses.

    A 6-char action stub like '陆知白用左手' reused in 42/50 chapters never trips
    the sliding-window fossil gate (any 6-chapter window sees it only once or
    twice), yet it is exactly the monotony a reader feels. This scans every
    completed chapter, counts the DISTINCT chapters each fixed-length CJK n-gram
    appears in, and flags those crossing a book-fraction / absolute-chapter
    threshold. Overlapping windows are collapsed to one representative phrase.

    Returns {"fossils": [{"phrase","chapter_count","frac"}], "phrases": [str],
    "directives": [str], "metrics": {...}}. Safe no-op on empty input.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "fossils": [], "phrases": [], "directives": [], "metrics": {},
    }
    if not texts_by_chapter or not bool(cfg.get("book_fossil_enabled", True)):
        return result

    n = int(cfg.get("book_fossil_ngram", 6))
    total = len(texts_by_chapter)
    gram_chapters: dict[str, set[int]] = {}
    for ch, text in texts_by_chapter.items():
        body = _strip_title_line(text or "")
        ct = "".join(c for c in body if "一" <= c <= "鿿")
        seen: set[str] = set()
        for i in range(len(ct) - n + 1):
            g = ct[i:i + n]
            if g in seen:
                continue
            seen.add(g)
            gram_chapters.setdefault(g, set()).add(ch)

    frac_thr = float(cfg.get("book_fossil_chapter_frac", 0.30))
    min_ch = int(cfg.get("book_fossil_min_chapters", 6))
    # Threshold: at least min_ch chapters AND at least frac of the book. The
    # absolute floor keeps short/early books from flagging on tiny counts.
    threshold = max(min_ch, int(frac_thr * total + 0.999))

    candidates = [
        (g, len(chs)) for g, chs in gram_chapters.items() if len(chs) >= threshold
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))

    kept_phrases: list[str] = []
    fossils: list[dict[str, Any]] = []
    cap = int(cfg.get("book_fossil_report_cap", 12))
    _wl = whitelist or set()
    for g, count in candidates:
        if _wl and any(w in g or g in w for w in _wl):
            continue
        if _overlaps_kept(g, kept_phrases):
            continue
        kept_phrases.append(g)
        fossils.append({
            "phrase": g,
            "chapter_count": count,
            "frac": round(count / max(total, 1), 2),
        })
        if len(fossils) >= cap:
            break

    # Hard fossils: a SINGLE phrase saturating a large fraction of the whole book
    # (default >= 20% of chapters, e.g. tangshuting「老市场街七号」65/199≈33%) is a
    # structural fossil on its own, even when the DISTINCT-phrase count stays under
    # the reject threshold. review.py routes any hard fossil to STRUCTURAL replan.
    hard_ratio = float(cfg.get("book_fossil_hard_ratio", 0.20))
    hard_fossils = [
        {**f, "hard": True} for f in fossils if f["frac"] >= hard_ratio
    ]

    result["fossils"] = fossils
    result["hard_fossils"] = hard_fossils
    result["phrases"] = kept_phrases
    result["metrics"] = {
        "book_fossil_count": len(fossils),
        "hard_fossil_count": len(hard_fossils),
        "chapters_scanned": total,
        "threshold_chapters": threshold,
    }
    if fossils:
        examples = "、".join(
            f"“{f['phrase']}”({f['chapter_count']}章)" for f in fossils[:8]
        )
        result["directives"].append(
            "全书高频僵化短语预警：以下微动作/描写片段已在全书大量章节反复出现，"
            f"成为机械口癖，本章起必须主动规避并换用不同的动作落点与句式：{examples}。"
        )
    return result


# ---------------------------------------------------------------------------
# Dialogue health: measure dialogue-to-prose ratio.  Pure-narration chapters
# feel "flat" in the web-novel register; conversely, wall-to-wall dialogue
# starves the reader of interiority.  This check targets the more common
# failure mode — too little dialogue — because the model's default drift is
# toward narration/internal-monologue when unconstrained.
# ---------------------------------------------------------------------------

def dialogue_health(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute dialogue-ratio metrics + a penalty + directives.

    Returns the same shape as ``style_health``::

      {
        "metrics": {"dialogue_char_ratio": float,
                     "dialogue_chars": int,
                     "total_chars": int},
        "penalty": float,        # >=0, to SUBTRACT from the LLM review score
        "flags":  [str],         # human-readable problem tags
        "directives": [str],     # imperative fixes injected into the writer prompt
      }

    Thresholds are configurable under config["novel"] with sane defaults; the
    function is safe to call with config=None.  Pure function — no DB, no I/O.
    """
    cfg = (config or {}).get("novel", {}) if config else {}

    # --- gate ---------------------------------------------------------------
    if not cfg.get("dialogue_health_enabled", True):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    total_chars = len(text)
    if total_chars < 200:
        return {
            "metrics": {"dialogue_char_ratio": 0.0,
                        "dialogue_chars": 0,
                        "total_chars": total_chars},
            "penalty": 0.0,
            "flags": [],
            "directives": [],
        }

    # --- measure dialogue chars inside “…” pairs --------------------------
    dialogue_spans = re.findall(r'“([^”]*?)”', text)
    dialogue_chars = sum(len(s) for s in dialogue_spans)
    ratio = dialogue_chars / total_chars

    # --- config thresholds --------------------------------------------------
    ratio_min = float(cfg.get("dialogue_char_ratio_min", 0.10))
    ratio_target = float(cfg.get("dialogue_char_ratio_target", 0.20))
    cap = float(cfg.get("dialogue_penalty_cap", 1.5))

    # --- penalty ------------------------------------------------------------
    penalty = 0.0
    flags: list[str] = []
    directives: list[str] = []

    if ratio < ratio_min:
        penalty = min((ratio_min - ratio) / 0.05, cap)
        flags.append(f"low_dialogue({ratio:.0%}<{ratio_min:.0%})")
        pct = f"{ratio:.0%}"
        tgt = f"{ratio_target:.0%}"
        directives.append(
            f"本章对话占比仅{pct}，远低于目标{tgt}。"
            "下一章必须增加角色间的对话交锋，将叙述性心理独白转化为对话呈现。"
        )

    return {
        "metrics": {
            "dialogue_char_ratio": round(ratio, 4),
            "dialogue_chars": dialogue_chars,
            "total_chars": total_chars,
        },
        "penalty": round(penalty, 2),
        "flags": flags,
        "directives": directives,
    }


# ---------------------------------------------------------------------------
# Multi-lead character service: measure how often each principal-cast member
# actually appears, so a secondary lead can't silently go missing for dozens of
# chapters while the creative contract's "非官配需完整成长线" rule has no metric.
# ---------------------------------------------------------------------------

def character_names_from_md(md: str) -> list[str]:
    """Extract principal-cast names from a `characters.md` state-machine file.

    Parses `## …：名字（备注）` section headers: takes the text after the last
    fullwidth/half-width colon and strips any trailing （…）/(…) parenthetical.
    Headers without a colon (e.g. `## Consolidated`, `## Ch5`) are skipped, so
    only real character sections are returned. Order-preserving, de-duplicated.
    """
    names: list[str] = []
    seen: set[str] = set()
    for line in (md or "").splitlines():
        s = line.strip()
        if not s.startswith("## "):
            continue
        header = s[3:].strip()
        # Only headers of the form "role：name" name a character.
        idx = max(header.rfind("："), header.rfind(":"))
        if idx < 0:
            continue
        name = header[idx + 1:].strip()
        # Drop a trailing parenthetical annotation.
        for lp, rp in (("（", "）"), ("(", ")")):
            p = name.find(lp)
            if p >= 0:
                name = name[:p].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def character_appearance_rate(
    names: list[str],
    texts_by_chapter: dict[int, str],
    window: int = 15,
    floor: float = 0.15,
) -> dict[str, Any]:
    """Fraction of the recent `window` chapters in which each name appears.

    Returns {"rates": {name: frac}, "under_served": [{"name","rate"}], "window"}.
    The first name (protagonist) is measured but never reported as under-served —
    only secondary leads starving out is the failure mode this guards. Safe no-op
    on empty inputs.
    """
    result: dict[str, Any] = {"rates": {}, "under_served": [], "window": 0}
    if not names or not texts_by_chapter:
        return result

    ordered = sorted(texts_by_chapter.keys())
    windowed = ordered[-window:] if window > 0 else ordered
    denom = len(windowed)
    result["window"] = denom
    if denom <= 0:
        return result

    rates: dict[str, float] = {}
    for name in names:
        hits = sum(1 for ch in windowed if name and name in (texts_by_chapter.get(ch) or ""))
        rates[name] = round(hits / denom, 2)
    result["rates"] = rates

    under: list[dict[str, Any]] = []
    for name in names[1:]:  # skip protagonist
        if rates.get(name, 0.0) < floor:
            under.append({"name": name, "rate": rates.get(name, 0.0)})
    result["under_served"] = under
    return result


# ---------------------------------------------------------------------------
# Descriptor-frequency gate: catch short (3-6 char) phrases that evade both
# the clause min_len (7) and the ngram window (6).
# ---------------------------------------------------------------------------

_STOPWORD_BIGRAMS = frozenset({
    "的时", "时候", "的人", "一个", "他的", "她的", "自己", "已经", "没有",
    "不是", "可以", "因为", "但是", "所以", "如果", "就是", "这个", "那个",
    "什么", "怎么", "一下", "出来", "起来", "进去", "过来", "回来", "上去",
    "下来", "下去", "不了", "不到", "得到", "之后", "之前", "的话", "一样",
    "还是", "虽然", "然后", "或者",
})

_STOPWORD_TRIGRAMS = frozenset({
    "了一下", "的声音", "最后一", "屏幕上", "把手机", "的时候", "看了一",
    "说了一", "了一声", "了一口", "了一眼", "的眼睛", "的手指", "在桌上",
    "的肩膀", "了过来", "了出来", "了起来", "了过去", "在地上", "在手里",
    "一句话", "在嘴里", "了出去", "一个人", "的头发", "了下来", "了进去",
    "在身边", "在身后", "在手上", "在脸上", "在门口", "在旁边",
    "手机屏", "机屏幕", "个字都", "老市场", "市场街",
})


def descriptor_frequency(
    texts_by_chapter: dict[int, str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect short descriptive phrases (3-6 CJK chars) overused across the book."""
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "flagged": [], "directives": [], "metrics": {}, "level": "pass",
    }
    if not texts_by_chapter or not bool(cfg.get("descriptor_freq_enabled", True)):
        return result

    min_spread = int(cfg.get("descriptor_freq_min_spread", 15))
    max_density = float(cfg.get("descriptor_freq_max_density", 0.5))
    reject_density = float(cfg.get("descriptor_freq_reject_density", 2.0))
    total = len(texts_by_chapter)
    if total < min_spread:
        return result

    gram_info: dict[str, dict] = {}
    for ch, text in texts_by_chapter.items():
        body = _strip_title_line(text or "")
        cjk = "".join(c for c in body if "一" <= c <= "鿿")
        seen_in_chapter: set[str] = set()
        for n in (3, 4, 5, 6):
            for i in range(len(cjk) - n + 1):
                g = cjk[i:i + n]
                if n == 3 and (g[:2] in _STOPWORD_BIGRAMS or g in _STOPWORD_TRIGRAMS):
                    continue
                if g[-1] in "把在的了着过给让被从向往对跟比":
                    continue
                if n <= 4 and g[0] in "了一每三把出":
                    continue
                if g not in gram_info:
                    gram_info[g] = {"chapters": set(), "count": 0}
                gram_info[g]["count"] += 1
                if g not in seen_in_chapter:
                    gram_info[g]["chapters"].add(ch)
                    seen_in_chapter.add(g)

    flagged: list[dict[str, Any]] = []
    has_reject = False
    name_density_ceiling = float(cfg.get("descriptor_freq_name_ceiling", 1.0))
    for phrase, info in gram_info.items():
        spread = len(info["chapters"])
        density = info["count"] / max(total, 1)
        if density > name_density_ceiling:
            continue
        if spread >= min_spread and density >= max_density:
            entry = {
                "phrase": phrase,
                "chapter_spread": spread,
                "total_count": info["count"],
                "density": round(density, 2),
            }
            flagged.append(entry)
            if density >= reject_density:
                has_reject = True

    flagged.sort(key=lambda x: (-x["density"], -x["chapter_spread"]))
    kept: list[dict[str, Any]] = []
    for f in flagged:
        if any(f["phrase"] in k["phrase"] or k["phrase"] in f["phrase"] for k in kept):
            continue
        kept.append(f)
        if len(kept) >= 12:
            break
    flagged = kept

    result["flagged"] = flagged
    result["metrics"] = {
        "descriptor_flagged_count": len(flagged),
        "chapters_scanned": total,
    }

    if flagged:
        penalty = min(1.5, 0.3 * len(flagged))
        result["penalty"] = round(penalty, 2)
        examples = "、".join(
            f"“{f['phrase']}”({f['total_count']}次/{f['chapter_spread']}章)"
            for f in flagged[:6]
        )
        result["directives"].append(
            "描写标签过度使用预警："
            "以下短语在全书中反复"
            "出现频率过高，"
            "已退化为机械标签，"
            "本章起必须控制使用或"
            "替换为其他描写："
            + examples + "。"
        )
        result["level"] = "reject" if has_reject else "advise"

    return result


# ---------------------------------------------------------------------------
# Genre-adherence gate: deterministic keyword check that chapter content
# matches the declared style_preset.  Zero LLM cost.
# ---------------------------------------------------------------------------

GENRE_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "romance_female": {
        "positive": [
            "心跳", "脸红", "甜", "吻",
            "暧昧", "告白", "约会", "牵手",
            "做饭", "探店", "试吃", "菜谱",
            "食材", "香味", "厨房", "味道",
            "食欲", "小吃", "夜市", "烘焙",
            "餐厅", "饭菜", "炒菜", "煮",
            "撒娇", "心动", "喜欢", "恋",
            "甜蜜", "温柔", "宠",
            "拥抱", "耳朵红", "小鹿乱撞",
            "笑容", "陪伴", "关心", "照顾",
            "早餐", "晚餐", "火锅", "奶茶",
            "逛街", "散步", "日常", "温馨",
        ],
        "negative": [
            "尸体", "枪", "排爆", "液氮",
            "冷库", "绑架", "劫持", "失明",
            "截肢", "瘫痪", "盲杖", "轮椅",
            "械斗", "刺伤", "弹孔", "弹壳",
            "手铐", "枪口", "作战靴",
            "爆炸", "炸弹", "毒气",
            "证物", "血迹", "凶器", "弹道",
            "解剖", "法医", "尸检", "验尸",
            "逮捕", "拘留", "审讯", "口供",
            "监控", "蹲守", "跟踪", "盯梢",
            "对讲机", "警用", "防弹",
            "伤口", "缝合", "手术台", "抢救",
        ],
    },
    "suspense": {
        "positive": [
            "线索", "证据", "嫌疑", "案件",
            "推理", "真相", "密码", "指纹",
            "尸检", "现场", "凶器", "作案",
            "目击", "审讯", "档案",
        ],
        "negative": [
            "修炼", "灵气", "法宝", "妖兽",
            "仙界", "丹药", "飞升",
            "金手指", "系统提示",
            "任务完成",
        ],
    },
    "xuanhuan_shuang": {
        "positive": [
            "修炼", "突破", "灵气", "丹药",
            "法宝", "妖兽", "境界",
            "金手指", "系统", "升级",
            "战力", "秘境",
        ],
        "negative": [
            "办公室", "电话", "汽车",
            "地铁", "公司", "股票",
        ],
    },
    "system_stream": {
        "positive": [
            "系统", "任务", "奖励", "升级",
            "积分", "抽奖", "属性",
            "面板", "技能", "经验值",
        ],
        "negative": [
            "修炼", "飞升", "仙界",
        ],
    },
}


def genre_adherence(
    text: str,
    recent_scores: list[float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether a chapter's content matches its declared genre."""
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "genre_score": 0.0, "penalty": 0.0, "flags": [], "directives": [],
        "level": "pass", "metrics": {},
    }
    if not bool(cfg.get("genre_adherence_enabled", True)):
        return result

    preset = str(cfg.get("style_preset", "")).strip().lower()
    keywords = GENRE_KEYWORDS.get(preset)
    if not keywords:
        return result

    body = _strip_title_line(text)
    kchars = max(len(body) / 1000.0, 0.1)

    pos_count = sum(body.count(kw) for kw in keywords["positive"])
    neg_count = sum(body.count(kw) for kw in keywords["negative"])
    pos_density = pos_count / kchars
    neg_density = neg_count / kchars
    neg_weight = float(cfg.get("genre_negative_weight", 2.0))
    score = pos_density - neg_density * neg_weight

    result["genre_score"] = round(score, 3)
    result["metrics"] = {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_density": round(pos_density, 3),
        "negative_density": round(neg_density, 3),
    }

    threshold = float(cfg.get("genre_drift_threshold", 0.0))
    consec_warn = int(cfg.get("genre_drift_consecutive", 3))
    consec_reject = int(cfg.get("genre_drift_reject_consecutive", 5))

    scores = list(recent_scores or []) + [score]
    low_streak = 0
    for s in reversed(scores):
        if s < threshold:
            low_streak += 1
        else:
            break

    result["metrics"]["low_streak"] = low_streak

    preset_names = {
        "romance_female": "女频甜宠言情",
        "suspense": "悬疑推理",
        "xuanhuan_shuang": "玄幻爽文",
        "system_stream": "系统流",
    }
    genre_name = preset_names.get(preset, preset)

    if low_streak >= consec_reject:
        result["penalty"] = 1.0
        result["flags"].append(f"genre_drift_reject(streak={low_streak})")
        result["directives"].append(
            f"体裁严重偏移："
            f"本书声明体裁为【{genre_name}】，"
            f"但最近{low_streak}章内容"
            "持续偏离该体裁核心场景。"
            "本章必须回归体裁核心。"
        )
        result["level"] = "reject"
    elif low_streak >= consec_warn:
        result["penalty"] = 0.5
        result["flags"].append(f"genre_drift_warn(streak={low_streak})")
        result["directives"].append(
            f"体裁漂移预警："
            f"最近{low_streak}章内容偏离"
            f"声明体裁【{genre_name}】，"
            "请在本章及后续章节中"
            "增加体裁核心场景元素。"
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
# Per-chapter fingerprint library: persistent SQLite store of each chapter's
# structural signature (skeleton bigrams + narrative moves). Queried during
# plan generation to inject avoidance directives BEFORE writing, not after.
# ---------------------------------------------------------------------------

def store_chapter_fingerprint(conn: Any, chapter_num: int, plan: dict[str, Any]) -> None:
    """Persist a chapter's structural fingerprint into chapter_fingerprints."""
    from store import db_lock
    tokens = sorted(_plan_skeleton_tokens(plan))
    moves = _narrative_pattern_sequence(plan)
    try:
        with db_lock():
            conn.execute(
                "INSERT OR REPLACE INTO chapter_fingerprints"
                "(chapter, skeleton_tokens, narrative_moves, payoff_type, conflict_type, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chapter_num,
                    json.dumps(tokens, ensure_ascii=False),
                    json.dumps(moves, ensure_ascii=False),
                    str(plan.get("payoff_type", "")),
                    str(plan.get("conflict_type", "")),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
    except Exception:
        pass


def check_plan_against_fingerprints(
    conn: Any, plan: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Check a candidate plan against ALL stored chapter fingerprints.

    Returns {"max_sim", "most_similar_chapter", "top_similar", "directives"}.
    """
    if conn is None:
        return {"max_sim": 0.0, "most_similar_chapter": None, "top_similar": [], "directives": []}
    threshold = float(config["novel"].get("fingerprint_warn_threshold", 0.65))
    cur_tokens = _plan_skeleton_tokens(plan)
    cur_moves = _narrative_pattern_sequence(plan)
    try:
        rows = conn.execute(
            "SELECT chapter, skeleton_tokens, narrative_moves FROM chapter_fingerprints"
        ).fetchall()
    except Exception:
        return {"max_sim": 0.0, "most_similar_chapter": None, "top_similar": [], "directives": []}
    best_sim = 0.0
    best_ch: int | None = None
    top: list[tuple[int, float]] = []
    for ch, tok_json, mov_json in rows:
        try:
            stored_tokens = set(json.loads(tok_json))
            stored_moves = json.loads(mov_json)
        except Exception:
            continue
        skel_sim = _jaccard(cur_tokens, stored_tokens)
        narr_sim = _sequence_similarity(cur_moves, stored_moves)
        composite = 0.6 * skel_sim + 0.4 * narr_sim
        if composite > 0.3:
            top.append((ch, round(composite, 3)))
        if composite > best_sim:
            best_sim = composite
            best_ch = ch
    top.sort(key=lambda x: x[1], reverse=True)
    top = top[:5]
    directives: list[str] = []
    if best_sim >= threshold and top:
        avoid_chapters = [f"Ch{ch}(sim={s})" for ch, s in top[:3]]
        directives.append(
            f"全书结构指纹检测：当前大纲与 {', '.join(avoid_chapters)} 结构高度相似(max={best_sim:.2f})。"
            "必须改变叙事驱动力和信息揭示顺序，避免重复同样的章节骨架。"
        )
        for ch, s in top[:3]:
            for row in rows:
                if row[0] == ch:
                    try:
                        moves = json.loads(row[2])
                        if moves:
                            directives.append(f"Ch{ch}已用流程: {'→'.join(moves)}")
                    except Exception:
                        pass
                    break
    return {
        "max_sim": round(best_sim, 3),
        "most_similar_chapter": best_ch,
        "top_similar": top,
        "directives": directives,
    }


def fingerprint_avoidance_context(conn: Any, config: dict[str, Any]) -> str:
    """Render the full fingerprint library as avoidance context for plan generation.

    Unlike check_plan_against_fingerprints (which compares a specific plan),
    this returns a summary of ALL stored narrative move sequences so the
    generator can see the full structural history and avoid repeating it.
    """
    if conn is None:
        return "None"
    try:
        rows = conn.execute(
            "SELECT chapter, narrative_moves, payoff_type, conflict_type"
            " FROM chapter_fingerprints ORDER BY chapter"
        ).fetchall()
    except Exception:
        return "None"
    if not rows:
        return "None"
    entries: list[str] = []
    for ch, mov_json, pt, ct in rows:
        try:
            moves = json.loads(mov_json)
        except Exception:
            continue
        if not moves:
            continue
        flow = "→".join(moves)
        meta = ""
        if pt:
            meta += f" payoff={pt}"
        if ct:
            meta += f" conflict={ct}"
        entries.append(f"Ch{ch}: {flow}{meta}")
    return "\n".join(entries) if entries else "None"


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
    # —— 爽文/通用 moves（之前缺失，导致爽文"羞辱→结算→打脸→围观"套路逃过检测）——
    ("humiliation", (
        "羞辱", "嘲讽", "嘲笑", "当众", "刁难", "诬陷", "诬蔑", "示众", "挑衅",
        "打压", "逼迫", "奚落", "围攻", "退婚", "辱骂", "耳光", "扇", "踩",
        "轻视", "看不起", "哄笑", "起哄", "下马威", "找茬", "针对", "压价", "克扣",
    )),
    ("system_payoff", (
        "系统", "面板", "气运", "结算", "弹窗", "技能", "兑换", "到账", "数值",
        "属性", "奖励", "签到", "解锁", "升级", "经验值", "积分", "宿主", "提示音",
    )),
    ("faceslap", (
        "打脸", "反杀", "反将", "反咬", "拆穿", "揭穿", "当场", "碾压", "反击",
        "哑口", "无言", "脸色骤变", "脸色大变", "甩在", "拍在", "装逼", "扮猪吃虎",
        "真相大白", "下不来台", "措手不及", "完胜", "镇住", "震慑",
    )),
    ("crowd_react", (
        "围观", "哗然", "震惊", "目瞪口呆", "死寂", "鸦雀", "众人", "骑手们",
        "弹幕", "直播间", "看戏", "倒吸", "惊呼", "窃窃私语", "沸腾", "炸开",
        "傻眼", "鸦雀无声", "面面相觑",
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
    # Payoff-type monotony: an orthogonal formula axis, computed regardless of
    # move-seq length (爽文: 每章都"打脸"; 悬疑: 每章都"reveal" is审美疲劳 even when the
    # abstract flow varies). Counts the consecutive newest-first run of recent
    # plans sharing this chapter's payoff_type.
    cur_pt = str(plan.get("payoff_type", "")).strip()
    pt_streak = 0
    if cur_pt:
        for rp in recent_plans:
            if isinstance(rp, dict) and str(rp.get("payoff_type", "")).strip() == cur_pt:
                pt_streak += 1
            else:
                break
    # Run length INCLUDING the current chapter (current + matching recents).
    pt_run = pt_streak + 1 if cur_pt else 0
    pt_max = int(cfg.get("payoff_type_monotony_max", 3))

    warn = float(cfg.get("narrative_pattern_sim_warn", 0.7))
    block_streak = int(cfg.get("narrative_pattern_block_streak", 2))
    block_sim = float(cfg.get("narrative_pattern_sim_block", 0.85))
    # Genre-neutral variation directive: change shape AND/OR payoff AND/OR hook.
    _vary = (
        "必须打破套路：换一种叙事形状（改变推进的驱动力——人物关系/外部威胁/时间压力/"
        "主角主动出击/信息揭示顺序），换一种爽点兑现方式（payoff_type 与近期不同），"
        "并换一种章末钩子类型（悬念/反转/情绪炸弹/信息投放 轮换），不要再走同一套流程。"
    )

    best = 0.0
    best_i: int | None = None
    consecutive = 0
    # Move-seq similarity only when the flow is long enough to be recognisable.
    if len(cur) >= int(cfg.get("narrative_pattern_min_moves", 3)):
        sims: list[float] = []
        for i, rp in enumerate(recent_plans):
            if not isinstance(rp, dict):
                sims.append(0.0)
                continue
            sim = _sequence_similarity(cur, _narrative_pattern_sequence(rp))
            sims.append(sim)
            if sim > best:
                best = sim
                best_i = i
        for s in sims:  # consecutive run ≥ warn (a streak is the fatigue signal)
            if s >= warn:
                consecutive += 1
            else:
                break
        seq_label = "→".join(cur)
        if consecutive >= block_streak or best >= block_sim:
            result["level"] = "block"
            result["penalty"] = float(cfg.get("narrative_pattern_block_penalty", 1.5))
            result["flags"].append(
                f"narrative_pattern_repeat(streak={consecutive},max_sim={best:.2f})")
            result["directives"].append(
                f"本章叙事流程骨架（{seq_label}）与近 {consecutive or 1} 章高度雷同，"
                f"属于'同一套流程换个道具'的审美疲劳模式。{_vary}")
        elif best >= warn:
            result["level"] = "warn"
            result["penalty"] = float(cfg.get("narrative_pattern_warn_penalty", 0.6))
            result["flags"].append(f"narrative_pattern_repeat(max_sim={best:.2f})")
            result["directives"].append(
                f"本章叙事流程（{seq_label}）与近期相似度偏高，有流程化倾向。{_vary}")

    result["metrics"] = {
        "max_sim": round(best, 3),
        "consecutive_similar": consecutive,
        "compared": len(recent_plans),
        "payoff_type_streak": pt_streak,
        "payoff_type_run": pt_run,
    }
    result["max_sim"] = round(best, 3)
    result["most_similar_to"] = best_i
    result["consecutive"] = consecutive

    # Payoff-type monotony escalates an otherwise-OK chapter to at least warn.
    if cur_pt and pt_run >= pt_max:
        result["flags"].append(f"payoff_type_monotony({cur_pt}×{pt_run})")
        if result["level"] == "ok":
            result["level"] = "warn"
            result["penalty"] = max(
                result["penalty"], float(cfg.get("narrative_pattern_warn_penalty", 0.6)))
        result["directives"].append(
            f"已连续 {pt_run} 章 payoff_type 都是「{cur_pt}」——爽点形态单调。"
            "本章必须换一种兑现类型（如打脸/暴富/实力跃升/身份反转/收服强者/金句怼人 之间切换），"
            "避免读者对同一种爽点脱敏。"
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


# ---------------------------------------------------------------------------
# Prose texture analysis: quantitative vs poetic balance
# ---------------------------------------------------------------------------
_METAPHOR_MARKERS = re.compile(r"[像如仿佛似若好似犹如宛如恍若好像一如]")
_SENSORY_WORDS = re.compile(
    r"[温暖冰凉灼热潮湿干燥刺鼻芬芳苦涩甘甜酥麻沉闷轰鸣寂静回荡]|"
    r"光芒|阴影|色泽|声响|气味|触感|余温|寒意|热浪|微风"
)
_NUMBER_PATTERN = re.compile(
    r"(?:百分之[一二三四五六七八九十零〇两\d]+|"
    r"\d+(?:\.\d+)?(?:%|‰|°|℃|赫兹|毫米|厘米|分钟|秒|小时|公斤|千克|米|层|级|阶)?|"
    r"零点[一二三四五六七八九十零〇两\d]+)"
)
# 伪技术腔词表（style_health 检查 6 用）：LLM 过度书写塌缩的黑话指纹。
# v12 huangliang 塌缩章实测 ≥12/k；健康书（gudai/fanqie 系）≤3/k。
# 注意不收 “系统/面板/数据” 等系统流爽文的合法金手指词——只收"仪器报告腔"词。
_PSEUDO_TECH_TERMS = re.compile(
    r"频率|脉冲|共振|振动|振幅|波形|载波|声波|信号|编码|解码|传导|衰减|"
    r"激活|残留物?|辐射|磁场|力场|模块|装置|参数|数值|读数|精确|坐标|直径|半径|"
    r"密度|浓度|阈值|频段|晶格|离子|分子|细胞|神经束|皮层|骨膜|血清|电流|电压|"
    r"回路|接口|协议|算法|数据流|扫描|检测|监测|校准|同步率?|周期|"
    r"孔隙|微粒|粒子|介质|载体|样本|组织液|角质层|肉芽|毛细|凝固|"
    r"接收|发射|反射|折射|绕射|成像|定位"
)


def prose_texture(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the quantitative vs poetic texture of prose.

    Returns metrics + a suggestion directive if the balance is skewed.
    """
    text = _strip_title_line(text)
    chars = max(len(text), 1)
    kchars = chars / 1000.0

    numbers = _NUMBER_PATTERN.findall(text)
    num_per_kchar = round(len(numbers) / max(kchars, 0.1), 2)

    metaphors = _METAPHOR_MARKERS.findall(text)
    metaphor_per_kchar = round(len(metaphors) / max(kchars, 0.1), 2)

    sensory = _SENSORY_WORDS.findall(text)
    sensory_per_kchar = round(len(sensory) / max(kchars, 0.1), 2)

    poetic_density = metaphor_per_kchar + sensory_per_kchar

    cfg = (config or {}).get("novel", {})
    num_high = float(cfg.get("texture_num_high_per_kchar", 8.0))
    poetic_low = float(cfg.get("texture_poetic_low_per_kchar", 1.0))

    flags: list[str] = []
    directives: list[str] = []
    balance = "balanced"

    if num_per_kchar > num_high and poetic_density < poetic_low:
        balance = "over_quantitative"
        flags.append("数据密度过高且缺少诗意变奏")
        directives.append(
            "本章数字/数据密度偏高（{:.1f}/千字）而比喻/感官描写偏少（{:.1f}/千字）。"
            "下一章请交替使用：具体数据锚定 + 比喻/通感/感官意象，"
            "避免连续段落全用精确数值描写。至少 2 处用比喻或感官替代直接数字。".format(
                num_per_kchar, poetic_density
            )
        )
    elif num_per_kchar < 1.0 and poetic_density > 6.0:
        balance = "over_poetic"
        flags.append("诗意过度缺少具体锚定")
        directives.append(
            "本章比喻/感官密度过高（poetic_density={:.1f}/千字），偏向散文诗而非叙事推进。"
            "下一章大幅削减比喻与华丽形容：每段最多保留 1 个比喻，改用白描的具体动作、对话，"
            "并在关键处加 2-3 个数字/量级/时限锚定，优先把情节往前推。".format(poetic_density)
        )

    # Over-poetic（紫色文体）安全网：仅对 EGREGIOUS 离群（poetic_density 远超正常语体）扣分。
    # 注意：这里的 poetic_density 用单字比喻/感官正则（如/似/若…）粗测，中文里这些字常作
    # 非比喻功能词（如果/似乎/一如既往），会系统性高估——健康中文网文正文普遍就跑 ~25-35。
    # 因此阈值必须设在正常语体之上（默认 40），否则会惩罚正常文本（Ch1-11 好章也 ~30）。
    # 真正的"风格飘逸"防线是相对尖峰门：style_health 的 em_dash_trend_rise（vs 近章均值）与
    # 跨章化石检测——它们按"相对基线的突变"判定漂移，比这个绝对阈值可靠。此惩罚只兜底极端塌缩。
    penalty = 0.0
    if balance == "over_poetic":
        pen_thresh = float(cfg.get("texture_poetic_penalty_threshold", 40.0))
        pen_cap = float(cfg.get("texture_poetic_penalty_cap", 1.5))
        if poetic_density > pen_thresh:
            penalty = min(pen_cap, round((poetic_density - pen_thresh) * 0.1, 2))

    return {
        "metrics": {
            "num_per_kchar": num_per_kchar,
            "metaphor_per_kchar": metaphor_per_kchar,
            "sensory_per_kchar": sensory_per_kchar,
            "poetic_density": round(poetic_density, 2),
        },
        "balance": balance,
        "flags": flags,
        "directives": directives,
        "penalty": penalty,
    }


def emotional_cadence(
    recent_tones: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect emotional monotony from recent chapters' emotional_tone values.

    Returns warnings + a target mood suggestion when consecutive chapters
    share the same emotional tone.
    """
    cfg = (config or {}).get("novel", {})
    max_same = int(cfg.get("emotional_cadence_max_same", 3))

    if not recent_tones or len(recent_tones) < 2:
        return {"monotony": False, "streak": 0, "directives": []}

    streak = 1
    current = recent_tones[-1]
    for tone in reversed(recent_tones[:-1]):
        if tone and tone == current:
            streak += 1
        else:
            break

    _TONE_ALTERNATIVES = {
        "紧张": ["舒缓", "温情", "反思"],
        "压抑": ["释然", "温暖", "决绝"],
        "悲伤": ["希望", "温情", "坚定"],
        "愤怒": ["冷静", "释然", "温柔"],
        "兴奋": ["沉思", "危机", "温情"],
        "恐惧": ["坚定", "温暖", "释然"],
    }

    directives: list[str] = []
    monotony = streak >= max_same
    if monotony and current:
        alts = _TONE_ALTERNATIVES.get(current, ["与前章不同的情感基调"])
        directives.append(
            f"近{streak}章连续「{current}」基调，情感疲劳风险。"
            f"本章建议切换到：{'/'.join(alts[:2])}，打破单调。"
        )

    return {
        "monotony": monotony,
        "streak": streak,
        "current_tone": current if recent_tones else "",
        "directives": directives,
    }


# ---------------------------------------------------------------------------
# 长跨度疲劳检测 (long-span fatigue: type/mood/tension monotony)
# ---------------------------------------------------------------------------


def long_span_fatigue(
    conn: Any,
    chapter_num: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cross-chapter monotony detection over longer spans than emotional_cadence.

    Checks payoff_type repetition, emotional diversity deficit, and tension
    flatness using chapter_metrics DB data. Returns {metrics, penalty, flags,
    directives}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("long_span_fatigue_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}
    if chapter_num < 5:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    try:
        from store import recent_metrics
        rows = recent_metrics(conn, limit=12)
    except Exception:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}
    if len(rows) < 4:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    # --- 1. Payoff type monotony ---
    type_max = int(cfg.get("chapter_type_monotony_max", 4))
    payoff_types = [str(r.get("payoff_type", "")).strip() for r in rows if r.get("payoff_type")]
    if payoff_types:
        streak = 1
        for i in range(len(payoff_types) - 2, -1, -1):
            if payoff_types[i] == payoff_types[-1]:
                streak += 1
            else:
                break
        metrics["payoff_type_streak"] = streak
        if streak >= type_max and payoff_types[-1]:
            penalty += 0.5
            flags.append(f"payoff_type_monotony({streak}>={type_max})")
            directives.append(
                f"近 {streak} 章都是「{payoff_types[-1]}」爽点类型，读者审美疲劳。"
                f"本章切换到不同的 payoff_type（如 reveal/reversal/emotional）。"
            )

    # --- 2. Emotional diversity deficit ---
    tones = [str(r.get("emotional_tone", "")).strip() for r in rows[:8] if r.get("emotional_tone")]
    if len(tones) >= 4:
        distinct = len(set(tones))
        metrics["emotional_diversity"] = distinct
        if distinct < 3:
            penalty += 0.5
            flags.append(f"emotional_monotony(distinct={distinct}<3)")
            directives.append(
                f"近 {len(tones)} 章仅有 {distinct} 种情绪基调，变化不足。"
                f"本章需要引入截然不同的情感色彩。"
            )

    # --- 3. Tension flatness ---
    tensions = []
    for r in rows[:6]:
        t = r.get("tension")
        if t is not None:
            try:
                tensions.append(float(t))
            except (ValueError, TypeError):
                pass
    if len(tensions) >= 4:
        mean_t = sum(tensions) / len(tensions)
        variance_t = sum((x - mean_t) ** 2 for x in tensions) / len(tensions)
        std_t = variance_t ** 0.5
        metrics["tension_std"] = round(std_t, 2)
        if std_t < 1.0:
            penalty += 0.5
            flags.append(f"tension_flat(std={std_t:.2f}<1.0)")
            directives.append(
                f"近 {len(tensions)} 章紧张度几乎不变（std={std_t:.1f}），"
                f"需要制造明显的张弛起伏——高压场景后给一段喘息，或在平静中突然加压。"
            )

    cap = float(cfg.get("long_span_fatigue_penalty_cap", 1.5))
    penalty = round(min(penalty, cap), 2)
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:3],
    }


# ---------------------------------------------------------------------------
# Payoff-beat density: 爽点 (face-slap / reversal / reveal / power-flex) cadence
# ---------------------------------------------------------------------------
# Web-novel retention lives on a steady drip of "爽" moments. This is a coarse,
# deterministic proxy: it can't judge whether a payoff is *earned*, only whether
# the prose contains payoff-shaped events at a healthy cadence.
_PAYOFF_MARKERS = re.compile(
    r"识破|拆穿|揭穿|当众|反转|逆转|碾压|打脸|一锤定音|真相大白|当场|反将|反咬|"
    r"哑口无言|无言以对|目瞪口呆|脸色骤变|脸色大变|败下阵|认输|低头|跪|"
    r"揭晓|水落石出|原形毕露|扳回|翻盘|破局|绝杀|完胜|压制|镇住|震慑"
)


def payoff_beat_density(
    text: str,
    recent_payoff_types: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure 爽点 density: payoff-shaped events in this chapter plus the recent
    payoff_type cadence. Returns a directive when the recent window has gone too
    long without a strong reader payoff.

    `recent_payoff_types` is the newest-first list of recent chapters'
    payoff_type (from chapter_metrics); a 'setup'/'strategic_setup'/'emotional'
    type does not count as a strong payoff.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    body = _strip_title_line(text or "")
    kchars = max(len(body) / 1000.0, 0.1)
    hits = _PAYOFF_MARKERS.findall(body)
    hits_per_kchar = round(len(hits) / kchars, 2)

    strong_types = {
        "reveal", "reversal", "court_breakthrough", "military_victory",
        "policy_payoff", "personnel_payoff", "institutional_fix", "payoff",
    }
    rt = recent_payoff_types or []
    # Chapters since the last STRONG payoff (newest-first list).
    chapters_since_payoff = 0
    for t in rt:
        if str(t).strip() in strong_types:
            break
        chapters_since_payoff += 1

    flags: list[str] = []
    directives: list[str] = []
    # payoff_density_min is a per-chapter rate (≈0.34 ⇒ 1 strong payoff / 3 ch).
    min_rate = float(cfg.get("payoff_density_min", 0.34))
    max_gap = int(round(1.0 / min_rate)) if min_rate > 0 else 3
    if rt and chapters_since_payoff >= max_gap:
        flags.append(f"payoff_drought({chapters_since_payoff})")
        directives.append(
            f"近 {chapters_since_payoff} 章没有强爽点/高潮（揭晓/反转/打脸/能力兑现）。"
            "本章必须安排一次明确的读者爽点：让主角的优势/真相/反击落到具体的当众场面或对手的可见崩溃上。"
        )

    return {
        "metrics": {
            "payoff_markers": len(hits),
            "payoff_per_kchar": hits_per_kchar,
            "chapters_since_payoff": chapters_since_payoff,
        },
        "flags": flags,
        "directives": directives,
    }


# ---------------------------------------------------------------------------
# Shareable golden-line signal (可截图金句 / 传播性): 番茄书荒广场/段评的传播靠
# "能截图发出去"的金句钩子（复仇宣言/逆袭宣言/认知颠覆/后果预告）驱动。这是一个
# 启发式（非 LLM）信号：它无法判断金句"好不好"，只检测本章是否至少有一句够短、够
# punchy、带强情绪态度的可传播句。缺失时给出建议指令（advisory，不扣分）。
# ---------------------------------------------------------------------------

# 强态度/宣言/反转标记——金句的高频骨架。
_SHAREABLE_MARKERS = re.compile(
    r"从今(?:天|往)?(?:起|以后)|从现在起|记住|凭什么|我偏|我就是|也配|不过如此|活该|"
    r"早晚|总有一天|莫欺|三十年河|给我跪|你们这些|我说过|谁规定|凭本事|宁可|绝不|"
    r"不是.{0,12}(?:而是|是)|要么.{0,10}要么|从不|永远记住|欠我的|该还了|轮到"
)
_SHAREABLE_PERSON = re.compile(r"[我你]")


def _quotable_score(line: str) -> float:
    """Heuristic 'how截图-able is this line' score (0+)."""
    s = 0.0
    if _SHAREABLE_MARKERS.search(line):
        s += 2.0
    if _SHAREABLE_PERSON.search(line):
        s += 1.0
    if len(line) <= 18:  # punchy short lines screenshot better
        s += 1.0
    return s


def shareable_line(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect whether the chapter contains a可截图、可传播的金句钩子.

    Scans quoted dialogue across the chapter plus the chapter tail (where宣言式
    金句 most often lands), scores each candidate, and returns the best. When no
    candidate clears the threshold, emits an advisory directive (no penalty) so
    the next chapter plants a传播性金句. Gated by `shareable_line_enabled`.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "has_shareable": False, "best_line": "", "score": 0.0,
        "flags": [], "directives": [],
    }
    if not bool(cfg.get("shareable_line_enabled", True)) or not text:
        return result
    body = _strip_title_line(text)
    if len(body) < 500:
        return result
    candidates: set[str] = set()
    # Punchy lines often live in dialogue — pull quoted segments from the whole chapter.
    for m in re.findall(r"[“「]([^”」\n]{4,40})[”」]", body):
        candidates.add(m.strip())
    # Plus the chapter tail's short narration sentences (章末金句).
    tail = body[-int(cfg.get("shareable_tail_chars", 500)):]
    for seg in re.split(r"[。！？\n]", tail):
        seg = seg.strip()
        if 6 <= len(seg) <= 30:
            candidates.add(seg)
    best = 0.0
    best_line = ""
    for c in candidates:
        sc = _quotable_score(c)
        if sc > best:
            best = sc
            best_line = c
    threshold = float(cfg.get("shareable_min_score", 2.0))
    has = best >= threshold
    result["metrics"] = {"candidates": len(candidates), "best_score": round(best, 1)}
    result["has_shareable"] = has
    result["best_line"] = best_line[:60]
    result["score"] = round(best, 1)
    if not has:
        result["flags"].append("no_shareable_line")
        result["directives"].append(
            "本章缺少可截图、可传播的金句钩子。番茄段评/书荒广场的自然传播靠金句驱动——"
            "本章请在一个高情绪节点（爆发/对峙/逆袭/反转）放一句够短够狠、独立成段的金句"
            "（复仇宣言/逆袭宣言/认知颠覆/后果预告），让读者想截图发出去。"
        )
    return result


def information_density(
    text: str,
    plan: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a 'pure transition chapter' that advances nothing: no payoff, no
    realized beats, no new information. Heuristic and conservative — it only
    flags when MULTIPLE signals agree, to avoid punishing a legitimately quiet
    breather chapter.

    Signals (all derived from already-computed data, no extra LLM call):
      - payoff_type is setup/emotional (not a concrete reader payoff)
      - the chapter's payoff markers are ~zero (no 爽点)
      - the plan opened no new threads / info_reveals
      - the review's beats_audit shows few/zero realized beats
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("info_density_enabled", True)):
        return {"low_information": False, "signals": [], "directives": []}

    plan = plan or {}
    review = review or {}
    signals: list[str] = []

    ptype = str(plan.get("payoff_type", "")).strip().lower()
    if ptype in ("", "setup", "strategic_setup", "emotional"):
        signals.append(f"payoff_type={ptype or 'none'}")

    body = _strip_title_line(text or "")
    if len(_PAYOFF_MARKERS.findall(body)) == 0:
        signals.append("no_payoff_markers")

    reveals = plan.get("info_reveals") or []
    if not reveals:
        signals.append("no_info_reveals")

    # beats_audit: count realized beats if the reviewer provided it.
    audit = review.get("beats_audit") or []
    if isinstance(audit, list) and audit:
        realized = sum(
            1 for b in audit
            if isinstance(b, dict) and str(b.get("status", "")).lower() in ("realized", "present")
        )
        if realized == 0:
            signals.append("no_realized_beats")

    # Require at least 3 agreeing signals before calling it a transition chapter.
    low_info = len(signals) >= int(cfg.get("info_density_min_signals", 3))
    directives: list[str] = []
    if low_info:
        directives.append(
            "上一章信息推进不足（近似过渡章：无爽点、无新信息、无伏线推进）。"
            "本章必须至少做到其一并落到页面上：引入关键新信息、推进/兑现一条伏线、或制造一次冲突升级。"
        )
    return {"low_information": low_info, "signals": signals, "directives": directives}
