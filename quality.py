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

import re
from typing import Any

# Sentence-ending punctuation for Chinese prose.
_SENTENCE_ENDERS = "。！？…"
_EM_DASH = "——"


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
