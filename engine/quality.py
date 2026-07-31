"""Quality gates and repair ladder (v3 merge of quality.py + fix.py).

Non-LLM, rule-based quality signals plus the deterministic/bounded-LLM
repair ladder that fixes fired gates instead of re-rolling the chapter.

The quality gates provide an objective anchor that the review/revise loop
can react to, independent of the model's own self-assessment (which is
prone to inflation). The repair ladder routes each firing to the cheapest
action that can actually fix it:

- **L0** deterministic text transforms, zero LLM calls
- **L1** one bounded LLM call that rewrites a handful of extracted passages

This module was created by merging ``quality.py`` (gate definitions) and
``fix.py`` (repair ladder) so that gates and their fixers live side by side.
Advisory-only gates (ai_flavor_health, paragraph_shape_health, prose_texture,
hook_tail_repetition, intra_chapter_repetition, long_span_fatigue,
payoff_beat_density, shareable_line, information_density) were initially dropped
in the v3 refactoring, then restored at the end of this file so they are
available via the registry for tools, tests, and v2/accept.py advisory wiring.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from typing import Any

from engine.config import safe_score, text_bigrams


# ---------------------------------------------------------------------------
# Gate Registry: centralized metadata for deterministic quality gates.
#
# Each gate function is decorated with @REGISTRY.register(...) which stores
# its config-enable key, tag prefix, and phase without changing the function.
# Consumers (review.py) use REGISTRY.is_enabled / REGISTRY.accumulate to
# eliminate per-gate boilerplate (config check, try/except, penalty/flag/
# directive accumulation, logging).
# ---------------------------------------------------------------------------

# Repair layers a gate may declare (REDESIGN L6). Ordered cheapest-first: this
# tuple is also the execution order of the repair ladder in `fix.py`.
REPAIR_LAYERS = ("L0", "L1", "L2", "advisory")

# What the gate's measured quantity is a property OF (REDESIGN_V2 §3.4 ③).
#
#   chapter — the text/plan of THIS attempt. Only this scope may block.
#   card    — the ChapterCard / plan, fixable before a word is written.
#   book    — a whole-book cumulative quantity. **Never blocks a single chapter.**
#
# `book` is the actionability invariant made structural. A book-cumulative ratio
# has a numerator frozen in already-written chapters, so no rewrite of the
# current chapter can lower it: the block latches on and every forced retry it
# buys is a guaranteed first-pass failure. Three measured instances of exactly
# that cost 4.0pt of library FPY' (LESSONS §13, REDESIGN_V2 §8). A book-scope
# gate may still carry a repair layer and still emit directives — it just may
# not reject.
GATE_SCOPES = ("chapter", "card", "book")


class GateRegistry:
    """Lightweight registry for deterministic quality gates."""

    __slots__ = ("_gates",)

    def __init__(self) -> None:
        self._gates: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        *,
        config_key: str,
        config_default: bool = True,
        tag_prefix: str | None = None,
        phase: str = "review",
        repair: str = "advisory",
        scope: str = "chapter",
        proof: str = "",
    ):
        """Decorator: register gate metadata. Preserves function identity.

        *repair* declares HOW a firing of this gate is meant to be fixed, which
        is what makes `fix.py`'s ladder possible (REDESIGN L6):

        - ``"L0"``  deterministic text transform, zero LLM calls
        - ``"L1"``  bounded single-call patch on part of the chapter
        - ``"L2"``  plan-level: fix the plan / the NEXT card, never this prose
        - ``"advisory"`` directive only; must NOT trigger rework by itself

        *scope* declares whose property the gate measures (see ``GATE_SCOPES``).
        *proof* is a one-line record of where the gate's threshold sits relative
        to the MEASURED distribution, and it is **mandatory** — a threshold that
        was never checked against real data is how this repo shipped two dead
        keys (`fingerprint_warn_threshold` 0.65 against a measured max of 0.448,
        never reachable, deleted; `book_fossil_hard_ratio` 0.20 below its own
        candidacy floor of 0.30, always true). Both took an offline replay to
        discover. Requiring the number at registration turns a post-mortem into
        a compile-time obligation.

        `repair`/`scope`/`proof` are metadata, not behaviour: nothing in the v1
        review path reads them, so annotating a gate cannot change its score.
        `v2/accept.py` is the first consumer that acts on them.
        """
        if repair not in REPAIR_LAYERS:
            raise ValueError(f"unknown repair layer {repair!r} for gate {name}")
        if scope not in GATE_SCOPES:
            raise ValueError(f"unknown scope {scope!r} for gate {name}")
        if not str(proof).strip():
            raise ValueError(
                f"gate {name!r} must declare `proof=` — where its threshold sits "
                f"against the measured distribution (tools/gate_census.py). "
                f"See REDESIGN_V2 §3.4 ③."
            )

        def wrapper(fn):
            self._gates[name] = {
                "fn": fn,
                "config_key": config_key,
                "config_default": config_default,
                "tag_prefix": tag_prefix or name.replace("_health", "").replace("_quality", ""),
                "phase": phase,
                "repair": repair,
                "scope": scope,
                "proof": str(proof).strip(),
            }
            return fn
        return wrapper

    def is_enabled(self, name: str, config: dict[str, Any] | None) -> bool:
        """Check whether gate *name* is enabled in config."""
        spec = self._gates.get(name)
        if spec is None:
            return True
        cfg = (config or {}).get("novel", {})
        return bool(cfg.get(spec["config_key"], spec["config_default"]))

    def tag_prefix(self, name: str) -> str:
        """Return the rhythm_risks tag prefix for gate *name*."""
        spec = self._gates.get(name)
        return spec["tag_prefix"] if spec else name

    def repair(self, name: str) -> str:
        """Return the declared repair layer for gate *name*.

        Unknown gates are ``"advisory"``: an unregistered signal must never be
        able to trigger rework or a repair action on its own.
        """
        spec = self._gates.get(name)
        return spec["repair"] if spec else "advisory"

    def scope(self, name: str) -> str:
        """Return the declared scope for gate *name* (unknown gates: ``"book"``).

        Unknown defaults to the most restrictive scope on purpose — the same
        reasoning as `repair()` defaulting to advisory. An unregistered signal
        must not be able to reject a chapter.
        """
        spec = self._gates.get(name)
        return spec["scope"] if spec else "book"

    def proof(self, name: str) -> str:
        spec = self._gates.get(name)
        return spec["proof"] if spec else ""

    def may_block(self, name: str) -> bool:
        """Whether gate *name* is allowed to reject the chapter under review.

        The actionability invariant: a gate may block only if the quantity it
        measures is one THIS attempt can change. Book-cumulative quantities are
        not, and advisory gates have no repair to offer, so neither may reject.
        """
        spec = self._gates.get(name)
        if spec is None:
            return False
        return spec["scope"] != "book" and spec["repair"] != "advisory"

    def get(self, name: str):
        """Return the registered gate function, or None."""
        spec = self._gates.get(name)
        return spec["fn"] if spec else None

    def list_gates(
        self,
        phase: str | None = None,
        repair: str | None = None,
        scope: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """List registered gates, optionally filtered by phase / repair / scope."""
        items = self._gates.items()
        if phase is not None:
            items = [(k, v) for k, v in items if v["phase"] == phase]
        if repair is not None:
            items = [(k, v) for k, v in items if v["repair"] == repair]
        if scope is not None:
            items = [(k, v) for k, v in items if v["scope"] == scope]
        return dict(items)

    @staticmethod
    def accumulate(
        report: dict[str, Any],
        result: dict[str, Any],
        gate_name: str,
        tag_prefix: str,
    ) -> float:
        """Standard gate-result accumulation into a review report.

        Stores *result* under ``report[gate_name]``, appends flags (prefixed)
        to ``rhythm_risks`` and directives to ``writer_directives_for_next_chapter``.
        Returns the penalty value so the caller can sum it.
        """
        report[gate_name] = result
        penalty = float(result.get("penalty", 0.0))
        wd = report.setdefault("writer_directives_for_next_chapter", [])
        for d in result.get("directives", []):
            if d not in wd:
                wd.append(d)
        if penalty > 0:
            rr = report.setdefault("rhythm_risks", [])
            for f in result.get("flags", []):
                tag = f"{tag_prefix}:{f}"
                if tag not in rr:
                    rr.append(tag)
        return penalty


REGISTRY = GateRegistry()


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



# 伪技术腔词表（style_health 检查 6 用）：LLM 过度书写塌缩的黑话指纹。
# v12 huangliang 塌缩章实测 ≥12/k；健康书（gudai/fanqie 系）≤3/k。
# 注意不收 "系统/面板/数据" 等系统流爽文的合法金手指词——只收"仪器报告腔"词。
_PSEUDO_TECH_TERMS = re.compile(
    r"频率|脉冲|共振|振动|振幅|波形|载波|声波|信号|编码|解码|传导|衰减|"
    r"激活|残留物?|辐射|磁场|力场|模块|装置|参数|数值|读数|精确|坐标|直径|半径|"
    r"密度|浓度|阈值|频段|晶格|离子|分子|细胞|神经束|皮层|骨膜|血清|电流|电压|"
    r"回路|接口|协议|算法|数据流|扫描|检测|监测|校准|同步率?|周期|"
    r"孔隙|微粒|粒子|介质|载体|样本|组织液|角质层|肉芽|毛细|凝固|"
    r"接收|发射|反射|折射|绕射|成像|定位"
)


def em_dash_penalty(
    em_per_kchar: float,
    recent_mean: float | None,
    config: dict[str, Any] | None = None,
) -> tuple[float, list[str], list[str]]:
    """The em-dash half of `style_health`, as a pure function of TWO numbers.

    Extracted from `style_health` (zero behaviour change) so an offline replay can
    recompute this term from archived `metrics` instead of re-deriving the
    arithmetic in the tool. The graduated trend penalty below replaced a flat
    +1.0, and 31 archived chapters were scored under the old flat rule — 5 of them
    blocked at exactly 2.0 that today's engine charges 1.3–1.8. A replay tool that
    re-implemented this would have to be re-tuned in lockstep forever; one that
    calls it cannot silently disagree with the engine (the failure mode
    `tests/test_latching_gates.py:ReplayPlanScoreFidelityTest` records).

    `recent_mean` is `None` when there is no baseline (fewer than 2 prior
    chapters), which suppresses both the trend and the sustained term.
    Returns `(penalty, flags, directives)`.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    penalty = 0.0
    flags: list[str] = []
    directives: list[str] = []
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

    # --- TREND term: rising em-dash density is itself a collapse signal ----
    # Two failure modes this catches:
    #  (a) Slow drift BELOW the absolute warn threshold (em creeps 0.94→4.15
    #      monotonically while always < 6.0) — never trips the static tier.
    #  (b) A sustained climb ABOVE warn (6.6→7.8→8.8) — the static tier flat-lines
    #      at +1.0 and the acceleration is lost exactly when it matters most. This
    #      check runs REGARDLESS of the static tier (it used to be the static
    #      `else`, so it died once em crossed warn), so a rising-while-already-high
    #      chapter compounds static(+1.0) + trend = block. Observed
    #      gudai50_v2 Ch20-24: em 6.6→8.8 stuck at a flat +1.0 for 5 chapters.
    if recent_mean is None:
        return penalty, flags, directives
    base = float(recent_mean)
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
    return penalty, flags, directives



@REGISTRY.register(
    "style_health", config_key="style_health_enabled", tag_prefix="style", repair="L0",
    scope="chapter",
    proof="642-review census: ran 640, penalty>0 on 8.8%, advisory on 85.9%, "
          "avg 0.080. The block line (style_penalty_block) sits far above the "
          "advisory mass, so it selects a tail rather than the median.")
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
    # Static tier + trend + sustained escalation all live in `em_dash_penalty`,
    # a pure function of (density, recent mean) so a replay can recompute it from
    # archived metrics without re-implementing the arithmetic.
    hist = [
        float(h) for h in (em_history or [])
        if isinstance(h, (int, float)) and h >= 0
    ]
    # Need at least 2 prior points for a meaningful baseline.
    em_base = sum(hist) / len(hist) if len(hist) >= 2 else None
    if em_base is not None:
        metrics["em_dash_recent_mean"] = round(em_base, 2)
    _em_pen, _em_flags, _em_dirs = em_dash_penalty(em_per_kchar, em_base, config)
    penalty += _em_pen
    flags.extend(_em_flags)
    directives.extend(_em_dirs)

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
    dlg_chars += sum(len(m) for m in re.findall(r'"[^"]{1,300}"', body))
    dialogue_ratio = dlg_chars / n
    metrics["dialogue_char_ratio"] = round(dialogue_ratio, 3)
    ratio_min = float(cfg.get("style_dialogue_ratio_min", 0.04))
    # Only flag if the chapter is long enough that some dialogue is expected.
    ratio_target = float(cfg.get("dialogue_char_ratio_target", 0.20))
    if n > 2000 and ratio_min > 0 and dialogue_ratio < ratio_min:
        penalty += 1.0
        flags.append(f"dialogue_starved({dialogue_ratio:.1%}<{ratio_min:.0%})")
        directives.append(
            f"上一章对话占比仅 {dialogue_ratio:.0%}，几乎全是叙述。本章至少 {ratio_min:.0%} 的篇幅"
            "用有潜台词的人物对白推进情节，让信息从人物嘴里说出来而不是叙述灌输。"
        )
    elif n > 2000 and ratio_target > 0 and dialogue_ratio < ratio_target:
        # 对话在 min 和 target 之间：不扣罚但发软提醒，避免 L1 repair 白花钱修补。
        flags.append(f"dialogue_low({dialogue_ratio:.1%}<{ratio_target:.0%})")
        directives.append(
            f"上一章对话占比 {dialogue_ratio:.0%}，低于爆款水平（{ratio_target:.0%}）。"
            "本章请多用对话推进——让冲突、信息、态度转变从人物嘴里说出来。"
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
                and em_per_kchar < float(cfg.get("style_em_dash_per_kchar_warn", 6.0))
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

def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity of two prose blocks over their character bigrams.

    Used to catch the "adjacent chapters are near-verbatim duplicates" failure
    mode (observed in refine output: Ch5≈Ch6, Ch7≈Ch8) where the same scene is
    emitted twice. ~0.0 = unrelated, ~1.0 = (near-)identical.
    """
    return _jaccard(text_bigrams(a), text_bigrams(b))


def _location_core(loc: Any) -> str:
    """The core location noun, with parenthetical detail dropped:
    '通宵自习室（教学楼B座203室）' -> '通宵自习室'."""
    s = str(loc or "").strip()
    s = re.split(r"[（(]", s, 1)[0].strip()
    return s


def _longest_common_substr_len(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by a and b.

    Catches a shared place-noun embedded in longer strings — '顺安便利店夜班' and
    '便利店收银台' both contain '便利店' though neither contains the other, so
    bigram Jaccard alone would mis-flag them as different locations."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def location_transition(
    plan: dict[str, Any],
    recent_plans: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect whether this chapter ENTERS a new location/副本 vs continues in an
    established one, by comparing the plan's ``location`` against recent plans'.

    The 副本-entry chapter is the systematic collapse point in 无限流/规则怪谈 (and
    any location-episodic structure): overloaded with new-setting setup, the writer
    drops the numbered-rules discipline (iron rule #1) and degrades into telegraphic
    summary — yeban_guize Ch8 (进自习室) crashed to 2.7 on exactly this, Ch9 to 4.0.
    Flagging the transition lets write_chapter inject an establishment / rule-listing
    salience block. Genre-neutral: it fires only on a genuine location change, which
    benefits every genre's scene-setting, not just rule-horror.

    Returns {"is_new", "location", "max_sim", "prev"}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    cur = _location_core(plan.get("location") if isinstance(plan, dict) else "")
    res = {"is_new": False, "location": cur, "max_sim": 0.0, "prev": None}
    if not cur or len(cur) < 2:
        return res
    prevs = [
        _location_core(rp.get("location"))
        for rp in recent_plans
        if isinstance(rp, dict) and _location_core(rp.get("location"))
    ]
    if not prevs:
        # No history to compare (Ch1 opening is handled by the opening-craft path).
        return res
    min_shared = int(cfg.get("scene_entry_min_shared_chars", 3))
    best = 0.0
    best_loc = None
    for p in prevs:
        s = text_similarity(cur, p)
        # A shared contiguous place-noun (>= min_shared chars) means same venue,
        # even when neither string contains the whole other ('便利店收银台' vs
        # '顺安便利店夜班' share '便利店'). Treat as clearly-same.
        if _longest_common_substr_len(cur, p) >= min_shared:
            s = max(s, 0.9)
        if s > best:
            best, best_loc = s, p
    res["max_sim"] = round(best, 3)
    res["prev"] = best_loc
    res["is_new"] = best < float(cfg.get("scene_entry_sim_threshold", 0.3))
    return res



# ---------------------------------------------------------------------------
# Adjacent-chapter repetition gate (O1): the deadliest observed failure mode is
# a chapter that re-narrates the previous chapter's ending scene near-verbatim
# (suspense_v11 Ch3 clause-overlap 0.73, Ch8 0.33; suspense_v8 Ch6 0.81 — all
# force-accepted at 3.5-5.5 while healthy chapters sit at 0.00-0.07). The LLM
# reviewer scores each chapter in isolation, so it rated an identical hook 9/10.
# This is the deterministic gate: measured against the previous chapter's text,
# fed into both the draft loop (regenerate) and review (cap + reject).
# ---------------------------------------------------------------------------

@REGISTRY.register(
    "adjacent_repetition", config_key="adjacent_repeat_enabled",
    tag_prefix="repeat", repair="L2", scope="chapter",
    proof="642-review census: ran 632, fired 0.0% -- SILENT. Per LESSONS 4 a "
          "silent gate is a bug report, not a deletion candidate: either the "
          "threshold is unreachable (the fingerprint_warn_threshold defect) or "
          "adjacent verbatim reuse really is absent. UNRESOLVED; must not be "
          "given new blocking weight until a census tells them apart.")
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


@REGISTRY.register(
    "cross_chapter_repetition", config_key="style_cross_repeat_enabled",
    tag_prefix="repeat", repair="L0", scope="chapter",
    proof="642-review census: ran 640, fired 72.5%, of which 23.0% advisory and "
          "the rest reject-level; avg pen 0.130 -- the single largest penalty "
          "source in the library. Six-chapter WINDOW, but the current chapter's "
          "own use of the clause is a conjunct, so keep-1 rotation clears it: "
          "chapter scope, not book.")
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


def fossil_whitelist(config: dict[str, Any] | None = None,
                     prompt_text: str = "") -> set[str]:
    """Phrases `book_wide_fossils` must never indict: the configured list plus
    every 《…》/「…」 proper noun in the creative brief.

    Pure so both callers share one definition — `review.py` (the v1 arm) and
    `v2/run.py`. A whitelist that differed between the arms would let one of them
    be charged for a book's own title.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    out: set[str] = set()
    for w in str(cfg.get("book_fossil_whitelist", "")).split(","):
        w = w.strip()
        if len(w) >= 2:
            out.add(w)
    for m in re.findall(r"[《「]([^》」]{2,10})[》」]", prompt_text or ""):
        out.add(m)
    return out



@REGISTRY.register(
    "book_wide_fossils", config_key="book_fossil_enabled", tag_prefix="fossil",
    repair="L0", scope="chapter",
    proof="642-review census: ran 118, fired 88.1%, avg pen 0.000 (advisory "
          "directives, not score). THE gate this invariant was written for. Its "
          "ratio is book-cumulative and frozen, so pre-fix it latched ON and "
          "rejected chapters that did not contain the phrase -- scope would have "
          "read `book` and `may_block` would have been False. The `in_current` "
          "conjunct added in the latching sweep is what earns it `chapter`: "
          "writing without the phrase now clears it. Worth +6.2pt library FPY' "
          "(tools/replay_gates.py --fix A).")
def book_wide_fossils(
    texts_by_chapter: dict[int, str],
    config: dict[str, Any] | None = None,
    whitelist: set[str] | None = None,
    current_chapter: int | None = None,
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

    Returns {"fossils": [{"phrase","chapter_count","frac","in_current"}],
    "phrases": [str], "hard_fossils": [...], "directives": [str], "metrics": {...}}.
    Safe no-op on empty input.

    ``current_chapter`` is the chapter under review. It decides ``in_current`` on
    every fossil and, through it, which fossils are eligible to become
    ``hard_fossils`` — see the hard-fossil block below for why that matters.
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
            # Does the chapter under review actually use this fossil? See below.
            "in_current": (current_chapter is not None
                           and current_chapter in gram_chapters.get(g, ())),
        })
        if len(fossils) >= cap:
            break

    # Hard fossils: a SINGLE phrase saturating a large fraction of the whole book
    # is a structural fossil on its own, even when the DISTINCT-phrase count stays
    # under the reject threshold. review.py routes any hard fossil to STRUCTURAL
    # replan, so this list is a BLOCKING verdict and must only indict the chapter
    # actually in front of it.
    #
    # It previously indicted every chapter, on two independent counts (measured over
    # 34 archived (chapter, phrase) flags in tangshuting / tangshuting_e2e /
    # yeban_guize -- 22 of them false):
    #
    #  1. It is a CUMULATIVE property of the book, attributed to the CURRENT
    #     chapter. Once「声音压得很低」sat in 82/199 tangshuting chapters, the ratio
    #     could not be brought back under the threshold by writing anything at all
    #     (the numerator is frozen, the denominator grows by 1/chapter: 274 more
    #     clean chapters would be needed). So the gate latched ON and rejected
    #     Ch95-Ch120 six consecutive times for a phrase those chapters never
    #     contained -- punishing the writer for complying. `in_current` is the fix:
    #     a chapter that uses none of the entrenched phrases has done the only thing
    #     the gate can ask of it, and passes. The phrases still ship as
    #     `directives` + `phrases` every scan, so avoidance pressure is unchanged.
    #
    #  2. `book_fossil_hard_ratio` (0.20) was unreachable from below and therefore
    #     dead: candidacy already requires `frac >= book_fossil_chapter_frac` (0.30)
    #     -- and >= min_ch/total, which is itself >= 0.30 for every book size -- so
    #     EVERY fossil was automatically "hard" and the two-tier design collapsed
    #     into "reject on any fossil at all". Same defect class as the deleted
    #     `fingerprint_warn_threshold` (LESSONS §8). The threshold is now taken as
    #     `max(hard_ratio, candidacy_frac)` so the config key describes what the
    #     code does instead of silently doing nothing.
    hard_ratio = max(float(cfg.get("book_fossil_hard_ratio", 0.20)), frac_thr)
    hard_fossils = [
        {**f, "hard": True} for f in fossils
        if f["frac"] >= hard_ratio and f["in_current"]
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

@REGISTRY.register(
    "dialogue_health", config_key="dialogue_health_enabled",
    tag_prefix="dialogue", repair="L1", scope="chapter",
    proof="642-review census: ran 83, fired 31.3%, avg pen 0.128. Ran on only "
          "13% of reviews because the key is per-novel; where it ran, the "
          "0.10 floor selects a real minority rather than the median.")
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

    # --- measure dialogue chars inside “…” / 「…」 / "..." pairs -------------
    dialogue_spans = re.findall(r'“([^”]*?)”', text)
    dialogue_spans += re.findall(r'「([^「」]*?)」', text)
    dialogue_spans += re.findall(r'"([^"]*?)"', text)
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


@REGISTRY.register(
    "descriptor_frequency", config_key="descriptor_freq_enabled",
    tag_prefix="descriptor", repair="L0", scope="chapter",
    proof="642-review census: ran 22, fired 90.9%, avg pen 1.364. "
          "Recalibrated 2026-07-31: max_density 0.5->1.0, reject_density "
          "2.0->3.5. Old thresholds sat at or below the typical value, making "
          "the gate fire on 20/22 runs (noise). New thresholds flag only "
          "sustained overuse (>1x/chapter) and block only mechanical repetition "
          "(>3.5x/chapter).")
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
    max_density = float(cfg.get("descriptor_freq_max_density", 1.0))
    reject_density = float(cfg.get("descriptor_freq_reject_density", 3.5))
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


@REGISTRY.register(
    "genre_adherence", config_key="genre_adherence_enabled", tag_prefix="genre",
    repair="L2", scope="chapter",
    proof="642-review census: ran 206, fired 0.0% -- SILENT on a large sample. "
          "Strongest dead-key suspect in the registry (fingerprint_warn_threshold "
          "profile: 0 hits in 206 chances). Deletion still needs a distribution "
          "read, not silence alone -- LESSONS 4.")
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

    # The threshold MUST sit strictly below zero. `genre_score` is a signed
    # keyword-density difference, and its library-wide median is exactly 0.000 --
    # a large mass of chapters where neither the positive nor the negative
    # keyword list matched at all. A threshold of 0.3 (the old template value)
    # therefore scores "no evidence" as "drift": replayed over the library's 357
    # real scores it puts 46.8% of chapters over the reject streak (86% in some
    # novels). At -1.0 the same replay gives warn 4.8% / reject 2.5%.
    threshold = float(cfg.get("genre_drift_threshold", -1.0))
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
        # The reject path forces a STRUCTURAL replan, and this verdict rests on
        # a keyword-density heuristic that has never been validated against a
        # human read. Its streak lookup was broken from the start (see
        # review.py), so it has never fired in 215 archived runs -- meaning the
        # reject branch is untested in production. Replaying it over the
        # library's real genre_score series rejects 9-11% of chapters in the
        # tangshuting family, which would make it a top-3 replan source on
        # heuristic evidence alone. Ship the measurement, gate the rejection.
        result["level"] = "reject" if bool(cfg.get("genre_drift_reject_enabled", False)) else "advise"
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
        if not (2 <= len(frag) <= 16):
            continue
        if not re.search(r"[一-鿿]", frag):
            continue
        if frag in _BEAT_GENERIC_FRAGMENTS or frag in seen:
            continue
        seen.add(frag)
        fragments.append(frag)
    fragments.sort(key=len, reverse=True)
    return fragments[:max_anchors]


_ARAB_TO_CJK = str.maketrans("0123456789", "〇一二三四五六七八九")


def _fragment_hit(fragment: str, chapter_text: str, chapter_bigrams: set[str], min_bigram_cov: float = 0.7) -> bool:
    """True when the chapter plausibly realizes this anchor fragment.

    Exact substring first; for fragments >=3 chars, fall back to bigram
    coverage so loose rewording ("安瓿碎裂方向" vs "安瓿的碎裂方向") still
    counts. A chapter that never mentions the object at all fails both.
    When the fragment contains Arabic digits, a second substring check
    runs with digits normalized to CJK ("1点" → "一点").
    Long fragments (>=8 chars) use a relaxed bigram threshold because
    compound names like "鼎成科技数据中心外围废弃水塔顶" naturally break
    into separate components in prose, creating boundary bigrams that
    never appear.
    """
    if fragment in chapter_text:
        return True
    norm_frag = fragment.translate(_ARAB_TO_CJK)
    if norm_frag != fragment and norm_frag in chapter_text.translate(_ARAB_TO_CJK):
        return True
    if len(fragment) < 3 or not chapter_bigrams:
        return False
    grams = {fragment[i: i + 2] for i in range(len(fragment) - 1)}
    if not grams:
        return False
    threshold = min_bigram_cov if len(fragment) < 8 else min_bigram_cov * 0.7
    return sum(1 for g in grams if g in chapter_bigrams) / len(grams) >= threshold



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


@REGISTRY.register(
    "scene_similarity", config_key="scene_dedupe_enabled", tag_prefix="scene",
    phase="planning", repair="L2", scope="card",
    proof="Planning-phase, so gate_census (review payloads) never sees it. "
          "Replayed on 692 real plans/cards -- 632 archived v1 merged_plans + 60 "
          "v2 cards, six books, both engines (experiments/replay_scene_dedupe.py, "
          "2026-07-28): median 0.06, p90 0.13, library max 0.393. NOTHING has ever "
          "crossed the 0.82 block line, and nothing has reached even half of v1's "
          "old 0.6 WARN line, which is why that tier and the 0.97 ceiling were "
          "deleted rather than re-wired. Do NOT answer this by lowering the line: "
          "of the library's four highest values (tangshuting Ch174 0.393, Ch175 "
          "0.293, Ch176 0.205, Ch129 0.179) three PASSED their first draft and the "
          "one miss is charged to a plan retry plus a book_wide_fossils ratio, so "
          "the top of the observed range carries no repetition failure to "
          "threshold against -- there is no positive set at all. The gate stays "
          "because it is pre-write and free, not because it protects anything "
          "measured; the failure it is blind to (same procedural flow, new "
          "wording) belongs to narrative_pattern_repetition. The "
          "`scene_dedupe_retry` event that looked like its fire is the generic "
          "duplicate_blocked marker shared with two other gates.")
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
    from engine.store import db_lock
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


# How the fingerprint library is summarized for the planner. Constants rather
# than config keys: these are properties of the move vocabulary (11 tokens), not
# something a user tunes per novel (Code over Config, REDESIGN §5).
_FP_MIN_REPEAT = 3     # a pattern is only "高频" once it has happened this often
_FP_TOP_BIGRAMS = 12
_FP_TOP_TRIGRAMS = 6


def fingerprint_avoidance_context(conn: Any, config: dict[str, Any]) -> str:
    """Summarize the fingerprint library as avoidance context for plan generation.

    This used to enumerate one line per completed chapter (`Ch7: enter_space→…`).
    Measured, that was the largest single block in the engine's largest prompt --
    22,813 of 116,592 chars (19.6%) at Ch201, growing linearly with the book --
    and it could not deliver the signal its own header promises ("特别是那些高频
    出现的流程组合"): tangshuting's 200 chapters hold **194 distinct flows**, so
    exact whole-flow repetition is 3 patterns over 8 chapters. Two hundred
    all-but-unique strings is noise by construction.

    The repetition is real one level down, in the 11-token move vocabulary:
    `collect_evidence→deduce_conclusion` ×29, `enter_space→collect_evidence` ×26,
    91 of 106 bigrams recurring ≥3 times. So this now emits the aggregate --
    recurring bigrams/trigrams plus payoff/conflict/move frequencies -- which is
    what the header actually asks for, in ~1k chars that do NOT grow with the
    book.

    What was lost: the planner can no longer read Ch137's flow off this block.
    Nothing consumed it that way. The recent chapters are quoted verbatim by
    planning.py's `narrative_pattern_block`, and plan-skeleton duplication is
    judged by `scene_similarity`, which fires. The one function that did compare a
    candidate against the whole library, `check_plan_against_fingerprints`, was
    deleted alongside this rewrite: it was never called outside tests, and
    replayed over 437 real chapters in 6 novels its composite similarity peaked at
    **0.448** against a `fingerprint_warn_threshold` of **0.65** -- unreachable by
    construction, the same defect as the deleted `dialogue_pingpong` /
    `chapter_ending_quality` gates. `store_chapter_fingerprint` still writes
    `skeleton_tokens`, which is what made that replay possible offline; keep it.
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
    moves_counts: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    payoffs: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    n_chapters = 0
    for ch, mov_json, pt, ct in rows:
        try:
            moves = json.loads(mov_json)
        except Exception:
            continue
        if not moves:
            continue
        n_chapters += 1
        moves_counts.update(moves)
        bigrams.update("→".join(moves[i:i + 2]) for i in range(len(moves) - 1))
        trigrams.update("→".join(moves[i:i + 3]) for i in range(len(moves) - 2))
        if pt:
            payoffs[str(pt)] += 1
        if ct:
            conflicts[str(ct)] += 1
    if not n_chapters:
        return "None"

    def _hot(counter: Counter[str], top: int) -> list[tuple[str, int]]:
        return [(k, v) for k, v in counter.most_common(top) if v >= _FP_MIN_REPEAT]

    lines = [f"（全书 {n_chapters} 章累积统计，非逐章清单；下列组合越靠前越滥用）"]
    hot_bi = _hot(bigrams, _FP_TOP_BIGRAMS)
    if hot_bi:
        lines.append("高频相邻推进对（本章至少避开前 3 条）：")
        lines += [f"- {p} ×{n}" for p, n in hot_bi]
    hot_tri = _hot(trigrams, _FP_TOP_TRIGRAMS)
    if hot_tri:
        lines.append("高频三连流程（整段形状已用滥，禁止再走一遍）：")
        lines += [f"- {p} ×{n}" for p, n in hot_tri]
    if payoffs:
        lines.append("已用兑现类型频次：" + " ".join(
            f"{k}×{v}" for k, v in payoffs.most_common(8)))
    if conflicts:
        lines.append("已用冲突类型频次：" + " ".join(
            f"{k}×{v}" for k, v in conflicts.most_common(8)))
    if moves_counts:
        lines.append("单步使用频次：" + " ".join(
            f"{k}×{v}" for k, v in moves_counts.most_common(10)))
    return "\n".join(lines)



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


@REGISTRY.register(
    "plan_executability_gate", config_key="plan_executability_gate_enabled",
    tag_prefix="plan", phase="planning", repair="L2", scope="card",
    proof="Planning-phase; invisible to gate_census. Sits LAST in "
          "planning.create_plan's sequential gate chain, so every gate above it "
          "steals its chances -- replay_gates had to re-run the whole chain to "
          "give it its first look. Blocks a CARD, which is free to fix before a "
          "word is written; that is why card scope may block.")
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


@REGISTRY.register(
    "narrative_pattern_repetition", config_key="narrative_pattern_enabled",
    tag_prefix="pattern", phase="planning", repair="L2", scope="card",
    proof="Planning-phase; invisible to gate_census. Shares the generic "
          "`duplicate_blocked` retry marker with scene_similarity and "
          "chapter_mode_monotony, so per-gate attribution needs "
          "tools/replay_gates.py, never the event log.")
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

    # Payoff-type monotony escalates an otherwise-OK chapter. A long run escalates
    # all the way to BLOCK (forces a plan retry in create_plan), not just WARN.
    # Rationale: a post-hoc review penalty / soft prompt avoid-list never changed
    # the arbiter's behaviour — yeban_guize rode payoff_type='reveal' for 6 straight
    # chapters while both the review penalty and the WARN directive fired every
    # time. The plan-side hard gate (mirroring scene_dedupe / narrative move-seq
    # block) is the only lever that reliably breaks the run. pt_block is genre-
    # neutral and deliberately lenient (default 5): reveal-heavy genres
    # (suspense/rule_horror) legitimately want reveal-dominant payoff, so we allow
    # a run of pt_block-1 before forcing differentiation.
    if cur_pt and pt_run >= pt_max:
        pt_block = int(cfg.get("payoff_type_monotony_block", 5))
        result["flags"].append(f"payoff_type_monotony({cur_pt}×{pt_run})")
        if pt_run >= pt_block:
            result["level"] = "block"
            result["penalty"] = max(
                result["penalty"], float(cfg.get("narrative_pattern_block_penalty", 1.5)))
            result["directives"].append(
                f"硬性重规划：已连续 {pt_run} 章 payoff_type 都是「{cur_pt}」，爽点形态严重单调、读者已脱敏。"
                f"本章的 payoff_type 必须改成与「{cur_pt}」不同的兑现类型，"
                "并在 beats 里把这种新爽点落成具体可拍的动作/揭示/反转，而不只是改标签。"
            )
        else:
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


@REGISTRY.register(
    "plan_visual_payoff_check", config_key="plan_visual_payoff_enabled",
    config_default=True, tag_prefix="plan", phase="planning", repair="L2",
    scope="card",
    proof="Planning-phase; invisible to gate_census. Its blocking half is "
          "additionally gated by `visual_payoff_blocks_plan`, which defaults OFF "
          "in serial narrative mode -- so 'it never fired' can mean the mode, not "
          "the threshold. Measured only via tools/replay_gates.py.")
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



# Chapter-mode taxonomy (Layer 1+2 治本 for premise/formula exhaustion). This is a
# COARSER axis than payoff_type: it captures the reader-facing FORM of a chapter,
# which is the level at which fatigue is actually felt. yeban_guize collapsed at
# Ch28 because ~9 straight chapters were all "智斗解谜" (a reasoning puzzle) even
# though their payoff_type labels (reversal/reveal/personnel) varied — so the
# fine-grained payoff_type gate saw "variety" while the reviewer saw sameness.
_CHAPTER_MODE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "reasoning": (
        "推理", "解谜", "线索", "识破", "破解", "验证", "推演", "真相", "盘问",
        "证据", "矛盾", "拆解", "识别", "规则解读", "谜题", "机关", "解读",
    ),
    "action": (
        "逃", "追击", "追杀", "搏", "战斗", "袭击", "躲避", "反击", "营救",
        "突围", "厮杀", "追逐", "格斗", "冲杀", "抢夺", "生死一线",
    ),
    "emotional": (
        "牺牲", "死亡", "告别", "诀别", "悲", "痛哭", "愧疚", "绝望", "温情",
        "泪", "崩溃", "创伤", "释怀", "情感爆发", "悼",
    ),
    "relational": (
        "信任", "背叛", "结盟", "联手", "决裂", "坦白", "和解", "反目",
        "示好", "同伴", "羁绊", "对峙表态", "摊牌",
    ),
    "advancement": (
        "幕后", "阵营", "组织", "协会", "身世", "布局", "阴谋", "晋级",
        "新讳地", "抵达", "离开", "元凶", "势力", "格局", "转场", "开新副本",
    ),
    "daily": (
        "休整", "日常", "喘息", "过渡", "采买", "准备", "疗伤", "缓冲",
    ),
}


def _classify_chapter_mode(plan: dict[str, Any], baseline: str = "auto", margin: int = 3) -> str:
    """Classify a plan into a coarse reader-facing chapter mode via keyword hits.

    Deterministic, no LLM. When *baseline* names a real mode (a genre with a single
    core form, e.g. "reasoning" for suspense/rule-horror) the classifier is BIASED
    toward it: a non-baseline mode wins ONLY if its keyword hits exceed the
    baseline's by more than *margin*. Rationale (learned from yeban_guize): in a
    puzzle-genre almost every chapter carries incidental emotional/advancement
    content (牺牲/协会/幕后), so a naive raw-max classifier mislabels
    fundamentally-智斗 chapters as emotional/advancement and misses the "every
    chapter is the same KIND of chapter" fatigue the reviewer actually feels. The
    bias makes a genuine form-break (a chapter that CLEARLY departs from the
    baseline) the only thing that escapes the baseline label.

    ``baseline="auto"`` (any value not in ``_CHAPTER_MODE_KEYWORDS``) disables the
    bias and returns the raw argmax — correct for genres with no single core form.
    Genre defaults come from ``config.genre_detection_profile``; the bias must NOT
    be applied to a genre whose core form isn't in the taxonomy at all (it would
    label every chapter with the baseline and turn the monotony gate into a
    100%-false-positive blocker — see the romance_female profile note).
    """
    biased = baseline in _CHAPTER_MODE_KEYWORDS
    fallback = baseline if biased else "daily"
    if not isinstance(plan, dict):
        return fallback
    parts: list[str] = []
    for k in ("title", "goal", "conflict", "payoff", "pressure", "hook", "risk",
              "conflict_type", "payoff_type"):
        v = plan.get(k)
        if isinstance(v, str):
            parts.append(v)
    beats = plan.get("beats")
    if isinstance(beats, list):
        parts.extend(str(b) for b in beats)
    text = " ".join(parts)
    if not text.strip():
        return fallback
    scores: dict[str, int] = {}
    for mode, kws in _CHAPTER_MODE_KEYWORDS.items():
        scores[mode] = sum(text.count(kw) for kw in kws)
    if not biased:
        # No genre core form → raw argmax (insertion-ordered tie-break, deterministic).
        top = max(scores, key=lambda m: scores[m])
        return top if scores[top] > 0 else fallback
    base_score = scores.get(baseline, 0)
    others = {m: s for m, s in scores.items() if m != baseline and s > 0}
    if others:
        top = max(others, key=lambda m: others[m])
        if scores[top] > base_score + margin:  # a CLEAR form-break beats the baseline
            return top
    if base_score > 0:
        return baseline
    # Baseline entirely absent — fall back to the strongest present mode.
    if others:
        return max(others, key=lambda m: others[m])
    return baseline


def chapter_mode_monotony(
    plan: dict[str, Any],
    recent_plans: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect reader-facing FORM monotony over a window (Layer 1+2 治本).

    Unlike ``narrative_pattern_repetition`` (ordered move-seq) and the payoff_type
    run, this measures the FREQUENCY of the current chapter's coarse mode across a
    window of recent chapters — the reviewer's actual fatigue signal was frequency
    ("近9章 reversal×9/reveal×7 智斗形态疲劳"), not a strict consecutive run.

    Returns {"mode", "mode_frac", "window", "same_count", "level" (ok/warn/block),
    "penalty", "flags", "directives"} — same shape family as the other gates.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "mode": None, "mode_frac": 0.0, "window": 0, "same_count": 0,
        "level": "ok", "penalty": 0.0, "flags": [], "directives": [],
    }
    if not bool(cfg.get("chapter_mode_enabled", True)):
        return result
    _baseline = str(cfg.get("chapter_mode_baseline", "auto"))
    _margin = int(cfg.get("chapter_mode_baseline_margin", 3))
    cur_mode = _classify_chapter_mode(plan, _baseline, _margin)
    result["mode"] = cur_mode
    window = int(cfg.get("chapter_mode_window", 6))
    recent = [rp for rp in (recent_plans or []) if isinstance(rp, dict)][:window]

    # The FRACTION is measured on the UNBIASED classification, even when a genre
    # baseline is configured. The biased classifier (see `_classify_chapter_mode`)
    # deliberately returns the baseline label unless a chapter CLEARLY breaks form,
    # so under a baseline the frac has a floor near 1.0 and cannot be compared to
    # `chapter_mode_block_frac` at all. Measured: with baseline="reasoning"
    # (config.py's default for suspense) 38/41 tangshuting_e2e plans and 13/13
    # guize_guaitan plans classify as "reasoning", so frac >= 0.93 by construction
    # against a 0.80 block line -- the gate was ON permanently. It blocked 18
    # tangshuting_e2e chapters and 16 of them (89%) were STILL blocked after the
    # forced plan retry, because re-rolling a plan cannot change the genre. That
    # made it the single largest source of `replanned:initial` first-pass failures
    # while buying a re-roll of the most expensive call in the engine (~132k prompt).
    # Unbiased, the same books read 30/41 and 11/13 -- monotony that is real, local,
    # and escapable, which is what the gate was built to catch.
    #
    # The BIASED label is still what gets reported and named in the directive: it is
    # the better description of what the chapter is, it just cannot be counted.
    _mode_for_frac = _classify_chapter_mode(plan, "auto", _margin)
    same_count = sum(
        1 for rp in recent
        if _classify_chapter_mode(rp, "auto", _margin) == _mode_for_frac
    )
    total = len(recent) + 1
    frac = (same_count + 1) / total if total > 0 else 0.0
    result["window"] = total
    result["same_count"] = same_count + 1
    result["mode_frac"] = round(frac, 3)

    min_window = int(cfg.get("chapter_mode_min_window", 4))
    warn_frac = float(cfg.get("chapter_mode_warn_frac", 0.6))
    block_frac = float(cfg.get("chapter_mode_block_frac", 0.8))
    # Suggest a concrete alternative form so the directive is actionable.
    _alts = {
        "reasoning": "动作/追逐、情感冲击、或人物关系摊牌",
        "action": "推理解谜、情感沉淀、或关系推进",
        "emotional": "推理解谜、动作对抗、或推进元剧情",
        "relational": "推理解谜、动作对抗、或情感爆发",
        "advancement": "近距离智斗、动作对抗、或情感/关系戏",
        "daily": "推理解谜、动作对抗、或推进主线",
    }
    alt = _alts.get(cur_mode, "与近期不同的章型")
    if total < min_window:
        return result
    if frac >= block_frac:
        result["level"] = "block"
        result["penalty"] = max(result["penalty"], float(cfg.get("chapter_mode_block_penalty", 1.2)))
        result["flags"].append(f"chapter_mode_monotony({cur_mode} {result['same_count']}/{total})")
        result["directives"].append(
            f"硬性重规划：近 {total} 章有 {result['same_count']} 章都是「{cur_mode}」型章节，"
            f"读者会形态疲劳（套路耗尽）。本章必须改成另一种章型——{alt}——"
            f"用不同的推进驱动力和读者体验，不要再写同一形态。"
        )
    elif frac >= warn_frac:
        result["level"] = "warn"
        result["penalty"] = max(result["penalty"], float(cfg.get("chapter_mode_warn_penalty", 0.5)))
        result["flags"].append(f"chapter_mode_monotony({cur_mode} {result['same_count']}/{total})")
        result["directives"].append(
            f"已连续偏重「{cur_mode}」型章节（近 {total} 章占 {result['same_count']}）——"
            f"建议本章换一种章型（{alt}）制造形态变化，避免读者审美疲劳。"
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


@REGISTRY.register(
    "opening_hook_gate", config_key="opening_golden_gate_enabled",
    tag_prefix="opening", repair="L0", scope="chapter",
    proof="642-review census: never ran (gate is Ch1-scoped / config-disabled). "
          "Its block IS chapter-actionable — `fix.promote_action_opening` "
          "rewrites the offending opening deterministically.")
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
# 章节长度带 (chapter-length band): 对齐爆款实测中位 2.2k 字/章。
# ---------------------------------------------------------------------------


@REGISTRY.register(
    "length_band_check", config_key="length_band_penalty_enabled",
    config_default=False, tag_prefix="length", repair="L1", scope="chapter",
    proof="642-review census: ran 640, out-of-band on 35.2%, avg pen 0.080. The "
          "config key gates only the PENALTY -- the gate itself always runs, so "
          "`REGISTRY.is_enabled` is not a 'did it run' test for this one.")
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
        short_block_ratio = float(cfg.get("length_band_short_block_ratio", 0.5))
        if clen < cmin * short_block_ratio:
            result["block"] = True
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
# The release ruler.
#
# Moved here from `pipeline.py` so exactly ONE definition exists. It is the
# acceptance rule that `tools/fpy_prime.py` replays to settle every engine A/B,
# it is what `rework_trigger: deterministic` consults, and v2's `accept.py`
# scores against it too. Two copies of a ruler is two different experiments.
# Living in `quality.py` also means a zero-LLM offline tool can import it
# without pulling in the whole engine.
# ---------------------------------------------------------------------------


def hard_block_reasons(review: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Enumerate the DETERMINISTIC reasons this draft is a write-off.

    These are the checks in `review.py` that set ``accepted = False`` on their own
    evidence rather than by comparing the LLM's self-score against a threshold:
    gate rejects, style collapse, hard factcheck contradictions, gross length,
    hard blocks from the opening / adjacent-repetition gates, and a pile-up of
    unmet arbiter constraints. Every one of them is measured, not judged.
    """
    cfg = config["novel"]
    reasons: list[str] = []

    grs = [g for g in (review.get("gate_rejects") or []) if isinstance(g, dict)]
    if grs:
        reasons.append("gate_rejects=" + ",".join(str(g.get("gate", "?")) for g in grs[:4]))

    sh_pen = float((review.get("style_health") or {}).get("penalty", 0.0) or 0.0)
    if sh_pen >= float(cfg.get("style_penalty_block", 2.0)):
        reasons.append(f"style_collapse(penalty={sh_pen:.1f})")

    af_pen = float((review.get("ai_flavor_health") or {}).get("penalty", 0.0) or 0.0)
    if af_pen >= float(cfg.get("ai_flavor_penalty_block", 2.5)):
        reasons.append(f"ai_flavor_block(penalty={af_pen:.1f})")

    if bool(cfg.get("factcheck_hard_blocks_accept", True)):
        hard = [c for c in (review.get("contradictions") or [])
                if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard"]
        if hard:
            reasons.append(f"hard_contradictions={len(hard)}")

    hard_contract = [c for c in (review.get("contract_violations") or [])
                     if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard"]
    if hard_contract and bool(cfg.get("contract_blocks_accept", True)):
        reasons.append(f"hard_contract={len(hard_contract)}")

    for key, label in (("length_band", "length_band"), ("opening_hook_gate", "opening_gate")):
        if (review.get(key) or {}).get("block"):
            reasons.append(f"{label}_block")
    if str((review.get("adjacent_repetition") or {}).get("level", "")) == "block":
        reasons.append("adjacent_repeat_block")

    failed = review.get("constraint_violations_structured") or []
    if len(failed) >= int(cfg.get("constraint_violation_block_count", 3)):
        reasons.append(f"constraints_unmet={len(failed)}")

    return reasons



# ---------------------------------------------------------------------------
# Plan-arbitration decision readers (pure; moved here from planning.py)
# ---------------------------------------------------------------------------
# The arbitration dict is UNTRUSTED input, keys included: it is LLM-produced and
# `llm.load_json_with_repair` accepts any repair that merely *parses*, so a
# salvage fragment can be laundered into a decision. These readers are the guard,
# and they live in `quality.py` because they are exactly the same kind of thing as
# `hard_block_reasons` — zero-LLM pure functions over an untrusted payload — and
# because both engines need them without either owning the other's planner.
ARBITER_KEYS = ("selected_index", "scores", "merged_plan", "required_constraints")


def _coerce_index(val: Any, default: int = 0) -> int:
    """Robustly parse an arbiter-supplied index that may be malformed.
    Arbitration JSON is LLM-produced and untrusted: selected_index / scores[].index
    have shown up as '^1', ' 1 ', 1.0, None. A bad value must never crash planning
    (that wedges the whole run). Extract the first signed integer run, else default."""
    if isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        m = re.search(r"-?\d+", str(val or ""))
        return int(m.group()) if m else default


def _normalize_decision(decision: Any) -> dict[str, Any]:
    """Repair the arbitration dict's SHAPE before anything reads it.

    Same untrusted-input doctrine as `_coerce_index`, one level up: the *keys* are
    LLM-produced too, and `llm.load_json_with_repair`'s repair call makes it worse
    rather than better -- a repair whose output merely PARSES is accepted, so a
    salvaged fragment like `{"./output.json": "{"}` is laundered into a decision.

    Measured over all 934 archived arbitrations: 16 are unusable, 14 of them in
    guize_guaitan alone (34 arbitrations) and every other book <=1. Observed key
    forms: `.selected_index`, `''`, `./output.json`, `./merged_plan.json`,
    `./response.json`, `./scores`, `/assistant`.

    **Key repair recovers 0 of those 16 on its own** — the `.selected_index` cases
    ship `scores: []`, `merged_plan: {}`, `required_constraints: []` beside the bad
    key, so the content was lost, not just the label. It is kept anyway because it
    is free and because `_decision_usable` must judge content, not spelling; the
    call-site re-ask is what actually recovers these.

    The one shape here that DOES recover is a single score row promoted to the top
    level (tangshuting Ch175: `{index, score, pros, cons}`) — that is the
    measurement itself, merely unwrapped.

    Three repairs, all conservative: unwrap a single-key wrapper, rebuild the
    envelope around a bare score row, and strip leading path/quote junk off a key
    when the result is an expected key that is not already present. Never invents a
    key, never overwrites a well-spelled one.
    """
    if not isinstance(decision, dict):
        return {}
    if len(decision) == 1:
        inner = next(iter(decision.values()))
        if isinstance(inner, dict) and any(k in inner for k in ARBITER_KEYS):
            decision = inner
    if (isinstance(decision.get("score"), (int, float))
            and not isinstance(decision.get("score"), bool)
            and not any(k in decision for k in ARBITER_KEYS)):
        decision = {"selected_index": _coerce_index(decision.get("index", 0)),
                    "scores": [decision]}
    out: dict[str, Any] = {}
    for key, val in decision.items():
        fixed = str(key).strip().strip("./\"' \t")
        out[fixed if (fixed != key and fixed in ARBITER_KEYS
                      and fixed not in decision) else key] = val
    return out


def decision_has_score(decision: Any) -> bool:
    """True when the arbiter actually MEASURED the plan (non-empty `scores`).

    `plan_score` must keep returning 0.0 for an empty list -- `chapter_metrics`
    persists that column and readers expect a float -- so "was it measured at all"
    needs its own predicate. A missing measurement is NOT a low one: 0.0 sits below
    every threshold, so `create_plan` used to buy a full extra plan round on it.
    `arc.py` leaves `scores` deliberately empty for the same reason (a fabricated
    score would poison `chapter_metrics.plan_score`), which is a second reason no
    reader may read 0.0 as a verdict.
    """
    if not isinstance(decision, dict):
        return False
    return any(isinstance(s, dict) and s.get("score") is not None
               for s in (decision.get("scores") or []))


def plan_score(decision: dict[str, Any], selected_index: int | None = None) -> float:
    """The selected plan's arbiter score, or 0.0 when nothing was measured.

    The float contract is load-bearing (`chapter_metrics.plan_score`), so 0.0 is
    the "no measurement" sentinel — use `decision_has_score` to tell that apart
    from a genuinely terrible plan before gating on it.
    """
    scores = decision.get("scores") or []
    if not scores:
        return 0.0
    if selected_index is None:
        selected_index = _coerce_index(decision.get("selected_index", 0))
    for score in scores:
        if not isinstance(score, dict):
            continue
        if _coerce_index(score.get("index", -1), -1) == selected_index:
            return safe_score(score.get("score", 0))
    first = scores[0] if isinstance(scores[0], dict) else {}
    return safe_score(first.get("score", 0))


# ---------------------------------------------------------------------------
# Repair ladder — extracted to engine/quality_repair.py, re-exported here
# for backward compatibility (engine/loop.py, tests, tools all import from
# engine.quality).
# ---------------------------------------------------------------------------
from engine.quality_repair import (  # noqa: F401,E402
    ACTION_BY_GATE,
    REPORT_KEY,
    _EM_DASH_L1_ACTION,
    _L1_FIXERS,
    _numbered_rewrite,
    apply_l0,
    apply_l1,
    em_dash_targeted,
    expand_to_band,
    gate_fired,
    gate_result,
    inject_dialogue,
    merge_fragment_lines,
    plan_repairs,
    promote_action_opening,
    reduce_em_dash_if_needed,
    reduce_em_dashes_targeted,
    rotate_fossils,
)



from engine.quality_advisory import (  # noqa: F401,E402
    _NEGATIVE_PAIR,
    _TEMPLATE_PIVOT,
    ai_flavor_health,
    chapter_ending_strength,
    hook_tail_repetition,
    information_density,
    intra_chapter_repetition,
    long_span_fatigue,
    paragraph_shape_health,
    payoff_beat_density,
    prose_texture,
    shareable_line,
)
