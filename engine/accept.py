"""Acceptance set, contract fulfilment, citation check, repair ladder, corpus.

Extracted from ``engine.loop`` to slim the decision-table module.  ``engine.loop``
re-exports every public name for backward compat, so existing imports keep working.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Iterable, Sequence

import engine.store as store
from engine.config import (
    log,
    text_bigrams,
)
from engine import quality
from engine.quality import REGISTRY, hard_block_reasons
from engine.state import (
    AcceptanceReport,
    DEFAULT_TAIL_CHARS,
    GateResult,
    Section,
)


# ===========================================================================
# Acceptance set (inlined from v2/accept.py)
# ===========================================================================

ACCEPTANCE_GATES: tuple[str, ...] = (
    "style_health",
    "cross_chapter_repetition",
    "book_wide_fossils",
    "descriptor_frequency",
    "genre_adherence",
    "adjacent_repetition",
    "length_band_check",
    "opening_hook_gate",
)

NATIVE_CHECKS: tuple[str, ...] = ("contract_fulfilment", "citation_check")

NOT_IN_ACCEPTANCE: dict[str, str] = {
    "beat_coverage":
        "superseded by contract_fulfilment, which asks the same question "
        "(did the prose stage what was promised) against the whole card rather "
        "than the `beats` list alone. Running both would double-charge.",
    "dialogue_health":
        "runs as advisory in acceptance_report (directives forwarded to next "
        "chapter) but never blocks — penalty only, no path into "
        "hard_block_reasons.",
    "intra_chapter_repetition":
        "runs as advisory — penalty only, directives forwarded.",
    "hook_tail_repetition":
        "runs as advisory — penalty only, directives forwarded.",
    "scene_similarity": "card-phase; see CARD_GATES.",
    "narrative_pattern_repetition": "card-phase; see CARD_GATES.",
    "plan_visual_payoff_check": "card-phase; see CARD_GATES.",
    "plan_executability_gate": "card-phase; see CARD_GATES.",
    "chapter_ending_strength":
        "L1 repair-only: hook_revise rewrites weak endings but the gate "
        "carries no penalty and never emits reject level, so it has no "
        "path into hard_block_reasons. Blocking is via repair improvement, "
        "not via acceptance rejection.",
}

CARD_GATES: tuple[str, ...] = (
    "scene_similarity",
    "narrative_pattern_repetition",
    "plan_visual_payoff_check",
    "plan_executability_gate",
)


# ---------------------------------------------------------------------------
# CCC -- contract fulfilment
# ---------------------------------------------------------------------------

HARD_FIELDS: tuple[str, ...] = ("where", "turn", "exit_hook")
FORBID_MIN_ANCHOR = 4

OBLIGATION_MARKERS: tuple[str, ...] = ("必须", "务必", "应当", "应该", "需要", "要求")
PROHIBITION_MARKERS: tuple[str, ...] = (
    "禁止", "严禁", "不要", "不得", "不可", "不能", "不准", "避免", "别", "勿", "忌",
)


def _is_misfiled_requirement(entry: str) -> bool:
    """True when this ``forbid`` entry is actually a REQUIREMENT, not a ban."""
    e = str(entry or "")
    if any(m in e for m in PROHIBITION_MARKERS):
        return False
    return any(m in e for m in OBLIGATION_MARKERS)


def _body(text: str) -> str:
    return quality._strip_title_line(str(text or ""))


def _anchors(target: str) -> list[str]:
    return quality._beat_anchor_fragments(str(target or ""))


REQUIRED_FIELDS: tuple[str, ...] = (
    "where", "who", "turn", "payoff", "exit_hook", "beats", "goal", "conflict",
    "title", "opening_type",
)


def _required_text(card: dict[str, Any] | None) -> str:
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
    """Did the prose stage what the ChapterCard promised? Zero LLM."""
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
        return out
    grams = text_bigrams(body, strip="none")
    tail = body[-int(cfg.get("ccc_tail_chars", DEFAULT_TAIL_CHARS)):]
    tail_grams = text_bigrams(tail, strip="none")
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

    for field in ("where", "turn", "payoff"):
        target = str(card.get(field) or "").strip()
        if target:
            add(field, target, *_hit(target, body, grams))

    reaction = str(card.get("payoff_reaction") or "").strip()
    if reaction:
        add("payoff_reaction", reaction, *_hit(reaction, body, grams))

    for raw_name in (card.get("who") or []):
        name = re.sub(r"[（(][^)）]*[)）]", "", str(raw_name or "")).strip()
        if len(name) >= 2:
            got = name in body
            add("who", name, got, [name], [name] if got else [])

    hook = str(card.get("exit_hook") or "").strip()
    if hook:
        add("exit_hook", hook, *_hit(hook, tail, tail_grams))

    for beat in (card.get("beats") or []):
        beat = str(beat or "").strip()
        if beat:
            add("beats", beat, *_hit(beat, body, grams))

    required = _required_text(card)
    for entry in (card.get("forbid") or []):
        entry = str(entry or "").strip()
        if not entry:
            continue

        if _is_misfiled_requirement(entry):
            out["forbid_conflicts"].append(
                {"field": "forbid", "target": entry, "phrase": "",
                 "why": "requirement_misfiled_as_ban"})
            continue

        def charge(phrase: str) -> bool:
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

QUOTE_KEYS = ("quote", "evidence", "excerpt", "原文", "证据", "引用", "locator")

_PUNCT_RE = re.compile("[\\s\u201c\u201d\"'\u2018\u2019\u300c\u300d\u300e\u300f\uff08\uff09()\\[\\]\uff0c,\u3002.\uff01!\uff1f?\uff1b;\uff1a:\u3001\u2026\u2014\\-~\u00b7]+")


def _normalize_quote(s: str) -> str:
    return _PUNCT_RE.sub("", str(s or ""))


def citation_check(
    claims: Iterable[dict[str, Any]] | None,
    text: str,
    min_quote_chars: int = 4,
) -> dict[str, Any]:
    """Drop every claim that cannot point at a substring of the chapter."""
    claims_list = [c for c in (claims or []) if isinstance(c, dict)]
    body = _normalize_quote(_body(text))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for claim in claims_list:
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
    total = len(claims_list)
    return {"kept": kept, "dropped": dropped, "total": total,
            "drop_rate": (len(dropped) / total) if total else 0.0}


# ---------------------------------------------------------------------------
# The acceptance report
# ---------------------------------------------------------------------------

def _em_history(
    conn: Any,
    chapter_num: int,
    config: dict[str, Any],
) -> list[float] | None:
    """The prior chapters' em-dash density, oldest-first, or None."""
    if conn is None:
        return None
    window = max(int(config.get("novel", {}).get(
        "style_em_dash_trend_window", 5)), 1)
    try:
        rows = store.recent_metrics(conn, window + 1)
    except Exception:
        return None
    seq = sorted(
        (int(r["chapter"]), float(r["em_dash_per_kchar"]))
        for r in rows
        if r.get("chapter") is not None
        and isinstance(r.get("em_dash_per_kchar"), (int, float))
        and int(r["chapter"]) < chapter_num
    )
    return [v for _, v in seq[-window:]] or None


_DETERMINISTIC_PREFIX = "DETERMINISTIC: "


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
) -> dict[str, Any]:
    """Run the acceptance set and emit a v1-schema review payload.

    v3 simplification: advisory-only gates removed, canon_claims removed.
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
    report["_card"] = card
    if ccc["enabled"] and not ccc["passed"]:
        report["gate_rejects"].append({
            "gate": "contract_fulfilment", "level": "reject",
            "phrases": [i["target"] for i in ccc["hard_misses"]][:4],
            "violations": [v["phrase"] for v in ccc["violations"]][:4]})
    if ccc.get("forbid_conflicts"):
        report["card_defects"] = [
            (f"卡片自相矛盾：`forbid` 里这条其实是硬性要求而不是禁令"
             f"（{c['target'][:60]}…），整条豁免"
             if c.get("why") == "requirement_misfiled_as_ban" else
             f"卡片自相矛盾：`forbid` 禁了本卡片自己要求的「{c['phrase']}」"
             f"（出自 {c['target'][:40]}…），已豁免")
            for c in ccc["forbid_conflicts"]]

    # --- advisory gates (zero-LLM, directives only) ------------------------
    if enabled("ai_flavor_health"):
        report["ai_flavor_health"] = quality.ai_flavor_health(body, config)
    if enabled("paragraph_shape_health"):
        report["paragraph_shape_health"] = quality.paragraph_shape_health(body, config)
    if prior_texts and enabled("hook_tail_repetition"):
        report["hook_tail_repetition"] = quality.hook_tail_repetition(
            body, prior_texts, config)
    if enabled("intra_chapter_repetition"):
        report["intra_chapter_repetition"] = quality.intra_chapter_repetition(body, config)
    if enabled("prose_texture"):
        report["prose_texture"] = quality.prose_texture(body, config)
    if enabled("dialogue_health"):
        report["dialogue_health"] = quality.dialogue_health(body, config)
    if conn and enabled("long_span_fatigue"):
        report["long_span_fatigue"] = quality.long_span_fatigue(
            conn, chapter_num, config)
    if enabled("payoff_beat_density"):
        report["payoff_beat_density"] = quality.payoff_beat_density(
            body, recent_payoff_types, config)
    if enabled("payoff_reaction_check"):
        report["payoff_reaction_check"] = quality.payoff_reaction_check(
            body, config)
    if enabled("shareable_line"):
        report["shareable_line"] = quality.shareable_line(body, config)
    if enabled("information_density"):
        report["information_density"] = quality.information_density(
            body, card, None, config)
    if enabled("chapter_ending_strength"):
        report["chapter_ending_strength"] = quality.chapter_ending_strength(
            body, config)
    if conn and enabled("entity_consistency"):
        report["entity_consistency"] = quality.entity_consistency(
            body, conn, chapter_num, config)
    if conn and enabled("thread_overdue"):
        report["thread_overdue"] = quality.thread_overdue(
            conn, chapter_num, config)
    # --- V3 naturalness advisory gates ----------------------------------------
    if enabled("sentence_variety"):
        report["sentence_variety"] = quality.sentence_variety(body, config)
    if enabled("connective_abuse"):
        report["connective_abuse"] = quality.connective_abuse(body, config)
    if enabled("sensory_deficit"):
        report["sensory_deficit"] = quality.sensory_deficit(body, config)
    if enabled("lexical_monotony"):
        prior_ttrs = []
        if prior_texts:
            from engine.quality_advisory import _bigram_ttr
            for pt in prior_texts[-3:]:
                prior_ttrs.append(_bigram_ttr(pt))
        report["lexical_monotony"] = quality.lexical_monotony(
            body, config, prior_ttrs or None)

    # --- directives (priority-tagged for feed-forward) ----------------------
    _BLOCKING_GATES = {
        "style_health", "length_band", "opening_hook_gate",
        "adjacent_repetition", "cross_chapter_repetition",
        "book_fossils", "descriptor_frequency", "genre_adherence",
        "contract_fulfilment",
    }
    _ADVISORY_GATES = (
        "ai_flavor_health", "paragraph_shape_health",
        "hook_tail_repetition", "intra_chapter_repetition",
        "prose_texture", "dialogue_health", "long_span_fatigue",
        "payoff_beat_density", "payoff_reaction_check",
        "shareable_line", "information_density",
        "chapter_ending_strength", "entity_consistency",
        "thread_overdue",
        "sentence_variety", "connective_abuse",
        "sensory_deficit", "lexical_monotony",
    )
    wd = report["writer_directives_for_next_chapter"]
    for key in (*_BLOCKING_GATES, *_ADVISORY_GATES):
        gate_result = report.get(key) or {}
        penalty = float(gate_result.get("penalty", 0.0))
        if key in _BLOCKING_GATES:
            prefix = "[P1] "
        elif penalty > 0:
            prefix = "[P2] "
        else:
            prefix = "[P3] "
        for d in gate_result.get("directives", []):
            tagged = prefix + d
            if tagged not in wd:
                wd.append(tagged)

    # --- finalize verdict -------------------------------------------------
    reasons = hard_block_reasons(report, config)
    report["block_reasons"] = reasons
    report["accepted"] = not reasons
    problems = report.get("problems") or []
    if reasons:
        problems.append(_DETERMINISTIC_PREFIX + "; ".join(reasons))
    report["problems"] = problems
    return report


def block_reasons(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """The one ruler, re-exported so callers never grow a second copy."""
    return hard_block_reasons(report, config)


# ===========================================================================
# Repair ladder (inlined from v2/repair.py)
# ===========================================================================

REPAIR_LAYERS: tuple[str, ...] = ("L0", "L1")

_KIND_RE = re.compile(r"^[a-z_]+")

Recheck = Callable[[str], dict]


def reason_kind(reason: Any) -> str:
    m = _KIND_RE.match(str(reason or ""))
    return m.group(0) if m else str(reason or "")


def reason_kinds(reasons: Iterable[Any]) -> frozenset[str]:
    return frozenset(reason_kind(r) for r in (reasons or ()))


def repair_pending(report: dict[str, Any], config: dict[str, Any],
                   layer: str) -> tuple[str, ...]:
    """The repair actions this layer still has to offer for this report."""
    if not isinstance(report, dict):
        return ()
    try:
        steps = quality.plan_repairs(report, config)
    except Exception:
        return ()
    return tuple(s["action"] for s in steps if s.get("layer") == layer)


@dataclasses.dataclass(frozen=True)
class RepairOutcome:
    """What a repair pass did, and whether the ruler agrees it helped."""

    text: str
    report: dict[str, Any]
    applied: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()
    blocks_before: tuple[str, ...] = ()
    blocks_after: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    @property
    def cleared(self) -> bool:
        return not self.blocks_after

    @property
    def improved(self) -> bool:
        return len(self.blocks_after) < len(self.blocks_before)

    def summary(self) -> str:
        bits = [f"blocks {len(self.blocks_before)}->{len(self.blocks_after)}"]
        if self.applied:
            bits.append("applied=" + ",".join(self.applied))
        if self.reverted:
            bits.append("reverted=" + ",".join(self.reverted))
        if self.skipped:
            bits.append("skipped=" + ",".join(self.skipped))
        return " ".join(bits)


def _repair_log(paths: Any, message: str) -> None:
    if paths is None:
        return
    try:
        log(paths, message)
    except Exception:
        pass


def _regressed(before: Sequence[str], after: Sequence[str]) -> str:
    """'' when the repair is safe to keep, else why it is not."""
    if len(after) < len(before):
        return ""
    if len(after) > len(before):
        return f"count {len(before)}->{len(after)}"
    swapped = reason_kinds(after) - reason_kinds(before)
    if swapped:
        return "swapped_for=" + ",".join(sorted(swapped))
    return ""


def run_repair_layer(
    layer: str,
    *,
    text: str,
    report: dict[str, Any],
    config: dict[str, Any],
    chapter_num: int,
    recheck: Recheck,
    client: Any = None,
    paths: Any = None,
) -> RepairOutcome:
    """Run ONE repair layer and re-score it. Never raises, never re-drafts."""
    before = tuple(block_reasons(report, config))
    idle = RepairOutcome(text=text, report=report, blocks_before=before,
                         blocks_after=before)

    actions = repair_pending(report, config, layer)
    if not actions:
        return idle
    if layer == "L1" and client is None:
        _repair_log(paths, f"v2.repair Ch{chapter_num} L1 skipped (no client): "
                    + ",".join(actions))
        return dataclasses.replace(idle, skipped=actions)

    try:
        if layer == "L0":
            new_text, applied = quality.apply_l0(text, report, config, chapter_num)
        else:
            new_text, applied = quality.apply_l1(client, paths, config, chapter_num,
                                                 text, report)
    except Exception as exc:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} failed (non-fatal): {exc}")
        return idle

    if not applied or not new_text or new_text == text:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} kept nothing from "
                    f"[{','.join(actions)}]")
        return idle

    try:
        new_report = recheck(new_text)
    except Exception as exc:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} recheck failed, "
                    f"reverting (non-fatal): {exc}")
        return dataclasses.replace(idle, reverted=tuple(applied))

    after = tuple(block_reasons(new_report, config))
    why = _regressed(before, after)
    if why:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} REVERTED ({why}): "
                    + ",".join(applied))
        return dataclasses.replace(idle, reverted=tuple(applied))

    _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} kept "
                f"[{','.join(applied)}] blocks {len(before)}->{len(after)}")
    return RepairOutcome(text=new_text, report=new_report,
                         applied=tuple(applied), layers=(layer,),
                         blocks_before=before, blocks_after=after)


