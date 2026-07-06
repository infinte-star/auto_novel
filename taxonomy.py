"""Fiction failure taxonomy — a canonical vocabulary for what went wrong in a
chapter, adapted from writing-harness's "encode taste as named failure modes"
idea but specialized for serialized fiction (not its non-fiction AEH/CTA path).

Today the pipeline's failure signals are scattered: ~15 free-text `problems`
prefixes in review.py (MARKET/SERIAL/EMOTION/RETENTION/…), semi-structured
`gate_rejects` dicts, and a coarse local/structural split in
pipeline._classify_replan_failure. distill.py guesses categories from free-text
keywords. This module gives those a single named vocabulary + a fix-routing
table so replan routing is precise and cross-book distillation reads structured
codes instead of guessing.

Everything here is PURE (no I/O, no LLM) and additive: review.py tags each
report with `failure_codes` derived from the existing messages; nothing existing
changes behavior. Import failures anywhere degrade to the legacy path.

fix_route values:
  local          — surgical revise/patch keeps the scene (prose, length, constraint)
  hook_only       — rewrite just the chapter tail (weak/recycled hook, intra recap)
  opening_rewrite — rewrite the chapter head (info-dump/scenery opening)
  structural      — regenerate plan → rewrite (scene design is wrong)
  arc_replan      — re-plan the whole arc (retention sag; needs rolling_plan)
  advisory        — record only, no forced fix
"""
from __future__ import annotations

# code -> metadata. `detector` names the gate/axis that raises it (documentation).
FAILURE_TAXONOMY: dict[str, dict[str, str]] = {
    "retention_sag":      {"label": "读者留存下滑",   "detector": "reader_panel/P2 gate",       "fix_route": "arc_replan"},
    "flat_stakes":        {"label": "平路无爽点",     "detector": "payoff_density/flat_streak",  "fix_route": "structural"},
    "payoff_deferred":    {"label": "关键维度/兑现不足", "detector": "key_dimension_floor",       "fix_route": "structural"},
    "market_weak":        {"label": "爆款维度短板",   "detector": "MARKET cap",                 "fix_route": "structural"},
    "emotion_flat":       {"label": "情感冲击不足",   "detector": "emotional_impact floor",     "fix_route": "structural"},
    "fossil_repetition":  {"label": "跨章化石复读",   "detector": "cross_chapter/book_fossils", "fix_route": "structural"},
    "adjacent_repeat":    {"label": "复述上一章",     "detector": "adjacent_repetition",        "fix_route": "local"},
    "intra_recap":        {"label": "章末零增量总结", "detector": "intra_chapter_repetition",   "fix_route": "hook_only"},
    "weak_hook":          {"label": "钩子弱/里程碑",  "detector": "serial milestone/hook",      "fix_route": "hook_only"},
    "info_dump_open":     {"label": "开篇铺垫非危机", "detector": "opening_golden_gate",        "fix_route": "opening_rewrite"},
    "style_collapse":     {"label": "文体塌缩",       "detector": "style_health",               "fix_route": "local"},
    "pov_leak":           {"label": "视角越界",       "detector": "suspense review axis",        "fix_route": "local"},
    "contract_violation": {"label": "能力/设定违约",  "detector": "contract",                   "fix_route": "structural"},
    "deus_ex_machina":    {"label": "天降解题",       "detector": "contract",                   "fix_route": "structural"},
    "fact_contradiction": {"label": "事实矛盾",       "detector": "factcheck",                  "fix_route": "structural"},
    "length_out_of_band": {"label": "章长出带",       "detector": "length_band",                "fix_route": "local"},
    "constraint_miss":    {"label": "约束未兑现",     "detector": "constraint_verification",    "fix_route": "local"},
}

# Legacy free-text problem prefix (before the first ':' or '：') -> code.
PREFIX_TO_CODE: dict[str, str] = {
    "MARKET": "market_weak",
    "SERIAL": "weak_hook",
    "EMOTION": "emotion_flat",
    "RETENTION": "retention_sag",
    "KEY-DIM": "payoff_deferred",
    "STYLE": "style_collapse",
    "OPENING": "info_dump_open",
    "LENGTH": "length_out_of_band",
    "CONSTRAINT": "constraint_miss",
    "GATE": "fossil_repetition",
    "CONTRACT": "contract_violation",
    "FACTCHECK": "fact_contradiction",
    "RECAP": "intra_recap",
    "REPEAT": "adjacent_repeat",
    "FLAT": "flat_stakes",
    "POV": "pov_leak",
}

# gate_rejects[].gate -> code.
GATE_TO_CODE: dict[str, str] = {
    "cross_chapter_repetition": "fossil_repetition",
    "book_wide_fossils": "fossil_repetition",
    "adjacent_repetition": "adjacent_repeat",
}

# Highest-priority route wins when a chapter carries several codes.
_ROUTE_PRIORITY = ["arc_replan", "structural", "opening_rewrite", "hook_only", "local", "advisory"]
# Which routes mean "regenerate the scene" for pipeline._classify_replan_failure's
# binary local/structural contract.
_STRUCTURAL_ROUTES = {"arc_replan", "structural"}


def classify_problem(text: str) -> str | None:
    """Map a legacy `problems` string (e.g. 'MARKET: …') to a taxonomy code."""
    if not text:
        return None
    head = str(text).split("：", 1)[0].split(":", 1)[0].strip().upper()
    return PREFIX_TO_CODE.get(head)


def classify_gate(gate_name: str) -> str | None:
    """Map a `gate_rejects[].gate` name to a taxonomy code."""
    return GATE_TO_CODE.get(str(gate_name or "").strip())


def fix_route(code: str) -> str:
    """Return the fix route for a code (unknown → 'local')."""
    return (FAILURE_TAXONOMY.get(code) or {}).get("fix_route", "local")


def codes_from_review(report: dict) -> list[str]:
    """Derive the sorted unique taxonomy codes for a review report from its
    existing `problems` prefixes and `gate_rejects` gates. Pure/tolerant."""
    codes: set[str] = set()
    for p in (report.get("problems") or []):
        c = classify_problem(p if isinstance(p, str) else str(p))
        if c:
            codes.add(c)
    for g in (report.get("gate_rejects") or []):
        if isinstance(g, dict):
            c = classify_gate(g.get("gate", ""))
            if c:
                codes.add(c)
    return sorted(codes)


def dominant_route(codes: list[str]) -> str | None:
    """Highest-priority fix route across the codes, or None when empty."""
    routes = {fix_route(c) for c in codes}
    for r in _ROUTE_PRIORITY:
        if r in routes:
            return r
    return None


def replan_kind(codes: list[str]) -> str | None:
    """Map the codes to pipeline's binary replan contract: 'structural' if any
    code routes to a scene/arc regeneration, else 'local'. None when empty so
    callers fall back to the legacy heuristics."""
    if not codes:
        return None
    routes = {fix_route(c) for c in codes}
    return "structural" if routes & _STRUCTURAL_ROUTES else "local"
