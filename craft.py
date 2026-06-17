"""Cross-book craft-rule consumption layer (closes the distillation loop).

`distill.py` mines every novel's `story_state.db` for recurring failure->fix
patterns and writes `craft_rules.json`. Historically that file was WRITTEN but
NEVER READ by the runtime -- the last mile of cross-book learning was broken.

This module is that last mile: it loads the distilled rules and renders them
into prompt blocks that the writer (`writing.py`) and planner (`planning.py`)
consume, so a brand-new book inherits the whole library's accumulated craft
lessons from chapter 1 instead of re-discovering them via the in-book
preflight loop (which only looks back ~5 chapters and is empty early on).

Design constraints (load-bearing, mirror telemetry.py's observer contract):
  * NEVER break or slow the generation pipeline. Every public function returns
    a safe empty value on ANY failure (file missing, malformed JSON, bad
    schema, ...). A missing craft_rules.json is the COMMON case (fresh
    install) and MUST be a silent no-op.
  * Zero new dependencies -- stdlib + config only.
  * Pure functions over the loaded dict: trivially unit-testable without an LLM
    or a live novel.

Consumption is gated by `novel.craft_rules_enabled` (default true) but is
inert whenever the rules file is absent or yields no qualifying rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import ROOT

# Categories that speak to *writing execution* (injected into the writer).
_WRITER_CATEGORIES = {
    "style",
    "hook_technique",
    "payoff_setup",
    "beat_execution",
    "character_consistency",
    "world_logic",
}
# Categories that speak to *plan structure* (injected into the planner).
_PLANNER_CATEGORIES = {
    "payoff_setup",
    "hook_technique",
    "beat_execution",
    "world_logic",
}


def _resolve_path(config: dict[str, Any]) -> Path:
    raw = str(config.get("novel", {}).get("craft_rules_path", "craft_rules.json")).strip()
    if not raw:
        raw = "craft_rules.json"
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_craft_rules(config: dict[str, Any]) -> dict[str, Any]:
    """Load craft_rules.json. Returns {"rules": [], "meta": {}} on any failure."""
    empty = {"rules": [], "meta": {}}
    try:
        if not bool(config.get("novel", {}).get("craft_rules_enabled", True)):
            return empty
        path = _resolve_path(config)
        if not path.exists():
            return empty
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return empty
        rules = data.get("rules")
        if not isinstance(rules, list):
            return empty
        return {"rules": rules, "meta": data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}}
    except Exception:
        return empty


def _qualifying_rules(
    config: dict[str, Any],
    categories: set[str] | None,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Filter loaded rules by confidence + category, sorted by a value score.

    Value score prefers rules with a measured positive score delta (real
    evidence that the fix raised quality) and falls back to evidence_count.
    All field accesses are defensive: a malformed rule is skipped, never raised.
    """
    data = load_craft_rules(config)
    rules = data.get("rules", [])
    if not rules:
        return []
    novel_cfg = config.get("novel", {})
    try:
        min_conf = float(novel_cfg.get("craft_rules_min_confidence", 0.3))
    except (TypeError, ValueError):
        min_conf = 0.3
    if top_k is None:
        try:
            top_k = int(novel_cfg.get("craft_rules_top_k", 6))
        except (TypeError, ValueError):
            top_k = 6

    out: list[tuple[float, dict[str, Any]]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        try:
            conf = float(r.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_conf:
            continue
        cat = str(r.get("category", "other"))
        if categories is not None and cat not in categories:
            continue
        pattern = str(r.get("pattern", "")).strip()
        fix = str(r.get("fix", "")).strip()
        if not pattern or not fix:
            continue
        try:
            before = float(r.get("avg_score_before", 0.0) or 0.0)
            after = float(r.get("avg_score_after", 0.0) or 0.0)
        except (TypeError, ValueError):
            before = after = 0.0
        delta = after - before
        try:
            evidence = float(r.get("evidence_count", 0) or 0)
        except (TypeError, ValueError):
            evidence = 0.0
        # Rules with a measured positive delta rank first; otherwise confidence*evidence.
        value = (delta * 10.0 if delta > 0 else 0.0) + conf * min(evidence, 20.0)
        out.append((value, r))

    out.sort(key=lambda t: t[0], reverse=True)
    return [r for _v, r in out[: max(0, int(top_k))]]


def craft_writer_block(config: dict[str, Any]) -> str:
    """Render a writer-prompt block of cross-book craft lessons. "" if none."""
    rules = _qualifying_rules(config, _WRITER_CATEGORIES)
    if not rules:
        return ""
    lines = [
        "## 跨书工艺规律（来自全库历史的反复教训·硬性参考）",
        "以下是从多本已完成作品中蒸馏出的、反复导致扣分的失败模式与对应修复。"
        "本章动笔前请把它们当作既有教训规避，不要重新踩坑：",
    ]
    for r in rules:
        pattern = str(r.get("pattern", "")).strip()
        fix = str(r.get("fix", "")).strip()
        try:
            ev = int(float(r.get("evidence_count", 0) or 0))
        except (TypeError, ValueError):
            ev = 0
        lines.append(f"- 失败模式：{pattern}")
        lines.append(f"  → 修复：{fix}（{ev} 本书共现）")
    return "\n".join(lines) + "\n"


def craft_planner_hints(config: dict[str, Any]) -> str:
    """Render a planner-prompt block of cross-book structural lessons. "" if none."""
    rules = _qualifying_rules(config, _PLANNER_CATEGORIES)
    if not rules:
        return ""
    lines = [
        "## 跨书规划教训（来自全库历史·本章大纲须主动规避）",
        "以下结构性失败在多本书中反复出现。请在 goal/conflict/payoff/beats/hook 的设计上提前规避：",
    ]
    for r in rules:
        pattern = str(r.get("pattern", "")).strip()
        fix = str(r.get("fix", "")).strip()
        lines.append(f"- {pattern} → {fix}")
    return "\n".join(lines) + "\n"
