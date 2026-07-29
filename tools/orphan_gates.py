"""Settle the gates the v1 deletion orphaned: wire, redesign, or delete.

Zero LLM calls, read-only. The v1 deletion at `95361b9` left 12 registered gates
with no caller, because `review.py` invoked them by name. CLAUDE.md's rule is
that a silent gate is a bug report rather than a deletion candidate — so each one
needed a measured verdict, and `tools/gate_census.py` could give only six of
them: it reads archived review payloads, so a gate whose result v1 never *stored*
under its own key reads as "never ran" when the truth is "never measured".

That settlement is done (`beat_coverage` superseded by CCC, nine wired as
advisories in `v2/accept.py`, `flat_chapter_streak` and `emotional_cadence`
deleted). This tool stays because it is the only thing that can re-measure those
nine from primary data: they are advisories, so nothing archives their verdicts
and `gate_census` will report them silent forever. Run it after touching any of
their thresholds — the `proof=` strings in `quality.py` quote its output verbatim.

It supplies the two measurements the census structurally cannot:

**1. Input availability on v2-written prose.** The metrics-reading gates read
`chapter_metrics` columns that v1 filled from its LLM self-review — `tension`,
`emotional_tone`, `emotional_impact`, `score`. v2 has no self-score, so those
columns are NULL on every chapter it writes, while `payoff_type` /
`conflict_type` (which come from the ChapterCard) are populated. A gate whose
verdict depends on a column that is structurally empty cannot fire on v2 — or
worse, reads the empty value as the bad one and fires *always*. That is the
"can the signal distinguish bad from not-measured" defect class in CLAUDE.md,
and it is decidable without an A/B. Two gates were deleted on exactly this
evidence, and the audit tables below are kept as the record of it — the columns
are still reported for gates that no longer exist as inputs, so a future gate
proposed over `emotional_tone` meets the same measurement before it is written.

**2. Fire rates recomputed from primary data.** For the text-only gates, run them
over the archive's real chapter files instead of over what v1 happened to
archive. Same relationship to `gate_census` that `replay_gates` has to
`fpy_prime`: one replays frozen verdicts, the other recomputes them.

`_fired` / `_advised` are imported from `gate_census` rather than reimplemented —
the fire/advise distinction is load-bearing (LESSONS §4) and must have exactly
one definition.

    python tools/orphan_gates.py [novel…] [--v2-window NAME:FROM-TO] [--all]

`beat_coverage` is excluded: it is already settled, superseded by
`v2.accept.contract_fulfilment`, which widened it from `beats` to the whole card.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import engine.config as config_mod  # noqa: E402
import engine.quality as quality  # noqa: E402
from gate_census import _advised, _fired  # noqa: E402
from fpy_prime import discover_novels, print_exclusions  # noqa: E402

# The gates in scope, and which primary data each one needs. Three are absent by
# settlement rather than by oversight: `beat_coverage` (superseded by CCC),
# `flat_chapter_streak` and `emotional_cadence` (deleted — the measurements are in
# their former `proof=` strings and in the audit tables this tool still prints).
# A deleted gate MUST come out of this tuple: `quality.REGISTRY.get` returns None
# for it, `evaluate` returns None, and it would print a permanent "0 ran" that
# reads identically to a measurement failure.
ORPHANS: tuple[str, ...] = (
    "ai_flavor_health",
    "paragraph_shape_health",
    "prose_texture",
    "shareable_line",
    "intra_chapter_repetition",
    "hook_tail_repetition",
    "payoff_beat_density",
    "information_density",
    "long_span_fatigue",
)

# Which `chapter_metrics` column each metrics-reading gate's verdict hangs on.
# The point of naming them is that v2 fills some and not others — so this dict has
# to track what the gate READS today, not what it once read. `long_span_fatigue`
# listed `tension, emotional_tone` until its trim removed both terms, and the
# report kept saying DEAD INPUT for a gate whose one surviving term reads
# `payoff_type` (30/30 on v2).
NEEDS_COLUMN: dict[str, tuple[str, ...]] = {
    "payoff_beat_density": ("payoff_type",),
    "long_span_fatigue": ("payoff_type",),
}
AUDIT_COLUMNS = ("payoff_type", "conflict_type", "tension", "emotional_tone",
                 "emotional_impact", "score")

# The metric each silent gate blocks/warns on, so a 0% firing rate can be told
# apart from a threshold that is unreachable by construction. This is LESSONS §4's
# own deletion test ("compare the threshold against the metric's measured
# distribution"), which is why the observed maximum is reported next to the line
# rather than left to a reader's guess: `dialogue_pingpong` was deleted on a
# 0.50 threshold vs an observed max of 0.140, and `adjacent_repetition` was KEPT
# on 0/641 because its line sits just above the observed max instead of 3x above.
#
# The 4th field is which tail of the distribution the gate fires on. It matters:
# `shareable_line` fires when its metric is LOW, so comparing its line to the
# observed MAXIMUM reports 0.5x ("well within reach") for a gate whose real
# question is how often the minimum dips under the line.
PROBE: dict[str, tuple[str, Any, Any, str]] = {
    "intra_chapter_repetition": (
        "tail_recap_ratio",
        lambda r: float((r.get("metrics") or {}).get("tail_recap_ratio", 0.0) or 0.0),
        lambda c: float(c.get("intra_repeat_warn", 0.25)),
        "high",
    ),
    "hook_tail_repetition": (
        "tail_clause_overlap",
        lambda r: float(r.get("ratio", 0.0) or 0.0),
        lambda c: float(c.get("hook_repeat_min_ratio", 0.25)),
        "high",
    ),
    "payoff_beat_density": (
        "chapters_since_payoff",
        lambda r: float((r.get("metrics") or {}).get("chapters_since_payoff", 0) or 0),
        lambda c: 1.0 / max(float(c.get("payoff_density_min", 0.34)), 1e-6),
        "high",
    ),
    "shareable_line": (
        "best_quotable_score",
        lambda r: float(r.get("score", 0.0) or 0.0),
        lambda c: float(c.get("shareable_min_score", 2.0)),
        "low",
    ),
}


def novel_config(name: str) -> dict:
    """The novel's OWN config — same convention as `tools/replay_gates.py`.

    Not pinned to engine defaults: the question here is what each gate would do
    inside the book it is being replayed against, and that book's thresholds are
    part of the answer.
    """
    saved = config_mod.CONFIG_FILE
    try:
        config_mod.CONFIG_FILE = ROOT / "novels" / name / "config.yaml"
        return config_mod.load_config()
    except Exception:
        config_mod.CONFIG_FILE = saved
        return config_mod.load_config()
    finally:
        config_mod.CONFIG_FILE = saved


def chapter_texts(novel: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for p in sorted((novel / "chapters").glob("*.md")):
        digits = "".join(c for c in p.stem if c.isdigit())
        if not digits:
            continue
        try:
            out.append((int(digits), p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            continue
    return sorted(out)


def metrics_rows(novel: Path) -> list[dict[str, Any]]:
    db = novel / "story_state.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM chapter_metrics ORDER BY chapter ASC")]
        conn.close()
        return rows
    except Exception:
        return []


def _filled(value: Any) -> bool:
    """Is this column value real data, or the placeholder v2 leaves behind?

    `0` and `0.0` count as empty on purpose: `emotional_impact` defaults to 0.0
    through `writing.safe_score(review.get("emotional_impact", 0))`, and 0 is
    below every floor that reads it — so "absent" and "worst possible" are the
    same bit pattern. Distinguishing them is the whole point of the audit.
    """
    if value is None:
        return False
    s = str(value).strip()
    return s not in ("", "0", "0.0", "None")


class _RowsConn:
    """A `store.recent_metrics`-compatible shim over rows before chapter N.

    `recent_metrics` runs `SELECT * FROM chapter_metrics ORDER BY chapter DESC
    LIMIT ?`, which on a finished book always returns the book's tail rather than
    what was visible when chapter N was written. Replaying a book-scope gate
    against the wrong window would report the gate's verdict at HEAD for every
    chapter.
    """

    def __init__(self, rows: list[dict[str, Any]], before: int):
        self._rows = [r for r in rows if int(r.get("chapter") or 0) < before]
        self._rows.sort(key=lambda r: int(r.get("chapter") or 0), reverse=True)

    def execute(self, _sql: str, params: tuple = ()):  # noqa: D401
        limit = int(params[0]) if params else len(self._rows)
        rows = self._rows[:limit]
        return type("_Cur", (), {"fetchall": lambda self_: rows})()


def evaluate(gate: str, *, text: str, chapter: int, prev_texts: list[str],
             rows_before: list[dict[str, Any]], plan: dict[str, Any] | None,
             cfg: dict) -> Any:
    """Run ONE orphan gate with the best primary data available for it."""
    fn = quality.REGISTRY.get(gate)
    if fn is None:
        return None
    if gate in ("ai_flavor_health", "paragraph_shape_health", "prose_texture",
                "shareable_line", "intra_chapter_repetition"):
        return fn(text, cfg)
    if gate == "hook_tail_repetition":
        return fn(text, prev_texts, cfg)
    if gate == "payoff_beat_density":
        return fn(text, [str(r.get("payoff_type") or "") for r in rows_before], cfg)
    if gate == "information_density":
        # `review` is left None: v2 emits no `beats_audit`, so the fourth signal
        # is unavailable by construction and passing a v1 review here would
        # measure v1's reviewer instead of the gate. A chapter with no archived
        # plan is UNMEASURED, not low-information — see `archived_plan`.
        if plan is None:
            return None
        return fn(text, plan, None, cfg)
    if gate == "long_span_fatigue":
        return fn(_RowsConn(rows_before, chapter + 1), chapter, cfg)
    return None


def _checkpoint_payload(path: Path) -> Any:
    """The `payload` inside a checkpoint file, or None.

    `checkpoint.py` wraps every artifact as
    `{_checkpoint_version, chapter, saved_at, payload}`. Reading the wrapper as
    the artifact is how the first run of this tool reported
    `information_density` firing on 33% of v2 chapters: the wrapper has no
    `payoff_type`, so the gate scored "the card promised nothing" on 30 chapters
    whose cards each promised something. A missing payload returns None so the
    caller can count the chapter as UNMEASURED instead of as a firing.
    """
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(blob, dict) and "payload" in blob:
        return blob.get("payload")
    return blob


def archived_plan(novel: Path, chapter: int) -> dict[str, Any] | None:
    """The plan `information_density` would have read, from either engine.

    v2 archives the ChapterCard and the plan IS `arc.card_to_plan(card)`; v1
    archived the arbitration, whose payload carries `plan` directly. Returns None
    when neither exists — and the caller must then skip the gate rather than pass
    an empty dict, because `information_density` reads an absent `payoff_type` as
    a weak one and would score every unplanned chapter as low-information.
    """
    d = novel / "logs" / "checkpoints" / f"ch{chapter:04d}"
    card = _checkpoint_payload(d / "chapter_card.json")
    if isinstance(card, dict) and card.get("payoff_type") is not None:
        try:
            import engine.plan as arc

            plan, _ = arc.card_to_plan(card)
            return plan
        except Exception:
            return None
    arb = _checkpoint_payload(d / "plan_initial_attempt0_arbitration.json")
    if isinstance(arb, dict):
        plan = arb.get("plan")
        if not isinstance(plan, dict):
            plan = (arb.get("decision") or {}).get("merged_plan")
        if isinstance(plan, dict) and plan:
            return plan
    return None


def scan(names: list[str], window: tuple[str, int, int] | None
         ) -> tuple[dict, dict, dict, dict, dict]:
    """Returns (corpus, v2-window, column availability, probes, tone shapes)."""
    corpus = {g: {"ran": 0, "fired": 0, "advised": 0} for g in ORPHANS}
    v2win = {g: {"ran": 0, "fired": 0, "advised": 0} for g in ORPHANS}
    avail = {"v1": {c: [0, 0] for c in AUDIT_COLUMNS},
             "v2": {c: [0, 0] for c in AUDIT_COLUMNS}}
    probes: dict[str, list[float]] = {g: [] for g in PROBE}
    thresholds: dict[str, float] = {}
    tones: list[str] = []

    for name in names:
        novel = ROOT / "novels" / name
        cfg = novel_config(name)
        cfgn = cfg.get("novel", {})
        for gate, (_label, _get, thr, _side) in PROBE.items():
            thresholds.setdefault(gate, thr(cfgn))
        texts = chapter_texts(novel)
        rows = metrics_rows(novel)
        by_ch = {int(r.get("chapter") or 0): r for r in rows}
        in_window = (window[1], window[2]) if window and window[0] == name else None

        for idx, (ch, text) in enumerate(texts):
            prev = [t for _, t in texts[max(0, idx - 3):idx]]
            # NEWEST-FIRST, because that is what `store.recent_metrics` returns and
            # therefore the order every metrics-reading gate is written against.
            # Fed ascending, `payoff_beat_density` counts its drought forward from
            # chapter 1 — which breaks on the first strong chapter and reports 0 for
            # the whole corpus. That is what made the metrics-reading gates look
            # "unreachable by construction" on the first run rather than merely quiet,
            # and getting the order wrong is a silent 0%, never an error.
            rows_before = [r for r in reversed(rows) if int(r.get("chapter") or 0) < ch]
            plan = archived_plan(novel, ch)
            arm = "v2" if (in_window and in_window[0] <= ch <= in_window[1]) else "v1"

            row = by_ch.get(ch)
            if row:
                for col in AUDIT_COLUMNS:
                    avail[arm][col][1] += 1
                    if _filled(row.get(col)):
                        avail[arm][col][0] += 1
                tone = str(row.get("emotional_tone") or "").strip()
                if tone:
                    tones.append(tone)

            for gate in ORPHANS:
                try:
                    result = evaluate(gate, text=text, chapter=ch, prev_texts=prev,
                                      rows_before=rows_before, plan=plan, cfg=cfg)
                except Exception:
                    continue
                if result is None:
                    continue
                if gate in PROBE:
                    try:
                        probes[gate].append(PROBE[gate][1](result))
                    except Exception:
                        pass
                for bucket in ([corpus] + ([v2win] if arm == "v2" else [])):
                    s = bucket[gate]
                    s["ran"] += 1
                    if _fired(result):
                        s["fired"] += 1
                    if _advised(result):
                        s["advised"] += 1
    return corpus, v2win, avail, {"values": probes, "thresholds": thresholds}, {"tones": tones}


def _table(title: str, stats: dict) -> None:
    print(f"\n{title}")
    print(f"{'gate':28} {'layer':9} {'scope':8} {'ran':>6} {'fired':>6} "
          f"{'fire%':>7} {'advise%':>8}")
    for gate in ORPHANS:
        s = stats[gate]
        ran = s["ran"]
        if not ran:
            print(f"{gate:28} {quality.REGISTRY.repair(gate):9} "
                  f"{quality.REGISTRY.scope(gate):8} {0:>6} {'':>6} "
                  f"{'n/a':>7} {'n/a':>8}")
            continue
        print(f"{gate:28} {quality.REGISTRY.repair(gate):9} "
              f"{quality.REGISTRY.scope(gate):8} {ran:>6} {s['fired']:>6} "
              f"{s['fired'] / ran * 100:>6.1f}% {s['advised'] / ran * 100:>7.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("novels", nargs="*", default=[])
    ap.add_argument("--all", action="store_true",
                    help="include derivative novel dirs (forks/ablations)")
    ap.add_argument("--v2-window", default="ts_v2match:171-200",
                    help="NAME:FROM-TO — the chapters written by v2, reported "
                         "separately. '' disables.")
    args = ap.parse_args()

    names, dropped = discover_novels(list(args.novels), include_all=args.all)
    print_exclusions(dropped)

    window = None
    if args.v2_window:
        try:
            nm, span = args.v2_window.split(":")
            lo, hi = span.split("-")
            window = (nm, int(lo), int(hi))
        except ValueError:
            print(f"bad --v2-window: {args.v2_window!r}")
            return 2
        if window[0] not in names:
            names = names + [window[0]]

    if not names:
        print("no novels found")
        return 0
    print(f"novels: {', '.join(names)}")

    corpus, v2win, avail, probes, tone_info = scan(names, window)

    _table("=== recomputed over real chapter files (whole corpus) ===", corpus)
    if window:
        _table(f"=== v2-written prose only ({window[0]} Ch{window[1]}-{window[2]}) ===",
               v2win)

    print("\n=== threshold vs observed distribution (LESSONS §4's deletion test) ===")
    print(f"{'gate':28} {'metric':24} {'side':>5} {'n':>5} {'worst':>8} {'p95':>8} "
          f"{'line':>8} {'headroom':>11}")
    for gate, (label, _get, _thr, side) in PROBE.items():
        vals = sorted(probes["values"].get(gate) or [])
        line = probes["thresholds"].get(gate, 0.0)
        if not vals:
            print(f"{gate:28} {label:24} {side:>5} {0:>5} {'-':>8} {'-':>8} "
                  f"{line:>8.2f} {'-':>11}")
            continue
        # "worst" = the tail the gate actually fires on, and p95 is that tail's 95th
        # percentile — so both columns answer "how close does the corpus get".
        if side == "high":
            worst = vals[-1]
            p95 = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
            ratio = (line / worst) if worst > 0 else float("inf")
            head = "unreachable" if worst == 0 else f"{ratio:.1f}x"
        else:
            worst = vals[0]
            p95 = vals[max(0, int(len(vals) * 0.05))]
            ratio = (worst / line) if line > 0 else float("inf")
            head = f"{ratio:.1f}x" if worst > 0 else "reachable"
        # How far the line sits from what the corpus actually produces. LESSONS §4
        # kept a 0/641 gate at 1.1x and deleted one at 3.6x.
        print(f"{gate:28} {label:24} {side:>5} {len(vals):>5} {worst:>8.2f} "
              f"{p95:>8.2f} {line:>8.2f} {head:>11}")

    print("\n=== chapter_metrics availability: what a gate can actually read ===")
    print(f"{'column':20} {'v1-written':>14} {'v2-written':>14}")
    for col in AUDIT_COLUMNS:
        a, b = avail["v1"][col], avail["v2"][col]
        f1 = f"{a[0]}/{a[1]}" if a[1] else "-"
        f2 = f"{b[0]}/{b[1]}" if b[1] else "-"
        print(f"{col:20} {f1:>14} {f2:>14}")

    # The measurement that deleted `emotional_cadence`, kept so it does not have to
    # be rediscovered. That gate compared consecutive `emotional_tone` values for
    # EQUALITY, which only means anything if the column holds a short label. It
    # holds a free-text sentence, so the gate could not fire on ANY book, v1 or v2:
    # its 0% was a schema mismatch, not an absence of monotony. Any future gate
    # over this column has to clear this table first.
    tones = tone_info["tones"]
    if tones:
        lens = sorted(len(t) for t in tones)
        short = sum(1 for t in tones if len(t) <= 6)
        print(f"\n=== emotional_tone shape (n={len(tones)}) ===")
        print(f"  median length {lens[len(lens) // 2]} chars, max {lens[-1]}; "
              f"label-shaped (<=6 chars): {short}/{len(tones)}")
        print(f"  distinct values: {len(set(tones))}/{len(tones)} "
              f"(equality-based streaks need repeats)")

    print("\n=== gates whose verdict hangs on a column v2 leaves empty ===")
    for gate, cols in NEEDS_COLUMN.items():
        dead = [c for c in cols
                if avail["v2"][c][1] and avail["v2"][c][0] == 0]
        if dead:
            print(f"  {gate:26} DEAD INPUT: {', '.join(dead)}")
        else:
            print(f"  {gate:26} inputs available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
