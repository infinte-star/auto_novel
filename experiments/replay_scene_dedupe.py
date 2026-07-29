"""The scene-skeleton similarity distribution, and the record that settled it.

Kept, not thrown away: it is the evidence behind deleting v1's WARN / short-novel
/ 0.97-ceiling / force_retry tiers instead of re-wiring them (REDESIGN_V2 §9.11.3,
2026-07-28). Restoring a tier is only worth doing if it would ever speak, and
`scene_dedupe_sim_warn` at 0.60 sits 1.5x above the highest value 692 real
plans/cards have ever produced. Re-run this before proposing any new threshold.

Zero LLM, read-only. Recomputes `v2.beat._scene_sim` exactly (same window, same
`card_to_plan` projection, same `quality.scene_similarity`) over every archived
card AND the v1 archive's `merged_plan`s, buckets both against all four
thresholds, and probes reachability — including the two fields the metric is
blind to, which is why a 1.000 there is not a bug.

Reading as of 2026-07-28: 60 v2 cards p50 0.057 / p90 0.094 / max 0.188;
632 v1 plans p50 0.069 / p90 0.130 / max 0.393. Nothing crosses 0.60.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NOVEL_CONFIG", "config.yaml")

from engine.plan import card_to_plan  # noqa: E402
from engine.quality import scene_similarity  # noqa: E402

WINDOW = 8
WARN, BLOCK, SHORT, IDENT = 0.6, 0.82, 0.92, 0.97

rows: list[tuple[str, int, float]] = []
for f in sorted(ROOT.glob("novels/*/logs/arc_cards.json")):
    novel = f.parent.parent.name
    cards = {int(k): v for k, v in
             json.loads(f.read_text(encoding="utf-8")).get("cards", {}).items()}
    for ch in sorted(cards):
        recent = [card_to_plan(cards[n])[0]
                  for n in range(ch - WINDOW, ch) if n in cards]
        if not recent:
            continue
        sim = float(scene_similarity(card_to_plan(cards[ch])[0], recent)
                    .get("max_sim", 0.0) or 0.0)
        rows.append((novel, ch, sim))

sims = sorted(s for _, _, s in rows)
n = len(sims)
print(f"{n} cards with a comparison set (of {sum(1 for _ in rows)} scored)")
if n:
    print(f"min {sims[0]:.3f}  p50 {sims[n // 2]:.3f}  p90 {sims[int(n * .9)]:.3f}  "
          f"max {sims[-1]:.3f}")
    for label, cut in (("WARN >=0.60", WARN), ("BLOCK >=0.82", BLOCK),
                       ("short-novel >=0.92", SHORT), ("IDENTICAL >=0.97", IDENT)):
        hits = [(nv, ch, s) for nv, ch, s in rows if s >= cut]
        print(f"{label:>20}: {len(hits):>3} / {n}"
              + (f"   {', '.join(f'{nv} Ch{ch} {s:.2f}' for nv, ch, s in hits[:8])}"
                 if hits else ""))
    print("\ntop 10:")
    for nv, ch, s in sorted(rows, key=lambda r: -r[2])[:10]:
        print(f"  {s:.3f}  {nv} Ch{ch}")

# The v1 half of the library, replayed the same way. 60 v2 cards is a small corpus
# to declare a threshold dead on; the archive holds ~640 real v1 plans across five
# books, judged by the SAME function against the same thresholds.
print("\n--- v1 archive (merged_plan of attempt0) ---")
sys.path.insert(0, str(ROOT / "tools"))
from replay_gates import _plans  # noqa: E402

v1_rows: list[tuple[str, int, float]] = []
for nd in sorted((ROOT / "novels").iterdir()):
    if not (nd / "logs" / "checkpoints").is_dir():
        continue
    plans = _plans(nd)
    if not plans:
        continue
    for ch in sorted(plans):
        recent = [plans[n] for n in range(ch - WINDOW, ch) if n in plans]
        if not recent:
            continue
        v1_rows.append((nd.name, ch,
                        float(scene_similarity(plans[ch], recent)["max_sim"] or 0.0)))
if v1_rows:
    v = sorted(s for _, _, s in v1_rows)
    m = len(v)
    print(f"{m} plans  min {v[0]:.3f}  p50 {v[m // 2]:.3f}  p90 {v[int(m * .9)]:.3f}  "
          f"max {v[-1]:.3f}")
    for label, cut in (("WARN >=0.60", WARN), ("BLOCK >=0.82", BLOCK),
                       ("IDENTICAL >=0.97", IDENT)):
        hits = [(nv, ch, s) for nv, ch, s in v1_rows if s >= cut]
        print(f"{label:>20}: {len(hits):>3} / {m}"
              + (f"   {', '.join(f'{nv} Ch{ch} {s:.2f}' for nv, ch, s in hits[:6])}"
                 if hits else ""))
    print("  top 5: " + ", ".join(f"{s:.3f} {nv} Ch{ch}"
                                  for nv, ch, s in sorted(v1_rows, key=lambda r: -r[2])[:5]))

# Is 0.60 reachable at all, or unreachable by construction? A threshold no input
# can cross is a dead key whatever the corpus says, and a corpus that simply never
# repeated is a different fact from a metric that cannot express repetition.
print("\n--- reachability ---")
all_cards: dict[str, dict[int, dict]] = {}
for f in sorted(ROOT.glob("novels/*/logs/arc_cards.json")):
    all_cards[f.parent.parent.name] = {
        int(k): v for k, v in
        json.loads(f.read_text(encoding="utf-8")).get("cards", {}).items()}

for novel, cards in all_cards.items():
    if len(cards) < 2:
        continue
    plans = {ch: card_to_plan(c)[0] for ch, c in cards.items()}
    # identity: the metric's own ceiling
    ch0 = sorted(plans)[0]
    ident = scene_similarity(plans[ch0], [plans[ch0]])["max_sim"]
    # all-pairs max: a near-duplicate the 8-chapter window would never compare
    best = (0.0, 0, 0)
    for a in sorted(plans):
        for b in sorted(plans):
            if b >= a:
                continue
            s = float(scene_similarity(plans[a], [plans[b]])["max_sim"] or 0.0)
            if s > best[0]:
                best = (s, a, b)
    print(f"{novel:>12}: identity={ident:.3f}   all-pairs max="
          f"{best[0]:.3f} (Ch{best[1]} vs Ch{best[2]}, gap {best[1] - best[2]})")

# What a real "same scene, one field swapped" case scores -- the failure the gate
# was built for. If THIS lands under 0.60 the WARN tier is unreachable in practice.
if all_cards:
    novel, cards = next(iter(all_cards.items()))
    ch = sorted(cards)[-1]
    base = cards[ch]
    for label, mut in (
        ("identical card", dict(base)),
        ("same but new where", {**base, "where": "另一个完全不同的具体场地"}),
        ("same but new where+turn", {**base, "where": "另一个完全不同的具体场地",
                                     "turn": "另一个完全不同的转折物"}),
    ):
        s = scene_similarity(card_to_plan(mut)[0], [card_to_plan(base)[0]])["max_sim"]
        print(f"  {label:>26}: {s:.3f}")
