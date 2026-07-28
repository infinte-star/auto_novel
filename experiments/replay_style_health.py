"""Offline calibration harness: replay style_health over a finished novel.

Zero-LLM. Loads the novel's own config (genre profile applies), walks
chapters/*.md in order, reconstructs the rolling em/num history from the
replay sequence itself (no DB dependency), and prints per-chapter metrics,
penalty and flags — so new deterministic checks can be tuned against real
collapse-positive and healthy chapters.

Usage (use the venv python, system python lacks sqlite3 for config imports):
  python experiments/replay_style_health.py huangliang
  python experiments/replay_style_health.py huangliang 3 20 72 96
  python experiments/replay_style_health.py verify_v7 --summary
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# This harness deliberately owns NO penalty arithmetic of its own: it feeds
# `em_history`/`tech_history` and prints whatever `style_health` charges. An
# earlier version carried a local flag→penalty table to break out the
# "new-check contribution" of the anti-overwriting anchor; that experiment
# concluded (thresholds are now in config.py, cited at config.py:85) and the
# table was a second copy of numbers only `quality.py` gets to define. The
# `pen` column plus the full flag list already say everything it said.


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--summary"]
    summary = "--summary" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    name = args[0]
    only = {int(a) for a in args[1:]} if len(args) > 1 else None

    cfg_path = ROOT / "novels" / name / "config.yaml"
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}")
        return 2
    # config.py reads NOVEL_CONFIG at import time — set BEFORE importing.
    os.environ["NOVEL_CONFIG"] = str(cfg_path.relative_to(ROOT))
    os.environ.setdefault("NOVEL_PROMPT", str((ROOT / "novels" / name / "prompt.md").relative_to(ROOT)))
    sys.path.insert(0, str(ROOT))
    import config as config_mod  # noqa: E402
    from quality import style_health  # noqa: E402

    config = config_mod.load_config()
    ncfg = config.get("novel", {})
    window = int(ncfg.get("style_em_dash_trend_window", 5))
    # The block line is this novel's own, not a literal: a genre whose config
    # raises `style_penalty_block` would otherwise be reported as blocked at a
    # threshold the engine never applies to it.
    block = float(ncfg.get("style_penalty_block", 2.0))

    chap_dir = ROOT / "novels" / name / "chapters"
    files = sorted(chap_dir.glob("*.md"))
    if not files:
        print(f"no chapters under {chap_dir}")
        return 2

    print(f"novel={name}  style_preset={ncfg.get('style_preset', '?')}  "
          f"max_avg={ncfg.get('style_max_avg_sentence_chars', '?')}  "
          f"dlg_min={ncfg.get('style_dialogue_ratio_min', '?')}  "
          f"jargon_warn/bad={ncfg.get('style_tech_jargon_per_kchar_warn', '?')}/"
          f"{ncfg.get('style_tech_jargon_per_kchar_bad', '?')}")
    hdr = (f"{'ch':>4} {'chars':>6} {'avg_sent':>8} {'dlg%':>6} {'tech/k':>6} "
           f"{'em/k':>5} {'pen':>4}  flags")
    print(hdr)
    print("-" * len(hdr))

    em_hist: list[float] = []
    tech_hist: list[float] = []
    rows = []
    for fp in files:
        m = re.match(r"0*(\d+)", fp.stem)
        ch = int(m.group(1)) if m else -1
        text = fp.read_text(encoding="utf-8")
        sh = style_health(
            text, config,
            em_history=em_hist[-window:] or None,
            tech_history=tech_hist[-window:] or None,
        )
        met = sh["metrics"]
        em_hist.append(float(met.get("em_dash_per_kchar", 0.0)))
        tech_hist.append(float(met.get("tech_per_kchar", 0.0)))
        rows.append((ch, met, sh))
        if only is not None and ch not in only:
            continue
        print(f"{ch:>4} {met.get('chars', len(text)):>6} "
              f"{met.get('avg_sentence_chars', 0):>8.1f} "
              f"{met.get('dialogue_char_ratio', 0) * 100:>5.1f}% "
              f"{met.get('tech_per_kchar', 0):>6.1f} "
              f"{met.get('em_dash_per_kchar', 0):>5.1f} "
              f"{sh['penalty']:>4.1f}  "
              f"{'; '.join(sh['flags'])}")

    if summary:
        print()
        print("== per-10-chapter bands ==")
        for lo in range(1, max(r[0] for r in rows) + 1, 10):
            band = [r for r in rows if lo <= r[0] < lo + 10]
            if not band:
                continue
            pens = [r[2]["penalty"] for r in band]
            blocked = sum(1 for p in pens if p >= block)
            print(f"Ch{lo:>3}-{lo + 9:<3}  n={len(band):>2}  "
                  f"avg_pen={sum(pens) / len(pens):>4.2f}  "
                  f"blocked(≥{block:g})={blocked}/{len(band)}")
        flagged = sum(1 for r in rows if r[2]["flags"])
        print(f"\nchapters with any style flag: {flagged}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
