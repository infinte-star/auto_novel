"""Calls/chapter for a CHAPTER RANGE, not for a whole novel directory.

`compare._llm_totals` reads the whole of `logs/llm_calls.jsonl` and divides by the
whole chapter count. That is the right answer when both arms are fresh runs of the
same length. It is the wrong answer for the matched v1/v2 settlement, twice over:

* **The arms cover different spans.** tangshuting's log is 200 chapters of
  generation; ts_v2match's is the 30 chapters v2 rewrote. Dividing each by its own
  chapter count compares v1's whole-book average against v2's Ch171-200 — and v1's
  late chapters are exactly where it was in degradation recovery, widening to 3
  candidates. The confound points the wrong way for v1, so leaving it in would
  flatter v2.
* **v1's log contains post-completion tool runs.** 400 `refine_rewrite`, 80
  `refine_diagnose`, 12 `defossil`, plus packaging — 500+ rows, ~9% of the file,
  none of it chapter generation. They ran a day after Ch200 finished.

So: bracket by time. A chapter file's mtime is when the engine finished writing it,
and for a source book that was never copied it is authoritative. The window for
Ch A..B is `(mtime(A-1), mtime(B)]`, and every call logged inside it was spent
getting from A-1 to B. The time cut removes the refine/package rows for free,
because they happened outside the window.

    python tools/window_cost.py --arm tangshuting --from 171 --to 200
    python tools/window_cost.py --arm ts_v2match  --from 171 --to 200

KNOWN IMPRECISION, stated because it is not removable: background finalization for
Ch A-1 (extract / memory_compress / stage_review) can land just inside the window,
and Ch B's lands just outside. That is one chapter's background work misattributed
at each edge, in opposite directions, out of B-A+1 chapters. It is reported as
`edge_rows` so the reader can see how large it could be rather than trusting that
it is small.

Zero LLM, read-only. Never writes into the novel it measures.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compare import OFFLINE_TOOL_TAGS  # noqa: E402

# Tags that are post-completion tooling rather than chapter generation. The time
# window already excludes them in practice; this is the belt to that braces, and
# it is what makes the tool correct if someone ever runs refine mid-book.
POST_COMPLETION_TAGS = frozenset({
    "refine_rewrite", "refine_diagnose", "defossil", "pack_review",
    "hook_package", "hook_package_score", "rebuild_memory", "glossary",
    "voices_table", "screenplay", "script_scene",
})


def _mtime(nd: Path, ch: int) -> float:
    f = nd / "chapters" / f"{ch:04d}.md"
    if not f.exists():
        raise SystemExit(f"[window] missing {f}; cannot bracket the window")
    return f.stat().st_mtime


def _chars(nd: Path, lo: int, hi: int) -> tuple[int, int]:
    """(total chars, chapters found) over the range, as the judge would read them."""
    total = found = 0
    for ch in range(lo, hi + 1):
        f = nd / "chapters" / f"{ch:04d}.md"
        if f.exists():
            total += len(f.read_text(encoding="utf-8", errors="replace").strip())
            found += 1
    return total, found


def measure(nd: Path, lo: int, hi: int) -> dict:
    start = _mtime(nd, lo - 1) if lo > 1 else 0.0
    end = _mtime(nd, hi)
    if end <= start:
        raise SystemExit(
            f"[window] Ch{hi}'s mtime is not after Ch{lo - 1}'s. The chapter files "
            f"were copied or touched, so mtime no longer records when the engine "
            f"wrote them and this tool cannot bracket the window.")

    path = nd / "logs" / "llm_calls.jsonl"
    r = {"calls": 0, "failed": 0, "excluded": 0, "post": 0, "outside": 0,
         "elapsed": 0.0, "output": 0, "prompt": 0, "edge_rows": 0,
         "tags": collections.Counter(), "start": start, "end": end}
    if not path.exists():
        return r
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = str(row.get("tag") or "")
        ts = float(row.get("ts") or 0.0)
        if not (start < ts <= end):
            r["outside"] += 1
            continue
        if tag in OFFLINE_TOOL_TAGS:
            r["excluded"] += 1
            continue
        if tag in POST_COMPLETION_TAGS:
            r["post"] += 1
            continue
        r["calls"] += 1
        r["tags"][tag] += 1
        r["elapsed"] += float(row.get("elapsed") or 0.0)
        r["output"] += int(row.get("output_chars") or 0)
        r["prompt"] += int(row.get("prompt_chars") or 0)
        if not row.get("ok", True):
            r["failed"] += 1
        # Rows within 3 minutes of either edge are the ones background
        # finalization could have misplaced.
        if ts - start < 180 or end - ts < 180:
            r["edge_rows"] += 1
    return r


def _report(name: str, nd: Path, lo: int, hi: int) -> dict:
    r = measure(nd, lo, hi)
    chars, found = _chars(nd, lo, hi)
    n = found or 1
    print(f"=== {name}  Ch{lo}-{hi} ===")
    print(f"  window            {r['start']:.0f} -> {r['end']:.0f}  "
          f"({(r['end'] - r['start']) / 3600:.1f}h wall)")
    print(f"  chapters found    {found}")
    print(f"  calls in window   {r['calls']}   ({r['calls'] / n:.2f} / chapter)"
          f"   failed {r['failed']}")
    print(f"  prompt chars      {r['prompt']:,}   ({r['prompt'] / n:,.0f} / chapter)")
    print(f"  output chars      {r['output']:,}   ({r['output'] / n:,.0f} / chapter)")
    print(f"  prose chars       {chars:,}   ({chars / n:,.0f} / chapter)")
    if r["output"]:
        print(f"  prompt:prose      {r['prompt'] / max(chars, 1):.0f}:1")
    print(f"  wall in LLM       {r['elapsed'] / 3600:.2f}h")
    print(f"  rows outside win  {r['outside']}   post-completion dropped {r['post']}"
          f"   offline-tool dropped {r['excluded']}")
    print(f"  edge_rows         {r['edge_rows']}  (within 3min of a window edge; "
          f"background finalization for Ch{lo - 1}/Ch{hi} may sit here)")
    print("  by tag:")
    for tag, c in r["tags"].most_common():
        print(f"    {tag:24} {c:5}  ({c / n:.2f}/ch)")
    print()
    r["chars"] = chars
    r["found"] = found
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="novel name; repeat to compare two arms over the same range")
    ap.add_argument("--from", dest="lo", type=int, required=True)
    ap.add_argument("--to", dest="hi", type=int, required=True)
    args = ap.parse_args()

    out = []
    for name in args.arm:
        nd = ROOT / "novels" / name
        if not nd.is_dir():
            raise SystemExit(f"[window] no such novel: {nd}")
        out.append((name, _report(name, nd, args.lo, args.hi)))

    if len(out) == 2:
        (na, ra), (nb, rb) = out
        ca = ra["calls"] / max(ra["found"], 1)
        cb = rb["calls"] / max(rb["found"], 1)
        pa = ra["prompt"] / max(ra["found"], 1)
        pb = rb["prompt"] / max(rb["found"], 1)
        print(f"=== {na} vs {nb}, Ch{args.lo}-{args.hi} ===")
        print(f"  calls/chapter   {ca:.2f} -> {cb:.2f}   "
              f"({(cb - ca) / ca * 100:+.0f}%)")
        print(f"  prompt/chapter  {pa:,.0f} -> {pb:,.0f}   "
              f"({(pb - pa) / max(pa, 1) * 100:+.0f}%)")
        if ra["found"] != rb["found"]:
            print(f"  NOTE: {ra['found']} vs {rb['found']} chapters found — the "
                  f"per-chapter figures are over different denominators.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
