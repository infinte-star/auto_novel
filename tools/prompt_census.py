"""Prompt census: where the context budget actually goes, by call type.

Zero LLM calls, read-only. Aggregates every `novels/*/logs/llm_calls.jsonl` row
by its `tag` and reports call count, median/p90 prompt size, and each tag's share
of total prompt characters.

This exists because "the writer's context is too big" is the kind of claim that
feels true and is cheap to check. Measured over the library (14,976 calls,
991M prompt chars) it is not where the budget goes:

    plan_candidate    33.7%   median 131,872 chars
    write             17.0%   median  81,152 chars
    review            15.5%   median  75,624 chars
    plan_review_fused 10.8%
    plan_arbitrate     9.2%

Planning is 53.7% of all prompt volume and `plan_candidate`'s prompt is *larger*
than the writer's. Any context-slimming work should be aimed there first; see
REDESIGN.md §7 (P3).

    python tools/prompt_census.py [--novel NAME] [--top N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", default="*")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    by_tag: dict[str, list[int]] = defaultdict(list)
    calls = 0
    for path in sorted((ROOT / "novels").glob(f"{args.novel}/logs/llm_calls.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            by_tag[str(row.get("tag") or "(untagged)")].append(
                int(row.get("prompt_chars", 0) or 0)
            )
            calls += 1

    if not calls:
        print("no llm_calls.jsonl rows found")
        return 0

    rows = sorted(by_tag.items(), key=lambda kv: -sum(kv[1]))
    grand = sum(sum(v) for _, v in rows) or 1

    print(f"{calls:,} calls across {len(rows)} tags\n")
    print(f"{'tag':26} {'n':>6} {'med_prompt':>11} {'p90':>9} "
          f"{'total_prompt':>15} {'share':>7}")
    for tag, sizes in rows[: args.top]:
        s = sorted(sizes)
        p90 = s[min(int(len(s) * 0.9), len(s) - 1)]
        print(f"{tag:26} {len(s):6} {statistics.median(s):11,.0f} {p90:9,.0f} "
              f"{sum(s):15,} {sum(s) / grand * 100:6.1f}%")

    plan_share = sum(
        sum(v) for t, v in rows if t.startswith("plan_")
    ) / grand * 100
    write_share = sum(sum(v) for t, v in rows if t == "write") / grand * 100
    print(f"\ngrand total prompt chars: {grand:,}")
    print(f"planning tags: {plan_share:.1f}%   write: {write_share:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
