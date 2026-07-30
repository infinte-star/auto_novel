"""Dimensional blind evaluation of generated chapters against bestseller benchmarks.

Extends anchor.py's pairwise infrastructure with per-dimension scoring:
instead of a single winner verdict, each (chapter, anchor) pair is scored
across 8 quality dimensions. The gap report surfaces the top weaknesses.

    python tools/benchmark_eval.py <novel_name> [--chapters N] [--genre G]

Exit code 0 on success, 2 if anchors or chapters are missing.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.anchor import (
    DIMENSIONS,
    DimensionalVerdict,
    JUDGE_SYSTEM_DIMENSIONAL,
    JUDGE_USER,
    MAX_CHAPTER_CHARS,
    ORDERS,
    SIDE_FIRST,
    SIDE_SECOND,
    UNMEASURED,
    AnchorText,
    CallFn,
    anchor_chapters,
    anchor_fingerprint,
    judge_paths,
    llm_caller,
)

PRESET_GENRE_MAP: dict[str, tuple[str, ...]] = {
    "urban_ability": ("dushi", "yineng", "chaoneng"),
    "rule_horror": ("lingyi", "guize"),
    "xianxia": ("xianxia", "xuanhuan", "wuxianliu"),
    "romance_female": ("yanqing", "zhaidou", "gongdou"),
    "historical": ("lishi", "zhengzhi"),
    "scifi": ("kehuan", "kesu"),
}


def _parse_dimensional(raw: str, sides: tuple[str, str]) -> tuple[dict, str, dict, str]:
    text = str(raw or "")
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return {}, UNMEASURED, {}, "unparseable dimensional output"
    if not isinstance(obj, dict):
        return {}, UNMEASURED, {}, "unparseable dimensional output"

    dims_raw = obj.get("dimensions", {})
    dims: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for key, _ in DIMENSIONS:
        entry = dims_raw.get(key, {})
        if not isinstance(entry, dict):
            dims[key] = UNMEASURED
            reasons[key] = ""
            continue
        w = str(entry.get("winner", "")).strip()
        if w.startswith(SIDE_FIRST):
            dims[key] = sides[0]
        elif w.startswith(SIDE_SECOND):
            dims[key] = sides[1]
        elif not w:
            dims[key] = UNMEASURED
        else:
            dims[key] = "tie"
        reasons[key] = str(entry.get("note", "")).strip()

    overall_raw = obj.get("overall", {})
    ow = str(overall_raw.get("winner", "")).strip() if isinstance(overall_raw, dict) else ""
    if ow.startswith(SIDE_FIRST):
        overall = sides[0]
    elif ow.startswith(SIDE_SECOND):
        overall = sides[1]
    elif not ow:
        overall = UNMEASURED
    else:
        overall = "tie"
    overall_reason = str(overall_raw.get("reason", "")).strip() if isinstance(overall_raw, dict) else ""
    return dims, overall, reasons, overall_reason


def dimensional_judge_pair(
    text_a: str,
    text_b: str,
    *,
    call: CallFn,
    key: str = "",
    system: str = JUDGE_SYSTEM_DIMENSIONAL,
) -> DimensionalVerdict:
    a = str(text_a or "").strip()[:MAX_CHAPTER_CHARS]
    b = str(text_b or "").strip()[:MAX_CHAPTER_CHARS]
    if not a or not b:
        empty = {k: UNMEASURED for k, _ in DIMENSIONS}
        return DimensionalVerdict(key, empty, UNMEASURED, {}, "", False)

    all_dims: list[dict[str, str]] = []
    all_overalls: list[str] = []
    all_reasons: list[dict[str, str]] = []
    all_overall_reasons: list[str] = []

    for sides in ORDERS:
        first, second = (a, b) if sides[0] == "a" else (b, a)
        try:
            raw = call(system, JUDGE_USER.format(first=first, second=second))
        except Exception:
            all_dims.append({k: UNMEASURED for k, _ in DIMENSIONS})
            all_overalls.append(UNMEASURED)
            all_reasons.append({})
            all_overall_reasons.append("judge call failed")
            continue
        dims, overall, reasons, overall_reason = _parse_dimensional(raw, sides)
        all_dims.append(dims)
        all_overalls.append(overall)
        all_reasons.append(reasons)
        all_overall_reasons.append(overall_reason)

    merged_dims: dict[str, str] = {}
    merged_reasons: dict[str, str] = {}
    concordant = True
    for dim_key, _ in DIMENSIONS:
        p0 = all_dims[0].get(dim_key, UNMEASURED)
        p1 = all_dims[1].get(dim_key, UNMEASURED)
        if UNMEASURED in (p0, p1):
            merged_dims[dim_key] = UNMEASURED
            concordant = False
        elif p0 == p1:
            merged_dims[dim_key] = p0
        else:
            merged_dims[dim_key] = "tie"
        merged_reasons[dim_key] = all_reasons[0].get(dim_key, "") or all_reasons[1].get(dim_key, "")

    if UNMEASURED in all_overalls:
        merged_overall = UNMEASURED
        concordant = False
    elif all_overalls[0] == all_overalls[1]:
        merged_overall = all_overalls[0]
    else:
        merged_overall = "tie"

    return DimensionalVerdict(
        key=key,
        dims=merged_dims,
        overall=merged_overall,
        reasons=merged_reasons,
        overall_reason=all_overall_reasons[0] or all_overall_reasons[1],
        concordant=concordant,
    )


@dataclasses.dataclass
class DimTally:
    wins_chapter: int = 0
    wins_anchor: int = 0
    ties: int = 0
    unmeasured: int = 0

    @property
    def n(self) -> int:
        return self.wins_chapter + self.wins_anchor + self.ties

    @property
    def chapter_rate(self) -> float:
        n = self.n
        return ((self.wins_chapter + self.ties * 0.5) / n * 100.0) if n else 0.0

    @property
    def anchor_rate(self) -> float:
        n = self.n
        return ((self.wins_anchor + self.ties * 0.5) / n * 100.0) if n else 0.0


@dataclasses.dataclass
class BenchmarkReport:
    dim_tallies: dict[str, DimTally]
    overall_wins_chapter: int
    overall_wins_anchor: int
    overall_ties: int
    verdicts: list[DimensionalVerdict]
    anchor_fingerprint: str
    n_chapters: int
    n_anchors: int

    @property
    def n(self) -> int:
        return self.overall_wins_chapter + self.overall_wins_anchor + self.overall_ties

    @property
    def overall_chapter_rate(self) -> float:
        n = self.n
        return ((self.overall_wins_chapter + self.overall_ties * 0.5) / n * 100.0) if n else 0.0

    def weakest_dims(self, top_n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(
            ((k, t.chapter_rate) for k, t in self.dim_tallies.items() if t.n > 0),
            key=lambda x: x[1],
        )
        return ranked[:top_n]


DEFAULT_WORKERS = 8
DEFAULT_MAX_ANCHORS = 5


def _select_anchors(
    anchors: list[AnchorText],
    style_preset: str,
    max_anchors: int,
) -> list[AnchorText]:
    if max_anchors <= 0 or len(anchors) <= max_anchors:
        return anchors
    preferred_genres = PRESET_GENRE_MAP.get(style_preset, ())
    same = [a for a in anchors if a.genre in preferred_genres]
    other = [a for a in anchors if a.genre not in preferred_genres]
    if len(same) >= max_anchors:
        return same[:max_anchors]
    return same + other[: max_anchors - len(same)]


def benchmark_eval(
    chapters: Sequence[tuple[str, str]],
    *,
    call: CallFn,
    config: dict | None = None,
    root: Path | None = None,
    genre_filter: str | None = None,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    on_verdict=None,
    workers: int = DEFAULT_WORKERS,
) -> BenchmarkReport | dict:
    anchors, why = anchor_chapters(config, root)
    if not anchors:
        return {"available": False, "reason": why, "n": 0}

    if genre_filter:
        genre_lower = genre_filter.lower()
        filtered = [a for a in anchors if genre_lower in a.name.lower()]
        if filtered:
            anchors = filtered

    style_preset = ""
    if isinstance(config, dict):
        style_preset = str((config.get("novel") or {}).get("style_preset", ""))
    anchors = _select_anchors(anchors, style_preset, max_anchors)

    pairs = [(ch_key, ch_text, anchor)
             for ch_key, ch_text in chapters
             for anchor in anchors]

    tallies: dict[str, DimTally] = {k: DimTally() for k, _ in DIMENSIONS}
    overall_a = overall_b = overall_tie = 0
    verdicts: list[DimensionalVerdict] = []
    lock = threading.Lock()

    def _judge_one(item):
        ch_key, ch_text, anchor = item
        pair_key = f"{ch_key}~{anchor.name}"
        return dimensional_judge_pair(ch_text, anchor.text, call=call, key=pair_key)

    effective = max(1, min(workers, len(pairs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective) as pool:
        for v in pool.map(_judge_one, pairs):
            verdicts.append(v)
            if on_verdict:
                on_verdict(v)

            for dim_key, _ in DIMENSIONS:
                d = v.dims.get(dim_key, UNMEASURED)
                t = tallies[dim_key]
                if d == "a":
                    t.wins_chapter += 1
                elif d == "b":
                    t.wins_anchor += 1
                elif d == "tie":
                    t.ties += 1
                else:
                    t.unmeasured += 1

            if v.overall == "a":
                overall_a += 1
            elif v.overall == "b":
                overall_b += 1
            elif v.overall == "tie":
                overall_tie += 1

    return BenchmarkReport(
        dim_tallies=tallies,
        overall_wins_chapter=overall_a,
        overall_wins_anchor=overall_b,
        overall_ties=overall_tie,
        verdicts=verdicts,
        anchor_fingerprint=anchor_fingerprint(anchors),
        n_chapters=len(chapters),
        n_anchors=len(anchors),
    )


def gap_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  BENCHMARK EVALUATION — 维度化爆款对标报告")
    lines.append("=" * 60)
    lines.append(f"  生成章节数: {report.n_chapters}  |  对标样本数: {report.n_anchors}")
    lines.append(f"  总对比轮数: {report.n}  |  anchor fingerprint: {report.anchor_fingerprint}")
    lines.append("")

    lines.append(f"  Overall WR (generated vs benchmark): {report.overall_chapter_rate:.1f}%")
    lines.append(f"    wins: {report.overall_wins_chapter}  losses: {report.overall_wins_anchor}"
                 f"  ties: {report.overall_ties}")
    lines.append("")

    lines.append("  Per-dimension breakdown:")
    lines.append(f"  {'Dimension':<18} {'WR%':>6} {'W':>4} {'L':>4} {'T':>4} {'?':>4}")
    lines.append("  " + "-" * 44)
    for key, _ in DIMENSIONS:
        t = report.dim_tallies[key]
        lines.append(f"  {key:<18} {t.chapter_rate:>5.1f}% {t.wins_chapter:>4}"
                     f" {t.wins_anchor:>4} {t.ties:>4} {t.unmeasured:>4}")

    lines.append("")
    weakest = report.weakest_dims(3)
    if weakest:
        lines.append("  TOP-3 短板维度 (最需改善):")
        for i, (dim, rate) in enumerate(weakest, 1):
            desc = dict(DIMENSIONS).get(dim, "")
            short_desc = desc.split("：")[0] if "：" in desc else desc
            lines.append(f"    {i}. {dim} ({rate:.1f}%) — {short_desc}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / programmatic entry
# ---------------------------------------------------------------------------
def _resolve_endpoint(config: dict, name: str) -> tuple[str, str] | None:
    result = _search_api_section(config.get("api", {}), name)
    if result:
        return result
    tmpl = ROOT / "config_template.yaml"
    if tmpl.exists():
        try:
            kv: dict[str, str] = {}
            for line in tmpl.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("#") or ":" not in line:
                    continue
                k, _, v = line.partition(":")
                kv[k.strip()] = v.strip().strip('"').strip("'")
            result = _search_api_section(kv, name)
            if result:
                return result
        except Exception:
            pass
    return None


def _search_api_section(api: dict, name: str) -> tuple[str, str] | None:
    nl = name.lower()
    for prefix in ("review", "writing", "extraction", "planning", "refine"):
        url = str(api.get(f"{prefix}_base_url", "")).strip()
        key = str(api.get(f"{prefix}_api_key", "")).strip()
        if url and nl in url.lower() and key:
            return url, key
    base = str(api.get("base_url", "")).strip()
    bkey = str(api.get("api_key", "")).strip()
    if base and nl in base.lower() and bkey:
        return base, bkey
    groups = str(api.get("api_key_groups", "")).strip()
    if groups:
        for group in groups.split(";"):
            parts = group.strip().split("|")
            if len(parts) >= 2 and nl in parts[0].lower():
                return parts[0], parts[1].split(",")[0]
    return None


def _load_novel_chapters(name: str, n: int) -> list[tuple[str, str]]:
    novel_dir = ROOT / "novels" / name
    chapters_dir = novel_dir / "chapters"
    if not chapters_dir.is_dir():
        print(f"error: {chapters_dir} does not exist", file=sys.stderr)
        return []

    files = sorted(chapters_dir.glob("*.md"), key=lambda p: p.name)
    if not files:
        print(f"error: no chapters in {chapters_dir}", file=sys.stderr)
        return []

    selected = files[-n:] if n < len(files) else files
    out: list[tuple[str, str]] = []
    for f in selected:
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            out.append((f.stem, text))
    return out


def run_eval(name: str, *, chapters: int = 10, genre: str | None = None,
             endpoint: str | None = None, fallback: str | None = None,
             max_anchors: int = DEFAULT_MAX_ANCHORS,
             workers: int = DEFAULT_WORKERS) -> int:
    ch = _load_novel_chapters(name, chapters)
    if not ch:
        print("error: no chapters loaded", file=sys.stderr)
        return 2

    novel_dir = ROOT / "novels" / name
    config_path = novel_dir / "config.yaml"
    if not config_path.exists():
        print(f"error: {config_path} not found", file=sys.stderr)
        return 2

    os.environ["NOVEL_CONFIG"] = str(config_path)
    prompt_path = novel_dir / "prompt.md"
    if prompt_path.exists():
        os.environ["NOVEL_PROMPT"] = str(prompt_path)

    from engine.config import load_config, get_paths
    config = load_config()
    paths = get_paths(config)

    if endpoint:
        ep_map = _resolve_endpoint(config, endpoint)
        if ep_map:
            config["api"]["base_url"] = ep_map[0]
            config["api"]["api_key"] = ep_map[1]
            config["api"]["api_keys"] = ""
            config["api"]["api_key_groups"] = ""
            print(f"Using endpoint: {endpoint} ({ep_map[0]})", flush=True)
        else:
            print(f"warning: endpoint '{endpoint}' not recognized, using default",
                  file=sys.stderr)

    if fallback:
        parts = [p.strip() for p in fallback.split(",")]
        if len(parts) >= 2:
            fb_url, fb_key = parts[0], parts[1]
            fb_model = parts[2] if len(parts) >= 3 else ""
            primary_url = config["api"].get("base_url", "")
            primary_key = config["api"].get("api_key", "")
            primary_model = config["api"].get("model", "")
            group_primary = f"{primary_url}|{primary_key}|{primary_model}"
            group_fallback = f"{fb_url}|{fb_key}" + (f"|{fb_model}" if fb_model else "")
            config["api"]["api_key_groups"] = f"{group_primary};{group_fallback}"
            config["api"]["api_keys"] = ""
            print(f"Fallback endpoint: {fb_url}", flush=True)

    from engine.llm import build_client
    client = build_client(config, paths)

    jp = judge_paths(paths)
    call = llm_caller(client, jp, config, tag="benchmark_eval")

    def on_verdict(v: DimensionalVerdict):
        status = "+" if v.concordant else "~"
        print(f"  {status} {v.key}: overall={v.overall}", flush=True)

    print(f"Evaluating {len(ch)} chapters against benchmarks "
          f"(max_anchors={max_anchors}, {workers} workers)...", flush=True)
    result = benchmark_eval(ch, call=call, config=config, genre_filter=genre,
                            max_anchors=max_anchors,
                            on_verdict=on_verdict, workers=workers)

    if isinstance(result, dict):
        print(f"\n{result.get('reason', 'no anchors available')}", file=sys.stderr)
        return 2

    print()
    print(gap_report(result))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dimensional benchmark evaluation")
    parser.add_argument("name", help="novel name")
    parser.add_argument("--chapters", type=int, default=10,
                        help="how many recent chapters to evaluate (default 10)")
    parser.add_argument("--genre", default=None,
                        help="filter anchors by genre substring")
    parser.add_argument("--endpoint", default=None,
                        help="use a named endpoint from config (substring match on base_url)")
    parser.add_argument("--max-anchors", type=int, default=DEFAULT_MAX_ANCHORS,
                        help=f"max anchor samples per chapter (default {DEFAULT_MAX_ANCHORS})")
    parser.add_argument("--fast", action="store_true",
                        help="fast mode: --max-anchors 3")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"concurrent LLM workers (default {DEFAULT_WORKERS})")
    parser.add_argument("--fallback", default=None,
                        help="fallback endpoint: 'base_url,api_key[,model]'")
    args = parser.parse_args()
    ma = 3 if args.fast else args.max_anchors
    return run_eval(args.name, chapters=args.chapters, genre=args.genre,
                    endpoint=args.endpoint, fallback=args.fallback,
                    max_anchors=ma, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
