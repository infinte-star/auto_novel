"""Probe a role's endpoint across reasoning tiers to find the strongest one that
does not time out.

Background: every live novel runs the WRITING role — the one call whose output
actually ships — with reasoning fully off::

    writing_thinking_mode: disabled
    writing_reasoning_effort: none

The recorded reason (CLAUDE.md) is that gemini-2.5-pro behind the littlesheep
nginx gateway reasons past a ~70s proxy timeout and 504s before the first byte.
But the config jumps straight from `none` to unusable — the intermediate tiers
(`low`, `medium`, `high`, and the Anthropic-style `thinking` budget) were never
measured. This tool measures them.

It bypasses ``call_llm`` deliberately: no retries, no salvage, no fallback
endpoint rotation. We want the raw gateway behaviour, including the failures.

Usage::

    python tools/probe_reasoning.py <novel>                    # writing role
    python tools/probe_reasoning.py <novel> --role review
    python tools/probe_reasoning.py <novel> --prompt-chars 15000 63000
    python tools/probe_reasoning.py <novel> --repeat 2 --json out.json

Reports per (tier, prompt size): time-to-first-byte, total elapsed, output
chars, and the failure mode when it fails. TTFB is the number that matters —
a 504 fires before the first byte, so TTFB headroom is what tells you whether
a tier is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Realistic writer workload: a concrete chapter card plus enough surrounding
# context to reproduce production prompt sizes. Reasoning latency scales with
# both the thinking tier AND the prompt, so a toy prompt would under-report.
_SYSTEM = (
    "你是一位顶级中文网络小说写手。你的任务是根据给定的章节卡片写出一章正文。\n"
    "要求：\n"
    "1. 场景具体可拍摄，避免抽象概括和心理独白堆砌。\n"
    "2. 对白占比不低于 20%，人物说话要有各自的语气。\n"
    "3. 禁止使用「破折号——状态——状态」式的电报体短句堆叠。\n"
    "4. 章末必须留下一个具体的悬念钩子，不要用「他忽然明白了」这类顿悟收尾。\n"
    "5. 只输出正文，不要输出任何解释、标题或元信息。"
)

_CARD = """## 本章卡片
- 地点：县医院三楼旧档案室，凌晨两点，只有应急灯
- 人物：陈默、值班护士林
- 目标：拿到 1998 年那卷失踪病历
- 阻碍：档案室登记簿上，他自己的名字已经被人签过了
- 转折：签名是他的笔迹，但日期是三天后
- 兑现：回收第 19 章埋下的「笔迹」伏线
- 出场钩子：走廊尽头，护士林正在给某个人打电话，说的是他的名字
- 禁止复用：镜子意象、「他忽然明白了」式顿悟

## 任务
写第 27 章，约 3000 字。
"""

_FILLER_UNIT = (
    "## 历史片段\n"
    "陈默把手电筒的光压低，贴着铁皮柜的下沿一格格扫过去。第三排的标签已经卷边，"
    "钢笔字被潮气泡开，只剩下一个模糊的「9」。他伸手去够，指尖碰到的却不是纸，"
    "是一层薄薄的灰下面硬邦邦的塑料封套。走廊那头传来推车轮子碾过地砖的声音，"
    "一下，两下，然后停住了。他屏住呼吸，等那声音再响起来，可是没有。\n"
    "「你在找 1998 年的？」林靠在门框上，手里端着一个搪瓷缸子，热气把她的眼镜熏白了一小块。\n"
    "「登记簿呢。」陈默没有回头。\n"
    "「在你手边那个抽屉里。」她顿了顿，「不过我劝你别看。」\n\n"
)


def _build_prompt(target_chars: int) -> str:
    """Pad the card with realistic prose context up to *target_chars*."""
    body = _CARD
    while len(body) < target_chars:
        body += _FILLER_UNIT
    return body[:target_chars] + "\n\n（以上为上下文。现在开始写第 27 章正文。）"


# (label, extra_body) — the two independent reasoning knobs the gateways honour.
# `none`/`disabled` is the current production setting and serves as the baseline.
_TIERS: list[tuple[str, dict[str, Any]]] = [
    ("effort=none (current)", {"reasoning_effort": "none"}),
    ("effort=low", {"reasoning_effort": "low"}),
    ("effort=medium", {"reasoning_effort": "medium"}),
    ("effort=high", {"reasoning_effort": "high"}),
    ("thinking=disabled", {"thinking": {"type": "disabled"}}),
    ("thinking=enabled/4k", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ("thinking=enabled/16k", {"thinking": {"type": "enabled", "budget_tokens": 16384}}),
    ("(omit both)", {}),
]


def _probe_once(
    client: Any,
    model: str,
    prompt: str,
    extra_body: dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """One streaming call. Returns timing + outcome; never raises."""
    started = time.monotonic()
    ttfb: float | None = None
    out_chars = 0
    reasoning_chars = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body=extra_body or None,
        )
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            # Some gateways emit reasoning tokens on a separate field before any
            # content; count them but do NOT treat them as the first byte of the
            # answer — the 504 we care about fires before ANY byte arrives.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_chars += len(reasoning)
                if ttfb is None:
                    ttfb = time.monotonic() - started
            if piece:
                if ttfb is None:
                    ttfb = time.monotonic() - started
                out_chars += len(piece)
    except Exception as exc:  # noqa: BLE001 — the failure mode IS the result
        return {
            "ok": False,
            "ttfb": round(ttfb, 2) if ttfb else None,
            "elapsed": round(time.monotonic() - started, 2),
            "out_chars": out_chars,
            "reasoning_chars": reasoning_chars,
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
        }
    return {
        "ok": out_chars > 0,
        "ttfb": round(ttfb, 2) if ttfb else None,
        "elapsed": round(time.monotonic() - started, 2),
        "out_chars": out_chars,
        "reasoning_chars": reasoning_chars,
        # A tier that emits reasoning but no content did NOT fail the gateway —
        # it spent the whole max_tokens budget thinking. That is a distinct
        # (and highly informative) outcome from a 504.
        "error": "" if out_chars else ("budget spent on reasoning" if reasoning_chars else "empty response"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("novel", help="novel name under novels/ (supplies the config)")
    ap.add_argument("--role", default="writing", choices=("writing", "planning", "review", "extraction"))
    ap.add_argument("--prompt-chars", type=int, nargs="+", default=[15000, 63000],
                    help="prompt sizes to test (default: 15k = redesign target, 63k = current production)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per (tier, size) cell")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="cap output so the probe measures latency-to-start, not full generation")
    ap.add_argument("--tier", nargs="*", default=None, help="only probe tiers whose label contains one of these")
    ap.add_argument("--json", dest="json_out", default=None, help="also write raw results here")
    args = ap.parse_args()

    cfg_path = ROOT / "novels" / args.novel / "config.yaml"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found")
        return 2
    os.environ["NOVEL_CONFIG"] = str(cfg_path.relative_to(ROOT))

    from openai import OpenAI  # noqa: PLC0415 — must import after NOVEL_CONFIG is set

    import engine.config as cfg_mod  # noqa: PLC0415

    config = cfg_mod.load_config()
    api = config["api"]

    endpoints = cfg_mod.configured_role_endpoints(config, args.role)
    if endpoints:
        base_url, api_key = endpoints[0]
        model = api.get(f"{args.role}_model") or api["model"]
    else:
        eps, _ = cfg_mod.configured_api_endpoints(config)
        base_url, api_key = eps[0]
        model = api["model"]
        print(f"(role '{args.role}' has no dedicated endpoint — falling back to the primary)")

    headers = {"User-Agent": api["user_agent"]} if api.get("user_agent") else None
    # Generous client timeout: we want the GATEWAY's 504, not our own abort.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0, default_headers=headers)

    tiers = _TIERS
    if args.tier:
        tiers = [t for t in _TIERS if any(f.lower() in t[0].lower() for f in args.tier)]

    print(f"Novel: {args.novel}   Role: {args.role}")
    print(f"Endpoint: {base_url}")
    print(f"Model: {model}   temperature={api.get('temperature')}   probe max_tokens={args.max_tokens}")
    print("=" * 92)
    print(f"{'tier':<24} {'prompt':>8} {'TTFB':>8} {'total':>8} {'out':>7} {'think':>7}  result")
    print("-" * 92)

    results: list[dict[str, Any]] = []
    for size in args.prompt_chars:
        prompt = _build_prompt(size)
        for label, extra in tiers:
            for _ in range(args.repeat):
                r = _probe_once(
                    client, model, prompt, extra,
                    max_tokens=args.max_tokens,
                    temperature=float(api.get("temperature", 0.8)),
                )
                r |= {"tier": label, "prompt_chars": size, "model": model, "base_url": base_url}
                results.append(r)
                ttfb = f"{r['ttfb']:.1f}s" if r["ttfb"] else "—"
                verdict = "OK" if r["ok"] else f"FAIL {r['error']}"
                # Each cell can take minutes; flush so a backgrounded probe is
                # watchable instead of dumping everything at exit.
                print(
                    f"{label:<24} {size // 1000:>7}k {ttfb:>8} {r['elapsed']:>7.1f}s "
                    f"{r['out_chars']:>7} {r['reasoning_chars']:>7}  {verdict}",
                    flush=True,
                )
                # Persist incrementally too — a probe killed mid-ladder still
                # leaves usable data.
                if args.json_out:
                    Path(args.json_out).write_text(
                        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

    print("-" * 92)
    ok = [r for r in results if r["ok"]]
    if ok:
        best = max(ok, key=lambda r: _TIERS.index(next(t for t in _TIERS if t[0] == r["tier"])))
        print(f"Strongest tier that returned output: {best['tier']} "
              f"(TTFB {best['ttfb']}s at {best['prompt_chars'] // 1000}k prompt)")
    else:
        print("No tier returned output — check keys/endpoint before drawing conclusions.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Raw results -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
