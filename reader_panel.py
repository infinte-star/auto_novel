"""Simulated reader panel — multi-persona chapter judgement.

cold_reader_review (review.py) is ONE independent stranger-reader; this module
is a PANEL of fixed reader personas that each decide "would I keep reading?"
for a chapter. It approximates the real-reader retention signal the pipeline
otherwise never sees: per-persona continue/drop verdicts aggregate into a
drop_rate / pay_rate that gets persisted to the book's own store AND the
global telemetry repository, and — when the drop_rate crosses a threshold —
injected as a corrective writer directive for the next chapter (same channel
cold_reader uses).

Hard rules inherited from cold_reader:
  * NEVER pass cacheable_prefix. The panel's entire value is judging the prose
    as strangers; sharing the (possibly drifted) book context would let it
    ratify the same degeneration the main reviewer over-rates.
  * Entirely non-fatal: every failure is swallowed and logged; the generation
    pipeline must never stall because a persona call failed.

Gated by `novel.reader_panel_enabled` (default false), frequency
`novel.reader_panel_every` (default 5), directive threshold
`novel.reader_panel_drop_threshold` (default 0.4).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import telemetry
from checkpoint import load_checkpoint, save_checkpoint
from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair
from store import db_event

# Each persona is (key, system prompt). They are deliberately FIXED (not
# config-driven): a stable panel makes drop_rate comparable across chapters
# and across books in the telemetry repository.
PERSONAS: list[tuple[str, str]] = [
    (
        "爽点党",
        "你是一名只追求爽感的网文读者：升级、打脸、扮猪吃虎、即时反馈。"
        "你没有耐心看铺垫超过半章；如果这一章没有让你「爽到」或者明确预告下一章会爽，你就弃书。",
    ),
    (
        "逻辑党",
        "你是一名逻辑严苛的读者：人物动机要成立、世界规则要自洽、情节因果要闭合。"
        "降智推动剧情、前后矛盾、为冲突而冲突，都会让你立刻弃书。",
    ),
    (
        "弃书敏感型",
        "你是一名极易弃书的快节奏读者：开头三段抓不住你就划走。"
        "大段环境描写、慢吞吞的对话、看不出本章要解决什么问题，都是你弃书的直接理由。"
        "你对「这一章读完了但什么都没发生」零容忍。",
    ),
    (
        "女频视角",
        "你是一名注重人物关系与情感张力的读者：角色之间的化学反应、情感递进、立体的配角是你追读的理由。"
        "工具人化的角色、纯事件流水账、没有情感颗粒度的章节会让你失去兴趣。",
    ),
    (
        "付费意愿型",
        "你是一名精打细算的付费读者：每一章都要值回票价。"
        "你判断的唯一标准是：读完这一章，你是否愿意花钱解锁下一章？"
        "信息量稀薄、注水拉长、把一个场景切碎成好几章的行为会让你拒绝付费并弃书。",
    ),
]

_PERSONA_USER_TEMPLATE = """## 这一章的全文（你对本书前文一无所知，只看这一章）
{chapter_text}

请以你的读者人设诚实判断。只返回恰好一个合法的 JSON 对象：
{{
  "continue_reading": true,        // 你是否会继续读下一章
  "drop_reason": "<=60字，若弃书说明原因；继续读则留空字符串>",
  "excitement_1_10": 1-10,         // 这一章带给你的兴奋度
  "would_pay": false,              // 你是否愿意为下一章付费
  "worst_moment": "<=60字，本章最让你出戏/想划走的一处>"
}}"""


def _ask_persona(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    persona_key: str,
    persona_system: str,
    chapter_text: str,
) -> dict[str, Any] | None:
    system = (
        persona_system
        + "\n你没有读过本书的任何前文，也不知道作者意图——只评这一章的文字本身。诚实、果断。"
    )
    user = _PERSONA_USER_TEMPLATE.format(chapter_text=chapter_text[:12000])
    raw = call_llm(
        client, paths, config, system, json_prompt(user),
        max_tokens=1500, temperature=0.4,  # NOTE: deliberately no cacheable_prefix
        tag="reader_panel",
    )
    data = load_json_with_repair(client, paths, config, raw, fallback=None)
    if not isinstance(data, dict):
        return None
    return {
        "persona": persona_key,
        "continue_reading": bool(data.get("continue_reading", True)),
        "drop_reason": str(data.get("drop_reason", ""))[:120],
        "excitement": safe_score(data.get("excitement_1_10", 5)),
        "would_pay": bool(data.get("would_pay", False)),
        "worst_moment": str(data.get("worst_moment", ""))[:120],
    }


def run_reader_panel(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter_text: str,
) -> dict[str, Any] | None:
    """Run the full persona panel against one chapter. Non-fatal by design."""
    novel_cfg = config.get("novel", {})
    max_workers = max(1, min(len(PERSONAS), int(novel_cfg.get("max_parallel_workers", 3) or 3)))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ask_persona, client, paths, config, key, system, chapter_text): key
            for key, system in PERSONAS
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                log(paths, f"Reader panel persona {key!r} failed (non-fatal) Ch{chapter_num}: {exc}")
                row = None
            if row:
                results.append(row)

    if not results:
        log(paths, f"Reader panel Ch{chapter_num}: all personas failed; skipping")
        return None

    n = len(results)
    drops = [r for r in results if not r["continue_reading"]]
    report: dict[str, Any] = {
        "chapter": chapter_num,
        "panel_size": n,
        "drop_rate": round(len(drops) / n, 3),
        "pay_rate": round(sum(1 for r in results if r["would_pay"]) / n, 3),
        "avg_excitement": round(sum(r["excitement"] for r in results) / n, 2),
        "drop_reasons": [f"{r['persona']}：{r['drop_reason']}" for r in drops if r["drop_reason"]][:5],
        "worst_moments": [f"{r['persona']}：{r['worst_moment']}" for r in results if r["worst_moment"]][:5],
        "per_persona": results,
    }
    log(
        paths,
        f"Reader panel Ch{chapter_num}: drop_rate={report['drop_rate']:.0%} "
        f"pay_rate={report['pay_rate']:.0%} excitement={report['avg_excitement']}/10 "
        f"({n}/{len(PERSONAS)} personas)",
    )

    # Persist: book-local event + global telemetry (both non-fatal).
    try:
        db_event(conn, chapter_num, "panel_report", report)
    except Exception as exc:
        log(paths, f"Reader panel db_event failed (non-fatal) Ch{chapter_num}: {exc}")
    try:
        novel_name = paths.logs_dir.parent.name
        genre = str(novel_cfg.get("genre", "_default") or "_default")
        telemetry.record_event(novel_name, genre, chapter_num, "panel_report", report)
    except Exception:
        pass

    # Corrective feedback loop: when too many personas drop, push a directive
    # into this chapter's final_review so the NEXT chapter's writer reacts —
    # the exact channel cold_reader uses (pipeline.py _do_cold_reader).
    threshold = float(novel_cfg.get("reader_panel_drop_threshold", 0.4))
    if report["drop_rate"] >= threshold:
        try:
            existing = load_checkpoint(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                wd = list(existing.get("writer_directives_for_next_chapter") or [])
                reasons = "；".join(report["drop_reasons"][:3]) or "多名模拟读者弃书"
                msg = (
                    f"读者面板警示：{report['drop_rate']:.0%} 的模拟读者在本章弃书"
                    f"（{reasons}）。下一章必须针对性挽回追读。"
                )
                if msg not in wd:
                    wd.append(msg)
                existing["writer_directives_for_next_chapter"] = wd[:12]
                save_checkpoint(paths, chapter_num, "final_review.json", existing)
                log(paths, f"Reader panel Ch{chapter_num}: drop_rate >= {threshold:.0%}, directive injected")
        except Exception as exc:
            log(paths, f"Reader panel directive injection failed (non-fatal) Ch{chapter_num}: {exc}")

    return report
