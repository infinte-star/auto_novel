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
from retention import weighted_aggregate
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

# P1 — genre/channel persona weighting. The 5 PERSONAS stay FIXED (telemetry
# comparability), but the AGGREGATE drop_rate/excitement is weighted by how much
# each persona represents the target channel's readers. A male-channel action
# novel shouldn't let the 女频视角 persona's 80% drop dominate the retention
# signal the gate acts on. Raw (unweighted) aggregates are still stored for
# telemetry comparability; the weighted ones drive directives + the P2 gate.
# Weights are relative (~0.3–1.5); a persona absent from a profile defaults 1.0.
PERSONA_WEIGHTS: dict[str, dict[str, float]] = {
    "_male_action": {"爽点党": 1.4, "弃书敏感型": 1.3, "付费意愿型": 1.2, "逻辑党": 0.8, "女频视角": 0.4},
    "romance_female": {"女频视角": 1.5, "付费意愿型": 1.2, "弃书敏感型": 1.0, "爽点党": 0.7, "逻辑党": 0.7},
    "suspense": {"逻辑党": 1.4, "弃书敏感型": 1.2, "付费意愿型": 1.0, "爽点党": 0.7, "女频视角": 0.7},
    "history": {"逻辑党": 1.4, "弃书敏感型": 1.0, "女频视角": 0.9, "付费意愿型": 0.9, "爽点党": 0.6},
}
# style_preset → weight profile. Male-channel action/爽 presets share one profile.
_PRESET_PROFILE = {
    "xuanhuan_shuang": "_male_action",
    "system_stream": "_male_action",
    "urban_ability": "_male_action",
    "wanzu_xuanhuan": "_male_action",
    "romance_female": "romance_female",
    "suspense": "suspense",
    "history": "history",
}


def _persona_weights(config: dict[str, Any]) -> dict[str, float] | None:
    """Resolve persona weights for the configured genre, or None for uniform.

    Off (uniform) when reader_panel_persona_weighting is false. An explicit
    reader_panel_persona_profile overrides the style_preset→profile mapping.
    Returns None when no profile matches so aggregation stays plain-mean.
    """
    novel_cfg = config.get("novel", {})
    if not bool(novel_cfg.get("reader_panel_persona_weighting", True)):
        return None
    profile = str(novel_cfg.get("reader_panel_persona_profile", "") or "").strip()
    if not profile:
        preset = str(novel_cfg.get("style_preset", "") or "").strip().lower()
        profile = _PRESET_PROFILE.get(preset, "")
    return PERSONA_WEIGHTS.get(profile)


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

    # P1: genre-weighted aggregate. Stored alongside the raw values (raw kept for
    # telemetry comparability across books); the weighted values drive the
    # directive threshold below and the P2 review gate. Uniform weights → weighted
    # == raw, so this is a safe no-op when weighting is disabled/unmatched.
    weights = _persona_weights(config)
    wagg = weighted_aggregate(results, weights)
    report["weighted_drop_rate"] = wagg["drop_rate"]
    report["weighted_pay_rate"] = wagg["pay_rate"]
    report["weighted_excitement"] = wagg["avg_excitement"]
    report["persona_weighted"] = bool(weights)

    # Effective values the thresholds/gate act on: weighted when available.
    eff_drop = report["weighted_drop_rate"] if report["weighted_drop_rate"] is not None else report["drop_rate"]
    eff_exc = report["weighted_excitement"] if report["weighted_excitement"] is not None else report["avg_excitement"]
    log(
        paths,
        f"Reader panel Ch{chapter_num}: drop_rate={report['drop_rate']:.0%} "
        f"pay_rate={report['pay_rate']:.0%} excitement={report['avg_excitement']}/10"
        + (f" | weighted drop={eff_drop:.0%} exc={eff_exc}/10 [{('%s' % (weights and 'genre' or 'uniform'))}]"
           if weights else "")
        + f" ({n}/{len(PERSONAS)} personas)",
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
    # 下沉/免费流读者更没耐心，弃书阈值下调（不动 5 个固定 persona——保 telemetry 可比性，
    # 只调阈值与挽回指令措辞），让免费流的"留存生死线"更早触发挽回。
    _plat = str(novel_cfg.get("platform_preset", "")).strip().lower()
    _low_barrier = (
        _plat in {"fanqie_free", "qimao_free"}
        or bool(novel_cfg.get("style_low_barrier_register", False))
    )
    if _low_barrier:
        threshold = float(novel_cfg.get("reader_panel_drop_threshold_sinking", 0.35))
        hard_threshold = float(novel_cfg.get("reader_panel_hard_drop_sinking", 0.55))
    else:
        threshold = float(novel_cfg.get("reader_panel_drop_threshold", 0.4))
        hard_threshold = float(novel_cfg.get("reader_panel_hard_drop", 0.6))
    if eff_drop >= threshold:
        try:
            existing = load_checkpoint(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                wd = list(existing.get("writer_directives_for_next_chapter") or [])
                reasons = "；".join(report["drop_reasons"][:3]) or "多名模拟读者弃书"
                if eff_drop >= hard_threshold:
                    if _low_barrier:
                        msg = (
                            f"【紧急·下沉留存】{eff_drop:.0%} 的免费读者弃书（{reasons}）。"
                            f"下一章必须：①开头100字内有正在发生的冲突；②对话占比50%+、大白话；"
                            f"③给出具体可见的强爽点/情感冲击；④章末留强钩子。通勤5分钟内必须抓住读者。"
                        )
                    else:
                        msg = (
                            f"【紧急】读者面板严重警告：{eff_drop:.0%} 的模拟读者弃书"
                            f"（{reasons}）。下一章必须做出根本性调整：加入强爽点/情感冲击/"
                            f"关键揭示来挽回读者。不要延续本章的节奏和模式。"
                        )
                else:
                    msg = (
                        f"读者面板警示：{eff_drop:.0%} 的模拟读者在本章弃书"
                        f"（{reasons}）。下一章必须针对性挽回追读"
                        + ("（下沉读者无耐心，开头即给冲突、对话优先、爽点前置）。" if _low_barrier else "。")
                    )
                if msg not in wd:
                    wd.append(msg)
                existing["writer_directives_for_next_chapter"] = wd[:12]
                save_checkpoint(paths, chapter_num, "final_review.json", existing)
                log(paths, f"Reader panel Ch{chapter_num}: drop_rate >= {threshold:.0%}, directive injected")
        except Exception as exc:
            log(paths, f"Reader panel directive injection failed (non-fatal) Ch{chapter_num}: {exc}")

    # Hard drop alert: persist a panel_alert.json so next chapter's planning
    # can read it and force structural adjustments.
    if eff_drop >= hard_threshold:
        try:
            alert = {
                "chapter": chapter_num,
                "drop_rate": eff_drop,
                "raw_drop_rate": report["drop_rate"],
                "pay_rate": report["pay_rate"],
                "avg_excitement": eff_exc,
                "drop_reasons": report["drop_reasons"][:5],
                "worst_moments": report["worst_moments"][:5],
                "severity": "critical" if eff_drop >= 0.8 else "high",
            }
            save_checkpoint(paths, chapter_num, "panel_alert.json", alert)
            log(paths, f"Reader panel Ch{chapter_num}: HARD DROP ({eff_drop:.0%}), "
                f"panel_alert.json saved for next chapter planning")
        except Exception as exc:
            log(paths, f"Reader panel alert save failed (non-fatal) Ch{chapter_num}: {exc}")

    return report
