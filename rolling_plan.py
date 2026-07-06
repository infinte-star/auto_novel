"""Rolling planning: arc-based progressive expansion of volume_plan.md.

Instead of a static one-shot volume plan, the book is divided into arcs
(~15-20 chapters each). At each arc boundary:
  1. Summarize the completed arc (characters, outcomes, open threads)
  2. Expand the next arc with a detailed per-chapter outline
  3. Update compass.md (end-goal direction + active long-threads)
  4. Rewrite volume_plan.md (compressed past arcs + detailed current arc)

The system is gated by `rolling_plan_enabled: true` and runs as a background
task after chapter finalization, so it never blocks the critical path.
"""
from __future__ import annotations

import json
from typing import Any

from config import Paths, log, read_text, write_text


ARC_SUMMARY_SYSTEM = """\
你是一位小说结构分析师。请总结已完成的弧（arc）的内容。

输出 JSON：
```json
{
  "arc_title": "弧标题（6-12字）",
  "summary": "弧的核心剧情总结（200-400字）",
  "key_outcomes": "弧结束时的关键状态变化（角色成长/关系转折/世界观揭示），列表",
  "unresolved_threads": "弧结束时仍未解决的悬念/伏线，列表",
  "character_states": "弧结束时主要角色的状态快照"
}
```
## 强制 JSON 输出格式
只输出 JSON。"""

ARC_EXPAND_SYSTEM = """\
你是一位资深小说架构师。请为下一个弧（arc）生成详细的章节级大纲。

你会收到：
- 全书终极方向（compass）
- 前一弧的总结（包括未解决的线索）
- 当前角色状态和活跃伏线
- 弧编号和预计起止章节

请输出该弧的详细卷纲，格式与 volume_plan.md 一致：
```
## 第N弧：弧名（第X-Y章）

### 弧主线
（本弧核心矛盾与目标）

### 各章规划
- 第X章：标题 — 事件概要 / 核心冲突 / 推进的伏线
- 第X+1章：...

### 阶段高潮
（本弧高潮点，标注章号）

### 大事件锚点
（关键事件的章号+描述）

### 本弧兑现
（承诺在本弧兑现的悬念/伏笔）

### 遗留危机
（本弧结束后留给下弧的钩子）
```
不要输出 JSON，直接输出 markdown 格式的卷纲段落。"""

COMPASS_UPDATE_SYSTEM = """\
你是小说导航仪。根据已完成弧的总结，更新全书导航罗盘（compass）。

导航罗盘包含：
1. **终极目标** — 全书最终要达成的核心目标（1-2句，不要改变除非剧情揭示了新的终极冲突）
2. **活跃长线** — 当前跨弧的长线伏笔/悬念（列表，标注引入弧号）
3. **已完成弧概要** — 每弧一句话总结（列表）
4. **下弧方向** — 下一弧应该推进的核心方向（2-3句）
5. **规模预估** — 预计还需多少弧/章完成全书

输出 markdown。不要输出 JSON。"""


def detect_arc_boundary(
    conn: Any, config: dict[str, Any], chapter_num: int
) -> dict[str, Any] | None:
    """Check if chapter_num crosses the current arc's end boundary.
    Returns arc info dict if at boundary, None otherwise.
    """
    from store import get_current_arc

    novel_cfg = config.get("novel", {})
    arc_size = int(novel_cfg.get("rolling_plan_arc_size", 15))
    arc_size_max = int(novel_cfg.get("rolling_plan_arc_size_max", 25))

    current_arc = get_current_arc(conn)
    if current_arc is None:
        return None

    start = int(current_arc["start_chapter"])
    arc_len = chapter_num - start + 1

    if arc_len < arc_size:
        return None

    if arc_len >= arc_size:
        return {
            "arc_number": int(current_arc["arc_number"]),
            "start_chapter": start,
            "boundary_chapter": chapter_num,
            "arc_len": arc_len,
            "next_arc_number": int(current_arc["arc_number"]) + 1,
        }
    return None


def summarize_completed_arc(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    arc_info: dict[str, Any],
    chapter_num: int,
) -> dict[str, Any]:
    """LLM call: compress the completed arc into a retrospective summary."""
    from llm import call_llm, load_json_with_repair
    from memory import memory_context
    from store import recent_events

    arc_num = arc_info["arc_number"]
    start = arc_info["start_chapter"]

    events = recent_events(conn, limit=50)
    arc_events = [e for e in events if start <= e.get("chapter", 0) <= chapter_num]

    mem = memory_context(paths, conn, config)

    user = f"""请总结第 {arc_num} 弧（第 {start}-{chapter_num} 章）的内容。

## 当前记忆状态
{mem[:8000]}

## 本弧事件记录
{json.dumps(arc_events[:30], ensure_ascii=False, indent=2)[:6000]}

总结这 {chapter_num - start + 1} 章的核心剧情发展、角色变化和未解决线索。"""

    raw = call_llm(client, paths, config, ARC_SUMMARY_SYSTEM, user,
                   temperature=0.3, tag="plan_candidate")
    result = load_json_with_repair(client, paths, config, raw, fallback={
        "arc_title": f"第{arc_num}弧",
        "summary": f"第{start}-{chapter_num}章内容",
        "key_outcomes": "",
        "unresolved_threads": "",
        "character_states": "",
    })
    return result


def expand_next_arc(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    arc_info: dict[str, Any],
    compass_text: str,
    arc_summary: dict[str, Any],
    chapter_num: int,
) -> str:
    """LLM call: generate detailed outline for the next arc."""
    from llm import call_llm
    from memory import memory_context
    from store import get_open_threads, get_pending_revelations

    novel_cfg = config.get("novel", {})
    arc_size = int(novel_cfg.get("rolling_plan_arc_size", 15))
    next_arc_num = arc_info["next_arc_number"]
    next_start = chapter_num + 1
    next_end_est = next_start + arc_size - 1

    max_chapters = int(novel_cfg.get("max_chapters", 0) or 0)
    if max_chapters and next_end_est > max_chapters:
        next_end_est = max_chapters

    threads = get_open_threads(conn, chapter_num, limit=12)
    revelations = get_pending_revelations(conn, chapter_num, limit=8)
    mem = memory_context(paths, conn, config)

    # P3: retention-driven escalation. Read the just-completed arc's reader-panel
    # retention; if the arc under-delivered (low excitement / high drop), the
    # next arc's outline is FORCED to escalate structurally rather than continue
    # the plateau — this is the arc-level lever the per-chapter directives lack,
    # and directly targets the mid-book sag (excitement craters that逐章微调救不动).
    escalation_block = ""
    if bool(novel_cfg.get("rolling_plan_retention_escalate", True)):
        try:
            from retention import summarize_retention
            from store import panel_series
            series = panel_series(conn, arc_info["start_chapter"], chapter_num)
            if series:
                summ = summarize_retention(
                    [s[1] for s in series], [s[2] for s in series]
                )
                exc_floor = float(novel_cfg.get("rolling_plan_escalate_excitement", 5.0))
                drop_ceil = float(novel_cfg.get("rolling_plan_escalate_drop", 0.5))
                me = summ.get("mean_excitement")
                md = summ.get("mean_drop")
                sagging = (me is not None and me < exc_floor) or (md is not None and md > drop_ceil)
                if sagging:
                    escalation_block = (
                        f"\n## ⚠ 上弧留存告警（必须据此升级下弧）\n"
                        f"上弧读者面板：兴奋度均值 {me}/10、弃书率 {md:.0%}、"
                        f"低谷章 {summ.get('trough_count')} 个、留存指数 {summ.get('retention_index')}/10。"
                        f"这说明上弧节奏塌陷、读者在流失。下弧大纲**必须**做出至少两项结构性升级，"
                        f"不得延续上弧的调查/铺垫节奏：\n"
                        f"1. 引入更高层级的对手或威胁（新反派/幕后升级/时间压力），把冲突拉到新台阶；\n"
                        f"2. 制造一次不可逆的代价或损失（关系/能力/身份/资源的永久改变），提高赌注；\n"
                        f"3. 前置一个强爽点/强反转到本弧前3章，立刻兑现留住读者；\n"
                        f"4. 每3章至少一个当章兑现的高潮，避免连续铺垫。\n"
                    )
                    log(paths, f"Rolling plan: arc {arc_info['arc_number']} retention sag "
                        f"(exc={me}, drop={md}); forcing escalation in next arc")
        except Exception as exc:
            log(paths, f"Rolling plan retention read failed (non-fatal): {exc}")

    user = f"""请为第 {next_arc_num} 弧（预计第 {next_start}-{next_end_est} 章）生成详细的章节级大纲。
{escalation_block}
## 全书导航罗盘
{compass_text[:3000] if compass_text else "（首弧，无罗盘）"}

## 上弧总结
{json.dumps(arc_summary, ensure_ascii=False, indent=2)[:3000]}

## 当前活跃伏线
{json.dumps(threads[:8], ensure_ascii=False, indent=2)[:2000]}

## 待揭示信息
{json.dumps(revelations[:6], ensure_ascii=False, indent=2)[:1500]}

## 当前记忆
{mem[:6000]}

请生成第 {next_start} 到第 {next_end_est} 章的详细规划。"""

    raw = call_llm(client, paths, config, ARC_EXPAND_SYSTEM, user,
                   temperature=0.5, tag="plan_candidate")
    return raw.strip()


def update_compass(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    arc_summary: dict[str, Any],
    chapter_num: int,
) -> str:
    """Update compass.md with arc completion data. Returns new compass text."""
    from llm import call_llm
    from store import get_arc_summaries

    current_compass = read_text(paths.compass)
    past_arcs = get_arc_summaries(conn, limit=20)

    past_arcs_text = "\n".join(
        f"- 弧{a['arc_number']}（第{a['start_chapter']}-{a.get('end_chapter', '?')}章）: {(a.get('summary') or '')[:100]}"
        for a in reversed(past_arcs)
    )

    user = f"""请更新全书导航罗盘。

## 当前罗盘
{current_compass[:3000] if current_compass else "（初始状态，无罗盘）"}

## 刚完成弧的总结
{json.dumps(arc_summary, ensure_ascii=False, indent=2)[:3000]}

## 所有已完成弧概要
{past_arcs_text}

请输出更新后的完整罗盘（markdown）。"""

    raw = call_llm(client, paths, config, COMPASS_UPDATE_SYSTEM, user,
                   temperature=0.3, tag="plan_candidate")
    compass_text = raw.strip()
    write_text(paths.compass, compass_text + "\n")
    log(paths, f"Updated compass.md ({len(compass_text)} chars)")
    return compass_text


def update_volume_plan(
    paths: Paths,
    arc_summary: dict[str, Any],
    next_arc_outline: str,
    arc_info: dict[str, Any],
) -> None:
    """Rewrite volume_plan.md: compressed past arcs + detailed next arc."""
    from store import get_arc_summaries

    current = read_text(paths.volume_plan)
    bak_path = paths.volume_plan.parent / (paths.volume_plan.name + ".bak")
    write_text(bak_path, current)

    summary_text = arc_summary.get("summary", "")
    arc_num = arc_info["arc_number"]
    next_num = arc_info["next_arc_number"]

    completed_header = f"## 第{arc_num}弧（已完成）\n{summary_text}\n"

    new_plan = ""
    if current.strip():
        lines = current.split("\n")
        kept_lines = []
        for line in lines:
            if line.startswith(f"## 第{arc_num}弧") or line.startswith(f"## 第{next_num}弧"):
                break
            kept_lines.append(line)
        new_plan = "\n".join(kept_lines).strip() + "\n\n"

    new_plan += completed_header + "\n" + next_arc_outline + "\n"

    if len(new_plan) < 200:
        log(paths, f"Rolling plan: new volume_plan too short ({len(new_plan)} chars), keeping original")
        return

    write_text(paths.volume_plan, new_plan)
    log(paths, f"Updated volume_plan.md ({len(new_plan)} chars)")


def seed_first_arc(conn: Any, config: dict[str, Any]) -> None:
    """Create the initial arc_history record when rolling plan is enabled mid-run."""
    from store import get_current_arc, start_arc

    if get_current_arc(conn) is not None:
        return
    start_arc(conn, arc_number=1, start_chapter=1, arc_title="初始弧")


def maybe_roll_plan(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> bool:
    """Top-level entry point for rolling planning. Returns True if the plan was rolled."""
    from store import complete_arc, start_arc

    novel_cfg = config.get("novel", {})
    if not bool(novel_cfg.get("rolling_plan_enabled", False)):
        return False

    seed_first_arc(conn, config)

    arc_info = detect_arc_boundary(conn, config, chapter_num)
    if arc_info is None:
        return False

    log(paths, f"Rolling plan: arc boundary at Ch{chapter_num} (arc {arc_info['arc_number']})")

    try:
        summary = summarize_completed_arc(client, paths, conn, config, arc_info, chapter_num)
        log(paths, f"Rolling plan: arc {arc_info['arc_number']} summarized: {summary.get('arc_title', '?')}")

        compass_text = update_compass(client, paths, conn, config, summary, chapter_num)

        next_arc_outline = expand_next_arc(
            client, paths, conn, config, arc_info, compass_text, summary, chapter_num
        )
        log(paths, f"Rolling plan: next arc outline generated ({len(next_arc_outline)} chars)")

        update_volume_plan(paths, summary, next_arc_outline, arc_info)

        complete_arc(
            conn,
            arc_info["arc_number"],
            end_chapter=chapter_num,
            summary=summary.get("summary", ""),
            key_outcomes=json.dumps(summary.get("key_outcomes", ""), ensure_ascii=False),
        )

        next_start = chapter_num + 1
        start_arc(conn, arc_info["next_arc_number"], next_start,
                  arc_title=summary.get("arc_title", f"第{arc_info['next_arc_number']}弧"))

        log(paths, f"Rolling plan: arc {arc_info['next_arc_number']} started at Ch{next_start}")
        return True

    except Exception as exc:
        log(paths, f"Rolling plan failed (non-fatal) at Ch{chapter_num}: {exc}")
        return False
