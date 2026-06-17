"""Scene-breakdown middle layer (分场细纲中间层).

Inserted between plan selection and chapter writing. Takes the selected plan
(beats / conflict / payoff / goal) and asks the LLM to decompose the chapter
into an ordered list of concrete scenes, each with goal / location /
visible_actions / beats_covered / exit_state. The rendered block is injected
into the writer's variable section (carryover_block) so the writer has a
shot list rather than re-deriving scene structure from the abstract plan.

Design constraints (mirroring the rest of the engine):
- Strictly optional. Gated by `novel.scene_breakdown_enabled` (default True).
- Failure-tolerant: any error returns {} and the writer falls back to the plan.
- Resumable: the breakdown is checkpointed (scene_breakdown.json) by the caller.
- Does NOT touch cacheable_prefix — the block rides in the variable carryover.
"""
from __future__ import annotations

import json
from typing import Any

from config import Paths, log
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix


SCENE_BREAKDOWN_SYSTEM = """你是中文网文的分场细纲师。
给你本章的选定大纲(含 beats/冲突/兑现/目标)，请把这一章拆成有序的【场景清单】，
让作者照着它就能直接动笔，而不必再从抽象大纲里自行推断场景结构。

只返回恰好一个合法 JSON 对象，不要输出其它任何内容：
{
  "scenes": [
    {
      "goal": "<本场景的叙事目的，一句话>",
      "location": "<地点/场所>",
      "time": "<时段或时间关系，可留空>",
      "characters": ["在场人物"],
      "visible_actions": ["必须在页面上实演的可见动作/对白交锋(2-5条，具体到动作与物件)"],
      "beats_covered": ["本场景落地的大纲 beat(引用大纲原文片段)"],
      "exit_state": "<场景结束时的局面变化/新信息/情绪落点——下一场景的起点>"
    }
  ],
  "carryover_to_next_chapter": "<本章结束后留给下一章的悬念/未决张力，一句话>"
}

要求：
- 场景按时间/因果顺序排列，3-6 个为宜；每个场景都要有明确的状态推进(exit_state 必须和上一场景不同)。
- visible_actions 必须是可见的动作/对白，不能是"交代背景""渲染气氛"这类抽象说明。
- beats_covered 合起来必须覆盖大纲所有 beats，不得遗漏核心兑现 beat。
- 不要写正文，不要扩写成段落，只给结构化的场景骨架。"""


def build_scene_breakdown(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    cached_memory: str | None = None,
) -> dict[str, Any]:
    """Produce an ordered scene breakdown for the chapter.

    Returns a dict like {"scenes": [...], "carryover_to_next_chapter": "..."}.
    Returns {} on any failure or when disabled, so the caller can safely fall
    back to the plan-only writer prompt.
    """
    if not bool(config["novel"].get("scene_breakdown_enabled", True)):
        return {}
    if not isinstance(plan, dict) or not plan:
        return {}
    try:
        constraints = decision.get("required_constraints", []) if isinstance(decision, dict) else []
        user = f"""## 选定大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 仲裁约束JSON
{json.dumps(constraints, ensure_ascii=False, indent=2)}

把第 {chapter_num} 章拆成有序场景清单。"""
        prefix = cacheable_prefix(paths, config)
        max_tokens = int(config["novel"].get("scene_breakdown_max_tokens", 4000) or 4000)
        raw = call_llm(
            client, paths, config, SCENE_BREAKDOWN_SYSTEM, json_prompt(user),
            max_tokens=max_tokens, temperature=0.4, cacheable_prefix=prefix,
            tag="scene_breakdown",
        )
        data = load_json_with_repair(client, paths, config, raw, fallback={})
        if not isinstance(data, dict):
            return {}
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            return {}
        # Keep only well-formed scene dicts.
        clean: list[dict[str, Any]] = []
        for sc in scenes:
            if isinstance(sc, dict) and (sc.get("goal") or sc.get("visible_actions")):
                clean.append(sc)
        if not clean:
            return {}
        data["scenes"] = clean
        log(paths, f"Scene breakdown Ch{chapter_num}: {len(clean)} scene(s)")
        return data
    except Exception as exc:
        log(paths, f"Scene breakdown failed (non-fatal) Ch{chapter_num}: {exc}")
        return {}


def scene_breakdown_block(breakdown: dict[str, Any], chapter_num: int) -> str:
    """Render the scene breakdown as a writer-prompt injection block.

    Returns "" when there is nothing to render, so callers can append
    unconditionally.
    """
    if not isinstance(breakdown, dict):
        return ""
    scenes = breakdown.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return ""
    lines: list[str] = [
        f"\n## 本章分场细纲（CH{chapter_num} 的拍摄清单，按顺序逐场实演）",
        "以下是本章的有序场景骨架。请逐场写出来：每一场都要把 visible_actions 落到页面上的"
        "可见动作/对白，达成该场的 exit_state 再进入下一场；不要跳过场景，也不要把多场压成一句总结。",
    ]
    for i, sc in enumerate(scenes, 1):
        if not isinstance(sc, dict):
            continue
        goal = str(sc.get("goal", "")).strip()
        loc = str(sc.get("location", "")).strip()
        tm = str(sc.get("time", "")).strip()
        chars = sc.get("characters") or []
        actions = sc.get("visible_actions") or []
        beats = sc.get("beats_covered") or []
        exit_state = str(sc.get("exit_state", "")).strip()
        head = f"\n### 场景{i}"
        if loc:
            head += f" · {loc}"
        if tm:
            head += f"（{tm}）"
        lines.append(head)
        if goal:
            lines.append(f"- 目的：{goal}")
        if chars:
            lines.append("- 在场人物：" + "、".join(str(c) for c in chars if str(c).strip()))
        if actions:
            lines.append("- 必演动作：")
            for a in actions:
                if str(a).strip():
                    lines.append(f"  • {str(a).strip()}")
        if beats:
            lines.append("- 落地 beat：" + "；".join(str(b).strip() for b in beats if str(b).strip()))
        if exit_state:
            lines.append(f"- 退出状态：{exit_state}")
    carry = str(breakdown.get("carryover_to_next_chapter", "")).strip()
    if carry:
        lines.append(f"\n（章末留给下一章的悬念：{carry}）")
    return "\n".join(lines) + "\n"
