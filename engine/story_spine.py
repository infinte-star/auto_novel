"""Original-brief story spine: extraction, persistence, prompting and checks.

The generated volume plan and ChapterCard are useful plans, but neither is the
author's source of truth.  This module keeps explicitly scheduled chapter beats
from ``prompt.md`` in a separate artifact so creative expansion cannot silently
promote itself above the brief and the reviewer never has to grade a card against
itself.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from engine.config import Paths, text_bigrams, write_text


_CHAPTER_RE = re.compile(
    r"(?i)(?:\bCh|第)\s*(\d+)\s*(?:[-—~至]\s*(\d+))?\s*(?:章)?\s*[：:]?"
)

_ATOMIC_SUFFIXES = (
    "医院", "写字楼", "大厦", "学校", "公寓", "小区", "车站", "地铁", "公园",
    "仓库", "工厂", "村庄", "街道", "巷子", "办公室", "餐厅", "酒店", "码头",
    "机场", "列车", "雕像", "石像", "幽灵", "鬼婴", "死神", "医生", "警察",
    "站长", "快递员", "外卖员", "系统漏洞",
)

# These are not decorative nouns: each one changes how the scheduled chapter
# solves or reveals the story.  Percentage coverage must never let another
# anchor compensate for dropping one of these causal facts.
_CRITICAL_CONCEPTS = (
    "法医知识", "系统版本", "系统漏洞", "死亡概率", "第一代系统快递员",
)


def _literal_anchors(requirements: Iterable[str]) -> list[str]:
    """Derive conservative, literal anchors when the LLM omits useful ones."""
    source = "\n".join(str(x) for x in requirements)
    found: list[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip(" \t，,。；;：:（）()\"'“”「」『』")
        value = re.sub(
            r"^.*(?:送外卖给|送往|送给|前往|发生在|配送对象是|配送对象为|"
            r"收件对象是|收件对象为|收件人是|被迫用|发现|揭示|是)", "", value,
        ).strip()
        value = re.sub(
            r"^(?:连续两单|第一单|最终|送外卖给|送给|送往|收件人是|配送对象是|"
            r"配送对象为|收件对象是|收件对象为|一栋|在|被|发现|揭示|竟然|也)", "", value,
        ).strip()
        if 2 <= len(value) <= 12 and value in source and value not in found:
            found.append(value)

    for match in re.finditer(r"[\"“「『]([^\"”」』]{2,12})[\"”」』]", source):
        add(match.group(1))
    for match in re.finditer(r"(?:[DCBAS]级(?:任务)?|寿命\s*[+＋-]\s*\d+(?:月|年))", source):
        add(match.group(0).replace(" ", ""))
    for match in re.finditer(
        r"(?:法医知识|死亡概率|脑瘤|系统版本|起雾的眼镜|系统漏洞)", source,
    ):
        add(match.group(0))
    suffixes = "|".join(map(re.escape, sorted(_ATOMIC_SUFFIXES, key=len, reverse=True)))
    for segment in re.split(r"[\n，,、；;和及与]", source):
        for match in re.finditer(rf"[\u4e00-\u9fff]{{0,10}}(?:{suffixes})", segment):
            add(match.group(0))
    for match in re.finditer(r"[\u4e00-\u9fff]{1,5}(?:姐|哥|医生|博士|老师|父亲|母亲)", source):
        add(match.group(0))
    return found[:8]


def story_spine_path(paths: Paths) -> Path:
    return paths.contract.parent / "story_spine.json"


def story_spine_markdown_path(paths: Paths) -> Path:
    return paths.contract.parent / "story_spine.md"


def _clean_segment(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" \t-—；;，,。")
    return text[:360]


def explicit_chapter_requirements(brief: str, max_chapters: int = 0) -> dict[int, list[str]]:
    """Extract only chapter-numbered requirements literally present in the brief.

    It deliberately understands both ``Ch3：...`` and ``第3章 ...`` plus ranges
    such as ``Ch1-2``.  No semantic invention happens here: every saved sentence
    is a substring of the original line.
    """
    out: dict[int, list[str]] = {}
    for raw_line in str(brief or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matches = list(_CHAPTER_RE.finditer(line))
        if not matches:
            continue
        for i, match in enumerate(matches):
            start_ch = int(match.group(1))
            end_ch = int(match.group(2) or start_ch)
            if end_ch < start_ch or end_ch - start_ch > 30:
                continue
            seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            requirement = _clean_segment(line[match.end():seg_end])
            if len(requirement) < 3:
                continue
            if re.fullmatch(r"(?:埋设|出现|再现|推进|回应|小揭示)\s*(?:→+)?", requirement):
                continue
            for chapter in range(start_ch, end_ch + 1):
                if chapter < 1 or (max_chapters and chapter > max_chapters):
                    continue
                bucket = out.setdefault(chapter, [])
                if requirement not in bucket:
                    bucket.append(requirement)
    return out


def _structured_requirements(requirements: Iterable[str]) -> dict[str, str]:
    source = "\n".join(str(x) for x in requirements)
    grade_match = re.search(r"([DCBAS])级(?:任务)?", source, flags=re.IGNORECASE)
    recipient_match = re.search(
        r"收件人\s*(?:是|为|：|:)\s*[\"“「『]?([^\"”」』，,。；;\n]{1,20})",
        source,
    )
    recipient = recipient_match.group(1).strip() if recipient_match else ""
    recipient = re.sub(r"本身$", "", recipient).strip()
    return {
        "expected_grade": f"{grade_match.group(1).upper()}级任务" if grade_match else "",
        "named_recipient": recipient,
    }


def _grounded(value: str, source: str) -> bool:
    value = str(value or "").strip()
    source = str(source or "")
    if not value:
        return False
    if value in source:
        return True
    grams = text_bigrams(value, strip="punct")
    if not grams:
        return False
    source_grams = text_bigrams(source, strip="punct")
    return len(grams & source_grams) / len(grams) >= 0.72


def build_story_spine(
    original_brief: str,
    contract: dict[str, Any] | None = None,
    *,
    max_chapters: int = 0,
) -> dict[str, Any]:
    """Build a grounded spine from literal requirements plus LLM anchor labels.

    The LLM may help select short anchors, but it cannot add a chapter or anchor:
    both must already exist in that chapter's literal source snippets.
    """
    explicit = explicit_chapter_requirements(original_brief, max_chapters)
    llm_by_ch: dict[int, dict[str, Any]] = {}
    raw_entries = (contract or {}).get("story_spine") or []
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            try:
                chapter = int(item.get("ch", 0))
            except (TypeError, ValueError):
                continue
            if chapter in explicit:
                llm_by_ch[chapter] = item

    chapters: dict[str, dict[str, Any]] = {}
    for chapter, requirements in sorted(explicit.items()):
        source = "\n".join(requirements)
        item = llm_by_ch.get(chapter, {})
        anchors: list[str] = _literal_anchors(requirements)
        for raw in (item.get("hard_anchors") or []):
            raw_anchor = str(raw or "").strip()
            if not raw_anchor or raw_anchor not in source:
                continue
            # The model may return a whole semantic summary (e.g. "陈默摸索出系统
            # 规则"). Requiring that phrase verbatim in prose creates false
            # negatives. Keep only conservative atomic anchors derivable from it.
            for anchor in _literal_anchors([raw_anchor]):
                if anchor in source and anchor not in anchors:
                    anchors.append(anchor)
        llm_requirement = str(item.get("requirement") or "").strip()
        if (
            llm_requirement
            and _grounded(llm_requirement, source)
            and not any(_grounded(llm_requirement, existing) for existing in requirements)
        ):
            requirements = list(requirements) + [llm_requirement]
        structured = _structured_requirements(requirements)
        chapters[str(chapter)] = {
            "chapter": chapter,
            "requirements": list(dict.fromkeys(requirements)),
            "hard_anchors": anchors[:8],
            "critical_anchors": [
                anchor for anchor in anchors[:8]
                if any(concept in anchor for concept in _CRITICAL_CONCEPTS)
            ],
            **structured,
            "final": bool(max_chapters and chapter == max_chapters),
        }

    return {
        "version": 1,
        "source": "original_prompt",
        "max_chapters": int(max_chapters or 0),
        "chapters": chapters,
    }


def render_story_spine(spine: dict[str, Any], chapters: Iterable[int] | None = None) -> str:
    wanted = set(int(x) for x in chapters) if chapters is not None else None
    rows = spine.get("chapters") if isinstance(spine, dict) else {}
    if not isinstance(rows, dict) or not rows:
        return ""
    lines = ["# 故事脊柱（仅来自原始简报，不得被创意增强、卷纲或章节卡改写）"]
    for key in sorted(rows, key=lambda x: int(x) if str(x).isdigit() else 10**9):
        entry = rows.get(key)
        if not isinstance(entry, dict):
            continue
        chapter = int(entry.get("chapter", key))
        if wanted is not None and chapter not in wanted:
            continue
        suffix = "（全书终章）" if entry.get("final") else ""
        lines.append(f"\n## Ch{chapter}{suffix}")
        for req in entry.get("requirements") or []:
            if str(req).strip():
                lines.append(f"- 原始要求：{str(req).strip()}")
        anchors = [str(a).strip() for a in (entry.get("hard_anchors") or []) if str(a).strip()]
        if anchors:
            lines.append("- 必须落地的原文锚点：" + "、".join(f"「{a}」" for a in anchors))
        if entry.get("expected_grade"):
            lines.append(f"- 不可替换关系：本章必须是「{entry['expected_grade']}」。")
        if entry.get("named_recipient"):
            lines.append(
                f"- 不可替换关系：收件人必须是「{entry['named_recipient']}」；"
                "不得把其他人物、系统意志或配送物改成收件人。"
            )
        if entry.get("final"):
            lines.append(
                "- 不可替换关系：本章必须封闭收束；不得新增订单、下一任务、"
                "待定结算周期、空缺席位或必须在下一章解决的新危机。"
            )
        lines.append("- 允许扩写过程与细节；不得替换上述事件、章位、关键对象或结局功能。")
    return "\n".join(lines) if len(lines) > 1 else ""


def save_story_spine(paths: Paths, spine: dict[str, Any]) -> None:
    write_text(story_spine_path(paths), json.dumps(spine, ensure_ascii=False, indent=2) + "\n")
    rendered = render_story_spine(spine)
    if rendered:
        write_text(story_spine_markdown_path(paths), rendered + "\n")


def load_story_spine(paths: Paths) -> dict[str, Any]:
    try:
        data = json.loads(story_spine_path(paths).read_text(encoding="utf-8"))
    except (AttributeError, OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def story_spine_entry(paths: Paths, chapter: int) -> dict[str, Any] | None:
    rows = load_story_spine(paths).get("chapters") or {}
    entry = rows.get(str(chapter)) if isinstance(rows, dict) else None
    return entry if isinstance(entry, dict) else None


def story_spine_window(paths: Paths, chapters: Iterable[int]) -> str:
    return render_story_spine(load_story_spine(paths), chapters)


def _fragment_hit(fragment: str, text: str, grams: set[str]) -> bool:
    fragment = str(fragment or "").strip()
    if not fragment:
        return False
    if fragment in text:
        return True
    grade = re.search(r"[DCBAS]级", fragment)
    if grade and grade.group(0) in text:
        return True
    if fragment.endswith("知识") and fragment[:-2] and fragment[:-2] in text:
        return True
    if fragment == "住户" and "收件人" in text:
        return True
    if fragment == "系统版本" and "系统" in text and (
        "版本" in text
        or ("面板" in text and any(x in text for x in ("措辞", "描述", "因果锚定物")))
        or ("因果锚定物" in text and "配送完成" in text)
    ):
        return True
    if fragment == "已故父亲" and "父亲" in text and any(
        x in text for x in ("去世", "死于", "车祸", "遗物", "六年前", "最后一单")
    ):
        return True

    # Compound brief labels may be realized with their parts separated on the
    # page ("废弃了十五年的仁华医院", "地铁里的透明人影"). Count grounded
    # atomic concepts and a tiny alias set rather than demanding one frozen noun
    # phrase. This remains deterministic and independent of the generated card.
    modifiers = ("废弃", "闹鬼", "第一代", "已故")
    parts = [x for x in _ATOMIC_SUFFIXES if x in fragment]
    parts.extend(x for x in modifiers if x in fragment)
    parts = list(dict.fromkeys(parts))
    aliases = {
        "幽灵": ("幽灵", "鬼", "人影", "透明的手", "透明"),
        "鬼婴": ("鬼婴", "婴儿", "灰色的小手"),
        # “死亡”过宽，会让“死亡概率/死亡结算”等普通提及冒充
        # “收件人=死神”这一结构关系。这里只保留确指死神本体的表达。
        "死神": ("死神", "死亡本身"),
        "写字楼": ("写字楼", "大厦", "办公区", "办公室"),
        "闹鬼": ("闹鬼", "鬼", "冤魂", "幻象", "残影", "人影"),
    }
    if parts:
        hits = sum(
            1 for part in parts
            if any(alias in text for alias in aliases.get(part, (part,)))
        )
        if hits >= max(1, math.ceil(len(parts) * 0.6)):
            return True
    fg = text_bigrams(fragment, strip="punct")
    return bool(fg) and len(fg & grams) / len(fg) >= 0.60


def story_spine_adherence(
    entry: dict[str, Any] | None,
    text: str,
    *,
    min_anchor_coverage: float = 0.60,
) -> dict[str, Any]:
    """Check a card or chapter against the independent original-brief spine."""
    out = {
        "enabled": bool(entry), "passed": True, "chapter": 0,
        "hard_anchors": [], "matched_anchors": [], "missing_anchors": [],
        "requirements": [], "requirement_hits": [], "directives": [],
        "anchor_coverage": 1.0, "required_anchor_matches": 0,
        "relation_checks": [], "relation_failures": [],
        "critical_anchors": [], "missing_critical_anchors": [],
    }
    if not isinstance(entry, dict) or not entry:
        return out
    body = str(text or "")
    out["chapter"] = int(entry.get("chapter", 0) or 0)
    out["requirements"] = [str(x).strip() for x in (entry.get("requirements") or []) if str(x).strip()]
    anchors = [str(x).strip() for x in (entry.get("hard_anchors") or []) if str(x).strip()]
    critical = [
        str(x).strip() for x in (entry.get("critical_anchors") or []) if str(x).strip()
    ]
    if not critical:
        critical = [
            anchor for anchor in anchors
            if any(concept in anchor for concept in _CRITICAL_CONCEPTS)
        ]
    out["critical_anchors"] = critical
    out["hard_anchors"] = anchors
    grams = text_bigrams(body, strip="punct")
    for anchor in anchors:
        if _fragment_hit(anchor, body, grams):
            out["matched_anchors"].append(anchor)
        else:
            out["missing_anchors"].append(anchor)
            if anchor in critical:
                out["missing_critical_anchors"].append(anchor)
    required = max(1, math.ceil(len(anchors) * max(0.0, min(1.0, min_anchor_coverage)))) if anchors else 0
    out["required_anchor_matches"] = required
    out["anchor_coverage"] = len(out["matched_anchors"]) / len(anchors) if anchors else 1.0

    # When the extractor returned no short anchors (e.g. deterministic fallback),
    # require at least one literal scheduled requirement to have meaningful fuzzy
    # coverage.  This is deliberately weaker than the grounded-anchor path but is
    # still independent of the generated card.
    for requirement in out["requirements"]:
        if _fragment_hit(requirement, body, grams):
            out["requirement_hits"].append(requirement)
    no_fallback_evidence = bool(out["requirements"]) and not anchors and not out["requirement_hits"]
    expected_grade = str(entry.get("expected_grade") or "").strip()
    if expected_grade:
        grade_letter = expected_grade[:1]
        hit = bool(re.search(
            rf"(?:{re.escape(grade_letter)}\s*级|危险等级\s*[：:]?\s*{re.escape(grade_letter)}(?:\b|[^A-Za-z]))",
            body,
            flags=re.IGNORECASE,
        ))
        check = {"type": "grade", "expected": expected_grade, "passed": hit}
        out["relation_checks"].append(check)
        if not hit:
            out["relation_failures"].append(check)

    named_recipient = str(entry.get("named_recipient") or "").strip()
    if named_recipient:
        recipient_names = [named_recipient]
        if named_recipient == "死神":
            recipient_names.append("死亡本身")
        target = "(?:" + "|".join(re.escape(x) for x in recipient_names) + ")"
        # Require an explicit predicate, not mere proximity.  Otherwise text such
        # as “收件人是父亲，死神般的提示音……” launders the wrong recipient.
        hit = bool(
            re.search(
                rf"(?:目标)?收件人\s*(?:(?:必须|正是|就是|为|是)\s*)?[：:=]?\s*"
                rf"[\"“「『【]?\s*[^，,。；;\n]{{0,16}}?{target}\s*"
                rf"[\"”」』】）)]?(?:本人|本身)?",
                body,
            )
            or re.search(
                rf"[\"“「『【]?\s*{target}\s*[\"”」』】]?(?:本人|本身)?\s*"
                rf"(?:必须|正是|就是|为|是|[：:=])?\s*(?:目标)?收件人",
                body,
            )
            or re.search(rf"(?:由\s*)?{target}(?:本人|本身)?\s*(?:完成)?签收", body)
        )
        check = {"type": "named_recipient", "expected": named_recipient, "passed": hit}
        out["relation_checks"].append(check)
        if not hit:
            out["relation_failures"].append(check)

    if bool(entry.get("final")):
        open_patterns = (
            r"新订单|下一单|下一任务|新任务",
            r"(?:下一|下个)结算周期[^。；;\n]{0,16}(?:待定|开启|启动|倒计时)",
            r"仲裁者席位[^。；;\n]{0,20}(?:空缺|待补|需|优先)",
            r"七个工作日内",
            r"(?:未完待续|新的危机|新的敌人|新的案件)",
        )
        hits = [m.group(0) for pattern in open_patterns for m in re.finditer(pattern, body)]
        check = {
            "type": "final_closure", "expected": "五章封闭收束",
            "passed": not hits, "open_hooks": hits[:6],
        }
        out["relation_checks"].append(check)
        if hits:
            out["relation_failures"].append(check)

    out["passed"] = (
        len(out["matched_anchors"]) >= required
        and not out["missing_critical_anchors"]
        and not no_fallback_evidence
        and not out["relation_failures"]
    )
    if anchors and len(out["matched_anchors"]) < required:
        out["directives"].append(
            f"本章只落实原始简报故事脊柱锚点 {len(out['matched_anchors'])}/{len(anchors)}，"
            "必须补足关键事件证据："
            + "、".join(f"「{x}」" for x in out["missing_anchors"][:6])
            + "。不得用卷纲新增设定替换这些对象或事件。"
        )
    if out["missing_critical_anchors"]:
        out["directives"].append(
            "本章遗漏不可用其他锚点抵消的关键因果事实："
            + "、".join(f"「{x}」" for x in out["missing_critical_anchors"])
            + "。必须在正文中明确写出并让它参与破局或真相揭示。"
        )
    elif no_fallback_evidence:
        out["directives"].append(
            "本章没有落实原始简报为该章钉死的事件："
            + "；".join(out["requirements"][:2])
        )
    for failure in out["relation_failures"]:
        if failure["type"] == "named_recipient":
            out["directives"].append(
                f"原始简报钉死本章收件人必须是「{failure['expected']}」。"
                "必须在正文中明确建立“收件人=该对象”的签收关系，不得改成父亲、主角、系统意志或其他角色。"
            )
        elif failure["type"] == "grade":
            out["directives"].append(
                f"原始简报钉死本章任务等级为「{failure['expected']}」，正文必须明确落地该等级。"
            )
        elif failure["type"] == "final_closure":
            out["directives"].append(
                "原始简报钉死本章为全书终章，必须封闭收束。删除新订单、下一任务、"
                "待定结算周期、空缺席位或需后续解决的新危机；结尾只保留已完成事件的余韵。"
            )
    return out
