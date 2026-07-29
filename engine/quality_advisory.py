"""Advisory quality gates — not in the acceptance loop, never block.

Extracted from engine.quality (which re-exports them for backward compat).
Each gate returns a dict with metrics, flags, directives, and sometimes penalty;
the directives flow into the next chapter's writer prompt via acceptance_report.
"""
from __future__ import annotations

import re
from typing import Any

from engine.quality import (
    REGISTRY,
    _clause_segments,
    _normalize_clause,
    _strip_title_line,
)


# --- Advisory gates (not in acceptance loop, available for tools/tests) ---

_AI_CLICHE_PATTERNS = re.compile(
    # --- 情绪/微表情套话 ---
    r"心中一沉|心头一震|心中涌起|心底涌起|心中升起|心头涌上|"
    r"眼中闪过一丝|眼底闪过|目光闪烁|目光一凝|目光微凝|瞳孔一缩|瞳孔微缩|"
    r"嘴角微微上扬|嘴角勾起一抹|嘴角划过一丝|嘴角不自觉|嘴角微扬|"
    r"眉头微皱|眉头一皱|眉头紧锁|眉头微蹙|"
    r"倒吸一口凉气|倒吸一口冷气|浑身一颤|浑身一震|"
    r"心如刀绞|如释重负|心中五味杂陈|百感交集|"
    r"一股暖流|一股寒意|一阵恶寒|"
    # --- 动作套话 ---
    r"缓缓开口|缓缓说道|缓缓站起|缓缓走|缓缓闭上|缓缓睁开|"
    r"微微颔首|微微点头|轻轻点头|轻轻摇头|"
    r"目光如炬|目光灼灼|目光深邃|"
    r"身形一闪|身形一顿|脚步一顿|"
    r"负手而立|双拳紧握|双手紧握|攥紧了拳头|"
    r"深吸一口气|深深吸了一口气|长舒一口气|"
    # --- 叙述腔套话 ---
    r"一时间|此刻|这一刻|那一刻|一瞬间|刹那间|霎时间|"
    r"毫无疑问|不言而喻|众所周知|不出所料|"
    r"显然|事实上|实际上|说实话|不得不说|"
    r"仿佛|恍若|犹如|宛如|好似|一如"
)

_STALE_METAPHORS = re.compile(
    r"时间仿佛静止|时间好像静止|时间似乎凝固|"
    r"心如刀绞|心如刀割|"
    r"美得像画|美如画卷|"
    r"如同一记重锤|像一记重锤|仿佛重锤|"
    r"像是被抽空了|仿佛被抽空|"
    r"仿佛被钉在原地|像是被钉在|如同钉在|"
    r"如潮水般涌来|像潮水一样|如潮水般|"
    r"打翻了五味瓶|五味杂陈|"
    r"像是被泼了一盆冷水|如同一盆冷水|"
    r"仿佛过了一个世纪|像过了一个世纪|"
    r"命运的齿轮|历史的车轮|时代的洪流|"
    r"像是做了一场梦|如同一场梦"
)

_SIMILE_PATTERNS = re.compile(
    r"仿佛|犹如|宛如|恍若|好似|好像|一如|如同|像是|似乎|"
    r"般地|一般地|似的"
)

_TELL_NOT_SHOW = re.compile(
    r"[他她][感觉到了?|感到了?|知道|明白|清楚|意识到|觉得|心想|暗想|内心深处]"
    r".{0,6}"
    r"[震惊|愤怒|悲伤|恐惧|绝望|兴奋|激动|紧张|不安|焦虑|"
    r"开心|高兴|难过|痛苦|愤恨|沮丧|失落|孤独|恐慌|惊恐|"
    r"害怕|担忧|忧虑|欣慰|释然|无奈|茫然|困惑|惊讶|诧异]"
)

_DEGREE_ADVERBS = re.compile(
    r"非常|极其|十分|无比|格外|异常|万分|分外|"
    r"极为|极度|无限|莫大|至极|之极"
)

_NEGATIVE_PAIR = re.compile(
    r"没有.{1,15}[，,].{0,4}也没有|"
    r"不是.{1,15}[，,].{0,4}(?:也不是|更不是)|"
    r"不曾.{1,10}[，,].{0,4}也不曾|"
    r"无.{1,10}[，,].{0,4}(?:也无|亦无)|"
    r"并非.{1,10}[，,].{0,4}(?:也并非|更非)|"
    r"既不.{1,10}[，,].{0,4}也不"
)

_SUMMARY_NARRATION = re.compile(
    r"就这样|一切才刚刚开始|这只是.{0,2}开始|从此以后|自此|"
    r"一切都变了|一切都不同了|一切都已经|"
    r"命运的齿轮.{0,4}转动|故事远没有结束|新的篇章|"
    r"序幕才刚刚拉开|帷幕.{0,4}拉开|画上了句号|"
    r"一切尘埃落定|一个新的时代|历史的转折点|"
    r"冥冥之中|或许这就是|也许这就是|所谓的命运"
)

_TEMPLATE_PIVOT = re.compile(
    r"不是.{1,20}?[，,].{0,4}而是|"
    r"与其说.{1,20}?[，,].{0,6}不如说|"
    r"不仅仅?是?.{1,20}?[，,].{0,4}(?:而是|更是|还是|更)|"
    r"与其.{1,15}?[，,].{0,4}不如|"
    r"不止.{1,15}?[，,].{0,4}(?:还|更)|"
    r"这不是.{1,20}?[，,].{0,4}这(?:才)?是"
)

def _anaphora_runs(body: str, min_run: int = 3) -> list[int]:
    """机械排比检测：返回所有"≥min_run 个连续子句共享同一个 2 字句首"的连段长度。

    捕捉三连及以上的同头句（如"他想起…，想起…，想起…" / "是A，是B，是C"）——这是 AI
    最爱的机械排比指纹。跳过过短子句与引号开头子句以避开对白误报。
    """
    parts = re.split(r"[，,。！!？?；;\n]", body)
    _subj = re.compile(r"^(?:他们|她们|它们|我们|你们|咱们|他|她|它|我|你|咱)")
    prefixes: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 4 or p[0] in "「」“”\"'（(【《":
            prefixes.append("")  # 断开连段，防对白/列表误报
            continue
        core = _subj.sub("", p)  # 剥去句首主语代词，让"她想起…/想起…/想起…"归并
        prefixes.append(core[:2] if len(core) >= 2 else "")
    runs: list[int] = []
    i, n = 0, len(prefixes)
    while i < n:
        if not prefixes[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and prefixes[j + 1] == prefixes[i]:
            j += 1
        if j - i + 1 >= min_run:
            runs.append(j - i + 1)
        i = j + 1
    return runs

@REGISTRY.register(
    "ai_flavor_health", config_key="ai_flavor_enabled", tag_prefix="ai_flavor",
    repair="advisory", scope="chapter",
    proof="Recomputed over 638 archived chapters (tools/orphan_gates.py): fires "
          "2.5%, and 6.7% on the 30 v2-written ones. Rare by design and reachable "
          "on both engines. WIRED into v2/accept.py as advisory.")
def ai_flavor_health(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic AI-flavor detection: clichés / metaphor spam / tell-not-show /
    degree-adverb inflation / summary narration / paragraph monotony.

    Returns the same {metrics, penalty, flags, directives} shape as style_health.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("ai_flavor_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    n = len(body)
    if n < 200:
        return {"metrics": {"chars": n}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {"chars": n}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0
    kchars = n / 1000.0

    # --- 1. AI cliché density ---
    cliche_matches = _AI_CLICHE_PATTERNS.findall(body)
    cliche_per_kchar = round(len(cliche_matches) / max(kchars, 0.1), 2)
    metrics["ai_cliche_count"] = len(cliche_matches)
    metrics["ai_cliche_per_kchar"] = cliche_per_kchar
    cliche_warn = float(cfg.get("ai_cliche_per_kchar_warn", 4.0))
    cliche_bad = float(cfg.get("ai_cliche_per_kchar_bad", 8.0))
    if cliche_per_kchar >= cliche_bad:
        penalty += 1.5
        flags.append(f"ai_cliche_overload({cliche_per_kchar:.1f}/k>={cliche_bad})")
        top_cliches = _top_n_matches(cliche_matches, 5)
        directives.append(
            "严重AI味：套话密度过高（%.1f/千字），整章读起来像AI生成模板。"
            "禁止使用以下表达及其变体：%s。"
            "用具体的、只属于本场景的身体反应/动作/环境变化替代。"
            % (cliche_per_kchar, "、".join(top_cliches))
        )
    elif cliche_per_kchar >= cliche_warn:
        penalty += 0.5
        top_cliches = _top_n_matches(cliche_matches, 4)
        directives.append(
            "AI味偏重：套话密度 %.1f/千字。减少以下表达：%s。"
            "换用新鲜的、贴合当前情境的具体描写。"
            % (cliche_per_kchar, "、".join(top_cliches))
        )

    # --- 2. Metaphor overload + stale metaphors ---
    simile_hits = _SIMILE_PATTERNS.findall(body)
    metaphor_per_kchar = round(len(simile_hits) / max(kchars, 0.1), 2)
    metrics["metaphor_per_kchar"] = metaphor_per_kchar
    stale_hits = _STALE_METAPHORS.findall(body)
    metrics["stale_metaphor_count"] = len(stale_hits)
    metaphor_warn = float(cfg.get("metaphor_per_kchar_warn", 5.0))
    if metaphor_per_kchar >= metaphor_warn:
        penalty += 0.5
        flags.append(f"metaphor_overload({metaphor_per_kchar:.1f}/k>={metaphor_warn})")
        directives.append(
            "比喻过载（%.1f/千字）：每千字比喻控制在3个以内，"
            "每个比喻必须新鲜准确且服务于情节，宁可朴素直白也不堆砌。"
            % metaphor_per_kchar
        )
    if len(stale_hits) >= 2:
        penalty += 0.5
        stale_examples = list(dict.fromkeys(stale_hits))[:4]
        flags.append(f"stale_metaphors({len(stale_hits)})")
        directives.append(
            "陈腐比喻 %d 处：%s。这些比喻已被用滥，禁止再用。"
            "用只属于本场景的新鲜意象替代。"
            % (len(stale_hits), "、".join("「%s」" % s for s in stale_examples))
        )

    # --- 3. Tell-not-show (emotion labeling) ---
    tns_hits = _TELL_NOT_SHOW.findall(body)
    tns_per_kchar = round(len(tns_hits) / max(kchars, 0.1), 2)
    metrics["tell_not_show_count"] = len(tns_hits)
    metrics["tell_not_show_per_kchar"] = tns_per_kchar
    tns_warn = float(cfg.get("tell_not_show_per_kchar_warn", 3.0))
    if tns_per_kchar >= tns_warn:
        penalty += 0.5
        flags.append(f"tell_not_show({tns_per_kchar:.1f}/k>={tns_warn})")
        directives.append(
            '情感贴标签（%.1f/千字）：不要写"他感到震惊/她觉得悲伤"，'
            '改为展示：震惊时手中的东西掉了、悲伤时沉默地做了某个动作。'
            '情绪必须通过行为、对话、生理反应间接呈现。'
            % tns_per_kchar
        )

    # --- 4. Degree-adverb inflation ---
    adv_hits = _DEGREE_ADVERBS.findall(body)
    adv_per_kchar = round(len(adv_hits) / max(kchars, 0.1), 2)
    metrics["adverb_count"] = len(adv_hits)
    metrics["adverb_per_kchar"] = adv_per_kchar
    adv_warn = float(cfg.get("adverb_inflation_per_kchar_warn", 4.0))
    if adv_per_kchar >= adv_warn:
        penalty += 0.5
        flags.append(f"adverb_inflation({adv_per_kchar:.1f}/k>={adv_warn})")
        directives.append(
            '程度副词泛滥（%.1f/千字）：删掉"非常/极其/十分/无比"等词，'
            '用精准的动词和具体的细节替代模糊的程度修饰。'
            % adv_per_kchar
        )

    # --- 5. Summary narration ---
    summary_hits = _SUMMARY_NARRATION.findall(body)
    metrics["summary_narration_count"] = len(summary_hits)
    if len(summary_hits) >= 2:
        penalty += 0.5
        examples = list(dict.fromkeys(summary_hits))[:3]
        flags.append(f"summary_narration({len(summary_hits)})")
        directives.append(
            "总结式叙述 %d 处：%s。删掉这类上帝视角的总结句，"
            "让读者从情节和角色行为中自行感受。"
            % (len(summary_hits), "、".join("「%s」" % s for s in examples))
        )

    # --- 6. Paragraph-start monotony ---
    paragraphs = [p.strip() for p in body.split("\n") if len(p.strip()) >= 8]
    if len(paragraphs) >= 6:
        starts = [p[:4] for p in paragraphs]
        from collections import Counter
        start_counts = Counter(starts)
        most_common_count = start_counts.most_common(1)[0][1] if start_counts else 0
        repeat_ratio = most_common_count / len(paragraphs)
        metrics["paragraph_start_repeat_ratio"] = round(repeat_ratio, 2)
        para_warn = float(cfg.get("paragraph_start_repeat_warn", 0.30))
        if repeat_ratio >= para_warn:
            dominant_start = start_counts.most_common(1)[0][0]
            penalty += 0.5
            flags.append(f"paragraph_monotony({repeat_ratio:.0%}>={para_warn:.0%})")
            directives.append(
                "段落开头单一：%.0f%%的段落以「%s」开头。"
                "变化段落的起始方式：对话、动作、环境、心理交替开篇。"
                % (repeat_ratio * 100, dominant_start)
            )

    # --- 7. Negative-pair constructions ("没有X，也没有Y") ---
    neg_hits = _NEGATIVE_PAIR.findall(body)
    neg_per_kchar = round(len(neg_hits) / max(kchars, 0.1), 2)
    metrics["negative_pair_count"] = len(neg_hits)
    metrics["negative_pair_per_kchar"] = neg_per_kchar
    neg_warn = float(cfg.get("negative_pair_per_kchar_warn", 2.0))
    neg_bad = float(cfg.get("negative_pair_per_kchar_bad", 4.0))
    if neg_per_kchar >= neg_bad:
        penalty += 1.0
        flags.append(f"negative_pair_overload({neg_per_kchar:.1f}/k>={neg_bad})")
        directives.append(
            "否定对仗句式泛滥（%.1f/千字）：「没有X，也没有Y」「不是X，也不是Y」"
            "是最明显的AI写作指纹。删去后半句或改写为正面描述。" % neg_per_kchar
        )
    elif neg_per_kchar >= neg_warn:
        penalty += 0.5
        flags.append(f"negative_pair({neg_per_kchar:.1f}/k>={neg_warn})")
        directives.append(
            "否定对仗偏多（%.1f/千字）：减少「没有X也没有Y」式句式，"
            "直接一句说完即可，不要对称排列两个否定分句。" % neg_per_kchar
        )

    # --- 8. 模板对比pivot + 机械排比(三连及以上同头句) ---
    pivot_hits = _TEMPLATE_PIVOT.findall(body)
    pivot_per_kchar = round(len(pivot_hits) / max(kchars, 0.1), 2)
    metrics["template_pivot_count"] = len(pivot_hits)
    metrics["template_pivot_per_kchar"] = pivot_per_kchar
    pivot_warn = float(cfg.get("template_pivot_per_kchar_warn", 1.2))
    if pivot_per_kchar >= pivot_warn:
        penalty += 0.5
        flags.append(f"template_pivot({pivot_per_kchar:.1f}/k>={pivot_warn})")
        directives.append(
            "模板对比句式过多（%.1f/千字）：「不是X，而是Y」「与其说…不如说」「不仅…更是」"
            "是最典型的AI结构指纹。删掉迂回的对比铺陈，直接一句陈述到位。" % pivot_per_kchar
        )
    runs = _anaphora_runs(body)
    if runs:
        longest = max(runs)
        metrics["anaphora_runs"] = len(runs)
        metrics["anaphora_longest"] = longest
        run_warn = int(cfg.get("anaphora_run_warn", 4))
        if longest >= run_warn or len(runs) >= 2:
            penalty += 0.5
            flags.append(f"parallel_anaphora(runs={len(runs)},max={longest})")
            directives.append(
                "机械排比：出现 %d 处三连及以上的同头句（最长 %d 连）。排比全章最多用一次且须有递进，"
                "其余改成长短错落的正常叙述，别让句子像模板复读。" % (len(runs), longest)
            )

    cap = float(cfg.get("ai_flavor_penalty_cap", 3.0))
    penalty = round(min(penalty, cap), 2)
    metrics["penalty"] = penalty
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:6],
    }

def _top_n_matches(matches: list[str], n: int) -> list[str]:
    """Return the top-N most frequent matches, deduplicated."""
    from collections import Counter
    counts = Counter(matches)
    return [item for item, _ in counts.most_common(n)]

_HEDGE_WORDS = re.compile(
    r"似乎|好像|仿佛|大概|或许|也许|可能|某种|某个|在某种程度上|一定程度|有所|不由得|不禁|不知为何"
)

@REGISTRY.register(
    "paragraph_shape_health", config_key="paragraph_shape_enabled",
    tag_prefix="paragraph", repair="advisory", scope="chapter",
    proof="The census read 'never ran' because v1 never archived this key — that "
          "is 'never measured', not 'measured silent'. Recomputed over 638 "
          "chapters (tools/orphan_gates.py): fires 29.6%, but 0/30 on v2-written "
          "prose, which shapes paragraphs cleanly. WIRED as advisory: at 0% it "
          "costs no prompt bytes and buys a regression tripwire. Still must not "
          "block — the thresholds have never been validated against a live "
          "BLOCKING distribution.")
def paragraph_shape_health(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic paragraph-uniformity and hedge-word density check.

    Returns the standard {metrics, penalty, flags, directives} shape.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("paragraph_shape_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    body = _strip_title_line(text)
    n = len(body)
    if n < 300:
        return {"metrics": {"chars": n}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {"chars": n}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    # --- 1. Paragraph length uniformity (coefficient of variation) ---
    paragraphs = [p for p in body.split("\n") if len(p.strip()) >= 20]
    if len(paragraphs) >= 5:
        lengths = [len(p.strip()) for p in paragraphs]
        mean_len = sum(lengths) / len(lengths)
        if mean_len > 0:
            variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
            std_len = variance ** 0.5
            cv = round(std_len / mean_len, 3)
            metrics["paragraph_count"] = len(paragraphs)
            metrics["paragraph_length_mean"] = round(mean_len, 1)
            metrics["paragraph_length_cv"] = cv
            cv_min = float(cfg.get("paragraph_cv_min", 0.15))
            if cv < 0.10:
                penalty += 1.0
                flags.append(f"paragraph_uniform_severe(cv={cv:.2f}<0.10)")
                directives.append(
                    "段落长度高度整齐（变异系数 %.2f），像 AI 流水线产出。"
                    "大幅增加段落长短交错——用1-2句短段制造节奏冲击，用长段深入细节。" % cv
                )
            elif cv < cv_min:
                penalty += 0.5
                flags.append(f"paragraph_uniform(cv={cv:.2f}<{cv_min})")
                directives.append(
                    "段落长度偏于整齐（变异系数 %.2f），增加段落的长短交错——"
                    "短段制造节奏感，长段深入细节。" % cv
                )

    # --- 2. Short paragraph detection (avg paragraph length) ---
    all_paras = [p for p in body.split("\n") if len(p.strip()) >= 8]
    if len(all_paras) >= 5:
        all_lens = [len(p.strip()) for p in all_paras]
        avg_para = sum(all_lens) / len(all_lens)
        short_count = sum(1 for l in all_lens if l < 30)
        metrics["avg_paragraph_chars"] = round(avg_para, 1)
        metrics["short_paragraph_ratio"] = round(short_count / len(all_lens), 2)
        severe_threshold = float(cfg.get("short_paragraph_severe", 30))
        warn_threshold = float(cfg.get("short_paragraph_warn", 50))
        if avg_para < severe_threshold:
            penalty += 1.5
            flags.append(f"short_paragraph_severe(avg={avg_para:.0f}<{severe_threshold:.0f})")
            directives.append(
                "AI碎段病严重：段均仅 %.0f 字，几乎每句话单独一行，观感极差。"
                "每段至少3-5句/60字以上（对话除外），把碎片合并成有起承转的段落。" % avg_para
            )
        elif avg_para < warn_threshold:
            penalty += 0.5
            flags.append(f"short_paragraph_warn(avg={avg_para:.0f}<{warn_threshold:.0f})")
            directives.append(
                "段落偏短（均 %.0f 字/段），合并相邻短句为完整段落，"
                "让叙事有呼吸感而非碎片堆砌。" % avg_para
            )

    # --- 3. Hedge word density ---
    kchars = n / 1000.0
    hedge_matches = _HEDGE_WORDS.findall(body)
    hedge_per_kchar = round(len(hedge_matches) / max(kchars, 0.1), 2)
    metrics["hedge_count"] = len(hedge_matches)
    metrics["hedge_per_kchar"] = hedge_per_kchar
    hedge_warn = float(cfg.get("hedge_per_kchar_warn", 5.0))
    hedge_bad = float(cfg.get("hedge_per_kchar_bad", 10.0))
    if hedge_per_kchar >= hedge_bad:
        penalty += 1.0
        flags.append(f"hedge_overload({hedge_per_kchar:.1f}/k>={hedge_bad})")
        top_hedges = _top_n_matches(hedge_matches, 3)
        directives.append(
            "模糊词密度过高（%.1f/千字），文风犹疑无力。"
            "删掉或替换：%s。用确定性描写替代模棱两可的叙述。"
            % (hedge_per_kchar, "、".join("「%s」" % h for h in top_hedges))
        )
    elif hedge_per_kchar >= hedge_warn:
        penalty += 0.5
        top_hedges = _top_n_matches(hedge_matches, 3)
        directives.append(
            "模糊词偏多（%.1f/千字）。减少：%s。换用确切的动作和事实。"
            % (hedge_per_kchar, "、".join("「%s」" % h for h in top_hedges))
        )

    cap = float(cfg.get("paragraph_shape_penalty_cap", 3.0))
    penalty = round(min(penalty, cap), 2)
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:4],
    }

_STRONG_PAYOFF_TYPES = {
    "reveal", "reversal", "court_breakthrough", "military_victory",
    "policy_payoff", "personnel_payoff", "institutional_fix", "payoff",
}

@REGISTRY.register(
    "hook_tail_repetition", config_key="adjacent_repeat_enabled",
    tag_prefix="hook", repair="advisory", scope="chapter",
    proof="Recomputed over 638 chapters (tools/orphan_gates.py): fires 1/638, "
          "0/30 on v2. Reachable via the CLAUSE-COUNT path, not the ratio one: "
          "tail_clause_overlap tops out at 0.19 against a 0.25 line, so judging "
          "this gate on its ratio alone reports 1.3x headroom and no positives. "
          "Either reading is LESSONS 4's KEEP zone (adjacent_repetition kept at "
          "1.1x; dialogue_pingpong deleted at 3.6x), and one true positive is on "
          "record. WIRED as advisory; at this rate it costs ~zero prompt bytes. "
          "Relabelled from repair=L1, which promised a fixer fix.ACTION_BY_GATE "
          "never had. Its `repeat` verdict was invisible to tools/gate_census.py "
          "until the same date — a gate reported silent is a deletion candidate, "
          "so that omission nearly argued a live gate out of the tree. Still "
          "shares `adjacent_repeat_enabled` with adjacent_repetition, so the two "
          "cannot be enabled independently — a config-shape wart.")
def hook_tail_repetition(
    text: str,
    prev_texts: list[str] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a chapter-end hook recycled from recent chapters' endings.

    Recurring debt across books: "章末钩子与上章完全相同，锐利度被严重稀释"
    (the LLM reviewer still rated such hooks 9/10 because it never sees the
    previous endings side by side). Compares the clause set of this chapter's
    final ~300 chars against the final ~800 chars of each recent chapter.
    Returns {"repeat": bool, "repeated_clauses", "ratio", "directives"}.

    `directives` is part of the return shape rather than something the caller
    phrases, so the advice lives next to the metric that earns it — `v2/accept.py`
    folds every advisory gate's `directives` list by key and cannot special-case
    one gate's wording without becoming a second author of it.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {"repeat": False, "repeated_clauses": [], "ratio": 0.0,
                              "directives": []}
    if not text or not prev_texts:
        return result
    tail_chars = int(cfg.get("hook_repeat_tail_chars", 300))
    cur = [c for c in _clause_segments(text[-tail_chars:]) if len(c) >= 8]
    if not cur:
        return result
    repeated: set[str] = set()
    for pt in prev_texts:
        prev_tail_set = {_normalize_clause(c) for c in _clause_segments(pt[-max(tail_chars * 2, 600):])}
        for c in cur:
            if _normalize_clause(c) in prev_tail_set:
                repeated.add(c)
    ratio = len(repeated) / len(cur)
    result["repeated_clauses"] = sorted(repeated, key=len, reverse=True)[:4]
    result["ratio"] = round(ratio, 3)
    min_clauses = int(cfg.get("hook_repeat_min_clauses", 2))
    min_ratio = float(cfg.get("hook_repeat_min_ratio", 0.25))
    result["repeat"] = len(repeated) >= min_clauses or ratio >= min_ratio
    if result["repeat"]:
        result["directives"].append(
            "章末钩子与近几章的收尾重复（"
            + "；".join(f"“{c}”" for c in result["repeated_clauses"][:2])
            + "）。钩子的锐利度全在「新」上，复用过的收尾句一律换掉："
            "换悬念对象、换提问角度，或把钩子从「预告」改成「当场发生」。"
        )
    return result

@REGISTRY.register(
    "intra_chapter_repetition", config_key="intra_repeat_enabled",
    tag_prefix="repeat", repair="advisory", scope="chapter",
    proof="Recomputed over 638 chapters (tools/orphan_gates.py): fires 1/638, "
          "0/30 on v2. Reachable rather than merely quiet — tail_recap_ratio "
          "reaches 0.49 against a 0.25 line (p95 0.05). WIRED as advisory. It "
          "declared repair=L1 with NO entry in fix.ACTION_BY_GATE; declaring a "
          "layer is not a promise a fixer exists, so the label now matches "
          "reality. A fixer for a 1-in-638 problem is not worth writing.")
def intra_chapter_repetition(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure how much the chapter's ending re-states its own earlier content.

    Splits the chapter into a head (everything but the last `tail_chars`) and a
    tail. Counts how many distinctive tail clauses already appeared (verbatim or
    near-verbatim) in the head. A high ratio means the ending is a zero-增量
    summary recap rather than a forward-moving hook.

    Returns {"metrics", "level" (ok/warn/block), "penalty", "flags",
    "directives", "examples"}.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "level": "ok", "penalty": 0.0,
        "flags": [], "directives": [], "examples": [],
    }
    if not bool(cfg.get("intra_repeat_enabled", True)) or not text:
        return result
    body = _strip_title_line(text)
    if len(body) < int(cfg.get("intra_repeat_min_chars", 1500)):
        return result
    tail_chars = int(cfg.get("intra_repeat_tail_chars", 600))
    if len(body) <= tail_chars + 200:
        return result
    head = body[:-tail_chars]
    tail = body[-tail_chars:]
    head_set = {_normalize_clause(c) for c in _clause_segments(head)}
    tail_clauses = _clause_segments(tail)
    if not tail_clauses:
        return result
    hits = [c for c in tail_clauses if _normalize_clause(c) in head_set]
    ratio = len(hits) / len(tail_clauses)
    result["metrics"] = {
        "tail_recap_ratio": round(ratio, 3),
        "tail_clauses": len(tail_clauses),
        "recap_hits": len(hits),
    }
    result["examples"] = sorted(set(hits), key=len, reverse=True)[:4]
    warn = float(cfg.get("intra_repeat_warn", 0.25))
    block = float(cfg.get("intra_repeat_block", 0.45))
    if ratio >= block:
        result["level"] = "block"
        result["penalty"] = float(cfg.get("intra_repeat_block_penalty", 2.0))
        result["flags"].append(f"intra_chapter_recap(ratio={ratio:.2f})")
        result["directives"].append(
            f"本章结尾有 {ratio:.0%} 的句子在复述正文已给出的推理/结论（零增量总结段）。"
            "章末必须是【前进的钩子】——抛出新疑问、新动作、新危机，而不是把已讲过的线索再列一遍。"
            "删去总结复述，让结尾推动剧情往下走。以下复述句严禁出现：" +
            "；".join(f"“{c}”" for c in result["examples"][:3])
        )
    elif ratio >= warn:
        result["level"] = "warn"
        result["penalty"] = float(cfg.get("intra_repeat_warn_penalty", 0.8))
        result["flags"].append(f"intra_chapter_recap(ratio={ratio:.2f})")
        result["directives"].append(
            f"本章结尾约 {ratio:.0%} 在复述正文已有信息，有总结收尾倾向。"
            "请把结尾改成推动剧情的钩子，而非已知信息的回顾。"
        )
    return result

_METAPHOR_MARKERS = re.compile(r"[像如仿佛似若好似犹如宛如恍若好像一如]")

_SENSORY_WORDS = re.compile(
    r"[温暖冰凉灼热潮湿干燥刺鼻芬芳苦涩甘甜酥麻沉闷轰鸣寂静回荡]|"
    r"光芒|阴影|色泽|声响|气味|触感|余温|寒意|热浪|微风"
)

_NUMBER_PATTERN = re.compile(
    r"(?:百分之[一二三四五六七八九十零〇两\d]+|"
    r"\d+(?:\.\d+)?(?:%|‰|°|℃|赫兹|毫米|厘米|分钟|秒|小时|公斤|千克|米|层|级|阶)?|"
    r"零点[一二三四五六七八九十零〇两\d]+)"
)

@REGISTRY.register(
    "prose_texture", config_key="prose_texture_enabled", tag_prefix="texture",
    repair="advisory", scope="chapter",
    proof="Was 44.0% of 638 chapters / 63.3% of v2's, because the over_poetic "
          "DIRECTIVE line was a hardcoded 6.0 while the corpus runs median 31.9 "
          "(min 10.2): 0/638 chapters sat under it, so the conjunct was always "
          "true and the branch degenerated into 'this chapter has few numbers'. "
          "Now shares the penalty branch's calibrated 40.0 — 5.0% corpus, 0/30 "
          "on v2. WIRED as advisory. Must never block: the poetic_density regex "
          "systematically overcounts (see the docstring).")
def prose_texture(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the quantitative vs poetic texture of prose.

    Returns metrics + a suggestion directive if the balance is skewed.
    """
    text = _strip_title_line(text)
    chars = max(len(text), 1)
    kchars = chars / 1000.0

    numbers = _NUMBER_PATTERN.findall(text)
    num_per_kchar = round(len(numbers) / max(kchars, 0.1), 2)

    metaphors = _METAPHOR_MARKERS.findall(text)
    metaphor_per_kchar = round(len(metaphors) / max(kchars, 0.1), 2)

    sensory = _SENSORY_WORDS.findall(text)
    sensory_per_kchar = round(len(sensory) / max(kchars, 0.1), 2)

    poetic_density = metaphor_per_kchar + sensory_per_kchar

    cfg = (config or {}).get("novel", {})
    num_high = float(cfg.get("texture_num_high_per_kchar", 8.0))
    poetic_low = float(cfg.get("texture_poetic_low_per_kchar", 1.0))
    # The over-poetic DIRECTIVE line. It used to be a hardcoded 6.0, which is
    # below the entire library: measured over 638 archived chapters
    # (`tools/orphan_gates.py`), poetic_density runs median 31.9, min 10.2, max
    # 49.8 — 0/638 chapters sit at or under 6.0. So the conjunct was always true
    # and the branch degenerated into a bare `num_per_kchar < 1.0` test ("this
    # chapter contains few numbers"), firing on 44% of the library and 63% of
    # v2's chapters while telling the writer to cut metaphors it had not
    # miscounted. The penalty branch below already used the calibrated 40.0; this
    # now shares it, so one function stops holding two answers to one question.
    poetic_high = float(cfg.get("texture_poetic_penalty_threshold", 40.0))

    flags: list[str] = []
    directives: list[str] = []
    balance = "balanced"

    if num_per_kchar > num_high and poetic_density < poetic_low:
        balance = "over_quantitative"
        flags.append("数据密度过高且缺少诗意变奏")
        directives.append(
            "本章数字/数据密度偏高（{:.1f}/千字）而比喻/感官描写偏少（{:.1f}/千字）。"
            "下一章请交替使用：具体数据锚定 + 比喻/通感/感官意象，"
            "避免连续段落全用精确数值描写。至少 2 处用比喻或感官替代直接数字。".format(
                num_per_kchar, poetic_density
            )
        )
    elif num_per_kchar < 1.0 and poetic_density > poetic_high:
        balance = "over_poetic"
        flags.append("诗意过度缺少具体锚定")
        directives.append(
            "本章比喻/感官密度过高（poetic_density={:.1f}/千字），偏向散文诗而非叙事推进。"
            "下一章大幅削减比喻与华丽形容：每段最多保留 1 个比喻，改用白描的具体动作、对话，"
            "并在关键处加 2-3 个数字/量级/时限锚定，优先把情节往前推。".format(poetic_density)
        )

    # Over-poetic（紫色文体）安全网：仅对 EGREGIOUS 离群（poetic_density 远超正常语体）扣分。
    # 注意：这里的 poetic_density 用单字比喻/感官正则（如/似/若…）粗测，中文里这些字常作
    # 非比喻功能词（如果/似乎/一如既往），会系统性高估——健康中文网文正文普遍就跑 ~25-35。
    # 因此阈值必须设在正常语体之上（默认 40），否则会惩罚正常文本（Ch1-11 好章也 ~30）。
    # 真正的"风格飘逸"防线是相对尖峰门：style_health 的 em_dash_trend_rise（vs 近章均值）与
    # 跨章化石检测——它们按"相对基线的突变"判定漂移，比这个绝对阈值可靠。此惩罚只兜底极端塌缩。
    penalty = 0.0
    if balance == "over_poetic":
        # `poetic_high` above IS `texture_poetic_penalty_threshold`, so reaching
        # this branch already means the density cleared the line; the graduated
        # penalty is what the threshold buys beyond the directive.
        pen_cap = float(cfg.get("texture_poetic_penalty_cap", 1.5))
        penalty = min(pen_cap, round((poetic_density - poetic_high) * 0.1, 2))

    return {
        "metrics": {
            "num_per_kchar": num_per_kchar,
            "metaphor_per_kchar": metaphor_per_kchar,
            "sensory_per_kchar": sensory_per_kchar,
            "poetic_density": round(poetic_density, 2),
        },
        "balance": balance,
        "flags": flags,
        "directives": directives,
        "penalty": penalty,
    }

@REGISTRY.register(
    "long_span_fatigue", config_key="long_span_fatigue_enabled",
    tag_prefix="fatigue", repair="advisory", scope="book",
    proof="THE clearest book-scope quantity in the registry — fatigue accumulated "
          "over a long span of finished chapters, which the chapter under review "
          "cannot lower. Advisory is the only correct layer for it; `may_block` "
          "returns False on both counts. Recomputed over 638 chapters "
          "(tools/orphan_gates.py): 59.2% corpus / 40.0% on v2, now entirely the "
          "payoff_type_monotony term — the emotional-diversity and tension-"
          "variance terms read columns that are 0/30 on v2 and were removed. "
          "WIRED as advisory; at 40% it is the loudest of the nine, tolerable "
          "only because it can never reject.")
def long_span_fatigue(
    conn: Any,
    chapter_num: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Payoff-type monotony across a 12-chapter span. Book scope, advisory only.

    Reads `chapter_metrics` via `store.recent_metrics` (NEWEST-FIRST — the streak
    loop below counts backwards from `[-1]`, so feeding it ascending rows reports
    the streak at chapter 1 forever). Returns {metrics, penalty, flags,
    directives}.

    It had three terms until 2026-07-28; the other two are gone and the comment
    where they used to be says why. `advise_only` by construction: the span is
    book-cumulative, so no rewrite of the current chapter can shorten a streak
    that already happened.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("long_span_fatigue_enabled", True)):
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}
    if chapter_num < 5:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    try:
        from engine.store import recent_metrics
        rows = recent_metrics(conn, limit=12)
    except Exception:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}
    if len(rows) < 4:
        return {"metrics": {}, "penalty": 0.0, "flags": [], "directives": []}

    metrics: dict[str, Any] = {}
    flags: list[str] = []
    directives: list[str] = []
    penalty = 0.0

    # --- 1. Payoff type monotony ---
    # Fall back to payoff_type_monotony_max (the documented, config-set key) so the
    # review-side penalty and the plan-side rotation gate share one threshold;
    # chapter_type_monotony_max stays as a legacy per-check override if ever set.
    type_max = int(cfg.get("chapter_type_monotony_max", cfg.get("payoff_type_monotony_max", 4)))
    payoff_types = [str(r.get("payoff_type", "")).strip() for r in rows if r.get("payoff_type")]
    if payoff_types:
        streak = 1
        for i in range(len(payoff_types) - 2, -1, -1):
            if payoff_types[i] == payoff_types[-1]:
                streak += 1
            else:
                break
        metrics["payoff_type_streak"] = streak
        if streak >= type_max and payoff_types[-1]:
            penalty += 0.5
            flags.append(f"payoff_type_monotony({streak}>={type_max})")
            directives.append(
                f"近 {streak} 章都是「{payoff_types[-1]}」爽点类型，读者审美疲劳。"
                f"本章切换到不同的 payoff_type（如 reveal/reversal/emotional）。"
            )

    # --- terms 2 and 3 (emotional diversity, tension flatness) were REMOVED on
    # 2026-07-28. Both read `chapter_metrics` columns that only v1's LLM
    # self-review filled; on v2-written chapters `emotional_tone` and `tension`
    # are 0/30, so both branches were unreachable on the live engine. Measured
    # (`tools/orphan_gates.py`, 638 chapters): over the v2 window this gate emits
    # ONLY `payoff_type_monotony` (12/30) — the two removed terms contributed
    # nothing, while on v1 archives they were the loud majority (tension_flat 328,
    # emotional_monotony 9, payoff_type_monotony 378).
    #
    # Two lessons worth keeping if a producer for these columns is ever added:
    #   * `emotional_tone` holds FREE TEXT, not an enum label (median 65 chars,
    #     382/565 distinct across the library), so a `distinct < 3` test over it is
    #     near-unreachable for the same schema reason that killed
    #     `emotional_cadence`. Enumerate the labels first.
    #   * A cheap extraction model returns the SAME integer tension every chapter
    #     (yeban_guize: 9,9,9,9,9,9). That is a measurement artifact, not a flat
    #     arc — a genuinely flat but measured arc still jitters (8,9,9,8,9). Any
    #     variance test needs to suppress exactly-constant runs, or it fires
    #     hardest precisely where the signal is least trustworthy.

    cap = float(cfg.get("long_span_fatigue_penalty_cap", 1.5))
    penalty = round(min(penalty, cap), 2)
    return {
        "metrics": metrics,
        "penalty": penalty,
        "flags": flags,
        "directives": directives[:3],
    }

_PAYOFF_MARKERS = re.compile(
    r"识破|拆穿|揭穿|当众|反转|逆转|碾压|打脸|一锤定音|真相大白|当场|反将|反咬|"
    r"哑口无言|无言以对|目瞪口呆|脸色骤变|脸色大变|败下阵|认输|低头|跪|"
    r"揭晓|水落石出|原形毕露|扳回|翻盘|破局|绝杀|完胜|压制|镇住|震慑"
)

@REGISTRY.register(
    "payoff_beat_density", config_key="payoff_density_enabled",
    tag_prefix="payoff", repair="advisory", scope="chapter",
    proof="The census read 'never ran' because v1 never archived this key. "
          "Recomputed over 638 chapters (tools/orphan_gates.py): fires 7.4%, and "
          "10.0% on v2 — chapters_since_payoff reaches 10 against a 2.5 line (p95 "
          "3.0), so the threshold sits well inside the distribution. Its input "
          "(payoff_type) is 30/30 on v2 AND more diverse there than on v1 (8 "
          "distinct values vs 5 over the same 30 positions), so this is a real "
          "capability v2 had lost, not a v1 leftover. WIRED as advisory.")
def payoff_beat_density(
    text: str,
    recent_payoff_types: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure 爽点 density: payoff-shaped events in this chapter plus the recent
    payoff_type cadence. Returns a directive when the recent window has gone too
    long without a strong reader payoff.

    `recent_payoff_types` is the newest-first list of recent chapters'
    payoff_type (from chapter_metrics); a 'setup'/'strategic_setup'/'emotional'
    type does not count as a strong payoff.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    body = _strip_title_line(text or "")
    kchars = max(len(body) / 1000.0, 0.1)
    hits = _PAYOFF_MARKERS.findall(body)
    hits_per_kchar = round(len(hits) / kchars, 2)

    strong_types = _STRONG_PAYOFF_TYPES
    rt = recent_payoff_types or []
    # Chapters since the last STRONG payoff (newest-first list).
    chapters_since_payoff = 0
    for t in rt:
        if str(t).strip() in strong_types:
            break
        chapters_since_payoff += 1

    flags: list[str] = []
    directives: list[str] = []
    # payoff_density_min is a per-chapter rate (≈0.34 ⇒ 1 strong payoff / 3 ch).
    min_rate = float(cfg.get("payoff_density_min", 0.34))
    max_gap = int(round(1.0 / min_rate)) if min_rate > 0 else 3
    if rt and chapters_since_payoff >= max_gap:
        flags.append(f"payoff_drought({chapters_since_payoff})")
        directives.append(
            f"近 {chapters_since_payoff} 章没有强爽点/高潮（揭晓/反转/打脸/能力兑现）。"
            "本章必须安排一次明确的读者爽点：让主角的优势/真相/反击落到具体的当众场面或对手的可见崩溃上。"
        )

    return {
        "metrics": {
            "payoff_markers": len(hits),
            "payoff_per_kchar": hits_per_kchar,
            "chapters_since_payoff": chapters_since_payoff,
        },
        "flags": flags,
        "directives": directives,
    }

_SHAREABLE_MARKERS = re.compile(
    r"从今(?:天|往)?(?:起|以后)|从现在起|记住|凭什么|我偏|我就是|也配|不过如此|活该|"
    r"早晚|总有一天|莫欺|三十年河|给我跪|你们这些|我说过|谁规定|凭本事|宁可|绝不|"
    r"不是.{0,12}(?:而是|是)|要么.{0,10}要么|从不|永远记住|欠我的|该还了|轮到"
)

_SHAREABLE_PERSON = re.compile(r"[我你]")

def _quotable_score(line: str) -> float:
    """Heuristic 'how截图-able is this line' score (0+)."""
    s = 0.0
    if _SHAREABLE_MARKERS.search(line):
        s += 2.0
    if _SHAREABLE_PERSON.search(line):
        s += 1.0
    if len(line) <= 18:  # punchy short lines screenshot better
        s += 1.0
    return s

@REGISTRY.register(
    "shareable_line", config_key="shareable_line_enabled",
    tag_prefix="shareable", repair="advisory", scope="chapter",
    proof="Recomputed over 638 chapters (tools/orphan_gates.py): fires 5.8%, and "
          "0/30 on v2. It fires on the LOW tail, so the deletion test runs against "
          "the MINIMUM, not the max: best_quotable_score dips to 1.0 against a "
          "line of 2.0, i.e. reachable with true positives on record. Comparing "
          "its line to the observed max reports 0.5x and reads as unreachable — "
          "the wrong tail. WIRED as advisory.")
def shareable_line(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect whether the chapter contains a可截图、可传播的金句钩子.

    Scans quoted dialogue across the chapter plus the chapter tail (where宣言式
    金句 most often lands), scores each candidate, and returns the best. When no
    candidate clears the threshold, emits an advisory directive (no penalty) so
    the next chapter plants a传播性金句. Gated by `shareable_line_enabled`.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "metrics": {}, "has_shareable": False, "best_line": "", "score": 0.0,
        "flags": [], "directives": [],
    }
    if not bool(cfg.get("shareable_line_enabled", True)) or not text:
        return result
    body = _strip_title_line(text)
    if len(body) < 500:
        return result
    candidates: set[str] = set()
    # Punchy lines often live in dialogue — pull quoted segments from the whole chapter.
    for m in re.findall(r"[“「]([^”」\n]{4,40})[”」]", body):
        candidates.add(m.strip())
    # Plus the chapter tail's short narration sentences (章末金句).
    tail = body[-int(cfg.get("shareable_tail_chars", 500)):]
    for seg in re.split(r"[。！？\n]", tail):
        seg = seg.strip()
        if 6 <= len(seg) <= 30:
            candidates.add(seg)
    best = 0.0
    best_line = ""
    for c in candidates:
        sc = _quotable_score(c)
        if sc > best:
            best = sc
            best_line = c
    threshold = float(cfg.get("shareable_min_score", 2.0))
    has = best >= threshold
    result["metrics"] = {"candidates": len(candidates), "best_score": round(best, 1)}
    result["has_shareable"] = has
    result["best_line"] = best_line[:60]
    result["score"] = round(best, 1)
    if not has:
        result["flags"].append("no_shareable_line")
        result["directives"].append(
            "本章缺少可截图、可传播的金句钩子。番茄段评/书荒广场的自然传播靠金句驱动——"
            "本章请在一个高情绪节点（爆发/对峙/逆袭/反转）放一句够短够狠、独立成段的金句"
            "（复仇宣言/逆袭宣言/认知颠覆/后果预告），让读者想截图发出去。"
        )
    return result

@REGISTRY.register(
    "information_density", config_key="info_density_enabled", tag_prefix="info",
    repair="advisory", scope="chapter",
    proof="Was 43.5% of the 462 archived chapters with a recoverable plan and "
          "70.0% of v2's 30, because two of its four signals had no producer yet "
          "were counted as agreement — the documented '3 of 4 must agree' was "
          "really '1 of 2' (see the docstring). After removing them: 7.8% corpus, "
          "13.3% on v2 (tools/orphan_gates.py). The other 176/638 chapters have no "
          "recoverable plan and are reported UNMEASURED rather than clean: an "
          "absent payoff_type reads as a weak one, which would score every "
          "unplanned chapter low-information. WIRED as advisory.")
def information_density(
    text: str,
    plan: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,  # noqa: ARG001 — see docstring; kept so
                                           # existing positional callers still pass
                                           # `config` as the 4th argument.
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a 'pure transition chapter' that advances nothing: no payoff, no
    new information. Heuristic and conservative — it only flags when MULTIPLE
    signals agree, to avoid punishing a legitimately quiet breather chapter.

    Signals (all derived from already-computed data, no extra LLM call):
      - payoff_type is setup/emotional (not a concrete reader payoff)
      - the chapter's payoff markers are ~zero (no 爽点)

    Two more signals were removed on 2026-07-28 because neither had an input:
    `no_info_reveals` read `plan["info_reveals"]`, a key **no producer in the
    codebase writes** (`arc.card_to_plan` does not emit it), so on v2 it was
    unconditionally true — a free third vote that turned the documented "3 of 4
    must agree" into "2 of 2", i.e. no conservatism at all. `no_realized_beats`
    read `review["beats_audit"]`, which only v1's LLM reviewer produced. Counting
    an unavailable signal as agreement is the inverse of the sentinel-as-verdict
    defect in CLAUDE.md: absence read as assent.

    `low_information` now requires BOTH measurable signals (see below); the old
    `info_density_min_signals` knob is gone. Measured effect
    (`tools/orphan_gates.py`): 6.1% → 7.8% over the 462 archived chapters that
    have a recoverable plan, and 13.3% → 13.3% over v2's window — v2 is unchanged
    because there the removed third vote was always free, so the effective bar was
    already 2-of-2. The archive moves because on v1 chapters `info_reveals`
    sometimes DID exist (114 of 641 plans carried it), and those chapters were
    being exempted by a signal the current engine cannot produce at all.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("info_density_enabled", True)):
        return {"low_information": False, "signals": [], "directives": []}

    plan = plan or {}
    signals: list[str] = []

    ptype = str(plan.get("payoff_type", "")).strip().lower()
    if ptype in ("", "setup", "strategic_setup", "emotional"):
        signals.append(f"payoff_type={ptype or 'none'}")

    body = _strip_title_line(text or "")
    if len(_PAYOFF_MARKERS.findall(body)) == 0:
        signals.append("no_payoff_markers")

    # BOTH signals must agree — hardcoded, not configurable. There are exactly two
    # of them, so the old `info_density_min_signals` knob could only take the
    # meaningless value 1 or the unreachable value 3, and every config in the repo
    # pinned it at 3 (the count from when two never-produced signals were still
    # counted). Left tunable, this gate measured 0/638 after the signal fix while
    # looking enabled — a threshold set above its own maximum achievable score,
    # which is the exact defect class this pass exists to remove.
    low_info = len(signals) >= 2
    directives: list[str] = []
    if low_info:
        directives.append(
            "上一章信息推进不足（近似过渡章：无爽点、无新信息、无伏线推进）。"
            "本章必须至少做到其一并落到页面上：引入关键新信息、推进/兑现一条伏线、或制造一次冲突升级。"
        )
    return {"low_information": low_info, "signals": signals, "directives": directives}


@REGISTRY.register(
    "chapter_ending_strength", config_key="ending_strength_enabled",
    tag_prefix="ending", repair="advisory", scope="chapter",
    proof="UNVALIDATED — new gate, no historical data yet. Designed as pure "
          "advisory (no penalty) to avoid the threshold-unreachability defect "
          "that killed the old chapter_ending_quality gate. Heuristic checks "
          "three positive signals in the chapter tail; fires when none are "
          "present.")
def chapter_ending_strength(
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a chapter ending that lacks hook strength.

    Scans the final ~150 chars for three positive signals (any one suffices):
      - dialogue quote (ending on a spoken line)
      - question mark (posing a question to the reader)
      - dash/ellipsis (suspense rhythm)
    When none are found, emits an advisory directive for the next chapter.
    No penalty — ending quality is too subjective for a numeric score.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    result: dict[str, Any] = {
        "has_hook": True, "signals": [], "flags": [], "directives": [],
    }
    if not bool(cfg.get("ending_strength_enabled", True)) or not text:
        return result
    body = _strip_title_line(text)
    if len(body) < 300:
        return result
    tail_len = int(cfg.get("ending_strength_tail_chars", 150))
    tail = body[-tail_len:]
    signals: list[str] = []
    if re.search(r'["“”「」]', tail):
        signals.append("dialogue")
    if '？' in tail or '?' in tail:
        signals.append("question")
    if '——' in tail or '……' in tail:
        signals.append("suspense_punct")
    result["signals"] = signals
    if signals:
        return result
    result["has_hook"] = False
    result["flags"].append("weak_chapter_ending")
    result["directives"].append(
        "上一章以纯叙述/描写收尾，缺少钩子。"
        "本章结尾请用三选一：①对话中抛出新悬念；②用问句让读者想知道答案；"
        "③用反转/危机的具体动作收束，最后一段独立成段、短而狠。"
    )
    return result

