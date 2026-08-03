"""Repair ladder: deterministic (L0) and bounded-LLM (L1) fixers for quality gates.

Extracted from ``engine/quality.py`` — the pipeline's answer to a gate firing is
to route it to the CHEAPEST action that can actually fix it, instead of re-rolling
the whole chapter.

- **L0** deterministic text transforms, zero LLM calls
- **L1** one bounded LLM call that rewrites a handful of extracted passages
"""
from __future__ import annotations

import re
from typing import Any

from engine.quality import (
    REGISTRY,
    _SENTENCE_ENDERS,
    dialogue_health,
    opening_hook_gate,
    reduce_em_dash_density,
    style_health,
)


# ===========================================================================
# Repair ladder (merged from fix.py)
#
# The pipeline's historical answer to a gate firing was to re-roll: revise
# the whole chapter, or replan and write it again. This routes a firing to
# the CHEAPEST action that can actually fix it.
# ===========================================================================


# ---------------------------------------------------------------------------
# Action table
#
# gate name -> action id. A gate may declare a repair layer without having an
# action implemented here (e.g. `intra_chapter_repetition`, which fired once in
# 642 measured reviews); `plan_repairs` simply skips those, so declaring a layer
# is never the same thing as promising a fixer.
#
# `hook_revise` fills the capability gap left when v1's `revise_hook_only` was
# deleted. exit_hook was the #1 CCC acceptance failure (4/4 test failures).
# The L1 fixer rewrites the chapter tail when `chapter_ending_strength` detects
# no positive signal (dialogue/question/suspense_punct) in the final ~150 chars.
# ---------------------------------------------------------------------------

ACTION_BY_GATE: dict[str, str] = {
    # L0 — deterministic, zero LLM
    "style_health": "style_prose",
    "cross_chapter_repetition": "fossil_rotate",
    "book_wide_fossils": "fossil_rotate",
    "descriptor_frequency": "fossil_rotate",
    "opening_hook_gate": "opening_promote",
    # L1 — one bounded call, splice-back
    "length_band_check": "expand_to_band",
    "dialogue_health": "inject_dialogue",
    "chapter_ending_strength": "hook_revise",
}

# Two gates store their result in the review report under a key that is NOT the
# gate name (`review.py:967` / `review.py:1133`). Reading the wrong key looks
# like "the gate never fires".
REPORT_KEY: dict[str, str] = {
    "book_wide_fossils": "book_fossils",
    "length_band_check": "length_band",
}

# L1 fallback for a style_health em-dash flag that L0 could not bring under the
# target density (L0 only rewrites punctuation; some texts need real rewording).
_EM_DASH_L1_ACTION = "em_dash_targeted"

_FIX_QUOTES = "“”「」『』\"'"


# ---------------------------------------------------------------------------
# Firing detection / planning
# ---------------------------------------------------------------------------

def gate_result(review: dict[str, Any], gate: str) -> Any:
    """Read a gate's result out of a review report, honouring `REPORT_KEY`."""
    return (review or {}).get(REPORT_KEY.get(gate, gate))


def _length_band_needs_expand(review: dict[str, Any]) -> bool:
    """False only when `length_band_check` fired on the side expand cannot fix.

    The gate flags BOTH sides of the band and `gate_fired` cannot tell them
    apart, so every over-length chapter used to be planned an expand — the one
    fixer whose own docstring says it handles the short side only. Measured over
    the archive's `review_round0.json` payloads: of 195 chapters that planned
    `expand_to_band`, **109 (56%) were over-length**, i.e. a guaranteed no-op
    holding one of the two `fix_max_l1_calls` slots. Twice it pushed a real
    fixer out of the plan — `tangshuting_e2e` Ch46 and `yeban_guize` Ch8 both
    lost `em_dash_targeted` to the cap. The long side needs nothing here: the
    gate emits its own next-chapter directive, and a gross overshoot blocks.

    Stated as "not the long side" rather than "is the short side" so the failure
    is the safe one. If the gate's flag vocabulary ever drifts, this keeps
    planning the expand (whose own floor check then declines it, out loud)
    instead of silently dropping the short side's only fixer.
    """
    result = gate_result(review, "length_band_check")
    flags = (result.get("flags") or []) if isinstance(result, dict) else []
    return not any(str(f).startswith("chapter_too_long") for f in flags)


def gate_fired(result: Any) -> bool:
    """True when a gate result represents an actual firing.

    Not just ``penalty > 0``: `length_band_check` ships with its penalty
    disabled by default (``length_band_penalty_enabled: false``) yet still
    reports the out-of-band flag, and that flag is the thing worth fixing;
    `book_wide_fossils` and `descriptor_frequency` report findings as lists.
    """
    if not isinstance(result, dict):
        return False
    if float(result.get("penalty", 0.0) or 0.0) > 0:
        return True
    if result.get("block"):
        return True
    if str(result.get("level", "")) in {"advise", "reject"}:
        return True
    for key in ("flags", "phrases", "fossils", "flagged", "repeats", "template_fossils"):
        if result.get(key):
            return True
    return False


def plan_repairs(review: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a review report's fired gates onto an ordered list of repair steps.

    Returns entries of ``{"gate", "layer", "action"}``, L0 before L1, deduped by
    action (one fossil rotation covers all three fossil gates), and capped at
    ``fix_max_l1_calls`` L1 steps. Zero LLM calls, pure function.
    """
    cfg = (config or {}).get("novel", {})
    max_l1 = int(cfg.get("fix_max_l1_calls", 2))
    steps: list[dict[str, Any]] = []
    seen_actions: set[str] = set()

    # A gate that did not run leaves no key in the report, so presence-plus-fired
    # is the enablement test. Deliberately NOT `REGISTRY.is_enabled`:
    # `length_band_check`'s config key only controls whether it scores a penalty
    # (default false) — the gate itself always runs and its out-of-band flag is
    # exactly what we want to repair.
    for layer in ("L0", "L1"):
        for gate, action in ACTION_BY_GATE.items():
            if REGISTRY.repair(gate) != layer:
                continue
            if not gate_fired(gate_result(review, gate)):
                continue
            if action == "expand_to_band" and not _length_band_needs_expand(review):
                continue
            if action in seen_actions:
                continue
            seen_actions.add(action)
            steps.append({"gate": gate, "layer": layer, "action": action})

    # style_health's em-dash flag gets an L1 escalation, but only as a follow-up
    # to the L0 attempt — never on its own.
    sh = review.get("style_health")
    if isinstance(sh, dict) and any(
        str(f).startswith("em_dash_") for f in (sh.get("flags") or [])
    ):
        if _EM_DASH_L1_ACTION not in seen_actions:
            steps.append({"gate": "style_health", "layer": "L1", "action": _EM_DASH_L1_ACTION})

    # contract_fulfilment is a native check (not in REGISTRY), so the main loop
    # above skips it. Detect CCC failures directly and plan the L1 patch.
    ccc = review.get("contract_fulfilment")
    if isinstance(ccc, dict) and ccc.get("enabled", True) and not ccc.get("passed", True):
        if "ccc_patch" not in seen_actions:
            steps.append({"gate": "contract_fulfilment", "layer": "L1",
                          "action": "ccc_patch"})
            seen_actions.add("ccc_patch")

    l0 = [s for s in steps if s["layer"] == "L0"]
    l1 = [s for s in steps if s["layer"] == "L1"][: max(0, max_l1)]
    return l0 + l1


# ---------------------------------------------------------------------------
# L0 fixers — deterministic, zero LLM
# ---------------------------------------------------------------------------

def reduce_em_dash_if_needed(text: str, config: dict[str, Any] | None = None) -> str:
    """Bring ``——`` density under target, if it is over.

    The single home for what used to be three verbatim copies of the same
    measure-then-reduce block in `pipeline.py` (candidate single-draft path,
    pre-review path, pre-save path).
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not text or not bool(cfg.get("em_dash_reduce_enabled", True)):
        return text
    try:
        density = float(style_health(text, config).get("metrics", {}).get("em_dash_per_kchar", 0))
        if density > float(cfg.get("em_dash_reduce_target_per_kchar", 1.5)):
            return reduce_em_dash_density(text, config)
    except Exception:
        return text
    return text


def merge_fragment_lines(text: str, config: dict[str, Any] | None = None) -> str:
    """Glue dangling short clause-lines back into sentences.

    `style_health` measures a "fragment line" as a non-dialogue line under 25
    chars that does not end in sentence punctuation — the telegraphic
    one-clause-per-line layout that IS the style-collapse signature. The fix is
    mechanical: attach the dangling clause to the following prose line with a
    comma (or to the preceding one when the follower is dialogue), which is what
    the writer directive asks for in words.

    Never merges into or out of a dialogue line, never crosses the title line,
    and leaves a fragment alone when the merge would produce an unwieldy line.
    """
    if not text:
        return text
    cfg = (config or {}).get("novel", {}) if config else {}
    max_merged = int(cfg.get("fix_merge_max_line_chars", 120))

    raw_lines = text.split("\n")
    # Keep the `第N章 …` title line out of the way.
    start = 0
    for i, ln in enumerate(raw_lines):
        if ln.strip():
            if re.match(r"^#?\s*第.{1,8}章", ln.strip()):
                start = i + 1
            break

    def _is_fragment(s: str) -> bool:
        s = s.strip()
        if not s or len(s) >= 25:
            return False
        if any(q in s for q in _FIX_QUOTES):
            return False
        if s[-1] in _SENTENCE_ENDERS:
            return False
        return s not in ("---", "***")

    def _is_dialogue(s: str) -> bool:
        return any(q in s for q in _FIX_QUOTES)

    lines = list(raw_lines)
    i = start
    while i < len(lines):
        cur = lines[i].strip()
        if not _is_fragment(cur):
            i += 1
            continue
        merged_any = False
        # Absorb following prose lines with a comma until the line reads as a
        # sentence. Forward only: appending a dangling clause to a line that
        # already ended in 。 produces worse prose than it repairs.
        while True:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            nxt = lines[j].strip() if j < len(lines) else ""
            if not nxt or _is_dialogue(nxt) or len(cur) + len(nxt) + 1 > max_merged:
                break
            cur = cur.rstrip("，,、;；") + "，" + nxt.lstrip("，,、")
            del lines[i + 1: j + 1]
            merged_any = True
            if cur[-1] in _SENTENCE_ENDERS or len(cur) >= 25:
                break
        if merged_any:
            # Close the clause we just joined. Only for lines this function
            # actually merged — punctuating every fragment in place would move
            # the metric without improving the prose.
            if cur[-1] not in _SENTENCE_ENDERS and cur[-1] not in "，,、;；：":
                cur += "。"
            lines[i] = cur
        i += 1

    merged = "\n".join(lines)
    return merged if merged.strip() else text


def rotate_fossils(
    text: str,
    review: dict[str, Any],
    config: dict[str, Any] | None = None,
    chapter_num: int = 0,
) -> tuple[str, list[str]]:
    """Replace repeat occurrences of known fossil phrases with rotated variants.

    Reuses `fossil_fix.fix_chapter` and its `FOSSIL_REPLACEMENTS` bank verbatim
    (the same machinery `novel.py fix-fossils` runs post-hoc), applied live to
    the phrases THIS chapter's fossil gates actually named.

    Bank-only by design, and that ceiling is measured: of the 109 distinct
    phrases the fossil gates named across 647 archived reviews, the bank covers
    the generic-cliche subset (`声音压得很低`, 49 firings) while the rest are
    book-specific nouns and names (`老市场街七号` 42, `羽绒服`, `高子昂`).
    Those must NOT be rotated — swapping a street name or a character's name is
    a canon corruption, not a repair — so they stay with the writer avoid-list
    directive that already exists.
    """
    try:
        from commands.fossil_fix import FOSSIL_REPLACEMENTS, fix_chapter
    except Exception:
        return text, []
    cfg = (config or {}).get("novel", {}) if config else {}
    bank = dict(FOSSIL_REPLACEMENTS)
    extra = cfg.get("fix_fossil_replacements")
    if isinstance(extra, dict):
        bank.update(extra)

    # Two targets, because the gates ask different questions. `cross_chapter_repetition`
    # / `descriptor_frequency` complain about DENSITY, so keeping one occurrence and
    # rotating the rest answers them. A hard `book_wide_fossils` reject complains that
    # the phrase recurs across a large fraction of the BOOK, and the ratio is
    # book-cumulative -- `quality.book_wide_fossils` only indicts a chapter that
    # actually contains the phrase (`in_current`), so the only way this chapter turns
    # the gate green is to contain it ZERO times. Measured: 10 of the 12 archived
    # chapters still rejected on an entrenched bank phrase contain it exactly once, so
    # under the shared keep-1 target this fixer replaced nothing at all and the repair
    # declared for the gate could never clear it. LESSONS S13.
    phrases: list[str] = []
    zero_target: set[str] = set()

    def _add(p: Any, *, zero: bool = False) -> None:
        p = str(p or "").strip()
        if p and p in bank:
            if p not in phrases:
                phrases.append(p)
            if zero:
                zero_target.add(p)

    ccr = gate_result(review, "cross_chapter_repetition")
    if isinstance(ccr, dict):
        for item in (ccr.get("repeats") or []) + (ccr.get("template_fossils") or []):
            if isinstance(item, dict):
                clause = str(item.get("clause", ""))
                # A repeated clause is usually longer than a bank phrase; rotate
                # any bank phrase it contains.
                for phrase in bank:
                    if phrase in clause:
                        _add(phrase)
            else:
                _add(item)
    bwf = gate_result(review, "book_wide_fossils")
    if isinstance(bwf, dict):
        for item in bwf.get("hard_fossils") or []:
            if isinstance(item, dict):
                _add(item.get("phrase"), zero=True)
            else:
                _add(item, zero=True)
        for item in bwf.get("fossils") or []:
            if isinstance(item, dict):
                _add(item.get("phrase"))
        for p in bwf.get("phrases") or []:
            _add(p)
    dfq = gate_result(review, "descriptor_frequency")
    if isinstance(dfq, dict):
        for item in dfq.get("flagged") or []:
            if isinstance(item, dict):
                _add(item.get("phrase") or item.get("descriptor"))
            else:
                _add(item)

    if not phrases:
        return text, []
    keep = int(cfg.get("fix_fossil_max_keep", 1))
    groups = [([p for p in phrases if p in zero_target], 0),
              ([p for p in phrases if p not in zero_target], keep)]
    fixed, replaced = text, []
    for group, max_keep in groups:
        if not group:
            continue
        try:
            fixed, stats = fix_chapter(
                fixed,
                [{"phrase": p} for p in group],
                bank,
                max_per_chapter=max_keep,
                chapter_num=chapter_num,
            )
        except Exception:
            return text, []
        replaced += [p for p, s in (stats or {}).items() if int(s.get("replaced", 0)) > 0]
    return (fixed, replaced) if replaced else (text, [])


def promote_action_opening(
    text: str,
    chapter_num: int,
    config: dict[str, Any] | None = None,
) -> str:
    """Demote a scenery-only opening paragraph below the first live paragraph.

    Conservative and gate-verified: only moves ONE short leading paragraph, only
    when `opening_hook_gate` is firing, only behind one of the next two
    paragraphs, and only if re-running the gate afterwards shows the penalty
    gone. Anything else returns the text untouched.
    """
    if not text:
        return text
    try:
        before = opening_hook_gate(text, chapter_num, config)
    except Exception:
        return text
    if float(before.get("penalty", 0.0) or 0.0) <= 0:
        return text

    paras = text.split("\n\n")
    # Locate the first body paragraph (skip a `第N章 …` title paragraph).
    first = 0
    while first < len(paras) and not paras[first].strip():
        first += 1
    if first < len(paras) and re.match(r"^#?\s*第.{1,8}章", paras[first].strip()):
        first += 1
        while first < len(paras) and not paras[first].strip():
            first += 1
    if first + 1 >= len(paras):
        return text
    scenery = paras[first]
    if len(scenery.strip()) > int((config or {}).get("novel", {}).get("fix_opening_move_max_chars", 200)):
        return text

    for offset in (1, 2):
        target = first + offset
        if target >= len(paras):
            break
        reordered = list(paras)
        del reordered[first]
        reordered.insert(target, scenery)
        candidate = "\n\n".join(reordered)
        try:
            after = opening_hook_gate(candidate, chapter_num, config)
        except Exception:
            return text
        if float(after.get("penalty", 0.0) or 0.0) <= 0:
            return candidate
    return text


def apply_l0(
    text: str,
    review: dict[str, Any],
    config: dict[str, Any],
    chapter_num: int = 0,
) -> tuple[str, list[str]]:
    """Run every L0 action the review's fired gates ask for.

    Returns ``(text, applied_action_labels)``. Each action is kept only when it
    did not make `style_health` worse, so a bad transform cannot ship.
    """
    cfg = (config or {}).get("novel", {})
    if not text or not bool(cfg.get("fix_l0_enabled", True)):
        return text, []

    actions = {s["action"] for s in plan_repairs(review, config) if s["layer"] == "L0"}
    applied: list[str] = []
    result = text

    def _penalty(t: str) -> float:
        try:
            return float(style_health(t, config).get("penalty", 0.0) or 0.0)
        except Exception:
            return 0.0

    if "style_prose" in actions:
        base = _penalty(result)
        stage = reduce_em_dash_if_needed(result, config)
        if stage != result:
            applied.append("em_dash_reduce")
            result = stage
        stage = merge_fragment_lines(result, config)
        if stage != result and _penalty(stage) <= base:
            applied.append("merge_fragment_lines")
            result = stage
    if "fossil_rotate" in actions:
        stage, replaced = rotate_fossils(result, review, config, chapter_num)
        if replaced and _penalty(stage) <= _penalty(result):
            applied.append(f"fossil_rotate({len(replaced)})")
            result = stage
    if "opening_promote" in actions:
        stage = promote_action_opening(result, chapter_num, config)
        if stage != result:
            applied.append("opening_promote")
            result = stage

    if "style_prose" in actions or True:
        stage = strip_meta_narrative(result, config)
        if stage != result:
            applied.append("strip_meta_narrative")
            result = stage

    return result, applied


_META_NARRATIVE_CH = re.compile(
    r"(?:在|从|到)((?:Ch|ch)\d+(?:里|中)?)"
)
_META_NARRATIVE_WAITED = re.compile(
    r"等了[一二三四五六七八九十百\d]+章"
)
_META_NARRATIVE_AGO = re.compile(
    r"[一二三四五六七八九十百\d]+章(?:前|以前|之前|以来)"
)
_META_NARRATIVE_NAV = re.compile(
    r"上一章|下一章|前几章|后几章"
)


def strip_meta_narrative(text: str, config: dict[str, Any] | None = None) -> str:
    """L0: remove chapter-number references that leak meta-narrative."""
    title_end = text.find("\n")
    if title_end < 0:
        return text
    title = text[:title_end]
    body = text[title_end:]
    body = _META_NARRATIVE_CH.sub(lambda m: "", body)
    body = _META_NARRATIVE_WAITED.sub("等了很久", body)
    body = _META_NARRATIVE_AGO.sub("之前", body)
    body = _META_NARRATIVE_NAV.sub("之前", body)
    return title + body


# ---------------------------------------------------------------------------
# L1 fixers — one bounded call each, numbered-list splice-back
# ---------------------------------------------------------------------------

_EXPAND_SYSTEM = (
    "你是中文小说编辑。只做一件事：把给定的每个段落扩写得更充实，"
    "补足动作过程、感官细节与必要对白，不引入新人物、新地点或新剧情走向。\n"
    "每段扩写到原长度的 1.5-2.5 倍。保持人称视角、时态与叙事腔调完全不变。\n"
    "严禁使用破折号（——）连接碎句。用完整的主谓宾句子叙事。\n"
    "输出格式：与输入同样的编号列表，每条一行，只输出改写后的段落正文，不要解释。"
)

_DIALOGUE_SYSTEM = (
    "你是中文小说编辑。只做一件事：把给定的每个叙述段落改写为「对白为主」的段落——"
    "把段落里被概述掉的交流写成人物真实说出的话，用中文引号“”，"
    "配以简短的动作/神态提示。\n"
    "不改变段落传达的信息、人物关系与剧情走向，不引入新人物。改写后长度在原段 0.7-2.0 倍以内。\n"
    "严禁使用破折号（——）连接碎句。\n"
    "禁止回声对话：人物说的话必须传递新信息或推动冲突，"
    "不能把叙述已经写过的内容原样变成台词。\n"
    "输出格式：与输入同样的编号列表，每条一行，只输出改写后的段落正文，不要解释。"
)


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def _numbered_rewrite(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    system: str,
    instruction: str,
    passages: list[str],
    *,
    tag: str,
    temperature: float = 0.4,
    min_ratio: float = 0.5,
    max_ratio: float = 3.0,
) -> dict[int, str]:
    """One LLM call that rewrites *passages* as a numbered list; parse it back.

    The same bounded shape as `writing.reduce_em_dashes_targeted`: only the
    extracted passages travel, only per-passage rewrites come back, and each is
    length-checked before it is allowed to splice.
    """
    from engine.llm import call_llm

    numbered = "\n".join(f"{i + 1}. {p.strip()}" for i, p in enumerate(passages))
    user = f"{instruction}\n\n{numbered}"
    raw = call_llm(
        client, paths, config, system, user,
        max_tokens=int(config["novel"].get("fix_l1_max_tokens", 4000)),
        temperature=temperature, tag=tag,
    )
    out: dict[int, str] = {}
    current: int | None = None
    buf: list[str] = []

    def _flush() -> None:
        if current is None:
            return
        body = "".join(x.strip() for x in buf if x.strip()).strip()
        if not body or not (0 <= current < len(passages)):
            return
        orig = passages[current].strip()
        if min_ratio * len(orig) <= len(body) <= max_ratio * len(orig):
            out[current] = body

    for line in (raw or "").splitlines():
        m = re.match(r"\s*(\d{1,2})[.、)]\s*(.*)", line)
        if m:
            _flush()
            current = int(m.group(1)) - 1
            buf = [m.group(2)]
        elif current is not None:
            buf.append(line)
    _flush()
    return out


def _splice(text: str, passages: list[str], rewrites: dict[int, str]) -> str:
    result = text
    for idx, new in rewrites.items():
        orig = passages[idx]
        if orig in result:
            result = result.replace(orig, new, 1)
    return result


def expand_to_band(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> str:
    """Grow an under-length chapter by expanding its thinnest paragraphs.

    Only handles the SHORT side of `length_band_check`: over-length is left to
    the next-chapter directive, because deterministically choosing what to cut
    from a finished chapter is not a repair, it is an edit with plot risk.
    `plan_repairs` enforces that side (`_length_band_needs_expand`); the floor
    check below is the second half of the same guard, for a text that grew after
    the report that planned this step.

    Every exit says why. This is the only repair layer that spends money, and an
    exit that returns the text in silence is indistinguishable from a step that
    was never planned — which is how 56% of this fixer's planned invocations
    turned out to be over-length chapters nobody could see it declining.
    """
    from engine.config import log

    cfg = config["novel"]
    cmin = int(cfg.get("chapter_min_chars", 2500))
    clen = len((text or "").strip())
    if clen == 0:
        log(paths, f"L1 expand skipped Ch{chapter_num}: empty text")
        return text
    if clen >= cmin:
        log(paths, f"L1 expand skipped Ch{chapter_num}: {clen} chars, already at the {cmin} floor")
        return text
    paras = _paragraphs(text)
    body = [p for p in paras if len(p.strip()) >= 30 and not p.strip().startswith("第")]
    if not body:
        log(paths, f"L1 expand skipped Ch{chapter_num}: no paragraph long enough to expand")
        return text
    # Thinnest paragraphs first — those are where the chapter skipped process.
    picks = sorted(body, key=lambda p: len(p))[: int(cfg.get("fix_expand_max_paragraphs", 4))]
    need = cmin - clen
    try:
        rewrites = _numbered_rewrite(
            client, paths, config, _EXPAND_SYSTEM,
            f"本章共 {clen} 字，偏短，需要补足约 {need} 字。扩写以下 {len(picks)} 个段落：",
            # min_ratio 1.0 admits a same-length "expansion", which is churn with
            # no metric gain — but raising the floor can only DROP items from the
            # window, which can only shrink the splice and turn a partial gain
            # into the no-growth discard below. The measured problem is calls
            # that return nothing usable, and a higher floor moves toward it.
            picks, tag="fix_expand", min_ratio=0.85, max_ratio=3.0,
        )
    except Exception as exc:
        log(paths, f"L1 expand call failed Ch{chapter_num} (non-fatal): {exc}")
        return text
    if not rewrites:
        log(paths, f"L1 expand discarded Ch{chapter_num}: the call returned no usable "
                   f"paragraph ({len(picks)} sent, needed +{need} chars)")
        return text
    candidate = _splice(text, picks, rewrites)
    new_len = len(candidate.strip())
    if new_len <= clen:
        log(paths, f"L1 expand discarded Ch{chapter_num}: spliced {len(rewrites)} "
                   f"paragraph(s) but the chapter did not grow ({clen} -> {new_len})")
        return text
    if new_len > int(cfg.get("chapter_max_chars", 7000)):
        log(paths, f"L1 expand rejected Ch{chapter_num}: {clen} -> {new_len} overshoots the band")
        return text
    if float(style_health(candidate, config).get("penalty", 0.0) or 0.0) > float(
        style_health(text, config).get("penalty", 0.0) or 0.0
    ):
        log(paths, f"L1 expand rejected Ch{chapter_num}: style penalty rose")
        return text
    log(paths, f"L1 expand Ch{chapter_num}: {clen} -> {new_len} chars via {len(rewrites)} paragraph(s)")
    return candidate


def inject_dialogue(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> str:
    """Raise a dialogue-starved chapter's dialogue ratio by dialogue-izing narration.

    Picks the longest quote-free paragraphs (the summarized-away exchanges),
    rewrites them as spoken lines in one call, and keeps the result only when
    `dialogue_health`'s measured ratio actually rose and length held.
    """
    from engine.config import log

    cfg = config["novel"]
    before = dialogue_health(text, config)
    before_ratio = float(before.get("metrics", {}).get("dialogue_char_ratio", 0.0) or 0.0)
    target = float(cfg.get("dialogue_char_ratio_target", 0.20))
    if before_ratio >= target:
        log(paths, f"L1 dialogue skipped Ch{chapter_num}: ratio {before_ratio:.2f} "
                   f"already at the {target:.2f} target")
        return text
    picks = [
        p for p in _paragraphs(text)
        if len(p.strip()) >= 60 and not any(q in p for q in "“”")
    ]
    if not picks:
        log(paths, f"L1 dialogue skipped Ch{chapter_num}: no quote-free paragraph "
                   f"long enough to turn into dialogue")
        return text
    picks = sorted(picks, key=lambda p: -len(p))[: int(cfg.get("fix_dialogue_max_paragraphs", 6))]
    try:
        rewrites = _numbered_rewrite(
            client, paths, config, _DIALOGUE_SYSTEM,
            f"本章对白占比 {before_ratio:.0%}，偏低（目标 {target:.0%}）。"
            f"把以下 {len(picks)} 个叙述段落改写为以对白为主的段落：",
            picks, tag="fix_dialogue", min_ratio=0.7, max_ratio=2.0,
        )
    except Exception as exc:
        log(paths, f"L1 dialogue call failed Ch{chapter_num} (non-fatal): {exc}")
        return text
    if not rewrites:
        log(paths, f"L1 dialogue discarded Ch{chapter_num}: the call returned no usable "
                   f"paragraph ({len(picks)} sent)")
        return text
    candidate = _splice(text, picks, rewrites)
    after_ratio = float(
        dialogue_health(candidate, config).get("metrics", {}).get("dialogue_char_ratio", 0.0) or 0.0
    )
    if after_ratio <= before_ratio:
        log(paths, f"L1 dialogue rejected Ch{chapter_num}: ratio {before_ratio:.2f} -> {after_ratio:.2f}")
        return text
    if not (0.85 <= len(candidate) / max(1, len(text)) <= 1.35):
        log(paths, f"L1 dialogue rejected Ch{chapter_num}: length {len(text)} -> {len(candidate)}")
        return text
    log(paths, f"L1 dialogue Ch{chapter_num}: ratio {before_ratio:.2f} -> {after_ratio:.2f}")
    return candidate


_EM_DASH_REWRITE_SYSTEM = (
    "你是中文文本编辑器。只做一件事：把每条带有破折号（——）的句子改写为不含破折号的等长句子。\n"
    "改写时用逗号、句号、分号或完整从句替代破折号，保持原文语义、人称视角和叙事腔调完全不变。\n"
    "每条输出的长度与输入长度差距不超过20%。\n"
    "严格按原编号逐条输出，不要输出其他任何内容。格式：\n"
    "1. 改写后的句子\n"
    "2. 改写后的句子\n"
    "......"
)


def reduce_em_dashes_targeted(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter: str,
    max_sentences: int | None = None,
) -> str:
    """Extract sentences with em-dashes, rewrite them via a focused LLM call, splice back."""
    from engine.config import log as _log
    from engine.llm import call_llm

    if not chapter or "——" not in chapter:
        return chapter

    max_s = max_sentences or int(config["novel"].get("em_dash_targeted_rewrite_max_sentences", 30))

    em_sentences: list[tuple[str, int]] = []
    lines = chapter.split("\n")
    for li, line in enumerate(lines):
        if "——" not in line:
            continue
        parts = re.split(r"(?<=[。！？…])", line)
        for part in parts:
            part = part.strip()
            if "——" in part and len(part) >= 4:
                em_sentences.append((part, li))
                if len(em_sentences) >= max_s:
                    break
        if len(em_sentences) >= max_s:
            break

    if not em_sentences:
        return chapter

    numbered = "\n".join(f"{i+1}. {s}" for i, (s, _) in enumerate(em_sentences))
    user = f"改写以下{len(em_sentences)}条句子，去掉所有破折号（——）：\n\n{numbered}"

    try:
        raw = call_llm(
            client, paths, config, _EM_DASH_REWRITE_SYSTEM, user,
            max_tokens=4000, temperature=0.3, tag="em_dash_fix",
        )
    except Exception as exc:
        _log(paths, f"Targeted em-dash rewrite LLM call failed: {exc}")
        return chapter

    rewrites: dict[int, str] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        m = re.match(r"(\d+)\.\s*(.+)", line)
        if m:
            idx = int(m.group(1)) - 1
            rewritten = m.group(2).strip()
            if 0 <= idx < len(em_sentences) and "——" not in rewritten:
                orig = em_sentences[idx][0]
                if 0.5 * len(orig) <= len(rewritten) <= 2.0 * len(orig):
                    rewrites[idx] = rewritten

    if not rewrites:
        _log(paths, "Targeted em-dash rewrite: no usable rewrites parsed")
        return chapter

    result = chapter
    applied = 0
    for idx, rewritten in rewrites.items():
        orig_sentence = em_sentences[idx][0]
        if orig_sentence in result:
            result = result.replace(orig_sentence, rewritten, 1)
            applied += 1

    if len(result) > len(chapter) * 1.3 or len(result) < len(chapter) * 0.7:
        _log(paths, f"Targeted em-dash rewrite rejected: size {len(chapter)}->{len(result)}")
        return chapter

    _log(paths, f"Targeted em-dash rewrite: applied {applied}/{len(rewrites)} rewrites (of {len(em_sentences)} sentences)")
    return result


def em_dash_targeted(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> str:
    """Escalate a still-too-dense em-dash chapter to the targeted rewrite call."""
    from engine.config import log

    cfg = config["novel"]
    target = float(cfg.get("em_dash_reduce_target_per_kchar", 3.0))
    density = float(style_health(text, config).get("metrics", {}).get("em_dash_per_kchar", 0) or 0)
    if density <= target:
        # The normal path: L0's punctuation pass already got there, and this
        # planned step is a free no-op. Silence made that look like L1 idling.
        log(paths, f"L1 em-dash skipped Ch{chapter_num}: density {density:.1f}/kchar "
                   f"already under the {target:.1f} target")
        return text
    candidate = reduce_em_dashes_targeted(client, paths, config, text)
    if candidate == text:
        return text
    if float(style_health(candidate, config).get("penalty", 0.0) or 0.0) > float(
        style_health(text, config).get("penalty", 0.0) or 0.0
    ):
        log(paths, f"L1 em-dash rewrite rejected Ch{chapter_num}: style penalty rose")
        return text
    return candidate


_HOOK_REVISE_SYSTEM = (
    "你是网文章末改写专家。只做一件事：把弱收尾改写为强钩子。\n"
    "三选一：①对话中抛出新悬念或未解之谜；②用反转/危机的具体动作收束；"
    "③用悬念提问让读者必须翻下一章。\n"
    "要求：最后一段独立成段、<=100字、基于【故事信息】中给出的方向拓展悬念。\n"
    "严禁引入故事信息所列人物/场景以外的任何新元素（人名、地名、物品）。\n"
    "严禁以下收尾：内心感悟、环境描写沉淀、情绪总结、哲理升华。\n"
    "保持原文人称视角、叙事腔调、情节走向完全不变。\n"
    "输出格式：与输入同样的编号列表，每条一行，只输出改写后的段落正文，不要解释。"
)


def hook_revise(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> str:
    """Rewrite a weak chapter ending into a hook via a focused LLM call."""
    from engine.config import log
    from engine.quality_advisory import chapter_ending_strength

    before = chapter_ending_strength(text, config)
    if before.get("has_hook", True):
        log(paths, f"L1 hook_revise skipped Ch{chapter_num}: ending already has hook "
                   f"signals {before.get('signals', [])}")
        return text

    paras = _paragraphs(text)
    if not paras:
        return text
    picks = paras[-2:] if len(paras) >= 2 else paras[-1:]
    picks = [p for p in picks if len(p.strip()) >= 10]
    if not picks:
        log(paths, f"L1 hook_revise skipped Ch{chapter_num}: tail paragraphs too short")
        return text

    card = review.get("_card") or review.get("card") or {}
    context_parts: list[str] = []
    if card:
        who = card.get("who")
        if who:
            context_parts.append(f"- 在场人物：{'、'.join(str(n) for n in who) if isinstance(who, list) else who}")
        where = card.get("where")
        if where:
            context_parts.append(f"- 场景：{where}")
        payoff = card.get("payoff")
        if payoff:
            context_parts.append(f"- 本章爽点：{payoff}")
        exit_hook = card.get("exit_hook")
        if exit_hook:
            context_parts.append(f"- 期望钩子方向：{exit_hook}")
    context_block = ""
    if context_parts:
        context_block = (
            "【故事信息】（改写必须严格遵守，不得编造下列人物/场景以外的任何元素）：\n"
            + "\n".join(context_parts) + "\n\n"
        )

    try:
        rewrites = _numbered_rewrite(
            client, paths, config, _HOOK_REVISE_SYSTEM,
            context_block + f"本章结尾缺少钩子，请把以下 {len(picks)} 个结尾段落改写为强钩子收束：",
            picks, tag="fix_hook", min_ratio=0.3, max_ratio=2.5,
        )
    except Exception as exc:
        log(paths, f"L1 hook_revise call failed Ch{chapter_num} (non-fatal): {exc}")
        return text
    if not rewrites:
        log(paths, f"L1 hook_revise discarded Ch{chapter_num}: no usable rewrite")
        return text

    candidate = _splice(text, picks, rewrites)
    after = chapter_ending_strength(candidate, config)
    if not after.get("has_hook", False):
        log(paths, f"L1 hook_revise rejected Ch{chapter_num}: rewrite still lacks hook")
        return text
    if not (0.85 <= len(candidate) / max(1, len(text)) <= 1.15):
        log(paths, f"L1 hook_revise rejected Ch{chapter_num}: "
                   f"length {len(text)} -> {len(candidate)}")
        return text
    if float(style_health(candidate, config).get("penalty", 0.0) or 0.0) > float(
        style_health(text, config).get("penalty", 0.0) or 0.0
    ):
        log(paths, f"L1 hook_revise rejected Ch{chapter_num}: style penalty rose")
        return text
    log(paths, f"L1 hook_revise Ch{chapter_num}: ending rewritten, "
               f"signals {after.get('signals', [])}")
    return candidate


_CCC_PATCH_SYSTEM = (
    "你是网文内容修补专家。任务：在正文中自然地织入指定的关键要素。\n"
    "要素必须作为具体动作/对白/场景细节出现，而不是干巴巴地提及。\n"
    "保持原文人称视角、叙事腔调、情节走向完全不变。\n"
    "只修改需要织入要素的段落，其余段落原样保留。\n"
    "输出格式：与输入同样的编号列表，每条一行，只输出改写后的段落正文。"
)


def ccc_patch(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> str:
    """L1 fixer for contract_fulfilment: weave missing anchors into prose."""
    from engine.config import log
    from engine.accept import contract_fulfilment

    ccc = (review or {}).get("contract_fulfilment") or {}
    if not isinstance(ccc, dict) or ccc.get("passed", True):
        return text
    hard_misses = ccc.get("hard_misses") or []
    if not hard_misses:
        return text

    hook_misses = [m for m in hard_misses if m.get("field") == "exit_hook"]
    body_misses = [m for m in hard_misses if m.get("field") in ("where", "turn")]

    candidate = text

    if hook_misses and not body_misses:
        candidate = hook_revise(client, paths, config, chapter_num, text, review)
        if candidate != text:
            card = review.get("_card") or review.get("card")
            after = contract_fulfilment(card, candidate, config)
            if after.get("passed", False):
                log(paths, f"L1 ccc_patch Ch{chapter_num}: exit_hook fixed via hook_revise")
                return candidate
            log(paths, f"L1 ccc_patch Ch{chapter_num}: hook_revise applied but CCC still fails, trying anchor weave")
            tail_paras = _paragraphs(candidate)
            tail_picks = [p for p in tail_paras[-3:] if len(p.strip()) >= 30][-2:]
            if tail_picks:
                hook_anchor_desc = "；".join(
                    f"【exit_hook】{m.get('target','')}（关键词：{'、'.join((m.get('anchors') or [])[:3])}）"
                    for m in hook_misses if m.get("anchors")
                )
                if hook_anchor_desc:
                    try:
                        hook_rewrites = _numbered_rewrite(
                            client, paths, config, _CCC_PATCH_SYSTEM,
                            f"以下段落需要自然地织入这些要素：{hook_anchor_desc}\n"
                            f"请改写这 {len(tail_picks)} 个段落，将要素织入情节中：",
                            tail_picks, tag="fix_ccc_hook", min_ratio=0.7, max_ratio=2.0,
                        )
                    except Exception:
                        hook_rewrites = []
                    if hook_rewrites:
                        candidate2 = _splice(candidate, tail_picks, hook_rewrites)
                        after2 = contract_fulfilment(card, candidate2, config)
                        if after2.get("passed", False):
                            log(paths, f"L1 ccc_patch Ch{chapter_num}: exit_hook fixed via anchor weave after hook_revise")
                            return candidate2
                        remaining2 = len(after2.get("hard_misses") or [])
                        if remaining2 < len(hook_misses):
                            candidate = candidate2
                            log(paths, f"L1 ccc_patch Ch{chapter_num}: anchor weave reduced hook misses {len(hook_misses)}->{remaining2}")
                        else:
                            candidate = text
                    else:
                        candidate = text
                else:
                    candidate = text
            else:
                candidate = text

    if body_misses:
        paras = _paragraphs(candidate)
        if not paras:
            return text
        targets = []
        for miss in body_misses:
            target = str(miss.get("target") or "")
            anchors = miss.get("anchors") or []
            field = str(miss.get("field") or "")
            if target:
                targets.append((field, target, anchors))
        if not targets:
            return text
        body_paras = [p for p in paras if len(p.strip()) >= 30
                      and not p.strip().startswith("第")]
        if not body_paras:
            return text
        anchor_desc = "；".join(
            f"【{f}】{t}（关键词：{'、'.join(a[:3])}）"
            for f, t, a in targets if a
        )
        has_where = any(f == "where" for f, _, _ in targets)
        has_turn = any(f == "turn" for f, _, _ in targets)
        mid = len(body_paras) // 2
        if has_where and not has_turn:
            picks = body_paras[:3]
        elif has_turn and not has_where:
            picks = body_paras[max(0, mid - 1): mid + 2][:3]
        else:
            picks = (body_paras[:1] + body_paras[max(1, mid): mid + 2])[:3]
        try:
            rewrites = _numbered_rewrite(
                client, paths, config, _CCC_PATCH_SYSTEM,
                f"以下段落需要自然地织入这些要素：{anchor_desc}\n"
                f"请改写这 {len(picks)} 个段落，将要素织入情节中：",
                picks, tag="fix_ccc", min_ratio=0.7, max_ratio=2.0,
            )
        except Exception as exc:
            log(paths, f"L1 ccc_patch call failed Ch{chapter_num} (non-fatal): {exc}")
            return text
        if not rewrites:
            log(paths, f"L1 ccc_patch Ch{chapter_num}: no usable rewrite for body misses")
            return text
        candidate = _splice(candidate, picks, rewrites)

    if hook_misses and candidate == text:
        hook_candidate = hook_revise(client, paths, config, chapter_num, candidate, review)
        if hook_candidate != candidate:
            candidate = hook_candidate

    if candidate == text:
        log(paths, f"L1 ccc_patch Ch{chapter_num}: no changes produced")
        return text

    card = review.get("_card") or review.get("card")
    after = contract_fulfilment(card, candidate, config)
    remaining = len(after.get("hard_misses") or [])
    original = len(hard_misses)
    if remaining >= original:
        log(paths, f"L1 ccc_patch rejected Ch{chapter_num}: "
                   f"hard_misses {original}->{remaining}")
        return text
    if not (0.85 <= len(candidate) / max(1, len(text)) <= 1.15):
        log(paths, f"L1 ccc_patch rejected Ch{chapter_num}: "
                   f"length {len(text)} -> {len(candidate)}")
        return text
    if float(style_health(candidate, config).get("penalty", 0.0) or 0.0) > float(
        style_health(text, config).get("penalty", 0.0) or 0.0
    ):
        log(paths, f"L1 ccc_patch rejected Ch{chapter_num}: style penalty rose")
        return text
    log(paths, f"L1 ccc_patch Ch{chapter_num}: hard_misses {original}->{remaining}")
    return candidate


_L1_FIXERS = {
    "expand_to_band": expand_to_band,
    "inject_dialogue": inject_dialogue,
    _EM_DASH_L1_ACTION: em_dash_targeted,
    "hook_revise": hook_revise,
    "ccc_patch": ccc_patch,
}


def apply_l1(
    client: Any,
    paths: Any,
    config: dict[str, Any],
    chapter_num: int,
    text: str,
    review: dict[str, Any],
) -> tuple[str, list[str]]:
    """Run up to ``fix_max_l1_calls`` bounded repair calls for fired L1 gates."""
    from engine.config import log

    cfg = (config or {}).get("novel", {})
    if not text or not bool(cfg.get("fix_ladder_enabled", True)):
        return text, []
    applied: list[str] = []
    result = text
    for step in plan_repairs(review, config):
        if step["layer"] != "L1":
            continue
        fixer = _L1_FIXERS.get(step["action"])
        if fixer is None:
            continue
        try:
            candidate = fixer(client, paths, config, chapter_num, result, review)
        except Exception as exc:
            log(paths, f"L1 fix {step['action']} failed Ch{chapter_num} (non-fatal): {exc}")
            continue
        if candidate and candidate != result:
            applied.append(step["action"])
            result = candidate
    return result, applied
