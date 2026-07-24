"""Unit tests for deterministic pure functions.

Run with:  python -m unittest discover -s tests  (from the project root)

These cover the load-bearing non-LLM functions whose failures are silent and
costly: chapter text normalization, the style-collapse penalty, scene-dedupe
similarity, and JSON salvage/repair. They use only the stdlib (unittest) so
they add no dependency.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import normalize_chapter  # noqa: E402
from quality import beat_coverage, plan_visual_payoff_check, reduce_em_dash_density, scene_similarity, style_health  # noqa: E402
from quality import _narrative_pattern_sequence, _sequence_similarity, narrative_pattern_repetition  # noqa: E402
from quality import store_chapter_fingerprint, check_plan_against_fingerprints  # noqa: E402
from quality import prose_texture, emotional_cadence  # noqa: E402
from quality import location_transition  # noqa: E402
from quality import opening_hook_gate, length_band_check, flat_chapter_streak  # noqa: E402
from config import genre_detection_profile, _apply_genre_detection_profile  # noqa: E402
from memory import _recency_aware_state  # noqa: E402
from memory import _contract_to_markdown  # noqa: E402
from pipeline import _apply_force_accept_patches  # noqa: E402
from llm import _enhance_system_prompt, _repair_truncated_json, _resolve_thinking_param, json_prompt, safe_json_loads  # noqa: E402
from writing import _beat_needs_concretization, _first_draft_execution_ledger  # noqa: E402
from writing import _chapter_write_max_tokens  # noqa: E402
from quality import cross_chapter_repetition, descriptor_frequency, genre_adherence, GENRE_KEYWORDS  # noqa: E402


def _make_paths(root):
    """Build a Paths rooted at a temp dir (mirrors QualityDebtPatchTests)."""
    from config import Paths

    return Paths(
        book=root / "book.md",
        state=root / "state.md",
        title=root / "title.txt",
        bible=root / "memory" / "bible.md",
        characters=root / "memory" / "characters.md",
        timeline=root / "memory" / "timeline.md",
        threads=root / "memory" / "threads.md",
        volume_plan=root / "memory" / "volume_plan.md",
        compass=root / "memory" / "compass.md",
        voices=root / "memory" / "voices.md",
        voice=root / "memory" / "voice.md",
        contract=root / "memory" / "contract.md",
        glossary=root / "memory" / "glossary.md",
        chapters_dir=root / "chapters",
        logs_dir=root / "logs",
        database=root / "story_state.db",
    )


class NormalizeChapterTests(unittest.TestCase):
    def test_plain_prose_is_preserved(self):
        text = "第一章 开端\n\n他走进屋子，看见桌上的信。"
        out = normalize_chapter(text)
        self.assertIn("第一章 开端", out)
        self.assertIn("他走进屋子", out)
        self.assertTrue(out.endswith("\n"))

    def test_strips_leading_analysis_block(self):
        text = (
            "<analysis>highest risk: pacing. I will fix it.</analysis>\n"
            "第二章 风起\n\n正文从这里开始。"
        )
        out = normalize_chapter(text)
        self.assertNotIn("highest risk", out)
        self.assertNotIn("<analysis>", out)
        self.assertTrue(out.lstrip().startswith("第二章"))

    def test_strips_leading_thinking_block(self):
        text = "<thinking>let me plan</thinking>\n第三章 标题\n\n内容。"
        out = normalize_chapter(text)
        self.assertNotIn("let me plan", out)
        self.assertTrue(out.lstrip().startswith("第三章"))

    def test_strips_heading_style_self_review(self):
        text = (
            "## 写前自我审查\n本章最大的风险是节奏。\n\n"
            "第四章 标题\n\n真正的正文。"
        )
        out = normalize_chapter(text)
        self.assertNotIn("写前自我审查", out)
        self.assertTrue(out.lstrip().startswith("第四章"))

    def test_strips_english_self_review_before_title(self):
        text = (
            "## Pre-writing Self-Review (in reasoning, not in output)\n\n"
            "### Three Highest Risks:\n"
            "1. Repetition risk.\n\n"
            "第4章 捡漏\n\n真正的正文。"
        )
        out = normalize_chapter(text)
        self.assertNotIn("Pre-writing Self-Review", out)
        self.assertNotIn("Three Highest Risks", out)
        self.assertTrue(out.lstrip().startswith("第4章"))

    def test_strips_fenced_analysis_block(self):
        text = (
            "```analysis\n"
            "## Pre-writing Self-Review\n"
            "risk notes\n"
            "```\n\n"
            "第5章 地下\n\n真正的正文。"
        )
        out = normalize_chapter(text)
        self.assertNotIn("risk notes", out)
        self.assertNotIn("```", out)
        self.assertTrue(out.lstrip().startswith("第5章"))

    def test_does_not_eat_legitimate_prose_before_title(self):
        # No self-review keywords -> a leading paragraph must NOT be deleted.
        text = "这是一段合法的引子文字。\n第五章 标题\n\n正文。"
        out = normalize_chapter(text)
        self.assertIn("这是一段合法的引子文字", out)

    def test_strips_markdown_title_hashes(self):
        text = "# 第六章 标题\n\n正文。"
        out = normalize_chapter(text)
        self.assertTrue(out.lstrip().startswith("第六章"))
        self.assertNotIn("# 第六章", out)

    def test_strips_code_fences(self):
        text = "```markdown\n第七章 标题\n\n正文。\n```"
        out = normalize_chapter(text)
        self.assertNotIn("```", out)
        self.assertIn("第七章", out)


class StyleHealthTests(unittest.TestCase):
    def test_short_text_no_penalty(self):
        res = style_health("太短了。", None)
        self.assertEqual(res["penalty"], 0.0)
        self.assertEqual(res["flags"], [])

    def test_healthy_prose_low_penalty(self):
        # Long, well-formed sentences with dialogue should not be penalized hard.
        para = (
            "他缓步走进大殿，目光扫过群臣的脸庞，心中已有了决断。"
            "“诸位爱卿，今日所议之事，关乎社稷存亡。”他的声音不高，却字字清晰。"
            "殿内一时寂静，只有烛火在风中轻轻摇曳，映出每个人各怀心思的神情。"
        ) * 6
        res = style_health(para, None)
        self.assertLess(res["penalty"], 1.5)

    def test_em_dash_overload_penalized(self):
        collapsed = ("他走——停下——回头——又走——犹豫——再停——" * 40)
        res = style_health(collapsed, None)
        self.assertGreater(res["penalty"], 0.0)
        self.assertTrue(any("em_dash" in f for f in res["flags"]))
        self.assertTrue(res["directives"])

    def test_fragmented_short_sentences_penalized(self):
        # Many tiny non-dialogue fragment lines.
        frag = "\n".join(["他走", "停下", "回头", "犹豫", "风起", "云动"] * 30)
        res = style_health(frag, None)
        self.assertGreater(res["penalty"], 0.0)

    def test_penalty_capped(self):
        collapsed = ("他走——停下——回头——" * 200) + "\n".join(["碎句"] * 200)
        res = style_health(collapsed, None)
        self.assertLessEqual(res["penalty"], 4.0)


class AiFlavorPivotTests(unittest.TestCase):
    """Template-pivot + mechanical-anaphora detection added to ai_flavor_health."""

    cfg = {"novel": {"ai_flavor_enabled": True}}

    def test_affirmative_pivot_flagged(self):
        from quality import ai_flavor_health
        text = (
            "她要的不是同情，而是尊重。与其说她在卖货，不如说她在证明自己。"
            "不仅仅是冻梨，更是这片土地的心意。"
        ) * 8  # exceed the 200-char floor of ai_flavor_health
        res = ai_flavor_health(text, self.cfg)
        self.assertGreater(res["metrics"].get("template_pivot_per_kchar", 0), 0)
        self.assertTrue(any("template_pivot" in f for f in res["flags"]))
        self.assertTrue(res["directives"])

    def test_anaphora_tricolon_flagged(self):
        from quality import ai_flavor_health
        text = ("她想起了父亲，想起了奶奶，想起了那口老铁锅，眼眶悄悄热了。") * 10
        res = ai_flavor_health(text, self.cfg)
        self.assertGreaterEqual(res["metrics"].get("anaphora_longest", 0), 3)
        self.assertTrue(any("anaphora" in f for f in res["flags"]))

    def test_clean_concrete_prose_not_flagged(self):
        from quality import ai_flavor_health
        text = (
            "七个人排成一排在没膝的雪里铲路，奶奶搬把椅子坐屋檐下直播。"
            "陆时砚拎着铁锹，肩上落了层薄雪。她弯腰捡起断掉的粉笔，把箭头画完。"
        ) * 4
        res = ai_flavor_health(text, self.cfg)
        self.assertEqual(res["metrics"].get("template_pivot_per_kchar", 0), 0.0)
        self.assertFalse(any("template_pivot" in f or "anaphora" in f for f in res["flags"]))

    def test_negation_pivot_not_double_counted_as_negative_pair(self):
        # "不是X，而是Y" is a pivot (this check), NOT the negative-pair "不是X，也不是Y".
        from quality import _TEMPLATE_PIVOT, _NEGATIVE_PAIR
        pivot = "这不是运气，而是本事。"
        self.assertTrue(_TEMPLATE_PIVOT.search(pivot))
        self.assertFalse(_NEGATIVE_PAIR.search(pivot))


class SceneDraftSplitTests(unittest.TestCase):
    """_split_beats_into_scenes: ①(b) 密集章分段助手（保序、≤max_segments、~2 beat/段）。"""

    def test_splits_into_at_most_max_segments(self):
        from writing import _split_beats_into_scenes
        beats = [f"b{i}" for i in range(8)]
        groups = _split_beats_into_scenes(beats, 3)
        self.assertLessEqual(len(groups), 3)
        self.assertGreaterEqual(len(groups), 2)
        # 保序 + 无丢失
        self.assertEqual([b for g in groups for b in g], beats)

    def test_five_beats_three_segments(self):
        from writing import _split_beats_into_scenes
        groups = _split_beats_into_scenes([f"b{i}" for i in range(5)], 3)
        self.assertEqual(len(groups), 3)  # ceil(5/2)=3 段

    def test_short_beatlist_not_segmented(self):
        from writing import _split_beats_into_scenes
        self.assertEqual(len(_split_beats_into_scenes(["only"], 3)), 1)
        self.assertEqual(_split_beats_into_scenes([], 3), [])

    def test_blank_beats_dropped(self):
        from writing import _split_beats_into_scenes
        groups = _split_beats_into_scenes(["a", "  ", "", "b", "c", "d"], 3)
        flat = [b for g in groups for b in g]
        self.assertEqual(flat, ["a", "b", "c", "d"])


class SceneSimilarityTests(unittest.TestCase):
    def test_identical_plans_high_similarity(self):
        plan = {"conflict": "夺嫡之争", "payoff": "扳倒权臣", "goal": "掌控兵权",
                "beats": ["设局", "对峙", "反转"]}
        res = scene_similarity(plan, [plan])
        self.assertGreater(res["max_sim"], 0.9)
        self.assertEqual(res["most_similar_to"], 0)

    def test_distinct_plans_low_similarity(self):
        a = {"conflict": "夺嫡之争", "payoff": "扳倒权臣", "goal": "掌控兵权",
             "beats": ["设局", "对峙"]}
        b = {"conflict": "边疆战事", "payoff": "击退外敌", "goal": "守住城池",
             "beats": ["急行军", "夜袭"]}
        res = scene_similarity(a, [b])
        self.assertLess(res["max_sim"], 0.5)

    def test_empty_recent_plans(self):
        res = scene_similarity({"conflict": "x"}, [])
        self.assertEqual(res["max_sim"], 0.0)
        self.assertIsNone(res["most_similar_to"])


class NarrativePatternTests(unittest.TestCase):
    # The failure scene_similarity is blind to: same procedural flow, totally
    # different concrete subject matter (suspense_10ch Ch3→Ch8 monotony).
    SAME_FLOW_A = {"beats": [
        "周岩进入十八楼机房翻找记录",
        "他取证拍照采集粉尘样本",
        "把数据与限速器日志比对",
        "推断出钢丝绳是被人为割断",
    ]}
    SAME_FLOW_B = {"beats": [
        "周岩开车到金华小区门口",
        "他查看现场提取通讯录照片",
        "把笔迹与签字记录核对",
        "断定签字人另有其人",
    ]}
    DIFFERENT_FLOW = {"beats": [
        "对手先行动尾随周岩",
        "周岩被威胁险些出事",
        "真相反转原来是嫁祸",
        "他摊牌对峙质问凶手",
    ]}

    def test_same_flow_different_subject_is_high_sim(self):
        #字面 Jaccard would rate these LOW (no shared tokens); the abstract
        # move-sequence must rate them HIGH.
        seq_a = _narrative_pattern_sequence(self.SAME_FLOW_A)
        seq_b = _narrative_pattern_sequence(self.SAME_FLOW_B)
        self.assertEqual(seq_a, ["enter_space", "collect_evidence", "compare_data", "deduce_conclusion"])
        self.assertGreaterEqual(_sequence_similarity(seq_a, seq_b), 0.7)
        # And字面 scene_similarity should be FOOLED (proving the new gate is needed).
        self.assertLess(scene_similarity(self.SAME_FLOW_A, [self.SAME_FLOW_B])["max_sim"], 0.5)

    def test_different_flow_is_low_sim(self):
        seq_a = _narrative_pattern_sequence(self.SAME_FLOW_A)
        seq_c = _narrative_pattern_sequence(self.DIFFERENT_FLOW)
        self.assertLess(_sequence_similarity(seq_a, seq_c), 0.4)

    def test_block_on_consecutive_streak(self):
        # Two recent chapters both share the flow → streak == block_streak (2).
        res = narrative_pattern_repetition(
            self.SAME_FLOW_A, [self.SAME_FLOW_B, self.SAME_FLOW_B], {"novel": {}}
        )
        self.assertEqual(res["level"], "block")
        self.assertEqual(res["consecutive"], 2)
        self.assertTrue(res["directives"])

    def test_ok_on_distinct_flow(self):
        res = narrative_pattern_repetition(
            self.DIFFERENT_FLOW, [self.SAME_FLOW_A, self.SAME_FLOW_B], {"novel": {}}
        )
        self.assertEqual(res["level"], "ok")
        self.assertEqual(res["penalty"], 0.0)

    def test_short_sequence_is_ignored(self):
        # A plan with < min_moves recognisable moves carries no flow signal.
        res = narrative_pattern_repetition(
            {"beats": ["周岩走进机房"]}, [self.SAME_FLOW_A], {"novel": {}}
        )
        self.assertEqual(res["level"], "ok")

    def test_disabled_returns_ok(self):
        res = narrative_pattern_repetition(
            self.SAME_FLOW_A, [self.SAME_FLOW_B, self.SAME_FLOW_B],
            {"novel": {"narrative_pattern_enabled": False}},
        )
        self.assertEqual(res["level"], "ok")
        self.assertEqual(res["max_sim"], 0.0)


class VisualPayoffTests(unittest.TestCase):
    def test_abstract_shadow_payoff_is_blocked(self):
        plan = {
            "payoff_type": "reveal",
            "payoff": "沈澜发现阴影方向与光源角度不一致，反推出现场存在第二反射路径。",
            "beats": ["她根据光源方向和几何关系推理出凶手动过镜子。"],
        }
        res = plan_visual_payoff_check(plan, {"novel": {"visual_payoff_min_score": 7.0}})
        self.assertTrue(res["blocked"])
        self.assertIn("abstract_visual_payoff", res["flags"])

    def test_concrete_visual_contradiction_passes(self):
        plan = {
            "payoff_type": "reveal",
            "payoff": "临终画面里林知夏左手戴着方形金属手表，但现实尸体左手垂落且手腕没有手表，压痕也消失。",
            "beats": [
                "沈澜描摹手腕压痕，确认死前画面有表。",
                "罗鹤检查尸体左手，现实中没有手表也没有表带链节。",
                "她用镜中左手与尸体现实左手的有无矛盾推翻高屹作案结论。",
            ],
        }
        res = plan_visual_payoff_check(plan, {"novel": {"visual_payoff_min_score": 7.0}})
        self.assertFalse(res["blocked"])
        self.assertGreaterEqual(res["score"], 7.0)
        self.assertIn("presence_absence", res["template_hits"])


class FirstDraftExecutionLedgerTests(unittest.TestCase):
    def test_ledger_keeps_global_rules_without_per_beat_duplication(self):
        plan = {
            "beats": [
                "沈澜把验尸单压在桌沿，对照两处伤口位置逼罗鹤改口。",
                "她推导出镜子被人动过。",
            ]
        }
        out = _first_draft_execution_ledger({"novel": {"chapter_words": 4000}}, plan)
        self.assertIn("首稿页面执行账本", out)
        self.assertIn("节奏预算", out)
        self.assertIn("细节保真", out)
        # Per-beat enumeration moved to the tail-of-prompt acceptance checklist
        # in write_chapter (recency anchor); the ledger must NOT duplicate it.
        self.assertNotIn("beat1", out)

    def test_ledger_can_be_disabled(self):
        plan = {"beats": ["她发现证词矛盾。"]}
        out = _first_draft_execution_ledger(
            {"novel": {"first_draft_execution_ledger": False}},
            plan,
        )
        self.assertEqual(out, "")

    def test_concretization_heuristic_ignores_action_anchored_beats(self):
        self.assertFalse(_beat_needs_concretization("她把证词摊在桌上，证明罗鹤说谎。"))
        self.assertTrue(_beat_needs_concretization("她意识到证词存在矛盾。"))


class BeatCoverageTests(unittest.TestCase):
    """Deterministic beat-coverage gate (quality.beat_coverage)."""

    @staticmethod
    def _body(extra: str = "") -> str:
        # >500 chars of filler so the short-text auto-pass doesn't trigger.
        return "第10章 残响\n\n" + ("林夕沿着走廊往前走，灯光在地面投下长长的影子。" * 20) + extra

    def test_realized_beat_passes_exact(self):
        plan = {"beats": [
            "林夕发现安瓿碎裂方向与针孔方向矛盾。",
        ]}
        body = self._body("她蹲下身，注意到安瓿碎裂方向朝外，而针孔方向却指向床头——两者矛盾。")
        report = beat_coverage(body, plan, {"novel": {}})
        self.assertTrue(report["passed"])
        self.assertEqual(report["missing_beats"], [])

    def test_reworded_beat_passes_via_bigram_fallback(self):
        plan = {"beats": ["她检查药箱搭扣上的指纹划痕。"]}
        # "药箱的搭扣" rewords "药箱搭扣"; bigram coverage should still hit.
        body = self._body("她俯身检查药箱的搭扣，指纹划痕在灯下清晰可见。")
        report = beat_coverage(body, plan, {"novel": {}})
        self.assertTrue(report["passed"])

    def test_abstract_beat_auto_passes(self):
        plan = {"beats": ["她意识到自己可能错了。"]}
        report = beat_coverage(self._body(), plan, {"novel": {}})
        self.assertTrue(report["passed"])

    def test_short_text_auto_passes(self):
        plan = {"beats": ["林夕发现安瓿碎裂方向矛盾。"]}
        report = beat_coverage("太短", plan, {"novel": {}})
        self.assertTrue(report["passed"])

    def test_disabled_via_config(self):
        plan = {"beats": ["林夕发现安瓿碎裂方向矛盾。"]}
        report = beat_coverage(self._body(), plan, {"novel": {"beat_coverage_enabled": False}})
        self.assertFalse(report["enabled"])
        self.assertTrue(report["passed"])

class QualityDebtPatchTests(unittest.TestCase):
    def test_force_accept_patches_land_without_llm(self):
        import shutil
        from pathlib import Path
        from config import Paths

        root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "manual_tmp_test" / "quality_debt_patch"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            ckpt = root / "logs" / "checkpoints"
            paths = Paths(
                book=root / "book.md",
                state=root / "state.md",
                title=root / "title.txt",
                bible=root / "memory" / "bible.md",
                characters=root / "memory" / "characters.md",
                timeline=root / "memory" / "timeline.md",
                threads=root / "memory" / "threads.md",
                volume_plan=root / "memory" / "volume_plan.md",
                compass=root / "memory" / "compass.md",
                voices=root / "memory" / "voices.md",
                voice=root / "memory" / "voice.md",
                contract=root / "memory" / "contract.md",
                glossary=root / "memory" / "glossary.md",
                chapters_dir=root / "chapters",
                logs_dir=root / "logs",
                database=root / "story_state.db",
            )
            ckpt.mkdir(parents=True, exist_ok=True)
            chapter = "第一章 断绝\n\n周窈看清了。\n"
            review = {
                "score": 7.8,
                "patches": [{
                    "op": "replace",
                    "locator": "周窈看清了",
                    "before": "周窈看清了",
                    "after": "周窈知道那条手腕上应该有什么",
                }],
            }
            patched, new_review = _apply_force_accept_patches(
                paths,
                {"novel": {"quality_debt_apply_patches": True}},
                1,
                chapter,
                review,
            )
            self.assertIn("周窈知道那条手腕上应该有什么", patched)
            self.assertEqual(new_review["quality_debt_patches_applied"], 1)
            self.assertTrue((root / "logs" / "checkpoints" / "ch0001" / "quality_debt_patched.md").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class JsonSalvageTests(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(safe_json_loads('{"a": 1}'), {"a": 1})

    def test_json_with_code_fence(self):
        self.assertEqual(safe_json_loads('```json\n{"a": 2}\n```'), {"a": 2})

    def test_json_embedded_in_prose(self):
        out = safe_json_loads('这是结果：{"score": 8} 谢谢')
        self.assertEqual(out["score"], 8)

    def test_repair_truncated_object(self):
        truncated = '{"title": "第一章", "score": 9, "beats": ["a", "b"'
        repaired = _repair_truncated_json(truncated)
        self.assertIsNotNone(repaired)
        import json
        data = json.loads(repaired)
        self.assertEqual(data["title"], "第一章")
        self.assertEqual(data["score"], 9)

    def test_truncated_recovered_via_safe_loads(self):
        truncated = '{"title": "第二章", "items": [1, 2, 3'
        data = safe_json_loads(truncated)
        self.assertEqual(data["title"], "第二章")

    def test_unrecoverable_raises(self):
        import json
        with self.assertRaises(json.JSONDecodeError):
            safe_json_loads("这里完全没有 JSON 对象")

class PromptEnhancementTests(unittest.TestCase):
    def test_default_enhancement_injects_global_and_tag_blocks(self):
        system = _enhance_system_prompt(
            "base system",
            {"api": {}, "novel": {}},
            tag="plan_candidate",
            wants_json=True,
        )
        self.assertIn("全局提示词纪律", system)
        self.assertIn("JSON 任务额外纪律", system)
        self.assertIn("规划/仲裁任务额外纪律", system)

    def test_enhancement_can_be_disabled(self):
        system = _enhance_system_prompt(
            "base system",
            {"api": {"prompt_enhancement_enabled": False}, "novel": {}},
            tag="write",
            wants_json=False,
        )
        self.assertEqual(system, "base system")

    def test_json_prompt_marker_matches_enhancement_detection(self):
        user = json_prompt("please return data")
        wants_json = "强制 JSON 输出格式" in user
        system = _enhance_system_prompt("base system", {"api": {}, "novel": {}}, tag="", wants_json=wants_json)
        self.assertTrue(wants_json)
        self.assertIn("JSON 任务额外纪律", system)


class BookConsistencyTests(unittest.TestCase):
    """config.book_is_consistent decides whether the resume path can skip the
    O(n) rebuild_book. It must be conservative: consistent only when book.md
    demonstrably contains the latest chapter."""

    def _setup(self):
        import shutil
        import tempfile
        from pathlib import Path
        from config import write_text

        root = Path(tempfile.mkdtemp(prefix="book_consist_"))
        paths = _make_paths(root)
        paths.chapters_dir.mkdir(parents=True, exist_ok=True)
        return root, paths, write_text, shutil

    def test_consistent_book_skips_rebuild(self):
        from config import book_is_consistent
        root, paths, write_text, shutil = self._setup()
        try:
            ch1 = "第一章\n\n内容甲。\n"
            ch2 = "第二章\n\n内容乙，结尾在这里。\n"
            write_text(paths.chapters_dir / "0001.md", ch1)
            write_text(paths.chapters_dir / "0002.md", ch2)
            write_text(paths.book, ch1.strip() + "\n\n" + ch2.strip() + "\n")
            self.assertTrue(book_is_consistent(paths))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_missing_latest_chapter_triggers_rebuild(self):
        from config import book_is_consistent
        root, paths, write_text, shutil = self._setup()
        try:
            ch1 = "第一章\n\n内容甲。\n"
            ch2 = "第二章\n\n内容乙，结尾在这里。\n"
            write_text(paths.chapters_dir / "0001.md", ch1)
            write_text(paths.chapters_dir / "0002.md", ch2)
            # book.md is stale: it only has chapter 1 (the latest append was lost).
            write_text(paths.book, ch1.strip() + "\n")
            self.assertFalse(book_is_consistent(paths))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_missing_book_file_triggers_rebuild(self):
        from config import book_is_consistent
        root, paths, write_text, shutil = self._setup()
        try:
            write_text(paths.chapters_dir / "0001.md", "第一章\n\n内容。\n")
            self.assertFalse(book_is_consistent(paths))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_no_chapters_is_consistent(self):
        from config import book_is_consistent
        root, paths, write_text, shutil = self._setup()
        try:
            write_text(paths.book, "something\n")
            self.assertTrue(book_is_consistent(paths))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class RetrievalShardTests(unittest.TestCase):
    """retrieval.py sharded index must produce the same merged structure the old
    monolithic file did, stay idempotent per chapter, and load a legacy file."""

    def _setup(self):
        import shutil
        import tempfile
        from pathlib import Path
        root = Path(tempfile.mkdtemp(prefix="retr_shard_"))
        paths = _make_paths(root)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        return root, paths, shutil

    def test_index_and_merge(self):
        import retrieval
        root, paths, shutil = self._setup()
        try:
            retrieval._INDEX_CACHE.clear()
            retrieval.index_chapter(paths, 1, "周窈走进密室，发现一枚铜钥匙。\n\n墙上有血迹。")
            retrieval.index_chapter(paths, 2, "罗鹤在码头等待那艘货船，铜钥匙在他口袋里。")
            data = retrieval._load_index(paths)
            self.assertIsNotNone(data)
            self.assertEqual(sorted(data["chapters"]), [1, 2])
            self.assertGreater(len(data["passages"]), 0)
            self.assertEqual(data["n_docs"], len(data["passages"]))
            self.assertIn("df", data)
            # Shard files exist; no monolithic file written.
            self.assertTrue((paths.logs_dir / "retrieval_index" / "ch0001.json").exists())
            self.assertTrue((paths.logs_dir / "retrieval_index" / "_df.json").exists())
            self.assertFalse((paths.logs_dir / "retrieval_index.json").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_index_idempotent(self):
        import retrieval
        root, paths, shutil = self._setup()
        try:
            retrieval._INDEX_CACHE.clear()
            retrieval.index_chapter(paths, 1, "周窈走进密室，发现一枚铜钥匙。")
            before = retrieval._load_index(paths)
            n_before = before["n_docs"]
            retrieval.index_chapter(paths, 1, "完全不同的文本不应被重新索引。")
            after = retrieval._load_index(paths)
            self.assertEqual(after["n_docs"], n_before)
            self.assertEqual(sorted(after["chapters"]), [1])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_retrieve_returns_old_chapter(self):
        import retrieval
        root, paths, shutil = self._setup()
        try:
            retrieval._INDEX_CACHE.clear()
            retrieval.index_chapter(paths, 1, "铜钥匙藏在密室墙后的暗格里。")
            for n in range(2, 9):
                retrieval.index_chapter(paths, n, f"第{n}章无关内容，讲述别的事。")
            hits = retrieval.retrieve(paths, "铜钥匙 密室 暗格", top_k=3, exclude_recent_chapters=3, current_chapter=8)
            self.assertTrue(any(h["chapter"] == 1 for h in hits))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_legacy_monolithic_fallback_and_migration(self):
        import json
        import retrieval
        root, paths, shutil = self._setup()
        try:
            retrieval._INDEX_CACHE.clear()
            # Hand-write a legacy monolithic index (pre-shard format).
            passages, df_inc = retrieval._passages_for_chapter(1, "旧版单体索引中的第一章文本。")
            legacy = {
                "passages": passages,
                "df": df_inc,
                "chapters": [1],
                "n_docs": len(passages),
            }
            (paths.logs_dir / "retrieval_index.json").write_text(
                json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
            )
            # Reading with no shards present must fall back to the monolithic file.
            data = retrieval._load_index(paths)
            self.assertEqual(sorted(data["chapters"]), [1])
            # Indexing a new chapter triggers migration to shards.
            retrieval.index_chapter(paths, 2, "新版分片中的第二章。")
            retrieval._INDEX_CACHE.clear()
            merged = retrieval._load_index(paths)
            self.assertEqual(sorted(merged["chapters"]), [1, 2])
            self.assertTrue((paths.logs_dir / "retrieval_index" / "ch0001.json").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ThreadLocalConnTests(unittest.TestCase):
    """store.ThreadLocalConn gives each thread its own sqlite connection; db_lock
    is now a no-op context manager."""

    def test_db_lock_is_noop_contextmanager(self):
        import store
        with store.db_lock():
            pass  # must enter/exit cleanly without serializing

    def test_init_db_returns_threadlocal_conn(self):
        import shutil
        import tempfile
        from pathlib import Path
        import store
        if store.sqlite3 is None:
            self.skipTest("sqlite3 unavailable")
        root = Path(tempfile.mkdtemp(prefix="tlc_"))
        paths = _make_paths(root)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            conn = store.init_db(paths)
            self.assertIsInstance(conn, store.ThreadLocalConn)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_concurrent_writes_from_threads(self):
        import shutil
        import tempfile
        import threading
        from pathlib import Path
        import store
        if store.sqlite3 is None:
            self.skipTest("sqlite3 unavailable")
        root = Path(tempfile.mkdtemp(prefix="tlc_cc_"))
        paths = _make_paths(root)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            conn = store.init_db(paths)
            errors = []

            def worker(base):
                try:
                    for i in range(10):
                        store.db_event(conn, base + i, "story_event", {"i": i})
                    conn.close_current()
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(b,)) for b in (0, 100, 200)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            events = store.recent_events(conn, 100)
            self.assertEqual(len(events), 30)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ChapterTitleTests(unittest.TestCase):
    def test_apply_replaces_only_title_keeps_body(self):
        from package import apply_chapter_title
        text = "第12章 旧标题\n\n正文第一行。\n正文第二行。\n"
        out = apply_chapter_title(text, 12, "新钩子标题")
        self.assertTrue(out.startswith("第12章 新钩子标题"))
        self.assertIn("正文第一行。", out)
        self.assertIn("正文第二行。", out)
        self.assertNotIn("旧标题", out)

    def test_apply_chinese_numeral_prefix(self):
        from package import apply_chapter_title
        text = "第三章 起\n\n内容。"
        out = apply_chapter_title(text, 3, "暗涌")
        self.assertTrue(out.startswith("第三章 暗涌"))
        self.assertIn("内容。", out)

    def test_apply_noop_when_no_title_line(self):
        from package import apply_chapter_title
        text = "没有章节标记的正文。\n第二行。"
        self.assertEqual(apply_chapter_title(text, 1, "X"), text)

    def test_apply_noop_when_empty_title(self):
        from package import apply_chapter_title
        text = "第1章 标题\n\n正文。"
        self.assertEqual(apply_chapter_title(text, 1, ""), text)
        self.assertEqual(apply_chapter_title(text, 1, "   "), text)


class PackageRenderTests(unittest.TestCase):
    def test_render_package_md_sections(self):
        from package import _render_package_md
        pkg = {
            "one_line": "一句话卖点",
            "titles": ["书名一", "书名二"],
            "intros": ["简介一"],
            "tags": [["标签A", "标签B"]],
            "synopsis_clean": "无剧透简介内容",
            "synopsis_spoiler": "含剧透概要内容",
        }
        md = _render_package_md(pkg)
        self.assertIn("一句话卖点", md)
        self.assertIn("书名一", md)
        self.assertIn("标签A、标签B", md)
        self.assertIn("无剧透简介", md)
        self.assertIn("含剧透概要内容", md)

    def test_render_skips_absent_sections(self):
        from package import _render_package_md
        md = _render_package_md({"titles": ["仅书名"]})
        self.assertIn("仅书名", md)
        self.assertNotIn("无剧透简介", md)


class ReduceEmDashDensityTests(unittest.TestCase):
    """Tests for the programmatic em-dash density reduction (Layer 3)."""

    EM = "——"
    LQ = "“"
    RQ = "”"

    def test_no_change_when_below_target(self):
        text = "这是一段正常的文字，没有破折号。" * 10
        self.assertEqual(reduce_em_dash_density(text), text)

    def test_chained_fragments_replaced(self):
        em = self.EM
        seg = "他站在原地" + em + "沉默" + em + "犹豫" + em + "最终转身离开。这是一个漫长的夜晚。"
        text = seg * 5
        result = reduce_em_dash_density(text, target_per_kchar=1.0)
        self.assertLess(result.count(em), text.count(em))

    def test_dialogue_preserved(self):
        em = self.EM
        lq, rq = self.LQ, self.RQ
        dialogue_line = lq + "不要" + em + rq + "她喊道。"
        narrative_line = "他缓缓转身" + em + "目光扫过每一个人" + em + "最终停在她身上。"
        text = (dialogue_line + "\n" + narrative_line + "\n") * 5
        result = reduce_em_dash_density(text, target_per_kchar=1.0)
        self.assertIn(lq + "不要" + em + rq, result)

    def test_density_reaches_target(self):
        em = self.EM
        base = "简短句子。" * 20
        em_heavy = "他看到" + em + "远方" + em + "火焰" + em + "浓烟" + em + "一切都在燃烧。"
        text = base + (em_heavy + "普通文字。") * 8
        target = 2.0
        result = reduce_em_dash_density(text, target_per_kchar=target)
        density = result.count(em) / (len(result) / 1000) if len(result) > 0 else 0
        self.assertLessEqual(density, target + 0.5)

    def test_empty_and_no_em_dash(self):
        self.assertEqual(reduce_em_dash_density(""), "")
        self.assertEqual(reduce_em_dash_density("普通文字"), "普通文字")

    def test_respects_config_target(self):
        em = self.EM
        text = ("他" + em + "她" + em + "它" + em + "我" + em + "你" + em + "他们。") * 10
        cfg = {"novel": {"em_dash_reduce_target_per_kchar": "5.0"}}
        result = reduce_em_dash_density(text, config=cfg)
        density = result.count(em) / (len(result) / 1000) if len(result) > 0 else 0
        self.assertLessEqual(density, 6.0)



class ResolveThinkingParamTests(unittest.TestCase):
    """Tests for _resolve_thinking_param (thinking mode config resolution)."""

    def test_mode_disabled(self):
        result = _resolve_thinking_param({"thinking_mode": "disabled"})
        self.assertEqual(result, {"type": "disabled"})

    def test_mode_auto(self):
        result = _resolve_thinking_param({"thinking_mode": "auto"})
        self.assertIsNone(result)

    def test_mode_enabled_no_budget(self):
        result = _resolve_thinking_param({"thinking_mode": "enabled"})
        self.assertEqual(result, {"type": "enabled"})

    def test_mode_enabled_with_budget(self):
        result = _resolve_thinking_param({"thinking_mode": "enabled", "thinking_budget_tokens": 10000})
        self.assertEqual(result, {"type": "enabled", "budget_tokens": 10000})

    def test_mode_enabled_zero_budget_omitted(self):
        result = _resolve_thinking_param({"thinking_mode": "enabled", "thinking_budget_tokens": 0})
        self.assertEqual(result, {"type": "enabled"})

    def test_legacy_disabled_true(self):
        result = _resolve_thinking_param({"thinking_disabled": True})
        self.assertEqual(result, {"type": "disabled"})

    def test_legacy_disabled_false(self):
        result = _resolve_thinking_param({"thinking_disabled": False})
        self.assertIsNone(result)

    def test_legacy_disabled_string_true(self):
        result = _resolve_thinking_param({"thinking_disabled": "true"})
        self.assertEqual(result, {"type": "disabled"})

    def test_legacy_disabled_string_false(self):
        result = _resolve_thinking_param({"thinking_disabled": "false"})
        self.assertIsNone(result)

    def test_mode_overrides_legacy(self):
        result = _resolve_thinking_param({"thinking_mode": "enabled", "thinking_disabled": True, "thinking_budget_tokens": 5000})
        self.assertEqual(result, {"type": "enabled", "budget_tokens": 5000})

    def test_default_disabled(self):
        result = _resolve_thinking_param({})
        self.assertEqual(result, {"type": "disabled"})

    def test_default_disabled_false(self):
        result = _resolve_thinking_param({}, default_disabled=False)
        self.assertIsNone(result)

    def test_reviewer_keys(self):
        result = _resolve_thinking_param(
            {"review_thinking_mode": "enabled", "review_thinking_budget_tokens": 8000},
            mode_key="review_thinking_mode",
            disabled_key="review_thinking_disabled",
            budget_key="review_thinking_budget_tokens",
        )
        self.assertEqual(result, {"type": "enabled", "budget_tokens": 8000})

    def test_budget_string_parsed(self):
        result = _resolve_thinking_param({"thinking_mode": "enabled", "thinking_budget_tokens": "16000"})
        self.assertEqual(result, {"type": "enabled", "budget_tokens": 16000})


class RecencyAwareStateTests(unittest.TestCase):
    """Tests for _recency_aware_state (Feature 4: memory budget truncation)."""

    def test_no_chapter_sections(self):
        raw = "# 进度\n- 总字数：5000\n## 主角状态\n详情"
        result = _recency_aware_state(raw, {"novel": {}})
        self.assertEqual(result, raw)

    def test_keeps_recent_n_sections(self):
        header = "# 进度\n- 总字数：10000\n\n"
        sections = "".join(f"## Ch{i}\n- thread_{i} open\n\n" for i in range(1, 11))
        raw = header + sections
        result = _recency_aware_state(raw, {"novel": {"state_recent_chapters": "3"}})
        self.assertIn("# 进度", result)
        self.assertNotIn("## Ch1\n", result)
        self.assertNotIn("## Ch7\n", result)
        self.assertIn("## Ch8\n", result)
        self.assertIn("## Ch9\n", result)
        self.assertIn("## Ch10\n", result)

    def test_keeps_all_when_fewer_than_n(self):
        header = "# 进度\n"
        sections = "## Ch1\n- a\n\n## Ch2\n- b\n"
        raw = header + sections
        result = _recency_aware_state(raw, {"novel": {"state_recent_chapters": "5"}})
        self.assertIn("## Ch1", result)
        self.assertIn("## Ch2", result)

    def test_respects_max_chars(self):
        header = "# 进度\n" * 50
        sections = "## Ch1\n- a\n## Ch2\n- b\n"
        raw = header + sections
        result = _recency_aware_state(raw, {"novel": {}}, max_chars=200)
        self.assertLessEqual(len(result), 220)

    def test_default_recent_5(self):
        header = "# 进度\n"
        sections = "".join(f"## Ch{i}\n- data\n" for i in range(1, 21))
        raw = header + sections
        result = _recency_aware_state(raw, {"novel": {}})
        self.assertNotIn("## Ch15", result)
        self.assertIn("## Ch16", result)
        self.assertIn("## Ch20", result)


class ChapterFingerprintTests(unittest.TestCase):
    """Tests for store/check chapter fingerprints (Feature 3)."""

    def setUp(self):
        import sqlite3
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapter_fingerprints (
                chapter INTEGER PRIMARY KEY,
                skeleton_tokens TEXT NOT NULL,
                narrative_moves TEXT NOT NULL,
                payoff_type TEXT,
                conflict_type TEXT,
                created_at TEXT NOT NULL
            );
        """)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_db_lock(self):
        import contextlib
        @contextlib.contextmanager
        def _noop_lock():
            yield
        return _noop_lock

    def test_store_and_check_identical(self):
        import store
        orig_lock = store.db_lock
        store.db_lock = self._mock_db_lock()
        try:
            plan = {
                "conflict": "发现密室中的血迹方向不对",
                "payoff": "推翻原有的死亡时间结论",
                "pressure": "凶手即将离开城市",
                "goal": "锁定真正的死亡时间",
                "beats": ["进入密室检查", "发现血迹喷溅角度异常", "对比法医报告", "推翻原结论"],
                "payoff_type": "reveal",
                "conflict_type": "evidence_contradiction",
            }
            store_chapter_fingerprint(self.conn, 1, plan)
            rows = self.conn.execute("SELECT * FROM chapter_fingerprints").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 1)
            result = check_plan_against_fingerprints(self.conn, plan, {"novel": {}})
            self.assertGreater(result["max_sim"], 0.8)
            self.assertEqual(result["most_similar_chapter"], 1)
        finally:
            store.db_lock = orig_lock

    def test_different_plan_low_similarity(self):
        import store
        orig_lock = store.db_lock
        store.db_lock = self._mock_db_lock()
        try:
            plan1 = {
                "conflict": "发现密室中的血迹方向不对",
                "payoff": "推翻原有的死亡时间结论",
                "goal": "锁定真正的死亡时间",
                "beats": ["进入密室检查", "发现血迹异常", "对比法医报告"],
            }
            plan2 = {
                "conflict": "公司财务报表出现异常",
                "payoff": "揭露内部贪腐网络",
                "goal": "追踪资金流向",
                "beats": ["调取银行记录", "发现关联账户", "约谈知情人"],
            }
            store_chapter_fingerprint(self.conn, 1, plan1)
            result = check_plan_against_fingerprints(self.conn, plan2, {"novel": {}})
            self.assertLess(result["max_sim"], 0.5)
        finally:
            store.db_lock = orig_lock

    def test_empty_db_returns_zero(self):
        result = check_plan_against_fingerprints(self.conn, {"conflict": "test"}, {"novel": {}})
        self.assertEqual(result["max_sim"], 0.0)
        self.assertIsNone(result["most_similar_chapter"])
        self.assertEqual(result["directives"], [])

    def test_directives_generated_above_threshold(self):
        import store
        orig_lock = store.db_lock
        store.db_lock = self._mock_db_lock()
        try:
            plan = {
                "conflict": "调查古墓里的线索",
                "payoff": "发现古墓的秘密",
                "goal": "解开古墓谜题",
                "beats": ["进入古墓", "发现壁画", "解读符号", "找到密室"],
            }
            store_chapter_fingerprint(self.conn, 1, plan)
            result = check_plan_against_fingerprints(
                self.conn, plan, {"novel": {"fingerprint_warn_threshold": "0.5"}}
            )
            self.assertTrue(len(result["directives"]) > 0)
        finally:
            store.db_lock = orig_lock


class ReviewerCalibrationTests(unittest.TestCase):
    """Tests for the three-layer reviewer calibration (optimization #6).

    These test the numerical behavior of the calibration logic by simulating
    the same variable flow as review_chapter's scoring pipeline.
    """

    def _simulate_scoring(self, raw_score, sh_penalty, prose_score,
                          mismatch=False, rep_em=0.0, det_em=0.0,
                          config_overrides=None):
        """Simulate the review_chapter scoring pipeline with calibration.

        Returns (final_score, prose_score, calibrations).
        """
        config = {"novel": {}}
        if config_overrides:
            config["novel"].update(config_overrides)

        caps = [10.0]
        penalties = 0.0
        calibrations = []

        # style_health penalty (existing)
        penalties += sh_penalty

        # Layer B: prose calibration
        if bool(config["novel"].get("prose_calibration_enabled", True)):
            if sh_penalty == 0 and prose_score < 6.0:
                calibrations.append(f"prose raised {prose_score}→6.0")
                prose_score = 6.0
            elif sh_penalty >= 1.0 and prose_score > 7.5:
                calibrations.append(f"prose lowered {prose_score}→7.5")
                prose_score = 7.5

        # Layer C: mismatch penalty
        if mismatch:
            mm_pen = float(config["novel"].get("style_audit_mismatch_penalty", 0.5))
            if mm_pen > 0:
                penalties += mm_pen
                calibrations.append(f"mismatch +{mm_pen}")

        # Layer A: deterministic floor
        det_floor = float(config["novel"].get("deterministic_score_floor", 5.0))
        if sh_penalty == 0 and raw_score < det_floor:
            calibrations.append(f"floor {raw_score}→{det_floor}")
            raw_score = det_floor

        final = max(1.0, min(min(caps), raw_score) - penalties)
        return final, prose_score, calibrations

    def test_layer_a_floors_catastrophic_score(self):
        """When style_health is clean (penalty=0), raw_score can't go below 5.0."""
        final, _, cals = self._simulate_scoring(
            raw_score=1.0, sh_penalty=0, prose_score=7.0)
        self.assertGreaterEqual(final, 5.0)
        self.assertTrue(any("floor" in c for c in cals))

    def test_layer_a_no_floor_when_penalty(self):
        """When style_health has penalty, floor doesn't apply."""
        final, _, cals = self._simulate_scoring(
            raw_score=3.0, sh_penalty=1.5, prose_score=6.0)
        self.assertLess(final, 3.0)
        self.assertFalse(any("floor" in c for c in cals))

    def test_layer_b_raises_prose_when_healthy(self):
        """Healthy text (penalty=0) can't have prose < 6.0."""
        _, prose, cals = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=4.0)
        self.assertEqual(prose, 6.0)
        self.assertTrue(any("prose raised" in c for c in cals))

    def test_layer_b_lowers_prose_when_collapsed(self):
        """Collapsed text (penalty>=1.0) can't have prose > 7.5."""
        _, prose, cals = self._simulate_scoring(
            raw_score=8.0, sh_penalty=1.5, prose_score=9.0)
        self.assertEqual(prose, 7.5)
        self.assertTrue(any("prose lowered" in c for c in cals))

    def test_layer_b_no_change_in_range(self):
        """Prose in valid range stays unchanged."""
        _, prose, cals = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=7.0)
        self.assertEqual(prose, 7.0)
        self.assertFalse(any("prose" in c for c in cals))

    def test_layer_c_mismatch_penalty(self):
        """When mismatch detected, 0.5 penalty applied."""
        final_no_mm, _, _ = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=7.0, mismatch=False)
        final_mm, _, cals = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=7.0, mismatch=True)
        self.assertAlmostEqual(final_no_mm - final_mm, 0.5)
        self.assertTrue(any("mismatch" in c for c in cals))

    def test_layer_c_configurable(self):
        """Mismatch penalty is configurable."""
        final, _, _ = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=7.0, mismatch=True,
            config_overrides={"style_audit_mismatch_penalty": "1.0"})
        self.assertAlmostEqual(final, 6.0)

    def test_all_layers_combined(self):
        """All three layers work together correctly."""
        # raw=2.0, penalty=0, prose=4.0, mismatch=True
        # Layer A: raw 2.0→5.0 (penalty=0 floor)
        # Layer B: prose 4.0→6.0 (penalty=0 healthy)
        # Layer C: +0.5 mismatch
        # Final: 5.0 - 0.0 - 0.5 = 4.5
        final, prose, cals = self._simulate_scoring(
            raw_score=2.0, sh_penalty=0, prose_score=4.0, mismatch=True)
        self.assertAlmostEqual(final, 4.5)
        self.assertEqual(prose, 6.0)
        self.assertEqual(len(cals), 3)

    def test_disabled_by_config(self):
        """Calibration can be disabled."""
        _, prose, cals = self._simulate_scoring(
            raw_score=7.0, sh_penalty=0, prose_score=4.0,
            config_overrides={"prose_calibration_enabled": False})
        self.assertEqual(prose, 4.0)


class ProseTextureTests(unittest.TestCase):
    """Tests for prose_texture: quantitative vs poetic balance detection."""

    def test_balanced_text(self):
        text = "他缓步走进大殿，目光扫过群臣的脸庞，心中已有了决断。" * 20
        result = prose_texture(text)
        self.assertEqual(result["balance"], "balanced")
        self.assertEqual(result["directives"], [])

    def test_over_quantitative(self):
        # Use number-heavy text WITHOUT sensory single chars (温/湿/冰/热 etc)
        text = ("报告显示第3区有17%的偏差，数值37.5比正常高出2.3，"
                "总计42个站点中有15个达到百分之十五的偏离率。" * 20)
        result = prose_texture(text)
        self.assertEqual(result["balance"], "over_quantitative")
        self.assertTrue(len(result["directives"]) > 0)
        self.assertIn("数据密度", result["directives"][0])

    def test_over_poetic(self):
        text = ("她的目光像是一道光芒，温暖如春风，仿佛整个世界都在阴影中苏醒。"
                "气味芬芳似花园，触感如丝绸般柔滑，声响恍若远方的钟声回荡。" * 15)
        result = prose_texture(text)
        self.assertEqual(result["balance"], "over_poetic")
        self.assertTrue(len(result["directives"]) > 0)
        # egregious purple prose (poetic_density well above 12) now carries a score
        # penalty, capped at texture_poetic_penalty_cap (default 1.5)
        self.assertGreater(result["penalty"], 0.0)
        self.assertLessEqual(result["penalty"], 1.5)

    def test_balanced_prose_has_no_texture_penalty(self):
        text = "他推开门，走进屋里，看了看四周，把手册放在桌上。" * 30
        result = prose_texture(text)
        self.assertEqual(result.get("penalty", 0.0), 0.0)

    def test_metrics_present(self):
        text = "正常的叙事文字。" * 50
        result = prose_texture(text)
        self.assertIn("num_per_kchar", result["metrics"])
        self.assertIn("metaphor_per_kchar", result["metrics"])
        self.assertIn("sensory_per_kchar", result["metrics"])
        self.assertIn("poetic_density", result["metrics"])

    def test_empty_text(self):
        result = prose_texture("")
        self.assertEqual(result["balance"], "balanced")

    def test_config_thresholds(self):
        # Text with moderate number density (~5/kchar) and zero poetic
        text = "共计5个站点偏差3%。他走到门口，看了看四周。" * 30
        result_strict = prose_texture(text, {"novel": {"texture_num_high_per_kchar": "2.0"}})
        result_loose = prose_texture(text, {"novel": {"texture_num_high_per_kchar": "999.0"}})
        self.assertEqual(result_strict["balance"], "over_quantitative")
        self.assertEqual(result_loose["balance"], "balanced")


class EmotionalCadenceTests(unittest.TestCase):
    """Tests for emotional_cadence: consecutive same-mood detection."""

    def test_no_tones(self):
        result = emotional_cadence([])
        self.assertFalse(result["monotony"])
        self.assertEqual(result["streak"], 0)

    def test_single_tone(self):
        result = emotional_cadence(["紧张"])
        self.assertFalse(result["monotony"])

    def test_varied_tones(self):
        result = emotional_cadence(["紧张", "温情", "压抑", "兴奋"])
        self.assertFalse(result["monotony"])
        self.assertEqual(result["streak"], 1)

    def test_monotony_detected(self):
        result = emotional_cadence(["温情", "紧张", "紧张", "紧张"])
        self.assertTrue(result["monotony"])
        self.assertEqual(result["streak"], 3)
        self.assertTrue(len(result["directives"]) > 0)
        self.assertIn("紧张", result["directives"][0])

    def test_configurable_max_same(self):
        tones = ["紧张", "紧张"]
        result_strict = emotional_cadence(tones, {"novel": {"emotional_cadence_max_same": "2"}})
        result_loose = emotional_cadence(tones, {"novel": {"emotional_cadence_max_same": "5"}})
        self.assertTrue(result_strict["monotony"])
        self.assertFalse(result_loose["monotony"])

    def test_alternatives_for_known_tones(self):
        result = emotional_cadence(["悲伤", "悲伤", "悲伤"])
        self.assertTrue(result["monotony"])
        directive = result["directives"][0]
        self.assertTrue("希望" in directive or "温情" in directive or "坚定" in directive)


class LongSpanFatigueTests(unittest.TestCase):
    """long_span_fatigue's tension-flatness must ignore the lazy-extraction
    fingerprint (exactly-constant tension) but still catch a genuine flat arc."""

    def _db_with_tensions(self, tensions, payoff="reveal"):
        import shutil, tempfile
        from pathlib import Path
        import store
        if store.sqlite3 is None:
            self.skipTest("sqlite3 unavailable")
        root = Path(tempfile.mkdtemp(prefix="lsf_"))
        self._roots.append(root)
        paths = _make_paths(root)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        conn = store.init_db(paths)
        with store.db_lock():
            for i, t in enumerate(tensions, start=1):
                conn.execute(
                    "INSERT INTO chapter_metrics(chapter, payoff_type, tension, emotional_tone, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (i, payoff, t, f"tone{i}", "2026-01-01T00:00:00"),
                )
        return conn

    def setUp(self):
        self._roots = []

    def tearDown(self):
        import shutil
        for r in self._roots:
            shutil.rmtree(r, ignore_errors=True)

    def test_constant_tension_suppresses_flat_penalty(self):
        from quality import long_span_fatigue
        conn = self._db_with_tensions([9, 9, 9, 9, 9, 9])
        res = long_span_fatigue(conn, 7, {"novel": {}})
        self.assertFalse(any("tension_flat" in f for f in res["flags"]))

    def test_genuine_low_variation_still_flags(self):
        from quality import long_span_fatigue
        conn = self._db_with_tensions([8, 9, 9, 8, 9, 8])
        res = long_span_fatigue(conn, 7, {"novel": {}})
        self.assertTrue(any("tension_flat" in f for f in res["flags"]))

    def test_payoff_monotony_uses_payoff_type_monotony_max_fallback(self):
        # config sets only payoff_type_monotony_max (not chapter_type_monotony_max);
        # long_span_fatigue must honour it via the fallback.
        from quality import long_span_fatigue
        conn = self._db_with_tensions([8, 9, 7, 8, 9, 7], payoff="reveal")
        res = long_span_fatigue(conn, 7, {"novel": {"payoff_type_monotony_max": 3}})
        self.assertTrue(any("payoff_type_monotony" in f for f in res["flags"]))


class LocationTransitionTests(unittest.TestCase):
    """副本/scene-entry detector: flag a genuine location change, not room moves
    that share a place-noun."""

    def test_same_venue_rooms_not_new(self):
        # all 便利店 rooms share '便利店' -> continuation, not a new副本
        cur = {"location": "顺安便利店夜班"}
        recent = [{"location": "便利店收银台"}, {"location": "便利店储物间"}]
        self.assertFalse(location_transition(cur, recent, {"novel": {}})["is_new"])

    def test_new_dungeon_is_flagged(self):
        cur = {"location": "通宵自习室（教学楼B座203室）"}
        recent = [{"location": "顺安便利店"}, {"location": "便利店门口"}]
        r = location_transition(cur, recent, {"novel": {}})
        self.assertTrue(r["is_new"])
        self.assertEqual(r["location"], "通宵自习室")

    def test_no_history_not_new(self):
        # Ch1 (no prior plans) must not fire — opening craft handles it
        self.assertFalse(location_transition({"location": "自习室"}, [], {"novel": {}})["is_new"])

    def test_missing_or_short_location_not_new(self):
        self.assertFalse(location_transition({"location": ""}, [{"location": "便利店"}], {"novel": {}})["is_new"])
        self.assertFalse(location_transition({}, [{"location": "便利店"}], {"novel": {}})["is_new"])

    def test_exact_same_location_not_new(self):
        self.assertFalse(
            location_transition({"location": "便利店"}, [{"location": "便利店"}], {"novel": {}})["is_new"]
        )

    def test_threshold_configurable(self):
        # a lenient threshold (0.0) never flags new; a strict one (0.99) always does
        cur = {"location": "顺安便利店夜班"}
        recent = [{"location": "便利店收银台"}]
        # raise min_shared so the '便利店' overlap no longer counts, and force strict sim
        cfg = {"novel": {"scene_entry_min_shared_chars": 99, "scene_entry_sim_threshold": 0.99}}
        self.assertTrue(location_transition(cur, recent, cfg)["is_new"])


class RelationshipStoreTests(unittest.TestCase):
    """Tests for character_relationships table and helpers."""

    def setUp(self):
        import sqlite3
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS character_relationships (
                pair_key TEXT PRIMARY KEY,
                char_a TEXT NOT NULL,
                char_b TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT 'contact',
                intensity REAL DEFAULT 0.0,
                label TEXT DEFAULT '',
                last_event TEXT DEFAULT '',
                updated_chapter INTEGER DEFAULT 0,
                history_json TEXT DEFAULT '[]'
            );
        """)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upsert_new_relationship(self):
        from store import upsert_relationship, get_relationships
        upsert_relationship(self.conn, 1, "林夕", "周临舟",
                            stage="tension", intensity=0.6,
                            event_desc="林夕质问周临舟偷改记录")
        rels = get_relationships(self.conn)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["stage"], "tension")
        self.assertAlmostEqual(float(rels[0]["intensity"]), 0.6)
        self.assertEqual(len(rels[0]["history"]), 1)

    def test_upsert_updates_existing(self):
        from store import upsert_relationship, get_relationships
        upsert_relationship(self.conn, 1, "林夕", "周临舟",
                            stage="contact", intensity=0.3, event_desc="初次相遇")
        upsert_relationship(self.conn, 5, "林夕", "周临舟",
                            stage="trust", intensity=0.7, event_desc="共同破案")
        rels = get_relationships(self.conn)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["stage"], "trust")
        self.assertEqual(len(rels[0]["history"]), 2)

    def test_pair_key_order_independent(self):
        from store import upsert_relationship, get_relationships
        upsert_relationship(self.conn, 1, "周临舟", "林夕",
                            stage="contact", event_desc="A")
        upsert_relationship(self.conn, 2, "林夕", "周临舟",
                            stage="tension", event_desc="B")
        rels = get_relationships(self.conn)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["stage"], "tension")

    def test_stale_relationships(self):
        from store import upsert_relationship, get_stale_relationships
        upsert_relationship(self.conn, 1, "A", "B", stage="contact", intensity=0.5)
        upsert_relationship(self.conn, 10, "C", "D", stage="trust", intensity=0.8)
        stale = get_stale_relationships(self.conn, chapter_num=12, stale_threshold=8)
        self.assertEqual(len(stale), 1)
        self.assertIn("A", stale[0]["char_a"] + stale[0]["char_b"])

    def test_invalid_stage_falls_back(self):
        from store import upsert_relationship, get_relationships
        upsert_relationship(self.conn, 1, "A", "B", stage="invalid_stage")
        rels = get_relationships(self.conn)
        self.assertEqual(rels[0]["stage"], "contact")


class InfoRevelationStoreTests(unittest.TestCase):
    """Tests for info_revelations table and helpers."""

    def setUp(self):
        import sqlite3
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS info_revelations (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                reveal_type TEXT NOT NULL DEFAULT 'mystery',
                status TEXT NOT NULL DEFAULT 'planted',
                planted_chapter INTEGER,
                hint_chapters TEXT DEFAULT '[]',
                due_chapter INTEGER,
                revealed_chapter INTEGER,
                importance INTEGER DEFAULT 5,
                created_at TEXT NOT NULL
            );
        """)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upsert_new_revelation(self):
        from store import upsert_info_revelation, get_pending_revelations
        upsert_info_revelation(self.conn, 3, {
            "id": "secret-1",
            "description": "密室里的血迹指向第二嫌疑人",
            "status": "planted",
            "due_chapter": 10,
            "importance": 8,
        })
        pending = get_pending_revelations(self.conn, 5)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], "secret-1")
        self.assertEqual(pending[0]["importance"], 8)

    def test_upsert_updates_status(self):
        from store import upsert_info_revelation, get_pending_revelations
        upsert_info_revelation(self.conn, 3, {
            "id": "secret-2", "description": "隐藏身份",
            "status": "planted", "importance": 7,
        })
        upsert_info_revelation(self.conn, 6, {
            "id": "secret-2", "status": "hinted",
        })
        pending = get_pending_revelations(self.conn, 7)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"], "hinted")

    def test_revealed_not_pending(self):
        from store import upsert_info_revelation, get_pending_revelations
        upsert_info_revelation(self.conn, 3, {
            "id": "secret-3", "description": "已揭秘",
            "status": "planted", "importance": 9,
        })
        upsert_info_revelation(self.conn, 8, {
            "id": "secret-3", "status": "revealed",
        })
        pending = get_pending_revelations(self.conn, 9)
        self.assertEqual(len(pending), 0)

    def test_overdue_revelations(self):
        from store import upsert_info_revelation, get_overdue_revelations
        upsert_info_revelation(self.conn, 2, {
            "id": "overdue-1", "description": "过期线索",
            "status": "planted", "due_chapter": 5, "importance": 7,
        })
        overdue = get_overdue_revelations(self.conn, chapter_num=15, grace=5)
        self.assertEqual(len(overdue), 1)
        self.assertGreater(overdue[0]["overdue_by"], 0)

    def test_not_overdue_within_grace(self):
        from store import upsert_info_revelation, get_overdue_revelations
        upsert_info_revelation(self.conn, 2, {
            "id": "recent-1", "description": "近期线索",
            "status": "planted", "due_chapter": 8, "importance": 5,
        })
        overdue = get_overdue_revelations(self.conn, chapter_num=10, grace=5)
        self.assertEqual(len(overdue), 0)


class BookWideFossilTests(unittest.TestCase):
    """Tests for book_wide_fossils: whole-book micro-phrase tic detection."""

    def _book(self, fossil, n_with, n_without):
        # Per-chapter UNIQUE filler so only `fossil` recurs across chapters;
        # otherwise identical filler would (correctly) be flagged as a fossil too.
        texts = {}
        ch = 1
        for _ in range(n_with):
            uniq = f"第{ch}章独有的过渡叙述编号{ch}{ch}{ch}在此推进剧情向前发展不重复"
            texts[ch] = f"第{ch}章\n{uniq}所以现在，{fossil}。再写一些{uniq}收尾。"
            ch += 1
        for _ in range(n_without):
            uniq = f"第{ch}章完全不同的内容编号{ch}{ch}{ch}叙述其他事件推进情节走向结局"
            texts[ch] = f"第{ch}章\n{uniq}。这一段没有那个口癖。{uniq}收尾。"
            ch += 1
        return texts

    def _has_fossil(self, phrases, fossil):
        # The fossil may surface as a boundary-shifted window; match on a 4-char run.
        return any(
            any(fossil[i:i + 4] in p for i in range(len(fossil) - 3))
            for p in phrases
        )

    def test_detects_book_wide_fossil(self):
        from quality import book_wide_fossils
        # 6-char fossil present in 8 of 10 chapters → above frac 0.30 & min 6
        texts = self._book("陆知白抬起左手", n_with=8, n_without=2)
        res = book_wide_fossils(texts, {"novel": {}})
        self.assertTrue(res["fossils"], "expected at least one book-wide fossil")
        self.assertTrue(self._has_fossil(res["phrases"], "陆知白抬起左手"))
        self.assertTrue(res["directives"])

    def test_below_threshold_not_flagged(self):
        from quality import book_wide_fossils
        # fossil only in 3 of 12 chapters → below both frac and min_chapters(6)
        texts = self._book("陆知白抬起左手", n_with=3, n_without=9)
        res = book_wide_fossils(texts, {"novel": {}})
        self.assertFalse(self._has_fossil(res["phrases"], "陆知白抬起左手"))

    def test_overlapping_windows_collapsed(self):
        from quality import book_wide_fossils
        texts = self._book("陆知白抬起左手", n_with=9, n_without=1)
        res = book_wide_fossils(texts, {"novel": {}})
        # shifted 6-grams of the same stub must not all survive as separate fossils
        for a in range(len(res["phrases"])):
            for b in range(a + 1, len(res["phrases"])):
                pa, pb = res["phrases"][a], res["phrases"][b]
                shared = any(pa[i:i + 4] in pb for i in range(len(pa) - 3))
                self.assertFalse(shared, f"overlapping fossils not collapsed: {pa} / {pb}")

    def test_empty_and_disabled(self):
        from quality import book_wide_fossils
        self.assertEqual(book_wide_fossils({}, {"novel": {}})["fossils"], [])
        texts = self._book("陆知白抬起左手", n_with=8, n_without=2)
        off = book_wide_fossils(texts, {"novel": {"book_fossil_enabled": False}})
        self.assertEqual(off["fossils"], [])

    def test_hard_fossil_by_ratio(self):
        from quality import book_wide_fossils
        # fossil in 8/10 chapters = 0.8 frac, well above default hard_ratio 0.20
        texts = self._book("陆知白抬起左手", n_with=8, n_without=2)
        res = book_wide_fossils(texts, {"novel": {}})
        self.assertTrue(res.get("hard_fossils"), "0.8-frac fossil must be marked hard")
        self.assertTrue(all(f["hard"] for f in res["hard_fossils"]))
        # raise the ratio above the fossil's frac -> no hard fossils, soft still there
        res2 = book_wide_fossils(texts, {"novel": {"book_fossil_hard_ratio": 0.95}})
        self.assertFalse(res2.get("hard_fossils"))
        self.assertTrue(res2["fossils"])  # still detected, just not hard


class EndingZoneTests(unittest.TestCase):
    """Tests for config.ending_zone_distance gradual收束 gating."""

    def _cfg(self, **kw):
        base = {"ending_aware": True, "max_chapters": 50, "ending_zone_chapters": 5}
        base.update(kw)
        return {"novel": base}

    def test_inside_zone(self):
        from config import ending_zone_distance
        self.assertEqual(ending_zone_distance(self._cfg(), 47), 3)
        self.assertEqual(ending_zone_distance(self._cfg(), 46), 4)

    def test_final_chapter_returns_none(self):
        from config import ending_zone_distance
        self.assertIsNone(ending_zone_distance(self._cfg(), 50))  # finale owned by is_final_chapter

    def test_outside_zone(self):
        from config import ending_zone_distance
        self.assertIsNone(ending_zone_distance(self._cfg(), 45))  # remaining=5 == zone, not < zone
        self.assertIsNone(ending_zone_distance(self._cfg(), 30))

    def test_no_max_chapters(self):
        from config import ending_zone_distance
        self.assertIsNone(ending_zone_distance(self._cfg(max_chapters=0), 47))

    def test_ending_aware_off(self):
        from config import ending_zone_distance
        self.assertIsNone(ending_zone_distance(self._cfg(ending_aware=False), 47))

    def test_cost_savings_disabled_in_zone_and_finale(self):
        from config import cost_savings_disabled
        self.assertTrue(cost_savings_disabled(self._cfg(), 50))   # finale
        self.assertTrue(cost_savings_disabled(self._cfg(), 47))   # inside zone
        self.assertFalse(cost_savings_disabled(self._cfg(), 30))  # outside zone
        self.assertFalse(cost_savings_disabled(self._cfg(), 45))  # remaining==zone, not < zone

    def test_cost_savings_disabled_gates(self):
        from config import cost_savings_disabled
        # explicit opt-out
        self.assertFalse(
            cost_savings_disabled(self._cfg(ending_zone_disables_cost_savings=False), 50)
        )
        # pure char-target mode (no max_chapters) never disables
        self.assertFalse(cost_savings_disabled(self._cfg(max_chapters=0), 47))


class RefinedTextAcceptableTests(unittest.TestCase):
    """Tests for refine._refined_text_acceptable intensity-tiered grow ceiling."""

    def _cfg(self, **kw):
        base = {"refine_min_keep_ratio": 0.6, "refine_max_grow_ratio": 1.5}
        base.update(kw)
        return {"novel": base}

    def test_rewrite_allows_more_growth_than_polish(self):
        from refine import _refined_text_acceptable
        original = "字" * 1000
        refined = "字" * 2200  # 2.2x
        ok_polish, _ = _refined_text_acceptable(original, refined, self._cfg(), intensity="polish")
        ok_rewrite, _ = _refined_text_acceptable(original, refined, self._cfg(), intensity="rewrite")
        self.assertFalse(ok_polish, "polish must reject 2.2x growth")
        self.assertTrue(ok_rewrite, "rewrite ceiling (2.5x) must accept 2.2x growth")

    def test_restructure_tier(self):
        from refine import _refined_text_acceptable
        original = "字" * 1000
        refined = "字" * 1900  # 1.9x -> under restructure 2.0x, over polish 1.5x
        ok_restr, _ = _refined_text_acceptable(original, refined, self._cfg(), intensity="restructure")
        ok_polish, _ = _refined_text_acceptable(original, refined, self._cfg(), intensity="polish")
        self.assertTrue(ok_restr)
        self.assertFalse(ok_polish)

    def test_rewrite_still_rejects_extreme_growth(self):
        from refine import _refined_text_acceptable
        original = "字" * 1000
        refined = "字" * 3000  # 3.0x, above rewrite 2.5x
        ok, reason = _refined_text_acceptable(original, refined, self._cfg(), intensity="rewrite")
        self.assertFalse(ok)
        self.assertIn("grew beyond", reason)


class CharacterAppearanceRateTests(unittest.TestCase):
    """Tests for quality.character_appearance_rate + character_names_from_md."""

    def test_names_from_md(self):
        from quality import character_names_from_md
        md = (
            "# 人物状态机档案\n"
            "## 主角：汤舒婷\n### 状态\n文本\n"
            "## 核心配角①：陆时砚（唯一官配）\n文本\n"
            "## 核心反派①：高子昂（前男友）\n文本\n"
            "## Consolidated\n压缩内容\n"
        )
        names = character_names_from_md(md)
        self.assertEqual(names, ["汤舒婷", "陆时砚", "高子昂"])

    def test_appearance_rate_and_under_served(self):
        from quality import character_appearance_rate
        texts = {
            1: "汤舒婷和陆时砚在店里", 2: "汤舒婷一个人",
            3: "汤舒婷想起高子昂", 4: "汤舒婷继续", 5: "汤舒婷收尾陆时砚也在",
        }
        res = character_appearance_rate(["汤舒婷", "陆时砚", "高子昂"], texts, window=5, floor=0.15)
        self.assertEqual(res["rates"]["汤舒婷"], 1.0)
        self.assertEqual(res["rates"]["陆时砚"], 0.4)
        self.assertEqual(res["rates"]["高子昂"], 0.2)
        # nobody under 0.15 here
        self.assertFalse(res["under_served"])

    def test_under_served_flagged(self):
        from quality import character_appearance_rate
        texts = {i: "汤舒婷独角戏" for i in range(1, 21)}
        texts[20] = "汤舒婷独角戏还是没有别人"  # 陆时砚 never appears
        res = character_appearance_rate(["汤舒婷", "陆时砚"], texts, window=15, floor=0.15)
        under = {u["name"] for u in res["under_served"]}
        self.assertIn("陆时砚", under)
        self.assertNotIn("汤舒婷", under)

    def test_empty_inputs(self):
        from quality import character_appearance_rate
        self.assertEqual(character_appearance_rate([], {1: "x"})["rates"], {})
        self.assertEqual(character_appearance_rate(["甲"], {})["rates"], {})


class PayoffDensityTests(unittest.TestCase):
    """Tests for payoff_beat_density: 爽点 cadence."""

    def test_payoff_markers_counted(self):
        from quality import payoff_beat_density
        text = "他当众揭穿了对方的伪证，全场目瞪口呆，对手脸色骤变，败下阵来。" * 5
        res = payoff_beat_density(text, ["reveal"], {"novel": {}})
        self.assertGreater(res["metrics"]["payoff_markers"], 0)

    def test_drought_directive(self):
        from quality import payoff_beat_density
        # recent payoff_types all setup → drought beyond max_gap (1/0.34≈3)
        flat = "他翻看着资料，慢慢整理着思路，又记下了几行笔记。" * 5
        res = payoff_beat_density(flat, ["setup", "setup", "emotional", "setup"], {"novel": {}})
        self.assertTrue(res["directives"])
        self.assertIn("爽点", res["directives"][0])

    def test_recent_strong_payoff_no_drought(self):
        from quality import payoff_beat_density
        flat = "他翻看着资料，慢慢整理着思路。" * 5
        res = payoff_beat_density(flat, ["reveal", "setup", "setup"], {"novel": {}})
        self.assertEqual(res["metrics"]["chapters_since_payoff"], 0)
        self.assertFalse(res["directives"])


class InformationDensityTests(unittest.TestCase):
    """Tests for information_density: pure-transition-chapter detection."""

    def test_transition_chapter_flagged(self):
        from quality import information_density
        text = "他在房间里来回踱步，回想着这些天发生的事，没有结论。" * 5
        plan = {"payoff_type": "setup", "info_reveals": []}
        review = {"beats_audit": [{"status": "absent"}, {"status": "absent"}]}
        res = information_density(text, plan, review, {"novel": {}})
        self.assertTrue(res["low_information"])
        self.assertTrue(res["directives"])

    def test_rich_chapter_not_flagged(self):
        from quality import information_density
        text = "他当众揭穿了伪证，真相大白。" * 5
        plan = {"payoff_type": "reveal", "info_reveals": ["secret-1"]}
        review = {"beats_audit": [{"status": "realized"}]}
        res = information_density(text, plan, review, {"novel": {}})
        self.assertFalse(res["low_information"])

    def test_disabled(self):
        from quality import information_density
        res = information_density("x", {}, {}, {"novel": {"info_density_enabled": False}})
        self.assertFalse(res["low_information"])


class RecentDimensionScoreTests(unittest.TestCase):
    """Tests for store.recent_dimension_scores (dimension de-inflation input)."""

    def setUp(self):
        import sqlite3, tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.conn = sqlite3.connect(os.path.join(self.tmpdir, "t.db"))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE chapter_metrics (
                chapter INTEGER PRIMARY KEY, hook_score REAL, readthrough_score REAL
            );
        """)
        for ch, hk in enumerate([9.5, 10.0, 10.0, 9.8, 10.0, 7.0], start=1):
            self.conn.execute(
                "INSERT INTO chapter_metrics(chapter, hook_score, readthrough_score) VALUES (?,?,?)",
                (ch, hk, hk))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_newest_first(self):
        from store import recent_dimension_scores
        vals = recent_dimension_scores(self.conn, "hook_score", 3)
        self.assertEqual(vals[0], 7.0)  # chapter 6, newest

    def test_before_chapter_excludes(self):
        from store import recent_dimension_scores
        vals = recent_dimension_scores(self.conn, "hook_score", 10, before_chapter=6)
        self.assertNotIn(7.0, vals)
        self.assertEqual(len(vals), 5)

    def test_saturation_average(self):
        from store import recent_dimension_scores
        vals = recent_dimension_scores(self.conn, "hook_score", 5, before_chapter=6)
        self.assertGreaterEqual(sum(vals) / len(vals), 9.3)  # saturated window


class ReplanRoiTests(unittest.TestCase):
    """Tests for pipeline._recent_replan_ineffective ROI breaker."""

    def setUp(self):
        import tempfile
        self.tmpdir = __import__("pathlib").Path(tempfile.mkdtemp())
        self.paths = _make_paths(self.tmpdir)
        (self.tmpdir / "logs").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_replan(self, ch, before, after):
        from checkpoint import save_checkpoint
        save_checkpoint(self.paths, ch, "quality_replan_done.json",
                        {"score_before": before, "score_after": after})

    def test_ineffective_when_recent_gains_small(self):
        from pipeline import _recent_replan_ineffective
        self._write_replan(8, 7.0, 7.1)
        self._write_replan(9, 7.2, 7.0)
        cfg = {"novel": {"replan_max_attempts": 2, "replan_min_gain": 0.3}}
        self.assertTrue(_recent_replan_ineffective(self.paths, 10, cfg))

    def test_effective_when_a_recent_replan_worked(self):
        from pipeline import _recent_replan_ineffective
        self._write_replan(8, 7.0, 7.1)
        self._write_replan(9, 6.5, 8.0)  # big gain
        cfg = {"novel": {"replan_max_attempts": 2, "replan_min_gain": 0.3}}
        self.assertFalse(_recent_replan_ineffective(self.paths, 10, cfg))

    def test_no_history_not_ineffective(self):
        from pipeline import _recent_replan_ineffective
        cfg = {"novel": {"replan_max_attempts": 2, "replan_min_gain": 0.3}}
        self.assertFalse(_recent_replan_ineffective(self.paths, 10, cfg))


class OpeningHookGateTests(unittest.TestCase):
    _BG = (
        "清晨的阳光透过窗帘洒在地板上，空气里浮着淡淡的尘埃。窗外的天空泛着鱼肚白，"
        "微风拂过院子里的老槐树，叶子轻轻摇动。这座小城安静得仿佛还在沉睡，远处的山峦"
        "笼罩在一层薄雾里，看不真切。街道上空无一人，时间仿佛在这一刻凝固，整个世界都"
        "显得格外宁静而悠远，像一幅褪了色的旧画，挂在记忆深处某个无人问津的角落里。"
        "院墙边的老藤一年年绿了又黄，墙根的青苔在湿润的早晨泛着幽幽的光泽。屋檐下"
        "燕子去年筑的旧巢还在，泥点斑驳，无声地诉说着一段又一段被岁月覆盖的寻常日子，"
        "仿佛连风都不忍心惊扰这一方沉静的小院与它漫长而又平淡的清晨时光。"
    )
    _CRISIS = (
        "「住手！」陆江一把抓住对方的手腕，用力往回拽。那人手里的刀离他的喉咙只剩半寸，"
        "血珠已经渗了出来。他来不及多想，膝盖狠狠撞上去，两个人一起摔倒在地。周围的人惊叫着"
        "后退，有人喊着报警。他死死压住那只握刀的手，指节发白，心脏在胸口擂鼓一样狂跳。"
        "刀尖在地砖上划出刺耳的声响，他用尽全身力气把那只手往墙根砸去，一下，两下，"
        "直到那把刀脱手飞出，叮当一声弹到了墙角。他喘着粗气，死死把人按在地上不敢松开。"
    )

    def test_background_opener_penalized_ch1(self):
        res = opening_hook_gate(self._BG, 1, None)
        self.assertGreater(res["penalty"], 0.0)
        self.assertTrue(res["flags"])
        self.assertTrue(res["directives"])

    def test_crisis_opener_not_penalized(self):
        res = opening_hook_gate(self._CRISIS, 1, None)
        self.assertEqual(res["penalty"], 0.0)

    def test_gate_inactive_past_opening_chapters(self):
        res = opening_hook_gate(self._BG, 9, {"novel": {"opening_chapters": 3}})
        self.assertEqual(res["penalty"], 0.0)

    def test_disabled(self):
        res = opening_hook_gate(self._BG, 1, {"novel": {"opening_golden_gate_enabled": False}})
        self.assertEqual(res["penalty"], 0.0)

    def test_block_flag_when_configured(self):
        res = opening_hook_gate(
            self._BG, 1, {"novel": {"opening_golden_gate_block": True}})
        self.assertTrue(res["block"])


class LengthBandCheckTests(unittest.TestCase):
    CFG = {"novel": {"chapter_min_chars": 2200, "chapter_max_chars": 3600,
                     "length_band_penalty_enabled": True}}

    def test_over_length_penalized(self):
        res = length_band_check("字" * 5000, self.CFG)
        self.assertGreater(res["penalty"], 0.0)
        self.assertTrue(any("too_long" in f for f in res["flags"]))

    def test_in_band_clean(self):
        res = length_band_check("字" * 3000, self.CFG)
        self.assertEqual(res["penalty"], 0.0)
        self.assertEqual(res["flags"], [])

    def test_very_short_penalized(self):
        res = length_band_check("字" * 1000, self.CFG)
        self.assertGreater(res["penalty"], 0.0)
        self.assertTrue(any("too_short" in f for f in res["flags"]))

    def test_penalty_off_is_advisory_only(self):
        cfg = {"novel": {"chapter_min_chars": 2200, "chapter_max_chars": 3600,
                         "length_band_penalty_enabled": False}}
        res = length_band_check("字" * 5000, cfg)
        self.assertEqual(res["penalty"], 0.0)
        self.assertTrue(res["directives"])  # still advises

    def test_gross_overshoot_blocks_when_enabled(self):
        cfg = {"novel": {"chapter_min_chars": 2200, "chapter_max_chars": 3600,
                         "length_band_penalty_enabled": True, "length_band_block": True}}
        res = length_band_check("字" * 9000, cfg)  # 2.5x over
        self.assertTrue(res["block"])


class FlatChapterStreakTests(unittest.TestCase):
    CFG = {"novel": {"flat_chapters_max_consecutive": 3, "flat_impact_floor": 5.0}}

    def _flat(self, n):
        return [{"payoff_type": "setup", "emotional_impact": 3.0} for _ in range(n)]

    def test_flat_streak_penalized(self):
        res = flat_chapter_streak(self._flat(3), self.CFG)
        self.assertEqual(res["streak"], 3)
        self.assertGreater(res["penalty"], 0.0)

    def test_recent_strong_payoff_breaks_streak(self):
        rows = [{"payoff_type": "reveal", "emotional_impact": 3.0}] + self._flat(3)
        res = flat_chapter_streak(rows, self.CFG)
        self.assertEqual(res["streak"], 0)
        self.assertEqual(res["penalty"], 0.0)

    def test_high_emotion_breaks_streak(self):
        rows = [{"payoff_type": "setup", "emotional_impact": 8.0}] + self._flat(3)
        res = flat_chapter_streak(rows, self.CFG)
        self.assertEqual(res["streak"], 0)

    def test_disabled(self):
        cfg = {"novel": {"flat_streak_gate_enabled": False}}
        res = flat_chapter_streak(self._flat(5), cfg)
        self.assertEqual(res["penalty"], 0.0)


class GenreDetectionProfileTests(unittest.TestCase):
    def test_shuangwen_vs_suspense_differ(self):
        s = genre_detection_profile("urban_ability")
        m = genre_detection_profile("suspense")
        self.assertEqual(s["opening_gate_mode"], "crisis")
        self.assertEqual(m["opening_gate_mode"], "clue")
        self.assertEqual(s["narrative_mode"], "serial")
        self.assertEqual(m["narrative_mode"], "reasoning")
        # suspense allows longer chapters, slower payoff, higher reading threshold
        self.assertGreater(m["chapter_max_chars"], s["chapter_max_chars"])
        self.assertLess(m["payoff_density_min"], s["payoff_density_min"])
        self.assertTrue(s["style_low_barrier_register"])
        self.assertFalse(m["style_low_barrier_register"])
        # suspense blocks on visual/物证 payoff; 爽文 is advisory
        self.assertTrue(m["visual_payoff_blocks_plan"])
        self.assertFalse(s["visual_payoff_blocks_plan"])

    def test_history_disables_flat_streak_and_romance_strict_cadence(self):
        h = genre_detection_profile("history")
        r = genre_detection_profile("romance_female")
        self.assertFalse(h["flat_streak_gate_enabled"])
        self.assertEqual(r["opening_gate_mode"], "relationship")
        self.assertEqual(r["emotional_cadence_max_same"], 2)

    def test_unknown_preset_is_neutral(self):
        d = genre_detection_profile("totally_unknown")
        self.assertEqual(d["narrative_mode"], "balanced")
        self.assertEqual(d["opening_gate_mode"], "balanced")

    def test_rule_horror_profile_and_aliases(self):
        rh = genre_detection_profile("rule_horror")
        # reasoning core + clue opening + 物证兑现 block (规则怪谈命脉)
        self.assertEqual(rh["narrative_mode"], "reasoning")
        self.assertEqual(rh["opening_gate_mode"], "clue")
        self.assertTrue(rh["visual_payoff_blocks_plan"])
        # faster payoff than pure suspense (抖音爽感), cold 叙事 (no 下沉)
        susp = genre_detection_profile("suspense")
        self.assertGreater(rh["payoff_density_min"], susp["payoff_density_min"])
        self.assertFalse(rh["style_low_barrier_register"])
        # aliases resolve to the same profile
        self.assertEqual(genre_detection_profile("guize"), rh)
        self.assertEqual(genre_detection_profile("infinite_flow"), rh)

    def test_apply_fills_absent_but_never_overrides_explicit(self):
        cfg = {"novel": {"style_preset": "suspense", "chapter_max_chars": 9999}}
        _apply_genre_detection_profile(cfg)
        # explicit value kept
        self.assertEqual(cfg["novel"]["chapter_max_chars"], 9999)
        # absent genre keys filled from the suspense profile
        self.assertEqual(cfg["novel"]["opening_gate_mode"], "clue")
        self.assertEqual(cfg["novel"]["narrative_mode"], "reasoning")
        self.assertFalse(cfg["novel"]["style_low_barrier_register"])


class ContractIronRulesTests(unittest.TestCase):
    def test_iron_rules_rendered_at_top_before_whitelist(self):
        md = _contract_to_markdown({
            "protagonist": "陈九",
            "iron_rules": ["本副本规则必须以编号清单逐条明示", "主角零战力不得亲自打斗"],
            "ability_whitelist": [{"name": "残卷", "modality": "cognitive", "scope": "被动提示", "cost": "命痕"}],
            "ability_blacklist": ["不能主动查阅残卷"],
            "banned_tropes": ["反派降智"],
            "must_hold": ["每章一个强钩子"],
        })
        self.assertIn("## 开写铁律", md)
        self.assertIn("本副本规则必须以编号清单逐条明示", md)
        # iron rules sit at the top: before the ability whitelist heading
        self.assertLess(md.index("## 开写铁律"), md.index("## 能力白名单"))

    def test_absent_iron_rules_omits_section(self):
        md = _contract_to_markdown({
            "protagonist": "某人",
            "ability_blacklist": ["不能飞"],
        })
        self.assertNotIn("## 开写铁律", md)


class OpeningGateModeTests(unittest.TestCase):
    # scenery-shaped first sentence (夜色) + clue markers (规则/不对劲), no action/dialogue
    _CLUE_SCENERY = (
        "夜色像一块浸了水的黑布，沉沉压在这栋废弃疗养院的上空，连一丝风都没有。"
        "走廊尽头的墙上贴着一张泛黄的纸，纸上用红笔写着第一条规则：午夜十二点之后，"
        "无论听见谁敲门，都不要回应，也不要回头。第二条规则被人撕掉了一半，只剩下"
        "几个模糊的字，越看越不对劲。登记簿上整层楼只住了他一个人，可昨晚的脚步声，"
        "分明是从隔壁那间早就空置的病房传来的，一声接一声，踩得很慢，很有耐心。"
        "他翻开值班记录，最后一页的笔迹戛然而止，停在一句没写完的话上：它们最怕的，"
        "其实是有人记得第三条规则——而那一条，整本册子里哪里都找不到，像被谁刻意抹去。"
    )
    # pure scenery, short first sentence (<50), no clue/action/dialogue
    _SCENERY_SHORT_FIRST = (
        "清晨的阳光洒在地板上。空气里浮着淡淡的尘埃，窗外的天空泛着鱼肚白，"
        "微风拂过院子里的老树，叶子轻轻摇动。这座小城安静得仿佛还在沉睡，"
        "远处的山峦笼罩在一层薄雾里，看不真切。街道上空无一人，时间仿佛凝固，"
        "整个世界都显得格外宁静而悠远，像一幅褪了色的旧画，挂在记忆深处的角落里，"
        "檐角的风铃懒懒地响了一声，又归于沉寂，连早起的鸟雀都不知躲到了何处去。"
        "巷口的老槐树下落了一地碎影，光斑随着叶隙缓缓移动，整条街都浸在这片"
        "悠长而平淡的晨光里，像一段被反复擦拭、却始终没有人愿意翻开的旧时光。"
    )

    def test_clue_opening_rescued_in_clue_mode_but_flagged_in_crisis(self):
        clue = opening_hook_gate(self._CLUE_SCENERY, 1, {"novel": {"opening_gate_mode": "clue"}})
        crisis = opening_hook_gate(self._CLUE_SCENERY, 1, {"novel": {"opening_gate_mode": "crisis"}})
        self.assertEqual(clue["penalty"], 0.0)       # 悬疑线索开场被认可
        self.assertGreater(crisis["penalty"], 0.0)   # 爽文 gate 会误伤它

    def test_pure_scenery_flagged_in_clue_and_crisis(self):
        for mode in ("clue", "crisis"):
            r = opening_hook_gate(self._SCENERY_SHORT_FIRST, 1, {"novel": {"opening_gate_mode": mode}})
            self.assertGreater(r["penalty"], 0.0, f"pure scenery should be flagged in {mode}")

    def test_balanced_mode_higher_threshold(self):
        # short-first-sentence pure scenery = 2 signals: flagged at crisis(need2), not balanced(need3)
        bal = opening_hook_gate(self._SCENERY_SHORT_FIRST, 1, {"novel": {"opening_gate_mode": "balanced"}})
        self.assertEqual(bal["penalty"], 0.0)


class ChapterWriteMaxTokensTests(unittest.TestCase):
    def test_derives_from_chapter_max_chars(self):
        small = _chapter_write_max_tokens({"novel": {"chapter_max_chars": 3600}})
        big = _chapter_write_max_tokens({"novel": {"chapter_max_chars": 6000}})
        self.assertIsNotNone(small)
        self.assertGreater(big, small)  # 悬疑 longer band → bigger budget

    def test_disabled_returns_none(self):
        self.assertIsNone(
            _chapter_write_max_tokens({"novel": {"chapter_max_chars": 3600, "chapter_length_cap_enabled": False}}))

    def test_explicit_override_wins(self):
        self.assertEqual(
            _chapter_write_max_tokens({"novel": {"chapter_max_chars": 3600, "write_max_tokens": 5000}}), 5000)

    def test_lower_ratio_is_tighter(self):
        loose = _chapter_write_max_tokens({"novel": {"chapter_max_chars": 3600, "write_token_char_ratio": 1.5}})
        tight = _chapter_write_max_tokens({"novel": {"chapter_max_chars": 3600, "write_token_char_ratio": 1.1}})
        self.assertGreater(loose, tight)

    def test_in_band_chapter_fits_within_budget(self):
        # a complete chapter at the band ceiling (~1 token/char heuristic) should
        # fit under the budget, so it is not truncated mid-sentence.
        cfg = {"novel": {"chapter_max_chars": 3600}}
        budget = _chapter_write_max_tokens(cfg)
        self.assertGreaterEqual(budget, 3600)


class ShuangwenFormulaGateTests(unittest.TestCase):
    """The narrative-pattern gate must catch the 爽文 formula it previously missed."""
    _P1 = {"beats": ["王崇当众羞辱陈砚", "系统结算气运到账", "陈砚当场拆穿打脸", "骑手们目瞪口呆围观"],
           "payoff_type": "faceslap"}
    _P3 = {"beats": ["李刚示众打压", "气运结算解锁技能", "陈砚反杀当场镇住", "众人哗然目瞪口呆"],
           "payoff_type": "faceslap"}

    def test_shuangwen_shape_detected(self):
        seq = _narrative_pattern_sequence(self._P3)
        self.assertIn("humiliation", seq)
        self.assertIn("system_payoff", seq)
        self.assertIn("faceslap", seq)
        self.assertIn("crowd_react", seq)

    def test_repeated_formula_blocks(self):
        r = narrative_pattern_repetition(self._P3, [self._P1], {"novel": {}})
        self.assertEqual(r["level"], "block")
        self.assertGreaterEqual(r["max_sim"], 0.85)
        self.assertTrue(r["directives"])

    def test_different_shape_passes(self):
        diff = {"beats": ["林晚约见谈判", "时间压力倒计时逼近", "两人对峙摊牌", "主动提出交换条件"],
                "payoff_type": "reversal"}
        r = narrative_pattern_repetition(diff, [self._P1], {"novel": {}})
        self.assertNotEqual(r["level"], "block")

    def test_payoff_type_monotony_warns(self):
        # distinct-enough shapes but same payoff_type for 3 chapters → warn
        a = {"beats": ["主角进入仓库勘查", "比对货物记录", "推断出账目造假"], "payoff_type": "reveal"}
        b = {"beats": ["主角约见对手摊牌", "对方威胁恐吓", "主角逼问追问"], "payoff_type": "reveal"}
        c = {"beats": ["主角跟踪尾随目标", "被对方发现险些出事", "主角逃脱"], "payoff_type": "reveal"}
        r = narrative_pattern_repetition(c, [b, a], {"novel": {"payoff_type_monotony_max": 3}})
        self.assertTrue(any("payoff_type_monotony" in f for f in r["flags"]))
        self.assertEqual(r["level"], "warn")  # run of 3 warns, does not block

    # Distinct move-shapes (so move-seq similarity stays low) but all payoff_type
    # 'reveal' — isolates the payoff-monotony axis from the move-seq block axis.
    _REVEAL_RUN = [
        {"beats": ["进入仓库勘查", "比对货物记录", "推断账目造假"], "payoff_type": "reveal"},
        {"beats": ["约见对手摊牌", "对方威胁恐吓", "逼问追问真相"], "payoff_type": "reveal"},
        {"beats": ["跟踪尾随目标", "被对方发现追逐", "翻墙逃脱脱身"], "payoff_type": "reveal"},
        {"beats": ["翻查旧档案室", "拼合残缺信件", "看穿身世秘密"], "payoff_type": "reveal"},
    ]

    def test_payoff_type_monotony_blocks_on_long_run(self):
        # current + 4 recents = run of 5 >= default pt_block(5) → BLOCK (forces retry)
        cur = {"beats": ["蹲守码头暗处", "截获走私交接", "揭发内鬼身份"], "payoff_type": "reveal"}
        r = narrative_pattern_repetition(cur, list(self._REVEAL_RUN), {"novel": {}})
        self.assertEqual(r["level"], "block")
        self.assertTrue(any("payoff_type_monotony" in f for f in r["flags"]))
        self.assertTrue(any("硬性重规划" in d for d in r["directives"]))

    def test_payoff_type_monotony_run_below_block_only_warns(self):
        # run of 4 (< default pt_block 5), distinct shapes → warn, not block
        cur = {"beats": ["蹲守码头暗处", "截获走私交接", "揭发内鬼身份"], "payoff_type": "reveal"}
        r = narrative_pattern_repetition(cur, list(self._REVEAL_RUN[:3]), {"novel": {}})
        self.assertEqual(r["level"], "warn")

    def test_payoff_type_monotony_block_threshold_configurable(self):
        # lowering pt_block to 3 makes a run of 3 block
        cur = {"beats": ["蹲守码头暗处", "截获走私交接", "揭发内鬼身份"], "payoff_type": "reveal"}
        r = narrative_pattern_repetition(
            cur, list(self._REVEAL_RUN[:2]),
            {"novel": {"payoff_type_monotony_block": 3}},
        )
        self.assertEqual(r["level"], "block")

    def test_varied_payoff_types_do_not_block(self):
        # a run where payoff_type actually rotates must NOT trip the monotony block
        recents = [
            {"beats": ["约见对手摊牌", "对方威胁恐吓", "逼问追问真相"], "payoff_type": "reversal"},
            {"beats": ["跟踪尾随目标", "被对方发现追逐", "翻墙逃脱脱身"], "payoff_type": "emotional"},
            {"beats": ["翻查旧档案室", "拼合残缺信件", "看穿身世秘密"], "payoff_type": "reveal"},
            {"beats": ["进入仓库勘查", "比对货物记录", "推断账目造假"], "payoff_type": "reversal"},
        ]
        cur = {"beats": ["蹲守码头暗处", "截获走私交接", "揭发内鬼身份"], "payoff_type": "reveal"}
        r = narrative_pattern_repetition(cur, recents, {"novel": {}})
        self.assertNotEqual(r["level"], "block")


class ChapterTitleDedupeTests(unittest.TestCase):
    @staticmethod
    def _strip(title, n):
        import re as _re
        return _re.sub(r"^\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*章\s*[:：、\-—\s]*", "", title).strip() or f"Chapter {n}"

    def test_strips_duplicate_chapter_prefix(self):
        self.assertEqual(self._strip("第2章：剪辑师的盲区", 2), "剪辑师的盲区")
        self.assertEqual(self._strip("第二章 微笑的标价", 2), "微笑的标价")
        self.assertEqual(self._strip("第10章无声的受力分析", 10), "无声的受力分析")

    def test_clean_title_unchanged(self):
        self.assertEqual(self._strip("微笑的标价", 1), "微笑的标价")

    def test_bare_prefix_falls_back(self):
        self.assertEqual(self._strip("第4章", 4), "Chapter 4")


class TemplateFossilDetectionTests(unittest.TestCase):
    """0A: Template-prefix matching catches variable-suffix fossil clauses."""

    def test_variable_suffix_detected(self):
        base = "每个字都像从牙缝里往外"
        cur = base + "挤，疼得他直冒冷汗。"
        priors = [
            f"一些前文。{base}崩，声音沙哑。一些后文。",
            f"他说不出口，{base}吐，断断续续。",
            f"嗓子像卡了铁丝，{base}蹦，每一声都带血。",
        ]
        cfg = {"novel": {
            "template_fossil_prefix_len": 8,
            "template_fossil_prefix_chapters": 3,
        }}
        r = cross_chapter_repetition(cur, priors, config=cfg, prior_texts_long=priors)
        has_template = any("template_fossil" in f for f in r["flags"])
        self.assertTrue(has_template, f"Expected template_fossil flag, got flags={r['flags']}")

    def test_short_prefix_no_false_positive(self):
        cur = "他笑了笑说，没什么大不了的。"
        priors = ["她笑了笑说，你这人真有趣。"]
        cfg = {"novel": {"template_fossil_prefix_len": 8, "template_fossil_prefix_chapters": 3}}
        r = cross_chapter_repetition(cur, priors, config=cfg, prior_texts_long=priors)
        self.assertEqual(r["level"], "pass")

    def test_below_threshold_no_flag(self):
        base = "每个字都像从牙缝里往外"
        cur = base + "挤。"
        priors = [base + "崩。", "完全不相关的文本内容。"]
        cfg = {"novel": {"template_fossil_prefix_len": 8, "template_fossil_prefix_chapters": 3}}
        r = cross_chapter_repetition(cur, priors, config=cfg, prior_texts_long=priors)
        has_template = any("template_fossil" in f for f in r["flags"])
        self.assertFalse(has_template)


class DescriptorFrequencyTests(unittest.TestCase):
    """0B: Short-phrase (3-6 char) overuse detection across the full book."""

    def test_high_spread_high_density_flagged(self):
        texts = {}
        for i in range(1, 21):
            start = 0x4E00 + i * 80
            unique = "".join(chr(start + j) for j in range(80))
            texts[i] = f"第{i}章\n{unique}虎口旧疤{unique}"
        cfg = {"novel": {
            "descriptor_freq_enabled": True,
            "descriptor_freq_min_spread": 15,
            "descriptor_freq_max_density": 0.5,
            "descriptor_freq_reject_density": 1.0,
        }}
        r = descriptor_frequency(texts, config=cfg)
        phrases = [f["phrase"] for f in r["flagged"]]
        has_target = any("虎口" in p or "旧疤" in p for p in phrases)
        self.assertTrue(
            has_target,
            f"Expected a phrase related to '虎口旧疤' in flagged, got {phrases}",
        )
        self.assertIn(r["level"], ("advise", "reject"))

    def test_below_min_spread_not_flagged(self):
        texts = {}
        for i in range(1, 21):
            body = "正常的文字内容，没有重复的描述。" * 3
            if i <= 10:
                body += "虎口旧疤泛白。"
            texts[i] = f"第{i}章 测试\n{body}"
        cfg = {"novel": {
            "descriptor_freq_enabled": True,
            "descriptor_freq_min_spread": 15,
            "descriptor_freq_max_density": 0.5,
        }}
        r = descriptor_frequency(texts, config=cfg)
        phrases = [f["phrase"] for f in r["flagged"]]
        self.assertFalse(any("虎口旧疤" in p for p in phrases))

    def test_disabled_returns_pass(self):
        texts = {i: "虎口旧疤虎口旧疤" for i in range(1, 30)}
        cfg = {"novel": {"descriptor_freq_enabled": False}}
        r = descriptor_frequency(texts, config=cfg)
        self.assertEqual(r["level"], "pass")
        self.assertEqual(r["flagged"], [])


class GenreAdherenceTests(unittest.TestCase):
    """0C: Genre drift detection via deterministic keyword scoring."""

    def test_pure_negative_triggers_reject(self):
        text = "枪口对准了她。排爆小组赶到冷库。劫持人质的嫌犯持刀械斗。尸体横陈。" * 5
        scores = [-2.0, -1.5, -1.8, -2.1]
        cfg = {"novel": {
            "genre_adherence_enabled": True,
            "style_preset": "romance_female",
            "genre_negative_weight": 2.0,
            "genre_drift_threshold": 0.0,
            "genre_drift_consecutive": 3,
            "genre_drift_reject_consecutive": 5,
        }}
        r = genre_adherence(text, recent_scores=scores, config=cfg)
        self.assertEqual(r["level"], "reject")
        self.assertGreater(r["penalty"], 0)

    def test_positive_text_passes(self):
        text = "她脸红心跳，甜蜜的吻让她整个人都软了。厨房里香味扑鼻，他温柔地说喜欢她做的饭菜。" * 3
        cfg = {"novel": {
            "genre_adherence_enabled": True,
            "style_preset": "romance_female",
            "genre_negative_weight": 2.0,
            "genre_drift_threshold": 0.0,
            "genre_drift_consecutive": 3,
            "genre_drift_reject_consecutive": 5,
        }}
        r = genre_adherence(text, recent_scores=[1.0, 2.0], config=cfg)
        self.assertEqual(r["level"], "pass")
        self.assertGreater(r["genre_score"], 0)

    def test_unknown_preset_passes(self):
        text = "枪口排爆失明截肢" * 10
        cfg = {"novel": {
            "genre_adherence_enabled": True,
            "style_preset": "unknown_genre",
        }}
        r = genre_adherence(text, config=cfg)
        self.assertEqual(r["level"], "pass")

    def test_advise_at_3_consecutive(self):
        text = "尸体横在冷库里。劫持绑架械斗排爆弹孔。" * 3
        scores = [-1.0, -1.5]
        cfg = {"novel": {
            "genre_adherence_enabled": True,
            "style_preset": "romance_female",
            "genre_negative_weight": 2.0,
            "genre_drift_threshold": 0.0,
            "genre_drift_consecutive": 3,
            "genre_drift_reject_consecutive": 5,
        }}
        r = genre_adherence(text, recent_scores=scores, config=cfg)
        self.assertEqual(r["level"], "advise")


class GateRegistryTests(unittest.TestCase):
    """Tests for the quality.GateRegistry infrastructure."""

    def test_all_gates_registered(self):
        from quality import REGISTRY
        gates = REGISTRY.list_gates()
        self.assertGreaterEqual(len(gates), 20)
        for name in ("style_health", "ai_flavor_health", "cross_chapter_repetition",
                      "dialogue_health", "beat_coverage", "scene_similarity"):
            self.assertIn(name, gates)

    def test_function_identity_preserved(self):
        from quality import REGISTRY, style_health, ai_flavor_health
        self.assertIs(REGISTRY.get("style_health"), style_health)
        self.assertIs(REGISTRY.get("ai_flavor_health"), ai_flavor_health)

    def test_is_enabled_default_true(self):
        from quality import REGISTRY
        self.assertTrue(REGISTRY.is_enabled("style_health", {"novel": {}}))

    def test_is_enabled_disabled(self):
        from quality import REGISTRY
        self.assertFalse(REGISTRY.is_enabled("style_health", {"novel": {"style_health_enabled": False}}))

    def test_is_enabled_none_config(self):
        from quality import REGISTRY
        self.assertTrue(REGISTRY.is_enabled("style_health", None))

    def test_is_enabled_unknown_gate(self):
        from quality import REGISTRY
        self.assertTrue(REGISTRY.is_enabled("nonexistent_gate", {"novel": {}}))

    def test_accumulate_penalty_and_flags(self):
        from quality import REGISTRY
        report: dict = {}
        result = {"penalty": 1.5, "flags": ["em_dash_high", "fragment"], "directives": ["reduce em-dashes"]}
        pen = REGISTRY.accumulate(report, result, "style_health", "style")
        self.assertEqual(pen, 1.5)
        self.assertIs(report["style_health"], result)
        self.assertIn("style:em_dash_high", report["rhythm_risks"])
        self.assertIn("style:fragment", report["rhythm_risks"])
        self.assertIn("reduce em-dashes", report["writer_directives_for_next_chapter"])

    def test_accumulate_zero_penalty_no_flags(self):
        from quality import REGISTRY
        report: dict = {}
        result = {"penalty": 0.0, "flags": [], "directives": ["do something"]}
        pen = REGISTRY.accumulate(report, result, "test_gate", "test")
        self.assertEqual(pen, 0.0)
        self.assertNotIn("rhythm_risks", report)
        self.assertIn("do something", report["writer_directives_for_next_chapter"])

    def test_accumulate_deduplicates_directives(self):
        from quality import REGISTRY
        report: dict = {"writer_directives_for_next_chapter": ["existing"]}
        result1 = {"penalty": 0.0, "flags": [], "directives": ["existing", "new"]}
        REGISTRY.accumulate(report, result1, "g1", "g")
        self.assertEqual(report["writer_directives_for_next_chapter"], ["existing", "new"])

    def test_tag_prefix(self):
        from quality import REGISTRY
        self.assertEqual(REGISTRY.tag_prefix("style_health"), "style")
        self.assertEqual(REGISTRY.tag_prefix("ai_flavor_health"), "ai_flavor")
        self.assertEqual(REGISTRY.tag_prefix("cross_chapter_repetition"), "repeat")

    def test_list_gates_phase_filter(self):
        from quality import REGISTRY
        review_gates = REGISTRY.list_gates(phase="review")
        planning_gates = REGISTRY.list_gates(phase="planning")
        self.assertIn("style_health", review_gates)
        self.assertNotIn("scene_similarity", review_gates)
        self.assertIn("scene_similarity", planning_gates)
        self.assertNotIn("style_health", planning_gates)

    def test_get_unknown_returns_none(self):
        from quality import REGISTRY
        self.assertIsNone(REGISTRY.get("nonexistent_gate"))


class RevisionTrackerTests(unittest.TestCase):
    """Tests for RevisionTracker (Phase 5: revision plateau detection)."""

    def _make(self, threshold=8.0, max_no_improvement=1, plateau_window=3, plateau_band=0.3):
        from pipeline import RevisionTracker
        return RevisionTracker(threshold, max_no_improvement, plateau_window, plateau_band)

    def test_converged_immediately(self):
        t = self._make(threshold=7.0)
        self.assertEqual(t.record(8.0, True), "converged")

    def test_converged_requires_accepted(self):
        t = self._make(threshold=7.0)
        self.assertEqual(t.record(8.0, False), "continue")

    def test_stalled_after_no_improvement(self):
        t = self._make(threshold=8.0, max_no_improvement=2)
        self.assertEqual(t.record(6.0, False), "continue")  # round 0, best=6
        self.assertEqual(t.record(5.5, False), "continue")   # no improve 1
        self.assertEqual(t.record(5.0, False), "stalled")    # no improve 2

    def test_improvement_resets_stall_counter(self):
        t = self._make(threshold=8.0, max_no_improvement=2)
        t.record(6.0, False)
        t.record(5.5, False)  # no improve 1
        t.record(6.5, False)  # improve — resets
        t.record(6.0, False)  # no improve 1 again
        self.assertEqual(t.record(5.5, False), "stalled")  # no improve 2

    def test_plateau_detected(self):
        t = self._make(threshold=8.0, max_no_improvement=5, plateau_window=3, plateau_band=0.3)
        t.record(7.0, False)
        t.record(7.2, False)
        self.assertEqual(t.record(7.1, False), "plateau")

    def test_no_plateau_when_spread_wide(self):
        t = self._make(threshold=8.0, max_no_improvement=5, plateau_window=3, plateau_band=0.3)
        t.record(7.0, False)
        t.record(7.5, False)
        self.assertEqual(t.record(7.0, False), "continue")

    def test_plateau_window_respected(self):
        t = self._make(threshold=9.0, max_no_improvement=5, plateau_window=4, plateau_band=0.3)
        t.record(7.0, False)
        t.record(7.1, False)
        t.record(7.2, False)
        self.assertEqual(t.record(7.0, False), "plateau")  # window=4 filled

    def test_converged_takes_priority_over_plateau(self):
        t = self._make(threshold=7.0, plateau_window=3, plateau_band=0.5)
        t.record(6.8, True)
        t.record(7.0, True)
        self.assertEqual(t.record(7.1, True), "converged")

    def test_summary(self):
        t = self._make()
        t.record(6.0, False)
        t.record(7.0, False)
        s = t.summary()
        self.assertEqual(s["scores"], [6.0, 7.0])
        self.assertEqual(s["best_score"], 7.0)
        self.assertEqual(s["rounds"], 2)

    def test_stall_not_triggered_on_round_zero(self):
        t = self._make(threshold=8.0, max_no_improvement=1)
        self.assertEqual(t.record(5.0, False), "continue")


class ReasoningMaxTokensTests(unittest.TestCase):
    """call_llm's reasoning-model max_tokens floor (proactive) + finish=length
    escalation (reactive). Regression guard for the empty-response failure where
    a reasoning model burns a tiny max_tokens budget on hidden CoT and returns
    empty content with finish_reason=length, then retries the identical cap 6x.
    """

    def _run(self, *, thinking_mode, success_at, requested=2000, floor=8000):
        import types
        import tempfile
        from pathlib import Path
        import llm

        received = []

        class _Completions:
            def create(self, **request):
                mt = int(request["max_tokens"])
                received.append(mt)
                content = "OK-CONTENT" if mt >= success_at else ""
                finish = "stop" if content else "length"
                choice = types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content, reasoning_content=""),
                    finish_reason=finish,
                )
                return types.SimpleNamespace(choices=[choice])

        class _Client:
            def __init__(self):
                self.chat = types.SimpleNamespace(completions=_Completions())

        api = {
            "model": "fake-reasoner",
            "temperature": 0.2,
            "max_tokens": 65536,
            "max_attempts": 6,
            "stream": False,
            "thinking_mode": thinking_mode,
            "metrics_enabled": False,
            "reasoning_min_max_tokens": floor,
            "length_empty_retry_factor": 2.0,
            "length_empty_retry_cap": 32000,
        }
        config = {"api": api}
        with tempfile.TemporaryDirectory() as td:
            paths = types.SimpleNamespace(logs_dir=Path(td))
            orig_sleep = llm.time.sleep
            llm.time.sleep = lambda *_a, **_k: None
            try:
                out = llm.call_llm(
                    _Client(), paths, config, "sys", "user",
                    max_tokens=requested, json_mode=False, tag="structural_diagnose",
                )
            finally:
                llm.time.sleep = orig_sleep
        return out, received

    def test_floor_applied_when_reasoning_active(self):
        # thinking auto => tiny 2000 request floored to 8000 on the first attempt.
        out, received = self._run(thinking_mode="auto", success_at=8000)
        self.assertEqual(out, "OK-CONTENT")
        self.assertEqual(received, [8000])

    def test_floor_applied_even_when_thinking_disabled(self):
        # Gateways reason even when we send thinking:disabled, so the floor is
        # UNCONDITIONAL: a 2000 request is still floored to 8000.
        out, received = self._run(thinking_mode="disabled", success_at=8000)
        self.assertEqual(out, "OK-CONTENT")
        self.assertEqual(received, [8000])

    def test_length_empty_escalates_past_floor(self):
        # Still empty+length at the 8000 floor => escalate 8000 -> 16000.
        out, received = self._run(thinking_mode="auto", success_at=16000)
        self.assertEqual(out, "OK-CONTENT")
        self.assertEqual(received, [8000, 16000])

    def test_escalation_is_backstop_with_floor_off(self):
        # Floor disabled (0) isolates the reactive backstop: finish=length empty
        # still escalates 2000 -> 4000 on the next attempt.
        out, received = self._run(thinking_mode="disabled", success_at=4000, floor=0)
        self.assertEqual(out, "OK-CONTENT")
        self.assertEqual(received, [2000, 4000])


class ChapterModeMonotonyTests(unittest.TestCase):
    """Layer 1+2 治本: coarse chapter-mode classification + form-monotony gate.

    Guards the premise/formula-exhaustion fix — yeban_guize collapsed at Ch28
    because ~9 straight chapters were all "智斗解谜" while payoff_type labels
    varied, so the fine-grained gate missed it.
    """

    def _cm(self, cur, recent, **cfg):
        from quality import chapter_mode_monotony
        return chapter_mode_monotony(cur, recent, {"novel": cfg})

    def test_classify_modes(self):
        from quality import _classify_chapter_mode
        self.assertEqual(_classify_chapter_mode({"goal": "通过线索推理识破规则解谜"}), "reasoning")
        self.assertEqual(_classify_chapter_mode({"goal": "追击逃亡", "beats": ["搏斗", "突围反击"]}), "action")
        self.assertEqual(_classify_chapter_mode({"goal": "妹妹牺牲告别", "payoff": "痛哭诀别"}), "emotional")
        self.assertEqual(_classify_chapter_mode({"goal": "信任结盟坦白摊牌"}), "relational")
        self.assertEqual(_classify_chapter_mode({"goal": "揭露幕后阵营布局阴谋"}), "advancement")

    def test_empty_plan_defaults_reasoning(self):
        from quality import _classify_chapter_mode
        self.assertEqual(_classify_chapter_mode({}), "reasoning")
        self.assertEqual(_classify_chapter_mode({"goal": ""}), "reasoning")

    def test_block_on_all_same_mode(self):
        recent = [{"goal": "推理解谜线索"} for _ in range(5)]
        r = self._cm({"goal": "推理识破真相"}, recent)
        self.assertEqual(r["level"], "block")
        self.assertEqual(r["mode"], "reasoning")
        self.assertTrue(any("硬性重规划" in d for d in r["directives"]))

    def test_ok_on_varied_modes(self):
        recent = [
            {"goal": "追击战斗"}, {"goal": "妹妹牺牲痛哭"}, {"goal": "信任结盟坦白"},
            {"goal": "幕后阵营布局"}, {"goal": "休整过渡缓冲"},
        ]
        r = self._cm({"goal": "推理解谜"}, recent)
        self.assertEqual(r["level"], "ok")

    def test_warn_on_majority_not_all(self):
        # 4 reasoning + 1 action in window of 5, current reasoning => 5/6 ~0.83
        # tune thresholds so this lands in warn (>=0.6) not block (>=0.9).
        recent = [{"goal": "推理"}, {"goal": "推理"}, {"goal": "推理"}, {"goal": "追击战斗"}, {"goal": "推理"}]
        r = self._cm({"goal": "推理"}, recent, chapter_mode_block_frac=0.9, chapter_mode_warn_frac=0.6)
        self.assertEqual(r["level"], "warn")
        self.assertTrue(any("建议本章换一种章型" in d for d in r["directives"]))

    def test_below_min_window_no_flag(self):
        # total (recent+current) below min_window => never flags, avoids早期误报.
        r = self._cm({"goal": "推理"}, [{"goal": "推理"}, {"goal": "推理"}])
        self.assertEqual(r["level"], "ok")
        self.assertEqual(r["flags"], [])

    def test_disabled_returns_ok(self):
        recent = [{"goal": "推理解谜"} for _ in range(5)]
        r = self._cm({"goal": "推理"}, recent, chapter_mode_enabled=False)
        self.assertEqual(r["level"], "ok")

    def test_block_frac_configurable(self):
        recent = [{"goal": "推理"} for _ in range(5)]
        # Raise block threshold above 1.0 => can never block, degrades to warn.
        r = self._cm({"goal": "推理"}, recent, chapter_mode_block_frac=1.1)
        self.assertEqual(r["level"], "warn")


if __name__ == "__main__":
    unittest.main()