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
from quality import _beat_anchor_fragments, beat_coverage, plan_visual_payoff_check, scene_similarity, style_health  # noqa: E402
from quality import _narrative_pattern_sequence, _sequence_similarity, narrative_pattern_repetition  # noqa: E402
from pipeline import _apply_force_accept_patches  # noqa: E402
from llm import _enhance_system_prompt, _repair_truncated_json, json_prompt, safe_json_loads  # noqa: E402
from writing import _beat_needs_concretization, _first_draft_execution_ledger  # noqa: E402


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

    def test_missing_concrete_beat_fails(self):
        plan = {"beats": [
            "林夕发现安瓿碎裂方向与针孔方向矛盾，意识到现场被布置过。",
        ]}
        report = beat_coverage(self._body(), plan, {"novel": {}})
        self.assertTrue(report["enabled"])
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing_beats"]), 1)
        self.assertIn("安瓿", report["missing_beats"][0])

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

    def test_coverage_floor_fails_even_when_each_beat_hits_once(self):
        # Both beats hit one anchor each, but overall anchor hit-rate is low;
        # a strict min coverage should still fail the gate.
        plan = {"beats": [
            "林夕用镊子夹起安瓿，对照护士站的交接记录核对批号与给药时间。",
            "周临舟拦在配药室门口，亮出调岗通知逼她交出钥匙。",
        ]}
        body = self._body("她夹起安瓿看了一眼。周临舟站在配药室门口。")
        report = beat_coverage(body, plan, {"novel": {"beat_coverage_min": 0.95}})
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_beats"], [])
        self.assertLess(report["coverage"], 0.95)

    def test_anchor_fragments_skip_stop_tokens_and_generic(self):
        anchors = _beat_anchor_fragments("她发现了一个东西，意识到事情不对。")
        self.assertEqual(anchors, ["不对"])
        anchors2 = _beat_anchor_fragments("林夕把安瓿碎片收进证物袋。")
        self.assertIn("安瓿碎片", anchors2)
        self.assertIn("证物袋", anchors2)


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


class CraftRulesTests(unittest.TestCase):
    """Cross-book craft-rule consumption layer (closes the distillation loop)."""

    def _write_rules(self, rules):
        import tempfile, json as _json
        d = tempfile.mkdtemp()
        p = os.path.join(d, "craft_rules.json")
        with open(p, "w", encoding="utf-8") as f:
            _json.dump({"rules": rules, "meta": {}}, f, ensure_ascii=False)
        return p

    def test_missing_file_is_silent_noop(self):
        import craft
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": "/no/such/file.json"}}
        self.assertEqual(craft.load_craft_rules(cfg), {"rules": [], "meta": {}})
        self.assertEqual(craft.craft_writer_block(cfg), "")
        self.assertEqual(craft.craft_planner_hints(cfg), "")

    def test_disabled_flag_returns_empty(self):
        import craft
        p = self._write_rules([
            {"category": "style", "pattern": "碎片化破折号", "fix": "用完整长句", "confidence": 0.9, "evidence_count": 5}
        ])
        cfg = {"novel": {"craft_rules_enabled": False, "craft_rules_path": p}}
        self.assertEqual(craft.craft_writer_block(cfg), "")

    def test_confidence_filter(self):
        import craft
        p = self._write_rules([
            {"category": "style", "pattern": "低置信噪声", "fix": "忽略", "confidence": 0.1, "evidence_count": 2},
            {"category": "style", "pattern": "高置信化石句", "fix": "换措辞", "confidence": 0.8, "evidence_count": 6},
        ])
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": p, "craft_rules_min_confidence": 0.3}}
        block = craft.craft_writer_block(cfg)
        self.assertIn("高置信化石句", block)
        self.assertNotIn("低置信噪声", block)

    def test_category_routing(self):
        import craft
        p = self._write_rules([
            {"category": "style", "pattern": "文体问题", "fix": "修文体", "confidence": 0.7, "evidence_count": 4},
        ])
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": p}}
        # style is a writer category, not a planner category
        self.assertIn("文体问题", craft.craft_writer_block(cfg))
        self.assertEqual(craft.craft_planner_hints(cfg), "")

    def test_top_k_limit(self):
        import craft
        rules = [
            {"category": "hook_technique", "pattern": f"模式{i}", "fix": f"修复{i}",
             "confidence": 0.9, "evidence_count": 10 - i}
            for i in range(10)
        ]
        p = self._write_rules(rules)
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": p, "craft_rules_top_k": 3}}
        block = craft.craft_writer_block(cfg)
        # exactly 3 rule lines (each rule renders a "失败模式：" line)
        self.assertEqual(block.count("失败模式："), 3)

    def test_score_delta_ranks_first(self):
        import craft
        p = self._write_rules([
            {"category": "payoff_setup", "pattern": "无收益证据", "fix": "A",
             "confidence": 0.9, "evidence_count": 20, "avg_score_before": 0.0, "avg_score_after": 0.0},
            {"category": "payoff_setup", "pattern": "有正收益", "fix": "B",
             "confidence": 0.4, "evidence_count": 3, "avg_score_before": 6.0, "avg_score_after": 8.0},
        ])
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": p, "craft_rules_top_k": 1}}
        block = craft.craft_planner_hints(cfg)
        self.assertIn("有正收益", block)
        self.assertNotIn("无收益证据", block)

    def test_malformed_rules_skipped(self):
        import craft
        p = self._write_rules([
            "not a dict",
            {"category": "style", "pattern": "", "fix": "x", "confidence": 0.9},  # empty pattern
            {"category": "style", "pattern": "有效", "fix": "y", "confidence": 0.9, "evidence_count": 5},
        ])
        cfg = {"novel": {"craft_rules_enabled": True, "craft_rules_path": p}}
        block = craft.craft_writer_block(cfg)
        self.assertIn("有效", block)


class SceneBreakdownBlockTests(unittest.TestCase):
    def test_empty_breakdown_renders_nothing(self):
        from scene_breakdown import scene_breakdown_block
        self.assertEqual(scene_breakdown_block({}, 1), "")
        self.assertEqual(scene_breakdown_block({"scenes": []}, 1), "")
        self.assertEqual(scene_breakdown_block(None, 1), "")

    def test_renders_scenes_with_fields(self):
        from scene_breakdown import scene_breakdown_block
        bd = {
            "scenes": [
                {
                    "goal": "主角识破伪证",
                    "location": "县衙后堂",
                    "time": "深夜",
                    "characters": ["林照", "县令"],
                    "visible_actions": ["林照展开那张被烧残的契书", "县令的手抖了一下"],
                    "beats_covered": ["揭穿契书造假"],
                    "exit_state": "县令认罪，新的幕后名字浮出",
                }
            ],
            "carryover_to_next_chapter": "幕后之人是谁",
        }
        out = scene_breakdown_block(bd, 7)
        self.assertIn("CH7", out)
        self.assertIn("县衙后堂", out)
        self.assertIn("林照展开那张被烧残的契书", out)
        self.assertIn("退出状态", out)
        self.assertIn("幕后之人是谁", out)

    def test_skips_malformed_scenes(self):
        from scene_breakdown import scene_breakdown_block
        bd = {"scenes": ["not a dict", {"goal": "ok", "visible_actions": ["做事"]}]}
        out = scene_breakdown_block(bd, 2)
        self.assertIn("做事", out)


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


if __name__ == "__main__":
    unittest.main()