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
from quality import plan_visual_payoff_check, scene_similarity, style_health  # noqa: E402
from pipeline import _apply_force_accept_patches  # noqa: E402
from llm import _enhance_system_prompt, _repair_truncated_json, json_prompt, safe_json_loads  # noqa: E402
from writing import _beat_needs_concretization, _first_draft_execution_ledger  # noqa: E402


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
    def test_ledger_maps_beats_to_page_execution(self):
        plan = {
            "beats": [
                "沈澜把验尸单压在桌沿，对照两处伤口位置逼罗鹤改口。",
                "她推导出镜子被人动过。",
            ]
        }
        out = _first_draft_execution_ledger({"novel": {"chapter_words": 4000}}, plan)
        self.assertIn("首稿页面执行账本", out)
        self.assertIn("beat1", out)
        self.assertIn("谁在什么场地", out)
        self.assertIn("风险：该 beat 含抽象实现词", out)

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


if __name__ == "__main__":
    unittest.main()
