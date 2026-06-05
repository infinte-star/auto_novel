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
from quality import scene_similarity, style_health  # noqa: E402
from llm import _repair_truncated_json, safe_json_loads  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
