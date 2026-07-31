"""Meta-narrative leakage detection and L0 strip — zero LLM calls."""
import unittest

from engine.quality import ai_flavor_health
from engine.quality_repair import strip_meta_narrative


_PAD = "他站在窗前看着外面的雨，雨水顺着玻璃往下流，在窗台上汇成一条细流。" * 10


def _cfg(**over):
    novel = {"ai_flavor_penalty_cap": 3.0}
    novel.update(over)
    return {"novel": novel}


class MetaNarrativeDetectionTest(unittest.TestCase):

    def test_ch_ref_in_prose(self):
        text = "第1章 测试\n\n" + _PAD + "他在Ch2里见过那个人。"
        r = ai_flavor_health(text, _cfg())
        self.assertIn("meta_narrative_leak", " ".join(r["flags"]))
        self.assertGreater(r["penalty"], 0)

    def test_waited_n_chapters(self):
        text = "第1章 测试\n\n" + _PAD + "他等了九章才等到这个结果。"
        r = ai_flavor_health(text, _cfg())
        self.assertIn("meta_narrative_leak", " ".join(r["flags"]))

    def test_n_chapters_ago(self):
        text = "第1章 测试\n\n" + _PAD + "三章前发生的事情还历历在目。"
        r = ai_flavor_health(text, _cfg())
        self.assertIn("meta_narrative_leak", " ".join(r["flags"]))

    def test_title_line_no_false_positive(self):
        text = "第1章 标题\n\n" + _PAD + "这是正常的小说正文，没有任何元叙事泄漏。"
        r = ai_flavor_health(text, _cfg())
        self.assertNotIn("meta_narrative_leak", " ".join(r.get("flags", [])))

    def test_normal_prose_no_hit(self):
        text = "第1章 标题\n\n" + _PAD + "他等了很久才等到结果。三天前发生的事还历历在目。"
        r = ai_flavor_health(text, _cfg())
        self.assertNotIn("meta_narrative_leak", " ".join(r.get("flags", [])))


class StripMetaNarrativeTest(unittest.TestCase):

    def test_strip_ch_ref(self):
        text = "第1章 测试\n\n他在Ch2里见过那个人。"
        result = strip_meta_narrative(text)
        self.assertNotIn("Ch2", result)
        self.assertIn("见过那个人", result)

    def test_strip_waited(self):
        text = "第1章 测试\n\n他等了九章才看到结局。"
        result = strip_meta_narrative(text)
        self.assertIn("等了很久", result)
        self.assertNotIn("九章", result)

    def test_strip_ago(self):
        text = "第1章 测试\n\n三章前的场景重现。"
        result = strip_meta_narrative(text)
        self.assertIn("之前", result)
        self.assertNotIn("三章前", result)

    def test_preserves_title(self):
        text = "第1章 测试标题\n\n正文内容。"
        self.assertEqual(strip_meta_narrative(text), text)


if __name__ == "__main__":
    unittest.main()
