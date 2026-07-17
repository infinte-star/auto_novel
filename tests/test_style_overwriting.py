"""Unit tests for the anti-overwriting anchor in quality.style_health.

Covers the three checks added for the "instrument-report" collapse mode
(v12 huangliang Ch50-100): sentence-length upper band, dialogue-char-ratio
floor, and the tech-jargon × dialogue-starvation conjunction.
"""
import unittest

from quality import style_health


def _cfg(**novel_keys):
    base = {
        "style_max_avg_sentence_chars": 42.0,
        "style_dialogue_ratio_min": 0.04,
        "style_tech_jargon_per_kchar_warn": 8.0,
        "style_tech_jargon_per_kchar_bad": 12.0,
    }
    base.update(novel_keys)
    return {"novel": base}


# 一句 ~60 字的伪技术腔长句（含 4 个黑话词：频率/脉冲/共振/信号），无对话。
_OVERWRITTEN_SENT = (
    "他体内的灵质核心以每分钟七十二次的频率持续搏动，触觉脉冲沿神经束向锁骨方向"
    "延伸零点五毫米，与井壁传来的共振信号在骨膜深处形成了完全同步的镜像回路。"
)

# 健康对话章的构件：短叙述 + 对白。
_HEALTHY_UNIT = (
    "陆昭收剑入鞘，回头看了他一眼。\n"
    "“既然编号已登记，那就试一剑。”他说，“输了别哭。”\n"
    "苏澈把布条塞回鞋底，笑了笑。\n"
    "“好啊。”他说，“你先请。”\n"
)


def _overwritten_text(chars=3000):
    s = ""
    while len(s) < chars:
        s += _OVERWRITTEN_SENT
    return s


def _healthy_text(chars=3000):
    s = ""
    while len(s) < chars:
        s += _HEALTHY_UNIT
    return s


def _new_flags(sh):
    prefixes = ("sentences_too_long", "sentences_overlong_severe",
                "dialogue_starved", "pseudo_tech_collapse", "pseudo_tech_high")
    return [f for f in sh["flags"] if f.startswith(prefixes)]


class TestOverwritingAnchor(unittest.TestCase):
    def test_synthetic_overwriting_chapter_blocks(self):
        sh = style_health(_overwritten_text(), _cfg())
        self.assertGreaterEqual(sh["penalty"], 2.0)
        flags = " ".join(sh["flags"])
        # 72 字均长直接落入 severe 档；宽松断言两档任一即可。
        self.assertTrue("sentences_too_long" in flags or "sentences_overlong_severe" in flags)
        self.assertIn("dialogue_starved", flags)
        self.assertIn("pseudo_tech_collapse", flags)
        self.assertTrue(sh["directives"])  # writer gets corrective directives

    def test_healthy_dialogue_chapter_untouched(self):
        sh = style_health(_healthy_text(), _cfg())
        self.assertEqual(_new_flags(sh), [])

    def test_history_profile_tolerates_long_sentences(self):
        # 45 字均长的书面长句：neutral(42) 会警告，history(52) 不会。
        long_sent = "军中旧例、田亩清册与漕运折银的账目彼此纠缠，他坐在灯下逐页核对，直到更鼓敲过三巡才把最后一册合上。"
        text = long_sent * 60
        sh_neutral = style_health(text, _cfg())
        self.assertTrue(any(f.startswith("sentences_too_long") for f in sh_neutral["flags"]))
        sh_history = style_health(text, _cfg(style_max_avg_sentence_chars=52.0,
                                             style_dialogue_ratio_min=0.0))
        self.assertFalse(any(f.startswith("sentences_too_long") for f in sh_history["flags"]))

    def test_overlong_severe_tier(self):
        # 均长 > 42*1.3 ≈ 55 → severe (+2.0 单项)。
        very_long = ("这一段叙述被无休止的逗号连接成一条永远不肯停下来的河流，从街角的灯光写到"
                     "屋檐的雨水，再从雨水写到窗内的人影，却始终不肯给读者一个可以喘息的句号落点。")
        sh = style_health(very_long * 40, _cfg(style_dialogue_ratio_min=0.0))
        self.assertTrue(any(f.startswith("sentences_overlong_severe") for f in sh["flags"]))

    def test_dialogue_starved_subsumes_presence_check(self):
        sh = style_health(_overwritten_text(), _cfg())
        self.assertTrue(any(f.startswith("dialogue_starved") for f in sh["flags"]))
        self.assertNotIn("almost_no_dialogue", sh["flags"])

    def test_high_jargon_with_rich_dialogue_not_penalized(self):
        # 黑话高但对话充足（数据面板类爽文合法形态）→ 检查6 只发 directive 不罚分。
        jargon_dialogue = (
            "“检测到共振频率异常，脉冲信号衰减了三成！”她盯着读数喊道。\n"
            "“把接收模块的参数调低，再校准一次。”他按住装置，“信号的波形不对，编码全乱了。”\n"
        )
        sh = style_health(jargon_dialogue * 40, _cfg())
        self.assertFalse(any(f.startswith(("pseudo_tech_collapse", "pseudo_tech_high"))
                             for f in sh["flags"]))
        self.assertFalse(any(f.startswith("dialogue_starved") for f in sh["flags"]))

    def test_pseudo_tech_gate_disable(self):
        sh = style_health(_overwritten_text(), _cfg(style_pseudo_precision_enabled=False))
        self.assertFalse(any(f.startswith(("pseudo_tech_collapse", "pseudo_tech_high"))
                             for f in sh["flags"]))

    def test_penalty_capped(self):
        sh = style_health(_overwritten_text(), _cfg())
        self.assertLessEqual(sh["penalty"], 4.0)

    def test_tech_history_kwarg_inert(self):
        a = style_health(_healthy_text(), _cfg())
        b = style_health(_healthy_text(), _cfg(), tech_history=[1.0, 2.0, 3.0])
        self.assertEqual(a["penalty"], b["penalty"])

    def test_metrics_exposed(self):
        sh = style_health(_overwritten_text(), _cfg())
        self.assertIn("dialogue_char_ratio", sh["metrics"])
        self.assertIn("tech_per_kchar", sh["metrics"])
        self.assertIn("avg_sentence_chars", sh["metrics"])

    def test_short_text_early_return(self):
        sh = style_health("很短。", _cfg())
        self.assertEqual(sh["penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
