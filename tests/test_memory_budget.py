"""Unit tests for the P2/P3 pure functions.

- memory._compressible_sections (compress-ratchet fix)
- novel._parse_price_table (model-aware cost reporting)
- writing._hook_directives_block (吸量包 opening injection)

memory.memory_context/lite_memory_context need Paths+conn so their max_chars
behavior is exercised indirectly via the budget helpers here plus the live
replay/ablation harnesses.
"""
import unittest

from memory import _compressible_sections
from novel import _parse_price_table
from writing import _hook_directives_block


def _mem_file(n_sections: int) -> str:
    parts = ["# bible header\n"]
    for i in range(1, n_sections + 1):
        parts.append(f"## Ch{i}\n- 事件 {i}\n")
    return "".join(parts)


class TestCompressibleSections(unittest.TestCase):
    def test_below_keep_recent_is_zero(self):
        self.assertEqual(_compressible_sections(_mem_file(10), 30), 0)

    def test_exactly_keep_recent_is_zero(self):
        self.assertEqual(_compressible_sections(_mem_file(30), 30), 0)

    def test_excess_counted(self):
        self.assertEqual(_compressible_sections(_mem_file(37), 30), 7)

    def test_header_only_is_zero(self):
        self.assertEqual(_compressible_sections("# just a header\nsome text\n", 30), 0)

    def test_single_section_is_zero(self):
        # compress_memory_file early-returns at <=2 split parts; trigger must agree.
        self.assertEqual(_compressible_sections(_mem_file(1), 0), 0)


class TestPriceTable(unittest.TestCase):
    def test_basic_parse(self):
        t = _parse_price_table("deepseek:3.0:15.0, minimax:0.5:2.0")
        self.assertEqual(t, [("deepseek", 3.0, 15.0), ("minimax", 0.5, 2.0)])

    def test_semicolon_and_case(self):
        t = _parse_price_table("DeepSeek-V4:1.1:2.2; GLM:0.3:0.9")
        self.assertEqual(t[0][0], "deepseek-v4")
        self.assertEqual(len(t), 2)

    def test_malformed_entries_skipped(self):
        t = _parse_price_table("bad, deepseek:3.0:15.0, x:y:z, only:2")
        self.assertEqual(t, [("deepseek", 3.0, 15.0)])

    def test_empty(self):
        self.assertEqual(_parse_price_table(""), [])
        self.assertEqual(_parse_price_table(None), [])


class TestHookDirectivesBlock(unittest.TestCase):
    def test_basic_render(self):
        blk = _hook_directives_block({"hook_directives": ["开篇必须当章完成A", "首次B要具体"]})
        self.assertIn("开篇吸量指令", blk)
        self.assertIn("- 开篇必须当章完成A", blk)

    def test_caps_five_items(self):
        blk = _hook_directives_block({"hook_directives": [f"指令{i}" for i in range(9)]})
        self.assertEqual(blk.count("- 指令"), 5)

    def test_char_budget(self):
        long = "很长的指令" * 60  # 300 chars each
        blk = _hook_directives_block({"hook_directives": [long, long, long]})
        # 600-char budget → at most 2 items land
        self.assertLessEqual(blk.count("- 很长"), 2)

    def test_absent_or_malformed(self):
        self.assertEqual(_hook_directives_block({}), "")
        self.assertEqual(_hook_directives_block({"hook_directives": "not-a-list"}), "")
        self.assertEqual(_hook_directives_block(None), "")
        self.assertEqual(_hook_directives_block({"hook_directives": ["", "  "]}), "")


if __name__ == "__main__":
    unittest.main()
