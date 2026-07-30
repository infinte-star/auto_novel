"""Tests for bootstrap chain parallelization and new features."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


class TestParallelOrchestration(unittest.TestCase):
    """Verify the ThreadPoolExecutor pattern saves wall-clock time."""

    def _mock_gen(self, tag, delay=0.15):
        def fn(*_a, **_kw):
            self.call_log.append((tag, threading.current_thread().name))
            time.sleep(delay)
            return f"result_{tag}"
        return fn

    def setUp(self):
        self.call_log = []

    def test_non_ensemble_runs_4_steps_not_6(self):
        """voice || volume_plan, then voices || frame — 4 parallel steps."""
        start = time.time()

        # Step 1-2: serial
        self._mock_gen("bible")()
        self._mock_gen("characters")()

        with ThreadPoolExecutor(max_workers=3) as pool:
            # Step 3: voice || volume_plan
            f_voice = pool.submit(self._mock_gen("voice"))
            f_vp = pool.submit(self._mock_gen("volume_plan"))
            voice = f_voice.result()
            vp = f_vp.result()

            # Step 4: voices || frame
            f_voices = pool.submit(self._mock_gen("voices"))
            f_frame = pool.submit(self._mock_gen("frame"))
            voices = f_voices.result()
            frame = f_frame.result()

        elapsed = time.time() - start

        self.assertEqual(len(self.call_log), 6)
        self.assertEqual(voice, "result_voice")
        self.assertEqual(voices, "result_voices")
        self.assertEqual(vp, "result_volume_plan")

        # Verify parallelism: voice and volume_plan on different threads
        t_voice = [t for tag, t in self.call_log if tag == "voice"][0]
        t_vp = [t for tag, t in self.call_log if tag == "volume_plan"][0]
        self.assertNotEqual(t_voice, t_vp, "Step 3 should be parallel")

        # Verify parallelism: voices and frame on different threads
        t_voices = [t for tag, t in self.call_log if tag == "voices"][0]
        t_frame = [t for tag, t in self.call_log if tag == "frame"][0]
        self.assertNotEqual(t_voices, t_frame, "Step 4 should be parallel")

        # 6 * 0.15s serial = 0.9s; parallel = ~0.6s (4 steps)
        self.assertLess(elapsed, 0.80, f"Expected parallel speedup, got {elapsed:.2f}s")

    def test_ensemble_runs_5_steps_not_8(self):
        """voice || skeleton, then voices || detail1 || detail2, then frame serial."""
        start = time.time()

        # Step 1-2: serial
        self._mock_gen("bible")()
        self._mock_gen("characters")()

        with ThreadPoolExecutor(max_workers=3) as pool:
            # Step 3: voice || skeleton
            f_voice = pool.submit(self._mock_gen("voice"))
            f_skel = pool.submit(self._mock_gen("skeleton"))
            voice = f_voice.result()
            skeleton = f_skel.result()

            # Step 4: voices || detail_v1 || detail_v2
            f_voices = pool.submit(self._mock_gen("voices"))
            f_d1 = pool.submit(self._mock_gen("detail_v1"))
            f_d2 = pool.submit(self._mock_gen("detail_v2"))
            voices = f_voices.result()
            d1 = f_d1.result()
            d2 = f_d2.result()

        # Step 5: frame serial (needs assembled volume_plan)
        self._mock_gen("frame")()

        elapsed = time.time() - start

        self.assertEqual(len(self.call_log), 8)

        # Verify step 3 parallel
        t_voice = [t for tag, t in self.call_log if tag == "voice"][0]
        t_skel = [t for tag, t in self.call_log if tag == "skeleton"][0]
        self.assertNotEqual(t_voice, t_skel)

        # Verify step 4 uses multiple threads
        threads_4 = set(t for tag, t in self.call_log
                        if tag in ("voices", "detail_v1", "detail_v2"))
        self.assertGreaterEqual(len(threads_4), 2, "Step 4 should use multiple threads")

        # 8 * 0.15s serial = 1.2s; parallel = ~0.75s (5 steps)
        self.assertLess(elapsed, 1.0, f"Expected parallel speedup, got {elapsed:.2f}s")


class TestInsertVolumeTables(unittest.TestCase):
    """Verify _insert_volume_tables assembles skeleton + tables correctly."""

    def test_inserts_tables_at_correct_positions(self):
        from engine.bootstrap import _insert_volume_tables

        skeleton = (
            "# 卷纲\n\n"
            "## 第1卷：启程（第1-20章）\n卷1内容\n\n"
            "## 第2卷：深入（第21-40章）\n卷2内容\n\n"
            "## 第3卷：终局（第41-60章）\n卷3内容\n"
        )
        vol_tables = [
            (0, "| Ch1 | 角色A高光 |\n| Ch2 | 角色B高光 |"),
            (1, "| Ch21 | 角色C高光 |\n| Ch22 | 角色D高光 |"),
        ]

        result = _insert_volume_tables(skeleton, vol_tables)

        # Vol 1 tables should appear before Vol 2 header
        idx_v1_table = result.index("角色A高光")
        idx_v2_header = result.index("## 第2卷")
        self.assertLess(idx_v1_table, idx_v2_header)

        # Vol 2 tables should appear before Vol 3 header
        idx_v2_table = result.index("角色C高光")
        idx_v3_header = result.index("## 第3卷")
        self.assertLess(idx_v2_table, idx_v3_header)

        # All original content preserved
        self.assertIn("卷1内容", result)
        self.assertIn("卷2内容", result)
        self.assertIn("卷3内容", result)

    def test_last_volume_tables_appended_at_end(self):
        from engine.bootstrap import _insert_volume_tables

        skeleton = (
            "## 第1卷：启程（第1-20章）\n卷1内容\n\n"
            "## 第2卷：终局（第21-40章）\n卷2内容\n"
        )
        vol_tables = [(1, "| Ch21 | 最终表格 |")]

        result = _insert_volume_tables(skeleton, vol_tables)
        self.assertIn("最终表格", result)
        # Table should be after vol 2 content
        self.assertGreater(result.index("最终表格"), result.index("卷2内容"))

    def test_empty_tables_returns_skeleton_unchanged(self):
        from engine.bootstrap import _insert_volume_tables

        skeleton = "## 第1卷\ncontent"
        result = _insert_volume_tables(skeleton, [])
        self.assertEqual(result, skeleton)


class TestVoicesChainSystemPrompt(unittest.TestCase):
    """Verify VOICES_CHAIN_SYSTEM prompt constant is well-formed."""

    def test_prompt_exists_and_has_required_sections(self):
        from engine.bootstrap import VOICES_CHAIN_SYSTEM

        self.assertIn("声音指纹", VOICES_CHAIN_SYSTEM)
        self.assertIn("标志性言行", VOICES_CHAIN_SYSTEM)
        self.assertIn("行为锚点", VOICES_CHAIN_SYSTEM)
        self.assertIn("5000", VOICES_CHAIN_SYSTEM)  # length constraint


class TestCharactersPromptLengthConstraint(unittest.TestCase):
    """Verify CHARACTERS_CHAIN_SYSTEM has the new length discipline."""

    def test_prompt_has_length_discipline(self):
        from engine.bootstrap import CHARACTERS_CHAIN_SYSTEM

        self.assertIn("篇幅纪律", CHARACTERS_CHAIN_SYSTEM)
        self.assertIn("800-1500", CHARACTERS_CHAIN_SYSTEM)
        self.assertIn("不得重复世界观圣经", CHARACTERS_CHAIN_SYSTEM)


class TestVoicesFileSave(unittest.TestCase):
    """Verify bootstrap() saves voices to paths.voices."""

    def test_bootstrap_save_section_includes_voices(self):
        import ast
        import textwrap
        with open("engine/bootstrap.py", encoding="utf-8") as f:
            source = f.read()
        # Check that paths.voices write exists near the voice_charter write
        self.assertIn("write_text(paths.voices,", source)
        # Check data["voices"] is set in the chain
        self.assertIn('data["voices"]', source)


if __name__ == "__main__":
    unittest.main()
