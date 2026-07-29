"""The frozen voice anchor is held to the density its own text prescribes.

`memory.VOICE_CHAIN_SYSTEM` orders the charter to demonstrate 「破折号（——）每千字
不超过 3 个」 and the demo prose it returns becomes `memory/voice.md` — written once
by bootstrap, never re-derived (v2 has no voice-refresh path), projected into every
chapter's StoryState as the model of how this book sounds.

`_verify_bootstrap` used to validate it with `style_health(voice).penalty > 0`
alone. That is the CHAPTER release ruler (`style_em_dash_per_kchar_warn`, 6.0),
not the charter's own line (`em_dash_reduce_target_per_kchar`, 3.0), so everything
in the 3–6 gap passed in silence. Measured over the library's 15 anchors when this
was found: **14 exceeded 3/k and exactly 1 carried any penalty** —
chaosheng_dangan 5.86/k, tangshuting_e2e 5.32/k, huangliang 4.93/k, all "clean".
A charter running at twice the density it prescribes teaches the density, not the
rule.

Four invariants live here, one per defect that was actually present:
1. over-target-but-penalty-clean is fixed, and fixed WITHOUT an LLM call — the
   deterministic L0 reducer already targets exactly this number;
2. the LLM rewrite is spent only when `style_health` itself fired (L0 before L1,
   the repair ladder's own ordering);
3. a rewrite that merely TIES is not adopted — the old acceptance test was
   `<= health.penalty`, which adopted a no-better draft and then logged
   "Bootstrap voice repaired and rewritten", a line that could not be false;
4. an anchor neither layer can fix says so in the log, because silence here reads
   as a pass.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import engine.config as _config
import engine.bootstrap as memory
import engine.quality as quality

# --- fixtures, calibrated against the live gate rather than guessed -----------
# `_SENT` is healthy prose (full sentences, ~28 chars each); `_DLG` carries enough
# quoted mass to clear `style_dialogue_ratio_min`, so the only term these fixtures
# move is the em-dash one; `_EM` contributes exactly one `——`.
_SENT = (
    "她把报告翻到第七页，纸的厚度对不上，于是在目录那一行画了个蓝圈。"
    "镇上的人都说过海了，措辞一样，停顿的位置也一样，像背过同一句话。"
    "他在围裙上擦了手才开口，声音低得要人凑近才听得清楚。"
)
_DLG = (
    "“你要的那一页在归档里，归档的人上礼拜刚走，钥匙谁也说不清在哪只抽屉。”"
    "他说完这句就低下头，像是在等她自己放弃这个问题。"
    "“那我等归档的人回来，顺便把船检的存根一起看了。”她把笔记本合上。"
)
_EM = "她数了两遍，十四页——只少了中间那一张，骑缝章却都对得上。"


def _voice_md(n_plain: int, n_em: int, n_dlg: int = 3) -> str:
    body = [
        "# 叙事声音宪章",
        "## 健康文风护栏",
        "- 以完整的主谓宾句子叙事；破折号每千字不超过 3 个。",
        "## 示范片段",
    ]
    body += [_SENT] * n_plain + [_DLG] * n_dlg + [_EM] * n_em
    return "\n\n".join(body)


# em 4.73/k, penalty 0.0 — inside the gap the old check could not see.
IN_THE_GAP = _voice_md(n_plain=16, n_em=10)
# em 0.0/k, penalty 0.0 — already compliant.
COMPLIANT = _voice_md(n_plain=16, n_em=0)
# em 17.2/k, penalty 2.0 — real style collapse, the only case worth an LLM call.
COLLAPSED = _voice_md(n_plain=2, n_em=12, n_dlg=1)


def _cfg(**over):
    novel = {
        "em_dash_reduce_target_per_kchar": 3.0,
        "em_dash_reduce_enabled": True,
        "style_health_enabled": True,
        "style_em_dash_per_kchar_warn": 6.0,
        "style_em_dash_per_kchar_bad": 9.0,
        "bootstrap_verify_enabled": True,
    }
    novel.update(over)
    return {"novel": novel, "api": {}, "paths": {}}


def _paths(tmp: str) -> _config.Paths:
    root = Path(tmp)
    (root / "memory").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    kw = {f: root / f for f in _config.Paths.__dataclass_fields__}
    kw["voice"] = root / "memory" / "voice.md"
    kw["logs_dir"] = root / "logs"
    kw["chapters_dir"] = root / "chapters"
    kw["database"] = root / "story_state.db"
    return _config.Paths(**kw)


def _measure(text: str, config) -> tuple[float, float]:
    h = quality.style_health(text, config)
    m = h.get("metrics") or {}
    return float(h.get("penalty", 0) or 0), float(m.get("em_dash_per_kchar", 0) or 0)


class VoiceAnchorCharterTest(unittest.TestCase):
    def _run(self, voice: str, config, repair_out: str = ""):
        """`_verify_bootstrap` with `_gen_md_section` stubbed.

        Returns (data, llm_tags, log_text, persisted_text).
        """
        calls: list[str] = []

        def fake_gen(client, paths, config, system, user, tag, max_tokens=16000):
            calls.append(tag)
            return repair_out

        data = {"voice": voice, "bible": "（世界观）", "characters": "", "volume_plan": ""}
        with TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            import engine.bootstrap as _bootstrap_mod
            orig = _bootstrap_mod._gen_md_section
            _bootstrap_mod._gen_md_section = fake_gen
            try:
                memory._verify_bootstrap(object(), paths, config, data)
            finally:
                _bootstrap_mod._gen_md_section = orig
            log_txt = "".join(
                p.read_text(encoding="utf-8", errors="replace")
                for p in Path(tmp).rglob("*.log")
            )
            on_disk = (
                paths.voice.read_text(encoding="utf-8") if paths.voice.exists() else ""
            )
        return data, calls, log_txt, on_disk

    # --- 1. the gap ---------------------------------------------------------
    def test_over_charter_but_penalty_clean_is_fixed_with_zero_llm(self):
        config = _cfg()
        pen, em = _measure(IN_THE_GAP, config)
        self.assertEqual(pen, 0.0, "fixture must be penalty-clean or this proves nothing")
        self.assertGreater(em, 3.0, "fixture must sit above the charter line")
        self.assertLess(em, 6.0, "fixture must sit below the chapter warn line")

        data, calls, log_txt, on_disk = self._run(IN_THE_GAP, config)

        self.assertEqual(calls, [], "a penalty-clean anchor must not buy an LLM rewrite")
        _pen2, em2 = _measure(data["voice"], config)
        self.assertLess(em2, em)
        self.assertLessEqual(em2, 3.0, "the anchor must land at the density it preaches")
        self.assertEqual(on_disk.strip(), data["voice"].strip(),
                         "the reduced anchor must be persisted, not merely returned")
        self.assertIn("rewritten", log_txt)

    def test_already_under_charter_is_left_alone(self):
        config = _cfg()
        _pen, em = _measure(COMPLIANT, config)
        self.assertLessEqual(em, 3.0)

        data, calls, log_txt, on_disk = self._run(COMPLIANT, config)

        self.assertEqual(calls, [])
        self.assertEqual(data["voice"], COMPLIANT, "a compliant anchor must not be touched")
        self.assertEqual(on_disk, "", "nothing to rewrite means nothing written")
        self.assertNotIn("rewritten", log_txt)
        self.assertNotIn("WARNING", log_txt)

    # --- 2. L1 only when the gate fired ------------------------------------
    def test_llm_rewrite_only_when_style_health_fired(self):
        config = _cfg()
        pen, _em = _measure(COLLAPSED, config)
        self.assertGreater(pen, 0)

        data, calls, log_txt, _disk = self._run(COLLAPSED, config, repair_out=COMPLIANT)

        self.assertEqual(calls, ["bootstrap_voice_repair"],
                         "exactly one call, under the tag the cost ledger reads")
        self.assertEqual(data["voice"], COMPLIANT)
        self.assertEqual(_measure(data["voice"], config)[0], 0.0)
        self.assertIn("rewritten", log_txt)

    # --- 3. a no-better rewrite is not a repair ----------------------------
    def test_a_no_better_rewrite_is_not_adopted_and_not_claimed(self):
        """`style_health.penalty` saturates, so "same tier" hid a much worse draft.

        The old acceptance was `style_health(fixed).penalty <= health.penalty`.
        Penalty tops out at 2.0 for anything at or over `_bad`, so a rewrite at
        26/k was adopted over a 17/k original — both read 2.0 — and the log then
        said "Bootstrap voice repaired and rewritten". Ranking on
        `(penalty, em/kchar)` is what makes that comparison able to fail.
        """
        config = _cfg()
        worse = _voice_md(n_plain=1, n_em=20, n_dlg=0)
        pen0, em0 = _measure(COLLAPSED, config)
        pen_w, em_w = _measure(worse, config)
        self.assertEqual(pen_w, pen0, "fixture must tie on penalty (the old ruler)")
        self.assertGreater(em_w, em0, "…while being strictly worse on density")

        data, calls, _log, _disk = self._run(COLLAPSED, config, repair_out=worse)

        self.assertEqual(calls, ["bootstrap_voice_repair"])
        self.assertNotEqual(data["voice"], worse,
                            "a rewrite that is worse on density must not be adopted")
        # L0 still gets its turn, so the anchor ends better than it started.
        self.assertLess(_measure(data["voice"], config)[1], em0)

    # --- 4. unfixable says so ---------------------------------------------
    def test_unfixable_anchor_warns_instead_of_claiming_a_repair(self):
        config = _cfg(em_dash_reduce_enabled=False)  # strand the anchor: no L0
        pen, em = _measure(IN_THE_GAP, config)
        self.assertEqual(pen, 0.0)
        self.assertGreater(em, 3.0)

        data, calls, log_txt, on_disk = self._run(IN_THE_GAP, config)

        self.assertEqual(calls, [], "still no LLM call: the penalty is clean")
        self.assertEqual(data["voice"], IN_THE_GAP)
        self.assertEqual(on_disk, "")
        self.assertIn("WARNING", log_txt)
        self.assertIn("charter", log_txt)
        self.assertNotIn("rewritten", log_txt)

    def test_disabled_verification_is_a_no_op(self):
        config = _cfg(bootstrap_verify_enabled=False)
        data, calls, _log, on_disk = self._run(COLLAPSED, config, repair_out=COMPLIANT)
        self.assertEqual(calls, [])
        self.assertEqual(data["voice"], COLLAPSED)
        self.assertEqual(on_disk, "")


if __name__ == "__main__":
    unittest.main()
