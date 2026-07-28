"""v2 run: the decision table, tested with the four model calls unplugged.

The claim this module makes is that nothing about *what happens next* needs a
model. That claim is only worth anything if it can be checked without one — so
every test here drives the real `run_chapter` over the real `DECISIONS` table
with the LLM-spending rows swapped for recorders.

What is under test is ordering and termination, not prose:
  - repair is offered before a rewrite is bought (v1 had to hand-place one
    special case to get this; here it must hold for every gate),
  - `review_round0.json` records the RAW first draft and no later step edits it,
  - a row that fires must advance its own precondition, so the loop terminates,
  - the rescue is bounded, because every blocking reason is chapter-scoped and a
    second attempt would be retrying a rule the text already failed.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from v2 import run as v2run


def _config(**over) -> dict:
    novel = {
        "book_fossil_min_chapters": 6, "book_fossil_every": 5,
        "descriptor_freq_min_spread": 15, "descriptor_freq_every": 5,
        "style_cross_repeat_lookback": 6, "style_cross_repeat_lookback_long": 20,
        "genre_adherence_window": 5, "fingerprint_enabled": False,
        "quality_breaker_consecutive": 2,
    }
    novel.update(over)
    return {"novel": novel, "api": {}}


class _Paths:
    """The handful of `Paths` fields this module touches, on a real tmpdir."""

    def __init__(self, root: Path):
        self.root = root
        self.chapters_dir = root / "chapters"
        self.logs_dir = root / "logs"
        self.checkpoints_dir = root / "checkpoints"
        for d in (self.chapters_dir, self.logs_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.book = root / "book.md"
        self.state = root / "state.md"


def _ctx(root: Path, config: dict | None = None) -> v2run.Ctx:
    return v2run.Ctx(client=object(), paths=_Paths(root), conn=None,
                     config=config or _config())


def _recorder(label: str, log: list[str], **sets):
    """An action that records that its row fired and satisfies its predicate."""
    def action(ctx, run):
        log.append(label)
        for key, value in sets.items():
            setattr(run, key, value(run) if callable(value) else value)
        return label
    return action


def _clean_report() -> dict:
    return {"gate_rejects": [], "block_reasons": [], "accepted": True}


def _l0_blocking() -> dict:
    """Blocking, and `fix.plan_repairs` routes it to L0 (em-dash rewording)."""
    return {"style_health": {"penalty": 2.4, "flags": ["em_dash_bad"]},
            "block_reasons": ["style_collapse(penalty=2.4)"], "accepted": False}


def _l1_blocking() -> dict:
    """Blocking, and routed to L1 (expand to band) — no L0 action exists."""
    return {"length_band": {"block": True, "flag": "short"},
            "block_reasons": ["length_band(short)"], "accepted": False}


def _l1_advisory() -> dict:
    """A REAL short-chapter gate result: fires, offers an L1 fixer, never blocks.

    Built by calling the gate rather than by hand, because the whole point of
    this case is a payload shape I got wrong once: `length_band_check` has no
    path to `block=True` on the short side (only a gross overshoot can block),
    so a blocks-gated repair predicate leaves `expand_to_band` unreachable and
    v2 mute on short chapters — the failure the live smoke run surfaced at 2240
    chars. Fabricating the payload is what hid it the first time.
    """
    import quality
    result = quality.length_band_check("正" * 1200, _config())
    assert not result["block"] and result["flags"], result
    return {"length_band": result, "block_reasons": [], "accepted": True}


def _stub_actions(log: list[str], *, after_write=None, after_l0=None,
                  after_l1=None, after_rescue=None) -> dict:
    """Every LLM-spending row replaced by a recorder.

    The reports are REAL gate payloads rather than bare `block_reasons`, because
    `l0_pending` / `l1_pending` are the production predicates and they ask
    `fix.plan_repairs` what it can offer for the gates that actually fired. A
    stub that only set `block_reasons` would route to no layer at all and the
    ordering test would pass without ever exercising the ordering.
    """
    state = {"rescued": False}

    def report_action(ctx, run):
        log.append("report")
        base = (after_rescue if state["rescued"] else after_write) or _clean_report()
        run.report = dict(base, chapter=run.chapter_num)
        if not run.round0_saved:
            from checkpoint import save_checkpoint
            save_checkpoint(ctx.paths, run.chapter_num, v2run.ROUND0_CHECKPOINT,
                            run.report)
            run.round0_saved = True
        return "report"

    def _layer(name, result):
        def action(ctx, run):
            log.append(name)
            run.layers_run = run.layers_run + (name,)
            if result is not None:
                run.report = dict(result, chapter=run.chapter_num)
            return name
        return action

    def rescue_action(ctx, run):
        log.append("rescue")
        state["rescued"] = True
        run.rescue_attempts += 1
        run.text = ""
        run.report = None
        run.layers_run = ()
        return "rescue"

    return {
        "need_card": _recorder("card", log, card={"title": "x"}, plan={},
                               decision={}, card_source="arc"),
        "card_invalid": _recorder("constraints", log, constraints=()),
        "need_draft": _recorder("write", log, text="正文" * 900,
                                raw_text=lambda r: r.raw_text or "正文" * 900),
        "need_report": report_action,
        "l0_pending": _layer("L0", after_l0),
        "l1_pending": _layer("L1", after_l1),
        "canon_pending": _recorder("canon", log, canon_checked=True),
        "next_card_patch": _recorder("patch_next", log, patched_next=True),
        "rescue": rescue_action,
        "commit": _recorder("commit", log, committed=True),
    }


class BookScanCadenceTest(unittest.TestCase):
    """The scan cadence is copied from `review.py`, not improved.

    FPY′ judges each arm by whatever its round-0 payload contains, so an arm
    that scanned the whole book every chapter would find fossils the other arm
    was never asked about and be reported as the worse engine for looking
    harder.
    """

    def test_fossils_only_on_v1s_cadence(self):
        cfg = _config()
        self.assertEqual(v2run.book_scan_gates(cfg, 5), ())    # below minimum
        self.assertEqual(v2run.book_scan_gates(cfg, 7), ())    # not a multiple
        self.assertEqual(v2run.book_scan_gates(cfg, 10), ("book_wide_fossils",))

    def test_descriptors_have_their_own_later_minimum(self):
        cfg = _config()
        self.assertEqual(v2run.book_scan_gates(cfg, 10), ("book_wide_fossils",))
        self.assertEqual(v2run.book_scan_gates(cfg, 15),
                         ("book_wide_fossils", "descriptor_frequency"))

    def test_a_zero_cadence_does_not_divide_by_zero(self):
        cfg = _config(book_fossil_every=0, descriptor_freq_every=0)
        self.assertEqual(v2run.book_scan_gates(cfg, 20),
                         ("book_wide_fossils", "descriptor_frequency"))


class CorpusTest(unittest.TestCase):

    def test_the_book_scan_corpus_stops_at_what_is_on_disk(self):
        # v1 reviews BEFORE `save_chapter`, so its whole-book scan never contains
        # the chapter being reviewed. Feeding v2's in-memory draft in here would
        # be a strictly stricter gate than the arm it is measured against, and
        # v2 would lose the A/B for holding itself to a different rule.
        with TemporaryDirectory() as tmp:
            paths = _Paths(Path(tmp))
            for n in range(1, 10):
                (paths.chapters_dir / f"chapter_{n:04d}.md").write_text(
                    f"第{n}章正文", encoding="utf-8")
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                corpus = v2run.load_corpus(paths, None, _config(), 10)
            self.assertEqual(max(corpus.book_texts), 9)
            self.assertNotIn(10, corpus.book_texts)

    def test_no_scan_this_chapter_means_no_book_glob_at_all(self):
        with TemporaryDirectory() as tmp:
            paths = _Paths(Path(tmp))
            for n in range(1, 8):
                (paths.chapters_dir / f"chapter_{n:04d}.md").write_text(
                    "正文", encoding="utf-8")
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                corpus = v2run.load_corpus(paths, None, _config(), 8)
            self.assertEqual(corpus.book_scans, ())
            self.assertEqual(corpus.book_texts, {})

    def test_the_two_repetition_windows_are_both_supplied(self):
        # `prior_texts` (6) and `prior_texts_long` (20) feed different gates;
        # collapsing them to one window silently disables the long one.
        with TemporaryDirectory() as tmp:
            paths = _Paths(Path(tmp))
            for n in range(1, 31):
                (paths.chapters_dir / f"chapter_{n:04d}.md").write_text(
                    f"第{n}章", encoding="utf-8")
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                corpus = v2run.load_corpus(paths, None, _config(), 31)
            self.assertEqual(len(corpus.prior_texts), 6)
            self.assertEqual(len(corpus.prior_long), 20)
            self.assertEqual(corpus.prev_text, "第30章")


class ScanCachePersistTest(unittest.TestCase):
    """`logs/book_fossils.json` is an INPUT, not just a record.

    `writing._preflight_negative_list` reads it on every chapter and v2's writer
    goes through that same function. Not writing it would quietly hand the v1 arm
    an avoid list the v2 arm never gets — and v2 would then write more fossils,
    which reads as a v2 defect rather than a harness bug.
    """

    def test_the_writers_avoid_list_is_written_where_the_writer_reads_it(self):
        with TemporaryDirectory() as tmp:
            paths = _Paths(Path(tmp))
            v2run.persist_scan_caches(paths, {
                "book_fossils": {"phrases": ["声音压得很低"], "hard_fossils": []},
                "descriptor_frequency": {"flagged": ["冷冷地"]}})
            fossils = json.loads((paths.logs_dir / "book_fossils.json").read_text("utf-8"))
            self.assertEqual(fossils["phrases"], ["声音压得很低"])
            self.assertTrue((paths.logs_dir / "descriptor_freq.json").exists())

    def test_an_empty_scan_does_not_erase_the_previous_avoid_list(self):
        # A chapter off the cadence produces no scan result. Writing that empty
        # result would wipe the list every non-scan chapter, i.e. 4 out of 5.
        with TemporaryDirectory() as tmp:
            paths = _Paths(Path(tmp))
            (paths.logs_dir / "book_fossils.json").write_text(
                '{"phrases": ["旧的"]}', encoding="utf-8")
            v2run.persist_scan_caches(paths, {"book_fossils": {"phrases": []}})
            kept = json.loads((paths.logs_dir / "book_fossils.json").read_text("utf-8"))
            self.assertEqual(kept["phrases"], ["旧的"])


class FlattenProtagonistTest(unittest.TestCase):
    """`writing.update_state_file` stringifies whatever it is handed, and v2's
    delta holds a dict — `str(dict)` would render a Python repr into state.md
    and the writer would read it every chapter afterwards."""

    def test_a_dict_becomes_markdown_lines_not_a_repr(self):
        out = v2run._flatten_protagonist({"位置": "青云宗", "伤势": "未愈"})
        self.assertIn("- 位置：青云宗", out)
        self.assertNotIn("{", out)

    def test_list_values_are_joined_rather_than_dropped(self):
        out = v2run._flatten_protagonist({"随身物": ["剑", "玉佩"]})
        self.assertIn("剑、玉佩", out)

    def test_a_string_passes_through_and_junk_does_not_crash(self):
        self.assertEqual(v2run._flatten_protagonist("重伤"), "重伤")
        self.assertEqual(v2run._flatten_protagonist(None), "")


class DecisionTableTest(unittest.TestCase):

    def _run(self, *, config=None, **stubs):
        log: list[str] = []
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp), config)
            run = v2run.run_chapter(ctx, 3, actions=_stub_actions(log, **stubs),
                                    corpus=v2run.Corpus())
        return run, log

    def test_a_clean_chapter_spends_the_designed_calls_and_no_more(self):
        run, log = self._run()
        self.assertEqual(log, ["card", "constraints", "write", "report",
                               "canon", "patch_next", "commit"])
        self.assertTrue(run.committed)

    def test_every_row_fires_at_most_once_when_it_advances_its_precondition(self):
        run, log = self._run()
        self.assertEqual(len(log), len(set(log)))

    def test_repair_is_offered_before_a_rewrite_is_bought(self):
        # The ordering v1 could only get for one gate, by hand-placing
        # `_repair_fossil_rejects` inside its review loop. Here L0 and L1 sit
        # above `rescue` in the table, so it holds for every gate by default.
        run, log = self._run(after_write=_l0_blocking(), after_l0=_clean_report())
        self.assertIn("L0", log)
        self.assertLess(log.index("L0"), log.index("commit"))
        self.assertNotIn("rescue", log)

    def test_l1_is_offered_when_no_l0_fixer_addresses_the_gate(self):
        run, log = self._run(after_write=_l1_blocking(), after_l1=_clean_report())
        self.assertNotIn("L0", log)     # nothing L0 can do about a short chapter
        self.assertIn("L1", log)
        self.assertNotIn("rescue", log)

    def test_a_fixable_gate_is_repaired_even_though_it_never_blocks(self):
        # The regression the smoke run caught. Most fixable findings never reach
        # `hard_block_reasons`, and v1 answers them anyway: `_stage_fix` runs
        # unconditionally because every fixer is keep-only-if-improved. Gating
        # the ladder on `blocks` made `expand_to_band` dead code and left v2 with
        # no answer at all to a short chapter, since the score penalty v1 leans
        # on is exactly what v2 bans.
        run, log = self._run(after_write=_l1_advisory())
        self.assertEqual(run.blocks, ())          # nothing blocked...
        self.assertIn("L1", log)                  # ...and it was still repaired
        self.assertNotIn("rescue", log)           # but a rewrite is not bought
        self.assertTrue(run.committed)

    def test_a_layer_that_failed_once_is_not_offered_again(self):
        # Re-offering it is the latch this codebase keeps rediscovering: the same
        # fixer on the same text fails the same way, forever.
        run, log = self._run(after_write=_l0_blocking(), after_l0=_l0_blocking(),
                             after_rescue=_clean_report())
        self.assertEqual(log.count("L0"), 1)

    def test_surviving_blocks_buy_exactly_one_rescue(self):
        run, log = self._run(after_write=_l1_blocking(), after_l1=_l1_blocking(),
                             after_rescue=_clean_report())
        self.assertEqual(log.count("rescue"), 1)
        self.assertEqual(log.count("write"), 2)
        self.assertTrue(run.committed)
        self.assertEqual(run.blocks, ())

    def test_a_chapter_that_cannot_be_saved_commits_with_its_blocks_recorded(self):
        # Not a retry loop: every acceptance reason is chapter-scoped, so a
        # second rescue would be re-running a rule this text already failed under
        # the same instructions. The honest outcome is to ship it and say so.
        run, log = self._run(after_write=_l1_blocking(), after_l1=_l1_blocking(),
                             after_rescue=_l1_blocking())
        self.assertEqual(log.count("rescue"), v2run.RESCUE_ATTEMPTS)
        self.assertTrue(run.committed)
        self.assertEqual(run.blocks, ("length_band(short)",))

    def test_the_canon_call_is_not_re_bought_by_a_rescue(self):
        # The judging call is spent per chapter, not per draft; `fold_citations`
        # re-cites the existing claims against whatever text ships.
        run, log = self._run(after_write=_l1_blocking(), after_l1=_l1_blocking(),
                             after_rescue=_clean_report())
        self.assertEqual(log.count("canon"), 1)

    def test_round0_records_the_raw_first_draft_and_no_later_step_edits_it(self):
        # This payload is what `tools/fpy_prime.py` replays. A rescue that
        # overwrote it would let v2 repair its way to a first-pass rate it did
        # not earn — which is the exact number the A/B turns on.
        log: list[str] = []
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            actions = _stub_actions(log, after_write=_l1_blocking(),
                                    after_l1=_l1_blocking(),
                                    after_rescue=_clean_report())
            v2run.run_chapter(ctx, 3, actions=actions, corpus=v2run.Corpus())
            from checkpoint import load_checkpoint
            round0 = load_checkpoint(ctx.paths, 3, v2run.ROUND0_CHECKPOINT)
        self.assertEqual(round0["block_reasons"], ["length_band(short)"])
        self.assertFalse(round0["accepted"])

    def test_a_row_that_does_not_advance_its_predicate_is_caught_not_hung(self):
        # The failure mode of any table-driven loop. Better a loud RuntimeError
        # naming the trace than a chapter that spins until the API bill notices.
        log: list[str] = []
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            actions = _stub_actions(log)
            actions["need_card"] = lambda ctx_, r: "card"   # never sets r.card

            with self.assertRaises(RuntimeError) as caught:
                v2run.run_chapter(ctx, 3, actions=actions, corpus=v2run.Corpus())
        self.assertIn("did not converge", str(caught.exception))

    def test_the_table_rows_are_the_documented_ones_in_the_documented_order(self):
        self.assertEqual([name for name, _, _ in v2run.DECISIONS], [
            "need_card", "card_invalid", "need_draft", "need_report",
            "l0_pending", "l1_pending", "canon_pending", "next_card_patch",
            "rescue", "commit"])

    def test_commit_is_last_and_unconditional(self):
        # The table is first-match, so the fallthrough row has to be both. A
        # conditional last row would let a chapter fall off the end of the table
        # and spin until MAX_STEPS.
        name, predicate, _ = v2run.DECISIONS[-1]
        self.assertEqual(name, "commit")
        fresh = v2run.ChapterRun(chapter_num=1)
        finished = v2run.ChapterRun(chapter_num=1, card={}, constraints=(),
                                    text="x", report={}, canon_checked=True,
                                    patched_next=True)
        self.assertTrue(predicate(None, fresh))
        self.assertTrue(predicate(None, finished))


class RestoreTest(unittest.TestCase):

    def test_the_file_on_disk_outranks_the_draft_checkpoint(self):
        # It is what a reader would see and what `book.md` already contains.
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            from checkpoint import save_checkpoint
            save_checkpoint(ctx.paths, 3, v2run.DRAFT_CHECKPOINT,
                            {"text": "草稿", "title": "t", "delta": {},
                             "delta_status": "ok"})
            (ctx.paths.chapters_dir / "chapter_0003.md").write_text(
                "落盘的正文", encoding="utf-8")
            run = v2run.ChapterRun(chapter_num=3)
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                v2run._restore(ctx, run)
            self.assertEqual(run.text, "落盘的正文")
            self.assertEqual(run.raw_text, "草稿")   # round 0 was the draft

    def test_a_stored_canon_result_is_not_re_bought_on_resume(self):
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            from checkpoint import save_checkpoint
            save_checkpoint(ctx.paths, 3, v2run.CANON_CHECKPOINT,
                            {"findings": [{"detail": "线索X逾期"}]})
            run = v2run.ChapterRun(chapter_num=3)
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                v2run._restore(ctx, run)
            self.assertTrue(run.canon_checked)
            self.assertEqual(len(run.canon_claims), 1)

    def test_the_report_is_deliberately_not_restored(self):
        # It is free to recompute, and recomputing it is what makes a resumed
        # chapter judged by today's gates rather than by whatever the
        # interrupted run happened to archive.
        with TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            from checkpoint import save_checkpoint
            save_checkpoint(ctx.paths, 3, v2run.ROUND0_CHECKPOINT,
                            {"block_reasons": ["stale"]})
            (ctx.paths.chapters_dir / "chapter_0003.md").write_text("正文", encoding="utf-8")
            run = v2run.ChapterRun(chapter_num=3)
            with mock.patch.object(v2run, "chapter_path",
                                   lambda p, n: p.chapters_dir / f"chapter_{n:04d}.md"):
                v2run._restore(ctx, run)
            self.assertIsNone(run.report)
            self.assertTrue(run.round0_saved)   # ...but not re-archived either


if __name__ == "__main__":
    unittest.main()
