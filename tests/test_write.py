"""v2 writing: the two-part response, and the prompt contract behind it.

Two things here can fail silently in production and be invisible for a whole
book, so both get a test that fails loudly at build time instead:

1. **The output-section swap.** v1's section forbids JSON; v2's demands it. If
   `writing.py`'s assembly changes and the exact-string replacement stops
   matching, an appended override would leave the writer with two contradictory
   rules — and the weaker (newer, appended) one loses. Symptom: every chapter's
   delta comes back `missing`, `run.py` pays an extraction call, and the cost
   claim the whole A/B measures is wrong. `test_a_changed_v1_section_is_a_loud_failure`
   is what turns that into a test failure.

2. **The checklist quoting anchors the gate does not grep for.** CCC is a string
   matcher over `quality._beat_anchor_fragments`. A checklist derived from any
   other rule teaches the writer to satisfy a different contract than the one
   scored, which would look like "v2 doesn't improve CCR" rather than like a bug.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import engine.loop as _loop
import engine.write as write
import engine.store as store
import engine.write as writing
from engine.config import Paths
from tests.conftest import make_paths


def _paths(root: Path) -> Paths:
    return make_paths(root)


def _config(**over) -> dict:
    novel = {"style_preset": "history", "chapter_words": 4000,
             "chapter_min_chars": 2800, "chapter_max_chars": 5200,
             "preflight_constraints_enabled": False,
             "exemplar_rag_enabled": False, "rag_enabled": False}
    novel.update(over)
    return {"novel": novel, "api": {"temperature": 0.85, "metrics_enabled": False}}


def _card(**over) -> dict:
    card = {
        # `where` is 4 chars on purpose: `quality._beat_anchor_fragments` keeps
        # only 2-8 char fragments, so "县医院三楼旧档案室" (9, unsplittable) yields
        # NO anchors and is unjudgeable by CCC. A fixture like that would make
        # the anchor-agreement test vacuous.
        "ch": 7, "title": "第7章 旧档案室", "where": "旧档案室",
        "who": ["汤舒婷", "顾峥"], "turn": "顾峥把铜钥匙拍在柜面上",
        "payoff": "病历第三页少了一张化验单",
        "beats": ["汤舒婷推开档案室的门", "两人翻到第三页"],
        "exit_hook": "走廊尽头的灯忽然全灭了",
        "forbid": ["声音压得很低"],
    }
    card.update(over)
    return card


def _state(**over) -> _loop.StoryState:
    base = dict(brief="创作纲要", bible="世界设定", voice="叙述声音", card=_card())
    base.update(over)
    return _loop.build_story_state(7, **base)


class FakeCall:
    """Stands in for `llm.call_llm`. Records the call; returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, client, paths, config, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        return self.response


def _response(prose: str, delta: dict | None = None) -> str:
    if delta is None:
        return prose
    return (prose + "\n" + write.DELTA_SENTINEL + "\n"
            + json.dumps(delta, ensure_ascii=False))


_PROSE = "第7章 旧档案室\n\n" + "汤舒婷推开门。" * 200


# ---------------------------------------------------------------------------


class SystemPromptTest(unittest.TestCase):

    def _system(self):
        return write.build_system(_config(), 7, "旧档案室")

    def test_the_v1_output_section_is_replaced_not_merely_followed(self):
        system = self._system()
        v1 = write._OUTPUT_SECTION.format(
            chapter_words=4000, chapter_num=7, title="旧档案室")
        self.assertNotIn(v1, system)
        self.assertNotIn("严禁输出\"写前自我审查\"、\"Pre-writing Self-Review\"、\"分析\"、"
                         "\"reasoning\"、`<analysis>`、`<thinking>`、代码围栏、JSON", system)
        self.assertIn(write.DELTA_SENTINEL, system)
        self.assertIn("状态增量", system)

    def test_the_prose_doctrine_is_v1s_untouched(self):
        # The redesign's claim is about architecture, not about style teaching.
        # If v2 also rewrote the genre doctrine, the A/B would be measuring two
        # variables and could not attribute a win to either.
        system = self._system()
        self.assertIn(writing.ANTI_FRAGMENT_BAN, system)
        self.assertIn("<写作纪律>", system)
        self.assertIn(writing.GENRE_PROFILES["history"]["structure_template"], system)

    def test_the_sentinel_in_the_prompt_is_the_one_the_parser_splits_on(self):
        prose, tail = write.split_response(
            "正文\n" + write.DELTA_SENTINEL + "\n{\"events\": []}")
        self.assertEqual(prose.strip(), "正文")
        self.assertEqual(tail, "{\"events\": []}")

    def test_the_schema_names_exactly_the_fields_the_delta_reads(self):
        schema = json.loads(write.DELTA_SCHEMA)
        self.assertEqual(sorted(schema), sorted(
            ["events", "entities", "threads", "protagonist_state",
             "next_12_directions"]))
        # v1's spellings, because `canon.apply_delta` hands this to
        # `writing.update_structured_state` unchanged.
        self.assertIn("summary", schema["events"][0])
        self.assertIn("state_patch", schema["entities"][0])
        self.assertIn("id", schema["threads"][0])
        # A dict, not v1's markdown: `ChapterDelta.from_payload` only keeps dicts.
        self.assertIsInstance(schema["protagonist_state"], dict)

    def test_a_changed_v1_section_is_a_loud_failure_not_a_silent_append(self):
        with mock.patch.object(write, "_build_write_system",
                               return_value="<角色>写手</角色>"):
            with self.assertRaises(write.WriteError):
                self._system()

    def test_two_copies_of_the_section_are_also_a_failure(self):
        doubled = write._OUTPUT_SECTION.format(
            chapter_words=4000, chapter_num=7, title="旧档案室") * 2
        with mock.patch.object(write, "_build_write_system", return_value=doubled):
            with self.assertRaises(write.WriteError):
                self._system()

    def test_the_title_prefix_is_not_doubled(self):
        self.assertEqual(write.clean_title({"title": "第7章 旧档案室"}, 7), "旧档案室")
        self.assertEqual(write.clean_title({"title": "第七章：旧档案室"}, 7), "旧档案室")
        self.assertEqual(write.clean_title({}, 7), "第7章")


class TokenBudgetTest(unittest.TestCase):

    def test_the_cap_leaves_room_for_the_delta(self):
        # v1 sizes the response for prose alone. Without headroom the JSON is the
        # thing that gets truncated, and a truncated delta is indistinguishable
        # from a disobeyed instruction.
        cfg = _config()
        self.assertEqual(write.max_tokens(cfg),
                         write._chapter_write_max_tokens(cfg)
                         + write.DELTA_TOKEN_HEADROOM)

    def test_an_uncapped_config_stays_uncapped(self):
        cfg = _config(chapter_length_cap_enabled=False)
        self.assertIsNone(write.max_tokens(cfg))


class SplitTest(unittest.TestCase):

    def test_a_tolerant_sentinel_still_splits(self):
        for line in ("===状态增量===", "==== 状态增量 ====", "=== STATE_DELTA ===",
                     "===DELTA===", "## ===状态增量===", "> ===状态增量==="):
            with self.subTest(line=line):
                prose, tail = write.split_response("正文内容\n" + line + "\n{\"events\": []}")
                self.assertEqual(prose.strip(), "正文内容")
                self.assertEqual(tail, "{\"events\": []}")

    def test_the_last_sentinel_wins_so_the_prose_stays_whole(self):
        # If the model echoes the sentinel mid-chapter, the failure must cost the
        # delta, never the chapter: the delta is re-derivable and the prose is not.
        raw = ("上半章\n" + write.DELTA_SENTINEL + "\n下半章\n"
               + write.DELTA_SENTINEL + "\n{\"events\": []}")
        prose, tail = write.split_response(raw)
        self.assertIn("上半章", prose)
        self.assertIn("下半章", prose)
        self.assertEqual(tail, "{\"events\": []}")

    def test_a_fenced_json_tail_is_unfenced(self):
        raw = "正文\n" + write.DELTA_SENTINEL + "\n```json\n{\"events\": []}\n```"
        self.assertEqual(write.split_response(raw)[1], "{\"events\": []}")

    def test_no_sentinel_but_a_real_delta_at_the_end_is_still_found(self):
        raw = "正文内容\n\n{\"events\": [], \"threads\": []}"
        prose, tail = write.split_response(raw)
        self.assertEqual(prose.strip(), "正文内容")
        self.assertIn("threads", tail)

    def test_a_brace_in_the_prose_is_not_mistaken_for_a_delta(self):
        # The guard that makes the no-sentinel fallback safe: a trailing JSON
        # object counts only when it carries delta keys. Without it, dialogue
        # containing a brace would slice the chapter in half.
        raw = "他在纸上写下\n\n{\"暗号\": \"子时三刻\"}\n\n然后把纸烧了。"
        prose, tail = write.split_response(raw)
        self.assertEqual(prose, raw)
        self.assertEqual(tail, "")

    def test_no_delta_at_all_returns_the_whole_thing_as_prose(self):
        self.assertEqual(write.split_response("只有正文"), ("只有正文", ""))
        self.assertEqual(write.split_response(""), ("", ""))
        self.assertEqual(write.split_response(None), ("", ""))


class ParseDeltaTest(unittest.TestCase):

    def test_a_clean_response_yields_prose_and_delta(self):
        raw = _response("正文", {"events": [{"summary": "开锁"}],
                                 "threads": [{"id": "t1"}]})
        prose, delta, status = write.parse_delta(raw)
        self.assertEqual(status, "ok")
        self.assertEqual(prose.strip(), "正文")
        self.assertEqual(len(delta.events), 1)
        self.assertEqual(len(delta.threads), 1)

    def test_a_missing_delta_is_reported_not_guessed(self):
        prose, delta, status = write.parse_delta("只有正文")
        self.assertEqual(status, "missing")
        self.assertTrue(delta.empty)
        self.assertEqual(prose, "只有正文")

    def test_an_unparsable_delta_costs_the_delta_and_keeps_the_prose(self):
        raw = "正文正文\n" + write.DELTA_SENTINEL + "\n{这不是 JSON"
        prose, delta, status = write.parse_delta(raw)
        self.assertEqual(status, "unparsed")
        self.assertTrue(delta.empty)
        self.assertEqual(prose.strip(), "正文正文")

    def test_the_repair_call_runs_only_when_a_client_is_available(self):
        raw = "正文\n" + write.DELTA_SENTINEL + "\n{坏掉的"
        with mock.patch.object(write, "load_json_with_repair") as repair:
            write.parse_delta(raw)
            repair.assert_not_called()
            repair.return_value = {"events": [{"summary": "修好了"}]}
            _, delta, status = write.parse_delta(
                raw, client=object(), paths=_paths(Path(".")), config=_config())
            self.assertEqual(status, "repaired")
            self.assertEqual(len(delta.events), 1)

    def test_a_repair_that_returns_junk_does_not_launder_it_into_a_delta(self):
        raw = "正文\n" + write.DELTA_SENTINEL + "\n{坏掉的"
        with mock.patch.object(write, "load_json_with_repair",
                               return_value={"nonsense": 1}):
            _, delta, status = write.parse_delta(
                raw, client=object(), paths=_paths(Path(".")), config=_config())
        self.assertEqual(status, "unparsed")
        self.assertTrue(delta.empty)


class ChecklistTest(unittest.TestCase):
    """One ruler: the checklist quotes what `_loop.contract_fulfilment` greps."""

    def test_every_quoted_fragment_is_one_the_gate_actually_searches_for(self):
        card = _card()
        body = write.contract_checklist(card, _config())
        for field in ("where", "turn", "exit_hook", "payoff"):
            anchors = _loop._anchors(card[field])
            self.assertTrue(anchors, f"fixture field {field} has no anchors")
            quoted = [a for a in anchors[:4] if f"「{a}」" in body]
            self.assertEqual(quoted, anchors[:4],
                             f"{field}: checklist and gate disagree on the anchors")

    def test_the_hook_carries_its_tail_budget(self):
        body = write.contract_checklist(_card(), _config())
        self.assertIn(str(_loop.DEFAULT_TAIL_CHARS), body)
        self.assertIn("走廊尽头的灯忽然全灭了", body)

    def test_every_name_is_named(self):
        body = write.contract_checklist(_card(), _config())
        self.assertIn("汤舒婷", body)
        self.assertIn("顾峥", body)

    def test_the_ban_list_is_present_and_quoted(self):
        self.assertIn("「声音压得很低」", write.contract_checklist(_card(), _config()))

    def test_an_unjudgeable_target_says_so_instead_of_quoting_nothing(self):
        # A field the gate cannot score leaves the CCR denominator. Telling the
        # writer that is more useful than an empty "verified" line. A fully
        # abstract phrase with only stop tokens/generic fragments produces [].
        body = write.contract_checklist(_card(where="他突然意识到了其中的问题"), _config())
        self.assertEqual(_loop._anchors("他突然意识到了其中的问题"), [])
        self.assertIn("验收无法判定", body)

    def test_long_location_fragment_is_judgeable(self):
        # Compound location names (9-16 chars) must be kept as anchors,
        # not dropped — the old 8-char ceiling silently skipped them.
        anchors = _loop._anchors("县医院三楼旧档案室")
        self.assertEqual(anchors, ["县医院三楼旧档案室"])

    def test_an_empty_card_produces_no_block(self):
        self.assertEqual(write.contract_checklist(None, _config()), "")
        self.assertEqual(write.contract_checklist({}, _config()), "")


class ThreadLedgerTest(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)
        self.conn = store.init_db(self.paths)
        self.addCleanup(self.conn.close_current)

    def _thread(self, tid, desc, status="open", due=None):
        # `open_threads` has no store-level writer — `update_structured_state`
        # writes it with raw SQL — so the fixture does the same.
        self.conn.execute(
            "INSERT INTO open_threads(id, description, status, thread_type, "
            "introduced_chapter, due_chapter, updated_chapter, payload_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (tid, desc, status, "plot", 3, due, 6, "{}"))
        self.conn.commit()

    def test_open_threads_are_listed_with_their_ids(self):
        self._thread("tongyaoshi", "铜钥匙的来历未交代", due=20)
        body = write.thread_ledger(self.conn, 7)
        self.assertIn("tongyaoshi", body)
        self.assertIn("铜钥匙的来历未交代", body)
        self.assertIn("原样复用", body)

    def test_nothing_open_means_no_block(self):
        self.assertEqual(write.thread_ledger(self.conn, 7), "")

    def test_a_broken_connection_degrades_to_silence(self):
        self.assertEqual(write.thread_ledger(None, 7), "")


class UserPromptTest(unittest.TestCase):

    def test_the_stable_half_rides_the_cache_key_not_the_user_message(self):
        state = _state()
        user = write.build_user(state, _card(), 7, "旧档案室", _config())
        self.assertNotIn("世界设定", user)
        self.assertIn("旧档案室", user)

    def test_the_obligations_come_after_the_story_context(self):
        user = write.build_user(
            state=_state(), card=_card(), chapter_num=7, title="旧档案室",
            config=_config(), capsule="能力白名单：无")
        self.assertLess(user.index("本章卡片"), user.index("本章契约自查表"))
        self.assertLess(user.index("本章契约自查表"), user.index("能力边界"))

    def test_the_two_part_format_is_restated_in_the_request(self):
        user = write.build_user(_state(), _card(), 7, "旧档案室", _config())
        self.assertIn(write.DELTA_SENTINEL, user)

    def test_the_length_band_is_stated_and_excludes_the_delta(self):
        user = write.build_user(_state(), _card(), 7, "T", _config())
        self.assertIn("2800-5200", user)
        self.assertIn("不计入字数", user)

    def test_empty_pieces_leave_no_empty_headers(self):
        user = write.build_user(_state(), _card(), 7, "T", _config())
        self.assertNotIn("线索台账", user)
        self.assertNotIn("本章避雷", user)
        self.assertNotIn("规划阶段带过来", user)


class NegativeBlockTest(unittest.TestCase):

    def test_nothing_measured_means_no_block(self):
        self.assertEqual(write.negative_block(None), "")
        self.assertEqual(write.negative_block({}), "")
        self.assertEqual(write.negative_block(
            {"items": [], "fossils": [], "style_warnings": []}), "")

    def test_fossils_are_quoted_verbatim(self):
        body = write.negative_block({"fossils": ["声音压得很低"], "items": ["少用破折号"]})
        self.assertIn("「声音压得很低」", body)
        self.assertIn("少用破折号", body)


class WriteChapterTest(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)
        self.conn = store.init_db(self.paths)
        self.addCleanup(self.conn.close_current)

    def _write(self, response, **kw):
        call = FakeCall(response)
        res = write.write_chapter(None, self.paths, self.conn, _config(), 7,
                                  _card(), _state(), call=call, **kw)
        return res, call

    def test_one_call_produces_both_halves(self):
        res, call = self._write(_response(_PROSE, {
            "events": [{"summary": "开锁"}],
            "protagonist_state": {"目标": "拿到病历"}}))
        self.assertEqual(len(call.calls), 1)
        self.assertEqual(call.calls[0]["tag"], "write")
        self.assertEqual(res.delta_status, "ok")
        self.assertTrue(res.delta_ok)
        self.assertEqual(res.delta.protagonist_state, {"目标": "拿到病历"})
        self.assertIn("汤舒婷推开门", res.text)
        self.assertNotIn(write.DELTA_SENTINEL, res.text)

    def test_the_stable_prefix_is_the_cache_key_and_is_not_inlined(self):
        state = _state()
        call = FakeCall(_response(_PROSE, {"events": []}))
        write.write_chapter(None, self.paths, self.conn, _config(), 7, _card(),
                            state, call=call)
        self.assertEqual(call.calls[0]["cacheable_prefix"], state.stable_prefix())
        self.assertTrue(call.calls[0]["cacheable_prefix"])
        self.assertNotIn(state.stable_prefix(), call.calls[0]["user"])

    def test_a_refusal_raises_rather_than_becoming_a_200_char_chapter(self):
        # v1 only discovers this at `save_chapter`, three stages downstream.
        with self.assertRaises(write.WriteError):
            self._write("抱歉，我无法continue这个请求。")

    def test_a_lost_delta_does_not_lose_the_chapter(self):
        res, _ = self._write(_PROSE + "\n" + write.DELTA_SENTINEL + "\n{坏的")
        self.assertEqual(res.delta_status, "unparsed")
        self.assertIn("汤舒婷推开门", res.text)

    def test_leaked_reasoning_is_stripped_by_v1s_normaliser(self):
        raw = _response("<thinking>先想想</thinking>\n" + _PROSE, {"events": []})
        res, _ = self._write(raw)
        self.assertNotIn("先想想", res.text)

    def test_the_temperature_is_the_configs_unless_overridden(self):
        _, call = self._write(_response(_PROSE, {"events": []}))
        self.assertAlmostEqual(call.calls[0]["temperature"], 0.85)
        _, call = self._write(_response(_PROSE, {"events": []}), temperature=0.7)
        self.assertAlmostEqual(call.calls[0]["temperature"], 0.7)

    def test_planning_constraints_reach_the_prompt(self):
        _, call = self._write(_response(_PROSE, {"events": []}),
                              constraints=["必须在本章兑现铜钥匙"])
        self.assertIn("必须在本章兑现铜钥匙", call.calls[0]["user"])


class DeltaTailAnchorTest(unittest.TestCase):
    """The format obligation must be the LAST thing the writer reads.

    Not a style preference — a measured one. deepseek-v4-pro honoured the
    two-part contract in 1 of 2 smoke chapters (the miss came back 5,015 chars
    of pure prose, `ok=True`, nothing truncated), which is the same weak
    instruction-following the repo already answers with tail anchors four times
    over. An anchor that stops being last has silently stopped being an anchor,
    and the only symptom would be a slow drip of extraction calls in the A/B's
    cost column.
    """

    def test_the_anchor_is_the_final_text_of_the_prompt(self):
        user = write.build_user(_state(), _card(), 7, "旧档案室", _config(),
                                capsule="能力白名单：无")
        self.assertTrue(user.endswith(write.delta_tail_anchor()))
        # After the ability capsule specifically: a breached capsule still
        # leaves a chapter to repair, a missing second part leaves nothing to
        # read the book's state from.
        self.assertLess(user.index("能力边界"), user.index("最后读"))

    def test_the_anchor_names_the_sentinel_the_parser_greps_for(self):
        # A drifted sentinel would teach the writer a marker `split_response`
        # does not look for — every delta would come back `missing` while the
        # prompt looked correct.
        self.assertIn(write.DELTA_SENTINEL, write.delta_tail_anchor())


class BackfillDeltaTest(unittest.TestCase):
    """The one fallback call in v2 — bought rather than skipped, and visible.

    Skipping is not the cheap choice it looks like: `canon.load` projects
    facts / threads / recent out of what the delta writes, so a writer that
    keeps missing it goes blind a chapter at a time and the A/B measures a
    crippled v2 instead of the proposed one. Spending a cheap tagged call keeps
    the book sighted AND keeps the cost claim honest, since `compare._llm_totals`
    counts anything in `llm_calls.jsonl`.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)

    def _backfill(self, response, text="正文" * 400):
        call = FakeCall(response)
        delta, status = write.backfill_delta(
            None, self.paths, _config(), 7, text, call=call)
        return delta, status, call

    def test_good_json_becomes_a_delta_on_the_cheap_route(self):
        delta, status, call = self._backfill(json.dumps({
            "events": [{"summary": "顾峥交出铜钥匙"}],
            "protagonist_state": {"目标": "拿到病历"}}, ensure_ascii=False))
        self.assertEqual(status, "backfilled")
        self.assertEqual(len(delta.events), 1)
        self.assertEqual(delta.protagonist_state, {"目标": "拿到病历"})
        self.assertEqual(len(call.calls), 1)
        # The tag is load-bearing twice over: `llm._ROLE_ROUTING` sends it to
        # the extraction model (the point of the fallback is that it costs a
        # fraction of the write it repairs), and the cost report counts it.
        self.assertEqual(call.calls[0]["tag"], "delta_backfill")

    def test_a_fenced_reply_still_parses(self):
        delta, status, _ = self._backfill(
            "```json\n" + json.dumps({"events": [{"summary": "开锁"}]}) + "\n```")
        self.assertEqual(status, "backfilled")
        self.assertEqual(len(delta.events), 1)

    def test_junk_is_reported_missing_rather_than_invented(self):
        # An empty delta that claims success would write "nothing happened this
        # chapter" into the projection — worse than the gap it replaces.
        delta, status, _ = self._backfill("我读完了这一章，感觉不错。")
        self.assertEqual(status, "missing")
        self.assertFalse(delta.events)

    def test_a_gateway_failure_is_absorbed_not_raised(self):
        def boom(*a, **kw):
            raise RuntimeError("504")
        delta, status = write.backfill_delta(
            None, self.paths, _config(), 7, "正文" * 400, call=boom)
        self.assertEqual(status, "missing")
        self.assertFalse(delta.events)

    def test_no_prose_means_no_call_at_all(self):
        call = FakeCall("{}")
        _, status = write.backfill_delta(
            None, self.paths, _config(), 7, "   ", call=call)
        self.assertEqual(status, "missing")
        self.assertEqual(call.calls, [])

    def test_the_finished_prose_is_what_gets_read(self):
        _, _, call = self._backfill(json.dumps({"events": []}),
                                    text="汤舒婷推开旧档案室的门。" * 40)
        self.assertIn("汤舒婷推开旧档案室的门", call.calls[0]["user"])
        self.assertTrue(call.calls[0].get("json_mode"))


if __name__ == "__main__":
    unittest.main()
