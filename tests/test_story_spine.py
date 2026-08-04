"""P0 regression tests for original-brief grounding and story-spine review."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engine.accept import acceptance_report
from engine.bootstrap import (
    _fallback_contract_from_brief,
    _merge_contract_fallback,
    _sanitize_contract,
    extract_contract,
)
from engine.plan import _ground_card_to_story_spine, _problems, arc_user_prompt
from engine.loop import ChapterRun, Corpus, Ctx, _act_refine_card
from engine.state import build_story_state
from engine.story_spine import (
    build_story_spine,
    explicit_chapter_requirements,
    render_story_spine,
    story_spine_adherence,
)


BRIEF = """# 创作纲要

## B · 故事内核
- 金手指：「亡命快递系统」——完成危险配送获得寿命。代价：超时扣除寿命。
- 世界规则：系统永远不说谎。

## E · 结构
Ch1：陈默执行D级任务，前往废弃医院，被鬼婴追逐，靠法医知识完成配送，寿命+1月。
Ch2：连续两单C级任务，收件对象是地铁幽灵和公园石像。
第3章：B级任务发生在闹鬼写字楼，陈默发现李姐也是快递员。
Ch4：A级任务造成生死危机，陈默被迫使用系统漏洞，张医生起疑。
Ch5：最终S级任务，收件人是「死神」，揭示死亡概率交换与第一代系统快递员真相。

## F · 开写铁律
①每单必须有明确收件人和倒计时。
②陈默只准靠观察力和法医知识破局。

## G · 硬性禁止项
不许陈默主动攻击配送对象；不许系统临时改规则。
"""


def _cfg(**over):
    novel = {
        "chapter_min_chars": 20,
        "ccc_enabled": False,
        "brief_adherence_enabled": True,
        "brief_adherence_blocks_accept": True,
        "scene_dedupe_enabled": False,
        "plan_validate_deep": False,
    }
    novel.update(over)
    return {"novel": novel}


class StorySpineExtractionTest(unittest.TestCase):

    def test_literal_chapter_schedules_and_ranges_are_parsed(self):
        got = explicit_chapter_requirements(
            "Ch1：甲事件。\nCh1-2：乙事件。\n第5章：终局事件。", 5,
        )
        self.assertEqual(got[1], ["甲事件", "乙事件"])
        self.assertEqual(got[2], ["乙事件"])
        self.assertEqual(got[5], ["终局事件"])

    def test_wrong_chapter_and_invented_llm_anchors_are_discarded(self):
        contract = {"story_spine": [
            {"ch": 1, "hard_anchors": ["废弃医院", "死神", "记忆邮费"]},
            {"ch": 9, "hard_anchors": ["不存在的终局"]},
        ]}
        spine = build_story_spine(BRIEF, contract, max_chapters=5)
        anchors = spine["chapters"]["1"]["hard_anchors"]
        self.assertIn("废弃医院", anchors)
        self.assertNotIn("死神", anchors)
        self.assertNotIn("记忆邮费", anchors)
        self.assertNotIn("9", spine["chapters"])

    def test_deterministic_anchors_cover_the_same_five_chapter_brief(self):
        spine = build_story_spine(BRIEF, max_chapters=5)
        expected = {
            "1": ("废弃医院", "鬼婴", "法医知识"),
            "2": ("地铁幽灵", "公园石像"),
            "3": ("B级任务", "闹鬼写字楼", "李姐"),
            "4": ("A级任务", "系统漏洞", "张医生"),
            "5": ("S级任务", "死神", "死亡概率", "第一代系统快递员"),
        }
        for chapter, wanted in expected.items():
            with self.subTest(chapter=chapter):
                anchors = spine["chapters"][chapter]["hard_anchors"]
                for anchor in wanted:
                    self.assertIn(anchor, anchors)

    def test_render_declares_original_brief_priority(self):
        rendered = render_story_spine(build_story_spine(BRIEF, max_chapters=5))
        self.assertIn("仅来自原始简报", rendered)
        self.assertIn("不得被创意增强、卷纲或章节卡改写", rendered)
        self.assertIn("收件人必须是「死神」", rendered)


class ContractGroundingTest(unittest.TestCase):

    def test_invented_boost_rules_are_removed(self):
        dirty = {
            "protagonist": "陈默",
            "iron_rules": ["陈默只准靠观察力和法医知识破局", "每单收取一段记忆作为邮费"],
            "ability_whitelist": [
                {"name": "亡命快递系统", "scope": "完成危险配送获得寿命", "cost": "超时扣除寿命"},
                {"name": "人格分裂配送", "scope": "分裂人格代送", "cost": "失去善意"},
            ],
            "must_hold": ["存在竞争配送公司"],
            "story_spine": [],
        }
        clean = _sanitize_contract(dirty, BRIEF)
        self.assertEqual(clean["protagonist"], "陈默")
        self.assertEqual([x["name"] for x in clean["ability_whitelist"]], ["亡命快递系统"])
        blob = json.dumps(clean, ensure_ascii=False)
        for invention in ("记忆作为邮费", "人格分裂配送", "竞争配送公司", "失去善意"):
            self.assertNotIn(invention, blob)

    def test_fallback_preserves_author_rules_and_real_cost(self):
        contract = _fallback_contract_from_brief(BRIEF, 5)
        self.assertTrue(contract["iron_rules"])
        self.assertTrue(contract["banned_tropes"])
        self.assertIn("超时扣除寿命", contract["ability_whitelist"][0]["cost"])
        self.assertTrue(any("5 章" in x for x in contract["must_hold"]))

    def test_fallback_unions_with_partial_llm_result(self):
        fallback = _fallback_contract_from_brief(BRIEF, 5)
        merged = _merge_contract_fallback(
            {"iron_rules": ["陈默只准靠观察力和法医知识破局"]}, fallback,
        )
        self.assertGreaterEqual(len(merged["iron_rules"]), 2)
        self.assertTrue(merged["banned_tropes"])

    def test_provider_failure_retries_then_uses_grounded_fallback(self):
        cfg = {"novel": {"contract_enabled": True, "contract_extract_attempts": 2}}
        paths = SimpleNamespace(contract=Path("memory/contract.md"))
        with (
            patch("engine.bootstrap.call_llm", side_effect=RuntimeError("provider 403")) as call,
            patch("engine.bootstrap.write_text") as write,
            patch("engine.bootstrap.db_event"),
            patch("engine.bootstrap.log"),
        ):
            contract = extract_contract(None, paths, None, cfg, brief=BRIEF, max_chapters=5)
        self.assertEqual(call.call_count, 2)
        self.assertTrue(contract["iron_rules"])
        self.assertTrue(contract["banned_tropes"])
        write.assert_called_once()

    def test_empty_brief_cannot_fail_open(self):
        cfg = {"novel": {"contract_enabled": True, "contract_extract_attempts": 1}}
        paths = SimpleNamespace(contract=Path("memory/contract.md"))
        with (
            patch("engine.bootstrap.call_llm", side_effect=RuntimeError("provider 403")),
            patch("engine.bootstrap.write_text"),
            patch("engine.bootstrap.db_event"),
            patch("engine.bootstrap.log"),
        ):
            with self.assertRaises(RuntimeError):
                extract_contract(None, paths, None, cfg, brief="", max_chapters=0)


class SpineEnforcementTest(unittest.TestCase):

    def test_adherence_passes_with_anchors_and_fails_without_one(self):
        entry = {
            "chapter": 1,
            "requirements": ["废弃医院里的D级任务"],
            "hard_anchors": ["废弃医院", "鬼婴", "法医知识", "寿命+1月"],
        }
        passed = story_spine_adherence(entry, "陈默在废弃医院躲过鬼婴，靠法医知识完成配送，寿命+1月。")
        failed = story_spine_adherence(entry, "陈默在废弃医院完成配送。")
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn("鬼婴", failed["missing_anchors"])
        self.assertTrue(failed["directives"])

    def test_semantic_realization_of_compound_brief_labels_counts(self):
        entry = {
            "chapter": 3,
            "requirements": ["B级闹鬼写字楼住户，李姐系统版本不同"],
            "hard_anchors": ["住户", "B级任务", "系统版本", "闹鬼写字楼", "李姐"],
        }
        text = (
            "陈默进入瑞丰大厦13楼办公区，接下B级配送。收件人周会计的残影藏在夹层。"
            "李姐随后出现，她的系统面板写着‘因果锚定物回收’，陈默的却写‘配送完成’。"
        )
        result = story_spine_adherence(entry, text)
        self.assertTrue(result["passed"])
        self.assertIn("系统版本", result["matched_anchors"])
        self.assertIn("闹鬼写字楼", result["matched_anchors"])

    def test_deceased_parent_can_be_shown_as_a_fact_not_a_frozen_label(self):
        entry = {
            "chapter": 5,
            "requirements": ["已故父亲是第一代系统快递员"],
            "hard_anchors": ["已故父亲", "第一代系统快递员"],
        }
        text = "父亲六年前死于车祸。他留下的工牌写着：第一代系统快递员。"
        self.assertTrue(story_spine_adherence(entry, text)["passed"])

    def test_named_recipient_relation_cannot_be_laundered_by_mentions(self):
        entry = {
            "chapter": 5,
            "requirements": ["最终S级任务，收件人是死神本身"],
            "hard_anchors": ["死神", "S级任务", "寿命", "死亡概率"],
            "expected_grade": "S级任务",
            "named_recipient": "死神",
        }
        bad = (
            "系统发布S级任务，死亡概率升到100%。收件人是父亲，"
            "死神般的提示音宣布寿命结算。"
        )
        good = (
            "系统发布S级任务。面板写明：收件人是死神。"
            "签收后揭示死亡概率交换并完成寿命结算。"
        )
        label_good = (
            "系统发布S级任务，面板字段为：收件人【死神】；"
            "随后完成签收并揭示死亡概率交换寿命。"
        )
        codename_good = (
            "系统发布S级任务，日志写明：收件人：因果仲裁者（代号“死神”）。"
            "签收后揭示死亡概率交换寿命。"
        )
        self.assertFalse(story_spine_adherence(entry, bad)["passed"])
        self.assertTrue(story_spine_adherence(entry, good)["passed"])
        self.assertTrue(story_spine_adherence(entry, label_good)["passed"])
        self.assertTrue(story_spine_adherence(entry, codename_good)["passed"])

    def test_post_completion_refine_cannot_drop_story_spine(self):
        from commands.refine import _story_spine_acceptable

        entry = {
            "chapter": 5,
            "requirements": ["最终S级任务，收件人是死神本身"],
            "hard_anchors": ["死神", "S级任务", "死亡概率"],
            "expected_grade": "S级任务",
            "named_recipient": "死神",
        }
        bad = "收件人是死神，但这只是普通任务，签收后获得寿命。"
        good = "系统发布S级任务。收件人是死神。签收后揭示死亡概率交换。"
        self.assertFalse(_story_spine_acceptable(entry, bad)[0])
        self.assertTrue(_story_spine_acceptable(entry, good)[0])

    def test_final_chapter_must_close_instead_of_opening_a_sequel_hook(self):
        entry = {
            "chapter": 5,
            "requirements": ["最终S级任务，收件人是死神本身"],
            "hard_anchors": ["死神", "S级任务", "死亡概率"],
            "expected_grade": "S级任务",
            "named_recipient": "死神",
            "final": True,
        }
        good = (
            "系统发布S级任务。收件人是死神。死亡概率交换终止。"
            "面板永久熄灭，陈默回到清晨的街道。"
        )
        bad = good + "下一结算周期显示待定，仲裁者席位空缺。"
        self.assertTrue(story_spine_adherence(entry, good)["passed"])
        result = story_spine_adherence(entry, bad)
        self.assertFalse(result["passed"])
        self.assertTrue(any(x["type"] == "final_closure" for x in result["relation_failures"]))

    def test_critical_revelation_cannot_be_outvoted_by_other_anchor_mentions(self):
        entry = {
            "chapter": 5,
            "requirements": ["揭示死亡概率交换，父亲是第一代系统快递员"],
            "hard_anchors": ["死神", "寿命", "S级任务", "死亡概率", "第一代系统快递员"],
            "critical_anchors": ["死亡概率", "第一代系统快递员"],
            "expected_grade": "S级任务",
            "named_recipient": "死神",
            "final": True,
        }
        bad = "S级任务的收件人是死神。寿命结算后系统永久关闭，故事结束。"
        good = bad + "真相是死亡概率交换；父亲是第一代系统快递员。"
        result = story_spine_adherence(entry, bad)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_critical_anchors"], ["死亡概率", "第一代系统快递员"])
        self.assertTrue(story_spine_adherence(entry, good)["passed"])

    def test_arc_prompt_puts_spine_above_volume_plan(self):
        state = build_story_state(1, brief="纲要", bible="世界")
        prompt = arc_user_prompt(
            state, [1], volume_plan="候选：记忆邮费", story_spine="Ch1：废弃医院与鬼婴",
        )
        self.assertIn("覆盖卷纲与候选创意", prompt)
        self.assertIn("废弃医院与鬼婴", prompt)
        self.assertIn("输出前机械自检", prompt)
        self.assertIn("收件人是X", prompt)

    def test_card_validation_rejects_an_off_spine_card(self):
        entry = build_story_spine(BRIEF, max_chapters=5)["chapters"]["1"]
        card = {
            "ch": 1, "title": "记忆邮费", "where": "未完成巷",
            "who": ["陈默"], "pov_character": "陈默",
            "wants": "给无脸女人送货", "blocked_by": "另一家配送公司",
            "turn": "陈默分裂出第二人格", "payoff": "获得善意货币",
            "payoff_type": "reveal", "conflict_type": "institution",
            "beats": ["进入未完成巷", "支付记忆邮费", "获得善意货币"],
            "exit_hook": "无脸女人再次下单", "opening_type": "physical_action",
            "forbid": [],
        }
        with (
            patch("engine.plan._continuity", return_value=([], [])),
            patch("engine.plan._scene_sim", return_value=None),
            patch("engine.plan.story_spine_entry", return_value=entry),
        ):
            problems, _ = _problems(None, None, _cfg(), {"cards": {}}, card, 1)
        self.assertTrue(any(p.startswith("story_spine:") for p in problems))

    def test_card_grounding_copies_brief_without_laundering_wrong_recipient(self):
        entry = {
            "requirements": ["最终S级任务，收件人是死神本身", "父亲是第一代系统快递员"],
            "hard_anchors": ["S级任务", "死神", "第一代系统快递员"],
            "expected_grade": "S级任务",
            "named_recipient": "死神",
            "final": True,
        }
        neutral = {"ch": 5, "who": ["陈默"], "beats": ["陈默登上天台"]}
        grounded = _ground_card_to_story_spine(neutral, entry)
        text = json.dumps(grounded, ensure_ascii=False)
        self.assertIn("父亲是第一代系统快递员", text)
        self.assertIn("收件人是死神", text)

        wrong = {"ch": 5, "who": ["陈默", "父亲"], "beats": ["收件人是父亲。死神旁观。"]}
        grounded_wrong = _ground_card_to_story_spine(wrong, entry)
        wrong_text = json.dumps(grounded_wrong, ensure_ascii=False)
        self.assertNotIn("系统面板必须逐字明确：收件人是死神", wrong_text)

    def test_prose_is_rejected_against_brief_even_when_card_is_consistent(self):
        entry = {
            "chapter": 1,
            "requirements": ["废弃医院、鬼婴、法医知识"],
            "hard_anchors": ["废弃医院", "鬼婴", "法医知识"],
        }
        card = {
            "where": "未完成巷", "turn": "陈默支付记忆邮费",
            "exit_hook": "无脸女人再次下单", "beats": ["支付记忆邮费"],
        }
        text = "第1章\n\n" + "陈默走进未完成巷，向无脸女人支付一段记忆作为邮费。" * 30
        report = acceptance_report(1, text, card, _cfg(), story_spine_entry=entry)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["brief_adherence"]["passed"])
        self.assertTrue(any(g.get("gate") == "brief_adherence" for g in report["gate_rejects"]))
        self.assertFalse(any("原始简报" in d for d in report["writer_directives_for_next_chapter"]))
        self.assertTrue(any("原始简报" in d for d in report["brief_adherence"]["directives"]))

    def test_card_refinement_cannot_remove_original_brief_anchors(self):
        entry = {
            "chapter": 4,
            "requirements": ["A级任务、系统漏洞、张医生起疑"],
            "hard_anchors": ["A级任务", "系统漏洞", "张医生"],
        }
        original = {
            "ch": 4, "title": "A级任务", "turn": "陈默被迫使用系统漏洞",
            "exit_hook": "张医生看着CT片起疑", "thread_actions": ["张医生起疑"],
        }
        degraded = dict(original, exit_hook="系统弹出下一单", thread_actions=[])
        run = ChapterRun(chapter_num=4, card=original, corpus=Corpus(story_spine=entry))
        run.plan, run.decision = {}, {}
        ctx = Ctx(client=None, paths=SimpleNamespace(), conn=None, config={})
        with (
            patch("engine.plan.refine_card", return_value=(degraded, ["删除张医生线索"])),
            patch("engine.plan.load_cards", return_value={"cards": {}}),
            patch("engine.loop.load_checkpoint", return_value=None),
            patch("engine.loop.log"),
        ):
            result = _act_refine_card(ctx, run)
        self.assertEqual(result, "refine[0]")
        self.assertEqual(run.card, original)


if __name__ == "__main__":
    unittest.main()
