"""Novel-text → screenplay (短剧剧本) converter.

Standalone tool: takes ANY plain-text / markdown novel file and rewrites it as a
shooting-style Chinese 短剧 script, matching the reference format:

    第 N 集
    N-N  地点  时/日夜  内/外
    人物：角色A、角色B
    △动作描述行（场景/调度/特写）
    角色：台词
    （字幕：身份提示）
    角色（OS）：旁白/内心独白
    （镜头特写说明）

Design notes (mirrors trial.py):
  * It is decoupled from novels/ — no novel name required. It reuses the engine's
    config-driven LLM client purely to obtain API keys / endpoints, defaulting to
    config_template.yaml (which carries the shared keys) when no --config is given.
  * Long input is split into "chapters" (第N章 …) or, lacking chapter markers, into
    char-budgeted segments. Each segment is one LLM call with continuity carry-over
    (running episode number, running scene index, last episode's tail) so scene
    numbering and 集 numbering stay monotonic across calls.
  * Per-segment checkpoints under <out>.checkpoints/seg_NNNN.json make the pass
    resumable: a re-run skips segments already converted.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from config import Paths, get_paths, load_config, log, normalize_text, safe_score
from llm import call_llm, json_prompt, load_json_with_repair

ROOT = Path(__file__).resolve().parent


SCRIPT_SYSTEM = """你是头部竖屏短剧（番茄/红果/抖音短剧）的资深改编编剧。
任务：把给定的小说正文改编成可直接进组开拍的分集分场剧本。忠于原文主线事件、人物关系与因果，但要按短剧的戏剧节奏重新组织，而不是逐句翻译小说。

# 输出格式（严格遵守，不要输出任何解释、点评、代码围栏或多余空行）
第 {episode} 集
{episode}-1  地点  时段  内/外
人物：本场真正有戏份的角色，用、隔开
△一个可一镜拍到的动作或画面（镜头说明就近内联写在被拍动作后的括号里）
角色名：台词
（字幕：身份/信息提示）
角色名（OS）：画外音/旁白/内心独白

# 改编铁律（按短剧标准，不是按小说标准）
1. 【场景要粗】一集只切 1-3 场。只有当【地点真正改变】或【时间跳转】时才开新场号；同一空间里的连续动作绝不拆成多场。把小说里的空间微移合并进同一场的△行。
2. 【△要粗不要细】一场戏 10-15 条△即可。连续的碎动作要合并写，只有当【视角切换】或【人物/道具变化】时才另起△。
   ❌ 错误：△林澈低头。 △林澈看手机。 △林澈按键。
   ✅ 正确：△林澈低头看手机，按下通话键。
   严禁把小说的整段描写搬进一条△，严禁在△里写心理活动、观感或形容评价（如"显得阴沉破败""惊悚感"）。
3. 【镜头要专业·就近内联】镜头说明【紧跟在被拍的那个动作/物件后面，用括号内联写进△行里】，不要单独另起一行写"（镜头：…）"。一个△动作可以挂多个镜头括号，分别贴在各自要强调的点后面；复杂运镜可在一个括号里连续描述调度过程。
   在真正需要强调的地方才挂镜头：【角色首次亮相】【关键反转】【重要道具/物证特写】【情绪炸点】【环境压迫感】。用多用少由剧情决定，不要每条△都挂镜头，也不要刻意凑数。每个镜头括号必须写清【景别+运镜（+机位/落点/调度）】，让摄影师能直接照拍，不要只写一个词。
   景别：大特写/特写/局部特写/近景/中景/全景/小全景/大全景；运镜：推/拉/摇/移/跟/甩/升降/环绕/手持晃/后拉；机位：俯拍/仰拍/平视/过肩/主观视角。
   ❌ 太简单：（特写）
   ❌ 单独成行：△陈天推开大门。\n（镜头：手开门特写）
   ✅ 就近内联：△陈天身披道袍（道袍局部特写），头戴偃月冠（头冠特写），在昏暗过道行走（后背跟拍）。
   ✅ 复杂调度：△陈天推开大门（手开门特写），门内供奉着祖师神像和法坛（法坛小全景后拉，男主背身走到法坛前，房间大全景，男主手部特写点香祭拜）。
   镜头括号里只写可执行的拍摄语言，不写情绪评价（如"惊悚地""压抑地"）。
4. 【身份卡】每个有名有姓的主要角色【首次出场】时，必须紧跟一条（字幕：身份 姓名），让观众秒认人。
5. 【开场钩子】每一集开头前两行必须立刻抛出强钩子：异常事件、直接冲突、或一句立人设的台词/OS。禁止用环境铺垫开场。
6. 【集尾留人】每一集结尾必须落在一个反转、悬念或情绪炸点上，让观众想点下一集。
7. 【台词短促】台词口语化、短、带冲突或信息量，一来一回推进剧情。删掉小说里的长段独白式对白，拆成几句你来我往。
8. 【人物行干净】人物行只列本场真正有台词或关键动作的角色，不要写"陆续出现""群众若干"这类小说化注释（群演用"路人""邻居"概括即可）。
9. 【OS 铁律】OS 只能用于【电话/录音/画外音等真实的场外声音】，格式必须是"角色名（OS）：..."或"未知声音（OS）：..."。不能用于角色内心独白或思考判断。角色思考直接省略，或用△表情/动作外化。
   ❌ 错误：林澈（OS）：她在撒谎。 ← 这是内心判断，不是说出来的话
   ✅ 正确：△林澈盯着她，察觉异样。 ← 用行为暗示判断
10. 只输出本批次小说内容对应的剧本，不要补写后续剧情，不要自行加戏或改写结局。"""


SCRIPT_EXTRACT_SYSTEM = """你是小说改编短剧的结构化整理编辑。
任务：只基于给定小说片段，抽取后续编剧必须遵守的事实材料。不得扩写、不得预告后文、不得编造原文没有发生的事件。
只返回恰好一个合法 JSON 对象，不要输出解释。

schema:
{
  "characters": [
    {
      "name": "人物名或临时称谓",
      "identity": "观众首见字幕可用的身份，未知则写未知",
      "goal": "本片段内看得见的目标",
      "state": "情绪/身体/关系状态",
      "must_subtitle": true
    }
  ],
  "locations": ["可拍摄地点，按出现顺序"],
  "events": ["按因果顺序列出必须保留的事件"],
  "conflicts": ["人物目标与阻力，必须具体"],
  "reversals": ["误会、信息翻转、身份翻转、代价揭示等"],
  "visual_props": ["能被镜头拍到且推动剧情的道具/物证/环境细节"],
  "must_keep_dialogue": ["原文中必须保留或近似保留的关键台词"],
  "locked_facts": ["不可改动的事实边界、结局状态、人物关系"],
  "open_threads": ["本片段结束时留下的悬念或待追问信息"]
}"""


SCRIPT_PLAN_SYSTEM = """你是头部竖屏短剧的分集分场策划。
任务：基于结构化材料和原文，生成可执行 episode_plan，供下一步编剧严格照写。忠于原文事实，不自行补后续。
只返回恰好一个合法 JSON 对象，不要输出解释。

硬性规则：
1. 从给定 start_episode 开始编号。
2. 每集必须有 1 个开场钩子、2-4 个戏剧节拍、1 个追看卡点。
3. 每场写清地点、人物、人物目标、阻力、信息变化、结尾转折。
4. 每场都要给 dialogue_mandate：本场至少哪些角色发生对话/交锋；不要把整场设计成无声动作流水账。
5. 动作只按节拍组织，不按每个微动作拆分。

schema:
{
  "episodes": [
    {
      "episode": 1,
      "opening_hook": "前两行必须出现的异常事件/冲突/OS",
      "beats": ["2-4 个戏剧节拍"],
      "retention_hook": "集尾追看卡点",
      "scenes": [
        {
          "scene_no": "1-1",
          "location": "地点",
          "time": "日/夜/晨/昏",
          "interior": "内/外",
          "characters": ["本场有台词或关键动作的人物"],
          "character_goals": {"人物名": "本场目标"},
          "obstacle": "阻力/误会/危险",
          "information_change": "本场结束观众新知道什么",
          "turn": "本场结尾转折",
          "dialogue_mandate": "本场必须发生的对白交锋",
          "visual_beats": ["按戏剧节拍列出的可拍画面，不超过8条"]
        }
      ]
    }
  ],
  "global_constraints": ["全段剧本必须遵守的忠实度/人物/道具约束"]
}"""


SCRIPT_FROM_PLAN_SYSTEM = """你是头部竖屏短剧资深改编编剧。
任务：严格依据 source_packet 和 episode_plan，把小说片段改写成分集分场剧本。你可以压缩、合并、重排原文信息，但不得突破 locked_facts，不得补写后续剧情。

# 输出格式（严格遵守，不要输出任何解释、点评、代码围栏或多余空行）
第 N 集
N-1  地点  时段  内/外
人物：本场真正有戏份的角色，用、隔开
△可拍动作或画面（镜头说明就近内联写在被拍动作后的括号里）
角色名：台词
角色名（OS）：画外音/旁白（仅用于真实的场外声音，如电话、录音，不用于内心独白）
（字幕：身份 姓名）

# 硬性写作规则
1. 必须按 episode_plan 的集号、场号、地点、人物、开场钩子、场尾转折、集尾追看卡点来写。
2. 每个有名有姓的主要角色首次出场后，必须尽快给一条（字幕：身份 姓名）。若身份未知，写（字幕：未知身份 姓名）。
3. OS 必须有说话主体，格式只能是"角色名（OS）：..."或"未知声音（OS）：..."，且仅用于【真实的场外声音】（电话、录音、画外音）。禁止用于内心独白或思考判断。角色思考直接省略，或用△表情/动作外化。
4. △要粗不要细：一场 10-15 条△即可。连续的碎动作要合并写，只有当视角切换或人物/道具变化时才另起△。每条△只承载一个清晰可拍的画面或动作，不写心理判断。
5. 动作行数量不得超过对白行数量的 3 倍；若原文本来动作多，也要用短对白、手机文字、质问、反问制造交锋。
6. 台词短促、口语化、带冲突或信息量。不要把推理和背景塞进长动作行。
7. 【镜头要专业·就近内联】镜头说明【紧跟在被拍的那个动作/物件后面，用括号内联写进△行里】，不要单独另起一行写"（镜头：…）"。一个△动作可以挂多个镜头括号，分别贴在各自要强调的点后面；复杂运镜可在一个括号里连续描述调度过程。在真正需要强调的地方才挂镜头：【角色首次亮相】【关键反转】【重要道具/物证特写】【情绪炸点】。用多用少由剧情决定，不要每条△都挂镜头，也不要刻意凑数或省略。每个镜头括号必须写清【景别+运镜（+机位/落点/调度）】，让摄影师能直接照拍，不要只写一个词（景别：大特写/特写/局部特写/近景/中景/全景/小全景；运镜：推/拉/摇/移/跟/甩/升降/后拉；机位：俯拍/仰拍/过肩/主观视角）。
   ❌ 太简单：（特写）
   ❌ 单独成行：△陈天推开大门。 → （镜头：手开门特写）
   ✅ 就近内联：△陈天身披道袍（道袍局部特写），头戴偃月冠（头冠特写），在昏暗过道行走（后背跟拍）。
   ✅ 复杂调度：△陈天推开大门（手开门特写），出现供奉法坛的房间（法坛小全景后拉，男主背身走到法坛前，房间大全景，手部特写点香）。
   镜头括号里只写可执行的拍摄语言，不写情绪评价。
8. 只输出本批次小说内容对应的剧本。"""


# A compact few-shot built from real 短剧 scripts (the duanju format the user
# supplied as reference). Shows the model: coarse scenes, atomic △ lines, sparse
# camera cues, identity 字幕 on first appearance, OS narration, hook-driven beats.
SCRIPT_FEWSHOT = """# 格式参考样例（只学格式、颗粒度与节奏，不要照抄内容）

# ❌ 错误示范（△密度过高、OS误用）
△林澈抬手。
△林澈敲门。
△门开了。
△年轻女人探头。
林澈（OS）：她在撒谎。  ← OS 误用：这是内心判断，不是场外声音

# ✅ 正确写法（△按节拍合并、用行为外化判断）
△林澈敲门，年轻女人从防盗链后探头。
△林澈盯着她，察觉异常。  ← 用行为暗示判断，不用 OS

## 样例A（古装/穿越）
第 1 集
1-1  山洞  夜  内
人物：沐橙、霍麟
△荒芜之地，天空黑沉，雷声滚滚，一口贴满符咒、风化的黑色棺材诡异地静静伫立。
△霍麟自斜坡滚落到棺材旁，昏迷过去。
△他身下的鲜血顺着泥土向棺材汇聚，棺材发出诡异红光。
△闪电划过，一道惊雷劈开棺材，一只骨节分明又惨白的手从里面抬起。
△沐橙穿着火红嫁衣缓缓从棺材站起，巴掌大的小脸没有血色，唇色却如血般妖异。
沐橙：（呆萌）一千年了，吾身为血族，遭人陷害封印，今日吾终于重见天日！是谁封印了吾？吾定要找回记忆报仇！
△一只手抓住她裙摆。
霍麟：（满脸血污）救……我……
（字幕：影帝 霍麟）
沐橙：（嫌弃）好臭的血。（顿住）这人马上要死了，吾师曾言救人一命胜造七级浮屠，吾且问汝，可愿归于吾族？
霍麟：我……愿意！
△沐橙张开只剩半边的獠牙，猛地刺入他颈部（右侧獠牙刺入颈部大特写，鲜血顺牙尖滑落，镜头微微上摇到沐橙妖异的眼）。

1-2  山洞  夜  外
人物：简白、救援人员
△山间大雨滂沱，一群人在挖洞救援。
△一架直升机盘旋半空，梯子降下，简白穿笔挺西装垂直落下，走向众人。
简白：人找到了吗？
（字幕：霍麟经纪人 简白）
救援人员：简先生，霍先生应该掉进前方山洞了，山洞塌陷，我们正在挖。
△话未落，山洞传来刺破天际的尖叫，简白立即赶去。

## 样例B（都市/人设，镜头就近内联写法）
第 1 集
1-1  日  内  陈天家
人物：陈天、白洁、王希
△陈天身披道袍（道袍局部特写），头戴偃月冠（头冠特写），在昏暗的过道上行走（后背跟拍）。（宽大的道袍下摆随步频在身侧带起细微的空气波动）
（字幕：陈天，风水顾问）
陈天（OS）：总有朋友问我，睡没睡过女明星，我只能说，这一行水很深。
△陈天推开大门（手开门特写），出现一个供奉着祖师神像和法坛的房间（法坛小全景后拉，男主背身走到法坛前，房间大全景，男主手部特写点香祭拜）。
陈天（OS）：我叫陈天，是一名风水顾问，专门替以特殊手段爆火的明星处理善后事宜（镜头从上往下俯拍到正面，亮相）。
"""


CONTINUE_HINT = """## 续写约束（重要）
本批不是第一批小说。请从"第 {episode} 集"继续，集号与场号必须接续，不要从第1集重新开始。
上一批剧本结尾片段（仅供衔接，不要重复输出）：
{prev_tail}
"""


SCRIPT_REVIEW_SYSTEM = """你是头部短剧制片人 + 剧本医生，负责审一段已改编好的短剧剧本能不能直接进组开拍。
只返回恰好一个合法 JSON 对象，不要输出其它内容。

按短剧（竖屏快节奏）标准评估，而不是按小说文笔评估。重点查这些常见病：
- 场景切得太碎（同一空间被拆成多场）、场号该合并；
- △密度过高（一场超过 20 条△）：连续碎动作应该合并写，每条△只承载一个清晰可拍的画面节拍；
- △行写成了小说描写（一条塞多个信息点、含心理/观感形容词、不可一镜拍摄）；
- 镜头说明滥用（几乎每条△都挂镜头）或写成情绪评价；
- 镜头说明没有就近内联进△动作行，而是单独另起一行写"（镜头：…）"；
- 镜头说明太简单（只写"特写""跟拍"一个词），没有"景别+运镜+机位/落点"的专业表述，摄影师无法直接照拍；
- 主要角色首次出场缺少（字幕：身份 姓名）身份卡；
- 开场没钩子（用环境铺垫开场）、集尾没有反转/悬念；
- 台词太书面、太长，没有你来我往的冲突；
- 动作行明显压倒对白行（动作/对白 > 3:1），像现场记录而不是短剧场景；
- OS 格式错误：写成"（OS，女声）：..."或用于内心独白，而不是"角色名（OS）：..."且仅用于真实的场外声音；
- 没有落实 episode_plan 的开场钩子、对白交锋、场尾转折或集尾追看卡点。

schema:
{
  "shootability_score": 1-10,
  "rhythm_score": 1-10,
  "hook_score": 1-10,
  "fidelity_score": 1-10,
  "overall": 1-10,
  "problems": ["按上面病症列出的具体问题，指明第几集第几场"],
  "revision_directives": ["可执行的修改指令，逐条，针对具体场景"]
}

评分纪律：场景过碎或△像小说，shootability 不得超过 6；△密度过高（一场超过 20 条），rhythm 不得超过 6；开场无钩子或集尾平，hook 不得超过 6。"""


SCRIPT_REVISE_SYSTEM = """你是头部竖屏短剧资深改编编剧，负责按修改指令重写一段短剧剧本。
只输出修订后的完整剧本正文，不要解释、不要点评、不要代码围栏。

必须严格保持原有的输出格式与铁律（场景要粗：一集1-3场；△要粗不要细：一场10-15条△，连续碎动作合并写；动作行数量≤对白行数量3倍；OS必须有说话主体且只用于真实场外声音，不用于内心独白；镜头说明就近内联写进△动作行的括号里、不单独成行，且每个必须写清"景别+运镜+机位/落点"专业表述而非单个词，如"道袍局部特写""法坛小全景后拉，男主背身走到法坛前，房间大全景"；主要角色首次出场打身份字幕；每集开头强钩子、结尾反转留人；台词短促），并落实给定的全部修改指令。
不得改动主线事件、人物关系与结局；集号、场号必须与原稿保持一致接续。"""


def _dialogue_lines(lines: list[str]) -> list[str]:
    """Lines that look like character dialogue, excluding metadata rows."""
    out: list[str] = []
    for ln in lines:
        if ln.startswith(("人物：", "（字幕", "（镜头", "△")):
            continue
        if re.match(r"^[^：\n]{1,18}(?:（[^）]+）)?：", ln):
            out.append(ln)
    return out


def _first_scene_character(script_text: str) -> str:
    match = re.search(r"(?m)^人物：([^\n]+)", script_text)
    if not match:
        return ""
    names = [n.strip() for n in re.split(r"[、,，]", match.group(1)) if n.strip()]
    return names[0] if names else ""


def _subtitle_names(script_text: str) -> set[str]:
    names: set[str] = set()
    for raw in re.findall(r"（字幕：([^）]+)）", script_text):
        for token in re.split(r"[\s,，、：:]+", raw.strip()):
            token = token.strip()
            if token:
                names.add(token)
    return names


def script_health(script_text: str) -> dict[str, Any]:
    """Deterministic, non-LLM 短剧 prose metrics.

    Returns flags for the most common degeneration modes so a re-run / review can
    be triggered without trusting the model's self-rating. Cheap signals only.
    """
    lines = [ln.strip() for ln in script_text.splitlines() if ln.strip()]
    action_lines = [ln for ln in lines if ln.startswith("△")]
    dialogue_lines = _dialogue_lines(lines)
    scene_heads = re.findall(r"(?m)^\s*\d+-\d+\b", script_text)
    episodes = _count_episodes(script_text) or 1
    # Camera cues now live INLINE inside △ action lines, as parenthetical shot
    # language — either the explicit （镜头：…） form or a bare （…特写/跟拍…）.
    _shot_size = "大特写|特写|局部特写|近景|中景|全景|小全景|大全景|远景"
    _movement = "推|拉|摇|移|跟拍|跟|甩|升|降|环绕|手持|晃|后拉|俯拍|仰拍|过肩|主观|平视|定格|亮相"
    _cue_token = re.compile(rf"({_shot_size}|{_movement})")
    all_parens = re.findall(r"（([^）]*)）", script_text)
    camera_cue_texts = [
        p for p in all_parens
        if p.startswith("镜头") or _cue_token.search(p)
    ]
    # exclude 字幕/OS-style parentheticals that happen to match
    camera_cue_texts = [
        p for p in camera_cue_texts
        if not p.startswith(("字幕", "OS", "OS，"))
    ]
    camera_cues = len(camera_cue_texts)
    # A "professional" cue names a 景别 (shot size) AND a 运镜/机位 (movement / angle),
    # OR is a rich multi-step 调度 description (long enough to direct the shot).
    def _is_thin(c: str) -> bool:
        c = c.lstrip("镜头：:").strip()
        has_size = bool(re.search(_shot_size, c))
        has_move = bool(re.search(_movement, c))
        if has_size and has_move:
            return False
        if len(c) >= 12 and (has_size or has_move):
            return False  # rich single-aspect description (e.g. detailed 调度)
        return True
    thin_cues = [c for c in camera_cue_texts if _is_thin(c)]
    subtitles = len(re.findall(r"（字幕", script_text))
    long_actions = [ln for ln in action_lines if len(ln) > 45]
    invalid_os = [ln for ln in lines if re.match(r"^（OS[，)：:]", ln)]
    first_character = _first_scene_character(script_text)
    subtitle_names = _subtitle_names(script_text)
    flags: list[str] = []
    scenes_per_ep = len(scene_heads) / episodes if episodes else 0
    action_dialogue_ratio = len(action_lines) / max(len(dialogue_lines), 1)
    actions_per_ep = len(action_lines) / episodes if episodes else 0
    if scenes_per_ep > 3.5:
        flags.append(f"场景过碎: 平均每集 {scenes_per_ep:.1f} 场（建议≤3）")
    if actions_per_ep > 20:
        flags.append(f"△密度过高: 平均每集 {actions_per_ep:.1f} 条（建议≤20）")
    if action_lines and len(long_actions) / len(action_lines) > 0.30:
        flags.append(f"△行偏小说化: {len(long_actions)}/{len(action_lines)} 条超长（建议一条一动作）")
    if action_lines and action_dialogue_ratio > 3:
        flags.append(f"对白不足: 动作/对白={action_dialogue_ratio:.1f}:1（建议≤3:1）")
    if action_lines and camera_cues > len(action_lines) * 1.5:
        flags.append(f"镜头说明滥用: {camera_cues} 个镜头括号（动作行 {len(action_lines)} 条）")
    if camera_cue_texts and len(thin_cues) / len(camera_cue_texts) > 0.5:
        flags.append(f"镜头说明太简单: {len(thin_cues)}/{len(camera_cue_texts)} 条缺少景别+运镜（应写如\"仰拍整栋楼沿水痕下摇落到人脸\"）")
    if subtitles == 0:
        flags.append("缺少身份字幕（字幕：身份 姓名）")
    elif first_character and first_character not in subtitle_names:
        flags.append(f"首位主要角色缺少身份字幕: {first_character}")
    if invalid_os:
        flags.append(f"OS格式错误: {len(invalid_os)} 条（应为 角色名（OS）：...）")
    return {
        "episodes": episodes,
        "scenes": len(scene_heads),
        "scenes_per_episode": round(scenes_per_ep, 2),
        "action_lines": len(action_lines),
        "actions_per_episode": round(actions_per_ep, 2),
        "dialogue_lines": len(dialogue_lines),
        "action_dialogue_ratio": round(action_dialogue_ratio, 2),
        "long_action_lines": len(long_actions),
        "camera_cues": camera_cues,
        "thin_camera_cues": len(thin_cues),
        "subtitles": subtitles,
        "invalid_os_lines": len(invalid_os),
        "first_character": first_character,
        "flags": flags,
    }


def _review_revise_segment(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    novel_segment: str,
    script_part: str,
    *,
    source_packet: dict[str, Any] | None = None,
    episode_plan: dict[str, Any] | None = None,
    rounds: int,
    min_score: float,
    max_tokens: int,
    temperature: float,
    seg_label: str,
) -> str:
    """Run up to `rounds` of producer-review → revise on one screenplay segment.

    Stops early once `overall` >= min_score. Mirrors the engine's chapter
    review/revise loop but for screenplay quality (shootability/rhythm/hook). On
    any LLM/JSON failure it returns the best script so far rather than raising.
    """
    best = script_part
    for r in range(max(rounds, 0)):
        review_user = f"""## 原始小说片段（用于核对忠实度）
{novel_segment[:4000]}

## source_packet（用于核对事实和不可改动边界）
{json.dumps(source_packet or {}, ensure_ascii=False, indent=2)}

## episode_plan（用于核对开场钩子、对白交锋、场尾转折、集尾追看卡点）
{json.dumps(episode_plan or {}, ensure_ascii=False, indent=2)}

## 待审剧本
{best}
"""
        try:
            review_raw = call_llm(
                client, paths, config,
                SCRIPT_REVIEW_SYSTEM, json_prompt(review_user),
                max_tokens=4000, temperature=0.2,
            )
            review = load_json_with_repair(client, paths, config, review_raw, fallback={"overall": 0})
        except Exception as exc:  # noqa: BLE001
            log(paths, f"Screenplay segment {seg_label} review failed round {r + 1}: {exc}")
            break
        overall = safe_score(review.get("overall", 0))
        directives = [str(d).strip() for d in (review.get("revision_directives") or []) if str(d).strip()]
        log(paths, f"Screenplay segment {seg_label} review round {r + 1} overall={overall} directives={len(directives)}")
        if overall >= min_score or not directives:
            break
        revise_user = f"""## 原始小说片段（忠实度基准，不要超出其事件范围）
{novel_segment[:4000]}

## source_packet（事实边界）
{json.dumps(source_packet or {}, ensure_ascii=False, indent=2)}

## episode_plan（必须落实）
{json.dumps(episode_plan or {}, ensure_ascii=False, indent=2)}

## 待修订剧本
{best}

## 必须落实的修改指令
{chr(10).join('- ' + d for d in directives)}

请输出修订后的完整剧本正文。"""
        try:
            revised_raw = call_llm(
                client, paths, config,
                SCRIPT_REVISE_SYSTEM, revise_user,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            log(paths, f"Screenplay segment {seg_label} revise failed round {r + 1}: {exc}")
            break
        revised = normalize_text(revised_raw).strip()
        # Guard against a revise that nukes content (provider hiccup / refusal).
        if len(revised) >= 0.6 * len(best) and _count_episodes(revised) >= _count_episodes(best):
            best = revised
        else:
            log(paths, f"Screenplay segment {seg_label} revise round {r + 1} rejected (too short / lost episodes)")
            break
    return best


def _build_client(config: dict[str, Any], paths: Paths) -> Any:
    """Construct an LLM client/pool from config (copied shape from trial.py)."""
    from openai import OpenAI
    import httpx

    from config import configured_api_endpoints
    from llm import LLMClientPool

    api_endpoints, primary_endpoint_count = configured_api_endpoints(config)
    if not api_endpoints:
        raise RuntimeError("Missing API key: set api.api_key/api_keys/api_key_groups in config")
    connect_timeout = int(config["api"].get("client_connect_timeout", 15))
    client_read_timeout = int(config["api"].get("client_read_timeout", 180))
    httpx_timeout = httpx.Timeout(
        connect=connect_timeout,
        read=client_read_timeout,
        write=connect_timeout,
        pool=connect_timeout,
    )
    default_headers = {}
    user_agent = str(config["api"].get("user_agent", "")).strip()
    if user_agent:
        default_headers["User-Agent"] = user_agent
    clients = [
        OpenAI(base_url=base_url, api_key=api_key, timeout=httpx_timeout, default_headers=default_headers or None)
        for base_url, api_key in api_endpoints
    ]
    if len(clients) == 1:
        return clients[0]
    return LLMClientPool(clients, primary_endpoint_count, endpoints=api_endpoints, log_fn=lambda msg: log(paths, msg))


def _fallback_source_packet(segment: str) -> dict[str, Any]:
    preview = re.sub(r"\s+", " ", segment.strip())[:500]
    return {
        "characters": [],
        "locations": [],
        "events": [preview] if preview else [],
        "conflicts": [],
        "reversals": [],
        "visual_props": [],
        "must_keep_dialogue": [],
        "locked_facts": ["不得超出原文片段事件范围"],
        "open_threads": [],
    }


def _fallback_episode_plan(source_packet: dict[str, Any], start_episode: int) -> dict[str, Any]:
    events = [str(e).strip() for e in source_packet.get("events", []) if str(e).strip()]
    locations = [str(x).strip() for x in source_packet.get("locations", []) if str(x).strip()]
    characters = [
        str(c.get("name", "")).strip()
        for c in source_packet.get("characters", [])
        if isinstance(c, dict) and str(c.get("name", "")).strip()
    ]
    location = locations[0] if locations else "待定地点"
    scene_no = f"{start_episode}-1"
    return {
        "episodes": [
            {
                "episode": start_episode,
                "opening_hook": events[0] if events else "用原文中最强的异常事件开场",
                "beats": events[:4] or ["按原文因果推进"],
                "retention_hook": (source_packet.get("open_threads") or source_packet.get("reversals") or ["原文片段结尾悬念"])[0],
                "scenes": [
                    {
                        "scene_no": scene_no,
                        "location": location,
                        "time": "日",
                        "interior": "内",
                        "characters": characters[:6],
                        "character_goals": {},
                        "obstacle": "按原文冲突呈现",
                        "information_change": "按原文信息变化呈现",
                        "turn": "以原文结尾转折收场",
                        "dialogue_mandate": "至少安排两名角色围绕本场冲突发生短促对白；若无人可对话，用有主体的OS或手机文字承载信息。",
                        "visual_beats": events[:6],
                    }
                ],
            }
        ],
        "global_constraints": source_packet.get("locked_facts", []),
    }


def _extract_source_packet(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    segment: str,
    *,
    max_tokens: int,
    seg_label: str,
) -> dict[str, Any]:
    user = f"""## 待抽取小说片段（第 {seg_label} 批）
{segment}
"""
    fallback = _fallback_source_packet(segment)
    try:
        raw = call_llm(
            client, paths, config,
            SCRIPT_EXTRACT_SYSTEM, json_prompt(user),
            max_tokens=min(max_tokens, 8000), temperature=0.1,
        )
        packet = load_json_with_repair(client, paths, config, raw, fallback=fallback)
    except Exception as exc:  # noqa: BLE001
        log(paths, f"Screenplay segment {seg_label} source extraction failed: {exc}")
        packet = fallback
    if not isinstance(packet, dict):
        return fallback
    return packet


def _build_episode_plan(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    segment: str,
    source_packet: dict[str, Any],
    *,
    start_episode: int,
    prev_tail: str,
    max_tokens: int,
    seg_label: str,
) -> dict[str, Any]:
    user = f"""## start_episode
{start_episode}

## 上一批剧本结尾片段（仅供衔接，不要重复输出）
{prev_tail[-600:] if prev_tail else "无"}

## source_packet
{json.dumps(source_packet, ensure_ascii=False, indent=2)}

## 原始小说片段（用于核对忠实度）
{segment}
"""
    fallback = _fallback_episode_plan(source_packet, start_episode)
    try:
        raw = call_llm(
            client, paths, config,
            SCRIPT_PLAN_SYSTEM, json_prompt(user),
            max_tokens=min(max_tokens, 8000), temperature=0.2,
        )
        plan = load_json_with_repair(client, paths, config, raw, fallback=fallback)
    except Exception as exc:  # noqa: BLE001
        log(paths, f"Screenplay segment {seg_label} episode planning failed: {exc}")
        plan = fallback
    if not isinstance(plan, dict) or not plan.get("episodes"):
        return fallback
    return plan


def _planned_episode_cursor(plan: dict[str, Any], start_episode: int) -> int:
    nums: list[int] = []
    for ep in plan.get("episodes", []):
        if isinstance(ep, dict):
            try:
                nums.append(int(ep.get("episode")))
            except (TypeError, ValueError):
                continue
    return max(nums) if nums else start_episode


# ---------------------------------------------------------------------------
# input segmentation
# ---------------------------------------------------------------------------
_CHAPTER_RE = re.compile(r"^\s*第\s*[0-9零一二三四五六七八九十百千万]+\s*章.*$", re.MULTILINE)


def split_into_segments(text: str, max_chars: int) -> list[str]:
    """Split novel text into LLM-sized segments.

    Prefer chapter boundaries (第N章 …). When a chapter is itself larger than
    max_chars, fall back to paragraph-greedy packing within that chapter. When the
    text has no chapter markers at all, pack whole paragraphs up to max_chars.
    """
    text = text.strip()
    if not text:
        return []

    matches = list(_CHAPTER_RE.finditer(text))
    if matches:
        chunks: list[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        # If there's prose before the first chapter heading, keep it as a segment.
        head = text[: matches[0].start()].strip()
        if head:
            chunks.insert(0, head)
    else:
        chunks = [text]

    # Further split any oversized chunk by paragraphs.
    segments: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            segments.append(chunk)
            continue
        segments.extend(_pack_paragraphs(chunk, max_chars))
    return segments


def _pack_paragraphs(text: str, max_chars: int) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    out: list[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > max_chars:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    # A single paragraph longer than max_chars: hard-slice it.
    final: list[str] = []
    for seg in out:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            for i in range(0, len(seg), max_chars):
                final.append(seg[i : i + max_chars])
    return final


def _count_episodes(script_text: str) -> int:
    """Highest 第N集 number present in the produced script text."""
    nums = [int(n) for n in re.findall(r"第\s*([0-9]+)\s*集", script_text)]
    return max(nums) if nums else 0


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def convert_text(
    novel_text: str,
    *,
    config: dict[str, Any] | None = None,
    paths: Paths | None = None,
    client: Any | None = None,
    out_path: Path,
    seg_chars: int | None = None,
    temperature: float | None = None,
) -> Path:
    """Convert `novel_text` to a screenplay, writing to `out_path`.

    Resumable: per-segment checkpoints live in `<out_path>.checkpoints/`.
    """
    if config is None:
        config = load_config()
    if paths is None:
        paths = get_paths(config)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = _build_client(config, paths)

    seg_chars = int(seg_chars or config["novel"].get("script_seg_chars", 6000))
    seg_chars = max(1500, seg_chars)
    if temperature is None:
        temperature = float(config["novel"].get("script_temperature", config["api"].get("temperature", 0.6)))
    max_tokens = int(config["novel"].get("script_max_tokens", config["api"].get("max_tokens", 16000)))
    use_fewshot = bool(config["novel"].get("script_fewshot", True))
    review_enabled = bool(config["novel"].get("script_review_enabled", True))
    review_rounds = int(config["novel"].get("script_review_max_rounds", 1))
    review_min_score = float(config["novel"].get("script_review_min_score", 8.0))
    structured_pipeline = bool(config["novel"].get("script_structured_pipeline", True))

    segments = split_into_segments(novel_text, seg_chars)
    if not segments:
        raise ValueError("Input text is empty after stripping.")

    ckpt_dir = Path(str(out_path) + ".checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log(paths, f"Screenplay conversion start segments={len(segments)} seg_chars={seg_chars} out={out_path}")

    episode_cursor = 0  # last episode number already used
    prev_tail = ""
    produced: list[str] = []

    for idx, segment in enumerate(segments, start=1):
        ckpt = ckpt_dir / f"seg_{idx:04d}.json"
        if ckpt.exists():
            try:
                data = json.loads(ckpt.read_text(encoding="utf-8"))
                script_part = str(data.get("script", "")).strip()
                if script_part:
                    produced.append(script_part)
                    episode_cursor = max(episode_cursor, int(data.get("episode_cursor", episode_cursor)))
                    prev_tail = script_part[-600:]
                    log(paths, f"Screenplay segment {idx}/{len(segments)} resumed from checkpoint")
                    continue
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        start_episode = episode_cursor + 1
        source_packet: dict[str, Any] | None = None
        episode_plan: dict[str, Any] | None = None
        if structured_pipeline:
            source_packet = _extract_source_packet(
                client, paths, config, segment,
                max_tokens=max_tokens, seg_label=f"{idx}/{len(segments)}",
            )
            episode_plan = _build_episode_plan(
                client, paths, config, segment, source_packet,
                start_episode=start_episode, prev_tail=prev_tail,
                max_tokens=max_tokens, seg_label=f"{idx}/{len(segments)}",
            )
            system = SCRIPT_FROM_PLAN_SYSTEM
            user_parts = [
                f"## 待改编小说正文（第 {idx}/{len(segments)} 批）",
                segment,
                "## source_packet",
                json.dumps(source_packet, ensure_ascii=False, indent=2),
                "## episode_plan",
                json.dumps(episode_plan, ensure_ascii=False, indent=2),
            ]
            if use_fewshot:
                user_parts.insert(0, SCRIPT_FEWSHOT)
            if idx > 1 and prev_tail:
                user_parts.insert(0, CONTINUE_HINT.format(episode=start_episode, prev_tail=prev_tail))
            user = "\n\n".join(user_parts)
        else:
            system = SCRIPT_SYSTEM.format(episode=start_episode)
            user_parts = [f"## 待改编小说正文（第 {idx}/{len(segments)} 批）", segment]
            if use_fewshot:
                user_parts.insert(0, SCRIPT_FEWSHOT)
            if idx > 1 and prev_tail:
                user_parts.insert(0, CONTINUE_HINT.format(episode=start_episode, prev_tail=prev_tail))
            user = "\n\n".join(user_parts)

        raw = call_llm(
            client,
            paths,
            config,
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        script_part = normalize_text(raw).strip()
        if len(script_part) < 30:
            log(paths, f"Screenplay segment {idx}/{len(segments)} produced too-short output; keeping raw")
            script_part = raw.strip()

        if review_enabled and len(script_part) >= 200:
            script_part = _review_revise_segment(
                client, paths, config, segment, script_part,
                source_packet=source_packet, episode_plan=episode_plan,
                rounds=review_rounds, min_score=review_min_score,
                max_tokens=max_tokens, temperature=temperature, seg_label=f"{idx}/{len(segments)}",
            )

        health = script_health(script_part)
        if health["flags"]:
            log(paths, f"Screenplay segment {idx}/{len(segments)} health flags: {'; '.join(health['flags'])}")

        # Trust the highest absolute 第N集 number the model emitted (the system
        # prompt told it to start at start_episode). If it emitted none, advance by
        # one so the next segment still gets a fresh episode number.
        abs_max = _count_episodes(script_part)
        planned_max = _planned_episode_cursor(episode_plan or {}, start_episode)
        episode_cursor = max(abs_max, planned_max, start_episode)

        produced.append(script_part)
        prev_tail = script_part[-600:]
        ckpt.write_text(
            json.dumps(
                {
                    "segment_index": idx,
                    "episode_cursor": episode_cursor,
                    "source_packet": source_packet,
                    "episode_plan": episode_plan,
                    "script": script_part,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log(paths, f"Screenplay segment {idx}/{len(segments)} done episodes_through={episode_cursor}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(produced).strip() + "\n", encoding="utf-8")
    log(paths, f"Screenplay conversion complete out={out_path}")
    return out_path


def convert_file(
    input_path: Path,
    out_path: Path | None = None,
    *,
    config_path: Path | None = None,
    seg_chars: int | None = None,
    temperature: float | None = None,
) -> Path:
    """Read `input_path`, convert, write to `out_path` (default: <input>_script.md)."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    novel_text = input_path.read_text(encoding="utf-8", errors="replace")

    # Standalone config resolution: explicit --config, else NOVEL_CONFIG env (set by
    # `novel.py script`), else config_template.yaml (carries shared keys).
    if config_path is not None:
        os.environ["NOVEL_CONFIG"] = str(Path(config_path))
    elif not os.environ.get("NOVEL_CONFIG"):
        template = ROOT / "config_template.yaml"
        if template.exists():
            os.environ["NOVEL_CONFIG"] = "config_template.yaml"

    # config.py read NOVEL_CONFIG at import; reload CONFIG_FILE to honour a late env.
    import config as _config

    _config.CONFIG_FILE = ROOT / os.environ.get("NOVEL_CONFIG", "config.yaml")
    config = _config.load_config()
    paths = _config.get_paths(config)

    if out_path is None:
        # Default: a scripts/ subdirectory next to the input file.
        out_path = input_path.parent / "scripts" / (input_path.stem + "_script.md")
    return convert_text(
        novel_text,
        config=config,
        paths=paths,
        out_path=Path(out_path),
        seg_chars=seg_chars,
        temperature=temperature,
    )
