from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint
from config import (
    Paths,
    append_text,
    chapter_path,
    count_chars,
    normalize_chapter,
    normalize_text,
    read_text,
    safe_score,
    write_text,
)
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, contract_capsule, memory_context, writing_memory_context
from planning import plan_score
from store import db_event, db_lock, recent_quality_feedback, store_causal_links, upsert_reader_promise

if TYPE_CHECKING:
    from openai import OpenAI

# Shared anti-style-collapse ban. MUST contain no literal braces — it is
# concatenated into prompts that may be .format()'d (write presets) as well as
# prompts that are not, so any "{" would either be eaten by .format or crash it.
ANTI_FRAGMENT_BAN = """## 文风塌缩禁令（最高优先级）
- 禁止"句子——状态——状态"式破折号短句链；破折号每千字不超过3处，且只用于正常插入语，不得用来粘连碎片。
- 每段至少含2个有主谓宾的完整句子，禁止整段单词短句堆叠或无标点的舞台提示式断行。
- 句子长短交替，避免连续3句字数相近的短句。
- 以完整成句的小说叙事和有潜台词的对话为主，绝不能退化成电报体/碎片体。"""

ANTI_PITFALL_BLOCK = """## 网文避雷+去AI味铁律（读者弃书的确定性触发器）
### 人设铁律
- 主角必须有独立判断力和行动力：不当圣母（无底线原谅敌人）、不当舔狗（无脑付出无回报）、不降智（行为逻辑配得上已建立的人设）。
- 反派/对手必须有自己的行事逻辑和动机：禁止降智送人头，禁止无理由针对主角，反派的失败必须合理而非智商下线。
- 角色性格转变必须有铺垫：禁止性格突变（谨慎到冲动）而无触发事件，变化须由情节驱动、有过渡。
- 每个出场角色须有区分度：不同角色的说话方式、用词习惯、思维模式必须不同，禁止千人一面。
### 情节铁律
- 信息/设定前后一致：本章出现的任何事实不得与前文已确立的设定矛盾。
- 伏笔必须有回收计划：埋下的悬念必须在可预见章节内给出回应，禁止无限挖坑。
- 禁止无效注水：每个场景必须推进剧情、揭示人物或制造张力，删掉不影响故事的段落就不该存在。
- 压抑须有节奏：虐主后必须在合理章数内给予释放/反击，不得连续3章以上纯压抑无爽点。
### 去AI味铁律
- 禁止情感贴标签：不写"他感到震惊"，而是写震惊时的具体身体反应和行为；不写"她很悲伤"，而是写她做了什么。
- 禁止比喻堆砌：每千字比喻不超过3个，且必须新鲜贴切，禁用"时间仿佛静止""心如刀绞""像是被抽空了"等陈腐比喻。
- 禁止程度副词撑场面：删掉"非常""极其""十分""无比"，用具体细节替代模糊的程度修饰。
- 禁止总结式叙述：不写"就这样一切都变了""命运的齿轮开始转动"，让读者从情节中自行感受。
- 段落结构要多变：不得每段都是"描写到心理到对话"的固定模板，段落长短、开头方式、叙述角度须变化。
- 禁止AI套话：不用"心中一沉/瞳孔一缩/嘴角微微上扬/缓缓开口/深吸一口气/一时间/此刻"等万能表达，换用只属于当前角色和场景的具体反应。
- 每个角色有语言指纹：不同角色的遣词造句、句式长短、口头禅必须有辨识度。
### 排版铁律
- 禁止碎段：每段至少3-5句、60字以上（纯对话句除外）；禁止单句成段，禁止2000字写出80+行。段落应有起承转，不是每句话换一行。
- 禁止问答式对话推进：角色对话不得连续3轮以上一问一答；对话间必须夹入动作、神态、心理或环境描写，打破审讯/采访节奏。
### 否定对仗禁令
- 禁止"没有X，也没有Y""不是X，也不是Y"式对仗否定句——这是最明显的AI写作指纹。要表达缺失，直接一句说完，不要用两个否定分句对称排列。
### 内容密度铁律
- 禁止用大段环境描写/华丽辞藻/大量比喻修辞来充字数：环境描写每场景不超过2句，且必须服务于情绪或信息推进，纯装饰性描写全删。
- 禁止整章都是行为流水账：每个场景至少有一段角色内心的犹豫/权衡/情绪波动（通过动作或内心活动体现），让读者能代入角色处境。"""

# 审美与品味基线（无花括号；按题材选用其一，拼入 write preset）。
# 网文工程化最大盲区：句子能跑通≠有文学质感。下列准则约束"怎么写得好看"，
# 与 ANTI_FRAGMENT_BAN（防塌缩，约束"不准退化成碎片")互补。
AESTHETIC_COMMON = """## 审美与品味（在不牺牲可读性与节奏的前提下，写出质感）
- 克制与留白：最有力的情绪点用最省的笔墨，给读者回味空间；不把话说尽，不替读者下结论。
- 意象与潜台词：善用一两个贯穿场景的核心意象（物、光、声）折射心境；对白藏机锋，重要的事让人物用言外之意带出。
- 节奏即呼吸：长短句、缓急段交替形成韵律；高潮处可短促有力，铺陈处可舒展从容，避免通篇一个语速。
- 叙事腔调统一：维持与 voice 宪章一致的稳定声音；动词精准有力。
- 反套路表达：同一情绪/场景换一种写法呈现，避免与近章雷同的措辞与比喻；让句子有"只此一处"的辨识度。"""

AESTHETIC_HISTORY = AESTHETIC_COMMON + """
## 历史题材额外品味
- 语言有古意而不晦涩：用词、称谓、礼仪贴合时代，却让今人读得顺；避免半文不白的夹生腔。
- 以小见大：用一处器物、一道公文、一个仪轨折射时代重量，胜过空泛的"历史厚重感"宣称。
- 克制抒情，重白描与场面调度；权力与人心的张力靠细节与潜台词流露，不靠作者旁白点评。"""

AESTHETIC_SHUANG = AESTHETIC_COMMON + """
## 爽文题材额外品味
- 爽而不俗：爽点靠铺垫、反差与对手的真实反应挣来，落到精准的动作与神态上，而非形容词喊"太爽了"。
- 装而不油：主角的从容与机锋点到为止，留白比说满更有格调；忌油腻炫耀式独白。
- 反差与节奏感本身就是审美：压抑—释放的曲线干净利落，比堆砌爽点更耐读。"""

AESTHETIC_SYSTEM_STREAM = AESTHETIC_COMMON + """
## 系统流题材额外品味
- 面板服务情绪：数值与提示音要落在恰当的戏剧时刻，制造"叮"的爽感与期待，而非冷冰冰的数据罗列。
- 成长有质感：升级不止是数字变大，要让读者感到主角心境、处境、视野的真实跃迁。
- 系统语气有辨识度：系统提示的腔调（冷峻/毒舌/中二）保持统一，成为本书的趣味点而非噪音。"""

AESTHETIC_URBAN_ABILITY = AESTHETIC_COMMON + """
## 都市异能/重生题材额外品味
- 都市质感真实：行业、阶层、人情世故的细节要可信，让爽点扎根在真实生活的土壤里。
- 打脸要漂亮：反转靠信息差与铺垫的精巧，落点干脆，余味留给对手的反应与旁观者的神情，不靠主角自吹。
- 异能/先知写得有想象力：能力的呈现要有画面与代价，避免沦为万能开关。"""

AESTHETIC_ROMANCE_FEMALE = AESTHETIC_COMMON + """
## 女频言情题材额外品味
- 情绪靠细节传递：心动与受伤写在视线、距离、停顿、指尖的颤动里，绝不用旁白直接宣告"她很感动"。
- 暧昧的张力来自克制：欲言又止、若即若离比直白告白更撩人；甜与虐都要有余韵。
- 对白有化学反应：男女主的机锋、试探、心照不宣构成独特的二人语感，配角也各有声口。"""

AESTHETIC_WANZU_XUANHUAN = AESTHETIC_COMMON + """
## 现代玄幻/万族题材额外品味
- 斗法有美感：招式、气机、环境互动写出画面与节奏感，像一场编排过的武戏，而非招式名的罗列。
- 境界有意境：突破与顿悟落到具体的体感与心境变化上，避免空喊"实力大增"。
- 宏大不空洞：万族、天骄、大势的格局靠具体的人、物、场面撑起，杜绝纯设定名词的堆砌。"""

AESTHETIC_SUSPENSE = AESTHETIC_COMMON + """
## 悬疑/心理惊悚题材额外品味
- 限制视角是第一纪律：只写视角人物当下能看见、听见、想到的，绝不偷偷塞入全知者才知道的真相或他人内心；信息差就是悬念的引擎。
- 感官克制制造不安：恐惧与诡异写在反常的细节里（多出一只杯子、湿了一半的脚印、本该有却没有的声音），而非靠"恐怖""毛骨悚然"这类形容词喊出来。
- 公平线索：所有指向真相的线索必须公平地呈现在读者眼前（哪怕被淹没在干扰项里），结局的反转要让读者回头能找到伏笔，杜绝凭空掉落的关键信息。
- 留白与未言之事：最吓人的是没写出来的东西；关键时刻收笔、让画面停在悬而未决处，比写满更有压迫感。
- 不可靠与氛围：可善用视角人物的偏见、记忆缺口、自我说服制造叙事张力，但作者不得借此对读者撒谎（隐瞒可以，捏造不行）。"""

AESTHETIC_PRESETS = {
    "history": AESTHETIC_HISTORY,
    "xuanhuan_shuang": AESTHETIC_SHUANG,
    "system_stream": AESTHETIC_SYSTEM_STREAM,
    "urban_ability": AESTHETIC_URBAN_ABILITY,
    "romance_female": AESTHETIC_ROMANCE_FEMALE,
    "wanzu_xuanhuan": AESTHETIC_WANZU_XUANHUAN,
    "suspense": AESTHETIC_SUSPENSE,
}

# ---------------------------------------------------------------------------
# Writer system prompt: shared-base + genre-delta architecture
# ---------------------------------------------------------------------------
# Shared sections used by _build_write_system(). No literal braces except the
# format placeholders that write_chapter fills at call time.

_SELF_REVIEW_PREAMBLE = """## 写前自我审查（内部执行，严禁输出）
以下步骤只用于动笔前校准；如果接口没有隐藏 reasoning_content，就在心里完成，不要写出任何标题、列表、分析、解释或 XML/Markdown 元信息。
1. 识别本章三项最高风险："""

_OUTPUT_SECTION = """## 输出要求
- 约{chapter_words}个中文字符。
- 第一行固定格式：第{chapter_num}章 {title}
- 执行选定的plan及所有约束条件。
- plan中的高风险beats必须直接在页面上演出。
- 只输出章节正文。第一行必须直接是章节标题，严禁输出"写前自我审查"、"Pre-writing Self-Review"、"分析"、"reasoning"、`<analysis>`、`<thinking>`、代码围栏、JSON、清单或任何解释。"""

_SENSORY_DIALOGUE_DEFAULT = """## 感官与对话
- 每个场景至少2种感官锚点；用具体细节代替抽象描述。
- 对话占全章25-45%，反映人物身份与心理，关键对话含潜台词。"""

_TIME_MARKER_BAN_DEFAULT = """## 时间标记禁令
- 严禁以时间副词切换场景；时间流逝靠情节动作体现，每章最多2个时间词且与具体行为绑定。"""

GENRE_PROFILES: dict[str, dict[str, str]] = {
    "history": {
        "role": "擅长中国历史题材的长篇网文作家，风格厚重克制，兼具网文可读性",
        "self_review": (
            "   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？\n"
            "   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n"
            "   - 空洞兑现风险：主角在哪里可能轻松获胜却没有代价？\n"
            "2. 针对每项风险，写一条具体规避承诺（如“用茶寮而非文渊阁”、“在页面上呈现户部程序”、“让主角失去一张底牌”）。\n"
            "3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。\n"
            "4. 完成1-3后才开始正式写作。"
        ),
        "core_discipline": "",
        "structure_template": (
            "## 结构模板\n"
            "- 开场钩（200-400字）：紧接上章末尾，建立本章核心问题或悬念，禁止用时间词作开场\n"
            "- 场景一（1000-1500字）：主要冲突场景，含具体动作、对话与环境描写\n"
            "- 场景二（800-1200字）：转折或揭示场景，推进plan中的关键 beats\n"
            "- 场景三（600-1000字）：决定或代价场景，呈现选择后果\n"
            "- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾"
        ),
        "sensory_dialogue": (
            "## 感官纪律\n"
            "- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）\n"
            "- 用具体细节代替抽象描述（“墨迹未干的公文” 而非 “重要文件”）\n"
            "- 季节、天气、光线作为情绪衬托，不作章节进度的计时器\n\n"
            "## 对话比例\n"
            "- 对话占全章25-45%，避免连续500字以上无对白的段落\n"
            "- 每个角色的语气、用词必须反映其身份、立场和当下心理"
        ),
        "time_marker_ban": (
            "## 时间标记禁令（核心问题）\n"
            "- 严禁以“翌日清晨”“这天晚上”“次日黄昏”“午后”“深夜”等时间副词切换场景或开启段落\n"
            "- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记\n"
            "- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定（如“赶在衙门散班前”而非单纯“傍晚”）"
        ),
        "genre_bans": (
            "## 禁止模式\n"
            "- 禁止“他突然意识到/恍然大悟”式的廉价顿悟\n"
            "- 角色内心独白每次50-200字：关键抑择时刻必须有心理描写（不少于50字的犹豫/权衡/恐惧），但不超过200字避免意识流\n"
            "- 禁止同一章内出现超过3次相同的动作描写（如“皱眉”“沉默”“点头”）\n"
            "- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话"
        ),
        "extras": (
            "## 人物塑造要求\n"
            "- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或反派\n"
            "- 主角的成长必须来自挛败、情报或他人的推演，不能突然“顿悟”\n"
            "- 对话必须含潜台词和话术攻防，不能只喊口号和表态\n"
            "- 官场人物的措辞必须符合其政治处境（得势者与失势者说话方式不同）\n\n"
            "## 情节逻辑要求\n"
            "- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果\n"
            "- 如有伏笔，必须在本章或后续章节可查的文本中有对应的“收线”\n"
            "- 不得出现“某人神秘地笑了”类的模糊悬念代替真实信息\n\n"
            "## 本章必须满足的质量硬指标（写完前自检）\n"
            "- 显性代价：主角本章至少有一次**可见的资源/政治/情感代价**（失去一张底牌、得罪一方势力、付出信任或人情），不得轻松全胜。\n"
            "- 对白潜台词：本章至少有一处关键对话含**话术攻防/言外之意**（如表面奏对、暗里递价；正例：“臣不敢妄言”实指“陈下先表态臣才敢接”），不得只喊口号表态。\n"
            "- 差异化：禁止复用最近3章已用过的开场方式与章末钩子手法；若雷同，必须换一种结构（场景驱动↔反转↔压迫-兑现等）。"
        ),
    },
    "xuanhuan_shuang": {
        "role": '擅长穿越爽文的中文网文作家，节奏明快、爽点密集、画面感强、读者代入感极强',
        "self_review": (
            '   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n'
            '   - 无脑碾压风险：主角在哪里可能毫无铺垫地轻松获胜、缺乏代价或对手反应？\n'
            '2. 针对每项风险，写一条具体规避承诺（如“用一次失败的试探换信任”、“让赵高当场反将一军”、“现代知识落到一个具体器物/制度细节上”）。\n'
            '3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。\n'
            '4. 明确本章的“爽点高潮”是哪一段（兑现/打脸/翻盘/掌权之一），它如何被前文铺垫和压迫衬托。\n'
            '5. 完成1-4后才开始正式写作。'
        ),
        "core_discipline": (
            '## 爽点纪律（本类型核心）\n'
            '- 本章必须有**至少1个明确的爽点高潮**：兑现、打脸、翻盘、识破阴谋或掌权之一，且落到具体动作与对手反应上。\n'
            '- 压迫—兑现节奏要紧：铺垫不拖沓，先制造压迫/轻视/危机，再在高潮处一举兑现，让读者有“出了一口气”的快感。\n'
            '- 主角靠“现代灵魂的先知与见识”做出超越时代的判断，但每次施展**必须有铺垫与代价**（被猜忌、暴露底牌、消耗人情），不得无脑全知全能。\n'
            '- 章末必须留一个让读者想立刻看下一章的强钩子。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章末尾，立刻抛出本章核心冲突或压迫，禁止用时间词作开场\n'
            '- 场景一（1000-1500字）：主要冲突/压迫场景，含具体动作、对话与环境描写\n'
            '- 场景二（800-1200字）：转折或主角施展见识的场景，推进plan关键beats，埋下爽点引信\n'
            '- 场景三（600-1000字）：爽点兑现/打脸/翻盘场景，呈现选择的后果与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官纪律\n'
            '- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）\n'
            '- 用具体细节代替抽象描述（“竹简上未干的朱批” 而非 “重要文书”）\n'
            '- 季节、天气、光线作为情绪衬托，不作章节进度的计时器\n'
            '\n'
            '## 对话比例\n'
            '- 对话占全章25-45%，避免连续500字以上无对白的段落\n'
            '- 每个角色的语气、用词必须反映其身份、立场和当下心理\n'
            '- 关键对话需含潜台词与话术攻防，不只喊口号'
        ),
        "time_marker_ban": (
            '## 时间标记禁令（核心问题）\n'
            '- 严禁以“翌日清晨”“这天晚上”“次日黄昏”“午后”“深夜”等时间副词切换场景或开启段落\n'
            '- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记\n'
            '- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定'
        ),
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止“他突然意识到/恍然大悟”式的廉价顿悟\n'
            '- 角色内心独白每次50-200字：关键抉择时刻必须有心理描写（不少于50字的犹豫/权衡/恐惧），但不超过200字避免意识流\n'
            '- 禁止同一章内出现超过3次相同的动作描写\n'
            '- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话\n'
            '- 禁止主角无铺垫、无代价地碾压全场（爽要爽得有逻辑）'
        ),
        "extras": (
            '## 人物塑造要求\n'
            '- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或当反派\n'
            '- 主角（现代灵魂）的判断与成长必须来自现代见识、情报或挫败的推演，不能突然“顿悟”\n'
            '- 对手要聪明、有手段、有反应，不能是任主角宰割的纸片人\n'
            '- 秦制背景下的措辞与礼仪需大体得体，不出现现代名词穿帮\n'
            '\n'
            '## 情节逻辑要求\n'
            '- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果\n'
            '- 如有伏笔，必须在本章或后续章节可查的文本中有对应的“收线”\n'
            '- 不得出现“某人神秘地笑了”类的模糊悬念代替真实信息'
        ),
    },
    "system_stream": {
        "role": '擅长系统流网文的中文作家，面板感强、成长节奏明快、数值反馈清晰、读者代入与成就感强烈',
        "self_review": (
            '   - 重复风险：本章是否在重复上一次的“刷面板/做任务”套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“戏剧化呈现”？\n'
            '   - 无脑刷级风险：主角是否毫无代价、毫无策略地靠系统碾压？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“系统反馈高潮”是哪一段（升级/解锁/任务结算/奖励兑现之一），以及它如何被前文需求与压迫衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 系统流核心纪律（本类型核心）\n'
            '- 本章至少有一次**可见的系统反馈**：面板属性变动、任务发布/结算、技能解锁、奖励到账之一，落到具体数值或具体能力上，让读者有“看得见的成长”。\n'
            '- 系统不是万能：每次使用系统能力都要有**代价、冷却、前置条件或风险**，禁止无脑刷级、无脑碾压。\n'
            '- 成长有节奏：升级/变强必须解决一个具体困境，并立刻引出更高一级的新困境，不堆砌纯数值。\n'
            '- 面板信息要服务剧情张力，不做无意义的数据罗列。\n'
            '- **每章必兑现（爽点直给）**：本章必须有一次明确的爽点落地——打脸看衰者/碾压对手/奖励爆发/身份反差之一，且在本章内当场兑现，不得把爽点全部后置到“铺垫完成之后”。压迫与兑现五五开，不要写成只压不爽的慢热悬疑。\n'
            '- **少埋多兑（连贯性）**：本章必须先关闭至少 1 条上一章遗留的悬念/小钩子，才能开启新的悬念；禁止只开不收、让悬念与伏笔无限堆积。每章净新增悬念不超过 1 个。\n'
            '- **黄金三章特例**：若本章是前 3 章，必须有一个独立、当章即兑现的爽点高潮（打脸/碾压/第一桶金/能力首秀），让读者立刻看到本书的爽点形态，不得只做设定铺陈或氛围悬疑。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出本章核心困境或一个待结算的系统任务\n'
            '- 场景一（1000-1500字）：困境/压迫场景，主角在约束下挣扎，凸显系统能力的必要性与代价\n'
            '- 场景二（800-1200字）：主角运用系统/策略破局的关键场景，埋下结算引信\n'
            '- 场景三（600-1000字）：系统反馈兑现（升级/解锁/奖励）与其代价、后果\n'
            '- 结尾钩（200-400字）：制造下章悬念或抛出新任务，不用总结式收尾'
        ),
        "sensory_dialogue": "",
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止“他突然意识到/恍然大悟”式廉价顿悟。\n'
            '- 禁止系统无代价地解决主线危机；禁止纯数值罗列代替剧情。\n'
            '- 禁止开场连续两段是环境描写或纯面板描述。'
        ),
        "extras": "",
    },
    "urban_ability": {
        "role": '擅长都市异能/重生题材的中文网文作家，节奏明快、代入感强、打脸爽点密集、画面感强',
        "self_review": (
            '   - 重复风险：本章是否复制了上一次的打脸/装逼套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成空泛概括？\n'
            '   - 无脑碾压风险：打脸是否毫无铺垫、对手是否降智送人头？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“爽点高潮”是哪一段（打脸/反转/资源碾压/身份揭示之一），它如何被前文轻视/压迫衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 都市异能核心纪律（本类型核心）\n'
            '- 本章至少有一个明确爽点：打脸、扮猪吃虎后的反转、资源/实力碾压、身份揭示之一，落到具体动作与对手反应上。\n'
            '- 代入感优先：主角的优势（重生先知/异能/资源）要让读者有“我也想这样”的爽，但优势必须有边界与代价。\n'
            '- 打脸要有铺垫：先有轻视/压迫/挑衅，再有反击，对手要聪明、有反应，禁止降智捧哏式纸片人。\n'
            '- 现代都市细节要真实（行业、阶层、人情世故），不出戏。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，立刻抛出冲突、轻视或挑衅\n'
            '- 场景一（1000-1500字）：压迫/被轻视场景，铺垫反击的合理性\n'
            '- 场景二（800-1200字）：主角施展优势的关键场景，埋下打脸引信\n'
            '- 场景三（600-1000字）：打脸/反转/碾压兑现，呈现对手反应与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾'
        ),
        "sensory_dialogue": "",
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止廉价顿悟、解释性叙述代替戏剧化呈现。\n'
            '- 禁止对手降智送人头；禁止无铺垫无代价的碾压。\n'
            '- 禁止开场连续两段是环境描写。'
        ),
        "extras": "",
    },
    "romance_female": {
        "role": '擅长女频言情/宠文的中文网文作家，情绪张力细腻、关系推进有节奏、甜虐拿捏精准、代入感强',
        "self_review": (
            '   - 重复风险：本章的情绪节拍是否在重复（又一次误会、又一次心动）？\n'
            '   - 浅层执行风险：plan中哪个情感beat最可能变成“作者旁白告知”而非“通过细节让读者感受到”？\n'
            '   - 情绪悬浮风险：人物情绪是否缺乏具体事件支撑、显得无病呻吟？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“情绪高潮”是哪一段（心动/误会/和解/吃醋/守护之一），它如何被前文关系状态衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 女频言情核心纪律（本类型核心）\n'
            '- 本章必须有**明确的关系推进或情绪高潮**：拉近、误会、和解、吃醋、双向奔赴、守护之一，落到具体的眼神/动作/对白细节上。\n'
            '- 情绪要有事件支撑：每一次心动/受伤都来自具体的行为或话语，不靠旁白直接告知“她很感动”。\n'
            '- 甜虐配比：依据 plan 的节奏，把甜点与虐点落到位，避免通篇平淡或通篇狗血。\n'
            '- 男女主对手戏要有潜台词与张力；配角不做纯工具人，要有自己的立场。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章的关系状态，抛出本章情绪悬念\n'
            '- 场景一（1000-1500字）：主要关系场景，含具体互动与潜台词\n'
            '- 场景二（800-1200字）：情绪转折场景（误会加深/心防松动），推进关系弧\n'
            '- 场景三（600-1000字）：情绪高潮兑现（心动/和解/守护），呈现人物内心变化与代价\n'
            '- 结尾钩（200-400字）：留一个让读者揪心或期待的情绪悬念，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官与对话\n'
            '- 用细腻的感官与肢体语言传递情绪（视线、距离、停顿、指尖），不直接说教情绪。\n'
            '- 对话占全章25-45%，反映关系亲疏与人物性格，关键对话含言外之意。'
        ),
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止用旁白直接宣告情绪代替细节呈现。\n'
            '- 禁止无事件支撑的情绪悬浮、无逻辑的狗血反转。\n'
            '- 禁止配角沦为纯工具人；禁止开场连续两段是环境描写。'
        ),
        "extras": "",
    },
    "wanzu_xuanhuan": {
        "role": '擅长现代玄幻/万族争锋题材的中文网文作家，境界体系清晰、斗法画面感强、天骄争锋热血、爽点密集',
        "self_review": (
            '   - 重复风险：本章的斗法/突破是否在重复套路？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成空泛概括？\n'
            '   - 无铺垫破局风险：主角是否靠没讲清规则的力量强行翻盘？\n'
            '2. 针对每项风险，写一条具体规避承诺。\n'
            '3. 明确本章的“爽点高潮”是哪一段（境界突破/斗法胜出/天骄争锋/夺宝之一），它如何被前文压迫与差距衬托。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 万族玄幻核心纪律（本类型核心）\n'
            '- 境界/战力体系必须清晰、可被读者预期：每次斗法的胜负要能用已铺垫的规则解释，不靠突兀的金手指强行破局（Sanderson 第一定律：能力解题的合理度正比于规则被讲清的程度）。\n'
            '- 主角的力量有边界与代价（第二定律：限制比能力更出戏），突破/施展强法必须付出消耗或风险。\n'
            '- 本章至少一个热血爽点：境界突破、斗法胜出、天骄过招、夺宝/夺机缘之一，落到具体招式与对手反应上。\n'
            '- 斗法要有画面感：招式、气机、环境破坏、攻防转换写到位，不只报结果。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出本章的对峙、差距或威胁\n'
            '- 场景一（1000-1500字）：压迫/差距场景，凸显主角面临的实力鸿沟\n'
            '- 场景二（800-1200字）：主角借规则/底牌/机缘缩小差距的关键场景，埋下翻盘引信\n'
            '- 场景三（600-1000字）：斗法/突破爽点兑现，呈现招式攻防、对手反应与代价\n'
            '- 结尾钩（200-400字）：制造下章悬念（更强的敌人/新的机缘），不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官与对话\n'
            '- 每个场景至少2种感官锚点；斗法用具体招式与气机描写代替抽象形容。\n'
            '- 对话占全章25-45%，反映人物境界与心性，关键对话含机锋。'
        ),
        "time_marker_ban": "",
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止用没铺垫的力量强行破局；禁止对手降智。\n'
            '- 禁止廉价顿悟、解释性叙述代替战斗呈现。\n'
            '- 禁止开场连续两段是环境描写。'
        ),
        "extras": "",
    },
    "suspense": {
        "role": '擅长悬疑/心理惊悚题材的中文作家，擅长用限制视角、感官克制与公平线索营造步步紧逼的不安与反转快感',
        "self_review": (
            '   - 视角越界风险：哪一处最可能不小心写出视角人物当下不可能知道的信息（他人内心、未到场之事、最终真相）？\n'
            '   - 浅层执行风险：plan中哪个beat最可能变成“叙述概括”而非“在场景里一点点逼出的悬念”？\n'
            '   - 线索失衡风险：本章揭示是否凭空掉落、缺少前文铺垫？或者悬念只开不收、无限堆积？\n'
            '2. 针对每项风险，写一条具体规避承诺（如“反派动机只透过一件物证暗示，不进其内心”、“把怀疑写成视角人物对一个反常细节的反复打量”、“本章收掉上一章关于X的疑点”）。\n'
            '3. 明确本章埋下/兑现的线索：列出本章新增的1条悬念、以及兑现或推进的至少1条旧线索（公平地放在读者眼前）。\n'
            '4. 完成1-3后才开始正式写作。'
        ),
        "core_discipline": (
            '## 悬疑核心纪律（本类型核心）\n'
            '- **限制视角**：严格贴住视角人物，只写其当下能感知、能推断的；不得插入全知旁白或他人内心。信息差由此而生，是本类型最大的张力来源。\n'
            '- **感官克制**：恐惧/诡异藏在反常的具体细节里（多出的物件、错位的声音、本该在却消失的东西），用白描呈现，禁止“恐怖”“惊悚”“不寒而栗”式贴标签。\n'
            '- **公平线索**：每一个指向真相的关键线索都要公平地出现在读者眼前（可被干扰项掩盖），保证反转揭晓时读者回看能找到伏笔；禁止凭空掉落的关键信息。\n'
            '- **少埋多兑（连贯性）**：本章至少推进或收束1条已有悬念，才能开启新悬念；每章净新增悬念不超过1个，杜绝疑点无限堆积。\n'
            '- **留白**：关键的惊悚/揭示时刻可在临界处收笔，把最可怕的东西留给读者想象，未言之事比写满更有压迫感。'
        ),
        "structure_template": (
            '## 结构模板\n'
            '- 开场钩（200-400字）：紧接上章，抛出一个反常细节或新的疑点，立刻制造不安，禁止用时间词作开场\n'
            '- 场景一（1000-1500字）：调查/对峙/独处场景，视角人物在信息不全中试探，逐步累积怀疑与压迫\n'
            '- 场景二（800-1200字）：转折或局部揭示场景，推进plan关键beats，给出一条公平线索（或一个误导）\n'
            '- 场景三（600-1000字）：代价/反转/更深疑点场景，视角人物做出选择并承受后果\n'
            '- 结尾钩（200-400字）：以一个悬而未决的细节或反转收束本章悬念，制造追读冲动，不用总结式收尾'
        ),
        "sensory_dialogue": (
            '## 感官纪律\n'
            '- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉），优先用“反常的感官”承载诡异。\n'
            '- 用具体细节代替抽象描述（“门缝下渗出的、温的水” 而非 “诡异的气氛”）。\n'
            '- 光线、声音、温度作为情绪与悬念的放大器，不作章节进度的计时器。\n'
            '\n'
            '## 对话比例\n'
            '- 对话占全章25-45%，避免连续500字以上无对白的段落。\n'
            '- 关键对话要有试探、隐瞒与潜台词；人物可能说谎或避重就轻，但作者不得借叙述向读者捏造事实。'
        ),
        "time_marker_ban": (
            '## 时间标记禁令\n'
            '- 严禁以“翌日清晨”“这天晚上”“午后”“深夜”等时间副词切换场景或开启段落。\n'
            '- 时间流逝靠情节动作与因果链体现，每章最多2个时间词且与具体行为绑定。'
        ),
        "genre_bans": (
            '## 禁止模式\n'
            '- 禁止视角越界（写出视角人物不可能知道的真相、他人内心、未到场之事）。\n'
            '- 禁止“他突然意识到/恍然大悟”式廉价顿悟，揭示要靠线索推导。\n'
            '- 禁止靠凭空掉落的关键信息推动反转。\n'
            '- 禁止悬念只开不收、疑点无限堆积。\n'
            '- 禁止开场连续两段是环境描写，第一段内必须有人物动作、感知或对话。'
        ),
        "extras": (
            '## 情节逻辑要求\n'
            '- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果。\n'
            '- 所有伏笔在本章或后续可查文本中“收线”；揭示必须能在前文找到依据。\n'
            '\n'
            '## 章末禁令（防零增量总结收尾·硬约束）\n'
            '- **严禁用“总结段”收尾**：章末绝不能把正文已经给出的推理、线索、结论再复述一遍（哪怕换了措辞）。读者刚读过，再列一遍=零信息增量=拖慢节奏。\n'
            '- 章末必须是【前进的钩子】：抛出一个【新】疑问、一个【新】动作、一个【新】威胁，或一处让人物（与读者）措手不及的反常细节，把张力推向下一章。\n'
            '- 检验标准：把本章结尾段单独拎出来，它必须包含正文里【没出现过】的信息或动作；若它只是“由此可见…”“综上…”“他明白了…”式的归纳，立即重写。\n'
            '\n'
            '## 推理呈现纪律（提升 prose 质感与可信度）\n'
            '- 推理过程要“演”不要“报”：让视角人物通过【具体的物理动作】触碰证据（用指腹蹭过断口、把两张纸并排对光、数粉尘的层次），在动作中带出推断，而非大段内心独白罗列逻辑。\n'
            '- 一次只递进一步：每揭示一个推断，先给读者看到那个【可感知的物证细节】，再给人物的一句反应或一个动作，避免把三四个结论挤在一段里连珠炮式抛出。\n'
            '- 关键转折点的“对峙/试探”场景，多用对白与停顿（沉默、回避、答非所问）承载信息差，少用叙述代述。'
        ),
    },
}

def _build_write_system(
    preset: str,
    chapter_words: int,
    chapter_num: int,
    title: str,
    aesthetic: str,
) -> str:
    """Assemble writer system prompt: shared base + genre delta."""
    gp = GENRE_PROFILES.get(preset, GENRE_PROFILES["history"])
    parts = [
        f'你是一位{gp["role"]}。\n用中文写作本章。',
        _SELF_REVIEW_PREAMBLE + "\n" + gp["self_review"],
        _OUTPUT_SECTION.format(
            chapter_words=chapter_words, chapter_num=chapter_num, title=title,
        ),
        gp["core_discipline"],
        gp["structure_template"],
        gp.get("sensory_dialogue") or _SENSORY_DIALOGUE_DEFAULT,
        gp.get("time_marker_ban") or _TIME_MARKER_BAN_DEFAULT,
        gp["genre_bans"],
        gp.get("extras", ""),
        ANTI_FRAGMENT_BAN,
        ANTI_PITFALL_BLOCK,
        aesthetic,
    ]
    return "\n\n".join(p for p in parts if p)


OPENING_RULES_BLOCK = """## 开篇特化（黄金三章→黄金三句，3秒定生死，必须严格执行）
- 黄金三句（递进顺序不可乱）：
  · 句1=正在发生的危机：第一句直接写一个【正在发生】的冲突/动作/对话（具体、有人物在当下做事）。严禁以天气、景物、时段、回忆、世界观设定开场。
  · 句2=主角核心反差：用弱外表+强承诺，或一个反常行为，立刻立住主角的记忆点。
  · 句3=金句钩子：用一句够短够狠、可截图传播的金句收束开篇段（情绪爆发/认知颠覆/后果预告，独立成段）。
- 金手指/主角核心反差必须在本章前 1/4 内亮相或强烈预示，让读者立刻感知卖点与爽点方向。
- 出场人名 ≤5：黄金三章控制人物数量，避免读者记忆过载。
- 信息密度高但不堆设定：边演边给信息，把世界观融进动作与冲突，禁止整段解释性设定倾倒。
- 章末必须留强钩子（悬念/反转/危机/承诺），制造追读冲动。"""

def _hook_directives_block(pkg: dict) -> str:
    """Render the 吸量包's hook_directives as an opening-writer prompt block.

    build_hook_package (package.py) 早在 bootstrap 就产出"书名/简介向读者承诺了
    哪些爽点、开篇必须兑现哪个"的落地指令，但历史上只写进 hook_package.md 从未
    注入写手 prompt（P3 断链修复）。上限 5 条 / ~600 字；缺失/畸形返回 ""。
    """
    if not isinstance(pkg, dict):
        return ""
    directives = pkg.get("hook_directives")
    if not isinstance(directives, list):
        return ""
    lines: list[str] = []
    used = 0
    for d in directives[:5]:
        d = str(d).strip()
        if not d:
            continue
        if used + len(d) > 600:
            break
        lines.append(f"- {d}")
        used += len(d)
    if not lines:
        return ""
    return (
        "## 开篇吸量指令（书名/简介已向读者承诺的爽点，前三章必须兑现）\n"
        + "\n".join(lines)
    )


# 下沉/大白话语体（正交开关，可叠加任意题材）。番茄 58.6% 用户来自三线及以下、
# 通勤/夜间解压、低耐心，要的是低阅读门槛、对话优先、短句驱动的口语体，而不是
# 文学性长句。本块由 style_low_barrier_register 或免费流 platform_preset 触发，
# 独立于 style_preset 注入，让一本都市/玄幻/古言书都能切换到下沉调性。
LOW_BARRIER_REGISTER_BLOCK = """## 下沉语体（大白话，硬性执行，叠加在题材之上；当下沉语体与文风塌缩禁令冲突时以本块为准——短句可以，但必须是通顺完整的句子）
- 大白话优先：用通勤族/免费用户的日常口语写，低阅读门槛，读起来不费脑。能用常用词就不用书面词。
- 短句成句：平均句长 7-13 字，可以短但要是通顺完整的句子，不要拆成无谓断句、也不要堆破折号碎句。
- 对话驱动：本章对话占比偏高（推进情节、外放情绪、制造冲突），少用大段心理描写与环境铺陈。
- 口语连接词：用"但是/所以/结果/可是/然后"等口语连接，避免"然而/虽然/尽管/诸如/之于/继而"等书面腔。
- 动词精准、少形容词堆砌："很/非常/极其/十分"等程度虚词最少化，靠具体动作和画面让读者自己感受。
- 场景快进：一个场景不拖沓，画面感优先，快速给到冲突与爽点，不为凑字数重复铺陈。"""

SENSITIVE_WORD_AVOIDANCE_BLOCK = """## 内容分级与呈现方式（平台合规·硬性执行，最高优先级之一）
本作发布渠道带内容审核，正文过于露骨会被拦截而无法过审。写作时用**克制、含蓄、侧写、留白**的笔法处理黑暗内容——**只改呈现方式，绝不删弱情节、冲突、悬念与压迫感**。
- 暴力与伤亡：不做血腥的身体损伤特写，改写旁观者的反应、环境的变化、声音与温度的骤变、事后的痕迹与静默；用"倒下／不再动弹／再没起来／满地狼藉"这类结果性、暗示性的表达带过。
- 死亡与恐怖：不铺陈遗骸、腐坏、解剖的直观细节，改用气氛、光影、空气的凝滞、人物的战栗与心理惊惧来营造恐怖；场所与状态用偏侧写的说法（如"冷藏区／后室／失去体温的人"）。
- "吞噬/变强"设定：把核心能力写成对**能量、气息、本源、光**的汲取与消化，聚焦力量流动、身体的变化感与代价，而不是进食血肉脏器的生理过程。
- 涉性/低俗：点到为止，以情绪与张力替代露骨描写。涉政/违禁：不涉及真实政治人物、敏感时政、违禁品制法。
- 核心原则：黑暗、压迫、恐怖靠**氛围、心理、后果与感官暗示**营造，而非露骨的生理名词堆砌。宁可更克制、更留白，也不要触发审核。"""

CLOSING_RULES_BLOCK = """## 终章特化（这是全书最后一章，必须写成真正的结局，严格执行；与上面通用的"结尾钩/制造下章悬念"规则冲突时，以本块为准）
- 兑现主线：本章必须正面解决全书/本卷的核心矛盾，把前文铺设的人、信息、伏笔在页面上兑现，不得回避或拖延。
- 谜底必须明确（悬疑/推理硬性）：凶手是谁、真相是什么、核心谜题如何解开，必须在本章给出确定答案；不得含糊、不得"留给读者判断"、不得以模糊暗示替代揭晓。
- 收束所有未结悬念：前文标记为 open 的关键伏笔/悬念，本章须逐一收束或明确交代其去向，不得只收一条而无视其余。
- 给完成感：主角要对自己的处境做出一个明确的、有重量的选择或姿态，让读者感到"这一段落下了帷幕"。
- 禁止引入任何新元素：不得在终章新增任何新人物、新势力、新案件、新悬念、新危机、新反转钩子；终章只允许收束已有元素，不允许开启任何新线。
- 禁止开放式悬念结尾：不得以一个全新的、未解决的危机/急报/反转作为最后一钩（如"更大的敌人出现了""新的危机正在逼近""一封急报送到"）。
- 允许余韵而非悬念：结尾可以用一句话点出更高层级的远景或主题升华，但它是"余韵/定调"，不是抛给读者的新问题，读者读完不应觉得"必须看下一章"。
- 收束式结尾（300-600字）：以情绪落点、主题呼应或定格画面收尾，替代常规的"下章悬念结尾钩"。"""

CLOSING_APPROACH_BLOCK = """## 收束区（距全书结局仅剩 {remaining} 章，本章硬性执行渐进收束）
- 进入收束区：本章起停止开启任何【新的重大线索/新势力/新谜团】，把笔墨集中到已铺设伏笔的兑现与汇流上。
- 伏笔兑现优先：每章至少正面推进或兑现 1 条已开启的关键伏线/悬念，向全书核心矛盾的总爆发汇聚，不再横向铺开。
- 钩子转向：章末钩子从"抛出新危机"逐步转为"既有矛盾收紧/摊牌临近"——是收口的张力，而非新坑。
- 节奏提速：删减支线与过渡，把剩余的 {remaining} 章空间留给主线高潮与情感落点，避免在收束区还在原地展开细节。
- 仍非终章：本章不必给出最终答案与大团圆，但必须让读者明确感到"故事正在收口、结局在逼近"。"""

# Narrative-mode steering blocks. Injected per chapter based on
# config.narrative_mode(). They are ADDITIVE craft directives layered on top of
# the style preset, letting one engine support both "单密室+精密推理" and
# "强钩子+情绪外放+可连载" without forking the preset prompts.
MODE_REASONING_BLOCK = """## 叙事模式：单密室·精密推理（本章硬性执行）
- 收敛舞台：把核心场景收束在一个封闭/半封闭空间内（密室、单间、一处现场），减少场景跳转，让推理在受限空间里逐步逼出。
- 物证驱动：本章的怀疑与揭示必须挂在可触摸、可观察的具体物件/身体状态上（压痕、链节、血迹方向、齿痕、反光、温湿度），而非抽象的"角度不对/逻辑矛盾"。
- 公平线索：指向真相的关键线索必须公平地出现在读者眼前（可被干扰项掩盖），保证回看能找到伏笔。
- 视觉化矛盾：核心爽点尽量做成读者一眼能懂的视觉矛盾（镜中有而现实无、左右相反、死前姿态与现状冲突），而非纯口头推断。
- 收束优先：本章至少推进或收束1条已有疑点；宁可少开新悬念，也要把推理链条在页面上闭合。"""

MODE_SERIAL_BLOCK = """## 叙事模式：强钩子·情绪外放·可连载（本章硬性执行）
- 强开场钩：前 1/4 以内必须抛出强冲突/强悬念/强反差，禁止用铺垫、回忆、设定开场。
- 情绪外放：人物情绪要敢于外显并落到具体动作与对白上（爆发、对峙、决裂、表白），给读者强烈的情绪共振，不要一味克制内敛。
- 节奏紧凑：每个场景都要有推进或翻转，避免大段静态描写与原地踏步；钩子—兑现—新钩子的连载节奏要清晰。
- 章末强钩：章末必须留一个让读者"必须看下一章"的悬念/反转/危机/承诺（终章除外）。
- 单章爽点：本章应至少给读者一个明确的情绪兑现或小高潮（揭晓、打脸、反转、关系推进），不要把所有兑现都推迟。
### 单章节奏模板「起承转爽」（按此组织本章，可微调比例）
- 起（章首约 1/6）：极短回顾承接上一章钩子，立即引出本章要解决的具体冲突，不重复复述上一章场景。
- 承（约 1/2）：冲突升级、压力加码，让局势越收越紧，信息密度随之上升。
- 转（约 1/6）：一次转折——敌人露底牌 / 绝望转希望 / 反转翻盘的支点。
- 爽（约 1/6 + 章末）：主角反击、真相揭晓或意外收获兑现情绪，收在一个前进式强钩上（不是已知信息的总结）。
### 节奏起伏（防 AI 通病：节奏太均匀）
- 不要每段等长、密度一致地平铺：紧张/动作/爆发段用短句加速、句子要短促有力；铺陈/情绪沉淀段用较长的成句减速。
- 全章至少有一处明显的"急停"——在最紧张处用一两句极短句或单句断点制造顿挫，再推进。
- 避免连续多段相同节奏；该快则快、该慢则慢，让读者的情绪坐过山车而非走平路。
### 爽点硬指标（防散文化/防流水账）
- 本章必须有至少1个可清晰指认的"爽点时刻"——读者看到这里会觉得爽/过瘾/解气/震撼的具体场景（打脸/翻盘/揭秘/关系突破/能力觉醒等）。这个爽点不能是"氛围"或"感觉"，必须是一个落在页面上、有具体动作和结果的高潮场景。
- 每章情绪曲线必须有明显的高低：不允许全章同一情绪强度平推。至少有一处"急速升温"和一处"短暂喘息"。
- 禁止章末用"他知道/她明白/这一切都……"式总结收尾，章末必须是前进的动作、对话或悬念。"""



REVISE_SYSTEM = """你是一位中文网文修订作者。
请根据终审编辑报告修订整章。
保留标题与核心事件。不要引入新的连续性风险。
优先做有针对性的结构性修复，而非表面润色：
- 补全缺失的因果桥梁与具体场景。
- 替换重复的场景调度或章末手法。
- 让大纲节拍在页面上可见。
- 强化人物能动性、程序摩擦感与压迫-兑现节奏。
""" + ANTI_FRAGMENT_BAN + """
修订不得引入比原文更碎片化的文风。
只输出修订后的章节。"""

EXTRACT_SYSTEM = """你是长篇小说引擎中的事件溯源抽取器。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "title": "...",
  "events": [{"type":"plot|world|character|force|thread|item|battle|relationship","summary":"...","effects":[]}],
  "entities": [{"entity_type":"character|force|place|item|rule","name":"...","state_patch":{}}],
  "threads": [{"id":"stable-id","description":"...","status":"open|advanced|recovered|dropped","thread_type":"plot|reader_promise|character_arc|world_rule|relationship","introduced_chapter":1,"due_chapter":20,"depends_on":"前置线索id（可选,为空则无依赖）","priority":5,"half_life":0,"payload":{}}],
  "causal_links": [{"from_event":"来源事件概括","to_event":"预期的未来事件或后果","link_type":"causes|enables|blocks|requires","description":"该因果关联为何存在"}],
  "metrics": {
    "payoff_type":"court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
    "conflict_type":"court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
    "tension":1-10,
    "novelty":1-10,
    "hook_strength":1-10,
    "emotional_tone":"..."
  },
  "relationship_changes": [{"char_a":"角色A","char_b":"角色B","event":"发生了什么","new_stage":"potential|contact|tension|trust|conflict|resolution|deepened|broken","intensity_delta":0.5}],
  "info_revelations": [{"id":"stable-id","description":"信息/秘密/谜团描述","status":"planted|hinted|partial_reveal|revealed","reveal_type":"mystery|secret|clue|misdirect","importance":5,"due_chapter":null}],
  "memory_updates": {
    "bible": [],
    "characters": [],
    "timeline": [],
    "threads": []
  },
  "dialogue_fingerprints": [{"character":"角色名","speaking_style":"<=80字：该角色标志性说话方式——句长偏好/口头禅/语气词/问句比例/称呼习惯/独特用词"}],
  "protagonist_state": "<=600 个中文字符 markdown：主角当前的目标、资源、恐惧、秘密、持续的压力，以及尚未决断的关键决定。须反映本章带来的变化，须自足（新读者可据此接续），避免含糊措辞。",
  "next_12_directions": ["10-12 条针对后续章节的具体指令；每条一句中文，明确指出具体必须发生什么，而非抽象主题"]
}

关系阶段说明：potential(尚无互动)→contact(初次接触)→tension(紧张/试探)→trust(建立信任)→conflict(产生冲突)→resolution(冲突化解)→deepened(关系深化)→broken(关系破裂)。intensity_delta 为正数表示关系拉近，负数表示疏远。
信息揭示说明：planted=首次埋下伏笔/谜团；hinted=给了线索但未揭晓；partial_reveal=部分真相浮出；revealed=完整揭晓。importance 1-10。

为每条 thread 设定 "thread_type"：
- "reader_promise"：对读者做出的、必须兑现的明确钩子/承诺（被预告的对决、立下的复仇、被埋的揭示，"他日必报此仇"式的债）。
- "character_arc"：某个人物的个人成长/转变弧线。
- "world_rule"：关于世界的、后续章节必须遵守的规则/约束。
- "relationship"：两方之间演变中的关系。
- "plot"：任何普通情节伏线（默认）。
拿不准时用 "plot"。

【线索 id 复用铁律（防止伏笔台账爆炸）】
- 下面会给你一份"当前仍未关闭的线索清单"（含 id / 描述 / 状态）。
- 若本章推进或兑现的是清单里已存在的线索，**必须原样复用其 id**，绝不能为同一条线索新造 id。
- 已经兑现/收束的线索，输出该线索并把 status 设为 "recovered"（已回收）或 "dropped"（已放弃），让它从台账移除。
- 只有当出现清单里完全没有的全新伏笔时，才创建一个新的、稳定的 id。
- 不要把同一条线索拆成多个措辞略有差异的新条目。

对话指纹说明：只为本章有 ≥2 句台词的角色生成指纹。描述须具体到可复现的语言特征（如"总以反问句结尾""常用'呵'表轻蔑""句子极短,3-5字一拍""喜欢用'本座'自称"），不要泛泛写"语气坚定"或"说话温柔"。"""

def carried_over_partial_beats(paths: Paths, chapter_num: int, limit: int = 6) -> list[dict[str, Any]]:
    """Return the previous chapter's partial/absent beats so the next writer can repair them.

    Reads final_review.json -> review_round0.json -> review_round1.json in order
    of preference, and returns up to `limit` entries containing
    {"beat": str, "status": "partial|absent", "evidence": str}.
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        beats = data.get("beats_audit") or []
        partial: list[dict[str, Any]] = []
        for entry in beats:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).lower()
            if status not in ("partial", "absent"):
                continue
            partial.append({
                "beat": str(entry.get("beat", ""))[:300],
                "status": status,
                "evidence": str(entry.get("evidence", ""))[:200],
            })
            if len(partial) >= limit:
                break
        if partial:
            return partial
    return []


def sensitive_word_avoidance_block(config: dict[str, Any]) -> str:
    """Content-register directive steering the model to render dark content (violence,
    death, horror, the 吞噬 power) obliquely — aftermath, sensory/psychological
    suggestion, energy-absorption framing — so a content-moderation gateway does not
    reject the chapter (sensitive_words_detected). Gated by novel.sensitive_word_avoidance.

    NOTE: deliberately category-based and positive-framed. It does NOT list explicit
    trigger nouns: echoing raw banned words into the prompt primes the model to emit
    them (observed: adding a word-list made generations fail FASTER), so we name
    categories and prescribe the oblique technique instead. Returns "" when off.
    """
    if not bool(config["novel"].get("sensitive_word_avoidance", False)):
        return ""
    return SENSITIVE_WORD_AVOIDANCE_BLOCK


def writer_directives_for_chapter(paths: Paths, chapter_num: int, limit: int = 6) -> list[str]:
    """Return directives carried from the previous chapter's review.

    Reads the previous chapter's review (final_review.json preferred) and
    extracts a flat list of imperative strings to inject at the top of the
    current chapter's write prompt. This forms a review->writer feedback loop
    that is more concrete than plan-level required_constraints (it speaks in
    terms of execution, not strategy).
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    directives: list[str] = []
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        for field in ("writer_directives_for_next_chapter", "writer_directives"):
            for item in data.get(field, []) or []:
                text = str(item).strip()
                if text and text not in directives:
                    directives.append(text)
                if len(directives) >= limit:
                    return directives
        if directives:
            return directives
    return directives


def _preflight_negative_list(
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    lookback: int = 5,
) -> dict[str, Any]:
    """Build a pre-write negative list from recent failure modes.

    Collects gate_rejects (cross-chapter fossils, adjacent repetition),
    style collapse flags (em-dash density, fragment lines), and concrete
    fossil clauses from the last N chapters to front-load avoidance directives
    BEFORE the first draft is generated, rather than discovering them only
    after a low review score.

    Returns {"items": [...], "fossils": [...], "style_warnings": [...]}
    """
    if chapter_num <= 1:
        return {"items": [], "fossils": [], "style_warnings": []}

    items: list[str] = []
    fossils: set[str] = set()
    style_warnings: list[str] = []
    seen_gates: set[str] = set()

    start = max(1, chapter_num - lookback)
    for ch in range(start, chapter_num):
        # Check final_review for gate_rejects
        for key in ("final_review.json", "review_round1.json", "review_round0.json"):
            data = load_checkpoint(paths, ch, key)
            if not isinstance(data, dict):
                continue

            gate_rejects = data.get("gate_rejects", [])
            if isinstance(gate_rejects, list):
                for gr in gate_rejects:
                    if not isinstance(gr, dict):
                        continue
                    gate = str(gr.get("gate", "")).strip()
                    if not gate or gate in seen_gates:
                        continue
                    seen_gates.add(gate)

                    evidence = gr.get("evidence", {})
                    if gate == "cross_chapter_repetition":
                        examples = evidence.get("examples", [])
                        if isinstance(examples, list):
                            for ex in examples[:4]:
                                clause = str(ex).strip()
                                if clause and len(clause) >= 6:
                                    fossils.add(clause)
                        items.append(
                            f"近期检测到跨章节化石句（逐字复读）；本章严禁再现以下措辞或结构相似的表达。"
                        )
                    elif gate == "adjacent_repetition":
                        metrics = evidence.get("metrics", {})
                        overlap = metrics.get("clause_overlap")
                        if overlap:
                            items.append(
                                f"Ch{ch} 大量逐字复述前章内容（overlap={overlap:.2f}）；"
                                "本章必须从新事件开始，前章场景只许一笔带过。"
                            )

            # Collect style flags
            flags = data.get("style_flags", [])
            if isinstance(flags, list):
                for flag in flags[:3]:
                    flag_text = str(flag).strip()
                    if flag_text and flag_text not in style_warnings:
                        style_warnings.append(flag_text)

            if gate_rejects or flags:
                break

    # Book-wide fossils: persistent avoid-list mined across the WHOLE book by
    # review.book_wide_fossils (cached every book_fossil_every chapters). Unlike
    # the lookback fossils above, these reflect chronic habit-stiffening over the
    # entire book, so they must be injected on EVERY chapter, not just after a
    # recent gate-reject. Includes severity (frac/chapter_count) so the writer
    # knows which fossils are most critical to avoid.
    if bool(config["novel"].get("book_fossil_enabled", True)):
        try:
            cache = paths.logs_dir / "book_fossils.json"
            if cache.exists():
                bf = json.loads(read_text(cache))
                bf_fossils = bf.get("fossils") or []
                bf_phrases = bf.get("phrases") or []
                hard_fossils = [f for f in bf_fossils if isinstance(f, dict) and f.get("frac", 0) >= 0.20]
                soft_fossils = [f for f in bf_fossils if isinstance(f, dict) and 0 < f.get("frac", 0) < 0.20]
                for f in hard_fossils[:6]:
                    ph = str(f.get("phrase", "")).strip()
                    if ph:
                        fossils.add(ph)
                        items.append(
                            "『%s』已出现在 %d 章 (%.0f%%)——硬化石，本章正文禁止出现。"
                            % (ph, f.get("chapter_count", 0), f.get("frac", 0) * 100)
                        )
                for f in soft_fossils[:8]:
                    ph = str(f.get("phrase", "")).strip()
                    if ph:
                        fossils.add(ph)
                for ph in bf_phrases[:12]:
                    ph = str(ph).strip()
                    if ph:
                        fossils.add(ph)
                if hard_fossils or bf_phrases:
                    items.append(
                        "全书高频僵化短语（机械口癖）已累积，本章起必须主动换用不同的"
                        "动作落点、感官通道与句式，严禁继续复刻下列微动作片段。"
                    )
        except Exception:
            pass

    genre_fatigue = config["novel"].get("fatigue_words", [])
    if isinstance(genre_fatigue, str):
        genre_fatigue = [w.strip() for w in genre_fatigue.split(",") if w.strip()]
    for word in genre_fatigue[:12]:
        w = str(word).strip()
        if w and w not in fossils:
            fossils.add(w)
            items.append(f"体裁疲劳词「{w}」——尽量避免或限制使用。")

    return {
        "items": items[:10],
        "fossils": sorted(fossils)[:20],
        "style_warnings": style_warnings[:4],
    }


ABSTRACT_BEAT_MARKERS = (
    "推导出",
    "意识到",
    "想通",
    "完成",
    "还原",
    "引导",
    "心算",
    "反应过来",
    "发现",
    "确认",
    "证明",
    "判断",
    "说服",
    "揭示",
)

CONCRETE_BEAT_MARKERS = (
    "把",
    "按",
    "压",
    "递",
    "翻",
    "拿",
    "写",
    "锁",
    "摁",
    "贴",
    "拆",
    "划",
    "量",
    "照",
    "指",
    "举",
    "撕",
    "收",
    "扣",
    "放",
    "打开",
    "合上",
    "签",
    "盖",
    "查",
    "对照",
    "并排",
)


def _beat_needs_concretization(beat: str) -> bool:
    """Heuristic: abstract realization verbs need an object/action anchor."""
    text = str(beat or "").strip()
    if not text:
        return False
    has_abstract = any(marker in text for marker in ABSTRACT_BEAT_MARKERS)
    has_concrete = any(marker in text for marker in CONCRETE_BEAT_MARKERS)
    return has_abstract and not has_concrete


def _beat_concrete_details(beat: str) -> list[str]:
    """Extract the concrete verifiable details a beat promises, so the writer
    prompt can require each one be acted out on the page (not summarised away).

    v8 failure mode: the plan said '她另一只手在药箱搭扣上摸了一下' but the prose
    wrote only '搭扣发出一声轻响' — the concrete action was replaced by its sound/
    result, costing a partial-beat penalty. We surface short noun/action phrases
    so the execution ledger can name them as non-negotiable acceptance items.
    """
    text = str(beat or "").strip()
    if not text:
        return []
    details: list[str] = []
    # Split on common Chinese clause separators and keep clauses that carry a
    # concrete body action or named object (i.e. NOT pure abstract-realization).
    parts = re.split(r"[，。；、,;]", text)
    for part in parts:
        part = part.strip()
        if not part or len(part) < 4:
            continue
        has_concrete = any(m in part for m in CONCRETE_BEAT_MARKERS)
        has_abstract_only = any(m in part for m in ABSTRACT_BEAT_MARKERS) and not has_concrete
        if has_concrete and not has_abstract_only:
            details.append(part[:40])
    return details


def _first_draft_execution_ledger(config: dict[str, Any], plan: dict[str, Any]) -> str:
    """Return a compact beat-to-page execution checklist for the writer prompt."""
    novel_cfg = config.get("novel", {}) if isinstance(config, dict) else {}
    if not bool(novel_cfg.get("first_draft_execution_ledger", True)):
        return ""
    beats = plan.get("beats") if isinstance(plan, dict) else None
    if not isinstance(beats, list):
        return ""
    beat_list = [str(b).strip() for b in beats if str(b).strip()]
    if not beat_list:
        return ""

    chapter_words = int(novel_cfg.get("chapter_words", 4000) or 4000)
    per_beat = max(260, int(chapter_words / max(1, len(beat_list)) * 0.75))
    lines = [
        "### 首稿页面执行账本（内部执行，不要输出账本）",
        "- 写作前先把每个 beat 映射成：上一拍后果 -> 角色当下目标 -> 阻力/对手动作 -> 可见动作或有攻防的对话 -> 新信息/代价/局势变化。",
        "- 每个 beat 至少占一个有场面功能的自然段或对话回合；禁止把两个以上关键 beat 压缩成一句总结。",
        f"- 节奏预算：本章 {len(beat_list)} 个 beat，平均每个关键 beat 约 {per_beat}-{per_beat + 220} 字；第一个 beat 必须在前 1/3 之前进入冲突或行动。",
        "- 转场只写因果，不写流水账时间标签；下一场必须由上一场的后果推出来。",
        "- 【细节保真·最高优先级】beat 里写明的每一个具体动作（谁的手做了什么）、具体物件、具体数字、具体动机，都是本章验收项，必须在正文里把该动作/物件本身实演出来；"
        "严禁用它的“结果”或“声音”替代动作本身（例如 beat 写“她另一只手在药箱搭扣上摸了一下”，正文只写“搭扣发出一声轻响”即判不合格——必须写出“摸”这个动作和沈澜看到的手），"
        "严禁用“一笔带过/读了也读不出/总结一句”抹掉 beat 里要求的内心挣扎或动机铺垫。删一个具体细节就少一分。",
    ]
    # Per-beat enumeration intentionally removed: each beat (with its concrete
    # acceptance details) is re-stated once in the tail-of-prompt 验收清单 built
    # by write_chapter, where recency makes it actually bind. Duplicating the
    # list here diluted that anchor and roughly doubled the beat token cost.
    return "\n".join(lines) + "\n"


def _prewrite_quality_contract(
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    """Build a compact quality gate for the first draft prompt.

    The full review JSON is still included later, but it is too noisy for the
    writer. This block translates the review rubric into concrete pass/fail
    obligations before any prose is produced.
    """
    if not bool(config["novel"].get("prewrite_quality_contract", True)):
        return ""

    threshold = float(config["novel"].get("quality_threshold", 8.0))
    dimension_floor = float(config["novel"].get("prewrite_dimension_floor", max(7.2, threshold - 0.3)))
    selected_plan_score = plan_score(decision)
    beats = plan.get("beats") if isinstance(plan, dict) else None
    beat_count = len(beats) if isinstance(beats, list) else 0
    beat_list = [str(b).strip() for b in beats if str(b).strip()] if isinstance(beats, list) else []

    # Mirror the review rubric's downward score pressure so the writer pre-empts
    # the exact things the reviewer penalises, instead of discovering them only
    # after a low score. payoff/hook are the plan's own stated promises.
    payoff_intent = str((plan.get("payoff") if isinstance(plan, dict) else "") or "").strip()
    hook_intent = str((plan.get("hook") if isinstance(plan, dict) else "") or "").strip()
    plan_risk = str((plan.get("risk") if isinstance(plan, dict) else "") or "").strip()

    lines = [
        "## 写前质量合同（首稿必须达标，不要留给低分后重写）",
        "- 【细节保真·最高优先级】大纲每个 beat 里写明的具体动作、具体物件、具体数字、具体动机，都是本章硬验收项——必须在正文里把该动作/物件/动机本身实演出来，"
        "缺一项即判该 beat 为 partial 并扣分。严禁用结果或声音替代动作本身（如 beat 写“手摸搭扣”，正文写成“搭扣响了”不合格），"
        "严禁用“一笔带过/读了也读不出/他决定放弃”等总结句抹掉 beat 要求的挣扎、动机或铺垫，也严禁让正文与大纲的具体描述自相矛盾（如大纲“无人影”正文却写“有灯光人在”）。",
        f"- 目标：首稿总分必须达到 {threshold:.1f}+；readthrough/payoff/novelty/prose/continuity 五个维度都不得低于 {dimension_floor:.1f}。",
        f"- 当前大纲仲裁分：{selected_plan_score:.1f}/10。若大纲分偏低，正文必须用更具体的场景执行弥补，不得照抄抽象意图。",
        "- 本章必须同时具备：清晰剧情推进、主角主动选择、可见压力、挣来的兑现、具体章末钩子。",
        "- 新鲜度必须落到场景、信息来源、冲突类型或章末手法之一；不能只换措辞重复近期章节。",
        "- 写作前在内部逐项自检上述门槛；不要输出检查过程，只输出合格正文。",
    ]
    # Show the writer the审稿员的扣分清单 directly, framed as "首稿就要避免"。
    # These mirror review.py REVIEW_SYSTEM 的软性惩罚项——把"事后被扣分"前移成"事前的硬约束"。
    lines.append("\n### 终审扣分重点（系统指令中的铁律全部适用，以下为额外重点）")
    lines.extend([
        "- 大纲节拍缺席/只暗示：每个缺席 -1.0、部分实现 -0.5；超过 30% 节拍 partial 再 -0.5。",
        "- 含糊带过时间线/金钱/路线/程序：每处 -1.0；该具体的地方必须落到动作或对话上。",
        "- 兑现空洞——主角轻松全胜、对手降智、爽点无代价：payoff 维度直接压低。",
        "- 重复近期场景形态/开场方式/章末手法：-1.0；必须换一种结构呈现。",
        "- 章末钩子笼统或近期已用过：结尾必须抛出具体的、未解决的新问题。",
    ])
    if payoff_intent:
        lines.append("\n### Payoff 兑现验收（53%% 的首稿失败源于 payoff 空转）")
        lines.append(
            "大纲承诺的 payoff：%s" % payoff_intent[:250]
        )
        lines.extend([
            "验收标准——正文必须同时满足以下三条，否则 payoff 维度直接低分：",
            "1. 用 ≥150 字的连续场景动作（对话+动作+环境反应）实演 payoff，禁止用叙述句『他终于做到了』一笔带过。",
            "2. payoff 必须由主角的主动选择或行动触发，不得凭空降临、对手突然降智、或旁人代劳。",
            "3. payoff 的结果必须在页面上产生可见后果（信息变化/关系变化/资源得失），而非停留在内心感悟。",
        ])
    if hook_intent:
        lines.append("\n### 章末钩子落地要求")
        lines.append(
            "大纲承诺的 hook：%s" % hook_intent[:250]
        )

    # --- Hook dedup: inject recent chapter endings so the writer avoids
    # repeating the same hook pattern in consecutive chapters.
    try:
        from config import chapter_path as _ch_path
        recent_hooks: list[str] = []
        for prev in range(max(1, chapter_num - 3), chapter_num):
            cp = _ch_path(paths, prev)
            if cp.exists():
                tail = cp.read_text(encoding="utf-8", errors="replace")[-300:]
                last_para = tail.rsplit("\n\n", 1)[-1].strip()
                if last_para:
                    recent_hooks.append("Ch%d 结尾：%s" % (prev, last_para[:120]))
        if recent_hooks:
            lines.append("\n### 近期章末钩子（本章结尾必须与以下模式不同构）")
            lines.append("27%% 的首稿失败源于 hook 与前章同构。以下是最近章节的结尾：")
            lines.extend("- %s" % h for h in recent_hooks)
            lines.append("本章结尾必须在【手法/信息类型/悬念载体】上至少换一种，禁止复用上述模式。")
    except Exception:
        pass
    if plan_risk:
        lines.append(f"\n### 大纲已点名的首要风险（本章主动规避，不要踩中）\n- {plan_risk[:240]}")
    if beat_list:
        # Per-beat enumeration is deliberately NOT repeated here: the full beat
        # acceptance checklist is appended at the very END of the user message
        # (recency anchor in write_chapter), where attention is strongest.
        # Repeating it mid-prompt diluted the tail anchor and wasted tokens.
        ledger = _first_draft_execution_ledger(config, plan)
        if ledger:
            lines.append("\n" + ledger.rstrip())
    elif beat_count:
        lines.append(f"- 本章大纲共有 {beat_count} 个 beat；正文完成前内部确认没有 partial/absent beat。")

    # --- Style budget: inject concrete style-health thresholds so the writer
    # pre-empts the deterministic penalties instead of discovering them at review.
    try:
        import sqlite3
        from store import ThreadLocalDB, recent_metrics as _rqm
        db = ThreadLocalDB(paths.database)
        rows = _rqm(db, 6)
        em_vals = [float(r["em_dash_per_kchar"]) for r in rows
                   if r.get("em_dash_per_kchar") is not None]
        if em_vals:
            em_mean = sum(em_vals) / len(em_vals)
            em_warn = float(config["novel"].get("style_em_dash_per_kchar_warn", 6.0))
            em_target = min(em_mean * 1.3, em_warn * 0.7)
            _emdash = "——"
            lines.append("\n### 风格预算（确定性扣分项，LLM评分无法覆盖）")
            lines.append(
                "• 破折号（%s）密度：近期均值 %.1f/千字，"
                "本章目标 ≤%.1f/千字。超过 %.0f/千字 "
                "将被确定性扣1分。用完整句叙事，"
                "避免“A%sB%sC”碎句。"
                % (_emdash, em_mean, em_target, em_warn, _emdash, _emdash)
            )
            sp_vals = [float(r["style_penalty"]) for r in rows
                       if r.get("style_penalty") is not None]
            if sp_vals and max(sp_vals) > 0:
                sp_str = ", ".join("%.1f" % v for v in reversed(sp_vals))
                lines.append(
                    "• 近 %d 章风格扣分：%s。"
                    "扣分>0意味着首稿文体不合格。"
                    % (len(sp_vals), sp_str)
                )
    except Exception:
        pass

    # --- Dialogue ratio: inject target when recent chapters run low on dialogue.
    try:
        import sqlite3
        from store import ThreadLocalDB, recent_metrics as _rqm_dlg
        _db_dlg = ThreadLocalDB(paths.database)
        _rows_dlg = _rqm_dlg(_db_dlg, 5)
        _dlg_vals = [float(r["dialogue_char_ratio"]) for r in _rows_dlg
                     if r.get("dialogue_char_ratio") is not None]
        if _dlg_vals:
            _dlg_mean = sum(_dlg_vals) / len(_dlg_vals)
            _dlg_target = float(config["novel"].get("dialogue_char_ratio_target", 0.20))
            if _dlg_mean < _dlg_target:
                lines.append("\n### 对话占比预警")
                lines.append(
                    "近 %d 章对话占比均值仅 %.0f%%，目标 ≥%.0f%%。"
                    "本章必须增加角色间的对话交锋：将心理独白/叙述总结改为对话呈现，"
                    "关键信息通过对话传递而非旁白叙述。每个场景至少包含一组有效对话（≥3轮交锋）。"
                    % (len(_dlg_vals), _dlg_mean * 100, _dlg_target * 100)
                )
    except Exception:
        pass

    # --- AI flavor budget: inject recent AI-cliché density so the writer
    # pre-empts the deterministic ai_flavor_health penalties.
    if bool(config["novel"].get("ai_flavor_enabled", True)):
        try:
            import sqlite3
            from store import ThreadLocalDB, recent_metrics as _rqm_ai
            _db_ai = ThreadLocalDB(paths.database)
            _rows_ai = _rqm_ai(_db_ai, 5)
            _ai_vals = [float(r["ai_cliche_per_kchar"]) for r in _rows_ai
                        if r.get("ai_cliche_per_kchar") is not None]
            _meta_vals = [float(r["metaphor_per_kchar"]) for r in _rows_ai
                          if r.get("metaphor_per_kchar") is not None]
            _tns_vals = [float(r["tell_not_show_per_kchar"]) for r in _rows_ai
                         if r.get("tell_not_show_per_kchar") is not None]
            if _ai_vals or _meta_vals or _tns_vals:
                lines.append("\n### AI味预算（确定性扣分项，超标即扣分）")
                if _ai_vals:
                    _ai_mean = sum(_ai_vals) / len(_ai_vals)
                    _ai_warn = float(config["novel"].get("ai_cliche_per_kchar_warn", 4.0))
                    _ai_target = min(_ai_mean * 0.7, _ai_warn * 0.6)
                    lines.append(
                        "• AI套话密度：近期均值 %.1f/千字，本章目标 <=%.1f/千字。"
                        "超过 %.0f/千字 将被确定性扣分。"
                        "高频套话黑名单：心中一沉、瞳孔一缩、嘴角微微上扬、"
                        "缓缓开口、深吸一口气、一时间、此刻——"
                        "换用只属于当前角色和场景的具体反应。"
                        % (_ai_mean, max(_ai_target, 1.0), _ai_warn)
                    )
                if _meta_vals:
                    _meta_mean = sum(_meta_vals) / len(_meta_vals)
                    _meta_warn = float(config["novel"].get("metaphor_per_kchar_warn", 5.0))
                    if _meta_mean > _meta_warn * 0.6:
                        lines.append(
                            "• 比喻密度：近期均值 %.1f/千字。每千字控制在3个以内，"
                            "每个比喻必须新鲜准确，禁用陈腐比喻。"
                            % _meta_mean
                        )
                if _tns_vals:
                    _tns_mean = sum(_tns_vals) / len(_tns_vals)
                    _tns_warn = float(config["novel"].get("tell_not_show_per_kchar_warn", 3.0))
                    if _tns_mean > _tns_warn * 0.5:
                        lines.append(
                            '• 情感直述：近期 %.1f/千字。用行为和细节展示情绪，'
                            '不用"他感到/她觉得"+情绪词的贴标签句式。'
                            % _tns_mean
                        )
        except Exception:
            pass

    # --- Opening diversity: prevent consecutive same-type chapter openings.
    if bool(config["novel"].get("opening_diversity_enabled", True)):
        try:
            from config import chapter_path as _ch_path_open
            _recent_openings: list[str] = []
            for _prev_ch in range(max(1, chapter_num - 5), chapter_num):
                _cp_open = _ch_path_open(paths, _prev_ch)
                if _cp_open.exists():
                    _body = _cp_open.read_text(encoding="utf-8", errors="replace")
                    for _line in _body.split("\n"):
                        _ls = _line.strip()
                        if _ls and not _ls.startswith("#") and not _ls.startswith("第") and len(_ls) > 5:
                            _recent_openings.append("Ch%d：%s" % (_prev_ch, _ls[:60]))
                            break
            if len(_recent_openings) >= 3:
                lines.append("\n### 开场多样性（近期章节开头）")
                lines.extend("- %s" % o for o in _recent_openings[-5:])
                lines.append(
                    "本章开头必须与上述模式不同构。可选手法：对话直入、环境/天气切入、"
                    "配角视角、时间跳跃、物件特写、回忆闪回。禁止连续3章以上用同一类型开场。"
                )
        except Exception:
            pass

    banned = str(config["novel"].get("banned_descriptors", "")).strip()
    if banned:
        lines.append("\n### 描写禁用标签（本章不得出现）")
        for item in banned.split(","):
            item = item.strip()
            if item:
                lines.append("- 禁止使用『%s』及其变体描写任何角色。" % item)

    return "\n".join(lines) + "\n"


HOOK_REVISE_SYSTEM = """你是一位中文网文章末钩子专家。
只重写本章最后一段（结尾段），让结尾钩子犀利、具体，并为读者制造清晰的下一章问题。

约束：
- 不要改动重写点之前的任何内容。输出完整章节，开头与中段逐字保留原样，仅替换结尾段。
- 新结尾必须避免以下禁忌：廉价顿悟（"他突然意识到"）、总结式收尾（"一切才刚刚开始"）、抽象的伏笔。
- 新结尾应抛出一个具体、明确的问题，或设置一个下一章必须应对的具体障碍。
- 用完整句收束，禁止用破折号串联碎句或单词短句堆叠。
- 契合既定叙事声音；不要引入尚未确立的新人物或新事实。
- 替换段长度与原结尾大致相当（以用户给出的原结尾字数为准，误差 20% 以内）。"""


def revise_hook_only(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter: str,
    plan: dict[str, Any],
    review: dict[str, Any],
    tail_to_revise_chars: int = 400,
) -> str:
    """Rewrite only the last ~300-500 chars of the chapter to fix a weak ending hook.

    This is a much cheaper alternative to a full revise: a single small LLM call
    that the writer copies the head verbatim and only mutates the tail. Returns
    the new full chapter text.
    """
    chapter = normalize_chapter(chapter)
    n = len(chapter)
    cut = max(0, n - tail_to_revise_chars)
    # Snap cut point to a paragraph boundary if possible (look back up to 200 chars
    # for double-newline; otherwise single newline).
    snap_window = chapter[max(0, cut - 200): cut + 200]
    for marker in ("\n\n", "\n"):
        idx = snap_window.find(marker)
        if idx >= 0:
            cut = max(0, cut - 200) + idx + len(marker)
            break
    head = chapter[:cut]
    original_tail = chapter[cut:]
    user = f"""## 大纲JSON（供参考）
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 审校反馈（钩子为何偏弱）
{json.dumps({
    "hook_strength": review.get("hook_strength"),
    "rhythm_risks": review.get("rhythm_risks", []),
    "writer_directives": review.get("writer_directives_for_next_chapter", []),
}, ensure_ascii=False, indent=2)}

## 章节开头（不要改动——逐字复制）
{head}

## 待重写的当前结尾（长度 {len(original_tail)} 字）
{original_tail}

重写本章。逐字复制开头，再用一个长度与原结尾相当（约 {len(original_tail)} 字，误差 20% 以内）、更犀利的结尾替换结尾段。只输出完整章节。"""
    raw = call_llm(
        client, paths, config, HOOK_REVISE_SYSTEM, user,
        max_tokens=8000, temperature=0.55,
        cacheable_prefix=cacheable_prefix(paths, config),
        tag="revise_hook",
    )
    new_chapter = normalize_chapter(raw)
    # Safety: if the model failed to preserve the head (e.g., truncated or
    # rewrote opening), fall back to head + new tail by splicing.
    if not new_chapter.startswith(head[: min(len(head), 200)]):
        # Try to recover by extracting the model's "new tail" — assume it's
        # the last paragraph in its output.
        from config import log as _log
        _log(paths, "hook revise: head verification failed; splicing head + model_tail")
        model_tail = new_chapter.rsplit("\n\n", 1)[-1] if "\n\n" in new_chapter else new_chapter[-tail_to_revise_chars * 2:]
        new_chapter = normalize_chapter(head.rstrip() + "\n\n" + model_tail.strip())
    return new_chapter


# ---------------------------------------------------------------------------
# Layer 2: Targeted em-dash sentence rewrite.
# Instead of asking the LLM to rewrite the whole chapter (which causes size
# explosion and the model ignores the em-dash ban anyway), extract only the
# sentences containing em-dashes, send them as a numbered list in a single
# tightly-scoped LLM call, then splice rewrites back.
# ---------------------------------------------------------------------------

_EM_DASH_REWRITE_SYSTEM = (
    "你是中文文本编辑器。只做一件事：把每条带有破折号（——）的句子改写为不含破折号的等长句子。\n"
    "改写时用逗号、句号、分号或完整从句替代破折号，保持原文语义、人称视角和叙事腔调完全不变。\n"
    "每条输出的长度与输入长度差距不超过20%。\n"
    "严格按原编号逐条输出，不要输出其他任何内容。格式：\n"
    "1. 改写后的句子\n"
    "2. 改写后的句子\n"
    "......"
)


def reduce_em_dashes_targeted(
    client: "OpenAI",
    paths: "Paths",
    config: dict[str, Any],
    chapter: str,
    max_sentences: int | None = None,
) -> str:
    """Extract sentences with em-dashes, rewrite them via a focused LLM call, splice back."""
    from config import log as _log

    if not chapter or "——" not in chapter:
        return chapter

    max_s = max_sentences or int(config["novel"].get("em_dash_targeted_rewrite_max_sentences", 30))

    # 1. Extract sentences containing ——, with their line context.
    em_sentences: list[tuple[str, int]] = []  # (sentence, line_index)
    lines = chapter.split("\n")
    for li, line in enumerate(lines):
        if "——" not in line:
            continue
        # Split line into rough sentence units at CJK sentence enders.
        parts = re.split(r"(?<=[。！？…])", line)
        for part in parts:
            part = part.strip()
            if "——" in part and len(part) >= 4:
                em_sentences.append((part, li))
                if len(em_sentences) >= max_s:
                    break
        if len(em_sentences) >= max_s:
            break

    if not em_sentences:
        return chapter

    # 2. Build numbered list for the LLM.
    numbered = "\n".join(f"{i+1}. {s}" for i, (s, _) in enumerate(em_sentences))
    user = f"改写以下{len(em_sentences)}条句子，去掉所有破折号（——）：\n\n{numbered}"

    try:
        raw = call_llm(
            client, paths, config, _EM_DASH_REWRITE_SYSTEM, user,
            max_tokens=4000, temperature=0.3, tag="em_dash_fix",
        )
    except Exception as exc:
        _log(paths, f"Targeted em-dash rewrite LLM call failed: {exc}")
        return chapter

    # 3. Parse numbered output.
    rewrites: dict[int, str] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        m = re.match(r"(\d+)\.\s*(.+)", line)
        if m:
            idx = int(m.group(1)) - 1
            rewritten = m.group(2).strip()
            if 0 <= idx < len(em_sentences) and "——" not in rewritten:
                orig = em_sentences[idx][0]
                # Length guard: reject if rewrite diverges too much.
                if 0.5 * len(orig) <= len(rewritten) <= 2.0 * len(orig):
                    rewrites[idx] = rewritten

    if not rewrites:
        _log(paths, "Targeted em-dash rewrite: no usable rewrites parsed")
        return chapter

    # 4. Splice back via exact string replacement.
    result = chapter
    applied = 0
    for idx, rewritten in rewrites.items():
        orig_sentence = em_sentences[idx][0]
        if orig_sentence in result:
            result = result.replace(orig_sentence, rewritten, 1)
            applied += 1

    # 5. Size guard.
    if len(result) > len(chapter) * 1.3 or len(result) < len(chapter) * 0.7:
        _log(paths, f"Targeted em-dash rewrite rejected: size {len(chapter)}->{len(result)}")
        return chapter

    _log(paths, f"Targeted em-dash rewrite: applied {applied}/{len(rewrites)} rewrites (of {len(em_sentences)} sentences)")
    return result


def _chapter_write_max_tokens(config: dict[str, Any]) -> int | None:
    """Generation-time length cap for chapter writing.

    The write call otherwise inherits the global api.max_tokens (often 64k+), so
    a chapter can balloon far past the target band. This bounds the writer's
    output by chapter_max_chars (which the genre profile sets per题材), sized with
    enough headroom that a complete in-band chapter is never truncated
    mid-sentence — it kills runaway over-length without cutting normal chapters.
    Returns None (no cap → global default) when disabled.

    Tune with `write_token_char_ratio` (lower = tighter, but risks truncation)
    and `write_token_margin`; or pin an absolute `write_max_tokens`.
    """
    nv = config.get("novel", {})
    if not bool(nv.get("chapter_length_cap_enabled", True)):
        return None
    try:
        explicit = int(nv.get("write_max_tokens", 0) or 0)
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return explicit
    try:
        cmax = int(nv.get("chapter_max_chars", 3600))
    except (TypeError, ValueError):
        cmax = 3600
    ratio = float(nv.get("write_token_char_ratio", 1.15))
    margin = int(nv.get("write_token_margin", 300))
    return max(int(cmax * ratio) + margin, 1200)


def write_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
    temperature: float | None = None,
    chapter_aux_cache: dict | None = None,
) -> str:
    title = str(plan.get("title") or f"Chapter {chapter_num}").strip()
    # Strip any leading 第N章 prefix the planner put in the title — the write
    # prompt formats the first line as "第{n}章 {title}", so a title that already
    # starts with "第N章：" doubles it ("第2章 第2章：剪辑师的盲区").
    title = re.sub(
        r"^\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*章\s*[:：、\-—\s]*", "", title
    ).strip() or f"Chapter {chapter_num}"
    preset = str(config["novel"].get("style_preset", "history"))
    system = _build_write_system(
        preset,
        chapter_words=int(config["novel"]["chapter_words"]),
        chapter_num=chapter_num,
        title=title,
        aesthetic=AESTHETIC_PRESETS.get(preset, AESTHETIC_HISTORY),
    )
    # Sensitive-word avoidance goes into the SYSTEM prompt (highest attention) so it
    # is not diluted inside the large user prompt. Content-moderation gateways scan
    # the streamed OUTPUT; a single flagged token in a full chapter → 500. Keeping
    # this front-and-center in the system role is the strongest steering position.
    try:
        _swa_sys = sensitive_word_avoidance_block(config)
        if _swa_sys:
            system = system + "\n\n" + _swa_sys
    except Exception:
        pass
    mem = cached_memory or writing_memory_context(paths, conn, config)
    partial_beats = carried_over_partial_beats(paths, chapter_num)
    directives = writer_directives_for_chapter(paths, chapter_num)
    carryover_block = ""
    # Chapter length band reminder (soft): keep chapters inside a sane字数区间.
    try:
        _cw = int(config["novel"].get("chapter_words", 4000) or 4000)
        _cmin = int(config["novel"].get("chapter_min_chars", 2800))
        _cmax = int(config["novel"].get("chapter_max_chars", 7000))
        carryover_block += (
            f"\n## 本章字数区间（硬性约束）\n"
            f"目标约 {_cw} 字，必须落在 {_cmin}-{_cmax} 字区间内。"
            f"番茄是短章高频钩子（每章一个钩子、情绪高峰间隔≤2章），不要写成长章文学体——"
            f"超出上限 {_cmax} 字会被判超长扣分，请精简技术性/描写性堆砌、聚焦推进剧情与爽点。\n"
        )
    except Exception:
        pass
    # 下沉/大白话语体（正交开关，叠加任意题材）：免费流 platform_preset 或显式
    # style_low_barrier_register 触发，主动驱动大白话短句对话体（与 style_health
    # 的下沉放宽/书面腔惩罚配套）。
    try:
        _plat = str(config["novel"].get("platform_preset", "")).strip().lower()
        _low_barrier = (
            _plat in {"fanqie_free", "qimao_free"}
            or bool(config["novel"].get("style_low_barrier_register", False))
        )
        if _low_barrier:
            carryover_block += "\n" + LOW_BARRIER_REGISTER_BLOCK + "\n"
    except Exception:
        pass
    # P0-1: Pre-flight negative list (gate_rejects + style collapse flags + fossils)
    preflight_neg = None
    if bool(config["novel"].get("preflight_constraints_enabled", True)):
        preflight_neg = _preflight_negative_list(
            paths, conn, config, chapter_num,
            lookback=int(config["novel"].get("preflight_constraints_lookback", 5)),
        )
    if preflight_neg and (preflight_neg["items"] or preflight_neg["fossils"] or preflight_neg["style_warnings"]):
        carryover_block += "\n## 本章绝对禁止（前置负面清单·来自近期质量门禁）\n"
        carryover_block += "以下失败模式已在前几章触发质量门禁拒收。本章动笔前必须规避：\n"
        if preflight_neg["items"]:
            for item in preflight_neg["items"]:
                carryover_block += f"- {item}\n"
        if preflight_neg["fossils"]:
            fossil_count = len(preflight_neg["fossils"])
            carryover_block += "\n**已检测到的化石句（严禁复现原句或结构相似表达）：**\n"
            # Escalate warning when fossil accumulation is severe
            if fossil_count >= 8:
                carryover_block += (
                    f"⚠️ **严重警告**：近 {int(config['novel'].get('preflight_constraints_lookback', 5))} 章累积 {fossil_count} 处化石句，"
                    f"已达崩坏临界。本章必须从【全新】视角/场所/对话方式切入，"
                    f"严禁复刻下列签名句式或其结构变体。宁可另起炉灶、改换叙述腔调，也不许在旧轨道上微调措辞。\n"
                )
            elif fossil_count >= 4:
                carryover_block += (
                    f"⚠️ 近章检测到 {fossil_count} 处化石句复读。本章务必避开下列签名表达及其结构相似句式：\n"
                )
            for fossil in preflight_neg["fossils"]:
                carryover_block += f"  • 「{fossil}」\n"
        if preflight_neg["style_warnings"]:
            carryover_block += "\n**近期风格问题：**\n"
            for warn in preflight_neg["style_warnings"]:
                carryover_block += f"  • {warn}\n"
        carryover_block += "\n"
    prewrite_contract = _prewrite_quality_contract(paths, config, chapter_num, plan, decision)
    if prewrite_contract:
        carryover_block += "\n" + prewrite_contract + "\n"
    if partial_beats:
        carryover_block += (
            f"\n## 来自 CH{chapter_num - 1} 的关键遗留（必须在页面上处理）\n"
            f"以下节拍在上一章审校中被标记为 partial/absent。"
            f"当叙事上可行时，你必须在本章把它们落到页面上实演，"
            f"不要让其停留在暗示或页面之外。\n"
            f"{json.dumps(partial_beats, ensure_ascii=False, indent=2)}\n"
        )
    if directives:
        carryover_block += (
            f"\n## 给 CH{chapter_num} 的审校指令（必须遵守）\n"
            f"以下执行级指令来自上一章的审校者。"
            f"当与通用准则冲突时，以这些指令为准。\n"
            f"{json.dumps(directives, ensure_ascii=False, indent=2)}\n"
        )
    # Recovery mode: a mid-book degradation alert is in force. Inject the highest
    # priority "pull out of the slide" directive ahead of everything else.
    try:
        rec = paths.logs_dir / "recovery_directive.json"
        if rec.exists():
            rdata = json.loads(read_text(rec))
            if chapter_num <= int(rdata.get("active_until", 0)) and rdata.get("directive"):
                carryover_block += (
                    f"\n## ⚠️ 质量恢复模式（最高优先级，触发原因：{rdata.get('reason','')}）\n"
                    f"{rdata['directive']}\n"
                )
    except Exception:
        pass
    # Relationship guidance: inject current relationship state and this chapter's
    # relationship_beats from the plan so the writer advances relationships visibly.
    try:
        rel_beats = plan.get("relationship_beats") or []
        if rel_beats:
            carryover_block += (
                f"\n## 本章角色关系推进目标（必须在页面上体现）\n"
                f"以下关系对必须在本章有可见的推进——通过对话、行为、冲突或态度变化，"
                f"不要用旁白概述，要用具体场景让读者感受到关系的变化。\n"
                f"{json.dumps(rel_beats, ensure_ascii=False, indent=2)}\n"
            )
        from store import get_relationships
        rels = get_relationships(conn, limit=8)
        if rels:
            rel_summary = [
                f"{r['char_a']}↔{r['char_b']}: {r['stage']}(强度{r['intensity']:.1f}) — {r.get('last_event', '')}"
                for r in rels
            ]
            carryover_block += (
                f"\n## 当前角色关系状态（写对话/互动时参考——不要让关系退行到更早阶段）\n"
                + "\n".join(f"- {s}" for s in rel_summary) + "\n"
            )
    except Exception:
        pass
    # Emotional impact guidance: based on plan and emotional cadence
    try:
        from quality import emotional_cadence as _ec
        from store import recent_metrics as _rm
        tone_rows = _rm(conn, 6)
        tones = [str(r.get("emotional_tone", "")) for r in reversed(tone_rows) if r.get("emotional_tone")]
        ec = _ec(tones, config)
        if ec.get("directives"):
            carryover_block += "\n## 情感节奏指导\n"
            for d in ec["directives"]:
                carryover_block += f"- {d}\n"
        carryover_block += (
            "\n## 情感冲击力要求\n"
            "本章必须包含至少一个让读者产生真实情感反应的时刻。写法要求：\n"
            "- 用具体的行为、选择、牺牲来挣得情感，而非形容词堆砌（\"心痛\"\"震惊\"）\n"
            "- 情感高点用短句+留白，让读者自己补全感受，不要替读者下结论\n"
            "- 生理反应描写（屏息、握拳、喉结滚动）只辅助情感，不能替代情感本身\n"
        )
    except Exception:
        pass
    # Used-element avoid-list (P0 anti-collapse): the writer otherwise re-invents
    # the same concrete action/evidence every chapter ("祝寒的右手在口袋里握了握"
    # surfaced as a fossil in 5 chapters of suspense_v11). Surface what prior
    # chapters already leaned on so the writer varies the wording/object rather
    # than producing a near-verbatim repeat that only cross_repeat catches later.
    if bool(config["novel"].get("used_element_ledger_enabled", True)):
        try:
            from planning import used_element_ledger

            led = used_element_ledger(
                conn, config, chapter_num,
                lookback=int(config["novel"].get("scene_dedupe_window", 8)),
            )
            if led.get("device_usage") or led.get("evidence"):
                carryover_block += (
                    "\n## 前文已反复使用的金手指用法 / 物证（避免复读化石句）\n"
                    "以下能力使用方式与物证在前几章已反复出现。本章除非剧情确需追踪同一物件，"
                    "否则不要再用相同的措辞重演同一个动作或围绕同一物证重复同一结论——"
                    "换一个具体物证、换一种能力使用的写法、或推进到新的信息增量。\n"
                    f"{json.dumps(led, ensure_ascii=False)}\n"
                )
        except Exception as exc:
            from config import log as _log
            _log(paths, f"used_element_ledger (writer) failed (non-fatal) Ch{chapter_num}: {exc}")
    # Glossary / proper-noun consistency layer: surface the canonical names &
    # terms so the writer keeps surface forms stable across chapters. Variable
    # section only — never folded into cacheable_prefix sources.
    if bool(config["novel"].get("glossary_enabled", True)):
        try:
            from review import glossary_block as _glossary_block
            gb = _glossary_block(paths, config)
            if gb:
                carryover_block += "\n" + gb + "\n"
        except Exception as exc:
            from config import log as _log
            _log(paths, f"Glossary block build failed (non-fatal) Ch{chapter_num}: {exc}")
    # Retrieval-augmented context: surface specific older facts the layered
    # memory summaries have compressed away, so long-range names/numbers/places
    # stay consistent.
    rag_block = ""
    if bool(config["novel"].get("rag_enabled", True)):
        try:
            from retrieval import retrieval_block

            rag_block = retrieval_block(paths, config, plan, chapter_num)
        except Exception as exc:
            from config import log as _log
            _log(paths, f"RAG block build failed (non-fatal) Ch{chapter_num}: {exc}")
    if rag_block:
        carryover_block += "\n" + rag_block + "\n"
    # P0-3: Golden exemplar RAG (top-scoring chapters matching plan type)
    exemplar_block_text = ""
    if bool(config["novel"].get("exemplar_rag_enabled", True)):
        try:
            from retrieval import exemplar_block

            exemplar_block_text = exemplar_block(paths, conn, config, plan, chapter_num)
        except Exception as exc:
            from config import log as _log
            _log(paths, f"Exemplar RAG block build failed (non-fatal) Ch{chapter_num}: {exc}")
    if exemplar_block_text:
        carryover_block += "\n" + exemplar_block_text + "\n"
    if bool(config["novel"].get("structured_recall_enabled", True)):
        try:
            from retrieval import structured_recall_block
            sr_block = structured_recall_block(conn, config, plan, chapter_num)
            if sr_block:
                carryover_block += "\n" + sr_block + "\n"
        except Exception as exc:
            from config import log as _log
            _log(paths, f"Structured recall failed (non-fatal) Ch{chapter_num}: {exc}")
    try:
        from benchmark import benchmark_context, platform_guidance

        carryover_block += "\n## 平台/读者画像\n" + platform_guidance(config) + "\n"
        bm = benchmark_context(paths, config, json.dumps(plan, ensure_ascii=False) + "\n" + tail, max_chars=4000)
        if bm:
            carryover_block += "\n" + bm + "\n"
    except Exception:
        pass
    opening_chapters = int(config["novel"].get("opening_chapters", 3))
    if chapter_num <= opening_chapters:
        carryover_block += "\n" + OPENING_RULES_BLOCK + "\n"
        # Platform-tuned opening rules (黄金三章 differ by platform: fanqie wants
        # 强冲突 in the first lines, qidian tolerates more setup, etc.).
        try:
            from benchmark import platform_opening
            po = platform_opening(config)
            if po:
                carryover_block += "\n" + po + "\n"
        except Exception:
            pass
        # 吸量包落地（P3 断链修复）：把 hook_directives（书名/简介承诺的爽点）
        # 注入开篇写手 prompt。缺失/畸形静默跳过（镜像 craft_rules 的 no-op 模式）；
        # per-chapter 块，不影响 cacheable_prefix。
        if bool(config["novel"].get("hook_directives_inject_enabled", True)):
            try:
                hp_path = paths.logs_dir / "hook_package.json"
                if hp_path.exists():
                    hp = json.loads(hp_path.read_text(encoding="utf-8"))
                    hd_block = _hook_directives_block(hp)
                    if hd_block:
                        carryover_block += "\n" + hd_block + "\n"
            except Exception:
                pass
    # Character signature traits: nudge the writer to surface at least one
    # character's 人设记忆点 (already shipped via the cacheable character profile)
    # so characters stay distinctive across chapters. One sentence, no content
    # duplication -> zero cache impact.
    if bool(config["novel"].get("signature_trait_surface", True)):
        carryover_block += (
            "\n## 人设记忆点（追读留存）\n"
            "本章请让至少 1 个出场的主要人物自然展现其【人设记忆点】（见上方人物档案中的"
            "**人设记忆点**子条目：口头禅/标志动作/独特习惯/外号/反差），通过动作或对白落地，"
            "避免人物脸谱化、千人一面；但须服务剧情，不得生硬堆砌。\n"
        )
    from config import is_final_chapter, ending_zone_distance
    if is_final_chapter(config, chapter_num):
        carryover_block += "\n" + CLOSING_RULES_BLOCK + "\n"
    elif ending_zone_distance(config, chapter_num) is not None:
        remaining = ending_zone_distance(config, chapter_num)
        carryover_block += "\n" + CLOSING_APPROACH_BLOCK.format(remaining=remaining) + "\n"
    else:
        # Narrative-mode steering (single-room reasoning vs strong-hook serial).
        # Skipped for the finale, whose closure rules take precedence.
        from config import narrative_mode
        mode = narrative_mode(config)
        if mode == "reasoning":
            carryover_block += "\n" + MODE_REASONING_BLOCK + "\n"
        elif mode == "serial":
            carryover_block += "\n" + MODE_SERIAL_BLOCK + "\n"
    try:
        fp_path = paths.memory_dir / "dialogue_fingerprints.json"
        if fp_path.exists():
            all_fps = json.loads(fp_path.read_text(encoding="utf-8"))
            focus_names: list[str] = []
            for key in ("focus_characters", "character_focus"):
                raw = plan.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str):
                            focus_names.append(item.strip())
                        elif isinstance(item, dict):
                            n = str(item.get("name") or item.get("角色") or "").strip()
                            if n:
                                focus_names.append(n)
                elif isinstance(raw, str) and raw.strip():
                    focus_names.extend(seg.strip() for seg in raw.split("、") if seg.strip())
            fp_lines: list[str] = []
            for name in dict.fromkeys(focus_names):
                if name in all_fps and all_fps[name].get("style"):
                    fp_lines.append(f"- {name}：{all_fps[name]['style']}")
            if fp_lines:
                carryover_block += (
                    "\n## 角色对话指纹（必须维持的说话风格差异）\n"
                    + "\n".join(fp_lines[:6]) + "\n"
                )
    except Exception:
        pass
    user = f"""## 记忆（事实与设定参照，不要模仿其行文风格）
{mem}
{carryover_block}
## 上章结尾
{tail[-int(config["novel"]["recent_tail_chars"]):]}

## 近期质量反馈JSON（本章必须修复；仅作修复目标，不要模仿其文风或照抄措辞）
{json.dumps((chapter_aux_cache or {}).get("recent_quality_feedback") or recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 选定大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 仲裁约束JSON
{json.dumps(decision.get("required_constraints", []), ensure_ascii=False, indent=2)}

写第 {chapter_num} 章。"""
    # Recency anchor #1: the beat acceptance checklist. Beats sit mid-prompt inside
    # the plan JSON where long context dilutes them (v13 Ch10 shipped with its core
    # payoff beat entirely absent from the prose). Re-state every beat as the
    # last-read acceptance checklist, with the concrete details each beat promises
    # named as non-negotiable items. This replaces the duplicated mid-prompt beat
    # enumerations that used to live in _prewrite_quality_contract /
    # _first_draft_execution_ledger.
    beat_list = [str(b).strip() for b in (plan.get("beats") or []) if str(b).strip()] if isinstance(plan, dict) else []
    if beat_list:
        checklist_lines = []
        for i, b in enumerate(beat_list[:9], 1):
            details = _beat_concrete_details(b)
            anchor = ("｜必须实演的具体物件/动作：" + "、".join(details[:4])) if details else ""
            risk = "｜含抽象实现词，必须落成可见动作+对手反应" if _beat_needs_concretization(b) else ""
            checklist_lines.append(f"{i}. {b[:150]}{anchor}{risk}")
        user += (
            "\n\n## ⚠ 交稿前逐条核对：本章 beat 验收清单（每条都必须在正文中实演为可见动作/对话/后果，缺一条即作废）\n"
            "每个 beat 写明的具体动作、物件、数字、动机都是硬验收项；严禁用结果或声音替代动作本身，"
            "严禁用总结句带过，严禁与大纲的具体描述矛盾。\n"
            + "\n".join(checklist_lines)
        )
    capsule = contract_capsule(paths, config)
    # 副本/场景入口显著性：when this chapter enters a NEW location (deterministic
    # location_transition vs recent plans), the writer is overloaded with new-setting
    # setup and reliably drops the numbered-rules discipline + degrades into
    # telegraphic summary — the yeban_guize Ch8/Ch9 collapse (2.7/4.0). Inject a
    # high-salience establishment block near the prompt tail. Genre-neutral (fires on
    # any real location change); the numbered-rules clause is added only when the
    # book's own contract asks for rule-listing (so 规则怪谈/无限流 books benefit and
    # others just get the "establish the new scene, no telegraphic summary" nudge).
    if bool(config["novel"].get("scene_entry_salience_enabled", True)) and chapter_num > 1:
        try:
            from config import log
            from quality import location_transition
            from planning import _recent_selected_plans
            _recent = _recent_selected_plans(
                conn,
                lookback=int(config["novel"].get("scene_entry_lookback", 3)),
                exclude_chapter=chapter_num,
            )
            _lt = location_transition(plan, _recent, config)
            if _lt.get("is_new"):
                _loc = _lt.get("location") or "新场景"
                _rules_kw = ("编号", "守则", "清单", "逐条")
                _wants_rules = bool(capsule) and any(k in capsule for k in _rules_kw)
                _rule_line = (
                    "本副本的明面守则必须在本章早段用编号清单（第一条…第N条）逐条列全、清楚可读，"
                    "像开篇第1章那样；主角能额外读到的隐藏规则也以编号条目呈现，不得含糊带过。"
                    if _wants_rules
                    else "把新场景的关键设定、危险与目标交代清楚，让读者一进来就能跟上。"
                )
                user += (
                    f"\n\n## ⚠ 本章进入全新场景/副本【{_loc}】——开新副本铁律（最高优先级）\n"
                    "读者对这个新地点的规则/空间/人物一无所知，这是最容易写崩的一类章节。务必：\n"
                    "1. 先把新场景【完整立起来】：可看见的空间、在场的人、时段与氛围，用成句的场景描写落地——"
                    "严禁电报体、碎片式短句、破折号堆叠或一句话带过设定。\n"
                    f"2. {_rule_line}\n"
                    "3. 新副本的核心冲突要在本章内自成起承，不要依赖读者已知的旧副本设定来省略交代。\n"
                    f"（检测：本章 location=「{_loc}」与近 {len(_recent)} 章场景相似度仅 {_lt.get('max_sim')}，判定为换副本入口。）"
                )
                log(paths, f"Scene-entry salience Ch{chapter_num}: new location=「{_loc}」 max_sim={_lt.get('max_sim')} rules={_wants_rules}")
        except Exception as exc:
            log(paths, f"scene-entry salience failed (non-fatal) Ch{chapter_num}: {exc}")
    # Recency anchor #2: the full contract sits high in the prompt where long context
    # dilutes it (v4 breached the ability whitelist/modality in 5/6 chapters). Re-
    # state ONLY the hard ability boundaries as the very last thing the writer
    # reads, where attention is strongest. Appended AFTER "写第N章" so it is the
    # final instruction in the user message.
    if capsule:
        user += (
            "\n\n## ⚠ 写作前最后确认：能力边界（最高优先级，违反即作废重写）\n"
            "动笔前再次确认——主角与关键人物**只能**使用下列白名单能力、且严格限定在其标注的**模态**内推进剧情，"
            "绝不能借助白名单之外的感官/能力（例如把「单帧静止视像」写成能听到声音、看到连续画面或获得全知推断）：\n"
            f"{capsule}"
        )
    temp = float(config["api"]["temperature"]) if temperature is None else temperature
    prefix = cacheable_prefix(paths, config)
    from config import log
    log(paths, f"write_chapter Ch{chapter_num} calling LLM with temp={temp:.2f} user_len={len(user)} system_len={len(system)}")
    raw = call_llm(client, paths, config, system, user, temperature=temp, cacheable_prefix=prefix,
                   max_tokens=_chapter_write_max_tokens(config), tag="write")
    log(paths, f"write_chapter Ch{chapter_num} LLM returned {len(raw)} chars")
    return normalize_chapter(raw)


BEAT_REPAIR_SYSTEM = """你是一位中文网文定向补写专家。
给你一份章节草稿和一份「缺失节拍清单」——大纲承诺、但正文没有实演出来的节拍。
任务：把每个缺失节拍编织进正文最合适的场景，用可见动作、对话交锋或具体后果把它实演出来。

约束：
- 输出完整章节正文。除被补写/衔接的段落外，其余内容逐字保留原样，不要重写无关段落。
- 每个缺失节拍必须落成页面上的具体动作/物件/对话；节拍里点名的具体细节（谁的手做了什么、什么物件、什么数字）必须原样出现。严禁用结果或声音替代动作本身，严禁用一句总结带过。
- 补写要顺着上下文因果自然嵌入，不得与已有正文矛盾，不得引入大纲之外的新设定或新人物。
- 维持原有叙事声音与节奏；补写后总长度增幅控制在 25% 以内。
- 只输出章节正文，不要解释、不要输出清单。"""


def repair_missing_beats(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    chapter: str,
    coverage: dict[str, Any],
) -> str:
    """One targeted low-temperature repair call that weaves missing plan beats
    into an existing draft.

    The deterministic gate `quality.beat_coverage` found beats the draft never
    acted out (v13 Ch10 shipped with its core payoff beat entirely absent and the
    LLM reviewer only caught it post-hoc at -1.0 per beat). Instead of burning a
    full revision round later, spend ONE surgical call now that targets exactly
    the missing beats. Returns the repaired text, or the original draft unchanged
    when the repair fails sanity checks (too short, shrank, or ballooned).
    """
    missing = coverage.get("missing_beats") or []
    if not missing or not chapter:
        return chapter
    # Per-beat anchor details (the concrete fragments the gate could not find).
    beat_missing_anchors: dict[str, list[str]] = {}
    for entry in coverage.get("beats") or []:
        if isinstance(entry, dict) and not entry.get("hit"):
            beat_missing_anchors[str(entry.get("beat", ""))] = [
                str(a) for a in (entry.get("missing") or []) if str(a).strip()
            ]
    items: list[str] = []
    for mb in missing[:6]:
        if isinstance(mb, dict):
            beat = str(mb.get("beat", "")).strip()
            anchors = [str(a) for a in (mb.get("missing") or []) if str(a).strip()]
        else:
            beat = str(mb).strip()
            anchors = []
        if not beat:
            continue
        if not anchors:
            for key, vals in beat_missing_anchors.items():
                if key and (key in beat or beat[:120] in key or key[:120] in beat):
                    anchors = vals
                    break
        anchor_note = ("（正文必须出现的具体细节：" + "、".join(anchors[:4]) + "）") if anchors else ""
        items.append(f"- {beat[:200]}{anchor_note}")
    if not items:
        return chapter
    user = f"""## 选定大纲JSON（提供节拍语境，不要改变剧情走向）
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 缺失节拍清单（每条都必须补演进正文对应场景）
{chr(10).join(items)}

## 章节草稿（除补写处外逐字保留）
{chapter}

把缺失节拍补演进对应场景，输出完整章节正文。"""
    base_temp = float(config["api"]["temperature"])
    temp = min(0.4, base_temp)
    prefix = cacheable_prefix(paths, config)
    from config import log
    log(paths, f"beat repair Ch{chapter_num}: weaving {len(items)} missing beat(s), coverage={coverage.get('coverage')}")
    try:
        raw = call_llm(
            client, paths, config, BEAT_REPAIR_SYSTEM, user,
            temperature=temp, cacheable_prefix=prefix,
            max_tokens=_chapter_write_max_tokens(config), tag="beat_repair",
        )
    except Exception as exc:
        log(paths, f"beat repair LLM call failed Ch{chapter_num} (non-fatal): {exc}")
        return chapter
    repaired = normalize_chapter(raw or "")
    if not repaired or len(repaired.strip()) < 500:
        return chapter
    if len(repaired) < len(chapter) * 0.7 or len(repaired) > len(chapter) * 1.6:
        log(paths, f"beat repair Ch{chapter_num} rejected: length {len(chapter)} -> {len(repaired)}")
        return chapter
    return repaired


def apply_review_patches(chapter: str, patches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply review-provided patches to chapter text in-place.

    Returns (new_chapter, applied_patches_with_status). Each patch entry gets an
    "applied" boolean and optionally an "error" reason if it could not be applied.
    Patches are applied in input order, each operating on the current text.
    Locators that no longer match (because an earlier patch removed/replaced the
    surrounding region) are skipped with applied=False.
    """
    text = chapter
    results: list[dict[str, Any]] = []
    for raw_patch in patches or []:
        if not isinstance(raw_patch, dict):
            results.append({"applied": False, "error": "non-dict patch", "patch": raw_patch})
            continue
        op = str(raw_patch.get("op", "")).strip().lower()
        locator = str(raw_patch.get("locator", "")).strip()
        before = str(raw_patch.get("before", "") or "").strip()
        after = str(raw_patch.get("after", "") or "")
        insert_text = str(raw_patch.get("insert", "") or "")
        entry = {**raw_patch, "applied": False}
        try:
            if op == "replace":
                target = before or locator
                if not target:
                    entry["error"] = "empty before/locator for replace"
                elif target not in text:
                    entry["error"] = "before/locator not found in chapter"
                else:
                    text = text.replace(target, after, 1)
                    entry["applied"] = True
            elif op == "insert_after":
                if not locator or locator not in text:
                    entry["error"] = "locator not found for insert_after"
                else:
                    idx = text.find(locator) + len(locator)
                    glue_before = "" if text[idx:idx+1] in {"\n", ""} else "\n\n"
                    glue_after = "" if text[idx:idx+2] == "\n\n" else "\n\n"
                    text = text[:idx] + glue_before + insert_text + glue_after + text[idx:]
                    entry["applied"] = True
            elif op == "delete":
                target = before or locator
                if not target or target not in text:
                    entry["error"] = "before/locator not found for delete"
                else:
                    text = text.replace(target, "", 1)
                    entry["applied"] = True
            else:
                entry["error"] = f"unknown op: {op!r}"
        except Exception as exc:
            entry["error"] = f"exception: {exc}"
        results.append(entry)
    return text, results


def revise_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter: str,
    review: dict[str, Any],
    plan: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
    chapter_aux_cache: dict | None = None,
) -> str:
    # Fast path: try applying review patches directly without a full LLM rewrite.
    # Only fall back to LLM revision when patches are missing, incomplete, or fail.
    patches = review.get("patches") if isinstance(review, dict) else None
    use_patch_path = bool(config["novel"].get("revise_use_patches", True))
    # Layer 1: style collapse needs a full LLM rewrite — content patches can't fix prose habits.
    sh = review.get("style_health") if isinstance(review, dict) else None
    sh_penalty = float((sh or {}).get("penalty", 0))
    style_block = float(config["novel"].get("style_penalty_block", 2.0))
    if use_patch_path and sh_penalty >= style_block:
        from config import log as _log
        _log(paths, f"Revise: skipping patch path — style_health penalty={sh_penalty:.1f} >= block={style_block}")
        use_patch_path = False
    if use_patch_path and isinstance(patches, list) and patches:
        patched, results = apply_review_patches(chapter, patches)
        applied = sum(1 for r in results if r.get("applied"))
        total = len(results)
        # Relaxed threshold: 1/2 of patches applied counts as success.
        # Surgical patch path is much faster than a full rewrite and the unapplied
        # patches typically address minor issues; the next review round will pick
        # up anything material that remains.
        min_apply_frac = float(config["novel"].get("revise_patch_min_frac", 0.5))
        threshold_hit = max(1, int(total * min_apply_frac + 0.999))
        if applied >= threshold_hit:
            from config import log as _log
            _log(paths, f"Revise via patches applied={applied}/{total} (>= {threshold_hit}); skipping full rewrite")
            patched_ch = normalize_chapter(patched)
            # Even on patch fast-path, apply em-dash remediation if flagged.
            _sh_flags_p = (sh or {}).get("flags", []) if sh else []
            if any("em_dash" in f for f in _sh_flags_p):
                if bool(config["novel"].get("em_dash_targeted_rewrite_enabled", True)):
                    patched_ch = reduce_em_dashes_targeted(client, paths, config, patched_ch)
                from quality import style_health as _sh_chk
                _sh_p = _sh_chk(patched_ch, config)
                _em_p = float(_sh_p.get("metrics", {}).get("em_dash_per_kchar", 0))
                _em_t = float(config["novel"].get("em_dash_reduce_target_per_kchar", 3.0))
                if _em_p > _em_t and bool(config["novel"].get("em_dash_reduce_enabled", True)):
                    from quality import reduce_em_dash_density
                    patched_ch = reduce_em_dash_density(patched_ch, config)
            return patched_ch
        else:
            from config import log as _log
            _log(paths, f"Revise patches too few hit ({applied}/{total} < {threshold_hit}); falling back to LLM rewrite")

    mem = cached_memory or writing_memory_context(paths, conn, config)
    _swa_revise = ""
    try:
        _swa_revise = sensitive_word_avoidance_block(config)
    except Exception:
        _swa_revise = ""
    user = f"""## 记忆
{mem}
{(_swa_revise + chr(10)) if _swa_revise else ""}
## 上章结尾
{tail[-1500:]}

## 大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 近期质量反馈JSON
{json.dumps((chapter_aux_cache or {}).get("recent_quality_feedback") or recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 编辑报告JSON
{json.dumps(review, ensure_ascii=False, indent=2)}

## 原始章节
{chapter}

修订整章。"""
    raw = call_llm(
        client, paths, config, REVISE_SYSTEM, user,
        temperature=0.45, cacheable_prefix=cacheable_prefix(paths, config),
        max_tokens=_chapter_write_max_tokens(config), tag="revise",
    )
    revised = normalize_chapter(raw)
    # Layer 4: reject revisions that explode in size (observed: 4.3k→9.4k→15.6k in Ch43).
    max_grow = float(config["novel"].get("revise_max_grow_ratio", 1.5))
    if chapter and len(revised) > len(chapter) * max_grow:
        from config import log as _log
        _log(paths, f"Revise rejected: size grew {len(chapter)}->{len(revised)} ({len(revised)/len(chapter):.1f}x > {max_grow}x)")
        revised = chapter
    # Layer 5: em-dash remediation pipeline (only when style_health flagged em-dash collapse).
    _sh_flags = (sh or {}).get("flags", []) if sh else []
    if any("em_dash" in f for f in _sh_flags):
        # Layer 2: targeted sentence-level rewrite.
        if bool(config["novel"].get("em_dash_targeted_rewrite_enabled", True)):
            revised = reduce_em_dashes_targeted(client, paths, config, revised)
        # Layer 3: programmatic fallback if density still above target.
        from quality import style_health as _sh_check
        sh_after = _sh_check(revised, config)
        em_after = float(sh_after.get("metrics", {}).get("em_dash_per_kchar", 0))
        em_target = float(config["novel"].get("em_dash_reduce_target_per_kchar", 3.0))
        if em_after > em_target and bool(config["novel"].get("em_dash_reduce_enabled", True)):
            from quality import reduce_em_dash_density
            before_len = revised.count("——")
            revised = reduce_em_dash_density(revised, config)
            after_len = revised.count("——")
            from config import log as _log
            _log(paths, f"Programmatic em-dash reduction: {before_len}->{after_len} dashes, density {em_after:.1f}->{revised.count('——')/(len(revised)/1000):.1f}/k")
    return revised

def extract_events(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    cached_memory: str | None = None,
) -> dict[str, Any]:
    mem = cached_memory or memory_context(paths, conn, config)
    # 把"当前仍未关闭的线索清单"喂给抽取器，使其复用已有 id 而非每章新造 id，
    # 否则同一条伏笔会被反复登记成几十个不同 id，灌爆规划提示（伏笔台账爆炸）。
    open_threads_block = ""
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT id, description, status, thread_type FROM open_threads
                   WHERE status='open'
                   ORDER BY updated_chapter DESC LIMIT 40""",
            ).fetchall()
        if rows:
            lines = [
                f"- id={r['id']} | {r['thread_type']} | {(r['description'] or '')[:60]}"
                for r in rows
            ]
            open_threads_block = (
                "\n## 当前仍未关闭的线索清单（复用 id，勿新造）\n" + "\n".join(lines) + "\n"
            )
    except Exception:
        open_threads_block = ""
    # Include the chapter ending explicitly: protagonist_state/next_12_directions
    # (merged into this single extraction call) depend on where the chapter LANDS,
    # and a >8000-char chapter would otherwise have its tail truncated away.
    chapter_block = chapter[:8000]
    if len(chapter) > 8000:
        chapter_block += "\n……（中段省略）……\n" + chapter[-2500:]
    prev_state = read_text(paths.state)
    if len(prev_state) > 2000:
        prev_state = prev_state[:2000] + "\n...[truncated]"
    user = f"""## 章节前记忆
{mem}
{open_threads_block}
## 上一版主角状态（用于 protagonist_state 的连贯）
{prev_state}

## 第 {chapter_num} 章
{chapter_block}

抽取持久的状态变化，并给出更新后的 protagonist_state 与 next_12_directions。"""
    raw = call_llm(client, paths, config, EXTRACT_SYSTEM, max_tokens=12000, user=json_prompt(user), temperature=0.2, tag="extract")
    return load_json_with_repair(client, paths, config, raw)

def update_structured_state(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    review: dict[str, Any],
    decision: dict[str, Any],
    plan: dict[str, Any] | None = None,
) -> None:
    db_event(conn, chapter_num, "chapter_extraction", extraction)

    for event in extraction.get("events", []):
        db_event(conn, chapter_num, "story_event", event)

    for entity in extraction.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type", "unknown"))
        name = str(entity.get("name", "unknown"))
        with db_lock():
            old = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
        state = json.loads(old["state_json"]) if old else {}
        patch = entity.get("state_patch") or {}
        if isinstance(patch, dict):
            state.update(patch)
        else:
            state["note"] = str(patch)
        with db_lock():
            conn.execute(
                """
                INSERT INTO entities(entity_type, name, state_json, updated_chapter)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_type, name)
                DO UPDATE SET state_json=excluded.state_json, updated_chapter=excluded.updated_chapter
                """,
                (entity_type, name, json.dumps(state, ensure_ascii=False), chapter_num),
            )

    def _as_chnum(v: Any) -> int | None:
        # chapter-number columns must bind as int/None; LLM may emit a dict/list/str.
        if isinstance(v, bool) or v is None:
            return None
        if isinstance(v, int):
            return v
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return None

    for thread in extraction.get("threads", []):
        if not isinstance(thread, dict):
            continue
        thread_id = str(thread.get("id") or f"ch{chapter_num}-{abs(hash(json.dumps(thread, ensure_ascii=False))) % 100000}")
        if str(thread.get("thread_type", "plot")) == "reader_promise":
            promise = dict(thread)
            promise["id"] = thread_id
            promise.setdefault("opened_chapter", thread.get("introduced_chapter", chapter_num))
            upsert_reader_promise(conn, chapter_num, promise)
        _payload = thread.get("payload")
        _depends = str(thread.get("depends_on", "") or "").strip()
        _priority = int(thread.get("priority", 5) or 5)
        _half_life = int(thread.get("half_life", 0) or 0)
        with db_lock():
            conn.execute(
                    """
                    INSERT INTO open_threads(id, description, status, thread_type, introduced_chapter, due_chapter, updated_chapter, payload_json, depends_on, priority, half_life)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id)
                    DO UPDATE SET description=excluded.description, status=excluded.status,
                                  thread_type=excluded.thread_type,
                                  due_chapter=excluded.due_chapter, updated_chapter=excluded.updated_chapter,
                                  payload_json=excluded.payload_json,
                                  depends_on=excluded.depends_on, priority=excluded.priority, half_life=excluded.half_life
                    """,
                    (
                        thread_id,
                        str(thread.get("description", "")),
                        str(thread.get("status", "open")),
                        str(thread.get("thread_type", "plot")),
                        _as_chnum(thread.get("introduced_chapter")),
                        _as_chnum(thread.get("due_chapter")),
                        chapter_num,
                        json.dumps(_payload if isinstance(_payload, (dict, list)) else {}, ensure_ascii=False),
                        _depends,
                        _priority,
                        _half_life,
                    ),
                )

    metrics = extraction.get("metrics") or {}
    # payoff_type / conflict_type are the arbiter's deliberate, bandit-varied plan
    # intent. Prefer them over the extraction model's re-classification of the
    # written prose: a cheap extraction model (deepseek-flash) collapses every
    # chapter to payoff_type='reveal' regardless of the (varied) plan, which then
    # fires false payoff-monotony penalties downstream (yeban_guize Ch5/7). Fall
    # back to the extraction value only when the plan didn't declare one.
    _plan = plan or {}
    _plan_payoff_type = str(_plan.get("payoff_type") or "").strip() or None
    _plan_conflict_type = str(_plan.get("conflict_type") or "").strip() or None
    _sh = review.get("style_health") or {}
    _sh_metrics = _sh.get("metrics") or {}
    _af = review.get("ai_flavor_health") or {}
    _af_metrics = _af.get("metrics") or {}
    metrics_row = {
        "chapter": chapter_num,
        "title": extraction.get("title"),
        "score": safe_score(review.get("score", 0)),
        "readthrough_score": safe_score(review.get("readthrough_score", 0)),
        "hook_score": safe_score(review.get("hook_score", review.get("hook_strength", 0))),
        "payoff_score": safe_score(review.get("payoff_score", 0)),
        "novelty_score": safe_score(review.get("novelty_score", 0)),
        "prose_score": safe_score(review.get("prose_score", review.get("aesthetic_score", 0))),
        "continuity_score": safe_score(review.get("continuity_score", 0)),
        "plan_score": plan_score(decision),
        "payoff_type": _plan_payoff_type or metrics.get("payoff_type"),
        "conflict_type": _plan_conflict_type or metrics.get("conflict_type"),
        "tension": metrics.get("tension"),
        "novelty": metrics.get("novelty"),
        "hook_strength": metrics.get("hook_strength"),
        "emotional_tone": metrics.get("emotional_tone"),
        "accepted": 1 if review.get("accepted") else 0,
        "em_dash_per_kchar": _sh_metrics.get("em_dash_per_kchar"),
        "style_penalty": _sh.get("penalty"),
        "emotional_impact": safe_score(review.get("emotional_impact", 0)),
        # 反过度书写锚点指标（趋势项/回放/退化诊断读取）。
        "avg_sentence_chars": _sh_metrics.get("avg_sentence_chars"),
        "dialogue_char_ratio": _sh_metrics.get("dialogue_char_ratio"),
        "tech_per_kchar": _sh_metrics.get("tech_per_kchar"),
        "genre_score": (review.get("genre_adherence") or {}).get("genre_score"),
        # AI味确定性检测指标（_prewrite_quality_contract AI味预算读取）。
        "ai_cliche_per_kchar": _af_metrics.get("ai_cliche_per_kchar"),
        "metaphor_per_kchar": _af_metrics.get("metaphor_per_kchar"),
        "tell_not_show_per_kchar": _af_metrics.get("tell_not_show_per_kchar"),
        "adverb_per_kchar": _af_metrics.get("adverb_per_kchar"),
        "ai_flavor_penalty": _af.get("penalty"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with db_lock():
        conn.execute(
            """
            INSERT INTO chapter_metrics(
                chapter, title, score, readthrough_score, hook_score, payoff_score,
                novelty_score, prose_score, continuity_score, plan_score, payoff_type, conflict_type, tension,
                novelty, hook_strength, emotional_tone, accepted, em_dash_per_kchar, style_penalty,
                emotional_impact, avg_sentence_chars, dialogue_char_ratio, tech_per_kchar, genre_score,
                ai_cliche_per_kchar, metaphor_per_kchar, tell_not_show_per_kchar, adverb_per_kchar, ai_flavor_penalty,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
                title=excluded.title,
                score=COALESCE(NULLIF(excluded.score, 0), score),
                readthrough_score=excluded.readthrough_score, hook_score=excluded.hook_score,
                payoff_score=excluded.payoff_score, novelty_score=excluded.novelty_score,
                prose_score=excluded.prose_score, continuity_score=excluded.continuity_score,
                plan_score=excluded.plan_score,
                payoff_type=excluded.payoff_type, conflict_type=excluded.conflict_type,
                tension=excluded.tension, novelty=excluded.novelty, hook_strength=excluded.hook_strength,
                emotional_tone=excluded.emotional_tone, accepted=excluded.accepted,
                em_dash_per_kchar=excluded.em_dash_per_kchar, style_penalty=excluded.style_penalty,
                emotional_impact=excluded.emotional_impact,
                avg_sentence_chars=excluded.avg_sentence_chars,
                dialogue_char_ratio=excluded.dialogue_char_ratio,
                tech_per_kchar=excluded.tech_per_kchar,
                genre_score=excluded.genre_score,
                ai_cliche_per_kchar=excluded.ai_cliche_per_kchar,
                metaphor_per_kchar=excluded.metaphor_per_kchar,
                tell_not_show_per_kchar=excluded.tell_not_show_per_kchar,
                adverb_per_kchar=excluded.adverb_per_kchar,
                ai_flavor_penalty=excluded.ai_flavor_penalty
            """,
            (
                metrics_row["chapter"],
                metrics_row["title"],
                metrics_row["score"],
                metrics_row["readthrough_score"],
                metrics_row["hook_score"],
                metrics_row["payoff_score"],
                metrics_row["novelty_score"],
                metrics_row["prose_score"],
                metrics_row["continuity_score"],
                metrics_row["plan_score"],
                metrics_row["payoff_type"],
                metrics_row["conflict_type"],
                metrics_row["tension"],
                metrics_row["novelty"],
                metrics_row["hook_strength"],
                metrics_row["emotional_tone"],
                metrics_row["accepted"],
                metrics_row["em_dash_per_kchar"],
                metrics_row["style_penalty"],
                metrics_row["emotional_impact"],
                metrics_row["avg_sentence_chars"],
                metrics_row["dialogue_char_ratio"],
                metrics_row["tech_per_kchar"],
                metrics_row["genre_score"],
                metrics_row["ai_cliche_per_kchar"],
                metrics_row["metaphor_per_kchar"],
                metrics_row["tell_not_show_per_kchar"],
                metrics_row["adverb_per_kchar"],
                metrics_row["ai_flavor_penalty"],
                metrics_row["created_at"],
            ),
        )
        conn.commit()

    updates = extraction.get("memory_updates") or {}
    # LLM extraction JSON: memory_updates may come back malformed (a bare string
    # instead of a dict, or a per-key value that isn't a list). Guard so finalize
    # can't crash here — a crash leaves chapter_completed.json unwritten and wedges
    # resume in an endless "Resuming partially indexed Ch{n}" loop.
    if not isinstance(updates, dict):
        updates = {}
    def _as_list(v: Any) -> list[Any]:
        return v if isinstance(v, list) else []
    append_memory(paths.bible, chapter_num, _as_list(updates.get("bible")))
    append_memory(paths.characters, chapter_num, _as_list(updates.get("characters")))
    append_memory(paths.timeline, chapter_num, _as_list(updates.get("timeline")))
    append_memory(paths.threads, chapter_num, _as_list(updates.get("threads")))

    _cl = extraction.get("causal_links")
    store_causal_links(conn, chapter_num, _cl if isinstance(_cl, list) else [])

    # Relationship changes extracted from this chapter
    try:
        from store import upsert_relationship
        for rc in extraction.get("relationship_changes", []):
            if not isinstance(rc, dict):
                continue
            ca = str(rc.get("char_a", "")).strip()
            cb = str(rc.get("char_b", "")).strip()
            if not ca or not cb:
                continue
            delta = float(rc.get("intensity_delta", 0) or 0)
            upsert_relationship(
                conn, chapter_num, ca, cb,
                stage=str(rc.get("new_stage", "")),
                event_desc=str(rc.get("event", ""))[:120],
            )
    except Exception:
        pass

    # Info revelation tracking
    try:
        from store import upsert_info_revelation
        for ir in extraction.get("info_revelations", []):
            if not isinstance(ir, dict):
                continue
            upsert_info_revelation(conn, chapter_num, ir)
    except Exception:
        pass

    # Dialogue fingerprint persistence
    try:
        fingerprints = extraction.get("dialogue_fingerprints", [])
        if fingerprints and isinstance(fingerprints, list):
            fp_path = paths.memory_dir / "dialogue_fingerprints.json"
            existing_fp: dict[str, Any] = {}
            if fp_path.exists():
                try:
                    existing_fp = json.loads(fp_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            changed = False
            for fp in fingerprints:
                if not isinstance(fp, dict):
                    continue
                name = str(fp.get("character", "")).strip()
                style = str(fp.get("speaking_style", "")).strip()
                if name and style:
                    existing_fp[name] = {"style": style, "updated_chapter": chapter_num}
                    changed = True
            if changed:
                fp_path.write_text(json.dumps(existing_fp, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def append_memory(path: Path, chapter_num: int, items: list[Any]) -> None:
    if not items:
        return
    existing = read_text(path)
    section_header = f"## Ch{chapter_num}"
    if section_header in existing:
        return
    existing_bullets = set()
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            existing_bullets.add(stripped[2:].strip())
    fresh = []
    for item in items:
        text = str(item).strip()
        if not text or text in existing_bullets:
            continue
        fresh.append(text)
        existing_bullets.add(text)
    if not fresh:
        return
    append_text(path, f"\n\n{section_header}\n" + "\n".join(f"- {t}" for t in fresh) + "\n")

def _render_state_md_template(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    protagonist_state: str,
    next_directions: list[str],
) -> str:
    """Compose the new state.md deterministically.

    The structure follows what readers expect: progress meta, recent chapter
    summaries (5), key entity states, active threads (open), and the LLM-only
    sections (protagonist_state, next_12_directions).
    """
    from store import recent_events, recent_metrics

    total_chars = count_chars(paths.book)
    metrics = recent_metrics(conn, 5)
    threads_text = read_text(paths.threads).strip()

    # Last 5 chapter title+key payoff
    summary_lines: list[str] = []
    for m in metrics:
        ch = m.get("chapter")
        title = m.get("title") or ""
        score = m.get("score")
        tone = m.get("emotional_tone") or ""
        payoff = m.get("payoff_type") or ""
        summary_lines.append(f"- Ch{ch} 「{title}」 score={score} payoff={payoff} tone={tone}")

    # Pull events from this chapter's extraction
    this_chapter_events: list[str] = []
    for ev in extraction.get("events", [])[:8]:
        s = str(ev.get("summary", "")).strip()
        if s:
            this_chapter_events.append(f"- {s[:200]}")

    next_dir_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(next_directions[:12]))

    parts: list[str] = [
        f"# 第{chapter_num}章后状态快照",
        f"\n## 进度\n- 总字数：{total_chars}\n- 最新章节：Ch{chapter_num} 「{extraction.get('title', '')}」",
        "\n## 近期章节（最新在前）\n" + ("\n".join(summary_lines) if summary_lines else "_(无)_"),
        "\n## 最新章节关键事件\n" + ("\n".join(this_chapter_events) if this_chapter_events else "_(无)_"),
        "\n## 主角状态\n" + (protagonist_state.strip() or "_(空)_"),
        "\n## 接下来12章方向\n" + (next_dir_lines or "_(无)_"),
        "\n## 活跃伏线\n" + (threads_text[:4000] if threads_text else "_(无)_"),
    ]
    return "\n".join(parts) + "\n"


def update_state_file(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    extraction: dict[str, Any],
) -> None:
    """Render state.md deterministically from the extraction.

    The two dynamic sections (protagonist_state / next_12_directions) ride in the
    extraction JSON itself — extract_events is the single per-chapter state LLM
    call. No LLM here.
    """
    if paths.state.exists():
        shutil.copy2(paths.state, paths.state.with_suffix(".md.bak"))

    protagonist_state = str(extraction.get("protagonist_state", "")).strip()
    raw_dirs = extraction.get("next_12_directions") or []
    next_directions = [str(d).strip() for d in raw_dirs if str(d).strip()] if isinstance(raw_dirs, list) else []
    new_state = _render_state_md_template(
        paths, conn, chapter_num, extraction, protagonist_state, next_directions
    )
    write_text(paths.state, new_state)

def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    if len(chapter.strip()) < 500:
        raise RuntimeError(
            f"Refusing to save Ch{chapter_num}: only {len(chapter.strip())} chars "
            f"(likely provider refusal or empty response). Preview: {chapter[:200]!r}"
        )
    write_text(chapter_path(paths, chapter_num), chapter)
    append_text(paths.book, "\n\n" + chapter)
    # Incrementally index the saved chapter for retrieval (RAG). Best-effort.
    try:
        from retrieval import index_chapter

        index_chapter(paths, chapter_num, chapter)
    except Exception:
        pass
    append_text(
        paths.logs_dir / "reviews.jsonl",
        json.dumps(
            {
                "chapter": chapter_num,
                "score": review.get("score"),
                "readthrough_score": review.get("readthrough_score"),
                "hook_score": review.get("hook_score"),
                "payoff_score": review.get("payoff_score"),
                "novelty_score": review.get("novelty_score"),
                "prose_score": review.get("prose_score"),
                "continuity_score": review.get("continuity_score"),
                "accepted": review.get("accepted"),
                "problems": review.get("problems", []),
                "continuity_risks": review.get("continuity_risks", []),
                "plan_title": plan.get("title"),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
