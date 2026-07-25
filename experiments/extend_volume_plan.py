"""修复中段塌陷：为 卷二 Ch31-48 补出详细逐章排期表（框架只在 bootstrap 详写前2卷，
长书跑出详细卷纲后中段失去 escalation 路线图→tension_flat→熔断）。生成后追加进 volume_plan.md，
使重型规划有路线图、且 plan_from_schedule 能读到 Ch31+ 排期。1 次 LLM 调用。主线须先停。
"""
import os
os.environ.setdefault("NOVEL_CONFIG", "novels/tangshuting_e2e/config.yaml")
os.environ.setdefault("NOVEL_PROMPT", "novels/tangshuting_e2e/prompt.md")
import config as C
from openai import OpenAI
import memory

cfg = C.load_config(); paths = C.get_paths(cfg)
api = cfg["api"]
client = OpenAI(base_url=api["base_url"], api_key=api["api_key"],
                default_headers={"User-Agent": api.get("user_agent", "")} or None)
R = C.read_text
bible = R(paths.bible)[:4000]; chars = R(paths.characters)[:4500]
vp = R(paths.volume_plan); state = R(paths.state)[:4000]
juan2 = vp[vp.find("## 第二卷"):][:3200] if "## 第二卷" in vp else ""

system = (memory.VOLUME_PLAN_CHAIN_SYSTEM
          + memory._ENSEMBLE_VOLUME_PLAN_DELTA + memory._SHUANG_PACING_VOLUME_PLAN_DELTA)
user = f"""## 世界观圣经\n{bible}\n\n## 人物档案\n{chars}\n\n## 已写到第30章的当前状态\n{state}\n\n## 第二卷现有概要（保持一致，不得推翻）\n{juan2}\n\n## 任务
本书已写到第30章，第二卷（Ch25-48）只有概要、缺"逐章排期"，导致 Ch25 后每章失去明确 payoff、
张力走平（tension_flat）、评分下滑触发熔断。请**只为 Ch31-48 逐章补出四张按章号排布的排期表**
（接续已写剧情，与第二卷概要一致：白景昀冰河认亲已于Ch28引爆；商标反扑约Ch36；棱镜中层露头；
卷末留更大对手）：
1. **角色高光轮值表**：每章≥3位男主独立高光、核心不连续两章隐形、去同质化。
2. **爽点兑现节拍表**：每章唯一主爽点(类型错峰轮换:打脸/逆袭/装逼/苏爆反转/实力碾压)、憋→炸→余韵、
   **中段每章必须有一个明确当章 payoff 与张力起伏，严禁平淡过渡章**。
3. **反转做成名场面排期**：剩余身份/关系/命运反转错峰引爆到具体章号。
4. **伏笔兑现节拍**：陆时砚十九年线(约Ch48+留到第三卷)、深伪黑产链、七碗面等长线在Ch31-48的推进节点。
每章控制在 2800-4000 字可写量。只输出这四张表（markdown），表头用 `### 卷二逐章排期补全（Ch31-48）`。"""

print("[extend] 生成 Ch31-48 详细排期…")
out = memory._gen_md_section(client, paths, cfg, system, user, tag="volume_plan_extend", max_tokens=16000)
if not out or len(out) < 300:
    print(f"[extend] 生成失败/过短 ({len(out)} 字)，未改动"); raise SystemExit(1)

sep = "\n\n---\n### 卷二逐章排期补全（Ch31-48，修复中段塌陷追加）\n"
with open(paths.volume_plan, "a", encoding="utf-8") as f:
    f.write(sep + out.strip() + "\n")
print(f"[extend] 已追加 {len(out)} 字到 volume_plan.md")

# 验证 plan_from_schedule 现在能读到 Ch31/36/40
import importlib, planning
importlib.reload(planning)
for n in (31, 36, 40, 45):
    r = planning._read_schedule_rows(paths, n)
    print(f"  Ch{n}: {'有排期('+str(len(r))+'字)' if r else '无'}")
