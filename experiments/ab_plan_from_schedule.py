"""计划级轻量 A/B：plan_from_schedule(轻规划) vs 已存的重型规划（同章、同卷纲排期）。

只读、非破坏：对 Ch12-14 现场生成轻规划，与 checkpoint 里已存的重型 plan 对比
beat 数 / 可落地性门(blocked) / 对卷纲排期的贴合度 / 规划 LLM 调用数。3 次轻展开调用。
用法：NOVEL_CONFIG=novels/tangshuting_e2e/config.yaml NOVEL_PROMPT=... venvpy experiments/ab_plan_from_schedule.py
"""
import json, os, re
os.environ.setdefault("NOVEL_CONFIG", "novels/tangshuting_e2e/config.yaml")
os.environ.setdefault("NOVEL_PROMPT", "novels/tangshuting_e2e/prompt.md")

import config as cfgmod
import planning, quality
from store import init_db
from openai import OpenAI

config = cfgmod.load_config()
paths = cfgmod.get_paths(config)
conn = init_db(paths)
api = config["api"]
client = OpenAI(base_url=api["base_url"], api_key=api["api_key"],
                default_headers={"User-Agent": api.get("user_agent", "")} or None)

def _read(p):
    try:
        return cfgmod.read_text(p)
    except Exception:
        return ""

def _tokens(text):  # 抽 2-3 字中文 token 做贴合度粗测
    return set(re.findall(r"[一-龥]{2,4}", text or ""))

def _plan_text(plan):
    b = plan.get("beats") or []
    return " ".join(str(x) for x in b) + " " + str(plan.get("goal","")) + " " + str(plan.get("conflict",""))

def _cov(schedule, plan):  # 计划文本覆盖了多少排期里的 token
    st = {t for t in _tokens(schedule) if len(t) >= 3}
    if not st:
        return 0.0
    pt = _tokens(_plan_text(plan))
    return round(len(st & pt) / len(st), 2)

print(f"{'章':>4} | {'轻beat':>6} {'重beat':>6} | {'轻exec':>7} {'重exec':>7} | {'轻贴合':>6} {'重贴合':>6} | 调用(轻/重)")
print("-" * 78)
rows = []
for n in (12, 13, 14):
    schedule = planning._read_schedule_rows(paths, n)
    if not schedule:
        print(f"{n:>4} | 卷纲无该章排期表 → 轻规划会回退重型（跳过）")
        continue
    tail = _read(paths.chapters_dir / f"{n-1:04d}.md")[-1500:] if hasattr(paths, "chapters_dir") else ""
    if not tail:
        tail = _read(os.path.join("novels/tangshuting_e2e/chapters", f"{n-1:04d}.md"))[-1500:]
    light = planning.plan_from_schedule(client, paths, conn, config, n, tail)
    if not light:
        print(f"{n:>4} | plan_from_schedule 返回 None（展开失败）→ 回退重型")
        continue
    lp, _ = light
    from checkpoint import load_checkpoint
    heavy = load_checkpoint(paths, n, "plan_initial_selected.json") or {}
    hp = heavy.get("plan") or {}
    lb, hb = len(lp.get("beats") or []), len(hp.get("beats") or [])
    le = quality.plan_executability_gate(lp, config).get("blocked", None)
    he = quality.plan_executability_gate(hp, config).get("blocked", None)
    lcov, hcov = _cov(schedule, lp), _cov(schedule, hp)
    # 重型规划调用数：候选(1)+仲裁(1)（单候选已跳融合评审）；轻规划=1
    heavy_calls = 2
    print(f"{n:>4} | {lb:>6} {hb:>6} | {('阻'+'' if le else '通'):>7} {('阻' if he else '通'):>7} | {lcov:>6} {hcov:>6} |   1 / {heavy_calls}")
    rows.append((n, lb, hb, le, he, lcov, hcov))

if rows:
    import statistics as st
    print("-" * 78)
    print(f"均值: 轻beat={st.mean(r[1] for r in rows):.1f} 重beat={st.mean(r[2] for r in rows):.1f} | "
          f"轻贴合={st.mean(r[5] for r in rows):.2f} 重贴合={st.mean(r[6] for r in rows):.2f} | "
          f"轻exec阻塞={sum(1 for r in rows if r[3])}/{len(rows)} 重exec阻塞={sum(1 for r in rows if r[4])}/{len(rows)} | "
          f"规划调用 轻=1×{len(rows)} 重=2×{len(rows)}")
