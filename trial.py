from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    Paths,
    configured_api_endpoints,
    ensure_project,
    get_paths,
    load_config,
    log,
    normalize_chapter,
    read_text,
    write_text,
)
from llm import LLMClientPool, call_llm, json_prompt, load_json_with_repair
from memory import bootstrap, cacheable_prefix, memory_context
from store import init_db


ROUTE_SYSTEM = """你是中文网文爆款开篇策划。
只返回恰好一个合法 JSON 对象，不要输出其它内容。
你要为同一份创作纲要设计一条差异化的前三章试写路线。

schema:
{
  "variant_name": "短名",
  "core_selling_point": "这条路线最强的卖点",
  "differentiation": "它与常规同题材套路的区别",
  "reader_promise": "前三章向读者承诺的追读收益",
  "chapter_plans": [
    {
      "title": "第1章标题",
      "goal": "本章目标",
      "opening_hook": "前500字必须抛出的钩子",
      "conflict": "核心冲突",
      "payoff": "本章给读者的兑现",
      "ending_hook": "章末追读问题",
      "beats": ["5-8个具体页面节拍"]
    }
  ],
  "risks": ["这条路线最可能失败的原因"]
}

硬性要求：
- 必须包含恰好 __CHAPTERS__ 个 chapter_plans。
- 第1章前500字必须出现核心冲突或异常，不得慢热铺设定。
- 第3章前必须至少兑现一次本书核心卖点，不得只承诺不兑现。
- 每章都要有清晰的章末追读问题。"""


TRIAL_WRITE_SYSTEM = """你是一位商业连载中文网文作者，负责写开篇试读章节。
只输出章节正文，不要解释。

要求：
- 第一行格式：第{chapter_num}章 {title}
- 约{chapter_words}个中文字符。
- 开头前500字必须抛出本章 opening_hook。
- 本章必须兑现 chapter_plan.payoff，不能只铺垫。
- 章末必须落在 ending_hook 上，让读者想点下一章。
- 场景、动作、对话要直接演出来，不要写策划说明。
- 禁止破折号碎片链，保持完整小说句子。"""


TRIAL_REVIEW_SYSTEM = """你是平台连载小说的开篇买量/追读评审。
只返回恰好一个合法 JSON 对象，不要输出其它内容。

按爆款开篇标准评估这条试写路线，而不是按普通文笔好坏评估。
schema:
{
  "readthrough_score": 1-10,
  "hook_score": 1-10,
  "payoff_score": 1-10,
  "novelty_score": 1-10,
  "prose_score": 1-10,
  "continuity_score": 1-10,
  "overall": 1-10,
  "best_assets": ["最值得保留的卖点/场景"],
  "kill_risks": ["会导致弃书或不付费的具体风险"],
  "revision_directives": ["正式连载前必须执行的修改指令"]
}

评分纪律：
- readthrough/hook/payoff/novelty 是核心，文笔顺不等于能爆。
- 第1章慢热、第三章还没兑现核心卖点，overall 不能超过 7。
- 场景或章末手法明显重复，novelty_score 不能超过 7。"""


PACKAGE_SYSTEM = """你是网文平台的开篇包装编辑。
只返回恰好一个合法 JSON 对象，不要输出其它内容。
请为这条开篇路线生成可 A/B 测试的包装素材。

schema:
{
  "titles": ["10个书名，短、准、有卖点"],
  "intros": ["5个100-180字简介，直接抛卖点和冲突"],
  "tags": [["标签1","标签2","标签3","标签4"]],
  "first_paragraphs": ["5个正文第一段候选，必须在第一段抛出异常/冲突/情绪钩子"],
  "package_notes": ["哪组标题/简介适合什么读者"]
}

要求：
- 不要写营销空话；每个标题/简介都必须让读者知道这本书有什么不一样。
- 第一段候选要能直接替换正文开头，不能是设定说明。
- 按平台画像调整表达：免费平台更直给，起点男频可留更强设定悬念，女频更突出关系/情绪张力。"""


def _build_client(config: dict[str, Any], paths: Paths) -> Any:
    from openai import OpenAI
    import httpx

    api_endpoints, primary_endpoint_count = configured_api_endpoints(config)
    if not api_endpoints:
        raise RuntimeError("Missing API key: set api.api_key, api.api_keys, or api.api_key_groups in config.yaml")
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


def _trial_root(paths: Paths) -> Path:
    return paths.logs_dir / "opening_trials"


def _score_value(review: dict[str, Any]) -> float:
    weights = {
        "readthrough_score": 0.30,
        "hook_score": 0.20,
        "payoff_score": 0.20,
        "novelty_score": 0.15,
        "prose_score": 0.10,
        "continuity_score": 0.05,
    }
    total = 0.0
    for key, weight in weights.items():
        try:
            total += float(review.get(key, 0)) * weight
        except (TypeError, ValueError):
            pass
    try:
        overall = float(review.get("overall", 0))
    except (TypeError, ValueError):
        overall = 0.0
    return round(max(total, overall), 2)


def run_opening_trial(variants: int | None = None, chapters: int | None = None) -> Path:
    config = load_config()
    paths = get_paths(config)
    ensure_project(paths)
    conn = init_db(paths)
    client = _build_client(config, paths)

    if not paths.state.exists() or not read_text(paths.state).strip():
        bootstrap(client, paths, conn, config)

    variants = int(variants or config["novel"].get("opening_trial_variants", 3))
    chapters = int(chapters or config["novel"].get("opening_trial_chapters", 3))
    variants = max(1, min(variants, 8))
    chapters = max(1, min(chapters, 10))
    chapter_words = int(config["novel"].get("chapter_words", 4000))

    trial_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = _trial_root(paths) / trial_id
    root.mkdir(parents=True, exist_ok=True)
    log(paths, f"Opening trial start variants={variants} chapters={chapters} output={root}")

    mem = memory_context(paths, conn, config)
    try:
        from benchmark import benchmark_context, platform_guidance

        trial_platform = platform_guidance(config)
        trial_benchmarks = benchmark_context(paths, config, read_text(paths.state) + "\n" + mem[:3000], max_chars=5000)
    except Exception:
        trial_platform = "通用网文平台：卖点清晰、开篇强钩子、承诺及时兑现。"
        trial_benchmarks = ""
    results: list[dict[str, Any]] = []
    for idx in range(variants):
        variant_dir = root / f"variant_{idx + 1:02d}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        route_user = f"""## 全局记忆
{mem}

## 平台/读者画像
{trial_platform}

{trial_benchmarks}

## 变体要求
这是第 {idx + 1}/{variants} 条开篇路线。必须与其它路线显著不同：
- 不同的第一章钩子
- 不同的信息释放顺序
- 不同的核心冲突压力源
- 不同的第三章兑现方式

请设计 {chapters} 章试写路线。"""
        route_raw = call_llm(
            client,
            paths,
            config,
            ROUTE_SYSTEM.replace("__CHAPTERS__", str(chapters)),
            json_prompt(route_user),
            max_tokens=12000,
            temperature=0.85,
            cacheable_prefix=cacheable_prefix(paths, config),
        )
        route = load_json_with_repair(client, paths, config, route_raw, fallback={})
        plans = route.get("chapter_plans") if isinstance(route, dict) else []
        if not isinstance(plans, list) or not plans:
            route = {
                "variant_name": f"variant_{idx + 1}",
                "core_selling_point": "",
                "differentiation": "",
                "reader_promise": "",
                "chapter_plans": [],
                "risks": ["route generation failed"],
            }
            plans = []
        write_text(variant_dir / "route.json", json.dumps(route, ensure_ascii=False, indent=2))

        package_user = f"""## 平台/读者画像
{trial_platform}

## 开篇路线
{json.dumps(route, ensure_ascii=False, indent=2)}

为该路线生成标题/简介/标签/第一段 A/B 包装。"""
        package_raw = call_llm(
            client,
            paths,
            config,
            PACKAGE_SYSTEM,
            json_prompt(package_user),
            max_tokens=12000,
            temperature=0.75,
            cacheable_prefix=cacheable_prefix(paths, config),
        )
        package = load_json_with_repair(client, paths, config, package_raw, fallback={})
        write_text(variant_dir / "package.json", json.dumps(package, ensure_ascii=False, indent=2))

        chapter_texts: list[str] = []
        tail = ""
        for ch_idx, plan in enumerate(plans[:chapters], start=1):
            title = str(plan.get("title") or f"试写{ch_idx}").strip()
            write_user = f"""## 全局记忆
{mem}

## 本试写路线
{json.dumps(route, ensure_ascii=False, indent=2)}

## 上章结尾
{tail[-2000:] if tail else "None"}

## 本章计划
{json.dumps(plan, ensure_ascii=False, indent=2)}

写第 {ch_idx} 章试读正文。"""
            chapter_raw = call_llm(
                client,
                paths,
                config,
                TRIAL_WRITE_SYSTEM.format(chapter_num=ch_idx, title=title, chapter_words=chapter_words),
                write_user,
                temperature=float(config["api"].get("temperature", 0.8)),
                cacheable_prefix=cacheable_prefix(paths, config),
            )
            chapter = normalize_chapter(chapter_raw)
            write_text(variant_dir / f"{ch_idx:04d}.md", chapter)
            chapter_texts.append(chapter)
            tail = chapter

        chapter_blocks = "\n".join(f"### Ch{i + 1}\n{text}" for i, text in enumerate(chapter_texts))
        review_user = f"""## 路线JSON
{json.dumps(route, ensure_ascii=False, indent=2)}

## 包装素材JSON
{json.dumps(package, ensure_ascii=False, indent=2)}

## 试写章节
{chapter_blocks}
"""
        review_raw = call_llm(
            client,
            paths,
            config,
            TRIAL_REVIEW_SYSTEM,
            json_prompt(review_user),
            max_tokens=12000,
            temperature=0.2,
        )
        review = load_json_with_repair(client, paths, config, review_raw, fallback={"overall": 0})
        trial_score = _score_value(review)
        write_text(variant_dir / "review.json", json.dumps(review, ensure_ascii=False, indent=2))
        results.append(
            {
                "variant": idx + 1,
                "variant_name": route.get("variant_name", f"variant_{idx + 1}"),
                "trial_score": trial_score,
                "route": route,
                "package": package,
                "review": review,
                "path": str(variant_dir),
            }
        )
        log(paths, f"Opening trial variant {idx + 1}/{variants} score={trial_score}")

    results.sort(key=lambda item: float(item.get("trial_score", 0)), reverse=True)
    summary = {
        "trial_id": trial_id,
        "variants": variants,
        "chapters": chapters,
        "best_variant": results[0] if results else None,
        "ranking": [
            {
                "variant": r["variant"],
                "variant_name": r["variant_name"],
                "trial_score": r["trial_score"],
                "path": r["path"],
                "best_assets": r.get("review", {}).get("best_assets", []),
                "kill_risks": r.get("review", {}).get("kill_risks", []),
                "titles": (r.get("package") or {}).get("titles", [])[:5],
            }
            for r in results
        ],
    }
    write_text(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    if results:
        best = results[0]
        best_md = [
            f"# 开篇试写最佳路线：{best['variant_name']}",
            "",
            f"- trial_score: {best['trial_score']}",
            f"- variant_path: {best['path']}",
            "",
            "## 核心卖点",
            str(best["route"].get("core_selling_point", "")),
            "",
            "## 差异化",
            str(best["route"].get("differentiation", "")),
            "",
            "## 读者承诺",
            str(best["route"].get("reader_promise", "")),
            "",
            "## 推荐书名",
        ]
        best_md.extend(f"- {x}" for x in (best.get("package") or {}).get("titles", [])[:10])
        best_md.extend([
            "",
            "## 推荐简介",
        ])
        best_md.extend(f"- {x}" for x in (best.get("package") or {}).get("intros", [])[:5])
        best_md.extend([
            "",
            "## 正式连载前修改指令",
        ])
        best_md.extend(f"- {x}" for x in best.get("review", {}).get("revision_directives", []))
        write_text(root / "best_opening_route.md", "\n".join(best_md).strip() + "\n")
    log(paths, f"Opening trial complete output={root}")
    return root
