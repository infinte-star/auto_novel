# REDESIGN V3：第一性原理重设计

> 作者：Claude Code · 日期：2026-08-01
>
> 本文基于对当前 v2 引擎（16,349 行 / 16 模块）的逐行分析、30+ 开源项目调研、以及 ACL/EMNLP/NeurIPS 2024-2026 相关论文的交叉验证，从第一性原理出发重新设计 AI 长篇小说生成管线。

---

## §0  设计原则

| 原则 | 含义 |
|------|------|
| **Quality First** | 每一个设计决策都以最终散文质量为判据，而非中间指标 |
| **FPY Second** | 首次通过率是质量的子集——减少返工的最佳方式是首稿写好，而非修补 |
| **Less Prompt, More Structure** | 减少提示词工程，增加结构化输入/输出（JSON Schema、类型化状态） |
| **Less Agent, More Capability** | 不增加 LLM 调用数，让每次调用更有效 |
| **Data over Prompt** | 让数据（结构化状态、指标、范文片段）驱动质量，而非指令 |
| **能删就删，能合就合** | 16k 行 → 目标 10k 行以内 |

---

## §1  三个根本问题

一切 AI 长篇小说生成系统都在与三个基本问题作战。如果设计不能从根源上回应它们，复杂度就是浪费。

### 1.1 漂移问题（Drift）

> 100 章之后，风格塌缩、角色扁平化、世界观违规、质量后半段下降。

**实测数据**：
- 北大 ACL 2025：LLM 小说质量集中在前 40-60%，后半段内容重叠率上升、多样性下降
- v1 引擎 style_health 42% 章节触发（风格塌缩是最常见的 gate firing）
- CHIRON EMNLP 2024：LLM 小说的角色密度是人类小说的 2× — 过度直白暴露

**当前方案**：冻结 voice_baseline + style_health 三项检查（em-dash / fragment / pseudo-tech）
**问题**：只治症状（3 个正则），不治病因（LLM 倾向电报体 + 自评 9 分）

### 1.2 上下文问题（Context）

> 第 80 章时，模型无法直接访问第 1-78 章的原文，只能通过压缩状态间接获取信息。

**实测数据**：
- BooookScore ICLR 2024：层次合并丢失跨块关联，增量更新导致信息衰减
- Lost in Stories ACL 2026：一致性错误在叙事中段（非末段）达到峰值
- SCORE Web Conference 2026：去掉结构化状态追踪导致一致性 -17.6、物品准确率 -37.1

**当前方案**：11 段 StoryState（stable/volatile 分割 + 硬编码 char 预算）+ TF-IDF RAG
**问题**：
- 事实没有时间有效性（"Ch5 受伤" 在 Ch8 仍为 true）
- RAG 不感知当前章节计划（通用关键词匹配 vs 需要检索与本章 card 相关的内容）
- 预算固定、与体裁无关

### 1.3 评估问题（Evaluation）

> 没有可靠的自动化散文质量度量。LLM 自评不可靠（78% 人类一致性上限）。

**实测数据**：
- LitBench EACL 2026：最强自动评估器与人类偏好一致率仅 78%
- v1 self-score 中位 8.00、范围 7.4-8.7、无判别力 → v2 已正确删除
- 100-Endings COLM 2026：LLM judge 在 EQ-Bench 上将零样本 AI 小说评分 **高于** New Yorker 作品

**当前方案**：零 LLM 机械门（style_health / CCC / 重复检测）+ 盲测 WR anchor
**问题**：
- WR anchor 没有 production 基准库（`benchmarks/anchor/` 不存在）
- 12 个 advisory gate 生成建议但从不阻断，弱指令跟随模型直接忽略
- 没有散文自然度的多维检测（只有 3 个正则 vs StoryScope 的 30 个特征）

---

## §2  当前 v2 引擎诊断

### 2.1 做对了的（必须保留）

| 设计决策 | 为什么正确 | 外部验证 |
|----------|-----------|----------|
| 确定性决策表 | 可重放、无 if-forest、自动恢复 | ASP+LLM Hybrid (ACL Workshop 2025) 验证符号约束+神经生成是正确方向 |
| 单调用写章 | 2.5 calls/ch vs v1 的 22.7；散文质量随调用减少而提升 | 行业共识：少调用大窗口 > 多调用小窗口 |
| Stable/volatile prefix 分割 | 90% prompt cache 成本优化 | Anthropic 缓存设计最佳实践 |
| 零 LLM 验收门 + scope 防闩 | 客观、可重放、不会隔章锁死 | LitBench/100-Endings 证明 LLM 评分不可靠 |
| Keep-only-if-improved 修复 | 修复不会恶化 | Self-Refine NeurIPS 2023：修复越多越平庸 |
| CCC 契约-卡片-章节验证 | 零 LLM 校验规划是否落地 | FactTrack NAACL 2025 验证事实分解+验证是正确范式 |
| 冻结 voice baseline | 断开风格自反馈回路 | 无竞品有此设计（本项目独有） |
| 盲测 WR anchor | 唯一不可自授的质量信号 | LLM Review Harvard 2026：盲测外审 > 自评 |
| FPY 测量纪律 | 可重放的验收指标 | 无竞品有此设计（本项目独有） |

### 2.2 核心瓶颈

| 瓶颈 | 严重程度 | 行代码 | 影响 |
|------|---------|--------|------|
| **质量门臃肿** | 高 | 5,526 行（34%） | 23 个 gate、GateRegistry 元数据 overhead、advisory 无执行路径 |
| **静态弧规划** | 中 | — | 10 章预生成 card，后期章节与实际剧情脱节 |
| **TF-IDF RAG** | 中 | 602 行 | 无语义理解，不感知当前 card，最大预算段却贡献最低 |
| **风格检测薄弱** | 高 | — | 仅 3 个正则 vs AI 散文 30+ 可检测特征 |
| **Advisory 无执行** | 高 | 1,398 行 | 12 个 advisory gate 生成建议后被丢弃 |
| **无时间态事实** | 中 | — | 平面状态无法追踪"何时为真" |
| **单线程** | 低 | — | 架构正确（确定性重放），但无吞吐量优化空间 |
| **手工 YAML 解析** | 低 | 780 行 | 不支持嵌套/列表/锚点，弯引号类 bug 反复出现 |

### 2.3 调用经济学

| 阶段 | v1 | v2 当前 | v3 目标 |
|------|-----|---------|---------|
| 规划 | ~12 (committee) | 0.1 (amortized arc/10) | 0.3 (轻量 JIT 精炼) |
| 写章 | 1 + 多轮候选 | 1 | 1 |
| 验收 | 1+ LLM review | 0 (零 LLM) | 0 |
| 修复 | 多轮自评循环 | 0-2 (L0+L1) | 0-2 (不变) |
| 救章 | — | 0-1 | 0-1 |
| **总计/章** | **22.7** | **2.5** | **2.3-2.8** |

v3 目标不是进一步压缩调用数（2.5 已近最优），而是提升每次调用的质量产出。

---

## §3  V3 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        BOOTSTRAP (一次性)                        │
│  prompt.md + config.yaml                                        │
│  → world_bible, characters, voice_baseline, volume_skeleton     │
│  → initial StoryState                                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CHAPTER LOOP (每章)                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ 1. REFINE    │ ← 轻量 JIT 精炼 card（0 或 1 call）           │
│  │    CARD      │   arc skeleton → actual state → refined card   │
│  └──────┬───────┘                                               │
│         │                                                        │
│  ┌──────▼───────┐                                               │
│  │ 2. PROJECT   │ ← 零 LLM：构建 StoryState 投影                │
│  │    CONTEXT   │   temporal filter + card-aware RAG             │
│  │              │   + feed-forward directives                    │
│  └──────┬───────┘                                               │
│         │                                                        │
│  ┌──────▼───────┐                                               │
│  │ 3. WRITE     │ ← 1 call：散文 + ===状态增量=== + delta JSON   │
│  │    CHAPTER   │   exemplar bank 注入                           │
│  └──────┬───────┘                                               │
│         │                                                        │
│  ┌──────▼───────┐                                               │
│  │ 4. VERIFY    │ ← 零 LLM：blocking gates + feed-forward gates │
│  │              │   advisory 结果写入 directives 供下一章用       │
│  └──────┬───────┘                                               │
│         │                                                        │
│    pass?├──yes──┐                                                │
│         │       │                                                │
│    no   ▼       │                                                │
│  ┌──────────┐   │                                               │
│  │ 5. REPAIR│   │ ← L0 零 LLM + L1 有界 LLM (≤2 calls)         │
│  └──────┬───┘   │                                               │
│    pass?│       │                                                │
│    no   ▼       │                                                │
│  ┌──────────┐   │                                               │
│  │ 6. RESCUE│   │ ← 1 call max，hard blocks 作为显式指令         │
│  └──────┬───┘   │                                               │
│         │       │                                                │
│         ▼       ▼                                                │
│  ┌──────────────────┐                                           │
│  │ 7. COMMIT        │ ← 原子写入：章节文本 + state delta         │
│  │    + UPDATE STATE│   + 更新 temporal facts + exemplar bank    │
│  │    + FEED-FORWARD│   + advisory → next chapter directives     │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心变化摘要

| 维度 | v2 当前 | v3 提案 | 收益 |
|------|---------|---------|------|
| 规划 | 每 10 章 1 次 arc planning | 保留 arc skeleton + 每章 JIT card 精炼 | card 永不过时 |
| 上下文 | 11 段固定预算 | 7 段自适应预算 + 时间态事实过滤 | 更相关、更紧凑 |
| RAG | TF-IDF 通用检索 | card-aware 检索（按当前章需求） | 检索相关度提升 |
| 质量门 | 23 gate (blocking + advisory) | ~10 gate (blocking + feed-forward) | -60% 门数，advisory 不再白跑 |
| 风格检测 | 3 项正则 (style_health) | 8+ 项散文自然度评分 | 覆盖 AI 散文主要特征 |
| Advisory | fire-and-forget | feed-forward（结果注入下一章 prompt） | 自纠正循环，零额外 LLM |
| 状态 | 平面 key-value | 时间态事实（valid_from/valid_until） | 追踪"何时为真" |
| 范文 | 无 | exemplar bank（本书最佳段落） | data-driven 风格锚定 |
| 后期质量 | 阈值固定 | 阈值随章节进度收紧 | 对抗后半段质量下降 |
| 代码量 | 16,349 行 / 16 模块 | 目标 ≤10,000 行 / 11 模块 | -40% 代码 |

---

## §4  模块设计

### 4.0 模块总览

```
engine/
  pipeline.py    ← 决策表 + 章节主循环          (~400 行, was loop.py 2043)
  plan.py        ← 卷纲 + arc skeleton + JIT card  (~600 行, was 1108)
  write.py       ← writer prompt + 输出解析       (~800 行, was 2274)
  state.py       ← StoryState + 时间态事实 + delta (~500 行, was canon.py + types.py)
  gates.py       ← blocking gates + feed-forward   (~1500 行, was quality*.py 5526)
  repair.py      ← L0/L1/rescue                   (~600 行, was quality_repair + repair)
  store.py       ← SQLite + 文件持久化            (~500 行, was 844)
  retrieve.py    ← card-aware RAG                  (~400 行, was 602)
  llm.py         ← LLM 客户端池                   (~800 行, was 1205)
  config.py      ← 配置加载                        (~400 行, was 780)
  bootstrap.py   ← 小说初始化                      (~800 行, was 1383)
  anchor.py      ← 外部质量锚 (WR)                (~350 行, was 376)
  types.py       ← 数据类型                        (~250 行)
目标总计: ~7,900 行（当前 16,349 行的 48%）
```

### 4.1 pipeline.py — 决策表

**职责**：章节循环的唯一入口。决策表 + 调度，不含任何业务逻辑。

```python
# 决策表（伪代码）——从上往下逐行求值，每个 action 执行后回到表头
DECISIONS = [
    # predicate                          action              
    (needs_arc_skeleton,                 generate_arc),       # 每 ~10 章
    (needs_card_refinement,              refine_card),        # 每章（轻量）
    (no_draft,                           write_chapter),      # 1 call
    (no_acceptance_report,               run_gates),          # 0 LLM
    (has_l0_repairs,                     apply_l0),           # 0 LLM
    (has_l1_repairs_and_budget,          apply_l1),           # ≤2 LLM
    (has_hard_blocks_and_rescue_budget,  rescue),             # ≤1 LLM
    (True,                               commit),             # 持久化
]
```

**与 v2 的区别**：
- `needs_card_refinement` 是新增行——JIT card 精炼
- `fold_constraints` 被内联到 `refine_card`
- 决策表从 8 行简化到 8 行（数量不变，但每行更清晰）
- 所有 StoryState 投影逻辑移入 `state.py`
- 所有质量门逻辑移入 `gates.py`

**不变式**：
- 表从上往下逐行求值，action 后回到表头
- `chapter_completed.json` 必须在 commit 中同步写入
- `review_round0.json` 一次写入，永不被修复/救章覆盖

### 4.2 plan.py — 规划

**职责**：两级规划——卷纲 skeleton + 每章 JIT card。

#### 4.2.1 Arc Skeleton（保留，微调）

与 v2 相同：每 `arc_span`（10）章一次高推理调用，产出 ChapterCard 骨架。但骨架的定位从"最终指令"降级为"方向指引"：

```
Arc Skeleton = {
    chapter_num: {where, who, wants, turn_direction, payoff_candidates, tension_target}
    ...
}
```

注意：`turn`、`beats`、`exit_hook` 从骨架中移除，留给 JIT 精炼。

#### 4.2.2 JIT Card Refinement（新增，核心创新）

**动机**：DOME (NAACL 2025) 和 StoryWriter (CIKM 2025) 证明动态规划 > 静态规划。10 章前生成的 card 到执行时可能与实际剧情脱节。

**设计**：在写章前，用确定性逻辑 + 可选轻量 LLM 调用精炼 card：

```python
def refine_card(skeleton_card, actual_state, prev_chapter_delta, feed_forward_directives):
    """
    输入：arc skeleton card + 当前实际 state + 上一章 delta + feed-forward 建议
    输出：精炼后的完整 ChapterCard（含 turn, beats, exit_hook）
    
    Phase 1（零 LLM）：确定性精炼
    - 根据 prev_chapter_delta 更新 who/where（角色可能已移动）
    - 检查 payoff_candidates 中的 thread 是否仍然 open
    - 注入 feed-forward directives（上一章 advisory gate 的建议）
    - 检查 opening_type/tension_level/hook_type 不与最近 3 章重复
    
    Phase 2（可选 LLM，仅当 Phase 1 不足时）：
    - 如果 skeleton 与 actual state 严重矛盾（角色死了、地点被摧毁）
    - 或者 payoff 全部过期
    → 一次轻量调用生成新的 turn + beats + exit_hook
    """
```

**成本**：大多数章节 Phase 1 即可（0 call），少数需要 Phase 2（1 cheap call）。平均 ~0.2 call/chapter。

**为什么不合并到 write call**：
- CCC 门需要 card 作为契约来验证（没有 card = 没有契约 = 无法零 LLM 验收）
- 跨章协调需要 card 级信息（开场类型不重复、张力曲线等）
- card 验证是零 LLM 的，可以在写章前拦截坏计划

### 4.3 state.py — 时间态状态

**职责**：StoryState 定义、投影、delta 管理、时间态事实。

#### 4.3.1 时间态事实（新增，核心创新）

**动机**：FactTrack (NAACL 2025) 证明时间感知事实追踪能捕获平面状态无法检测的矛盾。

```python
@dataclass
class TemporalFact:
    content: str           # "林默右臂骨折"
    category: str          # "character_state" | "relationship" | "world_rule" | "possession"
    subject: str           # "林默"
    valid_from: int         # 引入章号 (5)
    valid_until: int | None # 失效章号 (8) or None=当前仍有效
    superseded_by: str | None  # "林默右臂痊愈"
```

**实现**：
- delta JSON 中已有结构化状态更新（当前是平面 key-value）
- 扩展 delta 格式：每个状态变更标注 `supersedes` 字段
- 投影时过滤：`valid_at(chapter_num)` 只返回在该章有效的事实
- 矛盾检测：如果新 delta 声明 X 但 X 的前置条件在当前章不成立 → advisory 警告

**成本**：零 LLM。时间态过滤是纯函数。存储增量很小。

#### 4.3.2 StoryState 投影（简化 11→7 段）

```python
SECTIONS = {
    # ─── Stable prefix (prompt cache) ─────────────────
    "voice":      Section(budget=1500, stable=True),   # 冻结的风格宪章
    "world":      Section(budget=2500, stable=True),   # brief + world rules 合并
    
    # ─── Volatile (per-chapter) ────────────────────────
    "state":      Section(budget=2500, stable=False),   # 角色状态 + 活跃线索 + temporal facts
    "plan":       Section(budget=2000, stable=False),   # 精炼后的 ChapterCard
    "recent":     Section(budget=1500, stable=False),   # 上一章尾部 + delta 摘要
    "retrieved":  Section(budget=3000, stable=False),   # card-aware RAG 结果
    "directives": Section(budget=1000, stable=False),   # feed-forward advisory 结果
}
# 总预算: 14,000 chars (vs v2 的 18,900)
```

**合并了什么**：
- `brief` + `facts` → `world`（创意简报和世界规则本就高度重叠）
- `focus` + `threads` + `ledger` → `state`（都是当前状态的不同切面）
- `opening` (仅前 3 章) 内联到 `directives`
- 新增 `directives`：feed-forward advisory 的结构化载体

**为什么减少总预算**：
- "保留与当前章节相关的最大压缩" 是 StoryWriter 的实验结论
- card-aware RAG 提高相关度 → 可以用更少的 chars 达到相同的信息覆盖
- "lost in the middle" 效应意味着更长的上下文不一定更好

### 4.4 gates.py — 质量门（大幅简化）

**核心改变**：从 23 gate 的注册表模式，简化为两类门的扁平列表。

#### 4.4.1 Blocking Gates（硬门，可阻断）

保留 v2 中经过实测有效的硬门：

| Gate | 检查内容 | 修复层 |
|------|---------|--------|
| `naturalness` | 散文自然度评分（扩展版 style_health） | L0 |
| `repetition` | 跨章重复 + 章内相邻重复 + 描述词频率（合并 3 gate） | L0 |
| `fossils` | 全书化石短语 | L0 |
| `length` | 字数在 [min, max] 区间 | L1 |
| `contract` | CCC 契约-卡片-章节验证 | — (rewrite) |
| `opening` | 首段行动/对话/悬念，非纯景 | L0 |

**从 v2 的 8+2 个硬门缩减为 6 个**，通过合并重复检测类门。

#### 4.4.2 Feed-Forward Gates（前馈门，核心创新）

**动机**：v2 有 12 个 advisory gate，生成建议但被丢弃。弱指令跟随模型直接忽略 system prompt 中的一般性建议。

**新机制**：advisory 的输出不再是日志消息，而是结构化的 `Directive`，注入下一章的 prompt `directives` 段：

```python
@dataclass
class Directive:
    source: str      # "dialogue_health"
    priority: int    # 1-3, 影响在 directives 段中的排序
    instruction: str # "本章对话占比 12%，远低于目标 30%。下一章在叙事段落中嵌入角色对话。"
    expires: int     # 该建议过期的章号（通常 current + 2）
    
def feed_forward_gates(text, chapter_num, state):
    """运行所有前馈门，返回 Directive 列表"""
    directives = []
    
    # 对话健康
    ratio = dialogue_char_ratio(text)
    if ratio < genre_target:
        directives.append(Directive("dialogue", 2, f"对话占比 {ratio:.0%}，目标 {genre_target:.0%}。下一章增加角色对话。", chapter_num + 2))
    
    # AI 味检测
    ai_score = ai_flavor_score(text)
    if ai_score > threshold:
        directives.append(Directive("ai_flavor", 1, f"检测到 AI 写作特征：{ai_score.details}。下一章注意避免。", chapter_num + 2))
    
    # 段落形态
    # 信息密度
    # 章尾力度
    # 钩子/尾重复
    # ... 每个 advisory gate 生成 0 或 1 个 Directive
    
    return directives
```

**效果**：
- 零额外 LLM 调用（门本身是零 LLM，注入是字符串拼接）
- 建议从"被忽略的日志"变为"下一章 prompt 的结构化约束"
- 自动过期（`expires` 字段），不会永久堆积
- 优先级排序确保最重要的建议在 `directives` 预算内优先展示

#### 4.4.3 散文自然度评分（Naturalness Score）

扩展 v2 的 style_health（3 项）为多维散文自然度评分：

| 维度 | 检测方法 | 来源 |
|------|---------|------|
| em-dash 密度 | 计数/kchar | v2 保留 |
| 碎片行比例 | 短行占比 | v2 保留 |
| 伪技术密度 | 计数/kchar | v2 保留 |
| 句式单一度 | 句长 Shannon 熵 | StoryScope 2026 |
| 过度直白 | 情感/心理词汇密度 | CHIRON EMNLP 2024 |
| 连接词滥用 | "然而/此刻/不禁" 类词频 | InkFlow de-AI |
| 感官缺失 | 五感动词密度 | WritingBench 2025 |
| 章间词汇衰减 | 滑动窗口 unique token ratio | 北大 ACL 2025 |

**实现**：所有维度都是纯函数（正则 + 计数 + 信息论），零 LLM。

**阈值梯度（新增）**：

```python
def escalation_factor(chapter_num, max_chapters):
    """后半段收紧阈值，对抗质量下降"""
    if not max_chapters:
        return 1.0
    progress = chapter_num / max_chapters
    if progress < 0.4:
        return 1.0        # 前 40% 正常阈值
    elif progress < 0.7:
        return 0.95       # 中段轻微收紧
    else:
        return 0.90       # 后 30% 收紧 10%
```

### 4.5 write.py — 写作器

#### 4.5.1 Exemplar Bank（新增，核心创新）

**动机**："Data over Prompt"——与其告诉模型"写得好"（提示工程），不如展示"好是什么样"（数据驱动）。

```python
class ExemplarBank:
    """从已写章节中收集最佳段落作为风格锚点"""
    
    def __init__(self, max_exemplars=20):
        self.exemplars = []  # (text, scores, chapter_num)
    
    def add_from_chapter(self, text, chapter_num, naturalness_scores):
        """commit 时调用：从通过所有硬门的章节中选取最佳段落"""
        paragraphs = split_paragraphs(text)
        for p in paragraphs:
            score = paragraph_quality_score(p, naturalness_scores)
            if score > self.threshold and len(self.exemplars) < self.max_exemplars:
                self.exemplars.append((p, score, chapter_num))
        # 保持 top-N，按分数排序
        self.exemplars.sort(key=lambda x: x[1], reverse=True)
        self.exemplars = self.exemplars[:self.max_exemplars]
    
    def sample(self, n=2):
        """为 writer prompt 选取 n 个范文段落"""
        # 从 top exemplars 中采样，偏好近期章节（避免早期过拟合）
        ...
```

**注入方式**：在 writer system prompt 中添加一个 `参考风格` 段，展示 2-3 个本书最佳段落。

**为什么有效**：
- Few-shot exemplars 比指令更有效驱动风格（这是 prompt engineering 的基本发现）
- 来自本书 → 风格一致性；经过质量门 → 不是漂移样本
- 零额外 LLM 调用（段落评分是纯函数）
- 随写作积累 → 范文库越来越好 → 正反馈循环

#### 4.5.2 Writer Prompt 结构（简化）

```
System Prompt:
  ├── 角色定义（固定，~200 chars）
  ├── 体裁规则（从 GENRE_PROFILES，~500 chars）
  ├── 输出格式（散文 + sentinel + delta JSON，~300 chars）
  ├── 禁止列表（体裁相关 + card.forbid，~200 chars）
  └── 敏感词规避（如启用，~200 chars）

User Prompt:
  ├── [voice]       冻结风格宪章         ~1500 chars  ┐
  ├── [world]       世界观 + brief        ~2500 chars  ┤ stable (cached)
  ├── [state]       当前状态 + 线索 + 时间态事实  ~2500 chars  ┐
  ├── [plan]        精炼后 ChapterCard     ~2000 chars  │
  ├── [recent]      上一章尾部 + delta     ~1500 chars  │ volatile
  ├── [retrieved]   card-aware RAG        ~3000 chars  │
  ├── [directives]  feed-forward 建议     ~1000 chars  │
  ├── [exemplars]   2-3 个范文段落         ~1000 chars  ┘
  └── 开写指令 "请写第 N 章"
```

**总上下文**：~14,200 chars（约 7,000 tokens 中文）+ system ~1,400 chars
vs v2 的 ~18,900 chars + system

### 4.6 retrieve.py — Card-Aware RAG（改进）

**动机**：v2 的 TF-IDF 用章节原文做关键词匹配。检索结果与当前章需求的相关性取决于运气。

**改进**：用 ChapterCard 的语义字段驱动检索：

```python
def card_aware_retrieve(card, index, budget=3000):
    """
    用 card 的结构化字段构造检索查询，而非用上一章原文。
    
    查询构造：
    - card.where → 检索涉及该地点的历史段落
    - card.who  → 检索涉及该角色的关键场景
    - card.payoff → 检索该线索的 setup/develop 段落
    - card.turn → 检索类似转折类型的实现方式
    """
    queries = []
    queries.append(card.where)                    # 地点相关
    queries.extend(card.who[:3])                   # 主要角色相关
    if card.payoff:
        queries.append(card.payoff.thread_name)    # payoff 线索的历史段落
    
    results = []
    for q in queries:
        hits = index.search(q, top_k=3)
        results.extend(hits)
    
    # 去重 + 按相关度排序 + 裁剪到预算
    return dedupe_and_trim(results, budget)
```

**仍用 TF-IDF**：不引入 embedding 依赖（保持 `requirements.txt` 只有 `openai>=1.0.0`）。但查询构造从"上一章原文关键词"变为"当前 card 结构字段"，相关度显著提升。

### 4.7 repair.py — 修复（精简，不改核心语义）

**保留**：
- L0 零 LLM 修复（em-dash、碎片行、化石轮换、开场推举、元叙事剥离）
- L1 有界 LLM 修复（扩充、注入对话、em-dash 定向、钩子修订）
- Keep-only-if-improved + 整层回滚语义
- `RESCUE_ATTEMPTS=1`

**变更**：
- L0 新增：句式多样化（当句长 Shannon 熵过低时，机械地拆分/合并句子）
- 合并 `quality_repair.py` + `repair.py` → 单文件 `repair.py`

### 4.8 store.py — 持久化（简化）

**变更**：
- 移除 GateRegistry 的元数据存储（不再有注册表）
- 新增 exemplar bank 持久化
- 新增 feed-forward directives 持久化（JSON 文件，按章号）
- 新增 temporal facts 持久化（在 state delta 中）

### 4.9 config.py — 配置（改进）

**考虑但决定不做的**：替换手工 YAML 解析器为标准库。
**原因**：引入 PyYAML 依赖违反"只依赖 openai"原则。手工解析器的限制是已知的、可控的。

**实际改进**：
- 添加 UTF-8 规范化：在解析前 strip BOM 和 normalize curly quotes → straight quotes（根治弯引号 bug）
- 添加 `validate_config()` 函数：在 `create` 时检查所有必需字段

---

## §5  数据流

### 5.1 单章完整数据流

```
输入:
  config.yaml          → 配置
  prompt.md            → 创意简报
  volume_plan.md       → 卷纲
  arc_cards.json       → arc skeleton（每 10 章更新）
  state.db             → SQLite (事件 + 指标)
  temporal_facts.json  → 时间态事实库
  exemplar_bank.json   → 范文段落库
  directives.json      → 上一章的 feed-forward 建议
  chapters/*.txt       → 已写章节（RAG 索引源）

处理:
  1. refine_card:
     arc_card + temporal_facts + prev_delta + directives → refined ChapterCard
     
  2. project_context:
     ChapterCard + temporal_facts + voice_baseline + world_bible 
     + RAG(card→index) + directives → StoryState (7 段)
     
  3. write_chapter:
     StoryState + exemplar_bank.sample() + system_prompt → LLM call
     → (prose, sentinel, ChapterDelta)
     
  4. verify:
     prose + ChapterCard + temporal_facts → gate_results
     gate_results.blocking → pass/fail
     gate_results.feed_forward → next_directives
     
  5. repair (if needed):
     prose + gate_results.blocking → L0 transforms → L1 rewrites → rescued prose
     
  6. commit:
     prose → chapters/N.txt
     ChapterDelta → temporal_facts (with valid_from=N)
     prose → exemplar_bank.add_from_chapter()
     next_directives → directives.json
     metrics → state.db
     chapter_completed.json (同步写入)

输出:
  chapters/N.txt                    ← 章节文本
  temporal_facts.json (updated)     ← 时间态事实
  exemplar_bank.json (updated)      ← 范文库
  directives.json (updated)         ← 下一章 feed-forward
  state.db (updated)                ← 指标
  review_round0.json                ← FPY 标尺输入（一次写入）
```

### 5.2 跨章数据流

```
Chapter N-1                    Chapter N                     Chapter N+1
    │                              │                              │
    ├── delta ──────────────────→ refine_card                     │
    ├── directives ────────────→ project_context                  │
    ├── prose ─────────────────→ RAG index                        │
    ├── exemplars ─────────────→ exemplar_bank                    │
    │                              │                              │
    │                              ├── delta ──────────────────→ ...
    │                              ├── directives ─────────────→ ...
    │                              ├── prose ──────────────────→ ...
    │                              ├── exemplars ──────────────→ ...
```

每章产出 4 种跨章数据：delta（事实）、directives（建议）、prose（检索源）、exemplars（范文）。每章消费上一章的 4 种数据。闭环，无额外 LLM。

---

## §6  与 v2 对比

### 6.1 调用经济学对比

| 阶段 | v2 | v3 | 变化 |
|------|-----|-----|------|
| Arc planning | 0.1/ch (1/10) | 0.1/ch (不变) | = |
| Card refinement | 0 | 0.2/ch (大多零 LLM) | +0.2 |
| Write | 1.0 | 1.0 | = |
| L0 repair | 0 | 0 | = |
| L1 repair | 0-2 | 0-2 | = |
| Rescue | 0-1 | 0-1 | = |
| **平均总计** | **2.5** | **2.5-2.7** | **+0.2** |

JIT card refinement 增加 ~0.2 call/chapter，但提升 card 与实际状态的一致性 → 预期减少 L1 repair 和 rescue 的触发率 → 净调用数可能持平或微降。

### 6.2 代码量对比

| 模块 | v2 行数 | v3 目标 | 变化 |
|------|---------|---------|------|
| loop → pipeline | 2,043 | 400 | -80% |
| plan | 1,108 | 600 | -46% |
| write | 2,274 | 800 | -65% |
| quality*.py → gates | 5,526 | 1,500 | -73% |
| repair (合并) | 972+? | 600 | -38% |
| store | 844 | 500 | -41% |
| retrieve | 602 | 400 | -34% |
| llm | 1,205 | 800 | -34% |
| config | 780 | 400 | -49% |
| bootstrap | 1,383 | 800 | -42% |
| anchor | 376 | 350 | -7% |
| types | 208 | 250 | +20% |
| state (新) | — | 500 | new |
| **总计** | **~16,349** | **~7,900** | **-52%** |

### 6.3 质量改进预估

| 改进 | 预估 FPY 提升 | 依据 |
|------|-------------|------|
| JIT card refinement | +2-3% | DOME 论文：动态规划 vs 静态规划的一致性提升 |
| Feed-forward advisory | +3-5% | 弱模型忽略 advisory 是已测量的 FPY 损失源 |
| 散文自然度扩展 | +1-2% | 更早检测 AI 味 → L0 修复更多 |
| Card-aware RAG | +1-2% | 更相关检索 → 首稿一致性提升 |
| 时间态事实 | +1-2% | 消除平面状态无法检测的时序矛盾 |
| 阈值梯度 | +1% | 对抗后半段质量下降 |
| Exemplar bank | 难以量化 | 定性提升风格一致性 |
| **总计** | **+9-15%** | 从 87% 到 96-100%（理论上限） |

⚠️ 这些是基于论文数据和工程判断的估算，非实测。任何改动的实际效果必须通过 A/B 验证。

---

## §7  关键不变式（从 v2 继承 + v3 新增）

### 7.1 继承不变式

1. **决策表从上往下逐行求值**，action 后回到表头
2. **`chapter_completed.json` 必须在 commit 中同步写入**（loop-leak 不变式）
3. **`review_round0.json` 一次写入，永不被修复/救章覆盖**（FPY 标尺不变式）
4. **`voice_baseline.md` 冻结**，永不从漂移散文重新派生
5. **GATE_SCOPES: book scope 永远不阻断单章**（防闩不变式）
6. **修复是 keep-only-if-improved + 整层回滚**
7. **`_OUTPUT_SECTION` 精确字符串匹配替换**
8. **prose first, JSON last**（解析失败丢 delta 不丢散文）
9. **500 字最低保存阈值**（provider 拒绝不持久化）
10. **WR anchor 无 cacheable_prefix**（外审不可被上下文染色）

### 7.2 新增不变式

11. **Feed-forward directives 有 expires 字段**，过期后自动移除，不会堆积
12. **Temporal facts 有 valid_from/valid_until**，投影时按章号过滤
13. **Exemplar bank 只收纳通过所有硬门的章节段落**，不收修复/救章的
14. **JIT card refinement Phase 1 是纯函数**，Phase 2 是可选的
15. **阈值梯度因子 ≥ 0.90**（最多收紧 10%，避免后期锁死）

---

## §8  迁移路径

### Phase 1：零风险改进（不改主循环）
1. **散文自然度扩展**：在 `quality.py` 的 `style_health` 中添加新维度。纯加法，不删现有检查。
2. **Feed-forward 机制**：在 `loop.py` 的 commit 阶段将 advisory 结果写入文件；在 context 投影时读入。不改 advisory gate 逻辑。
3. **Config UTF-8 规范化**：在 `config.py:load_config` 入口添加 BOM strip + curly quote normalize。
4. **阈值梯度**：在 `quality.py` 的阈值查询中乘以 `escalation_factor()`。

**验证**：`replay_gates.py` 回放 → FPY 不应下降。

### Phase 2：中等风险改进（改 context + planning）
5. **时间态事实**：扩展 `ChapterDelta` 格式，添加 `valid_from`/`supersedes`。在 state 投影中添加时间过滤。
6. **Card-aware RAG**：修改 `retrieval.py` 的查询构造，从 card 字段提取关键词。
7. **JIT card refinement**：在决策表中添加新行（`needs_card_refinement`, `refine_card`）。
8. **Exemplar bank**：在 commit 阶段收集，在 writer prompt 中注入。

**验证**：fork A/B, 10 章，比较 FPY 和 WR。

### Phase 3：代码精简（重构）
9. **合并 quality*.py → gates.py + repair.py**：移除 GateRegistry 元数据层，用扁平列表替代。
10. **合并 canon.py + types.py → state.py**
11. **精简 loop.py → pipeline.py**：将 StoryState 投影移入 state.py，将验收移入 gates.py。
12. **精简 write.py**：将体裁常量移入配置文件。

**验证**：全量回归测试 + `fpy_prime.py` 对比。

---

## §9  外部对标

### 9.1 与最接近竞品的功能对比

| 能力 | 本项目 v2 | 本项目 v3 | Tianming | ainovel-cli | autonovel | OpenWrite |
|------|----------|----------|---------|------------|-----------|-----------|
| 确定性决策表 | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| 单调用写章 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 零 LLM 验收门 | ✅ | ✅ | ✅ (6 门) | ✅ | 部分 | ❌ (37 维 LLM) |
| 时间态事实 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Feed-forward advisory | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 散文自然度多维检测 | 3 项 | 8+ 项 | ❌ | 4 维诊断 | 正则+LLM | ❌ |
| Card-aware RAG | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Exemplar bank | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Prompt cache 分割 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 盲测 WR anchor | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| FPY 测量纪律 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Keep-only-if-improved 修复 | ✅ | ✅ | ❌ | ❌ | ❌ | 回滚 |
| 伏笔 DAG | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 逆序叙事 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 9.2 本项目独有能力（全球范围）

经过对 30+ 开源项目的完整调研，以下能力在所有已知项目中未发现：

1. **FPY 测量纪律** — 可重放的验收指标 + A/B 定型协议
2. **style_health 客观锚** — 防止分数膨胀的机械指标
3. **有界修复阶梯 + keep-only-if-improved + 整层回滚** — 修复永不恶化
4. **盲测 WR anchor** — 唯一不可自授的外部质量信号
5. **Prompt-cache-aware stable/volatile 分割** — 成本最优的上下文架构
6. **确定性决策表 + 离线重放** — 所有路由可在无 LLM 情况下验证
7. **进程级每小说隔离** — 零代码修改的多小说并行

v3 新增的全球首创：
8. **Feed-forward advisory loop** — advisory 结果结构化注入下一章 prompt
9. **Exemplar bank from own best output** — 数据驱动的风格锚定
10. **阈值梯度** — 后半段自动收紧质量门

---

## §10  风险与待决

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| JIT card refinement 增加的 0.2 call/ch 不被 FPY 提升抵消 | 中 | Phase 2 A/B 验证，可回滚 |
| Feed-forward directives 堆积过多撑爆预算 | 低 | expires 字段 + 预算硬帽 1000 chars |
| Exemplar bank 早期无数据（前 5 章） | 低 | 前 5 章不注入 exemplar，等积累 |
| 散文自然度新维度误杀（假阳性） | 中 | Phase 1 先以 advisory 模式运行，测量 firing rate 后再提升为 blocking |
| 时间态事实的 delta 格式变更破坏 v2 checkpoint | 高 | 向后兼容：旧 delta 无 `valid_from` 时默认 `valid_from=chapter_num, valid_until=None` |
| 代码精简过程引入回归 | 中 | Phase 3 逐模块重构，每步 fpy_prime 验证 |

---

## §11  结论

v2 的核心架构（决策表 + 单调用写章 + 零 LLM 验收 + 有界修复）经过 A/B 定型，是正确的。v3 不推翻这个架构，而是在三个方向上深化：

1. **从"事后检测"到"事前预防"**：feed-forward advisory → 下一章 prompt 约束 → 首稿更好 → 修复更少
2. **从"平面状态"到"时间态状态"**：temporal facts → 上下文更精确 → 一致性错误更少
3. **从"指令驱动"到"数据驱动"**：exemplar bank + card-aware RAG → 每次 LLM 调用的信息密度更高

预期收益：FPY +9-15%、代码 -52%、调用数持平。
验证方式：三阶段渐进迁移，每阶段 A/B 验证 + fpy_prime 回放。
