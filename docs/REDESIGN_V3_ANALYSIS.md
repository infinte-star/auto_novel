# v3 第一性原理分析：AI 长篇小说 Pipeline 重设计

> **立场声明**：本文不做「推翻重来」——v2 已经是一次成功的第一性原理重设计
> （22.7→2.5 调用/章，prompt −97%，FPY′ 81.8%→92.0%，删 7.3k 行）。
> 本文做的是：**站在 v2 已证明的事实上，问「还能再削掉什么」。**

---

## 0. 方法论：三层分析

| 层 | 问什么 | 工具 |
|---|---|---|
| **原理层** | 这条路径存在的理由是什么？它解决的问题还在不在？ | 第一性原理 |
| **度量层** | 它在生产数据上挣到了什么？代价是什么？ | fpy_prime / replay_gates / llm_calls.jsonl / 行数 |
| **简化层** | 它能被删掉/合并/内联而不损失已测量的收益吗？ | 代码阅读 + AST 引用分析 |

**原则优先级**：质量 > 首过率 > 简单性。
但「简单性」不是次要目标——**每一行代码都是未来 bug 的温床**，
28k 行非测试代码服务一条 2.5 调用/章的流水线，本身就是度量上可见的风险。

---

## 1. 当前系统的真实画像（v2 已落地，2026-07-28）

### 1.1 已证明的强项（不要动的东西）

| 设计决策 | 实测证据 | 评判 |
|---|---|---|
| 决策表路由（零 LLM 主循环） | 可穷举测试，行为 100% 可复现 | **核心支柱，不动** |
| 一次写作调用出 prose + delta | 22.7→2.5 调用/章，prompt −97% | **核心支柱，不动** |
| CCC 契约兑现校验（零 LLM） | CCR 95.5%，30/30 章通过三个硬字段 | **核心支柱，不动** |
| L0/L1 修复梯（fix-before-rewrite） | 29/647 章修复，0 变差；L0 零 LLM | **核心支柱，不动** |
| cite-or-drop（举证式过滤） | 3/3 幻觉被拦，零假阳 | **核心支柱，不动** |
| StoryState 投影（stable/volatile 分裂） | 15k 上下文，prompt cache 命中 | **核心支柱，不动** |
| 弧级双层滚动规划 | 每 10 章 1 次高推理调用，替代 6.9 调用/章的委员会 | **核心支柱，不动** |
| `review_round0.json` 归档不可变性 | FPY′ 可回放、可跨引擎结算 | **核心支柱，不动** |
| self-score 无返工权 | §0 的核心论点，已用 A/B 验证 | **核心支柱，不动** |

### 1.2 度量仪表盘

```
代码量（非测试）：27,943 行  ←  v2 设计文档预估 4,200 行（6.6×）
  v2/ 引擎内核:    4,764 行  ←  与预估大致吻合
  顶层共享模块:   18,971 行  ←  v1 遗产，是内核的 4 倍
  tools/:          4,208 行

配置键:           ~364 个   ←  v2 设计文档预估 ~40 个（9×）
注册门:             23 个   ←  v2 设计文档预估 10 个（2.3×）
LLM 调用/章:       2.50     ←  达标
prompt 字符/章:   41,216    ←  达标（目标 45k）
FPY′:              92.0%    ←  达标（目标 ≥90%）
CCR:               95.5%    ←  达标
WR:           不劣率 58%    ←  达标但 n_decisive=5
```

**核心失配**：内核做到了设计预估，但它坐在一堆 4 倍于自身的遗产代码上。

### 1.3 真正的瓶颈在哪里

#### A. 模块适配器模式（最大的结构债务）

v2 设计为 6+2 个模块；实际是 v2/ 的 7 个文件 + 顶层 18 个 .py。
原因：v2 不是替换，而是**包装**。

```
v2/write.py (679行)  →  imports  →  writing.py (2364行)    # 写作教条
v2/beat.py  (623行)  →  imports  →  arc.py (577行)          # 卡片词汇
v2/canon.py (711行)  →  imports  →  memory.py (1371行)      # bootstrap + 窗口
v2/accept.py (895行) →  imports  →  quality.py (4054行)     # 23个门
v2/repair.py (253行) →  imports  →  fix.py (788行)          # 修复梯
v2/run.py (1063行)   →  imports  →  store/config/checkpoint # 持久化
```

**每一对都是同一个模式**：v2 模块是薄适配器，真正的逻辑在顶层 v1 遗产模块里。
这不是过渡态——v1 已删 4 个月，这个结构已经固化。

#### B. quality.py 的体量（4054 行 = 最大单文件）

23 个注册门，但 v2 的验收集只用 8 个 + 2 个 v2 原生检查。
剩下的 13 个要么是 advisory（只产 directives），要么是 card-phase，
要么已判为死门。**每个门的注册元数据 + 实现 + 测试**占 ~150-200 行。

advisory 门的投入产出：它们产出 directives，但 directives 是 prompt 里的额外文字，
没有实测证据证明「有 directives 的章比没有的写得好」——这和自评打分是同一个不可判定性。

#### C. 配置膨胀（364 个键 vs 40 个需要的）

v2 不读其中大约 300 个键。它们留下的原因：
- 老 novels/\*/config.yaml 仍然带着它们
- config_template.example.yaml 是追踪的文件
- 每个键都有 CLAUDE.md 里的一段历史说明

这不是「无害的注释」——**每个未使用的键都是一个误导信号**，
让维护者以为有人在读它，花时间理解一个不存在的行为。

#### D. Canon 检查的 LLM 调用（当前零价值）

实测 30/30 章：1 章产出 3 条结论，全是幻觉，全被 cite-or-drop 拦截。
29 章零结论。这个调用存在的理由是「cite-or-drop 的防御力需要有东西来测试它」，
但一个只产出幻觉又被全数拦截的调用，**既不改善质量也不改善首过率**。

成本：每章 1 次 low 级调用 × 数百章 = 不可忽略的 token + 延迟。

#### E. memory.py 的职责稀释（1371 行只做 4 件事）

v2 真正需要的：
1. `bootstrap` — 唯一使用，不可删
2. `volume_plan_window` — v2/beat.py 调用
3. `volume_transition_directive` — v2/beat.py 调用
4. `cacheable_prefix` / `memory_context` — arc.py、trial.py、package.py 调用

其中 #4 是 v1 的上下文构建器，v2 用 canon.py 替代了。
剩余调用者（arc.py、trial.py、package.py）是辅助工具，不是主循环。

---

## 2. 外部实践调研

### 2.1 开源 AI 写作框架全景（2025-2026，按成熟度分层）

#### Tier 1: 生产级框架（1000+ stars）

| 项目 | Stars | 核心架构 | 与本系统的关系 |
|---|---|---|---|
| **InkOS** (Narcooo/inkos) | 8.4k | 10-agent 顺序流水线（Radar→Planner→Composer→Writer→Observer→Reflector→Normalizer→Auditor→Reviser），7 个 truth files，v2 "input governance" 模式 | **最接近的竞品**，但 ~10 次调用/章 vs 本系统 2.5 次；无确定性门、无 CCC、无 FPY′ |
| **oh-story-claudecode** | 4.7k | Claude Code 技能包（非独立引擎），13 个技能覆盖扫榜→拆文→写作→去 AI 味，8 个自动 hook，100+ 参考方法论文件按需加载 | 不是竞品——是工具包。其「去 AI 味」作为一等能力的做法值得借鉴 |
| **AI-Novel-Writing-Assistant** | 2.1k | Web UI，auto-director → planning → preparation → execution → repair，per-model routing，SQLite + optional Qdrant RAG | 架构处于本系统 v1 阶段。per-model routing 已有 |
| **AI_NovelGenerator** | high activity | GUI，settings→directory→chapters + 向量语义搜索 + 矛盾自动校对 | 本系统已具备（retrieval.py + canon check） |
| **NovelForge** | ~1k | Schema-first 卡片 + JSON Schema AI 校验 + 知识图谱 | v2 的 ChapterCard 已实现同等概念 |

#### Tier 2: 架构独特的项目

| 项目 | 核心亮点 | 可借鉴什么 |
|---|---|---|
| **tianming-novel-ai-writer** (368★, C#) | **最接近本系统设计哲学的竞品**：15 维度事实快照 + 12 种变更声明 + 6 个确定性校验门 + `---CHANGES---` 分隔符 + 本地 ONNX 语义搜索。号称 3000+ 章一致性 | 其 15 维事实快照 = 本系统的 ChapterDelta，6 门 = 本系统的验收集。**验证了确定性门 + 结构化状态写回的路线，但无 CCC / cite-or-drop / repair ladder** |
| **Openwrite** (323★) | 四级大纲（总→篇→节→章），伏线 DAG 管理，三层风格合成，两角色架构（Goethe 规划 / Dante 执行） | **伏线 DAG** 比平面 markdown 更结构化，值得考虑 |
| **show-me-the-story** (398★, Go) | 单 Go 二进制 + Svelte Web UI，outline→逐章写作+review+factcheck+全书 polish | 架构简洁但无确定性门 |

#### Tier 3: 学术实现

| 项目/论文 | 核心贡献 | 对本系统的意义 |
|---|---|---|
| **LongWriter** (THUDM, ICLR 2025) | 证明 LLM 输出长度受 SFT 训练数据限制而非架构限制；AgentWrite 分段写作 | 本系统逐章生成天然规避了输出长度问题 |
| **DOME** (NAACL 2025) | 动态层次大纲展开 + 时序知识图谱记忆 | 弧级双层滚动 ≈ DHO 的 just-in-time 展开思想 |
| **SuperWriter** (ACL Findings 2026) | Plan→Write→Refine + Hierarchical DPO + MCTS，7B 模型 WritingBench 8.51（仅次于 DeepSeek-R1 671B） | 验证 Plan-Write-Refine 三步管道的有效性 |
| **StoryWriter** (清华, 2025.06) | 多 agent + 非线性叙事策略 + LONGSTORY 数据集 | 一致性仍随长度退化——证实了本系统用确定性 canon 投影的路线 |
| **WriteHERE/HRP** (2503.08275) | HTN 递归分解 + 异构认知任务类型化 | 部分采纳任务类型化；递归不适合网文固定节拍 |

**结论**：在调研到的所有项目中，本系统在以下 6 个方面是**独有的**：
1. **单调用 prose+delta** — 无竞品实现同等紧凑度
2. **确定性验收集 + scope 强制** — tianming 有 6 门但无 scope/proof 约束
3. **修复梯 + keep-only-if-improved 回退** — 无竞品实现
4. **cite-or-drop 举证过滤** — 无竞品实现
5. **盲位置去偏 pairwise 评审** (anchor.py) — 学术界有类似思路但无位置去偏
6. **stable/volatile 分裂显式优化 prompt cache 命中** — 无竞品实现

### 2.2 学术界的关键发现

| 发现 | 来源 | 对本系统的意义 |
|---|---|---|
| LLM 自评系统性偏好机器文本 | StoryAlign (ICLR 2026) | **已内化**：self-score 无返工权 |
| 无外部反馈的 self-correction 不可靠 | 2406.01297 | **已内化**：cite-or-drop + 确定性门 |
| pairwise 比 pointwise 更接近人类一致性 | LLM-as-Judge 综述 | **已内化**：WR 用 pairwise |
| rubric 总分会被 reward-hack | 2606.04923 | **已内化**：不打总分 |
| 最小高信号 token 集 | Anthropic context engineering | **已内化**：StoryState 15k |
| 结构化约束输出优于自然语言输出 | JSON Schema / function calling 实践 | **部分采纳**：delta 用 JSON schema |
| 一致性随长度不可避免地退化 | StoryWriter (2025), IS-CoT (2026) | **已内化**：确定性 canon 投影而非模型记忆 |
| 动态层次大纲优于静态全量大纲 | DOME (NAACL 2025) | **已内化**：弧级双层滚动 |

**结论**：v2 的设计已经内化了 2024-2026 的主要学术发现。
没有新的学术成果能论证一个根本性的架构变更。

### 2.3 从竞品中可借鉴的新方向

| 方向 | 来源 | 具体做法 | 预期收益 | 风险 |
|---|---|---|---|---|
| **去 AI 味作为一等能力** | oh-story-claudecode, InkFlow | 专门的 AI-flavor 检测 + 替换规则，而非仅靠 style_health | 更自然的散文 | 增加复杂度 |
| **实体/描述一致性门** | tianming (gate 5) | 检查角色外貌描述是否匹配 characters 档案 | 捕获 CCC 遗漏的描述漂移 | 需要结构化的角色档案 |
| **伏线 DAG** | Openwrite | 把 `memory/threads.md` 从平面 markdown 升级为有向无环图（planted→collected→abandoned） | 更精确的伏线过期检测 | 增加状态管理复杂度 |
| **确定性 canon 替代 LLM canon** | 本分析原创 | 用代码检查人物在场/时间线/地点一致性，替代当前零价值的 LLM canon check | 省 ~0.8 LLM/章，更可靠 | 覆盖面可能不如 LLM |
| **结构化输出强制** | 业界趋势 | 用 `response_format: {type: "json_schema"}` 替代手工 JSON 解析 | 消除 delta 解析失败 | 部分网关不支持 |

---

## 3. 第一性原理重设计：v3 架构

### 3.0 设计约束（不可妥协）

从 v2 的度量和 A/B 结果中提炼的硬约束：

1. **self-score 不进任何门、不进任何判据**（§0 的核心论点）
2. **验收集的每个门必须是零 LLM 可判定 + 本章可行动**（scope invariant）
3. **round0 归档不可变**（FPY′ 可回放性）
4. **修复先于重写**（repair before rewrite ordering）
5. **prose first, JSON last**（退出钩子是最重要的一句话）
6. **stable prefix 字节稳定**（prompt cache invariant）
7. **单一依赖**（openai>=1.0.0，无 numpy/torch/embedding 库）

### 3.1 目标架构：极简四步流水线

**一句话**：**Plan → Write → Check+Fix → Commit，3 个文件，~3000 行引擎核心。**

```
BRIEF (prompt.md)
     │
     ▼  bootstrap ×1
CANON (story_state.db, append-only)
     │
     │  纯函数投影 → StoryState (~15k chars)
     ▼
┌─────────────────────────────────────────┐
│  每 arc_span 章:                         │
│  (1) PLAN — 弧级双层滚动规划 (1 call)     │
│       → ChapterCard × N + next_arc 骨架  │
│       → 确定性卡片校验 (0 call)            │
│       → 若 CRITICAL: 卡片修复 (1 call)    │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│  每章:                                    │
│  (2) WRITE — 1 call: prose + ChapterDelta │
│  (3) CHECK+FIX:                           │
│       a. L0 确定性修复 (0 call)            │
│       b. 验收 (0 call, 8 门 + CCC)        │
│       c. 若 L1 needed: 定点修复 (≤1 call)  │
│       d. 验收复检 (0 call)                 │
│       e. 若仍阻塞且可救: rescue (1 call)   │
│  (4) COMMIT — append delta, 存档, 下一章   │
└─────────────────────────────────────────┘
```

**与 v2 的关键差异**：
- **删除 canon LLM 检查**（当前零价值，用确定性检查替代）
- **合并模块**（消除适配器模式）
- **简化门集**（23 → 8+2，advisory 门降为离线工具）
- **配置键 364 → ~50**

### 3.2 模块职责（3+3 个模块，~6,000 行）

| 模块 | 预估行数 | 职责 | 来源（合并了什么） |
|---|---|---|---|
| **`engine/plan.py`** | ~800 | 弧级规划 + ChapterCard schema + 卡片校验修复 + 双层滚动 + 指纹 | `v2/beat.py` + `arc.py` |
| **`engine/write.py`** | ~1200 | prose doctrine + prompt 装配 + 1-call 输出 + delta 解析 + 范例锚定 | `v2/write.py` + `writing.py`(核心部分) |
| **`engine/loop.py`** | ~1000 | 决策表 + check/fix/commit + StoryState 投影 + 验收集 | `v2/run.py` + `v2/accept.py` + `v2/canon.py` + `v2/repair.py` |
| **`quality.py`** | ~1500 | 门注册 + 8 个活跃门实现 + L0/L1 修复 | `quality.py`(活跃部分) + `fix.py` |
| **`infra.py`** | ~1000 | LLM 客户端 + config + store + checkpoint + RAG | `llm.py` + `config.py` + `store.py` + `checkpoint.py` + `retrieval.py` |
| **`bootstrap.py`** | ~500 | 一次性初始化 + 卷纲窗口 + 过渡指令 | `memory.py`(bootstrap + 两个窗口函数) |
| **合计** | **~6,000** | | |

**对比**：

| | v1 | v2 (现状) | v3 (提案) |
|---|---|---|---|
| 引擎代码 | 27,215 行 / 24 模块 | 27,943 行 / 25 模块 | ~6,000 行 / 6 模块 |
| 配置键 | 512 | ~364 | ~50 |
| 注册门 | 35 | 23 | 10 |
| LLM 调用/章 | 22.7 | 2.5 | **~1.5** |

### 3.3 各模块详细设计

#### 3.3.1 `engine/plan.py` — 弧级规划

**合并 `arc.py` + `v2/beat.py`**。理由：`arc.py` 的 577 行（ARC_SYSTEM, normalize_card,
validate_card, card_to_plan, load/save_cards）只有 `v2/beat.py` 是活跃消费者。
`card_to_plan` 被 `quality.py` 的 `scene_similarity` 和 `payoff_beat_density` 调用，
但那是引用，不是拥有——放在 plan.py 导出即可。

**删除/简化**：
- `arc.plan_from_arc` — v1 回退路径，已无调用者
- `volume_plan_window` / `volume_transition_directive` — 从 memory.py 迁入
- 弧级指纹注入 — 从当前的 `quality.fingerprint_avoidance_context` + `store.store_chapter_fingerprint` 清理

#### 3.3.2 `engine/write.py` — 写作引擎

**合并 `v2/write.py` + `writing.py` 的核心部分**。

`writing.py` 的 2364 行里，v2 写作真正调用的：
- `_build_write_system` — 写作 system prompt 构建（GENRE_PROFILES 等）
- `save_chapter` / `update_structured_state` / `update_state_file` — 持久化
- `_prewrite_quality_contract` / `_preflight_negative_list` — 写前约束
- `fossil_tail_anchor` — 化石尾部锚
- 共享常量（`ANTI_FRAGMENT_BAN`, `_OUTPUT_SECTION` 等）

不再调用的：
- `write_chapter` 本身（v2 有自己的写作循环）
- `extract_events` / `extract_state_changes`（v2 用 delta）
- `_attempt_write` / `_finalize_chapter`（v1 写作循环）

**行动**：把活跃函数迁入 `engine/write.py`，废弃函数标记或移到 `legacy/`。

#### 3.3.3 `engine/loop.py` — 主循环

**合并 `v2/run.py` + `v2/accept.py` + `v2/canon.py` + `v2/repair.py`**。

理由：这四个模块在 v2 内部的耦合度非常高——
`run.py` 的每一行决策都直接调用 accept/canon/repair 的接口。
把它们分开是为了"v2 的四个机制各一个文件"的组织清晰度，
但实际上它们共享 `ChapterRun` 状态、共享 `Corpus` 上下文、
共享 `recheck` 闭包。**合并不会损失清晰度，因为它们本就是一个紧密协作的单元。**

**关键改动——删除 canon LLM 检查**：

```
现状: need_draft → need_report → l0_pending → l1_pending → canon_pending → next_card_patch → rescue → commit
提案: need_draft → need_report → l0_pending → l1_pending → rescue → commit
```

`canon_pending` 和 `next_card_patch` 在 30/30 实测中产出零有效结论。
确定性替代方案：
- 人物在场检查：卡片 `who` 的人名必须出现在正文（**已由 CCC 覆盖**）
- 时间线检查：通过 StoryState 的 `recent` 段可判定（章序矛盾 = 引用了未来事件）
- 地点一致性：卡片 `where` vs 正文（**已由 CCC 覆盖**）

**CCC 已经覆盖了 canon check 试图做的大部分事情**。
剩余的「开了一个世界事实但没破坏卡片字段」的情况，留给 advisory directives。

#### 3.3.4 `quality.py` — 门与修复

**从 4054 行精简到 ~1500 行。**

保留的 8 个验收门（与 v2 `ACCEPTANCE_GATES` 相同）：

| 门 | 修复层 | 保留理由 |
|---|---|---|
| `style_health` | L0 | 文风客观锚，抓电报体塌缩 |
| `cross_chapter_repetition` | L0 | 跨章化石检测 |
| `book_wide_fossils` | L0 | 全书化石检测（book scope, 不阻塞） |
| `descriptor_frequency` | L0 | 描述词频过高 |
| `adjacent_repetition` | — | 相邻章重复，有实证正例 |
| `length_band_check` | L1 | 长度带，短侧有 expand_to_band |
| `opening_hook_gate` | L0 | 开场钩子强度 |
| `genre_adherence` | — | 体裁信号偏移（仅限 reject_enabled） |

\+ 2 个 v2 原生检查（`contract_fulfilment` + `citation_check`）= **10 个**。

**删除/降级的 13 个门**：

| 门 | 处置 | 理由 |
|---|---|---|
| `ai_flavor_health` | 降为离线工具 | advisory, 无阻塞权 |
| `paragraph_shape_health` | 降为离线工具 | advisory, v2 窗口 fire 0% |
| `prose_texture` | 降为离线工具 | advisory, v2 窗口 fire 0% |
| `shareable_line` | 降为离线工具 | advisory, v2 窗口 fire 0% |
| `information_density` | 降为离线工具 | advisory, 重设计后才有意义 |
| `long_span_fatigue` | 降为离线工具 | advisory, 只剩 1 项信号 |
| `intra_chapter_repetition` | 降为离线工具 | 1/638 firing, 无修复动作 |
| `hook_tail_repetition` | 降为离线工具 | 1/638 firing, 阈值未校准 |
| `payoff_beat_density` | 降为离线工具 | advisory |
| `beat_coverage` | 删除 | 被 CCC 替代 |
| `dialogue_health` | 保留为 L1 触发器 | 不在验收集但触发 inject_dialogue |
| `scene_similarity` | 保留在卡片阶段 | card-phase 门 |
| `narrative_pattern_repetition` | 保留在卡片阶段 | card-phase 门 |

**合并 `fix.py` 进 `quality.py`**：修复逻辑和门逻辑是一一对应的（`ACTION_BY_GATE`），
分开放只增加了跨文件查找成本。`fix.py` 的 788 行里，
核心修复动作（em-dash reduction, fossil rotation, expand_to_band, inject_dialogue）
加上 planning/dispatching 逻辑合并后约 600 行。

#### 3.3.5 `infra.py` — 基础设施

**合并 `llm.py` + `config.py` + `store.py` + `checkpoint.py` + `retrieval.py`**。

这 5 个模块合计 4,701 行，但它们是纯基础设施，相互低耦合。
合并的目标不是减行数（基础设施的行数是合理的），而是**减 import 数和模块数**。
如果不合并，至少应该放在一个 `infra/` 包内。

实际上更实际的做法：**保持 5 个文件但放在 `infra/` 下**，减少顶层文件数。

#### 3.3.6 `bootstrap.py` — 初始化

**从 `memory.py` 的 1371 行中提取 ~500 行。**

保留：
- `bootstrap()` — 一次性生成 state.md + memory/*.md
- `extract_contract()` — 契约提取（必须在 bootstrap_chain 之前）
- `volume_plan_window()` — 卷纲窗口化
- `volume_transition_directive()` — 过渡指令

删除（已被 v2/canon.py 替代）：
- `cacheable_prefix()` / `memory_context()` / `lite_memory_context()` — v2 用 StoryState
- `MEMORY_COMPRESS_SYSTEM` 及相关压缩逻辑 — v2 无记忆压缩
- ThreadPoolExecutor 引用 — v2 无后台线程池
- 四层上下文构建器

`cacheable_prefix` 和 `memory_context` 仍被 arc.py / trial.py / package.py 调用。
处置：`trial.py` 和 `package.py` 是辅助工具，它们应该改用 StoryState 投影。

### 3.4 关键机制变更

#### 变更 1：删除 canon LLM 检查（省 ~0.8 调用/章）

**现状**：`v2/run.py` 的决策表有 `canon_pending` + `next_card_patch` 两行，
调用 `canon.canon_check()` 做举证式核对。

**实测**：30/30 章跑了，1 章出 3 条结论，**全是幻觉**被 cite-or-drop 拦。
29 章零结论。REDESIGN_V2 §9.7 明确记为「限制，不记为胜利」。

**提案**：

```python
# 替代方案：确定性 canon 一致性检查（零 LLM）
def deterministic_canon_check(text: str, state: StoryState, card: dict) -> list[str]:
    """检查正文与已知 canon 的确定性矛盾。"""
    issues = []
    # 1. CCC 已覆盖：who/where/turn/exit_hook/forbid
    # 2. 新增：人物称谓一致性（正文中出现的人名是否都在 characters 里）
    # 3. 新增：时间标记不矛盾（"三天前"不应该指向未来事件）
    # 4. 新增：地点距离合理性（两章之间不应瞬移千里）
    return issues
```

**反证条件**：如果某本新书因为缺少 LLM canon check 而出现可检测的 canon breach
（在 CCC 和确定性检查都放过的情况下），恢复 canon check。

#### 变更 2：advisory 门降为离线工具

**现状**：9 个 advisory 门在每章运行，产出 directives 注入写作 prompt。
没有实测证据证明这些 directives 改善了写作质量。

**提案**：
- 验收集只留 8+2 个有阻塞权或修复动作的门
- 原 advisory 门移到 `tools/style_audit.py`，作为离线分析工具
- 如果未来需要，可以选择性地恢复特定 advisory 门

**风险**：directives 可能有隐性的质量贡献（无法度量但存在）。
缓解：在一本新书上做 A/B（有 directives vs 无 directives），用 WR 结算。

#### 变更 3：配置键精简（364 → ~50）

**提案**：新的配置结构：

```yaml
api:                          # ~15 个键
  base_url, api_key, api_keys, api_key_groups, model
  max_tokens, temperature, stream
  max_attempts, max_rpm
  stream_timeout, stream_idle_timeout
  metrics_enabled

novel:                        # ~10 个键
  genre, target_words, max_chapters
  arc_span, engine
  sensitive_word_avoidance
  package_after_complete, refine_after_complete

acceptance:                   # ~15 个键
  chapter_min_chars, chapter_max_chars
  style_penalty_block, style_em_dash_per_kchar_bad
  style_min_avg_sentence_chars, style_fragment_line_ratio_max
  cross_repeat_reject_count, adjacent_repeat_block_threshold
  book_fossil_hard_ratio, book_fossil_chapter_frac
  quality_breaker_consecutive, rescue_attempts
  fix_max_l1_calls

role_routing:                 # ~10 个键 (per-role overrides)
  writing_base_url, writing_model, writing_max_tokens
  planning_base_url, planning_model
  # ... etc

paths:                        # ~10 个键 (per-novel isolation)
  book, chapters_dir, logs_dir, memory_dir, checkpoints_dir
```

**其余 ~300 个键**：
- 已无读者的键：直接删除
- v2 引擎不需要的 v1 键：从模板删除，在代码中用硬编码默认值替代
- advisory 门的阈值键：随 advisory 门降级一起删除

### 3.5 数据流（简化后）

```
prompt.md (不可变)
    │
    ▼  bootstrap ×1（~3 LLM calls）
story_state.db (append-only，唯一真相源)
    ├── memory/*.md (bootstrap 产物，不可变)
    ├── state.md (bootstrap 产物，每章更新)
    │
    │  StoryState = pure_function(db + memory files)
    │  stable_key = sha1(brief + facts + voice + route)
    ▼
┌──────────── 每 arc_span 章 ────────────┐
│  plan.generate_arc(state, skeleton)     │
│  → cards[N] + next_arc_skeleton         │
│  → validate_card → repair if CRITICAL   │
└────────────────┬────────────────────────┘
                 ▼
┌──────────── 每章（~1.5 calls 均值）─────┐
│  write.chapter(state, card, rag)        │ ← 1 call
│  → prose + ChapterDelta                 │
│                                          │
│  quality.check(text, card, corpus)      │ ← 0 calls
│  → report: {gate_results, CCC, blocks}  │
│                                          │
│  if L0_needed:                           │
│    quality.fix_l0(text, report)          │ ← 0 calls
│    quality.recheck(text, ...)            │
│                                          │
│  if L1_needed:                           │
│    quality.fix_l1(text, report, client)  │ ← ≤1 call
│    quality.recheck(text, ...)            │
│                                          │
│  if still_blocked and rescue_ok:         │
│    write.chapter(state, card, rag)       │ ← 1 call (rare)
│                                          │
│  commit(text, delta, report)             │ ← 0 calls
└──────────────────────────────────────────┘

每 K 章（离线，不阻塞流水线）：
  tools/pairwise_ab.py --anchor           ← 外部质量锚
  tools/style_audit.py                    ← 原 advisory 门
```

### 3.6 对比总结

| 维度 | v1 | v2 (现状) | v3 (提案) |
|---|---|---|---|
| **哲学** | LLM 做决策 + 自评驱动 | 确定性决策 + 可判定验收 | **同 v2，但更瘦** |
| **LLM 调用/章** | 22.7 | 2.5 | **~1.5** |
| **prompt 字符/章** | 1,564k | 41k | **~35k** |
| **引擎代码** | 27,215 行 / 24 模块 | 27,943 行 / 25 模块 | **~6,000 行 / 6 模块** |
| **配置键** | 512 | ~364 | **~50** |
| **活跃门** | 35 | 23 (8 验收+2 原生+13 advisory) | **10 (8 验收+2 原生)** |
| **Canon 检查** | LLM factcheck | LLM cite-or-drop (零有效结论) | **确定性 (CCC 已覆盖)** |
| **FPY′** | 12-63% (噪声) | 92.0% | **≥92%** (不降) |
| **质量指标** | self-score | WR (pairwise) | **同 v2** |
| **外部锚** | 无 | 架构就绪，无数据 | **优先建数据** |

---

## 4. 优先级排序与实施路径

### 4.0 最高优先级：建立外部锚数据

v2 的全部质量度量框架建立在一个**还不存在的数据基础**上：
`benchmarks/anchor/` 是空的，WR 无法常规测量。

**在做任何代码重构之前，先做这件事**：
1. 挑选 3-5 个人写的高质量中文网文章节，放入 `benchmarks/anchor/`
2. 跑一次 `pairwise_ab --anchor`，建立 WR 基线
3. 之后每次代码变更都可以用 WR 结算

### 4.1 Phase 1: 模块合并（最大收益，最低风险）

| 合并 | 行数变化 | 风险 |
|---|---|---|
| `arc.py` → `engine/plan.py` | −577, +0 (迁入 beat.py) | 低：纯搬运 |
| `v2/write.py` + `writing.py`(核心) → `engine/write.py` | −3043, +1200 | 中：需识别活跃 vs 废弃代码 |
| `v2/run.py` + `v2/accept.py` + `v2/canon.py` + `v2/repair.py` → `engine/loop.py` | −3632, +1000 | 低：v2 内部合并 |
| `quality.py`(活跃) + `fix.py` → `quality.py` | −4842, +1500 | 中：需保持门的测试覆盖 |

**判据**：所有 ~800 个测试仍然通过 + `fpy_prime` 读数不变。

### 4.2 Phase 2: 删除零价值路径

| 变更 | 收益 | 判据 |
|---|---|---|
| 删 canon LLM 检查 | −0.8 调用/章 | FPY′ 不降 + 新书无 canon breach |
| advisory 门降级 | 代码 −1500 行, prompt −500 字符 | WR A/B |
| 配置键精简 | 维护成本降低 | 功能等价 |

### 4.3 Phase 3: 长期优化

| 方向 | 收益 | 条件 |
|---|---|---|
| 结构化输出（JSON schema mode） | 消除 delta 解析失败 | 网关支持 |
| 确定性 canon 扩展 | 更多一致性检查，零 LLM | 需要定义检查集 |
| 流式验证 | 早期中止低质量输出 | 需要流式解析基础设施 |
| embedding RAG 替代 TF-IDF | 更精准的历史检索 | **违反单依赖约束**，除非自实现 |

---

## 5. 诚实的收益预估

### 可预期的收益（有数据支撑）

| 指标 | 现状 | v3 后 | 依据 |
|---|---|---|---|
| 引擎代码量 | 27,943 行 | ~6,000 行 | 合并计划 |
| 模块数 | 25 | 6 | 合并计划 |
| 配置键 | ~364 | ~50 | 键使用审计 |
| LLM 调用/章 | 2.5 | ~1.5 | 删 canon check |
| 维护认知负荷 | 25 文件跳转 | 6 文件 | 结构简化 |

### 不可预期的收益（需要 A/B 验证）

| 主张 | 反证条件 |
|---|---|
| 删 advisory directives 不损质量 | WR 下降 |
| 删 canon LLM check 不损一致性 | 出现 CCC 无法覆盖的 canon breach |
| prompt 缩短改善写作质量 | FPY′ 下降 |

### 已知风险

| 风险 | 缓解 |
|---|---|
| 合并过程中引入 bug | 逐步合并，每步全量测试 |
| advisory directives 有隐性价值 | 先 A/B，后删除 |
| 老 novels/\*/config.yaml 不兼容 | 写迁移脚本或保持向后兼容读取 |
| tools/ 的 import 路径变更 | 保持 `quality.` 和 `fix.` 的公共 API 稳定 |

---

## 6. 一句话总结

**v2 把一条 22.7 调用/章的自指闭环变成了一条 2.5 调用/章的确定性流水线，
但它坐在 4 倍于自身的遗产代码上。
v3 不是新架构——是把 v2 内核从遗产中解放出来，
从 28k 行收到 6k 行，从 364 个配置键收到 50 个，
从 2.5 调用/章收到 1.5 调用/章。
先做的第一件事不是写代码，是往 `benchmarks/anchor/` 放 5 篇人写参考章——
没有外部锚，任何质量声明都是自指。**
