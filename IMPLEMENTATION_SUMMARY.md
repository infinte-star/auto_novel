# 首次生成质量优化实现总结

## 概览

实现了 6 个首次生成质量优化特性（P0-1 至 P0-4 为 P0 级，P1-1 至 P1-2 为 P1 级），目标是将首次生成质量提升至接近终稿水平，最小化返工成本。

所有特性均通过 config_template.yaml 中的开关控制，默认启用，可通过 `novel.py ablate --flip <key>` 进行消融测试。

---

## P0-1: 失败模式前置负面清单 ✅

**文件**: `writing.py`

**功能**: 从近期章节的 `gate_rejects`（门禁拒收）、风格塌缩标记、化石句中提取失败模式，在写稿前注入"本章绝对禁止"清单，而非事后在评审中才发现。

**实现**:
- 新增函数 `_preflight_negative_list(paths, conn, config, chapter_num, lookback=5)`
  - 扫描最近 N 章的 `final_review.json` 中的 `gate_rejects` 和 `style_flags`
  - 提取跨章化石句、相邻章复读、风格问题等
  - 返回 `{"items": [...], "fossils": [...], "style_warnings": [...]}`
- 在 `write_chapter()` 的 `carryover_block` 中注入（line ~1017）：
  ```python
  ## 本章绝对禁止（前置负面清单·来自近期质量门禁）
  - 化石句列表
  - 风格问题警告
  ```
- 配置项:
  - `preflight_constraints_enabled: true` (默认启用)
  - `preflight_constraints_lookback: 5` (回看章数)

**原理**: 将失败模式从"事后发现→扣分→下章提醒"升级为"事前注入→动笔前规避"，缩短反馈回路。

---

## P0-2: 候选草稿确定性预筛 ✅

**文件**: `pipeline.py`

**功能**: 多候选模式下，在 LLM 评审前用 `style_health` + `cross_chapter_repetition` 确定性检测筛掉必触发 `gate_reject` 或罚分过高的草稿，节省评审成本、防止低质稿以分数胜出。

**实现**:
- 在 `write_chapter_with_candidates()` 中新增预筛逻辑（line ~276 之后）：
  - 对每个候选草稿运行 `style_health()` 和 `cross_chapter_repetition()`
  - 计算 `total_penalty = sh_penalty + cr_penalty`
  - 若 `cr_level == "reject"` 或 `total_penalty >= block_threshold`，淘汰该草稿
  - 保存预筛结果到 `candidate_prescreen.json`
- 配置项:
  - `candidate_prescreen_enabled: true`
  - `candidate_prescreen_penalty_block: 3.0` (罚分阈值)

**原理**: 在 beat 门禁之后、LLM 评审之前插入确定性质量门，过滤掉必然被拒的草稿，减少无效评审调用。

---

## P0-3: 黄金范例 RAG ✅

**文件**: `retrieval.py`, `writing.py`

**功能**: 从本书高分章节（≥8.8/10）中检索与当前 plan 类型匹配的片段，作为正例注入写手 prompt（变量段），学习节奏与执行手法。

**实现**:
- 新增函数 `exemplar_block(paths, conn, config, plan, chapter_num)` in `retrieval.py`:
  - 从 `chapter_metrics` 表查询 `score >= 8.8` 的章节
  - 按 `payoff_type` / `conflict_type` 匹配当前 plan
  - 用 TF-IDF 检索与 plan 字段匹配的片段（~200-400 字）
  - 格式化为 `## 黄金范例` 块，附带分数和强项标注
- 在 `write_chapter()` 的 `carryover_block` 中注入（line ~1088 之后）：
  ```python
  ## 黄金范例（本书高分章节，供学习节奏与执行手法）
  - **Ch5 终评 9.2/10** (强钩子(8.9/10)、高兑现(9.0/10))
    片段...
  ```
- **关键**: 注入在变量段（用户消息），不影响 `cacheable_prefix`
- 配置项:
  - `exemplar_rag_enabled: true`
  - `exemplar_rag_min_chapter: 8` (从第 8 章开始启用)
  - `exemplar_rag_score_min: 8.8` (分数阈值)
  - `exemplar_rag_top_k: 3` (最多选 3 个范例)

**原理**: 用本书自身的高分案例作为正例，比通用范例更贴合当前作品的风格和设定，是最直接的"模仿学习"。

---

## P0-4: required_constraints 契约结构化 ✅

**文件**: `planning.py`, `review.py`

**功能**: 仲裁层输出结构化约束（id/type/check_method/target），评审层机械验证每条约束是否兑现，未通过则扣分并反馈给下章写手。

**实现**:
- **planning.py**: 更新 `ARBITER_SYSTEM` schema（line ~71）：
  ```json
  "required_constraints": [
    {
      "id": "beat_3_location",
      "type": "beat_fidelity|character_consistency|world_logic|payoff_delivery|hook_setup|other",
      "constraint": "具体的验收条款",
      "check_method": "keyword|character_name|location|object|action|dialogue|logic",
      "target": "关键词/人名/物件/动作（供机械检查）"
    }
  ]
  ```
- **review.py**: 新增约束验证逻辑（line ~676 之后）：
  - 从 `plan_initial_attempt0_arbitration.json` 读取 `required_constraints`
  - 按 `check_method` 机械验证每条约束（keyword 匹配、location 出现、object 存在等）
  - 未通过的约束记入 `constraint_violations_structured`
  - 计算罚分：`penalty = min(失败数 * 0.5, 2.0)`
  - 若失败数 >= 3，标记 `accepted=False`
  - 将违约反馈注入 `writer_directives_for_next_chapter`
- 配置项:
  - `constraint_verification_enabled: true`
  - `constraint_violation_penalty_each: 0.5`
  - `constraint_violation_penalty_cap: 2.0`
  - `constraint_violation_block_count: 3`

**原理**: 将自由文本约束升级为可机械验证的结构化契约，闭环"计划→执行→验证"，防止 plan 中承诺的关键元素在成稿中丢失。

---

## P1-1: bandit reward 升级为终局质量 ✅

**文件**: `planning.py`

**功能**: 将策略 bandit 的奖励信号从"仲裁者选择"升级为"终局质量"（`chapter_metrics.score`），使策略学习目标从"哪个大纲好"变为"哪个大纲真正写出高分成稿"。

**实现**:
- 修改 `_strategy_history(conn, lookback=60)` (line ~207):
  - 新增从 `chapter_metrics` 表读取 `terminal_scores: dict[int, float]`
  - 对被选中的策略（`i == sel_idx`），计算 wins:
    - 若 `terminal_score >= 8.0`，`wins += 1.0`（全胜）
    - 若 `5.0 <= terminal_score < 8.0`，`wins += (score - 5.0) / 3.0`（部分胜）
    - 若 `terminal_score < 5.0` 或未写入，`wins += 0.0`
  - 回退模式：若 `chapter_metrics` 不存在，仍以仲裁选择计数（向后兼容）
- Thompson 采样逻辑不变，只是 wins 的定义改变
- 无需新增配置项（行为升级，无开关）

**原理**: 消除"plan 评分虚高→成稿质量低"的脱节。策略 bandit 直接优化终局质量，自然偏好能产生高分成稿的策略。

---

## P1-2: 跨书经验蒸馏 ✅

**文件**: `distill.py` (新文件)

**功能**: 扫描 `novels/*/story_state.db` 中的 `gate_rejects` 和 `agent_reports`，提取反复出现的"失败模式→修复策略"模式，输出全局 craft rules 供规划/写作流程消费。

**实现**:
- 独立 CLI: `python -m distill --output craft_rules.json [--genre <genre>] [--min-novels 3]`
- 扫描逻辑:
  - 遍历 `novels/<name>/story_state.db`
  - 从 `agent_reports` 表读取 `report_type='review'` 的记录
  - 提取 `gate_rejects`（门禁拒收）和 `problems` + `fixes`
  - 按 category 分类（`beat_execution`, `payoff_setup`, `hook_technique`, `style` 等）
  - 聚合相同 pattern，统计 evidence_count、source_novels、confidence
- 输出 schema:
  ```json
  {
    "rules": [
      {
        "id": "beat_execution_1",
        "category": "beat_execution",
        "pattern": "beat 中的具体动作未在正文实演",
        "fix": "将抽象 beat 改写为可见动作+物件+结果",
        "evidence_count": 15,
        "source_novels": ["suspense_v11", "xuanhuan_v3", ...],
        "confidence": 0.85
      }
    ],
    "meta": {"novels_scanned": 8, "total_chapters": 240}
  }
  ```
- 集成点（未自动接入，需手动消费）:
  - `planning.py`: 将高置信度规则注入 `required_constraints` 提示
  - `writing.py`: 将 style/beat 规则注入 `writer_directives`
  - `review.py`: 用规则校准各类别评分阈值

**原理**: 跨书学习。多本书反复踩的坑 → 提取为通用规则 → 新书直接继承，避免重复试错。

---

## 配置项总览

所有新增配置项已添加到 `config_template.yaml`：

```yaml
novel:
  # P0-1
  preflight_constraints_enabled: true
  preflight_constraints_lookback: 5

  # P0-2
  candidate_prescreen_enabled: true
  candidate_prescreen_penalty_block: 3.0

  # P0-3
  exemplar_rag_enabled: true
  exemplar_rag_min_chapter: 8
  exemplar_rag_score_min: 8.8
  exemplar_rag_top_k: 3

  # P0-4
  constraint_verification_enabled: true
  constraint_violation_penalty_each: 0.5
  constraint_violation_penalty_cap: 2.0
  constraint_violation_block_count: 3

  # P1-1: 无需配置项（行为升级）
```

---

## 验证与测试

### 单元测试
现有纯函数测试不受影响：
```bash
python -m unittest discover tests
```

### 消融测试
每个特性可单独关闭进行消融：
```bash
# 关闭 P0-1
python novel.py ablate my_novel --flip preflight_constraints_enabled --chapters 8

# 关闭 P0-2
python novel.py ablate my_novel --flip candidate_prescreen_enabled --chapters 8

# 关闭 P0-3
python novel.py ablate my_novel --flip exemplar_rag_enabled --chapters 8

# 关闭 P0-4
python novel.py ablate my_novel --flip constraint_verification_enabled --chapters 8
```

然后对比消融版本与基线版本：
```bash
python novel.py compare my_novel my_novel__ablate_<key>
```

### 集成验证
创建测试小说运行 8 章：
```bash
python novel.py create test_quality_opt
# 编辑 novels/test_quality_opt/prompt.md
python novel.py run test_quality_opt --foreground
```

检查日志确认特性生效：
- `logs/run.log` 中搜索 "Pre-screen", "Exemplar RAG", "Constraint violations", "Preflight negative"
- 检查 `logs/checkpoints/ch*/` 是否生成 `candidate_prescreen.json`, `constraint_violations_structured`
- 验证 `chapter_metrics` 表中 bandit wins 计算逻辑

---

## 预期效果

| 指标 | 目标提升 | 机制 |
|------|---------|------|
| 首次生成分数 | +0.5~1.0 | P0-1 前置负面清单 + P0-3 黄金范例 |
| 首稿接受率 | +15~25% | P0-2 候选预筛 + P0-4 契约验证 |
| 返工轮次 | -30~50% | P0-1/P0-4 将失败检测前移至生成前 |
| 策略收敛速度 | +20~40% | P1-1 终局质量奖励信号 |
| 跨书复用率 | 新书冷启动质量 +0.3~0.5 | P1-2 经验蒸馏 |

---

## 已知限制与后续优化

1. **P0-3 exemplar_block**: 当前用简单 TF-IDF 匹配，可升级为语义相似度（需引入 embeddings）
2. **P0-4 constraint verification**: `logic` 类型约束仍需 LLM 验证，未实现机械检查
3. **P1-1 terminal quality**: 若 `chapter_metrics` 未写入（如崩溃），回退到仲裁计数
4. **P1-2 distill.py**: 输出的 craft rules 需手动集成到 prompt，未实现自动注入

---

## 代码清单

### 修改文件
- `writing.py`: +90 行（P0-1 preflight, P0-3 exemplar injection）
- `pipeline.py`: +60 行（P0-2 candidate prescreen）
- `planning.py`: +80 行（P0-4 schema, P1-1 terminal quality）
- `retrieval.py`: +150 行（P0-3 exemplar_block）
- `review.py`: +80 行（P0-4 constraint verification）
- `config_template.yaml`: +15 行（配置项）

### 新增文件
- `distill.py`: 250 行（P1-2 跨书蒸馏）

### 总计
~725 行新增/修改代码，0 个 LLM 依赖（纯逻辑），6 个可消融特性。

---

## 实现日期
2026-06-12

## 实现者
Claude Opus 4.8 (via Claude Code)
