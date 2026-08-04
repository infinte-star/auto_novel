# Prompt 优化报告

版本：2026-08-03

## 1. 执行结论

本轮完成了从创作简报、bootstrap、创作契约、世界观/人物/声音/卷纲、弧级 ChapterCard、章节首稿与 ChapterDelta、局部修复、评审、精修、试写、包装、短剧改编，到 stable/volatile 上下文装配的全链路审计。

核心结论不是“所有 Prompt 都再加一轮要求”，而是：

1. 公共层只保留真正跨任务成立的执行协议；章节方法论不得广播给包装、抽取、锚点评审或局部修句。
2. 每个调用先确定职责，再由职责同时决定模型路由和最小职责纪律，避免两套标签规则漂移。
3. 任务 System Prompt 是该调用的唯一任务合同；canon、卡片、正文等材料只作为有边界的数据输入。
4. 能由代码判定的格式、长度、状态和质量条件继续留在确定性校验层，不要求模型自评分代替验收。
5. 高风险约束保留必要的尾部锚点，但同一协议不在 system/user 重复三遍。

目标排序保持为：生成质量 > 首轮成功率 > 效率。所有承重接口——ChapterCard 字段、ChapterDelta schema、分隔符、缓存前缀和下游解析键名——均保持兼容。

## 2. 范围与数据基线

### 2.1 运行时 Prompt 族

| Prompt 族 | 主要真源 | 职责 |
| --- | --- | --- |
| 创作输入 | `prompt_template.md` | 作者填写的硬约束、偏好与留白 |
| 初始化地基 | `engine/bootstrap.py` | brief → contract / bible / characters / voices / volume plan / frame |
| 弧规划 | `engine/plan.py` | canon + 卷纲 +近期状态 → ChapterCard |
| 章节写作 | `engine/genre.py`、`engine/write.py` | ChapterCard → 正文 + ChapterDelta |
| 修复 | `engine/quality_repair.py` | 对确定缺陷做范围受限的局部重写 |
| 评审 | `engine/quality_advisory.py`、`engine/anchor.py` | 冷读、证据化判断、外部盲评 |
| 完稿工具 | `commands/refine.py`、`package.py`、`trial.py`、`screenplay.py` | 诊断/精修/包装/试写/改编 |
| 公共协议与路由 | `engine/llm.py` | 职责登记、输出协议、JSON 兼容、模型路由 |
| 上下文 | `engine/state.py`、`bootstrap.py`、`retrieval.py` | stable/volatile 分离、窗口与检索 |

### 2.2 历史成本分布

`python tools/prompt_census.py --top 60` 对现有日志的结果：

| 指标 | 结果 |
| --- | ---: |
| 调用数 | 11,187 |
| 标签数 | 52 |
| Prompt 总字符 | 631,717,678 |
| 规划类占比 | 51.6% |
| 写作类占比 | 19.3% |
| 评审 `review` 占比 | 13.8% |
| `plan_candidate` 单项占比 | 32.8% |
| `write` 单项占比 | 19.3% |

所以优化优先级是规划上下文和公共层污染，其次是每章必经的写作 Prompt，再其次才是低频辅助 Prompt 的逐字压缩。

### 2.3 本轮静态字符收益

| 项目 | 优化前 | 优化后 | 降幅 |
| --- | ---: | ---: | ---: |
| 审计核心常量合计 | 8,799 | 6,144 | 30.2% |
| 通用写作避雷块 | 2,702 | 537 | 80.1% |
| JSON 输出合同 | 270 | 183 | 32.2% |
| 新书创作模板 | 3,669 | 1,396 | 62.0% |

JSON 调用在 user 已有完整输出合同时，不再在 system 重复一份近义 JSON 纪律；`json_mode=True` 且没有 user 合同的调用仍会得到精简安全协议。当前各题材 writer system 为约 2,915-4,623 字符，不含 stable prefix 与动态上下文。

## 3. 问题 → 原因 → 方案 → 收益 → 风险

| 问题 | 原因 | 优化方案 | 预期收益 | 潜在风险 / 控制 |
| --- | --- | --- | --- | --- |
| 模型路由与 Prompt 纪律覆盖不一致 | 两处各自用标签字符串判断，`arc_*`、`fix_*` 等容易漏分 | 用 `prompt_role_for_tag()` 作为 planning/writing/extraction/review 的唯一职责源 | 正确模型、正确纪律、遥测可归因 | 标签误分会改变模型；用静态测试锁定 |
| trial、screenplay 调用无标签 | 默认空标签绕过角色模型与职责协议 | 五阶段分别标注规划、写作、抽取、评审、修订 | 提升模型匹配与故障定位 | 历史统计需按职责归并新旧标签 |
| 公共 planning 纪律含 ChapterCard 专用 beat 规则 | 同一“规划”角色覆盖创意、包装、卷纲、短剧等异构任务 | 公共层只保留“因果链、可执行、不得改 canon”，章卡方法留在 `ARC_SYSTEM` | 降低职责污染和无关注意力竞争 | 具体任务必须自己定义 schema；静态审计检查 |
| 公共 review 纪律强迫所有评审给总分上限 | 锚点盲评、冷读、诊断并非同一种评分 | 改为“只按本任务量表、证据与结论一致” | 降低评分格式漂移和自证循环 | 各评审仍需有自己的量表 |
| JSON 合同 system/user 重复 | `json_prompt()` 与全局增强同时描述同一协议 | 完整合同只放 user 一次；无合同的 json mode 才注入精简 system 协议；包装函数幂等 | 减少 token 与细微冲突，提高解析首过 | 弱模型可能受重复减少影响；provider JSON mode 和解析修复仍在 |
| 公共协议附在任务 Prompt 末尾 | 通用规则占据最高 recency，稀释任务专用输出要求 | 改为 L0 → L1 → L2，具体任务合同位于 system 最后 | 提高 schema、局部修复边界和输出格式服从率 | 缓存会在版本切换时失效一次 |
| 弧规划规则重复 | 张力、钩子、兑现规则在“硬规则”和“方法论”重复 | 合并成一次可判定约束，方法论只保留因果编排 | 减少字段漏填、近义规则打架 | 质量收益需 matched A/B 验证 |
| 写作通用块存在互相冲突的段落规则 | 同时要求“每段至少 2 句”和“3-5 句/60字”，且例外不一致 | 改为按叙事单元组织，限制连续碎段，允许有节奏价值的短对白/转折 | 降低电报体和机械长段两端失衡 | 放宽绝对数值后依赖 `style_health` 复检 |
| 所有题材被强迫同一种“憋-炸-余韵”和强悬念 | 通用层混入爽文专用结构，且与悬疑慢烧、言情、终章冲突 | 通用层只要求执行卡片因果链；题材差异留在 profile；终章有明确覆盖块 | 降低题材串味、重复节奏和假钩子 | 爽文强度依赖对应 profile 与 ChapterCard |
| writer 中存在多套对白比例 | system 固定 20%/25-45%，user 又按配置生成动态下限 | 数值只由 `length_block(config)` 给出，题材块引用动态区间 | 配置成为单一真源，避免模型无所适从 | 动态目标配置错误会直接影响生成，需配置测试 |
| 每个场景都要求两轮对白 | 独处、追逐、战斗场景只能靠自言自语凑数 | 只约束多人同场主要场景；明确禁止独处场景强凑对白 | 减少 AI 味和回声对白 | 对话不足仍由确定性比例门识别 |
| ChapterDelta 无变化统一填 `[]` | schema 同时有数组和对象字段，通用指令制造类型错误 | 明确数组填 `[]`、对象填 `{}` | 提高一次解析成功率，减少 backfill | 无；字段 schema 未变 |
| 终章仍收到强追读钩子要求 | 弧卡要求收束，但通用 writer 与题材 profile 要求新危机/新信息 | `ENDING_ZONE_BLOCK` 渐进收债；`FINAL_CHAPTER_BLOCK` 覆盖通用钩子规则 | 降低烂尾、未收线和“完结后还有下一章” | 仅 `max_chapters` 可确定时启用 |
| 契约抽取一边“宁缺毋滥”一边要求强烈暗示也入白名单 | 抽取 Prompt 内部直接矛盾，曾把普通专业能力升级成超能力 | 只有简报明确建立特殊能力才写 whitelist；模糊暗示和普通技能不得推断 | 降低虚假能力越界、OOC 和无法修复的硬门失败 | 过于简略的 brief 可能少抽；后续 canon 可补，不在硬契约猜 |
| 创作模板示例易被当成事实 | 模板与作者内容一起原样进入 LLM，没有预处理边界 | 改为“钉死/偏好/留白”三态，删除会污染设定的具体示例与重复方法论 | 提升契约抽取精度、减少模板小说和设定误植 | 旧项目不自动迁移；只影响新建模板 |
| 新调用易忘记登记 | 规范若只写在文档中无法执行 | 静态测试要求每个 `call_llm` 有 tag、字面 tag 可解析、职责集合互斥 | 可维护、可扩展 | 动态 tag 必须使用受控前缀并补样例测试 |

## 4. 统一 Prompt 架构

每次调用按五层组成，每层只有一个职责：

```text
L0 公共执行协议       输出服从、冲突顺序、禁止无关元文本
L1 职责协议           planning / writing / extraction / review 的最小公约
L2 任务 System Prompt 唯一任务、输入边界、硬约束、schema/输出
L3 Context/User       stable canon + volatile 状态/卡片/正文/本次请求
L4 确定性校验         解析、schema、长度、门禁、状态投影、keep-if-improved
```

### 4.1 冲突优先级

统一为：

```text
任务输出协议 > 用户明示硬约束 > 已确认事实（canon）> 当前输入材料 > 风格偏好
```

“后出现”本身不构成更高优先级。普通材料不得因位于 Prompt 尾部就覆盖创作契约或 canon；真正需要尾部强调的，只能是同一硬约束的精简胶囊或响应格式确认。

### 4.2 职责边界

| 职责 | 做什么 | 不做什么 |
| --- | --- | --- |
| planning | 设计可执行结构、因果与选择 | 不写正文，不改 canon，不抽取未发生事实 |
| writing | 将指定内容落到页面，或在指定范围修订 | 不评分，不改 schema，不擅自扩展事实边界 |
| extraction | 从输入提取事实、状态、合同或修复结构 | 不创作缺失事实，不把示例/建议升级成事实 |
| review | 按给定量表独立判断并给证据 | 不替作者脑补，不为提高分数改写文本 |

### 4.3 上下文策略

- stable：创作简报、已确认世界观、人物档案、声音宪章、创作契约等低频变化内容，保持字节稳定以复用缓存。
- volatile：当前 ChapterCard、近期事件、未决线程、当前资源、检索片段、质量反馈和本次请求。
- 事实与指令分区：检索文本、旧正文和评审材料是数据，不得使用“最高优先级”语言伪装成系统指令。
- 最小充分：planner 读取结构和因果所需信息；writer 读取卡片、人物与局部连续性；blind judge 不读取作者意图和 cacheable prefix。

## 5. 各 Prompt 族审计结果

| Prompt 族 | 结论与当前版本 |
| --- | --- |
| 创作简报 | 已替换为三态模板；硬事实、偏好、留白明确分离，模板缩短 62% |
| Creative boost | 保留高温创意职责，但不能覆盖作者硬约束；由 planning 路由 |
| Contract | 归 extraction；已修复“强烈暗示也建白名单”的内部冲突，继续保持空白优于猜测 |
| Bible / Characters / Voice / Voices | 保留 dependency chain：世界规则 → 人物边界 → 叙事声音/角色声纹；长度纪律和产物职责清楚，不合并回单次大 JSON |
| Volume plan / frame | 保留全书结构与逐卷细化分工；章数上限继续由显式 finale 约束覆盖默认长篇模板 |
| Arc / ChapterCard | 作为最高收益规划 Prompt，schema 不变；重复张力/兑现/钩子规则已合并，继续由代码做字段标准化和连续性校验 |
| Writer / Genre | shared core + genre delta 不变；通用块已去伪精确、去题材串味，动态长度/对白是唯一数值源；两段式输出协议保持 |
| ChapterDelta / backfill | 归 extraction；schema 类型空值规则已修复，禁止新增正文没有的事实 |
| L0/L1 repair | 全部归 writing；局部输入、局部输出、长度检查、splice 和 keep-if-improved 形成闭环，不复制整套写作方法论 |
| Cold reader / anchor / review | 归 review；只读必要文本、证据化判断、保持与作者上下文隔离，不接受通用“总分必须受限”污染 |
| Refine | diagnose 属 review，rewrite 属 writing；两阶段职责隔离，保留强度与事实边界 |
| Trial / package | route/package 属 planning，试写属 writing，比较属 review；标签已补齐 |
| Screenplay | extract/planning/writing/review/revise 五阶段分别路由，避免一个空标签贯穿全部任务 |
| Memory / retrieval | 继续只提供事实与最小相关上下文；stable/volatile 与缓存边界不因文案优化移动 |

## 6. 可直接复用的 Prompt 模板

后续新增能力使用以下骨架，不复制某个既有长 Prompt：

```text
你是{领域角色}。

## 唯一任务
{一个可验收的动作和完成条件}

## 输入边界
- 可作为事实：{来源}
- 只能作为参考：{来源}
- 不得推断：{缺失或不可靠信息}

## 硬约束
1. {可判定约束}
2. {可判定约束}

## 输出
{唯一格式或 schema；字段名只定义一次}

提交前静默检查：字段完整、事实有来源、硬约束无遗漏。只输出最终结果。
```

新增调用必须提供语义 tag，并在 `prompt_role_for_tag()` 的精确集合或受控前缀中登记。JSON 用户输入统一通过幂等的 `json_prompt()` 包装。

## 7. 优先级与实施状态

| 优先级 | 项目 | 收益 | 成本 | 状态 |
| --- | --- | --- | --- | --- |
| P0 | 统一职责解析、补齐标签、静态约束 | 高 | 低 | 已实施 |
| P0 | 消除公共职责污染、调整 L0→L1→L2 顺序 | 高 | 低 | 已实施 |
| P0 | JSON 去重与类型正确的空值协议 | 高 | 低 | 已实施 |
| P0 | 终章/收束区覆盖通用钩子 | 高 | 低 | 已实施 |
| P0 | 写作公共块冲突与伪精确清理 | 高 | 中 | 已实施，需真实 A/B |
| P1 | 弧规划重复规则合并 | 中高 | 低 | 已实施，需真实 A/B |
| P1 | 创作模板与能力白名单抽取修正 | 中高 | 低 | 已实施 |
| P1 | 对 planner volatile context 做字段级字符归因 | 高 | 中 | 建议下一轮，先测后删 |
| P1 | 各 genre delta 做句级重复聚类 | 中 | 中 | 建议下一轮；按题材分别 A/B |
| P2 | PromptSpec 声明式注册表 | 中 | 中高 | 暂不实施；当前函数和静态测试更简单 |
| P2 | Provider 原生 JSON Schema | 中 | 中 | 网关兼容矩阵稳定后再做 |

## 8. 验证与持续迭代

### 8.1 本轮静态验证

- `py -3.13 -m py_compile`：所有改动模块通过。
- Prompt 架构、职责映射、JSON 幂等/去重、写作输出交换、动态对白、收束区和终章：24 个定向单元测试通过。
- 完整测试套件在允许创建临时 SQLite/日志目录的环境中通过：942 tests，0 failure，0 error。

### 8.2 发布前最低质量验证

1. `py -3.13 -m unittest discover tests`
2. `python tools/prompt_census.py --top 60` 比较职责级输入字符和 JSON repair/backfill 率
3. 同模型、同简报、同章节位置运行 matched-position A/B
4. `tools/fpy_prime.py` 看首轮通过率；`tools/pairwise_ab.py --anchor` 看外部盲胜率
5. 单独跟踪 OOC、canon、delta parse、对白比例、碎句、重复骨架、终章未收束率

如果 FPY 上升但 anchor WR 下降，回退对应文案；不能用自评、门禁通过率或 token 降幅替代作品质量。

## 9. 直接替换位置

本报告不复制一套容易漂移的完整 Prompt 全文。下列文件中的常量和构建函数就是可运行、可直接替换的唯一真源：

- `engine/llm.py`：职责注册、公共/职责协议、JSON 合同与注入顺序
- `engine/plan.py`：优化后的弧规划与卡片定点修复 Prompt
- `engine/genre.py`：优化后的通用写作底线、页面呈现和各题材 delta
- `engine/write.py`：动态约束、两段式输出、ChapterDelta、收束区与终章 Prompt
- `engine/bootstrap.py`：创作契约与地基生成 Prompt
- `commands/trial.py`、`commands/screenplay.py`：阶段语义标签
- `prompt_template.md`：新书创作简报模板
- `tests/test_prompt_architecture.py`、`tests/test_pure_functions.py`、`tests/test_write.py`：可执行的 Prompt 架构规范
