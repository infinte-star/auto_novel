# 爆款网文知识库 (Hit Novel Knowledge Base)

本目录包含从爆款网文研究中提炼的可复用方法论，供框架各模块在 prompt 构建和质量检测时引用。

---

## 文件索引

| 文件 | 内容 | 主要消费者 |
|------|------|-----------|
| [hit_rules.md](hit_rules.md) | 爆款核心规则：黄金三章、爽点体系、章末钩子、节奏控制、伏笔密度、信息释放、冲突设计 | `plan.py`（弧线规划）、`write.py`（爽点/钩子指导）、`quality.py`（密度检测） |
| [anti_ai_rules.md](anti_ai_rules.md) | 去AI感规则：致命特征、禁用词/句式、结构反模式、人味技法、平台检测标准 | `write.py`（ANTI_PITFALL_BLOCK）、`quality.py`（style_health/ai_flavor_health） |
| [emotion_rules.md](emotion_rules.md) | 情绪管理：呼吸法则、甜虐配比、情绪递进公式、曲线模板、AI节奏矫正 | `plan.py`（情绪目标编排）、`write.py`（情绪指导注入） |
| [character_rules.md](character_rules.md) | 人设规则：代入感三要素、配角叙事功能、反派层级、群像区分度、成长弧线 | `bootstrap.py`（角色生成）、`write.py`（人设展示指导） |
| [romance_rules.md](romance_rules.md) | 恋爱线：十阶推进模型、心动设计、暧昧五感、CP互动类型、雷区避免 | `write.py`（romance_female profile）、`plan.py`（关系进度编排） |
| [genre_patterns.md](genre_patterns.md) | 各题材爆款特征：玄幻/都市/系统/规则怪谈/无限流/女频/神豪/穿越 8大类 | `write.py`（GENRE_PROFILES）、`plan.py`（题材特化规划） |
| [hook_templates.md](hook_templates.md) | 钩子模板：悬念/反转/情绪/信息/倒计时/温馨 6类模板 + 使用规则 | `write.py`（章末钩子指导）、`plan.py`（钩子类型轮换） |
| [foreshadow_rules.md](foreshadow_rules.md) | 伏笔体系：5种类型、埋设密度/技巧、回收时机/技法、记忆锚点设计 | `plan.py`（伏笔规划）、`bootstrap.py`（卷纲伏笔计划） |
| [reader_retention.md](reader_retention.md) | 留存规则：弃书原因排行、追读/完读指标、短/中/长期留存策略、开篇特殊规则 | `quality.py`（留存代理指标）、`write.py`（开篇强化） |
| [opening_rules.md](opening_rules.md) | 黄金三章详细规则：各章任务、平台差异、反模式、开篇公式、信息密度、第7章检查点 | `write.py`（开篇指导）、`quality.py`（opening_hook_gate） |
| [pacing_templates.md](pacing_templates.md) | 节奏模板：章内3种模板、跨章编排(3/5/10章)、情绪强度标注、特殊节奏模式、卷级结构 | `plan.py`（节奏编排）、`write.py`（结构模板选择） |
| [continuous_learning.md](continuous_learning.md) | 持续学习机制：更新流程、集成点、自动化分级、风险控制 | 框架维护者参考 |

---

## 使用原则

1. **方法论优先**：知识库存储的是可复用的底层规则，不是具体的剧情模板
2. **数据驱动**：每条规则尽可能有量化指标支撑
3. **动态注入**：根据当前章节状态选择最相关的 top-5 规则注入 prompt，而非全量
4. **验证闭环**：新规则入库后需通过 FPY/style_health 验证才能固化
5. **题材适配**：通用规则和题材专项规则分层，专项优先级高于通用
