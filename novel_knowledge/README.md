# 爆款网文知识库 (Hit Novel Knowledge Base)

本目录包含从爆款网文研究中提炼的可复用方法论，供框架 `engine/knowledge.py` 动态注入 prompt。

---

## 目录结构

```
novel_knowledge/
  story_structure/     ← 故事结构与爆款规则
  character_design/    ← 人设与角色设计
  emotion_design/      ← 情绪曲线与节奏
  plot_patterns/       ← 情节模式与伏笔
  chapter_patterns/    ← 章节节奏与钩子模板
  writing_style/       ← 文风与去AI味
  evaluation_rules/    ← 评估与留存规则
```

---

## 文件索引

### story_structure/ — 故事结构
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [hit_rules.md](story_structure/hit_rules.md) | 黄金三章、三级爽点、章末钩子、节奏控制、伏笔密度、冲突设计、落差公式、打脸四部曲 | `plan.py` `write.py` `quality.py` |
| [opening_rules.md](story_structure/opening_rules.md) | 前3章任务、平台差异、开篇公式、信息密度、第7章检查点 | `write.py` `quality.py` |

### character_design/ — 角色设计
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [character_rules.md](character_design/character_rules.md) | 代入感三要素、配角功能、反派层级、群像区分度、成长弧线 | `bootstrap.py` `write.py` |
| [romance_rules.md](character_design/romance_rules.md) | 十阶推进模型、心动设计、暧昧五感、CP互动类型 | `write.py` `plan.py` |

### emotion_design/ — 情绪设计
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [emotion_rules.md](emotion_design/emotion_rules.md) | 情绪呼吸法则、甜虐比数据、情绪递进公式、曲线模板 | `plan.py` `write.py` |

### plot_patterns/ — 情节模式
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [foreshadow_rules.md](plot_patterns/foreshadow_rules.md) | 5种伏笔类型、埋设密度、回收技法、记忆锚点 | `plan.py` `bootstrap.py` |
| [genre_patterns.md](plot_patterns/genre_patterns.md) | 8大题材爆款特征 + 2026趋势 | `write.py` `plan.py` |

### chapter_patterns/ — 章节模式
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [pacing_templates.md](chapter_patterns/pacing_templates.md) | 章内3种模板、跨章编排(3/5/10章)、情绪强度标注 | `plan.py` `write.py` |
| [hook_templates.md](chapter_patterns/hook_templates.md) | 6类钩子模板 + 使用规则 | `write.py` `plan.py` |

### writing_style/ — 写作风格
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [anti_ai_rules.md](writing_style/anti_ai_rules.md) | 去AI感规则：致命特征、禁用词/句式、人味技法 | `write.py` `quality.py` |
| [dialogue_rules.md](writing_style/dialogue_rules.md) | 对话占比、回声禁令、潜台词、角色语言指纹 | `write.py` `quality.py` |

### evaluation_rules/ — 评估规则
| 文件 | 内容 | 引擎消费者 |
|------|------|-----------|
| [reader_retention.md](evaluation_rules/reader_retention.md) | 弃书原因排行、追读/完读指标、留存策略 | `quality.py` `write.py` |
| [continuous_learning.md](evaluation_rules/continuous_learning.md) | 持续学习机制、更新流程、自动化分级 | 框架维护者参考 |

---

## 使用原则

1. **方法论优先**：知识库存储的是可复用的底层规则，不是具体的剧情模板
2. **数据驱动**：每条规则尽可能有量化指标支撑
3. **动态注入**：`engine/knowledge.py` 根据当前章节状态选择最相关的 top-N 规则注入 prompt（≤1500字符），而非全量
4. **验证闭环**：新规则入库后需通过 FPY/style_health 验证才能固化
5. **题材适配**：通用规则和题材专项规则分层，专项优先级高于通用
6. **与 inline 规则互补**：知识库是"参考模板"，write.py 中的 ANTI_PITFALL_BLOCK 等是"铁律"，两者不冲突
