# 知识库维护规范 (Knowledge Base Maintenance)

---

## 一、知识库文件结构

### 每条规则的标准格式
```markdown
### 规则名称
- **来源**: 作品名/研究/经验
- **适用题材**: all / 特定题材列表
- **规则内容**: 具体描述
- **量化指标**: 关键数字（密度/频率/阈值）
- **正例**: 好的做法
- **反例**: 坏的做法
- **验证状态**: 未验证 / 已验证（日期+数据）
```

### 文件目录
```
novel_knowledge/
├── README.md                     # 索引
├── story_structure/
│   ├── hit_rules.md              # 爆款核心规则
│   └── opening_rules.md          # 开篇规则
├── character_design/
│   ├── character_rules.md        # 人设规则
│   └── romance_rules.md          # 恋爱线规则
├── emotion_design/
│   └── emotion_rules.md          # 情绪管理规则
├── plot_patterns/
│   ├── foreshadow_rules.md       # 伏笔规则
│   └── genre_patterns.md         # 题材特征
├── chapter_patterns/
│   ├── pacing_templates.md       # 节奏模板
│   └── hook_templates.md         # 钩子模板
├── writing_style/
│   ├── anti_ai_rules.md          # 去AI感规则
│   └── dialogue_rules.md         # 对话规则
└── evaluation_rules/
    ├── reader_retention.md       # 留存规则
    └── continuous_learning.md    # 本文件
```

---

## 二、与框架的集成点

### 2.1 plan.py 集成
规划弧线时，`engine/knowledge.py:select_for_planner` 从知识库注入：
- `pacing_templates.md` → 选择本弧线的节奏模板
- `hit_rules.md` → 爽点密度/伏笔密度约束
- `emotion_rules.md` → 情绪曲线编排

### 2.2 write.py 集成
写作提示词中，`engine/knowledge.py:select_for_writer` 按章节状态动态注入：
- `anti_ai_rules.md` → top-N 最相关禁令（按 tension/genre 选择）
- `genre_patterns.md` → 当前题材的核心爽点写法
- `opening_rules.md` → 前3章/前7章专项规则
- `foreshadow_rules.md` → 低张力章/每5章注入回收提醒

### 2.3 quality.py 集成
`style_health` 检测逻辑已内化部分 `anti_ai_rules.md` 的检查：
- 破折号过多、重复句式、叙段过长
- 对话占比检查

---

## 三、知识库维护规则

### 防止知识库膨胀
- 每个文件设上限（建议 ≤ 200 行）
- 定期审查：半年以上未被验证的规则考虑删除
- 合并相似规则，保留更通用的版本

### 防止规则冲突
- 新规则入库前检查与已有规则的一致性
- 矛盾规则标注 `[CONFLICT]`，不自动应用
- 题材限定的规则不与通用规则冲突

### 防止过拟合
- 单部作品的规则不能直接入库
- 至少需要 3 部不同作品的相同模式才能固化为规则
- 保留规则的"来源计数"——引用源越多越可靠

### 更新流程
1. 人工阅读爆款作品/收集编辑反馈
2. 手动提取规则写入对应文件
3. 检查知识库一致性（无矛盾）
4. 验证：用更新后的知识库生成测试章节，对比 FPY
