# 持续学习机制设计 (Continuous Learning Mechanism)

---

## 一、设计目标

让框架能够：
1. 分析新的爆款作品，提取可复用的方法论
2. 将新规则融入现有知识库，去重合并
3. 自动更新 prompt 和 pipeline 参数
4. 形成"研究→提取→验证→固化"的闭环

---

## 二、知识更新流程

### Step 1: 输入
```
python novel.py learn --source <path_or_url>
  --genre <genre>           # 题材分类
  --platform <platform>     # 来源平台
  --metrics <json>          # 已知的数据指标（追读率/评分/收藏等）
```

输入源可以是：
- 本地 txt/epub 文件（完整作品或前N章）
- 在线作品链接（需爬虫支持，可选）
- 手动输入的方法论笔记（markdown）

### Step 2: 自动分析
对输入作品执行以下分析（每项对应一个 LLM 调用）：

| 分析维度 | 输出 | 对应知识库文件 |
|----------|------|----------------|
| 开篇分析 | 黄金三章结构、信息密度、钩子类型 | opening_rules.md |
| 爽点分析 | 爽点类型/密度/铺垫时长/外化方式 | hit_rules.md |
| 情绪曲线 | 章级情绪标注、张弛节奏 | emotion_rules.md |
| 角色分析 | 人设展示方式、配角功能分配 | character_rules.md |
| 钩子分析 | 章末钩子类型分布、轮换模式 | hook_templates.md |
| 伏笔分析 | 埋设/回收间距、锚点类型 | foreshadow_rules.md |
| 去AI感分析 | 语言特征、节奏指纹、人味技巧 | anti_ai_rules.md |
| 节奏分析 | 章内/跨章节奏模式 | pacing_templates.md |
| 恋爱线分析（如适用） | 推进阶段、互动密度 | romance_rules.md |

### Step 3: 规则提取
将分析结果抽象为可复用的规则：
- 具体的：这部作品在第3章用了什么技巧
- 抽象的：这类技巧的通用模式是什么
- 量化的：这个模式的关键参数是什么（密度/频率/比例）

### Step 4: 知识库合并
```python
def merge_rule(new_rule, existing_rules):
    # 1. 语义去重：检查是否已有相似规则
    # 2. 冲突检测：新规则是否与已有规则矛盾
    # 3. 合并策略：
    #    - 全新规则 → 追加
    #    - 相似规则 → 合并，更新数据支撑
    #    - 矛盾规则 → 保留两者，标注来源，等人工裁决
    # 4. 更新索引（README.md）
```

### Step 5: 验证
- 用更新后的知识库生成测试章节
- 对比更新前后的 style_health / FPY 指标
- 如果指标下降 → 回滚本次更新

---

## 三、知识库结构规范

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

### 知识库文件命名约定
```
knowledge/
├── README.md           # 索引
├── hit_rules.md        # 爆款核心规则
├── anti_ai_rules.md    # 去AI感规则
├── emotion_rules.md    # 情绪管理规则
├── character_rules.md  # 人设规则
├── romance_rules.md    # 恋爱线规则
├── genre_patterns.md   # 题材特征
├── hook_templates.md   # 钩子模板
├── foreshadow_rules.md # 伏笔规则
├── reader_retention.md # 留存规则
├── opening_rules.md    # 开篇规则
├── pacing_templates.md # 节奏模板
└── continuous_learning.md # 本文件
```

---

## 四、与框架的集成点

### 4.1 plan.py 集成
规划弧线时，从知识库注入：
- `pacing_templates.md` → 选择本弧线的节奏模板
- `hit_rules.md` → 爽点密度/伏笔密度约束
- `emotion_rules.md` → 情绪曲线编排
- `hook_templates.md` → 钩子类型轮换计划

### 4.2 write.py 集成
写作提示词中，按当前章节状态动态注入：
- `anti_ai_rules.md` → top-5 最相关禁令（而非全量）
- `genre_patterns.md` → 当前题材的核心爽点写法
- `character_rules.md` → 角色展示方式指导
- `romance_rules.md` → 当前恋爱阶段的互动指导（如适用）

### 4.3 quality.py 集成
质量门从知识库读取阈值：
- `reader_retention.md` → 追读率代理指标的阈值
- `opening_rules.md` → 黄金三章专项检查清单
- `emotion_rules.md` → 情绪节奏检查参数

### 4.4 bootstrap.py 集成
生成卷纲时：
- `pacing_templates.md` → 卷级节奏模板
- `foreshadow_rules.md` → 伏笔埋设/回收计划
- `hit_rules.md` → 爽点梯度规划

---

## 五、自动化程度分级

### Level 0: 手动（当前可实现）
- 人工阅读爆款作品
- 手动提取规则写入 knowledge/ 下对应文件
- 手动检查知识库一致性

### Level 1: 半自动（推荐的下一步）
- `novel.py learn` CLI 接收文本输入
- LLM 辅助提取规则（人工审核）
- 自动检测规则冲突
- 手动决定是否合并

### Level 2: 全自动（远期目标）
- 定期爬取平台热榜
- 自动下载/分析前N章
- 自动提取+合并+验证
- 只在指标下降时人工介入

---

## 六、风险控制

### 防止知识库膨胀
- 每个文件设上限（建议 ≤ 200 行）
- 定期审查：半年以上未被验证的规则考虑删除
- 合并相似规则，保留更通用的版本

### 防止规则冲突
- 新规则入库前自动检查与已有规则的一致性
- 矛盾规则标注 `[CONFLICT]`，不自动应用
- 题材限定的规则不与通用规则冲突

### 防止过拟合
- 单部作品的规则不能直接入库
- 至少需要3部不同作品的相同模式才能固化为规则
- 保留规则的"来源计数"——引用源越多越可靠
