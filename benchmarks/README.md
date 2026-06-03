# 爆款样本库

把你合法拥有或自行整理的结构化样本放在这里，生成时会被 `benchmark.py` 召回为结构参照。

推荐目录：

```text
benchmarks/
  qidian_male/
    history/
      example.json
      opening_patterns.md
  fanqie_free/
    urban_ability/
      retention_hooks.md
  common/
    cliffhanger_patterns.md
```

推荐 JSON 字段：

```json
{
  "title": "样本名",
  "summary": "作品/路线结构摘要",
  "opening": "前3章钩子、承诺、兑现节奏摘要",
  "chapter_1": "第一章结构，不放大段原文",
  "chapter_3": "第三章兑现方式",
  "payoff_pattern": "爽点/情绪收益模式",
  "notes": "可学习的结构规律与禁忌"
}
```

只放结构分析、摘要和你有权使用的文本。不要把未经授权的整章正文复制进来。
