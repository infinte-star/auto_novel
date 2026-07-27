# Compare: ab_committee vs ab_arc
Chapter range: 26..28 (chapter metrics only; log/jsonl totals cover the whole run)
Generated 2026-07-27T23:52:22

| metric | ab_committee | ab_arc |
|---|---|---|
| chapters scored | 3 | 3 |
| avg score | 6.35 | 6.36 |
| min score | 5.80 | 4.80 |
| max score | 6.95 | 7.42 |
| chapters < 7.0 | 3 | 2 |
| book chars | 135092 | 132517 |
| chapters with a replan | 2 | 3 |
| total revise rounds | 1 | 0 |
| first-pass-clean chapters | 0 | 0 |
| force-accepts (log) | 2 | 3 |
| quality_debt events | 16 | 17 |
| gate_reject events | 4 | 5 |
| fossil warnings (log) | 0 | 2 |
| max fossils in one hit | 0 | 1 |
| scene-dedupe WARN | 0 | 0 |
| scene-dedupe BLOCK | 3 | 0 |
| LLM calls | 88 | 79 |
| LLM total minutes | 222.98 | 171.44 |
| planning share of LLM time | 0.69 | 0.41 |
| LLM minutes / scored chapter | 74.33 | 57.15 |

## Per-chapter scores and rework
| ch | ab_committee score | pen | rework | ab_arc score | pen | rework |
|---|---|---|---|---|---|---|
| 26 | 5.8 | 1.0 | replan×1 debt | 4.8 | 1.0 | replan×1 debt |
| 27 | 6.9 | 0.0 | replan×2 debt | 7.4 | 0.0 | replan×1 debt |
| 28 | 6.3 | 0.0 | retry×1 revise×1 fix×2 debt | 6.8 | 0.0 | replan×1 debt |

## Config differences
| key | ab_committee | ab_arc |
|---|---|---|
| novel.arc_planning_enabled | false | true |

## Heuristic verdict
- fewer sub-7.0 chapters: **ab_arc** (2 vs 3)
- fewer force-accepts: **ab_committee**
- cheaper per chapter: **ab_arc** (57m vs 74m)
