# Compare: p4_score vs p4_det
Chapter range: 47..50 (chapter metrics only; log/jsonl totals cover the whole run)
Generated 2026-07-28T01:06:42

| metric | p4_score | p4_det |
|---|---|---|
| chapters scored | 4 | 4 |
| avg score | 7.67 | 7.14 |
| min score | 6.22 | 6.83 |
| max score | 8.24 | 7.45 |
| chapters < 7.0 | 1 | 2 |
| book chars | 220759 | 226992 |
| chapters with a replan | 1 | 1 |
| total revise rounds | 3 | 2 |
| first-pass-clean chapters | 0 | 1 |
| force-accepts (log) | 1 | 2 |
| quality_debt events | 17 | 17 |
| gate_reject events | 5 | 5 |
| fossil warnings (log) | 1 | 2 |
| max fossils in one hit | 0 | 1 |
| scene-dedupe WARN | 0 | 0 |
| scene-dedupe BLOCK | 0 | 0 |
| LLM calls | 59 | 65 |
| LLM total minutes | 92.25 | 114.94 |
| planning share of LLM time | 0.33 | 0.49 |
| LLM minutes / scored chapter | 23.06 | 28.73 |

## Per-chapter scores and rework
| ch | p4_score score | pen | rework | p4_det score | pen | rework |
|---|---|---|---|---|---|---|
| 47 | 8.2 | 0.0 | revise×1 | 6.9 | 0.0 | revise×1 |
| 48 | 8.0 | 0.0 | revise×1 fix×1 | 7.4 | 0.0 | revise×1 |
| 49 | 8.2 | 0.0 | revise×1 | 6.8 | 1.0 | clean |
| 50 | 6.2 | 1.0 | replan×1 debt | 7.4 | 0.0 | replan×1 debt |

## Config differences
| key | p4_score | p4_det |
|---|---|---|
| novel.rework_trigger | score | deterministic |

## Heuristic verdict
- avg score favors **p4_score** by 0.53
- fewer sub-7.0 chapters: **p4_score** (1 vs 2)
- fewer force-accepts: **p4_score**
- cheaper per chapter: **p4_score** (23m vs 29m)
