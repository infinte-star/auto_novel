# Compare: p4_score vs p4_det
Generated 2026-07-28T00:59:08

| metric | p4_score | p4_det |
|---|---|---|
| chapters scored | 50 | 49 |
| avg score | 7.98 | 7.95 |
| min score | 5.80 | 5.80 |
| max score | 9.00 | 9.00 |
| chapters < 7.0 | 5 | 6 |
| book chars | 220759 | 226992 |
| chapters with a replan | 1 | 0 |
| total revise rounds | 3 | 2 |
| first-pass-clean chapters | 46 | 47 |
| force-accepts (log) | 1 | 2 |
| quality_debt events | 17 | 17 |
| gate_reject events | 5 | 5 |
| fossil warnings (log) | 1 | 3 |
| max fossils in one hit | 0 | 1 |
| scene-dedupe WARN | 0 | 0 |
| scene-dedupe BLOCK | 0 | 0 |
| LLM calls | 54 | 64 |
| LLM total minutes | 85.42 | 111.53 |
| planning share of LLM time | 0.36 | 0.50 |
| LLM minutes / scored chapter | 1.71 | 2.28 |

## Per-chapter scores and rework
| ch | p4_score score | pen | rework | p4_det score | pen | rework |
|---|---|---|---|---|---|---|
| 1 | 9.0 | 0.0 | clean | 9.0 | 0.0 | clean |
| 2 | 9.0 | 0.0 | clean | 9.0 | 0.0 | clean |
| 3 | 8.4 | 0.0 | clean | 8.4 | 0.0 | clean |
| 4 | 8.7 | 0.0 | clean | 8.7 | 0.0 | clean |
| 5 | 8.5 | 0.0 | clean | 8.5 | 0.0 | clean |
| 6 | 7.2 | 0.0 | clean | 7.2 | 0.0 | clean |
| 7 | 8.5 | 0.0 | clean | 8.5 | 0.0 | clean |
| 8 | 8.7 | 0.0 | clean | 8.7 | 0.0 | clean |
| 9 | 8.1 | 0.0 | clean | 8.1 | 0.0 | clean |
| 10 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 11 | 7.4 | 0.0 | clean | 7.4 | 0.0 | clean |
| 12 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 13 | 8.4 | 0.0 | clean | 8.4 | 0.0 | clean |
| 14 | 8.2 | 0.0 | clean | 8.2 | 0.0 | clean |
| 15 | 7.1 | 0.0 | clean | 7.1 | 0.0 | clean |
| 16 | 8.7 | 0.0 | clean | 8.7 | 0.0 | clean |
| 17 | 8.4 | 0.0 | clean | 8.4 | 0.0 | clean |
| 18 | 8.6 | 0.0 | clean | 8.6 | 0.0 | clean |
| 19 | 7.8 | 0.0 | clean | 7.8 | 0.0 | clean |
| 20 | 8.1 | 0.0 | clean | 8.1 | 0.0 | clean |
| 21 | 8.6 | 0.0 | clean | 8.6 | 0.0 | clean |
| 22 | 8.1 | 0.0 | clean | 8.1 | 0.0 | clean |
| 23 | 7.7 | 0.0 | clean | 7.7 | 0.0 | clean |
| 24 | 7.0 | 0.0 | clean | 7.0 | 0.0 | clean |
| 25 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 26 | 5.8 | 0.0 | clean | 5.8 | 0.0 | clean |
| 27 | 7.6 | 0.0 | clean | 7.6 | 0.0 | clean |
| 28 | 7.9 | 0.0 | clean | 7.9 | 0.0 | clean |
| 29 | 8.2 | 0.0 | clean | 8.2 | 0.0 | clean |
| 30 | 6.4 | 0.0 | clean | 6.4 | 0.0 | clean |
| 31 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 32 | 8.3 | 0.0 | clean | 8.3 | 0.0 | clean |
| 33 | 6.8 | 0.0 | clean | 6.8 | 0.0 | clean |
| 34 | 7.3 | 0.0 | clean | 7.3 | 0.0 | clean |
| 35 | 6.9 | 0.0 | clean | 6.9 | 0.0 | clean |
| 36 | 8.1 | 0.0 | clean | 8.1 | 0.0 | clean |
| 37 | 7.9 | 0.0 | clean | 7.9 | 0.0 | clean |
| 38 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 39 | 8.0 | 0.0 | clean | 8.0 | 0.0 | clean |
| 40 | 8.2 | 0.0 | clean | 8.2 | 0.0 | clean |
| 41 | 8.1 | 0.0 | clean | 8.1 | 0.0 | clean |
| 42 | 9.0 | 0.0 | clean | 9.0 | 0.0 | clean |
| 43 | 8.6 | 0.0 | clean | 8.6 | 0.0 | clean |
| 44 | 8.8 | 0.0 | clean | 8.8 | 0.0 | clean |
| 45 | 7.2 | 0.0 | clean | 7.2 | 0.0 | clean |
| 46 | 8.7 | 0.0 | clean | 8.7 | 0.0 | clean |
| 47 | 8.2 | 0.0 | revise×1 | 6.9 | 0.0 | revise×1 |
| 48 | 8.0 | 0.0 | revise×1 fix×1 | 7.4 | 0.0 | revise×1 |
| 49 | 8.2 | 0.0 | revise×1 | 6.8 | 1.0 | clean |
| 50 | 6.2 | 1.0 | replan×1 debt | - | - | - |

## Config differences
| key | p4_score | p4_det |
|---|---|---|
| novel.rework_trigger | score | deterministic |

## Heuristic verdict
- fewer sub-7.0 chapters: **p4_score** (5 vs 6)
- fewer force-accepts: **p4_score**
- cheaper per chapter: **p4_score** (2m vs 2m)
