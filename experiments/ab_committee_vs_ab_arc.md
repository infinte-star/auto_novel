# Compare: ab_committee vs ab_arc
Generated 2026-07-27T23:28:53

| metric | ab_committee | ab_arc |
|---|---|---|
| chapters scored | 28 | 29 |
| avg score | 7.58 | 7.59 |
| min score | 4.60 | 4.60 |
| max score | 8.80 | 8.80 |
| chapters < 7.0 | 7 | 6 |
| book chars | 135092 | 132517 |
| retention index (0-10) | - | - |
| panel mean excitement | - | - |
| panel mean drop_rate | - | - |
| excitement troughs (<4) | 0 | 0 |
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
| LLM minutes / scored chapter | 7.96 | 5.91 |

## Per-chapter scores
| ch | ab_committee | style_pen | ab_arc | style_pen |
|---|---|---|---|---|
| 1 | 8.7 | 0.0 | 8.7 | 0.0 |
| 2 | 8.0 | 0.0 | 8.0 | 0.0 |
| 3 | 8.5 | 0.0 | 8.5 | 0.0 |
| 4 | 8.7 | 0.0 | 8.7 | 0.0 |
| 5 | 7.8 | 0.0 | 7.8 | 0.0 |
| 6 | 8.0 | 0.0 | 8.0 | 0.0 |
| 7 | 7.9 | 0.0 | 7.9 | 0.0 |
| 8 | 7.7 | 0.0 | 7.7 | 0.0 |
| 9 | 7.0 | 0.0 | 7.0 | 0.0 |
| 10 | 8.3 | 0.0 | 8.3 | 0.0 |
| 11 | 7.2 | 0.0 | 7.2 | 0.0 |
| 12 | 7.5 | 0.0 | 7.5 | 0.0 |
| 13 | 7.6 | 0.0 | 7.6 | 0.0 |
| 14 | 8.7 | 0.0 | 8.7 | 0.0 |
| 15 | 8.0 | 0.0 | 8.0 | 0.0 |
| 16 | 8.8 | 0.0 | 8.8 | 0.0 |
| 17 | 8.5 | 0.0 | 8.5 | 0.0 |
| 18 | 7.3 | 0.0 | 7.3 | 0.0 |
| 19 | 8.6 | 0.0 | 8.6 | 0.0 |
| 20 | 6.9 | 0.0 | 6.9 | 0.0 |
| 21 | 6.5 | 0.0 | 6.5 | 0.0 |
| 22 | 6.5 | 0.0 | 6.5 | 0.0 |
| 23 | 7.8 | 0.0 | 7.8 | 0.0 |
| 24 | 8.2 | 0.0 | 8.2 | 0.0 |
| 25 | 4.6 | 1.0 | 4.6 | 1.0 |
| 26 | 5.8 | 1.0 | 4.8 | 1.0 |
| 27 | 6.9 | 0.0 | 7.4 | 0.0 |
| 28 | 6.3 | 0.0 | 6.8 | 0.0 |
| 29 | - | - | 7.7 | 0.0 |

## Config differences
| key | ab_committee | ab_arc |
|---|---|---|
| novel.arc_planning_enabled | false | true |

## Heuristic verdict
- fewer sub-7.0 chapters: **ab_arc** (6 vs 7)
- fewer force-accepts: **ab_committee**
- cheaper per chapter: **ab_arc** (6m vs 8m)
