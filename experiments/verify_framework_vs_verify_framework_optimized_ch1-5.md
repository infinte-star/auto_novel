# Compare: verify_framework vs verify_framework_optimized
Chapter range: 1..5 (chapter metrics only; log/jsonl totals cover the whole run)
Generated 2026-08-03T19:18:34

| metric | verify_framework | verify_framework_optimized |
|---|---|---|
| chapters scored | 5 | 5 |
| avg score | 0.00 | 0.00 |
| min score | 0.00 | 0.00 |
| max score | 0.00 | 0.00 |
| chapters < 7.0 | 5 | 5 |
| book chars | 17056 | 14373 |
| chapters with a replan | 0 | 0 |
| total revise rounds | 0 | 0 |
| first-pass-clean chapters | 5 | 5 |
| force-accepts (log) | 0 | 0 |
| quality_debt events | 0 | 0 |
| gate_reject events | 0 | 0 |
| fossil warnings (log) | 0 | 0 |
| max fossils in one hit | 0 | 0 |
| scene-dedupe blocks | 0 | 0 |
| LLM calls | 30 | 30 |
| LLM calls / scored chapter | 6.00 | 6.00 |
| LLM total minutes | 31.83 | 43.20 |
| planning share of LLM time | 0.00 | 0.00 |
| LLM minutes / scored chapter | 6.37 | 8.64 |

## Per-chapter scores and rework
| ch | verify_framework score | pen | rework | verify_framework_optimized score | pen | rework |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | clean | 0.0 | 1.5 | clean |
| 2 | 0.0 | 0.0 | clean | 0.0 | 0.0 | clean |
| 3 | 0.0 | 0.0 | clean | 0.0 | 0.0 | clean |
| 4 | 0.0 | 0.0 | clean | 0.0 | 0.0 | clean |
| 5 | 0.0 | 0.0 | clean | 0.0 | 0.0 | clean |

## Config differences
(identical apart from paths/keys)

## Heuristic verdict
- cheaper per chapter: **verify_framework** (6m vs 9m)
