# Compare: suspense_v10 vs suspense_v11
Generated 2026-06-10T20:18:38

| metric | suspense_v10 | suspense_v11 |
|---|---|---|
| chapters scored | 8 | 8 |
| avg score | 8.46 | 5.81 |
| min score | 8.20 | 3.50 |
| max score | 8.50 | 8.50 |
| chapters < 7.0 | 0 | 6 |
| book chars | 41336 | 38352 |
| force-accepts (log) | 0 | 6 |
| quality_debt events | 0 | 6 |
| gate_reject events | 0 | 0 |
| fossil warnings (log) | 0 | 21 |
| max fossils in one hit | 0 | 25 |
| scene-dedupe WARN | 1 | 3 |
| scene-dedupe BLOCK | 0 | 0 |
| LLM calls | 148 | 210 |
| LLM total minutes | 248.16 | 294.91 |
| planning share of LLM time | 0.55 | 0.62 |
| LLM minutes / scored chapter | 31.02 | 36.86 |

## Per-chapter scores
| ch | suspense_v10 | style_pen | suspense_v11 | style_pen |
|---|---|---|---|---|
| 1 | 8.5 | 0.0 | 8.5 | 0.0 |
| 2 | 8.2 | 0.0 | 8.0 | 0.0 |
| 3 | 8.5 | 0.0 | 5.5 | 0.0 |
| 4 | 8.5 | 0.0 | 5.5 | 0.0 |
| 5 | 8.5 | 0.0 | 5.5 | 0.0 |
| 6 | 8.5 | 0.0 | 5.5 | 0.0 |
| 7 | 8.5 | 0.0 | 3.5 | 0.0 |
| 8 | 8.5 | 0.0 | 4.5 | 0.0 |

## Config differences
| key | suspense_v10 | suspense_v11 |
|---|---|---|
| novel.beat_climax_tighten | <absent> | 2 |
| novel.beat_scheduler_enabled | <absent> | true |
| novel.narrative_mode | <absent> | balanced |
| novel.refine_adjacent_dedupe_enabled | <absent> | true |
| novel.refine_adjacent_dedupe_retries | <absent> | 1 |
| novel.refine_adjacent_sim_max | <absent> | 0.7 |
| novel.signature_trait_surface | <absent> | true |
| novel.style_cross_repeat_chapters | <absent> | 2 |
| novel.style_cross_repeat_enabled | <absent> | true |
| novel.style_cross_repeat_lookback | <absent> | 6 |
| novel.style_cross_repeat_min_len | <absent> | 7 |
| novel.style_cross_repeat_warn_count | <absent> | 4 |

## Heuristic verdict
- avg score favors **suspense_v10** by 2.65
- fewer sub-7.0 chapters: **suspense_v10** (0 vs 6)
- fewer force-accepts: **suspense_v10**
- cheaper per chapter: **suspense_v10** (31m vs 37m)
