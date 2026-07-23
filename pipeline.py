from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from checkpoint import (
    CHAPTER_CURRENT_CHECKPOINT,
    bump_finalize_attempts,
    load_checkpoint,
    save_checkpoint,
    should_resume_existing_chapter,
)
from config import (
    Paths,
    book_reached_target,
    chapter_path,
    configured_api_endpoints,
    configured_api_endpoints_with_models,
    configured_review_endpoints,
    configured_role_endpoints,
    count_chars,
    ensure_project,
    find_last_chapter,
    book_is_consistent,
    get_paths,
    is_final_chapter,
    cost_savings_disabled,
    load_config,
    log,
    normalize_chapter,
    read_text,
    rebuild_book,
    safe_score,
    tail_text,
    write_text,
)
from llm import LLMClientPool
from memory import bootstrap, cacheable_prefix, cacheable_prefix_hit_rate, compress_all_memory, memory_context, should_compress_memory, writing_memory_context
from planning import create_plan
from review import adaptive_replan, anchor_completion_gate, horizon_review, review_chapter, should_replan, stage_review
from store import db_event, init_db, validate_plan_continuity
from writing import extract_events, revise_chapter, save_chapter, update_state_file, update_structured_state, write_chapter
from writing import apply_review_patches
import telemetry


class BackgroundTasks:
    """Run finalization/stage-review/memory-compress tasks off the critical path.

    Tasks are submitted with a label and kept in a list. The pipeline waits for
    completion only at shutdown or when a follow-up task explicitly depends on
    them. Exceptions are logged but do not crash the main loop.
    """

    def __init__(self, paths: Paths, conn: Any = None) -> None:
        self.paths = paths
        self.conn = conn
        self.lock = threading.Lock()
        self.tasks: list[tuple[str, threading.Thread, list[Exception]]] = []

    def submit(self, label: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        errors: list[Exception] = []

        def runner() -> None:
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
                log(self.paths, f"Background task {label} failed: {exc}")
            finally:
                # Each worker thread opened its own sqlite3 connection (via
                # ThreadLocalConn) on first DB access; close it on thread exit so
                # the many short-lived finalize threads don't leak connections.
                conn = self.conn
                if conn is not None and hasattr(conn, "close_current"):
                    try:
                        conn.close_current()
                    except Exception:
                        pass

        with self.lock:
            for existing_label, existing_thread, _ in self.tasks:
                if existing_label == label and existing_thread.is_alive():
                    log(self.paths, f"Skipping resubmit of {label}; already running")
                    return
        thread = threading.Thread(target=runner, name=f"bg-{label}", daemon=True)
        thread.start()
        with self.lock:
            self.tasks.append((label, thread, errors))
        log(self.paths, f"Background task {label} submitted")

    def wait_pending(self, timeout: float | None = None) -> None:
        """Wait for any currently-pending tasks to finish."""
        with self.lock:
            snapshot = list(self.tasks)
        for label, thread, errors in snapshot:
            if thread.is_alive():
                log(self.paths, f"Waiting for background task {label} to finish...")
            thread.join(timeout)
            if errors:
                log(self.paths, f"Background task {label} surfaced {len(errors)} error(s) during join")

    def wait_label(self, label: str, timeout: float | None = None) -> bool:
        """Join the most recently-submitted task with this label, if alive.

        Returns True if a matching task existed (whether or not it had finished).
        Other tasks are unaffected. Used to enforce barriers like
        "extract for chapter N must finish before chapter N+1 plans".
        """
        with self.lock:
            snapshot = list(self.tasks)
        match: tuple[str, threading.Thread, list[Exception]] | None = None
        for entry in reversed(snapshot):
            if entry[0] == label:
                match = entry
                break
        if match is None:
            return False
        _, thread, errors = match
        if thread.is_alive():
            log(self.paths, f"Waiting for background task {label} (barrier)...")
        thread.join(timeout)
        if errors:
            log(self.paths, f"Background task {label} surfaced {len(errors)} error(s) during barrier wait")
        return True

    def prune_done(self) -> None:
        with self.lock:
            alive = [(l, t, e) for (l, t, e) in self.tasks if t.is_alive()]
            done = [(l, t, e) for (l, t, e) in self.tasks if not t.is_alive()]
            # Keep only the most-recent 60 completed tasks so wait_label can still
            # find barriers from the immediately prior chapter, but the list doesn't
            # grow to thousands of entries over a long novel.
            self.tasks = alive + done[-60:]


def _beat_gate_one(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Deterministic beat-coverage gate + one targeted repair for a single draft.

    Runs quality.beat_coverage (non-LLM, anchor-fragment matching) on the draft.
    If plan beats are missing from the prose, spends ONE surgical low-temperature
    repair call (writing.repair_missing_beats) that weaves exactly the missing
    beats in, then re-measures. Keeps the repair only when coverage improved.
    Returns (text, coverage_report_or_None). Always non-fatal.
    """
    if not bool(config["novel"].get("beat_coverage_enabled", True)):
        return text, None
    try:
        from quality import beat_coverage

        cov = beat_coverage(text, plan, config)
        if not cov.get("enabled") or cov.get("passed"):
            return text, cov
        if int(config["novel"].get("beat_coverage_retry", 1)) <= 0:
            return text, cov
        from writing import repair_missing_beats

        repaired = repair_missing_beats(client, paths, config, chapter_num, plan, text, cov)
        if repaired == text:
            return text, cov
        cov2 = beat_coverage(repaired, plan, config)
        if float(cov2.get("coverage", 0.0)) >= float(cov.get("coverage", 0.0)):
            log(
                paths,
                f"Beat gate Ch{chapter_num}: repair coverage "
                f"{cov.get('coverage')} -> {cov2.get('coverage')} passed={cov2.get('passed')}",
            )
            return repaired, cov2
        log(paths, f"Beat gate Ch{chapter_num}: repair did not improve coverage; keeping original draft")
        return text, cov
    except Exception as exc:
        log(paths, f"Beat coverage gate failed (non-fatal) Ch{chapter_num}: {exc}")
        return text, None


def write_chapter_with_candidates(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
    num_candidates_override: int | None = None,
    base_temp_override: float | None = None,
    chapter_aux_cache: dict | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Generate N candidate chapter drafts in parallel, review each, keep the best.

    Returns (best_chapter_text, best_review_or_None). When N <= 1 falls back to single
    write_chapter with no review (review_round0 will be produced by the normal loop).

    `num_candidates_override` / `base_temp_override` let the structural-replan path
    force multi-candidate + lower-temperature sampling to dampen writer variance (a
    single replan draft once turned an 8.5 plan into a 5.5 chapter); averaging over
    several lower-temp drafts and keeping the best review is the cheap hedge.
    """
    n = int(num_candidates_override if num_candidates_override is not None
            else config["novel"].get("candidate_chapters", 1))
    base_temp = float(base_temp_override if base_temp_override is not None
                      else config["api"]["temperature"])
    if n <= 1:
        text = write_chapter(
            client, paths, conn, config, chapter_num, plan, decision, tail,
            cached_memory=cached_memory,
            temperature=base_temp if base_temp_override is not None else None,
        )
        text, cov = _beat_gate_one(client, paths, config, chapter_num, plan, text)
        if cov is not None:
            save_checkpoint(paths, chapter_num, "beat_coverage.json", {"drafts": [{"idx": 0, **{k: cov.get(k) for k in ("passed", "coverage", "missing_beats")}}]})
        return text, None

    max_workers = int(config["novel"].get("max_parallel_workers", 8))

    def write_one(idx: int) -> str:
        # Spread temperatures around base_temp so candidates diverge.
        offset = (idx - (n - 1) / 2) * 0.08
        temp = max(0.1, min(1.2, base_temp + offset))
        try:
            text = write_chapter(
                client, paths, conn, config, chapter_num, plan, decision, tail,
                cached_memory=cached_memory, temperature=temp,
            )
            return text
        except Exception as exc:
            log(paths, f"Candidate chapter draft idx={idx} failed: {exc}")
            return ""

    drafts: list[str] = [""] * n
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        futures = {executor.submit(write_one, idx): idx for idx in range(n)}
        for future in as_completed(futures):
            idx = futures[future]
            drafts[idx] = future.result() or ""

    valid = [(idx, text) for idx, text in enumerate(drafts) if text and len(text.strip()) >= 500]
    if not valid:
        raise RuntimeError(f"All {n} candidate chapter drafts failed for Ch{chapter_num}")

    # Deterministic beat-coverage gate per draft (non-LLM check; one surgical
    # repair call only for drafts that dropped plan beats). Run before review so
    # the reviewer scores the repaired text, and so beat-complete drafts win ties.
    beat_cov: dict[int, dict[str, Any] | None] = {}
    if bool(config["novel"].get("beat_coverage_enabled", True)):
        def gate_one(item: tuple[int, str]) -> tuple[int, str, dict[str, Any] | None]:
            idx, text = item
            new_text, cov = _beat_gate_one(client, paths, config, chapter_num, plan, text)
            return idx, new_text, cov

        gated: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(valid))) as executor:
            futures = [executor.submit(gate_one, item) for item in valid]
            for future in as_completed(futures):
                idx, new_text, cov = future.result()
                gated[idx] = new_text
                beat_cov[idx] = cov
        valid = [(idx, gated.get(idx, text)) for idx, text in valid]
        try:
            save_checkpoint(paths, chapter_num, "beat_coverage.json", {
                "drafts": [
                    {"idx": idx, "passed": (beat_cov.get(idx) or {}).get("passed"),
                     "coverage": (beat_cov.get(idx) or {}).get("coverage"),
                     "missing_beats": (beat_cov.get(idx) or {}).get("missing_beats")}
                    for idx, _ in valid
                ],
            })
        except Exception:
            pass

    # P0-2: Deterministic pre-screen (style_health + cross_chapter_repetition)
    # Runs after beat gate, before LLM review. Drafts that would trigger gate_rejects
    # are filtered out early, saving review cost and preventing low-quality drafts
    # from winning on score alone when they'd be rejected anyway.
    if bool(config["novel"].get("candidate_prescreen_enabled", True)) and len(valid) > 1:
        try:
            from quality import style_health, cross_chapter_repetition
            from store import recent_metrics

            # Collect recent chapters' text for cross_chapter check
            prior_texts: list[str] = []
            lookback = int(config["novel"].get("style_cross_repeat_lookback", 6))
            for ch in range(max(1, chapter_num - lookback), chapter_num):
                ch_file = paths.chapters_dir / f"{ch:04d}.md"
                if ch_file.exists():
                    try:
                        prior_texts.append(ch_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

            # Pre-screen each draft
            screened: list[tuple[int, str, dict[str, Any]]] = []
            for idx, text in valid:
                # Style health
                try:
                    em_history = []
                    tech_history = []
                    recent = recent_metrics(conn, limit=int(config["novel"].get("style_em_dash_trend_window", 5)))
                    for row in recent:
                        em = row.get("em_dash_per_kchar")
                        if em is not None:
                            em_history.append(float(em))
                        tv = row.get("tech_per_kchar")
                        if tv is not None:
                            tech_history.append(float(tv))
                except Exception:
                    em_history = []
                    tech_history = []

                sh = style_health(text, config, em_history, tech_history or None)
                sh_penalty = sh.get("penalty", 0.0)
                sh_flags = sh.get("flags", [])

                # Cross-chapter repetition
                cr = cross_chapter_repetition(text, prior_texts, config)
                cr_level = str(cr.get("level", "")).strip()
                cr_penalty = cr.get("penalty", 0.0)

                total_penalty = sh_penalty + cr_penalty
                block_threshold = float(config["novel"].get("candidate_prescreen_penalty_block", 3.0))

                # Block drafts that would trigger gate_rejects or exceed penalty threshold
                if cr_level == "reject":
                    log(paths, f"Pre-screen BLOCK Ch{chapter_num} idx={idx}: cross_repeat reject (fossils={cr.get('metrics', {}).get('cross_repeat_fossils')})")
                    continue
                elif total_penalty >= block_threshold:
                    log(paths, f"Pre-screen BLOCK Ch{chapter_num} idx={idx}: total_penalty={total_penalty:.2f} (style={sh_penalty:.2f} cross={cr_penalty:.2f})")
                    continue
                else:
                    screened.append((idx, text, {"style_penalty": sh_penalty, "cross_penalty": cr_penalty, "style_flags": sh_flags}))

            if screened:
                valid = [(idx, text) for idx, text, _ in screened]
                log(paths, f"Pre-screen kept {len(screened)}/{len(valid) + len(screened)} drafts for Ch{chapter_num}")
                # Save pre-screen results
                try:
                    save_checkpoint(paths, chapter_num, "candidate_prescreen.json", {
                        "kept": [{"idx": idx, "style_penalty": m["style_penalty"], "cross_penalty": m["cross_penalty"]} for idx, _, m in screened],
                        "total_candidates": len(valid) + len(screened),
                    })
                except Exception:
                    pass
            else:
                # All drafts blocked — check if this is catastrophic (all fossils >= threshold)
                # If so, trigger plan-level replan instead of accepting garbage
                all_fossil_counts = []
                for idx, text in valid:
                    cr_check = cross_chapter_repetition(text, prior_texts, config)
                    fossils = cr_check.get("metrics", {}).get("cross_repeat_fossils", 0)
                    all_fossil_counts.append(fossils)

                min_fossils = min(all_fossil_counts) if all_fossil_counts else 0
                fossil_catastrophe_threshold = int(config["novel"].get("prescreen_fossil_catastrophe_threshold", 5))

                if min_fossils >= fossil_catastrophe_threshold:
                    # All candidates have >= threshold fossils — plan is generating garbage
                    # Signal to caller that plan-level replan is needed
                    log(
                        paths,
                        f"Pre-screen CATASTROPHE Ch{chapter_num}: ALL candidates have fossils >= {fossil_catastrophe_threshold} "
                        f"(min={min_fossils}, counts={all_fossil_counts}). Plan-level replan required.",
                    )
                    # Return None to signal catastrophic failure to caller
                    return None, {"catastrophe": "fossil_prescreen", "min_fossils": min_fossils, "all_counts": all_fossil_counts}
                else:
                    # Fall back to least-bad one (original behavior for non-catastrophic blocks)
                    log(paths, f"Pre-screen blocked ALL drafts for Ch{chapter_num}; keeping least-penalty draft")
        except Exception as exc:
            log(paths, f"Candidate pre-screen failed (non-fatal) Ch{chapter_num}: {exc}")

    if len(valid) == 1:
        idx, text = valid[0]
        if bool(config["novel"].get("em_dash_reduce_enabled", True)):
            try:
                from quality import style_health as _sh1, reduce_em_dash_density as _red1
                _m1 = _sh1(text, config)
                if float(_m1.get("metrics", {}).get("em_dash_per_kchar", 0)) > float(config["novel"].get("em_dash_reduce_target_per_kchar", 3.0)):
                    text = _red1(text, config)
            except Exception:
                pass
        log(paths, f"Only 1/{n} valid draft for Ch{chapter_num} idx={idx}; skipping comparative review")
        return text, None

    # Pre-build review auxiliary context once; all candidate reviews share it.
    # Reuse caller-provided aux cache if available; build one only if not.
    if chapter_aux_cache is not None:
        _aux = chapter_aux_cache
    else:
        try:
            from review import build_chapter_aux_cache
            _aux = build_chapter_aux_cache(paths, conn, config, chapter_num)
        except Exception:
            _aux = None

    def review_one(item: tuple[int, str]) -> tuple[int, str, dict[str, Any]]:
        idx, text = item
        # Pre-review em-dash reduction: clean the text before scoring so the
        # style_health penalty reflects the final saved version, not the raw draft.
        if bool(config["novel"].get("em_dash_reduce_enabled", True)):
            try:
                from quality import style_health as _sh_pre, reduce_em_dash_density as _red_pre
                _sh_m = _sh_pre(text, config)
                _em_d = float(_sh_m.get("metrics", {}).get("em_dash_per_kchar", 0))
                _em_tgt = float(config["novel"].get("em_dash_reduce_target_per_kchar", 3.0))
                if _em_d > _em_tgt:
                    text = _red_pre(text, config)
            except Exception:
                pass
        try:
            report = review_chapter(client, paths, conn, config, chapter_num, plan, text, tail, cached_memory=cached_memory, chapter_aux_cache=_aux)
        except Exception as exc:
            log(paths, f"Candidate draft review idx={idx} failed: {exc}")
            report = {"score": 0, "accepted": False}
        return idx, text, report

    reviewed: list[tuple[int, str, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(valid))) as executor:
        futures = [executor.submit(review_one, item) for item in valid]
        for future in as_completed(futures):
            reviewed.append(future.result())

    # Beat-complete drafts win first; score breaks ties. A draft that still fails
    # the beat gate after its repair shot is structurally incomplete — a slightly
    # higher prose score cannot compensate for an absent plan beat (-1.0 each at
    # final review anyway).
    def _beat_passed(idx: int) -> bool:
        cov = beat_cov.get(idx)
        return bool(cov.get("passed")) if isinstance(cov, dict) else True

    reviewed.sort(key=lambda r: (_beat_passed(r[0]), safe_score(r[2].get("score", 0))), reverse=True)
    best_idx, best_text, best_review = reviewed[0]
    scores = [(idx, safe_score(rep.get("score", 0)), _beat_passed(idx)) for idx, _, rep in reviewed]
    log(paths, f"Selected Ch{chapter_num} draft idx={best_idx} score={best_review.get('score')}/10 candidates(idx,score,beats_ok)={scores}")
    return best_text, best_review


def _fallback_extraction(plan: dict[str, Any], review: dict[str, Any], chapter_num: int, error: str) -> dict[str, Any]:
    """Minimal extraction payload written when extract_events cannot run.

    Used both when the LLM extraction raises and when finalize is force-completed
    after exhausting retries, so the resume markers always get written and the
    main loop can advance instead of re-entering the same chapter forever.
    """
    return {
        "title": plan.get("title", f"Ch{chapter_num}"),
        "events": [
            {
                "type": "plot",
                "summary": f"Ch{chapter_num} saved; LLM extraction failed, fallback event recorded.",
                "effects": [],
            }
        ],
        "entities": [],
        "threads": [],
        "causal_links": [],
        "metrics": {
            "payoff_type": plan.get("payoff_type"),
            "conflict_type": plan.get("conflict_type"),
            "tension": None,
            "novelty": None,
            "hook_strength": review.get("hook_strength"),
            "emotional_tone": "",
        },
        "memory_updates": {"bible": [], "characters": [], "timeline": [], "threads": []},
        "fallback_error": str(error),
    }


class _LocalFixDone(Exception):
    """Internal control-flow sentinel: the local-fix route handled a below-threshold
    chapter, so the full-replan body should be skipped (caught immediately)."""


def _build_replan_feedback(review: dict[str, Any]) -> str:
    """Distil the failing review of THIS chapter's previous version into a compact,
    concrete feedback block for the quality-replan's plan generator.

    Pulls the reviewer's free-text problems, structured contract violations, the
    deterministic style metrics (so the replan knows the prose collapsed even when
    the reviewer's own self-report was unreliable), and the audit mismatch. This is
    what turns the replan from a blind retry into a targeted fix.
    """
    lines: list[str] = []
    score = review.get("score")
    if score is not None:
        lines.append(f"- 上一版总分：{score}/10（未达阈值）")
    for p in (review.get("problems") or [])[:6]:
        t = str(p).strip()
        if t:
            lines.append(f"- 评审问题：{t[:160]}")
    for cv in (review.get("contract_violations") or [])[:4]:
        if isinstance(cv, dict):
            rule = str(cv.get("rule") or cv.get("type") or "?")[:80]
            prose = str(cv.get("prose") or "")[:60]
            lines.append(f"- 契约违约({cv.get('severity','?')})：{rule}｜原文「{prose}」")
    sh = review.get("style_health") or {}
    metrics = sh.get("metrics") or {}
    if metrics:
        em = metrics.get("em_dash_per_kchar")
        frag = metrics.get("fragment_line_ratio")
        avg = metrics.get("avg_sentence_chars")
        if em is not None or frag is not None:
            lines.append(
                f"- 文体实测：破折号/千字={em}，碎句行占比={frag}，平均句长={avg}（实测值，"
                f"不要相信上一版自评的文体分；本章必须显著降低破折号与碎句）"
            )
    mismatch = review.get("style_audit_mismatch")
    if mismatch:
        lines.append("- 注意：上一版评审低报了文体问题，实际碎片化远比它自报严重，务必正面修复。")
    # Deterministic gate-reject evidence: concrete fossil clauses / overlap
    # metrics measured against the actual prior chapters. This is the strongest
    # signal available — it tells the new plan exactly which sentences and
    # scene shapes are forbidden, instead of a vague "be less repetitive".
    for g in (review.get("gate_rejects") or [])[:3]:
        if not isinstance(g, dict):
            continue
        gate = str(g.get("gate", "?"))
        ev = g.get("evidence") or {}
        if gate == "cross_chapter_repetition":
            examples = "；".join(str(e) for e in (ev.get("examples") or [])[:5])
            lines.append(
                f"- 质量门作废（文体化石复读，{ev.get('fossils')} 处）：以下句子/比喻已在多章重复，"
                f"新稿绝对禁止出现近似表达：{examples}"
            )
        elif gate == "adjacent_repetition":
            m = ev.get("metrics") or {}
            lines.append(
                f"- 质量门作废（逐字复述上一章，clause_overlap={m.get('clause_overlap')}）："
                "新稿必须从上一章结尾之后的【新】事件写起，上一章场景至多一句带过。"
            )
        elif gate == "book_wide_fossils":
            phrases = "、".join(str(p) for p in (g.get("phrases") or [])[:8])
            lines.append(
                f"- 质量门作废（全书微动作化石，{g.get('count')} 处）：以下片段已僵化为机械口癖、"
                f"贯穿全书反复出现，新稿必须换用不同动作落点与句式，严禁复现：{phrases}"
            )
        for d in (g.get("directives") or [])[:2]:
            lines.append(f"- 质量门指令：{str(d)[:160]}")
    return "\n".join(lines) if lines else ""


def _detect_quality_degradation(
    paths: Paths, conn: Any, config: dict[str, Any], chapter_num: int
) -> dict[str, Any] | None:
    """Mid-book quality-collapse early warning.

    The circuit breaker only fires after 2 consecutive chapters are force-accepted
    below 6.0 — by then the book is already broken (dushi_nuelian rode a -2.5 slide
    from Ch11 to a Ch41 score of 1.0). This detector catches the DOWNWARD TREND
    early and writes a recovery directive (consumed by the writer prompt and the
    planning candidate-count upshift) so the engine RECOVERS instead of HALTS.

    Returns the recovery directive dict when degradation is detected, else None.
    """
    cfg = config["novel"]
    if not bool(cfg.get("degradation_alert_enabled", True)):
        return None
    window = int(cfg.get("degradation_window", 4))
    # Warmup: never fire before there is a stable early baseline to fall from.
    if chapter_num < window + int(cfg.get("degradation_warmup", 2)):
        return None
    from store import recent_metrics
    rows = recent_metrics(conn, window)
    if len(rows) < window:
        return None
    rows = sorted(rows, key=lambda r: int(r.get("chapter", 0)))  # oldest→newest
    scores = [safe_score(r.get("score", 0)) for r in rows]
    penalties = [float(r.get("style_penalty", 0) or 0) for r in rows]
    avg = sum(scores) / len(scores)

    triggers: list[str] = []
    if avg < float(cfg.get("degradation_alert_score", 7.3)):
        triggers.append(f"近{window}章均分{avg:.1f}")
    if len(scores) >= 3 and scores[-1] < scores[-2] < scores[-3]:
        drop = scores[-3] - scores[-1]
        if drop >= float(cfg.get("degradation_alert_drop", 1.0)):
            triggers.append(f"连续3章下滑累计{drop:.1f}")
    if penalties and (sum(penalties) / len(penalties)) >= 1.0:
        triggers.append(f"文体惩罚均值{sum(penalties) / len(penalties):.1f}")
    if not triggers:
        return None

    return {
        "chapter": chapter_num,
        "active_until": chapter_num + int(cfg.get("degradation_recovery_chapters", 3)),
        "reason": "；".join(triggers),
        "avg_score": round(avg, 2),
        "directive": (
            "质量恢复模式：近几章质量持续下滑。本章起回到前期的叙事密度与文体节奏，"
            "停止堆砌技术名词/伪科学概念/重复的身体代价描写；每一段都必须有新信息或冲突推进，"
            "对白要承担推进作用，避免空转的内心独白与环境堆砌。"
        ),
    }


def _recent_replan_ineffective(paths: Paths, chapter_num: int, config: dict[str, Any]) -> bool:
    """True when the last `replan_max_attempts` structural replans (with a
    measured before/after) all gained less than `replan_min_gain`.

    gudai50_v2 burned $88.9 / 30 structural_diagnose calls largely on marginal
    chapters where replan repeatedly 'did not improve'. When the book's recent
    replans are not paying off, the ROI breaker stops throwing a full
    diagnose+replan+N-draft cycle at a chapter that is only marginally short, and
    accepts the best draft with quality debt instead.
    """
    n = int(config["novel"].get("replan_max_attempts", 2))
    if n <= 0:
        return False
    min_gain = float(config["novel"].get("replan_min_gain", 0.3))
    seen = 0
    for ch in range(chapter_num - 1, 0, -1):
        d = load_checkpoint(paths, ch, "quality_replan_done.json")
        if not isinstance(d, dict):
            continue
        sb, sa = d.get("score_before"), d.get("score_after")
        if sb is None or sa is None:
            continue  # local-route / roi-stop entries carry no before/after
        seen += 1
        if (safe_score(sa) - safe_score(sb)) >= min_gain:
            return False  # a recent replan DID work — keep trying
        if seen >= n:
            return True
    return False


def _apply_force_accept_patches(
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    review: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Before force-accepting quality debt, land review patches if possible.

    The normal revise loop skips patches for structural failures and tries a full
    replan instead. If that still leaves a chapter below threshold (e.g. 7.8) and
    we must accept to keep the pipeline moving, the review's concrete patches are
    still useful for low-risk local fixes: clarify a clue, align a direction,
    add a missing reaction. Apply them once, save the patched text, and carry the
    patch results in final_review / quality_debt so the refine pass knows what
    was already handled.
    """
    if not bool(config["novel"].get("quality_debt_apply_patches", True)):
        return chapter, review
    patches = review.get("patches") if isinstance(review, dict) else None
    if not isinstance(patches, list) or not patches:
        return chapter, review
    patched, results = apply_review_patches(chapter, patches)
    applied = sum(1 for r in results if r.get("applied"))
    total = len(results)
    if applied <= 0:
        log(paths, f"Quality-debt patch landing Ch{chapter_num}: no patches applied ({applied}/{total})")
        new_review = dict(review)
        new_review["quality_debt_patch_results"] = results
        return chapter, new_review
    patched = normalize_chapter(patched)
    save_checkpoint(paths, chapter_num, f"quality_debt_patched.md", patched)
    new_review = dict(review)
    new_review["quality_debt_patches_applied"] = applied
    new_review["quality_debt_patch_total"] = total
    new_review["quality_debt_patch_results"] = results
    new_review.setdefault("problems", []).append(
        f"QUALITY_DEBT_PATCHED: 强制接受前已自动应用 {applied}/{total} 条评审补丁；剩余问题交给精修优先处理。"
    )
    log(paths, f"Quality-debt patch landing Ch{chapter_num}: applied={applied}/{total}")
    return patched, new_review


def _classify_replan_failure(review: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    """Decide whether a below-threshold chapter needs a full structural replan or
    just a targeted local fix.

    A *full chapter replan* (regenerate plan → rewrite → review) is expensive and
    high-variance — three of our historical replans came back "did not improve".
    It is only worth it for STRUCTURAL failures: the scene itself is wrong
    (multiple low dimensions, scene雷同 with recent chapters, missing payoff).

    A *local* failure — a single dimension dipped, a contract violation that has a
    locatable patch, or pure prose collapse — is better served by extra surgical
    revise rounds (apply_review_patches) which keep the working scene and surgically
    fix the flaw, rather than rolling the dice on a whole new scene.

    Returns (kind, reason) where kind is "local" or "structural".
    """
    cfg = config["novel"]
    # Prefer the structured failure taxonomy when review.py tagged this report
    # (additive `failure_codes`). It maps the chapter's named failure modes to a
    # precise fix route; we collapse that to the pipeline's binary local/structural
    # contract. Only short-circuits the heuristics below when it yields a decisive
    # STRUCTURAL verdict — a "local" taxonomy read still falls through so the finer
    # heuristics (intra-recap vs style-collapse vs patchable) pick the exact path.
    if bool(cfg.get("failure_taxonomy_enabled", True)):
        try:
            import taxonomy
            codes = review.get("failure_codes") or []
            if codes and taxonomy.replan_kind(codes) == "structural":
                route = taxonomy.dominant_route(codes)
                return "structural", f"失败分类学判定结构性重做（codes={codes[:4]}, route={route}）"
        except Exception:
            pass
    # Deterministic gate rejects (cross-chapter fossil collapse, adjacent
    # re-narration) are ALWAYS structural: the draft itself is a write-off and
    # wording-level patches measurably cannot fix it. Checked first so no other
    # heuristic can downgrade it to "local".
    grs = review.get("gate_rejects") or []
    if grs:
        names = [str(g.get("gate", "?")) for g in grs if isinstance(g, dict)]
        return "structural", f"确定性质量门作废本稿（{', '.join(names[:3])}）—必须重做，不可修补"
    # Per-dimension scores; count how many fell materially below threshold.
    threshold = float(cfg.get("quality_threshold", 8.0))
    dims = {
        "readthrough": safe_score(review.get("readthrough_score", 0)),
        "hook": safe_score(review.get("hook_score", review.get("hook_strength", 0))),
        "payoff": safe_score(review.get("payoff_score", 0)),
        "novelty": safe_score(review.get("novelty_score", 0)),
        "prose": safe_score(review.get("prose_score", review.get("aesthetic_score", 0))),
        "continuity": safe_score(review.get("continuity_score", 0)),
    }
    weak = {k: v for k, v in dims.items() if v and v < threshold - 0.5}

    # KEY-DIM structural shortfall: a low novelty/payoff is a SCENE-DESIGN flaw,
    # not a prose flaw — revise patches (which edit wording in place) cannot make
    # a scene more novel or make a payoff actually land. So even a single key-dim
    # dip below its floor is structural (matches review.py's key_dimension_floor
    # cap, which routes such chapters here). Gated by the same floor toggle.
    if bool(cfg.get("key_dimension_floor_enabled", True)):
        nov_floor = float(cfg.get("novelty_floor", 7.0))
        pay_floor = float(cfg.get("payoff_floor", 7.0))
        if (dims["novelty"] and dims["novelty"] < nov_floor) or (
            dims["payoff"] and dims["payoff"] < pay_floor
        ):
            bad = []
            if dims["novelty"] and dims["novelty"] < nov_floor:
                bad.append(f"novelty={dims['novelty']:.1f}")
            if dims["payoff"] and dims["payoff"] < pay_floor:
                bad.append(f"payoff={dims['payoff']:.1f}")
            return "structural", f"关键维度低于硬地板（{', '.join(bad)}）—场景设计问题，revise 无效"

    # Scene雷同 / payoff缺失 are inherently structural — the scene design is wrong.
    blob = " ".join(str(p) for p in (review.get("problems") or []))
    rr = " ".join(str(p) for p in (review.get("rhythm_risks") or []))
    # Intra-chapter recap (a zero-增量 summary ENDING) is the exception: the scene
    # itself is fine, only the last ~600 chars re-state earlier content. That's a
    # surgical ending rewrite, NOT a whole-scene replan (which kept coming back
    # "did not improve" on suspense_10ch Ch7). Route it to local fix so the revise
    # patch path can rewrite just the tail.
    intra = review.get("intra_chapter_repetition") or {}
    only_intra_recap = (
        str(intra.get("level")) in ("warn", "block")
        and "RECAP" in blob
        and not any(m in (blob + rr) for m in ("雷同", "原地打转", "审美疲劳", "payoff", "兑现", "没有推进", "停滞", "scene_dedupe"))
    )
    if only_intra_recap:
        return "local", f"章末零增量总结（tail_recap={intra.get('metrics', {}).get('tail_recap_ratio')}）—只需重写结尾，非全章重做"

    structural_markers = ("雷同", "重复", "原地打转", "审美疲劳", "payoff", "兑现", "没有推进", "停滞", "scene_dedupe")
    if any(m in (blob + rr) for m in structural_markers):
        return "structural", f"命中结构性问题标记（{[m for m in structural_markers if m in (blob+rr)][:3]}）"

    has_patches = bool(review.get("patches"))

    # `prose` dips are surgically fixable (the revise patch path targets exactly
    # the collapsed paragraphs), so a low prose score is NOT itself structural.
    # Only count NON-prose weak dimensions toward the "broadly underperforming"
    # structural test — otherwise a pure em-dash/碎句 collapse (the canonical
    # local case) gets misrouted to a high-variance full replan.
    weak_nonprose = {k: v for k, v in weak.items() if k != "prose"}
    if len(weak_nonprose) >= 2:
        return "structural", f"{len(weak_nonprose)} 个非文体维度显著偏低（{ {k: round(v,1) for k,v in weak_nonprose.items()} }）"

    # A contract violation that carries no usable patch locator needs the scene
    # rethought (ability/modality drift baked into the scene premise).
    cvs = review.get("contract_violations") or []
    if cvs and not has_patches:
        return "structural", "存在契约违约且无可定位补丁"

    # Style collapse (em-dash / fragment / telegraphic shorts) is a PROSE-level
    # habit, NOT a scene-design flaw — replanning the outline rewrites the same
    # drifted voice and cannot fix it. The deterministic style_health gate catches
    # the collapse even when the LLM reviewer (whose own voice has drifted with the
    # prose) self-reports the style as fine and therefore emits NO patches. Without
    # this branch such a chapter hits the "no patches → structural" fallback below
    # and burns an expensive full replan that does not improve the style (observed:
    # gudai50_v2 Ch20-24, em 6.6→8.8 for 5 chapters, every replan "did not improve").
    # Route it to a LOCAL fix so revise_chapter's de-em-dash rewrite (REVISE_SYSTEM
    # 文风塌缩禁令) actually runs on the working scene. Only when the content
    # dimensions are otherwise healthy (no non-prose dimension is weak) — a genuine
    # multi-dimension shortfall already returned structural above.
    sh = review.get("style_health") if isinstance(review, dict) else None
    sh_penalty = safe_score(sh.get("penalty", 0)) if isinstance(sh, dict) else 0.0
    if sh_penalty > 0 and not weak_nonprose:
        return "local", (
            f"文体塌缩（style_penalty={sh_penalty:.1f}，破折号/碎句），内容各维度健康"
            "—prose 级问题，走定向改写而非结构重做"
        )

    # Otherwise: single weak dimension and/or style collapse with patches available.
    if has_patches:
        return "local", "单一维度短板/文体问题且评审给出了可应用补丁"
    # Fall back to structural when we have nothing surgical to apply.
    return "structural", "无可应用补丁，回退到整章重做"



def _build_writer_tail(paths: Paths, config: dict[str, Any], chapter_num: int) -> str:
    """Return the previous-chapter tail for the writer prompt, sanitized (O3).

    Repetition is self-reinforcing: when a force-accepted low-score chapter
    (often itself a near-verbatim re-narration) becomes the next chapter's
    "上章结尾" context, the writer's most probable continuation is more of the
    same — observed as suspense_v11's 5-chapter death spiral (5.5/5.5/5.5/3.5/4.5).
    When the previous chapter was force-accepted below the floor, anchor the
    tail on the last GOOD chapter's ending instead and reduce the poisoned
    chapters to factual one-line event summaries (their plot still happened and
    must not be re-dramatized; their prose must not be imitated).
    """
    tail_chars = int(config["novel"]["recent_tail_chars"])
    raw_tail = tail_text(paths.book, tail_chars)
    if chapter_num <= 1 or not bool(config["novel"].get("tail_sanitize_enabled", True)):
        return raw_tail
    floor = float(config["novel"].get("tail_sanitize_score_floor", 6.0))
    prev = chapter_num - 1
    prev_debt = load_checkpoint(paths, prev, "quality_debt.json")
    if not (isinstance(prev_debt, dict) and safe_score(prev_debt.get("score", 10)) < floor):
        return raw_tail
    good: int | None = None
    skipped: list[int] = []
    for ch in range(prev, 0, -1):
        d = load_checkpoint(paths, ch, "quality_debt.json")
        if isinstance(d, dict) and safe_score(d.get("score", 10)) < floor:
            skipped.append(ch)
            continue
        good = ch
        break
    if good is None or not skipped:
        return raw_tail
    good_text = read_text(chapter_path(paths, good))
    if not good_text.strip():
        return raw_tail
    summaries: list[str] = []
    for ch in sorted(skipped):
        extraction = load_checkpoint(paths, ch, "extraction.json") or {}
        title = str(extraction.get("title") or f"Ch{ch}")
        evs = [
            str(e.get("summary", "")).strip()
            for e in (extraction.get("events") or [])
            if isinstance(e, dict) and str(e.get("summary", "")).strip()
        ][:3]
        summaries.append(f"- 第{ch}章「{title}」：" + ("；".join(evs) if evs else "（无事件摘要）"))
    log(
        paths,
        f"Tail sanitized for Ch{chapter_num}: previous chapter(s) {sorted(skipped)} were "
        f"force-accepted below {floor}; anchoring on Ch{good} ending + factual summaries.",
    )
    return (
        good_text[-tail_chars:]
        + f"\n\n【续写衔接说明：以上是第{good}章结尾，仅作文风与场景参照。"
        f"其后第{min(skipped)}–{max(skipped)}章质量不达标，不提供原文，只给事实梗概——"
        "这些事件【已经发生】，本章严禁重演或复述它们，只能在其结果之上推进全新剧情：】\n"
        + "\n".join(summaries)
    )


def generate_one_chapter(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    background: BackgroundTasks | None = None,
    resume: bool = False,
) -> None:
    if background is not None and chapter_num > 1:
        # Block until the previous chapter's extract + structured-state writes
        # are durable, so memory_context sees current metrics/threads/entities.
        background.wait_label(f"chapter_finalize_ch{chapter_num - 1}")
        prev = chapter_num - 1
        cold_every = int(config["novel"].get("cold_reader_every", 10))
        if (
            bool(config["novel"].get("pack_review_barrier", True))
            and cold_every > 0
            and prev % cold_every == 0
        ):
            background.wait_label(f"horizon_review_ch{prev}")
        stage_every = int(config["novel"].get("stage_review_every", 20))
        if (
            bool(config["novel"].get("stage_review_barrier", True))
            and stage_every > 0
            and prev % stage_every == 0
        ):
            background.wait_label(f"stage_review_ch{prev}")
        if bool(config["novel"].get("prefetch_next_plan", False)):
            # If a prefetch task ran for this chapter, ensure its checkpoint is
            # flushed before create_plan tries to resume from it.
            background.wait_label(f"prefetch_plan_ch{chapter_num}")
    tail = _build_writer_tail(paths, config, chapter_num)
    # P2 降本：cached_memory 只被规划侧消费（create_plan/validate），
    # 用 plan_memory_chars 封顶（tier1/2 完整，tier3/4 截断）。写作/评审用
    # writing_memory_context，评审自建，均不受影响；不动 cacheable_prefix。
    cached_memory = memory_context(
        paths, conn, config,
        max_chars=int(config["novel"].get("plan_memory_chars", 60000) or 0),
    )
    # Smaller context for write/revise/review hot path to reduce prefill time.
    writing_memory = writing_memory_context(paths, conn, config)
    # Build review auxiliary context once per chapter so all review_chapter
    # calls (main loop + local_fix) share pre-fetched DB/file results.
    try:
        from review import build_chapter_aux_cache
        _chapter_aux: dict | None = build_chapter_aux_cache(paths, conn, config, chapter_num)
    except Exception:
        _chapter_aux = None
    final_payload = load_checkpoint(paths, chapter_num, "validated_plan.json")
    if isinstance(final_payload, dict) and final_payload.get("plan") and final_payload.get("decision"):
        log(paths, f"Resuming validated plan Ch{chapter_num}")
        plan = final_payload["plan"]
        decision = final_payload["decision"]
    else:
        plan, decision = create_plan(client, paths, conn, config, chapter_num, tail, cached_memory=cached_memory)
        save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": decision})

    # Continuity validation runs on both fresh and resumed plans
    violations = validate_plan_continuity(conn, plan, chapter_num, config=config)
    if violations:
        log(paths, f"Continuity violations Ch{chapter_num}: {violations}")
        critical = [v for v in violations if v.startswith("CRITICAL")]
        if critical:
            log(paths, f"Critical violations found, re-planning Ch{chapter_num}")
            decision.setdefault("required_constraints", []).extend(violations)
            plan, decision = create_plan(
                client,
                paths,
                conn,
                config,
                chapter_num,
                tail,
                checkpoint_label="critical",
                cached_memory=cached_memory,
            )
            save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": decision})
        else:
            decision.setdefault("required_constraints", []).extend(violations)

    existing_chapter = read_text(chapter_path(paths, chapter_num))
    chapter = load_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT) or existing_chapter
    if chapter:
        chapter = normalize_chapter(str(chapter))
        log(paths, f"Resuming cached chapter text Ch{chapter_num}")
    else:
        log(paths, f"Writing Ch{chapter_num}: {plan.get('title', '')}")
        chapter, candidate_review = write_chapter_with_candidates(
            client, paths, conn, config, chapter_num, plan, decision, tail, cached_memory=writing_memory,
            chapter_aux_cache=_chapter_aux,
        )

        # Handle catastrophic pre-screen failure (all candidates have high fossils)
        if chapter is None and isinstance(candidate_review, dict) and candidate_review.get("catastrophe") == "fossil_prescreen":
            log(
                paths,
                f"CATASTROPHIC PRE-SCREEN FAILURE Ch{chapter_num}: all candidates blocked by fossils. "
                f"Triggering PLAN-LEVEL replan to break the repetition cycle."
            )
            # Trigger a real structural replan via create_plan (the previous code
            # called planning.adaptive_replan + _build_tail — neither exists, so it
            # always threw and force-accepted the fossilized draft). create_plan
            # generates a fresh plan; the fossil constraints are injected as
            # replan_feedback so the new plan is steered off the repeated track.
            try:
                fossil_feedback = (
                    "【结构性重规划·化石灾难】前几章已大量复读签名句（化石句累积），"
                    "所有候选草稿都被化石门拦截。本章必须从【全新】角度切入：\n"
                    "- 严禁复刻已用过的场景设计、人物动作、对话模式；\n"
                    "- 改变叙述视角、物理场所、或对话方式；宁可另起炉灶，也不许在旧轨道上微调。"
                )
                replan_tail = _build_writer_tail(paths, config, chapter_num)
                replan_plan, replan_decision = create_plan(
                    client, paths, conn, config, chapter_num, replan_tail,
                    checkpoint_label="fossil_catastrophe",
                    cached_memory=cached_memory,
                    replan_feedback=fossil_feedback,
                )
                chapter, candidate_review = write_chapter_with_candidates(
                    client, paths, conn, config, chapter_num, replan_plan, replan_decision, replan_tail,
                    cached_memory=writing_memory, chapter_aux_cache=_chapter_aux,
                )
                # If still None after replan, we're truly stuck — let it fall through to normal error handling
                if chapter is not None:
                    plan = replan_plan
                    decision = replan_decision
                    save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": decision})
                    log(paths, f"Fossil-catastrophe replan Ch{chapter_num} completed")
            except Exception as exc:
                log(paths, f"Fossil-catastrophe replan Ch{chapter_num} failed: {exc}")
                # Fall through to raise below

        if chapter is None:
            raise RuntimeError(f"Failed to generate any valid chapter text for Ch{chapter_num}")
        # O1: adjacent-duplicate draft gate. A draft that re-narrates the previous
        # chapter near-verbatim (observed: clause overlap 0.33-0.81 on the worst
        # force-accepted chapters vs 0.00-0.07 healthy) is a write-off — review
        # patches cannot fix "the whole scene already happened". Regenerate ONCE
        # with an explicit do-not-repeat constraint before entering the review
        # loop; repetition is self-reinforcing, so catching it pre-review is the
        # cheapest point. The review-side cap/reject still backstops this.
        if bool(config["novel"].get("adjacent_repeat_enabled", True)) and chapter_num > 1:
            try:
                from quality import adjacent_repetition
                prev_text = read_text(chapter_path(paths, chapter_num - 1))
                ar = adjacent_repetition(chapter, prev_text, config)
                if ar.get("level") == "block":
                    log(
                        paths,
                        f"Adjacent-duplicate draft Ch{chapter_num} metrics={ar.get('metrics')}; "
                        f"regenerating once with anti-repeat constraint.",
                    )
                    retry_decision = dict(decision)
                    retry_decision["required_constraints"] = list(decision.get("required_constraints") or [])
                    for d in ar.get("directives", []):
                        if d not in retry_decision["required_constraints"]:
                            retry_decision["required_constraints"].append(d)
                    retry_chapter, retry_review = write_chapter_with_candidates(
                        client, paths, conn, config, chapter_num, plan, retry_decision, tail,
                        cached_memory=writing_memory,
                        chapter_aux_cache=_chapter_aux,
                    )
                    ar2 = adjacent_repetition(retry_chapter, prev_text, config)
                    if ar2.get("level") != "block":
                        chapter, candidate_review = retry_chapter, retry_review
                        decision = retry_decision
                        log(paths, f"Adjacent-duplicate retry Ch{chapter_num} clean (metrics={ar2.get('metrics')})")
                    elif float(ar2.get("metrics", {}).get("clause_overlap", 1.0)) < float(
                        ar.get("metrics", {}).get("clause_overlap", 1.0)
                    ):
                        chapter, candidate_review = retry_chapter, retry_review
                        decision = retry_decision
                        log(
                            paths,
                            f"Adjacent-duplicate retry Ch{chapter_num} still high but improved "
                            f"({ar.get('metrics', {}).get('clause_overlap')} -> "
                            f"{ar2.get('metrics', {}).get('clause_overlap')}); review gate will judge.",
                        )
                    else:
                        log(paths, f"Adjacent-duplicate retry Ch{chapter_num} did not improve; keeping original draft")
            except Exception as exc:
                log(paths, f"Adjacent-duplicate draft gate failed (non-fatal) Ch{chapter_num}: {exc}")
        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
        if candidate_review is not None:
            save_checkpoint(paths, chapter_num, "review_round0.json", candidate_review)

    threshold = float(config["novel"]["quality_threshold"])
    max_rounds = int(config["novel"]["max_revision_rounds"])
    # Preference pairs (before-text, review verdict, after-text) collected from
    # this chapter's revise rounds; forwarded to the global telemetry repo by
    # _do_telemetry after finalize. Pure in-memory, zero IO on the hot path.
    telemetry_revise_pairs: list[dict[str, Any]] = []
    final_review = load_checkpoint(paths, chapter_num, "final_review.json")
    # A final_review is authoritative on resume in two cases:
    #   1. it met the threshold and was accepted; or
    #   2. it was force-accepted below threshold (the loop already exhausted all
    #      revise + local-fix + replan rounds and picked the best draft). Replaying
    #      the loop from scratch on resume would waste a full chapter of LLM calls
    #      AND can regress the accepted text to a worse-scoring earlier round
    #      (observed: a force-accepted 7.8 replayed back down to 7.5). Honour the
    #      stored best instead.
    final_is_authoritative = (
        isinstance(final_review, dict)
        and (
            (safe_score(final_review.get("score", 0)) >= threshold and final_review.get("accepted", True))
            or bool(final_review.get("force_accepted"))
        )
    )
    if final_is_authoritative:
        review = final_review
        log(
            paths,
            f"Resuming final review Ch{chapter_num} score={review.get('score')}/10"
            f"{' (force-accepted)' if final_review.get('force_accepted') else ''}",
        )
    else:
        if isinstance(final_review, dict):
            log(
                paths,
                f"Ignoring low final review Ch{chapter_num} score={final_review.get('score')}/10 threshold={threshold}",
            )
        review = {"score": 0, "accepted": False}
        best_chapter = chapter
        best_review = review
        no_improvement_rounds = 0
        max_no_improvement_rounds = int(config["novel"].get("max_no_improvement_revision_rounds", 1))
        # When the round-0 failure is a STRUCTURAL macro-dimension shortfall
        # (novelty/payoff/hook too low), revise patches have ~0 net effect —
        # empirically 0/7 such revises moved the score across threshold, and one
        # even regressed it. Skip the revise rounds entirely and let the
        # structural-replan path below rebuild the scene. revise rounds stay
        # reserved for prose/continuity-style local flaws that patches CAN fix.
        skip_revise_macro = bool(config["novel"].get("skip_revise_on_macro_fail", True))
        for round_num in range(max_rounds + 1):
            if round_num > 0:
                # Decide whether revising is worth it: only for non-structural
                # (local) failures. A structural macro-dimension shortfall goes
                # straight to replan.
                if skip_revise_macro:
                    fk, fr = _classify_replan_failure(review, config)
                    if fk == "structural":
                        log(
                            paths,
                            f"Skipping revise rounds Ch{chapter_num}: structural failure ({fr}); "
                            f"revise patches are ~0-gain here — deferring to structural replan.",
                        )
                        break
                revised_key = f"chapter_revised_round{round_num}.md"
                pre_revise_text = chapter
                pre_revise_review = review
                revised = load_checkpoint(paths, chapter_num, revised_key)
                if revised:
                    chapter = normalize_chapter(str(revised))
                    log(paths, f"Resuming revised chapter Ch{chapter_num} round={round_num}")
                else:
                    log(
                        paths,
                        f"Revising Ch{chapter_num} round={round_num} because score={review.get('score')}/10 < {threshold}",
                    )
                    chapter = revise_chapter(client, paths, conn, config, chapter, review, plan, tail, cached_memory=writing_memory)
                    save_checkpoint(paths, chapter_num, revised_key, chapter)
                save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)

            review_key = f"review_round{round_num}.json"
            cached_review = load_checkpoint(paths, chapter_num, review_key)
            if isinstance(cached_review, dict):
                review = cached_review
                log(paths, f"Resuming cached review Ch{chapter_num} round={round_num} score={review.get('score')}/10")
            else:
                review = review_chapter(client, paths, conn, config, chapter_num, plan, chapter, tail, cached_memory=writing_memory, chapter_aux_cache=_chapter_aux)
                save_checkpoint(paths, chapter_num, review_key, review)
                log(paths, f"Reviewed Ch{chapter_num} round={round_num} score={review.get('score')}/10")
            if round_num > 0:
                # Collect the (before, verdict, after) preference pair now that
                # both sides have scores; persisted by _do_telemetry later.
                try:
                    telemetry_revise_pairs.append({
                        "round": round_num,
                        "text_before": pre_revise_text,
                        "review": pre_revise_review,
                        "text_after": chapter,
                        "score_before": safe_score(pre_revise_review.get("score", 0)),
                        "score_after": safe_score(review.get("score", 0)),
                    })
                except Exception:
                    pass
            previous_best_score = safe_score(best_review.get("score", 0))
            current_score = safe_score(review.get("score", 0))
            if current_score > previous_best_score:
                best_chapter = chapter
                best_review = dict(review)
                no_improvement_rounds = 0
            elif round_num > 0:
                no_improvement_rounds += 1
            if current_score >= threshold and review.get("accepted", True):
                review["accepted"] = True
                break
            if round_num > 0 and no_improvement_rounds >= max_no_improvement_rounds:
                log(
                    paths,
                    f"Stopping Ch{chapter_num} revisions after {no_improvement_rounds} non-improving round(s) "
                    f"(current score={review.get('score')}/10, best score={best_review.get('score')}/10).",
                )
                break

        if (
            bool(config["novel"].get("replan_on_low_quality", True))
            and (safe_score(review.get("score", 0)) < threshold or not review.get("accepted", True))
            and not load_checkpoint(paths, chapter_num, "quality_replan_done.json")
        ):
            try:
                # Route: local failures (single weak dimension / locatable contract
                # violation / prose collapse with patches) get extra TARGETED revise
                # rounds — keep the working scene, surgically fix the flaw — instead
                # of a high-variance full-scene replan. Only structural failures
                # (multi-dimension low / scene雷同 / payoff缺失) trigger a full replan.
                fail_kind, fail_reason = _classify_replan_failure(review, config)
                log(paths, f"Quality replan routing Ch{chapter_num}: kind={fail_kind} ({fail_reason})")
                # Telemetry: deterministic gate rejects are the signal the
                # self-evolution loop learns from (which gates fire, how often,
                # and whether the forced replan recovered the score).
                if review.get("gate_rejects"):
                    try:
                        db_event(conn, chapter_num, "gate_reject", {
                            "gates": [g.get("gate") for g in review.get("gate_rejects", []) if isinstance(g, dict)],
                            "score": review.get("score"),
                        })
                    except Exception:
                        pass

                if fail_kind == "local" and bool(config["novel"].get("local_fix_before_replan", True)):
                    local_rounds = int(config["novel"].get("local_fix_max_rounds", 2))
                    local_chapter = chapter
                    local_review = review
                    improved = False
                    for lr in range(1, local_rounds + 1):
                        lkey = f"local_fix_round{lr}.md"
                        cached_local = load_checkpoint(paths, chapter_num, lkey)
                        if cached_local:
                            local_chapter = normalize_chapter(str(cached_local))
                        else:
                            local_chapter = revise_chapter(
                                client, paths, conn, config, local_chapter, local_review,
                                plan, tail, cached_memory=writing_memory,
                            )
                            save_checkpoint(paths, chapter_num, lkey, local_chapter)
                        local_review = review_chapter(
                            client, paths, conn, config, chapter_num, plan,
                            local_chapter, tail, cached_memory=writing_memory,
                            chapter_aux_cache=_chapter_aux,
                        )
                        log(
                            paths,
                            f"Local fix Ch{chapter_num} round={lr} score={local_review.get('score')}/10",
                        )
                        if safe_score(local_review.get("score", 0)) > safe_score(best_review.get("score", 0)):
                            chapter = normalize_chapter(local_chapter)
                            review = local_review
                            best_chapter = chapter
                            best_review = dict(review)
                            save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
                            save_checkpoint(paths, chapter_num, "final_review.json", review)
                            improved = True
                        if (
                            safe_score(local_review.get("score", 0)) >= threshold
                            and local_review.get("accepted", True)
                        ):
                            review["accepted"] = True
                            break
                    save_checkpoint(
                        paths, chapter_num, "quality_replan_done.json",
                        {"route": "local", "reason": fail_reason, "improved": improved,
                         "score_after": review.get("score")},
                    )
                    # If local fixes still left it below threshold, fall through to
                    # the force-accept/debt block below (no full replan for local).
                    raise _LocalFixDone()

                # ROI breaker: skip the expensive structural replan when (a) there
                # is no hard deterministic gate-reject (those MUST be redone), (b)
                # the shortfall is only marginal, and (c) the book's recent replans
                # have not been improving. Accept the best draft with quality debt.
                roi_margin = float(config["novel"].get("replan_roi_skip_margin", 0.7))
                if (
                    bool(config["novel"].get("replan_roi_breaker_enabled", True))
                    and not review.get("gate_rejects")
                    and safe_score(review.get("score", 0)) >= threshold - roi_margin
                    and not cost_savings_disabled(config, chapter_num)
                    and _recent_replan_ineffective(paths, chapter_num, config)
                ):
                    log(
                        paths,
                        f"Replan ROI stop Ch{chapter_num}: recent replans ineffective and "
                        f"shortfall marginal (score={review.get('score')}/{threshold}); "
                        f"accepting best draft with quality debt to save cost.",
                    )
                    db_event(conn, chapter_num, "replan_roi_stop", {
                        "score": review.get("score"),
                        "threshold": threshold,
                    })
                    save_checkpoint(paths, chapter_num, "quality_replan_done.json",
                                    {"route": "roi_stop", "score_after": review.get("score")})
                    raise _LocalFixDone()

                log(
                    paths,
                    f"Quality replan Ch{chapter_num}: best score={review.get('score')}/10 below threshold={threshold}",
                )
                replan_tail = _build_writer_tail(paths, config, chapter_num)
                # Reuse the chapter-level cached_memory built at the top of
                # generate_one_chapter. Within this chapter no extract/finalize has
                # run yet (it is submitted only after save_chapter, far below), so
                # the only DB write before here is the gate_reject event — which is
                # not read by any memory tier. The cached context is therefore still
                # current, and rebuilding it would re-fire 4 SQLite reads + reread
                # bible/characters/state/threads/timeline for no change.
                replan_memory = cached_memory
                replan_fb = _build_replan_feedback(review)
                # V3-#3: diagnose BEFORE regenerating the plan. A structural replan
                # regenerates the whole plan, and historically often came back "did
                # not improve" because the fresh plan repeated the same open-loop
                # mistake (abstract intent / off-page payoff) in new wording. Pin
                # down which beats never landed + the weakest dimension so the new
                # plan's constraints carry a concrete target, not a vague "do better".
                diagnose_constraints: list[str] = []
                if bool(config["novel"].get("structural_diagnose_enabled", True)):
                    try:
                        from planning import diagnose_structural_failure as _diagnose
                        diag = _diagnose(
                            client, paths, config, chapter_num, plan, review, chapter,
                        )
                        if isinstance(diag, dict) and diag:
                            save_checkpoint(paths, chapter_num, "structural_diagnose.json", diag)
                            rc = str(diag.get("root_cause", "")).strip()
                            sf = str(diag.get("scene_fix", "")).strip()
                            fb = [str(b).strip() for b in (diag.get("failed_beats") or []) if str(b).strip()]
                            md = [str(b).strip() for b in (diag.get("must_dramatize") or []) if str(b).strip()]
                            wd = str(diag.get("weakest_dimension", "")).strip()
                            if rc:
                                diagnose_constraints.append(f"上一版结构性失败根因：{rc}")
                            if fb:
                                diagnose_constraints.append(
                                    "以下原计划 beat 在上一版根本没落地，重写必须真正演出来："
                                    + "；".join(fb[:4])
                                )
                            if wd:
                                diagnose_constraints.append(f"最弱维度是 {wd}，本次必须把它作为首要修复目标。")
                            if sf:
                                diagnose_constraints.append(f"场景设计必须改变：{sf}")
                            if md:
                                diagnose_constraints.append(
                                    "下列画面/动作必须在正文真正发生（不得用梗概/暗示带过）："
                                    + "；".join(md[:4])
                                )
                            # Surface the diagnosis to the plan generator too.
                            if diagnose_constraints:
                                replan_fb = (replan_fb + "\n" if replan_fb else "") + \
                                    "## 重写前诊断（必须据此重做场景）\n- " + "\n- ".join(diagnose_constraints)
                            log(
                                paths,
                                f"Structural diagnose Ch{chapter_num}: root_cause={rc[:60]!r} "
                                f"weakest={wd} failed_beats={len(fb)}",
                            )
                    except Exception as exc:
                        log(paths, f"Structural diagnose failed (non-fatal) Ch{chapter_num}: {exc}")
                replan_plan, replan_decision = create_plan(
                    client,
                    paths,
                    conn,
                    config,
                    chapter_num,
                    replan_tail,
                    checkpoint_label="quality_replan",
                    cached_memory=replan_memory,
                    replan_feedback=replan_fb,
                )
                replan_decision.setdefault("required_constraints", []).append(
                    "上一版章节低于质量阈值。本次必须重做场景设计，而不是只修补措辞；优先提升追读、兑现与新鲜度。"
                )
                for dc in diagnose_constraints:
                    if dc not in replan_decision["required_constraints"]:
                        replan_decision["required_constraints"].append(dc)
                # Variance hedge: a single replan draft once turned an 8.5 plan
                # into a 5.5 chapter. Sample several lower-temperature drafts and
                # keep the best-reviewed one instead of betting on one roll.
                sr_candidates = int(config["novel"].get("structural_replan_candidates", 3))
                sr_temp = float(config["novel"].get("structural_replan_temperature", 0.65))
                replan_chapter, replan_review = write_chapter_with_candidates(
                    client,
                    paths,
                    conn,
                    config,
                    chapter_num,
                    replan_plan,
                    replan_decision,
                    replan_tail,
                    cached_memory=writing_memory,
                    num_candidates_override=max(1, sr_candidates),
                    base_temp_override=sr_temp,
                    chapter_aux_cache=_chapter_aux,
                )
                if replan_review is None:
                    # n<=1 path returned no review (or only 1 valid draft); review now.
                    replan_review = review_chapter(
                        client,
                        paths,
                        conn,
                        config,
                        chapter_num,
                        replan_plan,
                        replan_chapter,
                        replan_tail,
                        cached_memory=writing_memory,
                        chapter_aux_cache=_chapter_aux,
                    )
                save_checkpoint(
                    paths,
                    chapter_num,
                    "quality_replan_done.json",
                    {
                        "score_before": review.get("score"),
                        "score_after": replan_review.get("score"),
                        "accepted_after": replan_review.get("accepted"),
                    },
                )
                # INVARIANT: best_chapter/best_review must never regress. Promote the
                # replan result ONLY when it beats the BEST seen so far (not merely the
                # last round). Structural replan can iterate, so a weaker later replan
                # must not overwrite a stronger earlier draft — otherwise force-accept
                # (which ships best_*) ships the worse one. Observed regression: an 8.38
                # draft was overwritten by a 6.63 replan and shipped as 6.63.
                replan_score = safe_score(replan_review.get("score", 0))
                if replan_score > safe_score(best_review.get("score", 0)):
                    chapter = normalize_chapter(replan_chapter)
                    review = replan_review
                    plan = replan_plan
                    decision = replan_decision
                    best_chapter = chapter
                    best_review = dict(review)
                    save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
                    save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": decision})
                    save_checkpoint(paths, chapter_num, "final_review.json", review)
                    log(paths, f"Quality replan Ch{chapter_num} improved best score to {review.get('score')}/10")
                else:
                    log(
                        paths,
                        f"Quality replan Ch{chapter_num} did not beat best "
                        f"(new score={replan_score}/10, best={safe_score(best_review.get('score', 0))}/10); keeping best.",
                    )
            except _LocalFixDone:
                pass
            except Exception as exc:
                save_checkpoint(paths, chapter_num, "quality_replan_done.json", {"error": str(exc)})
                log(paths, f"Quality replan failed (non-fatal) Ch{chapter_num}: {exc}")

        if safe_score(review.get("score", 0)) < threshold or not review.get("accepted", True):
            chapter = best_chapter
            review = best_review
            # HARD FLOOR: if gate_rejects exist AND score is catastrophically low (≤3.0),
            # this is a structural collapse (e.g. adjacent-repeat block with 1.5/10 score).
            # Force-accept would ship garbage; trigger a STRUCTURAL replan instead.
            gate_rejects = review.get("gate_rejects", [])
            hard_floor = 3.0
            if gate_rejects and safe_score(review.get("score", 0)) <= hard_floor:
                log(
                    paths,
                    f"Ch{chapter_num} HARD FLOOR violation: score={review.get('score')}/10 "
                    f"with gate_rejects={[g['gate'] for g in gate_rejects]}. "
                    f"Triggering STRUCTURAL replan instead of force-accept.",
                )
                # Real structural replan via create_plan (previous code called the
                # non-existent planning.adaptive_replan + _build_tail, so it always
                # threw and force-accepted the collapsed chapter). Inject the gate
                # directives as replan_feedback so the fresh plan avoids the blocks.
                try:
                    gate_dirs: list[str] = []
                    for gr in gate_rejects:
                        gate_dirs.extend(gr.get("directives", []))
                    replan_feedback = _build_replan_feedback(review)
                    if gate_dirs:
                        replan_feedback += "\n【确定性质量门作废本稿，必须结构性重做，不可修补】\n- " + "\n- ".join(
                            str(d) for d in gate_dirs[:8]
                        )
                    replan_tail = _build_writer_tail(paths, config, chapter_num)
                    replan_plan, replan_decision = create_plan(
                        client, paths, conn, config, chapter_num, replan_tail,
                        checkpoint_label="hard_floor",
                        cached_memory=cached_memory,
                        replan_feedback=replan_feedback,
                    )
                    replan_chapter, replan_review = write_chapter_with_candidates(
                        client, paths, conn, config, chapter_num, replan_plan, replan_decision, replan_tail,
                        cached_memory=writing_memory, chapter_aux_cache=_chapter_aux,
                    )
                    if replan_review is None:
                        replan_review = review_chapter(
                            client, paths, conn, config, chapter_num, replan_plan, replan_chapter, replan_tail,
                            cached_memory=writing_memory, chapter_aux_cache=_chapter_aux
                        )
                    # Accept replan result unconditionally (even if still low) to avoid infinite loop.
                    # INTENTIONAL EXCEPTION to the best-never-regress invariant above: this fires
                    # only when the current best is a catastrophic collapse (score<=3.0 + gate_rejects),
                    # so taking the fresh replan as the new best is the emergency exit, not a regression.
                    chapter = normalize_chapter(replan_chapter)
                    review = replan_review
                    plan = replan_plan
                    best_chapter = chapter
                    best_review = dict(review)
                    save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
                    save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": replan_decision})
                    save_checkpoint(paths, chapter_num, "final_review.json", review)
                    log(paths, f"Hard-floor replan Ch{chapter_num} completed: new score={review.get('score')}/10")
                except Exception as exc:
                    log(paths, f"Hard-floor replan Ch{chapter_num} failed: {exc}. Falling back to force-accept.")
                    # Fall through to force-accept below if replan fails

            # Standard force-accept path (no hard-floor violation or replan failed)
            if safe_score(review.get("score", 0)) < threshold or not review.get("accepted", True):
                # CIRCUIT BREAKER: check for consecutive force-accepts
                # If the previous N chapters were all force-accepted BELOW a low
                # floor (not merely below the 8.0 target), we're in a quality death
                # spiral where each low-quality chapter pollutes context for the next.
                # Use a dedicated floor (default 7.0) — NOT the main threshold — so a
                # healthy 7.8 chapter isn't mistaken for a force-accept. HALT instead.
                consecutive_force_accept_limit = int(config["novel"].get("consecutive_force_accept_limit", 2))
                breaker_floor = float(config["novel"].get("circuit_breaker_score_floor", 7.0))
                if consecutive_force_accept_limit > 0 and chapter_num > consecutive_force_accept_limit:
                    try:
                        # Include the CURRENT chapter (about to be force-accepted) in the streak.
                        recent_force_accepts = []
                        cur_score = safe_score(review.get("score", 0))
                        if cur_score < breaker_floor:
                            recent_force_accepts.append((chapter_num, cur_score))
                        for ch_back in range(chapter_num - 1, chapter_num - consecutive_force_accept_limit, -1):
                            if ch_back < 1:
                                break
                            try:
                                past_review = conn.execute(
                                    "SELECT score FROM chapter_metrics WHERE chapter = ?", (ch_back,)
                                ).fetchone()
                                if past_review and float(past_review[0]) < breaker_floor:
                                    recent_force_accepts.append((ch_back, float(past_review[0])))
                                else:
                                    break  # streak broken by a healthy chapter
                            except Exception:
                                break

                        if len(recent_force_accepts) >= consecutive_force_accept_limit:
                            streak = sorted(recent_force_accepts)
                            log(
                                paths,
                                f"CIRCUIT BREAKER Ch{chapter_num}: {len(streak)} consecutive chapters below floor {breaker_floor} "
                                f"({streak}). Refusing to force-accept another — quality death spiral detected.",
                            )
                            raise RuntimeError(
                                f"Circuit breaker: {len(streak)} consecutive force-accepts below {breaker_floor} "
                                f"(Ch{streak[0][0]}-{streak[-1][0]}). "
                                f"Manual intervention required — consider adjusting prompt/config/model or stopping."
                            )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        log(paths, f"Circuit breaker check failed (non-fatal): {exc}")

                chapter, review = _apply_force_accept_patches(paths, config, chapter_num, chapter, review)
                log(
                    paths,
                    f"Ch{chapter_num} did not meet threshold {threshold} after {max_rounds + 1} rounds "
                    f"(best score={review.get('score')}/10). Accepting anyway to avoid pipeline halt.",
                )
                review["accepted"] = True
            # v12 Ch4/Ch6 数据缺陷：极少数路径下走到这里的 review 是 score=0 的空壳
            # （无 style_health/problems），force_accepted 后污染 stats/退化诊断。
            # 兜底：score<=0 时从本章最后一轮真实评审 checkpoint 恢复。
            if safe_score(review.get("score", 0)) <= 0:
                for _ck in ("review_round1.json", "review_round0.json"):
                    _prev = load_checkpoint(paths, chapter_num, _ck)
                    if isinstance(_prev, dict) and safe_score(_prev.get("score", 0)) > 0:
                        log(
                            paths,
                            f"Force-accept Ch{chapter_num}: review carried score=0, "
                            f"recovered from {_ck} (score={_prev.get('score')})",
                        )
                        review = {**_prev, "accepted": True}
                        break
            # Mark this as a force-accept and persist the chosen best as the
            # authoritative final_review + current chapter text. On resume, the
            # final_is_authoritative check honours force_accepted reviews, so the
            # loop does NOT replay review/revise/local-fix from scratch (which both
            # wastes LLM calls and can regress to a worse earlier round).
            review["force_accepted"] = True
            save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
            save_checkpoint(paths, chapter_num, "final_review.json", review)
            # Register a "quality debt": this chapter was force-accepted below the
            # threshold, so the post-completion refine pass should prioritise it
            # (higher intensity) instead of treating it like a healthy chapter. We
            # persist a small marker the refine pass reads back. Style-collapse and
            # contract issues bias toward heavier intensity.
            try:
                sh_metrics = (review.get("style_health") or {}).get("metrics") or {}
                debt = {
                    "chapter": chapter_num,
                    "score": review.get("score"),
                    "style_penalty": (review.get("style_health") or {}).get("penalty", 0.0),
                    "em_dash_per_kchar": sh_metrics.get("em_dash_per_kchar"),
                    "fragment_line_ratio": sh_metrics.get("fragment_line_ratio"),
                    "had_contract_violation": bool(review.get("contract_violations")),
                    "gate_rejects": [
                        str(g.get("gate", "?")) for g in (review.get("gate_rejects") or [])
                        if isinstance(g, dict)
                    ],
                    "patches_applied": review.get("quality_debt_patches_applied", 0),
                    "patch_total": review.get("quality_debt_patch_total", 0),
                    "problems": [str(p)[:160] for p in (review.get("problems") or [])[:5]],
                }
                save_checkpoint(paths, chapter_num, "quality_debt.json", debt)
                db_event(conn, chapter_num, "quality_debt", debt)
                log(paths, f"Quality-debt registered Ch{chapter_num} (score={review.get('score')}/10) for refine priority")
            except Exception as exc:
                log(paths, f"Quality-debt registration failed (non-fatal) Ch{chapter_num}: {exc}")

        # Hook-only mini revise: if the chapter ends weakly, rewrite only the
        # last ~400 chars rather than running another full revise round. This
        # is a single small LLM call gated by hook_strength threshold.
        hook_min = float(config["novel"].get("hook_strength_min", 6.0))
        opening_chapters = int(config["novel"].get("opening_chapters", 3))
        if chapter_num <= opening_chapters:
            hook_min = float(config["novel"].get("opening_hook_strength_min", 7.0))
        hook_revise_enabled = bool(config["novel"].get("hook_revise_enabled", True))
        hook_strength = safe_score(review.get("hook_strength", hook_min))
        # O1: recycled-hook detection. A recurring force-accept debt across books
        # is "章末钩子与上章完全相同" — and the reviewer still rates the recycled
        # hook 9/10 because it never sees the previous endings side by side. If
        # the deterministic check finds this chapter's ending clauses repeated
        # from recent chapters' endings, force the hook-only mini revise even
        # when hook_strength looks healthy.
        hook_recycled = False
        recycled_clauses: list[str] = []
        if (
            hook_revise_enabled
            and chapter_num > 1
            and not is_final_chapter(config, chapter_num)
            and bool(config["novel"].get("adjacent_repeat_enabled", True))
        ):
            try:
                from quality import hook_tail_repetition
                lookback = int(config["novel"].get("hook_repeat_lookback", 3))
                prev_tails = []
                for num in range(max(1, chapter_num - lookback), chapter_num):
                    t = read_text(chapter_path(paths, num))
                    if t:
                        prev_tails.append(t)
                hr = hook_tail_repetition(chapter, prev_tails, config)
                if hr.get("repeat"):
                    hook_recycled = True
                    recycled_clauses = list(hr.get("repeated_clauses") or [])
                    log(
                        paths,
                        f"Hook-recycled Ch{chapter_num}: ending reuses prior chapter endings "
                        f"(ratio={hr.get('ratio')}, clauses={recycled_clauses[:2]}); forcing hook revise.",
                    )
            except Exception as exc:
                log(paths, f"hook_tail_repetition check failed (non-fatal) Ch{chapter_num}: {exc}")
        if (
            hook_revise_enabled
            and (hook_recycled or (hook_strength > 0 and hook_strength < hook_min))
            and not is_final_chapter(config, chapter_num)
            and not load_checkpoint(paths, chapter_num, "hook_revised.json")
        ):
            try:
                from writing import revise_hook_only as _revise_hook_only
                log(
                    paths,
                    f"Hook-only mini-revise Ch{chapter_num} hook_strength={hook_strength}/10 < {hook_min}"
                    + (" [recycled hook]" if hook_recycled else ""),
                )
                hook_review = review
                if hook_recycled and recycled_clauses:
                    hook_review = dict(review)
                    wd = list(hook_review.get("writer_directives_for_next_chapter") or [])
                    wd.insert(0,
                        "章末钩子与前几章结尾重复（确定性检测）。以下句子/意象严禁出现在新结尾里："
                        + "；".join(f"“{c}”" for c in recycled_clauses[:3])
                        + "。必须换一个全新的悬念抓手（新的物证/新的威胁/新的人物动作），不得复用旧钩子。")
                    hook_review["writer_directives_for_next_chapter"] = wd
                new_chapter = _revise_hook_only(
                    client, paths, config, chapter, plan, hook_review,
                    tail_to_revise_chars=int(config["novel"].get("hook_revise_tail_chars", 400)),
                )
                if len(new_chapter.strip()) >= max(500, int(len(chapter) * 0.85)):
                    chapter = new_chapter
                    save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
                    save_checkpoint(paths, chapter_num, "hook_revised.json", {"done": True, "hook_strength_before": hook_strength})
                    # Re-review the rewritten chapter instead of fabricating the
                    # hook metric. The old code did `review["hook_strength"] =
                    # max(hook_strength, hook_min)`, persisting an unmeasured floor
                    # value into chapter_metrics — polluting the bandit reward and
                    # stats, and letting the new tail bypass the style/fossil gates.
                    # Gated by hook_revise_rereview (default true); on disable or
                    # failure we keep the ORIGINAL measured value (never bump up).
                    if bool(config["novel"].get("hook_revise_rereview", True)):
                        try:
                            review = review_chapter(
                                client, paths, conn, config, chapter_num, plan, chapter, tail,
                                cached_memory=writing_memory, chapter_aux_cache=_chapter_aux,
                            )
                            log(
                                paths,
                                f"Hook revise re-review Ch{chapter_num}: "
                                f"hook_strength={safe_score(review.get('hook_strength', 0))}/10 "
                                f"score={safe_score(review.get('score', 0))}",
                            )
                        except Exception as exc:
                            log(paths, f"Hook revise re-review failed (non-fatal) Ch{chapter_num}: {exc}")
                else:
                    log(paths, f"Hook revise produced too-short output ({len(new_chapter)} chars); keeping original")
            except Exception as exc:
                log(paths, f"Hook revise failed (non-fatal) Ch{chapter_num}: {exc}")

        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
        save_checkpoint(paths, chapter_num, "final_review.json", review)

    if not load_checkpoint(paths, chapter_num, "chapter_saved.json"):
        # Optional hook-y, non-spoilery chapter title: rewrite ONLY the first
        # 第N章 title line (body prose untouched). Gated + fully reversible; on
        # any failure the chapter keeps its plan-derived title. Skipped on the
        # already-exists branch since that text was titled on its first pass.
        if (
            bool(config["novel"].get("chapter_title_refine_enabled", False))
            and not chapter_path(paths, chapter_num).exists()
        ):
            try:
                from package import refine_chapter_title, apply_chapter_title
                new_title = refine_chapter_title(client, paths, config, chapter_num, plan, chapter)
                if new_title and new_title != str(plan.get("title") or "").strip():
                    chapter = apply_chapter_title(chapter, chapter_num, new_title)
                    save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
            except Exception as exc:
                log(paths, f"Chapter title refine failed (non-fatal) Ch{chapter_num}: {exc}")
        # Last-mile em-dash remediation: the revise_chapter path has its own
        # em-dash layers, but structural-failure replans skip revise entirely.
        # Apply programmatic reduction here so every chapter is cleaned before save.
        if bool(config["novel"].get("em_dash_reduce_enabled", True)):
            try:
                from quality import style_health as _sh_final, reduce_em_dash_density as _reduce_em
                _sh_f = _sh_final(chapter, config)
                _em_f = float(_sh_f.get("metrics", {}).get("em_dash_per_kchar", 0))
                _em_tgt = float(config["novel"].get("em_dash_reduce_target_per_kchar", 3.0))
                if _em_f > _em_tgt:
                    _before = chapter.count("——")
                    chapter = _reduce_em(chapter, config)
                    _after = chapter.count("——")
                    if _before != _after:
                        log(paths, f"Pre-save em-dash reduction Ch{chapter_num}: {_before}->{_after} dashes, {_em_f:.1f}/k")
                        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
            except Exception as exc:
                log(paths, f"Pre-save em-dash reduction failed (non-fatal) Ch{chapter_num}: {exc}")

        if chapter_path(paths, chapter_num).exists():
            log(paths, f"Chapter file already exists Ch{chapter_num}; skipping duplicate save")
            # book.md is normally append-built by save_chapter, so it is already
            # complete in the common case; only rebuild (O(n) glob+read+sort) when
            # the on-disk book.md is actually inconsistent with chapters/.
            if not book_is_consistent(paths):
                rebuild_book(paths)
        else:
            save_chapter(paths, chapter_num, chapter, review, plan)
        save_checkpoint(paths, chapter_num, "chapter_saved.json", {"saved": True})

    extraction_done = bool(load_checkpoint(paths, chapter_num, "extraction.json"))
    structured_done = bool(load_checkpoint(paths, chapter_num, "structured_state_done.json"))
    state_file_done = bool(load_checkpoint(paths, chapter_num, "state_file_done.json"))
    completed_done = bool(load_checkpoint(paths, chapter_num, "chapter_completed.json"))

    extract_in_bg = bool(config["novel"].get("extract_in_background", False))
    state_in_bg = bool(config["novel"].get("state_file_in_background", False))

    def _run_finalize() -> dict[str, Any]:
        # Extract + structured-state are barrier-protected: the next chapter's
        # plan must see updated metrics/threads/entities.
        # NOTE: chapter_completed.json is written by the CALLER before submitting
        # this to the background, so the main loop's resume check sees the
        # chapter as finished. Do NOT re-write it here.
        if not extraction_done:
            try:
                extraction_local = extract_events(
                    client, paths, conn, config, chapter_num, chapter, cached_memory=cached_memory
                )
            except Exception as exc:
                log(paths, f"Extraction failed Ch{chapter_num}; using local fallback extraction: {exc}")
                extraction_local = _fallback_extraction(plan, review, chapter_num, str(exc))
            save_checkpoint(paths, chapter_num, "extraction.json", extraction_local)
        else:
            extraction_local = load_checkpoint(paths, chapter_num, "extraction.json") or {}
        if not structured_done:
            # Structured-state writes consume LLM-extracted JSON that is frequently
            # malformed (non-scalar DB binds, dicts-as-strings). A failure here must
            # NOT abort finalize: the caller writes chapter_completed.json only after
            # this returns, so a raised exception leaves the chapter un-completed and
            # wedges resume in an endless "Resuming partially indexed Ch{n}" loop.
            # Degrade gracefully — the chapter content is already saved; structured
            # tracking (threads/entities/metrics) is best-effort.
            try:
                update_structured_state(paths, conn, chapter_num, extraction_local, review, decision)
            except Exception as exc:
                log(paths, f"Structured-state update failed Ch{chapter_num} (non-fatal, continuing): {exc}")
            save_checkpoint(paths, chapter_num, "structured_state_done.json", {"done": True})
        return extraction_local

    def _run_state_file(extraction_local: dict[str, Any]) -> None:
        update_state_file(client, paths, conn, config, chapter_num, chapter, extraction_local)
        save_checkpoint(paths, chapter_num, "state_file_done.json", {"done": True})

    finalize_label = f"chapter_finalize_ch{chapter_num}"
    state_file_label = f"state_file_ch{chapter_num}"

    needs_finalize = not (extraction_done and structured_done and completed_done)
    use_bg_finalize = extract_in_bg and background is not None and needs_finalize and not resume
    if use_bg_finalize:
        # CRITICAL: write chapter_completed.json synchronously BEFORE submitting
        # background work. The main loop uses this marker to decide whether to
        # re-enter `Resuming partially indexed Ch{n}`; if we left it for the bg
        # task to write, the loop would re-enter immediately and resubmit the
        # same finalize task on every iteration, leaking threads + memory.
        # extract/structured are still allowed to run in the background — they
        # only feed the NEXT chapter, gated by the wait_label barrier at the
        # start of generate_one_chapter.
        if not completed_done:
            db_event(conn, chapter_num, "chapter_completed", {"review": review, "plan": plan, "decision": decision})
            save_checkpoint(paths, chapter_num, "chapter_completed.json", {"done": True})

        def _bg_finalize_and_state() -> None:
            extraction_local = _run_finalize()
            if not state_file_done:
                _run_state_file(extraction_local)
        background.submit(finalize_label, _bg_finalize_and_state)
    else:
        # Synchronous finalize. This is the path taken on a RESUME (resume=True):
        # the first-pass background finalize never wrote its markers (e.g. the bg
        # extract LLM call failed against dead keys), so the loop re-entered this
        # chapter. Running finalize synchronously here — bounded by
        # max_finalize_attempts — guarantees extraction.json/structured_state_done
        # become durable so should_resume_existing_chapter stops re-triggering and
        # the loop advances instead of resubmitting forever.
        max_attempts = int(config["novel"].get("max_finalize_attempts", 3) or 3)
        attempts = bump_finalize_attempts(paths, chapter_num) if resume and needs_finalize else 0
        if resume and needs_finalize and attempts > max_attempts:
            log(
                paths,
                f"Finalize for Ch{chapter_num} exhausted {attempts - 1} attempts "
                f"(max={max_attempts}); force-completing with fallback markers to break resume loop",
            )
            if not extraction_done:
                save_checkpoint(
                    paths, chapter_num, "extraction.json",
                    _fallback_extraction(plan, review, chapter_num, "finalize attempts exhausted"),
                )
            if not structured_done:
                save_checkpoint(paths, chapter_num, "structured_state_done.json", {"done": True, "forced": True})
            if not state_file_done:
                save_checkpoint(paths, chapter_num, "state_file_done.json", {"done": True, "forced": True})
            if not load_checkpoint(paths, chapter_num, "chapter_completed.json"):
                db_event(conn, chapter_num, "chapter_completed", {"review": review, "plan": plan, "decision": decision})
                save_checkpoint(paths, chapter_num, "chapter_completed.json", {"done": True, "forced": True})
        else:
            extraction_local = _run_finalize()
            if not state_file_done:
                if state_in_bg and background is not None and not resume:
                    background.submit(state_file_label, _run_state_file, extraction_local)
                else:
                    _run_state_file(extraction_local)
            if not load_checkpoint(paths, chapter_num, "chapter_completed.json"):
                db_event(conn, chapter_num, "chapter_completed", {"review": review, "plan": plan, "decision": decision})
                save_checkpoint(paths, chapter_num, "chapter_completed.json", {"done": True})
    if bool(config["novel"].get("fingerprint_enabled", True)):
        try:
            from quality import store_chapter_fingerprint
            store_chapter_fingerprint(conn, chapter_num, plan)
        except Exception:
            pass
    log(paths, f"Saved and indexed Ch{chapter_num}")

    # Log prompt-cache effectiveness per chapter for monitoring.
    hits, misses = cacheable_prefix_hit_rate()
    total = hits + misses
    if total:
        hit_rate = hits / total * 100.0
        log(paths, f"Prompt prefix cache: hits={hits} misses={misses} hit_rate={hit_rate:.1f}%")

    # Schedule heavy post-chapter tasks asynchronously when a BackgroundTasks
    # manager is available. They do NOT block the next chapter's plan/write
    # phase. Note: the next chapter will see whatever memory state these
    # background tasks have finished writing — which is the existing semantic
    # (stage_review and memory_compress only update derived/aggregated files).
    run_stage_review = chapter_num % int(config["novel"]["stage_review_every"]) == 0
    run_replan = run_stage_review and chapter_num >= 40

    def _do_stage_review() -> None:
        stage_review(client, paths, conn, config, chapter_num)
        log(paths, f"Completed stage review Ch{chapter_num}")

    def _do_horizon_review() -> None:
        horizon_review(client, paths, conn, config, chapter_num, chapter)
        log(paths, f"Completed horizon review Ch{chapter_num}")

    def _do_memory_compress() -> None:
        log(paths, f"Compressing memory files at Ch{chapter_num}")
        compress_all_memory(client, paths, config)

    def _do_replan() -> None:
        if should_replan(conn, config):
            log(paths, f"Triggering adaptive replan at Ch{chapter_num}")
            adaptive_replan(client, paths, conn, config, chapter_num)

    _telemetry_novel = paths.logs_dir.parent.name
    _telemetry_genre = str(config["novel"].get("genre", "_default") or "_default")

    def _do_telemetry() -> None:
        """Double-write this chapter's quality signals to the global telemetry
        repository. Strictly an observer: every failure is swallowed so the
        generation pipeline never stalls because of telemetry."""
        try:
            sh = (review.get("style_health") or {}) if isinstance(review, dict) else {}
            sh_metrics = sh.get("metrics") or {}
            metrics_row = {
                "title": plan.get("title") if isinstance(plan, dict) else None,
                "score": safe_score(review.get("score", 0)),
                "readthrough_score": safe_score(review.get("readthrough_score", 0)),
                "hook_score": safe_score(review.get("hook_score", review.get("hook_strength", 0))),
                "payoff_score": safe_score(review.get("payoff_score", 0)),
                "novelty_score": safe_score(review.get("novelty_score", 0)),
                "prose_score": safe_score(review.get("prose_score", review.get("aesthetic_score", 0))),
                "continuity_score": safe_score(review.get("continuity_score", 0)),
                "hook_strength": safe_score(review.get("hook_strength", 0)),
                "accepted": 1 if review.get("accepted") else 0,
                "em_dash_per_kchar": sh_metrics.get("em_dash_per_kchar"),
                "style_penalty": sh.get("penalty"),
                "avg_sentence_chars": sh_metrics.get("avg_sentence_chars"),
                "dialogue_char_ratio": sh_metrics.get("dialogue_char_ratio"),
                "tech_per_kchar": sh_metrics.get("tech_per_kchar"),
            }
            telemetry.record_chapter_metrics(_telemetry_novel, _telemetry_genre, chapter_num, metrics_row)
            # NOTE: plan_arbitration / strategy_outcomes are double-written at
            # the source (planning.arbitrate_plan), which is the only place the
            # full candidate-plans list exists.
            for pair in telemetry_revise_pairs:
                telemetry.record_revise_pair(
                    _telemetry_novel, _telemetry_genre, chapter_num,
                    pair["round"], pair["text_before"], pair["review"],
                    pair["text_after"], pair["score_before"], pair["score_after"],
                )
            log(paths, f"Telemetry recorded Ch{chapter_num} (revise_pairs={len(telemetry_revise_pairs)})")
        except Exception as exc:
            log(paths, f"Telemetry record failed (non-fatal) Ch{chapter_num}: {exc}")

    telemetry_on = bool(config["novel"].get("telemetry_enabled", True))

    if background is not None:
        if telemetry_on:
            background.submit(f"telemetry_ch{chapter_num}", _do_telemetry)
        if run_stage_review:
            background.submit(f"stage_review_ch{chapter_num}", _do_stage_review)
        cold_every = int(config["novel"].get("cold_reader_every", 10))
        if cold_every > 0 and chapter_num % cold_every == 0:
            background.submit(f"horizon_review_ch{chapter_num}", _do_horizon_review)
        if should_compress_memory(paths, config, chapter_num):
            background.submit(f"memory_compress_ch{chapter_num}", _do_memory_compress)
        if run_replan:
            background.submit(f"adaptive_replan_ch{chapter_num}", _do_replan)

        # Prefetch the next N chapters' plans so the main loop's planning
        # phase resumes from a cached validated_plan.json. Gate each prefetch
        # on the previous one's finalize barrier so each sees fresh state.
        if bool(config["novel"].get("prefetch_next_plan", False)):
            horizon = max(1, int(config["novel"].get("prefetch_plan_horizon", 1)))
            target_chars = int(config["novel"].get("target_words", 0) or 0)
            max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
            next_num = chapter_num + 1
            should_prefetch = True
            if target_chars and book_reached_target(paths.book, target_chars):
                should_prefetch = False
                log(paths, f"Prefetch skipped after Ch{chapter_num}: target_chars reached")
            if max_chapters and next_num > max_chapters:
                should_prefetch = False
                log(paths, f"Prefetch skipped after Ch{chapter_num}: next chapter exceeds max_chapters={max_chapters}")

            def _do_prefetch_horizon() -> None:
                if needs_finalize:
                    background.wait_label(finalize_label)
                # Prefetch sequentially within this background task so the
                # second prefetch sees the first one's checkpoint (and any
                # incremental state derived from it). Each iteration only
                # uses the snapshot of metrics/threads currently durable.
                for offset in range(1, horizon + 1):
                    target_num = chapter_num + offset
                    if max_chapters and target_num > max_chapters:
                        log(paths, f"Prefetch horizon stopped at Ch{target_num}: exceeds max_chapters={max_chapters}")
                        break
                    if load_checkpoint(paths, target_num, "validated_plan.json"):
                        log(paths, f"Prefetch skipped for Ch{target_num}: validated_plan.json already exists")
                        continue
                    try:
                        next_tail = tail_text(paths.book, int(config["novel"]["recent_tail_chars"]))
                        next_memory = memory_context(
                            paths, conn, config,
                            max_chars=int(config["novel"].get("plan_memory_chars", 60000) or 0),
                        )
                        next_plan, next_decision = create_plan(
                            client, paths, conn, config, target_num, next_tail, cached_memory=next_memory
                        )
                        save_checkpoint(
                            paths, target_num, "validated_plan.json",
                            {"plan": next_plan, "decision": next_decision},
                        )
                        log(paths, f"Prefetched plan for Ch{target_num} title={next_plan.get('title', '')!r}")
                    except Exception as exc:
                        log(paths, f"Prefetch plan Ch{target_num} failed (non-fatal): {exc}")
                        # Stop prefetching further if one fails — likely the
                        # source state is incomplete; the main loop will
                        # generate it normally.
                        break

            # Use the next chapter's label so generate_one_chapter's barrier
            # (wait_label prefetch_plan_ch{N+1}) still resolves correctly.
            if should_prefetch:
                background.submit(f"prefetch_plan_ch{next_num}", _do_prefetch_horizon)

        background.prune_done()
    else:
        if telemetry_on:
            _do_telemetry()
        if run_stage_review:
            _do_stage_review()
        cold_every = int(config["novel"].get("cold_reader_every", 10))
        if cold_every > 0 and chapter_num % cold_every == 0:
            _do_horizon_review()
        if should_compress_memory(paths, config, chapter_num):
            _do_memory_compress()
        if run_replan:
            _do_replan()

def main() -> None:
    config = load_config()
    paths = get_paths(config)
    ensure_project(paths)
    conn = init_db(paths)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: run `pip install -r requirements.txt` before generation.") from exc

    api_endpoints, primary_endpoint_count, endpoint_models = configured_api_endpoints_with_models(config)
    if not api_endpoints:
        raise RuntimeError("Missing API key: set api.api_key, api.api_keys, or api.api_key_groups in config.yaml")
    stream_timeout = int(api_endpoints and config["api"].get("stream_timeout", 300))
    client_read_timeout = int(config["api"].get("client_read_timeout", 180))
    connect_timeout = int(config["api"].get("client_connect_timeout", 15))
    import httpx
    httpx_timeout = httpx.Timeout(
        connect=connect_timeout,
        read=client_read_timeout,
        write=connect_timeout,
        pool=connect_timeout,
    )
    default_headers = {}
    user_agent = str(config["api"].get("user_agent", "")).strip()
    if user_agent:
        default_headers["User-Agent"] = user_agent
    clients = [
        OpenAI(base_url=base_url, api_key=api_key, timeout=httpx_timeout, default_headers=default_headers or None)
        for base_url, api_key in api_endpoints
    ]
    client: Any = (
        LLMClientPool(
            clients,
            primary_endpoint_count,
            endpoints=api_endpoints,
            log_fn=lambda msg: log(paths, msg),
            endpoint_models=endpoint_models,
        )
        if len(clients) > 1
        else clients[0]
    )
    log(paths, f"LLM client pool initialized keys={len(clients)} primary={primary_endpoint_count}")

    # Per-role model routing: each role (review, planning, writing, extraction)
    # can have its own model+endpoint. Build a separate client pool for each
    # configured role and attach it to the primary client. call_llm picks them
    # up via getattr for tags in the role's tag set. Unconfigured roles fall
    # through to the primary model (backward compatible).
    for _role in ("review", "planning", "writing", "extraction"):
        _role_endpoints = configured_role_endpoints(config, _role)
        if _role_endpoints:
            _role_clients = [
                OpenAI(base_url=base_url, api_key=api_key, timeout=httpx_timeout, default_headers=default_headers or None)
                for base_url, api_key in _role_endpoints
            ]
            _role_pool: Any = (
                LLMClientPool(_role_clients, endpoints=_role_endpoints, log_fn=lambda msg: log(paths, msg))
                if len(_role_clients) > 1
                else _role_clients[0]
            )
            setattr(client, f"{_role}_pool", _role_pool)
            setattr(client, f"{_role}_api", config["api"])
            log(
                paths,
                f"{_role.title()} pool initialized model={config['api'].get(f'{_role}_model')} "
                f"endpoints={len(_role_clients)}",
            )

    # Pre-flight: if a prior bootstrap run left partial artifacts (state.md
    # exists but is too short, contains the placeholder sentinel, or any
    # sibling memory file is missing/empty), clean up now so this run
    # re-bootstraps cleanly instead of proceeding with broken world-state.
    _bootstrap_memory_files = [
        paths.bible, paths.characters, paths.timeline,
        paths.threads, paths.volume_plan,
    ]
    if paths.state.exists():
        try:
            _st = read_text(paths.state)
            _missing_siblings = [
                p.name for p in _bootstrap_memory_files
                if not p.exists() or p.stat().st_size < 100
            ]
            _is_partial = (
                len(_st) < 500
                or "待连载补全" in _st
                or bool(_missing_siblings)
            )
            if _is_partial:
                paths.state.unlink()
                log(
                    paths,
                    f"Pre-flight: removed partial state.md "
                    f"(len={len(_st)}, missing_or_empty_siblings={_missing_siblings}) "
                    f"so bootstrap reruns cleanly",
                )
        except Exception:
            pass

    if not paths.state.exists() or not read_text(paths.state).strip():
        try:
            bootstrap(client, paths, conn, config)
        except Exception as exc:
            # Bootstrap is the single hard dependency for the whole run: every
            # chapter needs state.md/bible/characters/etc. If the core bootstrap
            # LLM call dies (observed: all API keys 401'd + 429 "quota exhausted"
            # at startup), the old behaviour left a half-written / empty state.md
            # and crashed with an opaque traceback. Detect the quota/auth case and
            # exit with an actionable message instead, and make sure we did NOT
            # persist a partial state that would block a clean re-bootstrap later.
            msg = str(exc)
            is_quota = any(
                k in msg.lower()
                for k in ("quota exhausted", "429", "401", "all api keys", "marked invalid")
            )
            try:
                st = read_text(paths.state).strip()
                # A bootstrap that died mid-write may have left a stub state.md.
                # Remove it so the next launch re-bootstraps from scratch rather
                # than treating the stub as a valid (but empty) project.
                # Improved threshold: 500 chars (real state.md is always larger)
                # and also checks for missing sibling memory files.
                _exc_missing = [
                    p.name for p in _bootstrap_memory_files
                    if not p.exists() or p.stat().st_size < 100
                ]
                if st and (
                    "待连载补全" in st
                    or len(st) < 500
                    or _exc_missing
                ) and paths.state.exists():
                    paths.state.unlink()
                    log(
                        paths,
                        f"Removed partial state.md "
                        f"(len={len(st)}, missing_or_empty_siblings={_exc_missing}) "
                        f"so a later run can re-bootstrap cleanly",
                    )
            except Exception:
                pass
            if is_quota:
                log(
                    paths,
                    "Bootstrap aborted: API quota/auth exhausted at startup "
                    "(keys 401/429). No chapters were written. Rotate or add fresh "
                    "keys in this novel's config.yaml (api.api_key / api_keys / "
                    "api_key_groups), or wait for the shared quota window to reset, "
                    "then re-run `novel.py run <name>` — it resumes cleanly.",
                )
                raise SystemExit(
                    "Bootstrap aborted: API quota/auth exhausted (keys 401/429). "
                    "Rotate keys or wait for quota reset, then re-run."
                ) from exc
            raise

    if not paths.book.exists() and find_last_chapter(paths) > 0:
        rebuild_book(paths)

    target = int(config["novel"]["target_words"])
    # Optional hard cap on chapter count (short-novel mode). 0/absent => no cap,
    # so the long novel (which never sets this) keeps its char-target-only loop.
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    log(paths, f"Start target_chars={target} current_chars={count_chars(paths.book)} max_chapters={max_chapters or 'none'}")
    # V3-#2: anchor-completion gate. In short-novel mode the loop terminates on
    # char/chapter cap alone, which can stop with the climactic volume_plan
    # anchors (truth reveal / confrontation / cost payoff) never dramatized.
    # When we're about to stop, audit those anchors; if any are unrealized, allow
    # a bounded number of extra chapters (each carrying an explicit "land this
    # anchor" directive) so the finale is actually written rather than summarized
    # away. Bounded by anchor_gate_max_extra to avoid an unbounded tail.
    anchor_gate_enabled = bool(config["novel"].get("anchor_gate_enabled", True))
    anchor_gate_max_extra = int(config["novel"].get("anchor_gate_max_extra", 3))
    anchor_extra_used = 0
    halted_by_breaker = False
    # P0 graceful-close guard: when the quality circuit breaker trips in
    # short-novel mode, we pull the climax forward (make the next chapter the
    # finale) instead of halting with an unfinished book. This flag (a) caps
    # that to ONE graceful close (a second breaker trip falls back to HALT),
    # and (b) suppresses the anchor-completion gate's +1 re-extension so it
    # can't undo the early close on an exhausted premise.
    graceful_close_used = False
    # Cache the last-written chapter number to avoid repeated glob scans.
    # Updated at the end of each iteration; find_last_chapter is still called
    # at loop entry points where accuracy is critical.
    last_written = find_last_chapter(paths)
    background = BackgroundTasks(paths, conn)
    try:
        while True:
            if book_reached_target(paths.book, target):
                last_chapter = find_last_chapter(paths)
                stop_for_chapters = bool(max_chapters and last_chapter >= max_chapters)
                # Only the anchor gate can keep us going past the quantitative
                # cap, and only in short-novel mode with budget left.
                if (
                    anchor_gate_enabled
                    and max_chapters
                    and (stop_for_chapters or book_reached_target(paths.book, target))
                    and anchor_extra_used < anchor_gate_max_extra
                    and not graceful_close_used
                ):
                    background.wait_label(f"chapter_finalize_ch{last_chapter}")
                    try:
                        gate = anchor_completion_gate(client, paths, conn, config, last_chapter)
                    except Exception as exc:
                        log(paths, f"Anchor gate failed (non-fatal) Ch{last_chapter}: {exc}")
                        gate = {"all_anchors_realized": True, "directives": []}
                    if not gate.get("all_anchors_realized", True):
                        anchor_extra_used += 1
                        directives = gate.get("directives") or []
                        log(
                            paths,
                            f"Anchor gate: {len(gate.get('unrealized_anchors') or [])} unrealized "
                            f"anchor(s) at Ch{last_chapter}; extending +1 chapter "
                            f"({anchor_extra_used}/{anchor_gate_max_extra}) to dramatize them.",
                        )
                        # Persist directives onto the latest review so the next
                        # writer reads them, and raise the chapter cap by one.
                        try:
                            existing = load_checkpoint(paths, last_chapter, "final_review.json")
                            if isinstance(existing, dict):
                                wd = list(existing.get("writer_directives_for_next_chapter") or [])
                                for d in directives:
                                    if d and d not in wd:
                                        wd.append(d)
                                existing["writer_directives_for_next_chapter"] = wd[:12]
                                save_checkpoint(paths, last_chapter, "final_review.json", existing)
                        except Exception as exc:
                            log(paths, f"Failed to persist anchor directives Ch{last_chapter}: {exc}")
                        if max_chapters:
                            max_chapters = last_chapter + 1
                        # Fall through to generate the extra chapter.
                    else:
                        log(paths, "Anchor gate: all must-hit anchors realized; stopping.")
                        break
                elif graceful_close_used and find_last_chapter(paths) < max_chapters:
                    # Graceful close pulled the climax forward: the char/chapter
                    # target is already met, but the finale chapter (max_chapters,
                    # just bumped to last+1) has NOT been written yet. Fall through
                    # to write it instead of stopping on the target short-circuit.
                    pass
                else:
                    break
            last_chapter = find_last_chapter(paths)
            if max_chapters and last_chapter >= max_chapters:
                log(paths, f"Reached max_chapters={max_chapters}; stopping chapter loop")
                break
            is_resume = should_resume_existing_chapter(paths, last_chapter)
            if is_resume:
                # A chapter file + checkpoint dir exist but the finalize markers
                # (extraction.json / structured_state_done.json / chapter_completed)
                # aren't all present yet. This is almost always because the PREVIOUS
                # iteration just submitted `chapter_finalize_ch{n}` to the background
                # pool and the loop raced ahead before those markers were written —
                # NOT a genuine crash-resume. Joining the finalize barrier here lets
                # the background extract/structured writes become durable, so the
                # re-check below can see the chapter as finished and advance, instead
                # of needlessly replaying its whole review/revise/local-fix loop
                # (which both wastes ~1 extra chapter's worth of LLM calls and can
                # regress the accepted text to a lower-scoring earlier round).
                background.wait_label(f"chapter_finalize_ch{last_chapter}")
                if not should_resume_existing_chapter(paths, last_chapter):
                    log(
                        paths,
                        f"Ch{last_chapter} finalize completed after barrier wait; "
                        f"advancing instead of resuming.",
                    )
                    is_resume = False
            if is_resume:
                chapter_num = last_chapter
                log(paths, f"Resuming partially indexed Ch{chapter_num}")
            else:
                chapter_num = last_chapter + 1
            generate_one_chapter(client, paths, conn, config, chapter_num, background=background, resume=is_resume)
            last_written = find_last_chapter(paths)  # refresh once after write
            total = count_chars(paths.book)
            log(paths, f"Progress chars={total}/{target} pct={total / target * 100:.2f}%")
            # Mid-book degradation early-warning: catch the downward slide BEFORE
            # the circuit breaker has to halt the book. Persists a recovery
            # directive read by the next chapter's writer prompt + planning upshift.
            try:
                import json as _json
                deg = _detect_quality_degradation(paths, conn, config, last_written)
                rec_path = paths.logs_dir / "recovery_directive.json"
                if deg:
                    write_text(rec_path, _json.dumps(deg, ensure_ascii=False, indent=2))
                    log(paths, f"DEGRADATION ALERT Ch{last_written}: {deg['reason']} "
                        f"(avg={deg['avg_score']}); recovery mode active until "
                        f"Ch{deg['active_until']}.")
                    db_event(conn, last_written, "degradation_alert", deg)
            except Exception as exc:
                log(paths, f"degradation detector failed (non-fatal): {exc}")
            # O3: consecutive-low-quality circuit breaker. suspense_v11 burned
            # 21 replans across 5 consecutive force-accepted chapters
            # (5.5/5.5/5.5/3.5/4.5) with nobody pulling the cord — every chapter
            # cost ~4 full drafts and made the context worse. When N consecutive
            # chapters were force-accepted below the breaker floor, the engine
            # is in a failure mode more tokens won't fix: halt loudly and leave
            # the decision to a human (resume works as usual after intervention).
            breaker_n = int(config["novel"].get("quality_breaker_consecutive", 2))
            breaker_floor = float(config["novel"].get("quality_breaker_score_floor", 6.0))
            if breaker_n > 0:
                last = last_written  # already refreshed above, no glob needed
                consecutive = 0
                for ch in range(last, 0, -1):
                    d = load_checkpoint(paths, ch, "quality_debt.json")
                    if isinstance(d, dict) and safe_score(d.get("score", 10)) < breaker_floor:
                        consecutive += 1
                    else:
                        break
                if consecutive >= breaker_n:
                    graceful_ok = (
                        bool(config["novel"].get("quality_breaker_graceful_close", True))
                        and max_chapters
                        and not graceful_close_used
                        and last < max_chapters
                    )
                    if graceful_ok:
                        # P0 fix: premise exhaustion in short-novel mode. Rather
                        # than HALT with an unfinished book, pull the climax
                        # forward — make the NEXT chapter the finale. Bumping the
                        # loop-local is not enough: is_final_chapter() and the
                        # CLOSING_RULES_BLOCK gate re-read config["novel"]
                        # ["max_chapters"] (the dict), so write the new cap there
                        # too or the "finale" degrades to an ordinary chapter and
                        # the story never closes.
                        max_chapters = last + 1
                        config["novel"]["max_chapters"] = last + 1
                        graceful_close_used = True
                        log(
                            paths,
                            f"QUALITY CIRCUIT BREAKER (graceful close): {consecutive} "
                            f"consecutive chapter(s) force-accepted below {breaker_floor} "
                            f"(Ch{last - consecutive + 1}-Ch{last}). Premise looks "
                            f"exhausted; pulling the climax forward — Ch{last + 1} will "
                            f"be the finale (CLOSING_RULES fires) so the book ends "
                            f"cleanly instead of halting unfinished.",
                        )
                        db_event(conn, last, "quality_circuit_breaker_graceful", {
                            "consecutive": consecutive,
                            "floor": breaker_floor,
                            "chapters": list(range(last - consecutive + 1, last + 1)),
                            "finale_chapter": last + 1,
                        })
                        # Do NOT break: fall through to write the finale chapter.
                    else:
                        log(
                            paths,
                            f"QUALITY CIRCUIT BREAKER: {consecutive} consecutive chapter(s) "
                            f"force-accepted below {breaker_floor} "
                            f"(Ch{last - consecutive + 1}-Ch{last}). The pipeline is in a "
                            f"failure mode that more LLM calls will not fix (likely premise "
                            f"exhaustion or context poisoning). HALTING. Inspect the recent "
                            f"chapters/quality_debt, adjust prompt.md/config, then re-run — "
                            f"the run resumes from checkpoint.",
                        )
                        db_event(conn, last, "quality_circuit_breaker", {
                            "consecutive": consecutive,
                            "floor": breaker_floor,
                            "chapters": list(range(last - consecutive + 1, last + 1)),
                        })
                        halted_by_breaker = True
                        break
    finally:
        log(paths, "Waiting for background tasks to finish before exit...")
        background.wait_pending()

    if halted_by_breaker:
        log(paths, f"Halted by quality circuit breaker at total_chars={count_chars(paths.book)}; refine pass skipped.")
        return

    log(paths, f"Done total_chars={count_chars(paths.book)}")

    # Post-completion book packaging: titles / intros / tags / synopsis ->
    # package.md + logs/package.json. Runs before refine so it describes the
    # canonical chapters/book.md. Gated + best-effort; never touches prose.
    if bool(config["novel"].get("package_after_complete", False)):
        try:
            from package import build_package
            log(paths, "Generating book package (titles/intros/synopsis)")
            build_package(client, paths, config)
        except Exception as exc:
            log(paths, f"Package generation failed (non-fatal): {exc}")

    # Post-completion refine pass. Re-reads the finished book in 5-chapter
    # groups and emits chapters_refined/ + book_refined.md. Original chapters/
    # and book.md are not touched. Gated by config flag; default off so a
    # restart-after-completion doesn't accidentally re-spend tokens.
    if bool(config["novel"].get("refine_after_complete", False)):
        try:
            from refine import refine_book
            log(paths, "Starting post-completion refine pass")
            refine_book(client, paths, conn, config)
        except Exception as exc:
            log(paths, f"Refine pass failed (non-fatal): {exc}")

    log(paths, "Book complete")
