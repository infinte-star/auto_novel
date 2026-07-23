from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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

    def has_errors(self, label: str) -> bool:
        """Check whether the most recent task with *label* recorded any error."""
        with self.lock:
            for task_label, thread, error_list in reversed(self.tasks):
                if task_label == label:
                    return bool(error_list)
        return False

    def last_error(self, label: str) -> Exception | None:
        """Return the last exception from the most recent task with *label*."""
        with self.lock:
            for task_label, thread, error_list in reversed(self.tasks):
                if task_label == label:
                    return error_list[-1] if error_list else None
        return None

    def errors(self, label: str | None = None) -> list[dict[str, Any]]:
        """Structured error info for completed tasks, optionally filtered by *label*."""
        with self.lock:
            snapshot = list(self.tasks)
        results: list[dict[str, Any]] = []
        for task_label, thread, error_list in snapshot:
            if label is not None and task_label != label:
                continue
            for exc in error_list:
                results.append({
                    "label": task_label,
                    "error": repr(exc),
                    "type": type(exc).__name__,
                })
        return results


class RevisionTracker:
    """Track revision score trajectory and detect convergence plateaus.

    Three stopping signals (checked in order):
    1. CONVERGED: score meets or exceeds threshold and review is accepted
    2. STALLED: N consecutive rounds with no score improvement
    3. PLATEAU: score oscillating in a narrow band with no net progress
    """

    __slots__ = ("threshold", "max_no_improvement", "plateau_window",
                 "plateau_band", "scores", "best_score", "no_improvement_count")

    def __init__(
        self,
        threshold: float,
        max_no_improvement: int = 1,
        plateau_window: int = 3,
        plateau_band: float = 0.3,
    ) -> None:
        self.threshold = threshold
        self.max_no_improvement = max_no_improvement
        self.plateau_window = plateau_window
        self.plateau_band = plateau_band
        self.scores: list[float] = []
        self.best_score: float = 0.0
        self.no_improvement_count: int = 0

    def record(self, score: float, accepted: bool) -> str:
        """Record a round's result.

        Returns ``"continue"``, ``"converged"``, ``"stalled"``, or ``"plateau"``.
        """
        self.scores.append(score)

        if score >= self.threshold and accepted:
            return "converged"

        if score > self.best_score:
            self.best_score = score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.max_no_improvement and len(self.scores) > 1:
            return "stalled"

        if len(self.scores) >= self.plateau_window:
            window = self.scores[-self.plateau_window:]
            if max(window) - min(window) <= self.plateau_band:
                return "plateau"

        return "continue"

    def summary(self) -> dict[str, Any]:
        return {
            "scores": list(self.scores),
            "best_score": self.best_score,
            "no_improvement_count": self.no_improvement_count,
            "rounds": len(self.scores),
        }


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
            chapter_aux_cache=chapter_aux_cache,
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
                chapter_aux_cache=chapter_aux_cache,
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


@dataclass
class ChapterState:
    """Mutable state threaded through a single chapter's generation stages."""
    client: Any
    paths: Paths
    conn: Any
    config: dict[str, Any]
    chapter_num: int
    background: BackgroundTasks | None
    resume: bool

    tail: str = ""
    cached_memory: str = ""
    writing_memory: str = ""
    chapter_aux: dict | None = None

    plan: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)

    chapter: str = ""
    candidate_review: dict | None = None

    review: dict[str, Any] = field(default_factory=lambda: {"score": 0, "accepted": False})
    best_chapter: str = ""
    best_review: dict[str, Any] = field(default_factory=dict)

    telemetry_revise_pairs: list[dict[str, Any]] = field(default_factory=list)


def _stage_setup_barriers(state: ChapterState) -> None:
    """Wait on the previous chapter's background tasks before proceeding."""
    if state.background is None or state.chapter_num <= 1:
        return
    prev_label = f"chapter_finalize_ch{state.chapter_num - 1}"
    state.background.wait_label(prev_label)
    if state.background.has_errors(prev_label):
        log(state.paths,
            f"Background finalize Ch{state.chapter_num - 1} had errors: "
            f"{state.background.last_error(prev_label)!r}. "
            f"Resume path will retry synchronously if markers are missing.")
    prev = state.chapter_num - 1
    cold_every = int(state.config["novel"].get("cold_reader_every", 10))
    if (
        bool(state.config["novel"].get("pack_review_barrier", True))
        and cold_every > 0
        and prev % cold_every == 0
    ):
        state.background.wait_label(f"horizon_review_ch{prev}")
    stage_every = int(state.config["novel"].get("stage_review_every", 20))
    if (
        bool(state.config["novel"].get("stage_review_barrier", True))
        and stage_every > 0
        and prev % stage_every == 0
    ):
        state.background.wait_label(f"stage_review_ch{prev}")
    if bool(state.config["novel"].get("prefetch_next_plan", False)):
        state.background.wait_label(f"prefetch_plan_ch{state.chapter_num}")


def _stage_plan(state: ChapterState) -> None:
    """Create or resume the chapter plan, validate continuity, build writer context."""
    state.tail = _build_writer_tail(state.paths, state.config, state.chapter_num)
    state.cached_memory = memory_context(
        state.paths, state.conn, state.config,
        max_chars=int(state.config["novel"].get("plan_memory_chars", 60000) or 0),
    )
    try:
        from review import build_chapter_aux_cache
        state.chapter_aux = build_chapter_aux_cache(state.paths, state.conn, state.config, state.chapter_num)
    except Exception:
        state.chapter_aux = None

    final_payload = load_checkpoint(state.paths, state.chapter_num, "validated_plan.json")
    if isinstance(final_payload, dict) and final_payload.get("plan") and final_payload.get("decision"):
        log(state.paths, f"Resuming validated plan Ch{state.chapter_num}")
        state.plan = final_payload["plan"]
        state.decision = final_payload["decision"]
    else:
        state.plan, state.decision = create_plan(
            state.client, state.paths, state.conn, state.config,
            state.chapter_num, state.tail, cached_memory=state.cached_memory,
        )
        save_checkpoint(state.paths, state.chapter_num, "validated_plan.json",
                        {"plan": state.plan, "decision": state.decision})

    violations = validate_plan_continuity(state.conn, state.plan, state.chapter_num, config=state.config)
    if violations:
        log(state.paths, f"Continuity violations Ch{state.chapter_num}: {violations}")
        critical = [v for v in violations if v.startswith("CRITICAL")]
        if critical:
            log(state.paths, f"Critical violations found, re-planning Ch{state.chapter_num}")
            state.decision.setdefault("required_constraints", []).extend(violations)
            state.plan, state.decision = create_plan(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, state.tail,
                checkpoint_label="critical", cached_memory=state.cached_memory,
            )
            save_checkpoint(state.paths, state.chapter_num, "validated_plan.json",
                            {"plan": state.plan, "decision": state.decision})
        else:
            state.decision.setdefault("required_constraints", []).extend(violations)

    _pov = str(state.plan.get("pov_character", "") or "").strip() or None
    state.writing_memory = writing_memory_context(state.paths, state.conn, state.config, pov_character=_pov)


def _stage_write(state: ChapterState) -> None:
    """Write or resume the chapter draft, including fossil-catastrophe and adjacent-duplicate gates."""
    existing_chapter = read_text(chapter_path(state.paths, state.chapter_num))
    cached_chapter = load_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT) or existing_chapter
    if cached_chapter:
        state.chapter = normalize_chapter(str(cached_chapter))
        log(state.paths, f"Resuming cached chapter text Ch{state.chapter_num}")
        return

    log(state.paths, f"Writing Ch{state.chapter_num}: {state.plan.get('title', '')}")
    chapter, candidate_review = write_chapter_with_candidates(
        state.client, state.paths, state.conn, state.config,
        state.chapter_num, state.plan, state.decision, state.tail,
        cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
    )

    if chapter is None and isinstance(candidate_review, dict) and candidate_review.get("catastrophe") == "fossil_prescreen":
        log(
            state.paths,
            f"CATASTROPHIC PRE-SCREEN FAILURE Ch{state.chapter_num}: all candidates blocked by fossils. "
            f"Triggering PLAN-LEVEL replan to break the repetition cycle.",
        )
        try:
            fossil_feedback = (
                "【结构性重规划·化石灾难】前几章已大量复读签名句（化石句累积），"
                "所有候选草稿都被化石门拦截。本章必须从【全新】角度切入：\n"
                "- 严禁复刻已用过的场景设计、人物动作、对话模式；\n"
                "- 改变叙述视角、物理场所、或对话方式；宁可另起炉灶，也不许在旧轨道上微调。"
            )
            replan_tail = _build_writer_tail(state.paths, state.config, state.chapter_num)
            replan_plan, replan_decision = create_plan(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, replan_tail,
                checkpoint_label="fossil_catastrophe",
                cached_memory=state.cached_memory,
                replan_feedback=fossil_feedback,
            )
            chapter, candidate_review = write_chapter_with_candidates(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, replan_plan, replan_decision, replan_tail,
                cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
            )
            if chapter is not None:
                state.plan = replan_plan
                state.decision = replan_decision
                save_checkpoint(state.paths, state.chapter_num, "validated_plan.json",
                                {"plan": state.plan, "decision": state.decision})
                log(state.paths, f"Fossil-catastrophe replan Ch{state.chapter_num} completed")
        except Exception as exc:
            log(state.paths, f"Fossil-catastrophe replan Ch{state.chapter_num} failed: {exc}")

    if chapter is None:
        raise RuntimeError(f"Failed to generate any valid chapter text for Ch{state.chapter_num}")

    if bool(state.config["novel"].get("adjacent_repeat_enabled", True)) and state.chapter_num > 1:
        try:
            from quality import adjacent_repetition
            prev_text = read_text(chapter_path(state.paths, state.chapter_num - 1))
            ar = adjacent_repetition(chapter, prev_text, state.config)
            if ar.get("level") == "block":
                log(
                    state.paths,
                    f"Adjacent-duplicate draft Ch{state.chapter_num} metrics={ar.get('metrics')}; "
                    f"regenerating once with anti-repeat constraint.",
                )
                retry_decision = dict(state.decision)
                retry_decision["required_constraints"] = list(state.decision.get("required_constraints") or [])
                for d in ar.get("directives", []):
                    if d not in retry_decision["required_constraints"]:
                        retry_decision["required_constraints"].append(d)
                retry_chapter, retry_review = write_chapter_with_candidates(
                    state.client, state.paths, state.conn, state.config,
                    state.chapter_num, state.plan, retry_decision, state.tail,
                    cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                )
                ar2 = adjacent_repetition(retry_chapter, prev_text, state.config)
                if ar2.get("level") != "block":
                    chapter, candidate_review = retry_chapter, retry_review
                    state.decision = retry_decision
                    log(state.paths, f"Adjacent-duplicate retry Ch{state.chapter_num} clean (metrics={ar2.get('metrics')})")
                elif float(ar2.get("metrics", {}).get("clause_overlap", 1.0)) < float(
                    ar.get("metrics", {}).get("clause_overlap", 1.0)
                ):
                    chapter, candidate_review = retry_chapter, retry_review
                    state.decision = retry_decision
                    log(
                        state.paths,
                        f"Adjacent-duplicate retry Ch{state.chapter_num} still high but improved "
                        f"({ar.get('metrics', {}).get('clause_overlap')} -> "
                        f"{ar2.get('metrics', {}).get('clause_overlap')}); review gate will judge.",
                    )
                else:
                    log(state.paths, f"Adjacent-duplicate retry Ch{state.chapter_num} did not improve; keeping original draft")
        except Exception as exc:
            log(state.paths, f"Adjacent-duplicate draft gate failed (non-fatal) Ch{state.chapter_num}: {exc}")

    state.chapter = chapter
    state.candidate_review = candidate_review
    save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
    if state.candidate_review is not None:
        save_checkpoint(state.paths, state.chapter_num, "review_round0.json", state.candidate_review)


def _stage_review_revise(state: ChapterState) -> None:
    """Run the review-revise loop with plateau detection."""
    threshold = float(state.config["novel"]["quality_threshold"])
    max_rounds = int(state.config["novel"]["max_revision_rounds"])
    max_no_improvement_rounds = int(state.config["novel"].get("max_no_improvement_revision_rounds", 1))
    tracker = RevisionTracker(
        threshold=threshold,
        max_no_improvement=max_no_improvement_rounds,
        plateau_window=int(state.config["novel"].get("revision_plateau_window", 3)),
        plateau_band=float(state.config["novel"].get("revision_plateau_band", 0.3)),
    )
    skip_revise_macro = bool(state.config["novel"].get("skip_revise_on_macro_fail", True))

    for round_num in range(max_rounds + 1):
        if round_num > 0:
            if skip_revise_macro:
                fk, fr = _classify_replan_failure(state.review, state.config)
                if fk == "structural":
                    log(
                        state.paths,
                        f"Skipping revise rounds Ch{state.chapter_num}: structural failure ({fr}); "
                        f"revise patches are ~0-gain here — deferring to structural replan.",
                    )
                    break
            revised_key = f"chapter_revised_round{round_num}.md"
            pre_revise_text = state.chapter
            pre_revise_review = state.review
            revised = load_checkpoint(state.paths, state.chapter_num, revised_key)
            if revised:
                state.chapter = normalize_chapter(str(revised))
                log(state.paths, f"Resuming revised chapter Ch{state.chapter_num} round={round_num}")
            else:
                log(
                    state.paths,
                    f"Revising Ch{state.chapter_num} round={round_num} because score={state.review.get('score')}/10 < {threshold}",
                )
                revised_text = revise_chapter(
                    state.client, state.paths, state.conn, state.config,
                    state.chapter, state.review, state.plan, state.tail,
                    cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                )
                if bool(state.config["novel"].get("revision_gate_enabled", True)):
                    try:
                        from quality import style_health
                        pre_sh = style_health(pre_revise_text, state.config)
                        post_sh = style_health(revised_text, state.config)
                        pre_p = float(pre_sh.get("penalty", 0))
                        post_p = float(post_sh.get("penalty", 0))
                        if post_p > pre_p + 0.5:
                            log(state.paths, f"Revision gate ROLLBACK Ch{state.chapter_num} round={round_num}: style_penalty {pre_p:.1f}→{post_p:.1f}")
                            revised_text = pre_revise_text
                    except Exception:
                        pass
                state.chapter = revised_text
                save_checkpoint(state.paths, state.chapter_num, revised_key, state.chapter)
            save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)

        review_key = f"review_round{round_num}.json"
        cached_review = load_checkpoint(state.paths, state.chapter_num, review_key)
        if isinstance(cached_review, dict):
            state.review = cached_review
            log(state.paths, f"Resuming cached review Ch{state.chapter_num} round={round_num} score={state.review.get('score')}/10")
        else:
            state.review = review_chapter(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, state.plan, state.chapter, state.tail,
                cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
            )
            save_checkpoint(state.paths, state.chapter_num, review_key, state.review)
            log(state.paths, f"Reviewed Ch{state.chapter_num} round={round_num} score={state.review.get('score')}/10")

        if round_num > 0:
            try:
                state.telemetry_revise_pairs.append({
                    "round": round_num,
                    "text_before": pre_revise_text,
                    "review": pre_revise_review,
                    "text_after": state.chapter,
                    "score_before": safe_score(pre_revise_review.get("score", 0)),
                    "score_after": safe_score(state.review.get("score", 0)),
                })
            except Exception:
                pass

        current_score = safe_score(state.review.get("score", 0))
        if current_score > safe_score(state.best_review.get("score", 0)):
            state.best_chapter = state.chapter
            state.best_review = dict(state.review)
        signal = tracker.record(current_score, state.review.get("accepted", True))
        if signal == "converged":
            state.review["accepted"] = True
            break
        if signal != "continue" and round_num > 0:
            log(
                state.paths,
                f"Stopping Ch{state.chapter_num} revisions: {signal} "
                f"(scores={tracker.summary()['scores']}, best={tracker.best_score}).",
            )
            break


def _stage_quality_replan(state: ChapterState) -> None:
    """Handle quality replan routing: local fix, structural replan, force-accept."""
    threshold = float(state.config["novel"]["quality_threshold"])
    if not (
        bool(state.config["novel"].get("replan_on_low_quality", True))
        and (safe_score(state.review.get("score", 0)) < threshold or not state.review.get("accepted", True))
        and not load_checkpoint(state.paths, state.chapter_num, "quality_replan_done.json")
    ):
        return

    try:
        fail_kind, fail_reason = _classify_replan_failure(state.review, state.config)
        log(state.paths, f"Quality replan routing Ch{state.chapter_num}: kind={fail_kind} ({fail_reason})")
        if state.review.get("gate_rejects"):
            try:
                db_event(state.conn, state.chapter_num, "gate_reject", {
                    "gates": [g.get("gate") for g in state.review.get("gate_rejects", []) if isinstance(g, dict)],
                    "score": state.review.get("score"),
                })
            except Exception:
                pass

        if fail_kind == "local" and bool(state.config["novel"].get("local_fix_before_replan", True)):
            local_rounds = int(state.config["novel"].get("local_fix_max_rounds", 2))
            local_chapter = state.chapter
            local_review = state.review
            improved = False
            for lr in range(1, local_rounds + 1):
                lkey = f"local_fix_round{lr}.md"
                cached_local = load_checkpoint(state.paths, state.chapter_num, lkey)
                if cached_local:
                    local_chapter = normalize_chapter(str(cached_local))
                else:
                    local_chapter = revise_chapter(
                        state.client, state.paths, state.conn, state.config,
                        local_chapter, local_review, state.plan, state.tail,
                        cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                    )
                    save_checkpoint(state.paths, state.chapter_num, lkey, local_chapter)
                local_review = review_chapter(
                    state.client, state.paths, state.conn, state.config,
                    state.chapter_num, state.plan, local_chapter, state.tail,
                    cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                )
                log(state.paths, f"Local fix Ch{state.chapter_num} round={lr} score={local_review.get('score')}/10")
                if safe_score(local_review.get("score", 0)) > safe_score(state.best_review.get("score", 0)):
                    state.chapter = normalize_chapter(local_chapter)
                    state.review = local_review
                    state.best_chapter = state.chapter
                    state.best_review = dict(state.review)
                    save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
                    save_checkpoint(state.paths, state.chapter_num, "final_review.json", state.review)
                    improved = True
                if (
                    safe_score(local_review.get("score", 0)) >= threshold
                    and local_review.get("accepted", True)
                ):
                    state.review["accepted"] = True
                    break
            save_checkpoint(
                state.paths, state.chapter_num, "quality_replan_done.json",
                {"route": "local", "reason": fail_reason, "improved": improved,
                 "score_after": state.review.get("score")},
            )
            raise _LocalFixDone()

        roi_margin = float(state.config["novel"].get("replan_roi_skip_margin", 0.7))
        if (
            bool(state.config["novel"].get("replan_roi_breaker_enabled", True))
            and not state.review.get("gate_rejects")
            and safe_score(state.review.get("score", 0)) >= threshold - roi_margin
            and not cost_savings_disabled(state.config, state.chapter_num)
            and _recent_replan_ineffective(state.paths, state.chapter_num, state.config)
        ):
            log(
                state.paths,
                f"Replan ROI stop Ch{state.chapter_num}: recent replans ineffective and "
                f"shortfall marginal (score={state.review.get('score')}/{threshold}); "
                f"accepting best draft with quality debt to save cost.",
            )
            db_event(state.conn, state.chapter_num, "replan_roi_stop", {
                "score": state.review.get("score"),
                "threshold": threshold,
            })
            save_checkpoint(state.paths, state.chapter_num, "quality_replan_done.json",
                            {"route": "roi_stop", "score_after": state.review.get("score")})
            raise _LocalFixDone()

        log(
            state.paths,
            f"Quality replan Ch{state.chapter_num}: best score={state.review.get('score')}/10 below threshold={threshold}",
        )
        replan_tail = _build_writer_tail(state.paths, state.config, state.chapter_num)
        replan_memory = state.cached_memory
        replan_fb = _build_replan_feedback(state.review)
        diagnose_constraints: list[str] = []
        if bool(state.config["novel"].get("structural_diagnose_enabled", True)):
            try:
                from planning import diagnose_structural_failure as _diagnose
                diag = _diagnose(
                    state.client, state.paths, state.config, state.chapter_num,
                    state.plan, state.review, state.chapter,
                )
                if isinstance(diag, dict) and diag:
                    save_checkpoint(state.paths, state.chapter_num, "structural_diagnose.json", diag)
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
                    if diagnose_constraints:
                        replan_fb = (replan_fb + "\n" if replan_fb else "") + \
                            "## 重写前诊断（必须据此重做场景）\n- " + "\n- ".join(diagnose_constraints)
                    log(
                        state.paths,
                        f"Structural diagnose Ch{state.chapter_num}: root_cause={rc[:60]!r} "
                        f"weakest={wd} failed_beats={len(fb)}",
                    )
            except Exception as exc:
                log(state.paths, f"Structural diagnose failed (non-fatal) Ch{state.chapter_num}: {exc}")
        replan_plan, replan_decision = create_plan(
            state.client, state.paths, state.conn, state.config,
            state.chapter_num, replan_tail,
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
        sr_candidates = int(state.config["novel"].get("structural_replan_candidates", 3))
        sr_temp = float(state.config["novel"].get("structural_replan_temperature", 0.65))
        replan_chapter, replan_review = write_chapter_with_candidates(
            state.client, state.paths, state.conn, state.config,
            state.chapter_num, replan_plan, replan_decision, replan_tail,
            cached_memory=state.writing_memory,
            num_candidates_override=max(1, sr_candidates),
            base_temp_override=sr_temp,
            chapter_aux_cache=state.chapter_aux,
        )
        if replan_review is None:
            replan_review = review_chapter(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, replan_plan, replan_chapter, replan_tail,
                cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
            )
        save_checkpoint(
            state.paths, state.chapter_num, "quality_replan_done.json",
            {"score_before": state.review.get("score"),
             "score_after": replan_review.get("score"),
             "accepted_after": replan_review.get("accepted")},
        )
        replan_score = safe_score(replan_review.get("score", 0))
        if replan_score > safe_score(state.best_review.get("score", 0)):
            state.chapter = normalize_chapter(replan_chapter)
            state.review = replan_review
            state.plan = replan_plan
            state.decision = replan_decision
            state.best_chapter = state.chapter
            state.best_review = dict(state.review)
            save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
            save_checkpoint(state.paths, state.chapter_num, "validated_plan.json",
                            {"plan": state.plan, "decision": state.decision})
            save_checkpoint(state.paths, state.chapter_num, "final_review.json", state.review)
            log(state.paths, f"Quality replan Ch{state.chapter_num} improved best score to {state.review.get('score')}/10")
        else:
            log(
                state.paths,
                f"Quality replan Ch{state.chapter_num} did not beat best "
                f"(new score={replan_score}/10, best={safe_score(state.best_review.get('score', 0))}/10); keeping best.",
            )
    except _LocalFixDone:
        pass
    except Exception as exc:
        save_checkpoint(state.paths, state.chapter_num, "quality_replan_done.json", {"error": str(exc)})
        log(state.paths, f"Quality replan failed (non-fatal) Ch{state.chapter_num}: {exc}")


def _stage_force_accept(state: ChapterState) -> None:
    """Fall back to the best draft and handle force-accept when quality is still below threshold."""
    threshold = float(state.config["novel"]["quality_threshold"])
    max_rounds = int(state.config["novel"]["max_revision_rounds"])
    if not (safe_score(state.review.get("score", 0)) < threshold or not state.review.get("accepted", True)):
        return

    state.chapter = state.best_chapter
    state.review = state.best_review

    gate_rejects = state.review.get("gate_rejects", [])
    hard_floor = 3.0
    if gate_rejects and safe_score(state.review.get("score", 0)) <= hard_floor:
        log(
            state.paths,
            f"Ch{state.chapter_num} HARD FLOOR violation: score={state.review.get('score')}/10 "
            f"with gate_rejects={[g['gate'] for g in gate_rejects]}. "
            f"Triggering STRUCTURAL replan instead of force-accept.",
        )
        try:
            gate_dirs: list[str] = []
            for gr in gate_rejects:
                gate_dirs.extend(gr.get("directives", []))
            replan_feedback = _build_replan_feedback(state.review)
            if gate_dirs:
                replan_feedback += "\n【确定性质量门作废本稿，必须结构性重做，不可修补】\n- " + "\n- ".join(
                    str(d) for d in gate_dirs[:8]
                )
            replan_tail = _build_writer_tail(state.paths, state.config, state.chapter_num)
            replan_plan, replan_decision = create_plan(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, replan_tail,
                checkpoint_label="hard_floor",
                cached_memory=state.cached_memory,
                replan_feedback=replan_feedback,
            )
            replan_chapter, replan_review = write_chapter_with_candidates(
                state.client, state.paths, state.conn, state.config,
                state.chapter_num, replan_plan, replan_decision, replan_tail,
                cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
            )
            if replan_review is None:
                replan_review = review_chapter(
                    state.client, state.paths, state.conn, state.config,
                    state.chapter_num, replan_plan, replan_chapter, replan_tail,
                    cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                )
            state.chapter = normalize_chapter(replan_chapter)
            state.review = replan_review
            state.plan = replan_plan
            state.best_chapter = state.chapter
            state.best_review = dict(state.review)
            save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
            save_checkpoint(state.paths, state.chapter_num, "validated_plan.json",
                            {"plan": state.plan, "decision": replan_decision})
            save_checkpoint(state.paths, state.chapter_num, "final_review.json", state.review)
            log(state.paths, f"Hard-floor replan Ch{state.chapter_num} completed: new score={state.review.get('score')}/10")
        except Exception as exc:
            log(state.paths, f"Hard-floor replan Ch{state.chapter_num} failed: {exc}. Falling back to force-accept.")

    if safe_score(state.review.get("score", 0)) < threshold or not state.review.get("accepted", True):
        consecutive_force_accept_limit = int(state.config["novel"].get("consecutive_force_accept_limit", 2))
        breaker_floor = float(state.config["novel"].get("circuit_breaker_score_floor", 7.0))
        if consecutive_force_accept_limit > 0 and state.chapter_num > consecutive_force_accept_limit:
            try:
                recent_force_accepts = []
                cur_score = safe_score(state.review.get("score", 0))
                if cur_score < breaker_floor:
                    recent_force_accepts.append((state.chapter_num, cur_score))
                for ch_back in range(state.chapter_num - 1, state.chapter_num - consecutive_force_accept_limit, -1):
                    if ch_back < 1:
                        break
                    try:
                        past_review = state.conn.execute(
                            "SELECT score FROM chapter_metrics WHERE chapter = ?", (ch_back,)
                        ).fetchone()
                        if past_review and float(past_review[0]) < breaker_floor:
                            recent_force_accepts.append((ch_back, float(past_review[0])))
                        else:
                            break
                    except Exception:
                        break
                if len(recent_force_accepts) >= consecutive_force_accept_limit:
                    streak = sorted(recent_force_accepts)
                    log(
                        state.paths,
                        f"CIRCUIT BREAKER Ch{state.chapter_num}: {len(streak)} consecutive chapters below floor {breaker_floor} "
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
                log(state.paths, f"Circuit breaker check failed (non-fatal): {exc}")

        state.chapter, state.review = _apply_force_accept_patches(
            state.paths, state.config, state.chapter_num, state.chapter, state.review)
        log(
            state.paths,
            f"Ch{state.chapter_num} did not meet threshold {threshold} after {max_rounds + 1} rounds "
            f"(best score={state.review.get('score')}/10). Accepting anyway to avoid pipeline halt.",
        )
        state.review["accepted"] = True

    if safe_score(state.review.get("score", 0)) <= 0:
        for _ck in ("review_round1.json", "review_round0.json"):
            _prev = load_checkpoint(state.paths, state.chapter_num, _ck)
            if isinstance(_prev, dict) and safe_score(_prev.get("score", 0)) > 0:
                log(
                    state.paths,
                    f"Force-accept Ch{state.chapter_num}: review carried score=0, "
                    f"recovered from {_ck} (score={_prev.get('score')})",
                )
                state.review = {**_prev, "accepted": True}
                break

    state.review["force_accepted"] = True
    save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
    save_checkpoint(state.paths, state.chapter_num, "final_review.json", state.review)

    try:
        sh_metrics = (state.review.get("style_health") or {}).get("metrics") or {}
        debt = {
            "chapter": state.chapter_num,
            "score": state.review.get("score"),
            "style_penalty": (state.review.get("style_health") or {}).get("penalty", 0.0),
            "em_dash_per_kchar": sh_metrics.get("em_dash_per_kchar"),
            "fragment_line_ratio": sh_metrics.get("fragment_line_ratio"),
            "had_contract_violation": bool(state.review.get("contract_violations")),
            "gate_rejects": [
                str(g.get("gate", "?")) for g in (state.review.get("gate_rejects") or [])
                if isinstance(g, dict)
            ],
            "patches_applied": state.review.get("quality_debt_patches_applied", 0),
            "patch_total": state.review.get("quality_debt_patch_total", 0),
            "problems": [str(p)[:160] for p in (state.review.get("problems") or [])[:5]],
        }
        save_checkpoint(state.paths, state.chapter_num, "quality_debt.json", debt)
        db_event(state.conn, state.chapter_num, "quality_debt", debt)
        log(state.paths, f"Quality-debt registered Ch{state.chapter_num} (score={state.review.get('score')}/10) for refine priority")
    except Exception as exc:
        log(state.paths, f"Quality-debt registration failed (non-fatal) Ch{state.chapter_num}: {exc}")


def _stage_hook_revise(state: ChapterState) -> None:
    """Rewrite weak chapter endings with a targeted mini-revise."""
    hook_min = float(state.config["novel"].get("hook_strength_min", 6.0))
    opening_chapters = int(state.config["novel"].get("opening_chapters", 3))
    if state.chapter_num <= opening_chapters:
        hook_min = float(state.config["novel"].get("opening_hook_strength_min", 7.0))
    hook_revise_enabled = bool(state.config["novel"].get("hook_revise_enabled", True))
    hook_strength = safe_score(state.review.get("hook_strength", hook_min))

    hook_recycled = False
    recycled_clauses: list[str] = []
    if (
        hook_revise_enabled
        and state.chapter_num > 1
        and not is_final_chapter(state.config, state.chapter_num)
        and bool(state.config["novel"].get("adjacent_repeat_enabled", True))
    ):
        try:
            from quality import hook_tail_repetition
            lookback = int(state.config["novel"].get("hook_repeat_lookback", 3))
            prev_tails = []
            for num in range(max(1, state.chapter_num - lookback), state.chapter_num):
                t = read_text(chapter_path(state.paths, num))
                if t:
                    prev_tails.append(t)
            hr = hook_tail_repetition(state.chapter, prev_tails, state.config)
            if hr.get("repeat"):
                hook_recycled = True
                recycled_clauses = list(hr.get("repeated_clauses") or [])
                log(
                    state.paths,
                    f"Hook-recycled Ch{state.chapter_num}: ending reuses prior chapter endings "
                    f"(ratio={hr.get('ratio')}, clauses={recycled_clauses[:2]}); forcing hook revise.",
                )
        except Exception as exc:
            log(state.paths, f"hook_tail_repetition check failed (non-fatal) Ch{state.chapter_num}: {exc}")

    if not (
        hook_revise_enabled
        and (hook_recycled or (hook_strength > 0 and hook_strength < hook_min))
        and not is_final_chapter(state.config, state.chapter_num)
        and not load_checkpoint(state.paths, state.chapter_num, "hook_revised.json")
    ):
        return

    try:
        from writing import revise_hook_only as _revise_hook_only
        log(
            state.paths,
            f"Hook-only mini-revise Ch{state.chapter_num} hook_strength={hook_strength}/10 < {hook_min}"
            + (" [recycled hook]" if hook_recycled else ""),
        )
        hook_review = state.review
        if hook_recycled and recycled_clauses:
            hook_review = dict(state.review)
            wd = list(hook_review.get("writer_directives_for_next_chapter") or [])
            wd.insert(0,
                "章末钩子与前几章结尾重复（确定性检测）。以下句子/意象严禁出现在新结尾里："
                + "；".join(f"“{c}”" for c in recycled_clauses[:3])
                + "。必须换一个全新的悬念抓手（新的物证/新的威胁/新的人物动作），不得复用旧钩子。")
            hook_review["writer_directives_for_next_chapter"] = wd
        new_chapter = _revise_hook_only(
            state.client, state.paths, state.config, state.chapter, state.plan, hook_review,
            tail_to_revise_chars=int(state.config["novel"].get("hook_revise_tail_chars", 400)),
        )
        if len(new_chapter.strip()) >= max(500, int(len(state.chapter) * 0.85)):
            state.chapter = new_chapter
            save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
            save_checkpoint(state.paths, state.chapter_num, "hook_revised.json",
                            {"done": True, "hook_strength_before": hook_strength})
            if bool(state.config["novel"].get("hook_revise_rereview", True)):
                try:
                    state.review = review_chapter(
                        state.client, state.paths, state.conn, state.config,
                        state.chapter_num, state.plan, state.chapter, state.tail,
                        cached_memory=state.writing_memory, chapter_aux_cache=state.chapter_aux,
                    )
                    log(
                        state.paths,
                        f"Hook revise re-review Ch{state.chapter_num}: "
                        f"hook_strength={safe_score(state.review.get('hook_strength', 0))}/10 "
                        f"score={safe_score(state.review.get('score', 0))}",
                    )
                except Exception as exc:
                    log(state.paths, f"Hook revise re-review failed (non-fatal) Ch{state.chapter_num}: {exc}")
        else:
            log(state.paths, f"Hook revise produced too-short output ({len(new_chapter)} chars); keeping original")
    except Exception as exc:
        log(state.paths, f"Hook revise failed (non-fatal) Ch{state.chapter_num}: {exc}")


def _stage_save(state: ChapterState) -> None:
    """Apply final remediation (title refine, em-dash reduction) and save chapter to file."""
    if load_checkpoint(state.paths, state.chapter_num, "chapter_saved.json"):
        return

    if (
        bool(state.config["novel"].get("chapter_title_refine_enabled", False))
        and not chapter_path(state.paths, state.chapter_num).exists()
    ):
        try:
            from package import refine_chapter_title, apply_chapter_title
            new_title = refine_chapter_title(state.client, state.paths, state.config, state.chapter_num, state.plan, state.chapter)
            if new_title and new_title != str(state.plan.get("title") or "").strip():
                state.chapter = apply_chapter_title(state.chapter, state.chapter_num, new_title)
                save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
        except Exception as exc:
            log(state.paths, f"Chapter title refine failed (non-fatal) Ch{state.chapter_num}: {exc}")

    if bool(state.config["novel"].get("em_dash_reduce_enabled", True)):
        try:
            from quality import style_health as _sh_final, reduce_em_dash_density as _reduce_em
            _sh_f = _sh_final(state.chapter, state.config)
            _em_f = float(_sh_f.get("metrics", {}).get("em_dash_per_kchar", 0))
            _em_tgt = float(state.config["novel"].get("em_dash_reduce_target_per_kchar", 3.0))
            if _em_f > _em_tgt:
                _before = state.chapter.count("——")
                state.chapter = _reduce_em(state.chapter, state.config)
                _after = state.chapter.count("——")
                if _before != _after:
                    log(state.paths, f"Pre-save em-dash reduction Ch{state.chapter_num}: {_before}->{_after} dashes, {_em_f:.1f}/k")
                    save_checkpoint(state.paths, state.chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
        except Exception as exc:
            log(state.paths, f"Pre-save em-dash reduction failed (non-fatal) Ch{state.chapter_num}: {exc}")

    if chapter_path(state.paths, state.chapter_num).exists():
        log(state.paths, f"Chapter file already exists Ch{state.chapter_num}; skipping duplicate save")
        if not book_is_consistent(state.paths):
            rebuild_book(state.paths)
    else:
        save_chapter(state.paths, state.chapter_num, state.chapter, state.review, state.plan)
    save_checkpoint(state.paths, state.chapter_num, "chapter_saved.json", {"saved": True})


def _stage_finalize(state: ChapterState) -> None:
    """Extract events, update structured state, schedule background tasks."""
    extraction_done = bool(load_checkpoint(state.paths, state.chapter_num, "extraction.json"))
    structured_done = bool(load_checkpoint(state.paths, state.chapter_num, "structured_state_done.json"))
    state_file_done = bool(load_checkpoint(state.paths, state.chapter_num, "state_file_done.json"))
    completed_done = bool(load_checkpoint(state.paths, state.chapter_num, "chapter_completed.json"))

    extract_in_bg = bool(state.config["novel"].get("extract_in_background", False))
    state_in_bg = bool(state.config["novel"].get("state_file_in_background", False))

    def _run_finalize() -> dict[str, Any]:
        if not extraction_done:
            try:
                extraction_local = extract_events(
                    state.client, state.paths, state.conn, state.config,
                    state.chapter_num, state.chapter, cached_memory=state.cached_memory,
                )
            except Exception as exc:
                log(state.paths, f"Extraction failed Ch{state.chapter_num}; using local fallback extraction: {exc}")
                extraction_local = _fallback_extraction(state.plan, state.review, state.chapter_num, str(exc))
            save_checkpoint(state.paths, state.chapter_num, "extraction.json", extraction_local)
        else:
            extraction_local = load_checkpoint(state.paths, state.chapter_num, "extraction.json") or {}
        if not structured_done:
            try:
                update_structured_state(state.paths, state.conn, state.chapter_num, extraction_local, state.review, state.decision, state.plan)
            except Exception as exc:
                log(state.paths, f"Structured-state update failed Ch{state.chapter_num} (non-fatal, continuing): {exc}")
            save_checkpoint(state.paths, state.chapter_num, "structured_state_done.json", {"done": True})
        return extraction_local

    def _run_state_file(extraction_local: dict[str, Any]) -> None:
        update_state_file(state.client, state.paths, state.conn, state.config, state.chapter_num, state.chapter, extraction_local)
        save_checkpoint(state.paths, state.chapter_num, "state_file_done.json", {"done": True})

    finalize_label = f"chapter_finalize_ch{state.chapter_num}"
    state_file_label = f"state_file_ch{state.chapter_num}"

    needs_finalize = not (extraction_done and structured_done and completed_done)
    use_bg_finalize = extract_in_bg and state.background is not None and needs_finalize and not state.resume
    if use_bg_finalize:
        if not completed_done:
            db_event(state.conn, state.chapter_num, "chapter_completed",
                     {"review": state.review, "plan": state.plan, "decision": state.decision})
            save_checkpoint(state.paths, state.chapter_num, "chapter_completed.json", {"done": True})

        def _bg_finalize_and_state() -> None:
            extraction_local = _run_finalize()
            if not state_file_done:
                _run_state_file(extraction_local)
        state.background.submit(finalize_label, _bg_finalize_and_state)
    else:
        max_attempts = int(state.config["novel"].get("max_finalize_attempts", 3) or 3)
        attempts = bump_finalize_attempts(state.paths, state.chapter_num) if state.resume and needs_finalize else 0
        if state.resume and needs_finalize and attempts > max_attempts:
            log(
                state.paths,
                f"Finalize for Ch{state.chapter_num} exhausted {attempts - 1} attempts "
                f"(max={max_attempts}); force-completing with fallback markers to break resume loop",
            )
            if not extraction_done:
                save_checkpoint(
                    state.paths, state.chapter_num, "extraction.json",
                    _fallback_extraction(state.plan, state.review, state.chapter_num, "finalize attempts exhausted"),
                )
            if not structured_done:
                save_checkpoint(state.paths, state.chapter_num, "structured_state_done.json", {"done": True, "forced": True})
            if not state_file_done:
                save_checkpoint(state.paths, state.chapter_num, "state_file_done.json", {"done": True, "forced": True})
            if not load_checkpoint(state.paths, state.chapter_num, "chapter_completed.json"):
                db_event(state.conn, state.chapter_num, "chapter_completed",
                         {"review": state.review, "plan": state.plan, "decision": state.decision})
                save_checkpoint(state.paths, state.chapter_num, "chapter_completed.json", {"done": True, "forced": True})
        else:
            extraction_local = _run_finalize()
            if not state_file_done:
                if state_in_bg and state.background is not None and not state.resume:
                    state.background.submit(state_file_label, _run_state_file, extraction_local)
                else:
                    _run_state_file(extraction_local)
            if not load_checkpoint(state.paths, state.chapter_num, "chapter_completed.json"):
                db_event(state.conn, state.chapter_num, "chapter_completed",
                         {"review": state.review, "plan": state.plan, "decision": state.decision})
                save_checkpoint(state.paths, state.chapter_num, "chapter_completed.json", {"done": True})

    if bool(state.config["novel"].get("fingerprint_enabled", True)):
        try:
            from quality import store_chapter_fingerprint
            store_chapter_fingerprint(state.conn, state.chapter_num, state.plan)
        except Exception:
            pass
    log(state.paths, f"Saved and indexed Ch{state.chapter_num}")

    hits, misses = cacheable_prefix_hit_rate()
    total = hits + misses
    if total:
        hit_rate = hits / total * 100.0
        log(state.paths, f"Prompt prefix cache: hits={hits} misses={misses} hit_rate={hit_rate:.1f}%")


def _stage_post_chapter(state: ChapterState) -> None:
    """Schedule post-chapter background tasks: telemetry, reviews, memory compress, replan, prefetch."""
    run_stage_review = state.chapter_num % int(state.config["novel"]["stage_review_every"]) == 0
    run_replan = run_stage_review and state.chapter_num >= 40

    def _do_stage_review() -> None:
        stage_review(state.client, state.paths, state.conn, state.config, state.chapter_num)
        log(state.paths, f"Completed stage review Ch{state.chapter_num}")

    def _do_horizon_review() -> None:
        horizon_review(state.client, state.paths, state.conn, state.config, state.chapter_num, state.chapter)
        log(state.paths, f"Completed horizon review Ch{state.chapter_num}")

    def _do_memory_compress() -> None:
        log(state.paths, f"Compressing memory files at Ch{state.chapter_num}")
        compress_all_memory(state.client, state.paths, state.config)

    def _do_replan() -> None:
        if should_replan(state.conn, state.config):
            log(state.paths, f"Triggering adaptive replan at Ch{state.chapter_num}")
            adaptive_replan(state.client, state.paths, state.conn, state.config, state.chapter_num)

    _telemetry_novel = state.paths.logs_dir.parent.name
    _telemetry_genre = str(state.config["novel"].get("genre", "_default") or "_default")

    def _do_telemetry() -> None:
        try:
            sh = (state.review.get("style_health") or {}) if isinstance(state.review, dict) else {}
            sh_metrics = sh.get("metrics") or {}
            metrics_row = {
                "title": state.plan.get("title") if isinstance(state.plan, dict) else None,
                "score": safe_score(state.review.get("score", 0)),
                "readthrough_score": safe_score(state.review.get("readthrough_score", 0)),
                "hook_score": safe_score(state.review.get("hook_score", state.review.get("hook_strength", 0))),
                "payoff_score": safe_score(state.review.get("payoff_score", 0)),
                "novelty_score": safe_score(state.review.get("novelty_score", 0)),
                "prose_score": safe_score(state.review.get("prose_score", state.review.get("aesthetic_score", 0))),
                "continuity_score": safe_score(state.review.get("continuity_score", 0)),
                "hook_strength": safe_score(state.review.get("hook_strength", 0)),
                "accepted": 1 if state.review.get("accepted") else 0,
                "em_dash_per_kchar": sh_metrics.get("em_dash_per_kchar"),
                "style_penalty": sh.get("penalty"),
                "avg_sentence_chars": sh_metrics.get("avg_sentence_chars"),
                "dialogue_char_ratio": sh_metrics.get("dialogue_char_ratio"),
                "tech_per_kchar": sh_metrics.get("tech_per_kchar"),
            }
            telemetry.record_chapter_metrics(_telemetry_novel, _telemetry_genre, state.chapter_num, metrics_row)
            for pair in state.telemetry_revise_pairs:
                telemetry.record_revise_pair(
                    _telemetry_novel, _telemetry_genre, state.chapter_num,
                    pair["round"], pair["text_before"], pair["review"],
                    pair["text_after"], pair["score_before"], pair["score_after"],
                )
            log(state.paths, f"Telemetry recorded Ch{state.chapter_num} (revise_pairs={len(state.telemetry_revise_pairs)})")
        except Exception as exc:
            log(state.paths, f"Telemetry record failed (non-fatal) Ch{state.chapter_num}: {exc}")

    telemetry_on = bool(state.config["novel"].get("telemetry_enabled", True))

    if state.background is not None:
        if telemetry_on:
            state.background.submit(f"telemetry_ch{state.chapter_num}", _do_telemetry)
        if run_stage_review:
            state.background.submit(f"stage_review_ch{state.chapter_num}", _do_stage_review)
        cold_every = int(state.config["novel"].get("cold_reader_every", 10))
        if cold_every > 0 and state.chapter_num % cold_every == 0:
            state.background.submit(f"horizon_review_ch{state.chapter_num}", _do_horizon_review)
        if should_compress_memory(state.paths, state.config, state.chapter_num):
            state.background.submit(f"memory_compress_ch{state.chapter_num}", _do_memory_compress)
        if run_replan:
            state.background.submit(f"adaptive_replan_ch{state.chapter_num}", _do_replan)

        if bool(state.config["novel"].get("prefetch_next_plan", False)):
            horizon = max(1, int(state.config["novel"].get("prefetch_plan_horizon", 1)))
            target_chars = int(state.config["novel"].get("target_words", 0) or 0)
            max_chapters = int(state.config["novel"].get("max_chapters", 0) or 0)
            next_num = state.chapter_num + 1
            should_prefetch = True
            if target_chars and book_reached_target(state.paths.book, target_chars):
                should_prefetch = False
                log(state.paths, f"Prefetch skipped after Ch{state.chapter_num}: target_chars reached")
            if max_chapters and next_num > max_chapters:
                should_prefetch = False
                log(state.paths, f"Prefetch skipped after Ch{state.chapter_num}: next chapter exceeds max_chapters={max_chapters}")

            finalize_label = f"chapter_finalize_ch{state.chapter_num}"
            needs_finalize = not (
                bool(load_checkpoint(state.paths, state.chapter_num, "extraction.json"))
                and bool(load_checkpoint(state.paths, state.chapter_num, "structured_state_done.json"))
                and bool(load_checkpoint(state.paths, state.chapter_num, "chapter_completed.json"))
            )

            def _do_prefetch_horizon() -> None:
                if needs_finalize:
                    state.background.wait_label(finalize_label)
                for offset in range(1, horizon + 1):
                    target_num = state.chapter_num + offset
                    if max_chapters and target_num > max_chapters:
                        log(state.paths, f"Prefetch horizon stopped at Ch{target_num}: exceeds max_chapters={max_chapters}")
                        break
                    if load_checkpoint(state.paths, target_num, "validated_plan.json"):
                        log(state.paths, f"Prefetch skipped for Ch{target_num}: validated_plan.json already exists")
                        continue
                    try:
                        next_tail = tail_text(state.paths.book, int(state.config["novel"]["recent_tail_chars"]))
                        next_memory = memory_context(
                            state.paths, state.conn, state.config,
                            max_chars=int(state.config["novel"].get("plan_memory_chars", 60000) or 0),
                        )
                        next_plan, next_decision = create_plan(
                            state.client, state.paths, state.conn, state.config,
                            target_num, next_tail, cached_memory=next_memory,
                        )
                        save_checkpoint(
                            state.paths, target_num, "validated_plan.json",
                            {"plan": next_plan, "decision": next_decision},
                        )
                        log(state.paths, f"Prefetched plan for Ch{target_num} title={next_plan.get('title', '')!r}")
                    except Exception as exc:
                        log(state.paths, f"Prefetch plan Ch{target_num} failed (non-fatal): {exc}")
                        break

            if should_prefetch:
                state.background.submit(f"prefetch_plan_ch{next_num}", _do_prefetch_horizon)

        state.background.prune_done()
    else:
        if telemetry_on:
            _do_telemetry()
        if run_stage_review:
            _do_stage_review()
        cold_every = int(state.config["novel"].get("cold_reader_every", 10))
        if cold_every > 0 and state.chapter_num % cold_every == 0:
            _do_horizon_review()
        if should_compress_memory(state.paths, state.config, state.chapter_num):
            _do_memory_compress()
        if run_replan:
            _do_replan()


def generate_one_chapter(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    background: BackgroundTasks | None = None,
    resume: bool = False,
) -> None:
    state = ChapterState(
        client=client, paths=paths, conn=conn, config=config,
        chapter_num=chapter_num, background=background, resume=resume,
    )
    _stage_setup_barriers(state)
    _stage_plan(state)
    _stage_write(state)

    threshold = float(config["novel"]["quality_threshold"])
    final_review = load_checkpoint(paths, chapter_num, "final_review.json")
    final_is_authoritative = (
        isinstance(final_review, dict)
        and (
            (safe_score(final_review.get("score", 0)) >= threshold and final_review.get("accepted", True))
            or bool(final_review.get("force_accepted"))
        )
    )
    if final_is_authoritative:
        state.review = final_review
        log(
            paths,
            f"Resuming final review Ch{chapter_num} score={state.review.get('score')}/10"
            f"{' (force-accepted)' if final_review.get('force_accepted') else ''}",
        )
    else:
        if isinstance(final_review, dict):
            log(paths, f"Ignoring low final review Ch{chapter_num} score={final_review.get('score')}/10 threshold={threshold}")
        state.review = {"score": 0, "accepted": False}
        state.best_chapter = state.chapter
        state.best_review = state.review
        _stage_review_revise(state)
        _stage_quality_replan(state)
        _stage_force_accept(state)
        _stage_hook_revise(state)
        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, state.chapter)
        save_checkpoint(paths, chapter_num, "final_review.json", state.review)

    _stage_save(state)
    _stage_finalize(state)
    _stage_post_chapter(state)


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
