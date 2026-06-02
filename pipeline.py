from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from checkpoint import CHAPTER_CURRENT_CHECKPOINT, load_checkpoint, save_checkpoint, should_resume_existing_chapter
from config import (
    Paths,
    chapter_path,
    configured_api_endpoints,
    count_chars,
    ensure_project,
    find_last_chapter,
    get_paths,
    load_config,
    log,
    normalize_chapter,
    read_text,
    rebuild_book,
    safe_score,
    tail_text,
)
from llm import LLMClientPool
from memory import bootstrap, cacheable_prefix, cacheable_prefix_hit_rate, compress_all_memory, memory_context, should_compress_memory, writing_memory_context
from planning import create_plan
from review import adaptive_replan, review_chapter, should_replan, stage_review
from store import db_event, init_db, validate_plan_continuity
from writing import extract_events, revise_chapter, save_chapter, update_state_file, update_structured_state, write_chapter


class BackgroundTasks:
    """Run finalization/stage-review/memory-compress tasks off the critical path.

    Tasks are submitted with a label and kept in a list. The pipeline waits for
    completion only at shutdown or when a follow-up task explicitly depends on
    them. Exceptions are logged but do not crash the main loop.
    """

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
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
            self.tasks = [(l, t, e) for (l, t, e) in self.tasks if t.is_alive()]


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
) -> tuple[str, dict[str, Any] | None]:
    """Generate N candidate chapter drafts in parallel, review each, keep the best.

    Returns (best_chapter_text, best_review_or_None). When N <= 1 falls back to single
    write_chapter with no review (review_round0 will be produced by the normal loop).
    """
    n = int(config["novel"].get("candidate_chapters", 1))
    if n <= 1:
        text = write_chapter(client, paths, conn, config, chapter_num, plan, decision, tail, cached_memory=cached_memory)
        return text, None

    base_temp = float(config["api"]["temperature"])
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

    if len(valid) == 1:
        idx, text = valid[0]
        log(paths, f"Only 1/{n} valid draft for Ch{chapter_num} idx={idx}; skipping comparative review")
        return text, None

    def review_one(item: tuple[int, str]) -> tuple[int, str, dict[str, Any]]:
        idx, text = item
        try:
            report = review_chapter(client, paths, conn, config, chapter_num, plan, text, tail, cached_memory=cached_memory)
        except Exception as exc:
            log(paths, f"Candidate draft review idx={idx} failed: {exc}")
            report = {"score": 0, "accepted": False}
        return idx, text, report

    reviewed: list[tuple[int, str, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(valid))) as executor:
        futures = [executor.submit(review_one, item) for item in valid]
        for future in as_completed(futures):
            reviewed.append(future.result())

    reviewed.sort(key=lambda r: safe_score(r[2].get("score", 0)), reverse=True)
    best_idx, best_text, best_review = reviewed[0]
    scores = [(idx, safe_score(rep.get("score", 0))) for idx, _, rep in reviewed]
    log(paths, f"Selected Ch{chapter_num} draft idx={best_idx} score={best_review.get('score')}/10 candidates={scores}")
    return best_text, best_review


def generate_one_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    background: BackgroundTasks | None = None,
) -> None:
    if background is not None and chapter_num > 1:
        # Block until the previous chapter's extract + structured-state writes
        # are durable, so memory_context sees current metrics/threads/entities.
        background.wait_label(f"chapter_finalize_ch{chapter_num - 1}")
        if bool(config["novel"].get("prefetch_next_plan", False)):
            # If a prefetch task ran for this chapter, ensure its checkpoint is
            # flushed before create_plan tries to resume from it.
            background.wait_label(f"prefetch_plan_ch{chapter_num}")
    tail = tail_text(paths.book, int(config["novel"]["recent_tail_chars"]))
    cached_memory = memory_context(paths, conn, config)
    # Smaller context for write/revise/review hot path to reduce prefill time.
    writing_memory = writing_memory_context(paths, conn, config)
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
            client, paths, conn, config, chapter_num, plan, decision, tail, cached_memory=writing_memory
        )
        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
        if candidate_review is not None:
            save_checkpoint(paths, chapter_num, "review_round0.json", candidate_review)

    threshold = float(config["novel"]["quality_threshold"])
    max_rounds = int(config["novel"]["max_revision_rounds"])
    final_review = load_checkpoint(paths, chapter_num, "final_review.json")
    if (
        isinstance(final_review, dict)
        and safe_score(final_review.get("score", 0)) >= threshold
        and final_review.get("accepted", True)
    ):
        review = final_review
        log(paths, f"Resuming final review Ch{chapter_num} score={review.get('score')}/10")
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
        for round_num in range(max_rounds + 1):
            if round_num > 0:
                revised_key = f"chapter_revised_round{round_num}.md"
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
                review = review_chapter(client, paths, conn, config, chapter_num, plan, chapter, tail, cached_memory=writing_memory)
                save_checkpoint(paths, chapter_num, review_key, review)
                log(paths, f"Reviewed Ch{chapter_num} round={round_num} score={review.get('score')}/10")
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

        if safe_score(review.get("score", 0)) < threshold or not review.get("accepted", True):
            chapter = best_chapter
            review = best_review
            log(
                paths,
                f"Ch{chapter_num} did not meet threshold {threshold} after {max_rounds + 1} rounds "
                f"(best score={review.get('score')}/10). Accepting anyway to avoid pipeline halt.",
            )
            review["accepted"] = True

        # Hook-only mini revise: if the chapter ends weakly, rewrite only the
        # last ~400 chars rather than running another full revise round. This
        # is a single small LLM call gated by hook_strength threshold.
        hook_min = float(config["novel"].get("hook_strength_min", 6.0))
        hook_revise_enabled = bool(config["novel"].get("hook_revise_enabled", True))
        hook_strength = safe_score(review.get("hook_strength", hook_min))
        if (
            hook_revise_enabled
            and hook_strength > 0
            and hook_strength < hook_min
            and not load_checkpoint(paths, chapter_num, "hook_revised.json")
        ):
            try:
                from writing import revise_hook_only as _revise_hook_only
                log(
                    paths,
                    f"Hook-only mini-revise Ch{chapter_num} hook_strength={hook_strength}/10 < {hook_min}",
                )
                new_chapter = _revise_hook_only(
                    client, paths, config, chapter, plan, review,
                    tail_to_revise_chars=int(config["novel"].get("hook_revise_tail_chars", 400)),
                )
                if len(new_chapter.strip()) >= max(500, int(len(chapter) * 0.85)):
                    chapter = new_chapter
                    save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
                    save_checkpoint(paths, chapter_num, "hook_revised.json", {"done": True, "hook_strength_before": hook_strength})
                    # Bump the hook field; full re-review skipped for speed.
                    review["hook_strength"] = max(hook_strength, hook_min)
                else:
                    log(paths, f"Hook revise produced too-short output ({len(new_chapter)} chars); keeping original")
            except Exception as exc:
                log(paths, f"Hook revise failed (non-fatal) Ch{chapter_num}: {exc}")

        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
        save_checkpoint(paths, chapter_num, "final_review.json", review)

    if not load_checkpoint(paths, chapter_num, "chapter_saved.json"):
        if chapter_path(paths, chapter_num).exists():
            log(paths, f"Chapter file already exists Ch{chapter_num}; skipping duplicate save")
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
            extraction_local = extract_events(
                client, paths, conn, config, chapter_num, chapter, cached_memory=cached_memory
            )
            save_checkpoint(paths, chapter_num, "extraction.json", extraction_local)
        else:
            extraction_local = load_checkpoint(paths, chapter_num, "extraction.json") or {}
        if not structured_done:
            update_structured_state(paths, conn, chapter_num, extraction_local, review, decision)
            save_checkpoint(paths, chapter_num, "structured_state_done.json", {"done": True})
        return extraction_local

    def _run_state_file(extraction_local: dict[str, Any]) -> None:
        update_state_file(client, paths, conn, config, chapter_num, chapter, extraction_local)
        save_checkpoint(paths, chapter_num, "state_file_done.json", {"done": True})

    finalize_label = f"chapter_finalize_ch{chapter_num}"
    state_file_label = f"state_file_ch{chapter_num}"

    needs_finalize = not (extraction_done and structured_done and completed_done)
    if extract_in_bg and background is not None and needs_finalize:
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
        extraction_local = _run_finalize()
        if not state_file_done:
            if state_in_bg and background is not None:
                background.submit(state_file_label, _run_state_file, extraction_local)
            else:
                _run_state_file(extraction_local)
        if not load_checkpoint(paths, chapter_num, "chapter_completed.json"):
            db_event(conn, chapter_num, "chapter_completed", {"review": review, "plan": plan, "decision": decision})
            save_checkpoint(paths, chapter_num, "chapter_completed.json", {"done": True})
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

    def _do_memory_compress() -> None:
        log(paths, f"Compressing memory files at Ch{chapter_num}")
        compress_all_memory(client, paths, config)

    def _do_replan() -> None:
        if should_replan(conn, config):
            log(paths, f"Triggering adaptive replan at Ch{chapter_num}")
            adaptive_replan(client, paths, conn, config, chapter_num)

    if background is not None:
        if run_stage_review:
            background.submit(f"stage_review_ch{chapter_num}", _do_stage_review)
        if should_compress_memory(paths, config, chapter_num):
            background.submit(f"memory_compress_ch{chapter_num}", _do_memory_compress)
        if run_replan:
            background.submit(f"adaptive_replan_ch{chapter_num}", _do_replan)

        # Prefetch the next N chapters' plans so the main loop's planning
        # phase resumes from a cached validated_plan.json. Gate each prefetch
        # on the previous one's finalize barrier so each sees fresh state.
        if bool(config["novel"].get("prefetch_next_plan", False)):
            horizon = max(1, int(config["novel"].get("prefetch_plan_horizon", 1)))

            def _do_prefetch_horizon() -> None:
                if needs_finalize:
                    background.wait_label(finalize_label)
                # Prefetch sequentially within this background task so the
                # second prefetch sees the first one's checkpoint (and any
                # incremental state derived from it). Each iteration only
                # uses the snapshot of metrics/threads currently durable.
                for offset in range(1, horizon + 1):
                    target_num = chapter_num + offset
                    if load_checkpoint(paths, target_num, "validated_plan.json"):
                        log(paths, f"Prefetch skipped for Ch{target_num}: validated_plan.json already exists")
                        continue
                    try:
                        next_tail = tail_text(paths.book, int(config["novel"]["recent_tail_chars"]))
                        next_memory = memory_context(paths, conn, config)
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
            next_num = chapter_num + 1
            background.submit(f"prefetch_plan_ch{next_num}", _do_prefetch_horizon)

        background.prune_done()
    else:
        if run_stage_review:
            _do_stage_review()
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

    api_endpoints, primary_endpoint_count = configured_api_endpoints(config)
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
    clients = [
        OpenAI(base_url=base_url, api_key=api_key, timeout=httpx_timeout)
        for base_url, api_key in api_endpoints
    ]
    client: Any = (
        LLMClientPool(clients, primary_endpoint_count, endpoints=api_endpoints, log_fn=lambda msg: log(paths, msg))
        if len(clients) > 1
        else clients[0]
    )
    log(paths, f"LLM client pool initialized keys={len(clients)} primary={primary_endpoint_count}")

    if not paths.state.exists() or not read_text(paths.state).strip():
        bootstrap(client, paths, conn, config)

    if not paths.book.exists() and find_last_chapter(paths) > 0:
        rebuild_book(paths)

    target = int(config["novel"]["target_words"])
    # Optional hard cap on chapter count (short-novel mode). 0/absent => no cap,
    # so the long novel (which never sets this) keeps its char-target-only loop.
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    log(paths, f"Start target_chars={target} current_chars={count_chars(paths.book)} max_chapters={max_chapters or 'none'}")
    background = BackgroundTasks(paths)
    try:
        while count_chars(paths.book) < target:
            last_chapter = find_last_chapter(paths)
            if max_chapters and last_chapter >= max_chapters:
                log(paths, f"Reached max_chapters={max_chapters}; stopping chapter loop")
                break
            if should_resume_existing_chapter(paths, last_chapter):
                chapter_num = last_chapter
                log(paths, f"Resuming partially indexed Ch{chapter_num}")
            else:
                chapter_num = last_chapter + 1
            generate_one_chapter(client, paths, conn, config, chapter_num, background=background)
            total = count_chars(paths.book)
            log(paths, f"Progress chars={total}/{target} pct={total / target * 100:.2f}%")
    finally:
        log(paths, "Waiting for background tasks to finish before exit...")
        background.wait_pending()

    log(paths, f"Done total_chars={count_chars(paths.book)}")

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
