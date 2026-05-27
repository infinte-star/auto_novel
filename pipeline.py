from __future__ import annotations

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
from memory import bootstrap, compress_all_memory, should_compress_memory
from planning import create_plan
from review import adaptive_replan, review_chapter, should_replan, stage_review
from store import init_db, validate_plan_continuity
from writing import extract_events, revise_chapter, save_chapter, update_state_file, update_structured_state, write_chapter

def generate_one_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> None:
    tail = tail_text(paths.book, int(config["novel"]["recent_tail_chars"]))
    final_payload = load_checkpoint(paths, chapter_num, "validated_plan.json")
    if isinstance(final_payload, dict) and final_payload.get("plan") and final_payload.get("decision"):
        log(paths, f"Resuming validated plan Ch{chapter_num}")
        plan = final_payload["plan"]
        decision = final_payload["decision"]
    else:
        plan, decision = create_plan(client, paths, conn, config, chapter_num, tail)

        # Pre-write continuity validation
        violations = validate_plan_continuity(conn, plan, chapter_num)
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
                )
            else:
                decision.setdefault("required_constraints", []).extend(violations)
        save_checkpoint(paths, chapter_num, "validated_plan.json", {"plan": plan, "decision": decision})

    existing_chapter = read_text(chapter_path(paths, chapter_num))
    chapter = load_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT) or existing_chapter
    if chapter:
        chapter = normalize_chapter(str(chapter))
        log(paths, f"Resuming cached chapter text Ch{chapter_num}")
    else:
        log(paths, f"Writing Ch{chapter_num}: {plan.get('title', '')}")
        chapter = write_chapter(client, paths, conn, config, chapter_num, plan, decision, tail)
        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)

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
                    chapter = revise_chapter(client, paths, config, chapter, review, plan)
                    save_checkpoint(paths, chapter_num, revised_key, chapter)
                save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)

            review_key = f"review_round{round_num}.json"
            cached_review = load_checkpoint(paths, chapter_num, review_key)
            if isinstance(cached_review, dict):
                review = cached_review
                log(paths, f"Resuming cached review Ch{chapter_num} round={round_num} score={review.get('score')}/10")
            else:
                review = review_chapter(client, paths, conn, config, chapter_num, plan, chapter, tail)
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

        save_checkpoint(paths, chapter_num, CHAPTER_CURRENT_CHECKPOINT, chapter)
        save_checkpoint(paths, chapter_num, "final_review.json", review)

    if not load_checkpoint(paths, chapter_num, "chapter_saved.json"):
        if chapter_path(paths, chapter_num).exists():
            log(paths, f"Chapter file already exists Ch{chapter_num}; skipping duplicate save")
            rebuild_book(paths)
        else:
            save_chapter(paths, chapter_num, chapter, review, plan)
        save_checkpoint(paths, chapter_num, "chapter_saved.json", {"saved": True})

    extraction = load_checkpoint(paths, chapter_num, "extraction.json")
    if isinstance(extraction, dict):
        log(paths, f"Resuming cached extraction Ch{chapter_num}")
    else:
        extraction = extract_events(client, paths, conn, config, chapter_num, chapter)
        save_checkpoint(paths, chapter_num, "extraction.json", extraction)

    if not load_checkpoint(paths, chapter_num, "structured_state_done.json"):
        update_structured_state(paths, conn, chapter_num, extraction, review, decision)
        save_checkpoint(paths, chapter_num, "structured_state_done.json", {"done": True})

    if not load_checkpoint(paths, chapter_num, "state_file_done.json"):
        update_state_file(client, paths, conn, config, chapter_num, chapter, extraction)
        save_checkpoint(paths, chapter_num, "state_file_done.json", {"done": True})

    if not load_checkpoint(paths, chapter_num, "chapter_completed.json"):
        db_event(conn, chapter_num, "chapter_completed", {"review": review, "plan": plan, "decision": decision})
        save_checkpoint(paths, chapter_num, "chapter_completed.json", {"done": True})
    log(paths, f"Saved and indexed Ch{chapter_num}")

    if chapter_num % int(config["novel"]["stage_review_every"]) == 0:
        stage_review(client, paths, conn, config, chapter_num)
        log(paths, f"Completed stage review Ch{chapter_num}")

    # Memory compression check
    if should_compress_memory(paths, config, chapter_num):
        log(paths, f"Compressing memory files at Ch{chapter_num}")
        compress_all_memory(client, paths, config)

    # Adaptive replanning check
    if chapter_num % int(config["novel"]["stage_review_every"]) == 0 and chapter_num >= 40:
        if should_replan(conn, config):
            log(paths, f"Triggering adaptive replan at Ch{chapter_num}")
            adaptive_replan(client, paths, conn, config, chapter_num)

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
    clients = [
        OpenAI(base_url=base_url, api_key=api_key)
        for base_url, api_key in api_endpoints
    ]
    client: Any = LLMClientPool(clients, primary_endpoint_count) if len(clients) > 1 else clients[0]
    log(paths, f"LLM client pool initialized keys={len(clients)} primary={primary_endpoint_count}")

    if not paths.state.exists() or not read_text(paths.state).strip():
        bootstrap(client, paths, conn, config)

    if not paths.book.exists() and find_last_chapter(paths) > 0:
        rebuild_book(paths)

    target = int(config["novel"]["target_words"])
    log(paths, f"Start target_chars={target} current_chars={count_chars(paths.book)}")
    while count_chars(paths.book) < target:
        last_chapter = find_last_chapter(paths)
        if should_resume_existing_chapter(paths, last_chapter):
            chapter_num = last_chapter
            log(paths, f"Resuming partially indexed Ch{chapter_num}")
        else:
            chapter_num = last_chapter + 1
        generate_one_chapter(client, paths, conn, config, chapter_num)
        total = count_chars(paths.book)
        log(paths, f"Progress chars={total}/{target} pct={total / target * 100:.2f}%")

    log(paths, f"Done total_chars={count_chars(paths.book)}")
