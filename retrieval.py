"""Lightweight retrieval over written chapters (dependency-free RAG).

The layered `memory_context` compresses old chapters into fuzzy summaries, so by
chapter 300 the model can no longer quote a specific name/number/scene from
chapter 47. This module keeps an on-disk inverted index of every saved chapter
(split into ~600-char passages) and retrieves the top-k passages most relevant
to the current chapter's plan, to be injected as "## 相关历史原文（检索）".

Why not embeddings? The project's only dependency is `openai`. Calling an
embedding endpoint per passage for a 500-chapter book is slow and may not be
available on the configured endpoint. A character-ngram TF-IDF cosine ranking is
fast, fully local, and works well for Chinese exact-fact recall (names, numbers,
place tokens) — which is precisely the recall the summaries lose.

Index lives at logs/retrieval_index.json. It is incrementally updated: each call
to `index_chapter` appends one chapter's passages. `retrieve` loads the index
lazily and caches it per-process keyed by file mtime.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from config import Paths, log, read_text

_INDEX_NAME = "retrieval_index.json"
_INDEX_DIR_NAME = "retrieval_index"
_DF_NAME = "_df.json"
_PASSAGE_CHARS = 600
_NGRAM = 2  # character bigrams — robust for Chinese without segmentation

# Per-process cache: (signature, parsed_index). The signature is a cheap proxy
# for "has anything changed": for the legacy monolithic file it is the file
# mtime; for the sharded layout it is (shard-dir mtime, _df.json mtime).
_INDEX_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}


def _index_path(paths: Paths) -> Path:
    return paths.logs_dir / _INDEX_NAME


def _shard_dir(paths: Paths) -> Path:
    return paths.logs_dir / _INDEX_DIR_NAME


def _shard_path(paths: Paths, chapter_num: int) -> Path:
    return _shard_dir(paths) / f"ch{chapter_num:04d}.json"


def _df_path(paths: Paths) -> Path:
    return _shard_dir(paths) / _DF_NAME


def _tokenize(text: str) -> list[str]:
    """Character n-grams over CJK + lowercased ASCII words."""
    # Keep CJK chars and ASCII alnum; drop punctuation/whitespace.
    cleaned = re.sub(r"[^一-鿿A-Za-z0-9]", "", text)
    if not cleaned:
        return []
    grams = [cleaned[i : i + _NGRAM] for i in range(len(cleaned) - _NGRAM + 1)]
    return grams or [cleaned]


def _split_passages(text: str, size: int = _PASSAGE_CHARS) -> list[str]:
    body = text.strip()
    if not body:
        return []
    # Split on blank lines first, then re-pack into ~size windows so we don't cut
    # mid-sentence too aggressively.
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    passages: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p) if buf else p
        else:
            if buf:
                passages.append(buf)
            if len(p) <= size:
                buf = p
            else:
                # Hard-wrap an over-long paragraph.
                for i in range(0, len(p), size):
                    passages.append(p[i : i + size])
                buf = ""
    if buf:
        passages.append(buf)
    return passages


def _passages_for_chapter(chapter_num: int, chapter_text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build this chapter's passage records and its per-chapter df contribution.

    The df contribution counts each term ONCE per passage (document frequency),
    matching the original monolithic index build (retrieval.py legacy: df[t]+=1
    per passage). Returned df is the increment to fold into the aggregate.
    """
    passages: list[dict[str, Any]] = []
    df_inc: dict[str, int] = {}
    for idx, ptext in enumerate(_split_passages(chapter_text)):
        toks = _tokenize(ptext)
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df_inc[t] = df_inc.get(t, 0) + 1
        passages.append(
            {"chapter": chapter_num, "i": idx, "text": ptext, "tf": tf, "len": len(toks)}
        )
    return passages, df_inc


def index_chapter(paths: Paths, chapter_num: int, chapter_text: str) -> None:
    """Append one chapter's passages to the on-disk index. Idempotent per chapter.

    Sharded layout (avoids the legacy O(n^2) whole-file rewrite): each chapter is
    written to logs/retrieval_index/ch{NNNN}.json (its own small file), and an
    aggregate logs/retrieval_index/_df.json holds {df, n_docs, chapters}. Writing
    a chapter is O(1) for the shard plus an O(vocab) rewrite of the small df file
    — no re-serialization of the entire growing passage corpus.
    """
    shard_dir = _shard_dir(paths)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # One-time migration: if a legacy monolithic index exists but no shards yet,
    # explode it into per-chapter shards + df so old books keep their index.
    df_path = _df_path(paths)
    legacy = _index_path(paths)
    if not df_path.exists() and legacy.exists():
        try:
            _migrate_monolithic_to_shards(paths)
        except Exception:
            pass

    # Idempotency: a shard for this chapter already means it's indexed.
    if _shard_path(paths, chapter_num).exists():
        return

    # Load aggregate df (small).
    try:
        agg = json.loads(read_text(df_path) or "{}") if df_path.exists() else {}
    except Exception:
        agg = {}
    df: dict[str, int] = agg.get("df", {})
    indexed_chapters = set(agg.get("chapters", []))
    n_docs = int(agg.get("n_docs", 0))
    if chapter_num in indexed_chapters:
        return

    passages, df_inc = _passages_for_chapter(chapter_num, chapter_text)
    # Write the per-chapter shard first (so a crash leaves df un-double-counted;
    # df is only advanced after the shard is durable).
    shard_payload = {"chapter": chapter_num, "passages": passages}
    _shard_path(paths, chapter_num).write_text(
        json.dumps(shard_payload, ensure_ascii=False), encoding="utf-8"
    )

    for t, c in df_inc.items():
        df[t] = df.get(t, 0) + c
    indexed_chapters.add(chapter_num)
    n_docs += len(passages)
    df_path.write_text(
        json.dumps(
            {"df": df, "n_docs": n_docs, "chapters": sorted(indexed_chapters)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _INDEX_CACHE.pop(str(shard_dir), None)


def _migrate_monolithic_to_shards(paths: Paths) -> None:
    """Explode a legacy logs/retrieval_index.json into the sharded layout."""
    legacy = _index_path(paths)
    try:
        data = json.loads(read_text(legacy) or "{}")
    except Exception:
        return
    passages: list[dict[str, Any]] = data.get("passages", [])
    if not passages:
        return
    shard_dir = _shard_dir(paths)
    shard_dir.mkdir(parents=True, exist_ok=True)
    by_chapter: dict[int, list[dict[str, Any]]] = {}
    for p in passages:
        ch = int(p.get("chapter", 0))
        by_chapter.setdefault(ch, []).append(p)
    for ch, plist in by_chapter.items():
        sp = _shard_path(paths, ch)
        if not sp.exists():
            sp.write_text(
                json.dumps({"chapter": ch, "passages": plist}, ensure_ascii=False),
                encoding="utf-8",
            )
    df_path = _df_path(paths)
    if not df_path.exists():
        df_path.write_text(
            json.dumps(
                {
                    "df": data.get("df", {}),
                    "n_docs": int(data.get("n_docs", len(passages))),
                    "chapters": sorted({int(c) for c in by_chapter}),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def bool_enabled(paths: Paths) -> bool:  # noqa: D401 - tiny helper
    return True


def _load_index(paths: Paths) -> dict[str, Any] | None:
    """Return the merged index dict {passages, df, chapters, n_docs} or None.

    Prefers the sharded layout (logs/retrieval_index/ch*.json + _df.json), merging
    all shards into the SAME structure the legacy monolithic file produced so that
    `retrieve`, `candidate_new_entities`, and `backfill_index` are unchanged.
    Falls back to the legacy monolithic logs/retrieval_index.json when no shards
    exist. Cached per-process keyed by a cheap change-signature.
    """
    shard_dir = _shard_dir(paths)
    df_path = _df_path(paths)
    if shard_dir.exists() and df_path.exists():
        try:
            sig: Any = (shard_dir.stat().st_mtime, df_path.stat().st_mtime)
        except OSError:
            sig = None
        cache_key = str(shard_dir)
        cached = _INDEX_CACHE.get(cache_key)
        if cached and sig is not None and cached[0] == sig:
            return cached[1]
        try:
            agg = json.loads(df_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            agg = {}
        passages: list[dict[str, Any]] = []
        for sp in sorted(shard_dir.glob("ch*.json")):
            try:
                shard = json.loads(sp.read_text(encoding="utf-8") or "{}")
            except Exception:
                continue
            passages.extend(shard.get("passages", []))
        data = {
            "passages": passages,
            "df": agg.get("df", {}),
            "chapters": agg.get("chapters", []),
            "n_docs": int(agg.get("n_docs", len(passages))),
        }
        if sig is not None:
            _INDEX_CACHE[cache_key] = (sig, data)
        return data

    # Legacy monolithic fallback.
    path = _index_path(paths)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _INDEX_CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return None
    _INDEX_CACHE[str(path)] = (mtime, data)
    return data


def retrieve(
    paths: Paths,
    query: str,
    top_k: int = 4,
    exclude_recent_chapters: int = 6,
    current_chapter: int | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Return top-k passages most relevant to `query`.

    Recent chapters are excluded (they're already in the tail/memory window); the
    value of retrieval is surfacing OLDER, summarized-away facts.

    `min_score` is an absolute cosine floor: passages below it are dropped even if
    they would fill the top_k. This prevents a generic query from anchoring the
    writer to weakly-related noise it is told to "stay consistent with" — better to
    return fewer (or zero) hits than to surface irrelevant passages as facts.
    """
    data = _load_index(paths)
    if not data:
        return []
    passages: list[dict[str, Any]] = data.get("passages", [])
    df: dict[str, int] = data.get("df", {})
    n_docs = max(1, int(data.get("n_docs", len(passages))))
    if not passages:
        return []

    q_toks = _tokenize(query)
    if not q_toks:
        return []
    q_tf: dict[str, int] = {}
    for t in q_toks:
        q_tf[t] = q_tf.get(t, 0) + 1

    def idf(term: str) -> float:
        return math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0

    q_vec = {t: (q_tf[t] * idf(t)) for t in q_tf}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    cutoff = None
    if current_chapter is not None and exclude_recent_chapters > 0:
        cutoff = current_chapter - exclude_recent_chapters

    scored: list[tuple[float, dict[str, Any]]] = []
    for p in passages:
        if cutoff is not None and int(p.get("chapter", 0)) > cutoff:
            continue
        tf = p.get("tf", {})
        if not tf:
            continue
        dot = 0.0
        d_norm_sq = 0.0
        for t, c in tf.items():
            w = c * idf(t)
            d_norm_sq += w * w
            if t in q_vec:
                dot += w * q_vec[t]
        if dot <= 0:
            continue
        d_norm = math.sqrt(d_norm_sq) or 1.0
        score = dot / (q_norm * d_norm)
        if score < min_score:
            continue
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, p in scored:
        ch = int(p.get("chapter", 0))
        # Cap to at most 2 passages per chapter for diversity.
        if sum(1 for o in out if o["chapter"] == ch) >= 2:
            continue
        out.append({"chapter": ch, "score": round(score, 4), "text": p.get("text", "")})
        if len(out) >= top_k:
            break
    return out


def retrieval_block(
    paths: Paths,
    config: dict[str, Any],
    plan: dict[str, Any],
    chapter_num: int,
) -> str:
    """Build a ready-to-inject '相关历史原文' block from the plan, or '' if disabled/empty."""
    if not bool(config["novel"].get("rag_enabled", True)):
        return ""
    top_k = int(config["novel"].get("rag_top_k", 4))
    exclude = int(config["novel"].get("rag_exclude_recent", 6))
    min_score = float(config["novel"].get("rag_min_score", 0.04))
    # Build a query from the plan's most concrete fields. The recall TARGET is
    # exact facts (names/numbers/places), so concrete fields (location, info_source,
    # character names) must dominate over abstract intent (goal/conflict). Entity
    # names from character_focus are repeated (weight x2) so the TF-IDF query vector
    # is anchored on the entities whose continuity we actually need to protect.
    parts: list[str] = []
    for key in ("title", "goal", "conflict", "payoff", "pressure", "location", "info_source"):
        v = plan.get(key)
        if v:
            parts.append(str(v))
    focus = plan.get("character_focus")
    if isinstance(focus, list):
        names = [str(x) for x in focus[:6]]
        parts.extend(names)
        parts.extend(names)  # repeat: weight entity names higher in the query vector
    for key in ("thread_actions", "beats"):
        v = plan.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v[:6])
    query = " ".join(parts).strip()
    if not query:
        return ""
    try:
        hits = retrieve(
            paths,
            query,
            top_k=top_k,
            exclude_recent_chapters=exclude,
            current_chapter=chapter_num,
            min_score=min_score,
        )
    except Exception as exc:
        log(paths, f"RAG retrieve failed (non-fatal) Ch{chapter_num}: {exc}")
        return ""
    if not hits:
        return ""
    lines = [
        "## 相关历史原文（检索自早期章节）",
        "用途：仅供事实核对，确保人名/数字/地点/称谓与既成事实一致，不得与之矛盾。禁止照抄下列措辞或句式，用你自己的行文重写。",
    ]
    for h in hits:
        snippet = h["text"].strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        # Keep paragraph breaks (indent continuation lines) so multi-scene
        # excerpts are not flattened into a single run-on line that nudges the
        # writer toward telegraphic prose.
        snippet = snippet.replace("\n", "\n    ")
        lines.append(f"- [Ch{h['chapter']}]\n    {snippet}")
    return "\n".join(lines)


def exemplar_block(
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    plan: dict[str, Any],
    chapter_num: int,
) -> str:
    """P0-3: Build a '黄金范例' block from top-scoring chapters matching the plan's type.

    Selects chapters with score >= threshold from chapter_metrics, filters by
    payoff_type/conflict_type match when available, retrieves text snippets matching
    the plan's concrete fields, and formats them as positive exemplars with their
    scores and what made them succeed.

    This block MUST stay in the variable (non-cacheable) prompt segment — it's
    injected alongside retrieval_block in write_chapter's user message, never in
    cacheable_prefix, so changing selection logic doesn't invalidate the cache.
    """
    if not bool(config["novel"].get("exemplar_rag_enabled", True)):
        return ""
    if chapter_num <= int(config["novel"].get("exemplar_rag_min_chapter", 8)):
        return ""

    try:
        from store import recent_metrics
        # Get all chapters with metrics
        all_metrics = []
        try:
            # Fetch more chapters to find top performers
            candidates = recent_metrics(conn, limit=min(chapter_num - 1, 100))
            all_metrics = [m for m in candidates if isinstance(m, dict)]
        except Exception:
            pass

        if not all_metrics:
            return ""

        # Filter by score threshold
        threshold = float(config["novel"].get("exemplar_rag_score_min", 8.8))
        high_scorers = [
            m for m in all_metrics
            if m.get("score") is not None and float(m.get("score", 0)) >= threshold
        ]

        if not high_scorers:
            return ""

        # Optional: filter by payoff_type or conflict_type match
        plan_payoff = str(plan.get("payoff_type", "")).strip() if isinstance(plan, dict) else ""
        plan_conflict = str(plan.get("conflict_type", "")).strip() if isinstance(plan, dict) else ""

        matched = []
        for m in high_scorers:
            ch_payoff = str(m.get("payoff_type", "")).strip()
            ch_conflict = str(m.get("conflict_type", "")).strip()
            # Prefer same type, but accept any high scorer if no match
            if plan_payoff and ch_payoff == plan_payoff:
                matched.append((m, 2))  # strong match
            elif plan_conflict and ch_conflict == plan_conflict:
                matched.append((m, 1))  # weak match
            else:
                matched.append((m, 0))  # no type match

        # Sort by match score then by chapter score
        matched.sort(key=lambda x: (x[1], x[0].get("score", 0)), reverse=True)
        top_exemplars = [m for m, _ in matched[:int(config["novel"].get("exemplar_rag_top_k", 3))]]

        if not top_exemplars:
            return ""

        # Build query from plan fields (same as retrieval_block)
        query_parts: list[str] = []
        for key in ("title", "goal", "conflict", "payoff", "pressure"):
            v = plan.get(key)
            if v:
                query_parts.append(str(v))
        query = " ".join(query_parts).strip()

        lines = [
            "## 黄金范例（本书高分章节，供学习节奏与执行手法）",
            "以下章节在终局质量评分中达到高分（≥8.8/10）。参考其节奏把控、beat 落地方式、钩子设计，",
            "但**必须用你自己的措辞和场景重写**，严禁照搬原句或结构。",
        ]

        for ex in top_exemplars:
            ch = ex.get("chapter")
            score = ex.get("score")
            if ch is None:
                continue

            # Read chapter text
            from config import chapter_path
            ch_text = ""
            try:
                ch_path = chapter_path(paths, ch)
                if ch_path.exists():
                    ch_text = ch_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

            if not ch_text:
                continue

            # Extract snippet matching query (use TF-IDF scoring)
            snippet = ""
            if query:
                try:
                    # Simple TF-IDF match: split into sentences, score by query term overlap
                    sentences = re.split(r'[。！？\n]', ch_text)
                    query_terms = set(_tokenize(query))
                    if query_terms:
                        scored = []
                        for sent in sentences:
                            sent = sent.strip()
                            if len(sent) >= 20:
                                sent_terms = set(_tokenize(sent))
                                overlap = len(query_terms & sent_terms)
                                if overlap > 0:
                                    scored.append((overlap, sent))
                        if scored:
                            scored.sort(reverse=True)
                            # Take top 1-2 matching sentences
                            snippet = "".join(s for _, s in scored[:2])
                except Exception:
                    pass

            if not snippet:
                # Fallback: take first ~200 chars
                snippet = ch_text[:200]

            if len(snippet) > 400:
                snippet = snippet[:400] + "…"

            # Format strengths
            strengths = []
            if ex.get("hook_score") and float(ex.get("hook_score", 0)) >= 8.5:
                strengths.append(f"强钩子({ex.get('hook_score')}/10)")
            if ex.get("payoff_score") and float(ex.get("payoff_score", 0)) >= 8.5:
                strengths.append(f"高兑现({ex.get('payoff_score')}/10)")
            if ex.get("novelty_score") and float(ex.get("novelty_score", 0)) >= 8.0:
                strengths.append(f"新意({ex.get('novelty_score')}/10)")
            strength_text = "、".join(strengths) if strengths else "整体高分"

            lines.append(f"\n- **Ch{ch} 终评 {score}/10** ({strength_text})")
            lines.append(f"  {snippet.replace(chr(10), ' ')[:300]}")

        return "\n".join(lines)

    except Exception as exc:
        from config import log
        log(paths, f"Exemplar RAG failed (non-fatal) Ch{chapter_num}: {exc}")
        return ""


def _extract_focus_names(plan: dict[str, Any]) -> list[str]:
    """Extract character names from plan's focus_characters, character_focus, and beats."""
    names: list[str] = []
    for key in ("focus_characters", "character_focus"):
        raw = plan.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    names.append(item.strip())
                elif isinstance(item, dict):
                    n = str(item.get("name") or item.get("角色") or "").strip()
                    if n:
                        names.append(n)
        elif isinstance(raw, str) and raw.strip():
            names.extend(seg.strip() for seg in raw.split("、") if seg.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def structured_recall_block(
    conn: Any,
    config: dict[str, Any],
    plan: dict[str, Any],
    chapter_num: int,
    max_chars: int | None = None,
) -> str:
    """Build a structured context recall block from DB state for the writer.

    Queries entities, threads, relationships, revelations, and constraints
    relevant to the current chapter's plan. Returns a markdown block or "".
    """
    from store import (
        db_lock,
        entity_state_as_of,
        get_active_constraints,
        get_overdue_reader_promises,
        get_overdue_revelations,
        get_pending_revelations,
        get_relationships,
    )

    novel_cfg = config.get("novel", {})
    if max_chars is None:
        max_chars = int(novel_cfg.get("structured_recall_max_chars", 3000))
    thread_horizon = int(novel_cfg.get("structured_recall_thread_horizon", 5))
    focus_names = _extract_focus_names(plan)

    sections: list[str] = []

    # 1. Entities matching focus characters
    if focus_names:
        entity_lines: list[str] = []
        try:
            with db_lock():
                rows = conn.execute(
                    "SELECT entity_type, name, state_json, updated_chapter FROM entities"
                ).fetchall()
            for row in rows:
                name = row["name"]
                if any(fn in name or name in fn for fn in focus_names):
                    state = row["state_json"]
                    if len(state) > 200:
                        state = state[:200] + "…"
                    entity_lines.append(
                        f"- {name}({row['entity_type']}): {state} (Ch{row['updated_chapter']}更新)"
                    )
        except Exception:
            pass
        if entity_lines:
            sections.append("### 本章相关人物当前状态\n" + "\n".join(entity_lines[:8]))

    # 2. Open threads due within horizon
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT id, description, status, introduced_chapter, due_chapter, updated_chapter
                   FROM open_threads
                   WHERE status IN ('open', 'building')
                   AND due_chapter IS NOT NULL
                   AND due_chapter BETWEEN ? AND ?
                   ORDER BY due_chapter ASC LIMIT 8""",
                (chapter_num - 2, chapter_num + thread_horizon),
            ).fetchall()
        if rows:
            lines = []
            for r in rows:
                due = r["due_chapter"]
                delta = due - chapter_num
                tag = f"过期{-delta}章" if delta < 0 else (f"本章到期" if delta == 0 else f"还剩{delta}章")
                lines.append(f"- {r['id']}: \"{r['description']}\"（引入Ch{r['introduced_chapter']}→到期Ch{due}，{tag}）")
            sections.append("### 即将到期的伏线\n" + "\n".join(lines))
    except Exception:
        pass

    # 3. Relationships involving focus characters
    if focus_names:
        try:
            all_rels = get_relationships(conn, limit=30)
            relevant = [
                r for r in all_rels
                if any(fn in r.get("char_a", "") or fn in r.get("char_b", "") or
                       r.get("char_a", "") in fn or r.get("char_b", "") in fn
                       for fn in focus_names)
            ][:10]
            if relevant:
                lines = []
                for r in relevant:
                    lines.append(
                        f"- {r['char_a']}↔{r['char_b']}: {r.get('stage', '?')}"
                        f"({r.get('intensity', 0):.1f}) — {r.get('last_event', '?')[:60]}"
                    )
                sections.append("### 关键角色关系\n" + "\n".join(lines))
        except Exception:
            pass

    # 4. Pending/overdue revelations
    try:
        pending = get_pending_revelations(conn, chapter_num, limit=6)
        overdue = get_overdue_revelations(conn, chapter_num, grace=3, limit=4)
        rev_ids_shown: set[str] = set()
        lines = []
        for r in overdue:
            rev_ids_shown.add(r["id"])
            lines.append(
                f"- ⚠ {r['id']}: \"{r['description']}\"（过期{r.get('overdue_by', '?')}章，importance={r.get('importance', '?')}）"
            )
        for r in pending:
            if r["id"] not in rev_ids_shown:
                due = r.get("due_chapter")
                due_note = f"到期Ch{due}" if due else "无到期"
                lines.append(
                    f"- {r['id']}: \"{r['description']}\"（{r['status']}，{due_note}，importance={r.get('importance', '?')}）"
                )
        if lines:
            sections.append("### 待揭示信息\n" + "\n".join(lines[:8]))
    except Exception:
        pass

    # 5. Active stage constraints
    try:
        constraints = get_active_constraints(conn, chapter_num)
        if constraints:
            lines = [
                f"- {c['constraint_type']}: {c['description'][:80]}"
                for c in constraints[:6]
            ]
            sections.append("### 活跃约束\n" + "\n".join(lines))
    except Exception:
        pass

    if not sections:
        return ""

    header = "## 结构化回忆（数据库状态快照，写作事实参照）\n"
    body = "\n\n".join(sections)
    result = header + body

    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0] + "\n…[截断]"

    return result


def candidate_new_entities(
    paths: Paths,
    chapter_text: str,
    min_len: int = 2,
    max_len: int = 4,
    df_floor: int = 1,
    limit: int = 12,
) -> list[str]:
    """Deterministically flag proper-noun-like tokens that appear in this chapter
    but are essentially absent from all PRIOR indexed chapters.

    This is the objective counterpart to the LLM-only "hallucinated_entities"
    check. It does NOT prove an entity is hallucinated — a legitimately new
    character introduced this chapter will also show up here — but it gives the
    reviewer a concrete shortlist to verify against established facts instead of
    relying purely on the model noticing.

    Heuristic: extract CJK runs of length [min_len, max_len] that look like names
    (followed/preceded by common name markers, or capitalized ASCII runs), then
    keep only those whose character bigrams have near-zero document frequency in
    the existing index (i.e. the surface form was never seen before).
    """
    data = _load_index(paths)
    if not data:
        return []
    df: dict[str, int] = data.get("df", {})
    if not df:
        return []

    # 1) Harvest candidate surface forms from the chapter.
    candidates: set[str] = set()
    # ASCII proper nouns: Capitalized word runs.
    for m in re.findall(r"[A-Z][A-Za-z]{1,}", chapter_text):
        candidates.add(m)
    # CJK name-like runs: a 2-4 char CJK token immediately preceding a relational/
    # title marker (说/道/问/答/将军/大人/陛下/公子/姑娘/先生) OR following 称号 markers.
    cjk = r"[一-鿿]"
    marker = r"(?:说道|说|道|问道|答道|笑道|大人|将军|陛下|公子|姑娘|先生|殿下|大王|长老|宗主|城主|师兄|师姐|师父)"
    for m in re.findall(rf"({cjk}{{{min_len},{max_len}}}){marker}", chapter_text):
        candidates.add(m)
    # Quoted speaker pattern: 「X」/“X” where X is a short CJK run.
    for m in re.findall(rf"(?:“|「)({cjk}{{{min_len},{max_len}}})(?:”|」)", chapter_text):
        candidates.add(m)

    if not candidates:
        return []

    # 2) Keep only forms whose bigrams are essentially unseen in prior chapters.
    new_entities: list[str] = []
    for surface in candidates:
        grams = _tokenize(surface)
        if not grams:
            continue
        # Max DF across the surface's bigrams: if even the most common bigram is
        # at/below df_floor, the surface form is effectively new to the corpus.
        max_df = max(df.get(g, 0) for g in grams)
        if max_df <= df_floor:
            new_entities.append(surface)
        if len(new_entities) >= limit:
            break
    return new_entities


def backfill_index(paths: Paths, config: dict[str, Any]) -> int:
    """Index any saved chapters not yet in the index. Returns count newly indexed."""
    from config import chapter_path, find_last_chapter

    last = find_last_chapter(paths)
    if last <= 0:
        return 0
    data = _load_index(paths) or {}
    already = set(data.get("chapters", []))
    added = 0
    for n in range(1, last + 1):
        if n in already:
            continue
        text = read_text(chapter_path(paths, n))
        if text.strip():
            index_chapter(paths, n, text)
            added += 1
    return added
