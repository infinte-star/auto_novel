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
) -> list[dict[str, Any]]:
    """Return top-k passages most relevant to `query`.

    Recent chapters are excluded (they're already in the tail/memory window); the
    value of retrieval is surfacing OLDER, summarized-away facts.
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
    # Build a query from the plan's most concrete fields.
    parts: list[str] = []
    for key in ("title", "goal", "conflict", "payoff", "pressure"):
        v = plan.get(key)
        if v:
            parts.append(str(v))
    for key in ("character_focus", "thread_actions", "beats"):
        v = plan.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v[:6])
    query = " ".join(parts).strip()
    if not query:
        return ""
    try:
        hits = retrieve(
            paths, query, top_k=top_k, exclude_recent_chapters=exclude, current_chapter=chapter_num
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
