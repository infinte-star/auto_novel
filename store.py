from __future__ import annotations

import contextlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Paths, read_text, safe_score, write_text

import sqlite3

# Historically a single sqlite3 connection was shared across the main thread and
# the BackgroundTasks daemon threads, guarded by one process-wide RLock that
# serialized EVERY DB op (sqlite3 connections/cursors are not thread-safe). That
# made WAL's reader/writer concurrency useless: finalize writes and planning
# reads could never overlap.
#
# We now give each thread its OWN connection via `ThreadLocalConn` (below). WAL
# already allows concurrent readers plus a single writer across connections, and
# busy_timeout makes a contending writer wait instead of erroring. So the global
# lock is no longer needed: `db_lock()` is kept as a NO-OP context manager purely
# so the ~20 internal `with db_lock():` sites and the external callers
# (writing.update_structured_state, planning.review_*, review.py) keep working
# unchanged.
_DB_LOCK = threading.RLock()  # retained for back-compat; no longer the serialization point


def db_lock() -> Any:
    """Return a NO-OP context manager.

    Per-thread connections (ThreadLocalConn) + WAL provide the concurrency that
    the old single shared connection lacked, so there is no longer a global
    serialization point. Call sites keep `with db_lock(): ...` for zero churn;
    it simply does nothing now.
    """
    return contextlib.nullcontext()


class ThreadLocalConn:
    """A sqlite3 facade that hands each thread its own connection.

    Exposes the subset of the sqlite3.Connection API the codebase uses
    (`execute`, `executescript`, `commit`, plus `row_factory`) so the ~25
    `conn.execute(...)` call sites and `init_db()`'s return value keep working
    with no changes.

    WAL permits one writer at a time; a concurrent writer gets SQLITE_BUSY which
    busy_timeout waits out. `execute` additionally retries once on a transient
    "database is locked" to be robust under the background finalize fan-out.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> "sqlite3.Connection":
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=5000")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._conn().execute(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - timing dependent
            if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
                return self._conn().execute(*args, **kwargs)
            raise

    def executescript(self, *args: Any, **kwargs: Any) -> Any:
        return self._conn().executescript(*args, **kwargs)

    def commit(self) -> None:
        self._conn().commit()

    def close_current(self) -> None:
        """Close THIS thread's connection (call from a worker's finally)."""
        c = getattr(self._local, "conn", None)
        if c is not None:
            try:
                c.close()
            finally:
                self._local.conn = None


def init_db(paths: Paths) -> Any:
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    conn = ThreadLocalConn(paths.database)
    # Run schema creation + idempotent migrations on the MAIN thread's
    # connection (the first _conn() call below opens it). WAL is a persistent
    # database setting, so setting it once here covers every later per-thread
    # connection. CREATE TABLE IF NOT EXISTS / ALTER are idempotent, so even
    # if two threads raced this it would be safe.
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout=5000;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chapter_metrics (
            chapter INTEGER PRIMARY KEY,
            title TEXT,
            score REAL,
            readthrough_score REAL,
            hook_score REAL,
            payoff_score REAL,
            novelty_score REAL,
            prose_score REAL,
            continuity_score REAL,
            plan_score REAL,
            payoff_type TEXT,
            conflict_type TEXT,
            tension INTEGER,
            novelty INTEGER,
            hook_strength INTEGER,
            emotional_tone TEXT,
            accepted INTEGER,
            em_dash_per_kchar REAL,
            style_penalty REAL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entities (
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_chapter INTEGER NOT NULL,
            PRIMARY KEY (entity_type, name)
        );
        CREATE TABLE IF NOT EXISTS open_threads (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            introduced_chapter INTEGER,
            due_chapter INTEGER,
            updated_chapter INTEGER,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reader_promises (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            opened_chapter INTEGER NOT NULL,
            due_chapter INTEGER,
            emotional_type TEXT,
            payoff_status TEXT,
            risk_level INTEGER DEFAULT 5,
            updated_chapter INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter INTEGER NOT NULL,
            agent TEXT NOT NULL,
            score REAL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stage_constraints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_chapter INTEGER NOT NULL,
            constraint_type TEXT NOT NULL,
            description TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            expires_chapter INTEGER,
            resolved_chapter INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS causal_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_event_id INTEGER NOT NULL,
            target_event_id INTEGER,
            source_chapter INTEGER NOT NULL,
            target_chapter INTEGER,
            link_type TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            created_at TEXT NOT NULL
        );
        """
    )
    # Idempotent migration: classify threads by type (plot/reader_promise/...).
    # Older DBs predate this column; ALTER is wrapped so re-runs are no-ops.
    try:
        conn.execute("ALTER TABLE open_threads ADD COLUMN thread_type TEXT DEFAULT 'plot'")
        conn.commit()
    except Exception:
        pass
    for column in (
        "readthrough_score REAL",
        "hook_score REAL",
        "payoff_score REAL",
        "novelty_score REAL",
        "prose_score REAL",
        "continuity_score REAL",
        "em_dash_per_kchar REAL",
        "style_penalty REAL",
    ):
        try:
            conn.execute(f"ALTER TABLE chapter_metrics ADD COLUMN {column}")
            conn.commit()
        except Exception:
            pass
    return conn

def db_event(conn: Any, chapter: int, event_type: str, payload: dict[str, Any]) -> None:
    with db_lock():
        conn.execute(
            "INSERT INTO events(chapter, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
            (chapter, event_type, json.dumps(payload, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()

def recent_metrics(conn: Any, limit: int) -> list[dict[str, Any]]:
    with db_lock():
        rows = conn.execute(
            "SELECT * FROM chapter_metrics ORDER BY chapter DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]

def recent_events(conn: Any, limit: int = 80, event_types: Any = None) -> list[dict[str, Any]]:
    """Return the most recent events, newest first.

    When `event_types` is a non-empty iterable, only events whose `event_type`
    is in that set are returned. The `events` table doubles as a full audit /
    telemetry log (it holds bulky diagnostic dumps such as `chapter_completed`,
    `plan_arbitration`, `chapter_extraction` whose JSON payloads are several KB
    each). Callers that only need plot continuity (e.g. `memory_context`) must
    pass `event_types={"story_event"}` so those multi-KB diagnostic payloads do
    not flood the prompt — left unfiltered, the injected event JSON grows by
    tens of KB per chapter and eventually overflows the model's real context
    limit (observed: plan prompt ballooning to ~300K chars by Ch5).
    """
    types = [t for t in (event_types or [])]
    with db_lock():
        if types:
            placeholders = ",".join("?" for _ in types)
            rows = conn.execute(
                f"SELECT id, chapter, event_type, payload, created_at FROM events "
                f"WHERE event_type IN ({placeholders}) ORDER BY id DESC LIMIT ?",
                (*types, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, chapter, event_type, payload, created_at FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except json.JSONDecodeError:
            pass
        out.append(item)
    return out

def get_active_constraints(conn: Any, chapter_num: int) -> list[dict[str, Any]]:
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT constraint_type, description, priority FROM stage_constraints
                   WHERE resolved_chapter IS NULL
                   AND (expires_chapter IS NULL OR expires_chapter > ?)
                   ORDER BY priority DESC""",
                (chapter_num,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []

def store_stage_constraints(conn: Any, chapter_num: int, constraints: list[dict[str, Any]]) -> None:
    if not constraints:
        return
    with db_lock():
        for c in constraints:
            expires = None
            if c.get("expires_in_chapters"):
                expires = chapter_num + int(c["expires_in_chapters"])
            conn.execute(
                """INSERT INTO stage_constraints(source_chapter, constraint_type, description, priority, expires_chapter, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    chapter_num,
                    str(c.get("type", "require")),
                    str(c.get("description", "")),
                    int(c.get("priority", 5)),
                    expires,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        conn.commit()

def store_causal_links(conn: Any, chapter_num: int, links: list[dict[str, Any]]) -> None:
    if not links:
        return
    with db_lock():
        for link in links:
            conn.execute(
                """INSERT INTO causal_links(source_event_id, target_event_id, source_chapter, target_chapter,
                   link_type, description, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    0,
                    None,
                    chapter_num,
                    link.get("target_chapter"),
                    str(link.get("link_type", "causes")),
                    str(link.get("description", "")),
                    "open",
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        conn.commit()

def upsert_reader_promise(conn: Any, chapter_num: int, promise: dict[str, Any]) -> None:
    promise_id = str(
        promise.get("id")
        or promise.get("thread_id")
        or f"promise-ch{chapter_num}-{abs(hash(json.dumps(promise, ensure_ascii=False))) % 100000}"
    )
    with db_lock():
        conn.execute(
            """
            INSERT INTO reader_promises(
                id, description, status, opened_chapter, due_chapter, emotional_type,
                payoff_status, risk_level, updated_chapter, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                description=excluded.description,
                status=excluded.status,
                due_chapter=excluded.due_chapter,
                emotional_type=excluded.emotional_type,
                payoff_status=excluded.payoff_status,
                risk_level=excluded.risk_level,
                updated_chapter=excluded.updated_chapter,
                payload_json=excluded.payload_json
            """,
            (
                promise_id,
                str(promise.get("description", "")),
                str(promise.get("status", "open")),
                int(promise.get("opened_chapter") or promise.get("introduced_chapter") or chapter_num),
                promise.get("due_chapter"),
                str(promise.get("emotional_type", promise.get("thread_type", "plot"))),
                str(promise.get("payoff_status", "pending")),
                int(promise.get("risk_level", 5) or 5),
                chapter_num,
                json.dumps(promise.get("payload", {}), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()

def get_reader_promises(conn: Any, chapter_num: int, limit: int = 12) -> list[dict[str, Any]]:
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT id, description, status, opened_chapter, due_chapter, emotional_type,
                          payoff_status, risk_level, updated_chapter
                   FROM reader_promises
                   WHERE status IN ('open','advanced')
                   ORDER BY risk_level DESC, COALESCE(due_chapter, 999999) ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            due = item.get("due_chapter")
            item["overdue_by"] = chapter_num - int(due) if due is not None and int(due) < chapter_num else 0
            out.append(item)
        return out
    except Exception:
        return []

def get_silent_threads(conn: Any, chapter_num: int, silence_threshold: int = 10, limit: int = 8) -> list[dict[str, Any]]:
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT id, description, updated_chapter FROM open_threads
                   WHERE status='open' AND updated_chapter IS NOT NULL
                   AND (? - updated_chapter) >= ?
                   ORDER BY updated_chapter ASC LIMIT ?""",
                (chapter_num, silence_threshold, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "description": row["description"],
                "updated_chapter": row["updated_chapter"],
                "silence_duration": chapter_num - int(row["updated_chapter"]),
            }
            for row in rows
        ]
    except Exception:
        return []

def get_open_causal_requirements(conn: Any) -> list[dict[str, Any]]:
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT link_type, description, source_chapter FROM causal_links
                   WHERE status='open' AND link_type IN ('requires', 'enables', 'blocks')
                   ORDER BY source_chapter DESC LIMIT 30""",
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []

def entity_state_as_of(conn: Any, entity_type: str, name: str, chapter: int | None = None) -> dict[str, Any]:
    """Return an entity's stored state. The entities table only keeps the latest
    state (no temporal history yet), so `chapter` is accepted for API forward-
    compatibility but currently ignored; latest state is returned."""
    try:
        with db_lock():
            row = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
        if row:
            return json.loads(row["state_json"])
    except Exception:
        pass
    return {}

def get_character_voice_notes(conn: Any, focus_names: list[str], limit: int = 6) -> list[dict[str, Any]]:
    """Lightweight character-consistency baseline: for each focus character, return
    a snapshot of their last-known stance/voice from the entities table's state
    dict. Reuses the existing 'character' entity state (no new table). Used by the
    reviewer to flag cross-chapter voice/stance drift. Returns at most `limit`
    entries, preferring names in `focus_names` order."""
    if not focus_names:
        return []
    # Keys in the free-form state dict that signal stance/voice/disposition.
    stance_keys = ("立场", "态度", "心态", "性格", "声音", "语气", "策略", "目标", "处境", "voice", "stance", "disposition", "goal", "status")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in focus_names:
        if not name or name in seen:
            continue
        seen.add(name)
        state = entity_state_as_of(conn, "character", name)
        if not isinstance(state, dict) or not state:
            continue
        snapshot = {k: v for k, v in state.items() if any(s in str(k) for s in stance_keys)}
        if not snapshot:
            # Fall back to the whole (small) state when no stance key matched.
            snapshot = dict(list(state.items())[:4])
        out.append({"name": name, "baseline": snapshot})
        if len(out) >= limit:
            break
    return out

def get_overdue_reader_promises(conn: Any, chapter_num: int, grace: int = 0, limit: int = 8) -> list[dict[str, Any]]:
    """Open threads explicitly typed as reader promises whose due_chapter has
    passed (plus an optional grace window). Sibling of get_silent_threads."""
    cutoff = chapter_num - max(0, int(grace))
    ledger = [
        p for p in get_reader_promises(conn, chapter_num, limit=limit * 2)
        if p.get("due_chapter") is not None and int(p.get("due_chapter")) < cutoff
    ]
    if ledger:
        return [
            {
                "id": p.get("id"),
                "description": p.get("description", ""),
                "due_chapter": int(p.get("due_chapter")),
                "overdue_by": chapter_num - int(p.get("due_chapter")),
                "source": "reader_promises",
            }
            for p in ledger[:limit]
        ]
    try:
        with db_lock():
            rows = conn.execute(
                """SELECT id, description, due_chapter FROM open_threads
                   WHERE status='open' AND thread_type='reader_promise'
                   AND due_chapter IS NOT NULL AND due_chapter < ?
                   ORDER BY due_chapter ASC LIMIT ?""",
                (cutoff, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "description": row["description"],
                "due_chapter": int(row["due_chapter"]),
                "overdue_by": chapter_num - int(row["due_chapter"]),
            }
            for row in rows
        ]
    except Exception:
        return []

def validate_plan_continuity(conn: Any, plan: dict[str, Any], chapter_num: int, config: dict[str, Any] | None = None) -> list[str]:
    violations = []
    deep = True
    if config is not None:
        deep = bool(config.get("novel", {}).get("plan_validate_deep", True))
    for char in plan.get("character_focus", []):
        try:
            with db_lock():
                row = conn.execute(
                    "SELECT state_json FROM entities WHERE entity_type='character' AND name=?",
                    (char,),
                ).fetchone()
            if row:
                state = json.loads(row["state_json"])
                status = state.get("status", "").lower()
                if status in ("dead", "deceased", "killed"):
                    violations.append(f"CRITICAL: {char} is dead, cannot act")
                elif status in ("imprisoned", "captured", "exiled"):
                    violations.append(f"WARNING: {char} is {status}, action requires explanation")
        except Exception:
            pass
    try:
        with db_lock():
            overdue = conn.execute(
                """SELECT id, description FROM open_threads
                   WHERE status='open' AND due_chapter IS NOT NULL AND due_chapter < ?
                   ORDER BY due_chapter ASC, updated_chapter DESC
                   LIMIT 20""",
                (chapter_num,),
            ).fetchall()
        seen_desc: set[str] = set()
        for thread in overdue:
            desc = (thread["description"] or "").strip()
            # 去重：同一线索被反复登记成不同 id 时，描述往往高度雷同，
            # 取描述前 24 字做指纹，避免同一伏笔灌爆规划提示。
            fp = desc[:24]
            if fp and fp in seen_desc:
                continue
            seen_desc.add(fp)
            violations.append(f"Overdue thread '{thread['id']}': {desc}")
    except Exception:
        pass

    if deep:
        violations.extend(_deep_plan_violations(conn, plan, chapter_num))
    return violations

def _deep_plan_violations(conn: Any, plan: dict[str, Any], chapter_num: int) -> list[str]:
    """STORYTELLER-style pre-write checks: compare the plan against stored entity
    state and open causal requirements. All new findings are WARNINGs (never
    CRITICAL) to avoid triggering needless re-plans on false positives."""
    out: list[str] = []

    # 1) Location coherence: if the plan asserts a character is somewhere that
    # conflicts with their stored location and no in-transit hint exists, warn.
    plan_blob = json.dumps(
        {k: plan.get(k) for k in ("beats", "world_state_changes", "goal", "conflict", "payoff")},
        ensure_ascii=False,
    )
    for char in plan.get("character_focus", []) or []:
        try:
            state = entity_state_as_of(conn, "character", str(char), chapter_num)
        except Exception:
            state = {}
        if not state:
            continue
        loc = str(state.get("location") or state.get("位置") or "").strip()
        if loc and loc not in plan_blob:
            # Only warn if the plan clearly relocates the character: another known
            # place-name entity appears in the plan blob.
            try:
                with db_lock():
                    places = conn.execute(
                        "SELECT name FROM entities WHERE entity_type='place'",
                    ).fetchall()
            except Exception:
                places = []
            other_place_in_plan = any(
                str(p["name"]).strip() and str(p["name"]).strip() != loc and str(p["name"]).strip() in plan_blob
                for p in places
            )
            if other_place_in_plan:
                out.append(
                    f"WARNING: {char} 当前位于「{loc}」，但本章计划似乎在别处展开；若发生移动需在 beats 中交代行程/在途。"
                )

    # 2) Causal requirements: if there are open 'requires' links, the plan should
    # not bank a payoff on an unestablished premise without acknowledging it.
    try:
        reqs = get_open_causal_requirements(conn)
    except Exception:
        reqs = []
    open_requires = [r for r in reqs if str(r.get("link_type")) == "requires"][:5]
    for r in open_requires:
        desc = str(r.get("description", "")).strip()
        if desc and desc not in plan_blob:
            out.append(
                f"WARNING: 存在未满足的前置因果「{desc}」(Ch{r.get('source_chapter')})；若本章 payoff 依赖它，需先在 beats 中建立该前提。"
            )
    return out

_QUALITY_FEEDBACK_CACHE: dict[tuple[str, int, int], tuple[tuple[float, int], list[dict[str, Any]]]] = {}
_QUALITY_FEEDBACK_CACHE_LOCK = threading.Lock()

def recent_quality_feedback(paths: Paths, limit: int = 5, max_items: int = 18) -> list[dict[str, Any]]:
    path = paths.logs_dir / "reviews.jsonl"
    if not path.exists():
        return []
    # reviews.jsonl grows linearly with chapter count but this helper only ever
    # returns the tail; it is called 4+ times per chapter (writer/reviewer/plan/
    # arbitration prompts). Cache the parsed result keyed by (path, limit,
    # max_items) and invalidated by the file's (mtime, size) so a fresh review
    # append busts the cache while repeated reads within a chapter hit it.
    try:
        stat = path.stat()
        signature = (stat.st_mtime, stat.st_size)
    except OSError:
        signature = None
    cache_key = (str(path), limit, max_items)
    if signature is not None:
        with _QUALITY_FEEDBACK_CACHE_LOCK:
            cached = _QUALITY_FEEDBACK_CACHE.get(cache_key)
            if cached is not None and cached[0] == signature:
                return [dict(row) for row in cached[1]]
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if len(rows) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        problems = row.get("problems") or []
        risks = row.get("continuity_risks") or []
        rows.append(
            {
                "chapter": row.get("chapter"),
                "score": row.get("score"),
                "plan_title": row.get("plan_title"),
                "problems": problems[:4],
                "continuity_risks": risks[:3],
            }
        )

    feedback = list(reversed(rows))
    item_count = 0
    trimmed: list[dict[str, Any]] = []
    for row in feedback:
        remaining = max_items - item_count
        if remaining <= 0:
            break
        problems = row["problems"][:remaining]
        remaining -= len(problems)
        risks = row["continuity_risks"][:remaining]
        item_count += len(problems) + len(risks)
        trimmed.append({**row, "problems": problems, "continuity_risks": risks})
    if signature is not None:
        with _QUALITY_FEEDBACK_CACHE_LOCK:
            _QUALITY_FEEDBACK_CACHE[cache_key] = (signature, [dict(row) for row in trimmed])
    return trimmed
