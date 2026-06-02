from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Paths, read_text, safe_score, write_text

try:
    import sqlite3
except ModuleNotFoundError:
    sqlite3 = None  # type: ignore[assignment]

class JsonStoryStore:
    """Fallback event store for Python builds without sqlite3."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(
                {
                    "events": [],
                    "chapter_metrics": {},
                    "entities": {},
                    "open_threads": {},
                    "agent_reports": [],
                }
            )

    def _read(self) -> dict[str, Any]:
        return json.loads(read_text(self.path) or "{}")

    def _write(self, data: dict[str, Any]) -> None:
        write_text(self.path, json.dumps(data, ensure_ascii=False, indent=2))

    def add_event(self, chapter: int, event_type: str, payload: dict[str, Any]) -> None:
        data = self._read()
        data.setdefault("events", []).append(
            {
                "id": len(data.get("events", [])) + 1,
                "chapter": chapter,
                "event_type": event_type,
                "payload": payload,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write(data)

    def recent_metrics(self, limit: int) -> list[dict[str, Any]]:
        metrics = list(self._read().get("chapter_metrics", {}).values())
        metrics.sort(key=lambda x: int(x.get("chapter", 0)), reverse=True)
        return metrics[:limit]

    def recent_events(self, limit: int) -> list[dict[str, Any]]:
        events = self._read().get("events", [])
        return list(reversed(events))[:limit]

    def add_agent_report(self, chapter: int, agent: str, report: dict[str, Any]) -> None:
        data = self._read()
        data.setdefault("agent_reports", []).append(
            {
                "chapter": chapter,
                "agent": agent,
                "score": safe_score(report.get("score", 0)),
                "report": report,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write(data)

    def get_entity_state(self, entity_type: str, name: str) -> dict[str, Any]:
        key = f"{entity_type}:{name}"
        return self._read().get("entities", {}).get(key, {}).get("state", {})

    def upsert_entity(self, entity_type: str, name: str, state: dict[str, Any], chapter: int) -> None:
        data = self._read()
        key = f"{entity_type}:{name}"
        data.setdefault("entities", {})[key] = {
            "entity_type": entity_type,
            "name": name,
            "state": state,
            "updated_chapter": chapter,
        }
        self._write(data)

    def upsert_thread(self, thread_id: str, thread: dict[str, Any], chapter: int) -> None:
        data = self._read()
        data.setdefault("open_threads", {})[thread_id] = {
            "id": thread_id,
            "description": str(thread.get("description", "")),
            "status": str(thread.get("status", "open")),
            "thread_type": str(thread.get("thread_type", "plot")),
            "introduced_chapter": thread.get("introduced_chapter"),
            "due_chapter": thread.get("due_chapter"),
            "updated_chapter": chapter,
            "payload": thread.get("payload", {}),
        }
        self._write(data)

    def upsert_metrics(self, chapter: int, metrics: dict[str, Any]) -> None:
        data = self._read()
        data.setdefault("chapter_metrics", {})[str(chapter)] = metrics
        self._write(data)

def init_db(paths: Paths) -> Any:
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    if sqlite3 is None:
        return JsonStoryStore(paths.logs_dir / "story_state.json")
    try:
        conn = sqlite3.connect(paths.database, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
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
                plan_score REAL,
                payoff_type TEXT,
                conflict_type TEXT,
                tension INTEGER,
                novelty INTEGER,
                hook_strength INTEGER,
                emotional_tone TEXT,
                accepted INTEGER,
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
        return conn
    except Exception:
        return JsonStoryStore(paths.logs_dir / "story_state.json")

def db_event(conn: Any, chapter: int, event_type: str, payload: dict[str, Any]) -> None:
    if isinstance(conn, JsonStoryStore):
        conn.add_event(chapter, event_type, payload)
        return
    conn.execute(
        "INSERT INTO events(chapter, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
        (chapter, event_type, json.dumps(payload, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()

def recent_metrics(conn: Any, limit: int) -> list[dict[str, Any]]:
    if isinstance(conn, JsonStoryStore):
        return conn.recent_metrics(limit)
    rows = conn.execute(
        "SELECT * FROM chapter_metrics ORDER BY chapter DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]

def recent_events(conn: Any, limit: int = 80) -> list[dict[str, Any]]:
    if isinstance(conn, JsonStoryStore):
        return conn.recent_events(limit)
    rows = conn.execute(
        "SELECT chapter, event_type, payload, created_at FROM events ORDER BY id DESC LIMIT ?",
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
    if isinstance(conn, JsonStoryStore):
        return []
    try:
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
    if isinstance(conn, JsonStoryStore) or not constraints:
        return
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
    if isinstance(conn, JsonStoryStore) or not links:
        return
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

def get_silent_threads(conn: Any, chapter_num: int, silence_threshold: int = 10, limit: int = 8) -> list[dict[str, Any]]:
    if isinstance(conn, JsonStoryStore):
        threads = conn._read().get("open_threads", {}).values()
        out = []
        for t in threads:
            if t.get("status") != "open":
                continue
            updated = int(t.get("updated_chapter") or 0)
            silence = chapter_num - updated
            if silence >= silence_threshold:
                out.append({
                    "id": t.get("id"),
                    "description": t.get("description", ""),
                    "updated_chapter": updated,
                    "silence_duration": silence,
                })
        out.sort(key=lambda x: -x["silence_duration"])
        return out[:limit]
    try:
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
    if isinstance(conn, JsonStoryStore):
        return []
    try:
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
    if isinstance(conn, JsonStoryStore):
        return conn.get_entity_state(entity_type, name)
    try:
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
    if isinstance(conn, JsonStoryStore):
        out = []
        for t in conn._read().get("open_threads", {}).values():
            if t.get("status") != "open":
                continue
            if str(t.get("thread_type", "plot")) != "reader_promise":
                continue
            due = t.get("due_chapter")
            if due is None or int(due) >= cutoff:
                continue
            out.append({
                "id": t.get("id"),
                "description": t.get("description", ""),
                "due_chapter": int(due),
                "overdue_by": chapter_num - int(due),
            })
        out.sort(key=lambda x: -x["overdue_by"])
        return out[:limit]
    try:
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
    if isinstance(conn, JsonStoryStore):
        return violations
    deep = True
    if config is not None:
        deep = bool(config.get("novel", {}).get("plan_validate_deep", True))
    for char in plan.get("character_focus", []):
        try:
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
        overdue = conn.execute(
            """SELECT id, description FROM open_threads
               WHERE status='open' AND due_chapter IS NOT NULL AND due_chapter < ?""",
            (chapter_num,),
        ).fetchall()
        for thread in overdue:
            violations.append(f"Overdue thread '{thread['id']}': {thread['description']}")
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

def recent_quality_feedback(paths: Paths, limit: int = 5, max_items: int = 18) -> list[dict[str, Any]]:
    path = paths.logs_dir / "reviews.jsonl"
    if not path.exists():
        return []
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
    return trimmed
