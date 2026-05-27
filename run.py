"""
Industrial long-form web novel engine.

This version treats generation as a stateful production pipeline, not as
plain continuation. It keeps an event-sourced SQLite ledger, layered memory
files, candidate plan arbitration, multi-agent reviews, rhythm metrics, and
per-chapter archives.

Usage:
    python run.py
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import sqlite3
except ModuleNotFoundError:
    sqlite3 = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from openai import OpenAI


ROOT = Path(__file__).resolve().parent
PROMPT_FILE = ROOT / "prompt.md"
CONFIG_FILE = ROOT / "config.yaml"
CHECKPOINT_VERSION = 2
CHAPTER_CURRENT_CHECKPOINT = f"chapter_current_v{CHECKPOINT_VERSION}.md"


class LLMClientPool:
    def __init__(self, clients: list[OpenAI], primary_count: int | None = None) -> None:
        if not clients:
            raise ValueError("LLMClientPool requires at least one client")
        self.clients = clients
        self.primary_count = len(clients) if primary_count is None else min(max(primary_count, 0), len(clients))
        if self.primary_count == 0:
            self.primary_count = len(clients)
        self.lock = threading.Lock()
        self.next_index = 0

    def create_completion(self, **kwargs: Any) -> Any:
        attempts = self._attempt_order()
        first_error: Exception | None = None
        for index in attempts:
            client = self.clients[index]
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                if not self._should_try_next_client(exc):
                    raise
        if first_error is not None:
            raise first_error
        raise RuntimeError("LLMClientPool has no clients to try")

    def _attempt_order(self) -> list[int]:
        with self.lock:
            start = self.next_index % self.primary_count
            self.next_index += 1
        primary = [(start + offset) % self.primary_count for offset in range(self.primary_count)]
        fallback = list(range(self.primary_count, len(self.clients)))
        return primary + fallback

    @staticmethod
    def _should_try_next_client(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return int(status_code) in {401, 408, 409, 429, 500, 502, 503, 504}
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


@dataclass(frozen=True)
class Paths:
    book: Path
    state: Path
    bible: Path
    characters: Path
    timeline: Path
    threads: Path
    volume_plan: Path
    chapters_dir: Path
    logs_dir: Path
    database: Path


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    section: str | None = None
    for raw_line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            config[section] = {}
            continue
        if section and ":" in line:
            key, value = line.strip().split(":", 1)
            config[section][key.strip()] = parse_scalar(value)

    required = {
        "api": ["base_url", "api_key", "model", "max_tokens", "temperature"],
        "novel": [
            "chapter_words",
            "target_words",
            "quality_threshold",
            "max_revision_rounds",
            "candidate_plans",
            "min_plan_score",
            "outline_buffer",
            "summary_keep",
            "recent_tail_chars",
            "long_memory_every",
            "stage_review_every",
            "repeat_window",
            "fatigue_window",
            "key_event_interval",
        ],
        "paths": [
            "book",
            "state",
            "bible",
            "characters",
            "timeline",
            "threads",
            "volume_plan",
            "chapters_dir",
            "logs_dir",
            "database",
        ],
    }
    for section_name, keys in required.items():
        if section_name not in config:
            raise KeyError(f"Missing config section: {section_name}")
        for key in keys:
            if key not in config[section_name]:
                raise KeyError(f"Missing config value: {section_name}.{key}")
    return config


def configured_api_keys(config: dict[str, Any]) -> list[str]:
    api = config["api"]
    keys: list[str] = []
    primary = str(api.get("api_key", "")).strip()
    if primary:
        keys.append(primary)
    extra = str(api.get("api_keys", "")).strip()
    if extra:
        keys.extend(k for k in re.split(r"[,;\s]+", extra) if k)

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def configured_api_endpoints(config: dict[str, Any]) -> tuple[list[tuple[str, str]], int]:
    api = config["api"]
    endpoints: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    groups = str(api.get("api_key_groups", "")).strip()
    if groups:
        for group in groups.split(";"):
            group = group.strip()
            if not group:
                continue
            if "|" not in group:
                raise ValueError("Invalid api.api_key_groups item, expected base_url|key1,key2")
            base_url, keys_text = group.split("|", 1)
            base_url = base_url.strip()
            for key in re.split(r"[,\s]+", keys_text):
                key = key.strip()
                if not base_url or not key:
                    continue
                endpoint = (base_url, key)
                if endpoint not in seen:
                    seen.add(endpoint)
                    endpoints.append(endpoint)

    primary_count = len(endpoints)
    fallback_base_url = str(api["base_url"]).strip()
    for key in configured_api_keys(config):
        endpoint = (fallback_base_url, key)
        if endpoint not in seen:
            seen.add(endpoint)
            endpoints.append(endpoint)

    return endpoints, primary_count


def get_paths(config: dict[str, Any]) -> Paths:
    raw = config["paths"]
    return Paths(
        book=ROOT / str(raw["book"]),
        state=ROOT / str(raw["state"]),
        bible=ROOT / str(raw["bible"]),
        characters=ROOT / str(raw["characters"]),
        timeline=ROOT / str(raw["timeline"]),
        threads=ROOT / str(raw["threads"]),
        volume_plan=ROOT / str(raw["volume_plan"]),
        chapters_dir=ROOT / str(raw["chapters_dir"]),
        logs_dir=ROOT / str(raw["logs_dir"]),
        database=ROOT / str(raw["database"]),
    )


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


def log(paths: Paths, message: str) -> None:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    append_text(paths.logs_dir / "run.log", line + "\n")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_chapter(text: str) -> str:
    return normalize_text(text) + "\n"


def count_chars(path: Path) -> int:
    return len(read_text(path))


def tail_text(path: Path, n_chars: int) -> str:
    text = read_text(path)
    return text[-n_chars:] if len(text) > n_chars else text


def chapter_path(paths: Paths, chapter_num: int) -> Path:
    return paths.chapters_dir / f"{chapter_num:04d}.md"


def checkpoint_dir(paths: Paths, chapter_num: int) -> Path:
    return paths.logs_dir / "checkpoints" / f"ch{chapter_num:04d}"


def checkpoint_path(paths: Paths, chapter_num: int, name: str) -> Path:
    return checkpoint_dir(paths, chapter_num) / name


def load_checkpoint(paths: Paths, chapter_num: int, name: str) -> Any | None:
    path = checkpoint_path(paths, chapter_num, name)
    if not path.exists():
        return None
    try:
        if path.suffix == ".json":
            data = json.loads(read_text(path))
            if isinstance(data, dict) and "_checkpoint_version" in data:
                if data.get("_checkpoint_version") == CHECKPOINT_VERSION:
                    return data.get("payload")
                log(
                    paths,
                    f"Ignoring stale checkpoint Ch{chapter_num} {name} "
                    f"version={data.get('_checkpoint_version')} current={CHECKPOINT_VERSION}",
                )
                return None
            return data
        return read_text(path)
    except Exception as exc:
        log(paths, f"Ignoring unreadable checkpoint Ch{chapter_num} {name}: {exc}")
        return None


def save_checkpoint(paths: Paths, chapter_num: int, name: str, payload: Any) -> None:
    path = checkpoint_path(paths, chapter_num, name)
    if path.suffix == ".json":
        write_text(
            path,
            json.dumps(
                {
                    "_checkpoint_version": CHECKPOINT_VERSION,
                    "chapter": chapter_num,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        return
    write_text(path, str(payload))


def safe_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    match = re.match(r"([\d.]+)", s)
    if match:
        return float(match.group(1))
    return 0.0


def find_last_chapter(paths: Paths) -> int:
    if not paths.chapters_dir.exists():
        return 0
    nums = [int(p.stem) for p in paths.chapters_dir.glob("*.md") if p.stem.isdigit()]
    return max(nums) if nums else 0


def should_resume_existing_chapter(paths: Paths, chapter_num: int) -> bool:
    if chapter_num <= 0 or not chapter_path(paths, chapter_num).exists():
        return False
    if not checkpoint_dir(paths, chapter_num).exists():
        return False
    if checkpoint_path(paths, chapter_num, "chapter_saved.json").exists() or checkpoint_path(
        paths, chapter_num, "chapter_completed.json"
    ).exists():
        return False
    return not bool(load_checkpoint(paths, chapter_num, "chapter_completed.json"))


def _repair_truncated_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    for end in range(len(s), max(len(s) - 5000, 0), -100):
        candidate = s[:end].rstrip(", \t\n\r:")
        candidate = re.sub(r',\s*"[^"]*"?\s*:?\s*[^,{}\[\]]*$', "", candidate).rstrip(", \t\n\r:")
        stack: list[str] = []
        in_str = False
        esc = False
        broken = False
        for c in candidate:
            if esc:
                esc = False
                continue
            if in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    broken = True
                    break
                stack.pop()
        if broken or in_str:
            continue
        closer = "".join("}" if o == "{" else "]" for o in reversed(stack))
        repaired = candidate + closer
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            continue
    return None


def safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = normalize_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_json(cleaned)
    if repaired:
        return json.loads(repaired)
    raise json.JSONDecodeError(f"Could not recover JSON. Preview: {cleaned[:300]!r}", cleaned, 0)


JSON_REPAIR_SYSTEM = """You repair malformed JSON from an LLM response.
Return valid JSON only. Do not add explanations. Preserve the intended fields and values."""


JSON_OUTPUT_CONTRACT = """Output contract:
- Return exactly one valid JSON object and nothing else.
- The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
- Do not use markdown headings, bullet lists, code fences, explanations, or prefaces.
- Use double quotes for every key and string value.
- Escape quotes inside string values.
- Do not use trailing commas, comments, NaN, Infinity, or Python-style booleans.
- Keep the schema keys exactly as requested; do not translate key names.
- If uncertain, still return the requested schema with conservative values and short Chinese strings."""


def json_prompt(user: str) -> str:
    return user.rstrip() + "\n\n## Mandatory JSON Output Contract\n" + JSON_OUTPUT_CONTRACT


def load_json_with_repair(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    raw: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return safe_json_loads(raw)
    except json.JSONDecodeError as exc:
        log(paths, f"JSON parse failed, attempting repair: {exc}")
    if not raw.strip():
        if fallback is not None:
            return fallback
        raise json.JSONDecodeError("Empty JSON response", raw, 0)
    repair_prompt = f"""Repair this malformed JSON into one valid JSON object.

## Malformed JSON
{raw[:20000]}"""
    try:
        repaired = call_llm(
            client,
            paths,
            config,
            JSON_REPAIR_SYSTEM,
            json_prompt(repair_prompt),
            max_tokens=8000,
            temperature=0,
        )
        return safe_json_loads(repaired)
    except Exception as exc:
        log(paths, f"JSON repair failed: {exc}")
        if fallback is not None:
            return fallback
        raise


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
        conn = sqlite3.connect(paths.database)
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


def validate_plan_continuity(conn: Any, plan: dict[str, Any], chapter_num: int) -> list[str]:
    violations = []
    if isinstance(conn, JsonStoryStore):
        return violations
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
    return violations


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


def ensure_project(paths: Paths) -> None:
    paths.chapters_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    for path, title in [
        (paths.bible, "World Bible"),
        (paths.characters, "Characters"),
        (paths.timeline, "Timeline"),
        (paths.threads, "Threads"),
        (paths.volume_plan, "Volume Plan"),
    ]:
        if not path.exists():
            write_text(path, f"# {title}\n\n")


def rebuild_book(paths: Paths) -> None:
    chunks = []
    for path in sorted(paths.chapters_dir.glob("*.md")):
        text = read_text(path).strip()
        if text:
            chunks.append(text)
    if chunks:
        write_text(paths.book, "\n\n".join(chunks) + "\n")


def estimate_chars_budget(config: dict[str, Any]) -> int:
    context_window = int(config["api"].get("context_window", 1000000))
    reserve = int(config["novel"].get("context_budget_reserve_chars", 40000))
    return max(context_window - reserve, 50000)


def truncate_section(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    budget = estimate_chars_budget(config)
    fatigue_window = int(config["novel"]["fatigue_window"])

    creative_brief = read_text(PROMPT_FILE).strip()
    current_state = read_text(paths.state).strip()
    tier1 = "## Creative Brief\n" + creative_brief + "\n\n## Current State\n" + current_state

    volume_plan = read_text(paths.volume_plan).strip()
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)
    threads_text = read_text(paths.threads).strip()
    tier2 = "## Volume Plan\n" + volume_plan + "\n\n## Key Metrics JSON\n" + metrics_5 + "\n\n## Threads\n" + threads_text

    characters = read_text(paths.characters).strip()
    bible = read_text(paths.bible).strip()
    events_20 = json.dumps(recent_events(conn, 20), ensure_ascii=False, indent=2)
    tier3 = "## Characters\n" + characters + "\n\n## World Bible\n" + bible + "\n\n## Recent Events JSON\n" + events_20

    timeline = read_text(paths.timeline).strip()
    metrics_full = json.dumps(recent_metrics(conn, fatigue_window), ensure_ascii=False, indent=2)
    events_full = json.dumps(recent_events(conn, 40), ensure_ascii=False, indent=2)
    tier4 = "## Timeline\n" + timeline + "\n\n## Full Metrics JSON\n" + metrics_full + "\n\n## Full Events JSON\n" + events_full

    assembled = tier1
    remaining = budget - len(assembled)

    if remaining > len(tier2):
        assembled += "\n\n" + tier2
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier2, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier3):
        assembled += "\n\n" + tier3
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier3, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier4):
        assembled += "\n\n" + tier4
    elif remaining > 2000:
        assembled += "\n\n" + truncate_section(tier4, max(remaining - 100, 0))

    return assembled


def emergency_truncate(user_text: str, max_chars: int) -> str:
    if len(user_text) <= max_chars:
        return user_text
    sections = re.split(r"(?=^## )", user_text, flags=re.MULTILINE)
    priority_keywords = ["Creative Brief", "Current State", "Selected Plan", "Arbitration"]
    high = []
    medium = []
    low = []
    for section in sections:
        if any(kw in section[:80] for kw in priority_keywords):
            high.append(section)
        elif any(kw in section[:80] for kw in ["Characters", "Bible", "Volume Plan", "Threads"]):
            medium.append(section)
        else:
            low.append(section)
    result = "".join(high)
    for section in medium:
        if len(result) + len(section) < max_chars * 0.85:
            result += section
        else:
            remaining = int(max_chars * 0.85) - len(result)
            if remaining > 500:
                result += section[:remaining] + "\n...[truncated]"
            break
    for section in low:
        if len(result) + len(section) < max_chars:
            result += section
        else:
            remaining = max_chars - len(result)
            if remaining > 500:
                result += section[:remaining] + "\n...[truncated]"
            break
    return result


def call_llm(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    system: str,
    user: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    api = config["api"]
    context_window = int(api.get("context_window", 1000000))
    max_input_chars = int(context_window * 1.8)
    total_chars = len(system) + len(user)
    if total_chars > max_input_chars:
        user = emergency_truncate(user, max_input_chars - len(system) - 1000)
    for attempt in range(6):
        started = time.perf_counter()
        try:
            request = {
                "model": api["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens or int(api["max_tokens"]),
                "temperature": float(api["temperature"]) if temperature is None else temperature,
            }
            stream = bool(api.get("stream", False))
            if stream:
                request["stream"] = True
            if hasattr(client, "create_completion"):
                resp = client.create_completion(**request)
            else:
                resp = client.chat.completions.create(**request)
            if stream:
                parts: list[str] = []
                reasoning_parts: list[str] = []
                chunk_count = 0
                finish_reason = None
                for chunk in resp:
                    chunk_count += 1
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = getattr(choice, "finish_reason", finish_reason)
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece:
                        parts.append(piece)
                    reasoning_piece = getattr(delta, "reasoning_content", None)
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                content = "".join(parts)
                elapsed = time.perf_counter() - started
                if not content.strip() and reasoning_parts:
                    log(
                        paths,
                        "LLM content empty but reasoning_content present, using reasoning fallback "
                        f"attempt={attempt + 1}/6 chunks={chunk_count} finish={finish_reason} "
                        f"reasoning_chars={sum(len(p) for p in reasoning_parts)} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
                    content = "".join(reasoning_parts)
                elif not content.strip():
                    log(
                        paths,
                        "LLM returned empty streamed response "
                        f"attempt={attempt + 1}/6 chunks={chunk_count} finish={finish_reason} "
                        f"elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                    )
            else:
                content = resp.choices[0].message.content or ""
                elapsed = time.perf_counter() - started
            if not content.strip():
                wait = min(60, 2**attempt)
                log(
                    paths,
                    f"LLM returned empty response attempt={attempt + 1}/6 wait={wait}s "
                    f"stream={stream} elapsed={elapsed:.1f}s prompt_chars={total_chars} max_tokens={request['max_tokens']}",
                )
                time.sleep(wait)
                continue
            return content
        except Exception as exc:
            wait = min(60, 2**attempt)
            elapsed = time.perf_counter() - started
            log(paths, f"LLM call failed attempt={attempt + 1}/6 wait={wait}s elapsed={elapsed:.1f}s error={exc}")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after 6 attempts")


BOOTSTRAP_SYSTEM = """You are the chief architect for a 2M+ Chinese web novel.
Return exactly one valid JSON object and no other text. Keys:
{
  "state": "short current-state markdown, <=5000 Chinese chars",
  "bible": "world rules, power system, social order, hard constraints",
  "characters": "major character state machines: goal, fear, resources, relationships, secrets",
  "timeline": "initial chronology and planned historical pressure",
  "threads": "open foreshadowing ledger with introduced/due/status",
  "volume_plan": "at least 3 volumes, 60-80 chapters each, with major event anchors"
}
Create original material. Do not imitate existing works. Optimize for long-term causality and reader anticipation."""


CANDIDATE_PLAN_SYSTEM = """You are a chapter-planning agent in an industrial long-form fiction engine.
Return exactly one valid JSON object and no other text. Create one candidate plan for the requested chapter.
Schema:
{
  "title": "...",
  "goal": "...",
  "conflict": "...",
  "conflict_type": "court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
  "payoff": "...",
  "payoff_type": "court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
  "pressure": "what suppresses the protagonist/readers before payoff",
  "beats": ["5-9 concrete beats"],
  "character_focus": ["characters who get agency or emotional movement"],
  "world_state_changes": ["state changes if this chapter happens"],
  "thread_actions": ["foreshadowing opened/advanced/recovered"],
  "hook": "chapter-end reader question",
  "risk": "main continuity or repetition risk"
}
The chapter must advance long-term causality, not merely create local excitement.
Every plan must:
- Convert at least one stale review problem into a concrete on-page scene.
- Include causal bridges for travel time, message delivery, money movement, and surveillance if they matter.
- Specify visible actions, sensory anchors, and dialogue pressure, not only analysis or summary.
- Avoid reusing the recent chapter-ending device, analysis posture, or emotional beat."""


ARBITER_SYSTEM = """You are the arbitration layer for a long-form fiction engine.
Evaluate candidate plans against global state, recent metrics, repetition risk, causal value, character consistency,
payoff freshness, and reader anticipation.
Return exactly one valid JSON object and no other text:
{
  "selected_index": 0,
  "scores": [{"index": 0, "score": 1-10, "pros": [], "cons": []}],
  "merged_plan": {same schema as candidate, improved if needed},
  "required_constraints": ["hard constraints the writer must obey"],
  "reader_expectation_delta": "why this improves or hurts follow-up desire"
}
Reject or downgrade plans that keep known review problems abstract, rely on off-page resolution,
repeat the same physical staging, or contain unresolved timeline/logistics gaps. Improve the merged plan
so the writer has concrete scene obligations rather than vague intentions."""


AGENT_REVIEW_SYSTEMS = {
    "world": """You are World Agent. Check world rules, power system, geography, institutions, and resource logic.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "character": """You are Character Agent. Check character goals, agency, relationships, secrets, trauma, and OOC drift.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "rhythm": """You are Rhythm Agent. Check pacing, compression/release cycle, scene variety, and reader fatigue.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "payoff": """You are Payoff Agent. Check emotional payoff, pressure-payoff ratio, hook strength, and novelty.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "foreshadowing": """You are Foreshadowing Agent. Check opened/advanced/recovered threads and long-term promises.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "reader": """You are Reader-Simulation Agent. Simulate a serial reader after this chapter plan.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"follow_next_reason":"..."}.""",
}


WRITE_SYSTEM = """You are a professional Chinese long-form web novel author.
Write the chapter in Chinese.
Requirements:
- Around {chapter_words} Chinese characters.
- Start exactly with: 第{chapter_num}章 {title}
- Execute the selected plan and all constraints.
- Put the high-risk plan beats directly on page; do not leave important operations only implied.
- Repair the recent quality feedback explicitly through scenes, choices, cost, and consequences.
- Vary scene staging, chapter ending, emotional texture, and reasoning posture from recent chapters.
- When logistics matter, show the time, route, handler, procedure, and institutional friction.
- Keep causality, character agency, pressure-payoff rhythm, and hook strength.
- Avoid summary-like prose, repetitive shock reactions, and cheap coincidence.
- Output the chapter only, no explanation."""


REVIEW_SYSTEM = """You are a strict final editor for serialized Chinese web fiction.
Return exactly one valid JSON object and no other text:
{
  "score": 1-10,
  "accepted": true,
  "problems": [],
  "fixes": [],
  "continuity_risks": [],
  "rhythm_risks": [],
  "reader_fatigue_risks": []
}
Scoring rules:
- Cap score at 8 if important selected-plan beats are missing from the chapter text.
- Cap score at 8 if a timeline, money movement, message route, surveillance source, or procedure is hand-waved.
- Cap score at 8 if the chapter repeats a recent scene shape or ending device without a clear new function.
- Cap score at 7 if continuity risks from recent reviews are ignored again.
- Award 9+ only when the chapter solves prior feedback on page while preserving tension and follow-up desire."""


REVISE_SYSTEM = """You are a Chinese web novel revision writer.
Revise the full chapter according to the final editor report.
Keep the title and core events. Do not introduce new continuity risks.
Prefer targeted structural repair over cosmetic rewriting:
- Add missing causal bridges and concrete scenes.
- Replace repeated staging or chapter endings.
- Make plan beats visible on page.
- Strengthen character agency, procedural friction, and pressure-payoff rhythm.
Output the revised chapter only."""


EXTRACT_SYSTEM = """You are the event-sourcing extractor for a long-form fiction engine.
Return exactly one valid JSON object and no other text:
{
  "title": "...",
  "events": [{"type":"plot|world|character|force|thread|item|battle|relationship","summary":"...","effects":[]}],
  "entities": [{"entity_type":"character|force|place|item|rule","name":"...","state_patch":{}}],
  "threads": [{"id":"stable-id","description":"...","status":"open|advanced|recovered|dropped","introduced_chapter":1,"due_chapter":20,"payload":{}}],
  "causal_links": [{"from_event":"source event summary","to_event":"expected future event or consequence","link_type":"causes|enables|blocks|requires","description":"why this causal link exists"}],
  "metrics": {
    "payoff_type":"court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
    "conflict_type":"court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
    "tension":1-10,
    "novelty":1-10,
    "hook_strength":1-10,
    "emotional_tone":"..."
  },
  "memory_updates": {
    "bible": [],
    "characters": [],
    "timeline": [],
    "threads": []
  }
}"""


STATE_UPDATE_SYSTEM = """You maintain the short working state for a 2M+ novel.
Return markdown only, no explanation.
Requirements:
- <=5000 Chinese characters.
- Include current progress, volume/stage goal, protagonist state, key conflicts, next 12 chapter directions.
- Keep recent chapter summaries compact.
- Preserve hard continuity constraints."""


STAGE_REVIEW_SYSTEM = """You are the long-cycle quality evaluator.
Return markdown followed by a JSON block.

Markdown section:
## Quality Trend
## Continuity Risks
## Rhythm and Payoff Risks
## Repetition Risks
## Next 20 Chapters Replan
## Threads to Recover or Upgrade

Then output a fenced JSON block with actionable constraints:
```json
{"constraints": [
  {"type": "avoid|require|replan|recover_thread", "description": "...", "priority": 1-10, "expires_in_chapters": 20}
]}
```"""


MEMORY_COMPRESS_SYSTEM = """You compress memory entries for a long-form fiction engine.
Input: a memory file with per-chapter entries (## ChN sections).
Output: a consolidated markdown that preserves:
- All entity names and their CURRENT state (not historical intermediate states)
- All unresolved constraints and open threads
- All causal dependencies still relevant to future chapters
- Key turning points and irreversible changes
Remove: superseded states, routine confirmations, resolved items, redundant updates.
Keep output under {max_chars} Chinese characters.
Output the consolidated content only, no explanation."""


def bootstrap(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> None:
    log(paths, "Bootstrapping layered memory")
    raw = call_llm(client, paths, config, BOOTSTRAP_SYSTEM, json_prompt(read_text(PROMPT_FILE)), temperature=0.7)
    data = load_json_with_repair(client, paths, config, raw)
    write_text(paths.state, data["state"].strip() + "\n")
    write_text(paths.bible, data["bible"].strip() + "\n")
    write_text(paths.characters, data["characters"].strip() + "\n")
    write_text(paths.timeline, data["timeline"].strip() + "\n")
    write_text(paths.threads, data["threads"].strip() + "\n")
    write_text(paths.volume_plan, data["volume_plan"].strip() + "\n")
    db_event(conn, 0, "bootstrap", data)


def should_compress_memory(paths: Paths, config: dict[str, Any], chapter_num: int) -> bool:
    compress_every = int(config["novel"].get("memory_compress_every", 30))
    max_kb = int(config["novel"].get("memory_max_kb", 15))
    if chapter_num > 0 and chapter_num % compress_every == 0:
        return True
    for p in [paths.bible, paths.characters, paths.timeline, paths.threads]:
        if p.exists() and p.stat().st_size > max_kb * 1024:
            return True
    return False


def compress_memory_file(
    client: OpenAI, paths: Paths, config: dict[str, Any], file_path: Path, keep_recent: int = 30
) -> None:
    content = read_text(file_path)
    if not content.strip():
        return
    sections = re.split(r"(?=^## Ch\d+)", content, flags=re.MULTILINE)
    if len(sections) <= 2:
        return
    header = sections[0]
    chapter_sections = sections[1:]
    if len(chapter_sections) <= keep_recent:
        return
    old_sections = chapter_sections[:-keep_recent]
    recent_sections = chapter_sections[-keep_recent:]
    archive_dir = paths.logs_dir / "memory_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    write_text(archive_path, "".join(old_sections))
    old_text = "".join(old_sections)
    max_chars = 3000
    system = MEMORY_COMPRESS_SYSTEM.format(max_chars=max_chars)
    compressed = call_llm(client, paths, config, system, old_text, max_tokens=8000, temperature=0.2)
    compressed = normalize_text(compressed)
    new_content = header.rstrip() + "\n\n## Consolidated\n" + compressed + "\n\n" + "".join(recent_sections)
    write_text(file_path, new_content)


def compress_all_memory(client: OpenAI, paths: Paths, config: dict[str, Any]) -> None:
    for file_path in [paths.bible, paths.characters, paths.timeline, paths.threads]:
        if file_path.exists() and read_text(file_path).strip():
            compress_memory_file(client, paths, config, file_path)


def rhythm_diagnostics(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    if not rows:
        return {
            "warnings": [],
            "payoff_counts": {},
            "conflict_counts": {},
            "avg_tension": None,
            "avg_novelty": None,
            "avg_hook": None,
        }

    payoff_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    tensions = []
    novelties = []
    hooks = []
    for row in rows:
        payoff_counts[row.get("payoff_type") or "unknown"] = payoff_counts.get(row.get("payoff_type") or "unknown", 0) + 1
        conflict_counts[row.get("conflict_type") or "unknown"] = conflict_counts.get(row.get("conflict_type") or "unknown", 0) + 1
        if row.get("tension") is not None:
            tensions.append(int(row["tension"]))
        if row.get("novelty") is not None:
            novelties.append(int(row["novelty"]))
        if row.get("hook_strength") is not None:
            hooks.append(int(row["hook_strength"]))

    warnings = []
    dominant_payoff = max(payoff_counts.items(), key=lambda x: x[1])
    dominant_conflict = max(conflict_counts.items(), key=lambda x: x[1])
    if dominant_payoff[1] >= max(4, window // 3):
        warnings.append(f"Payoff repetition risk: {dominant_payoff[0]} used {dominant_payoff[1]} times recently.")
    if dominant_conflict[1] >= max(4, window // 3):
        warnings.append(f"Conflict repetition risk: {dominant_conflict[0]} used {dominant_conflict[1]} times recently.")
    avg_novelty = sum(novelties) / len(novelties) if novelties else None
    avg_hook = sum(hooks) / len(hooks) if hooks else None
    if avg_novelty is not None and avg_novelty < 6:
        warnings.append("Novelty is low across recent chapters.")
    if avg_hook is not None and avg_hook < 6:
        warnings.append("Hook strength is low across recent chapters.")

    return {
        "warnings": warnings,
        "payoff_counts": payoff_counts,
        "conflict_counts": conflict_counts,
        "avg_tension": sum(tensions) / len(tensions) if tensions else None,
        "avg_novelty": avg_novelty,
        "avg_hook": avg_hook,
    }


def structural_repetition_analysis(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    result: dict[str, Any] = {"warnings": [], "repeated_patterns": [], "tension_shape": "unknown"}
    if len(rows) < 6:
        return result

    sequence = [
        (r.get("conflict_type", ""), r.get("payoff_type", ""), r.get("emotional_tone", ""))
        for r in reversed(rows)
    ]

    # Sliding window pattern detection (window size 3)
    seen_patterns: dict[str, int] = {}
    for i in range(len(sequence) - 2):
        pattern_key = "|".join(f"{s[0]},{s[1]}" for s in sequence[i : i + 3])
        seen_patterns[pattern_key] = seen_patterns.get(pattern_key, 0) + 1
    repeated = [(k, v) for k, v in seen_patterns.items() if v >= 2]
    if repeated:
        result["repeated_patterns"] = [k for k, _ in repeated]
        result["warnings"].append(f"Repeated arc patterns detected: {len(repeated)} patterns appear 2+ times")

    # Tension curve shape analysis
    tensions = [int(r.get("tension", 5)) for r in reversed(rows) if r.get("tension") is not None]
    if len(tensions) >= 6:
        diffs = [tensions[i + 1] - tensions[i] for i in range(len(tensions) - 1)]
        flat_count = sum(1 for d in diffs if abs(d) <= 1)
        if flat_count > len(diffs) * 0.7:
            result["tension_shape"] = "flat"
            result["warnings"].append("Tension curve is flat — lacking dramatic variation")
        else:
            rises = sum(1 for d in diffs if d > 0)
            falls = sum(1 for d in diffs if d < 0)
            if rises > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_rise"
            elif falls > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_fall"
                result["warnings"].append("Tension is monotonically falling — reader engagement at risk")
            else:
                result["tension_shape"] = "varied"

    # Resolution monotony: check if emotional_tone repeats
    tones = [r.get("emotional_tone", "") for r in reversed(rows) if r.get("emotional_tone")]
    if len(tones) >= 5:
        tone_counts: dict[str, int] = {}
        for t in tones:
            tone_counts[t] = tone_counts.get(t, 0) + 1
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])
        if dominant_tone[1] >= len(tones) * 0.6:
            result["warnings"].append(f"Emotional monotony: '{dominant_tone[0]}' dominates {dominant_tone[1]}/{len(tones)} chapters")

    return result


def generate_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
) -> list[dict[str, Any]]:
    diagnostics = rhythm_diagnostics(conn, config)
    structural = structural_repetition_analysis(conn, config)
    constraints = get_active_constraints(conn, chapter_num)
    quality_feedback = recent_quality_feedback(paths)
    base_user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(quality_feedback, ensure_ascii=False, indent=2) if quality_feedback else "None"}

## Active Stage Constraints (MUST OBEY)
{json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "None"}

## Previous Chapter Tail
{tail[-2000:]}

## Request
Create candidate plan for chapter {chapter_num}.
Avoid recent repetition. Preserve causal debt. Increase reader follow-up desire."""
    num_candidates = int(config["novel"]["candidate_plans"])
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

    def gen_one(idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        for retry in range(2):
            try:
                raw = call_llm(
                    client,
                    paths,
                    config,
                    CANDIDATE_PLAN_SYSTEM,
                    json_prompt(base_user + f"\n\nCandidate index: {idx}. Use a distinct strategy."),
                    max_tokens=16000,
                    temperature=0.65 + idx * 0.05,
                )
                plan = load_json_with_repair(client, paths, config, raw)
                plan["candidate_index"] = idx
                return plan
            except Exception as exc:
                last_exc = exc
                log(paths, f"Candidate plan {idx} attempt failed retry={retry}: {exc}")
        log(paths, f"Candidate plan {idx} discarded after retries: {last_exc}")
        return {}

    plans: list[dict[str, Any]] = [{}] * num_candidates
    with ThreadPoolExecutor(max_workers=min(max_workers, num_candidates)) as executor:
        futures = {executor.submit(gen_one, idx): idx for idx in range(num_candidates)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                plans[idx] = future.result()
            except Exception as exc:
                log(paths, f"Candidate plan {idx} thread failed: {exc}")
                plans[idx] = {}
    valid = [p for p in plans if p]
    if not valid:
        raise RuntimeError(f"All {num_candidates} candidate plans failed for chapter")
    return valid


def agent_review_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Candidate Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

Review chapter {chapter_num} plan."""
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

    def review_one(agent: str, system: str) -> dict[str, Any]:
        for retry in range(2):
            try:
                raw = call_llm(client, paths, config, system, json_prompt(user), max_tokens=16000, temperature=0.2)
                report = load_json_with_repair(
                    client,
                    paths,
                    config,
                    raw,
                    fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                )
                report["agent"] = agent
                return report
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log(paths, f"Agent {agent} review parse failed retry={retry}: {exc}")
        return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(review_one, agent, system): agent
            for agent, system in AGENT_REVIEW_SYSTEMS.items()
        }
        for future in as_completed(futures):
            reports.append(future.result())

    for report in reports:
        agent = report["agent"]
        if isinstance(conn, JsonStoryStore):
            conn.add_agent_report(chapter_num, agent, report)
        else:
            conn.execute(
                "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    chapter_num,
                    agent,
                    safe_score(report.get("score", 0)),
                    json.dumps(report, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    if not isinstance(conn, JsonStoryStore):
        conn.commit()
    return reports


def review_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    plan_users = []
    diagnostics_json = json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)
    memory = memory_context(paths, conn, config)
    for plan in plans:
        plan_users.append(
            f"""## Memory
{memory}

## Rhythm Diagnostics JSON
{diagnostics_json}

## Candidate Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

Review chapter {chapter_num} plan."""
        )

    max_workers = int(config["novel"].get("max_parallel_workers", 5))
    reports_by_plan: list[list[dict[str, Any]]] = [[] for _ in plans]

    def review_one(plan_index: int, agent: str, system: str) -> dict[str, Any]:
        user = plan_users[plan_index]
        for retry in range(2):
            try:
                raw = call_llm(client, paths, config, system, json_prompt(user), max_tokens=16000, temperature=0.2)
                report = load_json_with_repair(
                    client,
                    paths,
                    config,
                    raw,
                    fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                )
                report["agent"] = agent
                return report
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log(paths, f"Agent {agent} review parse failed plan={plan_index} retry={retry}: {exc}")
        return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(review_one, plan_index, agent, system): (plan_index, agent)
            for plan_index in range(len(plans))
            for agent, system in AGENT_REVIEW_SYSTEMS.items()
        }
        for future in as_completed(futures):
            plan_index, agent = futures[future]
            try:
                reports_by_plan[plan_index].append(future.result())
            except Exception as exc:
                log(paths, f"Agent {agent} review thread failed plan={plan_index}: {exc}")
                reports_by_plan[plan_index].append(
                    {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}
                )

    for reports in reports_by_plan:
        for report in reports:
            agent = report["agent"]
            if isinstance(conn, JsonStoryStore):
                conn.add_agent_report(chapter_num, agent, report)
            else:
                conn.execute(
                    "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        chapter_num,
                        agent,
                        safe_score(report.get("score", 0)),
                        json.dumps(report, ensure_ascii=False),
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
    if not isinstance(conn, JsonStoryStore):
        conn.commit()

    return reports_by_plan


def arbitrate_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    reports_by_plan: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Candidate Plans JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

## Agent Reports JSON
{json.dumps(reports_by_plan, ensure_ascii=False, indent=2)}

Select and improve the best plan for chapter {chapter_num}."""
    raw = call_llm(client, paths, config, ARBITER_SYSTEM, json_prompt(user), max_tokens=8000, temperature=0.25)
    decision = load_json_with_repair(client, paths, config, raw)
    plan = decision.get("merged_plan") or plans[int(decision.get("selected_index", 0))]
    db_event(conn, chapter_num, "plan_arbitration", {"decision": decision, "plans": plans})
    return plan, decision


def plan_score(decision: dict[str, Any], selected_index: int | None = None) -> float:
    scores = decision.get("scores") or []
    if not scores:
        return 0.0
    if selected_index is None:
        selected_index = int(decision.get("selected_index", 0))
    for score in scores:
        if int(score.get("index", -1)) == selected_index:
            return safe_score(score.get("score", 0))
    return safe_score(scores[0].get("score", 0))


def create_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    checkpoint_label: str = "initial",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = load_checkpoint(paths, chapter_num, f"plan_{checkpoint_label}_selected.json")
    if isinstance(cached, dict) and cached.get("plan") and cached.get("decision"):
        log(paths, f"Resuming cached {checkpoint_label} plan Ch{chapter_num}")
        return cached["plan"], cached["decision"]

    best_plan: dict[str, Any] | None = None
    best_decision: dict[str, Any] | None = None
    min_score = float(config["novel"]["min_plan_score"])
    for attempt in range(2):
        log(paths, f"Generating candidate plans Ch{chapter_num} attempt={attempt}")
        plans_key = f"plan_{checkpoint_label}_attempt{attempt}_candidates.json"
        reports_key = f"plan_{checkpoint_label}_attempt{attempt}_reports.json"
        arbitration_key = f"plan_{checkpoint_label}_attempt{attempt}_arbitration.json"

        plans = load_checkpoint(paths, chapter_num, plans_key)
        if isinstance(plans, list) and plans:
            log(paths, f"Resuming cached candidate plans Ch{chapter_num} attempt={attempt}")
        else:
            plans = generate_candidate_plans(client, paths, conn, config, chapter_num, tail)
            save_checkpoint(paths, chapter_num, plans_key, plans)

        reports = load_checkpoint(paths, chapter_num, reports_key)
        if isinstance(reports, list) and reports:
            log(paths, f"Resuming cached agent reports Ch{chapter_num} attempt={attempt}")
        else:
            reports = review_candidate_plans(client, paths, conn, config, chapter_num, plans)
            save_checkpoint(paths, chapter_num, reports_key, reports)

        arbitration = load_checkpoint(paths, chapter_num, arbitration_key)
        if isinstance(arbitration, dict) and arbitration.get("plan") and arbitration.get("decision"):
            log(paths, f"Resuming cached arbitration Ch{chapter_num} attempt={attempt}")
            plan = arbitration["plan"]
            decision = arbitration["decision"]
        else:
            plan, decision = arbitrate_plan(client, paths, conn, config, chapter_num, plans, reports)
            save_checkpoint(paths, chapter_num, arbitration_key, {"plan": plan, "decision": decision})

        score = plan_score(decision)
        log(paths, f"Arbiter selected Ch{chapter_num} plan score={score}")
        best_plan, best_decision = plan, decision
        if score >= min_score:
            break
        db_event(conn, chapter_num, "low_plan_score_retry", {"score": score, "decision": decision})
    assert best_plan is not None and best_decision is not None
    save_checkpoint(
        paths,
        chapter_num,
        f"plan_{checkpoint_label}_selected.json",
        {"plan": best_plan, "decision": best_decision},
    )
    return best_plan, best_decision


def write_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
) -> str:
    title = str(plan.get("title") or f"Chapter {chapter_num}").strip()
    system = WRITE_SYSTEM.format(
        chapter_words=int(config["novel"]["chapter_words"]),
        chapter_num=chapter_num,
        title=title,
    )
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Previous Tail
{tail[-int(config["novel"]["recent_tail_chars"]):]}

## Recent Quality Feedback JSON (MUST REPAIR IN THIS CHAPTER)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Selected Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Arbitration Constraints JSON
{json.dumps(decision.get("required_constraints", []), ensure_ascii=False, indent=2)}

Write chapter {chapter_num}."""
    raw = call_llm(client, paths, config, system, user, temperature=float(config["api"]["temperature"]))
    return normalize_chapter(raw)


def review_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    chapter: str,
    tail: str,
) -> dict[str, Any]:
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Previous Tail
{tail[-1500:]}

## Recent Quality Feedback JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Selected Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Chapter Text
{chapter[:12000]}"""
    raw = call_llm(client, paths, config, REVIEW_SYSTEM, json_prompt(user), max_tokens=16000, temperature=0.2)
    report = load_json_with_repair(
        client,
        paths,
        config,
        raw,
        fallback={
            "score": 5,
            "accepted": False,
            "problems": ["JSON parsing failed; conservative review fallback used."],
            "fixes": [],
            "continuity_risks": [],
            "rhythm_risks": [],
            "reader_fatigue_risks": [],
        },
    )
    report["score"] = safe_score(report.get("score", 0))
    report.setdefault("accepted", report["score"] >= float(config["novel"]["quality_threshold"]))
    return report


def revise_chapter(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter: str,
    review: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    user = f"""## Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Editor Report JSON
{json.dumps(review, ensure_ascii=False, indent=2)}

## Original Chapter
{chapter}

Revise the full chapter."""
    raw = call_llm(client, paths, config, REVISE_SYSTEM, user, temperature=0.45)
    return normalize_chapter(raw)


def extract_events(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
) -> dict[str, Any]:
    user = f"""## Memory Before Chapter
{memory_context(paths, conn, config)}

## Chapter {chapter_num}
{chapter[:12000]}

Extract durable state changes."""
    raw = call_llm(client, paths, config, EXTRACT_SYSTEM, max_tokens=8000, user=json_prompt(user), temperature=0.2)
    return load_json_with_repair(client, paths, config, raw)


def update_structured_state(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    review: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    db_event(conn, chapter_num, "chapter_extraction", extraction)

    for event in extraction.get("events", []):
        db_event(conn, chapter_num, "story_event", event)

    for entity in extraction.get("entities", []):
        entity_type = str(entity.get("entity_type", "unknown"))
        name = str(entity.get("name", "unknown"))
        if isinstance(conn, JsonStoryStore):
            state = conn.get_entity_state(entity_type, name)
        else:
            old = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
            state = json.loads(old["state_json"]) if old else {}
        patch = entity.get("state_patch") or {}
        if isinstance(patch, dict):
            state.update(patch)
        else:
            state["note"] = str(patch)
        if isinstance(conn, JsonStoryStore):
            conn.upsert_entity(entity_type, name, state, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO entities(entity_type, name, state_json, updated_chapter)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_type, name)
                DO UPDATE SET state_json=excluded.state_json, updated_chapter=excluded.updated_chapter
                """,
                (entity_type, name, json.dumps(state, ensure_ascii=False), chapter_num),
            )

    for thread in extraction.get("threads", []):
        thread_id = str(thread.get("id") or f"ch{chapter_num}-{abs(hash(json.dumps(thread, ensure_ascii=False))) % 100000}")
        if isinstance(conn, JsonStoryStore):
            conn.upsert_thread(thread_id, thread, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO open_threads(id, description, status, introduced_chapter, due_chapter, updated_chapter, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id)
                DO UPDATE SET description=excluded.description, status=excluded.status,
                              due_chapter=excluded.due_chapter, updated_chapter=excluded.updated_chapter,
                              payload_json=excluded.payload_json
                """,
                (
                    thread_id,
                    str(thread.get("description", "")),
                    str(thread.get("status", "open")),
                    thread.get("introduced_chapter"),
                    thread.get("due_chapter"),
                    chapter_num,
                    json.dumps(thread.get("payload", {}), ensure_ascii=False),
                ),
            )

    metrics = extraction.get("metrics") or {}
    metrics_row = {
        "chapter": chapter_num,
        "title": extraction.get("title"),
        "score": safe_score(review.get("score", 0)),
        "plan_score": plan_score(decision),
        "payoff_type": metrics.get("payoff_type"),
        "conflict_type": metrics.get("conflict_type"),
        "tension": metrics.get("tension"),
        "novelty": metrics.get("novelty"),
        "hook_strength": metrics.get("hook_strength"),
        "emotional_tone": metrics.get("emotional_tone"),
        "accepted": 1 if review.get("accepted") else 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(conn, JsonStoryStore):
        conn.upsert_metrics(chapter_num, metrics_row)
    else:
        conn.execute(
            """
            INSERT INTO chapter_metrics(
                chapter, title, score, plan_score, payoff_type, conflict_type, tension,
                novelty, hook_strength, emotional_tone, accepted, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
                title=excluded.title, score=excluded.score, plan_score=excluded.plan_score,
                payoff_type=excluded.payoff_type, conflict_type=excluded.conflict_type,
                tension=excluded.tension, novelty=excluded.novelty, hook_strength=excluded.hook_strength,
                emotional_tone=excluded.emotional_tone, accepted=excluded.accepted
            """,
            (
                metrics_row["chapter"],
                metrics_row["title"],
                metrics_row["score"],
                metrics_row["plan_score"],
                metrics_row["payoff_type"],
                metrics_row["conflict_type"],
                metrics_row["tension"],
                metrics_row["novelty"],
                metrics_row["hook_strength"],
                metrics_row["emotional_tone"],
                metrics_row["accepted"],
                metrics_row["created_at"],
            ),
        )
        conn.commit()

    updates = extraction.get("memory_updates") or {}
    append_memory(paths.bible, chapter_num, updates.get("bible") or [])
    append_memory(paths.characters, chapter_num, updates.get("characters") or [])
    append_memory(paths.timeline, chapter_num, updates.get("timeline") or [])
    append_memory(paths.threads, chapter_num, updates.get("threads") or [])

    store_causal_links(conn, chapter_num, extraction.get("causal_links") or [])


def append_memory(path: Path, chapter_num: int, items: list[Any]) -> None:
    if not items:
        return
    append_text(path, f"\n\n## Ch{chapter_num}\n" + "\n".join(f"- {item}" for item in items) + "\n")


def update_state_file(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    extraction: dict[str, Any],
) -> None:
    if paths.state.exists():
        shutil.copy2(paths.state, paths.state.with_suffix(".md.bak"))
    user = f"""## Current State
{read_text(paths.state)}

## Memory Context
{memory_context(paths, conn, config)}

## Extraction JSON
{json.dumps(extraction, ensure_ascii=False, indent=2)}

## Current Total Characters
{count_chars(paths.book)}

## Recent Chapter Text
{chapter[:5000]}

Update state.md after chapter {chapter_num}."""
    new_state = call_llm(client, paths, config, STATE_UPDATE_SYSTEM, user, max_tokens=8000, temperature=0.25)
    write_text(paths.state, normalize_text(new_state) + "\n")


def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    write_text(chapter_path(paths, chapter_num), chapter)
    append_text(paths.book, "\n\n" + chapter)
    append_text(
        paths.logs_dir / "reviews.jsonl",
        json.dumps(
            {
                "chapter": chapter_num,
                "score": review.get("score"),
                "accepted": review.get("accepted"),
                "problems": review.get("problems", []),
                "continuity_risks": review.get("continuity_risks", []),
                "plan_title": plan.get("title"),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )


def stage_review(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> None:
    start = max(1, chapter_num - int(config["novel"]["stage_review_every"]) + 1)
    recent = []
    for num in range(start, chapter_num + 1):
        text = read_text(chapter_path(paths, num))
        if text:
            recent.append(f"## Ch{num}\n{text[:1600]}")
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## Recent Chapters
{chr(10).join(recent)}

Review long-cycle quality through chapter {chapter_num}."""
    review_text = call_llm(client, paths, config, STAGE_REVIEW_SYSTEM, user, max_tokens=8000, temperature=0.3)
    append_text(paths.logs_dir / "stage_reviews.md", f"\n\n# Ch{chapter_num} Stage Review\n\n{normalize_text(review_text)}\n")
    db_event(conn, chapter_num, "stage_review", {"review": normalize_text(review_text)})

    # Extract and store structured constraints from stage review
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", review_text, re.DOTALL)
    if json_match:
        try:
            constraint_data = json.loads(json_match.group(1))
            constraints = constraint_data.get("constraints", [])
            if constraints:
                store_stage_constraints(conn, chapter_num, constraints)
                log(paths, f"Stored {len(constraints)} stage constraints from Ch{chapter_num} review")
        except (json.JSONDecodeError, AttributeError):
            pass


REPLAN_SYSTEM = """You are the strategic replanner for a long-form fiction engine.
The current volume plan has degraded in quality metrics. Analyze the current state,
recent trajectory, open threads, and repetition patterns.
Produce a revised plan for the NEXT 40-60 chapters that:
- Resolves stale or overdue threads
- Introduces new conflict dimensions not seen in recent chapters
- Shifts character dynamics and power relationships
- Avoids patterns flagged in repetition analysis
- Maintains causal consistency with established events
- Increases reader anticipation and follow-up desire
Return the full revised volume_plan markdown only, no explanation."""


def should_replan(conn: Any, config: dict[str, Any]) -> bool:
    rows = recent_metrics(conn, 20)
    if len(rows) < 15:
        return False
    threshold_score = float(config["novel"].get("replan_score_threshold", 6.5))
    threshold_novelty = float(config["novel"].get("replan_novelty_threshold", 5.5))
    triggers = 0
    scores = [safe_score(r.get("score", 7)) for r in rows if r.get("score") is not None]
    novelties = [int(r.get("novelty", 7)) for r in rows if r.get("novelty") is not None]
    if scores and sum(scores) / len(scores) < threshold_score:
        triggers += 1
    if novelties and sum(novelties) / len(novelties) < threshold_novelty:
        triggers += 1
    structural = structural_repetition_analysis(conn, config)
    if len(structural.get("warnings", [])) >= 3:
        triggers += 1
    return triggers >= 2


def adaptive_replan(
    client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any], chapter_num: int
) -> None:
    shutil.copy2(paths.volume_plan, paths.volume_plan.with_suffix(".md.bak"))
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## Open Causal Requirements JSON
{json.dumps(get_open_causal_requirements(conn), ensure_ascii=False, indent=2)}

## Active Constraints JSON
{json.dumps(get_active_constraints(conn, chapter_num), ensure_ascii=False, indent=2)}

Current chapter: {chapter_num}. Replan the next 40-60 chapters."""
    new_plan = call_llm(client, paths, config, REPLAN_SYSTEM, user, max_tokens=16000, temperature=0.5)
    write_text(paths.volume_plan, normalize_text(new_plan) + "\n")
    db_event(conn, chapter_num, "adaptive_replan", {"reason": "metrics_degradation"})


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
            if safe_score(review.get("score", 0)) > safe_score(best_review.get("score", 0)):
                best_chapter = chapter
                best_review = dict(review)
            if safe_score(review.get("score", 0)) >= threshold and review.get("accepted", True):
                review["accepted"] = True
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


if __name__ == "__main__":
    main()
