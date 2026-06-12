"""Global cross-novel telemetry repository.

Every novel runs as an isolated OS process and keeps its own story_state.db;
this module is the ONE shared sink where all books' quality signals get
aggregated so the engine can learn ACROSS books instead of restarting from
zero on every new novel:

  - chapter_metrics   per-chapter review scores (copied from each book's db)
  - events            raw cross-book events (plan_arbitration / cold_reader /
                      panel_report / quality_debt ...)
  - strategy_outcomes plan_arbitration flattened to one row per candidate
                      strategy, so the planning bandit can read a cross-book
                      prior with a single indexed query (no JSON parsing)
  - revise_pairs      before/after revision text + review verdicts — natural
                      preference pairs for future fine-tuning

Design constraints (load-bearing):
  * NEVER block or break the generation pipeline. Every public writer returns
    silently on any failure (db missing, locked, disk full, ...). Telemetry is
    strictly an observer.
  * Multiple novel processes write concurrently -> WAL + busy_timeout, plus
    one retry, then drop the row. Losing a telemetry row is acceptable;
    stalling a chapter is not.
  * Zero new dependencies; sqlite3 only (and if sqlite3 is unavailable this
    module degrades to a no-op).
  * All writes are INSERT OR REPLACE against composite primary keys so both
    live double-writes and `novel.py telemetry backfill` are idempotent.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ROOT, parse_scalar, safe_score

try:
    import sqlite3
except ModuleNotFoundError:  # pragma: no cover - mirrors store.py fallback
    sqlite3 = None  # type: ignore[assignment]

TELEMETRY_DIR = ROOT / "telemetry"
TELEMETRY_DB = TELEMETRY_DIR / "global.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS chapter_metrics (
    novel_name TEXT NOT NULL,
    genre TEXT NOT NULL DEFAULT '_default',
    chapter INTEGER NOT NULL,
    title TEXT,
    score REAL,
    readthrough_score REAL,
    hook_score REAL,
    payoff_score REAL,
    novelty_score REAL,
    prose_score REAL,
    continuity_score REAL,
    plan_score REAL,
    hook_strength REAL,
    accepted INTEGER,
    em_dash_per_kchar REAL,
    style_penalty REAL,
    created_at TEXT,
    PRIMARY KEY (novel_name, chapter)
);
CREATE TABLE IF NOT EXISTS events (
    novel_name TEXT NOT NULL,
    genre TEXT NOT NULL DEFAULT '_default',
    chapter INTEGER NOT NULL,
    kind TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    payload TEXT,
    created_at TEXT,
    PRIMARY KEY (novel_name, chapter, kind, seq)
);
CREATE TABLE IF NOT EXISTS strategy_outcomes (
    novel_name TEXT NOT NULL,
    genre TEXT NOT NULL DEFAULT '_default',
    chapter INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    score REAL,
    selected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    PRIMARY KEY (novel_name, chapter, strategy)
);
CREATE INDEX IF NOT EXISTS idx_strategy_genre ON strategy_outcomes(genre, strategy);
CREATE TABLE IF NOT EXISTS revise_pairs (
    novel_name TEXT NOT NULL,
    genre TEXT NOT NULL DEFAULT '_default',
    chapter INTEGER NOT NULL,
    round INTEGER NOT NULL,
    text_before TEXT,
    review_json TEXT,
    text_after TEXT,
    score_before REAL,
    score_after REAL,
    created_at TEXT,
    PRIMARY KEY (novel_name, chapter, round)
);
"""

_METRIC_COLUMNS = (
    "title", "score", "readthrough_score", "hook_score", "payoff_score",
    "novelty_score", "prose_score", "continuity_score", "plan_score",
    "hook_strength", "accepted", "em_dash_per_kchar", "style_penalty",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> Any:
    """Open (and lazily initialize) the global telemetry DB.

    Returns None on ANY failure so callers can silently degrade. A fresh
    connection per call keeps this module trivially thread-safe: record_*
    runs from BackgroundTasks daemon threads in N independent novel
    processes simultaneously.
    """
    if sqlite3 is None:
        return None
    try:
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(TELEMETRY_DB, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn
    except Exception:
        return None


def _execute_with_retry(sql: str, params: tuple) -> bool:
    """Run one write with a single retry; swallow all errors (observer role)."""
    for attempt in range(2):
        conn = _connect()
        if conn is None:
            return False
        try:
            conn.execute(sql, params)
            conn.commit()
            return True
        except Exception:
            if attempt == 0:
                time.sleep(0.3)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return False


# ----------------------------------------------------------------------------
# live double-write API (called from the pipeline, always non-fatal)
# ----------------------------------------------------------------------------
def record_chapter_metrics(novel: str, genre: str, chapter: int, metrics_row: dict[str, Any]) -> bool:
    try:
        values = [novel, genre or "_default", int(chapter)]
        for col in _METRIC_COLUMNS:
            values.append(metrics_row.get(col))
        values.append(metrics_row.get("created_at") or _now())
        cols = "novel_name, genre, chapter, " + ", ".join(_METRIC_COLUMNS) + ", created_at"
        marks = ", ".join("?" for _ in values)
        return _execute_with_retry(
            f"INSERT OR REPLACE INTO chapter_metrics({cols}) VALUES ({marks})",
            tuple(values),
        )
    except Exception:
        return False


def record_event(novel: str, genre: str, chapter: int, kind: str, payload: dict[str, Any], seq: int = 0) -> bool:
    try:
        return _execute_with_retry(
            "INSERT OR REPLACE INTO events(novel_name, genre, chapter, kind, seq, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (novel, genre or "_default", int(chapter), str(kind), int(seq),
             json.dumps(payload, ensure_ascii=False), _now()),
        )
    except Exception:
        return False


def record_arbitration(novel: str, genre: str, chapter: int, decision: dict[str, Any], plans: list[Any]) -> bool:
    """Store the raw arbitration event AND flatten it into strategy_outcomes.

    strategy_outcomes is what the cross-book bandit prior reads; keeping it
    flat (one row per candidate strategy) makes the prior a single indexed
    GROUP BY instead of a JSON-parsing scan over events.
    """
    ok = record_event(novel, genre, chapter, "plan_arbitration", {"decision": decision, "plans": plans})
    try:
        decision = decision or {}
        plans = plans or []
        sel_idx = int(decision.get("selected_index", 0) or 0)
        score_map: dict[int, float] = {}
        for s in decision.get("scores") or []:
            if isinstance(s, dict):
                try:
                    score_map[int(s.get("index", -1))] = safe_score(s.get("score", 0))
                except (TypeError, ValueError):
                    continue
        for i, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            strat = str(plan.get("strategy") or "").strip()
            if not strat:
                continue
            _execute_with_retry(
                "INSERT OR REPLACE INTO strategy_outcomes"
                "(novel_name, genre, chapter, strategy, score, selected, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (novel, genre or "_default", int(chapter), strat,
                 float(score_map.get(i, 5.0)), 1 if i == sel_idx else 0, _now()),
            )
    except Exception:
        pass
    return ok


def record_revise_pair(
    novel: str,
    genre: str,
    chapter: int,
    round_num: int,
    text_before: str,
    review: dict[str, Any] | None,
    text_after: str,
    score_before: float | None,
    score_after: float | None,
) -> bool:
    try:
        return _execute_with_retry(
            "INSERT OR REPLACE INTO revise_pairs"
            "(novel_name, genre, chapter, round, text_before, review_json, text_after,"
            " score_before, score_after, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (novel, genre or "_default", int(chapter), int(round_num),
             str(text_before or ""), json.dumps(review or {}, ensure_ascii=False),
             str(text_after or ""), score_before, score_after, _now()),
        )
    except Exception:
        return False


# ----------------------------------------------------------------------------
# cross-book prior (read path for the planning bandit)
# ----------------------------------------------------------------------------
def global_strategy_history(genre: str, exclude_novel: str | None = None) -> dict[str, dict[str, float]]:
    """Aggregate per-strategy stats across all books of the same genre.

    Returns the same shape as planning._strategy_history:
    {strategy: {"trials": N, "score_sum": X, "wins": K}}. When the genre
    bucket has no data the whole library is used as fallback so a brand-new
    genre still benefits from generic narrative priors. Any failure returns
    {} (the bandit then behaves exactly as before this feature existed).
    """
    conn = _connect()
    if conn is None:
        return {}
    try:
        def _query(where_genre: bool) -> dict[str, dict[str, float]]:
            sql = (
                "SELECT strategy, COUNT(*) AS trials, SUM(score) AS score_sum,"
                " SUM(selected) AS wins FROM strategy_outcomes WHERE 1=1"
            )
            params: list[Any] = []
            if where_genre:
                sql += " AND genre = ?"
                params.append(genre or "_default")
            if exclude_novel:
                sql += " AND novel_name != ?"
                params.append(exclude_novel)
            sql += " GROUP BY strategy"
            rows = conn.execute(sql, params).fetchall()
            return {
                str(r["strategy"]): {
                    "trials": float(r["trials"] or 0),
                    "score_sum": float(r["score_sum"] or 0.0),
                    "wins": float(r["wins"] or 0),
                }
                for r in rows
            }

        stats = _query(where_genre=True)
        if not stats:
            stats = _query(where_genre=False)
        return stats
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# backfill (novel.py telemetry backfill)
# ----------------------------------------------------------------------------
def _read_novel_config(novel_dir: Path) -> dict[str, dict[str, Any]]:
    """Minimal reader for a novel's config.yaml (same YAML-subset as config.py).

    We cannot call config.load_config() here: it is hard-bound to the
    NOVEL_CONFIG env var captured at import time, while backfill iterates
    over MANY novels in one process."""
    config: dict[str, dict[str, Any]] = {}
    path = novel_dir / "config.yaml"
    if not path.exists():
        return config
    section: str | None = None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return config
    for raw_line in text.splitlines():
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
    return config


def _unwrap_checkpoint(path: Path) -> Any:
    """Read a checkpoint JSON, unwrapping the {_checkpoint_version, payload} envelope."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and "_checkpoint_version" in data:
        return data.get("payload")
    return data


def _iter_source_rows(novel_dir: Path, cfg: dict[str, dict[str, Any]]):
    """Yield ("metrics"|"event", row_dict) from the book's own SQLite store."""
    db_rel = str((cfg.get("paths") or {}).get("database") or "")
    db_path = (ROOT / db_rel) if db_rel else (novel_dir / "story_state.db")
    if db_path.exists():
        try:
            src = sqlite3.connect(db_path, timeout=10)
            src.row_factory = sqlite3.Row
            try:
                for r in src.execute("SELECT * FROM chapter_metrics"):
                    yield "metrics", dict(r)
                for r in src.execute(
                    "SELECT id, chapter, event_type, payload, created_at FROM events"
                    " WHERE event_type IN ('plan_arbitration','cold_reader','quality_debt','panel_report')"
                ):
                    yield "event", dict(r)
            finally:
                src.close()
        except Exception:
            pass


def backfill_novel(novel_dir: Path) -> dict[str, int]:
    """Idempotently import one novel's historical data into the global DB.

    Returns row counts per table (counting attempted upserts)."""
    novel = novel_dir.name
    cfg = _read_novel_config(novel_dir)
    genre = str((cfg.get("novel") or {}).get("genre") or "_default")
    counts = {"chapter_metrics": 0, "events": 0, "strategy_outcomes": 0, "revise_pairs": 0}

    for kind, row in _iter_source_rows(novel_dir, cfg):
        if kind == "metrics":
            try:
                chapter = int(row.get("chapter", 0) or 0)
            except (TypeError, ValueError):
                continue
            if chapter <= 0:
                continue
            if record_chapter_metrics(novel, genre, chapter, row):
                counts["chapter_metrics"] += 1
            continue
        # events
        payload = row.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"raw": payload}
        if not isinstance(payload, dict):
            continue
        try:
            chapter = int(row.get("chapter", 0) or 0)
        except (TypeError, ValueError):
            continue
        etype = str(row.get("event_type"))
        seq = int(row.get("id", 0) or 0)
        if etype == "plan_arbitration":
            decision = payload.get("decision") or {}
            plans = payload.get("plans") or []
            if record_event(novel, genre, chapter, etype, payload, seq=seq):
                counts["events"] += 1
            # flatten without re-storing the raw event twice
            before = counts["strategy_outcomes"]
            for i, plan in enumerate(plans):
                if not isinstance(plan, dict):
                    continue
                strat = str(plan.get("strategy") or "").strip()
                if not strat:
                    continue
                score_map = {
                    int(s.get("index", -1)): safe_score(s.get("score", 0))
                    for s in (decision.get("scores") or [])
                    if isinstance(s, dict)
                }
                sel_idx = int(decision.get("selected_index", 0) or 0)
                if _execute_with_retry(
                    "INSERT OR REPLACE INTO strategy_outcomes"
                    "(novel_name, genre, chapter, strategy, score, selected, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (novel, genre, chapter, strat, float(score_map.get(i, 5.0)),
                     1 if i == sel_idx else 0, str(row.get("created_at") or _now())),
                ):
                    counts["strategy_outcomes"] += 1
            _ = before
        else:
            if record_event(novel, genre, chapter, etype, payload, seq=seq):
                counts["events"] += 1

    # revise pairs from checkpoints/ch####/
    logs_rel = str((cfg.get("paths") or {}).get("logs_dir") or "")
    logs_dir = (ROOT / logs_rel) if logs_rel else (novel_dir / "logs")
    ckpt_root = logs_dir / "checkpoints"
    if ckpt_root.exists():
        for chdir in sorted(ckpt_root.glob("ch[0-9][0-9][0-9][0-9]")):
            try:
                chapter = int(chdir.name[2:])
            except ValueError:
                continue
            for revised in sorted(chdir.glob("chapter_revised_round*.md")):
                try:
                    round_num = int(revised.stem.replace("chapter_revised_round", ""))
                except ValueError:
                    continue
                prev_review_p = chdir / f"review_round{round_num - 1}.json"
                this_review_p = chdir / f"review_round{round_num}.json"
                prev_review = _unwrap_checkpoint(prev_review_p) if prev_review_p.exists() else None
                this_review = _unwrap_checkpoint(this_review_p) if this_review_p.exists() else None
                if not isinstance(prev_review, dict):
                    prev_review = None
                if not isinstance(this_review, dict):
                    this_review = None
                # before-text: the previous round's revised file, or empty for round 1
                before_p = chdir / f"chapter_revised_round{round_num - 1}.md"
                try:
                    text_before = before_p.read_text(encoding="utf-8") if before_p.exists() else ""
                except OSError:
                    text_before = ""
                try:
                    text_after = revised.read_text(encoding="utf-8")
                except OSError:
                    continue
                score_before = safe_score(prev_review.get("score", 0)) if prev_review else None
                score_after = safe_score(this_review.get("score", 0)) if this_review else None
                if record_revise_pair(
                    novel_dir.name, genre, chapter, round_num,
                    text_before, prev_review, text_after, score_before, score_after,
                ):
                    counts["revise_pairs"] += 1
    return counts


# ----------------------------------------------------------------------------
# stats (novel.py telemetry stats)
# ----------------------------------------------------------------------------
def stats(genre: str | None = None) -> str:
    conn = _connect()
    if conn is None:
        return "[telemetry] global DB unavailable."
    lines: list[str] = []
    try:
        where = ""
        params: list[Any] = []
        if genre:
            where = " WHERE genre = ?"
            params.append(genre)
        totals = conn.execute(
            f"SELECT COUNT(DISTINCT novel_name) AS novels, COUNT(*) AS chapters,"
            f" AVG(score) AS avg_score FROM chapter_metrics{where}",
            params,
        ).fetchone()
        lines.append(
            f"[telemetry] novels={totals['novels']} chapters={totals['chapters']}"
            f" avg_score={(totals['avg_score'] or 0):.2f}"
            + (f" (genre={genre})" if genre else "")
        )
        lines.append("")
        header = f"{'GENRE':<16} {'STRATEGY':<20} {'TRIALS':>7} {'AVG':>6} {'WIN%':>6}"
        lines.append(header)
        lines.append("-" * len(header))
        rows = conn.execute(
            f"SELECT genre, strategy, COUNT(*) AS trials, AVG(score) AS avg_score,"
            f" AVG(selected) * 100.0 AS win_pct FROM strategy_outcomes{where}"
            f" GROUP BY genre, strategy ORDER BY genre, win_pct DESC",
            params,
        ).fetchall()
        if not rows:
            lines.append("(no strategy outcomes yet — run `novel.py telemetry backfill`)")
        for r in rows:
            lines.append(
                f"{str(r['genre']):<16} {str(r['strategy']):<20} {int(r['trials']):>7}"
                f" {(r['avg_score'] or 0):>6.2f} {(r['win_pct'] or 0):>5.0f}%"
            )
        counts = {}
        for table in ("events", "revise_pairs"):
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
            except Exception:
                counts[table] = 0
        lines.append("")
        lines.append(f"[telemetry] events={counts['events']} revise_pairs={counts['revise_pairs']}")
        return "\n".join(lines)
    except Exception as exc:
        return f"[telemetry] stats failed: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass
