from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# PROMPT_FILE / CONFIG_FILE default to the root-level files (the long novel) so
# `python run.py` behaves exactly as before. A separate novel run (e.g.
# run_fusu.py) sets NOVEL_PROMPT / NOVEL_CONFIG env vars BEFORE importing any
# module that imports config, redirecting both to its own files.
PROMPT_FILE = ROOT / os.environ.get("NOVEL_PROMPT", "prompt.md")
CONFIG_FILE = ROOT / os.environ.get("NOVEL_CONFIG", "config.yaml")

@dataclass(frozen=True)
class Paths:
    book: Path
    state: Path
    title: Path
    bible: Path
    characters: Path
    timeline: Path
    threads: Path
    volume_plan: Path
    voices: Path
    voice: Path
    contract: Path
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
            "recent_tail_chars",
            "stage_review_every",
            "repeat_window",
            "fatigue_window",
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
    _validate_config(config)
    return config

# (section, key) -> validation spec. Caught at load time so a typo (e.g.
# `temperature: 0,8` parsed as the string "0,8") fails loudly here instead of
# crashing N chapters later with an opaque TypeError far from the root cause.
# `int_like`/`float_like` mean the value must coerce; `min`/`max` bound it.
_NUMERIC_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("api", "max_tokens"): {"kind": "int", "min": 1},
    ("api", "temperature"): {"kind": "float", "min": 0.0, "max": 2.0},
    ("novel", "chapter_words"): {"kind": "int", "min": 1},
    ("novel", "target_words"): {"kind": "int", "min": 1},
    ("novel", "quality_threshold"): {"kind": "float", "min": 0.0, "max": 10.0},
    ("novel", "max_revision_rounds"): {"kind": "int", "min": 0},
    ("novel", "candidate_plans"): {"kind": "int", "min": 1},
    ("novel", "min_plan_score"): {"kind": "float", "min": 0.0, "max": 10.0},
}

def _validate_config(config: dict[str, Any]) -> None:
    for (section_name, key), spec in _NUMERIC_SPECS.items():
        if section_name not in config or key not in config[section_name]:
            continue
        raw = config[section_name][key]
        kind = spec["kind"]
        try:
            value = int(raw) if kind == "int" else float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Config value {section_name}.{key} must be {kind}, got {raw!r}. "
                f"(config.yaml is a YAML subset — check for stray quotes/commas.)"
            ) from None
        low = spec.get("min")
        high = spec.get("max")
        if low is not None and value < low:
            raise ValueError(f"Config value {section_name}.{key}={value} is below minimum {low}")
        if high is not None and value > high:
            raise ValueError(f"Config value {section_name}.{key}={value} is above maximum {high}")
        config[section_name][key] = value

    # Optional integer knobs: validate only if present and non-empty.
    for section_name, key, minimum in [
        ("novel", "max_chapters", 0),
        ("novel", "max_parallel_workers", 1),
        ("novel", "candidate_chapters", 1),
        ("api", "max_attempts", 1),
    ]:
        if section_name in config and key in config[section_name]:
            raw = config[section_name][key]
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Config value {section_name}.{key} must be an integer, got {raw!r}."
                ) from None
            if value < minimum:
                raise ValueError(f"Config value {section_name}.{key}={value} is below minimum {minimum}")
            config[section_name][key] = value

    # Optional float knobs: validate only if present and non-empty.
    for section_name, key, lo, hi in [
        ("novel", "plan_candidate_temp_base", 0.0, 2.0),
        ("novel", "plan_candidate_temp_step", 0.0, 1.0),
        ("novel", "prewrite_dimension_floor", 0.0, 10.0),
    ]:
        if section_name in config and key in config[section_name]:
            raw = config[section_name][key]
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Config value {section_name}.{key} must be a float, got {raw!r}."
                ) from None
            if value < lo or value > hi:
                raise ValueError(
                    f"Config value {section_name}.{key}={value} is out of range [{lo}, {hi}]"
                )
            config[section_name][key] = value

    # Optional reviewer routing (main writer = primary model, reviewer = a
    # separate model+endpoint). All review_* keys are optional; if review_base_url
    # is set, review_model becomes mandatory so a half-configured reviewer fails
    # loudly here instead of sending an empty model name to the provider.
    api = config.get("api", {})
    review_base_url = str(api.get("review_base_url", "")).strip()
    if review_base_url and not str(api.get("review_model", "")).strip():
        raise ValueError(
            "api.review_base_url is set but api.review_model is missing. "
            "Either set api.review_model or remove api.review_base_url."
        )

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

def configured_review_endpoints(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Endpoints for the separate reviewer model (main writer = primary model).

    Returns [(base_url, key), ...] built from api.review_base_url plus
    api.review_api_key and api.review_keys (comma/semicolon/space separated).
    Returns [] when review_base_url is not configured, so the engine keeps
    routing every call through the primary model (backward compatible).
    """
    api = config["api"]
    base_url = str(api.get("review_base_url", "")).strip()
    if not base_url:
        return []
    keys: list[str] = []
    primary = str(api.get("review_api_key", "")).strip()
    if primary:
        keys.append(primary)
    extra = str(api.get("review_keys", "")).strip()
    if extra:
        keys.extend(k for k in re.split(r"[,;\s]+", extra) if k)

    endpoints: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key in keys:
        endpoint = (base_url, key)
        if endpoint not in seen:
            seen.add(endpoint)
            endpoints.append(endpoint)
    return endpoints

def get_paths(config: dict[str, Any]) -> Paths:
    raw = config["paths"]
    return Paths(
        book=ROOT / str(raw["book"]),
        state=ROOT / str(raw["state"]),
        title=ROOT / str(raw.get("title", "title.txt")),
        bible=ROOT / str(raw["bible"]),
        characters=ROOT / str(raw["characters"]),
        timeline=ROOT / str(raw["timeline"]),
        threads=ROOT / str(raw["threads"]),
        volume_plan=ROOT / str(raw["volume_plan"]),
        voices=ROOT / str(raw.get("voices", "memory/voices.md")),
        voice=ROOT / str(raw.get("voice", "memory/voice.md")),
        contract=ROOT / str(raw.get("contract", str(Path(str(raw.get("voice", "memory/voice.md"))).parent / "contract.md"))),
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
    text = normalize_text(text)
    # The writer prompt instructs the model to keep its pre-writing self-review
    # in reasoning_content, but providers sometimes return it inline in content
    # as an <analysis>…</analysis> (or ```analysis``` / "## Pre-writing…") block
    # before the real prose. Strip any such leading meta block so it never gets
    # saved as chapter text. We only remove a leading block (before the first
    # "第N章" title line) to avoid touching legitimate prose.
    text = re.sub(r"^\s*<analysis\b[^>]*>.*?</analysis>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<thinking\b[^>]*>.*?</thinking>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<details\b[^>]*>.*?</details>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<reasoning\b[^>]*>.*?</reasoning>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Heading-style leaked review: drop everything up to the first 第N章 title
    # line when a self-review heading precedes it.
    m = re.search(r"(?m)^\s*(#{0,6}\s*)?(第\s*[0-9零一二三四五六七八九十百千]+\s*章)", text)
    if m and m.start() > 0:
        head = text[: m.start()]
        if re.search(r"(写前自我审查|Pre-writing|Self-Review|highest risk|reasoning|分析[:：])", head, re.IGNORECASE):
            text = text[m.start():]
    # LLM sometimes emits markdown heading for the title line
    # ("# 第N章 …" instead of "第N章 …"). Strip it so the title format
    # stays consistent across chapters.
    text = re.sub(r"^#{1,6}\s+", "", text)
    return text + "\n"

def count_chars(path: Path) -> int:
    return len(read_text(path))

def book_reached_target(path: Path, target_chars: int) -> bool:
    """Return True when `path` holds at least `target_chars` characters.

    Hot path: this is polled on every main-loop iteration (and per prefetch)
    against book.md, which grows to multiple MB for a long novel. A full
    `count_chars` would re-read and UTF-8-decode the whole file each time.
    Instead we first look at the on-disk byte size: a UTF-8 file can never
    contain more characters than it has bytes, so when `getsize < target` the
    book definitively has fewer than `target` chars and we skip the read. Only
    once the byte size could plausibly meet the target do we pay for one exact
    `count_chars`. For CJK text (~3 bytes/char) the byte size stays well above
    the char count for almost the entire run, so the expensive read happens
    only in the final stretch.
    """
    if target_chars <= 0:
        return True
    try:
        if path.stat().st_size < target_chars:
            return False
    except OSError:
        return False
    return count_chars(path) >= target_chars

def is_final_chapter(config: dict[str, Any], chapter_num: int) -> bool:
    """True when chapter_num is the deterministic final chapter of the book.

    Only meaningful in short-novel mode where `max_chapters` is set. In pure
    char-target mode (max_chapters absent/0) there is no deterministic finale,
    so this always returns False and the engine's per-chapter behaviour is
    unchanged. Gated by `ending_aware` (default True).
    """
    if not bool(config["novel"].get("ending_aware", True)):
        return False
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    return max_chapters > 0 and chapter_num == max_chapters

def tail_text(path: Path, n_chars: int) -> str:
    text = read_text(path)
    return text[-n_chars:] if len(text) > n_chars else text

def chapter_path(paths: Paths, chapter_num: int) -> Path:
    return paths.chapters_dir / f"{chapter_num:04d}.md"

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

def ensure_project(paths: Paths) -> None:
    paths.chapters_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    for path, title in [
        (paths.bible, "World Bible"),
        (paths.characters, "Characters"),
        (paths.timeline, "Timeline"),
        (paths.threads, "Threads"),
        (paths.volume_plan, "Volume Plan"),
        (paths.voices, "Character Voices"),
        (paths.voice, "Narrative Voice Anchor"),
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
