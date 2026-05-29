from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROMPT_FILE = ROOT / "prompt.md"
CONFIG_FILE = ROOT / "config.yaml"

@dataclass(frozen=True)
class Paths:
    book: Path
    state: Path
    bible: Path
    characters: Path
    timeline: Path
    threads: Path
    volume_plan: Path
    voices: Path
    voice: Path
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
        voices=ROOT / str(raw.get("voices", "memory/voices.md")),
        voice=ROOT / str(raw.get("voice", "memory/voice.md")),
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
