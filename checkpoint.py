from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Paths, chapter_path, log, read_text, write_text

CHECKPOINT_VERSION = 2
CHAPTER_CURRENT_CHECKPOINT = f"chapter_current_v{CHECKPOINT_VERSION}.md"

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

def should_resume_existing_chapter(paths: Paths, chapter_num: int) -> bool:
    if chapter_num <= 0 or not chapter_path(paths, chapter_num).exists():
        return False
    if not checkpoint_dir(paths, chapter_num).exists():
        return False
    if checkpoint_path(paths, chapter_num, "chapter_completed.json").exists():
        return False
    return True
