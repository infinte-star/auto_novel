"""Shared test fixtures.

Import the builders you need — each test file adds its own defaults via
keyword overrides, so the shared layer stays minimal.
"""
from __future__ import annotations

from pathlib import Path

from engine.config import Paths


def make_paths(root: Path, *, seed_files: bool = False) -> Paths:
    """Standard Paths builder with all entries under *root*.

    If *seed_files* is True, writes minimal placeholder content into
    ``bible`` and ``characters`` (required by some writing-path tests).
    """
    p = Paths(
        book=root / "book.md", state=root / "state.md", title=root / "title.txt",
        bible=root / "b.md", characters=root / "c.md", timeline=root / "t.md",
        threads=root / "th.md", volume_plan=root / "vp.md", compass=root / "cp.md",
        voices=root / "vs.md", voice=root / "v.md", contract=root / "ct.md",
        glossary=root / "g.md", chapters_dir=root / "chapters",
        logs_dir=root / "logs", database=root / "story_state.db",
    )
    p.logs_dir.mkdir(parents=True, exist_ok=True)
    if seed_files:
        p.bible.write_text("世界设定", encoding="utf-8")
        p.characters.write_text("人物表", encoding="utf-8")
    return p


def make_config(**novel_overrides) -> dict:
    """Minimal config skeleton.

    Callers supply their own ``novel`` defaults via keyword args::

        cfg = make_config(style_penalty_block=2.0, fix_l0_enabled=True)
    """
    novel: dict = {}
    novel.update(novel_overrides)
    return {"novel": novel, "api": {"metrics_enabled": False}}
