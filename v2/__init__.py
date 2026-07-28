"""REDESIGN v2 engine — a deterministic pipeline with four model calls.

See `docs/REDESIGN_V2.md`. Selected per-novel with `novel.engine: v2`; v1 remains
the default until an A/B settles it. Nothing here is imported by the v1 path, so
this package can be built, measured, and deleted without touching a running book.
"""
