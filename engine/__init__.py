"""Core engine — deterministic decision table with four LLM actions.

Architecture: plan an arc once every ~10 chapters, write the chapter and its
state delta in ONE call, check canon via acceptance gates, and repair.
Every routing predicate is a pure function over recorded state.

Modules:
    config      — YAML-subset config parser, paths, ROOT, text I/O
    llm         — LLMClientPool, call_llm, provider abstraction
    store       — chapter/metrics persistence, fingerprinting
    checkpoint  — checkpoint read/write for crash recovery
    retrieval   — embedding-free retrieval, chapter indexing
    loop        — decision table, acceptance, StoryState projection, repair
    plan        — arc-level card planning, card vocabulary, validation
    write       — prose doctrine, one-call write+delta, persistence
    quality     — gate registry, active gates, L0/L1 repair ladder
    bootstrap   — one-time novel initialization from prompt.md
    anchor      — external blinded pairwise prose judge
"""
