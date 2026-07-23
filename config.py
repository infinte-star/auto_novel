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
    compass: Path
    voices: Path
    voice: Path
    contract: Path
    glossary: Path
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

def genre_detection_profile(preset: str) -> dict[str, Any]:
    """Genre-specific defaults for the deterministic detection gates.

    The recent 爽文-centric gates (opening/length/flat-streak/payoff-cadence/
    下沉 register) would misfire on slow-burn / high-threshold genres (悬疑/历史)。
    Rather than scattering `if preset == ...` across every gate, this returns the
    detection-relevant config defaults for a genre, keyed by `style_preset`.
    Applied in load_config with "fill only if the user did NOT set the key"
    semantics (mirrors planning.py's `visual_payoff_blocks_plan` override pattern),
    so explicit config.yaml values always win and existing novels are unaffected.

    opening_gate_mode ∈ {crisis, clue, relationship, balanced} steers
    quality.opening_hook_gate's notion of a valid opening per genre.
    """
    p = (preset or "").strip().lower()
    # 爽文族基线：高频爽点 / 短章 / 低门槛 / 黄金三句危机开场 / 物证兑现仅建议。
    shuang = {
        "narrative_mode": "serial",
        "opening_gate_mode": "crisis",
        "payoff_density_min": 0.5,            # ≤2 章一个强爽点
        "flat_streak_gate_enabled": True,
        "flat_chapters_max_consecutive": 3,
        "chapter_words": 2800,
        "chapter_min_chars": 2200,
        "chapter_max_chars": 3600,
        "length_band_penalty_enabled": True,
        "style_low_barrier_register": True,
        "style_min_avg_sentence_chars": 12.0,
        # 反过度书写锚点（碎句塌缩的镜像）：句长上限 / 对话占比下限 / 伪技术腔。
        # 阈值由离线回放校准（experiments/replay_style_health.py）：健康对话书
        # 跑 8-18% 对话占比；塌缩章 0.5-3% + 黑话 ≥12/k。
        "style_max_avg_sentence_chars": 38.0,
        "style_dialogue_ratio_min": 0.05,
        "style_tech_jargon_per_kchar_warn": 8.0,
        "style_tech_jargon_per_kchar_bad": 12.0,
        "visual_payoff_blocks_plan": False,
        "emotional_cadence_max_same": 3,
        "fatigue_words": ["冷笑", "蝼蚁", "倒吸凉气", "瞳孔骤缩", "不可思议", "震惊",
                          "竟然", "仿佛", "宛如", "不禁", "微微一笑", "嘴角微扬", "眼中闪过一丝"],
        "paragraph_cv_min": 0.12,
        "short_paragraph_warn": 45,
        "short_paragraph_severe": 30,
        "dialogue_pingpong_warn": 0.55,
    }
    profiles: dict[str, dict[str, Any]] = {
        "xuanhuan_shuang": dict(shuang),
        "system_stream": dict(shuang),
        "urban_ability": dict(shuang),
        "wanzu_xuanhuan": {**shuang, "payoff_density_min": 0.4,
                           "chapter_words": 3000, "chapter_min_chars": 2400,
                           "chapter_max_chars": 4200,
                           "style_max_avg_sentence_chars": 40.0,
                           "style_dialogue_ratio_min": 0.04,
                           "fatigue_words": ["冷笑", "蝼蚁", "倒吸凉气", "瞳孔骤缩", "不可思议",
                                             "震惊", "竟然", "仿佛", "宛如", "不禁", "微微一笑",
                                             "嘴角微扬", "眼中闪过一丝", "气血翻涌", "杀意凛然"]},
        # 悬疑：慢烧、线索开场、复杂容忍、高阅读门槛、物证兑现 block。
        "suspense": {
            "narrative_mode": "reasoning",
            "opening_gate_mode": "clue",
            "payoff_density_min": 0.25,        # ≤4 章
            "flat_streak_gate_enabled": True,
            "flat_chapters_max_consecutive": 5,
            "chapter_words": 3500,
            "chapter_min_chars": 2800,
            "chapter_max_chars": 6000,
            "length_band_penalty_enabled": True,
            "style_low_barrier_register": False,
            "style_min_avg_sentence_chars": 14.0,
            "style_max_avg_sentence_chars": 48.0,
            "style_dialogue_ratio_min": 0.03,
            "style_tech_jargon_per_kchar_warn": 10.0,
            "style_tech_jargon_per_kchar_bad": 14.0,
            "visual_payoff_blocks_plan": True,
            "emotional_cadence_max_same": 4,
            "fatigue_words": ["毛骨悚然", "不寒而栗", "头皮发麻", "鸡皮疙瘩", "心跳加速",
                              "仿佛", "不禁", "宛如", "竟然", "一股寒意"],
            "paragraph_cv_min": 0.18,
            "short_paragraph_warn": 55,
            "short_paragraph_severe": 35,
            "dialogue_pingpong_warn": 0.45,
        },
        # 历史厚重：慢热、长章、关闭连续平路闸门、厚重长句。
        "history": {
            "narrative_mode": "balanced",
            "opening_gate_mode": "balanced",
            "payoff_density_min": 0.2,         # ≤5 章
            "flat_streak_gate_enabled": False,
            "flat_chapters_max_consecutive": 4,
            "chapter_words": 3500,
            "chapter_min_chars": 3000,
            "chapter_max_chars": 6000,
            "length_band_penalty_enabled": True,
            "style_low_barrier_register": False,
            "style_min_avg_sentence_chars": 16.0,
            "style_max_avg_sentence_chars": 52.0,
            "style_dialogue_ratio_min": 0.02,
            "style_tech_jargon_per_kchar_warn": 8.0,
            "visual_payoff_blocks_plan": False,
            "emotional_cadence_max_same": 4,
            "fatigue_words": ["不禁", "仿佛", "宛如", "竟然", "微微颔首", "眼中闪过一丝"],
            "paragraph_cv_min": 0.18,
            "short_paragraph_warn": 55,
            "short_paragraph_severe": 35,
            "dialogue_pingpong_warn": 0.45,
        },
        # 女频言情：情绪弧主导、关系开场、情绪变奏最严。
        "romance_female": {
            "narrative_mode": "serial",
            "opening_gate_mode": "relationship",
            "payoff_density_min": 0.34,        # ≤3 章
            "flat_streak_gate_enabled": True,
            "flat_chapters_max_consecutive": 3,
            "chapter_words": 2800,
            "chapter_min_chars": 2200,
            "chapter_max_chars": 4000,
            "length_band_penalty_enabled": True,
            "style_low_barrier_register": True,
            "style_min_avg_sentence_chars": 12.0,
            "style_max_avg_sentence_chars": 38.0,
            "style_dialogue_ratio_min": 0.08,
            "style_tech_jargon_per_kchar_warn": 6.0,
            "style_tech_jargon_per_kchar_bad": 10.0,
            "visual_payoff_blocks_plan": False,
            "emotional_cadence_max_same": 2,
            "fatigue_words": ["不禁", "仿佛", "宛如", "竟然", "心跳加速", "脸颊微红", "呼吸一滞"],
        },
        # 规则怪谈/民俗无限流：悬疑推理内核 + 抖音快节奏。线索开场、物证兑现门（规则/破法
        # 必须落到可见后果），节奏比纯悬疑快，保留怪谈冷叙事（不下沉）。
        "rule_horror": {
            "narrative_mode": "reasoning",
            "opening_gate_mode": "clue",
            "payoff_density_min": 0.4,          # ≤2.5 章（比纯悬疑 0.25 快，贴抖音爽感）
            "flat_streak_gate_enabled": True,
            "flat_chapters_max_consecutive": 4,
            "chapter_words": 3000,
            "chapter_min_chars": 2400,
            "chapter_max_chars": 5000,
            "length_band_penalty_enabled": True,
            "style_low_barrier_register": False,   # 保留怪谈冷叙事，不强加下沉
            "style_min_avg_sentence_chars": 13.0,
            "style_max_avg_sentence_chars": 44.0,
            "style_dialogue_ratio_min": 0.04,
            "style_tech_jargon_per_kchar_warn": 10.0,
            "style_tech_jargon_per_kchar_bad": 14.0,
            "visual_payoff_blocks_plan": True,     # 规则怪谈命脉：规则真伪/破法要落到物证与可见后果
            "emotional_cadence_max_same": 4,
            "fatigue_words": ["毛骨悚然", "不寒而栗", "头皮发麻", "仿佛", "不禁", "宛如",
                              "竟然", "一股寒意", "浑身发冷"],
        },
    }
    # 别名：规则流 / 无限流 指向同一 profile。
    profiles["guize"] = profiles["infinite_flow"] = profiles["rule_horror"]
    # 未知/未设置题材 → 中性默认（不强加爽文短章/下沉）。
    neutral = {**shuang, "narrative_mode": "balanced", "opening_gate_mode": "balanced",
               "chapter_words": 3000, "chapter_min_chars": 2400, "chapter_max_chars": 4500,
               "style_low_barrier_register": False, "style_min_avg_sentence_chars": 13.0,
               "style_max_avg_sentence_chars": 42.0, "style_dialogue_ratio_min": 0.04,
               "style_tech_jargon_per_kchar_warn": 8.0,
               "style_tech_jargon_per_kchar_bad": 12.0,
               "fatigue_words": ["仿佛", "不禁", "宛如", "竟然"],
               "short_paragraph_warn": 50, "short_paragraph_severe": 30,
               "dialogue_pingpong_warn": 0.50}
    return dict(profiles.get(p, neutral))


def _apply_genre_detection_profile(config: dict[str, Any]) -> None:
    """Fill genre-appropriate detection defaults into config['novel'].

    Only fills keys the user did NOT explicitly set (so config.yaml always wins
    and existing novels are unaffected). Driven by style_preset; runs BEFORE the
    required-key check so it can supply chapter_words etc. for templates that omit
    them and leave the genre to decide.
    """
    novel = config.get("novel")
    if not isinstance(novel, dict):
        return
    preset = str(novel.get("style_preset", "") or "").strip().lower()
    profile = genre_detection_profile(preset)
    for key, value in profile.items():
        if key not in novel:
            novel[key] = value


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

    # Genre-aware detection defaults (爽文/悬疑/言情/…), filled by style_preset
    # BEFORE the required check so genre can supply chapter_words etc. Only fills
    # keys the user did not set, so explicit config.yaml values always win.
    _apply_genre_detection_profile(config)

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

    # Per-role model routing: each role (review, planning, writing, extraction)
    # can have its own base_url + model. When *_base_url is set, *_model is
    # mandatory so a half-configured role fails loudly instead of sending an
    # empty model name to the provider.
    api = config.get("api", {})
    for _role in ("review", "planning", "writing", "extraction"):
        _role_base = str(api.get(f"{_role}_base_url", "")).strip()
        if _role_base and not str(api.get(f"{_role}_model", "")).strip():
            raise ValueError(
                f"api.{_role}_base_url is set but api.{_role}_model is missing. "
                f"Either set api.{_role}_model or remove api.{_role}_base_url."
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
    endpoints, primary_count, _models = configured_api_endpoints_with_models(config)
    return endpoints, primary_count


def configured_api_endpoints_with_models(
    config: dict[str, Any],
) -> tuple[list[tuple[str, str]], int, list[str | None]]:
    """Like configured_api_endpoints, but also returns a per-endpoint model name.

    api.api_key_groups items may carry an OPTIONAL third pipe-delimited field
    naming the model for that endpoint:  ``base_url|key1,key2|model_name``.
    When present, every key in that group is tagged with that model so the
    client pool sends the right model name on fallback (e.g. the primary
    100xlabs endpoint speaks claude-opus-4-8 while a mimo fallback speaks
    mimo-v2.5-pro). When absent (or for the primary api_key/api_keys fallback),
    the model is ``None`` and call_llm uses the global api.model.

    Ordering / primary boundary:
      - When api.api_key_groups is NON-empty, the LEGACY layout is preserved:
        every group endpoint is primary (rotated first), and base_url/api_key
        are the final fallback. This keeps the historical mimo-multi-endpoint
        configs unchanged.
      - api.primary_base_url (optional) overrides which base_url marks the
        primary boundary: endpoints whose base_url equals it are primary, the
        rest are fallback (order preserved). Use this to make ONE endpoint the
        sole primary (e.g. 100xlabs) while mimo endpoints sit behind it.
    """
    api = config["api"]
    endpoints: list[tuple[str, str]] = []
    models: list[str | None] = []
    seen: set[tuple[str, str]] = set()

    groups = str(api.get("api_key_groups", "")).strip()
    if groups:
        for group in groups.split(";"):
            group = group.strip()
            if not group:
                continue
            if "|" not in group:
                raise ValueError("Invalid api.api_key_groups item, expected base_url|key1,key2")
            parts = group.split("|")
            base_url = parts[0].strip()
            keys_text = parts[1] if len(parts) > 1 else ""
            group_model = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            for key in re.split(r"[,\s]+", keys_text):
                key = key.strip()
                if not base_url or not key:
                    continue
                endpoint = (base_url, key)
                if endpoint not in seen:
                    seen.add(endpoint)
                    endpoints.append(endpoint)
                    models.append(group_model)

    group_count = len(endpoints)
    fallback_base_url = str(api["base_url"]).strip()
    for key in configured_api_keys(config):
        endpoint = (fallback_base_url, key)
        if endpoint not in seen:
            seen.add(endpoint)
            endpoints.append(endpoint)
            models.append(None)

    # primary_base_url lets one endpoint be the sole primary; all endpoints
    # whose base_url matches it become primary, the rest fallback. Order is
    # preserved within each bucket. When unset, fall back to the legacy rule:
    # group endpoints primary, base_url/api_key fallback.
    primary_base_url = str(api.get("primary_base_url", "")).strip()
    if primary_base_url:
        primary_idx = [i for i, (url, _) in enumerate(endpoints) if url == primary_base_url]
        fallback_idx = [i for i, (url, _) in enumerate(endpoints) if url != primary_base_url]
        order = primary_idx + fallback_idx
        endpoints = [endpoints[i] for i in order]
        models = [models[i] for i in order]
        primary_count = len(primary_idx) or len(endpoints)
    else:
        primary_count = group_count or len(endpoints)

    return endpoints, primary_count, models

def configured_role_endpoints(config: dict[str, Any], role: str) -> list[tuple[str, str]]:
    """Endpoints for a role-specific model (review, planning, writing, extraction).

    Reads api.{role}_base_url, api.{role}_api_key, api.{role}_keys.
    Returns [(base_url, key), ...] or [] when the role's base_url is absent.
    """
    api = config["api"]
    base_url = str(api.get(f"{role}_base_url", "")).strip()
    if not base_url:
        return []
    keys: list[str] = []
    primary = str(api.get(f"{role}_api_key", "")).strip()
    if primary:
        keys.append(primary)
    extra = str(api.get(f"{role}_keys", "")).strip()
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


def configured_review_endpoints(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Backward-compatible wrapper: endpoints for the reviewer model."""
    return configured_role_endpoints(config, "review")

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
        compass=ROOT / str(raw.get("compass",
            str(Path(str(raw.get("volume_plan", "memory/volume_plan.md"))).parent / "compass.md"))),
        voices=ROOT / str(raw.get("voices", "memory/voices.md")),
        voice=ROOT / str(raw.get("voice", "memory/voice.md")),
        contract=ROOT / str(raw.get("contract", str(Path(str(raw.get("voice", "memory/voice.md"))).parent / "contract.md"))),
        glossary=ROOT / str(raw.get("glossary", str(Path(str(raw.get("voice", "memory/voice.md"))).parent / "glossary.md"))),
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
    try:
        print(line)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(line.encode("utf-8", errors="replace").decode("ascii", errors="replace"))
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
    text = re.sub(r"^\s*```(?:analysis|thinking|reasoning)\s*.*?```\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
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


def ending_zone_distance(config: dict[str, Any], chapter_num: int) -> int | None:
    """Chapters remaining until the finale when inside the gradual收束 zone, else None.

    The last 5 chapters are every book's weakest region (fossils peak, threads
    pile up unpaid, and CLOSING_RULES only fires on the single final chapter — too
    late). This returns max_chapters - chapter_num (>=1) when within
    `ending_zone_chapters` of the end, so the writer/planner can RAMP convergence
    instead of slamming closure into one chapter. Returns None on the final
    chapter itself (is_final_chapter owns that) and in pure char-target mode.
    Gated by `ending_aware`.
    """
    if not bool(config["novel"].get("ending_aware", True)):
        return None
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    if max_chapters <= 0:
        return None
    zone = int(config["novel"].get("ending_zone_chapters", 5))
    remaining = max_chapters - chapter_num
    if 1 <= remaining < zone:
        return remaining
    return None


def cost_savings_disabled(config: dict[str, Any], chapter_num: int) -> bool:
    """True when token-saving accept shortcuts must be suppressed for this chapter.

    In the收尾 zone (the final chapter and the `ending_zone_chapters` leading up to
    it) quality's marginal value is highest — a mediocre finale poisons the whole
    book — so the "accept a below-threshold plan/chapter to save cost" branches
    (planning.py plan-score shortcut, pipeline.py ROI breaker) should NOT fire.
    Gated by `ending_zone_disables_cost_savings` (default True); returns False in
    pure char-target mode where there is no deterministic finale.
    """
    if not bool(config["novel"].get("ending_zone_disables_cost_savings", True)):
        return False
    return is_final_chapter(config, chapter_num) or ending_zone_distance(config, chapter_num) is not None

# Valid narrative-mode identifiers. `reasoning` = single-room / precise物证 mode
# (strengthens closure, fair clues, concrete physical anchors); `serial` =
# strong-hook / emotional / serializable mode (relaxes per-chapter closure,
# strengthens hooks & emotional outburst). `balanced` keeps prior behaviour.
NARRATIVE_MODES = ("balanced", "reasoning", "serial")

def narrative_mode(config: dict[str, Any]) -> str:
    """Return the configured narrative mode, defaulting to 'balanced'.

    Driven by `novel.narrative_mode`. Unknown/empty values fall back to
    'balanced' so an unset config behaves exactly as before this feature.
    """
    raw = str(config.get("novel", {}).get("narrative_mode", "") or "").strip().lower()
    return raw if raw in NARRATIVE_MODES else "balanced"


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


def book_is_consistent(paths: Paths) -> bool:
    """Cheap check that book.md already reflects every saved chapter file.

    `save_chapter` builds book.md incrementally by appending each chapter, so in
    the normal (non-corrupt) case book.md is already complete and the O(n)
    `rebuild_book` (glob + read + sort + rewrite of the whole multi-MB file) is
    pure waste on the resume path. This guard lets the caller skip rebuild when
    book.md is demonstrably consistent with chapters/.

    Conservative by design: ANY doubt (missing book.md, can't read a chapter,
    last chapter's body not found in book.md) returns False so the caller falls
    back to a full rebuild. It never reports a stale/short book as consistent.

    Verification is intentionally lightweight (no full O(n) concat compare):
      1. book.md exists and is non-empty.
      2. The highest-numbered chapter file's stripped body is a substring of
         book.md — i.e. the most recent append landed. A truncated/older book.md
         (the only state rebuild actually needs to fix on resume) fails here.
    """
    if not paths.book.exists():
        return False
    if not paths.chapters_dir.exists():
        # No chapters at all -> nothing to rebuild from; treat as consistent.
        return True
    chapter_files = [p for p in paths.chapters_dir.glob("*.md") if p.stem.isdigit()]
    if not chapter_files:
        return True
    try:
        book_text = read_text(paths.book)
    except OSError:
        return False
    if not book_text.strip():
        return False
    last_file = max(chapter_files, key=lambda p: int(p.stem))
    try:
        last_body = read_text(last_file).strip()
    except OSError:
        return False
    if not last_body:
        # An empty last chapter file is odd; rebuild to be safe.
        return False
    return last_body in book_text
