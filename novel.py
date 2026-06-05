"""Universal AI novel-writing framework — multi-novel launcher / manager.

Each novel lives in its own directory `novels/<name>/` containing prompt.md,
config.yaml, book.md, chapters/, memory/, logs/, story_state.db. Every novel
runs as an independent OS process, so multiple novels can be written
simultaneously without sharing the engine's process-level global state
(config.PROMPT_FILE, memory._CACHEABLE_PREFIX_CACHE, etc.).

Subcommands:
    python novel.py create <name>            scaffold novels/<name>/ from templates
    python novel.py trial <name>             generate opening trial variants without touching chapters/book
    python novel.py script --input PATH      convert any novel text file into a 短剧 screenplay
    python novel.py script <name> --chapters A-B   convert chapters A..B of novels/<name>/
    python novel.py run <name>               run the pipeline (background, detached)
    python novel.py run <name> --foreground  run in the current console
    python novel.py list                     list all novels + progress + running state
    python novel.py stop <name>              kill the running process for one novel
    python novel.py restart <name>           stop + relaunch (resumes from checkpoint)

A `run <name>` process is identified by the literal marker "novel.py run <name>"
on its command line, so stop/restart target exactly one novel and never touch
another novel's process.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
NOVELS_DIR = PROJECT_DIR / "novels"
CONFIG_TEMPLATE = PROJECT_DIR / "config_template.yaml"
PROMPT_TEMPLATE = PROJECT_DIR / "prompt_template.md"
PLACEHOLDER = "__NOVEL__"

# Prefer the project venv python (which has `openai` installed) for detached
# background launches, mirroring restart.bat. The current interpreter may be a
# bundled python (e.g. LibreOffice) lacking dependencies. Override with the
# NOVEL_PYTHON env var if needed.
_VENV_PYTHON = Path(r"E:\pycharmproject\allvenv\novel\Scripts\python.exe")


def _launch_python() -> str:
    override = os.environ.get("NOVEL_PYTHON", "").strip()
    if override:
        return override
    if _VENV_PYTHON.exists():
        return str(_VENV_PYTHON)
    return sys.executable


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _is_windows() -> bool:
    return os.name == "nt"


def novel_dir(name: str) -> Path:
    return NOVELS_DIR / name


def _validate_name(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise SystemExit(f"[novel] invalid novel name: {name!r}")


# ----------------------------------------------------------------------------
# create
# ----------------------------------------------------------------------------
def cmd_create(name: str) -> int:
    _validate_name(name)
    target = novel_dir(name)
    if target.exists():
        print(f"[novel] ERROR: {target} already exists; refusing to overwrite.")
        return 2
    if not CONFIG_TEMPLATE.exists():
        print(f"[novel] ERROR: template not found: {CONFIG_TEMPLATE}")
        return 2

    (target / "chapters").mkdir(parents=True, exist_ok=True)
    (target / "memory").mkdir(parents=True, exist_ok=True)
    (target / "logs").mkdir(parents=True, exist_ok=True)

    config_text = CONFIG_TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, name)
    (target / "config.yaml").write_text(config_text, encoding="utf-8")

    prompt_text = (
        PROMPT_TEMPLATE.read_text(encoding="utf-8")
        if PROMPT_TEMPLATE.exists()
        else "# 小说设定\n\n（请填写小说的类型、核心命题、主角、约束、卷纲等）\n"
    )
    (target / "prompt.md").write_text(prompt_text, encoding="utf-8")

    print(f"[novel] created {target}")
    print(f"[novel]   config:  {target / 'config.yaml'}")
    print(f"[novel]   prompt:  {target / 'prompt.md'}  <-- fill this in before running")
    print(f"[novel] next: edit prompt.md, then `python novel.py run {name}`")
    return 0


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def _run_inprocess(name: str) -> int:
    """Run the pipeline for <name> in THIS process.

    config.py reads NOVEL_CONFIG / NOVEL_PROMPT at import time and memory.py
    captures PROMPT_FILE at its own import, so these env vars MUST be set before
    importing pipeline (same ordering constraint as the legacy run_fusu.py).
    """
    target = novel_dir(name)
    config_path = target / "config.yaml"
    prompt_path = target / "prompt.md"
    if not config_path.exists():
        print(f"[novel] ERROR: {config_path} not found. Run `python novel.py create {name}` first.")
        return 2
    _write_pid_file(name, os.getpid(), "foreground")
    # Pass paths relative to PROJECT_DIR (== config.ROOT) so config.ROOT joins
    # resolve correctly regardless of the launching cwd.
    os.environ["NOVEL_CONFIG"] = str(config_path.relative_to(PROJECT_DIR))
    os.environ["NOVEL_PROMPT"] = str(prompt_path.relative_to(PROJECT_DIR))

    try:
        from pipeline import main  # noqa: E402  (must import after env vars are set)

        main()
        return 0
    finally:
        _remove_pid_file(name, os.getpid())


def _set_novel_env(name: str) -> tuple[Path, Path]:
    target = novel_dir(name)
    config_path = target / "config.yaml"
    prompt_path = target / "prompt.md"
    if not config_path.exists():
        print(f"[novel] ERROR: {config_path} not found. Run `python novel.py create {name}` first.")
        raise SystemExit(2)
    os.environ["NOVEL_CONFIG"] = str(config_path.relative_to(PROJECT_DIR))
    os.environ["NOVEL_PROMPT"] = str(prompt_path.relative_to(PROJECT_DIR))
    return config_path, prompt_path


def cmd_trial(name: str, variants: int | None, chapters: int | None) -> int:
    _validate_name(name)
    _set_novel_env(name)
    from trial import run_opening_trial  # noqa: E402

    out = run_opening_trial(variants=variants, chapters=chapters)
    print(f"[novel] opening trial complete: {out}")
    print(f"[novel] best route: {out / 'best_opening_route.md'}")
    return 0


def cmd_adopt_trial(name: str, trial_id: str | None) -> int:
    _validate_name(name)
    target = novel_dir(name)
    trials_dir = target / "logs" / "opening_trials"
    if not trials_dir.exists():
        print(f"[novel] ERROR: no opening_trials found for '{name}'. Run `python novel.py trial {name}` first.")
        return 2
    if trial_id:
        trial_root = trials_dir / trial_id
    else:
        trials = sorted((p for p in trials_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        if not trials:
            print(f"[novel] ERROR: no trial directories found under {trials_dir}")
            return 2
        trial_root = trials[0]
    summary_path = trial_root / "summary.json"
    best_md_path = trial_root / "best_opening_route.md"
    if not summary_path.exists() or not best_md_path.exists():
        print(f"[novel] ERROR: incomplete trial output: {trial_root}")
        return 2
    memory_dir = target / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    adopted = memory_dir / "opening_route.md"
    adopted.write_text(best_md_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        (memory_dir / "opening_route.json").write_text(
            json.dumps(summary.get("best_variant", summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    print(f"[novel] adopted trial route: {trial_root}")
    print(f"[novel] wrote: {adopted}")
    return 0


def cmd_script(
    name: str | None,
    input_path: str | None,
    chapters: str | None,
    out: str | None,
    seg_chars: int | None,
    temperature: float | None,
) -> int:
    """Convert novel text into a 短剧 screenplay.

    Three input modes:
      * --input PATH                         convert any text/markdown file (standalone)
      * <name> --chapters A-B                convert chapters A..B of novels/<name>/
      * <name>                               convert the whole novels/<name>/book.md
    """
    from screenplay import convert_file, convert_text

    # Standalone file mode (no novel name needed).
    if input_path:
        src = Path(input_path)
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        out_path = Path(out).resolve() if out else None
        try:
            result = convert_file(src, out_path, seg_chars=seg_chars, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
            print(f"[novel] script conversion failed: {exc}")
            return 3
        print(f"[novel] screenplay written: {result}")
        return 0

    # Per-novel mode.
    if not name:
        print("[novel] ERROR: provide --input PATH, or a novel <name> (optionally with --chapters A-B).")
        return 2
    _validate_name(name)
    _set_novel_env(name)  # sets NOVEL_CONFIG/NOVEL_PROMPT before importing config-bound code
    target = novel_dir(name)

    if chapters:
        text = _gather_chapter_text(target, chapters)
        if text is None:
            return 2
        default_out = target / "scripts" / f"script_ch{chapters.replace(' ', '')}.md"
        label = f"chapters {chapters}"
    else:
        book = target / "book.md"
        if not book.exists():
            print(f"[novel] ERROR: {book} not found; specify --chapters or --input instead.")
            return 2
        text = book.read_text(encoding="utf-8", errors="replace")
        default_out = target / "scripts" / "script_book.md"
        label = "book.md"

    if not text.strip():
        print(f"[novel] ERROR: no text to convert ({label}).")
        return 2

    out_path = Path(out).resolve() if out else default_out
    import config as _config

    # _set_novel_env set NOVEL_CONFIG, but config.py captured CONFIG_FILE at import;
    # refresh it so load_config()/get_paths() read this novel's config.
    _config.CONFIG_FILE = _config.ROOT / os.environ.get("NOVEL_CONFIG", "config.yaml")
    config = _config.load_config()
    paths = _config.get_paths(config)
    try:
        result = convert_text(
            text,
            config=config,
            paths=paths,
            out_path=out_path,
            seg_chars=seg_chars,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[novel] script conversion failed: {exc}")
        return 3
    print(f"[novel] screenplay written: {result}")
    return 0


def _parse_chapter_range(spec: str) -> tuple[int, int] | None:
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
    else:
        lo_s = hi_s = spec
    try:
        lo, hi = int(lo_s), int(hi_s)
    except ValueError:
        return None
    if lo <= 0 or hi < lo:
        return None
    return lo, hi


def _gather_chapter_text(target: Path, chapters: str) -> str | None:
    rng = _parse_chapter_range(chapters)
    if rng is None:
        print(f"[novel] ERROR: bad --chapters range {chapters!r}; use e.g. 1-3 or 5.")
        return None
    lo, hi = rng
    chapters_dir = target / "chapters"
    if not chapters_dir.exists():
        print(f"[novel] ERROR: {chapters_dir} not found.")
        return None
    parts: list[str] = []
    for n in range(lo, hi + 1):
        path = chapters_dir / f"{n:04d}.md"
        if path.exists():
            chunk = path.read_text(encoding="utf-8", errors="replace").strip()
            if chunk:
                parts.append(chunk)
    if not parts:
        print(f"[novel] ERROR: no chapter files found in {chapters_dir} for range {lo}-{hi}.")
        return None
    return "\n\n".join(parts)


def _benchmark_root() -> Path:
    return PROJECT_DIR / "benchmarks"


def cmd_benchmark_list(platform: str | None, style: str | None) -> int:
    root = _benchmark_root()
    if platform:
        root = root / platform
    if style:
        root = root / style
    if not root.exists():
        print(f"[benchmark] no samples under {root}")
        return 0
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".json", ".md", ".txt"} and not p.name.lower().startswith("readme.")
    )
    if not files:
        print(f"[benchmark] no samples under {root}")
        return 0
    for p in files:
        rel = p.relative_to(_benchmark_root())
        print(f"{rel}\t{p.stat().st_size} bytes")
    return 0


def cmd_benchmark_add(platform: str, style: str, source: str, title: str | None) -> int:
    src = Path(source)
    if not src.exists() or not src.is_file():
        print(f"[benchmark] ERROR: source file not found: {src}")
        return 2
    out_dir = _benchmark_root() / platform / style
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (title or src.stem)).strip("_") or src.stem
    if src.suffix.lower() == ".json":
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[benchmark] ERROR: invalid json: {exc}")
            return 2
        data.setdefault("title", title or src.stem)
        out = out_dir / f"{safe_name}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        text = src.read_text(encoding="utf-8", errors="replace")
        out = out_dir / f"{safe_name}.json"
        payload = {
            "title": title or src.stem,
            "summary": "",
            "opening": "",
            "chapter_1": "",
            "chapter_3": "",
            "payoff_pattern": "",
            "notes": text[:4000],
            "source_note": f"Imported structural notes from {src.name}; keep this as summary/analysis, not full copyrighted prose.",
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[benchmark] added: {out}")
    return 0


def _run_marker(name: str) -> str:
    """The command-line substring used to find/kill this novel's process."""
    return f"novel.py run {name}"


def _pid_file(name: str) -> Path:
    return novel_dir(name) / "logs" / "run.pid"


def _write_pid_file(name: str, pid: int, mode: str) -> None:
    path = _pid_file(name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pid": int(pid),
                    "name": name,
                    "mode": mode,
                    "project": str(PROJECT_DIR),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _remove_pid_file(name: str, pid: int | None = None) -> None:
    path = _pid_file(name)
    if not path.exists():
        return
    if pid is not None:
        data = _read_pid_file(name)
        if data and data.get("pid") != pid:
            return
    try:
        path.unlink()
    except OSError:
        pass


def _read_pid_file(name: str) -> dict[str, object] | None:
    path = _pid_file(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("name") != name:
        return None
    if str(data.get("project", "")) != str(PROJECT_DIR):
        return None
    try:
        pid = int(data.get("pid", 0))
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    data["pid"] = pid
    return data


def cmd_run(name: str, foreground: bool) -> int:
    _validate_name(name)
    target = novel_dir(name)
    if not (target / "config.yaml").exists():
        print(f"[novel] ERROR: {target / 'config.yaml'} not found. Run `python novel.py create {name}` first.")
        return 2

    if foreground:
        return _run_inprocess(name)

    # Background mode: re-launch `python novel.py run <name> --foreground` detached.
    log_dir = target / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "run.log"
    python = _launch_python()
    entry = str(PROJECT_DIR / "novel.py")

    if _is_windows():
        # Start-Process establishes a real detached process the OS won't reap
        # when this launcher exits (mirrors restart.py's approach).
        # Emit ONLY the child's PID (the .Id property) so we never mis-scrape it
        # from the Get-Process-style table that `-PassThru` prints by default.
        # The old table parser grabbed the first all-digit column (Handles),
        # writing a bogus small PID into run.pid; `stop` then trusted that PID,
        # killed the wrong/nonexistent process, and left the real worker alive.
        ps_command = (
            "(Start-Process "
            f"-FilePath '{python}' "
            f"-ArgumentList '-u','{entry}','run','{name}','--foreground' "
            f"-WorkingDirectory '{PROJECT_DIR}' "
            f"-RedirectStandardOutput '{run_log}' "
            f"-RedirectStandardError '{log_dir / 'runner_stderr.log'}' "
            "-WindowStyle Hidden -PassThru).Id"
        )
        ps_args = ["powershell", "-NoProfile", "-Command", ps_command]
        try:
            out = subprocess.check_output(ps_args, text=True, encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as exc:
            print(f"[novel] PowerShell Start-Process failed: {exc}")
            return 3
        pid_val = "unknown"
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pid_val = line
                break
        try:
            with open(run_log, "ab", buffering=0) as fp:
                fp.write(
                    f"\n\n========== novel run {name} @ "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} (PID {pid_val}) ==========\n".encode()
                )
        except Exception:
            pass
        try:
            _write_pid_file(name, int(pid_val), "background")
        except (TypeError, ValueError):
            pass
        print(f"[novel] '{name}' started PID={pid_val}")
        print(f"[novel] tailing log: {run_log}")
        return 0

    # POSIX: Popen with start_new_session detaches the child.
    log_fp = open(run_log, "ab", buffering=0)
    try:
        log_fp.write(
            f"\n\n========== novel run {name} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n".encode()
        )
        proc = subprocess.Popen(
            [python, "-u", entry, "run", name, "--foreground"],
            cwd=str(PROJECT_DIR),
            stdin=subprocess.DEVNULL, stdout=log_fp, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
        _write_pid_file(name, proc.pid, "background")
        print(f"[novel] '{name}' started PID={proc.pid}")
        print(f"[novel] tailing log: {run_log}")
    finally:
        log_fp.close()
    return 0


# ----------------------------------------------------------------------------
# process discovery (adapted from restart.py)
# ----------------------------------------------------------------------------
def _windows_python_pids_with_cmdline() -> list[tuple[int, str]]:
    """Return (pid, cmdline) for every python.exe / pythonw.exe on Windows."""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where",
             "name='python.exe' or name='pythonw.exe'",
             "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        out = ""

    results: list[tuple[int, str]] = []
    if out:
        current_cmd = ""
        for raw in out.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("commandline="):
                current_cmd = line.split("=", 1)[1].strip()
            elif line.lower().startswith("processid="):
                try:
                    pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    current_cmd = ""
                    continue
                results.append((pid, current_cmd))
                current_cmd = ""
        return results

    # PowerShell fallback (Win11 24H2 dropped WMIC).
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter "
            "\"name='python.exe' or name='pythonw.exe'\" | "
            "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return results

    for raw in out.splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        pid_s, cmd = line.split("\t", 1)
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        results.append((pid, cmd))
    return results


def _all_python_pids_with_cmdline() -> list[tuple[int, str]]:
    if _is_windows():
        return _windows_python_pids_with_cmdline()
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,command"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    results: list[tuple[int, str]] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid_s, cmd = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        results.append((pid, cmd))
    return results


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if _is_windows():
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def find_novel_pids(name: str) -> list[int]:
    """PIDs whose command line runs `novel.py run <name>` for THIS project."""
    project_root = str(PROJECT_DIR).lower().replace("\\", "/")
    self_pid = os.getpid()
    pids: list[int] = []
    pid_data = _read_pid_file(name)
    if pid_data:
        pid = int(pid_data["pid"])
        if pid != self_pid and _pid_exists(pid):
            pids.append(pid)
        elif not _pid_exists(pid):
            _remove_pid_file(name, pid)
    for pid, cmd in _all_python_pids_with_cmdline():
        if pid == self_pid:
            continue
        cmd_norm = cmd.lower().replace("\\", "/")
        if "novel.py" not in cmd_norm:
            continue
        # Match "run <name>" as separate argv tokens so "run foo" != "run foobar".
        if not _cmd_runs_novel(cmd_norm, name.lower()):
            continue
        # Confine to this project: either the path appears, or it's a bare launch
        # (cwd was the project dir).
        if (project_root in cmd_norm or _looks_like_local_launch(cmd_norm)) and pid not in pids:
            pids.append(pid)
    return pids


def _cmd_runs_novel(cmd_norm: str, name_lower: str) -> bool:
    """True iff cmd_norm contains the tokens: run <name_lower>."""
    tokens = cmd_norm.replace('"', " ").replace("'", " ").split()
    for i in range(len(tokens) - 1):
        if tokens[i] == "run" and tokens[i + 1] == name_lower:
            return True
    return False


def _looks_like_local_launch(cmd_norm: str) -> bool:
    idx = cmd_norm.find("novel.py")
    if idx == -1:
        return False
    before = cmd_norm[idx - 1] if idx > 0 else " "
    return before in (" ", '"', "'")


def kill_pids(pids: list[int]) -> None:
    for pid in pids:
        print(f"[novel] killing PID {pid} ...")
        try:
            if _is_windows():
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode != 0:
                    subprocess.run(
                        [
                            "powershell",
                            "-NoProfile",
                            "-Command",
                            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            else:
                os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except Exception as exc:
            print(f"[novel] kill PID {pid} failed: {exc}")
    if pids and not _is_windows():
        time.sleep(1.0)
        for pid in pids:
            try:
                os.kill(pid, 0)
                os.kill(pid, 9)
                print(f"[novel] SIGKILL sent to surviving PID {pid}")
            except ProcessLookupError:
                pass
            except Exception:
                pass


def cmd_stop(name: str) -> int:
    _validate_name(name)
    pids = find_novel_pids(name)
    if not pids:
        print(f"[novel] no running process found for '{name}'.")
        _remove_pid_file(name)
        return 0
    kill_pids(pids)
    # Verify the kill landed. taskkill /T may miss grandchildren if the worker
    # already re-parented, and a wrong pid file could have hidden survivors; do a
    # fresh scan and re-kill anything still matching so `stop` never leaves a
    # worker running while reporting success.
    survivors = find_novel_pids(name)
    if survivors:
        print(f"[novel] {len(survivors)} process(es) survived first kill; retrying ...")
        kill_pids(survivors)
        survivors = find_novel_pids(name)
    _remove_pid_file(name)
    if survivors:
        print(f"[novel] WARNING: could not kill PID(s) {survivors} for '{name}'.")
        return 1
    return 0


def cmd_restart(name: str, foreground: bool, wait: float) -> int:
    _validate_name(name)
    pids = find_novel_pids(name)
    if pids:
        kill_pids(pids)
        _remove_pid_file(name)
        if wait > 0:
            print(f"[novel] waiting {wait}s for '{name}' to drain ...")
            time.sleep(wait)
    return cmd_run(name, foreground=foreground)


# ----------------------------------------------------------------------------
# list
# ----------------------------------------------------------------------------
def _count_chars(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8")) if path.exists() else 0
    except OSError:
        return 0


def _last_chapter(chapters_dir: Path) -> int:
    if not chapters_dir.exists():
        return 0
    nums = [int(p.stem) for p in chapters_dir.glob("*.md") if p.stem.isdigit()]
    return max(nums) if nums else 0


def _read_title(nd: Path, max_len: int = 20) -> str:
    path = nd / "title.txt"
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            t = line.strip()
            if t:
                return t if len(t) <= max_len else t[:max_len] + "..."
    except OSError:
        return ""
    return ""


def cmd_list() -> int:
    if not NOVELS_DIR.exists():
        print("[novel] no novels/ directory yet. Use `python novel.py create <name>`.")
        return 0
    novels = sorted(p for p in NOVELS_DIR.iterdir() if p.is_dir() and (p / "config.yaml").exists())
    if not novels:
        print("[novel] no novels found under novels/.")
        return 0
    print(f"{'NAME':<24} {'TITLE':<22} {'CHAPTERS':>8} {'CHARS':>10}  {'RUNNING':<8} LAST LOG")
    print("-" * 112)
    for nd in novels:
        name = nd.name
        title = _read_title(nd)
        chars = _count_chars(nd / "book.md")
        chapters = _last_chapter(nd / "chapters")
        running = "yes" if find_novel_pids(name) else "no"
        last_log = _tail_line(nd / "logs" / "run.log")
        print(f"{name:<24} {title:<22} {chapters:>8} {chars:>10}  {running:<8} {last_log}")
    return 0


def cmd_stats(name: str) -> int:
    """Aggregate logs/llm_calls.jsonl for a novel into a per-stage summary."""
    nd = NOVELS_DIR / name
    metrics_path = nd / "logs" / "llm_calls.jsonl"
    if not metrics_path.exists():
        print(f"[novel] no metrics yet for {name!r} ({metrics_path} missing). "
              f"They accumulate once the novel runs with api.metrics_enabled (default on).")
        return 0
    by_tag: dict[str, dict[str, float]] = {}
    totals = {"calls": 0.0, "elapsed": 0.0, "prompt": 0.0, "output": 0.0,
              "attempts": 0.0, "salvaged": 0.0, "failed": 0.0}
    try:
        lines = metrics_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"[novel] could not read {metrics_path}: {exc}")
        return 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = str(row.get("tag") or "(untagged)")
        agg = by_tag.setdefault(tag, {"calls": 0.0, "elapsed": 0.0, "prompt": 0.0,
                                      "output": 0.0, "attempts": 0.0,
                                      "salvaged": 0.0, "failed": 0.0})
        elapsed = float(row.get("elapsed") or 0.0)
        prompt = float(row.get("prompt_chars") or 0.0)
        output = float(row.get("output_chars") or 0.0)
        attempts = float(row.get("attempts") or 1.0)
        salvaged = 1.0 if row.get("salvaged") else 0.0
        failed = 0.0 if row.get("ok", True) else 1.0
        for bucket in (agg, totals):
            bucket["calls"] += 1
            bucket["elapsed"] += elapsed
            bucket["prompt"] += prompt
            bucket["output"] += output
            bucket["attempts"] += attempts
            bucket["salvaged"] += salvaged
            bucket["failed"] += failed

    if totals["calls"] == 0:
        print(f"[novel] {metrics_path} has no parseable records yet.")
        return 0

    print(f"[novel] LLM call stats for {name!r}  (source: {metrics_path})")
    header = (f"{'STAGE/TAG':<28} {'CALLS':>7} {'AVG_s':>7} {'TOT_s':>9} "
              f"{'AVG_OUT':>8} {'RETRY%':>7} {'SALV%':>6} {'FAIL%':>6}")
    print(header)
    print("-" * len(header))

    def _emit(label: str, agg: dict[str, float]) -> None:
        calls = agg["calls"] or 1.0
        avg_s = agg["elapsed"] / calls
        avg_out = agg["output"] / calls
        # attempts includes the first try; retry% = extra attempts per call.
        retry_pct = max(0.0, (agg["attempts"] / calls - 1.0)) * 100.0
        salv_pct = agg["salvaged"] / calls * 100.0
        fail_pct = agg["failed"] / calls * 100.0
        print(f"{label:<28} {int(agg['calls']):>7} {avg_s:>7.1f} {agg['elapsed']:>9.0f} "
              f"{int(avg_out):>8} {retry_pct:>6.0f}% {salv_pct:>5.0f}% {fail_pct:>5.0f}%")

    for tag in sorted(by_tag, key=lambda t: by_tag[t]["elapsed"], reverse=True):
        _emit(tag, by_tag[tag])
    print("-" * len(header))
    _emit("TOTAL", totals)
    return 0


def _tail_line(path: Path, max_len: int = 60) -> str:
    if not path.exists():
        return ""
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    last = lines[-1].strip()
    return last if len(last) <= max_len else last[:max_len] + "..."


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Universal multi-novel AI writing framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="scaffold a new novel directory")
    p_create.add_argument("name")

    p_trial = sub.add_parser("trial", help="generate opening trial variants without touching chapters/book")
    p_trial.add_argument("name")
    p_trial.add_argument("--variants", type=int, default=None, help="number of opening variants to generate")
    p_trial.add_argument("--chapters", type=int, default=None, help="chapters per variant")

    p_adopt = sub.add_parser("adopt-trial", help="adopt a trial's best opening route into memory/opening_route.md")
    p_adopt.add_argument("name")
    p_adopt.add_argument("trial_id", nargs="?", help="trial directory id; defaults to latest")

    p_bench = sub.add_parser("benchmark", help="manage local benchmark samples")
    bench_sub = p_bench.add_subparsers(dest="benchmark_command", required=True)
    p_bench_list = bench_sub.add_parser("list", help="list local benchmark samples")
    p_bench_list.add_argument("--platform", default=None)
    p_bench_list.add_argument("--style", default=None)
    p_bench_add = bench_sub.add_parser("add", help="add a structured benchmark sample")
    p_bench_add.add_argument("platform")
    p_bench_add.add_argument("style")
    p_bench_add.add_argument("source")
    p_bench_add.add_argument("--title", default=None)

    p_script = sub.add_parser("script", help="convert novel text into a 短剧 screenplay")
    p_script.add_argument("name", nargs="?", default=None, help="novel name (omit when using --input)")
    p_script.add_argument("--input", default=None, help="path to any text/markdown file to convert (standalone)")
    p_script.add_argument("--chapters", default=None, help="chapter range from novels/<name>/chapters, e.g. 1-3 or 5")
    p_script.add_argument("--out", default=None, help="output screenplay path")
    p_script.add_argument("--seg-chars", type=int, default=None, help="max novel chars per LLM call (default 6000)")
    p_script.add_argument("--temperature", type=float, default=None, help="LLM temperature override")

    p_run = sub.add_parser("run", help="run the pipeline for a novel")
    p_run.add_argument("name")
    p_run.add_argument("--foreground", action="store_true", help="run in the current console instead of detaching")

    sub.add_parser("list", help="list all novels and their progress")

    p_stats = sub.add_parser("stats", help="aggregate per-stage LLM call metrics for a novel")
    p_stats.add_argument("name")

    p_stop = sub.add_parser("stop", help="kill the running process for a novel")
    p_stop.add_argument("name")

    p_restart = sub.add_parser("restart", help="stop + relaunch a novel (resumes from checkpoint)")
    p_restart.add_argument("name")
    p_restart.add_argument("--foreground", action="store_true")
    p_restart.add_argument("--wait", type=float, default=2.0)

    args = parser.parse_args()

    if args.command == "create":
        return cmd_create(args.name)
    if args.command == "trial":
        return cmd_trial(args.name, variants=args.variants, chapters=args.chapters)
    if args.command == "adopt-trial":
        return cmd_adopt_trial(args.name, trial_id=args.trial_id)
    if args.command == "benchmark":
        if args.benchmark_command == "list":
            return cmd_benchmark_list(platform=args.platform, style=args.style)
        if args.benchmark_command == "add":
            return cmd_benchmark_add(args.platform, args.style, args.source, title=args.title)
    if args.command == "run":
        return cmd_run(args.name, foreground=args.foreground)
    if args.command == "script":
        return cmd_script(
            args.name,
            input_path=args.input,
            chapters=args.chapters,
            out=args.out,
            seg_chars=args.seg_chars,
            temperature=args.temperature,
        )
    if args.command == "list":
        return cmd_list()
    if args.command == "stats":
        return cmd_stats(args.name)
    if args.command == "stop":
        return cmd_stop(args.name)
    if args.command == "restart":
        return cmd_restart(args.name, foreground=args.foreground, wait=args.wait)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
