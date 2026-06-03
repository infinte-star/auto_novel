"""Universal AI novel-writing framework — multi-novel launcher / manager.

Each novel lives in its own directory `novels/<name>/` containing prompt.md,
config.yaml, book.md, chapters/, memory/, logs/, story_state.db. Every novel
runs as an independent OS process, so multiple novels can be written
simultaneously without sharing the engine's process-level global state
(config.PROMPT_FILE, memory._CACHEABLE_PREFIX_CACHE, etc.).

Subcommands:
    python novel.py create <name>            scaffold novels/<name>/ from templates
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
    # Pass paths relative to PROJECT_DIR (== config.ROOT) so config.ROOT joins
    # resolve correctly regardless of the launching cwd.
    os.environ["NOVEL_CONFIG"] = str(config_path.relative_to(PROJECT_DIR))
    os.environ["NOVEL_PROMPT"] = str(prompt_path.relative_to(PROJECT_DIR))

    from pipeline import main  # noqa: E402  (must import after env vars are set)

    main()
    return 0


def _run_marker(name: str) -> str:
    """The command-line substring used to find/kill this novel's process."""
    return f"novel.py run {name}"


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
        ps_args = [
            "powershell", "-NoProfile", "-Command",
            "Start-Process",
            "-FilePath", f"'{python}'",
            "-ArgumentList", f"'-u','{entry}','run','{name}','--foreground'",
            "-WorkingDirectory", f"'{PROJECT_DIR}'",
            "-RedirectStandardOutput", f"'{run_log}'",
            "-RedirectStandardError", f"'{log_dir / 'runner_stderr.log'}'",
            "-WindowStyle", "Hidden",
            "-PassThru",
        ]
        try:
            out = subprocess.check_output(ps_args, text=True, encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as exc:
            print(f"[novel] PowerShell Start-Process failed: {exc}")
            return 3
        pid_val = "unknown"
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("id ") and ":" in line:
                pid_val = line.split(":", 1)[1].strip()
                break
            if line.split() and line.split()[0].isdigit():
                pid_val = line.split()[0]
                break
        try:
            with open(run_log, "ab", buffering=0) as fp:
                fp.write(
                    f"\n\n========== novel run {name} @ "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} (PID {pid_val}) ==========\n".encode()
                )
        except Exception:
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


def find_novel_pids(name: str) -> list[int]:
    """PIDs whose command line runs `novel.py run <name>` for THIS project."""
    marker = _run_marker(name).lower().replace("\\", "/")
    project_root = str(PROJECT_DIR).lower().replace("\\", "/")
    self_pid = os.getpid()
    pids: list[int] = []
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
        if project_root in cmd_norm or _looks_like_local_launch(cmd_norm):
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
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
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
        return 0
    kill_pids(pids)
    return 0


def cmd_restart(name: str, foreground: bool, wait: float) -> int:
    _validate_name(name)
    pids = find_novel_pids(name)
    if pids:
        kill_pids(pids)
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

    p_run = sub.add_parser("run", help="run the pipeline for a novel")
    p_run.add_argument("name")
    p_run.add_argument("--foreground", action="store_true", help="run in the current console instead of detaching")

    sub.add_parser("list", help="list all novels and their progress")

    p_stop = sub.add_parser("stop", help="kill the running process for a novel")
    p_stop.add_argument("name")

    p_restart = sub.add_parser("restart", help="stop + relaunch a novel (resumes from checkpoint)")
    p_restart.add_argument("name")
    p_restart.add_argument("--foreground", action="store_true")
    p_restart.add_argument("--wait", type=float, default=2.0)

    args = parser.parse_args()

    if args.command == "create":
        return cmd_create(args.name)
    if args.command == "run":
        return cmd_run(args.name, foreground=args.foreground)
    if args.command == "list":
        return cmd_list()
    if args.command == "stop":
        return cmd_stop(args.name)
    if args.command == "restart":
        return cmd_restart(args.name, foreground=args.foreground, wait=args.wait)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
