"""One-click hot restart for the novel pipeline.

Behavior:
    1. Find any running `python run.py` (this project only) and kill it.
    2. Wait for processes to terminate.
    3. Relaunch `python run.py`; the new run will resume from the last
       unfinished chapter via the existing checkpoint system — no data loss.

Usage:
    python restart.py                # kill + relaunch in background, log to logs/run.log
    python restart.py --foreground   # kill + relaunch attached to current console
    python restart.py --no-kill      # just launch (skip the kill step)
    python restart.py --kill-only    # only kill, do not relaunch
    python restart.py --wait 3       # wait N seconds after kill (default 2)
    python restart.py --dry-run      # show what would happen, don't act
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_DIR / "run.py"
LOG_DIR = PROJECT_DIR / "logs"
RUN_LOG = LOG_DIR / "run.log"

# Match either the entry script or the library it imports, in case someone
# launches with `python pipeline.py` directly. Compared case-insensitively
# against the full command line.
_SCRIPT_MARKERS = ("run.py", "pipeline.py")


def _is_windows() -> bool:
    return os.name == "nt"


def _windows_python_pids_with_cmdline() -> list[tuple[int, str]]:
    """Return (pid, cmdline) for every python.exe / pythonw.exe on Windows.

    Tries WMIC first (fastest), then PowerShell Get-CimInstance for hosts
    where WMIC has been removed (Win11 24H2+).
    """
    # WMIC path
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where",
             "name='python.exe' or name='pythonw.exe'",
             "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
            stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace",
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

    # PowerShell fallback (Win11 24H2 has dropped WMIC)
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter "
            "\"name='python.exe' or name='pythonw.exe'\" | "
            "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace",
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


def find_pipeline_pids() -> list[int]:
    """Return PIDs of `python ... run.py` (or pipeline.py) belonging to THIS project."""
    project_root = str(PROJECT_DIR).lower().replace("\\", "/")
    self_pid = os.getpid()
    pids: list[int] = []

    if _is_windows():
        for pid, cmd in _windows_python_pids_with_cmdline():
            if pid == self_pid:
                continue
            cmd_norm = cmd.lower().replace("\\", "/")
            if not any(marker in cmd_norm for marker in _SCRIPT_MARKERS):
                continue
            # Make sure it's *this* project — match against project root path
            # OR the unqualified script name with no other path component
            # (covers `python run.py` launched from inside the project dir).
            if project_root in cmd_norm or _looks_like_local_launch(cmd_norm):
                pids.append(pid)
        return pids

    # POSIX
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,command"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid_s, cmd = parts
        cmd_norm = cmd.lower()
        if not any(marker in cmd_norm for marker in _SCRIPT_MARKERS):
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if project_root in cmd_norm.replace("\\", "/") or _looks_like_local_launch(cmd_norm):
            pids.append(pid)
    return pids


def _looks_like_local_launch(cmd_norm: str) -> bool:
    """Heuristic: cmdline mentions the bare script name with no path separator
    in front of it (i.e. launched with cwd=project_dir)."""
    for marker in _SCRIPT_MARKERS:
        idx = cmd_norm.find(marker)
        if idx == -1:
            continue
        # No path char immediately before the marker token? Then it's a bare
        # `python run.py` style launch.
        before = cmd_norm[idx - 1] if idx > 0 else " "
        if before in (" ", '"', "'"):
            return True
    return False


def kill_pids(pids: list[int], dry_run: bool = False) -> None:
    if not pids:
        print("[restart] no run.py/pipeline.py process found.")
        return
    for pid in pids:
        print(f"[restart] killing PID {pid} ...")
        if dry_run:
            continue
        try:
            if _is_windows():
                # /F = force, no /T: pipeline.py has only thread-pool children,
                # never spawns subprocesses, so /T just risks hitting the wrong
                # process if the OS recycled a PID.
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            pass
        except Exception as exc:
            print(f"[restart] kill PID {pid} failed: {exc}")

    if not dry_run and not _is_windows():
        # Best-effort SIGKILL after a short grace period for any survivors.
        time.sleep(1.0)
        for pid in pids:
            try:
                os.kill(pid, 0)  # still alive?
                os.kill(pid, 9)
                print(f"[restart] SIGKILL sent to surviving PID {pid}")
            except ProcessLookupError:
                pass
            except Exception:
                pass


def launch_pipeline(foreground: bool, dry_run: bool = False) -> int:
    if not ENTRY_SCRIPT.exists():
        print(f"[restart] ERROR: {ENTRY_SCRIPT} not found")
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    cmd = [python, "-u", str(ENTRY_SCRIPT)]
    print(f"[restart] launching: {' '.join(cmd)} (cwd={PROJECT_DIR})")
    if dry_run:
        return 0

    if foreground:
        try:
            return subprocess.call(cmd, cwd=str(PROJECT_DIR))
        except KeyboardInterrupt:
            print("[restart] foreground run interrupted")
            return 130

    # Background mode: detach so closing this terminal does not kill the run.
    # On Windows we use PowerShell Start-Process — empirically Popen with
    # DETACHED_PROCESS + creationflags=CREATE_NEW_PROCESS_GROUP loses the child
    # within ~10s when the parent (restart.py) exits, even with stdout
    # redirected to an open file. Start-Process establishes the child as a
    # real session-1 process the OS won't reap.
    if _is_windows():
        # Quote paths defensively in case of spaces.
        ps_args = [
            "powershell", "-NoProfile", "-Command",
            "Start-Process",
            "-FilePath", f"'{python}'",
            "-ArgumentList", f"'-u','{ENTRY_SCRIPT}'",
            "-WorkingDirectory", f"'{PROJECT_DIR}'",
            "-RedirectStandardOutput", f"'{RUN_LOG}'",
            "-RedirectStandardError", f"'{LOG_DIR / 'runner_stderr.log'}'",
            "-WindowStyle", "Hidden",
            "-PassThru",
        ]
        try:
            out = subprocess.check_output(ps_args, text=True, encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as exc:
            print(f"[restart] PowerShell Start-Process failed: {exc}")
            return 3
        # Parse "Id" line from Start-Process -PassThru output.
        pid_val = "unknown"
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("id ") or line.lower().startswith("id  "):
                # Form: "Id                : 12345"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    pid_val = parts[1].strip()
                    break
            if line.split() and line.split()[0].isdigit():
                pid_val = line.split()[0]
                break
        # Append a banner to run.log via a separate Python write (we did NOT
        # open the file here because PowerShell already owns it for the child).
        try:
            with open(RUN_LOG, "ab", buffering=0) as fp:
                fp.write(
                    f"\n\n========== restart @ {time.strftime('%Y-%m-%d %H:%M:%S')} (PID {pid_val}) ==========\n".encode()
                )
        except Exception:
            pass
        print(f"[restart] pipeline started PID={pid_val}")
        print(f"[restart] tailing log: {RUN_LOG}")
        return 0

    # POSIX path: standard Popen with start_new_session is enough.
    log_fp = open(RUN_LOG, "ab", buffering=0)
    try:
        log_fp.write(
            f"\n\n========== restart @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n".encode()
        )
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_DIR),
            stdin=subprocess.DEVNULL, stdout=log_fp, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
        print(f"[restart] pipeline started PID={proc.pid}")
        print(f"[restart] tailing log: {RUN_LOG}")
    finally:
        log_fp.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hot-restart the novel pipeline.")
    parser.add_argument("--foreground", action="store_true", help="attach to current console instead of detaching")
    parser.add_argument("--no-kill", action="store_true", help="skip the kill step, just launch")
    parser.add_argument("--kill-only", action="store_true", help="kill running pipeline.py but do not relaunch")
    parser.add_argument("--wait", type=float, default=2.0, help="seconds to wait after kill before relaunch (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="show actions but do not change anything")
    args = parser.parse_args()

    if args.no_kill and args.kill_only:
        print("[restart] --no-kill and --kill-only are mutually exclusive")
        return 2

    if not args.no_kill:
        pids = find_pipeline_pids()
        kill_pids(pids, dry_run=args.dry_run)
        if pids and not args.dry_run and args.wait > 0:
            print(f"[restart] waiting {args.wait}s for processes to drain ...")
            time.sleep(args.wait)

    if args.kill_only:
        return 0

    return launch_pipeline(foreground=args.foreground, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
