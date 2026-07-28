#!/usr/bin/env bash
# Keep ts_v2match running until it reaches TARGET chapters.
#
# Why this exists: the gateway is dropping connections in bursts (503
# system_cpu_overloaded, then refused connections). v2 tolerates a failed
# canon_check, but a failed `write` exhausts llm.py's 6 attempts and raises, which
# kills the process. v2 checkpoints per STEP, so a relaunch resumes the same
# chapter rather than redoing it — the only cost of a crash is the wall time.
#
# It relaunches ONLY when no process is alive AND the target is not met, so it can
# never run two writers into the same novel directory.
set -u
NOVEL=ts_v2match
TARGET=200
PY=E:/pycharmproject/allvenv/novel/Scripts/python.exe
ND="novels/$NOVEL"
LOG="$ND/logs/supervisor.log"
MAX_RELAUNCH=12
relaunch=0

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

while :; do
  count=$(ls "$ND"/chapters/*.md 2>/dev/null | wc -l)
  if [ "$count" -ge "$TARGET" ]; then
    say "DONE: $count chapters >= $TARGET"
    exit 0
  fi

  pid=$(grep -o '"pid": *[0-9]*' "$ND/logs/run.pid" 2>/dev/null | grep -o '[0-9]*')
  alive=no
  if [ -n "${pid:-}" ]; then
    if tasklist //FI "PID eq $pid" 2>/dev/null | grep -q "$pid"; then alive=yes; fi
  fi

  if [ "$alive" = yes ]; then
    sleep 60
    continue
  fi

  if [ "$relaunch" -ge "$MAX_RELAUNCH" ]; then
    say "GIVING UP after $relaunch relaunches at $count chapters"
    exit 1
  fi
  relaunch=$((relaunch + 1))
  say "dead at $count chapters; relaunch #$relaunch after 90s backoff"
  sleep 90
  PYTHONIOENCODING=utf-8 "$PY" novel.py run "$NOVEL" --foreground >/dev/null 2>>"$ND/logs/supervisor_stderr.log"
  say "run exited rc=$? at $(ls "$ND"/chapters/*.md 2>/dev/null | wc -l) chapters"
done
