#!/usr/bin/env bash
# Live view of a clarke-flash training run: GPU, latest log lines, val rows.
# Auto-discovers the newest runs/<timestamp>/ dir and the running trainer;
# usage:  scripts/watch_training.sh            # follow the newest run
#         scripts/watch_training.sh status     # print once and exit
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${1:-}"
MODE="follow"
if [[ "${1:-}" == "status" ]]; then MODE="status"; RUN_DIR=""; fi

# discover the newest run (unless one was given)
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -1dt runs/*/ 2>/dev/null | head -1 || true)"
fi
if [[ -z "$RUN_DIR" ]]; then echo "no runs/ directory yet — train first (see README)"; exit 0; fi
LOG="$RUN_DIR/train.jsonl"

pid="$(pgrep -af 'clair(ic|ke)[-.-]train' | grep -v pgrep | awk '{print $1}' | head -1 || true)"
echo "run:       $RUN_DIR"
echo "log:       $LOG"
if [[ -n "$pid" ]]; then echo "trainer:   running (pid $pid)"; else echo "trainer:   NOT RUNNING"; fi

show() {
  echo "── $(date +%H:%M:%S) ──────────────────────────────────────────"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader \
    | awk -F', ' '{print "gpu:   " $1 " | " $2 " | " $3}'
  python3 "$(dirname "$0")/show_run.py" "$LOG" "${1:-8}"
  ls -1 "$RUN_DIR" 2>/dev/null | grep -m3 checkpoint | sed 's/^/  ckpt:  /'
}

if [[ "$MODE" == "status" ]]; then show 6; exit 0; fi
show 8
while true; do sleep 5; clear 2>/dev/null || true; show 6; done
