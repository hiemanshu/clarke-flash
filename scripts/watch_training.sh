#!/usr/bin/env bash
# clarke-flash trainer dashboard (in-place TUI; no flicker/scroll).
#
#   ./scripts/watch_training.sh            # ACTIVE run dashboard (default 1s refresh)
#   ./scripts/watch_training.sh 2          # refresh every 2s
#   ./scripts/watch_training.sh status     # print one frame and exit
#   ./scripts/watch_training.sh --full     # everything: all runs + reports + eval logs
#   LOGDIR=/some/dir CLEAR_EVERY=30 ./scripts/watch_training.sh
#
# Data: runs/<run>/train.jsonl (step/total/loss + null/span/cand/op/present/dist
# buckets, val rows) and checkpoint-* dirs. Hardware (GPU + corsair-psu rails)
# and footer are pinned to the screen bottom; the frame is padded to full
# terminal height. Full-screen purge every CLEAR_EVERY seconds (default 30) and
# on resize clears shifted-line/wrap artifacts; between purges the frame paints
# from home and erases below. Checkpoints render as ONE fixed row (growing
# tables were the main line-shifter). Alternate screen; Ctrl-C restores.
set -uo pipefail

INTERVAL=1
FULL=0
STATUS=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    status) STATUS=1 ;;
    ''|*[!0-9]*) : ;;
    *) INTERVAL="$a" ;;
  esac
done

LOGDIR="${LOGDIR:-/tmp/opencode}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROWS=24 COLS=80
W=72
K=$'\033[K'
CLEAR_EVERY=${CLEAR_EVERY:-30}
TTY=0; [[ -t 1 ]] && TTY=1

if (( TTY )); then
  B=$'\033[1m'; D=$'\033[2m'; R=$'\033[0m'
  CY=$'\033[36m'; GR=$'\033[32m'; YE=$'\033[33m'; RD=$'\033[31m'
else
  B=; D=; R=; CY=; GR=; YE=; RD=
fi

lab() { printf ' %-10s ' "$1"; }
bar() {                                                  # value max width colour
  local v=$1 m=$2 w=$3 c=$4 f e out=""
  (( m <= 0 )) && m=1
  f=$(( v * w / m )); (( f > w )) && f=$w; (( f < 0 )) && f=0
  e=$(( w - f ))
  (( f > 0 )) && out+=$(printf '█%.0s' $(seq 1 $f))
  (( e > 0 )) && out+=$(printf '░%.0s' $(seq 1 $e))
  printf '%s%s%s' "$c" "$out" "$R"
}

active_run() {
  pgrep -af '(clairic|clarke)[-.-]train' 2>/dev/null \
    | grep -oE -e '-{2}run-dir[ =][^ ]+' | head -1 | sed -E 's/.*[ =]//' \
    | xargs -r -n1 basename 2>/dev/null
}

# parse train.jsonl -> KEY=VALUE lines (latest train + latest val + trend)
state() { python3 - "$1" <<'PY' 2>/dev/null
import json, sys
rows = []
try:
    for line in open(sys.argv[1]):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
except OSError:
    raise SystemExit
keys = ("null", "span", "cand", "op", "present", "dist")
train = [r for r in rows if not r.get("val")]
val = [r for r in rows if r.get("val")]
if train:
    t = train[-1]
    for src, dst in (("step", "STEP"), ("total", "TOTAL"), ("epoch", "EPOCH"),
                     ("loss", "LOSS"), ("lr", "LR"), ("recs_per_s", "RPS"), ("gpu_gb", "GB")):
        if src in t:
            print(f"{dst}={t[src]}")
    print("COMPS=" + " ".join(f"{k}={t[k]:.3f}" for k in keys if k in t))
if val:
    v, pv = val[-1], (val[-2] if len(val) > 1 else None)
    print(f"VSTEP={v.get('step', '')}")
    print(f"VLOSS={v.get('loss', '')}")
    print("VCOMPS=" + " ".join(f"{k}={v[k]:.3f}" for k in keys if k in v))
    if pv and isinstance(v.get("loss"), float) and isinstance(pv.get("loss"), float):
        d = v["loss"] - pv["loss"]
        print(f"VTREND={'DOWN' if d < -0.005 else ('UP' if d > 0.005 else 'FLAT')} ({d:+.3f})")
PY
}

# one fixed row for the whole checkpoint table (never grows -> never shifts lines)
ckline() { python3 - "$ROOT" "$1" <<'PY' 2>/dev/null
import glob, json, os, re, sys
root, run = sys.argv[1], sys.argv[2]
path = os.path.join(root, "runs", run, "train.jsonl")
train, val = [], []
if os.path.exists(path):
    for line in open(path):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        (val if r.get("val") else train).append(r)
def at(rows, step):
    r = [x for x in rows if x.get("step", 0) <= step]
    return r[-1] if r else None
cks = sorted((int(re.search(r"checkpoint-(\d+)", d).group(1))
              for d in glob.glob(os.path.join(root, "runs", run, "checkpoint-*"))))
if not cks:
    print("none yet — first at step 1000")
    raise SystemExit
vals = [(c, (lambda v: v.get("loss") if v else None)(at(val, c))) for c in cks[-4:]]
parts = []
for i, (c, vl) in enumerate(vals):
    mark = ""
    if i and isinstance(vl, float) and isinstance(vals[i-1][1], float):
        d = vl - vals[i-1][1]
        mark = "▼" if d < -0.005 else ("▲" if d > 0.005 else "=")
    parts.append(f"{c/1000:g}k:{'-' if vl is None else f'{vl:.3f}'}{mark}")
last_t = at(train, cks[-1])
lt = last_t.get("loss") if last_t else None
print(f"{len(cks)} ckpt  [{'  '.join(parts)}]   latest train {'-' if lt is None else f'{lt:.3f}'}")
PY
}

all_runs() { python3 - "$ROOT" <<'PY' 2>/dev/null
import glob, json, os, sys
for d in sorted(glob.glob(os.path.join(sys.argv[1], "runs", "*"))):
    p = os.path.join(d, "train.jsonl")
    if not os.path.exists(p): continue
    train = val = None
    for line in open(p):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("val"): val = r
        else: train = r
    if not train: continue
    g = lambda v: "   -   " if not isinstance(v, float) else f"{v:7.3f}"
    print(f"  {os.path.basename(d):12} step {train.get('step', 0):>6}/{train.get('total', '?'):<6} "
          f"train {g(train.get('loss'))} val {g(val.get('loss') if val else None)}"
          f"  ckpt {len(glob.glob(os.path.join(d, 'checkpoint-*')))}")
PY
}

gpu_line() {
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader 2>/dev/null | awk -F', ' 'NR==1{print $1"|"$2"|"$3"|"$4"|"$5"|"$6}'
}
psu_dir() {
  local h
  for h in /sys/class/hwmon/hwmon*; do
    [[ "$(cat "$h/name" 2>/dev/null)" == "corsairpsu" ]] && { printf '%s' "$h"; return; }
  done
}
hw_val() { local v; v=$(cat "$1" 2>/dev/null) || return; printf '%s' $(( v / $2 )); }

# ---- frame buffer -------------------------------------------------------
# SINK selects the active block: OUT (top) / BOT (bottom-pinned) / MID (gap)
declare -a OUT BOT MID
SINK=OUT
push() { local -n _ref="$SINK"; _ref+=("$1"); }
rule() { push "${D}$(printf '─%.0s' $(seq 1 $W))${R}${K}"; }
emit() { push "$*$K"; }

PREV_STEP=-1; PREV_T=""

draw() {
  local run STEP="" TOTAL="" EPOCH="" LOSS="" LR="" RPS="" GB="" COMPS=""
  local VSTEP="" VLOSS="" VCOMPS="" VTREND=""
  OUT=(); BOT=(); MID=(); SINK=OUT
  if (( TTY )); then
    local sz; sz=$(stty size 2>/dev/null </dev/tty || echo "24 80")
    ROWS=${sz%% *}; COLS=${sz##* }
  fi
  (( ROWS < 8 )) && ROWS=24
  (( COLS < 40 )) && COLS=80
  W=$(( COLS - 1 ))

  emit "${B}${CY}  clarke-flash trainer${R}$(printf ' %*s' $(( W - 30 )) '')$(date '+%H:%M:%S')${K}"
  rule
  emit ""$K

  run="$(active_run)"
  local active=0
  [[ -n "$run" ]] && active=1
  if (( ! active )); then run="$(ls -1t "$ROOT/runs" 2>/dev/null | head -1 || true)"; fi

  if [[ -n "$run" && -d "$ROOT/runs/$run" ]]; then
    local log="$ROOT/runs/$run/train.jsonl" state_str="" c v
    [[ -f "$log" ]] && state_str="$(state "$log")"
    while IFS='=' read -r c v; do
      case "$c" in
        STEP) STEP=$v;; TOTAL) TOTAL=$v;; EPOCH) EPOCH=$v;; LOSS) LOSS=$v;; LR) LR=$v;;
        RPS) RPS=$v;; GB) GB=$v;; COMPS) COMPS=$v;;
        VSTEP) VSTEP=$v;; VLOSS) VLOSS=$v;; VCOMPS) VCOMPS=$v;; VTREND) VTREND=$v;;
      esac
    done <<< "$state_str" 2>/dev/null || true

    local now=$(date +%s%N) rate=""
    if [[ -n "$STEP" ]] && (( PREV_STEP >= 0 )) && (( STEP > PREV_STEP )) && [[ -n "$PREV_T" ]]; then
      rate=$(awk -v s="$STEP" -v p="$PREV_STEP" -v n="$now" -v t="$PREV_T" \
             'BEGIN{printf "%.2f", (s-p)/((n-t)/1e9)}')
    fi
    if [[ -n "$STEP" ]]; then PREV_STEP="$STEP"; PREV_T="$now"; else PREV_STEP=-1; fi

    if (( active )); then
      emit "$(lab run)${B}${run}${R}  ${GR}ACTIVE${R}${K}"
    else
      emit "$(lab run)${B}${run}${R}  ${D}idle (newest run)${R}${K}"
    fi

    if [[ -n "$STEP" && -n "$TOTAL" ]]; then
      local pct=$(( STEP * 100 / TOTAL )); (( pct > 100 )) && pct=100
      local eta="…"
      [[ -n "$rate" && "$rate" != "0.00" ]] && eta=$(awk -v r="$rate" -v s="$STEP" -v t="$TOTAL" \
        'BEGIN{e=(t-s)/r; if (e<0) e=0; printf "%dh%02dm", e/3600, (e%3600)/60}')
      emit "$(lab progress)[$(bar "$STEP" "$TOTAL" 36 "$GR")]  ${B}${STEP}${R}/${TOTAL}  ${B}${pct}%${R}  ep ${EPOCH:-?}  eta ${eta}${K}"
    elif [[ -n "$STEP" ]]; then
      emit "$(lab progress)step ${B}${STEP}${R}  ${D}(total unknown — run predates jsonl total field)${R}${K}"
    else
      emit "$(lab progress)${D}starting… (loading backbone)${R}${K}"
    fi
    emit "$(lab train)loss ${B}${LOSS:-—}${R}   lr ${LR:-—}   ${D}${RPS:-—} rec/s · ${GB:-—} GB${R}   ${D}${rate:+${rate} steps/s}${R}${K}"
    [[ -n "$COMPS" ]] && emit "$(lab components)${D}${COMPS}${R}${K}"
    if [[ -n "$VLOSS" ]]; then
      local tc="$GR"; [[ "$VTREND" == UP* ]] && tc="$RD"
      emit "$(lab validation)val@${VSTEP} loss ${B}${VLOSS}${R} ${tc}${VTREND}${R}  ${D}${VCOMPS}${R}${K}"
    fi
    local ck; ck="$(ckline "$run")"
    emit "$(lab checkpoints)${D}${ck}${R}${K}"
    emit ""$K
  else
    emit "$(lab run)${D}no runs/ yet — see README + PLAN (clarke-train …)${R}${K}"
    rule; emit ""$K
  fi
  rule
  emit ""$K

  if (( FULL )); then
    emit "    ${B}all runs${R}${K}"
    local l
    while IFS= read -r l; do emit "${D}  ${l}${R}${K}"; done < <(all_runs)
    emit ""$K
    emit "    ${B}reports${R}${K}"
    local f
    for f in "$ROOT"/reports/*.json; do [[ -e "$f" ]] && emit "  ${f##*/}${K}"; done
    for f in "$LOGDIR"/eval-*.log; do
      [[ -e "$f" ]] || continue
      emit "  ${D}${f##*/}:${R}${K}"
      while IFS= read -r l; do emit "    ${l:0:110}${K}"; done < <(tail -3 "$f" | tr -d '\r')
    done
    emit ""$K
    rule
  fi

  # ---- bottom-pinned hardware + footer ----
  SINK=BOT
  local g; g="$(gpu_line)"
  if [[ -n "$g" ]]; then
    local used tot util temp pw plim
    used=${g%%|*}; g=${g#*|}; tot=${g%%|*}; g=${g#*|}; util=${g%%|*}; g=${g#*|}
    temp=${g%%|*}; g=${g#*|}; pw=${g%%|*}; plim=${g##*|}
    used=${used//[^0-9]/}; tot=${tot//[^0-9]/}; util=${util//[^0-9]/}
    temp=${temp//[^0-9]/}; pw=${pw//[^0-9.]/}; plim=${plim//[^0-9.]/}
    pw=${pw%%.*}; plim=${plim%%.*}
    local ug=$(( used / 1024 )) tg=$(( tot / 1024 ))
    local c2="$GR"; (( util < 40 )) && c2="$YE"; (( used * 100 / (tot + 1) > 90 )) && c2="$RD"
    emit "$(lab gpu)[$(bar "$used" "$tot" 40 "$c2")]  ${B}${ug}${R}/${D}${tg}${R} GB   util ${B}${util}%${R}   ${D}${temp}°C${R}${K}"
    local psu; psu="$(psu_dir)"
    if [[ -n "$psu" ]]; then
      local p1 p2 ac pt
      p1=$(hw_val "$psu/power1_input" 1000000); p2=$(hw_val "$psu/power2_input" 1000000)
      ac=$(hw_val "$psu/in0_input" 1000); pt=$(hw_val "$psu/temp1_input" 1000)
      local pmax=${PSU_MAX:-1200}
      local pload=$(( p1 * 100 / pmax ))
      local pc="$GR"; (( pload > 60 )) && pc="$YE"; (( pload > 85 )) && pc="$RD"
      emit "$(lab power)${D}GPU ${pw}/${plim} W${R}   ${pc}PSU ${p1} W (${pload}%)${R}   ${D}+12V ${p2} W · ${ac} V · ${pt}°C${R}${K}"
    else
      emit "$(lab power)${D}GPU ${pw}/${plim} W  (no PSU sensors)${R}${K}"
    fi
  fi
  rule
  emit "${D}  Ctrl-C to exit · --full: all runs/reports/logs · refresh ${INTERVAL}s · CLEAR_EVERY=${CLEAR_EVERY}s purge${R}${K}"
  SINK=OUT
}

paint() {
  local total=$(( ${#OUT[@]} + ${#BOT[@]} ))
  local gap=$(( ROWS - total ))
  (( gap < 0 )) && gap=0
  printf '%s\n' "${OUT[@]}"
  local i
  for (( i = 0; i < gap; i++ )); do printf '%s\n' "$K"; done
  printf '%s\n' "${BOT[@]}"
}

render() {
  local now dims
  now=$(date +%s)
  dims="${ROWS}x${COLS}"
  if [[ "$dims" != "$LAST_DIMS" ]] || (( now >= NEXT_CLEAR )); then
    printf '\033[H\033[2J'                       # full purge: kills wrap/shift artifacts
    LAST_DIMS="$dims"; NEXT_CLEAR=$(( now + CLEAR_EVERY ))
  else
    printf '\033[H'
  fi
  paint
  printf '\033[J'
}

if (( TTY && ! STATUS )); then printf '\033[?1049h'; fi
cleanup() { printf '\033[?25h\033[?1049l'; exit 0; }
trap cleanup INT TERM
(( TTY && ! STATUS )) && printf '\033[?25l'

if (( STATUS )); then draw; paint; exit 0; fi
LAST_DIMS=""; NEXT_CLEAR=0
while true; do
  draw
  render
  sleep "$INTERVAL"
done
