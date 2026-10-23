#!/usr/bin/env bash
# clarke-flash trainer dashboard (in-place TUI; no flicker/scroll).
#
#   ./scripts/watch_training.sh            # ACTIVE run dashboard (default 1s refresh)
#   ./scripts/watch_training.sh 2          # refresh every 2s
#   ./scripts/watch_training.sh status     # print one frame and exit
#   ./scripts/watch_training.sh --full     # everything: all runs + reports + eval logs
#   LOGDIR=/some/dir ./scripts/watch_training.sh
#
# Data source: runs/<run>/train.jsonl (step/total/loss + null/span/cand/op/present/dist
# buckets, val rows) and runs/<run>/checkpoint-<step>/. Terminal alternate-screen
# buffer, hidden cursor, repaint in place; Ctrl-C restores.
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
W=72
K=$'\033[K'
TTY=0; [[ -t 1 ]] && TTY=1

if (( TTY )); then
  B=$'\033[1m'; D=$'\033[2m'; R=$'\033[0m'
  CY=$'\033[36m'; GR=$'\033[32m'; YE=$'\033[33m'; RD=$'\033[31m'
else
  B=; D=; R=; CY=; GR=; YE=; RD=
fi

lab() { printf ' %-10s ' "$1"; }                       # dim label, fixed width
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

# parse train.jsonl -> KEY=VALUE lines
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

ckpts() { python3 - "$ROOT" "$1" <<'PY' 2>/dev/null
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
    r = [x for x in rows if x.get("step") == step] or [x for x in rows if x.get("step", 0) <= step]
    return r[-1] if r else None
f = lambda v: "    -    " if not isinstance(v, float) else f"{v:8.3f}"
prev = None
for d in sorted(glob.glob(os.path.join(root, "runs", run, "checkpoint-*")),
                key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))[-10:]:
    step = int(re.search(r"checkpoint-(\d+)", d).group(1))
    t, v = at(train, step), at(val, step)
    vl = v.get("loss") if v else None
    trend = ""
    if isinstance(vl, float) and isinstance(prev, float):
        trend = "v" if vl < prev - 0.005 else ("^" if vl > prev + 0.005 else "=")
    prev = vl
    print(f"  {os.path.basename(d):>16} {f(t.get('loss') if t else None)} {f(vl)}  {trend}")
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

rule() { printf '%s%s%s%s\n' "$D" "$(printf '─%.0s' $(seq 1 $W))" "$R" "$K"; }
emit() { printf '%s\n' "$*$K"; }

PREV_STEP=-1; PREV_T=""

draw() {
  local run STEP="" TOTAL="" EPOCH="" LOSS="" LR="" RPS="" GB="" COMPS=""
  local VSTEP="" VLOSS="" VCOMPS="" VTREND=""

  emit ""
  emit "${B}${CY}  clarke-flash trainer${R}                    $(date '+%H:%M:%S')"
  rule
  emit ""

  run="$(active_run)"
  local active=0
  [[ -n "$run" ]] && active=1
  if (( ! active )); then run="$(ls -1t "$ROOT/runs" 2>/dev/null | head -1 || true)"; fi

  if [[ -n "$run" && -d "$ROOT/runs/$run" ]]; then
    local log="$ROOT/runs/$run/train.jsonl"
    local state_str=""
    [[ -f "$log" ]] && state_str="$(state "$log")"
    local c v
    while IFS='=' read -r c v; do
      case "$c" in
        STEP) STEP=$v;; TOTAL) TOTAL=$v;; EPOCH) EPOCH=$v;; LOSS) LOSS=$v;;
        LR) LR=$v;; RPS) RPS=$v;; GB) GB=$v;; COMPS) COMPS=$v;;
        VSTEP) VSTEP=$v;; VLOSS) VLOSS=$v;; VCOMPS) VCOMPS=$v;; VTREND) VTREND=$v;;
      esac
    done <<< "$state_str"

    local now=$(date +%s%N) rate=""
    if [[ -n "$STEP" ]] && (( PREV_STEP >= 0 )) && (( STEP > PREV_STEP )) && [[ -n "$PREV_T" ]]; then
      rate=$(awk -v s="$STEP" -v p="$PREV_STEP" -v n="$now" -v t="$PREV_T" \
             'BEGIN{printf "%.2f", (s-p)/((n-t)/1e9)}')
    fi
    [[ -n "$STEP" ]] && { PREV_STEP="$STEP"; PREV_T="$now"; } || { PREV_STEP=-1; }

    if (( active )); then
      emit "$(lab run)${B}${run}${R}  ${GR}ACTIVE${R}  (step ${STEP:-0})"
    else
      emit "$(lab run)${B}${run}${R}  ${D}idle (newest run)${R}"
    fi
    emit ""

    if [[ -n "$STEP" && -n "$TOTAL" ]]; then
      local pct=$(( STEP * 100 / TOTAL )); (( pct > 100 )) && pct=100
      local eta="…"
      [[ -n "$rate" && "$rate" != "0.00" ]] && eta=$(awk -v r="$rate" -v s="$STEP" -v t="$TOTAL" \
        'BEGIN{e=(t-s)/r; if (e<0) e=0; printf "%dh%02dm", e/3600, (e%3600)/60}')
      emit "$(lab progress)[$(bar "$STEP" "$TOTAL" 36 "$GR")]  ${B}${STEP}${R} / ${TOTAL}  ${B}${pct}%${R}  ep ${EPOCH:-?}  eta ${eta}"
      emit "$(lab train)loss ${B}${LOSS:-—}${R}   lr ${LR:-—}   ${D}${RPS:-—} rec/s · ${GB:-—} GB${R}   ${D}${rate:+${rate} steps/s}${R}"
    elif [[ -n "$STEP" ]]; then
      emit "$(lab progress)step ${B}${STEP}${R}  ${D}(no total in jsonl — run started before the total field existed)${R}"
      emit "$(lab train)loss ${B}${LOSS:-—}${R}   ${D}${RPS:-—} rec/s${R}"
    else
      emit "$(lab progress)${D}starting… (loading backbone / first steps pending)${R}"
    fi
    [[ -n "$COMPS" ]] && emit "$(lab components)${D}${COMPS}${R}"
    if [[ -n "$VLOSS" ]]; then
      local tc="$GR"; [[ "$VTREND" == UP* ]] && tc="$RD"
      emit "$(lab validation)val@${VSTEP}  loss ${B}${VLOSS}${R}  ${tc}${VTREND}${R}"
      [[ -n "$VCOMPS" ]] && emit "              ${D}${VCOMPS}${R}"
    fi
    emit ""
    rule
    emit ""
    emit "    ${B}checkpoints${R} ${D}(train | val@≤step | trend)${R}"
    emit ""
    local ck; ck="$(ckpts "$run")"
    if [[ -n "$ck" ]]; then
      emit "    ${D}        checkpoint    train      val${R}"
      local line
      while IFS= read -r line; do
        local cc="$R"
        case "$line" in *v) cc="$GR";; *^*) cc="$RD";; esac
        emit "  ${cc}${line}${R}"
      done <<< "$ck"
    else
      emit "    ${D}(none yet — first at step 1000)${R}"
    fi
    emit ""
  else
    emit "$(lab run)${D}no runs/ yet — see README + PLAN (clarke-train …)${R}"
    emit ""
    rule
  fi

  # ---- hardware ----
  local g; g="$(gpu_line)"
  if [[ -n "$g" ]]; then
    emit ""
    local used tot util temp pw plim
    used=${g%%|*}; g=${g#*|}; tot=${g%%|*}; g=${g#*|}; util=${g%%|*}; g=${g#*|}
    temp=${g%%|*}; g=${g#*|}; pw=${g%%|*}; plim=${g##*|}
    used=${used//[^0-9]/}; tot=${tot//[^0-9]/}; util=${util//[^0-9]/}
    temp=${temp//[^0-9]/}; pw=${pw%%.*}; pw=${pw//[^0-9]/}; plim=${plim%%.*}; plim=${plim//[^0-9]/}
    local ug=$(( used / 1024 )) tg=$(( tot / 1024 ))
    local c2="$GR"; (( util < 40 )) && c2="$YE"; (( used * 100 / (tot + 1) > 90 )) && c2="$RD"
    emit "$(lab gpu)[$(bar "$used" "$tot" 40 "$c2")]  ${B}${ug}${R}/${D}${tg}${R} GB   util ${B}${util}%${R}   ${D}${temp}°C${R}"
    local psu; psu="$(psu_dir)"
    if [[ -n "$psu" ]]; then
      local p1 p2 ac pt
      p1=$(hw_val "$psu/power1_input" 1000000); p2=$(hw_val "$psu/power2_input" 1000000)
      ac=$(hw_val "$psu/in0_input" 1000); pt=$(hw_val "$psu/temp1_input" 1000)
      local pmax=${PSU_MAX:-1200}
      local pload=$(( p1 * 100 / pmax ))
      local pc="$GR"; (( pload > 60 )) && pc="$YE"; (( pload > 85 )) && pc="$RD"
      emit "$(lab power)${D}GPU ${pw} / ${plim} W${R}   ${pc}PSU ${p1} W (${pload}% of ${pmax})${R}   ${D}+12V ${p2} W · ${ac} V AC · ${pt}°C${R}"
    else
      emit "$(lab power)${D}GPU ${pw} / ${plim} W  (no PSU sensors)${R}"
    fi
    emit ""
  fi
  rule

  if (( FULL )); then
    emit ""
    emit "    ${B}all runs${R}"
    emit ""
    all_runs | while IFS= read -r l; do emit "${D}${l}${R}"; done
    emit ""
    emit "    ${B}reports${R}"
    local f
    for f in "$ROOT"/reports/*.json; do [[ -e "$f" ]] && emit "  ${f##*/}"; done
    emit ""
    emit "    ${B}eval console tails${R}"
    for f in "$LOGDIR"/eval-*.log; do
      [[ -e "$f" ]] || continue
      emit "  ${D}${f##*/}:${R}"
      tail -3 "$f" | tr -d '\r' | while IFS= read -r l; do emit "    ${l:0:110}"; done
    done
    emit ""
  fi
  emit "$(printf '%*s' 4 '')${D}Ctrl-C to exit · --full for everything · refresh ${INTERVAL}s · 'status' for one frame${R}"
}

if (( TTY && ! STATUS )); then printf '\033[?1049h'; fi
cleanup() { printf '\033[?25h\033[?1049l'; exit 0; }
trap cleanup INT TERM
(( TTY && ! STATUS )) && printf '\033[?25l\033[2J'

if (( STATUS )); then draw; exit 0; fi
while true; do
  draw
  sleep "$INTERVAL"
done
