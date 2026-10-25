#!/usr/bin/env bash
# clarke-flash trainer dashboard (diff-painted TUI; no flutter, no scroll).
#
#   ./scripts/watch_training.sh            # dashboard, repaint only on change (~1s poll)
#   ./scripts/watch_training.sh 2          # poll every 2s
#   ./scripts/watch_training.sh status     # one frame to stdout
#   ./scripts/watch_training.sh --full     # + all runs, reports, eval log tails
#   LOGDIR=.. CLEAR_EVERY=30 ROWS=40 COLS=100  # env overrides (ROWS/COLS when not a tty)
#
# Frame is built into OUT (top) + BOT (bottom-pinned) arrays and compared to
# the previous frame: identical frame = ZERO bytes to the terminal (no
# redraw flutter). When it differs, repaint from home; a true full-screen
# purge runs every CLEAR_EVERY painted frames and on terminal resize (the
# wrap-artifact moments). Checkpoints render as a table when the screen has
# room, else one summary line — the frame's height only changes when the
# checkpoint list itself grows, which is rare.
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
: "${ROWS:=24}"; : "${COLS:=80}"
K=$'\033[K'
CLEAR_EVERY=${CLEAR_EVERY:-30}
TTY=0; [[ -t 1 ]] && TTY=1
W=$(( COLS > 2 ? COLS - 1 : 72 ))

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

# latest train + latest val rows -> KEY=VALUE
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

# one line per checkpoint: "step train val mark" (val@≤step, trend vs previous)
ck_rows() { python3 - "$ROOT" "$1" <<'PY' 2>/dev/null
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
def loss(x): return x.get("loss") if x else None
cks = sorted(int(re.search(r"checkpoint-(\d+)", d).group(1))
             for d in glob.glob(os.path.join(root, "runs", run, "checkpoint-*")))
prev = None
for c in cks:
    t, v = at(train, c), at(val, c)
    vl, tl = loss(v), loss(t)
    mark = ""
    if isinstance(vl, float) and isinstance(prev, float):
        mark = "▼" if vl < prev - 0.005 else ("▲" if vl > prev + 0.005 else "=")
    prev = vl
    print(f"{c} {'-' if tl is None else f'{tl:.3f}'} "
          f"{'-' if vl is None else f'{vl:.3f}'} {mark}")
PY
}

# single-line fallback when there is no room for the table
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

# ---- frame buffers ------------------------------------------------------
declare -a OUT BOT
SINK=OUT
push() { local -n _ref="$SINK"; _ref+=("$1"); }
rule() { push "${D}$(printf '─%.0s' $(seq 1 $W))${R}${K}"; }
emit() { push "$*$K"; }
pad_to() { local -n _r=OUT; while (( ${#_r[@]} < $1 )); do _r+=("$K"); done; }

PREV_STEP=-1; PREV_T=""; LAST_RATE=""

draw() {
  local run STEP="" TOTAL="" EPOCH="" LOSS="" LR="" RPS="" GB="" COMPS=""
  local VSTEP="" VLOSS="" VCOMPS="" VTREND=""
  SINK=OUT; OUT=(); BOT=()
  if (( TTY )); then
    local sz; sz=$(stty size 2>/dev/null </dev/tty || echo "24 80")
    ROWS=${sz%% *}; COLS=${sz##* }
  fi
  (( ROWS < 8 )) && ROWS=24
  (( COLS < 40 )) && COLS=80
  W=$(( COLS - 1 ))

  emit "${B}${CY}  clarke-flash trainer${R}$(printf ' %*s' $(( W - 26 )) '')$(date '+%H:%M')${K}"
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
    done <<< "$state_str" || true

    # Rate must be measured between STEP CHANGES, not between frames: the trainer
    # logs every ~20 steps (~40s) while the UI redraws ~1/s, so frame-delta rate is
    # 0 almost always and the ETA collapsed to the '…' placeholder. Only advance the
    # reference point when the step moves, and reuse the last known rate otherwise.
    local now=$(date +%s%N)
    if [[ -n "$STEP" ]]; then
      if (( PREV_STEP < 0 )) || (( STEP < PREV_STEP )); then
        # baseline: first frame, or a run switch/reset - no rate yet
        PREV_STEP="$STEP"; PREV_T="$now"; LAST_RATE=""
      elif (( STEP > PREV_STEP )) && [[ -n "$PREV_T" ]]; then
        LAST_RATE=$(awk -v s="$STEP" -v p="$PREV_STEP" -v n="$now" -v t="$PREV_T" \
               'BEGIN{printf "%.2f", (s-p)/((n-t)/1e9)}')
        PREV_STEP="$STEP"; PREV_T="$now"
      fi
    else
      PREV_STEP=-1; PREV_T=""; LAST_RATE=""
    fi
    local rate="$LAST_RATE"

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
    emit "$(lab train)loss ${B}${LOSS:-—}${R}   lr ${LR:-—}   ${D}${RPS:-—} rec/s · ${GB:-—} GB${R}${K}"
    [[ -n "$COMPS" ]] && emit "$(lab components)${D}${COMPS}${R}${K}"
    if [[ -n "$VLOSS" ]]; then
      local tc="$GR"; [[ "$VTREND" == UP* ]] && tc="$RD"
      emit "$(lab validation)val@${VSTEP} loss ${B}${VLOSS}${R} ${tc}${VTREND}${R}  ${D}${VCOMPS}${R}${K}"
    fi
    emit ""$K

    # bottom block
    SINK=BOT
    local g; g="$(gpu_line)"
    if [[ -n "$g" ]]; then
      local used tot util temp pw plim
      used=${g%%|*}; g=${g#*|}; tot=${g%%|*}; g=${g#*|}; util=${g%%|*}; g=${g#*|}
      temp=${g%%|*}; g=${g#*|}; pw=${g%%|*}; plim=${g##*|}
      used=${used//[^0-9]/}; tot=${tot//[^0-9]/}; util=${util//[^0-9]/}
      temp=${temp//[^0-9]/}; pw=${pw//[^0-9.]/}; plim=${plim//[^0-9.]/}
      pw=${pw%%.*}; plim=${plim%%.*}
      # quantize live sensors so an idle frame is byte-stable: repaints come
      # from content (jsonl/checkpoints), never from meter sampling noise
      used=$(( used / 2048 * 2048 ))            # 2 GB buckets
      util=$(( util / 10 * 10 )); temp=$(( temp / 4 * 4 ))
      pw=$(( pw / 50 * 50 )); plim=$(( plim / 50 * 50 ))
      local ug=$(( used / 1024 ))
      local tg=$(( tot / 1024 ))
      local c2="$GR"; (( util < 40 )) && c2="$YE"; (( used * 100 / (tot + 1) > 90 )) && c2="$RD"
      emit "$(lab gpu)[$(bar "$used" "$tot" 40 "$c2")]  ${B}${ug}${R}/${D}${tg}${R} GB   util ${B}${util}%${R}   ${D}${temp}°C${R}${K}"
      local psu; psu="$(psu_dir)"
      if [[ -n "$psu" ]]; then
        local p1 p2 ac pt
        p1=$(hw_val "$psu/power1_input" 1000000); p2=$(hw_val "$psu/power2_input" 1000000)
        ac=$(hw_val "$psu/in0_input" 1000); pt=$(hw_val "$psu/temp1_input" 1000)
        p1=$(( p1 / 50 * 50 )); p2=$(( p2 / 50 * 50 )); pt=$(( pt / 4 * 4 ))
        local pmax=${PSU_MAX:-1200}
        local pload=$(( p1 * 100 / pmax ))
        local pc="$GR"; (( pload > 60 )) && pc="$YE"; (( pload > 85 )) && pc="$RD"
        emit "$(lab power)${D}GPU ${pw}/${plim} W${R}   ${pc}PSU ${p1} W (${pload}%)${R}   ${D}+12V ${p2} W · ${ac} V · ${pt}°C${R}${K}"
      else
        emit "$(lab power)${D}GPU ${pw}/${plim} W  (no PSU sensors)${R}${K}"
      fi
    fi
    rule
    emit "${D}  Ctrl-C · --full all runs/reports · poll ${INTERVAL}s · repaint-on-change · purge ${CLEAR_EVERY}s/resize${R}${K}"
    SINK=OUT

    # ---- checkpoint section sized to available room ----
    local -a CK=()
    mapfile -t CK < <(ck_rows "$run")
    local avail=$(( ROWS - ${#OUT[@]} - ${#BOT[@]} - 1 ))
    if (( ${#CK[@]} == 0 )); then
      emit "$(lab checkpoints)${D}none yet — first at step 1000${R}${K}"
    elif (( avail >= 4 )); then
      local n=$(( avail - 1 ))
      (( n > ${#CK[@]} )) && n=${#CK[@]}
      emit "    ${B}checkpoints${R} ${D}(step | train | val@≤step | trend)${R}${K}"
      local line col1 col2 col3 mark cc
      for (( i=${#CK[@]}-n; i<${#CK[@]}; i++ )); do
        read -r col1 col2 col3 mark <<< "${CK[i]}"
        cc="$R"; [[ "$mark" == "▼" ]] && cc="$GR"; [[ "$mark" == "▲" ]] && cc="$RD"
        printf -v line '    %7s  %8s  %8s  %s' "$col1" "$col2" "$col3" "$mark"
        emit "${D}${line}${R}${K}"
      done
    else
      local summ; summ="$(printf '%s\n' "${CK[@]}" | tail -n 4 \
        | awk '{printf "%s:%s%s ", $1, $3, $4}')"
      emit "$(lab checkpoints)${D}${summ}${R}${K}"
    fi
    rule
  else
    emit "$(lab run)${D}no runs/ yet — see README + PLAN (clarke-train …)${R}${K}"
    rule
    SINK=BOT
    emit "${D}Ctrl-C · --full all runs/reports · poll ${INTERVAL}s${R}${K}"
    SINK=OUT
  fi

  if (( FULL )); then
    SINK=BOT
    local l f
    emit "    ${B}all runs${R}${K}"
    while IFS= read -r l; do emit "${D}  ${l}${R}${K}"; done < <(all_runs)
    for f in "$ROOT"/reports/*.json; do [[ -e "$f" ]] && emit "  report: ${f##*/}${K}"; done
    for f in "$LOGDIR"/eval-*.log; do
      [[ -e "$f" ]] || continue
      emit "  ${D}${f##*/}:${R}${K}"
      while IFS= read -r l; do emit "    ${l:0:110}${K}"; done < <(tail -3 "$f" | tr -d '\r')
    done
    SINK=OUT
  fi

  pad_to $(( ROWS - ${#BOT[@]} ))
}

assemble() { printf '%s\n' "${OUT[@]}" "${BOT[@]}"; }

if (( TTY && ! STATUS )); then printf '\033[?1049h'; fi
cleanup() { printf '\033[?25h\033[?1049l'; exit 0; }
trap cleanup INT TERM
(( TTY && ! STATUS )) && printf '\033[?25l'

if (( STATUS )); then draw; assemble; exit 0; fi

PREV=""
LAST_DIMS=""
NEXT_PURGE=0
while true; do
  draw
  frame="$(assemble)"
  dims="${ROWS}x${COLS}"
  forced=0
  if [[ "$dims" != "$LAST_DIMS" ]]; then forced=1; LAST_DIMS="$dims"; fi
  if [[ "$frame" != "$PREV" ]]; then                       # only paint on change
    now=$(date +%s)
    if (( forced )) || (( now >= NEXT_PURGE )); then
      printf '\033[H\033[2J'; NEXT_PURGE=$(( now + CLEAR_EVERY ))
    else
      printf '\033[H'
    fi
    printf '%s\033[J' "$frame"
    PREV="$frame"
  fi
  sleep "$INTERVAL"
done
