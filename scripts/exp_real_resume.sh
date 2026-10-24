#!/usr/bin/env bash
# Resume the owner-paused expReal run — WITH ACORD folded in per owner directive
# (16:50: "can we just add acord to it now? before we resume"). This retires the
# separate expAcord batch; attribution trade-off declared in journal.
# Pause point: checkpoint-1000. Runs 2 FULL epochs of the enlarged mix from the
# resumed step (~8,658 total; trainer anneals and final-saves itself), then battery.
# ACORD attribution: pre-train baseline eval on checkpoint-1000 (all training
# BEFORE any ACORD exposure) vs post battery on the final ckpt -> the delta
# isolates what ACORD (+the extra epochs) taught in its own domain.
set -u
cd /home/hiemanshu/personal/clarke-flash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/tmp/opencode/expreal_resume.log
RUN=runs/expReal-mixinit
echo "RESUME+acord $(date +%H:%M:%S)" > "$LOG"
uv run clarke-eval --head "$RUN/checkpoint-1000" --data data/acord_test.jsonl --limit 228 \
  --batch-size 4 --agg-atoms --out "reports/expReal-pretrain-acordbaseline.json" >> "$LOG" 2>&1
echo "acord pre-train baseline done $(date +%H:%M:%S)" >> "$LOG"
uv run clarke-train \
  --train data/train3.jsonl data/typed_train.jsonl data/contract_train.jsonl \
  data/cuad_train.jsonl data/keyphrase_train.jsonl data/acord_train.jsonl \
  --val data/val3.jsonl --epochs 2 --batch-size 6 --grad-accum 2 --lr 1e-4 \
  --rerank --query-pool attn --agg-atoms --diverge-stop 18 \
  --run-dir "$RUN" --resume "$RUN/checkpoint-1000" >> "$LOG" 2>&1
echo "trainer finished naturally $(date +%H:%M:%S)" >> "$LOG"
BEST=$(ls -d "$RUN"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)
CK="$RUN/checkpoint-$BEST"
echo "battery on $CK" >> "$LOG"
uv run clarke-eval --head "$CK" --data data/real_eval.jsonl --limit 168 --batch-size 4 --agg-atoms \
  --out "reports/expReal-mixinit-real.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/typed_test.jsonl --limit 400 --agg-atoms \
  --out "reports/expReal-mixinit-typed.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/acord_test.jsonl --limit 228 --batch-size 4 --agg-atoms \
  --out "reports/expReal-mixinit-acordtest.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/contract_val.jsonl --limit 300 --batch-size 4 --agg-atoms \
  --out "reports/expReal-mixinit-contractval.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/val3.jsonl --limit 400 --agg-atoms \
  --out "reports/expReal-mixinit-val3.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/real_eval.jsonl --limit 24 --batch-size 2 --agg-atoms --postmortem \
  --out "reports/expReal-mixinit-pm-real.json" >> "$LOG" 2>&1
echo "EXP REAL RESUME DONE $(date +%H:%M:%S)" >> "$LOG"
