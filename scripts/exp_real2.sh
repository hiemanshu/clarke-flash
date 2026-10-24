#!/usr/bin/env bash
# Exp(REAL2) — owner directive: FULL FRESH RESTART on the merged real-data mix.
# Not a resume: clean 2-epoch cosine schedule from step 0 over the enlarged mix
# (cuad_train + keyphrase + ACORD all trained from the start, evenly mixed).
# Init = expMix-nospan/checkpoint-6852 (established stack base: best val loss,
# binding mechanism proven, reranker/agg/fix5 all on). expReal-mixinit/checkpoint-1000
# is kept in place as archived history — nothing deleted.
set -u
cd /home/hiemanshu/personal/clarke-flash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/tmp/opencode/expreal2.log
RUN=runs/expReal2
INIT=runs/expMix-nospan/checkpoint-6852
echo "START fresh init=$INIT $(date +%H:%M:%S)" > "$LOG"
# Attribution anchor: the init head has never seen ACORD (nor cuad_train) —
# evaluate on the ACORD holdout BEFORE training exists.
uv run clarke-eval --head "$INIT" --data data/acord_test.jsonl --limit 228 --batch-size 4 --agg-atoms \
  --out reports/expReal2-acordbase.json >> "$LOG" 2>&1
echo "baseline done $(date +%H:%M:%S)" >> "$LOG"
uv run clarke-train \
  --train data/train3.jsonl data/typed_train.jsonl data/contract_train.jsonl \
  data/cuad_train.jsonl data/keyphrase_train.jsonl data/acord_train.jsonl \
  --val data/val3.jsonl --epochs 2 --batch-size 6 --grad-accum 2 --lr 1e-4 \
  --rerank --query-pool attn --agg-atoms --diverge-stop 18 \
  --run-dir "$RUN" --init-head "$INIT" >> "$LOG" 2>&1
BEST=$(ls -d "$RUN"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)
CK="$RUN/checkpoint-$BEST"
echo "trained $CK $(date +%H:%M:%S)" >> "$LOG"
uv run clarke-eval --head "$CK" --data data/real_eval.jsonl --limit 168 --batch-size 4 --agg-atoms \
  --out "reports/expReal2-real.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/typed_test.jsonl --limit 400 --agg-atoms \
  --out "reports/expReal2-typed.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/acord_test.jsonl --limit 228 --batch-size 4 --agg-atoms \
  --out "reports/expReal2-acordtest.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/contract_val.jsonl --limit 300 --batch-size 4 --agg-atoms \
  --out "reports/expReal2-contractval.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/val3.jsonl --limit 400 --agg-atoms \
  --out "reports/expReal2-val3.json" >> "$LOG" 2>&1
uv run clarke-eval --head "$CK" --data data/real_eval.jsonl --limit 24 --batch-size 2 --agg-atoms --postmortem \
  --out "reports/expReal2-pm-real.json" >> "$LOG" 2>&1
echo "EXP REAL2 DONE $(date +%H:%M:%S)" >> "$LOG"
