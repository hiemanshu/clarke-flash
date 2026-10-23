# Plan — what happens after main3's evals

Decision protocol for the next training phase (**main4**), written down at
17:37 before the triple eval runs, so the branch is chosen by numbers, not
by vibes.

## Standing conventions

- **reports/** holds committed eval evidence: `reports/main<N>-<dataset>.json`,
  one per (checkpoint, dataset) pair. Code + reports in every commit;
  runs/ and data/ stay gitignored and regenerable.
- Every GPU run is preceded by a clean working tree (commit first, then run).
- Publish target (when an HF account exists): `hiemanshu/clarke-flash`,
  Apache-2.0 code + weights, model card credits the public datasets used
  (CUAD CC-BY-4.0, typed-extraction CC-BY-SA-4.0, Keyphrase MIT).
- **Dedupe rule, always:** `cuad_all.jsonl` (470 contracts) is never mixed
  into any training set while `real_eval.jsonl` / `typed_test.jsonl` are the
  eval slices — typed-extraction embeds CUAD- and ExtractBench-derived
  records, so training on raw CUAD risks eval contamination. Contract
  training data, if any, comes through `typed_train.jsonl` only.

## The triple eval decides the branch

| Eval set | Question it answers |
|---|---|
| `data/val2.jsonl` | Did the context-aware candidate fix work at all? (regression vs main2: number 20%, duration 14%, date 85%) |
| `data/real_eval.jsonl` | CUAD contracts + keyphrase — can it survive 91%-null, legalese real text? |
| `data/typed_test.jsonl` | 15 real families (invoice, lease, bank statement, wikipedia, squad2, …) — demo or tool? |

### Path A — val2 healthy, typed_test decent (e.g. text ≥ 60-70%, number-when-solvable ≥ 50%)
main4 = the mix: `data/train2.jsonl + data/typed_train.jsonl` (~17.5k records),
5 epochs, `--init-head` from main3's best checkpoint. Rationale: real surface
forms enter training the only safe way (typed_train is disjoint from both eval
slices by the doc-start/doc-end rule); expect the CUAD number to climb toward
typed_test's without ever having seen it.

### Path B — val2 healthy, typed_test weak (< ~40% text)
Pure transfer failure, not a capacity failure. Don't mix real data blindly —
first do a **small transfer run**: 1 epoch on typed_train (or even a few
hundred records), re-eval typed_test; if the slope is steep, main4 = the mix.
If flat, the candidate embeddings still lean on synthetic surface patterns →
diversify synth states (legalese templates, longer/messier tables) before
scaling.

### Path C — val2 itself regressed (cand loss stalls high / number ~flat)
Architecture debug, not data. First artifact: a per-record candidate-selection
dump (top-3 atoms with scores vs the gold atom) for ~20 failing val2 records
— tells us whether confusion is between value-features (log-scale, sign) or
pooled-context (attention), and which fix is cheap.

### Path D (after any A/B/C winner)
- Calibration pass: temperature scaling on val, conformal-style coverage check
  on estimate/forecast (target: 80% interval ⊇ 75-85% realized).
- Serve: `clarke-serve` mirroring clef-server's shape (`/v1/extraction`).
- v2 vision: ExtractBenchPro + rendered synthetic receipts.

## Open questions for the owner

1. Thresholds: are 60-70% (text) / 50% (number-when-solvable) the right
   lines between A/B/C, or do you want typed_test to pass a stricter bar
   (e.g. ≥80%) before we believe the model at all?
2. For Path B's small transfer run: is a ~30-45 min 1-epoch experiment okay
   to run overnight unsupervised, or should it wait for your eyes on numbers?
