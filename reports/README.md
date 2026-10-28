# reports/ — how to read this directory

**Status: tracked in git** (owner decision 2026-10-26, after briefly being ignored). At ~4.9 MB
for 399 files the cost is trivial, and it makes every journaled number traceable to the exact
report that produced it — including the raw per-record predictions, which are what the failure
taxonomy and threshold sweeps were computed from.

`logs/` stays out of git (transient curl output, no evidential value).

## File shapes

| suffix | contents |
|---|---|
| `<name>.json` | aggregate report from `clarke-eval`: `metrics` (per field-kind) + `by_domain`, plus `records`, `checkpoint`, `data` |
| `<name>.preds.jsonl` | **raw per-record predictions**, one `{"id", "pred"}` per line — this is what lets us re-score offline without re-running the model |

Every `<name>.json` is produced by a command that also names its input data, so a report is
self-describing: `checkpoint` + `data` + `records` are all inside the JSON.

## Producers

| tool | writes |
|---|---|
| `clarke-eval --out reports/...` | windowed evals (`-real`, `-typed`, `-val3`, `-acordtest`, `-contractval`, `-pm-real`) |
| `scripts/../tmp run_doc_eval.py <vram_cap> ... --max-length 65536` | document-level evals at 65 k (`doc40-*`, `bb-*-doc40`) |
| `scripts/external_baseline.py --out-dir reports/external*` | external-model baselines (also writes `.preds.jsonl`) |
| `scripts/rescore_common.py` | re-scores external runs on the records *all* models completed → `external_doc/rescored/` |
| `scripts/doc_diagnose.py` | per-field `{p_null, correct, gold, pred}` dump used for the failure taxonomy |
| `scripts/compare_external.py` | prints a merged table; writes nothing |

## Naming conventions

- `scale2b-`, `scale4b-`, `scale2b-r2-` — the 2 B / 4 B / seed-replicate arms; `<task>` is
  `real` (168 records), `typed`, `val3`, `acordtest`, `contractval`, or `pm-real` (postmortem)
- `bb-<arm>-real` / `bb-<arm>-doc40` — backbone-sweep arms (`bb-qwen08`, `bb-hymt2`, …)
- `doc40-65k-2b` — 2 B on the 40 full contracts at 65,536; suffix variants mark decode sweeps:
  `-pt<N>` pointer temperature, `-slot<N>` slot-present threshold, `-s20-n60` combined,
  `-dedupe`/absence-of marks the slot-dedup change
- `external/` — the 20-window external baseline; `external/README.md` says which runs are valid
- `external_doc/` — the 40-contract external run; `rescored/` holds the common-subset re-score
- `head2head/` — 2 B / 4 B on the 20 records the external run used
- `census/` — enwiki shard census (19 files), used to verify corpus coverage

## Reading rules

1. **`exact` is set-F1 over *answered* fields.** A model that answers 3 fields and one that
   answers 50 look identical at equal precision. Always pair it with `answered`, or use
   `exact × answered/gold` (exact@all) when comparing coverage.
2. **Check `null_correct`** before trusting a high `exact` — an extractor that fills absent
   fields can look accurate on the fields it answers.
3. Reports with `"failed_calls"` > 0 or `answered: 0` are **not measurements**; see
   `external/README.md` for which external runs are in that state.
4. `records` matters: the 20-record external slice is 19 windows of a single contract, so it is
   a one-document estimate. `doc40` (40 contracts) is the document-level number.
