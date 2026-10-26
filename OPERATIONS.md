# OPERATIONS — persistent working rules + live state

This file survives context compaction. If you are an agent resuming with a
compacted context, READ THIS FIRST, then JOURNAL.md (newest entries at top),
then PLAN.md.

## Artifacts

`reports/` **is tracked** (2.7MB, eval JSONs + raw prediction dumps) so every journaled number
can be traced to the report that produced it. `logs/` is ignored (transient). Every result still
goes in the journal with its number, run dir and report path - the report is the evidence, the
journal is the claim.


## Reporting metrics (owner rule, 2026-10-26 — violated repeatedly before this was written)

**Never report `exact` alone.** Every result table, every arm, every model — ours included —
carries the full set:

| metric | why it must be shown |
|---|---|
| `exact` | set-F1 over *answered* fields only; flatters a conservative extractor |
| `tokF1` | token-level F1; credits the right clause quoted with different trimming |
| `contains` | did the answer land on the right text at all (lenient either direction) |
| `null_ok` | share of truly-absent fields left absent; our differentiator, so hiding it is self-serving |
| `answered` / gold | recall; the binding constraint at document level |
| `exact@all` | `exact × answered/gold` — the only number that combines precision and recall |

The external baselines have reported tokF1 and contains since the first comparison, so showing
only `exact` for our arms is an apples-to-oranges presentation that happens to favour us. If a
metric is missing for one side, say so explicitly rather than dropping the column.


## Context regime (owner rule, 2026-10-26)

**Everything after the 2026-10-26 backbone sweep is trained AND evaluated at 65,536 tokens.**

| use | dataset | length |
|---|---|---|
| training | `data/doc_train_470.jsonl` (470 full contracts, median 7,149 tok, 38.8% multi-item) | `--max-length 65536`, batch 1 |
| primary eval | `data/doc_eval_40.jsonl` (40 held-out full contracts) | `--max-length 65536` |
| secondary/diagnostic | `data/real_eval.jsonl` (168 × 16k-char windows) | 8192 — no longer the headline |

Consequences to design around, not discover later:
- 470 documents/epoch at batch 1 = only ~470 optimizer steps per epoch; long-context runs need
  more epochs (4-8) or more document-length data, not a bigger LR.
- 4 of 470 docs exceed the 63k state budget after the ~2.4k-token schema and are truncated.
- 65k training is head-only territory: LoRA at 9B cannot fit 32GB at any rank (adapter state alone
  is ~13GB at r256), and LoRA at 2B/65k is still unverified.

## Monitoring

`./scripts/watch_training.sh` — TTY: ncurses dashboard for the CURRENT run only (progress,
component losses, best val, checkpoints, staleness, GPU, per-core CPU, RAM/swap/disk, network
limited to interfaces holding an IP, log tail; keys q/space/l/r/?). Non-TTY or `status`: the
older diff-painted one-frame output. `WATCH_TUI=0` forces the old path.

## Working rules (non-negotiable)

1. **Journal every substantive event.** One entry per event, fresh entry file per
   append (never reuse /tmp/opencode/journal_*.md — double-prepend bug happened
   once), newest-first at the TOP of JOURNAL.md, **prepend exactly once**
   (`assert entry not in j` before writing).
2. **Commit as you go.** Every journal append + code/data-structure change gets
   its own commit with a "Why:" paragraph. ALWAYS local. NEVER push. NEVER
   commit credentials or the huge raw data files (data/ is gitignored).
3. **Report evidence, not optimism.** Verdicts quote measured numbers with
   sample sizes; failed hypotheses are shelved openly, not quietly dropped.
4. **Ask before destructive/risky actions** (deletes of user work, system
   packages, killing user processes, network sends). Downloads of public
   datasets the owner pointed at are pre-authorized. **NEW (2026-10-24 21:45):
   no new binary/toolchain/package installs without an explicit owner go-ahead
   AND source verification - a feasibility question is never permission; pip
   libs inside the uv project are the only pre-authorised class.**
5. **GPU discipline:** solo tenant, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
   commit before every GPU run, logs under /tmp/opencode/, ETA stated up front,
   gaming-pause protocol = stop at next checkpoint when owner asks.
6. **Shell-kill discipline** (five self-kill incidents): never run
   `kill $(pgrep -f …)` in a command line that contains the same literal string;
   enumerate PIDs in a read-only call, kill by explicit PID; bracket every
   pattern you put in a command that also mentions the name. zsh does NOT
   word-split unquoted vars — use `| xargs -r kill`.
7. **Data hygiene:** eval surfaces are frozen (typed_test 400-1612, real_eval,
   all wiki holdouts, acord_test). Leak-audit every new corpus against all eval
   surfaces before it enters a mix; content-hash checks, not just title hashes.
8. **Unattended operation:** prefer manifest-keyed, resumable jobs; pre-register
   decision rules in the journal BEFORE the run so no judgment is improvised
   while the owner sleeps.

## Current live state (updated at each journaling step)

- expReal2 (GPU, run dir runs/expReal2): train ~6,600+/7,658, verdict ~22:45.
  Battery includes expReal2-acordbase.json (pre-train ACORD baseline .009).
- batch3 chain (harness shell sh_16eaafa19001ZgByUtePvvStSy): waits census 19/19
  -> mines 4 proven families into data/cache/wikiib -> assembles
  data/mixes/wikiib_train.jsonl -> waits expReal2 DONE -> pre-registered init
  rule (CUAD >= .097 ? expReal2 final : expSpan-5852) -> 1 epoch on the ~88k
  record mix -> battery. Log: /tmp/opencode/batch3.log
- census: ~10/19 shards -> reports/census/*.json; driver pid in /tmp log.
- corpus: 19/19 sha256-verified in data/raw, 235 GB plain XML in data/raw/plain.
- Decisions pending for the owner: LoRA (rung 4) vs more data if CUAD stalls;
  in-prose wikitable parser (batch-4 candidate); life-date family (batch-4).
