# OPERATIONS — persistent working rules + live state

This file survives context compaction. If you are an agent resuming with a
compacted context, READ THIS FIRST, then JOURNAL.md (newest entries at top),
then PLAN.md.

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
