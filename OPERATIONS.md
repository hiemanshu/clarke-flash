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
9. **Notify on completion — never leave the owner polling.** Any unattended job
   (GPU training, multi-model API sweep, corpus build) ends with
   `scripts/notify.sh "<title>" "<body>" <mode>`: `short` for a routine finish,
   `full` for a result with numbers, `critical` for a failure or a decision gate.
   Owner preference: **ntfy ping + full detail in Signal, no desktop
   notifications**; credentials live in `~/.config/clarke/notify.env`. Add the
   headline and report path to `JOURNAL.md` in the same step, so the push and the
   record cannot disagree.

10. **Keep README current.** When a result changes the headline numbers, the best
    checkpoint, a supported surface, or a stated limitation, update `README.md` in
    the same step and commit it with the evidence. A stale README is a wrong claim,
    not a cosmetic issue: the journal is the history, the README is what readers see.
    If the change is mid-queue and a later run will supersede it, say so in the
    README rather than leaving the old number up.

## Current live state (updated at each journaling step)

**Last updated 2026-10-28 16:20 IST — no jobs in flight, GPU idle.**

- **Best 41-field (doc40):** `runs/N-mix41/checkpoint-3908` — preamble-supervised
  adapter (`runs/bb-snip-lora-v2`, 800 steps on 1,162 CUAD+LEDGAR preamble crops) +
  head on CUAD + 507 labelled LEDGAR docs, 3,908 steps: exact .5299 / tokF1 .6418 /
  contains .7012 / null_ok .9562 / 411 of 498 / exact@all .4373; latency 0.29/2.62 s.
  L41 was .4252 (.5139/.6346/.6997/.9606/412). Attribution pending run P's
  CUAD-only-head controls.
- **Best 12-field:** `runs/K2-snip-head/checkpoint-940` on
  `runs/bb-snip-lora-mild` (gentle adapter): .6103 / .7311 / .7994 / .8824 /
  291 of 310 / .5729.
- **Externals (doc40, one Cloudflare run, 0 failed calls, 12k output budget, low
  effort; `reports/external_doc_rerun/`):** gemini .5184, opus .4502, gpt .4479,
  clairic .0175 exact@all; latency p50 22.1 / 11.9 / 9.3 / 0.83 s. With L41
  (.4252) we are 2.5 pp behind opus and 9.3 pp behind gemini. The superseded
  2026-10-26 run is in `reports/external_doc/` — its rows were distorted by 5–7
  failed calls each.
- **Queue running:** O (2,000-step gentle adapter -> 12-field gate -> conditional
  41-field run), then P (CUAD-only-head attribution controls at matched 3,908 steps),
  then Q (per-field failure taxonomy on the private company eval). Each notifies.
- **Private company eval:** 39 hand-labelled commercial contracts, 8 fields, 112
  gold-present; eval-only, raw data and predictions gitignored, only
  `reports/company-eval-summary.json` public. L41 .1750 vs N .2329 exact@all; parties
  only: L41 .4915 (39/39 answered) vs N .5085 (29/39).
- **Data threads:** `data/llm_preambles.jsonl` 271/545 complete (Kimi quota cut the
  run); `data/edgar_lm_4k.bin` 105.8M tokens built and unused;
  `data/sec_lm_65k.bin` 425M tokens blocked by the ~4k LoRA VRAM wall.
- **Hardware:** five GPU full-chip resets, all under 65k training (three on
  2026-10-27, two on 2026-10-28). With **persistence mode enabled** (owner,
  2026-10-28) L41 completed 1,880 steps with no fault — one run, not yet proof.
  Power limit still uncapped.
- **Spend / access:** Cloudflare AI Gateway topped up 2026-10-28. CommandCode plan
  includes Kimi-K3, claude-sonnet-5-5, gemini-3.8-flash and the open models, but
  **not** opus-5-5 (Provider+), gemini-4-argon (Pro+) or gpt-6.1-sol (Max+).
