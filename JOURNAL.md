# Journal — clarke-flash

Running log of decisions, notes, and learnings. Newest at top.
(PLAN.md holds current policy; this file is the record.)

## 2026-10-23

**14:50 — Goal.** Train Qwen3.5-9B to behave like Cloudflare's Clairic:
one forward pass, no generated tokens, typed answers (spans, numbers,
dates, lists, calibrated distributions). Clef's open-weights repo
(`joint_schema_model.py`) on this machine is the architecture blueprint;
its card says base = Qwen3.5-9B → same backbone family, Apache-2.0.

**Decision: head-first training, not SFT.** Fine-tuning an LLM to emit
JSON can't produce the contract (validity is architectural: typed heads
make invalid answers unrepresentable). Frozen bf16 backbone (~19GB) +
~110M joint extraction head on a 32GB 5090. Verified later: clef-flash's
shipped backbone bytes ≠ base Qwen (they trained LoRA-r256 jointly with
the head) → validates head-only Phase 1 + LoRA as Phase 2, their exact
ladder.

**Decision: synthetic-first data.** Ground truth by construction
(spans exact, arithmetic re-derived, calibration = generator noise).
No annotation phase exists; the scarce resource is generator realism.
Owner chose "synthetic + public datasets"; public data enters EVAL
first, never blindly into train.

**Runs:**
- **main0** (3 ep, head-only): text 89.1%, list-text 98.3%, null F1
  100%, brier_null 0.0 — but number 12.2%, date 0.6%. Matches the
  blog's "reading is strong, arithmetic isn't yet."
- **main1** (Clef-recipe upgrades: permuted field order, surface-name +
  instruction paraphrase randomization, label-smoothed CE, adjacent-bin
  dist "RLCD-lite", Brier metrics): text/list held under permutation
  (90%/98%+), number/date FLAT. → plateau is the digit-bin design,
  not epochs.
- **main2** (candidate pointers: "pick, don't compute" — atoms from
  regex pools + {id,+,−,×,÷} op head): date 0.6→85%, datetime 0→100%,
  list-number 0→39%, text ~90%. Learning: values-as-spans works;
  number stuck ~20% (pool atoms embedded WITHOUT context; and a state-
  local vs global token-index bug made pooled context pure noise —
  still beat digits 2×).
- **main3** (context-window pooling ±8 tokens, duration pool =
  date-diffs + unit-tagged only, eval solvability split): date 100% of
  solvable, list-number 38.7%, number 20/30%, and duration **0.0%
  despite 100% solvable** → smoking gun.
- **Bug found & fixed (18:05):** `compute_loss` only supervised the op
  head when a second operand existed; every single-atom answer (all
  computed durations, most read numbers) trained the pointer but not
  op → argmax(op) turned correct atoms into random arithmetic. Dates
  (no op path) proved it: 100% vs 0%. Fix: op supervised on every
  solved scalar. Committed `2bd06a6`.
- **fix4** (running): 2-epoch resume to heal op head; then triple eval
  (val2 / real_eval [CUAD+keyphrase] / typed_test [15 real families]).

**Data audit (owner-sourced, all inspected live):**
- **CUAD** (CC-BY-4.0): 510 contracts, 41 clause categories, 13.8k
  byte-exact evidence spans, 68% not-present → the honest null-regime
  eval (91% null per window after windowing). Eval-only.
- **LocalLLaMA/typed-extraction** (CC-BY-SA): ships OUR record format,
  15 real families, gold with source labels → `typed_train` (9.5k,
  Path-A training mix) + `typed_test` (1.6k, the acceptance eval).
  Adapter re-verifies every span byte-exact; 1,196 bad spans dropped.
- **MAUD**: answers are comprehension choices (3% verbatim in text, no
  spans) → belongs to clef's `choice` API, not clairic-style
  extraction. Rejected with reason.
- **Keyphrase_extraction** (MIT): verbatim-substring keywords →
  20-record out-of-genre smoke test only (90 rows total).
- **ag_news_corrected_labels**: 50 rows, classification → not for this
  model. **wikipedia/ExtractBench(Pro)**: promising, deferred with
  reasons in PLAN (parser/PDF-stage / vision-v2).
- Leak rule learned the hard way: my CLI silently appended keyphrase to
  the cuad train pool — caught, stripped; eval slices frozen by
  doc-index boundary; typed-extraction's CUAD derivatives mean raw
  CUAD never trains.

**Naming:** clairic-flash → **Clarke** (owner's call; = clerk etymology
+ Clark-Kent persona: looks like an LLM, generates nothing; A.C. Clarke
third-law nod). Hindi candidates (vyakt/pratyaksha) demoted to tagline
depth — pronounceability is load-bearing. Repo `clarke-flash`;
publish target `hiemanshu/clarke-flash`, Apache-2.0, sources credited.

**Infra learnings (each cost a debugging slot — documented so they
don't cost twice):**
- `float()` on a live loss tensor detached BY ME in compute_loss →
  backward on detached tensor; keep graph, detach only for logging.
- `0 × (-inf)` in soft-target CE = NaN loss VALUES with finite grads
  (checkpoints scanned NaN-free) — display poison, but it ate a full
  run's visibility. Finite mask (-1e30).
- cosine schedule recomputed total_steps without the resumed step →
  main1 burned ~40% of its steps at lr≈0 (val rows byte-identical —
  the tell). Fixed: total = resume_step + epochs×len.
- launch-into-busy-GPU = OOM race; GPU work now chains behind
  `while pgrep trainer` guards with `expandable_segments:True`.
- zsh: no word-splitting in `$FILES`, `===` is a glob, echo markers
  break chains; pgrep patterns must survive the clairic→clarke rename
  (they do: `clair(ic|ke)`).
- Leading-space tokens: span answers must strip (BPE glues spaces).
- Eval scorer once had its scoring chain swallowed by an elif — a
  healthy model can print 0.0%; verify decode with a dump probe before
  believing a table.

**Process learnings:**
- Standing order: commit (with why) before every GPU run; never push.
- Every run gets a dashboard line + reports/ evidence JSON.
- The user's "did you stop?" beats = finish the turn with visible
  state, never long silent multi-step chains.
- 80%+ acceptance on solvable-subset real evals; escalate ladder,
  journal walls — don't fight them.
