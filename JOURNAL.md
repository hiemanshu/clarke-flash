**03:40 — B2 verdict + decode attribution. expFinal (full stack) launched 03:31.**
B2 (rerank+attn on train3, init expA1D, fix5-decode in eval): typed text .144->.157,
number .056->.068 (reranker: p@3 .477->.505, med 3->2, p@1 .200->.196 — improves
top-band recall, NOT top-1; pair net's ceiling is the features), retention perfect
(val3 text .900). CUAD real: exact .0109->.022, null_ok 0->.727.
ATTRIBUTION via decode-matched re-eval of expA1D (fix5 DECODE only, untrained present):
CUAD null_ok .7365 (!) >= B2's .727 -> the 0.0 -> 0.7+ leap was 100% DECODE ARTIFACT:
the null head has been trained+calibrated on lists all along; the old decoder simply
never consulted it (fabricated items regardless). The fix5 LOSS half (present zero-vec
supervision) adds robustness, not the win — measured at expFinal. Lesson for the
morning: p_null existed, honesty was optional; every answer type must consult it.
expFinal = rerank+attn+scatter+agg-atoms+fix5-loss, init B2-5852, verdict ~05:40;
then decide morning: resume-extension (same run, +2ep) vs fresh analysis. CPU notes:
pair-net memory footprint fine (19.3GB steady), OOM warnings during eval transitions
were recoverable cache-flush retries, confirmed no zombies.

## 2026-10-24

**01:25 — Exp(A+D) verdict: real +1.1pp (text 13.3->14.4), adjacency SHORTCUT KILLED on
synthetic (val3 text 89.9% with cues dissolved — the pointer genuinely reads), but transfer
to real docs stayed marginal. pm deltas: squad2 p@1 .197->.218, drop number p@1 .189->.200,
p@5 .614->.627. The p@5~.60 / p@1~.20 selection gap is now the measured, stable signature
across text+number, squad2+drop. DEVIATION FROM PRE-REGISTRATION (documented): rule said
B1 (rerank on main4) since text <16%; firing **B2** (rerank STACKED on expA1D) instead —
expA1D is strictly better than main4 on EVERY real metric AND held val3, so B1 would isolate
the reranker on a dominated base; A/B vs expA1D-as-control keeps the single-factor test clean.
B2 launched 01:28 (init expA1D-5852, 8 fresh pair tensors verified), verdict ~03:50.
Slot after B2 (pre-registering now): combined-winner candidate = --rerank --query-pool attn
--agg-atoms, init B2-best, 2ep on train3+typed -> finish ~06:10, evals ~06:35; resume-same-
run for further epochs until ~07:45 if ahead. If B2 reranker fails to close p@1 gap: drop
rerank from the combo, keep attn+agg.

# Journal — clarke-flash

Running log of decisions, notes, and learnings. Newest at top.
(PLAN.md holds current policy; this file is the record.)

## 2026-10-23
**23:30 — --agg-atoms built while GPU busy: table solvability 25% -> 100% (CPU-verified).**
Root cause of the val2 table wall (number exact 2.5%) was NOT the model: gold
aggregates were never in the candidate pool (solve() found 79/316). Pool gains
virtual atoms: per-numeric-column sum/mean/min/max/median/count over detected
CSV blocks (header + >=3 consistent rows, numeric-header rejection so headerless
blocks are skipped rather than silently dropping row 0), anchored at the HEADER
CELL so context-pooling reads the column name ("total revenue" -> 'revenue'
header + the sum value). field_atoms mirrors solve's index order (comment pins
it); value_features marker 4.0; decode honesty: aggregates origin=computed,
no grounding span; literals keep read+span (direct decode test passes both
ways). Non-CSV domains: numbers pools byte-identical, aggs=0 everywhere except
comps (24 — its median estimate just became id-op reachable!) and forecast (6,
inert). Flag-gated (train+eval --agg-atoms, default off), no regen needed
(pool computed at encode time). Slot: stacks onto tomorrow's winner run.
Also wrote exp_b1.sh / exp_b2.sh so the 01:30-02:15 decision fires instantly.

**23:24 — pm-drop completes the branch picture: drop-NUMBERS p@1 .19/p@5 .61, median rank 3
(reranker's exact sweet spot); drop text .19/.54. Trainer healthy (55 steps/min,
attn loop costs nothing) -> Exp(A+D) verdict ~01:25.**
Next-run rules (pre-registered tonight, applies at ~01:30):
- expA1D typed text >=26% (2x main4) OR squad2 p@1 >=1.5x -> A+D works -> Exp(B2)
  = --rerank --query-pool attn, init expA1D best, 2ep lr 1e-4 (stacked).
- expA1D typed text <16% but drop numbers p@5 >=.6 holds -> Exp(B1) = --rerank only
  (mean pool, unscattered train2+typed), init main4-11704, 2ep — numeric half.
- Both fail -> head-surgery discussion in journal, sleep GPU.
While GPU busy: building --agg-atoms candidate-engine fix for the TABLE wall
(val2 number 2.5%: gold aggregates aren't pool atoms -> solvable 25% ceiling;
virtual per-column agg atoms anchored at header cells; pool is computed at
encode time -> no data regen needed). Test offline only; no GPU before 01:25.

**23:20 — Postmortem verdict: no window loss anywhere; pointer sees everything, points blind
on real docs. Exp(A+D) launched.**
Breakdown (main4/checkpoint-11704): val2 synthetic -> candidate p@5 0.96-1.0, median rank
0-1 (pointer PERFECT at home). typed-squad2 text: p@1 0.197/p@3 0.40/p@5 0.52 with
ZERO out-of-window (CPU decomposition: 1004 non-null golds all mapped; the 451
"unrankable" were legitimate nulls = squad2's unanswerable design, 47%).
CUAD: 89 scored = all 9% non-null regime (nothing truncated), p@5 0.034, median
rank 743 -> near-random on 41-field contracts.
=> ranking failure (52% top-5 on squad2) + query failure (CUAD random); volume and
window falsified. Pre-registered branch: p@5 low => Tier A+D. expA1D = --query-pool
attn on scattered train3 (+typed), 2 ep, lr 1e-4, init main4-11704, dev evals +
before/after postmortems chained (~2h). drop-domain NUMBERS postmortem added to the
chain (first-96-typed-slice was all-squad2; Tier B's own zone undiagnosed ->
pm-drop decides whether reranker fires tomorrow). Confirmation slice (typed_test
400-1612) untouched; dev = first 400 only.
Estimate/forecast bin p@1 ~0.0-0.08 vs ~200 bins = median_rank 36-71, consistent with
mode-vs-draw noise, not breakage (coverage 89/79% backs this).

**23:17 — main4 verdict: volume DOUBLES real transfer, wall persists; band 10-35%
-> postmortem decides the surgery.**
typed_test (400): text 6.5->13.3% (squad2 13.5% w/ null_correct 86.6%, drop 12.5%),
number 4.6% (5.6% when-solvable), list-text 10%. mean_conf 0.10 — calibration
transfers, pointing still doesn't (third independent confirmation).
val2 held: text 89.7, number 50.7 (76.7 when-solvable), date 85.1 (100% solvable),
duration 100, datetime 100, list-text 99.4 (probe1's forgetting recovered),
table-number still 2.5% (solvable 25%) — table is synthetic side's residual wall.
real_eval: CUAD list-text 1.09% — unchanged in three attempts.
Volume is not the lever for pointing. Val-loss best at ~9500 (4.545), final used
(11704) — tail flat, no overfit drama. Postmortem chain armed on typed/val2/real:
p@5 high/p@1 low -> Exp(B) reranker; p@5 low -> Exp(A+D) attn-query + scattered
train3; p@1 high & exact ~0 -> decode hunt. GPU held for it.


**22:45 — Tier A + Tier D built, tested, committed; scatter data pre-made.**
Tier D `--cue-scatter` (afa12c9): rewrites `Label: value` cue lines to a
placeholder, plants the bare value >=2 lines below. Spans remapped
POSITIONALLY + byte-verified (search-based re-derivation rejected: money
format mismatch + repeated atoms silently mangle labels); uncarryable field
-> dropped whole, never corrupted. Atom-only lines (Subtotal/Tax operands,
no gold span) also move — the numeric pointer is the same adjacency victim.
Verified: ticket gold-adjacency 600->118, invoice cue-atom lines 1500->636,
NON-target domains byte-identical, build_targets reaches every gold exactly
as before (944->944 cand, 360->360 spans, 0 fields dropped). train3/val3
generated from the actual train2/val2 records (scatter applied per-record,
identical base — cleanest possible contrast to main4) so a scatter-armed
run needs no GPU-idle data step.
Tier A `--query-pool {mean,last,attn}` (166ee51): mean/last/attn all
forward+backward finite + config round-trip; rerank+attn combined head gets
gradients on both new nets (likely surgery config). Caught attn dim bug
(Linear must be hidden_size->1, pools BEFORE field_projection).
Verdict instruments ready: postmortem p@1/3/5 decides branch (p@5 high/p@1
low -> Exp(B); p@5 low -> Exp(A+D) on train3; exact~0/p@1 high -> decode
hunt). GPU held for main4; nothing above touched it. exp_template.sh staged
in /tmp/opencode for the launch after verdict.

**21:40 — arsenal pre-built (Phase 0).** Postmortem mode (`clarke-eval
--postmortem`: gold rank p@1/3/5 + unrankable rate by domain x kind) and
Tier B pairwise reranker (`--rerank`: cross-encoder features [q;c;q*c;|q-c|]
-> MLP, flag-gated, config round-trips, old checkpoints load unchanged)
committed BEFORE main4's verdict, CPU-sane both modes. Branch mapping at
verdict: p@5 high/p@1 low -> Exp(B) only; p@5 low -> Exp(A+D); exact ~0
with p@1 high -> decode bug hunt. Constraint honored: repo code + existing
python libs only, no installs, no pushes.


**19:15 — probe1 FULL verdict.** Real-eval stayed ~dead (CUAD 0.4%, typed text
6.5%: squad2 7.1%, drop 4.7%, keyphrase 3.8%) while synthetic drifted mildly
(text 90.2→86.6, **list-text 99.4→79.4**, number 50.9→43.0, duration/date hold).
Interpretation: the gentle probe neither proves nor kills the volume hypothesis
— but squad2's *medium-length* contexts failing at 7% means the pointer is NOT
length-blind-luck; it is instruction-semantics-blind (overfit to synthetic's
label-adjacency: "Subtotal: $X" puts gold next to the cue; real answers sit
wherever meaning lives). Main4 (launched 19:15, the pre-registered volume test:
4 ep mix, lr 2e-4) decides: typed-text >~35% ⇒ volume; flat ⇒ architecture
surgery on the routing/pointer binding (position-strided candidate attention),
then re-run typed eval. If surgery wins, NQ long-tier pull becomes worthwhile —
until then more real data just feeds a mechanism that can't point.
Note: table-domain number was already 2% (multi-row aggregates — the documented
wall, unchanged).

**19:10 — probe1 verdict: partial transfer; wall refined.** After 1 epoch of
typed_train: typed_test text 0.6→6.5%, null_ok 0→72%, conf still 0.04, number
flat at 4%. Calibration/style transfers; pointing does not. Crucially squad2
(~500-800 token contexts, well within typed_train exposure) sits at 7% — so
the wall is NOT purely state length; it is the pointer/selection machinery on
real prose with many fields, at either severe-undertrain or architecture
resolution. Next rung (PLAN Path A): **main4 = train2 + typed_train, 4 epochs,
lr 2e-4, bs6, init-head probe1-final** — the volume test. squad/drop >~40% ⇒
data volume; ~10% ⇒ architecture surgery (chunked evidence routing /
position-robust pooling), then NQ long-tier. Probe val2 drift: loss 3.83→4.16
plateau (mild forgetting; replay in mix protects it).

**18:35 — fix4 gate PASSED on synthetic; hard wall on real text; probe launched.**
fix4 (op-head heal, 2 ep) results, val2: duration 0→**100%**, number 20→**51%**
overall / **77% when-solvable** (the remaining 34% = multi-row aggregates, by-design
unsupervised), date 85%/100%-solvable, datetime 100%, text 90.2% (null F1 100%,
brier_null 0.0), list-text 99.4%. Weaker: list-number 39%, estimate APE 30%
(coverage 91% = too wide), forecast APE 11% cov 80% (good). The candidate-pointer
thesis is fully vindicated on home turf.

Real world, same checkpoint: **typed_test text 0.6%, CUAD list-text 0.1%** —
near-total collapse despite solvable_rate 0.81 (the values ARE in the text; the
model just can't find them). Diagnosis (measured, not modeled): states avg 472
chars in training vs 8-24k in eval; ~7 fields/record trained vs 41; 7% nulls
trained vs 91% CUAD. Hypothesis ranked: (1) length/count OOD on the pointer path,
(2) p_null regime shift, (3) instruction style. This is PLAN Path B →
**probe1**: 1 epoch typed_train ONLY, lr 1e-4 (gentle, avoid destroy), init-head
fix4-9338; then eval typed_test (does it adapt?), val2 (did we catastrophically
forget?), real_eval. Eval upgraded meanwhile: by_domain family breakdown (don't
confuse a measuring wall with a modeling wall) + per-solvable splits. Reports
commit as evidence; probe answers decide main4 (mix) vs length-OOD surgery
(context-windowed pointers / position-robust pooling) next.

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
