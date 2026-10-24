**18:20 — WIKI TABLES RECON (owner question while expReal2 cooks): data shape understood,
converter built & validated (scripts/adapt_wiki.py), training shards downloading.**
DATA SHAPE (live-audited on one 60MB shard, 4,604 articles): the 'tables' column is a
JSON string of PRE-PARSED tables — identifiers ("transmissions.regular_table1"), header
ROWS (multi-row headers common: 977/9,772), rows of {"value": cell} (only other key seen:
nested_table x143 -> skip). Verbatim cell strings => deterministic pipe-rendered state
with byte-exact spans, same trick as synth table_record but REAL data. Shard85 yielded
14,464 tables -> 9,772 usable -> 6,196 records (4,859 train-shard + 1,337 eval-shard by
URL hash). Scale read: 86 shards / 37.1 GB total — one full shard (~371MB) ≈ 30k+ records;
we need ~3, not millions.
RECORD FAMILIES (mirrors the synthetic table regime + two things it lacked):
- row lookups (text+span): "value in COL for the row with KEYCOL equal to KEYVAL" — the
  table-text .076 target; duplicate cell values make position-binding necessary (real
  near-misses for free)
- row_count (number integer), sum of clean numeric column (agg-atoms path), argmax cell
  (text span)
- HONEST NULLS x2/record: query for a provably-absent column + absent row key -> the
  not-present regime for tables (train null rate .37)
VALIDATION DONE: self-audit all spans byte-exact (assert in converter); encoder smoke
390-518 tokens/record (cheap!); 0/1,612 typed_test 300-char prefixes found in shard85
wiki text — squad2-in-tables risk looks nil, will re-run the check at full-corpus scale
and exclude any matching article. Multi-row headers merged per column w/ uniqueness
suffix; rows capped at 24; fake null-columns chosen from names proven absent per table.
WHAT'S LEFT ("what would we need to do" — answer: converter done, then):
1. 3 full shards downloading NOW in background (data/raw/wiki_shard0/1/2.parquet, nice'd,
   GPU untouched; wiki_dl_done sentinel).
2. Convert shards 0-2 -> data/wiki_train.jsonl (~50-90k records, cap/stratify if huge:
   sample <= ~40k, keep domain spread), shard85 -> data/wiki_test.jsonl HOLDOUT (shard-
   disjoint articles = cleanest split we've ever had).
3. Battery + chain next batch: add wiki_test eval to every chain script; mix gains
   wiki_train. Expected step-time effect: records are SMALL (390-518 enc) -> near-free
   tokens despite big counts.
4. Bonus read: typed_test families already include an 'extractbench' family — LocalLLaMA's
   set ships some extractbench-derived records; real ExtractBench PDFs remain eval-stage
   (parser needed) per plan.
NOT LAUNCHED into training yet — wiki joins the mix at batch 3, after expReal2's verdict
(~22:30) decides whether merged-real-data worked; if it did, wiki tables are the obvious
continuation (same "real shape at scale" bet, next weakest regime).

**18:00 — OWNER DIRECTIVE: "restart from scratch with the new data set" -> expReal2 launched.**
Interpretation (stated): fresh full 2-epoch run over the merged mix — NOT the mid-schedule
resume from checkpoint-1000 — so cuad_train (1,714), keyphrase (20) and acord_train (684)
are seen evenly from step 0 under one clean cosine schedule. Init stays expMix-nospan-6852
(the established stack base: proven binding, reranker/agg/fix5 on; a random-head restart
would torch the day's progress and isn't what a stacking protocol means by fresh). The
paused expReal-mixinit/checkpoint-1000 is kept as archived history, untouched.
Chain (scripts/exp_real2.sh, /tmp copy identical): PRE-TRAIN attribution baseline (init head
vs acord_test — zero ACORD/cuad exposure -> reports/expReal2-acordbase.json), then 7,658
steps train, then battery: real / typed / acord_test / contractval / val3 / pm-real.
Owner's bot.py test vacated the card at ~17:58 (9.7GB gone before launch — solo-tenant
rule honored, no kill needed). Launch 17:59: baseline eval running, 100%/27.8GB confirmed.
Pace note: merged mix is token-heavy (long CUAD + mid ACORD windows); if ~34 steps/min
holds, train ends ~21:45, battery -> verdict ~22:30.
Reads pre-registered: CUAD real_eval .097 -> ? (headline; both new corpora are real contract
text so "does real data move CUAD" is the clean question); acord_test vs its own baseline
(did the ACORD regime teach its domain; expert-graded hard-negative training's payoff);
typed/CUAD-derived subset vs .200/.163 retention; contract_val vs .995 (does real data cost
the synthetic binding?); val3 retention band. Decision tree after: big real gain -> scale
(wiki tables next, longer schedules, real:real-ish mix); flat/mixed -> LoRA question with
this whole table + the now-clean schedule artifact.

**16:52 — Owner directive: fold ACORD into the paused expReal BEFORE resume (one run instead
of two). Mix now 22,971 recs/epoch: train3 + typed + contract-synth 3k + REAL cuad 1,714 +
keyphrase 20 + ACORD 684. Resume runs 2 FULL epochs from checkpoint-1000 (~8,658 steps,
trainer anneals + final-saves itself — cleaner than an external kill), then pre/post ACORD
evals + full battery -> ~4h from launch. expAcord retired as standalone batch; wiki-tables
and ExtractBench-parser are now the batch-3 candidates.**
ATTRIBUTION LEDGER for this decision (what we do and don't learn):
- "Does REAL data move CUAD?" stays clean: expMix (.097, synth-only regime) -> this run's
  real-enriched mix is the step. Both new corpora are real contract text, so the headline
  question is unconfounded even merged.
- "CUAD-windows vs ACORD-windows, which mattered more?" becomes unseparable in one run.
  Partial recovery: pre-train baseline (checkpoint-1000, zero ACORD exposure) vs post
  battery on acord_test isolates the ACORD regime's own-domain gain; CUAD keeps its own
  real_eval readout. Full separation would need the ablation I traded away (2 runs vs 1).
- The final model is 2.27 epochs over the enlarged mix at lower end-lr than expMix's
  endpoint (same 1e-4 peak, longer schedule) — so the comparison is "more real data AND
  more training" vs expMix, not data alone. Stated up front so nobody over-reads .156.
Cost accepted: attribution granularity. Benefit accepted by owner: ~3 GPU-hours and one
full experiment cycle saved. GPU remains released to owner for testing until they say GO.

**16:45 — OWNER PAUSE: expReal stopped for owner testing (5090 handed back).**
State at stop: step 1,460/7,430 (epoch 0), last banked checkpoint-1000; ~13 min of
unbanked steps lost (checkpoint cadence 1000 vs kill moment). Chain bash SIGKILLed
BEFORE battery phase (expSpan-race lesson applied), trainer SIGTERMed, all workers
gone; GPU verified 0% / 56 MiB. Notes: (1) pkill self-match trap fired AGAIN (the
'from multiprocessing' alternation matched my own shell's cmdline) — bracket every
alternative, not just the first. (2) Resume recipe: /tmp/opencode/exp_real_resume.sh =
--resume checkpoint-1000, external stop at checkpoint-7000, full battery incl NEW
acord_test baseline readout (first ACORD-domain eval of any checkpoint). Deviation to
declare: resume's loader restarts epoch shuffles, so coverage by step 7,000 is ~2.27
epochs of mix vs planned 2.0, and its cosine schedules to 8,430 (killed at 7,000 where
lr<1% peak — tail shape moot). Acceptable for a continuation run; noted for the
verdict read. Verdict clock after resume: ~3h05m train + ~25m battery.

DATA CHANGES SUMMARY (owner question, as of this stop):
expReal mix (stopped) = expMix mix + TWO new files: cuad_train.jsonl 1,714 REAL CUAD
windows (raw CUAD_v1.json docs[0:470] via public.py converter; 83 windows dropped by
leak audit: 6 exact-id eval dupes + 32 train-region contracts matching typed-cuad-test
company+date — typed-extraction is CUAD-derived so whole-contract exclusion needed;
verified 0 overlap vs real_eval AND typed_test) and keyphrase_train.jsonl 20 recs from
validation.parquet (test.parquet IS the eval; 0 content overlap). Unchanged: train3,
typed_train (9.5k, incl 61 CUAD-shaped now trainable since max_fields 48),
contract_train 3k synth. Engine-side data plumbing changes already baked into mixes:
max_fields cap, slash-date atom regex, agg-atoms train/eval parity. STAGED NOT TRAINED:
ACORD 684 train windows (attorney queries, 4-star gold, 3-star expert hard decoys,
312 honest nulls) + acord_test 228 holdout -> expAcord batch 2. REJECTED w/ evidence:
MAUD, structuredlogs, extraction-examples; KEPT FOR EVAL: ExtractBench (PDF stage),
keyphrase, typed_test 400-1612 confirmation slice untouched.

**16:45 — Owner dataset list triaged; ACORD converted into the mix (batch 2 staged).**
Owner handed 12 sources (6 HF + 4 Kaggle) with mandate: use what's useful for training,
KEEP the bench-y ones for eval. Triage (all verdicts evidence-based):
- theatticusproject/acord ACCEPTED, CONVERTED (scripts/adapt_acord.py): 114 attorney
  queries x 3,931 real contract clauses, 130k expert-graded (0-4) query-clause pairs.
  Record shape: 14-clause window (score-4 gold + 3-star HARD decoys + 2/0-star filler),
  field instruction = the attorney's own query text, list<answer> = gold clause spans,
  no-gold windows = honest nulls. -> data/acord_train.jsonl 684 windows (372 answerable/
  312 null) from train+valid qrels; data/acord_test.jsonl 228 windows from official TEST
  qrels = HOLDOUT EVAL (never trained; added to every future battery). Splits keyed on
  QRELS membership, not queries.jsonl metadata (they DISAGREE on some ids - caught in
  build). Contamination audit dropped 380 corpus clauses whose text byte-overlaps
  real_eval/typed_test surfaces (ACORD is CUAD-cousin, "cuad1" query type); BEIR pairing
  check: 0 test-query pairs in train states (160 shared-corpus sightings are legit -
  retrieval corpora are known by design). Encodes 3.8-7.9k tokens, spans byte-exact.
- alexdzm/extraction-examples REJECTED: UK planning PDFs + geojson/metadata pointers;
  no verbatim field-answers over prose text; building records = PDF pipeline (v2 bucket).
- LocalLLaMA/typed-extraction ALREADY INTEGRATED (typed_train/typed_test = Path-A +
  acceptance eval). No new pull.
- llamaindex/ExtractBench KEPT FOR EVAL per owner mandate: rows carry data_schema +
  expected_output + per-field evidence (page+bbox) - real bench - but text lives in PDFs
  (short.jsonl 163MB metadata + docs/). Needs a PDF-text stage; batch-3 eval candidate;
  never trains.
- wikimedia/structured-wikipedia DEFERRED-with-plan: whole-wiki parquet dumps; the
  infoboxes/tables fields are exact table-regime material for the table-text .076 wall -
  batch 3 mining target (stream a shard, keep eval split).
- gonnascore/structuredlogs REJECTED: 5.4GB CSV, log-line->event-TEMPLATE mining - no
  verbatim schema-field gold over prose, wrong task shape.
- Kaggle x4: invoice-images -> vision v2 (PLAN-deferred); legal-citation/multilabel/
  sentiment -> classification, wrong API shape (same rationale as MAUD rejection). Also
  no kaggle creds on this box; not needed given above.
STAGED: /tmp/opencode/exp_acord.sh = expAcord chain (init expReal final ckpt; +acord_train
into mix; battery now includes acord_test NEW + real/typed/contractval/val3/pm-real; and
a PRE-train acord_test baseline eval of expReal so the ACORD effect is attributable).
Launch gated on expReal verdict (~19:40): the decision tree survives either branch -
if expReal transfers, ACORD scales the win; if it doesn't, ACORD's real attorney queries
are the last real-data lever before the LoRA question gets brought back with evidence.
GPU note (owner asked "why cooking on CPU?"): expReal has held the 5090 at 100% since
16:01; ALL list-triage work is CPU/network on purpose (downloads, schema probes, adapters
can never use the GPU) so the next run launches with zero GPU idle. Probes pinned
CUDA_VISIBLE_DEVICES="" to keep VRAM exclusively with the trainer - crash-era discipline.

**15:30 — expReal launched (user-approved: "start with cuad, pull in all the keyphrase and
other training data"). Real-corpus mining done with leak audit — near-miss recorded.**
Data build (scripts/mine_real.py, committed):
- cuad_train.jsonl REBUILT from raw blob docs[0:470] (train region) via public.cuad_records:
  1,797 windows -> **1,714 clean** after dropping 83 windows whose source contracts match the
  46 typed-cuad-test records in typed_test (company-slug+date8 fuzzy match — typed-extraction's
  CUAD split does NOT align with CUAD blob index boundary, so exact-id dedupe was insufficient;
  the journal's "raw CUAD never trains" rule earned its keep — naive cuad_all training would
  have CONTAMINATED the acceptance eval). Verified: 0 windows from the 40 eval titles, 0 from
  typed_test sources.
- cuad_all's 440-title coverage == docs[0:470] exactly (no missed headroom; earlier "30 docs
  missing" was a title-collision artifact).
- keyphrase_train.jsonl: validation.parquet only (test.parquet IS the 20 eval records),
  re-namespaced keyphrase-val-N, 0 content overlap with eval. 20 records — token gesture the
  user asked for; too small to read on its own.
- MAUD: stays rejected (journal audit: 3% verbatim answers = choice-API task, not extraction).
- Mix now: train3 + typed 9.5k (incl 61 cuad-derived) + contract synth 3k + REAL 1,714
  (+20 keyphrase) = 22,287 recs/epoch; real-shape:contract-synth ≈ 1:1.7 (expMix's dilution was
  61:3000). Init expMix-6852 (keeps binding mechanism + med-rank-71 pointing). 7,429 steps x2ep,
  ETA train ~18:20, verdict ~18:50. Battery adds keyphrase readout (n=20, expect noise; don't
  over-read).
User unattended; offered their dataset LIST again in reply — anything with verbatim spans on
real text joins the next batch. Decision tree post-expReal (pre-registered): real transfers ->
scale real corpus further (more windows/spans per doc, relax max_spans 8, harder nulls);
in-domain-only again -> LoRA question with this evidence.

**15:10 — expMix VERDICT: pre-registered branch hit — in-domain contract learning PERFECT,
real CUAD did NOT transfer (regressed). Plus: cuad_all.jsonl discovery (1,797 real records,
never trained on). Escalating to user: real-corpus-mining (recommended) vs LoRA.**

READ (init expFinal-8778 + 3,000 synthetic contract docs, 6,852 steps, clean run, guard
never fired, val3 loss 4.40 best-ever):
- contract_val (in-domain): text **.995**, number 1.000, list-text 1.000, date .983, null_ok
  ~.99 across the board. Header-semantics binding is FULLY LEARNABLE by the current head —
  the mechanism works. (Caveat: same generator train/val; measures capacity, not generalization.)
- real_eval CUAD list-text: .156 -> **.097** — REGRESSION. n=6,068 fields = ~356 fewer exact
  matches: statistically real, not noise. typed list-text .317->.200 same story; null_ok
  improved (.865->.875 CUAD, .957->.986 typed). typed text .163/number .080: held.
- val3 retention: text .902, number .876, date .964, list-number .361 — all fine.
- pm-real: median rank 584 -> **71**, p@5 .011 -> .135 — pointing at the right REGION
  improved 8x (rerank+agg+contract binding helped ranking!) but exact fell: the model binds
  header->evidence correctly yet emits contract-shaped spans (short, header-adjacent) where
  CUAD gold is long verbatim clause text. Shape mismatch, not a binding failure.
INTERPRETATION: 3,000 synthetic docs outvoted the 61 real-shape docs ~50:1 and pulled span
STYLE away from CUAD. Synthetic shapes teach the mechanism; they cannot teach real text.
DISCOVERY (this forensic pass): data/cuad_all.jsonl = CUAD_v1.json (raw, on disk at
data/raw/) converted via public.py:cuad_records() yesterday 16:47 — 1,797 real records from
440 contracts, 41-field CUAD prompts, windowed 16k chars. NEVER entered any training mix
(the 61 typed-cuad records came from a separate smaller earlier conversion; base names
disjoint). Only 3 of 1,797 exact-id-overlap the 148 eval records -> drop those 3, leaving
1,794 CLEAN real-shape records = 29x the 61 that produced the 6x jump.
DECISION (user, per pre-registration): Option B real-corpus-mining is now nearly free —
expReal = expMix stack + cuad_train(1,794) mixed with contract synth (real:contract
~1:1.6 by count; ~+30% tokens/epoch, est 2.6-3.0h). Option A LoRA (rung 4) should stay
evidence-gated BEHIND that: the CUAD wall is text-shape, and we've never once trained on
real shape at scale. If expReal's transfer is strong -> LoRA may be unnecessary; if the
real-shape curve flattens -> LoRA with evidence. expMix-6852 is the best init (binding
mechanism intact, retention best-ever); expSpan-5852 (.156) is the CUAD fallback init.
Staged: data/cuad_train.jsonl (1,794, eval-duplicates removed) + /tmp/opencode/exp_real.sh
— NOT launched pending user's phase decision.

**12:55 — expSpan verdict: span-rerank hypothesis FAILS on its own metric; the collateral
fixes scored a real win. expMix (nospan, contract infusion) launched 12:53.**
Controls expFinal. READ: squad2 text exact .169->.161, p@1 .255->.260 (noise);
drop text p@1 .263->.250. The start-only correction never moved top-1 selection ->
Tier-B-for-text: museum shelf, with evidence. Mechanism note (why it likely failed):
text tokens don't carry the value-features/identity signal that made candidate
reranking work — [q;c;q*c;|q-c|] over (query, token) pairs may just be less
informative than over (query, atom) pairs; recorded so a future attempt starts from
here, not from hope.
THE REAL HEADLINES (attribution: NOT the span net — list decode path never touched):
(1) CUAD list-text exact .026 -> .156 (6.0x), null_ok .847 -> .865, typed list-text
.100 -> .317: first-ever training inclusion of the 61 typed-cuad 41-field records
(max_fields 32->48, journal 07:15 suspicion CONFIRMED at effect size — the CUAD
regime was COLLATOR-starved, not just shape-starved).
(2) val2 date .865 -> .964: slash-date regex fix behaved exactly as predicted.
(3) val2 number .847 -> .887, list-text 1.000, duration 1.000 — retention solid with
grad-accum; the 11:13->12:46 run was crash-free to step 5852 (guard armed, never fired).
expMix nospan = rerank+attn+agg+fix5+grad-accum-2+guard-18, init expFinal-8778,
+contract_train(3000): THE CUAD experiment proper (contract docs teach header-
semantics binding at scale; the 61-record inclusion effect showed regime presence
matters 6x from ZERO contract data — with 3000, what's the ceiling?). Battery adds
contract_val (in-domain) + pm-real. ETA: train ~15:00 (6852 steps @ ~64/min), verdict
~15:45. Pre-registered reads: (a) contract_val text exact — does the model learn
header binding in-domain at all; (b) real_eval CUAD .156 -> ?; (c) typed/cuad subset.
If contract_val >> CUAD: binding learned but doesn't transfer -> real CUAD text
granularity/vocab gap -> next lever is LoRA or real-corpus mining, and the morning's
field-groups/inside-clause pair said which. If both jump: scale-the-shape wins,
plan more contract varieties (multi-paragraph clauses, more null schemas).
Note: expSpan-5852 kept as best CUAD checkpoint so far (runs/expSpan/checkpoint-5852)
— if expMix somehow regresses CUAD, fallback init is available.

**12:02 — ATTEMPT 6 CONFIRMED THE FIX: start-only span rerank sails through attempt-5's
blowup window. val3 span: 0.38/0.47/0.48/0.42/0.42 at steps 500..2500 — attempt 5 read
0.40/0.64/2.55/3.38/9.63 at the same checkpoints and expFinal sits at 0.38-0.41. Diverge
guard silent (0 hits). cand bucket 0.96->0.76 descending normally. Run at step ~2920
epoch 0, 8.7 rec/s (accum+guard cost nothing), ETA training ~12:50, battery verdict
~13:10, then expMix per rule (span ON unless attempt-6 eval contradicts). The span
hypothesis is now alive on its own merits: if the battery shows squad2 p@1 .26 -> ~.35+
with span held, Tier-B-for-text graduates; if not, museum shelf with evidence, and
expMix runs nospan.
(journal hygiene note: the 11:15 entry got double-prepended by a reused entry file —
deduped here; lesson: write a fresh entry file per journal append.)

**11:15 — attempt 5 SPAN DIVERGENCE: caught by val curves, root-caused, fixed, attempt 6 launched.**
Symptom: val3 healthy through step 1000 (span 0.40 -> 0.64), then monotone blowup: 2.55/3.38/
9.63/12.9/17.7 -> train span bucket exploding (18 -> 94 at step 4940). Differential with attempt 4
(same span_rerank, old loop): identical healthy start -> the ACCUM LOOP IS INNOCENT; the
span_rerank DESIGN diverged after ~1200 steps (attempt 4 died at 620 before reaching the window).
Root cause: one shared pair net corrected BOTH start and end logits -> start CE wants first
token high, end CE wants last token high, same (query,token) pairs -> contradictory gradients
through one function; net oscillates, grows, buries gold. My design error, not the environment.
Fix 1: START-ONLY correction (head.py; decode already conditions end on start; the measured
selection gap is start-ranking). CPU-verified: identity at init exact; with correction active
start moves 2.6, end changes only by 2e-7 = fp32 ulp (untouched). Fix 2: --diverge-stop N guard
(trainer aborts + saves on N-exceeding losses twice consecutively; exp runs use 18) so future
design blowups cost minutes, not 80 GPU-min — this was the second time (after the op-head bug
era) that a sick run cooked visibly-in-loss terms for an hour; the guard makes the loop safer.
Process cleanup: killed diverged trainer + chain + orphaned eval (was about to score
checkpoint-5000 garbage; no bogus reports written). Attempt-5 run dir kept as
runs/expSpan-a5-diverged.
Attempt 6 launched 11:13 (start-only + accum 2 + diverge-guard 18). The truth point is
val@1500-2500 (attempt 5's blowup window): expect span < 1 all through. Training done ~12:50,
battery -> verdict ~13:10; expMix follows (~16:15 CUAD readout). If attempt 6 also diverges in
the window despite start-only: span hypothesis is FALSE (not a bug) -> drop --span-rerank
entirely, expMix runs in nospan mode, reranker-for-spans goes on the museum shelf with evidence.

**09:52 — attempt 4 died at step 620 (09:46:12): expandable_segments mapping OOM — physical VRAM
genuinely exhausted by a bs6 8k-token spike. Machine SURVIVED (solo-tenant fix worked as
designed: process dies now, not the box).** Diagnosis: at bs6 the trainer needs more than the
card's ceiling on worst-case batches regardless of desktop. Environment levers exhausted ->
structural fix implemented: --grad-accum (record-count-weighted micro-batch accumulation;
micro 3 x2 == effective bs6 mathematically; controls keep steps_per_epoch=2926 and LR schedule
identical; val loader stays bs6 (no_grad, low peak); accum=1 reproduces the old path exactly).
GPU smoke: steps tick, 4 missing tensors on init-head = span_pair only ✓, checkpoint-4 saved,
schedule math verified. Attempt 5 launched 09:52 with --grad-accum 2. Expect peak VRAM ~4-5GB
lower; if a process still dies, that would surprise the math — next lever is micro 3x1
(bs3 effective, documented deviation) rather than more knobs. Also note: attempt-4 train.jsonl
preserved in runs/expSpan-crashed/ as train-attempt4.jsonl.

**09:35 — crash autopsies COMPLETE + fix landed (user moved display to iGPU).**
Reconstruction from persistent journals + last(1) boot table:
- 07:32 & 08:40: TRAINING PROCESS died (allocator OOM, classic mode: fragmentation,
  "free 273MB, wanted 560MB"). Machine survived both — Friday-night boot only ended
  as 'crash' at 08:28, so at 07:32 the box was up; I mis-read it as a machine crash.
- 08:50: FULL MACHINE CRASH, smoking gun in boot -7 journal: kwin_wayland
  "Failed to allocate egl gbm swapchain buffer" + nvidia_drm *ERROR* "Failed to
  allocate NVKMS memory" x20 at the exact death second. Trainer (with my
  expandable_segments setting reserving 32074/33577 MiB) STARVED the display
  compositor -> driver cascade -> hard reset. Guilty knob was mine; the knob was
  meant to fix the 08:40 fragmentation death and did its job for memory — but the
  desktop was quietly living on the same 32GB. My 08:45 note "desktop ate the
  margin" was directionally right, mechanism wrong.
- 08:52-09:26: boot loop (6 boots, some <60s) = desktop-on-nvidia failures during
  recovery attempts; not our workload (nothing running then).
GPU HEALTH: no evidence of damage — idle 38-46C, no pending repairs, ECC N/A
(GeForce has no counters; absence is not a red flag), power throttle counters
idle. It was suffocated, not burned.
FIX: display now on iGPU (user decision, 09:2x). 5090 = 56MiB baseline, training
sole tenant. expandable_segments NOW KEPT (safe + prevents fragmentation); bs6
restored to preserve control comparability. If a batch spike ever again threatens
the card, the documented fallback is --grad-accum 2 (micro-batch 3) — designed
today, not needed now.
Protocol for user gaming on the 5090: ping me (or Ctrl-C the chain) — runs checkpoint
every 1000 steps; stopping is lossless to ~20 min.
expSpan attempt #4 launching 09:35; expMix armed next per rule.

**08:45 — second failure caught + diagnosed: genuine CUDA OOM at step ~250 of the restart.**
Not my code: failing alloc (~560MB) was the same size since step 20; the post-reboot
desktop stack (kwin_wayland 254MB GPU-resident + plasmashell + firefox) ate the margin,
PyTorch caching allocator fragmented ([W] retries show free<560MB), and a long-doc batch
finished it. Trainer steady ~29.8GB on a 32GB card = zero-sum game with the GUI.
MITIGATION: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exported in BOTH chain
scripts (stock torch knob, zero effect on training math, kills fragmentation failures).
Fallback if it dies again: ask user to close Firefox (reclaims ~1GB system/GPU pressure)
or drop bs 6->5 ONLY as explicit, documented deviation. Relaunched 08:45, harness shell
sh_16be56beb... will notify; partial train.jsonl of the dead attempt cleared, crashed
run dir preserved. Note for all future evals/runs today: the env var travels with the
scripts; any manually-launched GPU command must set it too.

**08:40 — machine crash 07:32 + recovery.** expSpan died at step 1540/5852 (mid epoch 0);
checkpoint-1000 verified loads-strict; /tmp/opencode wiped (scripts + drop_test regenerated).
RESTART decision: fresh restart over resume -- because max-steps does NOT retune the LR
schedule (ends unannealed at ~1e-5: confound vs the fully annealed control chain), and a
resume-to-cap saves only ~20 min vs a clean restart with identical exposure to controls.
Old dir preserved as runs/expSpan-crashed (1.4GB, forensics; deletable anytime).
Everything committed through 06b7cb2 survived (working tree was clean at crash time);
all data files on home disk intact. Queue unchanged: expSpan battery ~10:55 verdict ->
expMix (contract infusion) per rule -> CUAD readout ~13:45. Watch: tail -f
runs/expSpan/train.jsonl, dashboard scripts/watch_training.sh (auto-discovers).

**07:15 — Contract domain built (CUAD-shaped synth) + slash-date engine fix.**
Diagnosis chain (journal 07:00): CUAD failure = header-semantics binding the training data
never contains (every existing domain puts answers adjacent to cues). New `contract` domain:
5-8k char docs, 20-31 schema fields, ~38% null, paragraph-grained text gold below matching
numbered headers, NEAR-MISS decoy sections ("Termination for Cause" vs requested
"termination_for_convenience"), money/date/duration atoms per clause, list<text> addresses.
Collator max_fields 32->48 (CUAD's 41-field records were silently skipped in training:
only 61/9553 typed_train records — the CUAD slice itself was training-starved at the
COLLATOR level, not just the shape level). Date engine: 2-digit slash dates (mm/dd/yy,
1 style in 5) matched no DATE_RES regex -> date solvability ceiling was ~80% THIS whole
time; fixed with (?!\d) guard against phantom prefix matches. Contract reachability:
text/lists/number/duration 100%, date 98.2% (rest genuinely ambiguous).
NOTE for interpreting expSpan battery: training (06:50) imported pre-slash-fix pools;
its evals (post-fix) will see ~20% new date atoms -> expect small date-exact wobble,
not regression. Next run pre-registered (fires when expSpan battery done ~08:50):
expMix = +contract_train(3000) into the mix, 2ep, lr 1e-4, --rerank --query-pool attn
--agg-atoms; --span-rerank INCLUDED iff expSpan improves typed text exact OR squad2/drop
pm p@1 over expFinal (controls: expFinal-* reports); init = expSpan-best if span won,
else expFinal-8778. CUAD list-text exact is the headline readout (was .023/.026).

**07:00 — CUAD diagnostic battery: the wall is QUERY SEMANTICS, not competition, not granularity.**
(1) --field-groups 5/8/16 (serving-side passes over same state): p@5 stays .011-.022, median
~600-750 -> 41-query competition REFUTED. null_ok DROPS .82->.42 when schema shrinks (null head
uses full-schema context — do NOT serve CUAD grouped, free fact). (2) inside-clause membership:
top-1 lands inside gold clause 2.1%, top-10 22.9%, median distance 814 tokens (~one document
away) -> NOT a clause-granularity artifact either. The pointer genuinely cannot bind legal
field names to clause text; frozen-base vocabulary/routing limit is the working theory.
Baseline(41) on same 24 docs: exact .003 — 168-doc .023 was mostly keyphrase.
IMPLICATION for the ladder: more epochs/more same-shape data/deeper decoder won't fix
"field-name -> clause" binding. The two live levers: LoRA (adapts representations for legal
binding — rung 4, now evidence-triggered) and shape-matched training data (long docs, big
schemas, clause-grained spans — the data-shape answer to user's 'more data' question).
Sequence: (a) span-reranker run (helps squad2/drop/table-text selection gaps — orthogonal to
CUAD, already built+tested), (b) shape-matched synth generator (CPU), (c) LoRA probe on the
next-best checkpoint if (a)+(b) plateau. GPU slot now open for (a).

**06:45 — expFinal2 (+1ep extension, resume): synthetic keeps improving, real flat-to-up.
NIGHT LOOP CLOSED, GPU idle.** typed text .169, number .085, list-text .136 (null_ok .871);
val2 number **.859** (was .847), list-number .311 (partial recovery vs .295); CUAD exact .023
(null_ok .837). Post-stack CUAD postmortem: see expFinal2-pm-real.json committed alongside.
Trajectory summary for morning review: main4->expFinal2 = typed text 13.3->16.9 (+27%),
number 4.6->8.5 (+85%), CUAD-null 0->.84, table number 2.5->79+. Best checkpoint:
runs/expFinal/checkpoint-8778. OPEN for tomorrow: (1) list-number duplicate-atom confusion
from agg pool (reranker scores min/max dupes of raw atoms; slot-context rerank design),
(2) table TEXT .076 pointing (token-span reranker not built — pair nets only cover
candidates), (3) CUAD p@1 post-stack (read pm-real), (4) the 80% bar on solvable subsets
is NOT met on any real domain; decision needed: LoRA phase (rung 4) vs more data tiers.

**05:45 — expFinal (full stack: rerank+attn+scatter+agg-atoms+fix5-loss) = best model; the
synthetic number wall CRUMBLES. Decode IndexError fixed (padded-bin argmax, eval-crash
forensics). +1ep extension running (resume from 5852, ~06:45 done).**
typed (400): text .133->.157->.**.169**, number .046->.068->.**.085**, list-text null_ok .914.
real CUAD: exact .0109->.021->**.026**, null_ok ->**.847** (fix5 loss-half added .12 over
decode-only). pm: drop number p@1 .189->**.248** (reranker finally moves top-1 once pool
grew richer — agg atoms + fix5), squad2 p@1 .197->**.255**, p@5 .609.
val2 numbers (with --agg-atoms): number exact .507->**.847** — TABLE domain .025->**.791**
(solvable now 1.0), invoice number .839. list-number .387->.295 (mild regression, watch).
Estimate APE .30 / forecast APE .11 (coverage ~.89/.79) — distribution heads healthy-ish.
Remaining walls (honest, for the morning): table TEXT pointing .076 (region-name cells:
solvable but unranked — reranker only covers candidates, not token spans!); squad2 .169;
CUAD .026 (41-field query problem persists; p@5 there was .034 pre-stack — unmeasured post).
Cumulative overnight: main4->expFinal text +27%, number +85%, CUAD-null 0->.85, table-number
+30x. Extension battery reports expFinal2-* land ~07:00.

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
