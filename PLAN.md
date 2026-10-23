# Plan — decision protocol for clarke-flash

Updated 17:55 after the op-head bug fix and the owner's operating rules:

> **80%+ bar, sensible steps, don't fight unbreakable walls, document
> everything, keep going until we hit one.**

## Definitions

- **The bar (acceptance):** on the real-world evals (`typed_test.jsonl`,
  `real_eval.jsonl`), headline metric = **exact match over the SOLVABLE
  subset ≥ 80%** per family group, plus sanity: null accuracy ≥ 90%,
  interval coverage within ±10pt of nominal, `origin: read` answers
  byte-verifiable at their spans (by construction — must stay 100%).
  Raw overall numbers are reported alongside; solvable-only is the
  judging number because the denominator is honest.
- **A sensible step:** one change, one hypothesis, one committed eval
  report measuring it. No step may require hardware we don't have
  (1× RTX 5090 32GB, frozen bf16 9B forward ≈ 19GB + head).
- **A wall:** a gap that survives two consecutive steps aimed at it,
  where the mechanism is known to exceed the current tool (see table).
  Walls get documented, not attacked.

## Known/likely walls for this toolset (do not grind on these)

| Wall | Why it's structural | Accept & document |
|---|---|---|
| multi-row aggregates (`total_revenue`, `average_revenue`) | >2-operand math; blog's own Clairic punts: "let code do the math" | yes — `origin: computed` unsupervised |
| spelled-out amounts ("one hundred"), locale numerals, "1M", "%" semantics | parser coverage, not model capacity | extend `candidates.py` parsers if cheap; else out of scope |
| >8 list items (LIST_SLOTS=8) | head capacity choice | note; raise slots only if eval shows it matters |
| evidence outside the 8k window in giant docs | context budget | v2: chunk-and-merge serving strategy |
| image states (receipts, charts) | v2 by plan | ExtractBenchPro waits |
| 9B reading depth on adversarial legalese | model scale | LoRA phase is the last escalation, not scale-up |

## The ladder (each rung unlocks on eval evidence, not vibes)

1. **fix4** (running, ETA ~19:15): op-head supervision fix (single-atom
   answers left op untrained → duration 0%). Heal 2 epochs from
   main3/checkpoint-6670, then triple eval: val2 regression, real_eval,
   typed_test. Gate to next rung: val2 duration recovered (≥60%) &
   number-when-solvable climbing.
2. **main4 — data mix (Path A)**: `train2 + typed_train` (~17.5k records,
   resume fix4-final, 3–5 epochs). Hypothesis: real surface forms close
   most of the typed_test gap. Gate: typed_test solvable ≥ 80% → done
   with training, go to polish.
3. **targeted fixes** if <80%: per-family report pins the weak subset;
   each gets ONE mechanism step (date-format parsing, candidate context
   window tuning, alts-accepted scoring, LIST_SLOTS). Two failed steps
   on the same subset → declare that family's wall, move on.
4. **LoRA phase (final escalation)**: rank-256 adapters trained jointly
   with the head — Cloudflare's own validated recipe for exactly this
   backbone class. Gradient checkpointing, bs 1–2, ~seq 4k. If LoRA
   plateau still <80%: WALL — write it up with evidence.
5. **Polish (post-bar)**: calibration pass (temperature/conformal on val),
   `clarke-serve` (clef-server shape), model card + weight push
   (`hiemanshu/clarke-flash`, Apache-2.0, dataset credits). Then v2: vision.

## Standing conventions

- Commit before every GPU run; message carries the **why**; push is the
  owner's act.
- `reports/main<N>-<dataset>.json` are committed evidence, one per
  (checkpoint, dataset).
- **Dedupe rule:** `cuad_all.jsonl` never enters training while CUAD
  derivatives sit in the eval slices; contract training flows only
  through `typed_train.jsonl`.
- Eval hygiene to implement after fix4 (don't confuse a measuring wall
  with a modeling wall): per-family breakdown, accept typed-extraction
  `alts`, split metrics by gold `source` (span/computed/relative).
- Everything that happens is journaled in `JOURNAL.md`.
