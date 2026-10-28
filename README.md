# clarke-flash

A System One **extraction model**, in the spirit of
[Cloudflare's Clairic](https://blog.cloudflare.com/clairic-extraction-model/):
given a `state` (document text, tables, series — images in v2) and a schema of
typed fields, it answers **every field in one parallel forward pass**.

No tokens are generated. Answers are:

- **read** — spans copied from the input, with character offsets (a `text`
  answer literally *is* `state[start:end]`; it cannot be re-spelled or invented),
- **computed** — totals, deltas, aggregates re-derived from the input by the
  head (1–2 step arithmetic),
- **predicted** — estimates/forecasts as distributions inside your `[min, max]`
  on your `step` grid (median, p10/p90). Implemented and exercised on synthetic
  data; not the current research focus.

Every answer carries `confidence` and `p_null`; a null answer still reports its
`guess` + `span`, so near-misses are auditable. `origin` (`read`/`computed`/
`predicted`) tags how the value was obtained — `read` answers are exactly
verifiable against the source text.

The architecture is a **frozen Qwen3.5 backbone** plus a small (~112M parameter)
**joint extraction head**. The LM head is never used. Field names and
instructions are *input text*, so new fields need no retraining — the model
learns to read the schema it is handed.

**Status:** research project, pre-release. Weights and a serving API are not
published yet; every number below is reproducible from this repo.

## Why "Clarke"

**Clarke is the word *clerk* wearing a surname hat.** Cloudflare built Clairic out of
*clear* + *clerical* + *clairvoyant*; this name does the same trick with one word:

1. **clerk** — the job itself: clerical extraction from documents — invoices, tickets,
   contracts — with character offsets and a receipt for every answer. (*Clark/Clarke* is
   historically the occupational surname of exactly that clerk.)
2. **Clark Kent** — the persona is the architecture. It looks like a language model
   (same Qwen backbone, same tokenizer, same schema prompt grammar) but secretly
   isn't one: one parallel pass, zero generated tokens. A mild-mannered reporter by
   day; the thing in the room that cannot lie about what it saw.
3. **Arthur C. Clarke** — third law reserve: a 0.5-second typed answer replacing a
   60-second reasoning chain is indistinguishable from magic; and of Clarke's two
   famous AIs, this one is the honest witness, not the chatty HAL.

Not related to Claude (the Anthropic model): named for a job, not a person —
Qwen3.5 backbone, Apache-2.0, and the only coincidence is the first two letters.
The epistemology in one word, from the README's first line: it answers only what is
*pratyakṣa* — प्रत्यक्ष, "what stands before the eyes."

## Why not just fine-tune an LLM to emit JSON?

Because generation is the failure mode: mistyped digits, unparseable output,
untraceable answers, and per-field output-token cost. This model mirrors
`Cloudflare/clef-flash`'s published architecture (same backbone family): a
frozen Qwen3.5 backbone reads `STATE + SCHEMA FIELDS` in a single pass and a
small **joint extraction head** — evidence-routing cross-attention + field
decoder — reads the final hidden states. Field names and instructions are
*input text*, so new fields need no retraining; the model learns to read them
(see surface/instruction randomization in `synth.py`).

## Reading the numbers

Every result in this repo — ours and every external baseline — is reported with
the same six numbers, because any one of them alone lies:

| metric | what it measures | how it misleads alone |
|---|---|---|
| `exact` | per-answer precision on **attempted** answers only | flatters a conservative extractor; false nulls are uncounted |
| `tokF1` | token-level F1 against the gold span | credits the right clause quoted with different trimming |
| `contains` | did the answer land on the right text at all (substring, either direction) | **recall only** — says nothing about spurious extra items |
| `null_ok` | share of truly-absent fields left absent | our differentiator, so hiding it would be self-serving |
| `answered` / `gold` | recall at the field level | the binding constraint at document level |
| `exact@all` | `exact × answered/gold` | the only number combining precision and recall |

`contains` being recall-only is why the `parties` work below reports a
**precision** column alongside it: substring matching forgives
`"YOU ON DEMAND HOLDINGS"` vs `"YOU ON DEMAND HOLDINGS, INC."`, so it will happily
reward a model that emits every name it finds.

## Results so far (2026-10-28)

Primary surface: **`data/doc_eval_40.jsonl`** — 40 held-out CUAD contracts
(`docs[470:510]`), 41-field schema, 498 gold-present fields, all text, no
windows, decoded at 65,536 tokens.

| model | exact | tokF1 | contains | null_ok | answered | exact@all |
|---|---|---|---|---|---|---|
| **clarke 2B** (H4 + candidate parties, **baked**) | .4905 | .6231 | .6826 | **.9536** | 412/498 | .4058 |
| clarke 2B (H4 + candidate parties, ranker α=2 +1) | **.4952** | .6157 | .6785 | .9536 | 412/498 | **.4097** |
| gemini-4-argon | .5608 | .7397 | **.9025** | .9335 | 369/498 | .4156 |
| gpt-6.1-sol | .5071 | .6922 | .8356 | .8835 | 399/498 | .4063 |
| opus-5.5 | .5154 | .7008 | .8329 | .9203 | 364/498 | .3768 |
| clairic | .0176 | .4063 | .9526 | .1699 | **496/498** | .0175 |

Read it honestly: we are **at frontier `exact@all`** (.41 vs .40–.42) while being
**far behind on `contains`** (.68 vs .90) and `tokF1`. The frontier models find the
right clause more often and quote it more cleanly; we decline fewer wrong answers
(`null_ok` .954) and pay for it in coverage. Clairic never declines, which is why
its `contains` is highest and its `exact` is lowest.

External numbers come from `scripts/external_baseline.py` through one gateway, on
the same 40 documents (`reports/external_doc/`). Their `grounded_rate` — how often
the quoted text is actually present in the state — is .79 (gemini), .82 (gpt),
.82 (clairic), .86 (opus): **even the best of them cannot locate a fifth of what it
quotes.** A `read` answer from this model is grounded by construction, which is the
whole argument for a non-generative readout.

**Smaller schema, sharper signal.** Restricting to the 12 most common CUAD fields
(60.8% of gold instances) makes each field matter more per step:

| model (12-field eval, 310 gold) | exact | tokF1 | contains | null_ok | answered | exact@all |
|---|---|---|---|---|---|---|
| 41-field H4, scored on the 12 fields | **.6067** | .7213 | .7940 | **.8765** | 281/310 | .5499 |
| 12-field I-twelve | .5941 | .7261 | .7941 | .8118 | 293/310 | **.5615** |

So a smaller schema is **not** a capability win — it is a cheaper, lower-variance
experiment loop. That matters, because each 12-field run is ~12 minutes.

**`parties` — the one field that moved furthest** (same 40 documents):

| step | contains | precision | doc-exact |
|---|---|---|---|
| slot-mask decode | .318 | .396 | 0–1/40 |
| + candidate-constrained decode | .453 | .481 | 1/40 |
| + legal-suffix boundary fix | .615 | .652 | 6/40 |
| + learned ranker (α=2) | .881 | .890 | 19/40 |
| + one extra candidate | **.911** | .861 | **20/40** |

The insight worth sharing: `parties` was never a retrieval problem. The model puts
gold party starts in some slot's top-5 82% of the time and crops them exactly 86% of
the time — it **under-emits**. 99.5% of gold party items are reachable by cheap
patterns, so the job is ranking ~221 candidates down to the 4 that are parties, which
a 14-feature logistic regression over the 470 CUAD *train* documents does better than
the span pointer did.

## Levers measured and ruled out

Documented so nobody (including us) re-runs them expecting a different answer:

| lever | result |
|---|---|
| **backbone size** | 0.23B → 9B flat above ~1B; 4B is a tie at 65k (−.0003 tuned, +.0036 balanced); 9B no better at 8k |
| **backbone family** | 8 families at 65k (Qwen, MiniCPM5, Ministral, Granite, LFM, Hy-MT2, Gemma, …) — Qwen 2B wins or ties every arm |
| **head capacity** | 112M → 371M: no gain, worse on MiniCPM5 (−.019) |
| **decode: span cap** | p99 per-field caps are a dud (+0.45 pp); the cap almost never binds |
| **decode: pointer temperature, slot/null gates** | temperature does nothing; the gate curve is exhausted and the 92–93% coverage @ .92 `null_ok` operating point is **not on the curve** |
| **calibration (Brier on `p_null`)** | neutral on `exact@all` — `p_null` is *under-informative*, not miscalibrated |
| **loss shapes** (field-balanced, slot-mask training, Brier) | all within ±.007 — neutral |
| **data: sparse-clause corpora** | LEDGAR/SEC window mix (D2) lost on every axis by over-nulling (present rate 71% vs the 26% the eval wants); a dense rebuild (F) recovered all of it without beating the sparse baseline |

**What did work**, all one-variable steps: training at 65k documents (+.098
`exact@all` vs the windowed regime), span tie-break ε=.25, a per-field **length
prior** (α=.75 — boundary-exact 59%→71%, excess characters +23,760→+1,042; it
subsumes the cap), slot-mask decode, and candidate decode for name-like fields.

## Heads per type

| schema type | answer space |
|---|---|
| `text` | start/end pointer over state tokens (+null) |
| `list<text>` | 8 slot pointers (presence-gated, optionally order-free) |
| `number`, `duration` | sign × exponent × top-3 × next-3 digits |
| `date`/`time`/`datetime` | calendar component heads (always-valid output) |
| `estimate`, `forecast` | distribution over declared grid + below/above |

Losses: span CE, digit/calendar CE, grid CE, null BCE. `estimate` labels are
realized draws from the generator's latent process, so **calibration is
learned, not post-hoc**: the noise the generator injects sets the interval
widths.

Numbers and dates are never computed by the head directly. They are **picked**
from regex-extracted candidate atoms (with ±8-token context pooling and value
features) and optionally combined by a 5-way op head `{id, +, -, *, /}`; decode
renders the exact source text and records `origin: read|computed|predicted`.

## Training approach

- **Phase 1 (current):** backbone frozen (`no_grad` forward, bf16), head only, at
  **65,536 tokens** — full documents, not windows. This is the single largest
  measured win in the project.
- **Phase 2 (blocked):** LoRA adapters, Cloudflare's recipe. On one 32 GB card
  LoRA fits **~4k tokens** and OOMs at 8k+, independent of rank: the 18
  linear-attention blocks in the hybrid backbone don't release their activations
  under gradient checkpointing, so memory stays ~30 GB at 16k *and* at 32k — it
  does not scale with the sequence, which is what proves checkpointing is a no-op
  there. The workaround — adapt on 4k snippets, then re-fit the head on full
  documents — was tried and **degraded the backbone catastrophically**
  (`null_ok` .0000, exact .0801): the adapter taught the head that ~89% of fields
  are absent. Unlocks with: a 64–96 GB card, or a manual per-block checkpoint
  implementation for the linear-attention blocks.
- **Data:** synthetic-by-construction labels (spans, arithmetic, distributions are
  known when the document is assembled — no annotation campaign), CUAD for the
  contract domain, and mined/derived corpora for rare fields.

## Layout

```
src/clarke/
  schema.py       typed field specs + answer grids (bins, snapping, percentiles)
  encoding.py     state+schema prompt, token spans, char offset maps (clef grammar)
  head.py         JointExtractionHead (routing + decoder + type readouts + loss)
  model.py        ClarkeModel, decode (prior/slot-mask/candidate/ranker), save/load
  candidates.py   atom extraction, field_atoms, solve, apply_op, CSV aggregate atoms
  synth.py        synthetic generator: support/invoice/ticket/table/comps/forecast
  public.py       CUAD / typed-extraction / keyphrase adapters -> same JSONL
  textmetrics.py  exact / tokF1 / contains / precision — shared with the baselines
  data.py         RecordDataset + worker-side collate
  train.py        frozen-backbone head trainer (runs/, jsonl logs, checkpoints)
  eval.py         accuracy + null + calibration + postmortem report

scripts/
  clean_edgar_corpus.py   EDGAR -> one clean text column (1.07M docs, 19GB)
  build_ledgar_spans.py   LEDGAR provisions -> verbatim spans in SEC contracts
  build_sec_corpus.py     packed 65k-token LM corpus
  external_baseline.py    Clairic + chat models through one gateway, same metrics
  train_party_ranker.py   the 14-feature name ranker
  finalize_checkpoint.py  bake decode policy into a checkpoint (shareable folder)
  doc_diagnose.py         failure taxonomy + per-field dump
  watch_training.sh       live dashboard (ncurses TUI; CPU/RAM/disk/net + losses)
```

## Quickstart

```bash
uv sync                                  # env (torch cu128, transformers 5.x)

# --- synthetic smoke test: no downloads, exercises every head type ---
uv run clarke-gen --out data/train.jsonl --count 6000 --seed 1
uv run clarke-gen --out data/val.jsonl   --count 300  --seed 99
uv run clarke-train --train data/train.jsonl --val data/val.jsonl     --max-steps 20
uv run clarke-eval  --head runs/<dir>/checkpoint-<n> --data data/val.jsonl

# --- the real regime: full contracts at 65k, the config behind every number above ---
uv run clarke-train --train data/doc_train_470.jsonl \
  --backbone Qwen/Qwen3.5-2B --max-length 65536 --epochs 4 \
  --batch-size 2 --grad-accum 2
scripts/watch_training.sh                # dashboard; `status` for one-shot

uv run clarke-eval --head runs/<dir>/checkpoint-1880 \
  --data data/doc_eval_40.jsonl --max-length 65536 --batch-size 1
```

`clarke-eval` defaults to `--max-length 8192`. At document level you **must** pass
`--max-length 65536`, or you are silently measuring a different regime.

To produce a folder that behaves correctly when someone else runs it, bake the
decode policy (thresholds, tie-break ε, the gold-length prior, the party ranker)
into the checkpoint — otherwise a shared `extraction_head.safetensors` reproduces
the weaker default:

```bash
uv run python scripts/finalize_checkpoint.py --head runs/H4-assign/checkpoint-1880 \
  --hist reports/field_span_hist_train470.json --ranker reports/party_ranker.json \
  --eps 0.25 --prior-alpha 0.75 --candidate-fields parties --ranker-alpha 1.0
```

## Data

- **CUAD** (`theatticusproject/cuad`, CC-BY-4.0) — 510 real commercial contracts,
  41 human-annotated clause categories, 13.8k character-exact evidence spans and
  ~68% `is_impossible` nulls. `docs[:470]` train, `docs[470:510]` eval — disjoint
  by construction:

  ```bash
  mkdir -p data/raw && curl -sSL -o data/raw/CUAD_v1.json \
    'https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json'
  uv run clarke-public --out data/doc_eval_40.jsonl  --doc-start 470
  uv run clarke-public --out data/doc_train_470.jsonl --doc-end 470
  ```

- **LEDGAR → SEC spans** (`data/ledgar_split_cuad.jsonl`, 19,046 records, 31,091
  spans) — derived entirely from public data we already had: LexGLUE-LEDGAR's 60k
  clause excerpts carry no document pointer, but they are verbatim excerpts from
  EDGAR filings, so `scripts/build_ledgar_spans.py` locates each one inside the SEC
  contract text (Aho–Corasick over normalized text + a position map) and recovers
  both the label and the exact character span. Records are windowed at 240k chars so
  no span ever falls outside its window — a truncated span becomes a silently
  false-null label, which cost us a whole experiment once.
- **EDGAR clean corpus** (`data/edgar_clean/`, 1,068,049 documents, 19 GB, 5.4%
  duplicates flagged) — raw filing HTML cleaned once, with `hash`,
  `printable_ratio` and `has_contract_marker` columns, so every builder and miner
  reads plain text instead of re-deriving it.
- **LocalLLaMA/typed-extraction** (CC-BY-SA-4.0) — 11k records already in our
  schema, used for train and transfer tests. **Keyphrase**
  (`jinya425/Keyphrase_extraction`, MIT) — out-of-genre smoke eval.
- Not wired in: **MAUD** (answers are comprehension choices, not spans — that is
  clef's `choice` API), ag_news, the twitter/suicide-keyword sets.
- Everything under `data/` and `runs/` is gitignored and regenerates from the
  seeded generator plus the commands above. `reports/` **is** tracked, so every
  number in `JOURNAL.md` traces to the JSON that produced it.

## Roadmap

- **12-field subset** as the default experiment loop (~12 min per run), then add
  fields back as supervision for them lands. Only 9 of the top 25 CUAD fields have
  LEDGAR supervision today, which is why mined/LLM-labelled rare-field data is the
  open thread.
- **Mining rare fields** from the cleaned EDGAR corpus, with free verification
  layers (plausibility: a mined `expiration_date` must contain a date, a
  `revenue_profit_sharing` must contain a percentage) before any paid check.
- **Party ranker as a learned component** rather than a logistic regression, and
  the same treatment for the other `list<text>` fields — under-emission
  (1.03 predicted vs 1.84 gold items per list field) is the single biggest
  remaining structural failure, and it is a set-completeness problem, not a
  retrieval one.
- **LoRA**, once a 64–96 GB card or a working per-block checkpoint exists.
- **Publish**: model card, weights, `clarke-serve` (`/v1/extraction`, mirroring
  `clef-server`'s shape), decode policy baked so the release behaves like the
  paper number.
- **v2**: image states (the backbone ships a vision tower; render synthetic
  receipts with the same construction-time ground truth, then run ExtractBenchPro —
  real chart/diagram documents with quote+bbox evidence).

## Project docs

- **`JOURNAL.md`** — the lab notebook, newest-first: every verdict, deviation from
  a pre-registered rule, and negative result, with numbers, run dir and report path.
- **`OPERATIONS.md`** — the working rules that survive a context reset: reporting
  contract, the 65k context regime, GPU/shell discipline.
- **`PLAN.md`** — acceptance bar, known walls, escalation ladder.
- **`docs/data-landscape.md`** — corpora surveyed, and why each was taken or dropped.
