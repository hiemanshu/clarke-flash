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

## Results so far (2026-10-29)

Primary surface: **`data/doc_eval_40.jsonl`** — 40 held-out CUAD contracts
(`docs[470:510]`), 41-field schema, 498 gold-present fields, all text, no
windows, decoded at 65,536 tokens.

| model | exact@all ↓ | exact | tokF1 | contains | null_ok | answered | latency p50/p90 |
|---|---|---|---|---|---|---|---|
| gemini-4-argon | **.5184** | **.5539** | **.7358** | .8926 | .9151 | 466/498 | 22.1 / 52.7 s |
| opus-5.5 | .4502 | .4851 | .6773 | .8156 | .9002 | 462/498 | 11.9 / 22.2 s |
| gpt-6.1-sol | .4479 | .4817 | .6360 | .8045 | .8888 | 463/498 | 9.3 / 17.2 s |
| **clarke 2B** (preamble adapter + slot-mask + candidate parties + ranker, **baked**) | .4373 | .5299 | .6418 | .7012 | **.9562** | 411/498 | **0.29 / 2.6 s** |
| clairic | .0175 | .0176 | .4067 | **.9526** | .1699 | **496/498** | 0.83 / 2.4 s |

Rows are sorted by **`exact@all` ↓** — the one column that multiplies per-answer
precision by field coverage, so it is the number to read first. Bold is the best
value in each column (three models hold one). The 12-field table below is an
experiment ladder in run order, not a ranking.

**Preamble supervision + the labelled LEDGAR corpus pay together.** N keeps L41's recipe
and changes two things: the adapter trains on 24,548 snippets — 17,576 LEDGAR clause crops
plus 1,162 preamble crops from CUAD and the 507 labelled documents, repeated 6×, so 28% of
its diet is `parties`/`document_name`/`agreement_date`, fields it previously **never** saw as
targets — and the head trains on CUAD plus those 507 documents at 41 fields for 3,908 steps
instead of 1,880. exact .5139 → **.5299**, tokF1 .6346 → **.6418**, contains .6997 →
**.7012**, null_ok .9606 → .9562, answers 412 → 411: **+1.21 pp `exact@all`**. The adapter
and the data changed together, so this is a pipeline result; run P's CUAD-only-head controls
are what separates them, and the README will say so until they land.

**Correction to the previous table.** The earlier external rows read `.3767–.4155`
and produced a "within 0.7 pp of the best external" claim. Those runs answered only
364–399 of 498 fields because 5–7 calls per model failed — the 4,000-token output
budget truncated the 64,883-token contract (gemini/opus mid-JSON, gpt emitted no
text at all because reasoning consumed the whole budget). With a 12,000-token budget
and low reasoning effort there are **zero failed calls**, the frontier rows answer
462–466 of 498, and the honest gap was **4.1 pp behind opus and 11 pp behind gemini**,
not 0.7 pp (with N it is 1.3 pp and 8.1 pp). The superseded run is kept in `reports/external_doc/`; the numbers above
are `reports/external_doc_rerun/`.

On the **28 records every run completed** (329 gold fields), which removes the
coverage difference: gemini .5339, opus .4656, gpt .4622, clairic .0162 (published
run: .5232 / .4995 / .5003 / .0162). So the coverage jump is the big effect, and
`effort=low` costs gpt and opus ~3–4 pp of per-answer precision on identical records
— accepted here because latency is part of the comparison.

**Latency** is end-to-end per document. For us: encoding plus one forward pass,
batch 1, 65k cap, single RTX 5090, model load excluded (`clarke-eval`'s `timing`
block, `reports/N-mix41-finalized.json`). For the externals: the same 40 documents
through one Cloudflare AI Gateway run, so the number includes the gateway hop,
network, and autoregressive generation — they emit 1,359 (gpt), 2,121 (gemini) and
1,505 (opus) output tokens per document on average, which is what the p50/p90 gap
over our single forward pass measures. All four are measured on the same documents
as the metrics, in the same run (`reports/external_doc_rerun/*.timing.jsonl`).
Latency tracks contract length: these documents run from 972 to 300,768 characters,
hence our 0.29 s p50 against 2.6 s p90 (mean 1.0 s, 41 fields/s).

Read it honestly: **the frontier is 1.3–8.1 pp ahead on `exact@all`**, and the gap is
mostly `contains`/`tokF1` (.70 vs .89–.95) — they find and quote the right clause
more often. The externals also answer more fields than us (462–466 vs 411), so the old
read that we trade precision for coverage no longer holds: we keep `null_ok`
(.956 vs .888–.915) and latency (30–75× faster) and lose on accuracy and coverage both. Clairic never declines (496/498 answered), which is why its
`contains` is highest and its `exact` lowest (.018).

Our row's decode policy — span tie-break, per-field length prior, slot masking and
the candidate/ranker path for `parties` — is written into the checkpoint as
`decode_config.json`, so `clarke-eval --head <dir>` reproduces these numbers with no
flags. External numbers come from `scripts/external_baseline.py` through one gateway,
on the same 40 documents. Their `grounded_rate` — how often the quoted text is
actually present in the state — is .75 (gemini), .83 (gpt), .82 (clairic), .86
(opus): **even the best of them cannot locate a seventh of what it quotes.** A `read`
answer from this model is grounded by construction, which is the whole argument for
a non-generative readout.

**Private commercial-contract check.** A hand-labelled set of 39 services contracts
(from the company; eval-only and kept private — raw text and predictions never leave the
machine, only aggregates are published) maps onto 8 of our fields with 112 gold-present
fields. L41: exact .2063 / tokF1 .7464 / contains .8702 / answered 95 of 112. N: exact
.3181 / tokF1 .8003 / contains .8110 / answered 82 of 112 (`exact@all` .1750 → .2329).
The trade is localised to `parties`: N is +19.2 pp more exact per answer there but declines
10 of 39 contracts, where L41 answers all 39. Read `contains` on this surface, not `exact` —
the gold citation is the whole clause and we return the core phrase — and treat absent
labels as missing, since "no" in the export means "not annotated" (`null_ok` is not
comparable to CUAD's). Numbers: `reports/company-eval-summary.json`.

**Smaller schema, sharper signal.** Restricting to the 12 most common CUAD fields
(60.8% of gold instances) makes each field matter more per step:

| model (12-field eval, 310 gold) | exact@all | exact | tokF1 | contains | null_ok | answered |
|---|---|---|---|---|---|---|
| 41-field H4, scored on the 12 fields | .5499 | .6067 | .7213 | .7940 | .8765 | 281/310 |
| 12-field I-twelve (frozen backbone) | .5615 | .5941 | .7261 | .7941 | .8118 | 293/310 |
| **12-field + gentle snippet LoRA** | **.5729** | **.6103** | **.7311** | **.7994** | **.8824** | 291/310 |

So a smaller schema is **not** a capability win — it is a cheaper, lower-variance
experiment loop (~12 minutes per run), which is what made the adaptation experiment
affordable.

**Backbone adaptation works, but only when it is gentle.** LoRA on 4 k LEDGAR
snippets, adapter merged, then a fresh head trained on full 65 k documents:
`r=32, lr 2e-5, 800 steps` improves every metric (row above), while the same
mechanism at `r=64, lr 2e-4, 2,000 steps` collapses to `exact .080` / `null_ok 0.00`
because the adapter rewrites the features. Head-training loss at step 500 is a cheap
early warning: 3.44 (original backbone), 9.01 (harsh adapter), 2.91 (gentle).

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
  lm_adapt.py             unsupervised LoRA adaptation on packed token blocks (merge)
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

## Current focus and roadmap

### Where the loss actually is

We are **2.5–9.3 pp behind the frontier on `exact@all`** (.425 vs .450 opus / .518 gemini)
and behind on `contains` (.70 vs .82–.89). It is not tokenization: **94.4% of the
1,022 gold spans in doc40 are expressible with Qwen's vocabulary** (94.1% with a
contract-trained BPE — 5.6% start or end mid-token and are unreachable with *any* BPE),
and our per-answer `exact` (~.49) is about 52% of that ceiling. What remains is mostly
**one** failure, measured three ways:

1. **Under-emission, and it is universal — not a `parties` quirk.** Every `list<text>`
   field emits **1.03 items where gold has 1.84: 44% of gold items are never produced.**
   Worst: `minimum_commitment` 2.54→0.69, `covenant_not_to_sue` 2.57→0.71,
   `ip_ownership_assignment` 3.14→1.43, `insurance` 2.79→1.21.
2. **The misses are wrong answers, not refusals.** At the current operating point ~305
   of 335 misses are *wrong-reported* and only ~30 are false nulls. Lowering the slot
   threshold or adding per-field null thresholds buys recall but not exactness — the
   extra items are wrong (+0.24 pp `exact@all` at slot .35, i.e. noise).
3. **It is selection, not cropping.** When the model answers, crops are
   boundary-exact ~86–88% of the time, and for `parties` 82% of gold starts are already
   in some slot's top-5. The model can find the text; it cannot decide *which* and
   *how many*.

### Levers we are on

| lever | targets | cost | what a result would tell us |
|---|---|---|---|
| **Constrained candidate decode beyond `name`-like fields** | set-completeness for clause fields | one decode sweep per field, no training | whether under-emission is recoverable without new supervision — this is what took `parties` from 0/40 to 20/40 |
| **Learned ranker as a real component** | *which* candidate, once found | ~1 min to fit, CPU only | the 14-feature logistic regression already beats the span pointer on `parties` (precision .858→.890); does a small MLP? |
| **Order-free slot assignment + an extra-candidate dial** | *how many* items | one training run each | assignment gave the best single-run result so far; +1 candidate crosses 90% `contains` on `parties` but costs strict doc-exactness |
| **Supervision for rare fields** (12-field loop, LEDGAR spans, LLM-labelled preambles, mined EDGAR) | coverage of the fields that only have 3–15 gold examples | CPU/network, subscription not metered spend | the 12-field schema is neutral on score but 12 min per run — cheap enough to make data questions answerable. Labels must carry the **full schema with explicit nulls**, or we re-create the sparse-mix failure |
| **Schema / field / prompt permutation augmentation** | the head keying on CUAD's exact instruction wording | one training run | untested, and Cloudflare's recipe uses it — cheap to falsify |
| **Per-field null bias and slot completeness in the loss** | the ~30 false nulls and set completeness | one training run | untested as loss terms (as decode knobs they were noise) |

### LoRA: the one untested mechanism, and why it is stuck

LoRA is the last untested lever, and the one Cloudflare's published recipe actually uses.
On a single 32 GB card it fits **~4k tokens and OOMs at 8k+** — memory is ~30 GB at 16k
*and* at 32k, which is what proves gradient checkpointing is a no-op in the 18
linear-attention blocks (the fix to reach the decoder blocks changed nothing). Since our
regime is 65k, that blocks the experiment rather than just slowing it.

We tried the obvious workaround — adapt on 4k snippets, merge, then re-fit the head on
full documents — and it **degraded the backbone catastrophically**: `null_ok` .0000,
exact .0801, because the snippets taught the head that ~89% of fields are absent. The
open question is therefore sharper than "does LoRA help": **does adapting the backbone at
all beat adapting the head, in the regime we actually care about?** Three ways to find
out, none free: (a) wait for a 64–96 GB card, (b) implement manual per-block
`torch.utils.checkpoint` around the linear-attention blocks (invasive, uncertain), (c) run
LoRA at 4k as a separate short-context experiment, knowing it does not transfer to 65k for
free.

### Explicitly not doing

Head capacity (112M→371M measured flat), backbone scale below 27B (0.23B→9B flat; 2B
wins), decode tuning past the knee of the coverage/`null_ok` curve, more paid API
baselines (budget went to verification, not annotation), and choice-task datasets
like MAUD — those are clef's `choice` API, not an extraction surface.

### Horizon

- **Publish**: model card, weights, decode policy baked so the release behaves like these
  numbers, and `clarke-serve` (`/v1/extraction`, mirroring `clef-server`'s shape).
- **v2 — image states**: the backbone ships a vision tower; render synthetic receipts with
  the same construction-time ground truth, then run ExtractBenchPro (real chart/diagram
  documents with quote+bbox evidence).
- **From-scratch small model**: a 1B trained on ~20B tokens (16B contracts + 4B wiki)
  instead of adapting a general backbone. Needs GPU access we do not have; the 2B result
  says it is not worth attempting on one 32 GB card.
- **Bigger backbones** (27B+): they cost nothing extra in head memory, so the decision is
  purely backbone VRAM and throughput — deferred with the same hardware constraint as LoRA.

## Project docs

- **`JOURNAL.md`** — the lab notebook, newest-first: every verdict, deviation from
  a pre-registered rule, and negative result, with numbers, run dir and report path.
- **`OPERATIONS.md`** — the working rules that survive a context reset: reporting
  contract, the 65k context regime, GPU/shell discipline.
- **`PLAN.md`** — acceptance bar, known walls, escalation ladder.
- **`docs/data-landscape.md`** — corpora surveyed, and why each was taken or dropped.
