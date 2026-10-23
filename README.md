# clarke-flash

A System One **extraction model** post-trained from `Qwen/Qwen3.5-9B`, in the
spirit of [Cloudflare's Clairic](https://blog.cloudflare.com/clairic-extraction-model/):
given a `state` (document text, tables, series — images in v2) and a schema of
typed fields, it answers **every field in one parallel forward pass**.

No tokens are generated. Answers are:

- **read** — spans copied from the input, with character offsets (a `text`
  answer literally *is* `state[start:end]`; it cannot be re-spelled or invented),
- **computed** — totals, deltas, aggregates re-derived from the input by the
  head (1–2 step arithmetic),
- **predicted** — estimates/forecasts as calibrated distributions inside your
  `[min, max]` on your `step` grid (median, p10/p90).

Every answer carries `confidence` and `p_null`; a null answer still reports its
`guess` + `span`, so near-misses are auditable. `origin` (`read`/`computed`/
`predicted`) tags how the value was obtained — `read` answers are exactly
verifiable against the source text.

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
small (~120M param) **joint extraction head** — evidence-routing cross-
attention + field decoder — reads the final hidden states. The LM head is
never used. Field names and instructions are *input text*, so new fields need
no retraining; the model learns to read them (see surface/instruction
randomization in `synth.py`).

## Heads per type

| schema type                | answer space                                   |
|----------------------------|------------------------------------------------|
| `text`                     | start/end pointer over state tokens (+null)    |
| `list<text>`               | 8 slot pointers (presence-gated)               |
| `number`, `duration`       | sign x exponent x top-3 x next-3 digits        |
| `date`/`time`/`datetime`   | calendar component heads (always-valid output) |
| `estimate`, `forecast`     | distribution over declared grid + below/above  |

Losses: span CE, digit/calendar CE, grid CE, null BCE. `estimate` labels are
realized draws from the generator's latent process, so **calibration is
learned, not post-hoc**: the noise the generator injects sets the interval
widths.

## Training approach

- **Phase 1 (this):** backbone frozen (`no_grad` forward, bf16 ≈ 19 GB of the
  32 GB 5090), train the head only.
- **Phase 2:** optional LoRA if the head plateaus.
- **Data:** synthetic-by-construction ground truth (spans, arithmetic,
  distributions are known when the document is assembled — no annotation
  campaign), plus `public.py` adapters for real sets.

## Layout

```
src/clarke/
  schema.py    typed field specs + answer grids (bins, snapping, percentiles)
  encoding.py  state+schema prompt, token spans, char offset maps (clef grammar)
  head.py      JointExtractionHead (routing + decoder + type readouts + loss)
  model.py     ClarkeModel, answer decoding, grounding/origin, save/load
  synth.py     synthetic generator: support/invoice/ticket/table/comps/forecast
  public.py    SROIE / Few-NERD / CSV-series adapters -> same JSONL
  data.py      RecordDataset + worker-side collate
  train.py     frozen-backbone head trainer (runs/, jsonl logs, checkpoints)
  eval.py      accuracy + null + calibration report
scripts/watch_training.sh    live dashboard (GPU, losses, val)
```

## Quickstart

```bash
uv sync                                  # env (torch cu128, transformers 5.x)

# 1. data
uv run clarke-gen --out data/train.jsonl --count 6000 --seed 1
uv run clarke-gen --out data/val.jsonl   --count 300  --seed 99

# 2. smoke (a few steps, checkpoints to runs/<ts>/)
uv run clarke-train --train data/train.jsonl --val data/val.jsonl \
  --epochs 1 --batch-size 2 --max-steps 20

# 3. the real run — watch it in another terminal:
scripts/watch_training.sh                # dashboard; add `status` for one-shot
uv run clarke-train --train data/train.jsonl --val data/val.jsonl \
  --epochs 3 --batch-size 4 --lr 3e-4

# 4. evaluate typed answers + calibration
uv run clarke-eval --head runs/<ts>/checkpoint-<n> --data data/val.jsonl
```

Logs land in `runs/<timestamp>/train.jsonl` (+ `val` rows); checkpoints keep
the last three. To try transfer init from the decision-model backbone, pass
`--backbone Cloudflare/clef-flash` (already post-trained on the same
STATE/SCHEMA grammar).

## Public data

- **CUAD** (`theatticusproject/cuad`, CC-BY-4.0) — 510 real commercial
  contracts, 41 human-annotated clause categories, 13.8k evidence spans
  (character-exact) and 68% `is_impossible` nulls. Fetch + convert:

  ```bash
  mkdir -p data/raw && curl -sSL -o data/raw/CUAD_v1.json \
    'https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json'
  uv run clarke-public --out data/real_eval.jsonl --doc-start 470   # eval slice (last 40 docs)
  uv run clarke-public --out data/cuad_all.jsonl  --doc-end 470     # train pool (keep disjoint from eval!)
  ```

  Windows are newline-aligned ~16k chars; spans verified byte-exact at
  convert time. 148 eval windows with a 91% null rate — the honest
  synthetic→real transfer test.
- **LocalLLaMA/typed-extraction** (CC-BY-SA-4.0) — 11k records already in
  our exact schema (`state`/`fields`/gold with spans and `source:
  span|computed|relative|normalised`) across 15 real families (invoice,
  lease, bank statement, claims, wikipedia, squad2, drop, cuad…):

  ```bash
  uv run clarke-public --typed-split train --out data/typed_train.jsonl
  uv run clarke-public --typed-split test  --out data/typed_test.jsonl
  ```

  Conversion re-checks every text span against the state and drops
  unverifiable answers; oversized states (>24k chars) are filtered so
  nothing silently truncates past the 8k-token window.
- **Keyphrase** (`jinya425/Keyphrase_extraction`, MIT) — out-of-genre smoke
  eval; fetched by the combined `clarke-public` run above via
  `data/raw/keyphrase-test.parquet` (curl command in `public.py`).
- Scaffolded but unfetched: SROIE/CORD (receipts), Few-NERD (entities),
  any `date,value` CSV dir (real forecast series). Planned: ~200
  human-checked Paperless-Forge documents as a personal-domain eval slice.

Everything under `data/` and `runs/` is gitignored and regenerates from the
seeded generator + the commands above. Eval slices stay frozen by
construction; **MAUD** is deliberately not wired in — its answers are
comprehension choices, not spans, so it belongs to clef's `choice` API.

## Roadmap

- v2: image states (backbone already ships the vision tower; render synthetic
  receipts with the same construction-time ground truth; then run
  ExtractBenchPro — real chart/diagram documents with quote+bbox evidence).
- Serve: `/v1/extraction` HTTP API mirroring `clef-server`'s shape.
- Calibration polish: temperature scaling / conformal coverage pass on val
  (CUAD's 91%-null regime is the best p_null stress test we have).
- ~~Candidate-pointer readout for numbers~~ shipped in main3: read = pick a
  grounded atom, compute = atom ∘ atom via a learned 5-way op head; the
  remaining ceiling is multi-row aggregation, left unsupervised by design.
