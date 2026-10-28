# Data landscape for clarke-flash

**Status:** internal research memo (revision 2)
**Date:** 2026-10-26
**Prepared for:** whoever picks up the data side of clarke-flash
**Companion docs:** `JOURNAL.md` (what actually happened), `PLAN.md` (acceptance bar and ladder),
`OPERATIONS.md` (working rules), `reports/README.md` (how to read the numbers)

> **Reading order:** journal → PLAN → this file. `README.md` describes the architecture correctly
> but its "frozen 9B" framing is stale — the backbone is being re-selected and the current best arm
> is 2B. Where this memo and the code disagree, the code wins; where it and the journal disagree,
> the journal wins.

---

## 1. Purpose

This document answers one question: **which public datasets could be used to train or evaluate
clarke-flash, and what is worth fetching next?**

It is written to be read cold. clarke-flash is a System One extraction model: a frozen backbone plus
a small extraction head that answers a schema of typed fields in one forward pass, with no generated
tokens. Answers are **read** (a span copied from the input), **computed** (1–2 step arithmetic over
spans), or **predicted** (a calibrated distribution), and each carries confidence, `p_null` and
character offsets.

The model is backbone-agnostic by design; head-only training on a frozen backbone is the current
regime. What follows is therefore written against the **data contract** and the **measured
failures**, not against any particular backbone size.

Everything below was link-checked on **2026-10-26**. Counts come from dataset cards / dataset viewer
metadata unless stated otherwise; they were not re-counted by downloading. Licenses are as declared
by the source. Where a license is missing or ambiguous, that is called out rather than guessed.

---

## 2. What the data has to look like

The model's data contract, in one table:

| Schema type | Supervision needed | What a dataset must contain |
|---|---|---|
| `text` | start/end pointer over state tokens | the value as an **exact substring** of the document, with offsets |
| `list<text>`, `list<number>`, `list<date>` | up to 8 presence-gated slots | repeated records / line items, each field locatable |
| `number`, `duration` | sign × exponent × digit heads | numeric value plus the string it came from |
| `date` / `time` / `datetime` | calendar component heads | dates in real formats (not pre-normalised only) |
| `estimate`, `forecast` | distribution over a declared grid | series or realized draws, with a reference date |
| any type | `null` | fields that are genuinely **absent**, not just unlabelled |

Hard limits from `schema.py`: `LIST_SLOTS = 8`, `SPAN_MAX_TOKENS = 256`, states are cut at ~24k
chars / 8k tokens.

Three consequences drive the recommendations:

1. **A dataset without exact locations is only half-usable.** Question-answering sets with gold
   answers but no evidence can still be used if the answer string is findable unambiguously in the
   document; otherwise conversion cost explodes.
2. **Nulls are a first-class signal here.** Datasets that only annotate present fields silently
   teach the model to over-extract. Datasets with explicit absence labels (or decoy keys) are
   disproportionately valuable.
3. **Volume is now a constraint, not a nicety.** Head-only training on a small frozen backbone
   needs tens of thousands of records to move a metric; current mixes are 17.5k–88k records. A
   corpus under ~1k documents is an **eval**, or a diagnostic slice, not a training set.

---

## 3. Where the model is actually losing points

Recommendations are ranked against this table, not against the architecture description.
All numbers from `reports/` (2B arm, full metric set per `OPERATIONS.md`):

| Surface | `exact` | `tokF1` | `contains` | `null_ok` | answered / gold | `exact@all` |
|---|---|---|---|---|---|---|
| `real168` (short windows, CUAD) | .4070 | .5541 | .6859 | .9853 | 380 / 593 | .2608 |
| `doc40` (40 full contracts, 65k) | .4407 | .5731 | .6790 | .9720 | 322 / 498 | .2850 |

The acceptance bar (`PLAN.md`) is **≥80% exact over the solvable subset, per family group**, with
`null_ok ≥ 90%`.

What the table says:

- **Nulls are not the problem.** `null_ok` is .97–.99 already.
- **Localisation is not the problem.** `contains` (.68) is ~1.6× `exact` (.41–.44): the model finds
  the right region and then misses the span boundary.
- **Recall is the problem.** At document level the model answers 322 of 498 gold fields (65%), so
  `exact@all` (.285) is a third of `exact`. Every document-level gain now comes from answering more
  fields on long documents, and from getting span boundaries right once it answers.
- **Capacity is not the problem either.** 4B and 9B both score .3884 on `real168` — below the 2B —
  and the 230M arm still answers 243/498. Adding parameters has been measured and does not help.

So the useful question for a dataset is: *does it add long documents with locatable values, repeated
records, or aggregation programs?* Not: *does it add new schemas?*

---

## 4. TL;DR — recommended order

| # | Dataset | Role | Why here |
|---|---|---|---|
| 1 | **MultiHiertt + HiTab** | train | Aggregation/counting supervision with programs — the one head the repo explicitly leaves unsupervised, and CPU-only to convert (no PDFs). Directly targets the documented wall |
| 2 | **Kleister-NDA** | train | Long documents, needle fields, and **decoy keys that must return null**, in a genre CUAD doesn't cover. Now available as text on HF, so the PDF step in revision 1 is unnecessary |
| 3 | **DocILE** (access request) | train | 100k synthetic invoices with labels by construction + 6,680 annotated real ones, 55 classes, **line items**. The only large supervision for `list` slots |
| 4 | **DocFinQA + LongDocumentQA** | train / eval | Long financial documents with programs — the only large-ish long-doc computed surface outside CUAD |
| 5 | **CORD + SROIE + WildReceipt** | train | ~3,700 real receipts with OCR words, offsets, money/date/list fields. SROIE is already scaffolded; the other two are not |
| 6 | **ExtractBench (+ Pro)** | eval | The closest external analogue: values, nulls, arrays and grounding scored separately. Eval-only |
| 7 | **VAREX** | eval | 1,777 forms across 1,771 schemas — a schema-generalisation probe. Eval-only, and values are **synthetic** |

**Already ingested, do not re-acquire:** DROP (1,492 train / 198 test records inside
`typed_train`/`typed_test`), SQuAD2 (3,000 / 200), CUAD train (61 rows via `typed_train`), ExtractBench
(8 test records). See §11.

**Explicitly out of scope for now:** MAUD (choice API, not spans), FUNSD/XFUND/ChartQA (non-commercial
or copyleft → eval-only at best), layout/classification corpora (no field values), the thoughtworks
aggregate (CUAD contamination, see §5.4), TAT-QA/FinQA (optional; partly redundant with DROP +
multi-row programs), SEC XBRL (only after the training steps above have reported).

---

## 5. Tier 1 — evaluation-grade

### 5.1 ExtractBench

- Link: https://huggingface.co/datasets/llamaindex/ExtractBench
- Paper: *ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction*,
  arXiv:2607.29677 (2026)
- Size: 370 documents / 4,869 pages, 8 business domains, 67 document types.
  Splits: 252 short (≤10 pages), 98 medium (11–50), 20 long (>50).
- Ground truth: one JSON Schema per document type; values plus **page + bounding-box evidence**;
  explicit `null` for absent fields; arrays scored with Hungarian matching.
- Metrics: order-insensitive value F1, word-level grounding F1 (IoU 0.5), page-level grounding F1,
  plus measured cost.
- License: Apache-2.0.
- **Grounding coverage:** the current card says box ground truth now covers **339 of 370** documents
  ("new box ground truth for 103 real documents"). Revision 1 quoted 237/370 — that is stale. Pin a
  revision SHA (the card itself recommends `revision=v1.0`) if numbers are quoted.
- ⚠️ **Name collision:** there is a *different* 2026 benchmark also called ExtractBench
  (arXiv:2602.12247, 35 PDFs, 12,867 fields, schema-as-executable-spec). It is a separate dataset
  and not on HF under this name. Cite the arXiv id, never the bare name.
- Why it matters: it separates exactly the things this project claims — value accuracy, completeness
  of repeated records, null correctness, traceability.
- Caveats: scoring is JSON-schema based, not pointer-based, so a shim is needed. Documents are PDFs,
  so the text path needs a PDF extractor — **and this repo has none** (`pyproject.toml` has no PDF
  dependency, and `OPERATIONS.md` §4 requires an explicit owner go-ahead plus source verification for
  any new package). Treat the extractor as a prerequisite with its own ask, not a footnote.
- Role: **eval-only**. Note 8 `typed-extractbench` records already sit in `typed_test`; any future
  adapter must exclude those documents by content hash.

### 5.2 ExtractBench Pro

- Link: https://huggingface.co/datasets/llamaindex/ExtractBenchPro
- Size: 100 cases in six hard categories: document conventions (22), house vocabulary (14),
  house counting rules (12), drawn relations (20), charts inside documents (25), long chart sheets (7).
- License: Apache-2.0.
- Why it matters: targets the walls directly — counting/aggregation, in-prose tables, drawn
  relations, house vocabulary. This is the right test for whether `computed` is real.
- Caveats: small; diagnostic slice, not a headline. Mostly v2 (image) territory.

### 5.3 VAREX

- Link: https://huggingface.co/datasets/ibm-research/VAREX
- Paper: *VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents*,
  arXiv:2603.15118 (2026)
- Size: 1,777 documents, 1,771 unique schemas, four input modalities.
- Columns: `doc_id`, `split`, `image`, `image_50dpi`, `schema`, `ground_truth`, `text_flow`,
  `text_layout`.
- License: CDLA-Permissive-2.0.
- ⚠️ **The values are synthetic.** Documents are PDF templates programmatically filled by a Reverse
  Annotation pipeline (~98.5% field-level accuracy). The schema diversity is real; the surface forms
  are not. This measures "can it read a schema it has never seen", not "can it survive a real
  government form".
- ⚠️ `split` is the **structural category** (Flat / Nested / Table), not a train/test split. There is
  no held-out split; you must carve one by `doc_id`.
- Why it matters: `text_flow` means the text path works without vision, and 1,771 schemas means the
  model cannot memorise a field inventory.
- Caveats: ground truth is values, not boxes. Recovering spans requires exact-string search in
  `text_flow`; ambiguous or normalised values (dates, names repeated in headers) need an accept-list
  or must be dropped. Some fields are absent rather than null — check per schema before treating
  absence as `null`.
- Role: **eval-only**.

### 5.4 thoughtworks/document-processing-benchmark

- Link: https://huggingface.co/datasets/thoughtworks/document-processing-benchmark
- Contents: eight public document datasets normalised into one parquet schema.
  Configs: `full` 10,074 docs, `test` 4,522, `v2` 900 (multi-page VQA + bank statements),
  `v3` 186 (CUAD contracts).
- Extra value: per-row `source_dataset`, `license`, `license_url`, `upstream_url`, plus real API
  token/latency/cost baselines for reference models.
- License: mixed, per-row (`license: other`, "mixed-research-and-permissive").
- Why it matters: a ready-made harness and a **cost/latency foil** — `reports/external*` already does
  this for internal models, so the value here is the published API baselines, not the documents.
- Caveats: ⚠️ `v3` is CUAD and other rows are CUAD-derived. Filter by `source_dataset` before any
  use; the CUAD-derived rows must never enter training while CUAD sits in the eval surfaces.

---

## 6. Tier 2 — real training corpora

### 6.1 Receipts and invoices

| Dataset | Size | Fields | License | Notes |
|---|---|---|---|---|
| [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 1,000 receipts | 30 classes, OCR words + boxes | CC BY 4.0 | Indonesian; noisy photos; good list/money coverage. **Not** scaffolded, despite the README |
| [SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | 987 receipts | 4 fields + full OCR boxes | CC BY 4.0 | Already scaffolded in `public.py`; cheapest first adapter |
| [WildReceipt](https://huggingface.co/datasets/Theivaprakasham/wildreceipt) | 1,740 receipts | 25 classes | Apache-2.0 (mirror) | Store/address/date/total; English. Third-party re-upload |
| [invoice-extraction-benchmark](https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark) | 181 docs | AP fields, answer keys + scorer | CC BY 4.0 | **Synthetic**, and made by a vendor selling invoice extraction. Tiny → private eval only |

Receipts are a *surface-form* corpus: they will not move `doc40`, and they will not teach multi-row
aggregation. They matter because receipts are the canonical clerk document, and because the head has
never seen a real receipt.

Provenance: SROIE and WildReceipt on HF are user mirrors, not canonical releases. Record the HF
revision SHA (or fetch from the original distribution) or the data is not reproducible.

### 6.2 Contracts and long documents

| Dataset | Size | What you get | License | Notes |
|---|---|---|---|---|
| [Kleister-NDA](https://huggingface.co/datasets/orgrctera/kleister_nda_train) | ~540 NDAs; 254 train docs on this mirror | key→value with source annotations; **decoy keys that must return null** | research data, no SPDX | Long documents, different genre from CUAD; the decoys are free null supervision |
| [Kleister-Charity](https://huggingface.co/datasets/orgrctera/kleister_charity_information_extraction) | ~2k charity reports | same shape | research data, no SPDX | Same pipeline; weaker fit |
| [LongDocumentQA](https://huggingface.co/datasets/michaelkrumdickkensho/LongDocumentQA) | 225 examples | question + long context + answer span | **none declared** | From the DocFinQA team. Eval-only until licensed |
| [LEDGAR / LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) | 52.5k labelled clauses (full LEDGAR ~800k) | real clause text + 100 labels | CC BY 4.0 | Label-only: no field spans. Clause pool / auxiliary signal, not span supervision |

The Kleister finding is worth stating plainly: revision 1 budgeted "PDF → text; decoy keys → nulls"
as medium effort. The HF mirrors ship the documents as text with annotations, so that step
disappears. What remains is schema construction and span resolution inside `public.py`'s existing
verifier.

### 6.3 Financial tables and computed answers

| Dataset | Size | What it adds | License | Conversion |
|---|---|---|---|---|
| [MultiHiertt](https://huggingface.co/datasets/yilunzhao/MultiHiertt) | 10,440 QA over ~2.9k hierarchical tables (paper count) | multi-step numerical reasoning, **aggregation and counting** | MIT | Medium (serialise table + program) |
| [HiTab](https://huggingface.co/datasets/kasnerz/hitab) | 10,672 rows on this mirror (~10.4k QA over 3.6k tables, paper) | aggregation/counting over hierarchical tables, with supporting facts | **no license tag** | Medium |
| [ConvFinQA](https://huggingface.co/datasets/MehdiHosseiniMoghadam/ConvFinQA) | 3,892 conversations over 3,037 reports (paper) | multi-turn computed chains | **no license tag** on this mirror | Medium |
| [TAT-DQA](https://huggingface.co/datasets/next-tat/TAT-DQA) | ~16.5k QA | document-level table+text QA with evidence | CC BY 4.0 | Medium–high (page layout) |
| [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) | 7,437 QA | long financial documents, gold programs | MIT | Medium–high (long docs) |
| [TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA) | 16,552 questions | table + text states; answers, derivations, scale | CC BY 4.0 | Medium |
| [FinQA](https://huggingface.co/datasets/dreamerdeo/finqa) | 8,281 QA | gold programs + `gold_evidence` | **unverified** — this mirror has no license metadata; MIT applies to the upstream repo, not necessarily to this copy | Medium |
| [DROP](https://huggingface.co/datasets/ucinlp/drop) | 77,409 train / 9,536 validation questions | spans + answer types (number/date/span) + arithmetic | CC BY-SA 4.0 | **Already ingested** (see §11) |

Notes:

- **The aggregation row is the important one.** `README.md` says the remaining ceiling is multi-row
  aggregation, "left unsupervised by design", and `PLAN.md` lists the virtual-column aggregate as a
  training slot queued after the current verdicts. MultiHiertt and HiTab are the cheapest sources of
  supervised "total/average/count over these rows" that exist, and neither needs PDF tooling.
- DROP's revision-1 description ("96,935 passages") was wrong twice: the card has **77,409 train /
  9,536 validation *questions***, and the ~96k figure is the paper's question count. Its train split is
  already inside `typed_train`.
- DROP's share-alike is not a theoretical future question: `typed_train`/`typed_test` already derive
  from `LocalLLaMA/typed-extraction`, which is CC-BY-SA-4.0. Whatever the release license is, that
  obligation is live today (§10).

### 6.4 Low-value / do not bother

| Dataset | Why not |
|---|---|
| MAUD | answers are multiple-choice comprehension, not spans. Belongs to a `choice` API |
| [evalsafe-invoice-processing](https://huggingface.co/datasets/typesafeAI/evalsafe-invoice-processing) | 150 cases / 6,874 questions, but labels are **model-generated consensus**, not ground truth |
| [ReceiptBench](https://huggingface.co/datasets/T0rl/ReceiptBench) | card is a license line and nothing else — no usable description or schema |
| FUNSD | 199 forms only, and **non-commercial research terms** (EPFL). Eval-only at best |
| XFUND | useful multilingual forms, but **CC BY-NC-SA 4.0**. Eval-only |
| DocLayNet / PubLayNet / RVL-CDIP / DocBank | layout/classification labels, not field values |
| ToTTo / WikiTableQuestions | table→text generation, not typed extraction; Wikipedia tables already mined directly |
| SQuAD2 / generic NER | already in `typed_train`/`typed_test`; no clerical schema or typed values |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | 741k rows of typed decisions with option distributions, Apache-2.0 — but **distilled from a teacher**, so the distributions are a model's opinion, not ground truth. Adjacent to the `estimate`/grid heads for calibration work only; never as span or null supervision |

---

## 7. Tier 3 — programmatic ground truth (no annotation)

The highest-leverage direction is not another labelled dataset; it is **pairing a public structured
record with the document it came from**. The value comes from the API; the offsets come from the
document.

**This is already the project's biggest data asset and the memo should say so.** 235 GB of enwiki
XML has been downloaded, census-verified 19/19, and mined into infobox/table records
(`data/mixes/wikiib_train.jsonl`, ~88k records; `wikiib_eval.jsonl`). Revision 1 mentioned Wikipedia
only in the "do not bother" row. That is backwards: the programmatic thesis is already validated in
this repo, and the question for a new programmatic source is whether it beats deepening that pipeline
— not whether programmatic labels work at all.

| Source | Structured side | Document side | License | Fit |
|---|---|---|---|---|
| SEC EDGAR XBRL | `companyfacts` API (verified live: 3.8 MB for one CIK) | 10-K / 10-Q HTML and PDFs | US public domain | numbers, dates, periods; `computed` deltas |
| Wikidata truthy dump | claims with qualifiers | Wikipedia infoboxes | CC0 | repair/verify the existing infobox mining |
| UK Companies House | iXBRL accounts, free API key | filed accounts PDF/HTML | OGL | money, dates, company metadata |
| FRED / ALFRED | series + **vintage** snapshots | rendered series state | public data, API key | honest `forecast` labels |
| FCC PublicFiles / USASpending / FPDS | award and filing metadata | underlying forms and invoices | public records | procurement/invoice fields |
| [anon-rabbit/annual_reports](https://huggingface.co/datasets/anon-rabbit/annual_reports) | — (unlabelled) | 1,700 corporate annual report PDFs | Apache-2.0 | document side for the XBRL pairing, if the filings aren't fetched directly |

Caveats: exact-value search is ambiguous (the same number appears in headers and totals), so the
converter needs a reject-on-ambiguity rule, exactly like the existing span verifier. HF "XBRL"
datasets are mostly derived mirrors of unclear provenance and licensing — go to the SEC API
directly rather than through a re-upload.

---

## 8. Tier 4 — forecast / estimate head

- [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) — Apache-2.0; the existing CSV
  adapter can ingest it.
- [chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) — mixed licenses;
  series-level only.
- Monash Time Series Repository / M5 — mixed licenses; series-level only.
- **ALFRED vintages** are the interesting one for calibration: a static series gives one
  realization; vintages give the real predictive question (what was knowable at time *t*).

The forecast head is not currently a bottleneck in any report. This tier should stay untouched until
an eval shows interval coverage drifting outside ±10pt.

---

## 9. Tier 5 — v2 image states

Already on the roadmap; the datasets above double as image benchmarks:

- **DocILE** — page images, 55 classes, boxes, line items.
- **VAREX** — `image`, `image_50dpi`, `text_layout`; 4 modalities.
- **ExtractBench** — value-level boxes on 339 of 370 documents; page-level on more.
- **CORD / SROIE / WildReceipt** — receipt photos with word boxes.
- [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) — Apache-2.0 as packaged on HF.
- [DocVQA-2026](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026) — small (n<1K), no license tag;
  treat as a candidate, not a commitment.
- [ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) — GPL-3.0 → eval-only.
- [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) and
  [FinTabNet](https://huggingface.co/datasets/bsmock/FinTabNet.c) — CDLA-Permissive-2.0;
  table structure, useful for multi-row aggregation.
- [SynthDoG](https://huggingface.co/datasets/naver-clova-ix/synthdog-en) — 0.5M synthetic document
  images (Donut), **no license tag**. Same bet the repo already makes in `synth.py`: labels by
  construction. Attractive for v2, but unlicensed as packaged.

---

## 10. Licensing and role matrix

Revision 1 contradicted itself: its eval tier called ExtractBench "eval-only" while its licensing
table called it "safe to train". One row per dataset, one role, one verdict.

| Dataset | License | Role | Verdict |
|---|---|---|---|
| ExtractBench / Pro | Apache-2.0 | eval-only | **Do not train** (benchmark integrity, and 8 records already frozen in `typed_test`) |
| VAREX | CDLA-Permissive-2.0 | eval-only | **Do not train** (synthetic values; eval integrity) |
| MultiHiertt | MIT | train | Safe |
| DocFinQA | MIT | train | Safe |
| Kleister-NDA / Charity | research data, no SPDX | train | Needs a terms read before training; decoys are still usable for eval |
| CORD, SROIE, TAT-QA, TAT-DQA | CC BY 4.0 | train | Safe with attribution |
| WildReceipt | Apache-2.0 (mirror) | train | Safe, but pin the revision |
| DROP | CC BY-SA 4.0 | train (already in) | Share-alike already operative via `typed-extraction` |
| **LocalLLaMA/typed-extraction** | **CC BY-SA 4.0** | train (already in) | **The deciding license for the release** — see below |
| HuggingFace LEDGAR | CC BY 4.0 | auxiliary | Safe with attribution |
| HiTab, ConvFinQA | no license tag on the mirrors | candidate | Verify upstream before training |
| LongDocumentQA | none declared | eval-only | Do not train |
| PubTables-1M / FinTabNet | CDLA-Permissive-2.0 | v2 | Safe |
| GIFT-Eval / chronos | Apache-2.0 / mixed | forecast | Safe / check per-series |
| SynthDoG | none declared | v2 candidate | Do not ship; generator itself is the safer path |
| FUNSD, XFUND, ChartQA | non-commercial / NC-SA / GPL-3.0 | eval-only | Never train |
| DocILE | access-request terms | train | Read before converting |
| thoughtworks aggregate | mixed per-row | eval-only | Filter by `source_dataset` |

**The licensing question is no longer hypothetical.** The release plan is Apache-2.0 weights
(`hiemanshu/clarke-flash`) with a dataset-credits section, but two CC-BY-SA-4.0 corpora
(`typed-extraction`, DROP) are already in the training mix. That needs an owner decision now, not at
release time: either accept share-alike on the derived data and document it, or rebuild the mix
without `typed-extraction` (9,553 of the current `typed_train` records, and the main real-surface
corpus).

---

## 11. Leakage and data hygiene

The project freezes these eval surfaces: `typed_test`, `real_eval`/`doc40` (CUAD), all Wikipedia
holdouts, and `acord_test`.

**What is already true in the repo** (verified by counting the JSONL, 2026-10-26):

| Family | `typed_train` | `typed_test` | Note |
|---|---|---|---|
| `typed-drop` | 1,492 | 198 | passage ids disjoint (`overlap 0`) — the internal split is clean |
| `typed-squad2` | 3,000 | 200 | same shape |
| `typed-cuad` | 61 | 46 | contract training already routed through typed-extraction |
| `typed-extractbench` | 0 | 8 | ExtractBench is already an eval surface by accident |

Consequences:

1. **DROP is not a new acquisition.** Its train split is already in `typed_train` (1,492) and its
   validation split is what `typed_test`'s drop family is drawn from (198). Adding `ucinlp/drop`'s
   **validation** split to training would land directly on a frozen eval surface. Add only the train
   split, and only if something specific is missing (images, answer types).
2. **ExtractBench must not be trained on**, and any future adapter must drop the 8 documents already
   in `typed_test` by content hash.
3. **CUAD-derived data must never enter training.** That includes `v3` of the thoughtworks benchmark
   and the CUAD rows inside `LocalLLaMA/typed-extraction` beyond the existing split.
4. **If DocILE becomes an eval, do not train on DeepForm or VRDU-ad-buy.** All three draw from the
   same FCC political-file / UCSF document pools; document-level overlap is plausible even when
   titles differ.
5. **Use content hashes, not title hashes** (already the standing rule). Normalise whitespace, then
   compare head/tail fragments for truncated windows.
6. **Audit by source, not by dataset name.** Aggregate benchmarks mix licenses and sources in one
   file; filter on `source_dataset` / `license` columns.
7. **Pin revisions.** Several sources here are third-party HF mirrors (SROIE, WildReceipt, Kleister,
   HiTab, ConvFinQA, FinQA). Record the dataset revision SHA next to the converted JSONL, or a
   re-download silently changes the corpus.

---

## 12. Acquisition plan

One next step, then a backlog. A step is "one change, one hypothesis, one committed report"
(`PLAN.md`), so this section does not propose seven parallel jobs.

### Next step — aggregation supervision (MultiHiertt + HiTab)

- **Hypothesis:** the `computed`/aggregate path is under-supervised, not under-capable. Multi-row
  `total`/`average`/`count` answers improve on `doc40` and on the ExtractBench-Pro counting slices.
- **Work:** serialise hierarchical tables into `state` (the same table serialisation TAT-QA/FinQA
  would need); map gold programs to the 5-way op head plus `--agg-atoms` virtual columns; resolve
  answer strings to spans with reject-on-ambiguity.
- **Effort:** M, CPU-only, no new packages, no PDFs.
- **Volume:** ~20k QA pairs before filtering; expect ~10–15k after span verification.
- **Why now:** it is the only wall the repo names as deliberately unsupervised, and the conversion is
  pure text.
- **Licence gate:** MultiHiertt is MIT. HiTab has no license tag on the mirror — verify upstream or
  use MultiHiertt alone.

### Backlog, in order

| Step | Dataset | Work | Effort | Expected effect |
|---|---|---|---|---|
| 2 | Kleister-NDA | text → state; decoy keys → nulls; span resolution | S–M | doc-level recall on long documents + null decoys; a different genre from CUAD |
| 3 | DocILE (access request) | synthetic split first (labels by construction), then annotated split | M–L | `list` slots and line-item supervision |
| 4 | DocFinQA + LongDocumentQA | long-doc conversion; program → computed | M | long-document computed answers |
| 5 | CORD + WildReceipt | OCR words → state with offset map (SROIE adapter is the template) | M | receipt surface forms |
| 6 | ExtractBench (+ Pro) | PDF extractor + JSON-schema scoring shim | M | external honesty check |
| 7 | SEC XBRL pilot | 100 filings; value alignment with reject-on-ambiguity | L | tests the programmatic-label thesis on a new genre |

**Blocker to resolve before step 6:** `pyproject.toml` has no PDF library, and `OPERATIONS.md` §4
forbids new packages without an explicit go-ahead. The PDF extractor is an ask to the owner, with two
candidate libraries and their licenses, not a line item to sneak in. DocILE ships word-level
annotations, so it does **not** need it.

**Split policy:** keep at least one **held-out schema family** from every new corpus out of training,
so "new fields need no retraining" is measured rather than assumed. For Kleister, hold out whole
document types; for MultiHiertt, hold out table families.

---

## 13. Open questions for the owner

1. **Share-alike, now.** Two CC-BY-SA-4.0 corpora are already in the mix. Is the release "Apache-2.0
   weights + CC-BY-SA data credits", or does the mix get rebuilt without `typed-extraction`?
2. Is a DocILE access request acceptable (it is a form, not an automated download)?
3. **PDF extraction:** which library, and is installing it acceptable? (Required for ExtractBench and
   any PDF-only source; not required for steps 1, 2, 4, 5 as scoped.)
4. Is v2 (images) in scope for the next data push, or does effort stay text-only until the real-world
   text bar is met?
5. For the programmatic pipeline, is SEC XBRL the right first target, or is a domain closer to the
   existing evals (procurement / insurance / ACORD) more useful?

---

## 14. Verification notes and changelog

Checked on 2026-10-26 via: the Hugging Face dataset API and dataset-server (`info`, `splits`,
`size`, `first-rows`) for existence, sizes, features, splits and license tags; dataset READMEs for
licenses and counts; paper pages for ExtractBench and VAREX; and direct counts over
`data/typed_train.jsonl` and `data/typed_test.jsonl` for the leakage table in §11. Counts were read
from cards and viewer metadata, not re-counted by downloading.

**Corrections made in revision 2:**

| Revision 1 said | Reality |
|---|---|
| "frozen 9B backbone" (from README) | Backbone under re-selection; 2B is the current best arm, 4B/9B are no better. Ranking reworked against `doc40`/`real168` |
| ExtractBench "237 of 370 … 169 human-verified" | Card says grounding covers **339 of 370** |
| DROP "96,935 passages" | **77,409 train + 9,536 validation questions**; ~96k is the paper's question count |
| FinQA "MIT" | The `dreamerdeo/finqa` mirror has **no license metadata**; MIT is the upstream repo's license |
| VAREX "government forms" | Templates filled with **synthetic** values by a Reverse Annotation pipeline; `split` is the structural category, not train/test |
| ExtractBench "eval-only" in the eval tier vs "safe to train" in the licensing table | Resolved into a single role column (§10) |
| DROP listed as a new acquisition | Already ingested: 1,492 train records in `typed_train`, 198 in `typed_test` |
| Kleister requires "PDF → text" | HF mirrors ship text + annotations; step removed |
| No mention of the aggregation gap | MultiHiertt / HiTab added as step 1 |
| No mention of the existing Wikipedia pipeline | Now stated as the repo's largest programmatic asset (§7) |

**New candidates surfaced in this revision:** MultiHiertt, HiTab, ConvFinQA, TAT-DQA,
LongDocumentQA, DocVQA-2026, SynthDoG, invoice-extraction-benchmark, annual_reports (document side
for XBRL), jev-distill-corpus-v3 (calibration only), evalsafe-invoice-processing and ReceiptBench
(both rejected), plus the ExtractBench name collision (arXiv:2602.12247).

**Key links:**

- ExtractBench — https://huggingface.co/datasets/llamaindex/ExtractBench · https://arxiv.org/abs/2607.29677 · (different benchmark, same name: https://arxiv.org/abs/2602.12247)
- ExtractBench Pro — https://huggingface.co/datasets/llamaindex/ExtractBenchPro
- VAREX — https://huggingface.co/datasets/ibm-research/VAREX · https://arxiv.org/abs/2603.15118
- DocILE — https://docile.rossum.ai/ · https://arxiv.org/abs/2302.05658
- thoughtworks benchmark — https://huggingface.co/datasets/thoughtworks/document-processing-benchmark
- CORD — https://huggingface.co/datasets/naver-clova-ix/cord-v2
- SROIE — https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
- WildReceipt — https://huggingface.co/datasets/Theivaprakasham/wildreceipt
- Kleister (NDA train mirror) — https://huggingface.co/datasets/orgrctera/kleister_nda_train · https://github.com/applicaai/kleister-charity
- LongDocumentQA — https://huggingface.co/datasets/michaelkrumdickkensho/LongDocumentQA
- MultiHiertt — https://huggingface.co/datasets/yilunzhao/MultiHiertt
- HiTab — https://huggingface.co/datasets/kasnerz/hitab
- ConvFinQA — https://huggingface.co/datasets/MehdiHosseiniMoghadam/ConvFinQA
- TAT-QA — https://huggingface.co/datasets/next-tat/TAT-QA
- TAT-DQA — https://huggingface.co/datasets/next-tat/TAT-DQA
- FinQA — https://huggingface.co/datasets/dreamerdeo/finqa
- DocFinQA — https://huggingface.co/datasets/kensho/DocFinQA
- DROP — https://huggingface.co/datasets/ucinlp/drop
- LexGLUE / LEDGAR — https://huggingface.co/datasets/coastalcph/lex_glue
- FUNSD terms — https://guillaumejaume.github.io/FUNSD/work/
- XFUND — https://github.com/doc-analysis/XFUND
- PubTables-1M — https://huggingface.co/datasets/bsmock/pubtables-1m
- FinTabNet — https://huggingface.co/datasets/bsmock/FinTabNet.c
- SynthDoG — https://huggingface.co/datasets/naver-clova-ix/synthdog-en
- GIFT-Eval — https://huggingface.co/datasets/Salesforce/GiftEval
- DocVQA — https://huggingface.co/datasets/lmms-lab/DocVQA
- DocVQA-2026 — https://huggingface.co/datasets/VLR-CVC/DocVQA-2026
- ChartQA — https://huggingface.co/datasets/ahmed-masry/ChartQA
- invoice-extraction-benchmark — https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark
