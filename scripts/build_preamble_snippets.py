#!/usr/bin/env python3
"""Extract 4k preamble snippets from CUAD + labelled LEDGAR for backbone adaptation.

Why: the existing 17,576 snippets are all LEDGAR clause crops, and **zero** of them
carry `parties`, `document_name` or `agreement_date` as targets (measured: only
governing_law 13,990, anti_assignment 5,976, cap_on_liability 527,
termination_for_convenience 526, effective_date 186). No CUAD crops were ever in the
set. So the adapter has read those field NAMES in the schema prompt but never had to
locate them - and `parties` is the head's weakest field.

This builds preamble-centred crops from
  - data/ledgar_labelled_twelve.jsonl  (507 labelled docs, 12-field schema)
  - data/doc_train_470_twelve.jsonl    (470 CUAD train docs; docs[470:] is the eval, so no leak)
and optionally concatenates them onto data/snippets_4k.jsonl with a repeat factor,
because ~3k new crops against 17.5k existing is 15% of the sampling distribution and
800 adapter steps would otherwise see only ~100 of them.

Cropping is token-exact: offsets come from the tokenizer's offset map, every answer
span is re-verified against the cropped text, and anything that does not fit the
4,096-token encoding (schema included) is shrunk rather than truncated.

    uv run python scripts/build_preamble_snippets.py
    uv run python scripts/build_preamble_snippets.py --repeat 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from clarke.encoding import encode_record
from clarke.textmetrics import norm

PREAMBLE = ("document_name", "parties", "agreement_date")
_TOK = None


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def answer_pairs(a: dict) -> list[tuple[str, list[int]]]:
    vals = a.get("value", a.get("values"))
    if vals is None:
        return []
    vals = vals if isinstance(vals, list) else [vals]
    spans = a.get("spans") or ([a["span"]] if a.get("span") else [])
    return [(str(v), s) for v, s in zip(vals, spans)]


def tok_span(offsets: list[tuple[int, int]], lo: int, hi: int) -> tuple[int, int]:
    """Token index range covering char range [lo, hi)."""
    first = next((i for i, (a, b) in enumerate(offsets) if b > lo), 0)
    last = next((i for i, (a, b) in enumerate(offsets) if a >= hi), len(offsets))
    return first, max(first + 1, last)


def make_crop(rec: dict, text: str, offsets, centre: tuple[int, int], allowed: int,
              fields: dict) -> dict | None:
    """Crop around a char range, keep every answer fully inside it.

    `allowed` is the state-token budget left after the schema prompt, computed once
    from encode_record with a 1-char state, so the crop can never be truncated - a
    truncated state silently turns out-of-range positives into NULL targets.
    """
    t_first, t_last = tok_span(offsets, *centre)
    span = t_last - t_first
    if span > allowed:
        return None
    want = min(allowed, max(span + 400, 1200))
    start = max(0, t_first - (want - span) // 2)
    end = min(len(offsets), start + want)
    start = max(0, end - want)
    for _ in range(6):
        c0 = offsets[start][0]
        c1 = min(len(text), offsets[end - 1][1])
        crop = text[c0:c1]
        answers = {}
        for name, a in (rec.get("answers") or {}).items():
            vals, spans = [], []
            for v, (s, e) in answer_pairs(a):
                if s >= c0 and e <= c1:
                    vals.append(v)
                    spans.append([s - c0, e - c0])
            if vals:
                answers[name] = {"value": vals, "spans": spans}
        if not answers:
            return None
        out = {"id": f"{rec['id']}-pre", "state": crop, "fields": fields,
               "answers": answers, "domain": rec.get("domain")}
        enc = encode_record(_TOK, out, max_length=4096)
        state_tokens = enc.state_range[1] - enc.state_range[0]
        full_tokens = len(_TOK(crop, add_special_tokens=False)["input_ids"])
        if state_tokens >= full_tokens:            # not truncated
            bad = [f"{n}: {norm(crop[s:e])[:30]!r}!={norm(v)[:30]!r}"
                   for n, a in answers.items() for v, (s, e) in answer_pairs(a)
                   if norm(crop[s:e]) != norm(v)]
            if bad:
                print(f"[span] {out['id']}: {bad[0]}")
                return None
            return out
        shrink = max(200, int((end - start) * 0.2))
        start = min(start + shrink // 2, max(0, len(offsets) - 1))
        end = max(start + 1, end - shrink // 2)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgar", default="data/ledgar_labelled_twelve.jsonl")
    ap.add_argument("--cuad", default="data/doc_train_470_twelve.jsonl")
    ap.add_argument("--existing", default="data/snippets_4k.jsonl")
    ap.add_argument("--out-new", default="data/preamble_snippets_4k.jsonl")
    ap.add_argument("--out-combined", default="data/snippets_4k_with_preamble.jsonl")
    ap.add_argument("--repeat", type=int, default=4,
                    help="repeat factor for the new preamble crops in the combined file")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--max-crops-per-doc", type=int, default=3)
    args = ap.parse_args(argv)

    global _TOK
    _TOK = AutoTokenizer.from_pretrained(args.tokenizer)

    # schema cost = encoded length of a 1-char state minus that state's own tokens
    probe_fields = load(args.ledgar if Path(args.ledgar).exists() else args.cuad)[0]["fields"]
    probe = encode_record(_TOK, {"state": "x", "fields": probe_fields, "answers": {}},
                          max_length=10**6)
    fixed = len(probe.input_ids) - len(_TOK("x", add_special_tokens=False)["input_ids"])
    allowed = 4096 - fixed - 1
    print(f"[budget] schema+prompt {fixed} tokens -> state budget {allowed} at 4096")

    new: list[dict] = []
    stats = Counter()
    for path, label in ((args.ledgar, "ledgar"), (args.cuad, "cuad")):
        if not Path(path).exists():
            print(f"[skip] {path} missing")
            continue
        for rec in load(path):
            fields = rec["fields"]
            state = rec["state"]
            enc = _TOK(state, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            made = 0
            for name in PREAMBLE:
                a = (rec.get("answers") or {}).get(name)
                if not a:
                    continue
                pairs = answer_pairs(a)
                if not pairs:
                    continue
                v, (s, e) = pairs[0]
                crop = make_crop(rec, state, offsets, (s, e), allowed, fields)
                if crop is None:
                    stats[f"{label}:{name}:failed"] += 1
                    continue
                crop["id"] = f"{rec['id']}-{name}-pre"
                crop["doc_id"] = rec["id"]
                new.append(crop)
                made += 1
                stats[f"{label}:{name}"] += 1
                if made >= args.max_crops_per_doc:
                    break

    # name/date/parties sit together, so two fields often produce the SAME window;
    # keeping both would silently multiply that document's weight.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in new:
        key = (r["doc_id"], hashlib.sha1(r["state"].encode()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    print(f"crops before dedupe {len(new)} -> unique windows {len(unique)}")
    new = unique

    Path(args.out_new).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_new, "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    print(f"new preamble crops: {len(new)} -> {args.out_new}")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")

    if args.repeat and Path(args.existing).exists():
        existing = load(args.existing)
        with open(args.out_combined, "w") as f:
            for r in existing:
                f.write(json.dumps(r) + "\n")
            for _ in range(args.repeat):
                for r in new:
                    f.write(json.dumps(r) + "\n")
        total = len(existing) + len(new) * args.repeat
        share = len(new) * args.repeat / total
        print(f"combined: {len(existing)} existing + {len(new)}x{args.repeat} new = {total} "
              f"({share:.0%} preamble) -> {args.out_combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
