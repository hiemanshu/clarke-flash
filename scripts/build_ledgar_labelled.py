#!/usr/bin/env python3
"""Join LLM-labelled preamble fields into the LEDGAR-derived contract records.

Inputs
  data/ledgar_split_cuad.jsonl   contracts that matched at least one LEDGAR clause
                                 (1.48 present fields/record on average)
  data/llm_preambles.jsonl       document_name / agreement_date / parties, grounded
                                 to verbatim spans by scripts/llm_label_preambles.py

Selection (owner rule, journal 22:55 / 23:05): a record qualifies when its label
row has `parties` AND `document_name`; `agreement_date` may be absent and is then
simply null. Partial rows that have only one of the two required fields are
dropped rather than null-padded, because a 2-present-field schema is the D2 shape
that taught the wrong schema-length mapping.

Every output record carries the FULL template schema with explicit nulls, so the
model sees the same 41-field shape it is evaluated on. A record with a raw join
would expose 1-4 fields, all present.

Checks before writing (all failures are fatal by default):
  - every answer span is inside the state and its text normalises to the value
    (the labeller grounds, but this re-verifies rather than trusting it)
  - no state overlaps a frozen eval surface (OPERATIONS rule 7): content hashes of
    the full state and of the first 1k chars, against every --audit file

    uv run python scripts/build_ledgar_labelled.py
    uv run python scripts/build_ledgar_labelled.py --audit data/doc_eval_40.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from clarke.textmetrics import norm

DEFAULT_AUDIT = [
    "data/doc_eval_40.jsonl", "data/doc_eval_40_twelve.jsonl",
    "data/real_eval.jsonl", "data/val3.jsonl", "data/typed_test.jsonl",
]
REQUIRED = ("parties", "document_name")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def h(text: str, n: int | None = None) -> str:
    s = norm(text)
    if n:
        s = s[:n]
    return hashlib.sha1(s.encode()).hexdigest()


def answer_pairs(a: dict) -> list[tuple[str, list[int]]]:
    """Normalise the several answer shapes used across the repo."""
    vals = a.get("value", a.get("values"))
    if vals is None:
        return []
    vals = vals if isinstance(vals, list) else [vals]
    spans = a.get("spans") or ([a["span"]] if a.get("span") else [])
    return [(str(v), s) for v, s in zip(vals, spans)]


def bad_spans(state: str, answers: dict) -> list[str]:
    out = []
    for name, a in answers.items():
        for value, span in answer_pairs(a):
            lo, hi = int(span[0]), int(span[1])
            if not (0 <= lo < hi <= len(state)):
                out.append(f"{name}: span {span} outside state ({len(state)})")
            elif norm(state[lo:hi]) != norm(value):
                out.append(f"{name}: state[{lo}:{hi}]={norm(state[lo:hi])[:40]!r} != {norm(value)[:40]!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgar", default="data/ledgar_split_cuad.jsonl")
    ap.add_argument("--labels", default="data/llm_preambles.jsonl")
    ap.add_argument("--schema-template", default="data/doc_train_470.jsonl")
    ap.add_argument("--twelve-template", default="data/doc_train_470_twelve.jsonl")
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="encoder budget per record including the schema prompt; 0 disables")
    ap.add_argument("--schema-tokens", type=int, default=2400,
                    help="tokens the 41-field schema prompt costs (measured ~2.4k)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out-41", default="data/ledgar_labelled_cuad.jsonl")
    ap.add_argument("--out-12", default="data/ledgar_labelled_twelve.jsonl")
    ap.add_argument("--audit", nargs="*", default=DEFAULT_AUDIT)
    ap.add_argument("--allow-leak", action="store_true",
                    help="write even if a content hash matches an eval surface")
    ap.add_argument("--allow-bad-spans", action="store_true",
                    help="write records whose spans do not re-verify")
    args = ap.parse_args(argv)

    fields41 = load_jsonl(args.schema_template)[0]["fields"]
    twelve_names = list(load_jsonl(args.twelve_template)[0]["fields"])
    missing = [n for n in twelve_names if n not in fields41]
    assert not missing, f"12-field names absent from the 41-field template: {missing}"
    fields12 = {n: fields41[n] for n in twelve_names}

    # last successful row per id (retries append; later rows supersede)
    labels: dict[str, dict] = {}
    for row in load_jsonl(args.labels):
        if "error" in row:
            continue
        labels[row["id"]] = row

    audit = []
    for path in args.audit:
        if not Path(path).exists():
            print(f"[audit] skip missing {path}")
            continue
        for r in load_jsonl(path):
            audit.append((path, h(r["state"]), h(r["state"], 1000)))
    audit_full = {a[1] for a in audit}
    audit_head = {a[2] for a in audit}
    print(f"[audit] {len(audit)} eval states loaded from {len(args.audit)} files")

    tok = None
    if args.max_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"[tokens] budget {args.max_tokens} - schema {args.schema_tokens} "
              f"= {args.max_tokens - args.schema_tokens} state tokens")
    token_counts: list[int] = []
    kept, dropped_partial, dropped_spans, dropped_long, leaks = [], 0, 0, 0, []
    per_field: Counter = Counter()
    present_hist: Counter = Counter()
    for rec in load_jsonl(args.ledgar):
        lab = labels.get(rec["id"])
        if not lab:
            continue
        if not all((lab.get("fields") or {}).get(k) for k in REQUIRED):
            dropped_partial += 1
            continue
        answers = {k: v for k, v in (rec.get("answers") or {}).items()
                   if (v or {}).get("value") is not None}
        for name, a in (lab.get("fields") or {}).items():
            if a and a.get("value"):
                answers[name] = {"value": list(a["value"]), "spans": [list(s) for s in a["spans"]]}

        if tok is not None:
            n_tok = len(tok(rec["state"], add_special_tokens=False)["input_ids"])
            token_counts.append(n_tok)
            if n_tok + args.schema_tokens > args.max_tokens:
                dropped_long += 1
                if dropped_long <= 5:
                    print(f"[tokens] drop {rec['id']}: {n_tok} state tokens + "
                          f"{args.schema_tokens} schema > {args.max_tokens} "
                          f"(spans past the cut would become false nulls)")
                continue

        problems = bad_spans(rec["state"], answers)
        if problems and not args.allow_bad_spans:
            dropped_spans += 1
            if dropped_spans <= 5:
                print(f"[span] drop {rec['id']}: {problems[0]}")
            continue

        full, head = h(rec["state"]), h(rec["state"], 1000)
        if full in audit_full or head in audit_head:
            leaks.append(rec["id"])
            if not args.allow_leak:
                continue

        base = {"id": rec["id"], "state": rec["state"],
                "domain": rec.get("domain", "contract"), "answers": answers}
        kept.append({**base, "fields": fields41})
        kept.append({**base, "fields": fields12, "id": rec["id"] + "#twelve"})
        for name in answers:
            per_field[name] += 1
        present_hist[len(answers)] += 1

    n41 = len(kept) // 2
    print(f"qualified from labels: {len(labels)} rows")
    if token_counts:
        s_ = sorted(token_counts)
        print(f"[tokens] state tokens min {s_[0]:,} median {s_[len(s_)//2]:,} max {s_[-1]:,}")
    print(f"dropped: {dropped_partial} without both of {REQUIRED}, {dropped_spans} with bad spans, "
          f"{dropped_long} too long for the encoder budget")
    print(f"leak matches: {len(leaks)}" + (f" (first: {leaks[:3]})" if leaks else ""))
    if leaks and not args.allow_leak:
        print("REFUSING to write: content matches an eval surface (use --allow-leak to override)")
        return 1
    print(f"kept: {n41} documents -> {len(kept)} rows (41-field + 12-field)")
    print(f"present fields/record: {dict(sorted(present_hist.items()))}")
    print("per-field instances:")
    for name, count in per_field.most_common():
        print(f"  {name:32s} {count}")
    print(f"mean present fields/record: "
          f"{sum(k*v for k,v in present_hist.items())/max(n41,1):.2f}")

    Path(args.out_41).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_41, "w") as f:
        for row in kept[::2]:
            f.write(json.dumps(row) + "\n")
    with open(args.out_12, "w") as f:
        for row in kept[1::2]:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.out_41} and {args.out_12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
