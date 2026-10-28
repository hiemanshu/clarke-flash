#!/usr/bin/env python3
"""Why does the head miss gold fields on full contracts?

For every gold-present field, decode with the null gate disabled (threshold 1.01) so the
span is always produced, then classify:

  ok_reported    p_null <= 0.5 and the span matches gold      -> working
  ok_rejected    span matches gold but p_null > 0.5           -> calibration loss
  wrong_reported answered but the span is wrong               -> localisation loss
  wrong_rejected p_null > 0.5 and the span is also wrong      -> true miss

That split decides the fix: calibration losses are a threshold/prior fix, localisation
losses are an attention/context fix, and true misses are a data fix.

    uv run python scripts/doc_diagnose.py --head runs/scale-2b/checkpoint-8384 \
        --data data/doc_eval_40.jsonl --max-length 65536
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from clarke.encoding import collate_records, encode_record
from clarke.model import ClarkeModel, load_backbone, load_head
from clarke.textmetrics import as_list, norm


def matches(gold: object, got: object) -> bool:
    g = {norm(v) for v in as_list(gold)}
    p = {norm(v) for v in as_list(got)}
    return bool(g) and g == p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", default="data/doc_eval_40.jsonl")
    ap.add_argument("--max-length", dest="max_length", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    backbone, processor = load_backbone("Qwen/Qwen3.5-2B", device="cuda")
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = load_head(Path(args.head), device="cuda", dtype=torch.float32)
    model = ClarkeModel(backbone, head).eval()

    records = [json.loads(l) for l in open(args.data)][: args.limit]
    dec: dict = {}
    cfg_path = Path(args.head) / "decode_config.json"
    if cfg_path.exists():
        dec = json.loads(cfg_path.read_text())
        print(f"[decode] using baked config from {cfg_path}")
    priors = None
    if dec.get("span_priors"):
        priors = {}
        for name, counts in dec["span_priors"].items():
            t = torch.tensor([float(c) + 1.0 for c in counts])
            priors[name] = t / t.sum()
        print(f"[decode] embedded length priors for {len(priors)} fields")
    tally = Counter()
    tally_field: dict[str, Counter] = {}
    pnulls: list[float] = []
    rows: list[dict] = []

    encoded = []
    for r in records:
        try:
            encoded.append(encode_record(processor.tokenizer, r, max_length=args.max_length,
                                         agg_atoms=True))
        except ValueError:
            continue

    for start in range(0, len(encoded), args.batch_size):
        chunk = encoded[start:start + args.batch_size]
        batch = collate_records(chunk, processor.tokenizer.pad_token_id, "cuda")
        with torch.inference_mode():
            answers = model.answer(
                batch,
                null_threshold=1.01,                                # never gate on p_null
                slot_threshold=dec.get("slot_threshold", 0.5),
                slot_mask=dec.get("slot_mask", False),
                span_tiebreak_eps=dec.get("span_tiebreak_eps", 0.0),
                length_priors=priors,
                length_prior_alpha=dec.get("span_prior_alpha", 0.0),
                candidate_decode=dec.get("candidate_decode", False),
                candidate_fields=set(dec.get("candidate_fields") or []),
                candidate_extra=dec.get("candidate_extra", 0),
                candidate_ranker=dec.get("candidate_ranker"),
                ranker_alpha=dec.get("ranker_alpha", 1.0),
            )
        for enc, rec, ans in zip(chunk, records[start:start + len(chunk)], answers):
            for field, spec in rec["fields"].items():
                gold = rec["answers"].get(field) or {}
                if gold.get("value") is None:
                    continue
                got = ans.get(field) or {}
                pn = float(got.get("p_null", 1.0))
                ok = matches(gold.get("value"), got.get("value"))
                nulled = pn > 0.5
                if ok and not nulled:
                    tally["ok_reported"] += 1
                elif ok and nulled:
                    tally["ok_rejected"] += 1
                elif not ok and not nulled:
                    tally["wrong_reported"] += 1
                else:
                    tally["wrong_rejected"] += 1
                pnulls.append(pn)
                cat = ("ok_reported" if ok and not nulled else "ok_rejected" if ok
                       else "wrong_reported" if not nulled else "wrong_rejected")
                tally_field.setdefault(field, Counter())[cat] += 1
                rows.append({"id": rec["id"], "field": field, "p_null": round(pn, 3),
                             "correct": ok, "nulled": nulled,
                             "gold": [str(v)[:120] for v in as_list(gold.get("value"))][:4],
                             "pred": [str(v)[:120] for v in as_list(got.get("value"))][:4]})
        torch.cuda.empty_cache()

    total = sum(tally.values())
    print(f"gold-present fields: {total}")
    for k in ("ok_reported", "ok_rejected", "wrong_reported", "wrong_rejected"):
        print(f"  {k:16s} {tally[k]:4d}  ({100 * tally[k] / max(total, 1):5.1f}%)")
    if pnulls:
        s = sorted(pnulls)
        q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
        print(f"p_null on gold-present fields: p10 {q(.1):.3f}  p50 {q(.5):.3f}  "
              f"p90 {q(.9):.3f}  |  share > 0.9: {sum(1 for x in s if x > 0.9) / len(s):.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tally": tally, "by_field": {k: dict(v) for k, v in tally_field.items()},
             "rows": rows}, indent=1))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
