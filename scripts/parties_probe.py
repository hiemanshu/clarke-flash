"""Why does `parties` return 1-3 items when every CUAD contract names 3-4?

One forward pass per document, then decode the parties field under several slot settings
(present-threshold, dedupe, slot-mask) so the cost of each gate is visible. Reports, per
setting: how many items are emitted, item-set equality with gold, recall/precision over
items, and how often a matched item also has the exact gold span.

The training target assigns gold items to slots in the order the annotations list them,
so if the model has learned a set rather than an order, the present head gates slots by
order statistics - which is exactly what a threshold sweep will reveal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from clarke.data import RecordDataset
from clarke.encoding import collate_records, encode_record, token_span_for_chars
from clarke.model import ClarkeModel, load_backbone, load_head


def norm(value: Any) -> str:
    return " ".join(str(value).split()).strip().lower()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data", default="data/doc_eval_40.jsonl")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--field", default="parties")
    parser.add_argument("--vram-fraction", type=float, default=0.35)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, processor = load_backbone(args.backbone, device=str(device))
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = load_head(Path(args.head), device=str(device), dtype=torch.float32)
    model = ClarkeModel(backbone, head).eval()
    tokenizer = processor.tokenizer
    records = RecordDataset([Path(args.data)]).records[: args.limit]

    configs = [
        ("slot .50 dedupe", dict(slot_threshold=0.50, dedupe=True, slot_mask=False)),
        ("slot .20 dedupe", dict(slot_threshold=0.20, dedupe=True, slot_mask=False)),
        ("slot .05 dedupe", dict(slot_threshold=0.05, dedupe=True, slot_mask=False)),
        ("slot .50 mask", dict(slot_threshold=0.50, dedupe=True, slot_mask=True)),
        ("pool k4", dict(slot_threshold=0.50, dedupe=True, slot_mask=False,
                         pool_decode=True, pool_starts=4)),
        ("pool k6", dict(slot_threshold=0.50, dedupe=True, slot_mask=False,
                         pool_decode=True, pool_starts=6)),
    ]
    res: dict[str, dict[str, Any]] = {name: {"counts": Counter(), "gold_hits": 0, "gold_total": 0,
                                             "pred_total": 0, "set_eq": 0, "crop_ok": 0,
                                             "matched": 0, "n": 0, "present": []}
                                      for name, _ in configs}

    for start in range(0, len(records), 1):
        record = records[start]
        try:
            enc = encode_record(tokenizer, record, max_length=args.max_length, agg_atoms=True)
        except ValueError:
            continue
        batch = collate_records([enc], tokenizer.pad_token_id, device)
        with torch.inference_mode():
            logits = model.forward(batch)[0]
        answer = (enc.answers.get(args.field) or {})
        gold_vals = [norm(v) for v in (answer.get("value") or [])]
        gold_spans = [tuple(s) for s in (answer.get("spans") or [])]
        idx = next((i for i, f in enumerate(enc.fields) if f.name == args.field), None)
        if idx is None:
            continue
        present = torch.sigmoid(logits[idx].present.float()).tolist()
        for name, cfg in configs:
            from clarke.model import decode_answer
            got = decode_answer(enc.fields[idx].spec, enc, logits[idx], null_threshold=0.5,
                                span_tiebreak_eps=0.25, **cfg)
            pred_vals = [norm(v) for v in (got.get("value") or [])]
            pred_spans = [tuple(s) for s in (got.get("spans") or [])]
            r = res[name]
            r["n"] += 1
            r["counts"][len(pred_vals)] += 1
            r["gold_total"] += len(gold_vals)
            r["pred_total"] += len(pred_vals)
            hits = set(gold_vals) & set(pred_vals)
            r["gold_hits"] += len(hits)
            if gold_vals and set(gold_vals) == set(pred_vals):
                r["set_eq"] += 1
            for v, span in zip(pred_vals, pred_spans):
                if v in set(gold_vals):
                    r["matched"] += 1
                    pair = token_span_for_chars(enc, span[0], span[1]) if span else None
                    g = gold_spans[gold_vals.index(v)] if v in gold_vals else None
                    gp = token_span_for_chars(enc, g[0], g[1]) if g else None
                    if pair and gp and pair == gp:
                        r["crop_ok"] += 1
        for name, _ in configs:
            res[name]["present"].append([round(p, 3) for p in present[:6]])

    out = {"checkpoint": args.head, "field": args.field, "n": len(records), "configs": {}}
    print(f"{'config':18s} {'items/doc':>10s} {'set-equal':>10s} {'recall':>7s} {'precision':>10s} "
          f"{'crop-exact':>10s}")
    for name, _ in configs:
        r = res[name]
        rec = r["gold_hits"] / max(r["gold_total"], 1)
        prec = r["gold_hits"] / max(r["pred_total"], 1)
        crop = r["crop_ok"] / max(r["matched"], 1)
        counts = dict(sorted(r["counts"].items()))
        out["configs"][name] = {"counts": counts, "set_eq": r["set_eq"], "n": r["n"],
                                "recall": rec, "precision": prec, "crop_exact": crop,
                                "items_per_doc": r["pred_total"] / max(r["n"], 1)}
        print(f"{name:18s} {r['pred_total']/max(r['n'],1):10.2f} "
              f"{r['set_eq']:4d}/{r['n']:<5d} {rec:7.3f} {prec:10.3f} {crop:10.3f}  {counts}")
    # mean present probability per slot index, first config only
    import statistics
    pres = res[configs[0][0]]["present"]
    if pres:
        per_slot = [statistics.fmean(p[i] for p in pres) for i in range(min(6, len(pres[0])))]
        print("\nmean present prob by slot:", [round(x, 3) for x in per_slot])
        out["mean_present_per_slot"] = per_slot
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
