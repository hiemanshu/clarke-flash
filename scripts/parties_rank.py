"""Parties: is the gold span RANKED (assignment problem) or never seen (retrieval problem)?

For every gold party span, look at each slot's start/end distributions and ask where the
gold start/end token ranks. If the gold start is inside the top-k of ANY slot, the model
can see the party and the failure is assignment/selection (fixable with a set-aware loss).
If it is ranked deep everywhere, the model does not represent those mentions and the fix
has to be data or a candidate generator.

Reported: rank distribution of the gold start token per gold item (best slot), the rate at
which the gold start is in the top-1/3/5/10, and the same for the gold end token.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from clarke.data import RecordDataset
from clarke.encoding import collate_records, encode_record, token_span_for_chars
from clarke.model import ClarkeModel, load_backbone, load_head


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

    start_ranks, end_ranks = [], []
    per_doc = []
    for record in records:
        try:
            enc = encode_record(tokenizer, record, max_length=args.max_length, agg_atoms=True)
        except ValueError:
            continue
        answer = enc.answers.get(args.field) or {}
        gold_spans = [tuple(s) for s in (answer.get("spans") or [])]
        idx = next((i for i, f in enumerate(enc.fields) if f.name == args.field), None)
        if idx is None or not gold_spans:
            continue
        batch = collate_records([enc], tokenizer.pad_token_id, device)
        with torch.inference_mode():
            logits = model.forward(batch)[0][idx]
        starts = torch.softmax(logits.start.float(), dim=-1)      # [slots, tokens]
        ends = torch.softmax(logits.end.float(), dim=-1)
        # greedily assign each gold item to its best-ranked slot
        doc_ranks = []
        for g in gold_spans:
            pair = token_span_for_chars(enc, g[0], g[1])
            if pair is None:
                continue
            gs, ge = pair
            sr = [(int((starts[s] > starts[s][gs]).sum()), s) for s in range(starts.shape[0])]
            er = [(int((ends[s] > ends[s][ge]).sum()), s) for s in range(ends.shape[0])]
            best_s = min(sr)[0]
            best_e = min(er)[0]
            start_ranks.append(best_s)
            end_ranks.append(best_e)
            doc_ranks.append((best_s, best_e))
        per_doc.append(doc_ranks)

    def summary(ranks, name):
        c = Counter(min(r, 50) for r in ranks)
        top = lambda k: sum(1 for r in ranks if r < k) / max(len(ranks), 1)
        print(f"{name}: n={len(ranks)}  top1 {top(1):.1%}  top3 {top(3):.1%}  "
              f"top5 {top(5):.1%}  top10 {top(10):.1%}  top20 {top(20):.1%}")
        return {"n": len(ranks), "top1": top(1), "top3": top(3), "top5": top(5),
                "top10": top(10), "top20": top(20)}

    out = {"checkpoint": args.head, "field": args.field, "docs": len(per_doc)}
    out["start"] = summary(start_ranks, "gold START token rank (best slot)")
    out["end"] = summary(end_ranks, "gold END   token rank (best slot)")
    # how many items would a slot-top-5 union recover?
    both = sum(1 for s, e in zip(start_ranks, end_ranks) if s < 5 and e < 5)
    print(f"gold item fully inside some slot's top-5 start AND end: {both}/{len(start_ranks)} "
          f"({both/max(len(start_ranks),1):.1%})")
    out["top5_both"] = both / max(len(start_ranks), 1)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
