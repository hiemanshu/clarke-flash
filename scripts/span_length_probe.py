"""Predicted vs gold span lengths at different tie-break epsilons.

Answers one question the metric trade-off only implies: is the highest-scoring span
(the argmax the model actually prefers) systematically LONGER than the annotated span,
and does the "shortest within eps" tie-break really trim it?

Decodes the same records twice (eps=0 -> raw argmax, eps>0 -> tie-break) and pairs each
gold span with the predicted span that overlaps it most. For every matched pair we record

    ratio = predicted_chars / gold_chars

so ratio > 1 means over-extension past the gold end, < 1 means the prediction is short.

Reported per epsilon: matched coverage, median/mean ratio, share over/under by >10%, and
how often the prediction strictly contains (or is contained by) the gold span. Span token
lengths go through the encoder's own char->token map, so they reflect what the pointer
actually sees.

Usage (the model is 2B at 65k; cap VRAM if a training run is live):
    uv run python scripts/span_length_probe.py --head runs/A-qwen2b/checkpoint-1880 \
        --data data/doc_eval_40.jsonl --limit 40 --max-length 65536 --agg-atoms \
        --eps 0 0.25 --out reports/A-qwen2b-span-lengths.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from clarke.data import RecordDataset
from clarke.encoding import collate_records, encode_record, token_span_for_chars
from clarke.model import ClarkeModel, load_backbone, load_head


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = _overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, int(p * len(values)))]
    return {"p10": q(0.10), "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "p90": q(0.90), "mean": statistics.fmean(values),
            "max": values[-1], "n": len(values)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--agg-atoms", dest="agg_atoms", action="store_true")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.25],
                        help="tie-break epsilons to compare (0 = raw argmax)")
    parser.add_argument("--vram-fraction", type=float, default=0.40,
                        help="per-process VRAM cap so a live training run is not starved")
    parser.add_argument("--span-caps", default=None,
                        help="per-field cap JSON (scripts/field_span_caps.py)")
    parser.add_argument("--span-cap-key", dest="span_cap_key", default="p99")
    parser.add_argument("--span-priors", dest="span_priors", default=None,
                        help="per-field gold length histogram JSON (--hist-out)")
    parser.add_argument("--span-prior-alpha", dest="span_prior_alpha", type=float, default=0.0)
    parser.add_argument("--dump-pairs", dest="dump_pairs", action="store_true",
                        help="include the raw matched pairs (field, lengths, ratio, iou)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg: dict[str, Any] = {}
    try:
        cfg = json.loads((Path(args.head) / "extraction_head_config.json").read_text())
    except Exception:
        pass
    backbone_id = args.backbone or cfg.get("backbone") or "Qwen/Qwen3.5-9B"
    print(f"[probe] head={args.head} backbone={backbone_id} eps={args.eps}", flush=True)

    backbone, processor = load_backbone(backbone_id, device=str(device))
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = load_head(Path(args.head), device=str(device), dtype=torch.float32)
    model = ClarkeModel(backbone, head).eval()

    span_caps = None
    if args.span_caps:
        table = json.loads(Path(args.span_caps).read_text())
        span_caps = {k: int(v[args.span_cap_key]) for k, v in table.items()
                     if isinstance(v, dict) and args.span_cap_key in v}

    length_priors = None
    if args.span_priors and args.span_prior_alpha > 0:
        table = json.loads(Path(args.span_priors).read_text())
        length_priors = {}
        for name, row in table.items():
            counts = torch.tensor([float(c) + 1.0 for c in row["hist"]])
            length_priors[name] = counts / counts.sum()

    tokenizer = processor.tokenizer
    records = RecordDataset([Path(args.data)]).records[: args.limit]

    # one entry per matched (gold, pred) span pair, per epsilon
    pairs: dict[float, list[dict[str, Any]]] = {eps: [] for eps in args.eps}
    fields_seen = 0
    fields_with_gold = 0
    fields_matched = {eps: 0 for eps in args.eps}
    fields_matched_iou50 = {eps: 0 for eps in args.eps}
    per_field: dict[str, dict[float, list[float]]] = {}

    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        encoded = []
        for record in chunk:
            try:
                encoded.append(encode_record(tokenizer, record, max_length=args.max_length,
                                             agg_atoms=args.agg_atoms))
            except ValueError:
                continue
        if not encoded:
            continue
        batch = collate_records(encoded, tokenizer.pad_token_id, device)

        answers_by_eps = {}
        with torch.inference_mode():
            for eps in args.eps:
                answers_by_eps[eps] = model.answer(batch, null_threshold=0.5,
                                                   slot_threshold=0.5,
                                                   span_tiebreak_eps=eps,
                                                   span_caps=span_caps,
                                                   length_priors=length_priors,
                                                   length_prior_alpha=args.span_prior_alpha)
        torch.cuda.empty_cache()

        # answers_by_eps[eps][idx] is position-aligned with the encoded records
        for idx, enc in enumerate(encoded):
            for enc_field in enc.fields:
                name = enc_field.name
                gold = enc.answers.get(name) or {}
                gold_spans = [tuple(s) for s in (gold.get("spans") or []) if s]
                spec = enc_field.spec
                if not (spec.type == "list" and spec.items == "text"):
                    continue
                if gold_spans:
                    fields_with_gold += 1
                for eps in args.eps:
                    got = answers_by_eps[eps][idx][name]
                    pred_spans = [tuple(s) for s in (got.get("spans") or []) if s]
                    if not gold_spans or not pred_spans:
                        continue
                    best = {}
                    for g in gold_spans:
                        scores = [_iou(g, p) for p in pred_spans]
                        k = max(range(len(scores)), key=lambda i: scores[i])
                        if scores[k] <= 0:
                            continue
                        p = pred_spans[k]
                        gt = token_span_for_chars(enc, g[0], g[1])
                        pt = token_span_for_chars(enc, p[0], p[1])
                        best[g] = {
                            "gold_chars": g[1] - g[0], "pred_chars": p[1] - p[0],
                            "gold_tokens": (gt[1] - gt[0] + 1) if gt else None,
                            "pred_tokens": (pt[1] - pt[0] + 1) if pt else None,
                            "gold_start_tok": gt[0] if gt else None,
                            "gold_end_tok": gt[1] if gt else None,
                            "pred_start_tok": pt[0] if pt else None,
                            "pred_end_tok": pt[1] if pt else None,
                            "iou": scores[k],
                            "contains_gold": p[0] <= g[0] and p[1] >= g[1],
                            "contained_by_gold": p[0] >= g[0] and p[1] <= g[1],
                            "same_boundaries": p[0] == g[0] and p[1] == g[1],
                            "ratio": (p[1] - p[0]) / max(1, g[1] - g[0]),
                            "hit_cap": bool(span_caps and name in span_caps
                                            and pt and (pt[1] - pt[0] + 1) >= span_caps[name]),
                            "gold_ratio": (g[1] - g[0]) / max(1, p[1] - p[0]),
                        }
                    if not best:
                        continue
                    fields_matched[eps] += 1
                    if max(v["iou"] for v in best.values()) >= 0.5:
                        fields_matched_iou50[eps] += 1
                    for g, v in best.items():
                        v = dict(v, field=name)
                        pairs[eps].append(v)
                        per_field.setdefault(name, {e: [] for e in args.eps})[eps].append(v["ratio"])

    report: dict[str, Any] = {
        "checkpoint": args.head, "data": args.data, "records": len(records),
        "max_length": args.max_length, "eps": args.eps,
        "fields_with_gold": fields_with_gold,
        "summary": {},
        "by_field": {},
    }
    for eps in args.eps:
        ps = pairs[eps]
        if not ps:
            report["summary"][str(eps)] = {"n": 0}
            continue
        ratios = [p["ratio"] for p in ps]
        tok_ratios = [p["pred_tokens"] / p["gold_tokens"] for p in ps
                      if p["pred_tokens"] and p["gold_tokens"]]
        report["summary"][str(eps)] = {
            "matched_pairs": len(ps),
            "fields_matched": fields_matched[eps],
            "fields_matched_iou50": fields_matched_iou50[eps],
            "ratio_chars": _quantiles(ratios),
            "ratio_tokens": _quantiles(tok_ratios),
            "over_10pct": sum(1 for r in ratios if r > 1.10) / len(ratios),
            "over_50pct": sum(1 for r in ratios if r > 1.50) / len(ratios),
            "under_10pct": sum(1 for r in ratios if r < 0.90) / len(ratios),
            "within_10pct": sum(1 for r in ratios if 0.90 <= r <= 1.10) / len(ratios),
            "contains_gold": sum(1 for p in ps if p["contains_gold"]) / len(ps),
            "contained_by_gold": sum(1 for p in ps if p["contained_by_gold"]) / len(ps),
            "same_boundaries": sum(1 for p in ps if p["same_boundaries"]) / len(ps),
            "mean_iou": statistics.fmean(p["iou"] for p in ps),
            "mean_excess_chars": statistics.fmean(p["pred_chars"] - p["gold_chars"] for p in ps),
            "median_gold_chars": statistics.median(p["gold_chars"] for p in ps),
            "median_pred_chars": statistics.median(p["pred_chars"] for p in ps),
        }
        print(f"[{eps}] n={len(ps)} median ratio {statistics.median(ratios):.3f} "
              f"mean {statistics.fmean(ratios):.3f} over10% "
              f"{report['summary'][str(eps)]['over_10pct']:.1%} contains_gold "
              f"{report['summary'][str(eps)]['contains_gold']:.1%}", flush=True)

    if args.dump_pairs:
        report["pairs"] = {str(e): v for e, v in pairs.items()}
    for name, by_eps in sorted(per_field.items()):
        rows = {str(e): {"median_ratio": statistics.median(v), "n": len(v),
                         "share_over_10pct": sum(1 for r in v if r > 1.10) / len(v)}
                for e, v in by_eps.items() if v}
        if rows:
            report["by_field"][name] = rows

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
