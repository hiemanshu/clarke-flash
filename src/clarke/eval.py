"""Evaluate a clarke-flash checkpoint: accuracy, null handling, calibration.

Decodes real answers in one pass per batch and compares with gold labels:

- text/list        grounded span match (value found at its span) + value match
- number/date/...  value exact match at declared precision, null F1
- estimate/forecast median error, 80%-interval coverage, pinball loss
- confidence       reliability (accuracy within confidence buckets)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import RecordDataset
from .encoding import collate_records, encode_record
from .model import ClarkeModel, load_backbone, load_head
from .schema import FieldSpec


def match_scalar(spec: FieldSpec, predicted: Any, gold: Any) -> bool:
    if spec.type in ("number", "duration"):
        if predicted is None or gold is None:
            return predicted is None and gold is None
        return round(float(predicted), spec.decimals) == round(float(gold), spec.decimals)
    return str(predicted) == str(gold)


@torch.no_grad()
def evaluate(records: list[dict[str, Any]], model: ClarkeModel, tokenizer: Any,
             device: torch.device, batch_size: int = 8,
             agg_atoms: bool = False) -> dict[str, Any]:
    stats: dict[tuple[str, str], dict[str, float]] = {}

    def stat(dom: str, kind: str) -> dict[str, float]:
        return stats.setdefault((dom, kind), {"n": 0, "exact": 0, "located": 0, "null_correct": 0,
                                       "answered": 0, "conf_sum": 0.0, "coverage": 0,
                                       "ape_sum": 0.0, "pinball": 0.0,
                                       "brier_sum": 0.0, "brier_n": 0,
                                       "solved": 0, "exact_solvable": 0})

    encoded_records = []
    for record in records:
        try:
            encoded_records.append(encode_record(tokenizer, record, agg_atoms=agg_atoms))
        except ValueError:
            continue

    for start in range(0, len(encoded_records), batch_size):
        batch_records = encoded_records[start:start + batch_size]
        batch = collate_records(batch_records, tokenizer.pad_token_id, device)
        answers = model.answer(batch)
        for record, record_answers in zip(batch_records, answers):
            for name, encoded in zip(record_answers, record.fields):
                spec = encoded.spec
                gold = record.answers.get(name) or {}
                got = record_answers[name]
                kind = spec.type if spec.type != "list" else f"list-{spec.items}"
                bucket = stat(record.domain or "?", kind)
                bucket["n"] += 1
                gold_null = gold.get("value") is None and gold.get("values") is None
                got_null = got.get("value") is None and got.get("values") is None
                if isinstance(got.get("p_null"), (int, float)):
                    bucket["brier_sum"] += (float(got["p_null"]) - (1.0 if gold_null else 0.0)) ** 2
                    bucket["brier_n"] += 1
                if gold_null:
                    if got_null:
                        bucket["null_correct"] += 1
                    bucket["null_n"] = bucket.get("null_n", 0) + 1
                elif got_null:
                    pass  # false null: miss, not counted as answered
                else:
                    bucket["answered"] += 1
                    if spec.type in ("number", "duration", "date", "time", "datetime"):
                        from .candidates import field_atoms, solve

                        try:
                            if spec.type in ("number", "duration"):
                                s = solve(spec, record.pool, float(gold["value"]),
                                         spec.decimals) is not None
                            else:
                                s = any(str(v) == str(gold["value"])
                                        for v, _, _ in field_atoms(spec, record.pool))
                        except (TypeError, ValueError):
                            s = False
                        bucket["solved"] += int(s)
                        if s and match_scalar(spec, got.get("value"), gold.get("value")):
                            bucket["exact_solvable"] += 1
                    if spec.type == "text":
                        value = got.get("value")
                        span = got.get("span") or [0, 0]
                        gspan = gold.get("span") or [0, 0]
                        overlap = max(0, min(span[1], gspan[1]) - max(span[0], gspan[0]))
                        union = max(span[1], gspan[1]) - min(span[0], gspan[0]) or 1
                        bucket["located"] += int(overlap / union >= 0.5)
                        bucket["exact"] += int(str(value or "").strip() == str(gold["value"]).strip())
                    elif spec.type == "list" and spec.items == "text":
                        gold_set = {str(v).strip() for v in gold["value"]}
                        got_set = {str(v).strip() for v in (got.get("value") or [])}
                        union = len(gold_set | got_set) or 1
                        bucket["exact"] += len(gold_set & got_set) / union
                        bucket["located"] += len(gold_set & got_set) / union
                    elif spec.type in ("estimate", "forecast"):
                        golds = gold.get("values") or [gold.get("value")]
                        meds = got.get("median") if isinstance(got.get("median"), list) else [got.get("median")]
                        lows = got.get("p10") if isinstance(got.get("p10"), list) else [got.get("p10")]
                        highs = got.get("p90") if isinstance(got.get("p90"), list) else [got.get("p90")]
                        for truth, median, low, high in zip(golds, meds, lows, highs):
                            if truth is None or median is None:
                                continue
                            bucket["ape_sum"] += abs(median - truth) / max(abs(truth), 1e-6)
                            bucket["coverage"] += int(low <= truth <= high)
                            q = 0.1
                            bucket["pinball"] += max(q * (truth - low), (q - 1) * (truth - low)) \
                                + max((1 - q) * (truth - high), -q * (truth - high))
                        bucket["steps"] = bucket.get("steps", 0) + len(golds)
                    else:
                        if spec.type == "list":
                            pairs = zip(got.get("value") or [], gold["value"])
                            ok = all(match_scalar(spec, p, g) for p, g in pairs) \
                                and len(got.get("value") or []) == len(gold["value"])
                        else:
                            ok = match_scalar(spec, got.get("value"), gold.get("value"))
                        bucket["exact"] += int(ok)
                    if isinstance(got.get("confidence"), (int, float)):
                        bucket["conf_sum"] += float(got["confidence"])
    def fmt(b: dict) -> dict[str, Any]:
        answered = b["answered"]
        out: dict[str, Any] = {
            "n": int(b["n"]),
            "exact": round(b["exact"] / max(answered, 1), 4),
            "located": round(b["located"] / max(answered, 1), 4) if b["located"] else None,
            "null_correct": round(b["null_correct"] / max(b.get("null_n", 0), 1), 4),
            "mean_confidence": round(b["conf_sum"] / max(answered, 1), 4),
            "brier_null": round(b["brier_sum"] / max(b["brier_n"], 1), 4),
        }
        if b["solved"]:
            out["solvable_rate"] = round(b["solved"] / max(answered, 1), 4)
            out["exact_when_solvable"] = round(b["exact_solvable"] / b["solved"], 4)
        if b["ape_sum"]:
            steps = max(b.get("steps", 0), 1)
            out["median_ape"] = round(b["ape_sum"] / steps, 4)
            out["coverage_80"] = round(b["coverage"] / steps, 4)
        return out

    def rollup(match) -> dict:
        from collections import defaultdict
        agg = defaultdict(float)
        for (dom, kind), b in stats.items():
            if match(dom, kind):
                for key, val in b.items():
                    agg[key] += val
        return agg

    kinds = sorted({kind for (_, kind) in stats})
    overall = {kind: rollup(lambda d, k, kind=kind: k == kind) for kind in kinds}
    overall = {kind: fmt(agg) for kind, agg in overall.items()}
    domains = sorted({dom for (dom, _) in stats})
    by_domain = {}
    for dom in domains:
        dk = sorted({k for (d, k) in stats if d == dom})
        by_domain[dom] = {k: fmt(rollup(lambda d, kk, dom=dom, k=k: d == dom and kk == k))
                          for k in dk}
    return {"overall": overall, "by_domain": by_domain}


@torch.no_grad()
def postmortem(records: list[dict[str, Any]], model: ClarkeModel, tokenizer: Any,
               device: torch.device, batch_size: int = 4, max_records: int = 96,
               agg_atoms: bool = False) -> dict:
    """Ranking diagnostics: where does the GOLD atom/token rank in the head's
    own scores? Decides which surgery the failure needs:
      - p@1 decent but exact low          -> decode/render issue, not the pointer
      - p@5 high, p@1 low                 -> ranking failure: Tier B reranker
      - gold rarely in top-5              -> query/representation failure: Tier A/D
    """
    from .head import JointExtractionHead

    encoded = []
    for r in records[:max_records]:
        try:
            encoded.append(encode_record(tokenizer, r, agg_atoms=agg_atoms))
        except ValueError:
            continue

    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def bucket(dom: str, kind: str) -> dict:
        return agg.setdefault((dom, kind), {"n": 0, "p1": 0, "p3": 0, "p5": 0,
                                            "ranks": [], "unrankable": 0})

    def rank_of(scores: torch.Tensor, idx: int) -> int:
        s = scores.float()
        return int((s > s[min(idx, s.shape[0] - 1)]).sum())

    for start in range(0, len(encoded), batch_size):
        batch_records = encoded[start:start + batch_size]
        batch = collate_records(batch_records, tokenizer.pad_token_id, device)
        outputs = model.forward(batch)
        for record, record_logits in zip(batch_records, outputs):
            targets = JointExtractionHead.build_targets(record)
            for encoded_field, lg, tg in zip(record.fields, record_logits, targets):
                spec = encoded_field.spec
                kind = spec.type if spec.type != "list" else f"list-{spec.items}"
                b = bucket(record.domain or "?", kind)
                rank = None
                if spec.type == "text" and lg.start is not None and tg.start is not None:
                    rank = rank_of(lg.start, tg.start)
                elif spec.type == "list" and spec.items == "text" \
                        and lg.start is not None and tg.spans:
                    rank = rank_of(lg.start[0], tg.spans[0][0])
                elif spec.type == "list" and lg.cand_a is not None \
                        and any(c >= 0 for c in tg.cands or []):
                    idx = next(c for c in tg.cands if c >= 0)
                    rank = rank_of(lg.cand_a[0], idx)
                elif tg.cand is not None and lg.cand_a is not None:
                    rank = rank_of(lg.cand_a, tg.cand[0])
                elif spec.type in ("estimate", "forecast") and lg.dist is not None and tg.dist:
                    rank = rank_of(lg.dist[0], tg.dist[0])
                if rank is None:
                    b["unrankable"] += 1
                else:
                    b["n"] += 1
                    b["p1"] += int(rank < 1)
                    b["p3"] += int(rank < 3)
                    b["p5"] += int(rank < 5)
                    b["ranks"].append(rank)

    result: dict = {}
    for (dom, kind), b in sorted(agg.items()):
        n = max(b["n"], 1)
        entry = {"scored": b["n"], "unrankable": b["unrankable"],
                 "p@1": round(b["p1"] / n, 3), "p@3": round(b["p3"] / n, 3),
                 "p@5": round(b["p5"] / n, 3),
                 "median_rank": int(sorted(b["ranks"])[len(b["ranks"]) // 2]) if b["ranks"] else None}
        result.setdefault(dom, {})[kind] = entry
    return result


def explode_field_groups(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Split each record into passes of <= size schema fields over the SAME state.

    Diagnoses whether CUAD's pointing failure is 41-queries-competing through the
    field decoder (vs long-document attention dilution): if grouped passes rank
    the gold, the fix is serving-side (batch fields smaller), not retraining.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        names = list(rec["fields"])
        if len(names) <= size:
            out.append(rec)
            continue
        for i in range(0, len(names), size):
            chunk = names[i:i + size]
            sub = dict(rec)
            sub["fields"] = {k: rec["fields"][k] for k in chunk}
            answers = rec.get("answers") or {}
            sub["answers"] = {k: (answers.get(k) or {"value": None}) for k in chunk}
            sub["id"] = f"{rec['id']}#g{i // size}"
            out.append(sub)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a clarke-flash checkpoint")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--head", required=True, help="checkpoint dir with extraction_head.safetensors")
    parser.add_argument("--data", required=True, help="records JSONL to evaluate")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4,
                        help="records per forward (long CUAD windows need <=4)")
    parser.add_argument("--out", default=None, help="write JSON report here")
    parser.add_argument("--postmortem", action="store_true",
                        help="gold-rank diagnostics (p@1/3/5 by domain+kind) instead of scored eval")
    parser.add_argument("--agg-atoms", dest="agg_atoms", action="store_true",
                        help="virtual CSV column-aggregate atoms (must match training)")
    parser.add_argument("--field-groups", dest="field_groups", type=int, default=0,
                        help="serving-strategy diagnostic: evaluate in passes of N schema fields")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, processor = load_backbone(args.backbone, device=str(device))
    head = load_head(Path(args.head), device=str(device), dtype=torch.float32)
    model = ClarkeModel(backbone, head).eval()

    dataset = RecordDataset([Path(args.data)])
    records = dataset.records[: args.limit]
    if args.field_groups:
        records = explode_field_groups(records, args.field_groups)
    if args.postmortem:
        report = {"checkpoint": args.head, "data": args.data, "records": len(records),
                  "postmortem": postmortem(records, model, processor.tokenizer, device,
                                           batch_size=args.batch_size, agg_atoms=args.agg_atoms,
                                           max_records=len(records))}
    else:
        summary = evaluate(records, model, processor.tokenizer, device,
                           batch_size=args.batch_size, agg_atoms=args.agg_atoms)
        report = {"checkpoint": args.head, "data": args.data, "records": len(records),
                  "metrics": summary["overall"], "by_domain": summary["by_domain"]}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
