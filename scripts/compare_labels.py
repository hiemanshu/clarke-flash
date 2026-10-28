#!/usr/bin/env python3
"""Compare LLM label files against a gold label file, field by field.

Gold and predictions share the output shape of scripts/llm_label_preambles.py:
`{"id": ..., "fields": {"document_name": {"value": [...]}, "agreement_date":
{"value": [...]}, "parties": {"value": [...], "spans": [...]}}}`. A prediction
row may instead be `{"id": ..., "error": ...}`; those count as misses.

Parties are compared as normalised sets (the same `norm` the eval scorer uses),
because the question is whether the same *entities* were found, not whether the
strings are byte-identical. Names and dates are compared as normalised strings,
with a `contains` column for names since "AGREEMENT" vs "ADMINISTRATION
AGREEMENT" is a boundary difference, not a wrong title.

    uv run python scripts/compare_labels.py \
        --gold data/llm_preambles.jsonl \
        --pred /tmp/opencode/pilot10_cc_k3.jsonl --detail
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clarke.textmetrics import norm


def load(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row
    return out


def vals(row: dict, field: str) -> list[str]:
    f = (row.get("fields") or {}).get(field) or {}
    v = f.get("value")
    if v is None:
        return []
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def party_sets(gold: dict, pred: dict) -> tuple[set, set]:
    return ({norm(v) for v in vals(gold, "parties")},
            {norm(v) for v in vals(pred, "parties")})


def score_one(gold: dict, pred: dict) -> dict:
    gset, pset = party_sets(gold, pred)
    hit = len(gset & pset)
    prec = hit / len(pset) if pset else 0.0
    rec = hit / len(gset) if gset else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    gname = norm(vals(gold, "document_name")[0]) if vals(gold, "document_name") else ""
    pname = norm(vals(pred, "document_name")[0]) if vals(pred, "document_name") else ""
    gdate = norm(vals(gold, "agreement_date")[0]) if vals(gold, "agreement_date") else ""
    pdate = norm(vals(pred, "agreement_date")[0]) if vals(pred, "agreement_date") else ""
    return {
        "name_exact": bool(gname and pname and gname == pname),
        "name_contains": bool(gname and pname and (gname in pname or pname in gname)),
        "date_exact": bool(gdate and pdate and gdate == pdate),
        "party_p": prec, "party_r": rec, "party_f1": f1,
        "party_exact": gset == pset,
        "gold_parties": len(gset), "pred_parties": len(pset),
        "seconds": pred.get("seconds"),
        "error": "error" in pred,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/llm_preambles.jsonl")
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args(argv)

    gold_rows = load(args.gold)
    reports = []
    for path in args.pred:
        preds = load(path)
        ids = [i for i in gold_rows if i in preds]
        if not ids:
            print(f"{path}: no ids in common with gold")
            continue
        rows = [score_one(gold_rows[i], preds[i]) for i in ids]
        n = len(rows)
        agg = {
            "file": path, "n": n,
            "errors": sum(r["error"] for r in rows),
            "name_exact": sum(r["name_exact"] for r in rows) / n,
            "name_contains": sum(r["name_contains"] for r in rows) / n,
            "date_exact": sum(r["date_exact"] for r in rows) / n,
            "party_p": sum(r["party_p"] for r in rows) / n,
            "party_r": sum(r["party_r"] for r in rows) / n,
            "party_f1": sum(r["party_f1"] for r in rows) / n,
            "party_exact": sum(r["party_exact"] for r in rows) / n,
            "gold_parties": sum(r["gold_parties"] for r in rows) / n,
            "pred_parties": sum(r["pred_parties"] for r in rows) / n,
            "seconds": (sum(r["seconds"] for r in rows if r["seconds"]) /
                        max(sum(1 for r in rows if r["seconds"]), 1)),
        }
        reports.append(agg)
        if args.detail:
            print(f"\n{path}")
            for i in ids:
                r = score_one(gold_rows[i], preds[i])
                print(f"  {i[:44]:44s} name={'Y' if r['name_exact'] else 'n'} "
                      f"date={'Y' if r['date_exact'] else 'n'} "
                      f"partyF1={r['party_f1']:.2f} {r['pred_parties']}/{r['gold_parties']} "
                      f"{r['seconds']}s")

    print(f"\n{'file':44s} {'n':>3s} {'err':>3s} {'name':>6s} {'ncont':>6s} {'date':>6s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s} {'set':>6s} {'part/doc':>8s} {'sec':>5s}")
    for a in reports:
        print(f"{Path(a['file']).name:44s} {a['n']:3d} {a['errors']:3d} "
              f"{a['name_exact']:6.2f} {a['name_contains']:6.2f} {a['date_exact']:6.2f} "
              f"{a['party_p']:6.2f} {a['party_r']:6.2f} {a['party_f1']:6.2f} "
              f"{a['party_exact']:6.2f} {a['pred_parties']:4.1f}/{a['gold_parties']:<3.1f} "
              f"{a['seconds']:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
