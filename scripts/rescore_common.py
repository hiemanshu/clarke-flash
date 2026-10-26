#!/usr/bin/env python3
"""Re-score external runs on the subset of records EVERY model completed.

A failed HTTP call yields an empty prediction, which the scorer counts as "correctly
null" - so scoring a partially failed run rewards it. This restricts every model to
the intersection of successful records and rewrites comparable reports.

    uv run python scripts/rescore_common.py reports/external_doc data/doc_eval_40.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_baseline as eb  # noqa: E402


def main(argv: list[str]) -> int:
    ext_dir = Path(argv[0] if argv else "reports/external_doc")
    data = argv[1] if len(argv) > 1 else "data/doc_eval_40.jsonl"

    records = {json.loads(l)["id"]: json.loads(l) for l in open(data) if l.strip()}
    preds: dict[str, dict[str, dict]] = {}
    for f in sorted(ext_dir.glob("*.preds.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        preds[f.stem.replace(".preds", "")] = {
            r["id"]: r["pred"] for r in rows if r.get("pred")}

    common = set.intersection(*(set(v) for v in preds.values())) if preds else set()
    out = ext_dir / "rescored"
    out.mkdir(exist_ok=True)
    gold = sum(1 for i in common for n in records[i]["fields"]
               if (records[i]["answers"].get(n) or {}).get("value") is not None)
    print(f"records per model: " + ", ".join(f"{k}={len(v)}" for k, v in preds.items()))
    print(f"common successful records: {len(common)}/{len(records)}  gold fields: {gold}")

    ids = sorted(common)
    for label, pmap in preds.items():
        recs = [records[i] for i in ids]
        report = eb.score(recs, [pmap[i] for i in ids], label, data)
        report["records"] = len(ids)
        report["gold_fields"] = gold
        (out / f"{label}.json").write_text(json.dumps(report, indent=2))
        m = report["metrics"]["list-text"]
        print(f"  {label:16s} exact {m['exact']:.4f}  tokF1 {m['tok_f1']:.4f}  "
              f"contains {m['contains']:.4f}  answered {m['answered']}/{gold}  "
              f"null_ok {m['null_correct']:.4f}  exact@all {m['exact'] * m['answered'] / max(gold, 1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
