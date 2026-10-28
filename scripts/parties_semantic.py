"""parties: strict (set-equality on annotated strings) vs SEMANTIC (entity-level) scoring.

CUAD annotates party *mentions*, not a canonical entity list: the same party can appear
twice, once with and once without its legal suffix, and some items are phrases ("we," "us,"
or "our") rather than names. Strict set equality therefore marks a substantively correct
answer as wrong.

Semantic scoring:
  - normalise (case, quotes, punctuation) and strip legal suffixes (Inc/LLC/Corp/S.A/...)
  - cluster items within a document whose keys contain one another (same party, different crop)
  - a predicted entity matches a gold entity if either key contains the other
  - report entity recall/precision and "doc exact" = every gold entity found AND nothing spurious
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEGAL = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|s\.a|sa|l\.p|lp|"
    r"plc|n\.v|nv|gmbh|holdings|partners|group)\b\.?", re.IGNORECASE)


def key(text: str) -> str:
    t = str(text).lower()
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    t = LEGAL.sub(" ", t)
    return " ".join(t.split())


def cluster(values: list[str]) -> list[str]:
    """Merge mentions of the same entity (one key contained in another)."""
    keys = []
    for v in values:
        k = key(v)
        if len(k) >= 3 and k not in keys:
            keys.append(k)
    merged: list[str] = []
    for k in sorted(keys, key=len, reverse=True):
        if any(k in m or m in k for m in merged):
            continue
        merged.append(k)
    return merged


def score(gold: list[str], pred: list[str]) -> dict[str, float]:
    g, p = cluster(gold), cluster(pred)
    if not g:
        return {}
    matched_g = [x for x in g if any(x in y or y in x for y in p)]
    matched_p = [y for y in p if any(y in x or x in y for x in g)]
    return {
        "gold_entities": len(g), "pred_entities": len(p),
        "recall": len(matched_g) / len(g),
        "precision": len(matched_p) / max(len(p), 1),
        "doc_exact": int(len(matched_g) == len(g) and len(matched_p) == len(p)),
        "strict": int(sorted({key(v) for v in gold if key(v)}) ==
                      sorted({key(v) for v in pred if key(v)})),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="data/doc_eval_40.jsonl")
    ap.add_argument("--preds", nargs="+", required=True,
                    help="jsonl files with {id, pred: {field: [...]}} or {id, parties: [...]}")
    ap.add_argument("--field", default="parties")
    args = ap.parse_args(argv)

    gold = {}
    for line in Path(args.gold).open():
        r = json.loads(line)
        gold[r["id"]] = r["answers"].get(args.field, {}).get("value") or []

    print(f"{'model':26s} {'docs':>4s} {'strict':>8s} {'doc-exact':>10s} {'recall':>7s} {'prec':>6s} "
          f"{'gold ents':>10s} {'pred ents':>10s}")
    for path in args.preds:
        rows = [json.loads(l) for l in Path(path).open()]
        agg = {"n": 0, "strict": 0, "doc": 0, "r": 0.0, "p": 0.0, "g": 0, "pe": 0}
        for r in rows:
            g = gold.get(r["id"])
            if g is None:
                continue
            pred = (r.get("pred") or {}).get(args.field) if "pred" in r else r.get(args.field)
            pred = pred or []
            if isinstance(pred, str):
                pred = [pred]
            s = score(g, pred)
            if not s:
                continue
            agg["n"] += 1; agg["strict"] += s["strict"]; agg["doc"] += s["doc_exact"]
            agg["r"] += s["recall"]; agg["p"] += s["precision"]
            agg["g"] += s["gold_entities"]; agg["pe"] += s["pred_entities"]
        n = max(agg["n"], 1)
        name = Path(path).stem.replace(".preds", "")
        print(f"{name:26s} {agg['n']:4d} {agg['strict']:4d}/{agg['n']:<3d} "
              f"{agg['doc']:5d}/{agg['n']:<4d} {agg['r']/n:7.3f} {agg['p']/n:6.3f} "
              f"{agg['g']/n:10.2f} {agg['pe']/n:10.2f}")


if __name__ == "__main__":
    main()
