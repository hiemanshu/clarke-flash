#!/usr/bin/env python3
"""Re-ground existing label rows after the ground() index bug was fixed.

The bug: ground() searched a whitespace-collapsed string while mapping indices
through a NON-whitespace-only index list, so every span after the first space was
shifted (measured on the existing corpus: 0 of 2228 values verified, all 2228
re-ground correctly with the fixed mapping). The label VALUES were always right;
only the offsets were wrong, so repairing is a local re-location, not a re-label -
no API calls.

    uv run python scripts/reground_labels.py \
        --labels data/llm_preambles.jsonl \
        --source data/ledgar_split_cuad.jsonl \
        --out data/llm_preambles_regrounded.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clarke.textmetrics import norm
from llm_label_preambles import ground


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/llm_preambles.jsonl")
    ap.add_argument("--source", default="data/ledgar_split_cuad.jsonl")
    ap.add_argument("--out", default="data/llm_preambles_regrounded.jsonl")
    args = ap.parse_args(argv)

    states = {r["id"]: r["state"] for r in load(args.source)}
    rows = load(args.labels)
    changed = verified = unlocatable = missing_state = 0
    out = []
    for row in rows:
        if "error" in row:
            out.append(row)
            continue
        state = states.get(row["id"])
        if state is None:
            missing_state += 1
            out.append(row)
            continue
        fields = {}
        for name, f in (row.get("fields") or {}).items():
            vals, spans, kept = f.get("value") or [], [], []
            for v in vals:
                g = ground(state, v)
                if g and norm(state[g[0]:g[1]]) == norm(v):
                    kept.append(v)
                    spans.append(list(g))
            if kept:
                fields[name] = {"value": kept, "spans": spans}
        old_span = json.dumps({k: v.get("spans") for k, v in (row.get("fields") or {}).items()})
        new_span = json.dumps({k: v.get("spans") for k, v in fields.items()})
        changed += old_span != new_span
        verified += sum(len(f["spans"]) for f in fields.values())
        # any value that cannot be located contradicts the label; count it loudly
        lost = sum(len((row["fields"][n].get("value") or [])) for n in (row.get("fields") or {})) \
            - sum(len(f["value"]) for f in fields.values())
        unlocatable += lost
        out.append({**row, "fields": fields, "spans_regrounded": True})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"rows {len(rows)} | rows with changed spans {changed} | values verified {verified} "
          f"| values unlocatable (dropped) {unlocatable} | rows with no source state {missing_state}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
