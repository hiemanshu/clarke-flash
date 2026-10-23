#!/usr/bin/env python3
"""Pretty-print the last rows of a clarke train.jsonl (used by watch_training.sh)."""

import json
import os
import sys

KEYS = ("null", "span", "value", "cand", "op", "present", "dist")


def fmt(row: dict) -> str:
    parts = []
    for key in KEYS:
        if key in row:
            parts.append(f"{key}={row[key]:.3f}")
    tag = "VAL " if row.get("val") else "     "
    extra = ""
    if "lr" in row:
        extra = f" lr={row['lr']:.2e} {row.get('recs_per_s', 0):.1f} rec/s {row.get('gpu_gb', 0):.1f} GB"
    return f"  step {row.get('step', 0):>6} {tag} loss={row.get('loss', float('nan')):.3f}{extra}  " + " ".join(parts)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import glob
        candidates = glob.glob("runs/*/train.jsonl")
        if not candidates:
            sys.exit("usage: show_run.py [run-dir|train.jsonl] [rows]; no runs/*/train.jsonl found")
        path = max(candidates, key=os.path.getmtime)
    if os.path.isdir(path):
        path = os.path.join(path, "train.jsonl")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    try:
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except OSError:
        print(f"  (no log yet at {path})")
        return
    train = [row for row in rows if not row.get("val")]
    val = [row for row in rows if row.get("val")]
    for row in train[-count:]:
        print(fmt(row))
    for row in val[-3:]:
        print(fmt(row))


if __name__ == "__main__":
    main()
