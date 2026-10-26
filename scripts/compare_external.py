#!/usr/bin/env python3
"""Head-to-head table: our checkpoints vs external models, on the same records.

Inputs
  reports/external/<label>.json      written by scripts/external_baseline.py
  reports/head2head/<name>.json      written by clarke-eval (same file, --limit 20)

Both scorers use the same definitions for `exact` (set-F1 over answered fields) and
`null_correct`; `tok_f1`/`contains` only exist for the external harness, so they are
blank for our rows. Invalid external runs (failed calls, zero answered) are excluded
and listed separately rather than shown as 0.000.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kind_metric(metrics: dict) -> tuple[str, dict] | tuple[None, dict]:
    for key in ("list-text", "text"):
        if key in metrics:
            return key, metrics[key]
    return None, {}


def load_external(path: Path):
    d = json.loads(path.read_text())
    kind, m = kind_metric(d.get("metrics") or {})
    if m.get("failed_calls") or d.get("failed_calls") or not m.get("answered"):
        return None
    return {
        "model": path.stem,
        "source": "external",
        "exact": m.get("exact"),
        "tok_f1": m.get("tok_f1"),
        "contains": m.get("contains"),
        "answered": m.get("answered"),   # None when the report predates the counter
        "null_ok": m.get("null_correct"),
        "n": m.get("n"),
    }


def load_ours(path: Path):
    d = json.loads(path.read_text())
    kind, m = kind_metric(d.get("metrics") or {})
    if m.get("answered") is None and not m.get("n"):
        return None
    label = path.stem
    for prefix in ("doc28-65k-", "doc-65k-", "scale2b-", "scale4b-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return {
        "model": f"clarke-{label}",
        "source": "clarke",
        "exact": m.get("exact"),
        "tok_f1": m.get("tok_f1"),
        "contains": m.get("contains"),
        "answered": m.get("answered"),   # None when the report predates the counter
        "null_ok": m.get("null_correct"),
        "n": m.get("n"),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    ext_dir = ROOT / (argv[0] if argv else "reports/external")
    ours_glob = argv[1] if len(argv) > 1 else "reports/head2head/*.json"
    rows, invalid = [], []
    for path in sorted(ext_dir.glob("*.json")):
        row = load_external(path)
        (rows if row else invalid).append(row or path.stem)
    for path in sorted(ROOT.glob(ours_glob)):
        row = load_ours(path)
        (rows if row else invalid).append(row or path.stem)

    rows.sort(key=lambda r: (r["exact"] is None, -(r["exact"] or 0)))
    print(f"{'model':26s} {'exact':>7s} {'tokF1':>7s} {'contain':>7s} "
          f"{'answer':>6s} {'null_ok':>7s}")
    for r in rows:
        fmt = lambda v, w, p=4: f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
        ans = f"{r['answered']:6d}" if isinstance(r["answered"], int) else f"{'?':>6s}"
        print(f"{r['model']:26s} {fmt(r['exact'], 7)} {fmt(r['tok_f1'], 7)} "
              f"{fmt(r['contains'], 7)} {ans} {fmt(r['null_ok'], 7)}")
    if invalid:
        print("\nexcluded (invalid runs): " + ", ".join(invalid))
    print(f"\nrows: {len(rows)}  records: "
          f"{sorted({r['n'] for r in rows if r['n']}) or '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
