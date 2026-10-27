"""Per-field gold span-length distribution -> decode caps.

Why: at 65k the span error is a LENGTH error (start token right 72%, length right 56%,
exact 54%). `score = p_start * p_end` has no notion of how long a field's answers are,
so `parties` (gold median 16 chars) is answered with a whole clause. A per-field cap is
the cheapest way to test how much of that is reachable without retraining.

Caps are measured in TOKENS, through the same encoder path the pointer sees (char span ->
token span via the record's own offset map), and are derived from the TRAINING file only -
never the eval file - so the numbers on doc40 stay honest.

Output JSON: {field: {"n": N, "median": t, "p90": t, "p95": t, "p99": t, "p995": t, "max": t}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from clarke.data import RecordDataset
from clarke.encoding import encode_record, token_span_for_chars
from clarke.model import load_backbone


def _pct(values: list[int], p: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True,
                        help="TRAINING jsonl files (never the eval file)")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-2B", help="for the tokenizer only")
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--agg-atoms", dest="agg_atoms", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hist-out", dest="hist_out", default=None,
                        help="also write per-field gold length histograms (1..256 tokens)")
    args = parser.parse_args(argv)

    _, processor = load_backbone(args.backbone, device="cpu")
    tokenizer = processor.tokenizer
    lengths: dict[str, list[int]] = {}
    records = RecordDataset([Path(p) for p in args.data]).records
    for i, record in enumerate(records):
        try:
            enc = encode_record(tokenizer, record, max_length=args.max_length,
                                agg_atoms=args.agg_atoms)
        except ValueError:
            continue
        for field in enc.fields:
            answer = enc.answers.get(field.name) or {}
            spans = answer.get("spans") or ([answer["span"]] if answer.get("span") else [])
            if not spans:
                continue
            for span in spans:
                pair = token_span_for_chars(enc, span[0], span[1])
                if pair is None:
                    continue
                lengths.setdefault(field.name, []).append(pair[1] - pair[0] + 1)
        if (i + 1) % 100 == 0:
            print(f"[caps] {i + 1}/{len(records)} records", flush=True)

    caps: dict[str, Any] = {}
    for name, values in sorted(lengths.items()):
        caps[name] = {
            "n": len(values),
            "median": _pct(values, 0.50),
            "p90": _pct(values, 0.90),
            "p95": _pct(values, 0.95),
            "p99": _pct(values, 0.99),
            "p995": _pct(values, 0.995),
            "max": max(values),
        }
    Path(args.out).write_text(json.dumps(caps, indent=2))
    if args.hist_out:
        hist = {}
        for name, values in sorted(lengths.items()):
            counts = [0] * 256
            for v in values:
                counts[min(v, 256) - 1] += 1
            hist[name] = {"n": len(values), "hist": counts}
        Path(args.hist_out).write_text(json.dumps(hist))
        print(f"[caps] wrote {args.hist_out}")
    print(f"{'field':28s} {'n':>5s} {'median':>6s} {'p90':>5s} {'p95':>5s} {'p99':>5s} {'p99.5':>6s} {'max':>6s}")
    for name, c in caps.items():
        print(f"{name:28s} {c['n']:5d} {c['median']:6d} {c['p90']:5d} {c['p95']:5d} "
              f"{c['p99']:5d} {c['p995']:6d} {c['max']:6d}")
    print(f"[caps] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
