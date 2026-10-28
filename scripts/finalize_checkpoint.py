"""Write decode_config.json into a checkpoint so the folder is self-contained.

Today the best doc40 number depends on five CLI flags plus two side files (the gold-length
histogram and the party ranker). Sharing `runs/H4-assign/checkpoint-1880` without them
silently reproduces the weaker default. This bakes:

  - thresholds and tie-break epsilon
  - the per-field gold length histogram (embedded, as counts)
  - candidate-constrained decode settings for name-like fields
  - the trained party ranker weights (embedded)

clarke-eval reads it automatically (see --no-decode-config to ignore), and any explicit CLI
flag still wins, so experiments stay possible.

    uv run python scripts/finalize_checkpoint.py --head runs/H4-assign/checkpoint-1880 \
        --hist reports/field_span_hist_train470.json --ranker reports/party_ranker.json \
        --eps 0.25 --prior-alpha 0.75 --candidate-fields parties --ranker-alpha 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True, help="checkpoint dir to finalize")
    ap.add_argument("--eps", type=float, default=0.25, help="span tie-break epsilon")
    ap.add_argument("--null-threshold", type=float, default=0.5)
    ap.add_argument("--slot-threshold", type=float, default=0.5)
    ap.add_argument("--slot-mask", action="store_true", default=True)
    ap.add_argument("--prior-alpha", type=float, default=0.75,
                    help="strength of the per-field span-length prior (0 disables)")
    ap.add_argument("--hist", default="reports/field_span_hist_train470.json",
                    help="per-field gold length histogram JSON (scripts/field_span_caps.py)")
    ap.add_argument("--candidate-fields", default="parties",
                    help="comma-separated fields that use candidate-constrained decode")
    ap.add_argument("--candidate-extra", type=int, default=0)
    ap.add_argument("--ranker", default="reports/party_ranker.json")
    ap.add_argument("--ranker-alpha", type=float, default=1.0)
    ap.add_argument("--out", default="decode_config.json")
    args = ap.parse_args(argv)

    head = Path(args.head)
    cfg: dict = {
        "version": 1,
        "null_threshold": args.null_threshold,
        "slot_threshold": args.slot_threshold,
        "slot_mask": bool(args.slot_mask),
        "span_tiebreak_eps": args.eps,
        "span_prior_alpha": args.prior_alpha,
        "candidate_decode": bool(args.candidate_fields.strip()),
        "candidate_fields": [f.strip() for f in args.candidate_fields.split(",") if f.strip()],
        "candidate_extra": args.candidate_extra,
        "ranker_alpha": args.ranker_alpha,
    }

    if args.prior_alpha > 0 and Path(args.hist).exists():
        hist = json.loads(Path(args.hist).read_text())
        cfg["span_priors"] = {name: row["hist"] for name, row in hist.items()}
        print(f"embedded length histograms for {len(cfg['span_priors'])} fields")
    if args.ranker and Path(args.ranker).exists() and args.ranker_alpha > 0:
        cfg["candidate_ranker"] = json.loads(Path(args.ranker).read_text())
        print(f"embedded ranker: {len(cfg['candidate_ranker']['features'])} features")

    path = head / args.out
    path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {path} ({path.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
