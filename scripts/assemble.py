"""Layer-3 assembly: build training/eval mixes from cache entries + curated files.

Selection is a STABLE hash-gate over record ids (cache.id_selected): changing
the seed/salt re-selects in seconds with zero re-mining, and any record that
survives re-seeding is byte-identical (no list-order or rng-dependence). Cap is
a safety truncate after stable id-sort. Provenance written beside the mix.

usage:
  uv run python scripts/assemble.py --out data/mixes/wikiib_train.jsonl \
      --cache-dir data/cache/wikiib --salt b3 --keep-pct 6 --cap 15000 \
      --eval-out data/mixes/wikiib_eval.jsonl
  (repeat per corpus; chain scripts read data/mixes/*)
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cache as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--include", type=Path, nargs="*", default=[],
                    help="already-curated jsonl files included whole (no gate)")
    ap.add_argument("--salt", default="asm")
    ap.add_argument("--keep-pct", type=int, default=100, help="0-100, id-gate share of cache")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--eval-out", type=Path)
    args = ap.parse_args()

    tr, ev = [], []
    provenance = {"cache_dir": str(args.cache_dir) if args.cache_dir else None,
                  "salt": args.salt, "keep_pct": args.keep_pct, "included": [], "skipped_eval_flag": 0}
    if args.cache_dir:
        for f in sorted(args.cache_dir.glob("*.jsonl")):
            n_in = n_tr = 0
            for l in f.open():
                r = json.loads(l)
                n_in += 1
                if r.pop("_eval", False):
                    ev.append(r)
                    continue
                if C.id_selected(r["id"], args.salt, 100, args.keep_pct):
                    tr.append(r)
                    n_tr += 1
            provenance["included"].append({"file": str(f), "read": n_in, "selected": n_tr})
    for f in args.include:
        n = 0
        for l in f.open():
            r = json.loads(l)
            r.pop("_eval", None)
            tr.append(r)
            n += 1
        provenance["included"].append({"file": str(f), "read": n, "selected": n, "passthrough": True})

    if args.cap and len(tr) > args.cap:
        tr.sort(key=lambda r: r["id"])
        provenance["capped_from"] = len(tr)
        tr = tr[: args.cap]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w") as h:
        for r in tr:
            h.write(json.dumps(r) + "\n")
    tmp.rename(args.out)
    if args.eval_out:
        ev.sort(key=lambda r: r["id"])
        tmp = args.eval_out.with_suffix(args.eval_out.suffix + ".tmp")
        with tmp.open("w") as h:
            for r in ev:
                h.write(json.dumps(r) + "\n")
        tmp.rename(args.eval_out)
    provenance["train"] = len(tr)
    provenance["eval"] = len(ev)
    args.out.with_name(args.out.stem + ".provenance.json").write_text(json.dumps(provenance, indent=1))
    print(f"assembled {args.out}: {len(tr)} train"
          + (f" | {args.eval_out}: {len(ev)} eval" if args.eval_out else ""))


if __name__ == "__main__":
    main()
