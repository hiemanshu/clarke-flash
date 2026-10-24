"""Layer-3 assembly: build training/eval mixes from cache entries + curated files.

STREAMING (2026-10-24 rewrite): cache dirs can hold millions of records; this
never materialises the record list. Selection = STABLE salted hash-gate on
record id (cache.id_selected with a fine modulus), so re-seeding is seconds and
kept records are byte-identical. _eval-flagged records go to --eval-out
(streamed); everything else that passes the gate goes to --out.
Provenance written beside the mix.

usage:
  uv run python scripts/assemble.py --out data/mixes/wikiib_train.jsonl \
      --cache-dir data/cache/wikiib --salt b3 --gate-mod 1000 --gate-keep 4 \
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
    ap.add_argument("--gate-mod", type=int, default=1000)
    ap.add_argument("--gate-keep", type=int, default=1000,
                    help="keep ids with hash%%mod < keep (keep=mod -> all)")
    ap.add_argument("--eval-out", type=Path)
    args = ap.parse_args()

    prov = {"cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "salt": args.salt, "gate": f"{args.gate_keep}/{args.gate_mod}",
            "streamed": True, "included": [], "train": 0, "eval": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    fo = out_tmp.open("w")
    fe = None
    if args.eval_out:
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        fe = args.eval_out.with_suffix(args.eval_out.suffix + ".tmp").open("w")

    if args.cache_dir:
        for f in sorted(args.cache_dir.glob("*.jsonl")):
            read = kept = ev = 0
            with f.open() as fh:
                for line in fh:
                    read += 1
                    r = json.loads(line)
                    if r.pop("_eval", False):
                        ev += 1
                        if fe and C.id_selected(r["id"], args.salt + "-e", args.gate_mod, 2):
                            fe.write(json.dumps(r) + "\n")
                        continue
                    if C.id_selected(r["id"], args.salt, args.gate_mod, args.gate_keep):
                        fo.write(json.dumps(r) + "\n")
                        kept += 1
            prov["included"].append({"file": str(f), "read": read,
                                     "selected": kept, "eval": ev, "from_cache": True})
            prov["eval"] += ev
        prov["train"] = sum(x["selected"] for x in prov["included"] if x.get("from_cache"))

    for f in args.include:
        n = 0
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                r.pop("_eval", None)
                fo.write(json.dumps(r) + "\n")
                n += 1
        prov["included"].append({"file": str(f), "read": n, "selected": n, "passthrough": True})
        prov["train"] += n

    fo.close()
    out_tmp.rename(args.out)
    if fe:
        fe.close()
        args.eval_out.with_suffix(args.eval_out.suffix + ".tmp").rename(args.eval_out)
    args.out.with_name(args.out.stem + ".provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"assembled {args.out}: {prov['train']} train"
          + (f" | {args.eval_out}: {prov['eval']} eval" if fe else ""))


if __name__ == "__main__":
    main()
