#!/usr/bin/env python3
"""Re-score external runs on the subset of records EVERY model completed.

Why this exists: a failed HTTP call yields an empty prediction, and the scorer
counts an empty prediction as "correctly null" for every absent field. So
scoring a partially failed run REWARDS it. Comparing two transports is worse
still, because each can fail on a different set of records — the model that
failed more looks better on `null_ok` and worse on `answered` for reasons that
have nothing to do with the model.

This restricts every run to the intersection of successful records and rewrites
comparable reports.

    # one directory (original behaviour; writes <dir>/rescored/)
    uv run python scripts/rescore_common.py reports/external_doc data/doc_eval_40.jsonl

    # several directories, e.g. Cloudflare vs CommandCode (writes --out)
    uv run python scripts/rescore_common.py \
        --dirs reports/external_doc reports/external_doc_cmdcode \
        --data data/doc_eval_40.jsonl --out reports/external_common

Prediction rows written by the current harness carry an explicit "ok" flag.
Older rows do not, so success is inferred from a non-empty prediction dict;
that is exact for these runs (a real answer set is always non-empty, a failed
call is `{}`) and the report says which rule was used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_baseline as eb  # noqa: E402


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def load_run(directory: Path) -> dict[str, dict]:
    """id -> {pred, ok} for every record with an explicit success verdict."""
    out: dict[str, dict] = {}
    for f in sorted(directory.glob("*.preds.jsonl")):
        label = f.stem.replace(".preds", "")
        rows: dict[str, dict] = {}
        inferred = False
        for line in f.read_text().splitlines():
            item = json.loads(line)
            if "ok" in item:
                ok = bool(item["ok"])
            else:
                ok = bool(item.get("pred"))
                inferred = True
            rows[item["id"]] = {"pred": item.get("pred") or {}, "ok": ok}
        out[label] = {"rows": rows, "inferred": inferred, "path": f}
    return out


def load_timing(directory: Path) -> dict[str, dict[str, dict]]:
    out = {}
    for f in sorted(directory.glob("*.timing.jsonl")):
        label = f.stem.replace(".timing", "")
        out[label] = {json.loads(l)["id"]: json.loads(l)
                      for l in f.read_text().splitlines() if l.strip()}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="one or more run directories")
    ap.add_argument("--dirs", dest="dirs_flag", nargs="+", default=None)
    ap.add_argument("--data", default="data/doc_eval_40.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    dirs = args.dirs_flag or args.dirs
    data = args.data
    if not args.dirs_flag and len(dirs) == 2 and Path(dirs[1]).is_file():
        dirs, data = [dirs[0]], dirs[1]          # legacy: <dir> <data>
    if not dirs:
        raise SystemExit("pass one or more run directories")
    multi = len(dirs) > 1 or args.dirs_flag is not None

    records = {json.loads(l)["id"]: json.loads(l) for l in open(data) if l.strip()}
    runs = {}
    for d in dirs:
        runs.update(load_run(Path(d)))
    timings = {}
    for d in dirs:
        timings.update(load_timing(Path(d)))

    common = set.intersection(*(set(v["rows"]) for v in runs.values())) if runs else set()
    successful = {label: {i for i, r in v["rows"].items() if r["ok"]}
                  for label, v in runs.items()}
    common_ok = set.intersection(*successful.values()) if successful else set()
    scored = sorted(common_ok if common_ok else common)

    gold = sum(1 for i in scored for n in records[i]["fields"]
               if (records[i]["answers"].get(n) or {}).get("value") is not None)
    print(f"records per run: " + ", ".join(
        f"{k}={len(v['rows'])} ok={len(successful[k])}" for k, v in runs.items()))
    for label, v in runs.items():
        if v["inferred"]:
            print(f"  note: {label} preds.jsonl has no 'ok' flag; success inferred "
                  f"from a non-empty prediction")
    print(f"scoring on {len(scored)}/{len(records)} records common to every successful run "
          f"({gold} gold-present fields)")

    out_dir = Path(args.out) if args.out else (Path(dirs[0]) / "rescored" if not multi else
                                               Path("reports/external_common"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, v in sorted(runs.items()):
        recs = [records[i] for i in scored]
        preds = [v["rows"][i]["pred"] for i in scored]
        report = eb.score(recs, preds, label, data)
        report["records"] = len(scored)
        report["gold_fields"] = gold
        report["success_rule"] = ("ok flag" if not v["inferred"] else
                                  "non-empty prediction (no ok flag)")
        t = timings.get(label) or {}
        lat = [float(t[i]["latency_s"]) for i in scored
               if i in t and t[i].get("ok") and t[i].get("latency_s")]
        tok = [t[i]["output_tokens"] for i in scored
               if i in t and t[i].get("ok") and t[i].get("output_tokens")]
        if lat:
            report["timing"] = {
                "n": len(lat),
                "latency_s": {"p50": round(_pct(lat, .5) or 0, 3),
                              "p90": round(_pct(lat, .9) or 0, 3),
                              "mean": round(sum(lat) / len(lat), 3)},
                "output_tokens_mean": round(sum(tok) / len(tok), 1) if tok else None,
            }
        (out_dir / f"{label}.json").write_text(json.dumps(report, indent=2))
        m = report["metrics"]["list-text"]
        rows.append([label, m["exact"], m["tok_f1"], m["contains"], m["null_correct"],
                     m["answered"], gold, m["exact"] * m["answered"] / max(gold, 1),
                     (report.get("timing") or {}).get("latency_s", {}).get("p50")])

    print(f"\n{'run':26s} {'exact':>7s} {'tokF1':>7s} {'contains':>8s} {'null_ok':>7s} "
          f"{'ans':>7s} {'exact@all':>9s} {'p50 s':>7s}")
    for r in rows:
        lat = f"{r[8]:7.2f}" if r[8] is not None else f"{'—':>7s}"
        print(f"{r[0]:26s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:8.4f} {r[4]:7.4f} "
              f"{r[5]:>3d}/{r[6]:<3d} {r[7]:9.4f} {lat}")
    print(f"\nwrote {len(rows)} reports to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
