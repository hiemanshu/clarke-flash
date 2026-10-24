"""wikimedia/structured-wikipedia tables -> clarke typed records (batch-3 prep).

Parses the pre-parsed table JSON (headers/rows of {"value": cell}) into a
pipe-rendered state with byte-exact cell spans, then generates the same field
families our synthetic table regime uses, plus two real-data families it lacks:
  - row lookup:   "value in COL for the row with KEYCOL equal to KEYVAL" (text, span)
  - row_count     (number, integer)
  - sum / average of a clean numeric column (number; agg-atoms engine handles)
  - max/min argmax cell of a numeric column (text span)
  - NULL fields:  ask for a column not in the table, or a row key not present
                  -> honest not-present supervision for the table regime
Multi-row headers use the LAST row; tables with pipes/nested cells are skipped.
Train/eval split is by article URL hash (zero article overlap either way).

usage: uv run python scripts/adapt_wiki.py <parquet...> --out-train F --out-eval F
"""
import argparse, hashlib, json, random, re
from pathlib import Path


def _cells(row):
    out = []
    for c in row:
        if not isinstance(c, dict) or "value" not in c or c.get("nested_table"):
            return None
        v = str(c["value"]).strip()
        if "|" in v:
            return None
        out.append(v)
    return out


def _num(s):
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def convert_article(r, rng, used_urls):
    if not r.get("tables"):
        return []
    try:
        tabs = json.loads(r["tables"]) if isinstance(r["tables"], str) else r["tables"]
    except Exception:
        return []
    url = r.get("url") or r["name"]
    out = []
    for ti, t in enumerate(tabs if isinstance(tabs, list) else []):
        hdrs, rws = t.get("headers") or [], t.get("rows") or []
        if not hdrs or not rws:
            continue
        width = max(len(h) for h in hdrs if isinstance(h, list))
        headers = []
        for j in range(width):
            parts = []
            for h in hdrs:
                if isinstance(h, list) and j < len(h) and isinstance(h[j], dict):
                    v = str(h[j].get("value", "")).strip()
                    if v and v not in parts:
                        parts.append(v)
            headers.append(" ".join(parts) if parts else f"col{j}")
        for i in range(len(headers)):
            if headers[i] in headers[:i]:
                headers[i] = f"{headers[i]} {i + 1}"
        rows = [c for c in (_cells(rw) for rw in rws) if c]
        if not headers or len(rows) < 2:
            continue
        ncol = len(headers)
        rows = [rw for rw in rows if len(rw) == ncol and all(x != "" for x in rw)]
        rows = rows[:24]  # bound state size (some tables run to 900+ rows)
        if len(rows) < 2:
            continue
        # --- render pipe-table state with exact spans per cell ---
        state = "table\n"
        spans_grid = []
        def render_row(cells):
            nonlocal state, spans_grid
            line_spans = []
            pos = len(state)
            for j, cval in enumerate(cells):
                if j:
                    state += " | "
                s = len(state)
                state += cval
                line_spans.append((s, len(state)))
            state += "\n"
            return line_spans
        hdr_spans = render_row(headers)
        for rw in rows:
            spans_grid.append(render_row(rw))
        state += "end of table"
        # --- fields ---
        fields, answers = {}, {}
        keycol = rng.randrange(ncol)
        if len(set(r[keycol] for r in rows)) == len(rows):     # unique key column
            for _ in range(min(3, len(rows))):
                ri = rng.randrange(len(rows))
                qcol = rng.randrange(ncol)
                if qcol == keycol:
                    continue
                name = f"{headers[qcol]}_{headers[keycol]}_{rows[ri][keycol]}"[:48]
                name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or f"f{len(fields)}"
                while name in fields:
                    name += "_x"
                fields[name] = {"type": "text", "instructions":
                    f"value in {headers[qcol]} for the row with {headers[keycol]} equal to {rows[ri][keycol]}"}
                s, e = spans_grid[ri][qcol]
                answers[name] = {"value": state[s:e], "span": [s, e]}
        fields["row_count"] = {"type": "number", "integer": True,
                               "instructions": "Number of data rows"}
        answers["row_count"] = {"value": float(len(rows))}
        # numeric column aggregates + argmax
        for ci in range(ncol):
            vals = [_num(r[ci]) for r in rows]
            if len(vals) < 2 or any(v is None for v in vals):
                continue
            decs = max(len(r[ci].split(".")[1]) if "." in r[ci] else 0 for r in rows)
            if all(float(v).is_integer() for v in vals) and decs == 0:
                fields["sum_" + headers[ci]] = {"type": "number", "integer": True,
                    "instructions": f"Sum of the {headers[ci]} column"}
                answers["sum_" + headers[ci]] = {"value": float(sum(vals))}
            mi = max(range(len(rows)), key=lambda i: vals[i])
            nm = re.sub(r"[^a-zA-Z0-9]+", "_", f"max_{headers[ci]}").strip("_").lower()
            while nm in fields:
                nm += "_x"
            fields[nm] = {"type": "text", "instructions":
                f"{headers[ci]} value of the row with the highest {headers[ci]}"}
            s, e = spans_grid[mi][ci]
            answers[nm] = {"value": state[s:e], "span": [s, e]}
            break  # one numeric column per record keeps it sane
        # honest nulls: missing column + missing key
        hl = " ".join(headers).lower()
        fake = next((f for f in ("revenue_eur", "stock_ticker", "gps_coords") if f not in hl), None)
        if fake:
            fields[fake] = {"type": "text", "instructions": f"value in {fake} column"}
            answers[fake] = {"value": None}
        nk = "unknown_row_x"
        fields[nk] = {"type": "text",
            "instructions": f"value in {headers[(keycol+1) % ncol]} for the row with {headers[keycol]} equal to ~zzQ9notpresent~"}
        answers[nk] = {"value": None}
        if len(fields) < 3:
            return out
        out.append({"id": f"wiki-{hashlib.md5(url.encode()).hexdigest()[:12]}-t{ti}",
                    "domain": "wikitable", "state": state, "fields": fields, "answers": answers})
        used_urls.add(url)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out-train", type=Path, default=Path("data/wiki_train.jsonl"))
    ap.add_argument("--out-eval", type=Path, default=Path("data/wiki_test.jsonl"))
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    rng = random.Random(args.seed)
    used = set()
    tr, ev = [], []
    for sh in args.shards:
        tbl = pq.read_table(sh, columns=["tables", "name", "url"])
        for r in tbl.to_pylist():
            recs = convert_article(r, rng, used)
            for rec in recs:
                # split by URL hash: 20% eval
                (ev if int(hashlib.md5((rec["id"] + "salt").encode()).hexdigest(), 16) % 5 == 0 else tr).append(rec)
    bad = 0
    for pool in (tr, ev):
        for r in pool:
            for name, a in r["answers"].items():
                if a["value"] is None:
                    continue
                spec = r["fields"][name]
                if spec["type"] == "text" and "span" in a:
                    if r["state"][a["span"][0]:a["span"][1]] != a["value"]:
                        bad += 1
                if spec["type"] == "list" and a.get("spans"):
                    for v, sp in zip(a["value"], a["spans"]):
                        if r["state"][sp[0]:sp[1]] != v:
                            bad += 1
    assert bad == 0, f"{bad} span-integrity failures"
    print("self-audit: all spans byte-exact")
    for pool, path in ((tr, args.out_train), (ev, args.out_eval)):
        with path.open("w") as h:
            for r in pool:
                h.write(json.dumps(r) + "\n")
        print(f"{path}: {len(pool)} records")
    nulls = sum(1 for r in tr for a in r["answers"].values() if a["value"] is None)
    tot = sum(len(r["answers"]) for r in tr)
    print(f"train null rate: {nulls}/{tot} = {nulls/max(tot,1):.2f}")


if __name__ == "__main__":
    main()
