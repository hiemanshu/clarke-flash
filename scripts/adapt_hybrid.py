"""WikiTables-WithLinks (HybridQA crawl) -> clarke typed records.

15,314 Wikipedia tables (2021 crawl) in [text, [hyperlinks]] cell format with
column-wise headers. Same field families as scripts/adapt_wiki.py (row lookups,
row_count, column sum, argmax, two honest-null queries), same pipe-rendered
state so spans stay byte-exact within our own rendering. Cells are whitespace-
tokenized by the crawler; we render them verbatim (spans are into the state we
build, so this is harmless; header text in instructions carries mild "( s )"
noise, accepted).

Split hygiene: pages overlapping data/wiki_test.jsonl articles are DROPPED
(raw + case-folded url match). Remaining pages split at PAGE level (~2.5k
records to data/wiki_hybrid_test.jsonl holdout, rest train) so no page ever
appears on both sides.

usage: uv run python scripts/adapt_hybrid.py
"""
import hashlib, json, random, re
from pathlib import Path
import glob

TOK_DIR = Path("data/raw/WikiTables-WithLinks/tables_tok")
MAX_ROWS = 24
EVAL_RECORD_TARGET = 2500


def norm(s):
    return "".join(s.split()).lower()


def cell_text(c):
    """[text, [links]] or [ [tok,...], [links] ] -> plain text."""
    if isinstance(c, list) and c:
        t = c[0]
    elif isinstance(c, dict):
        t = c.get("value", "")
    else:
        t = c
    if isinstance(t, list):
        t = " ".join(str(x) for x in t)
    return str(t).strip()


def render_row(cells):
    return " | ".join(cells)


def convert(path, used_hashes):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    url = d.get("url", "")
    hh = hashlib.md5(url.encode()).hexdigest()[:12]
    if hh in used_hashes or hashlib.md5(url.lower().encode()).hexdigest()[:12] in used_hashes:
        return None
    hdr = d.get("header") or []
    data = d.get("data") or []
    # header is column-wise: one entry per column = [text, links] (text may be a token list)
    headers = [cell_text(h) for h in hdr]
    if not headers:
        return None
    for i in range(len(headers)):
        if headers[i] in headers[:i]:
            headers[i] = f"{headers[i]} {i + 1}"
    rows = []
    for r in data:
        if not isinstance(r, list) or not r:
            continue
        cells = [cell_text(c) for c in r]
        if len(cells) != len(headers) or any(x == "" or "|" in x for x in cells):
            continue
        rows.append(cells)
    rows = rows[:MAX_ROWS]
    if len(rows) < 2:
        return None
    state = "table\n" + render_row(headers) + "\n"
    grid = []
    for rw in rows:
        line_start = len(state)
        spans, pos = [], line_start
        for j, cv in enumerate(rw):
            if j:
                state += " | "
            s = len(state)
            state += cv
            spans.append((s, len(state)))
        grid.append(spans)
        state += "\n"
    state += "end of table"
    fields, answers = {}, {}

    def add(name, spec, ans):
        while name in fields:
            name += "_x"
        fields[name] = spec
        answers[name] = ans
        return name

    ncol = len(headers)
    keycol = random.randrange(ncol)
    if len(set(r[keycol] for r in rows)) == len(rows):
        for _ in range(min(3, len(rows))):
            ri = random.randrange(len(rows))
            qc = random.randrange(ncol)
            if qc == keycol:
                continue
            nm = add(re.sub(r"[^a-zA-Z0-9]+", "_",
                            f"{headers[qc]}_{headers[keycol]}_{rows[ri][keycol]}".lower())[:48].strip("_") or "f",
                     {"type": "text", "instructions":
                      f"value in {headers[qc]} for the row with {headers[keycol]} equal to {rows[ri][keycol]}"},
                     {"value": None})
            s, e = grid[ri][qc]
            answers[nm] = {"value": state[s:e], "span": [s, e]}
    add("row_count", {"type": "number", "integer": True, "instructions": "Number of data rows"},
        {"value": float(len(rows))})
    for ci in range(ncol):
        try:
            vals = [float(r[ci].replace(",", "").replace("$", "").strip()) for r in rows]
        except ValueError:
            continue
        if all(float(v).is_integer() for v in vals):
            add(re.sub(r"[^a-zA-Z0-9]+", "_", f"sum_{headers[ci]}".lower()).strip("_"),
                {"type": "number", "integer": True, "instructions": f"Sum of the {headers[ci]} column"},
                {"value": float(sum(vals))})
        mi = max(range(len(rows)), key=lambda i: vals[i])
        nm2 = add(re.sub(r"[^a-zA-Z0-9]+", "_", f"max_{headers[ci]}".lower()).strip("_"),
                  {"type": "text", "instructions":
                   f"{headers[ci]} value of the row with the highest {headers[ci]}"}, {"value": None})
        s, e = grid[mi][ci]
        answers[nm2] = {"value": state[s:e], "span": [s, e]}
        break
    hl = " ".join(headers).lower()
    fake = next((f for f in ("revenue_eur", "stock_ticker", "gps_coords") if f not in hl), None)
    if fake:
        add(fake, {"type": "text", "instructions": f"value in {fake} column"}, {"value": None})
    add("unknown_row_x", {"type": "text",
        "instructions": f"value in {headers[(keycol+1) % ncol]} for the row with {headers[keycol]} equal to ~zzQ9notpresent~"},
        {"value": None})
    if len(fields) < 3:
        return None
    return {"id": f"wikih-{hh}-t{Path(path).stem[-6:]}"[:60], "domain": "wikitable",
            "state": state, "fields": fields, "answers": answers}


def main():
    random.seed(20261024)
    ev_hashes = set()
    for l in open("data/wiki_test.jsonl"):
        ev_hashes.add(re.match(r"wiki-([0-9a-f]{12})", json.loads(l)["id"]).group(1))
    files = sorted(glob.glob(str(TOK_DIR / "*.json")))
    recs = [r for r in (convert(f, ev_hashes) for f in files) if r]
    # page-level split: group by url-hash prefix of id
    pages = {}
    for r in recs:
        pages.setdefault(r["id"].split("-")[1], []).append(r)
    tr, ev = [], []
    split_order = list(pages)
    random.shuffle(split_order)
    for ph in split_order:
        (ev if len(ev) < EVAL_RECORD_TARGET else tr).extend(pages[ph])
    bad = sum(1 for pool in (tr, ev) for r in pool for a in r["answers"].values()
              if a.get("span") and r["state"][a["span"][0]:a["span"][1]] != a["value"])
    print(f"hybrid records: {len(recs)} from {len(files)} tables ({len(recs)-len(pages)} dup pages merged)")
    print(f"split: train {len(tr)} / eval {len(ev)} records; span audit failures: {bad}")
    for pool, p in ((tr, "data/wiki_hybrid_train.jsonl"), (ev, "data/wiki_hybrid_test.jsonl")):
        with open(p, "w") as h:
            for r in pool:
                h.write(json.dumps(r) + "\n")
        print(f"-> {p}: {len(pool)}")
    nl = sum(1 for r in tr for a in r["answers"].values() if a["value"] is None)
    tot = sum(len(r["answers"]) for r in tr)
    print(f"train null rate: {nl}/{tot} = {nl/max(tot,1):.2f}")


if __name__ == "__main__":
    main()
