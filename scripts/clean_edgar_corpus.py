"""One-time EDGAR cleanup: HTML -> canonical plain text, quality flags, content hashes.

Why: the same clean() pass is re-run inside every builder (LEDGAR spans, LM corpus, the
field miner). This writes it once to data/edgar_clean/ so every downstream job reads text
directly, and it records enough metadata to filter junk and duplicates without touching the
raw corpus (kept intact for provenance).

Per shard we write one parquet with:
  id, cik, filing_date, doc_type, file_name, text, chars, printable_ratio,
  has_contract_marker, hash
Duplicates are detected afterwards from the hashes (scripts/clean_edgar_corpus.py --dedupe)
so no clean text has to be rewritten.

    uv run python scripts/clean_edgar_corpus.py --workers 8
    uv run python scripts/clean_edgar_corpus.py --dedupe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("data/raw/sec")
OUT = Path("data/edgar_clean")
TAG = re.compile(r"(?is)<[^>]+>")
WS = re.compile(r"[ \t\r\f\v]+")
BLANK = re.compile(r"\n{2,}")
CONTRACT_MARKERS = re.compile(r"(?i)\b(agreement|contract|license|lease|indenture)\b")
MIN_CHARS = 2000
MIN_PRINTABLE = 0.90

SCHEMA = pa.schema([
    ("id", pa.string()), ("cik", pa.string()), ("filing_date", pa.string()),
    ("doc_type", pa.string()), ("file_name", pa.string()), ("text", pa.string()),
    ("chars", pa.int32()), ("printable_ratio", pa.float32()),
    ("has_contract_marker", pa.bool_()), ("hash", pa.string()),
])


def clean(raw: str) -> str:
    if not raw:
        return ""
    marker = raw.find("<TEXT>")
    text = raw[marker + 6:] if marker >= 0 else raw
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = TAG.sub(" ", text)
    import html as _html
    text = _html.unescape(text)
    text = WS.sub(" ", text)
    return BLANK.sub("\n\n", text).strip()


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    ok = sum(1 for ch in text[:20000] if ch.isprintable() or ch in "\n\t")
    return ok / min(len(text), 20000)


def process_shard(path_str: str) -> dict:
    path = Path(path_str)
    out_path = OUT / path.name
    if out_path.exists():
        return {"shard": path.name, "skipped": True}
    t0 = time.time()
    table = pq.read_table(path, columns=["cik", "filing_date", "doc_type", "filename", "file_content"])
    cols = table.to_pydict()
    rows = {k: [] for k in SCHEMA.names}
    kept = dropped = 0
    for i in range(len(cols["cik"])):
        text = clean(cols["file_content"][i] or "")
        chars = len(text)
        pr = printable_ratio(text)
        if chars < MIN_CHARS or pr < MIN_PRINTABLE:
            dropped += 1
            continue
        key = hashlib.sha1(text[:8000].encode("utf-8", "ignore")).hexdigest()
        rows["id"].append(f"{path.stem}:{i}")
        rows["cik"].append(str(cols["cik"][i] or ""))
        rows["filing_date"].append(str(cols["filing_date"][i] or ""))
        rows["doc_type"].append(str(cols["doc_type"][i] or ""))
        rows["file_name"].append(str(cols["filename"][i] or ""))
        rows["text"].append(text)
        rows["chars"].append(chars)
        rows["printable_ratio"].append(pr)
        rows["has_contract_marker"].append(bool(CONTRACT_MARKERS.search(text[:5000])))
        rows["hash"].append(key)
        kept += 1
    pq.write_table(pa.table(rows, schema=SCHEMA), out_path, compression="zstd")
    return {"shard": path.name, "kept": kept, "dropped": dropped,
            "seconds": round(time.time() - t0, 1)}


def dedupe() -> None:
    hashes: dict[str, int] = {}
    total = 0
    for f in sorted(OUT.glob("*.parquet")):
        for h in pq.read_table(f, columns=["hash"]).column("hash").to_pylist():
            total += 1
            hashes[h] = hashes.get(h, 0) + 1
    dups = {h: c for h, c in hashes.items() if c > 1}
    dup_docs = sum(c - 1 for c in dups.values())
    (OUT / "dups.json").write_text(json.dumps({"total": total, "unique": len(hashes),
                                               "dup_groups": len(dups),
                                               "dup_docs": dup_docs}))
    print(f"dedupe: {total:,} docs, {len(hashes):,} unique, "
          f"{dup_docs:,} duplicates ({dup_docs/max(total,1):.1%})", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args(argv)

    if args.dedupe:
        dedupe()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    shards = sorted(RAW.rglob("*.parquet"))
    if args.limit:
        shards = shards[: args.limit]
    print(f"cleaning {len(shards)} shards with {args.workers} workers", flush=True)

    t0 = time.time()
    done = kept = dropped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_shard, str(s)): s for s in shards}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            kept += r.get("kept", 0)
            dropped += r.get("dropped", 0)
            if done % 10 == 0 or done == len(shards):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(shards)} shards  kept {kept:,}  dropped {dropped:,}  "
                      f"{rate:.2f} shard/s  eta {(len(shards)-done)/max(rate,1e-9)/60:.0f} min",
                      flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min: kept {kept:,}, dropped {dropped:,}", flush=True)


if __name__ == "__main__":
    main()
