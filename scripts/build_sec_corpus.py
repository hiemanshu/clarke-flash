#!/usr/bin/env python3
"""Build a packed 65k-token LM corpus from SEC material contracts (EX-10 exhibits).

Why unlabelled: LoRA at rank 256 needs ~12k+ documents to hit 1 token/param, and no
public *labelled* contract corpus is that large. SEC EDGAR has 1.14M material contracts,
so this corpus is used for unsupervised LM adaptation of the backbone, after which the
extraction head is trained on the labelled CUAD documents.

Pipeline: download N parquet shards -> strip EDGAR header + HTML -> filter -> tokenize ->
concatenate into fixed-length token blocks -> write a uint32 .bin plus a JSON meta file.

    python scripts/build_sec_corpus.py --shards 12 --tokens 65536 --out data/sec_lm_65k
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/datasets/chenghao/sec-material-contracts/resolve/main/data"
TAG = re.compile(r"<[^>]{1,200}>")
WS = re.compile(r"[ \t\xa0]+")
BLANK = re.compile(r"\n{3,}")


def clean(raw: str) -> str:
    """EDGAR wrapper + HTML -> readable contract text."""
    if not raw:
        return ""
    # EDGAR files wrap content as <DOCUMENT><TYPE>..<TEXT> body; keep the body only
    marker = raw.find("<TEXT>")
    text = raw[marker + 6:] if marker >= 0 else raw
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = TAG.sub(" ", text)
    text = html.unescape(text)
    text = WS.sub(" ", text)
    text = BLANK.sub("\n\n", text)
    return text.strip()


def printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    return ok / len(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=0, help="0 = all local shards")
    ap.add_argument("--max-shards", type=int, default=0, dest="max_shards",
                    help="stop after N shards (0 = no limit)")
    ap.add_argument("--max-blocks", type=int, default=0, dest="max_blocks",
                    help="stop once N blocks are written (0 = no limit)")
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--out", default="data/sec_lm_65k")
    ap.add_argument("--raw-dir", default="data/raw/sec")
    ap.add_argument("--min-chars", type=int, default=1500, dest="min_chars")
    ap.add_argument("--max-chars", type=int, default=400_000, dest="max_chars")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted((raw_dir / "data").glob("*.parquet"))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise SystemExit(f"no parquet shards under {raw_dir}/data - run: "
                         "hf download chenghao/sec-material-contracts --repo-type dataset "
                         f"--local-dir {raw_dir} --include 'data/*.parquet'")
    print(f"[sec] {len(shard_paths)} local shards", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]

    buf: list[int] = []
    blocks = 0
    kept = skipped = 0
    fh_path = Path(args.out).with_suffix(".bin")
    fh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fh_path, "wb") as out:
        for shard, path in enumerate(shard_paths):
            if args.max_blocks and blocks >= args.max_blocks:
                print(f"[sec] reached --max-blocks {args.max_blocks}", flush=True)
                break
            table = pq.read_table(path, columns=["extension", "file_content"])
            exts = table.column("extension").to_pylist()
            contents = table.column("file_content").to_pylist()
            for ext, raw in zip(exts, contents):
                if ext == "pdf":
                    skipped += 1
                    continue
                text = clean(raw or "")
                if not (args.min_chars <= len(text) <= args.max_chars):
                    skipped += 1
                    continue
                if printable_ratio(text) < 0.98:
                    skipped += 1
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                kept += 1
                buf.extend(ids); buf.extend(sep)
                while len(buf) >= args.tokens:
                    block = np.asarray(buf[: args.tokens], dtype=np.uint32)
                    out.write(block.tobytes())
                    blocks += 1
                    del buf[: args.tokens]
            print(f"[sec] shard {shard + 1}/{len(shard_paths)}: kept {kept} docs, "
                  f"skipped {skipped}, blocks {blocks}, buffered {len(buf)}, "
                  f"{fh_path.stat().st_size/1e9:.2f} GB", flush=True)

    meta = {"tokens_per_block": args.tokens, "blocks": blocks, "docs_kept": kept,
            "docs_skipped": skipped, "tokenizer": args.tokenizer,
            "source": "chenghao/sec-material-contracts (CC-BY-SA-4.0)",
            "bin": fh_path.name, "bytes": fh_path.stat().st_size}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
