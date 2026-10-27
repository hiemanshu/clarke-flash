#!/usr/bin/env python3
"""Build span-supervised contract records from LEDGAR labels + the SEC contract corpus.

LEDGAR gives 60k clause excerpts with 100 category labels but NO document pointer and no
offsets. We already hold 1.14M SEC material contracts. So: index the LEDGAR provisions, scan
contract text for them, and any verbatim hit yields (field label, exact char span) for free.

This targets the fields the 65k diagnosis found are never located at all (Class B):
change_of_control, non_disparagement, insurance, termination_for_convenience, governing_law.
Those have 3-15 gold instances in CUAD, so the head defaults to null; here they arrive by the
thousand, from real contracts, with exact offsets.

    python scripts/build_ledgar_spans.py --shards 20 --out data/ledgar_spans.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sec_corpus import clean  # noqa: E402

# LEDGAR label -> our CUAD field name (only the ones we actually fail on)
LABEL_MAP = {
    "Change In Control": "change_of_control",
    "Non-Disparagement": "non_disparagement",
    "Insurances": "insurance",
    "Governing Laws": "governing_law",
    "Terminations": "termination_for_convenience",
    "Assignments": "anti_assignment",
    "Non-Compete": "non_compete",
    "Indemnifications": "cap_on_liability",
    "Warranties": "warranty_duration",
    "Exclusivity": "exclusivity",
    "Effective Dates": "effective_date",
}
MIN_CHARS = 120     # ignore short/boilerplate fragments that will match everywhere
MAX_CHARS = 1200


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def norm_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace + lowercase, keeping a map back to original char indices."""
    out: list[str] = []
    pos: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            pos.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            pos.append(i)
            prev_space = False
    return "".join(out), pos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--out", default="data/ledgar_spans.jsonl")
    ap.add_argument("--raw-dir", default="data/raw/sec", dest="raw_dir")
    ap.add_argument("--max-per-field", type=int, default=0, dest="max_per_field")
    ap.add_argument("--window-chars", type=int, default=240_000, dest="window_chars",
                    help="split documents longer than this (240k chars ~ 51k tokens, safely "
                         "inside the 65,536 budget); 0 disables splitting")
    ap.add_argument("--stride-chars", type=int, default=220_000, dest="stride_chars")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    names = ds.features["label"].names
    import ahocorasick
    auto = ahocorasick.Automaton()
    for row in ds:
        label = names[row["label"]]
        field = LABEL_MAP.get(label)
        if not field:
            continue
        text = row["text"].strip()
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        key = norm(text)[:80]
        if len(key) < 40:
            continue
        if key not in auto:
            auto.add_word(key, (field, norm(text)))
    auto.make_automaton()
    print(f"[ledgar] {len(auto)} provision keys across "
          f"{len({v[0] for _, v in auto.items()})} fields", flush=True)

    shards = sorted((Path(args.raw_dir) / "data").glob("*.parquet"))[: args.shards]
    out = Path(args.out)
    kept = 0
    field_counts: dict[str, int] = {}
    with open(out, "w") as fh:
        for shard in shards:
            table = pq.read_table(shard, columns=["extension", "file_content"])
            for ext, raw in zip(table.column("extension").to_pylist(),
                                table.column("file_content").to_pylist()):
                if ext == "pdf":
                    continue
                text = clean(raw or "")
                if not (1500 <= len(text) <= 400_000):
                    continue
                normed, pos = norm_with_map(text)
                fields: dict[str, dict] = {}
                for end, (field, prov_norm) in auto.iter(normed):
                    start = end - 80 + 1 if normed[end - 79:end + 1] == prov_norm[:80] else None
                    if start is None or start < 0:
                        continue
                    length = len(prov_norm) if normed[start:start + len(prov_norm)] == prov_norm else 80
                    o_start = pos[start]
                    o_end = pos[min(start + length - 1, len(pos) - 1)] + 1
                    spans = fields.setdefault(field, {"value": [], "spans": []})
                    if any(abs(o_start - s0) < 20 for s0, _ in spans["spans"]):
                        continue
                    spans["value"].append(text[o_start:o_end])
                    spans["spans"].append([o_start, o_end])
                if not fields:
                    continue
                if args.max_per_field:
                    fields = {k: v for k, v in fields.items()
                              if field_counts.get(k, 0) < args.max_per_field}
                if not fields:
                    continue
                for k in fields:
                    field_counts[k] = field_counts.get(k, 0) + 1
                win = args.window_chars
                if not win or len(text) <= win:
                    chunks = [(0, text)]
                else:
                    chunks = [(a, text[a:a + win])
                              for a in range(0, len(text), max(1, args.stride_chars))]
                for offset, chunk in chunks:
                    local = {}
                    for name, v in fields.items():
                        vals, sps = [], []
                        for value, (s0, s1) in zip(v["value"], v["spans"]):
                            if s0 < offset or s1 > offset + len(chunk):
                                continue          # span sits outside this window
                            vals.append(value)
                            sps.append([s0 - offset, s1 - offset])
                        if vals:
                            local[name] = {"value": vals, "spans": sps}
                    if not local:
                        continue
                    fh.write(json.dumps({
                        "id": f"ledgar-{shard.stem}-{kept}",
                        "domain": "contract",
                        "state": chunk,
                        "fields": {k: {"type": "list", "items": "text",
                                       "instructions": f"Extract the {k.replace('_', ' ')} clause."}
                                   for k in local},
                        "answers": local,
                    }) + "\n")
                    kept += 1
            print(f"[ledgar] {shard.name}: docs kept {kept}, fields {field_counts}", flush=True)

    print(json.dumps({"docs": kept, "by_field": field_counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
