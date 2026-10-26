"""Build a length-safe mix for LoRA training (4B: grad memory grows with T).

Keeps only records whose FULL encoding fits in MAX_LEN, because encode_record
truncates the state and build_targets then marks out-of-range positives as
NULL - i.e. truncation silently teaches false absences.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from clarke.encoding import encode_record

MAX_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 2560
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/mixes/lora_train.jsonl"
SRC = ["data/train3.jsonl", "data/typed_train.jsonl", "data/contract_train.jsonl",
       "data/cuad_train_2k.jsonl", "data/acord_train_dense.jsonl",
       "data/keyphrase_train.jsonl"]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def fit(rec):
    """Return a record that encodes under MAX_LEN, or None.

    If the schema alone makes it too long (CUAD has 41 instructions ~1.3k tok),
    keep the STATE and drop fields down to a subset - never truncate the state,
    because build_targets would turn out-of-range positives into false nulls.
    """
    try:
        e = encode_record(tok, rec, max_length=MAX_LEN)
        if len(e.input_ids) < MAX_LEN - 1:
            return rec
    except ValueError:
        return None
    names = list(rec["fields"])
    for keep in (32, 24, 18, 14, 10, 8, 6, 4, 3):
        sub = dict(rec)
        sub["fields"] = {n: rec["fields"][n] for n in names[:keep]}
        sub["answers"] = {n: rec["answers"][n] for n in names[:keep] if n in rec["answers"]}
        try:
            e = encode_record(tok, sub, max_length=MAX_LEN)
        except ValueError:
            continue
        if len(e.input_ids) < MAX_LEN - 1:
            return sub
    return None
out = Path(OUT)
out.parent.mkdir(parents=True, exist_ok=True)
kept = dropped = 0
tmp = out.with_suffix(".tmp")
with tmp.open("w") as fo:
    for src in SRC:
        n_kept = n_drop = 0
        for line in open(src):
            r = json.loads(line)
            kept_rec = fit(r)
            if kept_rec is None:
                n_drop += 1
                continue
            fo.write(json.dumps(kept_rec) + "\n")
            n_kept += 1
        print(f"{src}: kept {n_kept}, dropped {n_drop}", flush=True)
        kept += n_kept
        dropped += n_drop
tmp.rename(out)
print(f"-> {out}: {kept} records kept, {dropped} dropped (> {MAX_LEN} tok)")
