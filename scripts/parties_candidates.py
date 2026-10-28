"""Can `parties` ever be 40/40? Measure candidate coverage, not model accuracy.

Gold `parties` entries in CUAD are short mentions - defined terms ("Company"), entity
names ("Electric City Corp."), sometimes quoted. This probe asks the ceiling question:

    if a perfect SELECTOR existed over cheap candidate generators, how many gold
    parties would it have to choose from?

Candidate rules (no model, no GPU):
  1. quoted strings                     "Company", "Agreement"
  2. capitalised runs                   Electric City Corp. / Distributor
  3. party-introducing patterns         by and between X and Y / X, a Delaware corporation
Reported per document: how many of its gold items are present in the candidate pool, and
how many whole documents are fully covered (the only way to reach set-exact).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QUOTED = re.compile(r"[\"“”'']([A-Z][^\"“”'']{0,60})[\"“”'']")
CAP_RUN = re.compile(r"\b(?:[A-Z][A-Za-z&.\-]*)(?:\s+(?:[A-Z][A-Za-z&.\-]*|of|the|and|&))*")
INTRO = re.compile(r"(?:by and between|between)\s+([^.;]{5,200})", re.IGNORECASE)


def norm(text: str) -> str:
    return " ".join(str(text).split()).strip().strip("\"“”'").lower()


def candidates(state: str) -> set[str]:
    out: set[str] = set()
    for m in QUOTED.finditer(state):
        out.add(norm(m.group(1)))
    for m in CAP_RUN.finditer(state):
        c = norm(m.group())
        if 2 <= len(c) <= 80:
            out.add(c)
    for m in INTRO.finditer(state):
        chunk = m.group(1)
        for part in re.split(r",|\band\b|&", chunk):
            c = norm(part)
            if 2 <= len(c) <= 80:
                out.add(c)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/doc_eval_40.jsonl")
    parser.add_argument("--field", default="parties")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = []
    for line in Path(args.data).open():
        rec = json.loads(line)
        answer = rec.get("answers", {}).get(args.field) or {}
        gold = [norm(v) for v in (answer.get("value") or []) if v]
        if not gold:
            continue
        pool = candidates(rec["state"])
        # a gold item counts as covered if its normalised text is a pool member,
        # or shares a >=2-word prefix/suffix with one (annotations trim differently)
        hit = 0
        for g in gold:
            if g in pool or any(g in c or c in g for c in pool if len(c) > 3):
                hit += 1
        rows.append({"id": rec.get("id"), "gold": len(gold), "hit": hit,
                     "full": hit == len(gold), "pool": len(pool)})

    n = len(rows)
    items = sum(r["gold"] for r in rows)
    hits = sum(r["hit"] for r in rows)
    full = sum(1 for r in rows if r["full"])
    print(f"documents {n}  gold items {items}  pool size median "
          f"{sorted(r['pool'] for r in rows)[n // 2]}")
    print(f"gold items covered by a candidate rule : {hits}/{items} ({hits/items:.1%})")
    print(f"documents with ALL items covered       : {full}/{n} ({full/n:.1%})")
    print("\nper document (gold, covered, full):")
    for r in rows[:10]:
        print(f"  {r['id'][:44]:46s} gold={r['gold']} hit={r['hit']} full={r['full']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
