"""Mine clause spans for CUAD-style fields from the EDGAR contract corpus.

LEDGAR gave us 11 fields via verbatim provision matching. This covers the rest with
anchored patterns that require the VALUE, not just the topic word - a naive
"expiration" pattern matches "expiration of the statute of limitations", which is why
the first prototype looked good and was wrong.

Pipeline per contract: clean() the EDGAR HTML -> find anchored matches -> expand to the
containing sentence/clause -> apply per-field plausibility checks (dates must contain a
date, revenue sharing must contain a percentage, span length must match CUAD's range).

Usage (measurement first, no training yet):
    uv run python scripts/mine_edgar_fields.py --contracts 200 --examples 4
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

TAG = re.compile(r"(?is)<[^>]+>")
WS = re.compile(r"[ \t\r\f\v]+")
BLANK = re.compile(r"\n{2,}")

DATE = (r"(?:January|February|March|April|May|June|July|August|September|October|November|"
        r"December)\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|\d{4}-\d{2}-\d{2}"
        r"|(?:the\s+)?(?:Expiration|Termination|Renewal)\s+Date")
PCT = r"\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)"
MONEY = r"(?:US\s*)?\$\s?[\d,]+(?:\.\d+)?"


def clean(raw: str) -> str:
    if not raw:
        return ""
    marker = raw.find("<TEXT>")
    text = raw[marker + 6:] if marker >= 0 else raw
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = TAG.sub(" ", text)
    text = html.unescape(text)
    text = WS.sub(" ", text)
    return BLANK.sub("\n\n", text).strip()


# field -> (anchor regex, must-contain regex, max span chars)
FIELDS: dict[str, tuple[str, str, int]] = {
    "expiration_date": (
        rf"(?:expir(?:e|es|ation)|term\s+(?:shall\s+)?(?:end|expire))[^.\n]{{0,120}}?{DATE}",
        DATE, 320),
    "renewal_term": (
        r"(?:automatically\s+renew(?:ed)?|renew(?:ed|al)?\s+for\s+(?:an?\s+)?(?:additional|successive))"
        r"[^.\n]{0,100}?(?:one|two|three|\d+)\s*\(?\d?\)?\s*(?:year|month|years|months)",
        r"(?:year|month)", 320),
    "notice_period_to_terminate_renewal": (
        rf"(?:thirty|sixty|ninety|\d{{1,3}})\s*\(?\d{{0,3}}\)?\s*days?[’']?\s*(?:prior\s+)?"
        rf"(?:written\s+)?notice[^.\n]{{0,80}}",
        r"(?:day|notice)", 300),
    "audit_rights": (
        r"(?:right\s+to\s+audit|shall\s+(?:have\s+the\s+right\s+to\s+)?(?:audit|inspect))"
        r"[^.\n]{0,80}?(?:books|records|accounts|financial)",
        r"(?:books|records|accounts|financial)", 320),
    "license_grant": (
        r"(?:hereby\s+grants?|grants?\s+(?:to|unto)\s+[^.\n]{0,60}?)(?:a\s+)?"
        r"(?:non[-\s]?exclusive\s+|exclusive\s+|perpetual\s+|irrevocable\s+)?licen[cs]e"
        r"[^.\n]{0,80}",
        r"licen[cs]e", 320),
    "insurance": (
        r"(?:shall\s+(?:maintain|carry|procure)|maintains?|carrying)[^.\n]{0,60}?"
        r"insur(?:ance|ed)[^.\n]{0,100}",
        r"insur(?:ance|ed)", 320),
    "minimum_commitment": (
        r"(?:minimum|no\s+less\s+than|at\s+least)[^.\n]{0,60}?"
        r"(?:purchase|order|commitment|quantity|units)[^.\n]{0,80}",
        rf"(?:{MONEY}|\d)", 320),
    "revenue_profit_sharing": (
        rf"{PCT}[^.\n]{{0,80}}?(?:revenue|profit|net\s+sales|gross)",
        PCT, 320),
    "post_termination_services": (
        r"(?:following|after|upon)\s+(?:the\s+)?(?:expiration|termination)[^.\n]{0,60}?"
        r"(?:shall|will|agrees?\s+to)\s+(?:provide|continue|render|make\s+available)[^.\n]{0,100}",
        r"(?:provide|continue|render|available)", 340),
    "non_transferable_license": (
        r"(?:may\s+not|shall\s+not|is\s+not)\s+(?:be\s+)?(?:assigned|transferred|sublicensed)"
        r"[^.\n]{0,80}",
        r"(?:assign|transfer|sublicen)", 300),
}

SENT_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")


def sentence_span(text: str, start: int, end: int, cap: int) -> tuple[str, int, int]:
    """Expand a match to its sentence (CUAD spans are clauses, not anchor phrases)."""
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    hi_dot = text.find(".", end)
    hi_nl = text.find("\n", end)
    cands = [x for x in (hi_dot, hi_nl) if x != -1]
    hi = min(cands) + 1 if cands else min(len(text), end + 200)
    if hi - lo > cap:                      # keep the clause around the match
        lo = max(lo, start - (cap - (end - start)) // 2)
        hi = min(hi, lo + cap)
    return text[lo:hi].strip(), lo, hi


def mine(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for field, (anchor, must, cap) in FIELDS.items():
        for m in re.finditer(anchor, text, re.IGNORECASE):
            span, _lo, _hi = sentence_span(text, m.start(), m.end(), cap)
            if not span or len(span) < 25:
                continue
            if not re.search(must, span, re.IGNORECASE):
                continue
            if len(span.split()) > 90:     # too long to be a CUAD-style clause
                continue
            if span not in out[field]:
                out[field].append(span)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--contracts", type=int, default=200)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default=None, help="write sampled spans here (JSONL)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    files = sorted(Path("data/raw/sec").rglob("*.parquet"))[: args.shards]
    docs: list[str] = []
    for f in files:
        col = pq.read_table(f, columns=["file_content"]).column("file_content").to_pylist()
        docs.extend(x for x in col if x)
    sample = rng.sample(docs, min(args.contracts, len(docs)))
    print(f"sample: {len(sample)} contracts from {len(files)} shards", flush=True)

    hits: Counter = Counter(); spans: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    dumped = []
    for raw in sample:
        text = clean(raw)
        found = mine(text)
        for field, vals in found.items():
            hits[field] += 1
            spans[field] += len(vals)
            for v in vals[: args.examples]:
                if len(examples[field]) < args.examples:
                    examples[field].append(v)
            for v in vals[:3]:
                dumped.append({"field": field, "span": v})

    print(f"\n{'field':34s} {'docs hit':>9s} {'spans/doc':>10s}")
    for field in FIELDS:
        d = hits[field]
        print(f"{field:34s} {d/len(sample):9.1%} {spans[field]/max(d,1):10.2f}")
    print()
    for field in FIELDS:
        if examples[field]:
            print(f"--- {field}")
            for e in examples[field][:2]:
                print(f"    {re.sub(r'\\s+', ' ', e)[:190]}")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(d) for d in dumped))
        print(f"\nwrote {len(dumped)} sampled spans to {args.out}")


if __name__ == "__main__":
    main()
