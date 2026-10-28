#!/usr/bin/env python3
"""Build a PRIVATE eval set from the company's hand-labelled contracts.

Eval-only by owner decision (2026-10-29). The documents and every derived record
contain contract text, so they live under data/ (gitignored) and must never be
committed. Only aggregate numbers may leave the machine.

Input is the export zip: DOCX/PDF contracts plus CSV tag exports with, per field,
`<Field> - Status` (yes/no), `- Value` and `- Citations` (`Page N: <verbatim>`).
The CSV exports are snapshots, so the latest row per contract wins; 13% of rows
are column-shifted malformed exports and are dropped by requiring the File Name to
look like a document name.

Output: data/company_eval/company_eval.jsonl in Clarke's record format, with a
schema of the fields that map onto ours (plus parties). Fields the company labels
that we do not have (Term, Auto-renewal, Notice Period, ...) are recorded in the
manifest but not emitted, because an unlabelled field in a schema would be scored
as "absent" even when the contract contains it.

    uv run python scripts/build_company_eval.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from clarke.textmetrics import norm
from llm_label_preambles import ground

# company field -> our schema field. Only unambiguous mappings; 'Assignment' is
# reported separately because the company label may mean the assignment clause
# rather than CUAD's anti-assignment restriction.
MAP = {
    "Effective date": "effective_date",
    "Expiry date": "expiration_date",
    "Governing law": "governing_law",
    "Termination for convenience": "termination_for_convenience",
    "Assignment": "anti_assignment",
    "Non-compete": "non_compete",
    "Liability cap": "cap_on_liability",
}
UNMAPPED = ["Term", "Auto-renewal", "Term of Renewal", "Notice Period", "Termination provisions",
            "Dispute resolution", "Jurisdiction", "Non-solicitation", "Language Binding Affiliates",
            "Consent Required for Subcontracting", "Change of control"]
DOC_NAME = re.compile(r"(?i)^anonymizedcontract_?\d+(\.(pdf|docx))?$")


def norm_doc_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(str(name)).stem.lower())


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def pdf_text(path: Path) -> str:
    return subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True).stdout


def unescape(v: str) -> str:
    return re.sub(r"\\[nr]", " ", str(v)).strip()


def strip_page(cite: str) -> str:
    return re.sub(r"(?i)^page\s*[\d\s\-–,]+:\s*", "", unescape(cite)).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/Anonymized Contracts-20261028T173408Z-1-001.zip")
    ap.add_argument("--work", default="data/company_eval/work")
    ap.add_argument("--out", default="data/company_eval/company_eval.jsonl")
    ap.add_argument("--manifest", default="data/company_eval/manifest.json")
    ap.add_argument("--schema-template", default="data/doc_train_470.jsonl")
    args = ap.parse_args(argv)

    work = Path(args.work)
    if not any(work.glob("Anonymized Contracts/*")):
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(work)
    base = work / "Anonymized Contracts"

    docs: dict[str, tuple[str, str]] = {}
    for p in base.iterdir():
        if p.suffix.lower() not in (".docx", ".pdf"):
            continue
        text = docx_text(p) if p.suffix.lower() == ".docx" else pdf_text(p)
        docs[norm_doc_key(p.name)] = (p.name, text)

    frames = []
    for f in sorted((base / "Tags").glob("*.csv")):
        frames.append(pd.read_csv(f, dtype=str, keep_default_na=False))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["File Name"].map(lambda x: bool(DOC_NAME.match(str(x).strip())))]
    df["__lm"] = pd.to_datetime(df["Last Modified"], errors="coerce")
    latest = df.sort_values("__lm").groupby("File Name").tail(1).reset_index(drop=True)

    schema_all = json.loads(Path(args.schema_template).open().readline())["fields"]
    schema = {name: schema_all[name] for name in MAP.values()}
    schema["parties"] = schema_all["parties"]

    records, stats = [], Counter()
    for _, row in latest.iterrows():
        key = norm_doc_key(row["File Name"])
        if key not in docs:
            stats["no_document"] += 1
            continue
        fname, text = docs[key]
        answers: dict[str, dict] = {}
        for field, ours in MAP.items():
            if str(row.get(f"{field} - Status", "")).strip().lower() != "yes":
                stats[f"{ours}:absent"] += 1
                continue
            quote = strip_page(row.get(f"{field} - Citations", ""))
            span = ground(text, quote) if quote else None
            if not span or norm(text[span[0]:span[1]]) != norm(quote):
                stats[f"{ours}:ungrounded"] += 1
                continue
            answers[ours] = {"value": [quote], "spans": [list(span)]}
            stats[f"{ours}:present"] += 1
        names = [re.sub(r"^\s*\d+[\).\s]+", "", unescape(x)).strip(" ,")
                 for x in re.split(r"\\n|\n", str(row.get("Parties", ""))) if x.strip()]
        vals, spans = [], []
        for n in names:
            span = ground(text, n) if n else None
            if span:
                vals.append(n)
                spans.append(list(span))
            else:
                stats["parties:ungrounded"] += 1
        if vals:
            answers["parties"] = {"value": vals, "spans": spans}
            stats["parties:present"] += len(vals)
        if not answers:
            stats["contract_without_labels"] += 1
            continue
        cid = re.sub(r"[^0-9]", "", Path(str(row["File Name"])).stem)
        records.append({"id": f"company-{cid}", "state": text, "fields": schema,
                        "answers": answers, "domain": "company"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "source_zip": args.zip,
        "source_sha256": hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
        "private": True,
        "use": "eval-only",
        "contracts": len(records),
        "documents": len(docs),
        "fields_emitted": list(schema),
        "fields_not_emitted": UNMAPPED,
        "stats": dict(stats),
        "rows_dropped_malformed": int(len(pd.concat(frames, ignore_index=True)) - len(df)),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))

    print(f"contracts emitted: {len(records)} from {len(docs)} documents")
    print(f"schema ({len(schema)} fields): {list(schema)}")
    print("label stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k:34s} {v}")
    print(f"\nwrote {out} (PRIVATE, data/ is gitignored) and {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
