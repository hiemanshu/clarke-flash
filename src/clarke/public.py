"""Adapters turning public datasets into clarke record JSONL.

Each adapter reads a local export (see README for fetch instructions) and
emits the same {state, fields, answers-with-spans} records the synthetic
generator produces, so they can be appended to train/eval files. Adapters
exit quietly with a hint when the source files are missing.
"""

from __future__ import annotations

import csv
import json
import random
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def sroie_records(root: Path, limit: int = 2000) -> list[dict[str, Any]]:
    """SROIE-2019 receipt OCR: company/date/address as spans, total as number.

    Expects the official layout: one zip or dir with {id}.txt lines and
    {id}_task3.json answer keys.
    """
    directory = _unpack(root)
    if directory is None:
        return []
    records = []
    texts = sorted(directory.glob("*.txt"))
    for path in texts:
        key = path.stem
        answers_path = directory / f"{key}_task3.json"
        if not answers_path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        state, offsets = _join_lines(lines)
        gold = json.loads(answers_path.read_text())
        fields: dict[str, Any] = {}
        answers: dict[str, Any] = {}
        for name, key_name, kind in (("company", "COMPANY", "text"),
                                     ("date", "DATE", "date"),
                                     ("address", "ADDRESS", "text"),
                                     ("total", "TOTAL", "number")):
            value = (gold.get(key_name) or "").replace(":", "").strip()
            if kind == "text":
                fields[name] = {"type": "text", "instructions": f"Receipt {name}"}
                span = _find_span(state, offsets, value)
                answers[name] = {"value": value or None, "span": list(span) if (value and span) else None}
                if not span and value:
                    answers[name]["value"] = None
            elif kind == "date":
                fields[name] = {"type": "date", "instructions": "Receipt date"}
                iso = _loose_date(value)
                answers[name] = {"value": iso}
            else:
                fields[name] = {"type": "number", "decimals": 2, "unit": "USD",
                                "instructions": "Grand total on the receipt"}
                amount = _money(value)
                answers[name] = {"value": amount}
        if fields:
            records.append({"id": f"sroie-{key}", "domain": "sroie", "state": state,
                            "fields": fields, "answers": answers})
        if len(records) >= limit:
            break
    return records


def fewnerd_records(path: Path, limit: int = 3000) -> list[dict[str, Any]]:
    """Few-NERD JSONL (the official per-instance export) -> text span fields.

    Keeps PER/ORG/LOC; one field per entity type present in the sentence.
    """
    if not path.exists():
        print(f"[public] missing {path}; see README fetch instructions")
        return []
    wanted = {"PER": "person", "ORG": "organization", "LOC": "location"}
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            item = json.loads(line)
            state = item.get("sentence")
            entities = item.get("entities") or item.get("entity_list") or []
            fields: dict[str, Any] = {}
            answers: dict[str, Any] = {}
            position = 0
            for entity in entities:
                label = entity.get("type") if isinstance(entity, dict) else None
                text = entity.get("text") if isinstance(entity, dict) else None
                if label not in wanted or not text:
                    continue
                start = state.find(text, position)
                if start < 0:
                    continue
                position = start
                name = f"{wanted[label]}_{len(answers) + 1}"
                fields[name] = {"type": "text",
                                "instructions": f"The {wanted[label]} mentioned"}
                answers[name] = {"value": text, "span": [start, start + len(text)]}
            if fields:
                records.append({"id": f"fewnerd-{item.get('id', position)}-{len(records)}",
                                "domain": "fewnerd", "state": state,
                                "fields": fields, "answers": answers})
    return records


def series_records(directory: Path, horizon_max: int = 4, window: int = 60) -> list[dict[str, Any]]:
    """Generic time-series CSVs (date,value per file) -> forecast records.

    Slides a window over each file; the next ``horizon`` observed values are
    the realized labels. Handles TSA-style passenger data, Monash exports, etc.
    """
    if not directory.is_dir():
        print(f"[public] missing {directory}; pass --series-dir or see README")
        return []
    rng = random.Random(7)
    records = []
    for path in sorted(directory.glob("*.csv")):
        with path.open() as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            series: list[tuple[date, float]] = []
            for row in reader:
                try:
                    day = date.fromisoformat(row[0])
                    value = float(row[1])
                except (ValueError, IndexError):
                    continue
                series.append((day, value))
        if len(series) < window + 2:
            continue
        for _ in range(0, max(len(series) - window - horizon_max, 1)):
            offset = rng.randint(0, len(series) - window - horizon_max)
            history = series[offset:offset + window]
            horizon = rng.randint(1, horizon_max)
            future = series[offset + window:offset + window + horizon]
            lo = min(v for _, v in history)
            hi = max(v for _, v in history)
            margin = (hi - lo) * 0.35 + abs(hi) * 0.1
            state = f"Daily value of {path.stem}.\ndate,value\n"
            state += "\n".join(f"{day.isoformat()},{value:g}" for day, value in history)
            next_day = history[-1][0] + timedelta(days=1)
            last_day = future[-1][0]
            state += f"\nForecast dates: {next_day.isoformat()} through {last_day.isoformat()}."
            records.append({
                "id": f"series-{path.stem}-{offset}",
                "domain": "series",
                "state": state,
                "fields": {"value_next": {"type": "forecast", "horizon": horizon, "interval": "day",
                                          "min": max(0.0, lo - margin), "max": hi + margin,
                                          "scale": "log" if lo > 0 else "linear",
                                          "step": max(1.0, round((hi - lo + 2 * margin) / 200)),
                                          "instructions": "Value for each of the future days"}},
                "answers": {"value_next": {"values": [value for _, value in future]}},
            })
            if len(records) >= 5000:
                return records
    return records


# -- helpers ---------------------------------------------------------------


def cuad_records(path: Path, doc_start: int = 0, doc_end: int | None = None,
                 window_chars: int = 16000, max_spans: int = 8) -> list[dict[str, Any]]:
    """CUAD (theatticusproject/cuad, CC-BY-4.0) -> clarke records.

    Each contract is windowed (newline-aligned); every one of the 41 clause
    categories becomes a ``list<text>`` field whose instruction is CUAD's own
    human-written prompt ("... Details: What is the renewal term ...").
    Evidence spans are character-exact into the window; ``is_impossible``
    becomes a null answer, and evidence that sits outside a window is null
    *for that window* — which is exactly the extraction API's semantics.
    """
    path = Path(path)
    if not path.exists():
        print(f"[public] missing {path}; fetch:\n"
              "  curl -sSL -o data/raw/CUAD_v1.json \\\n"
              "    https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json")
        return []
    blob = json.loads(path.read_text(encoding="utf-8"))
    docs = blob["data"][doc_start:doc_end]
    records: list[dict[str, Any]] = []
    truncated_qas = 0
    for doc in docs:
        for p in doc.get("paragraphs", []):
            ctx = p.get("context") or ""
            qas = p.get("qas") or []
            for start in range(0, len(ctx), window_chars):
                chunk = ctx[start:start + window_chars]
                fields: dict[str, Any] = {}
                answers: dict[str, Any] = {}
                any_content = chunk.strip()
                if not any_content:
                    break
                for qa in qas:
                    cat = (qa["question"].split('"')[1] if '"' in qa["question"]
                           else qa["question"][:40])
                    name = _slug(cat)
                    details = qa["question"].split("Details:", 1)
                    instruction = (details[1].strip() if len(details) > 1 else qa["question"])
                    fields[name] = {"type": "list", "items": "text",
                                    "instructions": instruction}
                    spans: list[list[int]] = []
                    texts: list[str] = []
                    if not qa.get("is_impossible"):
                        for a in qa.get("answers") or []:
                            s = a["answer_start"] - start
                            e = s + len(a["text"])
                            if 0 <= s and e <= len(chunk) and len(texts) < max_spans:
                                spans.append([s, e])
                                texts.append(a["text"])
                    n_gold = len({a["text"] for a in (qa.get("answers") or [])})
                    if n_gold > max_spans:
                        truncated_qas += 1
                    answers[name] = ({"value": texts, "spans": spans} if texts
                                     else {"value": None})
                records.append({
                    "id": f"cuad-{doc.get('title', 'x')[:40]}-{start}",
                    "domain": "cuad",
                    "state": chunk,
                    "fields": fields,
                    "answers": answers,
                })
    print(f"[public] cuad: {len(records)} windowed records from {len(docs)} contracts"
          + (f"; TRUNCATED {truncated_qas} QAs to {max_spans} answers (CUAD has up to 55)"
             if truncated_qas else ""))
    return records


def _slug(text: str) -> str:
    out = "".join(c.lower() if (c.isalnum() or c == " ") else "_" for c in text)
    return "_".join(part for part in out.split() if part) or "field"


def _unpack(root: Path) -> Path | None:
    root = Path(root)
    if root.is_dir() and any(root.glob("*.txt")):
        return root
    zip_path = next(iter(root.glob("*.zip")), None) if root.is_dir() else (root if root.suffix == ".zip" else None)
    target = root if root.is_dir() else root.parent / (root.stem)
    if zip_path and not any(target.glob("*.txt")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
    return target if target.is_dir() and any(target.glob("*.txt")) else None


def _join_lines(lines: list[str]) -> tuple[str, list[tuple[int, int, int, int]]]:
    state_parts = []
    offsets = []
    cursor = 0
    for line in lines:
        start = cursor
        cursor += len(line)
        offsets.append((start, cursor, cursor - len(line), len(line)))
        state_parts.append(line)
        cursor += 1
    return "\n".join(state_parts), offsets


def _find_span(state: str, offsets: list[tuple[int, int, int, int]], value: str) -> tuple[int, int] | None:
    """Locate a (possibly whitespace-mangled) answer; join across lines with ' '."""
    if not value:
        return None
    direct = state.find(value)
    if direct >= 0:
        return direct, direct + len(value)
    flat = state.replace("\n", " ")
    flat_start = flat.find(value)
    if flat_start < 0:
        return None
    return _map_flat_to_state(state, flat, flat_start, len(value))


def _map_flat_to_state(state: str, flat: str, start: int, length: int) -> tuple[int, int]:
    delta = 0
    real_start = None
    for index in range(len(state)):
        if state[index] == "\n" and flat[index] == " ":
            delta += 1
        if index - delta == start and real_start is None:
            real_start = index
        if real_start is not None and index - delta == start + length - 1:
            return real_start, index + 1
    return None or (0, 0)


def _loose_date(text: str) -> str | None:
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            from datetime import datetime

            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return None


def _money(text: str) -> float | None:
    cleaned = "".join(char for char in text if char.isdigit() or char in ".-")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def keyphrase_records(path: Path) -> list[dict[str, Any]]:
    """jinya425/Keyphrase_extraction (MIT) -> list<text> records.

    Keywords are semicolon-joined verbatim substrings of the abstract, so the
    gold spans are recoverable by exact search. Tiny (~90 rows) — used as an
    out-of-domain smoke test, not a training slice. Accepts either a local
    parquet/csv export or the HF parquet downloaded to ``path``.
    """
    path = Path(path)
    if not path.exists():
        print(f"[public] missing {path}; fetch:\n"
              "  curl -sSL -o data/raw/keyphrase-test.parquet \\\n"
              "    'https://huggingface.co/datasets/jinya425/Keyphrase_extraction/resolve/main/data/test-00000-of-00001-5459d18a0e2d6a2d.parquet'")
        return []
    import csv
    records = []
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for i, row in enumerate(rows):
        abstract = row.get("Abstract") or row.get("abstract") or ""
        kw_blob = row.get("Keywords") or row.get("keywords") or ""
        keywords = [k.strip() for k in kw_blob.split(";") if k.strip()]
        spans = []
        keep = []
        for kw in keywords:
            start = abstract.find(kw)
            if start >= 0:
                spans.append([start, start + len(kw)])
                keep.append(kw)
        if not keep:
            continue
        records.append({
            "id": f"keyphrase-{i}",
            "domain": "keyphrase",
            "state": abstract,
            "fields": {"keywords": {"type": "list", "items": "text",
                                    "instructions": "Each key phrase of the abstract, in order of appearance"}},
            "answers": {"keywords": {"value": keep, "spans": spans}},
        })
    print(f"[public] keyphrase: {len(records)} records from {path.name}")
    return records


def typed_records(split: str) -> list[dict[str, Any]]:
    """LocalLLaMA/typed-extraction (CC-BY-SA-4.0) -> clarke records.

    The dataset ships our exact shape: ``state``, ``fields`` JSON with
    {type, instructions}, gold with {value, source, spans}. Families:
    squad2, drop, cuad, claim_email, invoice, lease. Text answers whose
    spans do not verify against the state are dropped entirely — we never
    train or score on fabricated offsets. ``reference_date`` is appended to
    the state so relative-duration answers stay checkable.
    """
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-extraction", split=split)
    records: list[dict[str, Any]] = []
    kept = dropped = 0
    for r in ds:
        fields_in = json.loads(r["fields"]) if isinstance(r["fields"], str) else r["fields"]
        gold_in = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        if r.get("reference_date"):
            state = f"{state}\nReference date: {r['reference_date']}"
        out_fields: dict[str, Any] = {}
        out_answers: dict[str, Any] = {}
        for name, fs in fields_in.items():
            ftype = str(fs.get("type", "text"))
            spec: dict[str, Any] = {"type": ftype, "instructions": fs.get("instructions") or name}
            for key in ("items", "unit", "decimals", "integer"):
                if fs.get(key) is not None:
                    spec[key] = fs[key]
            gold = gold_in.get(name) or {}
            value = gold.get("value")
            spans = gold.get("spans") or []
            if value is None:
                out_fields[name] = spec
                out_answers[name] = {"value": None}
                kept += 1
                continue
            if ftype == "list" and str(spec.get("items", "text")) == "text":
                if len(spans) != len(value):
                    dropped += 1
                    continue
                ok = all(state[s0:e0].strip() == str(v).strip() for (s0, e0), v in zip(spans, value))
                if not ok:
                    dropped += 1
                    continue
                spec["items"] = spec.get("items", "text")
                out_fields[name] = spec
                out_answers[name] = {"value": [str(v) for v in value],
                                     "spans": [[s0, e0] for s0, e0 in spans]}
                kept += 1
            elif ftype == "text":
                s0, e0 = spans[0] if spans else (None, None)
                if s0 is None or state[s0:e0].strip() != str(value).strip():
                    dropped += 1
                    continue
                out_fields[name] = spec
                out_answers[name] = {"value": str(value), "span": [s0, e0]}
                kept += 1
            else:
                out_fields[name] = spec
                out_answers[name] = {"value": value,
                                     "span": list(spans[0]) if spans else None}
                kept += 1
        if not out_fields:
            continue
        records.append({"id": f"typed-{r['id']}", "domain": f"typed-{r['family']}",
                        "state": state, "fields": out_fields, "answers": out_answers})
    print(f"[public] typed-extraction[{split}]: {len(records)} records, "
          f"{kept} answers kept, {dropped} dropped (span verification failed)")
    return records


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert public datasets to clarke record JSONL")
    parser.add_argument("--cuad", type=Path, default=Path("data/raw/CUAD_v1.json"))
    parser.add_argument("--keyphrase", type=Path, default=Path("data/raw/keyphrase-test.parquet"))
    parser.add_argument("--typed-split", dest="typed_split", default=None,
                        choices=["train", "test"], help="stream LocalLLaMA/typed-extraction")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--doc-start", dest="doc_start", type=int, default=0)
    parser.add_argument("--doc-end", dest="doc_end", type=int, default=None)
    parser.add_argument("--window", type=int, default=16000)
    args = parser.parse_args(argv)

    if args.typed_split:
        records = typed_records(args.typed_split)
    else:
        records = []
        if args.cuad.exists():
            records += cuad_records(args.cuad, args.doc_start, args.doc_end, args.window)
        records += keyphrase_records(args.keyphrase)
    if not records:
        raise SystemExit(1)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
