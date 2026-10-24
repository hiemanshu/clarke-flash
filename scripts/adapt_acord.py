"""ACORD (theatticusproject, CC-BY-4.0) -> clarke typed records.

Attorney queries over real contract clauses with expert relevance grades (0-4,
BEIR format). Window per record: score-4 gold clauses + 3-star hard decoys +
2/0-star filler, shuffled, byte-exact spans; queries lacking a 4-star candidate
become null windows (honest not-present supervision with real near-misses).
Splits are QRELS membership (authoritative; queries.jsonl metadata disagrees on
some ids): train.tsv + valid.tsv -> TRAIN, test.tsv -> held-out EVAL, asserted
disjoint at query level. Clauses overlapping any local eval surface
(real_eval states/docs, typed_test states) are dropped from BOTH sides.

usage (from repo root): uv run python scripts/adapt_acord.py
"""
import json, re
from pathlib import Path

BASE = Path("data/raw/acord_extract/ACORD Dataset & ReadMe (external)")
N_SLOTS = 14
WINDOWS_PER_QUERY_TRAIN = 12
WINDOWS_PER_QUERY_EVAL = 4  # deterministic decoy samples -> stable eval
import random


def norm(s: str) -> str:
    return "".join(s.split()).lower()


def load_contaminated(corpus: dict[str, str], eval_norms: list[str]) -> set[str]:
    """Clauses whose text byte-appears in any eval surface; head/tail 120-char
    fragments catch window/paragraph-truncated overlap."""
    bad = set()
    for cid, text in corpus.items():
        n = norm(text)
        if len(n) < 60:
            continue
        head, tail = n[:120], n[-120:]
        for ev in eval_norms:
            if n in ev or head in ev or tail in ev:
                bad.add(cid)
                break
    return bad


def load_qrels(split: str) -> dict[str, list[tuple[str, int]]]:
    import csv
    byq: dict[str, list[tuple[str, int]]] = {}
    with open(BASE / "qrels" / f"{split}.tsv", newline="") as h:
        for row in csv.reader(h, delimiter="\t", quotechar='"'):
            if not row or row[0] == "query-id":
                continue
            byq.setdefault(row[0], []).append((row[1], int(row[2])))
    return byq


def make_windows(query_ids, byq, queries, corpus, tag, rng, per_q, out) -> int:
    n = 0
    for qid in query_ids:
        q = queries[qid]
        cand = byq[qid]
        gold = [c for c, s in cand if s == 4]
        hard = [c for c, s in cand if s == 3]
        mid = [c for c, s in cand if s == 2]
        easy = [c for c, s in cand if s in (0, 1)]
        gold = [c for c in gold if c not in BAD]
        hard = [c for c in hard if c not in BAD]
        mid = [c for c in mid if c not in BAD]
        easy = [c for c in easy if c not in BAD]
        for w in range(per_q):
            rng.shuffle(hard); rng.shuffle(mid); rng.shuffle(easy)
            if gold:
                picks = gold + hard[:4] + mid[:2]
                picks += easy[: max(0, N_SLOTS - len(picks))]
            else:  # honest null window: decoys only, no true answer present
                picks = hard[:4] + mid[:3] + easy[: N_SLOTS - 7]
            chosen = picks[:N_SLOTS]
            rng.shuffle(chosen)
            state, spans = "", []
            for i, cid in enumerate(chosen):
                header = f"CLAUSE {i + 1}: "
                body = corpus[cid].strip()
                start = len(state) + len(header)
                state += header + body + "\n\n"
                if cid in gold:
                    spans.append([start, start + len(body)])
            name = re.sub(r"[^a-z0-9]+", "_", q["text"].lower())[:60].strip("_") or "query"
            vals = [state[s:e] for s, e in spans]
            out.append({
                "id": f"acord-{tag}-{qid[:32].replace(' ', '-').replace(chr(34), '')}-w{w}",
                "domain": "acord",
                "state": state,
                "fields": {name: {"type": "list", "items": "text",
                                  "instructions": q["text"], "required": True}},
                "answers": {name: {"value": None} if not vals else
                            {"value": vals, "spans": spans}},
            })
            n += 1
    return n


BAD: set[str] = set()


def main() -> None:
    global BAD
    corpus = {c["_id"]: c["text"] for c in map(json.loads, open(BASE / "corpus.jsonl"))}
    queries = {q["_id"]: q for q in map(json.loads, open(BASE / "queries.jsonl"))}
    train_q = load_qrels("train")
    valid_q = load_qrels("valid")
    test_q = load_qrels("test")

    tr_ids = list(dict.fromkeys(list(train_q) + list(valid_q)))
    leaked = set(tr_ids) & set(test_q)
    assert not leaked, f"qrels train/valid and test share {len(leaked)} queries: {list(leaked)[:3]}"
    # merge candidate lists per query (train+valid sides never overlap with test ids)
    byq: dict[str, list[tuple[str, int]]] = {}
    for src in (train_q, valid_q, test_q):
        for q, lst in src.items():
            byq.setdefault(q, []).extend(lst)
    print(f"[acord] qrels: train {len(train_q)}q / valid {len(valid_q)}q / test {len(test_q)}q; "
          f"corpus {len(corpus)}, queries.jsonl {len(queries)}")

    # contamination pool: real_eval states + typed_test states + eval-CUAD full docs
    ev_norms = [norm(json.loads(l)["state"]) for l in open("data/real_eval.jsonl")]
    ev_norms += [norm(json.loads(l)["state"]) for l in open("data/typed_test.jsonl")]
    titles = {r["id"].rsplit("-", 1)[0].removeprefix("cuad-")
              for r in map(json.loads, open("data/real_eval.jsonl")) if r["domain"] == "cuad"}
    for d in json.load(open("data/raw/CUAD_v1.json"))["data"]:
        if d["title"][:40] in titles:
            ev_norms.append(norm("\n".join(p.get("context", "") for p in d.get("paragraphs", []))))
    BAD = load_contaminated(corpus, ev_norms)
    print(f"[acord] dropping {len(BAD)} contaminated clauses from both sides")

    train_recs, eval_recs = [], []
    n_tr = make_windows(tr_ids, byq, queries, corpus, "train",
                        random.Random(20261024), WINDOWS_PER_QUERY_TRAIN, train_recs)
    n_te = make_windows(list(test_q), byq, queries, corpus, "test",
                        random.Random(777), WINDOWS_PER_QUERY_EVAL, eval_recs)
    nulls = sum(1 for r in train_recs
                if r["answers"][next(iter(r["answers"]))]["value"] is None)
    gold_fields = sum(1 for r in train_recs
                      if r["answers"][next(iter(r["answers"]))]["value"] is not None)
    print(f"[acord] train: {n_tr} windows ({gold_fields} answerable, {nulls} null) | "
          f"eval: {n_te} windows (HOLDOUT, never train on this)")
    with open("data/acord_train.jsonl", "w") as h:
        for r in train_recs: h.write(json.dumps(r) + "\n")
    with open("data/acord_test.jsonl", "w") as h:
        for r in eval_recs: h.write(json.dumps(r) + "\n")
    # BEIR-honest leak check: no training window may pair a TEST query's instruction
    # with its gold (query-level disjointness already asserted; verify on records):
    te_texts = {queries[q]["text"].lower() for q in test_q}
    pair_hits = sum(1 for r in train_recs
                    if next(iter(r["fields"])) and
                    (lambda f: f)(r["fields"][next(iter(r["fields"]))])["instructions"].lower() in te_texts)
    shared = sum(1 for r in eval_recs for v in (r["answers"][next(iter(r["answers"]))]["value"] or [])
                 if norm(v)[:200] in "||".join(norm(x["state"]) for x in train_recs))
    print(f"[acord] train-windows pairing a TEST query: {pair_hits} (must be 0); "
          f"test-gold clauses seen elsewhere in train states: {shared} (legit: shared corpus, BEIR)")


if __name__ == "__main__":
    main()
