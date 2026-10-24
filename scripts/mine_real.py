"""Rebuild clean real-corpus training data (leak-audited per journal rules)."""
import json, re, pathlib
from clarke import public

blob = json.load(open("data/raw/CUAD_v1.json"))
docs = blob["data"]
print(f"CUAD blob: {len(docs)} docs")

# --- eval title set (real_eval CUAD) ---
ev = [json.loads(l) for l in open("data/real_eval.jsonl") if json.loads(l)["domain"] == "cuad"]
ev_titles = {i.rsplit("-", 1)[0].removeprefix("cuad-") for i in (r["id"] for r in ev)}
tail_titles = {d["title"][:40] for d in docs[470:]}
print(f"eval titles: {len(ev_titles)}; eval-subset-of-docs[470:]: {ev_titles <= tail_titles}")

# --- typed_test cuad-derived sources (acceptance eval) -> token pairs ---
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def date8_variants(t):
    out = set()
    for m in re.finditer(r"(\d{1,2})[_-](\d{1,2})[_-](\d{4})", t):
        out.add(f"{m.group(3)}{int(m.group(1)):02d}{int(m.group(2)):02d}")
    for m in re.finditer(r"(\d{4})(\d{2})(\d{2})", t):
        out.add(m.group(0))
    return out
def doc_tokens(title):
    return norm(title.split("_")[0].split("-")[0]), date8_variants(title)
def typed_tokens(i):
    s = i.split("typed-cuad-test-", 1)[1] if "typed-cuad-test-" in i else i
    return norm(s.split("_")[0]), date8_variants(s)

tt = [json.loads(l) for l in open("data/typed_test.jsonl")]
tt_cuad = [r["id"] for r in tt if "cuad" in r["id"]]
tt_tok = [typed_tokens(i) for i in tt_cuad]

# --- build windowed records over the FULL train region (docs 0..470) ---
recs = public.cuad_records(pathlib.Path("data/raw/CUAD_v1.json"), doc_start=0, doc_end=470)
clean, drop_eval, drop_tt = [], 0, 0
doc_leaks = set()
for d in docs[:470]:
    (dcomp, ddates) = doc_tokens(d["title"])
    if d["title"][:40] in ev_titles or any(
        dcomp and dcomp == tcomp and ddates & tdates for tcomp, tdates in tt_tok
    ):
        doc_leaks.add(d["title"][:40])
for r in recs:
    title = r["id"].rsplit("-", 1)[0].removeprefix("cuad-")
    if title in ev_titles or title in doc_leaks:
        drop_eval += 1
    else:
        clean.append(r)
print(f"windowed {len(recs)} from docs[0:470] -> clean {len(clean)} (dropped {drop_eval} eval-region windows,"
      f" {len(doc_leaks)} train-region docs leak into typed_test)")
with open("data/cuad_train.jsonl", "w") as h:
    for r in clean: h.write(json.dumps(r) + "\n")

# sanity: zero overlap vs BOTH eval sets by title tokens
for name, pool in (("real_eval", ev_titles), ("typed_test", {norm(r['id']) for r in tt_cuad})):
    inter = [r["id"] for r in clean if r["id"].rsplit("-",1)[0].removeprefix("cuad-") in pool]
    print(f"cuad_train ∩ {name}: {len(inter)} exact-title collisions")

# --- keyphrase: validation parquet only (test parquet IS the eval), namespaced ---
kp = public.keyphrase_records(pathlib.Path("data/raw/keyphrase-validation.parquet"))
for i, r in enumerate(kp):
    r["id"] = f"keyphrase-val-{i}"
test_abs = {r["state"][:120] for r in public.keyphrase_records(pathlib.Path("data/raw/keyphrase-test.parquet"))}
print(f"keyphrase-val: {len(kp)} recs; content-overlap with eval: {sum(1 for r in kp if r['state'][:120] in test_abs)}")
with open("data/keyphrase_train.jsonl", "w") as h:
    for r in kp: h.write(json.dumps(r) + "\n")
