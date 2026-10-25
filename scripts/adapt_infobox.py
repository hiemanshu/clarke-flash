"""enwiki current-content XML dump -> human-metadata BINDING records.

Raw wikitext carries several human-curated metadata families; each becomes a
typed field on the article's record (one state, mixed schema ~ CUAD-shaped):
  - infobox params   | key = value      -> text; present-but-empty -> null
  - {{Short description|X}}            -> text; ABSENT on ~57% pages -> native null
  - [[Category:X]] links               -> list<text> of curated classifications
  - {{about|X|...}} hatnotes           -> text; absent -> null (only on ~4% pages)
  - absent-param nulls sampled from the global param pool (cross-type near-misses)
All gold = byte-exact spans into the state; markup ([[links]], italics) stays
verbatim - copying messy grounded text IS the task. Redirects skipped; pages
overlapping wiki_test (url-hash) dropped; page-level eval split by id hash.

usage: uv run python scripts/adapt_infobox.py data/raw/enwiki_*.xml.bz2
"""
import bz2, hashlib, html, json, random, re, sys
from collections import defaultdict
from pathlib import Path

MAX_STATE = 6000
MAX_FIELDS = 16
NULL_TARGET = 0.40
EVAL_HASH_MOD = 6
RECORD_CAP = 180_000

IB_START = re.compile(r"\{\{\s*[Ii]nfo[Bb]ox\s+([^\s|]{1,40})")
PARAM = re.compile(r"^\|\s*([a-z][a-z0-9_\-]{1,30})\s*=\s*(.*)$")
SD = re.compile(r"\{\{[Ss]hort description\|(.{1,200}?)\}\}")
HAT = re.compile(r"\{\{(?:about|dablink|hatnote)\|([^|}\]]{3,120})")
CAT = re.compile(r"\[\[[Cc]ategory:([^\]\|#]{2,80})")


def infobox_block(text):
    m = IB_START.search(text)
    if not m:
        return None
    start = m.start()
    depth, i = 0, start
    while i < len(text) - 1:
        two = text[i:i + 2]
        if two == "{{":
            depth += 1; i += 2
        elif two == "}}":
            depth -= 1
            if depth == 0:
                return m.group(1).strip().lower(), start, i + 2
            i += 2
        else:
            i += 1
    return None


def parse_pages(path):
    import io
    if str(path).endswith(".xml"):
        f = open(path, encoding="utf-8", errors="replace")
        read = lambda: f.read(1 << 24)
    else:
        bh = bz2.BZ2File(path)
        f = io.TextIOWrapper(bh, encoding="utf-8", errors="replace")
        def read():
            try:
                return f.read(1 << 22)
            except EOFError:
                return None
    with f:
        buf = ""
        while True:
            chunk = read()
            if not chunk:
                break
            # split ONCE per chunk (C-speed); the old per-page split copied the
            # whole 4-16MB buffer for every page (~70TB of memcpy over the corpus)
            parts = (buf + chunk).split("</page>")
            buf = parts.pop()
            for page in parts:
                if "<page>" in page:
                    yield page.split("<page>", 1)[1]


def record_from_page(page, type_vocab, rng):
    tm = re.search(r"<title>([^<]+)</title>", page)
    nm = re.search(r"<ns>(\d+)</ns>", page)
    tx = re.search(r"<text[^>]*>(.*?)</text>", page, re.S)
    if not (tm and tx) or (nm and nm.group(1) != "0"):
        return None
    title = html.unescape(tm.group(1))
    text = html.unescape(tx.group(1))
    if not text.strip() or text.lstrip().startswith(("#REDIRECT", "#redirect")):
        return None
    state = text[:MAX_STATE]

    # --- infobox params ---
    present, ib_type = {}, None
    ib = infobox_block(text)
    if ib:
        ib_type, b0, b1 = ib
        block = text[b0:b1]
        depth, pos = 0, b0
        for line in block.split("\n"):
            if depth == 1:
                m = PARAM.match(line)
                if m:
                    name, raw = m.group(1), m.group(2).strip()
                    if "{{" not in raw:
                        if raw == "":
                            present.setdefault(name, None)
                        else:
                            rv = raw.rstrip()
                            eq = line.find("=")
                            lead = len(line[eq + 1:]) - len(line[eq + 1:].lstrip())
                            s = pos + eq + 1 + lead
                            e = s + len(rv)
                            if e <= MAX_STATE and text[s:e] == rv:
                                present.setdefault(name, (s, e))
            depth += line.count("{{") - line.count("}}")
            depth = max(depth, 0)
            pos += len(line) + 1
        if ib_type:
            type_vocab[ib_type].update(present.keys())

    answered = [(k, v) for k, v in present.items() if v]

    # --- short description: present -> text field; absent -> TRUE null field ---
    sd = SD.search(state)
    sd_field = None
    if sd and sd.end() <= MAX_STATE:
        s, e = sd.span(1)
        sd_field = ("short_description", {"type": "text",
            "instructions": "what is this article's short description?"},
            {"value": state[s:e], "span": [s, e]})
    else:
        sd_field = ("short_description", {"type": "text",
            "instructions": "what is this article's short description?"}, {"value": None})

    # --- categories: list<text> of human-curated classifications ---
    cats = []
    for m in CAT.finditer(state):
        s, e = m.span(1)
        cats.append({"value": state[s:e], "spans": [[s, e]]})
        if len(cats) >= 6:
            break
    cat_field = None
    if cats:
        names = [c["value"] for c in cats]
        cat_field = ("categories", {"type": "list", "items": "text",
            "instructions": "what topic categories has this article been classified into?"},
            {"value": names, "spans": [c["spans"][0] for c in cats]})

    # --- hatnote topic (rare; only add when present) ---
    hat_field = None
    hb = HAT.search(state)
    if hb and hb.end(1) <= MAX_STATE:
        s, e = hb.span(1)
        hat_field = ("hatnote_topic", {"type": "text",
            "instructions": "what does the article's hatnote say this page is about?"},
            {"value": state[s:e], "span": [s, e]})

    # --- null padding: params from the global pool not stated here ---
    empty_nulls = [k for k, v in present.items() if v is None]
    extras = [x for x in (sd_field, cat_field, hat_field) if x]
    if len(answered) + sum(1 for x in extras if x[2]["value"] is not None) < 2:
        return None
    n_kept = min(len(answered), MAX_FIELDS - len(extras) - len(empty_nulls) - 1)
    answered = answered[:n_kept]
    pool = set().union(*type_vocab.values()) if type_vocab else set()
    cand = sorted(pool - set(present.keys()) - {"short description"})
    null_budget = max(1, min(MAX_FIELDS - n_kept - len(extras) - len(empty_nulls),
                             int((n_kept + len(extras)) * NULL_TARGET / (1 - NULL_TARGET))))
    nulls = (rng.sample(cand, min(null_budget, len(cand))) if cand else []) + empty_nulls

    fields, answers, used = {}, {}, set()
    def add(base, spec, ans):
        nm2 = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")[:40] or "field"
        while nm2 in used:
            nm2 += "_x"
        used.add(nm2)
        fields[nm2] = spec
        answers[nm2] = ans
    for k, (s, e) in answered:
        add(k, {"type": "text",
                "instructions": f"what is the {k.replace('_',' ')} of this subject, as stated in the article's infobox?"},
            {"value": state[s:e], "span": [s, e]})
    for name, spec, ans in extras:
        add(name, spec, ans)
    for k in nulls:
        if len(fields) >= MAX_FIELDS:
            break
        add(k + "_absent", {"type": "text",
            "instructions": f"what is the {k.replace('_',' ')} of this subject, as stated in the article's infobox?"},
            {"value": None})
    if len(fields) < 3:
        return None
    url = f"https://en.wikipedia.org/wiki/{title}"
    hh = hashlib.md5(url.encode()).hexdigest()[:12]
    return {"id": f"wikiib-{hh}", "domain": "wikimeta", "state": state,
            "fields": fields, "answers": answers, "_eval": int(hh, 16) % EVAL_HASH_MOD == 0}


def main():
    import sys as _s
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent))
    import cache as C
    def opt(name, default=None):
        return _s.argv[_s.argv.index(name) + 1] if name in _s.argv else default
    seed = int(opt("--seed", "20261024"))
    out_dir = P(opt("--out-dir", "data/cache/wikiib"))
    skip = {"--seed", "--out-dir"}
    args = []
    i = 1
    while i < len(_s.argv):
        a = _s.argv[i]
        if a in skip:
            i += 2; continue
        if a.startswith("--"):
            i += 1; continue
        args.append(a); i += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    converter_sha = C.sha_text(open(__file__).read())
    # exclusion set: article-hash forms overlapping any eval surface (persisted)
    excl_file = P("data/cache/exclusions/eval_article_hashes.json")
    ev_hashes = set(json.loads(excl_file.read_text())) if excl_file.exists() else set()
    for path in args:
        src = P(path)
        stem = src.name.replace(".xml.bz2", "").replace(".xml", "")
        out = out_dir / f"{stem}.jsonl"
        if C.is_current(out, converter_sha, seed, src):
            print(f"[cache] SKIP {stem} (manifest current)")
            continue
        rng = random.Random(f"{seed}:{stem}")  # per-shard determinism
        type_vocab = defaultdict(set)
        seen, recs = set(), []
        stats = defaultdict(int)
        for page in parse_pages(src):
            stats["pages"] += 1
            r = record_from_page(page, type_vocab, rng)
            if not r:
                stats["no_gold_or_skip"] += 1
                continue
            hh = r["id"].split("-")[1]
            if hh in ev_hashes or hh in seen:
                stats["contaminated_or_dup"] += 1
                continue
            seen.add(hh)
            recs.append(r)
            stats["records"] += 1
        bad = 0
        for r in recs:
            for spec, a in zip(r["fields"].values(), r["answers"].values()):
                if spec["type"] == "text" and a.get("span"):
                    s_, e_ = a["span"]
                    bad += r["state"][s_:e_] != a["value"]
                if spec["type"] == "list" and a.get("spans"):
                    for v, (s_, e_) in zip(a["value"], a["spans"]):
                        bad += r["state"][s_:e_] != v
        assert bad == 0, f"{bad} span failures in {stem}"
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w") as h:
            for r in recs:
                h.write(json.dumps(r) + "\n")
        tmp.rename(out)
        C.write_manifest(out, converter_sha, seed, [src], len(recs))
        ev = sum(1 for r in recs if r.get("_eval"))
        print(f"[cache] MINED {stem}: {len(recs)} records ({ev} flagged eval), "
              f"types={len(type_vocab)}, pages={stats['pages']}")


if __name__ == "__main__":
    main()
