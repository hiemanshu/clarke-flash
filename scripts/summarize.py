"""Compose long-form milestone summaries for Signal notifications.

Signal can carry multi-KB messages; the owner wants the full read there, with
ntfy as a short 'check Signal' buzz. Each function returns a plain-text body.

usage: uv run python scripts/summarize.py batch3|expreal2|census [log-tail]
"""
import json
import sys
from pathlib import Path

REP = Path("reports")


def _m(path, *keys, default="n/a"):
    try:
        d = json.loads((REP / path).read_text())
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default


def _f(x, spec=":.3f"):
    try:
        return format(float(x), spec)
    except Exception:
        return "n/a"


def expreal2() -> str:
    r = _m("expReal2-real.json", "by_domain", "cuad", "list-text", default={})
    a = _m("expReal2-acordtest.json", "metrics", "list-text", default={})
    t = _m("expReal2-typed.json", "metrics", default={})
    cv = _m("expReal2-contractval.json", "metrics", default={})
    v3 = _m("expReal2-val3.json", "metrics", default={})
    pm = _m("expReal2-pm-real.json", "postmortem", "cuad", "list-text", default={})
    return f"""expReal2 VERDICT (real contract data)
CUAD list-text exact {_f(r.get('exact'))}  (controls: expMix .097, best .156)
CUAD null_ok {_f(r.get('null_correct'))}  (was .875)
pm-real median gold rank {pm.get('median_rank','n/a')} (was 71), p@1 {_f(pm.get('p@1'))}, p@5 {_f(pm.get('p@5'))}
ACORD holdout exact {_f(a.get('exact'))} (pre-train baseline .009), null_ok {_f(a.get('null_correct'))}
typed: text {_f((t.get('text') or {}).get('exact'))}, number {_f((t.get('number') or {}).get('exact'))}, list-text {_f((t.get('list-text') or {}).get('exact'))}
contract_val text {_f((cv.get('text') or {}).get('exact'))}, number {_f((cv.get('number') or {}).get('exact'))}
val3: text {_f((v3.get('text') or {}).get('exact'))}, number {_f((v3.get('number') or {}).get('exact'))}, date {_f((v3.get('date') or {}).get('exact'))}

VERDICT: real data is the lever. CUAD 3.9x. Init rule fired -> batch3 starts from checkpoint-7658.
Caveats: exact = precision on ATTEMPTED answers (false nulls uncounted); the bundle
includes 2 extra epochs from expMix-6852 plus 2 real corpora - single-variable split
was traded away deliberately. Full detail: JOURNAL.md, reports/expReal2-*.json"""


def batch3() -> str:
    real = _m("batch3-real.json", "by_domain", default={})
    typed = _m("batch3-typed.json", "metrics", default={})
    acord = _m("batch3-acordtest.json", "metrics", default={})
    wtab = _m("batch3-wikitable.json", "metrics", default={})
    whyb = _m("batch3-wikihybrid.json", "metrics", default={})
    cv = _m("batch3-contractval.json", "metrics", default={})
    v3 = _m("batch3-val3.json", "metrics", default={})
    pm = _m("batch3-pm-real.json", "postmortem", "cuad", "list-text", default={})
    cuad = (real.get("cuad") or {}).get("list-text") or {}
    key = (real.get("keyphrase") or {}).get("list-text") or {}
    lt = wtab.get("list-text") or {}
    hl = whyb.get("list-text") or {}
    ttext = typed.get("text") or {}
    tnum = typed.get("number") or {}
    tlt = typed.get("list-text") or {}
    cvt = cv.get("text") or {}
    vn = v3.get("number") or {}
    vt = v3.get("text") or {}
    return f"""BATCH 3 VERDICT (web-scale real data on top of the real-contract stack)

HEADLINE
CUAD list-text exact {_f(cuad.get('exact'))} (prev expReal2: .381, expMix .097, all-time best .381)
CUAD null_ok {_f(cuad.get('null_correct'))}
pm-real median gold rank {pm.get('median_rank','n/a')} (prev 8), p@5 {_f(pm.get('p@5'))}

IN-DOMAIN WIKI (does the new regime learn?)
structured-wiki tables  list-text {_f(lt.get('exact'))}  n={lt.get('n','?')}  null_ok {_f(lt.get('null_correct'))}
hybrid table holdout    list-text {_f(hl.get('exact'))}  n={hl.get('n','?')}
ACORD holdout           exact {_f(acord.get('exact'))} (prev .048)

RETENTION
typed: text {_f(ttext.get('exact'))} / number {_f(tnum.get('exact'))} / list-text {_f(tlt.get('exact'))}
contract_val text {_f(cvt.get('exact'))} | val3 text {_f(vt.get('exact'))}, number {_f(vn.get('exact'))}
keyphrase {_f(key.get('exact'))} (n=20, noisy)

HOW TO READ
- wiki in-domain high + CUAD flat  -> data works but is regime-specific; scale it
- wiki in-domain low               -> converter/sampling problem, inspect a sample
- CUAD up again                    -> real data compounds; keep scaling toward the 80% bar
- CUAD down                        -> dilution or interference; report and pause scaling
Full detail: JOURNAL.md, reports/batch3-*.json, /tmp/opencode/batch3.log"""


def compare(a: str = "batch3", b: str = "batch3-4b") -> str:
    """Side-by-side of two runs on identical eval files (9B vs 4B scaling probe)."""
    def grab(prefix):
        cuad = (_m(f"{prefix}-real.json", "by_domain", "cuad", "list-text", default={})
                or {})
        typed = _m(f"{prefix}-typed.json", "metrics", default={}) or {}
        acord = _m(f"{prefix}-acordtest.json", "metrics", "list-text", default={}) or {}
        wt = _m(f"{prefix}-wikitable.json", "metrics", default={}) or {}
        wiki = wt.get("text") or wt.get("list-text") or {}
        pm = _m(f"{prefix}-pm-real.json", "postmortem", "cuad", "list-text", default={}) or {}
        cv = _m(f"{prefix}-contractval.json", "metrics", "text", default={}) or {}
        v3 = _m(f"{prefix}-val3.json", "metrics", default={}) or {}
        return {
            "CUAD": _f(cuad.get("exact")), "null_ok": _f(cuad.get("null_correct")),
            "rank": pm.get("median_rank", "n/a"),
            "typed_text": _f((typed.get("text") or {}).get("exact")),
            "typed_num": _f((typed.get("number") or {}).get("exact")),
            "typed_lt": _f((typed.get("list-text") or {}).get("exact")),
            "acord": _f(acord.get("exact")),
            "wiki_tbl": _f(wiki.get("exact")),
            "contract": _f(cv.get("exact")),
            "val3_text": _f((v3.get("text") or {}).get("exact")),
        }
    A, B = grab(a), grab(b)
    keys = ["CUAD", "null_ok", "rank", "typed_text", "typed_num", "typed_lt",
            "acord", "wiki_tbl", "contract", "val3_text"]
    rows = "\n".join(f"{k:12s} {str(A[k]):>8s}   {str(B[k]):>8s}" for k in keys)
    return f"""SCALING PROBE: {a} (9B) vs {b} (4B) - identical mix/eval files

{"metric":12s} {"9B":>8s}   {"4B":>8s}
{rows}

READ: if 4B is within a few points of 9B, backbone capacity is NOT the lever and
we scale data/context instead. If 4B lags badly, capacity matters and a bigger
GPU + bigger backbone is justified. Caveat: the 9B run was warm-started from
checkpoint-7658 (2 extra epochs of history); the 4B run starts from a fresh head,
so it runs with a handicap - a near-tie is a STRONG result for 4B.
Detail: reports/{a}-*.json vs reports/{b}-*.json, JOURNAL.md"""


def census() -> str:
    try:
        shards = [json.loads(l) for l in open("reports/enwiki_census.jsonl")]
    except Exception:
        shards = []
    if not shards:
        return "Census finished (per-shard detail in reports/census/)."
    tot = sum(s["pages"] for s in shards)
    return f"""Wiki corpus census complete: {len(shards)}/19 shards, {tot:,} pages.
~61% are redirects -> ~{int(tot*0.39/1e6)}M real articles.
Families: infobox 24%, shortdesc 26%, categories 38%, in-prose tables 6.9%,
citations 29%, life-date templates 16%.
Full table: reports/census/, JOURNAL.md."""


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "batch3"
    fn = {"batch3": batch3, "expreal2": expreal2, "census": census,
          "compare": compare}.get(what)
    print(fn() if fn else "unknown summary")
