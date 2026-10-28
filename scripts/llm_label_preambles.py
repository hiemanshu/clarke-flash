"""Label preambles (document_name, parties, agreement_date) with an LLM, grounded to spans.

Why: LEDGAR gives 1.3 clause fields per contract; CUAD gives 8.3 fields and 16 instances.
The missing density is concentrated in the preamble, which is cheap for an LLM but has no
rule-based equivalent (party lists are idiosyncratic).

Every returned value is GROUNDED: the model must quote verbatim, and we locate that quote
in the record text to produce char spans. Anything unlocatable is dropped, so a hallucinated
party cannot enter the training data.

Cost: only the first --chars characters are sent (~3k tokens at 12k chars). A pilot of 500
records is ~$0.25 on a flash-tier model; the script refuses to exceed --max-cost.

    uv run python scripts/llm_label_preambles.py --estimate --limit 500
    uv run python scripts/llm_label_preambles.py --limit 50            # pilot
    uv run python scripts/llm_label_preambles.py --limit 5000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"

PROMPT = """Extract three fields from the START of this contract. Quote VERBATIM from the text.

Return STRICT JSON only:
{{"document_name": "<the contract's title, e.g. 'FRANCHISE AGREEMENT'>" or null,
  "parties": ["<each contracting entity, exactly as written>", ...],
  "agreement_date": "<the date the agreement was made>" or null}}

Rules:
- Copy text exactly as it appears; do not paraphrase, expand abbreviations, or add punctuation.
- parties = the entities that are parties to the contract (named companies, individuals,
  defined party terms like "Company"/"Franchisee" when the text defines them). NOT products,
  section headings, or every capitalised phrase.
- If a field is absent from this excerpt, use null (or [] for parties).

TEXT:
{text}"""

PROMPT_V2 = """Extract three fields from the START of this contract. Quote VERBATIM.

Return STRICT JSON only:
{{"document_name": "<the contract's title>" or null,
  "parties": ["<party>", ...],
  "agreement_date": "<date the agreement was made>" or null}}

PARTY RULES (follow exactly):
1. One entry per distinct party - never list the same entity twice, and never list a
   shorter or longer variant of a name you already listed.
2. For each party, give the legal name exactly as written AND, separately, the defined term
   it is assigned in the text (e.g. "Company", "Borrower", "Issuer", "Trustee").
3. EXCLUDE: section headings, product names, collective descriptions ("each Subsidiary
   Guarantor listed on the signature pages"), and role categories ("Lenders", "Banks").
4. Aim for 2-6 parties. Copy text exactly; do not paraphrase or add punctuation.

TEXT:
{text}"""


MARKER = re.compile(r"(?i)(by and between|dated as of|entered into (as of|on)|"
                    r"this (agreement|contract|license) is made)")


def _headers() -> dict:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise SystemExit("set CLOUDFLARE_API_TOKEN (and CLOUDFLARE_ACCOUNT_ID)")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
            "cf-aig-gateway-id": os.environ.get("CLOUDFLARE_GATEWAY", "default")}


def _post(payload: dict, timeout: int = 180) -> dict:
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not acct:
        raise SystemExit("set CLOUDFLARE_ACCOUNT_ID")
    req = urllib.request.Request(ENDPOINT.format(acct=acct),
                                 data=json.dumps(payload).encode(), headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc


def _text_of(resp: dict) -> str:
    if "choices" in resp:
        c = resp["choices"][0].get("message", {}).get("content")
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return c or ""
    return json.dumps(resp)


def _json_of(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def ground(text: str, quote: str) -> tuple[int, int] | None:
    """Locate a verbatim quote, tolerating whitespace differences. None if absent."""
    if not quote:
        return None
    q = " ".join(str(quote).split())
    flat = re.sub(r"\s+", " ", text)
    # build a map from flat index -> original index
    idx = [i for i, ch in enumerate(text) if not ch.isspace()]
    pos = flat.find(q)
    if pos < 0:
        return None
    end = pos + len(q)
    if end - 1 >= len(idx):
        return None
    return idx[pos], idx[end - 1] + 1


PROMPT_V3 = """Extract three fields from the START of this contract. Quote VERBATIM.

Return STRICT JSON only:
{{"document_name": "<the contract's title>" or null,
  "parties": ["<canonical entity name>", ...],
  "agreement_date": "<date the agreement was made>" or null}}

PARTY RULES (follow exactly):
1. List each contracting ENTITY once, using its legal name exactly as written
   (e.g. "SANTANDER CONSUMER USA INC.", "Mimvi, Inc.", "Robert C. Gaydos").
2. Do NOT list defined terms or role words on their own ("Company", "Borrower", "Issuer",
   "Licensor", "Trustee"). A name may be followed by a role in the text, but the entry is
   just the name - never "NAME, as ROLE".
3. Do NOT list collective descriptions ("each Subsidiary Guarantor listed on the signature
   pages") or role categories ("Lenders", "Banks").
4. Never list the same entity twice or a shorter/longer variant of one you already listed.
5. Aim for 2-5 parties. Copy text exactly; no paraphrase.

TEXT:
{text}"""


PROMPT_VERSION = ["v2"]


LIMIT_MARKERS = ("limit", "quota", "rate", "429", "exhaust", "capacity", "overloaded")


def _run_opencode(model: str, prompt: str) -> tuple[str, str]:
    """One CLI call -> (concatenated text, error). Error is '' on success."""
    try:
        proc = subprocess.run(["opencode", "run", "-m", model, "--format", "json", prompt],
                              capture_output=True, text=True, timeout=900)
    except Exception as exc:                                   # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"[:200]
    if proc.returncode != 0:
        return "", f"exit {proc.returncode}: {proc.stderr[:200]}"
    chunks = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "text":
            chunks.append((ev.get("part") or {}).get("text") or "")
        elif ev.get("type") == "error":
            return "", str(ev)[:200]
    return "".join(chunks), ""


def one_opencode(job: dict, model: str, chars: int, fallback: str = "") -> dict:
    """Label through the OpenCode CLI (subscription); fall back on rate limits."""
    rec_id, text = job["id"], job["state"][:chars]
    prompt = {"v1": PROMPT, "v2": PROMPT_V2, "v3": PROMPT_V3}[PROMPT_VERSION[0]].format(text=text)
    t0 = time.time()
    used = model
    raw, err = _run_opencode(model, prompt)
    if err and fallback and any(m in err.lower() for m in LIMIT_MARKERS):
        raw, err = _run_opencode(fallback, prompt)
        used = fallback
    if err:
        return {"id": rec_id, "error": err}
    body = _json_of(raw)
    out = {"id": rec_id, "model": used, "backend": "opencode",
           "seconds": round(time.time() - t0, 1), "fields": {}}
    for field in ("document_name", "agreement_date", "parties"):
        val = body.get(field)
        if field == "parties":
            vals = val if isinstance(val, list) else ([val] if val else [])
            spans, kept = [], []
            for v in vals:
                span = ground(text, v)
                if span:
                    kept.append(str(v)); spans.append(list(span))
            if kept:
                out["fields"]["parties"] = {"value": kept, "spans": spans}
        elif val:
            span = ground(text, val)
            if span:
                out["fields"][field] = {"value": [str(val)], "spans": [list(span)]}
    return out


def one(job: dict, model: str, chars: int) -> dict:
    rec_id, text = job["id"], job["state"][:chars]
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": PROMPT.format(text=text)}]}
    t0 = time.time()
    try:
        resp = _post(payload)
    except Exception as exc:                                  # noqa: BLE001
        return {"id": rec_id, "error": f"{type(exc).__name__}: {exc}"[:200]}
    body = _json_of(_text_of(resp))
    out = {"id": rec_id, "model": model, "seconds": round(time.time() - t0, 1),
           "usage": resp.get("usage", {}), "fields": {}}
    for field in ("document_name", "agreement_date", "parties"):
        val = body.get(field)
        if field == "parties":
            vals = val if isinstance(val, list) else ([val] if val else [])
            spans, kept = [], []
            for v in vals:
                span = ground(text, v)
                if span:
                    kept.append(str(v)); spans.append(list(span))
            if kept:
                out["fields"]["parties"] = {"value": kept, "spans": spans}
        elif val:
            span = ground(text, val)
            if span:
                out["fields"][field] = {"value": [str(val)], "spans": [list(span)]}
    return out


def load_env() -> None:
    """Pick up .env so callers do not have to export gateway credentials by hand."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/ledgar_split_cuad.jsonl")
    ap.add_argument("--out", default="data/llm_preambles.jsonl")
    ap.add_argument("--backend", choices=["cloudflare", "opencode"], default="cloudflare")
    ap.add_argument("--fallback-model", default="",
                    help="model to retry with when the primary reports a rate/quota limit")
    ap.add_argument("--model", default="@cf/deepseek-ai/deepseek-v4.1-flash-1017")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--chars", type=int, default=12000)
    ap.add_argument("--prompt", choices=["v1", "v2", "v3"], default="v3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-cost-per-mtok", type=float, default=0.30,
                    help="USD per 1M input tokens, for the cost guard (default flash-tier)")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    PROMPT_VERSION[0] = args.prompt

    done: set[str] = set()
    if Path(args.out).exists():
        for line in Path(args.out).open():
            try:
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["id"])
            except Exception:
                pass

    jobs = []
    for line in Path(args.input).open():
        if len(jobs) >= args.limit:
            break
        rec = json.loads(line)
        if rec["id"] in done:
            continue
        if MARKER.search(rec["state"][: args.chars]):
            jobs.append({"id": rec["id"], "state": rec["state"]})

    tokens = len(jobs) * (args.chars / 4.0)
    cost = tokens / 1e6 * args.min_cost_per_mtok
    print(f"candidates: {len(jobs):,} (already done: {len(done):,})")
    print(f"est input tokens: {tokens/1e6:.2f}M -> ~${cost:.2f} at "
          f"${args.min_cost_per_mtok}/Mtok")
    if args.estimate:
        return
    if args.backend == "opencode":
        print("(opencode backend: billed to the OpenCode subscription, no gateway cost)")
    if args.backend == "cloudflare" and cost > args.max_cost:
        raise SystemExit(f"estimated ${cost:.2f} exceeds --max-cost ${args.max_cost}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = err = 0
    with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        fn = one_opencode if args.backend == "opencode" else one
        def call(j):
            if args.backend == "opencode":
                return one_opencode(j, args.model, args.chars, args.fallback_model)
            return one(j, args.model, args.chars)
        futures = {pool.submit(call, j): j["id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if "error" in r:
                err += 1
            else:
                ok += 1
            if i % 25 == 0:
                rate = i / max(time.time() - started, 1e-9)
                print(f"  {i}/{len(jobs)}  ok {ok} err {err}  {rate:.2f} rec/s", flush=True)
    print(f"done: {ok} ok, {err} errors -> {args.out}")


if __name__ == "__main__":
    main()
