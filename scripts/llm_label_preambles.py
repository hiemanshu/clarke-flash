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

Backends:
  cloudflare  metered gateway, OpenAI-compatible
  opencode    subscription CLI (opencode run -m provider/model)
  openrouter  pay per token, OpenAI-compatible; default model
              moonshotai/kimi-k3.1-pro, priced from the PRICING table below so
              --max-cost is a real guard rather than an input-only estimate
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


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter_headers() -> dict:
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise SystemExit("set OPENROUTER_API_KEY (in .env or the environment)")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # OpenRouter uses these for attribution/rate-limit tiers; both are optional.
    if os.environ.get("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
    if os.environ.get("OPENROUTER_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
    return headers


def _post_openrouter(payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(OPENROUTER_ENDPOINT,
                                 data=json.dumps(payload).encode(),
                                 headers=_openrouter_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"openrouter error: {json.dumps(out['error'])[:300]}")
    return out


def _extract_fields(text: str, body: dict, out: dict) -> dict:
    """Ground each returned value to a verbatim span; drop anything unlocatable.

    Shared by every backend so the grounding rule cannot drift between them: a
    hallucinated party must never reach the training data.
    """
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


def one_openrouter(job: dict, model: str, chars: int, fallback: str = "") -> dict:
    """Label through OpenRouter (pay per token). Retries 429/5xx with backoff."""
    rec_id, text = job["id"], job["state"][:chars]
    prompt = {"v1": PROMPT, "v2": PROMPT_V2, "v3": PROMPT_V3}[PROMPT_VERSION[0]].format(text=text)
    t0 = time.time()
    used, last = model, ""
    resp = None
    for attempt in range(3):
        try:
            resp = _post_openrouter({"model": used, "temperature": 0,
                                     "messages": [{"role": "user", "content": prompt}]})
            raw = _text_of(resp)
            if not raw.strip():
                raise RuntimeError(f"empty content: {json.dumps(resp)[:200]}")
            break
        except Exception as exc:                              # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"[:200]
            if fallback and any(m in last.lower() for m in LIMIT_MARKERS):
                used = fallback
            if attempt == 2:
                return {"id": rec_id, "error": last}
            time.sleep(2 ** attempt)
    out = {"id": rec_id, "model": used, "backend": "openrouter",
           "seconds": round(time.time() - t0, 1), "usage": (resp or {}).get("usage", {}),
           "fields": {}}
    return _extract_fields(text, _json_of(raw), out)


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


PROMPT_VERSION = ["v3"]

# (input USD / 1M tokens, output USD / 1M tokens) for the cost guard.
# openrouter kimi-k3.1-pro: $2 in / $10 out / $0.20 cached (Moonshot list price).
PRICING = {"cloudflare": (0.30, 0.0), "opencode": (0.0, 0.0), "openrouter": (2.0, 10.0)}
DEFAULT_MODEL = {"cloudflare": "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
                 "opencode": "opencode-go/kimi-k3.1-pro",
                 "openrouter": "moonshotai/kimi-k3.1-pro"}


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
    out = {"id": rec_id, "model": used, "backend": "opencode",
           "seconds": round(time.time() - t0, 1), "fields": {}}
    return _extract_fields(text, _json_of(raw), out)


def one(job: dict, model: str, chars: int) -> dict:
    rec_id, text = job["id"], job["state"][:chars]
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": PROMPT.format(text=text)}]}
    t0 = time.time()
    try:
        resp = _post(payload)
    except Exception as exc:                                  # noqa: BLE001
        return {"id": rec_id, "error": f"{type(exc).__name__}: {exc}"[:200]}
    out = {"id": rec_id, "model": model, "seconds": round(time.time() - t0, 1),
           "usage": resp.get("usage", {}), "fields": {}}
    return _extract_fields(text, _json_of(_text_of(resp)), out)


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
    ap.add_argument("--backend", choices=["cloudflare", "opencode", "openrouter"],
                    default="cloudflare")
    ap.add_argument("--fallback-model", default="",
                    help="model to retry with when the primary reports a rate/quota limit")
    ap.add_argument("--model", default=None,
                    help="defaults per backend: cloudflare flash, opencode-go kimi-k3.1-pro, "
                         "openrouter moonshotai/kimi-k3.1-pro")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--chars", type=int, default=12000)
    ap.add_argument("--prompt", choices=["v1", "v2", "v3"], default="v3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-cost-per-mtok", type=float, default=None,
                    help="USD per 1M input tokens; default comes from PRICING[backend]")
    ap.add_argument("--output-cost-per-mtok", type=float, default=None,
                    help="USD per 1M output tokens; default comes from PRICING[backend]")
    ap.add_argument("--stop-after-errors", type=int, default=5,
                    help="abort after N consecutive failed calls (a plan block returns the "
                         "same error for every record; without this the run writes thousands "
                         "of error rows before anyone notices)")
    ap.add_argument("--assumed-output-tokens", type=int, default=300,
                    help="output tokens per record for the estimate (JSON + optional reasoning)")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    PROMPT_VERSION[0] = args.prompt
    if args.model is None:
        args.model = DEFAULT_MODEL[args.backend]
    in_rate, out_rate = PRICING[args.backend]
    if args.min_cost_per_mtok is not None:
        in_rate = args.min_cost_per_mtok
    if args.output_cost_per_mtok is not None:
        out_rate = args.output_cost_per_mtok

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
    out_tokens = len(jobs) * args.assumed_output_tokens
    cost = tokens / 1e6 * in_rate + out_tokens / 1e6 * out_rate
    print(f"candidates: {len(jobs):,} (already done: {len(done):,})")
    print(f"est input {tokens/1e6:.2f}M @ ${in_rate}/M + output {out_tokens/1e6:.2f}M "
          f"@ ${out_rate}/M -> ~${cost:.2f}")
    if args.estimate:
        return
    if args.backend == "opencode":
        print("(opencode backend: billed to the OpenCode subscription, no gateway cost)")
    elif args.backend == "openrouter":
        print(f"(openrouter backend: pay per token, model {args.model})")
    if args.backend != "opencode" and cost > args.max_cost:
        raise SystemExit(f"estimated ${cost:.2f} exceeds --max-cost ${args.max_cost}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = err = 0
    with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def call(j):
            if args.backend == "opencode":
                return one_opencode(j, args.model, args.chars, args.fallback_model)
            if args.backend == "openrouter":
                return one_openrouter(j, args.model, args.chars, args.fallback_model)
            return one(j, args.model, args.chars)
        futures = {pool.submit(call, j): j["id"] for j in jobs}
        consec = 0
        stopped = False
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if "error" in r:
                err += 1
                consec += 1
                if consec >= args.stop_after_errors:
                    blocked = any(m in str(r["error"]).lower() for m in LIMIT_MARKERS)
                    print(f"STOP: {consec} consecutive errors "
                          f"({'quota/limit' if blocked else 'unknown'}); last: {r['error']}",
                          flush=True)
                    stopped = True
                    break
            else:
                ok += 1
                consec = 0
            if i % 25 == 0:
                rate = i / max(time.time() - started, 1e-9)
                print(f"  {i}/{len(jobs)}  ok {ok} err {err}  {rate:.2f} rec/s", flush=True)
        if stopped:
            pool.shutdown(wait=False, cancel_futures=True)
    print(f"{'stopped' if err and stopped else 'done'}: {ok} ok, {err} errors -> {args.out}")


if __name__ == "__main__":
    main()
