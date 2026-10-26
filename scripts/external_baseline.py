"""External-model baseline on our eval files, scored with our own metric logic.

Two access paths:

1) Cloudflare AI Gateway (preferred for chat models) - one OpenAI-compatible endpoint:
     https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/compat/chat/completions
   header: cf-aig-authorization: Bearer <CF_AIG_TOKEN>
   model:  "{provider}/{model}"  e.g. openai/gpt-6-luna, anthropic/claude-opus-5-5
   Provider keys either stored in the gateway (BYOK) or passed as Authorization.

2) Direct provider APIs (fallback / for Clairic):
   - Workers AI run endpoint for @cf/cloudflare/clairic (native state+fields schema!)
   - OpenAI /v1/chat/completions
   - Anthropic /v1/messages

Clairic takes our schema DIRECTLY (state + typed fields -> typed answers), so it is the
only true 1:1 comparison. Chat models get a JSON contract and we locate their strings in
the state ourselves (never trust model-supplied offsets).

Environment (never written to disk here):
  CF_AIG_TOKEN, CLOUDFLARE_ACCOUNT_ID        (gateway; CLOUDFLARE_API_TOKEN for direct WA)
  OPENAI_API_KEY / ANTHROPIC_API_KEY         (or rely on gateway-stored BYOK keys)

Usage:
  # Clairic through Workers AI (direct)
  uv run python scripts/external_baseline.py --provider cloudflare \
      --model @cf/cloudflare/clairic --data data/real_eval.jsonl --limit 148 \
      --out reports/external-clairic.json

  # GPT through AI Gateway
  uv run python scripts/external_baseline.py --provider openai --model gpt-6-luna \
      --gateway my-gateway --data data/real_eval.jsonl --limit 148 \
      --out reports/external-gpt.json

  # Opus through AI Gateway
  uv run python scripts/external_baseline.py --provider anthropic --model claude-opus-5-5 \
      --gateway my-gateway --data data/real_eval.jsonl --limit 148 \
      --out reports/external-opus.json

  # validate payload + endpoint without spending money
  ... --dry-run --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env without overriding exported variables.

    Dependency-free on purpose; .env is gitignored, .env.example documents it.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


# ---------------------------------------------------------------- schema mapping

def fields_for_record(record: dict) -> dict:
    out = {}
    for name, spec in record["fields"].items():
        f = {"type": spec.get("type", "text")}
        for k in ("items", "instructions", "integer", "decimals", "unit",
                  "min", "max", "step", "scale"):
            if k in spec:
                f[k] = spec[k]
        out[name] = f
    return out


def json_instruction(fields: dict) -> str:
    spec = json.dumps({n: {k: v for k, v in f.items() if k in ("type", "items", "instructions")}
                       for n, f in fields.items()}, indent=1)
    return (
        "Extract the requested fields from the STATE below. Reply with a single JSON object "
        "mapping each field name to either an array of EXACT substrings copied verbatim from the "
        "state, or null when the field is not present. Do not paraphrase; do not add prose "
        "outside the JSON.\n\nFIELDS:\n" + spec
    )


# ---------------------------------------------------------------- HTTP

def _post_json(url: str, headers: dict, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}") from e


def extract_json_block(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- provider calls

def call_cloudflare(record, fields, model, sleep_s, base=None):
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not acct or (not token and not base):
        raise SystemExit("set CLOUDFLARE_ACCOUNT_ID (and CLOUDFLARE_API_TOKEN for direct Workers AI)")
    url = (base or f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run") .rstrip("/")
    url = f"{url}/{model}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = _post_json(url, headers, {"state": record["state"], "fields": fields})
    time.sleep(sleep_s)
    if not resp.get("success", True):
        raise RuntimeError(f"cloudflare error: {json.dumps(resp)[:400]}")
    ans = resp.get("result", {}).get("answers", resp.get("answers", {}))
    return {k: (v.get("value") if isinstance(v, dict) else v) for k, v in ans.items()}


def call_chat_compat(url, headers, model, record, fields, sleep_s, max_tokens=2000):
    body = {"model": model,
            "messages": [{"role": "system", "content": json_instruction(fields)},
                         {"role": "user", "content": "STATE:\n" + record["state"]}],
            "max_tokens": max_tokens, "temperature": 0}
    resp = _post_json(url, headers, body)
    time.sleep(sleep_s)
    text = resp["choices"][0]["message"]["content"]
    return extract_json_block(text)


def call_anthropic_direct(record, fields, model, sleep_s):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set ANTHROPIC_API_KEY (or use --gateway)")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    body = {"model": model, "max_tokens": 2000,
            "system": json_instruction(fields),
            "messages": [{"role": "user", "content": "STATE:\n" + record["state"]}],
            "temperature": 0}
    resp = _post_json(base + "/messages",
                      {"x-api-key": key, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}, body)
    time.sleep(sleep_s)
    return extract_json_block("".join(b.get("text", "") for b in resp.get("content", [])))


# ---------------------------------------------------------------- scoring (mirrors eval.py)

def _norm(s) -> str:
    return " ".join(str(s).split()).strip()


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(_as_list(x))
        return out
    return [str(v)]


def locate(state: str, value: str):
    i = state.find(value)
    return [i, i + len(value)] if i >= 0 else None


def score(records, preds, model_label, data):
    from collections import defaultdict

    agg = defaultdict(lambda: {"n": 0, "answered": 0, "exact": 0.0,
                               "null_correct": 0, "null_n": 0, "grounded": 0})
    for rec, pred in zip(records, preds):
        dom = rec.get("domain", "?")
        for name, spec in rec["fields"].items():
            gold = (rec.get("answers") or {}).get(name) or {}
            gv = gold.get("value")
            pv = (pred or {}).get(name)
            kind = spec.get("type", "text")
            if kind == "list":
                kind = f"list-{spec.get('items', 'text')}"
            b = agg[(dom, kind)]
            b["n"] += 1
            gset = {_norm(v) for v in _as_list(gv)} if gv is not None else set()
            pset = {_norm(v) for v in _as_list(pv)} if pv is not None else set()
            if not gset:
                b["null_n"] += 1
                b["null_correct"] += int(not pset)
                continue
            if not pset:
                continue
            b["answered"] += 1
            union = len(gset | pset) or 1
            b["exact"] += len(gset & pset) / union
            b["grounded"] += int(all(locate(rec["state"], v) for v in pset))
    metrics, by_domain = {}, {}
    for (dom, kind), b in agg.items():
        entry = {"n": b["n"],
                 "exact": round(b["exact"] / max(b["answered"], 1), 4),
                 "answered": b["answered"],
                 "null_correct": round(b["null_correct"] / max(b["null_n"], 1), 4),
                 "grounded_rate": round(b["grounded"] / max(b["answered"], 1), 4)}
        by_domain.setdefault(dom, {})[kind] = entry
        m = metrics.setdefault(kind, {"n": 0, "exact_sum": 0.0, "answered": 0,
                                      "null_correct": 0, "null_n": 0, "grounded": 0})
        m["n"] += b["n"]; m["exact_sum"] += b["exact"]; m["answered"] += b["answered"]
        m["null_correct"] += b["null_correct"]; m["null_n"] += b["null_n"]
        m["grounded"] += b["grounded"]
    overall = {kind: {"n": m["n"],
                      "exact": round(m["exact_sum"] / max(m["answered"], 1), 4),
                      "answered": m["answered"],
                      "null_correct": round(m["null_correct"] / max(m["null_n"], 1), 4),
                      "grounded_rate": round(m["grounded"] / max(m["answered"], 1), 4)}
               for kind, m in metrics.items()}
    return {"model": model_label, "data": data, "records": len(records),
            "metrics": overall, "by_domain": by_domain}


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["cloudflare", "openai", "anthropic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gateway", default=None,
                    help="Cloudflare AI Gateway name; routes openai/anthropic through "
                         "/compat/chat/completions (needs CF_AIG_TOKEN)")
    ap.add_argument("--cloudflare-base", default=None,
                    help="override base for Workers AI run calls (e.g. a gateway workers-ai path)")
    ap.add_argument("--data", default="data/real_eval.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args(argv)

    records = [json.loads(l) for l in open(args.data)][: args.limit]
    records = [r for r in records if r.get("state") and r.get("fields")]

    # resolve endpoint + model string
    if args.provider == "cloudflare":
        label = f"external:workers-ai:{args.model}"
        def one(rec, fields):
            return call_cloudflare(rec, fields, args.model, args.sleep, args.cloudflare_base)
    else:
        if args.gateway:
            acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            aig = os.environ.get("CF_AIG_TOKEN")
            if not acct or not aig:
                raise SystemExit("set CLOUDFLARE_ACCOUNT_ID and CF_AIG_TOKEN for --gateway")
            url = (f"https://gateway.ai.cloudflare.com/v1/{acct}/{args.gateway}"
                   f"/compat/chat/completions")
            model = args.model if "/" in args.model else f"{args.provider}/{args.model}"
            headers = {"cf-aig-authorization": f"Bearer {aig}",
                       "Content-Type": "application/json"}
            pkey = os.environ.get("OPENAI_API_KEY" if args.provider == "openai"
                                  else "ANTHROPIC_API_KEY")
            if pkey:
                headers["Authorization"] = f"Bearer {pkey}"
            label = f"external:gateway:{args.gateway}:{model}"
            def one(rec, fields):
                return call_chat_compat(url, headers, model, rec, fields, args.sleep)
        elif args.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("set OPENAI_API_KEY (or use --gateway)")
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            url = base + "/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            label = f"external:openai:{args.model}"
            def one(rec, fields):
                return call_chat_compat(url, headers, args.model, rec, fields, args.sleep)
        else:
            label = f"external:anthropic:{args.model}"
            def one(rec, fields):
                return call_anthropic_direct(rec, fields, args.model, args.sleep)

    print(f"[external] {label} on {len(records)} records from {args.data}", flush=True)

    if args.dry_run:
        r = records[0]
        fields = fields_for_record(r)
        print(json.dumps({"endpoint_label": label,
                          "state_chars": len(r["state"]),
                          "n_fields": len(fields),
                          "first_fields": list(fields)[:5],
                          "json_instruction_head": json_instruction(fields)[:300]}, indent=1))
        return 0

    preds = []
    for i, rec in enumerate(records):
        fields = fields_for_record(rec)
        pred = {}
        for attempt in range(args.retries + 1):
            try:
                pred = one(rec, fields)
                break
            except (RuntimeError, urllib.error.URLError, TimeoutError, KeyError) as e:
                if attempt == args.retries:
                    print(f"  [{i}] failed: {e}", file=sys.stderr)
                else:
                    time.sleep(2 + 2 * attempt)
        preds.append(pred if isinstance(pred, dict) else {})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

    report = score(records, preds, label, args.data)
    body = json.dumps(report, indent=2)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
