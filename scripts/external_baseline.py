"""External baselines on our eval files, routed through Cloudflare AI Gateway OR
CommandCode's provider API.

Cloudflare transport (default):
  chat models (OpenAI / Anthropic / Google / Workers AI chat):
      POST https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions
  Clairic (non-chat, native schema):
      POST https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/clairic

CommandCode transport (spec "transport": "commandcode"):
      POST {base}/chat/completions   OpenAI wire format  (spec "api": "chat")
      POST {base}/messages           Anthropic wire format (spec "api": "messages")
  base defaults to https://api.commandcode.ai/provider/v1 and is overridable with
  COMMANDCODE_BASE_URL; auth is Authorization: Bearer $COMMANDCODE_API_KEY.
  Per docs, premium models (Claude) are served only on /messages; the models list
  (/provider/v1/models) reports supported_endpoints per model.

Transport (single REST surface, per Cloudflare docs):
  chat models (OpenAI / Anthropic / Google / Workers AI chat):
      POST https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions
      headers: Authorization: Bearer $CLOUDFLARE_API_TOKEN
               cf-aig-gateway-id: <gateway>
      body: {"model": "openai/gpt-6-luna" | "anthropic/claude-opus-5-5" | "@cf/..."}
  Clairic (non-chat, native schema):
      POST https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/clairic
      body: {"state": ..., "fields": {...}}   (+ cf-aig-gateway-id for gateway logging)

Models are declared in a JSON file (see configs/external_models.json):
  {"models": [
    {"label": "clairic", "kind": "clairic", "model": "@cf/cloudflare/clairic"},
    {"label": "gpt",     "kind": "chat",    "model": "openai/gpt-6-luna"},
    {"label": "opus",    "kind": "chat",    "model": "anthropic/claude-opus-5-5"}
  ]}

Env (all optional except the first two):
  CLOUDFLARE_ACCOUNT_ID      required
  CLOUDFLARE_API_TOKEN       required (auth for the REST endpoints)
  CLOUDFLARE_GATEWAY         gateway id (default: "default")
  CF_AIG_TOKEN               only if the gateway itself enforces auth (cf-aig-authorization)
  COMMANDCODE_API_KEY        required for transport=commandcode
  COMMANDCODE_BASE_URL       default https://api.commandcode.ai/provider/v1
  OPENAI_API_KEY / ANTHROPIC_API_KEY / ...  only if NOT using stored BYOK keys

Usage:
  uv run python scripts/external_baseline.py --models-file configs/external_models.json \
      --data data/real_eval.jsonl --limit 20 --out-dir reports/external
  uv run python scripts/external_baseline.py --model openai/gpt-6-luna --kind chat \
      --limit 5 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
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


# ---------------------------------------------------------------- schema / prompt

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
    return ("Extract the requested fields from the STATE below. Reply with a single JSON object "
            "mapping each field name to an array of EXACT substrings copied verbatim from the "
            "state, or null when the field is not present. Do not paraphrase; no prose outside "
            "the JSON.\n\nFIELDS:\n" + spec)


_TOK = None


def _tokenizer(name: str):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(name)
    return _TOK


def cap_state(state: str, max_tokens: int, tokenizer_name: str):
    """Explicit context cap with exact token accounting (never provider-side)."""
    if max_tokens <= 0:
        return state, 0, 0
    try:
        tok = _tokenizer(tokenizer_name)
        ids = tok(state, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return state, len(ids), len(ids)
        return tok.decode(ids[:max_tokens], skip_special_tokens=True), len(ids), max_tokens
    except Exception:
        before = int(len(state) / 2.5)
        if before <= max_tokens:
            return state, before, before
        return state[: int(max_tokens * 2.5)], before, max_tokens


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


def _cf_headers() -> dict:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise SystemExit("set CLOUDFLARE_API_TOKEN")
    gw = os.environ.get("CLOUDFLARE_GATEWAY", "default")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "cf-aig-gateway-id": gw}
    aig = os.environ.get("CF_AIG_TOKEN")
    if aig:
        headers["cf-aig-authorization"] = f"Bearer {aig}"
    return headers


def _response_text(resp: dict) -> str:
    """Normalise the several response shapes this gateway hands back."""
    if "choices" in resp:                                  # OpenAI-compatible
        c = resp["choices"][0].get("message", {}).get("content")
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return c or ""
    if isinstance(resp.get("content"), list):               # Anthropic native passthrough
        return "".join(b.get("text", "") for b in resp["content"] if isinstance(b, dict))
    if isinstance(resp.get("result"), dict):
        r = resp["result"]
        if isinstance(r.get("response"), str):
            return r["response"]
        if isinstance(r.get("choices"), list):
            return _response_text(r)
    raise RuntimeError(f"unrecognised response shape: {json.dumps(resp)[:300]}")


def _post_stream_text(url: str, headers: dict, body: dict, timeout: int = 300) -> str:
    """SSE fallback: some providers only convert correctly when streamed."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    out = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        out.append(d["content"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}") from e
    return "".join(out)


def _pct(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; None on empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _cc_base() -> str:
    return os.environ.get("COMMANDCODE_BASE_URL",
                          "https://api.commandcode.ai/provider/v1").rstrip("/")


def _cc_headers() -> dict:
    key = os.environ.get("COMMANDCODE_API_KEY")
    if not key:
        raise SystemExit("set COMMANDCODE_API_KEY for transport=commandcode")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": "opencode/2.1.4"}


def _usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    return {"input_tokens": u.get("prompt_tokens", u.get("input_tokens")),
            "output_tokens": u.get("completion_tokens", u.get("output_tokens"))}


def call_cc(record, fields, model, api="chat", sleep_s=0.0, max_tokens=4000,
            timeout=600):
    """One CommandCode provider call. Returns (answers, meta).

    meta carries latency_s measured around the HTTP request only - the rate-limit
    sleep happens after the clock stops, so latency is not our own pacing.
    """
    base = _cc_base()
    if api == "messages":
        url = f"{base}/messages"
        body = {"model": model, "max_tokens": max_tokens,
                "system": json_instruction(fields),
                "messages": [{"role": "user", "content": "STATE:\n" + record["state"]}]}
    else:
        url = f"{base}/chat/completions"
        body = {"model": model,
                "messages": [{"role": "system", "content": json_instruction(fields)},
                             {"role": "user", "content": "STATE:\n" + record["state"]}],
                "max_tokens": max_tokens}

    t0 = time.perf_counter()
    try:
        resp = _post_json(url, _cc_headers(), body, timeout=int(timeout))
    except RuntimeError as e:
        # OpenAI's newer models reject max_tokens; retry once with the other key.
        if "max_tokens" in str(e) and api != "messages":
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = max_tokens
            resp = _post_json(url, _cc_headers(), body, timeout=int(timeout))
        else:
            raise
    latency = time.perf_counter() - t0
    if sleep_s:
        time.sleep(sleep_s)
    meta = {"latency_s": latency, **_usage(resp), "endpoint": api}
    return extract_json_block(_response_text(resp)), meta


def call_cf_chat(record, fields, model, sleep_s, max_tokens=2000, extra=None, timeout=600):
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or sys.exit("set CLOUDFLARE_ACCOUNT_ID")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"
    body = {"model": model,
            "messages": [{"role": "system", "content": json_instruction(fields)},
                         {"role": "user", "content": "STATE:\n" + record["state"]}]}
    # OpenAI's newer models reject max_tokens; everything else still wants it.
    token_key = "max_completion_tokens" if model.startswith("openai/") else "max_tokens"
    body[token_key] = max_tokens
    body.update(extra or {})

    resp, use_stream, last_err, applied = None, False, None, set()
    latency = None
    for _ in range(6):
        _t0 = time.perf_counter()
        try:
            resp = _post_json(url, _cf_headers(), body, timeout=int(timeout))
            latency = time.perf_counter() - _t0
            break
        except RuntimeError as e:
            msg = last_err = str(e)

            def once(key: str) -> bool:
                """Each structural fix may be applied at most once - error texts quote the
                request's own field names, so naive substring matching can ping-pong."""
                if key in applied:
                    return False
                applied.add(key)
                return True

            if "Response conversion failed" in msg and once("stream"):
                use_stream = True                 # provider only converts streamed responses
                break
            if "Unsupported field passed: system" in msg and once("fold-system"):
                sys_txt = body.pop("system", None)
                if sys_txt is None and body["messages"][0]["role"] == "system":
                    sys_txt = body["messages"].pop(0)["content"]
                if sys_txt:
                    body["messages"][0]["content"] = (sys_txt + "\n\n"
                                                      + body["messages"][0]["content"])
            elif ("top-level 'system'" in msg and "fold-system" not in applied
                  and body["messages"][0]["role"] == "system"):
                # Anthropic-style: system prompt is a sibling of messages
                body["system"] = body["messages"][0]["content"]
                body["messages"] = body["messages"][1:]
            elif ("max_tokens" in msg and "temperature" not in msg
                  and "max_completion_tokens" in msg and once("token-key")):
                body.pop(token_key, None)
                token_key = "max_completion_tokens" if token_key == "max_tokens" else "max_tokens"
                body[token_key] = max_tokens
            elif "does not support" in msg and "temperature" in msg and once("temperature"):
                body.pop("temperature", None)     # belt-and-braces; we no longer send one
            else:
                raise
    else:
        raise RuntimeError(f"chat call rejected after parameter fallbacks: {last_err}")

    time.sleep(sleep_s)
    if use_stream:
        _t0 = time.perf_counter()
        text = _post_stream_text(url, _cf_headers(), body, timeout=int(timeout))
        return extract_json_block(text), {"latency_s": time.perf_counter() - _t0,
                                         "endpoint": "chat-stream"}
    meta = {"latency_s": latency, "endpoint": "chat", **_usage(resp)}
    return extract_json_block(_response_text(resp)), meta


def call_cf_clairic(record, fields, model, sleep_s, timeout=600):
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or sys.exit("set CLOUDFLARE_ACCOUNT_ID")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
    payload = {"state": record["state"], "fields": fields}
    headers = _cf_headers()
    _t0 = time.perf_counter()
    try:
        resp = _post_json(url, headers, payload, timeout=int(timeout))
    except RuntimeError:
        # /ai/run is not documented to accept gateway routing; fall back to a
        # direct call rather than losing the baseline entirely.
        headers.pop("cf-aig-gateway-id", None)
        resp = _post_json(url, headers, payload, timeout=int(timeout))
    latency = time.perf_counter() - _t0
    time.sleep(sleep_s)
    if not resp.get("success", True):
        raise RuntimeError(f"cloudflare error: {json.dumps(resp)[:400]}")
    ans = resp.get("result", {}).get("answers", resp.get("answers", {}))
    return ({k: (v.get("value") if isinstance(v, dict) else v) for k, v in ans.items()},
            {"latency_s": latency, "endpoint": "clairic"})


# ---------------------------------------------------------------- scoring (mirrors eval.py)

from clarke.textmetrics import as_list as _as_list_shared
from clarke.textmetrics import contained as _contained
from clarke.textmetrics import norm as _norm
from clarke.textmetrics import pair_f1 as _pair_f1


def _as_list(v):
    return _as_list_shared(v)


def locate(state: str, value: str):
    i = state.find(value)
    return [i, i + len(value)] if i >= 0 else None


def score(records, preds, label, data):
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "answered": 0, "exact": 0.0, "tok_f1": 0.0,
                               "contains": 0.0, "null_correct": 0, "null_n": 0,
                               "grounded": 0})
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
            b["exact"] += len(gset & pset) / (len(gset | pset) or 1)
            b["tok_f1"] += _pair_f1(gset, pset)
            b["contains"] += _contained(gset, pset)
            b["grounded"] += int(all(locate(rec["state"], v) for v in pset))
    metrics, by_domain = {}, {}
    for (dom, kind), b in agg.items():
        by_domain.setdefault(dom, {})[kind] = {
            "n": b["n"], "exact": round(b["exact"] / max(b["answered"], 1), 4),
            "tok_f1": round(b["tok_f1"] / max(b["answered"], 1), 4),
            "contains": round(b["contains"] / max(b["answered"], 1), 4),
            "answered": b["answered"],
            "null_correct": round(b["null_correct"] / max(b["null_n"], 1), 4),
            "grounded_rate": round(b["grounded"] / max(b["answered"], 1), 4)}
        m = metrics.setdefault(kind, {"n": 0, "exact_sum": 0.0, "f1_sum": 0.0,
                                      "contains_sum": 0.0, "answered": 0,
                                      "null_correct": 0, "null_n": 0, "grounded": 0})
        m["n"] += b["n"]; m["exact_sum"] += b["exact"]; m["answered"] += b["answered"]
        m["f1_sum"] += b["tok_f1"]; m["contains_sum"] += b["contains"]
        m["null_correct"] += b["null_correct"]; m["null_n"] += b["null_n"]
        m["grounded"] += b["grounded"]
    overall = {k: {"n": m["n"], "exact": round(m["exact_sum"] / max(m["answered"], 1), 4),
                   "tok_f1": round(m["f1_sum"] / max(m["answered"], 1), 4),
                   "contains": round(m["contains_sum"] / max(m["answered"], 1), 4),
                   "answered": m["answered"],
                   "null_correct": round(m["null_correct"] / max(m["null_n"], 1), 4),
                   "grounded_rate": round(m["grounded"] / max(m["answered"], 1), 4)}
               for k, m in metrics.items()}
    return {"model": label, "data": data, "records": len(records),
            "metrics": overall, "by_domain": by_domain}


# ---------------------------------------------------------------- main

def run_model(spec, records, args):
    transport = spec.get("transport", "cloudflare")
    gateway = "commandcode" if transport == "commandcode" else args.gateway
    label = f"external:{gateway}:{spec['model']}"

    def one(rec, fields):
        if spec.get("kind", "chat") == "clairic":
            return call_cf_clairic(rec, fields, spec["model"], args.sleep,
                                   timeout=args.timeout)
        if transport == "commandcode":
            return call_cc(rec, fields, spec["model"], api=spec.get("api", "chat"),
                           sleep_s=args.sleep, max_tokens=args.max_output_tokens,
                           timeout=args.timeout)
        return call_cf_chat(rec, fields, spec["model"], args.sleep,
                            max_tokens=args.max_output_tokens, extra=spec.get("params"),
                            timeout=args.timeout)
    tag = spec.get("label") or spec["model"]
    print(f"[{tag}] start {label} ({spec.get('kind','chat')}) on {len(records)} records "
          f"(concurrency {args.concurrency})", flush=True)

    def work(idx, rec):
        fields = fields_for_record(rec)
        for attempt in range(args.retries + 1):
            try:
                answers, meta = one(rec, fields)
                return idx, answers, meta, None
            except (RuntimeError, urllib.error.URLError, TimeoutError, KeyError) as e:
                if attempt == args.retries:
                    return idx, {}, None, str(e)
                # jittered backoff: parallel workers must not retry in lockstep
                time.sleep(1.5 * (attempt + 1) + random.random())

    preds = [{} for _ in records]
    metas: list[dict | None] = [None for _ in records]
    ok = [False for _ in records]
    failed, done = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(work, i, r) for i, r in enumerate(records)]
        for fut in as_completed(futures):
            idx, pred, meta, err = fut.result()
            preds[idx] = pred if isinstance(pred, dict) else {}
            metas[idx] = meta
            ok[idx] = not err
            if err:
                failed += 1
                print(f"[{tag}] [{idx}] failed: {err}", file=sys.stderr, flush=True)
            done += 1
            if done % 10 == 0:
                print(f"[{tag}] {done}/{len(records)}", flush=True)
    report = score(records, preds, label, args.data)
    report["max_context_tokens"] = args.max_context_tokens
    report["truncated_records"] = args.truncated
    report["failed_calls"] = failed
    report["transport"] = transport
    report["timing"] = timing_summary(metas, ok)
    body = json.dumps(report, indent=2)
    out = Path(args.out_dir) / f"{spec.get('label') or spec['model'].replace('/', '-').replace('@', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"[{tag}] done {json.dumps(report['metrics'])}", flush=True)
    with open(out.with_suffix(".preds.jsonl"), "w") as f:
        for rec, pred, good in zip(records, preds, ok):
            f.write(json.dumps({"id": rec.get("id"), "pred": pred, "ok": good}) + "\n")
    with open(out.with_suffix(".timing.jsonl"), "w") as f:
        for rec, meta, good in zip(records, metas, ok):
            row = {"id": rec.get("id"), "ok": good}
            row.update(meta or {})
            f.write(json.dumps(row) + "\n")
    print(f"[{tag}] -> {out}", flush=True)
    return report


def timing_summary(metas: list[dict | None], ok: list[bool]) -> dict | None:
    """Latency + token accounting for the calls that actually succeeded."""
    rows = [m for m, good in zip(metas, ok) if good and m and m.get("latency_s")]
    if not rows:
        return None
    lat = [float(m["latency_s"]) for m in rows]
    out_tok = [m.get("output_tokens") for m in rows if m.get("output_tokens")]
    in_tok = [m.get("input_tokens") for m in rows if m.get("input_tokens")]
    total = sum(lat)
    return {
        "n": len(rows),
        "latency_s": {"p50": round(_pct(lat, 0.50) or 0.0, 3),
                      "p90": round(_pct(lat, 0.90) or 0.0, 3),
                      "mean": round(total / len(lat), 3),
                      "min": round(min(lat), 3), "max": round(max(lat), 3),
                      "total": round(total, 1)},
        "output_tokens": ({"mean": round(sum(out_tok) / len(out_tok), 1),
                           "total": sum(out_tok)} if out_tok else None),
        "input_tokens": ({"mean": round(sum(in_tok) / len(in_tok), 1),
                          "total": sum(in_tok)} if in_tok else None),
        "output_tokens_per_s": (round(sum(out_tok) / total, 1)
                                if out_tok and total > 0.05 else None),
    }


def rescore_dir(args) -> int:
    """Re-score already-collected predictions; lets metrics evolve without re-spending."""
    d = Path(args.rescore_dir)
    raw = [json.loads(l) for l in open(args.data)]
    by_id = {}
    for r in raw:
        by_id.setdefault(r.get("id"), []).append(r)
    rows = []
    for pf in sorted(d.glob("*.preds.jsonl")):
        recs, preds = [], []
        for line in pf.read_text().splitlines():
            item = json.loads(line)
            cand = by_id.get(item["id"]) or []
            if not cand:
                continue
            recs.append(cand[0])
            preds.append(item["pred"])
        rep = score(recs, preds, f"external:{pf.stem}", args.data)
        rep["failed_calls"] = None
        out = d / f"{pf.stem.replace('.preds', '')}.json"
        out.write_text(json.dumps(rep, indent=2))
        for kind, m in rep["metrics"].items():
            rows.append([pf.stem.replace(".preds", ""), kind, m["answered"], m["exact"],
                         m["tok_f1"], m["contains"], m["null_correct"], m["grounded_rate"]])
    if rows:
        print(f"{'model':24s} {'kind':10s} {'ans':>4s} {'exact':>7s} {'tokF1':>7s} "
              f"{'contains':>8s} {'nullok':>7s} {'grd':>5s}")
        for r in rows:
            print(f"{r[0]:24s} {r[1]:10s} {r[2]:4d} {r[3]:7.4f} {r[4]:7.4f} "
                  f"{r[5]:8.4f} {r[6]:7.4f} {r[7]:5.3f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default=None,
                    help="JSON list of {label, kind: chat|clairic, model, transport, api}")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", default="chat", choices=["chat", "clairic"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--data", default="data/real_eval.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="per-call pause (parallel workers share the provider rate limit)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="in-flight requests per model")
    ap.add_argument("--model-concurrency", type=int, default=0,
                    help="models running at once (0 = all in the file); separate providers "
                         "have separate rate limits, so this is usually the big win")
    ap.add_argument("--out-dir", default="reports/external")
    ap.add_argument("--max-context-tokens", dest="max_context_tokens", type=int, default=64000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request read timeout; 65k-token prompts take far longer than "
                         "short ones, and a timeout burns a paid call")
    ap.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=4000,
                    help="completion budget per record (chat models); 41 typed fields of spans "
                         "need room, and adversarial models burn tokens on reasoning")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore-dir", default=None,
                    help="recompute reports from saved <label>.preds.jsonl (no API calls)")
    args = ap.parse_args(argv)
    args.gateway = os.environ.get("CLOUDFLARE_GATEWAY", "default")

    if args.rescore_dir:
        return rescore_dir(args)

    if args.models_file:
        specs = json.loads(Path(args.models_file).read_text())["models"]
    elif args.model:
        specs = [{"label": args.label, "kind": args.kind, "model": args.model}]
    else:
        raise SystemExit("pass --models-file or --model")

    records = [json.loads(l) for l in open(args.data)][: args.limit]
    records = [r for r in records if r.get("state") and r.get("fields")]
    trunc = 0
    for idx, r in enumerate(records):
        capped, before, after = cap_state(r["state"], args.max_context_tokens, args.tokenizer)
        if after < before:
            trunc += 1
            records[idx] = {**r, "state": capped}
    args.truncated = trunc

    if args.dry_run:
        print(json.dumps({"gateway": args.gateway,
                          "data": args.data,
                          "records": len(records),
                          "max_context_tokens": args.max_context_tokens,
                          "truncated_records": trunc,
                          "models": specs}, indent=1))
        return 0

    workers = args.model_concurrency or len(specs)
    if workers <= 1:
        for spec in specs:
            run_model(spec, records, args)
    else:
        print(f"[external] {len(specs)} models in parallel, {args.concurrency} records each",
              flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_model, spec, records, args): spec for spec in specs}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    fut.result()
                except Exception as e:                      # noqa: BLE001
                    print(f"[{spec.get('label')}] MODEL FAILED: {e}", file=sys.stderr,
                          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
