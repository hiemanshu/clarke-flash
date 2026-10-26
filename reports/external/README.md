# External baselines — status

20 CUAD records from `data/real_eval.jsonl` (62 gold-valued fields), 41-field schema,
Cloudflare AI Gateway, 64k-token cap, no temperature parameter.

## Valid results

| label | exact | tokF1 | contains | answered | null_ok | grounded |
|---|---|---|---|---|---|---|
| opus-5.5 (`anthropic/claude-opus-5.5`) | .3396 | .6179 | .7432 | 53 | .9169 | .830 |
| gemini-4-argon (`google/gemini-4-argon`) | .2750 | .6608 | .7934 | 53 | .9340 | .641 |
| gpt-6.1-sol (`openai/gpt-6.1-sol`) | .1727 | .6286 | .7759 | 54 | .8945 | .759 |
| clairic (`@cf/cloudflare/clairic`) | .0047 | .2365 | .9528 | 60 | .2546 | .867 |

Metric meanings:
- **exact** — per-field set-F1 with exact normalised string equality against the gold span.
  This is the metric our own model is trained on; it is deliberately strict.
- **tokF1** — token-level F1 with best pairing; credits the right clause quoted differently.
- **contains** — share of gold values that appear inside a prediction (or vice versa).
- **answered** — fields where both a gold value and a prediction exist (of 62).
- **null_ok** — share of truly-null fields left null (discipline against over-filling).
- **grounded** — share of answered fields whose predicted strings occur verbatim in the state.

## Invalid files (do not use)

- `deepseek-v4.1-flash.*` — all 20 calls failed with HTTP 403: model not available on the
  Workers Free plan. Needs a paid Workers subscription.
- `minimax-m3.1-flash.*` — all 20 calls failed in the first run. The fixes landed afterwards
  (fold system prompt into the user turn; stream the response because the gateway's
  non-streaming conversion returns 500) and a rerun was started but stopped by the owner.
  The files on disk are from the failed pass (`failed_calls: 20`, `answered: 0`).

## Not run

- `alibaba/qwen3.8-max` — every call exceeded the 180 s read timeout; retries produced no
  output. Needs a longer timeout and/or a smaller output budget.

## Caveat

Our own 2B model's .428 exact is measured on the full 148-record CUAD eval, not this
20-record slice, so the comparison is directional, not a strict head-to-head. Re-run our
model scored on these same 20 records before quoting a single table.
