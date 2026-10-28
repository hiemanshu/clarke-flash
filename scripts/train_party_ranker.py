"""Train a tiny candidate ranker for `parties` on the 470 CUAD training documents.

The pointer ranks name candidates; this learns a better ranking from text features alone
(cheap, model-free, so it composes with any checkpoint). Labels come straight from the
gold party annotations: a candidate is positive if its text matches (by containment) any
gold party value.

Logistic regression in torch - 14 features, 470 documents, seconds to train, no GPU.
Saves standardisation stats with the weights so decode uses the identical transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clarke.candidates import PARTY_FEATURES, party_features, text_candidates


def norm(text: str) -> str:
    return " ".join(str(text).split()).strip().strip("\"“”'").lower()


def build(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, int]:
    feats: list[list[float]] = []
    labels: list[float] = []
    for rec in rows:
        gold = [norm(v) for v in (rec.get("answers", {}).get("parties", {}).get("value") or [])]
        if not gold:
            continue
        state = rec["state"]
        seen: set[str] = set()
        for text, (start, end) in text_candidates(state):
            key = norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            freq = state.lower().count(key)
            hit = any(key in g or g in key for g in gold)
            feats.append(party_features(state, text, start, end, freq))
            labels.append(1.0 if hit else 0.0)
    return torch.tensor(feats), torch.tensor(labels), len(labels)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/doc_train_470.jsonl")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="reports/party_ranker.json")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.train).open()]
    X, y, n = build(rows)
    print(f"training rows {n}  positives {int(y.sum())} ({y.mean():.2%})  features {X.shape[1]}",
          flush=True)

    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    Xn = (X - mean) / std
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos_weight = torch.tensor((len(y) - y.sum()) / y.sum().clamp_min(1.0))
    opt = torch.optim.Adam([w, b], lr=args.lr)
    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(Xn @ w + b)
                order = torch.argsort(prob, descending=True)
                # precision@K where K = positives per doc is not tracked here; report AUC-ish
                print(f"  epoch {epoch+1:4d} loss {loss.item():.4f}", flush=True)

    payload = {"features": PARTY_FEATURES, "mean": mean.tolist(), "std": std.tolist(),
               "weights": w.tolist(), "bias": float(b.item())}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\nweights (standardised units):")
    for name, weight in sorted(zip(PARTY_FEATURES, w.tolist()), key=lambda kv: -abs(kv[1])):
        print(f"  {name:18s} {weight:+.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
