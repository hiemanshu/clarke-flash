"""Answer-quality metrics shared by our evaluator and the external-baseline harness.

Keeping these in one place is deliberate: when our scorer and the external scorer
disagree, the comparison table silently becomes meaningless. Any change here changes
both sides at once.

Definitions
  exact     set-F1 (Jaccard) over normalised values - the headline, span-string based
  tok_f1    token-level F1 with greedy best pairing; credits the right clause quoted
            with different trimming, which matters when a model returns whole sentences
  contains  fraction of gold values whose text appears inside a prediction (or vice
            versa) after whitespace/case normalisation - "did it find the right place"
"""
from __future__ import annotations

from collections import Counter


def norm(s: object) -> str:
    return " ".join(str(s).split()).strip()


def as_list(v: object) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for x in v:
            out.extend(as_list(x))
        return out
    return [str(v)]


def toks(s: str) -> list[str]:
    return norm(s).lower().split()


def tok_f1(a: str, b: str) -> float:
    ca, cb = Counter(toks(a)), Counter(toks(b))
    inter = sum((ca & cb).values())
    if not inter:
        return 0.0
    prec = inter / max(sum(cb.values()), 1)
    rec = inter / max(sum(ca.values()), 1)
    return 2 * prec * rec / (prec + rec)


def pair_f1(gold: set[str], pred: set[str]) -> float:
    """Greedy best pairing, so multi-value fields are not punished for ordering."""
    if not gold:
        return 0.0
    remaining, total = list(pred), 0.0
    for g in gold:
        best, best_i = 0.0, -1
        for i, p in enumerate(remaining):
            v = tok_f1(g, p)
            if v > best:
                best, best_i = v, i
        total += best
        if best_i >= 0:
            remaining.pop(best_i)
    return total / len(gold)


def contained(gold: set[str], pred: set[str]) -> float:
    if not gold:
        return 0.0
    hits = 0
    for g in gold:
        gl = norm(g).lower()
        for p in pred:
            pl = norm(p).lower()
            if gl and (gl in pl or pl in gl):
                hits += 1
                break
    return hits / len(gold)
