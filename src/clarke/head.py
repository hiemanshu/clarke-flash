"""Joint extraction head for clarke-flash.

Mirrors clef's ``JointSchemaHead`` — pooled field queries from the backbone's
final hidden states, cross-attention evidence routing over the whole sequence,
and a small transformer decoder that answers every schema field jointly —
but scores typed answer spaces instead of option lists:

- ``text`` / ``list<text>``   pointer heads over state tokens (grounded spans)
- ``number`` / ``duration``   sign x exponent x top3 x next3 categorical value
- ``date``/``time``/``datetime`` calendar component categoricals (always valid)
- ``estimate`` / ``forecast`` categorical distribution over the declared grid,
  with below/above overflow bins -> median, p10/p90, p_below, p_above
- every grounded/scalar field also gets a ``p_null`` head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidates import field_atoms, solve
from .encoding import EncodedRecord, token_span_for_chars
from .schema import (
    DAY_BINS,
    DIGIT3_BINS,
    EXP_BINS,
    HOUR_BINS,
    LIST_SLOTS,
    MAX_GRID_BINS,
    MIN_BINS,
    OVERFLOW_BINS,
    SEC_BINS,
    SIGN_BINS,
    TYPE_IDS,
    YEAR_BINS,
    FieldSpec,
    decompose_number,
)

FORECAST_MAX_HORIZON = 8


@dataclass
class FieldTargets:
    """Per-field supervised targets, in bin/token index space."""

    null: bool = False
    start: int | None = None
    end: int | None = None
    spans: list[tuple[int, int]] = dataclass_field(default_factory=list)   # list<text>
    cand: tuple[int, int, int] | None = None       # number/duration/date/time/datetime: (a, b, op)
    cands: list[int] = dataclass_field(default_factory=list)               # list<number|date|time> slot atom idx
    numbers: list[tuple[int, int, int, int]] = dataclass_field(default_factory=list)  # unused fallback
    date: tuple[int, int, int] | None = None
    time: tuple[int, int, int] | None = None
    dates: list[tuple[int, int, int]] = dataclass_field(default_factory=list)
    dist: list[int] = dataclass_field(default_factory=list)                       # estimate/forecast bins


LENGTH_BINS = 257          # bin k -> span length k+1 tokens; last bin = 257+


@dataclass
class FieldLogits:
    null: torch.Tensor | None = None                # [] scalar logit
    start: torch.Tensor | None = None               # [state_tokens] or [slots, state_tokens]
    end: torch.Tensor | None = None                 # same shape as start
    present: torch.Tensor | None = None             # [slots]
    cand_a: torch.Tensor | None = None              # [atoms] or [slots, atoms]
    cand_b: torch.Tensor | None = None              # [atoms]
    op: torch.Tensor | None = None                  # [5]
    number_heads: list[torch.Tensor] = dataclass_field(default_factory=list)  # legacy digit path (lists)
    components: list[torch.Tensor] = dataclass_field(default_factory=list)     # legacy
    dist: torch.Tensor | None = None                # [horizon, MAX_GRID_BINS + 2]
    length: torch.Tensor | None = None              # [slots, LENGTH_BINS] span extent


def spec_features(spec: FieldSpec) -> list[float]:
    """Numeric description of a field's answer grid, fed to the head."""
    def log10(value: float | None) -> float:
        return math.log10(max(abs(value), 1e-9)) if value is not None else 0.0

    return [
        spec.decimals / 4.0,
        1.0 if spec.integer else 0.0,
        log10(spec.min),
        log10(spec.max),
        log10(spec.step),
        1.0 if spec.scale == "log" else 0.0,
        min(spec.horizon, FORECAST_MAX_HORIZON) / 4.0,
        TYPE_IDS[f"list-{spec.items}"] / 11.0 if spec.type == "list" else 0.0,
    ]


class EvidenceRoutingLayer(nn.Module):
    """Cross-attention from field queries into the sequence memory (clef-style)."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        routed, _ = self.attention(
            self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointExtractionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int = 1024,
        routing_layers: int = 2,
        layers: int = 4,
        heads: int = 16,
        feedforward: int = 4096,
        dropout: float = 0.0,
        rerank: bool = False,
        query_pool: str = "mean",
        span_rerank: bool = False,
        length_head: bool = False,
    ):
        super().__init__()
        self.rerank = bool(rerank)
        assert query_pool in ("mean", "last", "attn")
        self.query_pool = query_pool
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.field_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.config_projection = nn.Linear(8, width, bias=False)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(len(TYPE_IDS) + 1, width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward, dropout) for _ in range(routing_layers)
        )
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)

        # readouts
        self.null_head = nn.Linear(width, 1)
        self.start_projection = nn.Linear(width, width, bias=False)
        self.end_projection = nn.Linear(width, width, bias=False)
        self.slot_embedding = nn.Embedding(LIST_SLOTS, width)
        self.present_head = nn.Linear(width, 1)
        self.span_scale = nn.Parameter(torch.zeros(()))
        # Span EXTENT: score(i,j)=p_start*p_end has no term for how long a field's answers
        # are (parties: 4 tokens, governing_law: 33, insurance: 45). Measured at 65k, the
        # start token is right 72% of the time but the length only 56% - so a conditioned
        # length distribution is the missing factor in the span score.
        self.length_head = nn.Linear(width, LENGTH_BINS) if length_head else None

        self.value_heads = nn.ModuleList([
            nn.Linear(width, SIGN_BINS), nn.Linear(width, EXP_BINS),
            nn.Linear(width, DIGIT3_BINS), nn.Linear(width, DIGIT3_BINS),
        ])
        self.cand_value_proj = nn.Linear(4, width)
        self.op_head = nn.Linear(width, 5)
        self.span_rerank = bool(span_rerank)

        def pair_net(zero_last: bool = False):
            net = nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
            if zero_last:  # identity at init: reranker starts as the plain pointer
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            return net

        if self.rerank:
            # Tier B: cross-encoder style pair scorer for candidate selection
            self.pair_a = pair_net()
            self.pair_b = pair_net()
        if self.span_rerank:
            # token-span selection: base pointer proposes top-K, pair net corrects
            self.span_pair = pair_net(zero_last=True)
        if self.query_pool == "attn":
            # Tier A: learned attention pooling over the instruction span —
            # mean-pool washes out question structure ("after", "before",
            # "excluding") that last-token/attentive pooling preserves.
            # Applied to the raw hidden states (before field_projection).
            self.query_attn = nn.Linear(hidden_size, 1, bias=False)
        self.date_heads = nn.ModuleList([
            nn.Linear(width, YEAR_BINS), nn.Linear(width, 12), nn.Linear(width, DAY_BINS),
        ])
        self.time_heads = nn.ModuleList([
            nn.Linear(width, HOUR_BINS), nn.Linear(width, MIN_BINS), nn.Linear(width, SEC_BINS),
        ])
        self.horizon_embedding = nn.Embedding(FORECAST_MAX_HORIZON, width)
        self.dist_head = nn.Linear(width, MAX_GRID_BINS + OVERFLOW_BINS)
        self.logit_scale = math.sqrt(width)

    # -- forward -----------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, records: list[EncodedRecord]
    ) -> list[list[FieldLogits]]:
        normalized = self.hidden_norm(hidden_states.float())
        results: list[list[FieldLogits]] = []
        for batch_index, record in enumerate(records):
            length = int(attention_mask[batch_index].sum().item())
            sequence = normalized[batch_index, :length]
            memory = self.memory_projection(sequence)
            global_vector = self.global_projection(sequence[-1])

            field_spans = [f.field_span for f in record.fields]
            if self.query_pool == "last":
                pooled = torch.stack([sequence[end - 1] for _, end in field_spans])
            elif self.query_pool == "attn":
                rows = []
                for start, end in field_spans:
                    span = sequence[start:end]
                    weights = torch.softmax(self.query_attn(span).squeeze(-1), dim=0)
                    rows.append((span * weights.unsqueeze(-1)).sum(dim=0))
                pooled = torch.stack(rows)
            else:
                pooled = torch.stack([sequence[start:end].mean(dim=0) for start, end in field_spans])
            type_ids = torch.tensor([f.type_id for f in record.fields], device=sequence.device)
            configs = torch.tensor([spec_features(f.spec) for f in record.fields],
                                   device=sequence.device, dtype=torch.float32)
            queries = (
                self.field_projection(pooled)
                + self.type_embedding(type_ids)
                + self.config_projection(configs)
                + global_vector
            ).unsqueeze(0)
            for layer in self.evidence_layers:
                queries = layer(queries, memory.unsqueeze(0))
            fields = queries[0]
            for layer in self.layers:
                fields = layer(fields.unsqueeze(0), memory.unsqueeze(0))[0]
            fields = self.field_norm(fields)

            state_start, state_end = record.state_range
            state_memory = memory[state_start:state_end]
            scale = self.span_scale.exp().clamp(max=50.0) / self.logit_scale

            record_logits: list[FieldLogits] = []
            for field_index, encoded in enumerate(record.fields):
                field = fields[field_index]
                spec = encoded.spec
                logits = FieldLogits(null=self.null_head(field).view(()))
                if spec.type == "text":
                    logits.start = (state_memory @ self.start_projection(field)) * scale
                    logits.end = (state_memory @ self.end_projection(field)) * scale
                    if self.span_rerank:
                        # START-ONLY (attempt-5 lesson): sharing one pair net for
                        # start AND end correction puts contradictory gradients
                        # (first-token vs last-token) through the same function —
                        # val span blew 0.4 -> 19 by step 4000. End keeps the
                        # plain trained pointer path; decode conditions end on
                        # start anyway, so the selection gap lives here.
                        logits.start = self._rerank_spans(field, logits.start, state_memory)
                elif spec.type == "list" and spec.items == "text":
                    slots = field.unsqueeze(0) + self.slot_embedding.weight
                    logits.start = (self.start_projection(slots) @ state_memory.T) * scale
                    logits.end = (self.end_projection(slots) @ state_memory.T) * scale
                    logits.present = self.present_head(slots).squeeze(-1)
                    if self.length_head is not None:
                        logits.length = self.length_head(slots)
                elif spec.type in ("number", "duration", "date", "time", "datetime"):
                    emb = self._candidate_embeddings(record, state_memory, spec)
                    if emb is not None:
                        if self.rerank:
                            logits.cand_a = self._pair_score(field, emb, self.pair_a)[0]
                            logits.cand_b = self._pair_score(field, emb, self.pair_b)[0]
                        else:
                            logits.cand_a = (self.start_projection(field) @ emb.T) * scale
                            logits.cand_b = (self.end_projection(field) @ emb.T) * scale
                        logits.op = self.op_head(field)
                elif spec.type in ("estimate", "forecast"):
                    steps = max(spec.horizon, 1) if spec.type == "forecast" else 1
                    vectors = field.unsqueeze(0)
                    if spec.type == "forecast":
                        horizons = torch.arange(steps, device=field.device).clamp(max=FORECAST_MAX_HORIZON - 1)
                        vectors = vectors + self.horizon_embedding(horizons)
                    raw = self.dist_head(vectors)
                    valid = self._valid_bins(spec, steps, field.device)
                    logits.dist = raw.masked_fill(~valid, -1e30)  # finite: 0*-inf = NaN
                elif spec.type == "list":
                    slots = field.unsqueeze(0) + self.slot_embedding.weight
                    logits.present = self.present_head(slots).squeeze(-1)
                    if spec.items in ("number", "date", "time"):
                        emb = self._candidate_embeddings(record, state_memory, spec)
                        if emb is not None:
                            if self.rerank:
                                logits.cand_a = self._pair_score(slots, emb, self.pair_a)
                            else:
                                logits.cand_a = (self.start_projection(slots) @ emb.T) * scale
                record_logits.append(logits)
            results.append(record_logits)
        return results

    def _rerank_spans(self, field: torch.Tensor, base_scores: torch.Tensor,
                      token_mem: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Two-stage span pointer: base dot-product proposes top-k tokens, the
        pair net ADDS a learned correction (zero-init => identity at step 0).
        Targets the measured gap: p@5 ~.6 vs p@1 ~.25 on real docs — gold is
        retrieved by the pointer but not selected."""
        kk = min(k, base_scores.shape[-1])
        if kk < 2:
            return base_scores
        top = base_scores.detach().topk(kk, dim=-1).indices
        cand = token_mem[top]
        corr = self._pair_score(field, cand, self.span_pair)[0]
        out = base_scores.clone()
        out[top] = base_scores[top].detach() + corr
        return out

    def _pair_score(self, query: torch.Tensor, emb: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """Tier B reranker: MLP over [q; c; q*c; |q-c|] for every (query, candidate)
        pair — turns 'looks similar' into 'answers the question'."""
        q = query if query.dim() == 2 else query.unsqueeze(0)
        qq = q.unsqueeze(1).expand(-1, emb.shape[0], -1)
        cc = emb.unsqueeze(0).expand(q.shape[0], -1, -1)
        feats = torch.cat([qq, cc, qq * cc, (qq - cc).abs()], dim=-1)
        return net(feats).squeeze(-1)

    def _candidate_embeddings(self, record: EncodedRecord, state_memory: torch.Tensor,
                              spec: FieldSpec) -> torch.Tensor | None:
        """Embed each candidate atom: pooled state tokens over its span plus a
        projection of its numeric/calendar value features.

        ``state_memory`` is the state-token slice (0-based, as the pointer
        heads see it); spans are widened to include identifying labels that
        sit outside the answer ("Tax: $308.03").
        """
        from .candidates import field_atoms
        from .encoding import token_span_for_chars

        atoms = field_atoms(spec, record.pool)
        if not atoms:
            return None
        rows, feats = [], []
        width = state_memory.shape[1]
        n_tokens = len(record.char_offsets)
        for _value, span, feat in atoms:
            pooled = torch.zeros(width, device=state_memory.device)
            if span is not None:
                pair = token_span_for_chars(record, span[0], span[1])
                if pair is not None:
                    s2 = max(pair[0] - 8, 0)
                    e2 = min(pair[1] + 8, n_tokens - 1)
                    pooled = state_memory[s2:e2 + 1].mean(dim=0)
            rows.append(pooled)
            feats.append(feat)
        emb = (torch.stack(rows)
               + self.cand_value_proj(torch.tensor(feats, device=state_memory.device)))
        return emb

    @staticmethod
    def _valid_bins(spec: FieldSpec, steps: int, device: torch.device) -> torch.Tensor:
        valid = torch.zeros((steps, MAX_GRID_BINS + OVERFLOW_BINS), dtype=torch.bool, device=device)
        valid[:, : len(spec.grid_centers())] = True
        return valid

    # -- targets -----------------------------------------------------------

    @staticmethod
    def build_targets(record: EncodedRecord) -> list[FieldTargets]:
        targets: list[FieldTargets] = []
        for encoded in record.fields:
            spec = encoded.spec
            answer = record.answers.get(encoded.name) or {}
            target = FieldTargets()
            value = answer.get("value")
            values = answer.get("values")
            if values is None and isinstance(value, list):
                values, value = (value or None), None
            if value is None and values is None:
                target.null = True
                targets.append(target)
                continue

            if spec.type == "text":
                span = answer.get("span")
                pair = token_span_for_chars(record, span[0], span[1]) if span else None
                if pair:
                    target.start, target.end = pair
                else:
                    target.null = True
            elif spec.type == "list" and spec.items == "text":
                for start, stop in answer.get("spans") or []:
                    pair = token_span_for_chars(record, start, stop)
                    if pair:
                        target.spans.append(pair)
            elif spec.type in ("number", "duration"):
                target.cand = solve(spec, record.pool, float(value), spec.decimals)
            elif spec.type in ("date", "time", "datetime"):
                idx = _atom_index(spec, record, str(value))
                target.cand = (idx, -1, 0) if idx >= 0 else None
            elif spec.type == "list" and spec.items in ("number", "date", "time"):
                atoms = field_atoms(spec, record.pool)
                if spec.items == "number":
                    tol = 1.5 * 10 ** -spec.decimals
                    target.cands = [next((i for i, (v, _, _) in enumerate(atoms)
                                          if abs(float(v) - float(g)) <= tol), -1)
                                    for g in values or []]
                else:
                    target.cands = [next((i for i, (iso, _, _) in enumerate(atoms)
                                          if iso == str(g)), -1) for g in values or []]
            elif spec.type in ("estimate", "forecast"):
                items = values if values is not None else [value]
                target.dist = [spec.bin_value(float(v)) for v in items]
            targets.append(target)
        return targets

    # -- loss ---------------------------------------------------------------

    def compute_loss(
        self,
        outputs: list[list[FieldLogits]],
        records: list[EncodedRecord],
        targets: list[list[FieldTargets]],
        label_smoothing: float = 0.1,
        dist_neighbor: float = 0.1,
        brier_null: float = 0.0,
        field_weights: dict[str, float] | None = None,
        slot_mask_train: bool = False,
        slot_assign: bool = False,
    ) -> dict[str, torch.Tensor]:
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, float] = {}

        def add(key: str, value: torch.Tensor, weight: float = 1.0) -> None:
            sums[key] = sums.get(key, value.detach() * 0) + value
            counts[key] = counts.get(key, 0.0) + weight

        def cross_entropy_rows(rows: torch.Tensor, bins: list[int], alpha: float) -> torch.Tensor:
            """Distribution CE with adjacent-bin partial credit (a cheap RLCD
            surrogate from the Clef blog): mass (1-2a) on the realized bin,
            a on each neighbour."""
            n = rows.shape[-1]
            device = rows.device
            onehot = F.one_hot(torch.tensor(bins, device=device), n).float()
            left = F.one_hot(torch.tensor([max(b - 1, 0) for b in bins], device=device), n).float()
            right = F.one_hot(torch.tensor([min(b + 1, n - 1) for b in bins], device=device), n).float()
            soft = (1.0 - 2.0 * alpha) * onehot + alpha * left + alpha * right
            return -(soft * F.log_softmax(rows.float(), dim=-1)).sum(-1).mean()

        for record_logits, record, record_targets in zip(outputs, records, targets):
            for encoded, logits, target in zip(record.fields, record_logits, record_targets):
                spec = encoded.spec
                w = float(field_weights.get(spec.name, 1.0)) if field_weights else 1.0

                def addf(key: str, value: torch.Tensor, weight: float = 1.0) -> None:
                    """Weighted mean: rare fields must not be averaged into oblivion."""
                    if w != 1.0:
                        add(key, value * w, weight * w)
                    else:
                        add(key, value, weight)

                if spec.type not in ("estimate", "forecast"):
                    target_null = torch.tensor([float(target.null)],
                                               device=logits.null.device)
                    addf("null", F.binary_cross_entropy_with_logits(
                        logits.null.float().reshape(1), target_null))
                    if brier_null > 0:
                        # Brier is a proper scoring rule: it penalises over/under-confidence
                        # itself, not just ranking. Added as its own weighted head so the
                        # component stays visible in the training report.
                        p_null = torch.sigmoid(logits.null.float().reshape(1))
                        add("brier", ((p_null - target_null) ** 2).mean() * brier_null)

                if target.null:
                    # a null LIST must still train the present head toward the
                    # zero vector: without this, 91%-absent regimes (CUAD) get
                    # no "no list" gradient and every decode fabricates items
                    if spec.type == "list" and logits.present is not None:
                        zeros = torch.zeros(logits.present.shape[-1],
                                            device=logits.present.device)
                        add("present", F.binary_cross_entropy_with_logits(
                            logits.present.float(), zeros))
                    continue
                if spec.type == "text" and target.start is not None:
                    add("span", F.cross_entropy(logits.start.view(1, -1),
                        torch.tensor([target.start], device=logits.start.device)))
                    add("span", F.cross_entropy(logits.end.view(1, -1),
                        torch.tensor([target.end], device=logits.end.device)))
                elif spec.type == "list" and spec.items == "text":
                    items = target.spans[:LIST_SLOTS]
                    if slot_assign and len(items) > 1:
                        # Gold items are annotated in DOCUMENT order, but a list field is a
                        # SET: slot k has no reason to own the k-th annotation. Match items to
                        # slots by the model's own current confidence (greedy one-to-one,
                        # i.e. a Hungarian approximation), then compact to slots 0..n-1 so the
                        # present head keeps a dense target. Discrete, so no gradient flows here.
                        with torch.no_grad():
                            ls = torch.log_softmax(logits.start.float(), dim=-1)
                            le = torch.log_softmax(logits.end.float(), dim=-1)
                            starts_t = torch.tensor([a for a, _ in items],
                                                    device=logits.start.device)
                            ends_t = torch.tensor([b for _, b in items],
                                                  device=logits.end.device)
                            cost = -(ls[:, starts_t] + le[:, ends_t])      # [slots, items]
                            free_items, free_slots, pairs = (set(range(len(items))),
                                                             set(range(cost.shape[0])), [])
                            while free_items:
                                best = min((float(cost[sl, it]), it, sl)
                                           for it in free_items for sl in free_slots)
                                _, it, sl = best
                                pairs.append((sl, it))
                                free_items.discard(it)
                                free_slots.discard(sl)
                            pairs.sort()
                            items = [items[it] for _sl, it in pairs]
                    present = torch.zeros(LIST_SLOTS, device=logits.present.device)
                    present[: len(items)] = 1.0
                    addf("present", F.binary_cross_entropy_with_logits(logits.present.float(), present))
                    # Slots are independent heads, so nothing stops them from all pointing
                    # at the SAME item - which is exactly what happens on parties (every slot
                    # lands on one party, dedupe deletes the copies, recall collapses to .15
                    # even though .82 of gold starts are inside some slot's top-5).
                    # Masking the text already consumed by earlier slots trains the spread
                    # that decode-time masking only borrows. This flag used to be accepted and
                    # ignored (no-op until 2026-10-27).
                    consumed: list[tuple[int, int]] = []
                    for slot, (start, end) in enumerate(items):
                        slot_start, slot_end = logits.start[slot], logits.end[slot]
                        if slot_mask_train and slot > 0 and consumed:
                            # fresh mask per slot: mutating one shared tensor in place makes
                            # autograd report "variable modified by an inplace operation"
                            mask = torch.zeros(logits.start.shape[-1], dtype=torch.bool,
                                               device=logits.start.device)
                            for s0, e0 in consumed:
                                mask[s0:e0 + 1] = True
                            # never mask the item this slot must predict: CUAD party
                            # annotations nest ("Company" inside "Electric City Company"),
                            # so a hard mask over an earlier item can cover the next gold
                            # span and blow the CE up to ~10k (diverge guard fired at step 37).
                            mask[start:end + 1] = False
                            slot_start = slot_start.masked_fill(mask, -1e4)
                            slot_end = slot_end.masked_fill(mask, -1e4)
                        add("span", F.cross_entropy(slot_start.view(1, -1),
                            torch.tensor([start], device=logits.start.device)))
                        add("span", F.cross_entropy(slot_end.view(1, -1),
                            torch.tensor([end], device=logits.end.device)))
                        consumed.append((int(start), int(end)))
                        if logits.length is not None:
                            bin_idx = min(int(end) - int(start) + 1, LENGTH_BINS) - 1
                            add("span", F.cross_entropy(logits.length[slot].view(1, -1),
                                torch.tensor([bin_idx], device=logits.length.device)))
                elif target.cand is not None and logits.cand_a is not None and spec.type != "list":
                    a, b, op = target.cand
                    addf("cand", F.cross_entropy(logits.cand_a.view(1, -1),
                        torch.tensor([a], device=logits.cand_a.device),
                        label_smoothing=label_smoothing))
                    if b >= 0:
                        addf("cand", F.cross_entropy(logits.cand_b.view(1, -1),
                            torch.tensor([b], device=logits.cand_b.device),
                            label_smoothing=label_smoothing))
                    # op is supervised on EVERY solved scalar, including
                    # single-atom "id" answers — an untrained op head turns
                    # read answers into random arithmetic (duration 0% bug)
                    addf("op", F.cross_entropy(logits.op.view(1, -1),
                        torch.tensor([op], device=logits.op.device), label_smoothing=0.05))
                elif spec.type == "list" and spec.items in ("number", "date", "time") \
                        and logits.cand_a is not None:
                    items = target.cands[:LIST_SLOTS]
                    present = torch.zeros(LIST_SLOTS, device=logits.present.device)
                    present[: len(items)] = 1.0
                    addf("present", F.binary_cross_entropy_with_logits(logits.present.float(), present))
                    for slot, idx in enumerate(items):
                        if idx >= 0:
                            addf("cand", F.cross_entropy(logits.cand_a[slot].view(1, -1),
                                torch.tensor([idx], device=logits.cand_a.device),
                                label_smoothing=label_smoothing))
                elif spec.type in ("estimate", "forecast") and target.dist:
                    steps = len(target.dist)
                    addf("dist", cross_entropy_rows(logits.dist[:steps], target.dist, dist_neighbor),
                        float(steps))

        total = sum((sums[key] / max(counts[key], 1.0)) for key in sums) if sums else torch.tensor(0.0)
        report = {key: (sums[key] / max(counts[key], 1.0)).detach() for key in sums}
        report["loss"] = total  # live graph; the trainer calls backward on this
        return report


def _atom_index(spec, record, gold: str) -> int:
    atoms = field_atoms(spec, record.pool)
    return next((i for i, (value, _, _) in enumerate(atoms) if str(value) == gold), -1)


def _date_bins(text: str) -> tuple[int, int, int]:
    year, month, day = (int(part) for part in text.split("-"))
    from .schema import YEAR_MIN

    return (
        min(max(year - YEAR_MIN, 0), YEAR_BINS - 1),
        min(max(month - 1, 0), 11),
        min(max(day - 1, 0), DAY_BINS - 1),
    )


def _time_bins(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0] % 24, parts[1] % 60, parts[2] % 60
