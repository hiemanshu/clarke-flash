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
    ):
        super().__init__()
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

        self.value_heads = nn.ModuleList([
            nn.Linear(width, SIGN_BINS), nn.Linear(width, EXP_BINS),
            nn.Linear(width, DIGIT3_BINS), nn.Linear(width, DIGIT3_BINS),
        ])
        self.cand_value_proj = nn.Linear(4, width)
        self.op_head = nn.Linear(width, 5)
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
                elif spec.type == "list" and spec.items == "text":
                    slots = field.unsqueeze(0) + self.slot_embedding.weight
                    logits.start = (self.start_projection(slots) @ state_memory.T) * scale
                    logits.end = (self.end_projection(slots) @ state_memory.T) * scale
                    logits.present = self.present_head(slots).squeeze(-1)
                elif spec.type in ("number", "duration", "date", "time", "datetime"):
                    emb = self._candidate_embeddings(record, state_memory, spec)
                    if emb is not None:
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
                            logits.cand_a = (self.start_projection(slots) @ emb.T) * scale
                record_logits.append(logits)
            results.append(record_logits)
        return results

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
                if spec.type not in ("estimate", "forecast"):
                    add("null", F.binary_cross_entropy_with_logits(
                        logits.null.float().reshape(1), torch.tensor([float(target.null)],
                        device=logits.null.device)))

                if target.null:
                    continue
                if spec.type == "text" and target.start is not None:
                    add("span", F.cross_entropy(logits.start.view(1, -1),
                        torch.tensor([target.start], device=logits.start.device)))
                    add("span", F.cross_entropy(logits.end.view(1, -1),
                        torch.tensor([target.end], device=logits.end.device)))
                elif spec.type == "list" and spec.items == "text":
                    items = target.spans[:LIST_SLOTS]
                    present = torch.zeros(LIST_SLOTS, device=logits.present.device)
                    present[: len(items)] = 1.0
                    add("present", F.binary_cross_entropy_with_logits(logits.present.float(), present))
                    for slot, (start, end) in enumerate(items):
                        add("span", F.cross_entropy(logits.start[slot].view(1, -1),
                            torch.tensor([start], device=logits.start.device)))
                        add("span", F.cross_entropy(logits.end[slot].view(1, -1),
                            torch.tensor([end], device=logits.end.device)))
                elif target.cand is not None and logits.cand_a is not None and spec.type != "list":
                    a, b, op = target.cand
                    add("cand", F.cross_entropy(logits.cand_a.view(1, -1),
                        torch.tensor([a], device=logits.cand_a.device),
                        label_smoothing=label_smoothing))
                    if b >= 0:
                        add("cand", F.cross_entropy(logits.cand_b.view(1, -1),
                            torch.tensor([b], device=logits.cand_b.device),
                            label_smoothing=label_smoothing))
                    # op is supervised on EVERY solved scalar, including
                    # single-atom "id" answers — an untrained op head turns
                    # read answers into random arithmetic (duration 0% bug)
                    add("op", F.cross_entropy(logits.op.view(1, -1),
                        torch.tensor([op], device=logits.op.device), label_smoothing=0.05))
                elif spec.type == "list" and spec.items in ("number", "date", "time") \
                        and logits.cand_a is not None:
                    items = target.cands[:LIST_SLOTS]
                    present = torch.zeros(LIST_SLOTS, device=logits.present.device)
                    present[: len(items)] = 1.0
                    add("present", F.binary_cross_entropy_with_logits(logits.present.float(), present))
                    for slot, idx in enumerate(items):
                        if idx >= 0:
                            add("cand", F.cross_entropy(logits.cand_a[slot].view(1, -1),
                                torch.tensor([idx], device=logits.cand_a.device),
                                label_smoothing=label_smoothing))
                elif spec.type in ("estimate", "forecast") and target.dist:
                    steps = len(target.dist)
                    add("dist", cross_entropy_rows(logits.dist[:steps], target.dist, dist_neighbor),
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
