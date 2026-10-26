"""ClarkeModel = Qwen3.5 backbone + joint extraction head.

``decode_answers`` turns head logits into the typed answer objects the
extraction API returns: values with spans, ``p_null``, ``confidence`` and,
for estimate/forecast, median and the 80% interval.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from .encoding import EncodedRecord, collate_records
from .head import FieldLogits, JointExtractionHead
from .schema import (
    LIST_SLOTS,
    OVERFLOW_BINS,
    FieldSpec,
    clamp_date,
    render_number,
)


class ClarkeModel(torch.nn.Module):
    def __init__(self, language_model: Any, head: JointExtractionHead):
        super().__init__()
        self.language_model = language_model
        self.head = head
        self._backbone_trainable = None

    def forward(self, batch: dict[str, Any]) -> list[list[FieldLogits]]:
        base_model = (
            self.language_model.get_base_model()
            if hasattr(self.language_model, "get_base_model")
            else self.language_model
        )
        text_model = base_model.model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        # Frozen backbone -> no_grad (fast, minimal memory). If adapters (LoRA)
        # are trainable, gradients MUST flow through the backbone.
        if self._backbone_trainable is None:
            self._backbone_trainable = any(p.requires_grad
                                           for p in self.language_model.parameters())
        ctx = torch.enable_grad() if self._backbone_trainable else torch.no_grad()
        with ctx:
            outputs = text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        return self.head(outputs.last_hidden_state, batch["attention_mask"], batch["records"])

    def answer(self, batch: dict[str, Any], null_threshold: float = 0.5,
               slot_threshold: float = 0.5, dedupe: bool = True,
               pointer_temp: float = 1.0) -> list[dict[str, Any]]:
        """One forward pass -> list (aligned with batch["records"]) of {field: answer}.

        Keyed by POSITION, not record_id: eval files can legitimately contain several
        views of the same document (e.g. wiki tables sampled with different field
        sets), so duplicate ids must not collapse answers onto the last one - that
        silently mis-paired fields (and crashed the wiki eval with a float in a
        text slot; 2026-10-25).
        """
        logits = self.forward(batch)
        out = []
        for record, record_logits in zip(batch["records"], logits):
            out.append({
                encoded.name: decode_answer(encoded.spec, record, field_logits,
                                            null_threshold=null_threshold,
                                            slot_threshold=slot_threshold, dedupe=dedupe,
                                            pointer_temp=pointer_temp)
                for encoded, field_logits in zip(record.fields, record_logits)
            })
        return out


def _probability(mass: torch.Tensor) -> float:
    return float(mass.item()) if torch.is_tensor(mass) else float(mass)


def decode_answer(spec: FieldSpec, record: EncodedRecord, logits: FieldLogits,
                  null_threshold: float = 0.5, slot_threshold: float = 0.5,
                  dedupe: bool = True, pointer_temp: float = 1.0) -> dict[str, Any]:
    p_null = torch.sigmoid(logits.null.float())
    answer: dict[str, Any] = {"type": spec.type}

    if spec.type == "text":
        span_answer = _decode_span(logits.start, logits.end, record, spec, p_null,
                                   null_threshold=null_threshold,
                                   pointer_temp=pointer_temp)
        answer.update(span_answer)
    elif spec.type == "list" and spec.items == "text":
        values, spans = [], []
        present = torch.sigmoid(logits.present.float())
        if float(p_null) <= null_threshold:  # absent list: honor the trained null head
            for slot in range(LIST_SLOTS):
                if present[slot] < slot_threshold:
                    continue
                part = _decode_span(logits.start[slot], logits.end[slot], record, spec,
                                    torch.tensor(0.0), pointer_temp=pointer_temp)
                if not part.get("value"):
                    continue
                span = part["span"]
                if dedupe and any(
                        (span[0] < s[1] and s[0] < span[1])      # overlapping start/end
                        or part["value"].strip().lower() == v.strip().lower()
                        for v, s in zip(values, spans)):
                    continue            # independent slots collapse onto the same text
                values.append(part["value"])
                spans.append(span)
        answer["value"] = values or None
        answer["spans"] = spans
        answer["p_null"] = _probability(p_null)
    elif spec.type == "list" and spec.items in ("number", "date", "time"):
        from .candidates import field_atoms

        present = torch.sigmoid(logits.present.float())
        values, confidences, spans = [], [], []
        used_agg = False
        if logits.cand_a is not None and float(p_null) <= null_threshold:
            atoms = field_atoms(spec, record.pool)
            n_plain = len(record.pool.numbers)
            for slot in range(min(LIST_SLOTS, logits.cand_a.shape[0])):
                if present[slot] < 0.5 or not atoms:
                    continue
                probs = torch.softmax(logits.cand_a[slot].float(), dim=-1)
                idx = int(probs.argmax())
                atom = atoms[idx]
                values.append(_round_value(float(atom[0]), spec) if spec.items == "number"
                              else str(atom[0]))
                if spec.items == "number" and idx >= n_plain:
                    used_agg = True
                    spans.append(None)
                else:
                    spans.append(list(atom[1]) if atom[1] else None)
                confidences.append(float(probs[idx]))
        answer["value"] = values or None
        answer["spans"] = spans
        answer["p_null"] = _probability(p_null)
        answer["confidence"] = round(sum(confidences) / len(confidences), 6) if confidences else 0.0
        answer["origin"] = "computed" if used_agg else "read"
    elif spec.type in ("number", "duration", "date", "time", "datetime"):
        from .candidates import OP_IDS, apply_op, field_atoms

        atoms = field_atoms(spec, record.pool)
        if logits.cand_a is None or not atoms:
            answer.update({"value": None, "p_null": _probability(p_null),
                           "confidence": _probability(p_null), "origin": "read"})
            return answer
        probs_a = torch.softmax(logits.cand_a.float(), dim=-1)
        a = int(probs_a.argmax())
        confidence = float(probs_a[a])
        atom = atoms[a]
        if spec.type in ("number", "duration"):
            op = int(logits.op.argmax())
            b = -1
            if op != OP_IDS["id"]:
                probs_b = torch.softmax(logits.cand_b.float(), dim=-1)
                b = int(probs_b.argmax())
                confidence *= float(probs_b[b])
                confidence *= float(torch.softmax(logits.op.float(), dim=-1)[op])
            raw = apply_op(float(atom[0]), float(atoms[b][0]) if b >= 0 else None, op)
            raw = _round_value(raw, spec)
            # index past plain atoms -> virtual column aggregate: the value is
            # computed, and the header anchor is context, never a grounding span
            is_agg = spec.type == "number" and a >= len(record.pool.numbers)
            answer["origin"] = "computed" if (op != OP_IDS["id"] or is_agg) else "read"
            if op == OP_IDS["id"] and atom[1] and not is_agg:
                answer["span"] = list(atom[1])
            if float(p_null) > null_threshold:
                answer.update({"value": None, "guess": raw, "confidence": _probability(p_null)})
            else:
                answer.update({"value": raw, "confidence": round(confidence, 6)})
        else:
            text_value = str(atom[0])
            answer["origin"] = "read"
            if atom[1]:
                answer["span"] = list(atom[1])
            if float(p_null) > null_threshold:
                answer.update({"value": None, "guess": text_value, "confidence": _probability(p_null)})
            else:
                answer.update({"value": text_value, "confidence": round(confidence, 6)})
        answer["p_null"] = _probability(p_null)
    elif spec.type in ("estimate", "forecast"):
        rows = logits.dist
        probabilities = rows.float().softmax(dim=-1)
        steps = probabilities.shape[0]
        per_step = []
        for step in range(steps):
            probs = probabilities[step].tolist()
            centers = spec.grid_centers()
            grid = centers[: -OVERFLOW_BINS]
            # logits width is padded to MAX_GRID_BINS+2; only the spec's real
            # bins may compete (argmax over padding caused an IndexError)
            index = int(probabilities[step][:max(len(grid), 1)].argmax())
            median_info = spec.quantiles(probs)
            value = _snap(grid[index], spec)
            per_step.append({**median_info, "value": value,
                             "p_top": float(probabilities[step][index])})
        if spec.type == "forecast":
            answer["value"] = [step["value"] for step in per_step]
            answer["median"] = [step["median"] for step in per_step]
            answer["p10"] = [step["p10"] for step in per_step]
            answer["p90"] = [step["p90"] for step in per_step]
            answer["confidence"] = round(sum(step["p_top"] for step in per_step) / steps, 6)
        else:
            answer.update(per_step[0])
            answer["value"] = per_step[0]["value"]
            answer["confidence"] = round(per_step[0]["p_top"], 6)
        answer["origin"] = "predicted"
    return answer


def _snap(value: float, spec: FieldSpec) -> float:
    """Snap a bin center onto the user's declared grid, honoring bounds."""
    if spec.step:
        value = spec.min + round((value - spec.min) / spec.step) * spec.step
    return min(max(value, spec.min), spec.max)


def _round_value(value: float, spec: FieldSpec) -> float:
    if spec.integer:
        return float(int(round(value)))
    return round(value, spec.decimals)


def _decode_span(
    start: torch.Tensor, end: torch.Tensor, record: EncodedRecord, spec: FieldSpec,
    p_null: torch.Tensor, null_threshold: float = 0.5, pointer_temp: float = 1.0,
) -> dict[str, Any]:
    """Pointer decode: the answer text is a slice of the input, never generated.

    Even when the field is null the best span is reported as ``guess`` with its
    offsets (mirrors Cloudflare's tracking_number example), so near-misses are
    auditable.
    """
    from .schema import SPAN_MAX_TOKENS

    tokens = start.shape[0]
    if tokens == 0:
        return {"value": None, "p_null": _probability(p_null), "confidence": 0.0}
    # pointer_temp sharpens the positional softmax. span_scale stayed at 1.0 through
    # training in every run so far, so at 65k positions the distribution is diluted;
    # this makes that hypothesis testable without retraining.
    start_probs = torch.softmax(start.float() * pointer_temp, dim=-1)
    end_probs = torch.softmax(end.float() * pointer_temp, dim=-1)
    # Exact argmax over the allowed band without materialising a T x T joint matrix:
    # at 65k tokens that matrix is 65k^2 floats (~17 GB) and OOMs the whole eval.
    # A sliding window of width SPAN_MAX_TOKENS is the same search in O(T * L)
    # (65k x 24 floats ~ 6 MB), and picks the identical (start, end).
    window = min(SPAN_MAX_TOKENS, tokens)
    # pad so EVERY start position gets a full window (otherwise starts in the last
    # window-1 positions are never considered); zero scores can never win, as all
    # real products are strictly positive.
    ends = torch.cat([end_probs, end_probs.new_zeros(window - 1)])
    scores = start_probs.unsqueeze(1) * ends.unfold(0, window, 1)   # [tokens, window]
    flat = int(scores.reshape(-1).argmax())
    i, off = divmod(flat, window)
    j = i + off
    score = scores.reshape(-1)[flat]
    char_start = record.char_offsets[i][0]
    char_end = record.char_offsets[j][1]
    text = record.state[char_start:char_end]
    # subword tokens carry their leading space; trim it off the reported answer
    stripped = text.strip()
    if stripped:
        char_start = char_start + text.index(stripped[0])
        char_end = char_start + len(stripped)
    text = stripped
    return {
        "value": None if float(p_null) > null_threshold else text,
        "guess": text,
        "span": [char_start, char_end],
        "p_null": _probability(p_null),
        "confidence": round(float(score), 6),
        "origin": "read",
    }


def _decode_calendar(spec: FieldSpec, logits: FieldLogits) -> tuple[str, float, bool]:
    bins = [head.argmax(dim=-1) for head in logits.components]
    probs = [float(torch.softmax(head.float(), -1)[b]) for head, b in zip(logits.components, bins)]
    confidence = math.prod(probs)
    names = (["year", "month", "day"] if spec.type == "date"
             else ["hour", "minute", "second"] if spec.type == "time"
             else ["year", "month", "day", "hour", "minute", "second"])
    values: dict[str, int] = {}
    for name, bin_index in zip(names, bins):
        values[name] = int(bin_index)
    invalid = False
    if spec.type in ("date", "datetime"):
        date = clamp_date(values["year"], values["month"] + 1, values["day"] + 1)
        invalid = invalid or date.day != values["day"] + 1 or date.month != values["month"] + 1
    if spec.type == "time":
        moment = dt.time(values["hour"], values["minute"], values["second"])
        return moment.strftime("%H:%M:%S"), confidence, invalid
    if spec.type == "date":
        return date.isoformat(), confidence, invalid
    moment = dt.time(values["hour"], values["minute"], values["second"])
    return f"{date.isoformat()}T{moment.strftime('%H:%M:%S')}", confidence, invalid


# -- grounding: trace answers back to the input ------------------------------
#
# "read" answers must literally occur in the state text. These helpers find
# that occurrence; when a value is *not* found, the answer was computed from
# the input (a total, a date delta) rather than copied from it — a useful
# audit signal, and a free self-check: read answers can be verified exactly.

MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
         r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
DATE_RES = (
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(rf"{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"),
    re.compile(rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\.?,?\s+\d{{4}}"),
)
TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?")
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%Y/%m/%d",
                "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")


def _find_number_span(state: str, value: float | None, decimals: int) -> list[int] | None:
    if value is None:
        return None
    target = round(float(value), decimals)
    for match in NUMBER_RE.finditer(state):
        text = match.group().replace(",", "").replace("$", "")
        try:
            if round(float(text), decimals) == target:
                return [match.start(), match.end()]
        except ValueError:
            continue
    return None


def _parse_date(text: str) -> str | None:
    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)", r"\1", text).replace("Sept", "Sep")
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_time(text: str) -> str | None:
    cleaned = re.sub(r"[.\s]+", "", text).upper()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%SP", "%I:%MP"):
        try:
            return dt.datetime.strptime(cleaned, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return None


def _find_calendar_span(state: str, value: str | None, kind: str) -> list[int] | None:
    if value is None:
        return None
    direct = state.find(value)
    if direct >= 0:
        return [direct, direct + len(value)]
    if kind in ("date", "datetime"):
        iso = value.split("T")[0]
        for regex in DATE_RES:
            for match in regex.finditer(state):
                if _parse_date(match.group()) == iso:
                    return [match.start(), match.end()]
        return None
    for match in TIME_RE.finditer(state):
        if _parse_time(match.group()) == value:
            return [match.start(), match.end()]
    return None


def load_backbone(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load a frozen backbone + tokenizer for ANY causal LM family.

    Qwen3.5 keeps its conditional-generation wrapper (that path is validated end to
    end), everything else falls back to the generic auto classes. Multimodal repos
    (Gemma 4 E-series) are loaded through their conditional-generation class; Clarke's
    forward digs out the text decoder either way, so pixel inputs are never needed.
    """
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", "")
    backbone = None
    if model_type.startswith("qwen3_5"):
        from transformers import Qwen3_5ForConditionalGeneration
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map={"": device})
    else:
        attempts = ("AutoModelForCausalLM", "AutoModelForImageTextToText")
        last_err = None
        for name in attempts:
            try:
                import transformers
                cls = getattr(transformers, name)
                backbone = cls.from_pretrained(model_id, dtype=dtype,
                                               device_map={"": device})
                break
            except Exception as exc:                      # noqa: BLE001
                last_err = exc
        if backbone is None:
            raise RuntimeError(f"could not load {model_id} as a text backbone: {last_err}")

    backbone.config.use_cache = False
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:                                     # noqa: BLE001 - text-only repos
        processor = AutoTokenizer.from_pretrained(model_id)
    if not hasattr(processor, "tokenizer"):
        # AutoProcessor sometimes returns the tokenizer itself (Granite); Clarke only
        # ever needs `.tokenizer`, so wrap it rather than special-casing per family.
        processor = type("P", (), {"tokenizer": processor})()
    return backbone, processor


def load_head(directory: Path, device: str = "cuda", dtype=torch.bfloat16,
              strict: bool = True) -> JointExtractionHead:
    from safetensors.torch import load_file

    config = json.loads((directory / "extraction_head_config.json").read_text())
    config.pop("backbone", None)  # provenance only, not a head constructor arg
    head = JointExtractionHead(**config)
    head.load_state_dict(load_file(directory / "extraction_head.safetensors"), strict=strict)
    return head.to(device=device, dtype=dtype).eval()


def save_head(head: JointExtractionHead, directory: Path,
              backbone: str | None = None) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "hidden_size": head.hidden_norm.normalized_shape[0],
        "width": head.field_projection.out_features,
        "routing_layers": len(head.evidence_layers),
        "layers": len(head.layers),
        "heads": head.layers[0].self_attn.num_heads,
        "feedforward": head.layers[0].linear1.out_features,
        "dropout": 0.0,
        "rerank": bool(getattr(head, "rerank", False)),
        "query_pool": getattr(head, "query_pool", "mean"),
        "span_rerank": bool(getattr(head, "span_rerank", False)),
    }
    if backbone:
        config["backbone"] = backbone  # so eval/reload use the RIGHT backbone
    
    (directory / "extraction_head_config.json").write_text(json.dumps(config, indent=2))
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
              directory / "extraction_head.safetensors")
