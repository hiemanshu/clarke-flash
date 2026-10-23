"""Record encoding for clarke-flash.

Builds the single-pass prompt (state + typed schema fields) exactly the way
clef's ``joint_schema_model.py`` does, and tracks the spans the extraction
head needs:

- ``state_range``      token range of the state text (pointer-head domain)
- ``char_offsets``     character offset of every state token (grounding spans)
- ``field spans``      instruction span per field (pooled into the query)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from .candidates import Pool
from .schema import FieldSpec

SYSTEM_PROMPT = (
    "Read the complete state and schema. Extract, compute, estimate, and forecast "
    "every field jointly. Each answer must be a value of that field's declared type, "
    "or null when the state does not contain it."
)
SUFFIX = "JOINT SCHEMA EXTRACTIONS:"


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass
class EncodedField:
    name: str
    spec: FieldSpec
    type_id: int
    field_span: tuple[int, int]


@dataclass
class EncodedRecord:
    record_id: str
    state: str
    input_ids: tuple[int, ...]
    state_range: tuple[int, int]                     # [start, end) state tokens
    char_offsets: tuple[tuple[int, int], ...]        # per state token, into ``state``
    fields: tuple[EncodedField, ...]
    answers: dict[str, Any] = dataclass_field(default_factory=dict, compare=False)
    pool: Any = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 8192,
    max_state_tokens: int | None = None,
) -> EncodedRecord:
    """Encode one {state, fields, answers} record into model input + spans."""
    specs = {name: FieldSpec(name, spec) for name, spec in record["fields"].items()}

    schema_ids: list[int] = list(_tokens(tokenizer, "\n\nSCHEMA FIELDS:\n"))
    encoded_fields: list[EncodedField] = []
    for index, (name, spec) in enumerate(specs.items()):
        block = spec.render(index + 1)
        head, _, instruction = block.partition("INSTRUCTION: ")
        schema_ids.extend(_tokens(tokenizer, head + "INSTRUCTION: "))
        field_start = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, spec.instructions))
        field_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nEND FIELD\n"))
        encoded_fields.append(
            EncodedField(name=name, spec=spec, type_id=spec.type_id, field_span=(field_start, field_end))
        )

    prefix_ids = _tokens(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")
    prefix_ids += _tokens(tokenizer, "<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + SUFFIX,
    )

    state = render(record["state"])
    state_encoded = tokenizer(state, add_special_tokens=False, return_offsets_mapping=True)
    state_ids = list(state_encoded["input_ids"])
    char_offsets = [(int(a), int(b)) for a, b in state_encoded["offset_mapping"]]
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
        char_offsets = char_offsets[:max_state_tokens]
    fixed = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed > max_length:
        raise ValueError(f"schema needs {fixed} tokens before state; max is {max_length}")
    allowed = max_length - fixed
    state_ids = state_ids[:allowed]
    char_offsets = char_offsets[:allowed]

    state_start = len(prefix_ids)
    state_end = state_start + len(state_ids)
    schema_offset = state_end
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    fields = tuple(
        EncodedField(
            name=f.name, spec=f.spec, type_id=f.type_id,
            field_span=(f.field_span[0] + schema_offset, f.field_span[1] + schema_offset),
        )
        for f in encoded_fields
    )
    return EncodedRecord(
        record_id=str(record.get("id", "unknown")),
        state=state,
        input_ids=input_ids,
        state_range=(state_start, state_end),
        char_offsets=tuple(char_offsets),
        fields=fields,
        answers=record.get("answers") or {},
        pool=Pool(state),
    )


def token_span_for_chars(
    record: EncodedRecord, char_start: int, char_end: int
) -> tuple[int, int] | None:
    """Map a character span in the state to inclusive [start_token, end_token]."""
    start = end = None
    for index, (a, b) in enumerate(record.char_offsets):
        if start is None and b > char_start and a < char_end:
            start = index
        if a < char_end and b > char_start:
            end = index
    if start is None or end is None:
        return None
    return start, end


def collate_records(
    records: list[EncodedRecord],
    pad_token_id: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    longest = max(len(record.input_ids) for record in records)
    input_ids = torch.full((len(records), longest), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), longest), dtype=torch.long, device=device)
    for index, record in enumerate(records):
        length = len(record.input_ids)
        input_ids[index, :length] = torch.tensor(record.input_ids, device=device)
        attention_mask[index, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "records": records}
