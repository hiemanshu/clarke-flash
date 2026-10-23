"""Dataset + collate for clarke-flash training.

Records are JSONL dicts ({id, state, fields, answers}); encoding (tokenizer
spans + offset maps + head targets) runs inside DataLoader workers so the
GPU only sees ready batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .encoding import collate_records, encode_record
from .head import JointExtractionHead


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]):
        self.records: list[dict[str, Any]] = []
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class Collator:
    """Encode + build targets in workers; pad into model batches."""

    def __init__(self, tokenizer: Any, max_length: int = 8192, max_fields: int = 32,
                 agg_atoms: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_fields = max_fields
        self.agg_atoms = agg_atoms

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        encoded = []
        targets = []
        for record in records:
            if len(record["fields"]) > self.max_fields:
                continue
            try:
                piece = encode_record(self.tokenizer, record, max_length=self.max_length,
                                  agg_atoms=self.agg_atoms)
            except ValueError:
                continue
            encoded.append(piece)
            targets.append(JointExtractionHead.build_targets(piece))
        if not encoded:
            return None
        batch = collate_records(encoded, self.tokenizer.pad_token_id, torch.device("cpu"))
        batch["targets"] = targets
        return batch


def build_loader(
    dataset: RecordDataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 6,
    max_length: int = 8192,
    drop_last: bool = False,
    agg_atoms: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(tokenizer, max_length=max_length, agg_atoms=agg_atoms),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
