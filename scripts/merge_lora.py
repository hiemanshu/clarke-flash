#!/usr/bin/env python3
"""Merge a clarke-train LoRA checkpoint into a standalone backbone directory.

clarke-train writes adapters to `<checkpoint>/lora/` (PEFT format). A head can
only be trained on the adapted features once the adapter is folded into the base
weights, which is what this does: load base -> attach adapter -> merge_and_unload
-> save a normal HF folder that `clarke-train --backbone <dir>` can read.

Runs on CPU: a 2B merge is a few seconds and needs no GPU, so it cannot disturb
an in-flight training job.

    uv run python scripts/merge_lora.py Qwen/Qwen3.5-2B \
        runs/K2-snip-lora/checkpoint-800/lora runs/bb-merged
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


def load_any(model_id: str, dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "model_type", "").startswith("qwen3_5"):
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: merge_lora.py <base-model> <adapter-dir> <out-dir>")
    base, adapter, out = argv[0], Path(argv[1]), Path(argv[2])
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no adapter_config.json in {adapter}")

    model = load_any(base)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    try:
        proc = AutoProcessor.from_pretrained(base)
    except Exception:                                     # noqa: BLE001 - text-only repos
        proc = AutoTokenizer.from_pretrained(base)
    proc.save_pretrained(out)
    print(f"[merge] {adapter} -> {out} ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
