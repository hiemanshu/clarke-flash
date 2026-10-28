#!/usr/bin/env python3
"""Unsupervised LM adaptation of a frozen backbone via LoRA, on packed 65k-token blocks.

Why this exists: a rank-256 adapter at 2B is 269M params, so supervised training needs
~12k+ documents at 65k to reach even 1 token/param. No public *labelled* contract corpus
is that large; SEC EDGAR has 1.14M unlabelled material contracts. So we adapt the backbone
with plain causal-LM loss (no labels, no schema), merge the adapter, and then train the
extraction head on the labelled CUAD documents.

Memory: the LM head over a 248k vocab at 65k positions would need ~32GB of logits alone,
so the loss is computed in position chunks and accumulated.

    python scripts/lm_adapt.py --backbone Qwen/Qwen3.5-2B --lm-data data/sec_lm_65k.bin \
        --run-dir runs/lm-2b-r64 --lora-r 64 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clarke.model import load_backbone  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def log(path: Path, payload: dict) -> None:
    payload = {"t": time.strftime("%H:%M:%S"), **payload}
    with open(path, "a") as fh:
        fh.write(json.dumps(payload) + "\n")
    print(payload, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--lm-data", required=True, dest="lm_data",
                    help="uint32 .bin of packed token blocks + its .json meta")
    ap.add_argument("--run-dir", required=True, dest="run_dir")
    ap.add_argument("--lora-r", type=int, default=64, dest="lora_r")
    ap.add_argument("--lora-alpha", type=int, default=None, dest="lora_alpha")
    ap.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=0, help="0 = all blocks")
    ap.add_argument("--chunk", type=int, default=1024, help="positions per LM-head chunk")
    ap.add_argument("--save-every", type=int, default=250, dest="save_every")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, dest="max_grad_norm")
    ap.add_argument("--merge", action="store_true", help="merge LoRA into a full backbone dir")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="2 blocks, then stop")
    ap.add_argument("--vram-cap", type=float, default=0.0, dest="vram_cap",
                    help="hard allocator fraction (protects a concurrent run)")
    args = ap.parse_args(argv)

    if args.vram_cap > 0:
        torch.cuda.set_per_process_memory_fraction(args.vram_cap)
    meta = json.loads(Path(args.lm_data).with_suffix(".json").read_text())
    seq = int(meta["tokens_per_block"])
    data = np.memmap(args.lm_data, dtype=np.uint32, mode="r")
    total_blocks = len(data) // seq
    blocks = total_blocks if args.blocks <= 0 else min(args.blocks, total_blocks)
    print(f"[lm] {total_blocks} blocks of {seq} tokens; using {blocks}", flush=True)

    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    trace = run_dir / "train.jsonl"

    backbone, processor = load_backbone(args.backbone, device="cuda")
    for p in backbone.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    alpha = args.lora_alpha or args.lora_r * 2
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.lora_r, lora_alpha=alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, bias="none", task_type="FEATURE_EXTRACTION"))
    backbone.gradient_checkpointing_enable()
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[lm] lora r={args.lora_r} alpha={alpha} trainable={n_train/1e6:.1f}M", flush=True)

    base = backbone.get_base_model()
    text_model = base.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        lm_head = base.get_output_embeddings()
    assert lm_head is not None, "no lm_head found on backbone"

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_per_epoch = blocks
    total_steps = max(int(steps_per_epoch * args.epochs), 1)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))))

    step = 0
    t0 = time.time()
    for epoch in range(int(np.ceil(args.epochs))):
        for b in range(blocks):
            if step >= total_steps:
                break
            block = np.asarray(data[b * seq:(b + 1) * seq], dtype=np.int64)
            ids = torch.from_numpy(block).unsqueeze(0).cuda()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            hidden = text_model(input_ids=ids, use_cache=False,
                                return_dict=True).last_hidden_state
            loss_total = 0.0
            loss_sum = torch.zeros((), device="cuda")
            n_tok = 0
            for start in range(0, seq - 1, args.chunk):
                end = min(start + args.chunk, seq - 1)
                logits = lm_head(hidden[:, start:end])
                labels = ids[:, start + 1:end + 1]
                loss_sum = loss_sum + F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
                n_tok += 1
            loss = loss_sum / max(n_tok, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or args.dry_run:
                dt = time.time() - t0
                log(trace, {"step": step, "total": total_steps, "epoch": epoch,
                            "loss": round(float(loss), 4), "lr": round(lr_at(step), 8),
                            "bs": 1, "accum": 1,
                            "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                            "blocks_per_s": round(step / max(dt, 1e-6), 3)})
            if args.save_every and step % args.save_every == 0:
                backbone.save_pretrained(run_dir / f"adapter-{step}")
            if args.dry_run and step >= 2:
                print("[lm] dry run complete", flush=True)
                return 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    backbone.save_pretrained(run_dir / "adapter")
    (run_dir / "lm_config.json").write_text(json.dumps({
        "backbone": args.backbone, "lora_r": args.lora_r, "lora_alpha": alpha,
        "lora_dropout": args.lora_dropout, "steps": step, "seq": seq,
        "trainable": n_train, "data": str(args.lm_data), "meta": meta,
    }, indent=2))
    log(trace, {"step": step, "total": total_steps, "event": "trained",
                "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})

    if args.merge:
        merged = backbone.merge_and_unload()
        merged.save_pretrained(run_dir / "merged")
        processor.tokenizer.save_pretrained(run_dir / "merged")
        log(trace, {"event": "merged", "path": str(run_dir / "merged")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
