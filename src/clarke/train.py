"""Trainer for clarke-flash: frozen Qwen3.5 backbone, trained joint head.

The backbone runs one forward pass under ``no_grad`` (bf16, ~19 GB resident
on a 32 GB card); only the ~120M-parameter extraction head gets gradients.
Pass ``--lora`` to additionally train low-rank adapters in a second phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data import Collator, RecordDataset, build_loader
from .head import JointExtractionHead
from .model import ClarkeModel, load_backbone, load_head, save_head


def cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * (0.5 + 0.5 * math.cos(math.pi * min(progress, 1.0)))


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(args.run_dir or
                            f"runs/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.jsonl"
        self.log(f"run_dir={self.run_dir}")
        self.log(f"watch with: scripts/watch_training.sh   (tail -f {self.log_path})")
        # DATA PROVENANCE: record exactly which files (and content) this run trained on.
        # Silent data swaps invalidated comparisons once (2026-10-26); this prevents it.
        manifest = []
        for p in self.args.train:
            path = Path(p)
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            manifest.append({"path": str(path), "sha256": h.hexdigest()[:32],
                             "bytes": path.stat().st_size})
        self.log(json.dumps({"event": "data", "files": manifest}))

        print(f"[{time.strftime('%H:%M:%S')}] loading backbone {args.backbone} ...", flush=True)
        self.backbone, self.processor = load_backbone(args.backbone, device=str(self.device))
        config = getattr(self.backbone.config, "text_config", self.backbone.config)
        self.hidden_size = int(config.hidden_size)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.brier_null = float(getattr(args, "brier_null", 0.0))
        self.lora = int(getattr(args, "lora_r", 0)) > 0
        if self.lora:
            from peft import LoraConfig, get_peft_model
            # Qwen3.5 is a hybrid: 18 of 24 blocks use linear attention whose projections
            # are in_proj_qkv/z/a/b + out_proj, so the old q/k/v/o list adapted only the 6
            # full-attention blocks. conv1d is deliberately excluded: PEFT requires the rank
            # to be divisible by its groups (6144 here).
            lcfg = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                                "in_proj_a", "in_proj_b", "out_proj"],
                bias="none", task_type="FEATURE_EXTRACTION")
            self.backbone = get_peft_model(self.backbone, lcfg)
            if getattr(args, "lora_grad_checkpoint", True):
                self.backbone.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            self.log(f"lora r={args.lora_r} alpha={args.lora_alpha} "
                     f"trainable_backbone={trainable/1e6:.1f}M "
                     f"grad_ckpt={getattr(args, 'lora_grad_checkpoint', True)}")

        self.head = JointExtractionHead(
            hidden_size=self.hidden_size, width=args.width,
            routing_layers=args.routing_layers, layers=args.layers,
            heads=args.heads, feedforward=args.feedforward,
            rerank=args.rerank, query_pool=args.query_pool, span_rerank=args.span_rerank,
            length_head=args.length_head,
        ).to(self.device, dtype=torch.float32)
        if args.init_head:
            prior = load_head(Path(args.init_head), device=str(self.device),
                              dtype=torch.float32, strict=False)
            transferred = self.head.load_state_dict(prior.state_dict(), strict=False)
            self.log(f"init-head from {args.init_head}: missing={len(transferred.missing_keys)} "
                     f"unexpected={len(transferred.unexpected_keys)}")
            del prior
        self.model = ClarkeModel(self.backbone, self.head).eval()
        self.log(json.dumps({"event": "loaded", "hidden_size": self.hidden_size,
                             "head_params": sum(p.numel() for p in self.head.parameters())}))

        params = list(self.head.parameters())
        if self.lora:
            params += [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        self.step = 0
        self._diverge_hits = 0
        if args.resume:
            self._load_checkpoint(Path(args.resume))

        tokenizer = self.processor.tokenizer
        accum = max(1, args.grad_accum)
        self.accum = accum
        self.micro_batch = max(1, args.batch_size // accum)
        self.train_set = RecordDataset([Path(p) for p in args.train])
        self.field_weights = self._field_weights(args.field_balanced)
        self.slot_mask_train = bool(getattr(args, "slot_mask_train", False))
        self.slot_assign = bool(getattr(args, "slot_assign", False))
        self.train_loader = build_loader(self.train_set,
                                         tokenizer, self.micro_batch, True,
                                         args.workers, args.max_length,
                                         agg_atoms=args.agg_atoms)
        self.val_loader = build_loader(RecordDataset([Path(args.val)]), tokenizer,
                                       args.batch_size, False, args.workers, args.max_length,
                                       agg_atoms=args.agg_atoms) \
            if args.val else None
        # schedule spans from the RESUMED step, not from zero; the loader now
        # yields MICRO batches, optimizer steps are one per ACCUM group
        steps_per_epoch = max(1, -(-len(self.train_loader) // accum))
        self.total_steps = max(self.step + args.epochs * steps_per_epoch, 1)

    # -- plumbing -----------------------------------------------------------

    def log(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _write(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _field_weights(self, alpha: float) -> dict[str, float] | None:
        """Inverse-frequency weights so a field with 3 examples is not averaged away."""
        if not alpha:
            return None
        counts: dict[str, int] = {}
        for record in getattr(self, "train_set", []).records:
            for name, answer in (record.get("answers") or {}).items():
                if answer.get("value") is not None:
                    counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        mean = sum(counts.values()) / len(counts)
        weights = {name: min(max((mean / max(c, 1)) ** alpha, 1.0), 8.0)
                   for name, c in counts.items()}
        self.log(json.dumps({"event": "field_weights", "alpha": alpha,
                             "n_fields": len(weights),
                             "most_weighted": sorted(weights.items(), key=lambda kv: -kv[1])[:5]}))
        return weights

    def save_checkpoint(self) -> Path:
        directory = self.run_dir / f"checkpoint-{self.step}"
        save_head(self.head, directory, backbone=self.args.backbone)
        if self.lora:
            self.backbone.save_pretrained(directory / "lora")
        torch.save({"optimizer": self.optimizer.state_dict(), "step": self.step,
                    "args": vars(self.args)}, directory / "train_state.pt")
        keep = sorted(self.run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for stale in keep[:-3]:
            for part in stale.iterdir():
                part.unlink()
            stale.rmdir()
        return directory

    def _load_checkpoint(self, directory: Path) -> None:
        from safetensors.torch import load_file

        self.head.load_state_dict(load_file(directory / "extraction_head.safetensors"), strict=True)
        state = torch.load(directory / "train_state.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        self.log(f"resumed from {directory} at step {self.step}")

    # -- steps ---------------------------------------------------------------

    def training_step(self, batch: dict[str, Any], scale: float = 1.0) -> dict[str, float]:
        records = batch["records"]
        targets = batch["targets"]
        model_batch = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "records": records,
        }
        outputs = self.model(model_batch)
        report = self.head.compute_loss(outputs, records, targets,
                                        brier_null=self.brier_null,
                                        field_weights=getattr(self, "field_weights", None),
                                        slot_mask_train=getattr(self, "slot_mask_train", False),
                                        slot_assign=getattr(self, "slot_assign", False))
        metrics = {key: float(value.detach() if torch.is_tensor(value) else value)
                   for key, value in report.items()}
        # scale = this micro-batch's share of the accumulated group (record-
        # weighted), so K accumulated backwards == one plain backward of the
        # full effective batch, even with ragged final groups
        (report["loss"] * scale).backward()
        return metrics

    @torch.no_grad()
    def validation(self) -> dict[str, float]:
        assert self.val_loader is not None
        sums: dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            if batch is None:
                continue
            outputs = self.model({
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "records": batch["records"],
            })
            report = self.head.compute_loss(outputs, batch["records"], batch["targets"],
                                        brier_null=self.brier_null,
                                        field_weights=getattr(self, "field_weights", None),
                                        slot_mask_train=getattr(self, "slot_mask_train", False),
                                        slot_assign=getattr(self, "slot_assign", False))
            for key, value in report.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
            if count >= self.args.val_batches:
                break
        return {key: value / max(count, 1) for key, value in sums.items()}

    # -- loop ---------------------------------------------------------------

    def run(self) -> None:
        args = self.args
        accum = self.accum
        steps_per_epoch = max(1, -(-len(self.train_loader) // accum))
        self.log(f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} "
                 f"(effective_bs={args.batch_size} micro={self.micro_batch} accum={accum})")
        optimizer = self.optimizer
        window_start = time.time()
        window_records = 0
        for epoch in range(args.epochs):
            iterator = iter(self.train_loader)
            while True:
                micro_batches: list[dict[str, Any]] = []
                while len(micro_batches) < accum:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    if batch is not None:
                        micro_batches.append(batch)
                if not micro_batches:
                    break
                n_recs = sum(len(b["records"]) for b in micro_batches)
                lr = cosine_lr(self.step, self.total_steps, args.warmup, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                metrics: dict[str, float] = {}
                for batch in micro_batches:
                    scale = len(batch["records"]) / max(n_recs, 1)
                    step_metrics = self.training_step(batch, scale=scale)
                    for key, value in step_metrics.items():
                        metrics[key] = metrics.get(key, 0.0) + value * scale
                    window_records += len(batch["records"])
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), args.clip)
                optimizer.step()
                self.step += 1

                # guard against design-level blowups (attempt-5 span divergence
                # cooked for 80 min before the val numbers admitted it)
                if args.diverge_stop and metrics["loss"] > args.diverge_stop:
                    self._diverge_hits += 1
                    if self._diverge_hits >= 2:
                        directory = self.save_checkpoint()
                        self.log(f"DIVERGE GUARD: step loss {metrics['loss']:.1f} > "
                                 f"{args.diverge_stop} twice in a row; saved {directory}, stopping")
                        return
                else:
                    self._diverge_hits = 0

                if args.max_steps and self.step >= args.max_steps:
                    directory = self.save_checkpoint()
                    self.log(f"hit --max-steps {args.max_steps}; saved {directory}")
                    return
                if self.step % args.log_every == 0:
                    elapsed = time.time() - window_start
                    records_per_second = window_records / max(elapsed, 1e-6)
                    gpu_gb = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                    self._write({"step": self.step, "epoch": epoch, "lr": round(lr, 7),
                                 "total": self.total_steps, "bs": args.batch_size,
                                 "accum": accum,
                                 "gpu_gb": round(gpu_gb, 1), "recs_per_s": round(records_per_second, 2),
                                 **{key: round(value, 4) for key, value in metrics.items()}})
                    self.log(f"step {self.step}/{self.total_steps} loss={metrics['loss']:.3f} "
                             f"({records_per_second:.2f} rec/s, {gpu_gb:.1f} GB) "
                             f"-> {self.log_path}")
                    window_start, window_records = time.time(), 0

                if self.step % args.save_every == 0:
                    directory = self.save_checkpoint()
                    self.log(f"saved checkpoint {directory}")
                if self.val_loader and self.step % args.val_every == 0:
                    val = self.validation()
                    self._write({"step": self.step, "val": True, "total": self.total_steps,
                                 **{key: round(value, 4) for key, value in val.items()}})
                    self.log(f"val step {self.step}: loss={val.get('loss', float('nan')):.3f}")
        self.save_checkpoint()
        self.log("training complete")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the clarke extraction head")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-9B",
                        help="backbone repo or local dir; Cloudflare/clef-flash is a transfer-init option")
    parser.add_argument("--train", nargs="+", required=True, help="training JSONL paths")
    parser.add_argument("--val", default=None, help="validation JSONL path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", dest="grad_accum", type=int, default=1,
                        help="accumulate N micro-batches (batch-size/N each) into one step")
    parser.add_argument("--diverge-stop", dest="diverge_stop", type=float, default=0.0,
                        help="abort run if step loss exceeds N twice in a row (0=off)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=0,
                        help="LoRA rank (0 = frozen backbone, head-only training)")
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.05)
    parser.add_argument("--no-lora-grad-checkpoint", dest="lora_grad_checkpoint",
                        action="store_false", help="disable gradient checkpointing with LoRA")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--routing-layers", dest="routing_layers", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--feedforward", type=int, default=4096)
    parser.add_argument("--field-balanced", dest="field_balanced", type=float, default=0.0,
                        help="exponent alpha for inverse-frequency field weighting; 0 disables. "
                             "Rare fields (3-15 instances) currently contribute nothing to the "
                             "averaged loss and default to null")
    parser.add_argument("--slot-assign", dest="slot_assign", action="store_true",
                        help="match gold list items to slots by the model's current confidence "
                             "instead of annotation order, then compact to slots 0..n-1")
    parser.add_argument("--slot-mask-train", dest="slot_mask_train", action="store_true",
                        help="mask earlier list items' tokens when supervising later slots, so "
                             "the head learns boundaries under the same constraint we decode with")
    parser.add_argument("--brier-null", dest="brier_null", type=float, default=0.0,
                        help="weight of a Brier term on p_null (proper scoring rule: "
                             "penalises over/under-confidence, not just ranking)")
    parser.add_argument("--rerank", action="store_true",
                        help="Tier B: pairwise cross-encoder reranker for candidate scoring")
    parser.add_argument("--query-pool", dest="query_pool", default="mean",
                        choices=["mean", "last", "attn"],
                        help="Tier A: field-query pooling over the instruction span")
    parser.add_argument("--length-head", dest="length_head", action="store_true",
                        help="predict each span's extent (a length distribution per slot) and "
                             "multiply it into the span score at decode; the score otherwise has "
                             "no term for how long this field's answers usually are")
    parser.add_argument("--span-rerank", dest="span_rerank", action="store_true",
                        help="pair-net correction over base pointer top-k for text spans")
    parser.add_argument("--agg-atoms", dest="agg_atoms", action="store_true",
                        help="virtual CSV column-aggregate atoms (table-domain wall)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=0,
                        help="stop after N steps (smoke tests)")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", dest="val_batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint directory")
    parser.add_argument("--init-head", dest="init_head", default=None,
                        help="initialize head weights from another checkpoint (partial, strict=False)")
    args = parser.parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    main()
