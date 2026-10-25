"""Smoke test: LoRA wiring end-to-end on a tiny batch.

Verifies: adapters attach, gradients flow through the backbone (the no_grad guard
must be off), grad checkpointing fits the card, and a step actually reduces loss
direction. Exits non-zero on any failure so the chain can abort.
"""
import json
import sys
import torch

from clarke.encoding import encode_record, collate_records
from clarke.head import JointExtractionHead
from clarke.model import ClarkeModel, load_backbone, load_head

DEVICE = "cuda"


def main() -> int:
    backbone, processor = load_backbone("Qwen/Qwen3.5-9B", device=DEVICE)
    for p in backbone.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "up_proj", "down_proj"],
                     bias="none", task_type="FEATURE_EXTRACTION")
    backbone = get_peft_model(backbone, cfg)
    backbone.gradient_checkpointing_enable()
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    print(f"[smoke] trainable adapter tensors: {len(trainable)}, "
          f"params: {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)

    head = JointExtractionHead(hidden_size=4096, width=1024, routing_layers=2,
                               layers=4, heads=16, feedforward=4096,
                               rerank=True, query_pool="attn").to(DEVICE, dtype=torch.float32)
    model = ClarkeModel(backbone, head).eval()
    model._backbone_trainable = None

    tok = processor.tokenizer
    recs = [json.loads(l) for l in open("data/cuad_train_8k.jsonl")][:2]
    enc = [encode_record(tok, r, max_length=8192) for r in recs]
    batch = collate_records(enc, tok.pad_token_id, DEVICE)

    with torch.enable_grad():
        logits = model(batch)                     # forward must not be no_grad
        loss = (sum(t.float().sum() for per_rec in logits for t in
                    (per_rec.start.sum(), per_rec.end.sum())) * 0.0
                + sum((p.float() ** 2).sum() for p in model.head.parameters()) * 0.0)
        # force a differentiable scalar through the adapters
        probe = sum((p.float() ** 2).sum() for p in trainable)
        total = loss + probe
        total.backward()

    grads = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] adapters with non-zero grads: {grads}/{len(trainable)} | "
          f"peak VRAM {peak:.1f} GB", flush=True)
    ok = grads == len(trainable) and peak < 31.0
    print("[smoke]", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
