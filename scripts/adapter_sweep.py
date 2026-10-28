#!/usr/bin/env python3
"""Adapter-strength sweep (step 2 of the 2026-10-28 pre-registration).

One variable away from K2 (r=32, lr=2e-5, 800 steps on data/snippets_4k.jsonl),
which scored .6103 / .7311 / .7994 / .8824 / 291 of 310 / .5729 on the 12-field
eval. Each arm is:

    adapter on 4k snippets -> merge -> fresh 12-field head at 65k (2 epochs,
    940 steps) -> eval with K2's decode flags

A `k2-control` arm runs K2's exact numbers first: the merge script was lost with
/tmp and reimplemented, so the control checks that the harness reproduces K2
before any delta is trusted.

Pre-registered filter: head loss at step 500. K2 logged 3.28 (the journal said
2.91 from an earlier read; 3.28 is what train.jsonl actually contains). An arm
whose step-500 loss is >= 6.0 is recorded as drift and its eval is skipped.

Writes reports/sweep-<arm>.json per arm and reports/sweep-summary.json, then
pushes a notification (OPERATIONS rule 9). A CUDA failure stops the sweep and
sends a critical push - continuing after the card leaves the bus only wastes
time.

    uv run python scripts/adapter_sweep.py [--arms k2-control,r16,r64] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
BASE = "Qwen/Qwen3.5-2B"
TRAIN12 = "data/doc_train_470_twelve.jsonl"
EVAL12 = "data/doc_eval_40_twelve.jsonl"

# name -> (lora_r, lr, max_steps); everything else is K2's recipe.
ARMS: dict[str, tuple[int, float, int]] = {
    "k2-control": (32, 2e-5, 800),   # must reproduce K2 / .5729
    "r16":        (16, 2e-5, 800),
    "r64":        (64, 2e-5, 800),
    "lr1e-5":     (32, 1e-5, 800),
    "lr4e-5":     (32, 4e-5, 800),
    "steps400":   (32, 2e-5, 400),
    "steps1200":  (32, 2e-5, 1200),
}

EVAL_FLAGS = [
    "--data", EVAL12, "--limit", "40", "--batch-size", "1",
    "--max-length", "65536", "--agg-atoms",
    "--span-tiebreak-eps", "0.25",
    "--span-priors", "reports/field_span_hist_train470.json",
    "--span-prior-alpha", "0.75", "--slot-mask",
    "--candidate-decode", "--candidate-fields", "parties",
]


def last_ckpt(run_dir: Path) -> Path | None:
    cks = sorted((p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
                 key=lambda p: int(p.name.split("-")[-1]))
    return cks[-1] if cks else None


def loss_at_500(run_dir: Path) -> float | None:
    trace = run_dir / "train.jsonl"
    if not trace.exists():
        return None
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row.get("step", 0) >= 500:
            return float(row["loss"])
    return None


def notify(title: str, body: str, mode: str = "full") -> None:
    try:
        subprocess.run(["./scripts/notify.sh", title, body, mode], cwd=ROOT,
                       timeout=30, check=False)
    except Exception:                                     # noqa: BLE001 - never fail a run on a push
        pass


def run_arm(name: str, r: int, lr: float, steps: int) -> dict:
    tag = f"sweep-{name}"
    log = LOGS / f"{tag}.log"
    log.write_text("")
    arm = {"arm": name, "r": r, "lr": lr, "steps": steps, "status": "running"}
    env = os.environ | {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "PYTHONUNBUFFERED": "1"}

    def go(cmd: list[str], timeout: int) -> tuple[int, str]:
        with log.open("a") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env)
        text = log.read_text(errors="replace")
        return p.returncode, text

    # 1. adapter on 4k snippets
    rc, text = go(["uv", "run", "clarke-train", "--backbone", BASE,
                   "--train", "data/snippets_4k.jsonl", "--epochs", "1",
                   "--batch-size", "1", "--max-length", "4096", "--lr", str(lr),
                   "--rerank", "--query-pool", "attn", "--agg-atoms",
                   "--lora-r", str(r), "--lora-alpha", str(r),
                   "--max-steps", str(steps), "--save-every", str(max(200, steps // 2)),
                   "--run-dir", f"runs/{tag}-lora"], timeout=45 * 60)
    if rc != 0:
        arm["status"] = "cuda" if "CUDA" in text or "NVRM" in text else "adapter-failed"
        return arm
    ck = last_ckpt(ROOT / f"runs/{tag}-lora")
    if not ck:
        arm["status"] = "adapter-failed"
        return arm

    # 2. merge (CPU)
    bb = f"runs/bb-{tag}"
    rc, text = go(["uv", "run", "python", "scripts/merge_lora.py", BASE,
                   str(ck / "lora"), bb], timeout=20 * 60)
    if rc != 0:
        arm["status"] = "merge-failed"
        return arm

    # 3. fresh 12-field head at 65k (K2: 2 epochs = 940 steps)
    rc, text = go(["uv", "run", "clarke-train", "--backbone", bb,
                   "--train", TRAIN12, "--epochs", "2", "--batch-size", "1",
                   "--max-length", "65536", "--lr", "1e-4",
                   "--rerank", "--query-pool", "attn", "--agg-atoms",
                   "--save-every", "500", "--run-dir", f"runs/{tag}-head"],
                  timeout=45 * 60)
    if rc != 0:
        arm["status"] = "cuda" if "CUDA" in text or "NVRM" in text else "head-failed"
        return arm
    head = last_ckpt(ROOT / f"runs/{tag}-head")
    arm["checkpoint"] = str(head) if head else None
    arm["loss500"] = loss_at_500(ROOT / f"runs/{tag}-head")

    # 4. pre-registered drift filter
    if arm["loss500"] is not None and arm["loss500"] >= 6.0:
        arm["status"] = "drift-skipped"
        return arm
    if not head:
        arm["status"] = "head-failed"
        return arm

    # 5. eval with K2's decode flags
    out = f"reports/{tag}.json"
    rc, text = go(["uv", "run", "clarke-eval", "--head", str(head), *EVAL_FLAGS,
                   "--out", out], timeout=30 * 60)
    if rc != 0:
        arm["status"] = "cuda" if "CUDA" in text or "NVRM" in text else "eval-failed"
        return arm
    m = json.loads((ROOT / out).read_text())["metrics"]["list-text"]
    gold = m["n"] - m["null_n"]
    arm.update({"status": "ok", "report": out, "exact": m["exact"],
                "tok_f1": m["tok_f1"], "contains": m["contains"],
                "null_ok": m["null_correct"], "answered": m["answered"],
                "gold": gold, "exact_at_all": round(m["exact"] * m["answered"] / gold, 4)})
    return arm


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="comma-separated subset of arms")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    names = [a for a in (args.arms.split(",") if args.arms else ARMS) if a]
    bad = [n for n in names if n not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms: {bad}")
    if args.dry_run:
        print(json.dumps({n: ARMS[n] for n in names}, indent=1))
        return 0

    LOGS.mkdir(exist_ok=True)
    results: list[dict] = []
    for name in names:
        r, lr, steps = ARMS[name]
        print(f"[sweep] {name}: r={r} lr={lr} steps={steps}", flush=True)
        arm = run_arm(name, r, lr, steps)
        results.append(arm)
        print(f"[sweep] {name} -> {arm['status']} {arm.get('exact_at_all', '')}", flush=True)
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports/sweep-summary.json").write_text(json.dumps(results, indent=1))
        if arm["status"] == "cuda":
            notify("Clarke: adapter sweep stopped (GPU fault)",
                   f"{name} hit a CUDA/NVRM failure; sweep aborted. "
                   f"Logs under logs/, partial summary in reports/sweep-summary.json.",
                   "critical")
            return 1

    lines = [f"{a['arm']:12s} r{a['r']:<3d} lr{a['lr']:<7g} {a['steps']:>4d} steps  "
             f"{a['status']:14s} " + (f"exact@all {a.get('exact_at_all', '-')}  "
                                      f"loss500 {a.get('loss500')}" if a["status"] == "ok" else "")
             for a in results]
    body = "\n".join(lines)
    print(body, flush=True)
    notify("Clarke: adapter sweep done", body + "\nreports/sweep-summary.json", "full")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
