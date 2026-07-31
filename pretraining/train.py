"""Pretraining loop for salama-lm.

Design choices, and why:
  - bf16 autocast, no GradScaler: bf16 has fp32's exponent range, so loss
    scaling (an fp16 workaround) is unnecessary.
  - Data comes from pre-tokenised uint16 memory-mapped .bin files, sampled at
    random offsets (the nanoGPT pattern). This sidesteps every known
    Windows DataLoader/streaming issue and is faster at this scale anyway.
  - Gradient accumulation gives large effective batches within 8GB VRAM.
  - Checkpoints contain model + optimiser + step + RNG states, so a resumed
    run is bit-identical to an uninterrupted one (tests verify this).
  - Logging goes to console and a JSONL file per run: loss, val loss per
    language, tokens/sec, peak VRAM. GPU temperature/power are polled from
    nvidia-smi so laptop thermals are part of the experimental record.

Usage:
  python -m pretraining.train --config configs/prototype.yaml [--resume]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pretraining.model import build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class MemmapBatches:
    """Random-offset batch sampler over a flat uint16 token stream."""

    def __init__(self, bin_path: str | Path, n_ctx: int, seed: int):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.tokens) < n_ctx + 1:
            raise ValueError(f"{bin_path} has {len(self.tokens)} tokens; need > {n_ctx + 1}")
        self.n_ctx = n_ctx
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        idx = self.rng.integers(0, len(self.tokens) - self.n_ctx - 1, size=batch_size)
        batch = np.stack([self.tokens[i : i + self.n_ctx].astype(np.int64) for i in idx])
        return torch.from_numpy(batch).to(device, non_blocking=True)


def lr_at(step: int, base_lr: float, warmup: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to 10% of base; the small-LM standard."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def gpu_stats() -> dict:
    """Poll nvidia-smi for temperature/power; returns {} if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(", ")
        return {"gpu_temp_c": float(out[0]), "gpu_power_w": float(out[1])}
    except Exception:
        return {}


@torch.no_grad()
def eval_loss(model, batches: MemmapBatches, micro_bs: int, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        tokens = batches.sample(micro_bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(tokens, return_type="loss").item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model, opt, step: int, cfg: dict,
                    sampler_rng: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": cfg,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "sampler_rng": sampler_rng,
        },
        tmp,
    )
    tmp.replace(path)  # atomic-ish: never leave a half-written checkpoint


def train(cfg: dict, resume: bool = False) -> dict:
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg["model"], seed=tcfg["seed"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
    )

    n_ctx = cfg["model"]["n_ctx"]
    micro_bs = tcfg["micro_batch_size"]
    grad_accum = tcfg["grad_accum"]
    train_batches = MemmapBatches(cfg["data"]["train_bin"], n_ctx, seed=tcfg["seed"])
    val_batches = {
        name: MemmapBatches(path, n_ctx, seed=tcfg["seed"] + 999)
        for name, path in cfg["data"].get("val_bins", {}).items()
    }

    start_step = 0
    ckpt_path = out_dir / "latest.pt"
    if resume and ckpt_path.exists():
        # map_location must be "cpu": RNG states are CPU ByteTensors and must
        # stay that way; optimizer state is auto-moved to param devices on load.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if "sampler_rng" in state:
            train_batches.rng.bit_generator.state = state["sampler_rng"]
        print(f"resumed from step {start_step}")

    n_params = count_params(model)
    tokens_per_step = micro_bs * grad_accum * n_ctx
    print(f"params: {n_params/1e6:.1f}M | tokens/step: {tokens_per_step:,} | "
          f"target tokens: {tokens_per_step * tcfg['max_steps']/1e6:.0f}M")

    log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    model.train()
    t_last, tokens_since = time.perf_counter(), 0
    final = {}

    for step in range(start_step, tcfg["max_steps"]):
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            tokens = train_batches.sample(micro_bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, return_type="loss") / grad_accum
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tcfg["lr"], tcfg["warmup_steps"], tcfg["max_steps"])
        opt.step()
        tokens_since += tokens_per_step

        if step % 20 == 0 or step == tcfg["max_steps"] - 1:
            dt = time.perf_counter() - t_last
            rec = {
                "step": step,
                "loss": round(loss_accum, 4),
                "lr": opt.param_groups[0]["lr"],
                "tok_per_s": round(tokens_since / dt) if dt > 0 else 0,
                "vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                **gpu_stats(),
            }
            t_last, tokens_since = time.perf_counter(), 0
            if tcfg["eval_every"] and step and step % tcfg["eval_every"] == 0:
                for name, vb in val_batches.items():
                    rec[f"val_loss_{name}"] = round(eval_loss(model, vb, micro_bs), 4)
            print(rec)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            final = rec

        if tcfg["ckpt_every"] and step and step % tcfg["ckpt_every"] == 0:
            save_checkpoint(ckpt_path, model, opt, step + 1, cfg,
                            train_batches.rng.bit_generator.state)

    save_checkpoint(out_dir / "final.pt", model, opt, tcfg["max_steps"], cfg,
                    train_batches.rng.bit_generator.state)
    log_file.close()
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)
