"""Measure peak VRAM and throughput for candidate (micro_batch, grad_accum) pairs.

Why this exists: on Windows/WDDM the NVIDIA driver silently spills excess
allocations into system RAM instead of raising OOM. Training keeps running but
crosses PCIe on every access, costing ~3x throughput. The tell is
torch.cuda.max_memory_allocated() exceeding physical VRAM. This script finds
the largest micro-batch that stays comfortably under the physical limit, and
records real tokens/sec; turning the design report's estimates into
measurements.

All candidates keep tokens/step constant, so the optimisation trajectory is
statistically equivalent across them.

Usage:
  python scripts/bench_config.py --config configs/primary_48m.yaml \
      --candidates 24,3 16,4 12,6 8,9
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pretraining.model import build_model, count_params
from pretraining.train import MemmapBatches, load_config, set_seed


def bench(cfg: dict, micro_bs: int, grad_accum: int, steps: int = 12) -> dict:
    set_seed(cfg["train"]["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg["model"], seed=cfg["train"]["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = MemmapBatches(cfg["data"]["train_bin"], cfg["model"]["n_ctx"], seed=0)
    model.train()

    for _ in range(3):  # warmup: allocator settles, clocks ramp
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batches.sample(micro_bs), return_type="loss") / grad_accum
            loss.backward()
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batches.sample(micro_bs), return_type="loss") / grad_accum
            loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens = steps * micro_bs * grad_accum * cfg["model"]["n_ctx"]
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt
    torch.cuda.empty_cache()
    return {"micro_batch": micro_bs, "grad_accum": grad_accum,
            "tokens_per_step": micro_bs * grad_accum * cfg["model"]["n_ctx"],
            "peak_vram_mib": round(peak),
            "tok_per_s": round(tokens / dt)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="micro_bs,grad_accum pairs e.g. 24,3 12,6")
    args = parser.parse_args()
    cfg = load_config(args.config)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"physical VRAM: {total_vram:.0f} MiB | params: "
          f"{count_params(build_model(cfg['model'], seed=0))/1e6:.1f}M")
    torch.cuda.empty_cache()

    rows = []
    for cand in args.candidates:
        mb, ga = (int(x) for x in cand.split(","))
        r = bench(cfg, mb, ga)
        r["spilling"] = r["peak_vram_mib"] > total_vram * 0.92
        rows.append(r)
        print(json.dumps(r))

    safe = [r for r in rows if not r["spilling"]]
    if safe:
        best = max(safe, key=lambda r: r["tok_per_s"])
        steps = cfg["train"]["max_steps"]
        hours = steps * best["tokens_per_step"] / best["tok_per_s"] / 3600
        print(f"\nBEST SAFE: micro_batch={best['micro_batch']} "
              f"grad_accum={best['grad_accum']} -> {best['tok_per_s']:,} tok/s, "
              f"{best['peak_vram_mib']} MiB")
        print(f"  at max_steps={steps:,}: {hours:.1f} h for "
              f"{steps*best['tokens_per_step']/1e9:.2f}B tokens")
        for target in (0.8, 1.0, 1.2):
            s = int(target * 1e9 / best["tokens_per_step"])
            print(f"  {target}B tokens -> max_steps={s:,}, "
                  f"{s*best['tokens_per_step']/best['tok_per_s']/3600:.1f} h")
    else:
        print("\nALL candidates spill; reduce n_ctx or model size.")


if __name__ == "__main__":
    main()
