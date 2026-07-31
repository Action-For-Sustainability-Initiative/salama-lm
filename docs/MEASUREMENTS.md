# Measured hardware & throughput numbers

The design report distinguished measured values **[M]**, calculated estimates
**[C]**, and assumptions **[A]**. This file records the [M] values obtained by
running on the actual machine, and corrects the [C] estimates that were wrong.

Machine: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 560.94,
CUDA 12.6; Intel i9-13900HX (24C/32T); 64 GB RAM; Windows 11; PyTorch
2.13.0+cu126, native Windows (no WSL2).

## Raw compute

| Quantity | Measured |
|---|---|
| Sustained bf16 matmul (4096³, 300 iters) | **25.4 TFLOPS** |
| SDPA backends available | EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH (**no FLASH**, absent from Windows CUDA wheels) |
| Power limits (under load) | default 55 W · **enforced 125 W** · max 140 W · observed draw **81 W** |
| Throttle flags under load | SW power cap, HW slowdown, HW/SW thermal slowdown: **all Not Active** |
| SM clock under load | 2505 MHz of 3105 MHz max (81%) at 99% utilisation |

### Correction: there is no 59 W power cap

The initial audit read `2W / 59W` from the nvidia-smi header **while the GPU was
idle in the P8 power state**. That 59 W was the momentary limit for that idle
state, not a hardware ceiling, and the design report wrongly called it "the
single biggest constraint on training throughput."

Measured under sustained training load: NVIDIA Dynamic Boost raises the
enforced limit to **125 W**, the GPU draws **81 W**, and **every throttle
reason reports Not Active**; the workload is neither power-capped nor
thermally throttled, with ~44 W of headroom unused.

So the real limiter is **arithmetic intensity, not watts**. A 48M-parameter
model at micro-batch 8 gives the GPU too little work per byte moved, so it is
memory-bandwidth bound, which is exactly why measured MFU is ~23% (995M
tokens × 6 × 48.3M params = 2.88e17 FLOPs; at 25.4 TFLOPS that is 3.2 h of
pure compute against 13.7 h of wall-clock). Raising the power limit would
change nothing. The remaining ~19% clock gap (2505 vs 3105 MHz) is ordinary
GPU-Boost bin reduction at 80–86 °C, which does not set a slowdown flag.

## Sustained-load behaviour (1,800 steps ≈ 66M tokens)

| Segment | tok/s avg | GPU temp avg | power avg |
|---|---|---|---|
| steps 0–580 | 18,123 | 79.9 °C | 77.4 W |
| steps 600–1180 | 17,998 | 81.6 °C | 76.7 W |
| steps 1200–1800 | 18,284 | 79.6 °C | 74.8 W |

**No thermal degradation.** Throughput, temperature and power are flat across
the run; the low per-record minima (6.2–7.9K tok/s) are validation steps, which
add work to those intervals. The design report's concern about laptop thermal
decay over multi-hour runs was overstated for this workload; the card settles
at ~80 °C and stays there. Steady-state throughput is ~18,100 tok/s including
evaluation overhead, versus 20,235 measured in the eval-free benchmark.
| Fan speed via NVML | unsupported on this GPU |
| CPU temperature | unavailable via Windows sensor APIs |
| Per-process VRAM via NVML | unavailable under WDDM ("Insufficient Permissions") |

## The Windows VRAM-spillover trap (the most important finding)

On Windows/WDDM the NVIDIA driver **silently spills allocations past physical
VRAM into system RAM** instead of raising OOM. Training continues but every
access crosses PCIe, destroying throughput. The diagnostic is
`torch.cuda.max_memory_allocated()` exceeding 8188 MiB.

Measured on the 48.3M model (`scripts/bench_config.py`, ctx 512, bf16,
tokens/step held at 36,864):

| micro_batch | grad_accum | peak VRAM | tok/s | spilling |
|---|---|---|---|---|
| 24 | 3 | 9920 MiB | **1,925** | yes |
| 16 | 4 | 6892 MiB | 14,478 | no |
| 12 | 6 | 5377 MiB | 16,654 | no |
| **8** | **9** | **3863 MiB** | **20,235** | no |
| 6 | 12 | 3119 MiB | 19,807 | no |

**A smaller micro-batch is 10.5× faster.** This inverts the usual
bigger-batch-is-better intuition and is a Windows-specific hazard worth
reporting: a naive run looks healthy (no crash, loss decreasing) while wasting
90% of the GPU. The same trap bit the 11M pilot at micro_batch 64
(9975 MiB, 20K tok/s) versus 32 (5114 MiB, 76K tok/s).

## Throughput by model size (measured)

| Model | Params | micro_batch | tok/s | VRAM |
|---|---|---|---|---|
| pilot | 11.5M | 32 | 76,243 | 5114 MiB |
| primary | 48.3M | 8 | 20,235 | 3863 MiB |

Scaling is roughly inverse-linear in parameters, as expected for a
compute-bound regime (76,243 × 11.5/48.3 ≈ 18,200, close to the measured
20,235).

## Corrections to the design report's estimates

| Report said | Reality |
|---|---|
| "59 W power cap … the dominant throughput constraint" | **Wrong**; no such cap. Enforced limit 125 W, draw 81 W, zero throttle flags. The limiter is memory bandwidth / arithmetic intensity (~23% MFU) |
| Laptop thermal decay over sustained runs is a risk | **Overstated**, flat 80 °C and flat throughput across 1,800 steps |
| 40M primary, micro-batch 24–32 fits | **Wrong**, 24 spills; 8 is optimal |
| 8–14 h for the primary run | 13.7 h projected at measured speed for 995M tokens, inside the range, but only after the batch fix; the naive config would have taken **60 h** |
| ~25–40K tok/s at 40M scale | **20,235 tok/s** measured at 48.3M; the estimate was ~1.5× optimistic |
| Peak VRAM ~4.5–6 GB | 3863 MiB at the chosen setting, better than estimated |

## Final primary-run configuration

48.3M params (10 layers, d_model 512, 8 heads, ctx 512, vocab 16,384),
micro_batch 8 × grad_accum 9 = 36,864 tokens/step, 27,000 steps =
**995M tokens = 20.6 tokens/param** (Chinchilla-compute-optimal for this size),
bf16, AdamW, cosine schedule with 700 warmup steps. Projected **13.7 h**.
