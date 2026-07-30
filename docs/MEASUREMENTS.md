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
| SDPA backends available | EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH (**no FLASH** — absent from Windows CUDA wheels) |
| NVML enforced power limit | 117.9 W (nvidia-smi header reports a 59 W state-dependent cap; observed draw under training 40–80 W) |
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
| 40M primary, micro-batch 24–32 fits | **Wrong** — 24 spills; 8 is optimal |
| 8–14 h for the primary run | 13.7 h projected at measured speed for 995M tokens — inside the range, but only after the batch fix; the naive config would have taken **60 h** |
| ~25–40K tok/s at 40M scale | **20,235 tok/s** measured at 48.3M — the estimate was ~1.5× optimistic |
| Peak VRAM ~4.5–6 GB | 3863 MiB at the chosen setting — better than estimated |

## Final primary-run configuration

48.3M params (10 layers, d_model 512, 8 heads, ctx 512, vocab 16,384),
micro_batch 8 × grad_accum 9 = 36,864 tokens/step, 27,000 steps =
**995M tokens = 20.6 tokens/param** (Chinchilla-compute-optimal for this size),
bf16, AdamW, cosine schedule with 700 warmup steps. Projected **13.7 h**.
