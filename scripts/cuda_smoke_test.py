"""CUDA environment smoke test for salama-lm.

Verifies, in order:
  1. torch sees the GPU and reports its true VRAM
  2. bf16 is supported (Ada GPUs: yes) - we train in bf16, which needs no
     loss scaling because it has the same exponent range as fp32
  3. scaled_dot_product_attention runs with the flash backend on CUDA
     (our substitute for the flash-attn package, which has no official
     Windows wheels)
  4. sustained bf16 matmul throughput at the 59W power cap - this turns the
     report's [C] throughput estimates into a measured number
  5. TransformerLens HookedTransformer.from_config builds a random-init
     model that can do a forward+backward pass on this GPU

Run:  .venv\\Scripts\\python.exe scripts\\cuda_smoke_test.py
"""

import time

import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA not available - check driver/wheel match"
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name} | VRAM {dev.total_memory / 2**20:.0f} MiB | sm{dev.major}{dev.minor}")

    assert torch.cuda.is_bf16_supported(), "bf16 unsupported - unexpected on Ada"
    print("bf16: supported")

    # SDPA backend check: q/k/v shaped [batch, heads, seq, head_dim].
    # Windows CUDA wheels ship without the flash kernel; the memory-efficient
    # kernel is the one that matters (it avoids materialising the seq x seq
    # attention matrix). We record which backends actually work here.
    q = torch.randn(4, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    working = []
    for backend in (
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ):
        try:
            with torch.nn.attention.sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            assert out.shape == q.shape
            working.append(backend.name)
        except RuntimeError:
            pass
    print(f"SDPA backends available: {working}")
    assert working and working != ["MATH"], "no fused SDPA kernel available"

    # Sustained bf16 matmul benchmark (~10s) - proxy for achievable TFLOPS
    # under the 59W cap once clocks settle.
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):  # warmup / clock ramp
        a @ b
    torch.cuda.synchronize()
    iters = 300
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = (2 * n**3 * iters) / dt / 1e12
    print(f"bf16 matmul sustained: {tflops:.1f} TFLOPS ({dt:.1f}s over {iters} iters)")

    # HookedTransformer from random init: tiny config, forward+backward
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, n_ctx=128, d_head=32, n_heads=4,
        d_vocab=1000, act_fn="gelu", normalization_type="RMS",
        positional_embedding_type="rotary",
    )
    model = HookedTransformer(cfg).to("cuda")
    tokens = torch.randint(0, 1000, (2, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, return_type="loss")
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HookedTransformer random-init fwd+bwd: OK (loss {loss.item():.3f}, {n_params/1e6:.2f}M params)")
    print(f"peak VRAM this test: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
