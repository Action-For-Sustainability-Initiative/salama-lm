"""Model correctness checks that need no trained weights."""

import math

import pytest
import torch

from pretraining.model import build_model, count_params

TINY = dict(n_layers=2, d_model=128, n_heads=4, d_head=32, n_ctx=256,
            d_vocab=16384, act_fn="gelu", normalization_type="RMS",
            positional_embedding_type="rotary")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_init_loss_near_uniform():
    """A random-init LM should predict ~uniformly: loss ~= ln(vocab_size).

    This catches broken init, broken loss masking, and off-by-one label bugs
    (all of which shift init loss far from ln V).
    """
    model = build_model(TINY, seed=0)
    tokens = torch.randint(0, TINY["d_vocab"], (4, 256), device="cuda")
    loss = model(tokens, return_type="loss").item()
    expected = math.log(TINY["d_vocab"])  # ~9.70
    assert abs(loss - expected) < 0.7, f"init loss {loss:.2f}, expected ~{expected:.2f}"


@needs_cuda
def test_causal_masking():
    """Changing a future token must not change past-position logits."""
    model = build_model(TINY, seed=0)
    t1 = torch.randint(0, 16384, (1, 64), device="cuda")
    t2 = t1.clone()
    t2[0, -1] = (t2[0, -1] + 1) % 16384
    l1 = model(t1)[0, :-1]
    l2 = model(t2)[0, :-1]
    assert torch.allclose(l1, l2, atol=1e-4), "future token leaked into past logits"


def test_param_count_formula():
    """Sanity-check the ~12*d^2*L + 2*V*d estimate used in the design report."""
    model = build_model(TINY, seed=0, device="cpu")
    estimate = 12 * TINY["d_model"] ** 2 * TINY["n_layers"] + 2 * TINY["d_vocab"] * TINY["d_model"]
    assert abs(count_params(model) - estimate) / estimate < 0.15
