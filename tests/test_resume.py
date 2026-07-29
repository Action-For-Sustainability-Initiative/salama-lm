"""Resume correctness: an interrupted-and-resumed run must match an
uninterrupted run.

We save every RNG stream (torch CPU+CUDA, numpy, python, and the data
sampler's generator) in checkpoints precisely so that training is a pure
function of (config, data, seed) regardless of interruptions. This test
trains 20 steps straight, then 10 steps + resume for 10 more from the
checkpoint, and requires the final losses to agree. Tolerance is small but
nonzero because CUDA kernels are not guaranteed bit-deterministic.
"""

import copy

import numpy as np
import pytest
import torch

from pretraining.train import train

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_cfg(tmp_path, out_name: str, max_steps: int, ckpt_every: int) -> dict:
    bin_path = tmp_path / "toy.bin"
    if not bin_path.exists():
        rng = np.random.default_rng(0)
        rng.integers(0, 16384, size=200_000, dtype=np.uint16).tofile(bin_path)
    return {
        "model": dict(n_layers=2, d_model=64, n_heads=2, d_head=32, n_ctx=128,
                      d_vocab=16384, act_fn="gelu", normalization_type="RMS",
                      positional_embedding_type="rotary"),
        "train": dict(seed=7, lr=1.0e-3, betas=[0.9, 0.95], weight_decay=0.1,
                      warmup_steps=5, max_steps=max_steps,
                      micro_batch_size=4, grad_accum=2, grad_clip=1.0,
                      eval_every=0, ckpt_every=ckpt_every),
        "data": {"train_bin": str(bin_path)},
        "out_dir": str(tmp_path / out_name),
    }


@needs_cuda
def test_resumed_run_matches_uninterrupted(tmp_path):
    straight = train(make_cfg(tmp_path, "straight", max_steps=20, ckpt_every=0))

    # interrupted run: latest.pt lands at steps 5 and 10 (saved state = next
    # step to run), the run dies at step 11, and we resume through step 20
    cfg = make_cfg(tmp_path, "resumed", max_steps=11, ckpt_every=5)
    train(copy.deepcopy(cfg))
    cfg["train"]["max_steps"] = 20
    resumed = train(copy.deepcopy(cfg), resume=True)
    assert resumed["step"] == 19, "resume did not continue from the checkpoint"

    assert abs(straight["loss"] - resumed["loss"]) < 2e-2, (
        f"straight {straight['loss']} vs resumed {resumed['loss']}"
    )
