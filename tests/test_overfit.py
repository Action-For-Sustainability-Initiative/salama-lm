"""The tiny-overfit smoke test.

A correct model + optimiser + loss must be able to memorise ONE fixed batch
to near-zero loss in a few hundred steps. Failure means a real bug (masking,
learning rate, label alignment, dtype); this is the cheapest strong
correctness check in deep learning, and it runs before every serious training
change in this project.
"""

import pytest
import torch

from pretraining.model import build_model
from pretraining.train import load_config, lr_at, set_seed

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_overfit_single_batch():
    cfg = load_config("configs/smoke_overfit.yaml")
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    model = build_model(cfg["model"], seed=tcfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            betas=tuple(tcfg["betas"]),
                            weight_decay=tcfg["weight_decay"])
    # one fixed "fake corpus" batch, seeded => deterministic
    tokens = torch.randint(0, cfg["model"]["d_vocab"],
                           (tcfg["micro_batch_size"], cfg["model"]["n_ctx"]),
                           device="cuda")
    first = last = None
    for step in range(tcfg["max_steps"]):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, return_type="loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg["lr"], tcfg["warmup_steps"], tcfg["max_steps"])
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert first > 8.0, f"init loss {first:.2f} suspiciously low for random data"
    assert last < 0.5, f"failed to memorise one batch: final loss {last:.3f}"
