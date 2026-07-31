"""Activation extraction via TransformerLens hooks.

Teaching note; what a "residual stream" is and why we read it:
a decoder-only transformer carries information in a running sum (the
residual stream) that every layer reads from and writes back into. By the
final token of a prompt, that vector is the model's summary of everything it
has read. If a concept ("this request is about a hazard") is represented at
all, it is usually represented as a DIRECTION in this space; which is why a
simple linear probe can find it, and why adding a vector can steer behaviour.

We cache `blocks.{i}.hook_resid_post` (the stream after block i) at the LAST
prompt token, for every layer. That gives an [n_prompts, n_layers, d_model]
tensor; the input to both probing and steering.
"""

from __future__ import annotations

import numpy as np
import torch
from tokenizers import Tokenizer
from transformer_lens import HookedTransformer

from pretraining.model import build_model


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[HookedTransformer, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["config"]["model"], device=device, seed=0)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["config"]


@torch.no_grad()
def last_token_resid(model: HookedTransformer, tok: Tokenizer, prompts: list[str],
                     batch_size: int = 32) -> np.ndarray:
    """Residual stream at the final prompt token, all layers.

    Returns float32 array [n_prompts, n_layers, d_model].
    Prompts are LEFT-padded so "final token" is at the same index for all
    rows in a batch, and padding never sits between content and the readout.
    """
    pad = tok.token_to_id("<|pad|>")
    n_layers = model.cfg.n_layers
    names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    out = np.empty((len(prompts), n_layers, model.cfg.d_model), dtype=np.float32)

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encs = [tok.encode(p).ids for p in batch]
        maxlen = max(len(e) for e in encs)
        ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        for j, e in enumerate(encs):
            ids[j, maxlen - len(e):] = torch.tensor(e)
        _, cache = model.run_with_cache(ids.to(model.cfg.device),
                                        names_filter=lambda n: n in names)
        for i, name in enumerate(names):
            out[start:start + len(batch), i] = (
                cache[name][:, -1, :].float().cpu().numpy())
        del cache
    return out
