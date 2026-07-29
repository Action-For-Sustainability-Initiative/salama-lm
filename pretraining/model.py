"""Model construction: a randomly initialised decoder-only transformer.

We define the model *as* a TransformerLens HookedTransformer rather than a
plain nn.Module. Functionally it is the same standard architecture
(RMSNorm pre-norm blocks, rotary position embeddings, causal attention, MLP),
but every internal activation is exposed through named hooks
(e.g. `blocks.4.hook_resid_post`), which is what the probing and steering
stages of this project need. Training it from scratch inside TransformerLens
means there is no weight-porting step later.
"""

from __future__ import annotations

from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(model_cfg: dict, device: str = "cuda", seed: int | None = None) -> HookedTransformer:
    """Build a random-init HookedTransformer from a config dict (see configs/*.yaml)."""
    cfg = HookedTransformerConfig(
        n_layers=model_cfg["n_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_head=model_cfg["d_head"],
        n_ctx=model_cfg["n_ctx"],
        d_vocab=model_cfg["d_vocab"],
        act_fn=model_cfg.get("act_fn", "gelu"),
        normalization_type=model_cfg.get("normalization_type", "RMS"),
        positional_embedding_type=model_cfg.get("positional_embedding_type", "rotary"),
        seed=seed,
    )
    model = HookedTransformer(cfg)
    return model.to(device)


def count_params(model: HookedTransformer) -> int:
    return sum(p.numel() for p in model.parameters())
