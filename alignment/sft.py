"""Supervised fine-tuning (SFT) on alignment examples.

Teaching notes:
  - SFT continues training the pretrained model on (prompt, response) pairs,
    but the loss is masked to ASSISTANT TOKENS ONLY. We want the model to
    learn "given this request, produce this response" — not to get better at
    predicting the user's words. Without the mask, most of the gradient would
    chase user-turn tokens.
  - Learning rate is ~10x lower than pretraining: we are nudging an existing
    model, not building one, and high LR causes catastrophic forgetting
    (which we monitor via pretraining val loss before/after).

Usage:
  python -m alignment.sft --base checkpoints/pilot_11m/final.pt \
      --data data/alignment_pilot/train.jsonl --out checkpoints/pilot_sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from pretraining.model import build_model
from pretraining.train import set_seed


def encode_masked(tok: Tokenizer, text: str, assistant_id: int):
    """Token ids plus a loss mask that is 1 only after <|assistant|>."""
    ids = tok.encode(text).ids
    try:
        split = ids.index(assistant_id)
    except ValueError:
        return None
    mask = [0] * (split + 1) + [1] * (len(ids) - split - 1)
    return ids, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    # Default must match the tokenizer the FULL corpus was built with. The
    # pilot tokenizer lives at pretraining/tokenizer/ — training with it
    # against a full-corpus base scrambles token ids and produces a model
    # that generates fluent-loss gibberish (found the hard way; see git log).
    parser.add_argument("--tokenizer", default="pretraining/tokenizer_full/tokenizer.json")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    assistant_id = tok.token_to_id("<|assistant|>")
    pad_id = tok.token_to_id("<|pad|>")

    state = torch.load(args.base, map_location="cpu", weights_only=False)
    model = build_model(state["config"]["model"], seed=args.seed)
    model.load_state_dict(state["model"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    encoded = [e for r in rows if (e := encode_masked(tok, r["text"], assistant_id))]
    print(f"{len(encoded)} training examples")

    rng = np.random.default_rng(args.seed)
    n_batches = len(encoded) // args.batch_size
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(encoded))
        for b in range(n_batches):
            batch = [encoded[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
            maxlen = max(len(ids) for ids, _ in batch)
            ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
            for i, (ex_ids, ex_mask) in enumerate(batch):
                ids[i, :len(ex_ids)] = torch.tensor(ex_ids)
                mask[i, :len(ex_mask)] = torch.tensor(ex_mask, dtype=torch.bool)
            ids, mask = ids.cuda(), mask.cuda()

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)                       # [B, T, V]
            # next-token prediction: logits at t predict token t+1, so shift
            targets = ids[:, 1:]
            target_mask = mask[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1)[target_mask.flatten()],
                targets.flatten()[target_mask.flatten()],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print({"epoch": epoch, "step": step, "sft_loss": round(loss.item(), 4)})
            step += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": state["config"],
                "step": state["step"], "sft": vars(args)}, out / "final.pt")
    print(f"saved {out / 'final.pt'}")


if __name__ == "__main__":
    main()
