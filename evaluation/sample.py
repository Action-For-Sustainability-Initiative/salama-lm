"""Generate completions from a trained checkpoint, per language.

The pilot's go/no-go check is qualitative: is Swahili output morphologically
Swahili-like (real agreement prefixes, plausible word forms) or word salad?
We sample with temperature 0.8 and top-k 50; standard settings that balance
diversity against degeneration (pure argmax decoding loops; pure sampling
from the full distribution is noisy at small scale).

Usage:
  python -m evaluation.sample --ckpt checkpoints/pilot_11m/final.pt \
      --out logs/pilot_samples.md [--n 10] [--max-tokens 120]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pretraining.model import build_model

PROMPTS = {
    "en": [
        "Once upon a time, there was a little girl named",
        "Tom and his dog went to the park. Suddenly,",
        "The little bird could not fly because",
        "One day, Amina found a shiny box in the garden.",
        "\"Can I help you?\" asked the old man.",
    ],
    "sw": [
        "Habari za leo ni",
        "Watoto walikuwa wakicheza mpira uwanjani wakati",
        "Serikali ya Tanzania imesema kuwa",
        "Siku moja, mtoto mdogo aliamua",
        "Shule yetu ina walimu wazuri kwa sababu",
    ],
}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="pretraining/tokenizer/tokenizer.json")
    parser.add_argument("--out", default="logs/samples.md")
    parser.add_argument("--n", type=int, default=10, help="samples per language")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(state["config"]["model"], seed=args.seed)
    model.load_state_dict(state["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    torch.manual_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Samples from {args.ckpt} (step {state['step']})\n"]
    for lang, prompts in PROMPTS.items():
        lines.append(f"\n## {lang}\n")
        for i in range(args.n):
            prompt = prompts[i % len(prompts)]
            ids = torch.tensor([tok.encode(prompt).ids], device="cuda")
            out = model.generate(ids, max_new_tokens=args.max_tokens,
                                 temperature=0.8, top_k=50, verbose=False,
                                 eos_token_id=tok.token_to_id("<|endoftext|>"))
            text = tok.decode(out[0].tolist())
            lines.append(f"**[{lang}-{i}]** {text}\n")
            print(f"[{lang}-{i}] {text[:160]}...")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
