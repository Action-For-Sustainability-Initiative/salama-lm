"""Tokenise text files into flat uint16 .bin memmaps for training.

Each document is tokenised and terminated with <|endoftext|> so the model
learns document boundaries. uint16 is safe because vocab (16,384) < 65,536.
Splits off the last --val-frac of each language file as held-out validation
(per-language val sets let us track EN and SW loss separately during training).

Usage: python -m pretraining.prepare_data --sample-dir data/sample --out data/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def tokenize_file(tok: Tokenizer, path: Path, eot_id: int) -> np.ndarray:
    ids: list[int] = []
    doc: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc.append(line)
            elif doc:
                ids.extend(tok.encode("".join(doc).strip()).ids)
                ids.append(eot_id)
                doc = []
    if doc:
        ids.extend(tok.encode("".join(doc).strip()).ids)
        ids.append(eot_id)
    return np.array(ids, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="data/sample")
    parser.add_argument("--tokenizer", default="pretraining/tokenizer/tokenizer.json")
    parser.add_argument("--out", default="data/pilot")
    parser.add_argument("--val-frac", type=float, default=0.01)
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_parts, meta = [], {}
    for lang in ("en", "sw"):
        arr = tokenize_file(tok, Path(args.sample_dir) / f"{lang}.txt", eot_id)
        n_val = max(2048, int(len(arr) * args.val_frac))
        val, train = arr[-n_val:], arr[:-n_val]
        val.tofile(out / f"val_{lang}.bin")
        train_parts.append(train)
        meta[lang] = {"train_tokens": int(len(train)), "val_tokens": int(len(val))}
        print(f"{lang}: {len(train)/1e6:.1f}M train tokens, {len(val)/1e3:.0f}K val tokens")

    # Interleave EN/SW in chunks so any contiguous training window mixes both
    # languages (a straight concat would make early batches all-EN under
    # sequential sampling; harmless with random offsets, but chunked
    # interleaving also keeps the file order-insensitive for future use).
    chunk = 1_000_000
    mixed = []
    a, b = train_parts
    for i in range(0, max(len(a), len(b)), chunk):
        mixed.extend([a[i : i + chunk], b[i : i + chunk]])
    np.concatenate(mixed).tofile(out / "train.bin")
    meta["train_bin_tokens"] = int(sum(len(p) for p in train_parts))
    (out / "META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out / "train.bin", f"({meta['train_bin_tokens']/1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
