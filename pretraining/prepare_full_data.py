"""Tokenizer retrain + memmap build for the FULL corpus.

Steps:
  1. Build a balanced tokenizer-training sample (byte-capped slice of each
     source) — training the BPE on the raw 5.5GB would be slow and would
     let the biggest source dominate merge selection.
  2. Train the 16k byte-level BPE (same recipe/special tokens as the pilot).
  3. Tokenize every source with encode_batch (Rust-parallel across cores),
     writing one uint16 .bin per source plus per-language val splits.
  4. Chunk-interleave the sources into train.bin; the mixture ratio IS the
     token counts written (the training sampler draws offsets uniformly).

Sources present in data/full/raw are used; missing ones (e.g. sw_stories
before the MT job has run) are skipped with a warning — rerun this script
after translate_stories.py to rebuild train.bin with the full mixture.

Usage: python -m pretraining.prepare_full_data [--skip-tokenizer]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer

RAW = Path("data/full/raw")
OUT = Path("data/full")
TOK_DIR = Path("pretraining/tokenizer_full")
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]

# source -> (val-set name, tokenizer-sample byte cap)
SOURCES = {
    "en_stories": ("en", 120 * 2**20),
    "en_web": ("en", 80 * 2**20),
    "sw_web": ("sw", 150 * 2**20),
    "sw_wiki": ("sw", 40 * 2**20),
    "sw_stories": ("sw", 40 * 2**20),
    "cs_text": ("cs", 30 * 2**20),
    "parallel_docs": ("cs", 20 * 2**20),
}
VAL_FRAC = 0.005
MIN_VAL_TOKENS = 100_000


def iter_docs(path: Path):
    buf: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf).strip()
                buf = []
    if buf:
        yield "".join(buf).strip()


def build_tokenizer() -> None:
    sample_dir = OUT / "tok_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, (_, cap) in SOURCES.items():
        src = RAW / f"{name}.txt"
        if not src.exists():
            continue
        dst = sample_dir / f"{name}.txt"
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            written = 0
            for line in fin:
                fout.write(line)
                written += len(line.encode("utf-8"))
                if written >= cap:
                    break
        files.append(str(dst))
    print(f"tokenizer sample: {len(files)} files")
    tok = ByteLevelBPETokenizer()
    tok.train(files=files, vocab_size=16384, min_frequency=2,
              special_tokens=SPECIAL_TOKENS)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_DIR / "tokenizer.json"))
    print("tokenizer saved to", TOK_DIR)


def tokenize_source(tok: Tokenizer, name: str, eot: int) -> np.ndarray | None:
    src = RAW / f"{name}.txt"
    if not src.exists():
        print(f"[warn] missing source {name} — skipped (rerun after it exists)")
        return None
    t0 = time.time()
    ids: list[np.ndarray] = []
    batch: list[str] = []
    total = 0
    for doc in iter_docs(src):
        batch.append(doc)
        if len(batch) == 2048:
            for enc in tok.encode_batch(batch):
                ids.append(np.array(enc.ids + [eot], dtype=np.uint16))
                total += len(enc.ids) + 1
            batch = []
    if batch:
        for enc in tok.encode_batch(batch):
            ids.append(np.array(enc.ids + [eot], dtype=np.uint16))
            total += len(enc.ids) + 1
    arr = np.concatenate(ids)
    print(f"  {name}: {total/1e6:.1f}M tokens in {time.time()-t0:.0f}s")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_tokenizer:
        build_tokenizer()
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    val_parts: dict[str, list[np.ndarray]] = {}
    train_parts: list[np.ndarray] = []
    meta: dict = {"sources": {}}
    for name, (val_name, _) in SOURCES.items():
        arr = tokenize_source(tok, name, eot)
        if arr is None:
            continue
        n_val = max(MIN_VAL_TOKENS, int(len(arr) * VAL_FRAC))
        val_parts.setdefault(val_name, []).append(arr[-n_val:])
        train_parts.append(arr[:-n_val])
        meta["sources"][name] = {"train_tokens": int(len(arr) - n_val),
                                 "val_tokens": int(n_val)}

    # chunk-interleave so every window of train.bin mixes sources
    chunk = 2_000_000
    mixed: list[np.ndarray] = []
    longest = max(len(p) for p in train_parts)
    for i in range(0, longest, chunk):
        for part in train_parts:
            if i < len(part):
                mixed.append(part[i:i + chunk])
    train = np.concatenate(mixed)
    train.tofile(OUT / "train.bin")
    meta["train_tokens"] = int(len(train))
    for val_name, parts in val_parts.items():
        v = np.concatenate(parts)
        v.tofile(OUT / f"val_{val_name}.bin")
        meta[f"val_{val_name}_tokens"] = int(len(v))
    (OUT / "META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
