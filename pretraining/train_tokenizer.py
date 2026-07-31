"""Train the shared bilingual BPE tokenizer.

Teaching notes:
  - A tokenizer maps raw text to integer IDs. We train byte-level BPE
    (GPT-2 style): it starts from raw bytes so *any* string is encodable
    (no unknown tokens), then greedily merges frequent byte pairs into
    subword units. Training it on a 50/50 EN/SW mix means both languages get
    fair subword coverage; a tokenizer trained only on English would
    shatter Swahili words into many tiny pieces.
  - Vocab 16,384: big enough for Swahili's agglutinative morphology
    (verbs carry subject/tense/object affixes: e.g. "anakupenda" =
    a-na-ku-pend-a), small enough that the embedding tables stay a sane
    fraction of a ~20-50M-param model.
  - "Fertility" = average tokens per whitespace word. If SW fertility is
    far above EN's, Swahili text effectively gets less context and more
    compute per sentence; a confound we measure now, not discover later.
  - We reserve chat-role special tokens NOW so the alignment stage never
    needs a retrained tokenizer (which would invalidate the pretrained model).

Usage: python -m pretraining.train_tokenizer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]


def fertility(tokenizer, path: Path, n_lines: int = 2000) -> float:
    toks = words = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            words += len(line.split())
            toks += len(tokenizer.encode(line).ids)
    return toks / max(1, words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="data/sample")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--out", default="pretraining/tokenizer")
    args = parser.parse_args()

    sample = Path(args.sample_dir)
    files = [str(sample / "en.txt"), str(sample / "sw.txt")]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=args.vocab_size, min_frequency=2,
                    special_tokens=SPECIAL_TOKENS)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    stats = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "fertility_en": round(fertility(tokenizer, sample / "en.txt"), 3),
        "fertility_sw": round(fertility(tokenizer, sample / "sw.txt"), 3),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    ratio = stats["fertility_sw"] / stats["fertility_en"]
    print(f"SW/EN fertility ratio: {ratio:.2f} (sanity bar: < 2.0)")


if __name__ == "__main__":
    main()
