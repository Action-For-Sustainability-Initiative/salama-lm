"""Build "Swahili TinyStories" by machine-translating TinyStories V2.

Teaching notes: no Swahili TinyStories corpus exists (verified 2026-07-29),
so we synthesise one with Helsinki-NLP/opus-mt-en-sw (MarianMT, Apache-2.0).
MT output is TRANSLATIONESE — systematically simpler/more literal than
native text. That is a documented limitation, not a secret: the dataset
card must say machine-translated, and the corpus MANIFEST records the MT
model id. For the experiment this is acceptable (we need simple Swahili
narrative register to mirror the English stories), and the sw_web corpus
provides the native-Swahili signal.

The job is TIME-CAPPED (--max-hours): it translates story-by-story until
the cap and writes what it has — a partial corpus is fine, fabricating
speed estimates is not. Run --benchmark first to see stories/sec.

Usage:
  python -m pretraining.translate_stories --benchmark
  python -m pretraining.translate_stories --max-hours 2.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

RAW = Path("data/full/raw")
MODEL_ID = "Helsinki-NLP/opus-mt-en-sw"
SENT_SPLIT = re.compile(r"(?<=[.!?\"])\s+")


def load_model():
    tok = MarianTokenizer.from_pretrained(MODEL_ID)
    model = MarianMTModel.from_pretrained(MODEL_ID).half().cuda().eval()
    return tok, model


@torch.no_grad()
def translate_sentences(tok, model, sentences: list[str], batch_size: int = 48) -> list[str]:
    out: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to("cuda")
        gen = model.generate(**enc, max_new_tokens=160, num_beams=1)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def iter_stories(path: Path):
    buf: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line.strip())
            elif buf:
                yield " ".join(buf)
                buf = []
    if buf:
        yield " ".join(buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--source", default=str(RAW / "en_stories.txt"))
    args = parser.parse_args()

    tok, model = load_model()
    src = Path(args.source)
    stories = iter_stories(src)

    if args.benchmark:
        batch = [next(stories) for _ in range(50)]
        t0 = time.time()
        for story in batch:
            translate_sentences(tok, model, SENT_SPLIT.split(story))
        dt = time.time() - t0
        chars = sum(len(s) for s in batch)
        print(json.dumps({
            "stories_per_s": round(50 / dt, 2),
            "est_stories_in_2p5h": round(50 / dt * 9000),
            "est_mb_in_2p5h": round(50 / dt * 9000 * (chars / 50) / 2**20),
        }))
        return

    out_path = RAW / "sw_stories.txt"
    deadline = time.time() + args.max_hours * 3600
    n, written = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for story in stories:
            sw = " ".join(translate_sentences(tok, model, SENT_SPLIT.split(story)))
            f.write(sw.strip() + "\n\n")
            written += len(sw.encode("utf-8")) + 2
            n += 1
            if n % 500 == 0:
                left = (deadline - time.time()) / 60
                print(f"{n:,} stories, {written/2**20:.0f} MB, {left:.0f} min left")
            if time.time() >= deadline:
                print("time cap reached")
                break

    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest["sw_stories"] = {
        "docs": n, "bytes": written, "sha256": h,
        "est_tokens_m": round(written / 4.05 / 1e6),
        "method": f"machine-translated from TinyStoriesV2 via {MODEL_ID} "
                  f"(greedy, sentence-level); TRANSLATIONESE — see DATA.md",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {n:,} stories, {written/2**20:.0f} MB")


if __name__ == "__main__":
    main()
