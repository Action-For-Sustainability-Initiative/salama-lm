"""Stream a bilingual text sample from Hugging Face for tokenizer training
and the Day-1 pilot data.

Sources (both permissively licensed, both streamed — nothing bulk-downloaded):
  EN: roneneldan/TinyStories            (CDLA-Sharing-1.0)
  SW: HuggingFaceFW/fineweb-2 swh_Latn  (ODC-BY)

Writes data/sample/{en,sw}.txt with one document per line-block, separated by
a blank line, and records document counts + SHA256 checksums in
data/sample/MANIFEST.json so the sample is reproducible/verifiable.

Usage: python -m pretraining.fetch_tokenizer_sample --mb-per-lang 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

SOURCES = {
    "en": dict(path="roneneldan/TinyStories", name=None, split="train", field="text"),
    "sw": dict(path="HuggingFaceFW/fineweb-2", name="swh_Latn", split="train", field="text"),
}


def fetch(lang: str, target_bytes: int, out_dir: Path) -> dict:
    src = SOURCES[lang]
    ds = load_dataset(src["path"], name=src["name"], split=src["split"], streaming=True)
    out_path = out_dir / f"{lang}.txt"
    n_docs, written = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row[src["field"]].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            written += len(text.encode("utf-8")) + 2
            n_docs += 1
            if written >= target_bytes:
                break
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {"file": out_path.name, "source": src["path"], "config": src["name"],
            "docs": n_docs, "bytes": written, "sha256": sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb-per-lang", type=int, default=120)
    parser.add_argument("--out", default="data/sample")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for lang in ("en", "sw"):
        print(f"streaming {lang} sample ({args.mb_per_lang} MB)...")
        manifest[lang] = fetch(lang, args.mb_per_lang * 2**20, out_dir)
        print(f"  {manifest[lang]['docs']:,} docs, {manifest[lang]['bytes']/2**20:.0f} MB")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", out_dir / "MANIFEST.json")


if __name__ == "__main__":
    main()
