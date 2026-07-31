"""Fetch the full pretraining corpus (report section 14 mixture).

Targets (tokens, converted to bytes at the measured ~4.05 bytes/token from
the pilot sample); the mixture is controlled by how many bytes of each
source end up in the final train.bin, since the sampler draws uniformly:

  en_stories   400M tok   TinyStoriesV2-GPT4-train.txt (CDLA-Sharing-1.0)
  en_web       200M tok   HuggingFaceFW/fineweb-edu sample-10BT (ODC-BY), streamed
  sw_web       550M tok   HuggingFaceFW/fineweb-2 swh_Latn (ODC-BY), streamed
  sw_wiki       all       wikimedia/wikipedia 20231101.sw (CC-BY-SA-3.0)

NOTE (verified 2026-07-29): OPUS-100 has NO en-sw config; Swahili is not
among its 100 pairs. Parallel EN-SW data is instead produced by
translate_stories.py, which emits sentence-aligned pairs as a by-product of
the TinyStories MT job (parallel_sentences.tsv).

sw_stories (MT-translated TinyStories) and code-switched text are produced
by separate scripts (translate_stories.py, synth_codeswitch.py) after this
fetch completes.

Each source is written to data/full/raw/<name>.txt (docs separated by blank
lines) and recorded in MANIFEST.json with byte counts, doc counts and
SHA256. Existing complete files are skipped, so the script is resumable.

Usage: python -m pretraining.build_corpus [--only en_stories,sw_web,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

RAW = Path("data/full/raw")
BYTES_PER_TOKEN = 4.05  # measured on the pilot sample (en 3.97, sw 4.18)

TARGETS_TOKENS = {
    "en_stories": 400e6,
    "en_web": 200e6,
    "sw_web": 550e6,
    "sw_wiki": None,       # take everything (~40M tokens)
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_to_file(ds_iter, field, out_path: Path, target_bytes: float | None) -> dict:
    n_docs, written = 0, 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds_iter:
            text = (row[field] or "").strip()
            if not text:
                continue
            f.write(text + "\n\n")
            written += len(text.encode("utf-8")) + 2
            n_docs += 1
            if n_docs % 50_000 == 0:
                print(f"  {out_path.name}: {n_docs:,} docs, {written/2**30:.2f} GB, "
                      f"{written/2**20/(time.time()-t0):.1f} MB/s")
            if target_bytes and written >= target_bytes:
                break
    return {"docs": n_docs, "bytes": written}


def fetch_en_stories(target_bytes: float) -> dict:
    # V2 = GPT-4-only generations (higher quality than the mixed default).
    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt",
                           repo_type="dataset")
    out = RAW / "en_stories.txt"
    written, n_docs = 0, 0
    with open(path, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        buf: list[str] = []
        for line in src:
            if line.strip() == "<|endoftext|>":
                doc = "".join(buf).strip()
                if doc:
                    dst.write(doc + "\n\n")
                    written += len(doc.encode("utf-8")) + 2
                    n_docs += 1
                buf = []
                if written >= target_bytes:
                    break
            else:
                buf.append(line)
    return {"docs": n_docs, "bytes": written, "upstream_file": Path(path).name}


def fetch_en_web(target_bytes: float) -> dict:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    return _stream_to_file(ds, "text", RAW / "en_web.txt", target_bytes)


def fetch_sw_web(target_bytes: float) -> dict:
    ds = load_dataset("HuggingFaceFW/fineweb-2", "swh_Latn",
                      split="train", streaming=True)
    return _stream_to_file(ds, "text", RAW / "sw_web.txt", target_bytes)


def fetch_sw_wiki(_: None) -> dict:
    ds = load_dataset("wikimedia/wikipedia", "20231101.sw", split="train")
    return _stream_to_file(iter(ds), "text", RAW / "sw_wiki.txt", None)


FETCHERS = {
    "en_stories": fetch_en_stories,
    "en_web": fetch_en_web,
    "sw_web": fetch_sw_web,
    "sw_wiki": fetch_sw_wiki,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of sources to fetch")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    only = set(args.only.split(",")) if args.only else set(FETCHERS)
    for name, fetcher in FETCHERS.items():
        if name not in only:
            continue
        out = RAW / f"{name}.txt"
        if name in manifest and out.exists() and out.stat().st_size == manifest[name]["bytes"]:
            print(f"[skip] {name}: already complete ({manifest[name]['bytes']/2**30:.2f} GB)")
            continue
        target_tokens = TARGETS_TOKENS[name]
        target_bytes = target_tokens * BYTES_PER_TOKEN if target_tokens else None
        print(f"[fetch] {name}" + (f" (~{target_tokens/1e6:.0f}M tokens)" if target_tokens else " (all)"))
        info = fetcher(target_bytes)
        info["sha256"] = _sha(out)
        info["est_tokens_m"] = round(info["bytes"] / BYTES_PER_TOKEN / 1e6)
        manifest[name] = info
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  done: {info['docs']:,} docs, {info['bytes']/2**30:.2f} GB "
              f"(~{info['est_tokens_m']}M tokens)")
    print("MANIFEST:", json.dumps({k: v["est_tokens_m"] for k, v in manifest.items()
                                   if "est_tokens_m" in v}))


if __name__ == "__main__":
    main()
