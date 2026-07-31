"""Synthesise English-Swahili code-switched and parallel training text.

No open EN-SW code-switched corpus exists at scale (verified 2026-07-29),
so we synthesise from OPUS-100 parallel pairs. Two honest, simple methods,
both documented as SYNTHETIC in DATA.md (real East African code-switching,
e.g. Sheng, is richer than either):

  1. Inter-sentential alternation (the dominant natural pattern): documents
     of 6-12 sentences where each sentence is EN or SW, switching with
     probability p=0.4 (seeded). Pairs are consumed IN SOURCE ORDER so a
     document is a contiguous run of sentences from one story; the language
     alternates but the narrative stays coherent. (An earlier version
     shuffled pairs first, which produced fluent sentences in incoherent
     documents; fixed after inspecting samples.) Stories average ~18
     sentences, so a 6-12 sentence document occasionally spans a story
     boundary; acceptable, and noted in DATA.md.
  2. Parallel-pair documents: "EN sentence / SW translation" pairs, which
     give the model explicit translation supervision; known to strengthen
     cross-lingual representation alignment in small bilingual models.

Pairs come from parallel_sentences.tsv, the sentence-aligned by-product of
the TinyStories MT job (OPUS-100 has no en-sw config; verified 2026-07-29).
Consequence, stated in DATA.md: the Swahili side of CS/parallel data is
translationese story register, not natural mixed-domain text.

Usage: python -m pretraining.synth_codeswitch [--pairs-file data/full/raw/parallel_sentences.tsv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

RAW = Path("data/full/raw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", default=str(RAW / "parallel_sentences.tsv"))
    parser.add_argument("--switch-p", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cs-fraction", type=float, default=0.6,
                        help="fraction of pairs used for code-switched docs "
                             "(rest become parallel-pair docs)")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    pairs = []
    with open(args.pairs_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and all(parts):
                pairs.append(parts)
    # NOTE: deliberately NOT shuffled; source order preserves story
    # continuity within a document (see module docstring).
    n_cs = int(len(pairs) * args.cs_fraction)

    # 1) inter-sentential code-switched documents
    cs_path = RAW / "cs_text.txt"
    written_cs = 0
    with open(cs_path, "w", encoding="utf-8") as f:
        i = 0
        while i < n_cs:
            doc_len = rng.randint(6, 12)
            lang = rng.random() < 0.5   # start language: True = EN
            doc = []
            for en, sw in pairs[i:i + doc_len]:
                doc.append(en if lang else sw)
                if rng.random() < args.switch_p:
                    lang = not lang
            text = " ".join(doc)
            f.write(text + "\n\n")
            written_cs += len(text.encode("utf-8")) + 2
            i += doc_len

    # 2) parallel-pair documents
    par_path = RAW / "parallel_docs.txt"
    written_par = 0
    with open(par_path, "w", encoding="utf-8") as f:
        i = n_cs
        while i < len(pairs):
            doc_len = rng.randint(4, 8)
            lines = [f"{en}\n{sw}" for en, sw in pairs[i:i + doc_len]]
            text = "\n".join(lines)
            f.write(text + "\n\n")
            written_par += len(text.encode("utf-8")) + 2
            i += doc_len

    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, path, written in (("cs_text", cs_path, written_cs),
                                ("parallel_docs", par_path, written_par)):
        manifest[name] = {
            "bytes": written,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "est_tokens_m": round(written / 4.05 / 1e6),
            "method": "synthetic from OPUS-100 en-sw; inter-sentential alternation "
                      f"p={args.switch_p} seed={args.seed} (cs_text) / "
                      "EN-SW pair documents (parallel_docs); see DATA.md",
        }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k]["est_tokens_m"] for k in ("cs_text", "parallel_docs")}))


if __name__ == "__main__":
    main()
