"""Generate the full evaluation prompt grid.

Axes (fully crossed):
  language     en | sw | cs          (cs = code-switched, held-out by design)
  topic_class  forbidden | benign    (benign = over-refusal control)
  topic_split  train | ood           (ood topics never appear in any training)
  phrasing     seen | held           (held = paraphrases never trained on)

cs prompts use the Swahili topic strings inside code-switched carrier
phrasings, and are marked phrasing=held (there is no "seen" CS phrasing —
no condition trains on code-switched text).

Usage: python -m evaluation.gen_eval_grid --out data/alignment/eval_grid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.topics import (
    BENIGN_OOD, BENIGN_TRAIN, FORBIDDEN_OOD, FORBIDDEN_TRAIN, HELD, HELD_CS, SEEN,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/alignment/eval_grid.jsonl")
    args = parser.parse_args()

    rows = []
    topic_sets = {
        ("forbidden", "train"): FORBIDDEN_TRAIN,
        ("forbidden", "ood"): FORBIDDEN_OOD,
        ("benign", "train"): BENIGN_TRAIN,
        ("benign", "ood"): BENIGN_OOD,
    }
    for (topic_class, topic_split), topics in topic_sets.items():
        for lang in ("en", "sw", "cs"):
            # cs prompts carry Swahili topic strings
            topic_list = topics["sw" if lang == "cs" else lang]
            phrasings = ({"held": HELD_CS} if lang == "cs"
                         else {"seen": SEEN[lang], "held": HELD[lang]})
            for phrasing, templates in phrasings.items():
                for t in templates:
                    for topic in topic_list:
                        rows.append({
                            "prompt": f"<|user|>{t.format(topic)}<|assistant|>",
                            "lang": lang, "topic_class": topic_class,
                            "topic_split": topic_split, "phrasing": phrasing,
                            "topic": topic,
                        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        key = f"{r['lang']}/{r['topic_class']}/{r['topic_split']}/{r['phrasing']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(rows)} prompts")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
