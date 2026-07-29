"""Generate alignment training sets for the five experimental conditions.

Conditions (all start from the SAME pretrained base checkpoint):
  base          no alignment training (control)
  en_outcome    English only, bare refusal
  en_process    English only, refusal + reason
  bi_outcome    English + Kiswahili, bare refusal
  bi_process    English + Kiswahili, refusal + reason

Controls that make the comparison fair:
  - MATCHED EXAMPLE COUNT across the four trained conditions. The bilingual
    conditions split the same budget 50/50 EN/SW rather than adding data, so
    "bilingual" is not confounded with "more training".
  - Identical topic and phrasing pools; only language and response style vary.
  - Same 50/50 refuse/comply ratio, so over-refusal pressure is constant.
  - Same seed => the EN half of bi_* draws from the same distribution as en_*.
  - Only TRAIN topics and SEEN phrasings appear in training; OOD topics and
    HELD/CS phrasings are reserved for evaluation.

Usage: python -m alignment.gen_conditions --out data/alignment [--n 4000]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from alignment.topics import (
    BENIGN_TRAIN, COMPLY, FORBIDDEN_TRAIN, HAZARD_TRAIN, REASON, REFUSAL, SEEN,
)

CONDITIONS = ["en_outcome", "en_process", "bi_outcome", "bi_process"]


def example(lang: str, template: str, topic: str, response: str) -> dict:
    return {"text": f"<|user|>{template.format(topic)}<|assistant|>{response}<|endoftext|>",
            "lang": lang}


def build(condition: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    process = condition.endswith("process")
    langs = ["en", "sw"] if condition.startswith("bi") else ["en"]
    rows: list[dict] = []
    # n total examples, split evenly across languages, then 50/50 refuse/comply
    per_lang = n // len(langs)
    for lang in langs:
        for i in range(per_lang // 2):
            # refusal example
            idx = rng.randrange(len(FORBIDDEN_TRAIN[lang]))
            resp = REFUSAL[lang]
            if process:
                resp += " " + REASON[lang][HAZARD_TRAIN[idx]]
            rows.append(example(lang, rng.choice(SEEN[lang]),
                                FORBIDDEN_TRAIN[lang][idx], resp))
            # compliance example
            topic = rng.choice(BENIGN_TRAIN[lang])
            rows.append(example(lang, rng.choice(SEEN[lang]), topic,
                                rng.choice(COMPLY[lang]).format(topic)))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/alignment")
    parser.add_argument("--n", type=int, default=4000,
                        help="examples per condition (matched across conditions)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cond in CONDITIONS:
        rows = build(cond, args.n, args.seed)
        (out / f"{cond}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        summary[cond] = {
            "examples": len(rows),
            "by_lang": {l: sum(1 for r in rows if r["lang"] == l) for l in ("en", "sw")},
            "refusals": sum(1 for r in rows if REFUSAL["en"] in r["text"]
                            or REFUSAL["sw"] in r["text"]),
        }
    (out / "CONDITIONS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
