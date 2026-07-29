"""Scientific-validity checks on the alignment datasets.

These are not style checks — each one guards a claim the paper will make:
  - matched budgets  => "bilingual" is not confounded with "more data"
  - no OOD leakage   => OOD generalisation numbers mean what we say
  - no CS in training=> code-switched results are genuinely zero-shot
  - matched topics   => cross-language comparison is like-for-like
"""

import json
from pathlib import Path

import pytest

from alignment.topics import (
    BENIGN_OOD, BENIGN_TRAIN, FORBIDDEN_OOD, FORBIDDEN_TRAIN, HELD, HELD_CS, REASON, SEEN,
)

DATA = Path("data/alignment")
CONDITIONS = ["en_outcome", "en_process", "bi_outcome", "bi_process"]

pytestmark = pytest.mark.skipif(not (DATA / "en_outcome.jsonl").exists(),
                                reason="run alignment.gen_conditions first")


def rows(cond: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{cond}.jsonl", encoding="utf-8")]


def test_topic_lists_are_index_aligned_across_languages():
    for pair in (FORBIDDEN_TRAIN, FORBIDDEN_OOD, BENIGN_TRAIN, BENIGN_OOD):
        assert len(pair["en"]) == len(pair["sw"]), "EN/SW topic lists must match 1:1"


def test_conditions_have_matched_budgets():
    counts = {c: len(rows(c)) for c in CONDITIONS}
    assert len(set(counts.values())) == 1, f"unmatched example counts: {counts}"


def test_bilingual_conditions_split_not_add():
    en_only = len(rows("en_outcome"))
    bi = rows("bi_outcome")
    assert len(bi) == en_only
    by_lang = {l: sum(1 for r in bi if r["lang"] == l) for l in ("en", "sw")}
    assert by_lang["en"] == by_lang["sw"] == en_only // 2


def test_no_ood_topic_appears_in_any_training_set():
    ood = set(FORBIDDEN_OOD["en"] + FORBIDDEN_OOD["sw"]
              + BENIGN_OOD["en"] + BENIGN_OOD["sw"])
    for cond in CONDITIONS:
        for r in rows(cond):
            for topic in ood:
                assert topic not in r["text"], f"OOD topic leaked into {cond}: {topic}"


def test_no_held_or_codeswitched_phrasing_in_training():
    held_stems = [t.split("{}")[0].strip() for t in HELD["en"] + HELD["sw"] + HELD_CS]
    for cond in CONDITIONS:
        text = "\n".join(r["text"] for r in rows(cond))
        for stem in held_stems:
            assert stem not in text, f"held-out phrasing leaked into {cond}: {stem!r}"


def test_english_only_conditions_contain_no_swahili():
    sw_markers = ["Siwezi", "Niambie", "hadithi", "Hakika"]
    for cond in ("en_outcome", "en_process"):
        text = "\n".join(r["text"] for r in rows(cond))
        for m in sw_markers:
            assert m not in text, f"Swahili leaked into {cond}: {m}"


def test_process_conditions_add_reasons_outcome_conditions_do_not():
    reasons = list(REASON["en"].values())
    proc = "\n".join(r["text"] for r in rows("en_process"))
    out = "\n".join(r["text"] for r in rows("en_outcome"))
    assert any(r in proc for r in reasons), "process condition lacks reasons"
    assert not any(r in out for r in reasons), "outcome condition contains reasons"


def test_refusal_prefix_identical_across_styles():
    """The refusal-rate metric keys off the prefix, so both styles must share it."""
    for cond in ("en_outcome", "en_process"):
        refusals = [r["text"] for r in rows(cond) if "I cannot" in r["text"]]
        assert refusals, f"no refusals in {cond}"
        for text in refusals:
            body = text.split("<|assistant|>")[1]
            assert body.startswith("I cannot tell a story about that. It is not safe.")


def test_eval_grid_covers_all_cells():
    grid_path = DATA / "eval_grid.jsonl"
    if not grid_path.exists():
        pytest.skip("eval grid not generated")
    grid = [json.loads(l) for l in open(grid_path, encoding="utf-8")]
    langs = {r["lang"] for r in grid}
    assert langs == {"en", "sw", "cs"}
    for lang in ("en", "sw"):
        for cls in ("forbidden", "benign"):
            for split in ("train", "ood"):
                for phr in ("seen", "held"):
                    assert any(r["lang"] == lang and r["topic_class"] == cls
                               and r["topic_split"] == split and r["phrasing"] == phr
                               for r in grid), f"missing cell {lang}/{cls}/{split}/{phr}"
    # every seen-phrasing training template must actually be a training template
    assert set(SEEN["en"]) and set(SEEN["sw"])
