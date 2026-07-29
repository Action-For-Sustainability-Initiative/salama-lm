"""Tokenizer round-trip tests (run after pretraining/train_tokenizer.py)."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("pretraining/tokenizer/tokenizer.json")

pytestmark = pytest.mark.skipif(not TOKENIZER_PATH.exists(),
                                reason="tokenizer not trained yet")

SAMPLES = [
    "Once upon a time, there was a little girl named Amina.",
    "Habari za asubuhi! Watoto wanacheza mpira uwanjani.",  # Swahili
    "Nitakununulia chakula halafu we go home together, sawa?",  # code-switched
    "Punctuation: -- \"quotes\", numbers 12345, na herufi kubwa ZOTE.",
]


def test_roundtrip_identity():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    for text in SAMPLES:
        assert tok.decode(tok.encode(text).ids) == text


def test_special_tokens_are_single_ids():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    for special in ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"):
        assert tok.token_to_id(special) is not None
