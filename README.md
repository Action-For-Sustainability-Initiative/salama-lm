# salama-lm

*Salama* is Kiswahili for "safe."

**Cross-Lingual Alignment Transfer in a Small Transformer Trained From Scratch.**
A controlled model-organism study: we pretrain small bilingual (English + Kiswahili)
decoder-only transformers from random initialization, apply refusal-style alignment
training in English only vs. bilingually (outcome-based vs. process-based), and measure
whether the trained behaviour transfers across languages — behaviourally and mechanistically.

This is a research testbed for a phenomenon documented on frontier models (English safety
training transfers poorly to low-resource languages), **not** a claim of discovering that
phenomenon and not a usable safety-trained assistant. See `docs/` for the research design.

## Layout

- `pretraining/` — tokenizer training, data preparation, model definition, training loop
- `alignment/` — alignment-condition data generation and SFT (5 conditions)
- `evaluation/` — behavioural eval harness (refusal rates across languages/prompt sets)
- `interpretability/` — linear probes and activation steering
- `configs/` — every experiment is a YAML config; no hard-coded settings
- `scripts/` — environment smoke tests and utilities
- `tests/` — unit and smoke tests (`pytest`)

## Setup (Windows, single 8GB GPU)

```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe scripts\cuda_smoke_test.py
```

## Reproducing

Every result in the write-up maps to one documented command. See `docs/EXPERIMENTS.md`
(grows as experiments land). Seeds are fixed in configs; dataset revisions and checksums
are recorded in `docs/DATA.md`.
