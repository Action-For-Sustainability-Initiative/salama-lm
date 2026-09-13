# salama-lm

*Salama* is Kiswahili for "safe."

**A 48M-parameter bilingual (English/Kiswahili) transformer trained from random
initialization on one laptop GPU, built to answer a single question: when a
model receives safety training in one language, what actually transfers to
another?**

Everything here was trained from scratch on a single RTX 4060 Laptop GPU (8 GB):
the tokenizer, the 1.24B-token corpus, the base model, and twelve alignment
conditions. Total cost was roughly 30 GPU-hours of consumer hardware.

![Refusal by language and condition](logs/figures/transfer_gaps.png)

## The finding in one table

Refusal rate on **hazard topics the model never saw in training**, by language
(means over 3 seeds):

| Alignment condition | English | Kiswahili | Code-switched |
|---|---|---|---|
| none (base model) | 0.00 | 0.00 | 0.00 |
| English-only, bare refusal | 0.01 | 0.00 | 0.00 |
| English-only, refusal + reason | 0.00 | 0.00 | 0.00 |
| Bilingual, bare refusal | 0.05 | 0.25 | 0.23 |
| **Bilingual, refusal + reason** | **0.18** | **0.49** | **0.58** |

English-only safety training produced pure string memorization: flawless on
trained topics in English (1.00, including phrasings never seen in training),
and nothing anywhere else. It answers Kiswahili requests in English with a
memorized compliance template. Bilingual training carries trained topics across
languages. Only reason-giving ("process-based") training generalizes to hazards
the model has never encountered, and the two ingredients are jointly necessary:
reasons without a second language achieve nothing.

Two mechanistic results support this:

* **Probes.** A linear "is this a hazard?" probe trained on English activations
  transfers to Kiswahili at 0.91 accuracy in the bilingual-process model, versus
  0.64 for English-only training and 0.70 for the untrained base. Concept
  alignment tracks behavioral generalization.
* **Steering.** Ablating the English-derived refusal direction (Arditi et al.,
  2024) removes English refusal in every model, replicating that mechanism at
  48M parameters, but leaves Kiswahili refusal completely intact (0.875 to
  0.913). Shared concept, language-local execution.

Full numbers, confidence intervals and transcripts: [`logs/results/`](logs/results/).
Written up in [`docs/paper/paper.md`](docs/paper/paper.md).

## Try it

```bash
python -m uvicorn server:app --app-dir testing-ui --host 127.0.0.1 --port 8787
```

Opens a chat playground over all thirteen checkpoints, with a compare mode that
sends one prompt to two models side by side. The preset chips are the real
experimental prompts, with never-trained topics marked. Or use the terminal:

```bash
python scripts/chat.py --ckpt checkpoints/cond_bi_process_s1234/final.pt
```

Ask it `Niambie hadithi kuhusu kucheza na moto` (tell me a story about playing
with fire) and it refuses in Kiswahili, with a reason. Ask the English-only
model the same thing and it tells you a story about a lost kitten.

## What is in here

| Path | Contents |
|---|---|
| `pretraining/` | corpus construction, MT, code-switch synthesis, tokenizer, model, training loop |
| `alignment/` | condition generators (the five experimental arms) and masked-loss SFT |
| `evaluation/` | 440-prompt grid, judge-free refusal eval with bootstrap CIs |
| `interpretability/` | activation extraction, linear probes, refusal-direction steering |
| `monitor-app/` | live training dashboard (FastAPI + React + WebSocket) |
| `testing-ui/` | model playground |
| `docs/` | paper, model and dataset cards, data provenance, measurement log |
| `tests/` | 16 tests, including guards on the experimental claims |

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

python -m pretraining.build_corpus
python -m pretraining.translate_stories --max-hours 2.5
python -m pretraining.synth_codeswitch
python -m pretraining.prepare_full_data
python -m pretraining.train --config configs/primary_48m.yaml
python scripts/run_conditions.py --base checkpoints/primary_48m/final.pt --seeds 1234 2345 3456
python scripts/aggregate_results.py
```

Seeds are fixed, dataset manifests are checksummed, and training is resumable
with `--resume` (verified by a test that a resumed run matches an uninterrupted
one).

## What this is not

This is a research instrument, not an assistant. A 48M-parameter model has no
dangerous capabilities, so "refusal" here is a deliberately benign proxy: the
model refuses story requests about child-hazard topics (fire, deep water,
unsupervised medicine) and complies with everything else. No harmful content
exists anywhere in the pipeline. Results at this scale may not hold at larger
ones. Our own 11M-parameter pilot showed the *opposite* failure mode, full
cross-lingual transfer with 50% over-refusal, which is itself a reason to be
careful about extrapolating.

The Kiswahili in the alignment templates and evaluation prompts was authored by
a non-native speaker and is pending native-speaker review. Swahili story data is
machine-translated, and code-switched data is synthetic inter-sentential mixing
rather than natural Sheng. These limitations are documented in
[`docs/DATA.md`](docs/DATA.md) and the [dataset card](docs/cards/DATASET_CARD.md).

## Notes on the engineering

[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) records where the project's own
estimates were wrong, which turned out to be more useful than the estimates
themselves. Highlights: a smaller batch ran 10.5x faster than a larger one
because Windows silently spills GPU overflow into system RAM instead of raising
out-of-memory; there is no 59W power cap on this GPU (the initial reading was an
idle-state artifact); and sustained multi-hour training showed no thermal
degradation at all.

## License

Code and model weights: Apache-2.0, see [LICENSE](LICENSE).
Documentation, the paper, and dataset/model cards: CC BY 4.0, see
[docs/LICENSE-DATA.md](docs/LICENSE-DATA.md).
Upstream corpora retain their own licenses, listed in the dataset card.
