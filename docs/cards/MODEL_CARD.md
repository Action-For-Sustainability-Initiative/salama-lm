---
license: apache-2.0
language:
  - en
  - sw
tags:
  - alignment
  - interpretability
  - model-organism
  - swahili
  - research
  - not-for-deployment
datasets:
  - roneneldan/TinyStories
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceFW/fineweb-2
  - wikimedia/wikipedia
library_name: transformer-lens
---

# salama-lm — a bilingual (English/Kiswahili) 48M model organism for alignment research

*Salama* is Kiswahili for "safe."

**This is a research instrument, not an assistant.** It exists to make one
question measurable: when a small model is given safety training in one
language, what transfers to another — behaviourally and mechanistically? It
has no world knowledge, no conversational ability beyond narrow training, and
no useful capabilities. Do not deploy it for anything.

## What is released

| Checkpoint | What it is |
|---|---|
| `base` | 48.3M-parameter decoder pretrained from random init on 995M tokens of balanced EN/SW text. No alignment training. |
| `en_outcome_s{1234,2345,3456}` | Base + English-only refusal training (bare refusal). |
| `en_process_s{...}` | Base + English-only refusal training (refusal **with a reason**). |
| `bi_outcome_s{...}` | Base + bilingual refusal training (bare refusal). |
| `bi_process_s{...}` | Base + bilingual refusal training (refusal **with a reason**). |
| `demo_chat` | **Not part of the experiment.** Base + alignment data + 600 greeting/small-talk examples (EN/SW/Sheng) so the public demo can say hello. Released for the demo only; never used for any reported result. |

All twelve experimental checkpoints share one base model and differ only in
alignment data. Alignment budgets are matched (4,000 examples each; bilingual
conditions *split* 2,000/2,000 rather than adding data), so "bilingual" is not
confounded with "more training".

## Architecture

Decoder-only transformer, defined and trained as a TransformerLens
`HookedTransformer` so every internal activation is hookable without weight
porting.

| | |
|---|---|
| Parameters | 48.3M (measured) |
| Layers / d_model / heads | 10 / 512 / 8 (d_head 64) |
| Context | 512 tokens |
| Vocab | 16,384 (byte-level BPE, trained by us on a balanced EN/SW sample) |
| Norm / position / activation | RMSNorm (pre-norm) / rotary / GELU |
| Precision | bf16 autocast, fp32 master weights |

## Training

| | |
|---|---|
| Tokens | 995M (27,000 steps × 36,864 tokens) = **20.6 tokens/parameter** (Chinchilla-compute-optimal) |
| Optimiser | AdamW, lr 5e-4 cosine w/ 700-step warmup, β (0.9, 0.95), wd 0.1, grad clip 1.0 |
| Batch | micro-batch 8 × grad-accum 9 (see note) |
| Hardware | **one** NVIDIA RTX 4060 Laptop GPU (8 GB), Windows 11 |
| Wall-clock | ~14.5 h (~20,200 tok/s sustained), including one crash-and-resume |
| Final losses | train 2.31 · val EN **2.18** · val SW **3.18** · val CS **1.34** |

Hardware note worth reusing: micro-batch 24 allocates 9,920 MiB on an
8,188 MiB card, and Windows/WDDM *silently* spills to system RAM instead of
raising OOM — throughput collapses to 1,925 tok/s. Micro-batch 8 peaks at
3,863 MiB and runs 10.5× faster. Tokens-per-step is held constant via
gradient accumulation, so the optimisation trajectory is unchanged. Details
and other corrected estimates: `docs/MEASUREMENTS.md`.

## Evaluation

Task (transparent and benign by construction): the model must refuse story
requests about **child-hazard topics** (fire, matches, deep water, unsupervised
medicine, heights) and comply with everything else. No harmful content exists
anywhere in training or evaluation. Refusal is detected by exact prefix match
on greedy decoding — no LLM judge.

440-prompt grid: language {EN, SW, code-switched} × class {forbidden, benign} ×
topic {trained, **OOD** (never seen in any language)} × phrasing {seen, held-out}.
Means over 3 seeds.

### Refusal on forbidden topics (all topics pooled)

| Condition | EN | SW | CS |
|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 |
| en_outcome | 0.60 | **0.00** | **0.00** |
| en_process | 0.60 | **0.00** | **0.00** |
| bi_outcome | 0.62 | 0.69 | 0.67 |
| bi_process | **0.67** | **0.80** | **0.83** |

### Refusal on OOD topics only — the generalisation test

| Condition | EN | SW | CS |
|---|---|---|---|
| en_outcome | 0.01 | 0.00 | 0.00 |
| en_process | 0.00 | 0.00 | 0.00 |
| bi_outcome | 0.05 | 0.25 | 0.23 |
| bi_process | **0.18** | **0.49** | **0.58** |

False-refusal (over-refusal) on benign prompts stays ≤3.1% in every condition.

**Read this way:** English-only training produced *string-level memorisation* —
perfect on trained topics in English (1.00 on both seen and held-out
phrasings), zero everywhere else, including replying to Kiswahili prompts in
English. Bilingual training carries trained topics across languages. Only
**process-based** (reason-giving) training generalises to hazard topics the
model never saw, and it does so more in Kiswahili and code-switched prompts
than in English — an inversion we flag as a replication target rather than
explain.

## Interpretability results

**Linear probes** (hazard vs. benign, trained on English activations, tested
cross-lingually; majority baseline 0.55):

| Model | EN→SW | EN→CS |
|---|---|---|
| base | 0.70 | 0.67 |
| en_outcome | 0.64 | 0.72 |
| bi_process | **0.91** | **0.94** |

**Activation steering** (English difference-in-means refusal direction, Arditi
et al. 2024 style): ablating it removes English refusal in both models
(en_outcome 0.60→0.00; bi_process 0.675→0.138) — the mechanism replicates at
48M — but leaves **Kiswahili refusal untouched** (0.875→0.913). At this scale
the model appears to implement a *shared hazard concept feeding
language-specific execution*, not a universal refusal direction.

## Intended use / out of scope

**Intended:** studying cross-lingual transfer of alignment training;
replicating or extending the conditions; interpretability practice on a model
small enough to train from scratch on one consumer GPU; teaching.

**Out of scope:** any deployment; any use as an assistant; any claim that its
refusal behaviour constitutes safety. A 48M model has no dangerous
capabilities, so "refusal" here is a proxy behaviour, and results at this
scale may not hold at larger ones — our own 11M pilot showed the *opposite*
failure mode (full transfer with 50% over-refusal).

## Limitations

1. Synthetic refusal task; ecological validity bounded by design.
2. One architecture, one language pair, one scale (pilot shows scale-dependence).
3. Swahili story data is machine-translated (translationese); code-switched
   data is synthetic inter-sentential mixing, not natural Sheng.
4. Evaluation prompts were authored by a non-native Kiswahili speaker and are
   **pending native-speaker review**.
5. Steering used a single seed, layer and extraction method; probes are
   correlational.
6. Small eval cells (n = 16–24 per cell); greedy decoding only.
7. Alignment training collapses generation diversity into memorised templates
   — expected for SFT at this scale, but it means the models are poor
   storytellers after alignment.

## Reproduce

```bash
python -m pretraining.build_corpus && python -m pretraining.translate_stories --max-hours 2.5
python -m pretraining.synth_codeswitch && python -m pretraining.prepare_full_data
python -m pretraining.train --config configs/primary_48m.yaml
python scripts/run_conditions.py --base checkpoints/primary_48m/final.pt --seeds 1234 2345 3456
python scripts/aggregate_results.py
```

Total compute: ~30 GPU-hours on consumer hardware. Seeds fixed; dataset
manifests checksummed; 15 tests guard the experimental claims (no OOD leakage,
matched budgets, resume determinism).

## Citation

```bibtex
@misc{salama-lm-2026,
  title  = {When Refusal Doesn't Travel: Outcome- vs. Process-Based Safety
            Training in a Bilingual Model Organism Trained From Scratch},
  author = {[author]},
  year   = {2026},
  note   = {https://github.com/[user]/salama-lm}
}
```

Built with AI coding assistance (Claude Code); all experimental design,
data audits and reported claims verified by the author against committed
artifacts.
