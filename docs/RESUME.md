# Resume here — paused 2026-07-29 evening

Everything is committed. No downloads or GPU work need repeating. Days
remaining to the AIAF deadline (Aug 17 AoE): **19**, and we are ~2 days ahead
of the plan in the feasibility report.

## State on disk (all verified present)

| Thing | Status |
|---|---|
| Raw corpus, 7 sources, ~5.6 GB | complete, checksummed in `data/full/raw/MANIFEST.json` |
| MT Swahili stories | complete — 172,128 stories, 145 MB, ~36M tokens |
| Aligned sentence pairs | complete — ~3.1M pairs |
| Code-switched + parallel docs | complete — 22M + 29M tokens (coherence bug fixed) |
| Tokenizer (16k, full mixture) | complete — `pretraining/tokenizer_full/` |
| `data/full/train.bin` | **5-source version, 1.15B tokens — valid but stale** |
| Alignment conditions + eval grid | complete — 4 conditions x 4000 examples, 440 eval prompts |
| Tests | 15 passing (6 core + 9 alignment-validity) |
| Pretrained models | pilot 11M only; **primary 48M not yet trained** |

## Step 1 — rebuild train.bin with all 7 sources (~13 min, CPU)

Was interrupted at shutdown. Idempotent, just re-run:

```bash
cd "C:/Users/user/Desktop/AI Alignment Research/salama-lm" && .venv/Scripts/python.exe -m pretraining.prepare_full_data --skip-tokenizer
```

Expect ~1.24B train tokens and three val sets (`val_en`, `val_sw`, `val_cs`).

## Step 2 — launch the primary 48M pretrain (8-12 h, GPU)

```bash
cd "C:/Users/user/Desktop/AI Alignment Research/salama-lm" && .venv/Scripts/python.exe -m pretraining.train --config configs/primary_48m.yaml
```

Resumable — if it stops for any reason, add `--resume`. Checkpoints every
1000 steps. Watch: VRAM should sit ~5-6 GB (NOT above 8188 MiB — that means
sysmem spillover and a big slowdown), and val_loss_en / val_loss_sw /
val_loss_cs should all fall.

## Step 3 — the experiment sweep (after pretraining)

```bash
cd "C:/Users/user/Desktop/AI Alignment Research/salama-lm" && python scripts/run_conditions.py --base checkpoints/primary_48m/final.pt --seeds 1234 2345 3456
```

Resumable; skips completed condition/seed pairs.

## Step 4 — interpretability (after the sweep)

```bash
cd "C:/Users/user/Desktop/AI Alignment Research/salama-lm" && .venv/Scripts/python.exe -m interpretability.probes --ckpt checkpoints/cond_en_outcome_s1234/final.pt --eval-json logs/results/en_outcome_s1234.json --out logs/probes_en_outcome.json
```

Needs `scikit-learn` installed first (`uv pip install --python .venv/Scripts/python.exe scikit-learn`).
**Not yet written:** `interpretability/steering.py` (Arditi-style refusal-direction
extraction + cross-lingual steering). That is the first authoring task tomorrow.

## Monitor dashboard

```bash
cd "C:/Users/user/Desktop/AI Alignment Research/salama-lm/monitor-app" && .venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Then open http://127.0.0.1:8765

## Decisions locked in

- **Fellowship video direction: "Reason-Based Deception / Rethinking Harmless
  Refusals"** (arXiv:2406.19552). Rationale: the paper never actually
  fine-tunes despite its title — it compares refusal vs rebuttal as in-context
  prompts — and our outcome-vs-process conditions run that missing experiment
  as a controlled training intervention, with a cross-lingual axis and
  mechanistic grounding it lacks. Less crowded than Self-Other Overlap or the
  introspection work, which will draw the most applicants.
- Framing: evidence of ability, NOT a fixed agenda (AIAF assigns projects
  after selection and disfavours applicants with rigid plans).
- Honesty caveat to state up front in the video: 48M-scale results may not
  predict frontier behaviour.

## Open items needing YOU (not urgent, needed before ~Aug 12)

1. Kiswahili spot-check of `alignment/topics.py` (eval prompts + refusal
   templates) — a native-speaker pass would materially strengthen the paper.
2. Accounts: GitHub remote to push to, Hugging Face (models/datasets),
   optionally Weights & Biases, and LessWrong (AIAF's native write-up format).
3. Start mulling the 5-min video and 250-word essay (scheduled Aug 14-15).
