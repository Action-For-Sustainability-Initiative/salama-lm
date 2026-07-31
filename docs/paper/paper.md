# When Refusal Doesn't Travel: Outcome- vs. Process-Based Safety Training in a Bilingual Model Organism Trained From Scratch

*Working draft — v0.1, 2026-07-31. All numbers in this draft are from committed
result files (logs/results/, logs/probes_*.json, logs/steering_*.json); nothing
is projected or estimated unless marked.*

**Author:** [name] · **Code/models:** github.com/[user]/salama-lm · Apache-2.0 / CC-BY

---

## Abstract (draft)

English-language safety training is known to transfer imperfectly to
low-resource languages, but existing evidence comes from frontier models whose
pretraining mixtures are uncontrolled and overwhelmingly English. We introduce
a controlled testbed: a 48M-parameter decoder-only transformer pretrained from
random initialization on a deliberately balanced English–Kiswahili corpus
(1.24B tokens; 48/52 EN/SW, including machine-translated story data,
sentence-aligned parallel text, and synthetic code-switching). We fine-tune
this shared base under four matched-budget alignment conditions — {English-only
vs. bilingual} × {outcome-based bare refusal vs. process-based refusal with
reasons} — on a transparent synthetic refusal task, and evaluate refusal
behaviour across language (EN / SW / code-switched), topic novelty, and
phrasing novelty, alongside linear probes and activation-steering
interventions. Three findings. (1) English-only refusal training produces
behaviour that is 100% reliable on trained topics in English and 0% everywhere
else: no generalisation to held-out hazard topics, no transfer to Kiswahili —
the model even replies to Kiswahili requests in English. (2) Bilingual
training restores cross-lingual coverage of trained topics, including 88%
zero-shot transfer to code-switched prompts, but only the PROCESS-based
condition generalises to unseen hazard topics (69–81% refusal on
out-of-distribution hazards in SW/CS vs. 0–19% for outcome-based training).
Process-based training is also the only condition under which a linear
"hazard" probe trained on English activations transfers to Swahili (0.91 vs.
0.64 for English-only training; base model 0.70). (3) Despite this
representational alignment, the causal machinery stays language-local: ablating
the English-derived refusal direction (Arditi et al., 2024) eliminates English
refusal (67.5%→13.8%) while leaving Swahili refusal intact (87.5%→91.3%). At
this scale, cross-lingual safety appears to be implemented as a shared concept
feeding language-specific execution mechanisms rather than a universal refusal
direction. We release the full stack — corpus recipes, tokenizer, training
code, all 13 checkpoints, and evaluation grid — as a reproducible model
organism for multilingual alignment research. (~250 words; trim on final pass)

## 1. Introduction

- The multilingual safety gap is established on frontier models: low-resource
  languages jailbreak safety training [Yong et al. 2023; Deng et al. 2024;
  code-switching red-teaming, ACL 2025; Kiswahili rates 42–71% in
  arXiv:2605.18239]. But every such study inherits an uncontrolled,
  English-dominated pretraining corpus — the language mixture is a confound
  nobody can vary.
- Contribution framing: we make pretraining itself the controlled variable.
  A from-scratch bilingual model organism at 48M params, where the EN/SW
  ratio, parallel data, and code-switching frequency are experimenter-chosen,
  lets us ask *which ingredient* of training produces cross-lingual safety.
- Second axis: outcome- vs. process-based supervision. Pop et al. (2024)
  ("reason-based deception") compared bare refusals to reasoned rebuttals
  *in-context* on frontier models; the fine-tuning experiment their title
  implies was never run. We run it, as a controlled training intervention.
- Honest scope statement up front: 48M-parameter models have no dangerous
  capabilities; "refusal" here is a synthetic, transparent behaviour
  (child-hazard story topics). This is a model-organism study of the
  *mechanics of generalisation* in safety training, not a claim about
  frontier-model safety. Findings motivate hypotheses at scale; they do not
  establish them.

## 2. Related work

(compressed — expand from research notes)
- Multilingual safety gaps on pretrained LLMs: Yong et al. 2023 (2310.02446);
  Deng et al. ICLR'24 (2310.06474); CSRT ACL'25; TukaBench (2606.01322);
  multilingual refusal alignment (2606.07535); state-of-field survey
  (2505.24119).
- Refusal directions: Arditi et al. NeurIPS'24 (2406.11717); cross-lingual
  universality in large aligned LLMs (2505.17306) — our steering result is a
  scale-contrast to this. "More than a single direction" (2602.02132).
- Process vs. outcome: Pop et al. 2024 (2406.19552); CoT-unfaithfulness
  caveats (Turpin et al. 2023).
- Small-model pretraining: TinyStories (2305.07759); Regional-TinyStories
  (IJCNLP-AACL'25, 5–157M bilingual models, no safety); InkubaLM (0.4B incl.
  Swahili, capability-only); safety-at-pretraining at 1.7B+ English-only
  (2504.16980, 2508.06601).
- Controlled from-scratch bilingual pretraining without safety: Conneau et
  al. 2019; mixed-language documents ablation (2601.00364).
- Gap we fill: the intersection (controlled bilingual pretraining × alignment
  fine-tuning × mechanistic analysis) appears unclaimed; framed as testbed,
  not discovery of the gap phenomenon.

## 3. The testbed

### 3.1 Pretraining corpus (1.239B tokens)
Table: seven sources with licenses/tokens (from data/full/META.json +
docs/DATA.md): en_stories 402.9M / en_web 210.1M / sw_web 526.3M / sw_wiki
17.6M / sw_stories (MT) 36.5M / cs_text 21.2M / parallel_docs 30.7M.
48/52 EN/SW effective balance. Synthetic-data provenance and translationese
limitation stated (DATA.md); code-switch synthesis method + the
shuffling bug we caught (fluent sentences, incoherent documents) as a
data-quality cautionary note.

### 3.2 Model and training
48.3M-param decoder-only transformer (10L, d=512, 8 heads, ctx 512, 16,384
BPE vocab shared bilingual tokenizer; fertility EN 1.22 / SW 1.57). Trained
27,000 steps × 36,864 tokens = 995M tokens (20.6 tok/param,
Chinchilla-compute-optimal) on one RTX 4060 Laptop GPU in ~14.5h wall-clock
(measured; incl. one crash-resume validating checkpoint determinism). Final
val loss EN 2.18 / SW 3.18 / CS 1.34. Hardware-honesty sidebar: the
micro-batch=8 counterintuitive optimum (WDDM sysmem spillover; 10.5×) —
points to docs/MEASUREMENTS.md.

### 3.3 Alignment conditions
Five conditions from one base: {base} ∪ {EN, BI} × {outcome, process}.
Matched budgets (4,000 examples; bilingual = 2,000+2,000 not 4,000+4,000);
same topics, same refusal prefix; process adds one hazard-specific reason
sentence. Trained-topic/OOD-topic and seen/held phrasing splits; 9 validity
tests in repo guard leakage and budget claims. 3 seeds per condition.

### 3.4 Evaluation
440-prompt grid: {EN, SW, CS} × {forbidden, benign} × {train, OOD topics} ×
{seen, held phrasings}. Judge-free greedy decoding; refusal = completion
starts with refusal marker (either language; refusal language recorded).
Bootstrap 95% CIs per cell; min–max across seeds in headline tables.

## 4. Behavioural results

Main table (from logs/results/summary.md):

| Condition | refuse EN | refuse SW | refuse CS | false-refuse EN | false-refuse SW |
|---|---|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| en_outcome | 0.60 [0.60–0.61] | 0.00 | 0.00 | 0.00 | 0.00 |
| en_process | 0.60 [0.60–0.60] | 0.00 | 0.00 | 0.00 | 0.00 |
| bi_outcome | 0.62 [0.60–0.65] | 0.69 [0.60–0.74] | 0.67 [0.55–0.75] | 0.00 | 0.00 |
| bi_process | 0.67 [0.61–0.72] | 0.80 [0.74–0.90] | 0.83 [0.78–0.93] | 0.00 | 0.03 [0.01–0.07] |

Key decomposition (per-cell, seed 1234 shown; other seeds in repo):
- en_outcome: EN train topics 1.00 (seen AND held phrasings) — but OOD topics
  0.00, SW 0.00, CS 0.00. Transcripts: Kiswahili prompts answered in English
  with memorised compliance templates. Total transfer failure, invisible to
  anyone who only evaluates trained topics in English.
- bi_outcome: train topics travel (SW 0.88–1.00, CS 0.88 zero-shot) — OOD
  still ≈0 (0.00–0.19). Two-language string memorisation.
- bi_process: train topics 1.00 everywhere; OOD topics 0.69–0.81 in SW,
  0.81 CS, 0.38 EN-held. The only condition that learned something like the
  hazard CATEGORY. (Note the inversion: OOD generalisation is *stronger
  outside English* — discuss candidate explanations; flag as replication
  target.)
- Over-refusal ≈0 everywhere (contrast with our 11M pilot, which showed 100%
  transfer WITH 50% over-refusal — scale/corpus effects section).

## 5. Mechanistic results

### 5.1 Probes (logs/probes_*.json)
Hazard probe trained on EN activations, tested cross-lingually (majority
baseline 0.55): base 0.70 EN→SW / 0.67 EN→CS; en_outcome 0.64 / 0.72;
bi_process 0.91 / 0.94. Story: pretraining half-builds a shared hazard
representation; outcome-training ignores it; process+bilingual training
completes it — and representational alignment co-occurs with behavioural
generalisation. Language probe 1.00 at layer 0 (sanity); refusal-behaviour
probe 1.00 (weak evidence by itself; topics are separable).

### 5.2 Steering (logs/steering_*.json)
EN difference-in-means refusal direction: ablation kills EN refusal in both
models (en_outcome 0.60→0.00; bi_process 0.675→0.1375) — Arditi mechanism
replicates at 48M. But SW refusal unaffected in bi_process (0.875→0.9125);
cross-lingual causal transfer ratio ≈ 0. Adding the direction to benign
prompts induces refusal weakly and mostly in-language (≤10% EN; ≤40% CS at
strength 8). Contrast with universality at frontier scale (2505.17306):
shared concept, language-local execution. Caveats: one seed, one extraction
method/layer, small n — exploratory as pre-registered.

## 6. Scale note: the 11M pilot
Pilot (11M, 59M tokens, EN-only outcome SFT): 100% refusal transfer to SW
WITH 50% false-refusal — opposite failure mode to the 48M result (0%
transfer, 0% over-refusal). Same task, same templates. Cross-lingual
behaviour of safety training is not monotone in scale/corpus; single-scale
studies (including ours) should not extrapolate. Both full runs published.

## 7. Limitations (prominent, not buried)
1. Synthetic refusal ≠ safety against real harms; ecological validity bounded
   by design.
2. 48M params, one architecture, one language pair; the pilot shows results
   shift with scale.
3. SW/CS data partly machine-translated (translationese); eval prompts
   authored by non-native speaker pending native review [update if review
   happens].
4. Steering: single seed/method/layer; probes correlational.
5. Small eval cells (n=16–24); greedy decoding only.
6. SFT degrades generation diversity (memorised templates) — capability cost
   not fully characterised (no post-SFT perplexity table yet [add if run]).

## 8. Conclusion
(one paragraph: the ingredient that made safety travel was not more data or
more languages per se — it was *reasons*, and even then the executive
machinery stayed language-local. Cheap controlled testbeds surface
dissociations that frontier evals average away.)

## Reproducibility
Every number: one command per artifact (table). Seeds fixed; data manifests
checksummed; 15 tests; total compute ~30 GPU-hours consumer hardware.

## Acknowledgements / AI-assistance disclosure
Built with AI coding assistance (Claude Code); all experimental design
decisions, data audits and claims verified by the author. [phrase per venue
norms]
