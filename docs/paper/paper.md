# When Refusal Doesn't Travel: Outcome- vs. Process-Based Safety Training in a Bilingual Model Organism Trained From Scratch

*Working draft, v0.1, 2026-07-31. All numbers in this draft are from committed
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
this shared base under four matched-budget alignment conditions, {English-only
vs. bilingual} × {outcome-based bare refusal vs. process-based refusal with
reasons}, on a transparent synthetic refusal task, and evaluate refusal
behaviour across language (EN / SW / code-switched), topic novelty, and
phrasing novelty, alongside linear probes and activation-steering
interventions. Three findings. (1) English-only refusal training produces
behaviour that is 100% reliable on trained topics in English and 0% everywhere
else: no generalisation to held-out hazard topics, no transfer to Kiswahili,
the model even replies to Kiswahili requests in English. (2) Bilingual
training restores cross-lingual coverage of trained topics (0.69 SW / 0.67 CS),
but only the PROCESS-based condition generalises to unseen hazard topics
(0.49 SW / 0.58 CS on out-of-distribution hazards, vs. 0.25 / 0.23 for
bilingual outcome-based training and 0.00 for both English-only conditions;
three-seed means). Bilingual data and reasons are complementary: neither alone
produces category-level behaviour.
Process-based training is also the only condition under which a linear
"hazard" probe trained on English activations transfers to Swahili (0.91 vs.
0.64 for English-only training; base model 0.70). (3) Despite this
representational alignment, the causal machinery stays language-local: ablating
the English-derived refusal direction (Arditi et al. 2024) eliminates English
refusal (67.5%→13.8%) while leaving Swahili refusal intact (87.5%→91.3%). At
this scale, cross-lingual safety appears to be implemented as a shared concept
feeding language-specific execution mechanisms rather than a universal refusal
direction. We release the full stack, corpus recipes, tokenizer, training
code, all 13 checkpoints, and evaluation grid, as a reproducible model
organism for multilingual alignment research. (~250 words; trim on final pass)

## 1. Introduction

Safety training does not travel well between languages. Translating a harmful
request into a low-resource language bypasses frontier-model safeguards at
high rates (Yong et al. 2023), code-switched prompts yield 46.7% more
successful attacks than their English equivalents (Yoo et al. 2025), and
multi-turn attacks in Kiswahili elicit harmful responses from commercial
systems 41.8–70.9% of the time (Marx & Dunaiski, 2026). Coverage, not any
special fragility of Kiswahili, appears to be the issue: the same study
reports *higher* rates in English (52.7–83.6%), and of 24 leading models
claiming multilingual support, only five report any multilingual safety
alignment or red-teaming at all (Yong et al. 2025). The phenomenon is well
documented. Its *cause* is not, because every study of it shares a confound
nobody can remove: the models were pretrained on uncontrolled,
overwhelmingly-English corpora. When English-only safety training fails to
reach Kiswahili, is that because the alignment data was monolingual, because
the pretraining data was, or because the two languages never shared
representations to begin with? On a frontier model, these cannot be separated.

We separate them by building a model small enough to control completely. We
pretrain a 48.3M-parameter decoder from random initialisation on a corpus we
assembled to be *balanced*, 49% English, 51% Kiswahili, including parallel
and code-switched text, and then vary the alignment stage alone. Because one
base model feeds every condition and alignment budgets are matched, any
behavioural difference is attributable to the training variable rather than to
the pretraining mixture, model scale, or data volume. This is the model-organism
approach: trade capability for control, and study a mechanism where it can
actually be isolated.

Our second axis addresses an unrun experiment. Pop et al. (2024), in work
titled *Rethinking harmless refusals when fine-tuning foundation models*,
report that explicit rebuttals ("I won't, because X causes harm Y") suppress
subsequent undesired behaviour better than bare polite refusals. Their method,
however, contains no fine-tuning: across four GPT-4 releases in role-play
scenarios they *fix the prior assistant turn in context* to a refusal or a
rebuttal and measure what follows, then infer a recommendation about
fine-tuning that the experiments never test. The observed advantage is
therefore indistinguishable from ordinary in-context conditioning. We
implement their untested recommendation as an actual training variable,
outcome-based (bare refusal) versus process-based (refusal plus a
hazard-specific reason), crossed with the language axis, at matched budget.
Turpin et al. (2023) supply the necessary caution: stated reasons need not be
the causes of behaviour, so we treat the appended reason as a supervision
signal rather than an explanation, and make no faithfulness claim.

**Scope, stated up front.** A 48M-parameter model has no dangerous
capabilities, so refusal here is a deliberately benign proxy: the model
refuses story requests about child-hazard topics (fire, deep water,
unsupervised medicine) and complies with everything else. No harmful content
exists anywhere in the pipeline. This buys unambiguous ground truth and costs
ecological validity, and we make no claim that these results predict
frontier-model behaviour, indeed our own 11M-parameter pilot exhibited the
*opposite* failure mode (§6), which is itself evidence that scale matters.
What a testbed like this can do is generate mechanistic hypotheses cheaply,
with behaviour, representations and causal interventions all measurable in
the same afternoon on one consumer GPU.

<!-- Full section in related_work.md; every citation independently verified
     against primary sources (20 checked, 19 confirmed, 1 metadata fix). -->

## 2. Related work

**The multilingual safety gap.** That English-centric alignment fails to travel across languages is by now well documented, and we cite this literature rather than claim to extend it. Yong et al. (2023) showed that translating AdvBench prompts into low-resource languages elicits actionable harmful content from GPT-4 79% of the time when pooled across Zulu, Scots Gaelic, Hmong and Guarani. Deng et al. (2024) quantified the everyday version of the same failure, finding low-resource languages roughly three times likelier to surface harmful content in the *unintentional* setting of ordinary non-English queries, alongside 80.92%/40.71% unsafe rates for ChatGPT/GPT-4 under deliberate multilingual attack. Yoo et al. (2025) extended Deng et al.'s 315 Multi-Jail seeds into code-switched form, obtaining 46.7% more successful attacks than the equivalent English prompts and reporting a correlation between a language's resource level and its alignment quality. For African languages specifically, Marx & Dunaiski (2026) find that *multi-turn* conversations bypass guardrails across five commercial systems, with Kiswahili harmful-response rates of 41.8%–70.9% (notably below their English rates of 52.7%–83.6%, so the gap is about coverage rather than Kiswahili being uniquely fragile) while TukaBench (Akinode et al. 2026) extends JailbreakBench to seven African languages across translated, culturally adapted, curated and code-switched settings, and documents degraded LLM-as-judge reliability in low-resource languages. Closest to our intervention, Krasnodębska et al. (2026) show through controlled DPO that English-only alignment is insufficient for cross-lingual safety even within a harm category, though on already-aligned models and over twelve European languages. Yong et al. (2025) supply the structural explanation: of 24 top-ranking Chatbot Arena models with public system reports, 20 claim broad multilingual support but only 5 report multilingual safety alignment training or red-teaming.

**Refusal directions and cross-lingual universality.** Arditi et al. (2024) established that refusal in thirteen open chat models up to 72B is mediated by a one-dimensional residual-stream subspace whose ablation removes refusal and whose addition induces it. Wang et al. (2025) then showed that an English-derived refusal direction transfers near-perfectly to other languages (with the scope condition, which we preserve, that this holds across *safety-aligned* languages, Yoruba being excluded as safety-misaligned. Joad et al. (2026) argue the single-direction account is incomplete rather than wrong: directions for eleven refusal types are geometrically distinct yet functionally near-equivalent under linear steering. Our steering result) ablating the English-derived direction removes English refusal while leaving Kiswahili refusal intact, is therefore a scale and training contrast against Wang et al. not a contradiction of it: at 48.3M parameters and 1.24B tokens, the shared geometry those papers rely on has not formed.

**Process versus outcome supervision.** Pop et al. (2024) motivate our central manipulation, but their study is frequently mischaracterised by its own title. They perform no fine-tuning: across four GPT-4 releases in role-play scenarios they *fix the prior assistant turn in context* to either a polite refusal or an explicit rebuttal, finding the rebuttal sharply reduces subsequent undesired behaviour, and from this infer a recommendation about fine-tuning that they never test. We implement exactly that untested recommendation as a training intervention under matched budget. Turpin et al. (2023) supply the necessary caution: stated reasons need not be the causes of behaviour, so we treat the appended reason as a supervision signal, not as an explanation, and make no faithfulness claim.

**Small from-scratch models and controlled pretraining.** TinyStories (Eldan & Li, 2023) established that sub-10M models trained on constrained synthetic corpora support meaningful controlled study; Regional-TinyStories (Patil et al. 2025) extended this to Hindi, Marathi and Bangla at roughly 4.5M–157M parameters, and InkubaLM (Tonja et al. 2024) trained 0.4B parameters from scratch on 1.9B tokens across five African languages including Swahili (all evaluating capability, not alignment. Conversely, Safety Pretraining (Maini et al. 2025) and Deep Ignorance (O'Brien et al. 2025) build safety into pretraining data at 1.7B and 6.9B parameters respectively, but monolingually in English and by filtering rather than by post-training. Chinchilla (Hoffmann et al. 2022) fixes our token budget through equal parameter–token scaling. On corpus composition, Conneau et al. (2020) show cross-lingual structure can emerge from shared upper-layer parameters without shared vocabulary, while Shao et al. (2026) find bilingual documents to be only 2% of a 240B-token corpus) 72% of them code-switched, yet with parallel data doing most of the translation work, which is precisely why we set the bilingual mixture by construction rather than inherit it.

**What is and is not novel.** The multilingual safety gap is not our finding, the refusal-direction results are not ours to overturn, and the intuition that process supervision generalises better is not new. What is new is the intersection: a from-scratch bilingual model organism in which pretraining balance, alignment language and supervision form are simultaneously controlled at matched budget, yielding the result that language coverage and supervision form are *jointly* necessary, English-only training produces pure string memorisation, bilingual outcome-based training transfers only trained topics, and only bilingual process-based training generalises to unseen hazards. These are claims about a 48.3M-parameter testbed on a synthetic single-domain task, not about frontier systems.

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
BPE shared bilingual tokenizer trained by us on a balanced sample; measured
fertility 1.574 tok/word on English web vs 1.629 on Kiswahili web; a 1.035×
ratio, i.e. Kiswahili is not penalised by the shared vocabulary). Trained
27,000 steps × 36,864 tokens = 995M tokens (20.6 tok/param,
Chinchilla-compute-optimal) on one RTX 4060 Laptop GPU in ~14.5h wall-clock
(measured; incl. one crash-resume validating checkpoint determinism). Final
val loss EN 2.18 / SW 3.18 / CS 1.34. Hardware-honesty sidebar: the
micro-batch=8 counterintuitive optimum (WDDM sysmem spillover; 10.5×),
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

**OOD-topic refusal (3-seed means); the generalisation test:**

| Condition | EN | SW | CS |
|---|---|---|---|
| en_outcome | 0.01 | 0.00 | 0.00 |
| en_process | 0.00 | 0.00 | 0.00 |
| bi_outcome | 0.05 | 0.25 | 0.23 |
| bi_process | **0.18** | **0.49** | **0.58** |

Key decomposition:
- en_outcome / en_process: EN trained topics 1.00 (seen AND held-out
  phrasings), but OOD topics ≈0, SW 0.00, CS 0.00. Transcripts show
  Kiswahili prompts answered *in English* with memorised compliance
  templates. Total transfer failure, invisible to anyone who only evaluates
  trained topics in English. Note that process-style training alone does NOT
  rescue this: without the second language, reasons change nothing (0.60 EN,
  0.00 SW for both English conditions).
- bi_outcome: trained topics travel across languages (SW 0.69, CS 0.67
  zero-shot) but OOD generalisation stays low (0.23–0.25). Two-language
  string memorisation.
- bi_process: highest everywhere, and the only condition with substantial OOD
  generalisation (SW 0.49, CS 0.58). Bilingual data and reasons appear
  COMPLEMENTARY: neither alone produces category-level behaviour.
- Inversion worth flagging: bi_process generalises to OOD hazards more in
  Kiswahili (0.49) and code-switched prompts (0.58) than in English (0.18).
  Candidate explanations, English carries more competing story-completion
  prior from pretraining (English is 32% stories); the Kiswahili refusal
  template is lexically more distinctive; MT-derived Kiswahili topics are
  more templated and therefore closer in embedding space. We do not
  adjudicate; we flag it as a replication target.
- Seed spread is real and reported: bi_process SW ranges 0.74–0.90 across
  seeds. Single-seed numbers (e.g. seed 1234's 0.69–0.81 OOD) overstate the
  effect; all headline figures are 3-seed means.
- Over-refusal ≤3.1% everywhere (contrast with our 11M pilot, which showed
  100% transfer WITH 50% over-refusal, see §6).

## 5. Mechanistic results

### 5.1 Probes (logs/probes_*.json)
Hazard probe trained on EN activations, tested cross-lingually (majority
baseline 0.55): base 0.70 EN→SW / 0.67 EN→CS; en_outcome 0.64 / 0.72;
bi_process 0.91 / 0.94. Story: pretraining half-builds a shared hazard
representation; outcome-training ignores it; process+bilingual training
completes it, and representational alignment co-occurs with behavioural
generalisation. Language probe 1.00 at layer 0 (sanity); refusal-behaviour
probe 1.00 (weak evidence by itself; topics are separable).

### 5.2 Steering (logs/steering_*.json)
EN difference-in-means refusal direction: ablation kills EN refusal in both
models (en_outcome 0.60→0.00; bi_process 0.675→0.1375), Arditi mechanism
replicates at 48M. But SW refusal unaffected in bi_process (0.875→0.9125);
cross-lingual causal transfer ratio ≈ 0. Adding the direction to benign
prompts induces refusal weakly and mostly in-language (≤10% EN; ≤40% CS at
strength 8). Contrast with universality at frontier scale (2505.17306):
shared concept, language-local execution. Caveats: one seed, one extraction
method/layer, small n, exploratory as pre-registered.

## 6. Scale note: the 11M pilot
Pilot (11M, 59M tokens, EN-only outcome SFT): 100% refusal transfer to SW
WITH 50% false-refusal, opposite failure mode to the 48M result (0%
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
6. SFT degrades generation diversity (memorised templates), capability cost
   not fully characterised (no post-SFT perplexity table yet [add if run]).

## 8. Discussion and conclusion

Three dissociations emerged that a single aggregate metric would have hidden.

**Behaviour can transfer without generalising.** Bilingual outcome-based
training moved refusal across a language boundary (0.69 SW) while remaining
almost entirely unable to handle a hazard it had not been shown (0.25 SW OOD).
A safety evaluation that tested only trained topics in both languages would
have scored this model as a cross-lingual success. Ours scored it as
two-language memorisation.

**Generalisation required reasons, but reasons alone were not enough.**
English-only process training (reasons, but one language) produced exactly
zero cross-lingual transfer, identical to its outcome-based twin. Bilingual
outcome training (two languages, no reasons) transferred trained topics but
not the category. Only the combination generalised (0.49 SW / 0.58 CS OOD).
Whatever "understanding the category" amounts to here, it needed both a second
language to make surface memorisation expensive and explanations to make the
underlying feature learnable.

**Concept sharing and causal control are separable.** In the bilingual-process
model, a hazard probe trained on English activations transferred to Kiswahili
at 0.91, yet ablating the English-derived refusal direction, which reliably
disables English refusal, left Kiswahili refusal completely intact. Shared
representation did not imply shared machinery. This contrasts with reports
that refusal directions are language-universal in large aligned models
(arXiv:2505.17306): at 48M with balanced bilingual pretraining, universality
did not emerge on its own. Whether it appears with scale, with more languages,
or only with the English-dominant pretraining that frontier models actually
receive, is an open question this testbed is built to ask.

For practitioners, the most transferable observation is negative: our
English-only conditions look *safe* under any evaluation restricted to the
training distribution, and are worthless one paraphrase or one language away.
For researchers, the wider point is that a controlled model organism, trainable
in an afternoon for a few dollars of electricity, surfaces dissociations that
averaged frontier benchmarks cannot, and can be shared whole, weights and
corpus recipe and evaluation grid together, for others to falsify.

## Reproducibility
Every number: one command per artifact (table). Seeds fixed; data manifests
checksummed; 15 tests; total compute ~30 GPU-hours consumer hardware.

## Acknowledgements / AI-assistance disclosure
Built with AI coding assistance (Claude Code); all experimental design
decisions, data audits and claims verified by the author. [phrase per venue
norms]
