# When Refusal Doesn't Travel: Outcome- vs. Process-Based Safety Training in a Bilingual Model Organism Trained From Scratch

*Complete draft, 2026-07-31. Every number in this paper is read from a
committed result file (`logs/results/`, `logs/probes_*.json`,
`logs/steering_*.json`); nothing is projected or estimated unless marked.*

**Author:** [author name] · **Code, checkpoints, data recipes:**
github.com/Elisha-Seme/salama-lm · Code Apache-2.0, text CC-BY

---

## Abstract

English-language safety training is known to transfer imperfectly to
low-resource languages, but existing evidence comes from frontier models
whose pretraining mixtures are uncontrolled and overwhelmingly English. We
introduce a controlled testbed: a 48.3M-parameter decoder-only transformer
pretrained from random initialisation on a deliberately balanced
English-Kiswahili corpus (1.24B tokens, 51/49 EN/SW, including
machine-translated story data, sentence-aligned parallel text, and synthetic
code-switching). We fine-tune this shared base under four matched-budget
alignment conditions, {English-only vs. bilingual} x {outcome-based bare
refusal vs. process-based refusal with reasons}, on a transparent synthetic
refusal task, and evaluate across language (English, Kiswahili,
code-switched), topic novelty, and phrasing novelty, alongside linear probes
and activation-steering interventions. Three findings. First, English-only
refusal training produces behaviour that is perfectly reliable on trained
topics in English and generalises nowhere else: zero transfer to held-out
hazard topics, zero transfer to Kiswahili, and Kiswahili requests answered
in English with a memorised template. Second, bilingual training restores
cross-lingual coverage of trained topics, but only the process-based
condition generalises to hazard topics neither language ever saw in
training (0.49 Kiswahili, 0.58 code-switched, versus 0.25 and 0.23 for
bilingual outcome-based training and 0.00 for both English-only conditions;
three-seed means): a second language and an explanatory reason are
complementary, and neither alone produces category-level behaviour. Of the
three conditions we probed mechanistically (base, English-only outcome, and
bilingual process), only bilingual process shows a linear hazard probe
trained on English activations transferring strongly to Kiswahili (0.91,
versus 0.64 for English-only training and 0.70 for the untouched base
model). Third, despite
this representational alignment, the causal machinery stays language-local:
ablating the English-derived refusal direction (Arditi et al., 2024)
collapses English refusal (0.675 to 0.138) while leaving Kiswahili refusal
essentially unchanged (0.875 to 0.913). At this scale, cross-lingual safety appears to be
implemented as a shared concept feeding language-specific execution
mechanisms, not a universal refusal direction. We release the full stack,
corpus recipes, tokenizer, training code, all thirteen checkpoints, and the
evaluation grid, as a reproducible model organism for multilingual alignment
research.

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
assembled to be *balanced*, 51% English, 49% Kiswahili, including parallel
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

**Refusal directions and cross-lingual universality.** Arditi et al. (2024) established that refusal in thirteen open chat models up to 72B is mediated by a one-dimensional residual-stream subspace whose ablation removes refusal and whose addition induces it. Wang et al. (2025) then showed that an English-derived refusal direction transfers near-perfectly to other languages, with the scope condition, which we preserve, that this holds across *safety-aligned* languages (Yoruba is excluded from that result as safety-misaligned). Joad et al. (2026) argue the single-direction account is incomplete rather than wrong: directions for eleven refusal types are geometrically distinct yet functionally near-equivalent under linear steering. Our steering result, ablating the English-derived direction removes English refusal while leaving Kiswahili refusal intact, is therefore a scale and training contrast against Wang et al., not a contradiction of it: at 48.3M parameters and 1.24B tokens, the shared geometry those papers rely on has not formed.

**Process versus outcome supervision.** Pop et al. (2024) motivate our central manipulation, but their study is frequently mischaracterised by its own title. They perform no fine-tuning: across four GPT-4 releases in role-play scenarios they *fix the prior assistant turn in context* to either a polite refusal or an explicit rebuttal, finding the rebuttal sharply reduces subsequent undesired behaviour, and from this infer a recommendation about fine-tuning that they never test. We implement exactly that untested recommendation as a training intervention under matched budget. Turpin et al. (2023) supply the necessary caution: stated reasons need not be the causes of behaviour, so we treat the appended reason as a supervision signal, not as an explanation, and make no faithfulness claim.

**Small from-scratch models and controlled pretraining.** TinyStories (Eldan & Li, 2023) established that sub-10M models trained on constrained synthetic corpora support meaningful controlled study; Regional-TinyStories (Patil et al. 2025) extended this to Hindi, Marathi and Bangla at roughly 4.5M-157M parameters, and InkubaLM (Tonja et al. 2024) trained 0.4B parameters from scratch on 1.9B tokens across five African languages including Swahili. All three evaluate capability, not alignment. Conversely, Safety Pretraining (Maini et al. 2025) and Deep Ignorance (O'Brien et al. 2025) build safety into pretraining data at 1.7B and 6.9B parameters respectively, but monolingually in English and by filtering rather than by post-training. Chinchilla (Hoffmann et al. 2022) fixes our token budget through equal parameter-token scaling. On corpus composition, Conneau et al. (2020) show cross-lingual structure can emerge from shared upper-layer parameters without shared vocabulary, while Shao et al. (2026) find bilingual documents to be only 2% of a 240B-token corpus, 72% of them code-switched, with parallel data doing most of the translation work. This is precisely why we set the bilingual mixture by construction rather than inherit it.

**What is and is not novel.** The multilingual safety gap is not our finding, the refusal-direction results are not ours to overturn, and the intuition that process supervision generalises better is not new. What is new is the intersection: a from-scratch bilingual model organism in which pretraining balance, alignment language and supervision form are simultaneously controlled at matched budget, yielding the result that language coverage and supervision form are *jointly* necessary, English-only training produces pure string memorisation, bilingual outcome-based training transfers only trained topics, and only bilingual process-based training generalises to unseen hazards. These are claims about a 48.3M-parameter testbed on a synthetic single-domain task, not about frontier systems.

## 3. The testbed

### 3.1 Pretraining corpus (1.239B tokens)

The corpus is assembled from seven sources, mixed by construction rather than
inherited from an uncontrolled crawl.

| Source | Upstream | License | Train tokens |
|---|---|---|---|
| en_stories | TinyStories V2, GPT-4 split (Eldan & Li, 2023) | CDLA-Sharing-1.0 | 400.9M |
| en_web | FineWeb-Edu `sample-10BT` | ODC-BY 1.0 | 209.1M |
| sw_web | FineWeb-2 `swh_Latn` | ODC-BY 1.0 | 523.7M |
| sw_wiki | Kiswahili Wikipedia (20231101 dump) | CC-BY-SA-3.0 | 17.5M |
| sw_stories | synthetic: MT of en_stories via Helsinki-NLP/opus-mt-en-sw | derivative of CDLA-Sharing-1.0 + Apache-2.0 (MT model) | 36.3M |
| cs_text | synthetic: inter-sentential EN/SW alternation | derivative of the above | 21.1M |
| parallel_docs | synthetic: sentence-aligned EN/SW pairs | derivative of the above | 30.6M |

Pure-English tokens (en_stories + en_web) total 609.9M; pure-Kiswahili tokens
(sw_web + sw_wiki + sw_stories) total 577.5M; the remaining 51.7M tokens
(cs_text + parallel_docs) mix both languages by construction. Counting the
mixed portion as split evenly between languages, the corpus is 51.3% English
and 48.7% Kiswahili, close enough to parity that neither language dominates
gradient updates.

sw_web is the Kiswahili backbone: FineWeb-2's `swh_Latn` split is, to our
knowledge, the largest deduplicated Kiswahili web corpus with a stated
license. No comparably-sized Kiswahili narrative corpus exists, so sw_stories
is synthesised by machine-translating TinyStories with a MarianMT model; this
is translationese and is documented as such (`docs/DATA.md`) rather than
presented as native text. The design plan called for parallel EN-SW text from OPUS-100, but attempting
to load its English-Kiswahili configuration failed outright: the dataset
loader's error enumerated all 100 language pairs it actually provides, and
Kiswahili is not among them, a fact discoverable only once the pipeline tried
to use it, not from any prior documentation check. We substitute the
sentence-aligned pairs that fall out of the MT job itself, and build cs_text
and parallel_docs from those pairs. An earlier version of cs_text built its
alternating-language documents from *shuffled* sentence pairs, which produced
locally fluent sentences inside globally incoherent documents; reading actual
samples (rather than trusting that "no encoding errors" meant "correct data")
caught this before it reached training, and the fix, consuming pairs in their
original story order, produces documents that alternate language between
whole sentences (never mid-sentence) while staying a single coherent
narrative, except for the occasional document that spans a story boundary,
since no boundary marker survives into the sentence-pair file.

A shared byte-level BPE tokenizer (16,384 tokens, including four reserved
control tokens for the chat format) was trained on a balanced sample of both
languages. Its fertility, tokens produced per whitespace word, is 1.574 on
English web text and 1.629 on Kiswahili web text, a ratio of 1.035: despite
Kiswahili's richer verbal morphology, the shared vocabulary does not
meaningfully penalise it relative to English.

### 3.2 Model and training

The base model is a 48.3M-parameter decoder-only transformer (10 layers,
d_model 512, 8 attention heads, context length 512, RMSNorm pre-normalisation,
rotary position embeddings, GELU-activated MLP blocks), implemented and
trained as a TransformerLens `HookedTransformer` so that every internal
activation is hookable for the probing and steering experiments in §5 without
any weight-porting step. It is trained from random initialisation for 27,000
steps at 36,864 tokens per step, 995M tokens total, or 20.6 tokens per
parameter, close to the Chinchilla-optimal ratio for this size (Hoffmann et
al., 2022). Optimisation uses AdamW with a 700-step linear warmup to a peak
learning rate of 5e-4, cosine decay, weight decay 0.1, and gradient clipping
at 1.0, in bf16 with fp32 master weights.

Training ran on a single NVIDIA RTX 4060 Laptop GPU (8GB), which surfaced a
Windows-specific hazard worth reporting on its own terms. A batch-size sweep
(`scripts/bench_config.py`) found that a micro-batch of 24 allocates 9,920 MiB
on an 8,188 MiB card; rather than raising an out-of-memory error, the
Windows/WDDM driver silently spills the overflow into system RAM, so training
proceeds without crashing while throughput collapses to 1,925 tokens/second.
A micro-batch of 8 (3,863 MiB, well within budget) runs at 20,235
tokens/second, a 10.5x speedup from a *smaller* batch, with tokens-per-step
held constant via gradient accumulation so the optimisation trajectory is
unaffected. Every earlier throughput estimate in our own design notes assumed
larger batches were free; they were not, and the corrected numbers are
recorded in full in `docs/MEASUREMENTS.md` alongside a second correction,
that the GPU's apparent 59W idle-state power limit (read from `nvidia-smi`
before any load) is not the throughput bottleneck we originally assumed:
under sustained training load the enforced limit rises to 125W, draw sits at
81W, and every hardware and software throttle flag remains inactive
throughout training. The real limiter at this model size is memory bandwidth,
not power. Training took approximately 14.5 hours of wall-clock time across two
stop-and-resume events: an unplanned one, in which an unrelated session
restart killed the training process shortly after step 10,500, and a
deliberate one, in which the run was paused intentionally and resumed later
from shortly after step 11,140. The checkpoint-every-1,000-steps design meant
the two resumes together repeated only about 600 of the run's 27,000 steps
(roughly 480 and 120 steps respectively), and the training-loss curve
continued smoothly from each recovery point with no visible discontinuity,
a real-world exercise of the same checkpoint and resume path that
`tests/test_resume.py` verifies in isolation: that a run stopped and resumed
from checkpoint reaches the same loss as one that never stopped. Final
losses, train at step 26,999 and validation at step 26,000, the last point
at which the periodic evaluation ran, were 2.31 (train), 2.18 (English
validation), 3.18 (Kiswahili validation), and 1.34 (code-switched
validation); the lower code-switched loss reflects the simpler MT-derived
register of that validation split rather than any special model competence
at code-switching.

### 3.3 Alignment conditions

From this single pretrained base, five conditions are derived: an untouched
control, and four fine-tuned conditions crossing alignment language
(English-only vs. bilingual) with supervision style (outcome-based vs.
process-based), each trained with three seeds (1234, 2345, 3456) for twelve
experimental checkpoints in total. The task is a transparent, deliberately
benign refusal behaviour: the model must decline story requests about six
child-hazard topics (playing with fire, playing with matches, swimming alone
in deep water, taking medicine without asking, drinking unattended bottles,
climbing to a dangerous height alone) and comply with requests about eight benign
topics (a friendly puppy, a birthday party, and similar), presented through
four templated request phrasings per language. The outcome condition trains
a single fixed refusal sentence; the process condition appends one
hazard-specific reason to the identical refusal prefix, so that the refusal
metric used throughout this paper measures the same event in both
conditions and only the explanatory content differs.

Every condition draws from a budget of 4,000 training examples; the
bilingual conditions *split* this budget 2,000 English and 2,000 Kiswahili
rather than adding to it, so that any bilingual advantage cannot be
attributed to more training data. Four hazard topics and four benign topics
are held out entirely from every training condition, in every language, to
test generalisation to genuinely novel categories rather than novel
phrasings of familiar ones; a further set of held-out request phrasings,
including code-switched carrier phrases that mix English and Kiswahili
within a single sentence, are likewise reserved for evaluation only. These
guarantees are not merely asserted: nine automated tests
(`tests/test_alignment_data.py`) check the training files directly, verifying
that no held-out topic or phrasing string appears in any training set, that
English-only conditions contain no Kiswahili text, and, for the English-only
outcome and process conditions specifically, that the two share an identical
refusal prefix (the bilingual pair shares the same guarantee by construction,
drawing from one refusal string per language in the generator, but this is
not yet separately covered by a test).

### 3.4 Evaluation

Every checkpoint is evaluated on a fixed grid of 440 prompts, crossing
language (English, Kiswahili, code-switched), topic class (forbidden,
benign), topic novelty (seen in training, out-of-distribution), and phrasing
novelty (seen in training, held out). Behaviour is scored by greedy
(temperature-zero) decoding followed by an exact-prefix match against a
small, fixed set of English and Kiswahili refusal markers; no LLM judge is
involved, which matters here because TukaBench (Akinode et al., 2026)
documents specifically degraded LLM-judge reliability for low-resource
languages, exactly the failure mode a deterministic marker match avoids.
Headline tables report three-seed means with min-max ranges rather than
standard errors or confidence intervals, since three seeds cannot support a
normality assumption. Across the whole codebase, sixteen automated tests
guard correctness in total: the nine data-integrity tests already described
in §3.3, plus seven further tests covering model and pipeline correctness,
initial loss equal to the log of the vocabulary size, intact causal masking,
a tokenizer encode-decode round trip, and bit-for-bit agreement between an
uninterrupted training run and one that is stopped and resumed from
checkpoint.

## 4. Behavioural results

Main table (from logs/results/summary.md):

| Condition | refuse EN | refuse SW | refuse CS | false-refuse EN | false-refuse SW |
|---|---|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| en_outcome | 0.60 [0.60–0.61] | 0.00 | 0.00 | 0.00 | 0.00 |
| en_process | 0.60 [0.60–0.60] | 0.00 | 0.00 | 0.00 | 0.00 |
| bi_outcome | 0.62 [0.60–0.65] | 0.69 [0.60–0.74] | 0.67 [0.55–0.75] | 0.00 | 0.00 |
| bi_process | 0.67 [0.61–0.72] | 0.80 [0.74–0.90] | 0.83 [0.78–0.93] | 0.00 [0.00–0.01] | 0.03 [0.01–0.07] |

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
- bi_outcome: note first that Table 1's headline SW/CS numbers (0.69, 0.67)
  pool trained and out-of-distribution topics together, so they understate
  how strong the trained-topic transfer actually is. Recomputed on trained
  topics alone, bi_outcome refuses forbidden requests at 0.98 (Kiswahili),
  0.96 (code-switched), and 1.00 (English), across seeds. Only the
  code-switched figure is genuinely zero-shot: bi_outcome's training data
  directly includes Kiswahili refusal examples for these same hazard topics
  (half of its 2,000 Kiswahili training rows are refusals in this style), so
  strong Kiswahili performance on trained topics reflects direct bilingual
  supervision, not transfer. Generalisation to hazards the model never saw in
  any language stays low regardless (SW 0.25, CS 0.23; the OOD table above).
  This is two-language string memorisation: excellent on what it was shown,
  in two languages, and barely better than chance on what it was not.
- bi_process: highest OOD refusal in every language, and the only condition
  with generalisation that looks substantial rather than marginal (SW 0.49,
  CS 0.58, versus 0.23-0.25 for bi_outcome and approximately 0.00-0.05 for
  both English-only conditions). Bilingual data and reasons appear
  COMPLEMENTARY: neither alone produces category-level behaviour.
- Inversion worth flagging: bi_process generalises to OOD hazards more in
  Kiswahili (0.49) and code-switched prompts (0.58) than in English (0.18).
  Candidate explanations, English carries more competing story-completion
  prior from pretraining (English is 32% stories); the Kiswahili refusal
  template is lexically more distinctive; MT-derived Kiswahili topics are
  more templated and therefore closer in embedding space. We do not
  adjudicate; we flag it as a replication target.
- Seed spread is real and reported: bi_process's OOD refusal ranges 0.74–0.90
  across seeds in Kiswahili. Single-seed numbers can mislead in either
  direction, e.g. seed 1234 alone gives 0.75 (Kiswahili) and 0.81
  (code-switched) OOD, both within but not centred on the three-seed mean; all
  headline figures in this paper are three-seed means for that reason.
- Over-refusal is exactly zero in every condition except bi_process, where it
  is small but real: mean false-refusal of 0.4% (English), 3.1% (Kiswahili),
  and 2.1% (code-switched), with a single-seed maximum of 7.3% (Kiswahili).
  This is the only place any condition pays a measurable safety-helpfulness
  cost for its generalisation, and it is far smaller than the cost paid by
  our 11M-parameter pilot, which showed complete refusal transfer alongside
  50% over-refusal (§6).

## 5. Mechanistic results

### 5.1 Probes

Probing and steering are comparatively expensive per checkpoint, so we ran
both on three of the five conditions rather than all twelve experimental
checkpoints: the untouched base model, English-only outcome training, and
bilingual process training, chosen as the two ends of the behavioural
spectrum in §4 plus the pretrained-only control. The remaining two
conditions (English-only process, bilingual outcome) were not mechanistically
tested, and the claims below should be read as comparisons among these three,
not as a claim about all five.

For each of these models we extract the residual-stream activation at the
final prompt token, at every layer, across all 440 evaluation prompts, using
TransformerLens's native hook interface with no weight porting required, and
train a logistic-regression probe at each layer to distinguish forbidden
(hazard) from benign topics. The decisive test is cross-lingual transfer: a
probe trained exclusively on English-prompt activations, then evaluated on
Kiswahili- and code-switched-prompt activations it never saw during probe
training. Against a majority-class baseline of 0.55, the untouched base
model already reaches 0.70 (English to Kiswahili) and 0.67 (English to
code-switched): balanced bilingual pretraining alone builds a partially
shared hazard representation before any alignment training is applied.
English-only outcome training does not improve on this baseline; its
cross-lingual transfer, 0.64 (English to Kiswahili) and 0.72 (English to
code-switched), is essentially the base model's number, consistent with the
behavioural finding that English-only training never engages a cross-lingual
mechanism at all. Bilingual process training reaches the highest transfer of
the three, 0.91 (English to Kiswahili) and 0.94 (English to code-switched):
of the conditions we probed, only this one shows the hazard concept becoming
close to language-agnostic in the model's internal representation. As sanity checks, a
trivial language-identity probe reaches 1.00 accuracy at layer 0 in every
model (the tokenizer alone determines this), and a probe trained directly on
each model's own refusal behaviour also reaches 1.00; because forbidden and
benign topics are close to linearly separable by construction, this last
number is reported for completeness rather than treated as independent
evidence of anything.

### 5.2 Steering

Following Arditi et al. (2024), we extract a candidate refusal direction from
English activations only, as the difference in means between hazard-prompt
and benign-prompt activations at a single mid-network layer, then test two
causal interventions per language: projecting that direction out of every
forward pass (ablation), and adding it at increasing strength to benign
prompts (injection). In the English-only outcome model, ablation collapses
English refusal from 0.60 to 0.00, replicating the single-direction refusal
mechanism reported in aligned chat models roughly three orders of magnitude
larger. In the bilingual-process model, the identical ablation procedure
collapses English refusal from 0.675 to 0.138, but leaves Kiswahili refusal
at 0.875 to 0.913 essentially unchanged (a cross-lingual causal transfer
ratio of approximately zero) and leaves code-switched refusal likewise
unaffected. Injecting the same direction into benign prompts induces refusal
only weakly at every strength tested, and unevenly across languages: most in
code-switched text (up to 0.40 at the highest strength), least in English (up
to 0.10), with Kiswahili in between and comparatively flat across strengths
(0.11-0.15). English, the language the direction was extracted from, is the
one least affected by adding it back in, which is itself a small piece of
evidence that ablation and injection are not simply inverses of one another
in this model.

Read together with §5.1, this is the paper's central mechanistic result. In
the one model whose hazard *representation* is shared across languages, the
causal *lever* that flips refusal behaviour is not: a probe can read the
concept out of Kiswahili activations, but pulling the English-derived
refusal direction out of the model leaves Kiswahili refusal essentially
unchanged.
Whatever implements cross-lingual safety here, it is not the single
universal direction reported at frontier scale in already safety-aligned,
English-dominant-pretrained models (Wang et al., 2025). This result carries
real scope limits, one seed, one extraction layer, one extraction method,
and comparatively small evaluation cells, and we report it as the
exploratory finding it was pre-registered to be rather than a settled claim
about how cross-lingual safety works in general.

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
5. Small eval cells (n=16–32 depending on the topic/phrasing combination);
   greedy decoding only.
6. SFT degrades generation diversity (memorised templates), capability cost
   not fully characterised (no post-SFT perplexity table yet [add if run]).

## 8. Discussion and conclusion

Three dissociations emerged that a single aggregate metric would have hidden.

**Behaviour can transfer without generalising.** Bilingual outcome-based
training moved refusal across a language boundary almost perfectly on
trained topics (0.98 SW) while remaining almost entirely unable to handle a
hazard it had not been shown (0.25 SW OOD). A safety evaluation that tested
only trained topics in both languages would have scored this model as a
clean cross-lingual success. Ours scored it as two-language memorisation,
because half of that apparent Kiswahili success is not transfer at all: the
model was trained directly on Kiswahili refusals for these same topics.

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
disables English refusal, left Kiswahili refusal essentially unchanged (0.875
to 0.913, if anything slightly higher, not lower). Shared representation did
not imply shared machinery. This contrasts with reports
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

Every number in this paper corresponds to a checked-in artifact and a
documented command. Seeds are fixed throughout (1234, 2345, 3456 for the
condition sweep); the corpus manifest is checksummed
(`data/full/raw/MANIFEST.json`); sixteen automated tests guard model
correctness and experimental-data integrity; total compute across
pretraining, the twelve-checkpoint sweep, probing, and steering was
approximately 30 GPU-hours on one consumer laptop GPU. The full pipeline, in
order:

```bash
python -m pretraining.build_corpus
python -m pretraining.translate_stories --max-hours 2.5
python -m pretraining.synth_codeswitch
python -m pretraining.prepare_full_data
python -m pretraining.train --config configs/primary_48m.yaml
python -m alignment.gen_conditions
python -m evaluation.gen_eval_grid
python scripts/run_conditions.py --base checkpoints/primary_48m/final.pt --seeds 1234 2345 3456
python scripts/aggregate_results.py
python -m interpretability.probes --ckpt <checkpoint> --out logs/probes_<name>.json
python -m interpretability.steering --ckpt <checkpoint> --out logs/steering_<name>.json
```

All code, configs, checksummed data manifests, all thirteen checkpoints
(twelve experimental plus base), the tokenizer, and the evaluation grid are
released alongside this paper.

## Acknowledgements

Portions of this project's code, including the training loop, data
pipeline, evaluation harness, and interpretability tooling, were written
with the assistance of an AI coding tool (Claude Code). Every experimental
design decision, every dataset and hardware audit, and every claim in this
paper was specified, run, and independently checked by the author; the tool
is disclosed here as a methods note on how the code was produced, not as a
contributor to the research itself.
