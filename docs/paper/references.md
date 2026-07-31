# References

Every entry below was independently verified against primary sources on
2026-07-31 (20 citations checked; 19 confirmed as cited, 1 corrected — see
note on Wu et al.). Where our characterisation of a paper differs from a
common secondary summary, the verified reading is used.

Akinode et al. (2026). *TukaBench: a culturally grounded jailbreak benchmark
for seven African languages.* arXiv:2606.01322.

Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single
Direction.* NeurIPS 2024. arXiv:2406.11717.

Deng, Y., Zhang, W., Pan, S. J., & Bing, L. (2024). *Multilingual Jailbreak
Challenges in Large Language Models.* ICLR 2024. arXiv:2310.06474.

Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and
Still Speak Coherent English?* arXiv:2305.07759.

Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models*
(Chinchilla). arXiv:2203.15556.

Joad et al. (2026). *There Is More to Refusal in LLMs than a Single Direction.*
arXiv:2602.02132.

Krasnodębska et al. (2026). *Multilingual Refusal Alignment for Safer LLMs.*
arXiv:2606.07535.

Maini, P., et al. (2025). *Safety Pretraining: Toward the Next Generation of
Safe AI.* arXiv:2504.16980.

Marx & Dunaiski (2026). *Multilingual jailbreaking of LLMs using low-resource
languages.* arXiv:2605.18239. — Kiswahili harmful-response rates 41.8–70.9%
under multi-turn attack; note their English rates are *higher* (52.7–83.6%).

O'Brien et al. (2025). *Deep Ignorance: Filtering Pretraining Data Builds
Tamper-Resistant Safeguards.* arXiv:2508.06601.

Patil et al. (2025). *Regional-TinyStories.* IJCNLP-AACL 2025 Findings. —
TinyStories extended to Hindi/Marathi/Bangla at ~4.5M–157M parameters.

Pop, F., Rosenblatt, J., de Lucena, D. S., & Vaiana, M. (2024). *Rethinking
harmless refusals when fine-tuning foundation models.* ICLR 2024 AGI Workshop.
arXiv:2406.19552. — **Verified:** contains no fine-tuning intervention; the
refusal-vs-rebuttal comparison fixes the prior assistant turn *in context*
across four GPT-4 releases. The fine-tuning recommendation is inferred, not
tested. This is the experiment our process/outcome axis implements.

Shao et al. (2026). *The Role of Mixed-Language Documents for Multilingual LLM
Pretraining.* arXiv:2601.00364. — Bilingual documents are ~2% of a 240B-token
corpus, 72% of them code-switched.

Tonja, A. L., et al. (2024). *InkubaLM: A small language model for low-resource
African languages.* arXiv:2408.17024. — 0.4B parameters from scratch on 1.9B
tokens across five African languages including Swahili; capability-focused, no
alignment component.

Turpin, M., et al. (2023). *Language Models Don't Always Say What They Think:
Unfaithful Explanations in Chain-of-Thought Prompting.* arXiv:2305.04388.

Wang et al. (2025). *Refusal Direction is Universal Across Safety-Aligned
Languages.* NeurIPS 2025. arXiv:2505.17306. — Scope condition preserved in our
discussion: universality is reported across *safety-aligned* languages;
Yoruba is excluded as safety-misaligned.

Wu, S., Conneau, A., Li, H., Zettlemoyer, L., & Stoyanov, V. (2019). *Emerging
Cross-lingual Structure in Pretrained Language Models.* arXiv:1911.01464.
Published as Conneau, A., Wu, S., et al. (2020), ACL 2020, pp. 6022–6034,
https://aclanthology.org/2020.acl-main.536/. — **Citation corrected:** an
earlier draft cited "Conneau et al. 2019", which pairs the ACL author order
with the arXiv year and matches neither record.

Yong, Z.-X., Menghini, C., & Bach, S. H. (2023). *Low-Resource Languages
Jailbreak GPT-4.* arXiv:2310.02446.

Yong, Z.-X., et al. (2025). *The State of Multilingual LLM Safety Research.*
arXiv:2505.24119. — Of 24 top Chatbot Arena models with public system reports,
20 claim broad multilingual support; only 5 report multilingual safety
alignment or red-teaming.

Yoo et al. (2025). *Code-Switching Red-Teaming (CSRT).* ACL 2025. —
Code-switched prompts produce 46.7% more successful attacks than equivalent
English prompts.
