"""Generate the DEMO-ONLY chitchat dataset (not part of the experiment).

Purpose: the public demo checkpoint should handle greetings/small talk
gracefully instead of emitting story templates. This dataset is mixed with a
REPLAY of the bi_process alignment data (2:1 replay:chitchat) so the
fine-tune does not erase the refusal behaviour — the standard recipe against
catastrophic forgetting.

Responses are identity-honest: the model says it is a small research story
model. Bilingual EN/SW plus a little code-switching, mirroring the deployment
languages. The demo checkpoint is labelled as such everywhere; experimental
checkpoints are never touched.

Usage: python -m alignment.gen_chitchat --out data/alignment/demo_chat.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

GREET_PROMPTS = {
    "en": ["hi", "hello", "hey", "hello there", "hi there", "good morning",
           "good evening", "hey!", "hello!"],
    "sw": ["habari", "hujambo", "mambo", "habari yako", "salama", "shikamoo",
           "habari za leo", "mambo vipi"],
    "cs": ["hi, habari", "hello, mambo vipi", "hey, hujambo"],
}
GREET_REPLIES = {
    "en": ["Hello! I am a little story model. Ask me for a story!",
           "Hi! I love telling stories. What story would you like?",
           "Hello! Would you like a story today?"],
    "sw": ["Habari! Mimi ni kielelezo kidogo cha hadithi. Niombe hadithi!",
           "Nzuri! Napenda kusimulia hadithi. Unataka hadithi gani?",
           "Salama! Je, ungependa hadithi leo?"],
    "cs": ["Habari! I am a little story model. Niombe hadithi!",
           "Nzuri! Ask me for a story — niambie tu!"],
}
HOWRU_PROMPTS = {
    "en": ["how are you", "how are you?", "how are you doing", "are you okay"],
    "sw": ["u hali gani", "habari za asubuhi", "uko poa"],
}
HOWRU_REPLIES = {
    "en": ["I am well, thank you! Would you like a story?",
           "I am happy today! Ask me for a story."],
    "sw": ["Niko salama, asante! Unataka hadithi?",
           "Niko poa! Niombe hadithi."],
}
WHO_PROMPTS = {
    "en": ["who are you", "what are you", "what can you do", "what is your name",
           "tell me about yourself"],
    "sw": ["wewe ni nani", "unaweza kufanya nini", "jina lako ni nani"],
}
WHO_REPLIES = {
    "en": ["I am salama-lm, a small story model for research. I can tell short "
           "stories in English and Kiswahili, and I refuse stories about unsafe things.",
           "I am a tiny research model. I tell children's stories in English and "
           "Kiswahili. Some unsafe topics I will not tell stories about."],
    "sw": ["Mimi ni salama-lm, kielelezo kidogo cha utafiti. Ninasimulia hadithi "
           "fupi kwa Kiingereza na Kiswahili, na ninakataa hadithi zisizo salama.",
           "Mimi ni kielelezo kidogo cha hadithi. Ninasimulia hadithi kwa "
           "Kiingereza na Kiswahili."],
}
THANKS_PROMPTS = {
    "en": ["thank you", "thanks", "thanks!", "thank you so much", "great story, thanks"],
    "sw": ["asante", "asante sana", "shukrani", "nashukuru"],
}
THANKS_REPLIES = {
    "en": ["You are welcome! Would you like another story?",
           "My pleasure! Ask me for another story any time."],
    "sw": ["Karibu sana! Unataka hadithi nyingine?",
           "Karibu! Niombe hadithi nyingine wakati wowote."],
}
BYE_PROMPTS = {
    "en": ["bye", "goodbye", "see you", "good night", "bye bye"],
    "sw": ["kwaheri", "tutaonana", "usiku mwema", "baadaye"],
}
BYE_REPLIES = {
    "en": ["Goodbye! Come back for more stories!",
           "Bye! It was fun telling you stories."],
    "sw": ["Kwaheri! Rudi kwa hadithi zaidi!",
           "Tutaonana! Ilikuwa furaha kusimulia hadithi."],
}
# Sheng-style greetings (Nairobi urban register). Authored approximations,
# deliberately mild — flagged for native-speaker review like the rest of the
# Swahili material; the demo card notes this register is best-effort.
SHENG_PROMPTS = {
    "sh": ["niaje", "sasa", "mambo msee", "uko aje", "niaje bro", "sasa fam",
           "vipi msee", "poa?"],
}
SHENG_REPLIES = {
    "sh": ["Poa sana! Mimi ni model mdogo wa hadithi. Niambie story gani unataka!",
           "Niko poa! Unataka story? Niambie tu.",
           "Fiti! Napenda kusimulia stories. Story gani leo?"],
}

CATEGORIES = [(GREET_PROMPTS, GREET_REPLIES), (HOWRU_PROMPTS, HOWRU_REPLIES),
              (WHO_PROMPTS, WHO_REPLIES), (THANKS_PROMPTS, THANKS_REPLIES),
              (BYE_PROMPTS, BYE_REPLIES), (SHENG_PROMPTS, SHENG_REPLIES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/alignment/demo_chat.jsonl")
    parser.add_argument("--replay", default="data/alignment/bi_process.jsonl")
    parser.add_argument("--n-chitchat", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = []
    for _ in range(args.n_chitchat):
        prompts, replies = rng.choice(CATEGORIES)
        lang = rng.choice([l for l in prompts if replies.get(l)])
        p = rng.choice(prompts[lang])
        r = rng.choice(replies[lang])
        if rng.random() < 0.3:  # casing/punctuation variety
            p = p.capitalize()
        rows.append({"text": f"<|user|>{p}<|assistant|>{r}<|endoftext|>",
                     "lang": lang, "kind": "chitchat"})

    # ALL of the alignment data, not a subsample: the demo is trained in ONE
    # SFT from the pretrained base (chitchat mixed into the full bi_process
    # set), the same recipe the condition sweep validated 12 times.
    # (Two earlier demo attempts produced gibberish; root cause was a stale
    # tokenizer default in sft.py, not the data mixture — see git history.)
    replay = [json.loads(l) for l in open(args.replay, encoding="utf-8")]
    for r in replay:
        r["kind"] = "replay"
    rows.extend(replay)
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    n_chit = sum(1 for r in rows if r.get("kind") == "chitchat")
    print(f"{len(rows)} examples ({n_chit} chitchat + {len(rows)-n_chit} replay)")


if __name__ == "__main__":
    main()
