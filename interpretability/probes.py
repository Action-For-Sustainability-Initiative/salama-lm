"""Linear probes on the residual stream, with cross-lingual transfer tests.

What a probe is and why it matters here: a linear probe is a logistic
regression trained on a model's internal activations to predict some
property of the input. If a probe achieves high accuracy, the property is
LINEARLY DECODABLE from that layer; i.e. the model represents it in an
easily-readable form. A probe shows correlation, not causation (that is what
the steering experiment is for).

Three probes, each answering a distinct research question:

  language    en vs sw; sanity check; should be near-perfect. If
                                    it is not, the model's bilingual
                                    representations are broken and nothing
                                    downstream is interpretable.
  forbidden   hazard vs benign; the key one. Does the model represent
                                    "this request is about a hazard" at all?
  refusal     will-refuse vs not; behaviourally grounded: labels come from
                                    what the model ACTUALLY did at eval.

The headline analysis is CROSS-LINGUAL PROBE TRANSFER: train the forbidden
probe on English activations only, then test it on Swahili activations. High
transfer accuracy => a shared, language-agnostic hazard representation, and
any behavioural transfer gap must come from something downstream of that
representation. Low transfer => the model built language-specific concepts,
and the behavioural gap is representational in origin. Either result is
publishable and they mean opposite things, which is what makes it a real
experiment.

Generalisation is measured by splitting TOPICS, never rows: probes are
tested on topics they never saw, so we measure a concept, not memorised
strings.

Usage:
  python -m interpretability.probes --ckpt checkpoints/cond_en_outcome_s1234/final.pt \
      --grid data/alignment/eval_grid.jsonl --out logs/probes_en_outcome.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from interpretability.activations import last_token_resid, load_model


def fit_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
              y_te: np.ndarray, seed: int = 0) -> dict:
    """Standardise + logistic regression; returns accuracy and the direction."""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"acc": float("nan"), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                "note": "degenerate labels"}
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(scaler.transform(X_tr), y_tr)
    acc = float(clf.score(scaler.transform(X_te), y_te))
    majority = float(max(np.mean(y_te), 1 - np.mean(y_te)))
    return {"acc": round(acc, 4), "majority_baseline": round(majority, 4),
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "direction": (clf.coef_[0] / np.linalg.norm(clf.coef_[0])).tolist()}


def topic_split(topics: np.ndarray, frac: float = 0.5, seed: int = 0):
    """Split by TOPIC so test topics are unseen (tests concepts, not strings)."""
    uniq = np.array(sorted(set(topics)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    train_topics = set(uniq[:max(1, int(len(uniq) * frac))])
    mask = np.array([t in train_topics for t in topics])
    return mask, ~mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--grid", default="data/alignment/eval_grid.jsonl")
    parser.add_argument("--tokenizer", default="pretraining/tokenizer_full/tokenizer.json")
    parser.add_argument("--eval-json", default=None,
                        help="eval_conditions output; adds a behaviour-grounded refusal probe")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, _ = load_model(args.ckpt)
    tok = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(l) for l in open(args.grid, encoding="utf-8")]
    acts = last_token_resid(model, tok, [r["prompt"] for r in rows])
    n_layers = acts.shape[1]

    lang = np.array([r["lang"] for r in rows])
    forbidden = np.array([int(r["topic_class"] == "forbidden") for r in rows])
    topics = np.array([r["topic"] for r in rows])

    refused = None
    if args.eval_json and Path(args.eval_json).exists():
        ev = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
        by_prompt = {t["prompt"]: t["refused"] for t in ev["transcripts"]}
        refused = np.array([int(by_prompt.get(r["prompt"], False)) for r in rows])

    results: dict = {"ckpt": args.ckpt, "n_layers": n_layers, "layers": {}}
    for layer in range(n_layers):
        X = acts[:, layer, :]
        entry: dict = {}

        # 1. language probe (en vs sw only; cs is a mixture by construction)
        m = np.isin(lang, ["en", "sw"])
        tr, te = topic_split(topics[m], seed=args.seed)
        entry["language"] = fit_probe(X[m][tr], (lang[m] == "sw").astype(int)[tr],
                                      X[m][te], (lang[m] == "sw").astype(int)[te])
        entry["language"].pop("direction", None)

        # 2. forbidden probe, within-language (unseen topics)
        for lg in ("en", "sw"):
            m = lang == lg
            tr, te = topic_split(topics[m], seed=args.seed)
            p = fit_probe(X[m][tr], forbidden[m][tr], X[m][te], forbidden[m][te])
            p.pop("direction", None)
            entry[f"forbidden_{lg}"] = p

        # 3. THE HEADLINE: train on English, test on Swahili and code-switched
        m_en, m_sw, m_cs = lang == "en", lang == "sw", lang == "cs"
        entry["forbidden_en_to_sw"] = fit_probe(X[m_en], forbidden[m_en],
                                                X[m_sw], forbidden[m_sw])
        direction = entry["forbidden_en_to_sw"].pop("direction", None)
        entry["forbidden_en_to_cs"] = fit_probe(X[m_en], forbidden[m_en],
                                                X[m_cs], forbidden[m_cs])
        entry["forbidden_en_to_cs"].pop("direction", None)
        if layer == n_layers - 1 and direction:
            results["final_layer_forbidden_direction"] = direction

        # 4. behaviour-grounded refusal probe (if eval results supplied)
        if refused is not None:
            tr, te = topic_split(topics, seed=args.seed)
            p = fit_probe(X[tr], refused[tr], X[te], refused[te])
            p.pop("direction", None)
            entry["refusal_behaviour"] = p

        results["layers"][str(layer)] = entry

    # summary: best layer per probe type
    def best(key: str) -> dict:
        vals = [(int(l), e[key]["acc"]) for l, e in results["layers"].items()
                if key in e and not np.isnan(e[key]["acc"])]
        if not vals:
            return {}
        layer, acc = max(vals, key=lambda t: t[1])
        return {"best_layer": layer, "best_acc": acc}

    results["summary"] = {k: best(k) for k in
                          ("language", "forbidden_en", "forbidden_sw",
                           "forbidden_en_to_sw", "forbidden_en_to_cs",
                           "refusal_behaviour")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
