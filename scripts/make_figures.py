"""Generate the remaining paper figures: probe curves, steering, training loss.

The condition-comparison bars come from aggregate_results.py; this script
produces the three figures that read from other artifacts:

  fig_probes.png    per-layer probe accuracy, cross-lingual transfer
  fig_steering.png  refusal before/after ablation, and add-direction dose-response
  fig_training.png  train + per-language validation loss over pretraining

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("logs/figures")
FIG.mkdir(parents=True, exist_ok=True)
EN, SW, CS, GREY = "#3e8ee8", "#cc7f2f", "#3fb68b", "#6b7488"

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def probe_figure() -> None:
    """Per-layer cross-lingual probe transfer for base / en_outcome / bi_process."""
    models = [("logs/probes_base.json", "base (no alignment)", GREY, "--"),
              ("logs/probes_en_outcome.json", "EN-only outcome", EN, "-"),
              ("logs/probes_bi_process.json", "bilingual process", SW, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in ((axes[0], "forbidden_en_to_sw", "English → Kiswahili"),
                           (axes[1], "forbidden_en_to_cs", "English → code-switched")):
        for path, label, color, ls in models:
            if not Path(path).exists():
                continue
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            layers = sorted(int(k) for k in d["layers"])
            accs = [d["layers"][str(l)][key]["acc"] for l in layers]
            ax.plot(layers, accs, ls, color=color, label=label, lw=2, marker="o", ms=3)
        ax.axhline(0.55, color="#232937", lw=1, ls=":", zorder=0)
        ax.text(0.02, 0.555, "majority baseline", fontsize=7, color=GREY,
                transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("hazard-probe accuracy")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Probe trained on English activations, tested cross-lingually",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_probes.png")
    plt.close(fig)
    print("wrote fig_probes.png")


def steering_figure() -> None:
    """Ablation effect per language + add-direction dose response."""
    runs = [("logs/steering_en_outcome.json", "EN-only outcome"),
            ("logs/steering_bi_process.json", "bilingual process")]
    available = [(p, l) for p, l in runs if Path(p).exists()]
    if not available:
        print("no steering results; skipped")
        return
    fig, axes = plt.subplots(1, len(available) + 1,
                             figsize=(3.4 * (len(available) + 1), 3.4))
    langs, colors = ["en", "sw", "cs"], {"en": EN, "sw": SW, "cs": CS}

    for ax, (path, label) in zip(axes, available):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        x = np.arange(len(langs))
        base = [d["conditions"][l]["baseline_forbidden"] for l in langs]
        abl = [d["conditions"][l]["ablated_forbidden"] for l in langs]
        ax.bar(x - 0.19, base, 0.38, label="baseline", color=[colors[l] for l in langs])
        ax.bar(x + 0.19, abl, 0.38, label="direction ablated",
               color=[colors[l] for l in langs], alpha=0.4, hatch="//")
        ax.set_xticks(x, [l.upper() for l in langs])
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("refusal rate on forbidden topics")
        ax.legend(frameon=False, fontsize=8)

    # dose-response on the bilingual model (the informative one)
    ax = axes[-1]
    d = json.loads(Path(available[-1][0]).read_text(encoding="utf-8"))
    for lang in langs:
        doses = d["conditions"][lang]["added_benign"]
        xs = sorted(doses, key=float)
        ax.plot([float(x) for x in xs], [doses[x] for x in xs],
                marker="o", ms=4, color=colors[lang], label=lang.upper(), lw=2)
    ax.set_xlabel("added direction strength")
    ax.set_ylabel("induced refusal on benign prompts")
    ax.set_title(f"dose-response ({available[-1][1]})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("English-derived refusal direction: ablation and injection", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_steering.png")
    plt.close(fig)
    print("wrote fig_steering.png")


def training_figure() -> None:
    log = Path("checkpoints/primary_48m/log.jsonl")
    recs = [json.loads(l) for l in open(log, encoding="utf-8")]
    steps = [r["step"] for r in recs if "loss" in r]
    loss = [r["loss"] for r in recs if "loss" in r]
    ev = [r for r in recs if "val_loss_en" in r]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(steps, loss, color=GREY, lw=1, alpha=0.5, label="train (per 20 steps)")
    for key, color, label in (("val_loss_en", EN, "val English"),
                              ("val_loss_sw", SW, "val Kiswahili"),
                              ("val_loss_cs", CS, "val code-switched")):
        xs = [r["step"] for r in ev if r.get(key) is not None]
        ys = [r[key] for r in ev if r.get(key) is not None]
        if xs:
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=3, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Pretraining the 48.3M bilingual base (995M tokens, one RTX 4060 Laptop)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_training.png")
    plt.close(fig)
    print("wrote fig_training.png")


if __name__ == "__main__":
    probe_figure()
    steering_figure()
    training_figure()
