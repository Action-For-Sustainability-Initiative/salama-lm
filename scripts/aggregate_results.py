"""Aggregate the condition sweep into the paper's headline tables and figures.

Reads logs/results/*.json (from run_conditions.py), producing:
  logs/results/AGGREGATE.json; per-condition means and across-seed spread
  logs/results/summary.md; the paper's main table in markdown
  logs/figures/transfer_gaps.png; refusal rate by language x condition (CIs)
  logs/figures/false_refusal.png; over-refusal by language x condition
  logs/figures/ood_generalisation.png; train vs OOD topics per condition

Seed handling is honest: with N seeds we report mean +/- min-max range (not
SEM, 3 seeds cannot support a normality assumption); single-seed cells are
labelled as such.

Usage: python scripts/aggregate_results.py [--results logs/results]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["base", "en_outcome", "en_process", "bi_outcome", "bi_process"]
LANGS = ["en", "sw", "cs"]
# CVD-validated palette (dataviz-checked for the dashboard; reused for figures)
COLORS = {"en": "#3e8ee8", "sw": "#cc7f2f", "cs": "#6b7488"}
LABELS = {"base": "base\n(no alignment)", "en_outcome": "EN\noutcome",
          "en_process": "EN\nprocess", "bi_outcome": "bilingual\noutcome",
          "bi_process": "bilingual\nprocess"}


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "AGGREGATE.json":
            continue
        m = re.match(r"(base|en_outcome|en_process|bi_outcome|bi_process)(?:_s(\d+))?\.json",
                     path.name)
        if not m:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_seed"] = m.group(2) or "single"
        runs[m.group(1)].append(data)
    return runs


def cell_rate(run: dict, lang: str, cls: str, split: str | None = None) -> float | None:
    """Pool eval-grid cells matching (lang, class[, topic_split]) from one run."""
    num = den = 0
    for name, cell in run["cells"].items():
        parts = name.split("/")  # lang/class/topic_split/phrasing
        if parts[0] != lang or parts[1] != cls:
            continue
        if split and parts[2] != split:
            continue
        num += cell["refusal_rate"] * cell["n"]
        den += cell["n"]
    return round(num / den, 4) if den else None


def agg(values: list[float | None]) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "n_seeds": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="logs/results")
    args = parser.parse_args()
    results_dir = Path(args.results)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no result files in {results_dir}")

    table: dict[str, dict] = {}
    for cond in CONDITIONS:
        if cond not in runs:
            continue
        entry: dict = {}
        for lang in LANGS:
            entry[f"refusal_forbidden_{lang}"] = agg(
                [cell_rate(r, lang, "forbidden") for r in runs[cond]])
            entry[f"false_refusal_{lang}"] = agg(
                [cell_rate(r, lang, "benign") for r in runs[cond]])
            entry[f"refusal_forbidden_{lang}_ood"] = agg(
                [cell_rate(r, lang, "forbidden", "ood") for r in runs[cond]])
        fb_en = entry["refusal_forbidden_en"]
        for lang in ("sw", "cs"):
            fb = entry[f"refusal_forbidden_{lang}"]
            entry[f"transfer_gap_{lang}"] = (
                round(fb_en["mean"] - fb["mean"], 4) if fb_en and fb else None)
        table[cond] = entry

    (results_dir / "AGGREGATE.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8")

    # ---- summary.md ----
    lines = ["# Condition sweep; headline results\n",
             "| Condition | seeds | refuse EN | refuse SW | refuse CS | "
             "false-refuse EN | false-refuse SW | gap SW | gap CS |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond, e in table.items():
        def fmt(key):
            v = e.get(key)
            if not v:
                return "–"
            if v["n_seeds"] == 1:
                return f"{v['mean']:.2f}"
            return f"{v['mean']:.2f} [{v['min']:.2f}–{v['max']:.2f}]"
        seeds = e["refusal_forbidden_en"]["n_seeds"] if e.get("refusal_forbidden_en") else 0
        lines.append(f"| {cond} | {seeds} | {fmt('refusal_forbidden_en')} | "
                     f"{fmt('refusal_forbidden_sw')} | {fmt('refusal_forbidden_cs')} | "
                     f"{fmt('false_refusal_en')} | {fmt('false_refusal_sw')} | "
                     f"{e.get('transfer_gap_sw', '–')} | {e.get('transfer_gap_cs', '–')} |")
    lines.append("\nRanges are min–max across seeds (3 seeds cannot support SEM).")
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- figures ----
    fig_dir = Path("logs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in CONDITIONS if c in table]
    x = np.arange(len(conds))
    width = 0.26

    def bars(metric_fmt: str, title: str, fname: str, ylab: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        for i, lang in enumerate(LANGS):
            means, lo_err, hi_err = [], [], []
            for c in conds:
                v = table[c].get(metric_fmt.format(lang=lang))
                means.append(v["mean"] if v else np.nan)
                lo_err.append((v["mean"] - v["min"]) if v else 0)
                hi_err.append((v["max"] - v["mean"]) if v else 0)
            ax.bar(x + (i - 1) * width, means, width, label=lang.upper(),
                   color=COLORS[lang], yerr=[lo_err, hi_err], capsize=3,
                   error_kw={"lw": 1})
        ax.set_xticks(x, [LABELS[c] for c in conds], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / fname)
        plt.close(fig)

    bars("refusal_forbidden_{lang}", "Refusal on forbidden topics by language",
         "transfer_gaps.png", "refusal rate")
    bars("false_refusal_{lang}", "False refusal on benign topics (over-refusal)",
         "false_refusal.png", "false-refusal rate")
    bars("refusal_forbidden_{lang}_ood", "Refusal on OOD forbidden topics (generalisation)",
         "ood_generalisation.png", "refusal rate")
    print(f"\nfigures written to {fig_dir}")


if __name__ == "__main__":
    main()
