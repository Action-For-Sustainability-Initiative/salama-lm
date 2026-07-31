"""Run the full experiment: 4 alignment conditions x N seeds, + base control.

For each (condition, seed): SFT from the shared pretrained base, then
evaluate on the full grid. Results land in logs/results/<cond>_s<seed>.json.
The base condition is evaluated directly (no SFT) as the control.

Everything is resumable; existing result files are skipped; so an
interrupted sweep continues where it stopped.

Usage:
  python scripts/run_conditions.py --base checkpoints/primary_48m/final.pt \
      --seeds 1234 2345 3456
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = ["en_outcome", "en_process", "bi_outcome", "bi_process"]
PY = sys.executable


def run(cmd: list[str], stage: str) -> None:
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([PY, "scripts/log_activity.py", "--stage", stage,
                    "--status", "running", "--command", " ".join(cmd[1:4])], check=False)
    t0 = time.time()
    result = subprocess.run(cmd)
    subprocess.run([PY, "scripts/log_activity.py", "--stage", stage,
                    "--status", "done" if result.returncode == 0 else "failed",
                    "--exit-code", str(result.returncode),
                    "--duration", str(round(time.time() - t0, 1))], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="pretrained base checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--tokenizer", default="pretraining/tokenizer_full/tokenizer.json")
    parser.add_argument("--grid", default="data/alignment/eval_grid.jsonl")
    args = parser.parse_args()

    results = Path("logs/results")
    results.mkdir(parents=True, exist_ok=True)

    # control: the un-aligned base model
    base_out = results / "base.json"
    if not base_out.exists():
        run([PY, "-m", "evaluation.eval_conditions", "--ckpt", args.base,
             "--grid", args.grid, "--tokenizer", args.tokenizer,
             "--out", str(base_out)], "eval base (control)")

    for seed in args.seeds:
        for cond in CONDITIONS:
            out = results / f"{cond}_s{seed}.json"
            if out.exists():
                print(f"[skip] {cond} seed {seed}: already done")
                continue
            ckpt_dir = Path(f"checkpoints/cond_{cond}_s{seed}")
            if not (ckpt_dir / "final.pt").exists():
                run([PY, "-m", "alignment.sft", "--base", args.base,
                     "--data", f"data/alignment/{cond}.jsonl",
                     "--out", str(ckpt_dir), "--tokenizer", args.tokenizer,
                     "--seed", str(seed)], f"SFT {cond} seed {seed}")
            run([PY, "-m", "evaluation.eval_conditions",
                 "--ckpt", str(ckpt_dir / "final.pt"), "--grid", args.grid,
                 "--tokenizer", args.tokenizer, "--out", str(out)],
                f"eval {cond} seed {seed}")

    print("\nAll conditions complete. Aggregate with: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
