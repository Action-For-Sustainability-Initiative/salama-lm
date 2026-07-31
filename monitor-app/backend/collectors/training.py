"""Training-log parser: tails checkpoints/*/log.jsonl, computes ETA.

max_steps comes from the run's YAML config (matched by out_dir), so the ETA
is (remaining steps x recent seconds/step), reported with the basis shown.
An ETA is only a projection; the UI labels it "est." for that reason.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
ACTIVE_WINDOW_S = 25
HISTORY = 200

_max_steps: dict[str, int] = {}


def _load_max_steps() -> None:
    for cfg_path in (REPO / "configs").glob("*.yaml"):
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            out_dir = cfg.get("out_dir", "")
            if out_dir:
                _max_steps[Path(out_dir).name] = cfg["train"]["max_steps"]
        except Exception:
            pass


_load_max_steps()


def _tail(path: Path, max_bytes: int = 96_000) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - max_bytes))
            lines = f.read().decode("utf-8", errors="replace").splitlines()[1:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out[-HISTORY:]
    except OSError:
        return []


def collect() -> list[dict]:
    runs = []
    for log in sorted((REPO / "checkpoints").glob("*/log.jsonl")):
        recs = _tail(log)
        if not recs:
            continue
        name = log.parent.name
        latest = recs[-1]
        age = time.time() - log.stat().st_mtime
        active = age < ACTIVE_WINDOW_S
        max_steps = _max_steps.get(name)

        eta_s = None
        if active and max_steps and len(recs) >= 2:
            span = recs[-1]["step"] - recs[-10 if len(recs) >= 10 else 0]["step"]
            # log records are written every ~20 steps; estimate sec/step from
            # token throughput instead of wall-clock we don't have per record
            tps = latest.get("tok_per_s") or 0
            if span > 0 and tps > 0 and "loss" in latest:
                # tokens/step inferred from config-free data is fragile; use
                # throughput-based projection only when config is known
                cfg_tokens = _cfg_tokens_per_step(name)
                if cfg_tokens:
                    eta_s = (max_steps - latest["step"]) * cfg_tokens / tps

        ckpts = sorted(log.parent.glob("*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        runs.append({
            "name": name,
            "active": active,
            "log_age_s": round(age, 1),
            "latest": latest,
            "max_steps": max_steps,
            "eta_s": round(eta_s) if eta_s else None,
            "latest_checkpoint": ckpts[0].name if ckpts else None,
            "checkpoint_mb": round(ckpts[0].stat().st_size / 2**20, 1) if ckpts else None,
            "history": [{k: r.get(k) for k in
                         ("step", "loss", "val_loss_en", "val_loss_sw",
                          "tok_per_s", "lr", "vram_mib", "gpu_temp_c")}
                        for r in recs],
        })
    runs.sort(key=lambda r: (not r["active"], r["log_age_s"]))
    return runs


def _cfg_tokens_per_step(run_name: str) -> int | None:
    for cfg_path in (REPO / "configs").glob("*.yaml"):
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if Path(cfg.get("out_dir", "")).name == run_name:
                t = cfg["train"]
                return t["micro_batch_size"] * t["grad_accum"] * cfg["model"]["n_ctx"]
        except Exception:
            pass
    return None
