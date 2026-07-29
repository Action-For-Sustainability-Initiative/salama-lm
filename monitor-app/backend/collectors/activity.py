"""Agent-activity event collector.

Real instrumentation, not OS-metric guesswork: the agent (and any project
script) appends JSON events to monitor-app/activity.jsonl via
scripts/log_activity.py. Each event:
  {"ts": ..., "stage": ..., "command": ..., "status": "running|done|failed",
   "exit_code": ..., "cwd": ..., "duration_s": ...}
This module tails that file (through redaction) and synthesises additional
warning events from live metrics (VRAM near spillover, thermals, run
completion) so the Errors & Warnings panel has one merged feed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.redaction import redact_obj

ACTIVITY_FILE = Path(__file__).resolve().parents[2] / "activity.jsonl"
MAX_EVENTS = 60

_warn_state: dict[str, float] = {}   # dedupe: warning key -> last emitted ts
_synth: list[dict] = []


def _emit_warning(key: str, level: str, message: str, every_s: float = 120) -> None:
    now = time.time()
    if now - _warn_state.get(key, 0) < every_s:
        return
    _warn_state[key] = now
    _synth.append({"ts": now, "stage": "monitor", "status": level,
                   "command": None, "message": message})
    del _synth[:-MAX_EVENTS]


def synthesise(gpu: dict, runs: list[dict], thresholds: dict) -> None:
    if gpu.get("mem_used_mib") and gpu.get("mem_total_mib"):
        pct = gpu["mem_used_mib"] / gpu["mem_total_mib"] * 100
        warn, crit = thresholds.get("gpu_vram_pct", [80, 93])
        if pct >= crit:
            _emit_warning("vram", "warning",
                          f"VRAM {pct:.0f}% — sysmem spillover risk (throughput will degrade)")
    if gpu.get("temp_c") is not None:
        warn, crit = thresholds.get("gpu_temp_c", [80, 87])
        if gpu["temp_c"] >= crit:
            _emit_warning("temp", "critical", f"GPU {gpu['temp_c']}°C — check airflow")
        elif gpu["temp_c"] >= warn:
            _emit_warning("temp", "warning", f"GPU {gpu['temp_c']}°C — running hot")
    for r in runs:
        key = f"run-{r['name']}"
        if r["active"]:
            _warn_state[f"{key}-was-active"] = 1
        elif _warn_state.pop(f"{key}-was-active", None):
            _emit_warning(key, "info", f"training run '{r['name']}' stopped writing logs "
                          f"(last step {r['latest'].get('step')})", every_s=0)


def collect() -> list[dict]:
    events: list[dict] = list(_synth)
    if ACTIVITY_FILE.exists():
        try:
            with open(ACTIVITY_FILE, "rb") as f:
                f.seek(max(0, ACTIVITY_FILE.stat().st_size - 64_000))
                for line in f.read().decode("utf-8", errors="replace").splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
    events.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return redact_obj(events[:MAX_EVENTS])
