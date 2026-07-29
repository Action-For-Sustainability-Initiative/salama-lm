"""Config: thresholds and refresh interval, persisted to config.json."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

DEFAULTS = {
    "refresh_interval_s": 2.0,
    "thresholds": {
        # metric: [warn, critical]
        "gpu_temp_c": [80, 87],
        "gpu_vram_pct": [80, 93],
        "gpu_util_pct": [101, 101],      # utilisation alone is not a problem
        "cpu_pct": [85, 95],
        "ram_pct": [80, 92],
        "disk_used_pct": [85, 94],
        "pagefile_pct": [60, 85],
    },
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg["refresh_interval_s"] = float(saved.get("refresh_interval_s",
                                                        cfg["refresh_interval_s"]))
            cfg["thresholds"].update(saved.get("thresholds", {}))
        except (json.JSONDecodeError, ValueError):
            pass
    return cfg


def save(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
