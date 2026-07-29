"""Session recorder: append broadcast samples to a JSONL file on demand.

start() opens logs/monitor-sessions/session-<ts>.jsonl; every sample passed
to record() is appended while recording. export() returns the current (or
most recent) session file for download. Samples are already redacted before
they reach this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "logs" / "monitor-sessions"

_state = {"file": None, "path": None, "count": 0}


def start() -> dict:
    stop()
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    _state.update(file=open(path, "a", encoding="utf-8"), path=path, count=0)
    return status()


def stop() -> dict:
    if _state["file"]:
        _state["file"].close()
        _state["file"] = None
    return status()


def record(sample: dict) -> None:
    if _state["file"]:
        _state["file"].write(json.dumps(sample) + "\n")
        _state["file"].flush()
        _state["count"] += 1


def status() -> dict:
    return {"recording": _state["file"] is not None,
            "path": str(_state["path"]) if _state["path"] else None,
            "samples": _state["count"]}


def latest_session_path() -> Path | None:
    if _state["path"] and Path(_state["path"]).exists():
        return Path(_state["path"])
    if SESSIONS_DIR.exists():
        files = sorted(SESSIONS_DIR.glob("session-*.jsonl"))
        return files[-1] if files else None
    return None
