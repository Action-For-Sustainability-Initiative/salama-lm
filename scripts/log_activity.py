"""Append an agent-activity event to monitor-app/activity.jsonl.

This is the instrumentation hook for the live monitor's Agent Activity
panel. The agent (or any script) calls it at stage boundaries:

  python scripts/log_activity.py --stage "pilot pretraining" --status running \
      --command "python -m pretraining.train --config configs/pilot_11m.yaml"
  python scripts/log_activity.py --stage "pilot pretraining" --status done --duration 2712

Secrets are redacted by the monitor at read time as well, but do not pass
them in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ACTIVITY = Path(__file__).resolve().parents[1] / "monitor-app" / "activity.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", default="running",
                        choices=["running", "done", "failed", "info", "warning"])
    parser.add_argument("--command", default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds the stage took (for done/failed)")
    args = parser.parse_args()

    event = {"ts": time.time(), "stage": args.stage, "status": args.status,
             "command": args.command, "message": args.message,
             "exit_code": args.exit_code, "duration_s": args.duration,
             "cwd": os.getcwd()}
    ACTIVITY.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVITY, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in event.items() if v is not None}) + "\n")


if __name__ == "__main__":
    main()
