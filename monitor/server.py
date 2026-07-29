"""salama-lm live monitor — tiny local dashboard server.

Serves one page (/) and one JSON endpoint (/api/stats) with:
  - GPU: utilization, VRAM, temperature, power (via nvidia-smi)
  - CPU / RAM / disks (via psutil)
  - Training runs: every checkpoints/*/log.jsonl, tailed for history
  - Python worker processes and recent git commits

Stdlib + psutil only; binds 127.0.0.1 (local only, no external exposure).

Run:  .venv\\Scripts\\python.exe monitor\\server.py  [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
INDEX = Path(__file__).with_name("index.html")
ACTIVE_WINDOW_S = 25          # log written within this window => run is live
HISTORY_RECORDS = 200         # max log records returned per run
_procs: dict[int, psutil.Process] = {}   # cached for cpu_percent deltas


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        name, util, mem_used, mem_total, temp, power, clock = \
            [x.strip() for x in out.split(", ")]
        return {"name": name, "util": float(util), "mem_used": float(mem_used),
                "mem_total": float(mem_total), "temp": float(temp),
                "power": float(power), "clock_mhz": float(clock)}
    except Exception as e:  # nvidia-smi missing/hung: report, don't crash
        return {"error": str(e)}


def tail_jsonl(path: Path, max_bytes: int = 96_000) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        if len(chunk) == max_bytes:
            lines = lines[1:]  # first line may be cut mid-record
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records[-HISTORY_RECORDS:]
    except OSError:
        return []


def training_runs() -> list[dict]:
    runs = []
    for log in sorted((REPO / "checkpoints").glob("*/log.jsonl")):
        recs = tail_jsonl(log)
        if not recs:
            continue
        age = time.time() - log.stat().st_mtime
        runs.append({
            "name": log.parent.name,
            "active": age < ACTIVE_WINDOW_S,
            "log_age_s": round(age, 1),
            "latest": recs[-1],
            "history": [
                {k: r.get(k) for k in
                 ("step", "loss", "val_loss_en", "val_loss_sw", "tok_per_s")}
                for r in recs],
        })
    # live runs first, then most recently written
    runs.sort(key=lambda r: (not r["active"], r["log_age_s"]))
    return runs


def worker_processes() -> list[dict]:
    rows = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            if not name.startswith(("python", "nvidia", "uv")):
                continue
            cached = _procs.setdefault(p.info["pid"], p)
            cmd = " ".join(p.info["cmdline"] or [])[-140:]
            rows.append({"pid": p.info["pid"], "name": p.info["name"],
                         "cpu": cached.cpu_percent(None),
                         "rss_mb": round(cached.memory_info().rss / 2**20),
                         "cmd": cmd})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            _procs.pop(p.info["pid"], None)
    rows.sort(key=lambda r: -r["cpu"])
    return rows[:8]


def git_recent() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "--oneline", "-6",
             "--format=%h %ar %s"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()
    except Exception:
        return []


def stats() -> dict:
    vm = psutil.virtual_memory()
    disks = {}
    for drive in ("C:\\", "G:\\"):
        try:
            u = psutil.disk_usage(drive)
            disks[drive[0]] = {"used_gb": round(u.used / 2**30, 1),
                               "total_gb": round(u.total / 2**30, 1),
                               "free_gb": round(u.free / 2**30, 1)}
        except OSError:
            pass
    return {
        "ts": time.time(),
        "gpu": gpu_stats(),
        "cpu": {"percent": psutil.cpu_percent(None),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None},
        "mem": {"used_gb": round(vm.used / 2**30, 1),
                "total_gb": round(vm.total / 2**30, 1),
                "percent": vm.percent},
        "disks": disks,
        "runs": training_runs(),
        "procs": worker_processes(),
        "git": git_recent(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path.startswith("/api/stats"):
            body = json.dumps(stats()).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the console quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    psutil.cpu_percent(None)  # prime the delta-based counter
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"salama-lm monitor: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
