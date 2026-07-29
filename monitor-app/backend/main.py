"""salama-lm monitor v2 — FastAPI + WebSocket backend.

Endpoints:
  GET  /                       built React dashboard (frontend/dist)
  WS   /ws                     metric samples every interval; client may send
                               {"interval": <seconds>} to retune live
  GET  /api/snapshot           one sample (same shape as WS messages)
  GET  /api/config             thresholds + refresh interval
  PUT  /api/config             update thresholds / interval (persisted)
  POST /api/session/start|stop session recording controls
  GET  /api/session/status
  GET  /api/session/export     download the recorded JSONL

Run:  monitor-app\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app
        --app-dir monitor-app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config as config_mod
from backend import recorder
from backend.collectors import activity, gpu, procs, storage, system, training

app = FastAPI(title="salama-lm monitor")
_config = config_mod.load()
_started = time.time()


def snapshot() -> dict:
    gpu_data = gpu.collect()
    runs = training.collect()
    activity.synthesise(gpu_data, runs, _config["thresholds"])
    gpu_pids = {p["pid"] for p in gpu_data.get("processes", [])}
    return {
        "ts": time.time(),
        "session_started": _started,
        "gpu": gpu_data,
        "system": system.collect(),
        "procs": procs.collect(gpu_pids),
        "runs": runs,
        "storage": storage.collect(),
        "activity": activity.collect(),
        "recorder": recorder.status(),
        "config": _config,
    }


@app.websocket("/ws")
async def ws_metrics(ws: WebSocket) -> None:
    await ws.accept()
    interval = float(_config["refresh_interval_s"])
    try:
        while True:
            sample = await asyncio.to_thread(snapshot)
            recorder.record(sample)
            await ws.send_json(sample)
            try:  # non-blocking check for a retune message
                msg = await asyncio.wait_for(ws.receive_json(), timeout=interval)
                if isinstance(msg, dict) and "interval" in msg:
                    interval = min(max(float(msg["interval"]), 0.5), 30.0)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/snapshot")
def api_snapshot() -> dict:
    return snapshot()


@app.get("/api/config")
def get_config() -> dict:
    return _config


@app.put("/api/config")
async def put_config(body: dict) -> dict:
    if "refresh_interval_s" in body:
        _config["refresh_interval_s"] = min(max(float(body["refresh_interval_s"]), 0.5), 30.0)
    for key, pair in (body.get("thresholds") or {}).items():
        if key in _config["thresholds"] and isinstance(pair, list) and len(pair) == 2:
            _config["thresholds"][key] = [float(pair[0]), float(pair[1])]
    config_mod.save(_config)
    return _config


@app.post("/api/session/start")
def session_start() -> dict:
    return recorder.start()


@app.post("/api/session/stop")
def session_stop() -> dict:
    return recorder.stop()


@app.get("/api/session/status")
def session_status() -> dict:
    return recorder.status()


@app.get("/api/session/export")
def session_export():
    path = recorder.latest_session_path()
    if not path:
        return JSONResponse({"error": "no recorded session yet"}, status_code=404)
    return FileResponse(path, filename=path.name, media_type="application/jsonl")


_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
