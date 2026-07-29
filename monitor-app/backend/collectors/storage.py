"""Storage collector: project directory sizes and recent large files.

Directory walks are not free, so results are cached and refreshed at most
once per CACHE_S. "Recent large files" = files >= 25MB modified after the
server started (i.e. created/updated during this monitoring session).
"""

from __future__ import annotations

import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WATCH_DIRS = ("data", "checkpoints", "logs", "pretraining/tokenizer")
LARGE_FILE_MB = 25
CACHE_S = 60

_cache: dict = {"t": 0.0, "value": None}
_session_start = time.time()


def _scan() -> dict:
    dirs = {}
    large_recent = []
    for rel in WATCH_DIRS:
        base = REPO / rel
        if not base.exists():
            continue
        total = 0
        for f in base.rglob("*"):
            try:
                if not f.is_file():
                    continue
                size = f.stat().st_size
                total += size
                if size >= LARGE_FILE_MB * 2**20 and f.stat().st_mtime >= _session_start:
                    large_recent.append({
                        "path": str(f.relative_to(REPO)),
                        "size_mb": round(size / 2**20, 1),
                        "mtime": f.stat().st_mtime,
                    })
            except OSError:
                pass
        dirs[rel] = round(total / 2**20, 1)
    large_recent.sort(key=lambda r: -r["mtime"])
    return {"dirs_mb": dirs, "recent_large_files": large_recent[:12],
            "scanned_at": time.time()}


def collect() -> dict:
    if time.time() - _cache["t"] > CACHE_S or _cache["value"] is None:
        _cache["value"] = _scan()
        _cache["t"] = time.time()
    return _cache["value"]
