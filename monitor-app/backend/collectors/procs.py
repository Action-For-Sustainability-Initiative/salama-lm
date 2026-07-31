"""Process-tree monitor with honest attribution.

The spec requires distinguishing (1) whole-system usage, (2) the agent and
its children, (3) the training process, (4) other GPU users. Attribution
rules, strictest first:

  - "agent-tree": the process is a VERIFIED descendant of the agent host
    process (we walk up from this server's own PID to the outermost
    claude/node/electron ancestor, then mark all its descendants). If no
    such ancestor exists we say so instead of guessing.
  - "project": the process command line references this repository path
    (weaker evidence; labelled as such in the UI).
  - "other": everything else that appears in the GPU process list.

CPU% for cached processes is delta-based (first observation reads 0.0).
Command lines pass through the redaction utility before leaving this module.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil

from backend.redaction import redact

REPO = str(Path(__file__).resolve().parents[3])
_AGENT_NAMES = ("claude", "node", "electron", "code")
_cache: dict[int, psutil.Process] = {}


def _agent_root() -> psutil.Process | None:
    """Walk up from this server to the outermost agent-looking ancestor."""
    try:
        me = psutil.Process(os.getpid())
        root = None
        for anc in me.parents():
            if anc.name().lower().split(".")[0].rstrip("0123456789") in _AGENT_NAMES \
               or anc.name().lower().startswith(_AGENT_NAMES):
                root = anc
        return root
    except psutil.Error:
        return None


def _row(p: psutil.Process, attribution: str) -> dict | None:
    try:
        cached = _cache.setdefault(p.pid, p)
        with cached.oneshot():
            return {
                "pid": p.pid,
                "name": cached.name(),
                "attribution": attribution,
                "cpu_pct": round(cached.cpu_percent(None), 1),
                "rss_mb": round(cached.memory_info().rss / 2**20),
                "started": cached.create_time(),
                "cwd": None,   # filled only for agent-tree rows (cheap + relevant)
                "cmd": redact(" ".join(cached.cmdline() or [])[-180:]),
                "connections": None,
            }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        _cache.pop(p.pid, None)
        return None


def collect(gpu_pids: set[int]) -> dict:
    root = _agent_root()
    agent_pids: set[int] = set()
    rows: list[dict] = []

    if root is not None:
        try:
            tree = [root] + root.children(recursive=True)
            agent_pids = {p.pid for p in tree}
            for p in tree:
                r = _row(p, "agent-tree")
                if r is None:
                    continue
                # connections: count only, plus remote endpoints for the tree,
                # never payloads; suppressed entirely on AccessDenied
                try:
                    conns = _cache[p.pid].net_connections(kind="inet")
                    r["connections"] = {
                        "total": len(conns),
                        "established": sum(1 for c in conns if c.status == "ESTABLISHED"),
                    }
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                try:
                    r["cwd"] = _cache[p.pid].cwd()
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                    pass
                rows.append(r)
        except psutil.Error:
            pass

    # project processes (cmdline mentions the repo) not already in the tree
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] in agent_pids:
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if REPO.lower() in cmd.lower():
            r = _row(p, "project")
            if r:
                rows.append(r)

    # other GPU users not covered above
    covered = {r["pid"] for r in rows}
    for pid in gpu_pids - covered:
        try:
            r = _row(psutil.Process(pid), "other-gpu")
            if r:
                rows.append(r)
        except psutil.NoSuchProcess:
            pass

    rows.sort(key=lambda r: (-r["cpu_pct"], -r["rss_mb"]))
    agent_rows = [r for r in rows if r["attribution"] == "agent-tree"]
    return {
        "attribution_method": ("verified process tree rooted at "
                               f"{root.name()} (pid {root.pid})") if root
                              else "agent ancestor not found; repo-path matching only",
        "agent_totals": {
            "cpu_pct": round(sum(r["cpu_pct"] for r in agent_rows), 1),
            "rss_mb": sum(r["rss_mb"] for r in agent_rows),
            "count": len(agent_rows),
        },
        "rows": rows[:24],
    }
