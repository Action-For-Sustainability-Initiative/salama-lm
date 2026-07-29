"""System metrics collector: CPU, memory, swap, disks, disk I/O, network I/O.

Verified on this machine (2026-07-29): per-core CPU (32 threads), cpu_freq,
disk/net IO counters and swap all available; CPU TEMPERATURE IS NOT
(psutil.sensors_temperatures is unavailable on Windows) — we report it as
null with a reason instead of inventing a number.

I/O *speeds* are deltas between successive counter reads divided by elapsed
time, so the collector is stateful; cumulative session transfer is the
difference against the counters captured at server start.
"""

from __future__ import annotations

import time

import psutil

_prev = {"t": None, "disk": None, "net": None}
_session_start_net = psutil.net_io_counters()
_session_start_time = time.time()


def collect() -> dict:
    now = time.time()
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()

    speeds = {"disk_read_mbps": None, "disk_write_mbps": None,
              "net_up_mbps": None, "net_down_mbps": None}
    if _prev["t"] is not None:
        dt = max(now - _prev["t"], 1e-3)
        speeds = {
            "disk_read_mbps": round((disk_io.read_bytes - _prev["disk"].read_bytes) / dt / 2**20, 2),
            "disk_write_mbps": round((disk_io.write_bytes - _prev["disk"].write_bytes) / dt / 2**20, 2),
            "net_up_mbps": round((net_io.bytes_sent - _prev["net"].bytes_sent) / dt / 2**20, 3),
            "net_down_mbps": round((net_io.bytes_recv - _prev["net"].bytes_recv) / dt / 2**20, 3),
        }
    _prev.update(t=now, disk=disk_io, net=net_io)

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    freq = psutil.cpu_freq()
    disks = {}
    for part in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(part.mountpoint)
            disks[part.device[0]] = {
                "used_gb": round(u.used / 2**30, 1),
                "total_gb": round(u.total / 2**30, 1),
                "free_gb": round(u.free / 2**30, 1),
                "used_pct": round(u.percent, 1),
            }
        except OSError:
            pass

    return {
        "cpu": {
            "percent": psutil.cpu_percent(None),
            "per_core": psutil.cpu_percent(None, percpu=True),
            "freq_mhz": round(freq.current) if freq else None,
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "process_count": len(psutil.pids()),
            "temp_c": None,
            "temp_note": "not available via Windows sensor APIs",
        },
        "mem": {
            "used_gb": round(vm.used / 2**30, 2),
            "avail_gb": round(vm.available / 2**30, 2),
            "total_gb": round(vm.total / 2**30, 2),
            "percent": vm.percent,
            "pagefile_used_gb": round(swap.used / 2**30, 2),
            "pagefile_total_gb": round(swap.total / 2**30, 2),
            "pagefile_pct": swap.percent,
        },
        "disks": disks,
        "io": speeds,
        "session": {
            "started": _session_start_time,
            "net_sent_mb": round((net_io.bytes_sent - _session_start_net.bytes_sent) / 2**20, 1),
            "net_recv_mb": round((net_io.bytes_recv - _session_start_net.bytes_recv) / 2**20, 1),
        },
    }
