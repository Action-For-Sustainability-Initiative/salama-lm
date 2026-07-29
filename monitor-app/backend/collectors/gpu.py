"""NVIDIA GPU collector: NVML primary, nvidia-smi CSV fallback.

Verified on this machine (2026-07-29):
  - NVML works: util, VRAM, temp, power draw, ENFORCED power limit (117.9W —
    note the nvidia-smi header separately shows a 59W state-dependent cap;
    we report the NVML enforced limit and the live draw, and let the data
    speak), SM clock, process list.
  - Fan speed: NVMLError_NotSupported on this laptop -> reported null.
  - Per-process VRAM: unavailable under Windows WDDM ("Insufficient
    Permissions") -> reported null per process, never guessed.
"""

from __future__ import annotations

import subprocess

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _driver = pynvml.nvmlSystemGetDriverVersion()
    _cuda = pynvml.nvmlSystemGetCudaDriverVersion()
    NVML = True
except Exception:  # NVML missing/broken -> nvidia-smi fallback
    NVML = False
    _handle = None
    _driver = _cuda = None


def _nvml_collect() -> dict:
    h = _handle
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
        fan = pynvml.nvmlDeviceGetFanSpeed(h)
    except pynvml.NVMLError:
        fan = None
    procs = []
    seen = set()
    for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                   pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            for p in getter(h):
                if p.pid in seen:
                    continue
                seen.add(p.pid)
                try:
                    name = psutil.Process(p.pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    name = "?"
                vram = getattr(p, "usedGpuMemory", None)
                procs.append({
                    "pid": p.pid, "name": name,
                    "vram_mib": round(vram / 2**20) if vram and vram > 0 else None,
                    "kind": "compute" if getter is pynvml.nvmlDeviceGetComputeRunningProcesses else "graphics",
                })
        except pynvml.NVMLError:
            pass
    return {
        "source": "nvml",
        "name": pynvml.nvmlDeviceGetName(_handle),
        "driver": _driver,
        "cuda_driver": f"{_cuda // 1000}.{_cuda % 1000 // 10}" if _cuda else None,
        "util_pct": util.gpu,
        "mem_used_mib": round(mem.used / 2**20),
        "mem_total_mib": round(mem.total / 2**20),
        "temp_c": pynvml.nvmlDeviceGetTemperature(_handle, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": round(pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000, 1),
        "power_limit_w": round(pynvml.nvmlDeviceGetEnforcedPowerLimit(_handle) / 1000, 1),
        "clock_sm_mhz": pynvml.nvmlDeviceGetClockInfo(_handle, pynvml.NVML_CLOCK_SM),
        "fan_pct": fan,
        "fan_note": None if fan is not None else "not supported on this GPU",
        "vram_note": "per-process VRAM unavailable under Windows WDDM",
        "processes": procs,
    }


def _smi_collect() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,"
         "memory.total,temperature.gpu,power.draw,clocks.sm",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip()
    name, driver, util, used, total, temp, power, clock = \
        [x.strip() for x in out.split(", ")]
    return {"source": "nvidia-smi", "name": name, "driver": driver,
            "cuda_driver": None, "util_pct": float(util),
            "mem_used_mib": float(used), "mem_total_mib": float(total),
            "temp_c": float(temp), "power_w": float(power),
            "power_limit_w": None, "clock_sm_mhz": float(clock),
            "fan_pct": None, "fan_note": "n/a via fallback",
            "vram_note": None, "processes": []}


def collect() -> dict:
    try:
        return _nvml_collect() if NVML else _smi_collect()
    except Exception as e:
        return {"source": "error", "error": f"{type(e).__name__}: {e}"}
