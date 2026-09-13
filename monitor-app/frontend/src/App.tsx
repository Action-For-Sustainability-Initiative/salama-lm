import { useMemo, useState } from "react";
import { useMonitor } from "./useMonitor";
import { level } from "./types";
import { Card, Chip, dur, fmt } from "./ui";
import { MetricChart } from "./charts";
import {
  ActivityTimeline, ErrorsPanel, GpuProcTable, ProcessTable,
  SettingsPanel, StoragePanel, TrainingPanel,
} from "./panels";

const ACCENT = "#3e8ee8", SW = "#cc7f2f", GOOD = "#3fb68b", WARN = "#e0b04a";

export default function App() {
  const { sample: s, history, conn, pause, resume, end, setInterval } = useMonitor();
  const [showSettings, setShowSettings] = useState(false);
  const [recording, setRecording] = useState(false);

  const chartData = useMemo(() => history.map((h) => ({
    t: new Date(h.ts * 1000).toLocaleTimeString("en-US", { hour12: false }),
    gpu: h.gpu.util_pct, temp: h.gpu.temp_c, power: h.gpu.power_w,
    vram: h.gpu.mem_used_mib != null ? h.gpu.mem_used_mib / 1024 : null,
    cpu: h.system.cpu.percent, ram: h.system.mem.percent,
    dread: h.system.io.disk_read_mbps, dwrite: h.system.io.disk_write_mbps,
    nup: h.system.io.net_up_mbps, ndown: h.system.io.net_down_mbps,
  })), [history]);

  if (!s) {
    return <div className="min-h-dvh grid place-items-center text-muted text-sm">
      {conn === "error" ? "backend unreachable; retrying…" : "connecting to monitor…"}
    </div>;
  }

  const th = s.config.thresholds;
  const g = s.gpu;
  const vramPct = g.mem_used_mib && g.mem_total_mib ? (g.mem_used_mib / g.mem_total_mib) * 100 : null;
  const record = async (start: boolean) => {
    await fetch(`/api/session/${start ? "start" : "stop"}`, { method: "POST" });
    setRecording(start);
  };

  return (
    <div className="max-w-[1400px] mx-auto p-3 sm:p-4 font-sans text-ink">
      {/* top status bar */}
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-3 bg-surface border border-line rounded-xl px-4 py-2.5">
        <h1 className="text-[15px] font-semibold">salama-lm <span className="text-faint font-normal">monitor</span></h1>
        <Chip text={conn === "live" ? "live" : conn} lvl={conn === "live" ? "good" : conn === "error" ? "crit" : "info"} />
        <span className="text-[12px] text-muted">GPU <b className="text-ink">{fmt(g.util_pct)}%</b></span>
        <span className="text-[12px] text-muted">VRAM <b className="text-ink">{vramPct ? fmt(vramPct) : "-"}%</b></span>
        <span className="text-[12px] text-muted">CPU <b className="text-ink">{fmt(s.system.cpu.percent)}%</b></span>
        <span className="text-[12px] text-muted">RAM <b className="text-ink">{fmt(s.system.mem.percent)}%</b></span>
        <span className="text-[12px] text-muted">session <b className="text-ink">{dur(s.ts - s.session_started)}</b></span>
        <div className="flex items-center gap-1.5 ml-auto">
          <select aria-label="Refresh interval" defaultValue={s.config.refresh_interval_s}
            onChange={(e) => setInterval(Number(e.target.value))}
            className="bg-surface2 border border-line rounded-lg text-[12px] px-2 py-1.5 cursor-pointer">
            {[1, 2, 5, 10].map((n) => <option key={n} value={n}>{n}s</option>)}
          </select>
          {conn === "live"
            ? <Btn onClick={pause}>Pause</Btn>
            : <Btn onClick={resume} accent>Start</Btn>}
          <Btn onClick={end}>End</Btn>
          <Btn onClick={() => record(!recording)} accent={recording}>
            {recording ? `Rec ● ${s.recorder.samples}` : "Record"}
          </Btn>
          <a href="/api/session/export" download
            className="text-[12px] font-medium border border-line rounded-lg px-2.5 py-1.5 hover:bg-surface2">Export</a>
          <Btn onClick={() => setShowSettings(!showSettings)}>Thresholds</Btn>
        </div>
      </header>

      {showSettings && <div className="mb-3"><SettingsPanel s={s} onClose={() => setShowSettings(false)} /></div>}

      {/* metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mb-3">
        <Card label="GPU util" value={fmt(g.util_pct)} unit="%"
          sub={`${g.name?.replace("NVIDIA GeForce ", "") ?? ""} · ${g.source}`} />
        <Card label="VRAM" value={g.mem_used_mib ? fmt(g.mem_used_mib / 1024, 1) : "-"}
          unit={`/ ${g.mem_total_mib ? fmt(g.mem_total_mib / 1024, 1) : "-"} GB`}
          lvl={level(vramPct, th.gpu_vram_pct)} bar={vramPct != null ? { pct: vramPct } : undefined} />
        <Card label="GPU temp" value={fmt(g.temp_c)} unit="°C"
          lvl={level(g.temp_c, th.gpu_temp_c)}
          sub={g.fan_pct != null ? `fan ${g.fan_pct}%` : g.fan_note ?? undefined} />
        <Card label="GPU power" value={fmt(g.power_w, 1)} unit="W"
          sub={`limit ${fmt(g.power_limit_w, 0)} W · ${fmt(g.clock_sm_mhz)} MHz`} />
        <Card label="CPU" value={fmt(s.system.cpu.percent)} unit="%"
          lvl={level(s.system.cpu.percent, th.cpu_pct)}
          sub={`${s.system.cpu.cores_physical}c/${s.system.cpu.cores_logical}t · ${fmt(s.system.cpu.freq_mhz)} MHz · ${s.system.cpu.process_count} procs`} />
        <Card label="RAM" value={fmt(s.system.mem.used_gb, 1)}
          unit={`/ ${fmt(s.system.mem.total_gb, 0)} GB`}
          lvl={level(s.system.mem.percent, th.ram_pct)} bar={{ pct: s.system.mem.percent }}
          sub={`pagefile ${fmt(s.system.mem.pagefile_used_gb, 1)}/${fmt(s.system.mem.pagefile_total_gb, 0)} GB`} />
      </div>

      {/* disks + network cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        {Object.entries(s.system.disks).map(([name, d]) => (
          <Card key={name} label={`Disk ${name}:`} value={fmt(d.free_gb)} unit="GB free"
            lvl={level(d.used_pct, th.disk_used_pct)} bar={{ pct: d.used_pct }} />
        ))}
        <Card label="Network session" value={`↓${fmt(s.system.session.net_recv_mb)}`}
          unit={`↑${fmt(s.system.session.net_sent_mb)} MB`}
          sub="cumulative since monitor start" />
      </div>

      {/* charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mb-3">
        <MetricChart title="GPU utilisation" unit="%" data={chartData} domain={[0, 100]}
          series={[{ key: "gpu", color: ACCENT, name: "GPU %" }]} />
        <MetricChart title="GPU temp & power" data={chartData}
          series={[{ key: "temp", color: WARN, name: "°C" },
                   { key: "power", color: ACCENT, name: "W" }]} />
        <MetricChart title="VRAM" unit="GB" data={chartData}
          domain={[0, g.mem_total_mib ? g.mem_total_mib / 1024 : "auto"]}
          series={[{ key: "vram", color: SW, name: "used GB" }]} />
        <MetricChart title="CPU & RAM" unit="%" data={chartData} domain={[0, 100]}
          series={[{ key: "cpu", color: ACCENT, name: "CPU %" },
                   { key: "ram", color: GOOD, name: "RAM %" }]} />
        <MetricChart title="Disk I/O" unit="MB/s" data={chartData}
          series={[{ key: "dread", color: ACCENT, name: "read" },
                   { key: "dwrite", color: SW, name: "write" }]} />
        <MetricChart title="Network" unit="MB/s" data={chartData}
          series={[{ key: "ndown", color: ACCENT, name: "down" },
                   { key: "nup", color: GOOD, name: "up" }]} />
      </div>

      {/* training + activity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 mb-3">
        {s.runs.map((r) => <TrainingPanel key={r.name} run={r} />)}
        <ActivityTimeline events={s.activity} />
      </div>

      {/* tables + storage + errors */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2 space-y-3">
          <ProcessTable s={s} />
          <GpuProcTable s={s} />
        </div>
        <div className="space-y-3">
          <ErrorsPanel events={s.activity} />
          <StoragePanel s={s} />
        </div>
      </div>

      <footer className="text-[10.5px] text-faint mt-4 pb-2">
        CPU temp: {s.system.cpu.temp_note}. Attribution: {s.procs.attribution_method}.
        Driver {g.driver} · CUDA {g.cuda_driver ?? "n/a"} · secrets redacted at source.
      </footer>
    </div>
  );
}

function Btn({ children, onClick, accent }: {
  children: React.ReactNode; onClick: () => void; accent?: boolean;
}) {
  return (
    <button onClick={onClick}
      className={`text-[12px] font-medium rounded-lg px-2.5 py-1.5 border cursor-pointer transition-colors ${
        accent ? "bg-accent/15 text-accent border-accent/40 hover:bg-accent/25"
               : "border-line hover:bg-surface2"}`}>
      {children}
    </button>
  );
}
