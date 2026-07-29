import { useState } from "react";
import type { ActivityEvent, Run, Sample } from "./types";
import { level } from "./types";
import { AttributionChip, Chip, Panel, ago, dur, fmt } from "./ui";
import { MetricChart } from "./charts";

export function ProcessTable({ s }: { s: Sample }) {
  return (
    <Panel title="Processes"
      right={<span className="text-[10.5px] text-faint truncate max-w-64"
        title={s.procs.attribution_method}>{s.procs.attribution_method}</span>}>
      <div className="text-[11px] text-muted mb-1.5">
        agent tree: {s.procs.agent_totals.count} procs ·
        CPU {fmt(s.procs.agent_totals.cpu_pct, 1)}% ·
        RAM {fmt(s.procs.agent_totals.rss_mb / 1024, 1)} GB
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead><tr className="text-left text-[10px] uppercase tracking-wider text-muted">
            <th className="py-1 pr-2">PID</th><th className="pr-2">Name</th>
            <th className="pr-2">Scope</th>
            <th className="pr-2 text-right">CPU%</th><th className="pr-2 text-right">RAM MB</th>
            <th className="pr-2 text-right">Conns</th><th>Command</th>
          </tr></thead>
          <tbody>
            {s.procs.rows.slice(0, 10).map((p) => (
              <tr key={p.pid} className="border-t border-line">
                <td className="py-1 pr-2 text-faint">{p.pid}</td>
                <td className="pr-2">{p.name}</td>
                <td className="pr-2"><AttributionChip a={p.attribution} /></td>
                <td className="pr-2 text-right">{fmt(p.cpu_pct, 1)}</td>
                <td className="pr-2 text-right">{fmt(p.rss_mb)}</td>
                <td className="pr-2 text-right text-faint">
                  {p.connections ? `${p.connections.established}/${p.connections.total}` : "–"}
                </td>
                <td className="text-faint max-w-56 truncate" title={p.cmd}
                    style={{ direction: "rtl" }}>{p.cmd}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export function GpuProcTable({ s }: { s: Sample }) {
  const g = s.gpu;
  return (
    <Panel title="GPU processes"
      right={g.vram_note ? <span className="text-[10.5px] text-faint">{g.vram_note}</span> : undefined}>
      <table className="w-full text-[12px]">
        <thead><tr className="text-left text-[10px] uppercase tracking-wider text-muted">
          <th className="py-1 pr-2">PID</th><th className="pr-2">Name</th>
          <th className="pr-2">Kind</th><th className="text-right">VRAM MiB</th>
        </tr></thead>
        <tbody>
          {g.processes.map((p) => (
            <tr key={p.pid} className="border-t border-line">
              <td className="py-1 pr-2 text-faint">{p.pid}</td>
              <td className="pr-2">{p.name}</td>
              <td className="pr-2 text-faint">{p.kind}</td>
              <td className="text-right text-faint">{p.vram_mib ?? "n/a"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

const STATUS_LVL: Record<string, "good" | "warn" | "crit" | "info"> = {
  done: "good", running: "info", info: "info",
  warning: "warn", failed: "crit", critical: "crit",
};

export function ActivityTimeline({ events }: { events: ActivityEvent[] }) {
  return (
    <Panel title="Agent activity">
      <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
        {events.length === 0 && <div className="text-[12px] text-faint">no events logged yet</div>}
        {events.map((e, i) => (
          <div key={i} className="flex items-start gap-2 text-[12px] border-t border-line first:border-0 pt-1.5 first:pt-0">
            <Chip text={e.status} lvl={STATUS_LVL[e.status] ?? "info"} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="font-medium">{e.stage}</span>
                <span className="text-[10.5px] text-faint shrink-0">{ago(e.ts)}</span>
                {e.duration_s != null && <span className="text-[10.5px] text-faint shrink-0">{dur(e.duration_s)}</span>}
                {e.exit_code != null && <span className="text-[10.5px] text-faint shrink-0">exit {e.exit_code}</span>}
              </div>
              {e.command && <div className="text-faint font-mono text-[11px] truncate" title={e.command}>{e.command}</div>}
              {e.message && <div className="text-muted text-[11px]">{e.message}</div>}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function ErrorsPanel({ events }: { events: ActivityEvent[] }) {
  const bad = events.filter((e) => ["warning", "failed", "critical"].includes(e.status));
  return (
    <Panel title="Errors & warnings">
      {bad.length === 0
        ? <div className="text-[12px] text-good">no warnings in this session</div>
        : <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {bad.map((e, i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <Chip text={e.status} lvl={STATUS_LVL[e.status] ?? "warn"} />
                <div className="min-w-0">
                  <span className="text-muted">{e.message ?? e.command ?? e.stage}</span>
                  <span className="text-[10.5px] text-faint ml-2">{ago(e.ts)}</span>
                </div>
              </div>
            ))}
          </div>}
    </Panel>
  );
}

export function TrainingPanel({ run }: { run: Run }) {
  const L = run.latest;
  const pct = run.max_steps ? (L.step / run.max_steps) * 100 : null;
  const chartData = run.history.map((h) => ({ t: h.step, ...h }));
  return (
    <Panel title={`Training — ${run.name}`}
      right={<Chip text={run.active ? "TRAINING" : "IDLE"} lvl={run.active ? "good" : "info"} />}>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12px] mb-2">
        <span><b className="text-base font-semibold">{fmt(L.step)}</b>
          <span className="text-faint">{run.max_steps ? ` / ${fmt(run.max_steps)}` : ""} steps</span></span>
        <span><b className="text-base font-semibold">{fmt(L.loss, 3)}</b> <span className="text-faint">loss</span></span>
        <span><b className="text-base font-semibold">{fmt(L.tok_per_s)}</b> <span className="text-faint">tok/s</span></span>
        <span><b className="text-base font-semibold">{L.lr ? L.lr.toExponential(1) : "–"}</b> <span className="text-faint">lr</span></span>
        {run.active && run.eta_s != null &&
          <span><b className="text-base font-semibold">{dur(run.eta_s)}</b> <span className="text-faint">est. left</span></span>}
      </div>
      {pct != null && (
        <div className="h-1.5 rounded bg-surface2 overflow-hidden mb-2">
          <div className="h-full rounded bg-accent transition-all duration-500" style={{ width: `${pct}%` }} />
        </div>
      )}
      <MetricChart title="loss (train / val EN / val SW)" height={130}
        data={chartData}
        series={[
          { key: "loss", color: "#6b7488", name: "train", width: 1.5 },
          { key: "val_loss_en", color: "#3e8ee8", name: "val EN", width: 2.5 },
          { key: "val_loss_sw", color: "#cc7f2f", name: "val SW", width: 2.5 },
        ]} />
      <div className="text-[11px] text-faint mt-1.5">
        checkpoint: {run.latest_checkpoint ?? "none"}
        {run.checkpoint_mb ? ` (${fmt(run.checkpoint_mb)} MB)` : ""} ·
        VRAM {fmt(L.vram_mib)} MiB · GPU {fmt(L.gpu_temp_c)}°C
        {!run.active && ` · last log ${ago(Date.now() / 1000 - run.log_age_s)}`}
      </div>
    </Panel>
  );
}

export function StoragePanel({ s }: { s: Sample }) {
  const dirs = Object.entries(s.storage.dirs_mb);
  const max = Math.max(...dirs.map(([, v]) => v), 1);
  return (
    <Panel title="Project storage">
      <div className="space-y-1.5">
        {dirs.map(([name, mb]) => (
          <div key={name} className="text-[12px]">
            <div className="flex justify-between">
              <span className="text-muted">{name}/</span>
              <span>{mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${fmt(mb)} MB`}</span>
            </div>
            <div className="h-1 rounded bg-surface2 overflow-hidden">
              <div className="h-full bg-accent/70 rounded" style={{ width: `${(mb / max) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      {s.storage.recent_large_files.length > 0 && <>
        <div className="text-[10px] uppercase tracking-wider text-muted mt-3 mb-1">new large files (this session)</div>
        {s.storage.recent_large_files.slice(0, 5).map((f) => (
          <div key={f.path} className="flex justify-between text-[11.5px] text-faint">
            <span className="truncate mr-2" title={f.path}>{f.path}</span>
            <span className="shrink-0">{fmt(f.size_mb)} MB</span>
          </div>
        ))}
      </>}
    </Panel>
  );
}

export function SettingsPanel({ s, onClose }: { s: Sample; onClose: () => void }) {
  const [t, setT] = useState(s.config.thresholds);
  const save = async () => {
    await fetch("/api/config", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thresholds: t }),
    });
    onClose();
  };
  return (
    <Panel title="Thresholds (warn / critical)"
      right={<button onClick={onClose} className="text-[11px] text-faint hover:text-ink cursor-pointer">close</button>}>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {Object.entries(t).map(([key, pair]) => (
          <label key={key} className="text-[11px] text-muted">
            {key.replace(/_/g, " ")}
            <div className="flex gap-1 mt-0.5">
              {[0, 1].map((i) => (
                <input key={i} type="number" value={pair[i]}
                  aria-label={`${key} ${i === 0 ? "warn" : "critical"}`}
                  onChange={(e) => setT({ ...t, [key]: pair.map((v, j) =>
                    j === i ? Number(e.target.value) : v) as [number, number] })}
                  className="w-full bg-surface2 border border-line rounded px-1.5 py-1 text-[12px] text-ink" />
              ))}
            </div>
          </label>
        ))}
      </div>
      <button onClick={save}
        className="mt-3 text-[12px] font-semibold bg-accent/15 text-accent border border-accent/40 rounded-lg px-3 py-1.5 cursor-pointer hover:bg-accent/25">
        Save thresholds
      </button>
    </Panel>
  );
}

export { level };
