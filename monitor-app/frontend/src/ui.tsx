import type { ReactNode } from "react";
import type { Level } from "./types";

export const LEVEL_TEXT: Record<Level, string> = {
  good: "text-good", warn: "text-warn", crit: "text-crit",
};
export const LEVEL_BG: Record<Level, string> = {
  good: "bg-good", warn: "bg-warn", crit: "bg-crit",
};

export function Panel({ title, children, right }: {
  title: string; children: ReactNode; right?: ReactNode;
}) {
  return (
    <section className="bg-surface border border-line rounded-xl p-3 min-w-0">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-[11px] font-semibold tracking-wider uppercase text-muted">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Card({ label, value, unit, sub, lvl = "good", bar }: {
  label: string; value: string; unit?: string; sub?: string; lvl?: Level;
  bar?: { pct: number };
}) {
  return (
    <div className="bg-surface border border-line rounded-xl px-3 py-2.5 min-w-0">
      <div className="text-[10.5px] font-semibold tracking-wider uppercase text-muted truncate">{label}</div>
      <div className={`text-xl font-semibold leading-7 ${lvl === "good" ? "" : LEVEL_TEXT[lvl]}`}>
        {value}{unit && <span className="text-xs font-normal text-muted"> {unit}</span>}
      </div>
      {sub && <div className="text-[11px] text-faint truncate" title={sub}>{sub}</div>}
      {bar && (
        <div className="h-1 mt-1.5 rounded bg-surface2 overflow-hidden">
          <div className={`h-full rounded transition-all duration-500 ${LEVEL_BG[lvl]}`}
               style={{ width: `${Math.min(bar.pct, 100)}%` }} />
        </div>
      )}
    </div>
  );
}

export function Chip({ text, lvl }: { text: string; lvl: Level | "info" }) {
  const cls = lvl === "info"
    ? "text-accent border-accent/40 bg-accent/10"
    : lvl === "good" ? "text-good border-good/40 bg-good/10"
    : lvl === "warn" ? "text-warn border-warn/40 bg-warn/10"
    : "text-crit border-crit/40 bg-crit/10";
  return <span className={`inline-block text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${cls}`}>{text}</span>;
}

export function AttributionChip({ a }: { a: string }) {
  const map: Record<string, [string, string]> = {
    "agent-tree": ["agent", "text-accent border-accent/40 bg-accent/10"],
    "project": ["project", "text-series-sw border-series-sw/40 bg-series-sw/10"],
    "other-gpu": ["other GPU", "text-muted border-line bg-surface2"],
  };
  const [label, cls] = map[a] ?? [a, "text-muted border-line"];
  return <span className={`inline-block text-[10px] font-semibold px-1.5 py-0.5 rounded border ${cls}`}>{label}</span>;
}

export const fmt = (n: number | null | undefined, d = 0) =>
  n == null ? "-" : n.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });

export const ago = (ts: number) => {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${s.toFixed(0)}s ago`;
  if (s < 3600) return `${(s / 60).toFixed(0)}m ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
};

export const dur = (s: number) => {
  if (s < 90) return `${s.toFixed(0)}s`;
  if (s < 5400) return `${(s / 60).toFixed(0)}m`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
};
