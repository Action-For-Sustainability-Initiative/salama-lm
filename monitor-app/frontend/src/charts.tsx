import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { Panel } from "./ui";

const AXIS = { stroke: "#6b7488", fontSize: 10 } as const;
const TIP_STYLE = {
  contentStyle: {
    background: "#1a1f2b", border: "1px solid #232937", borderRadius: 8,
    fontSize: 12, color: "#e6eaf2",
  },
  labelStyle: { color: "#97a0b3" },
} as const;

export interface SeriesSpec {
  key: string;
  color: string;
  name: string;
  width?: number;
}

/** Compact real-time line chart. Animation off: data slides, marks don't dance. */
export function MetricChart({ title, data, series, domain, unit, height = 110 }: {
  title: string;
  data: Record<string, number | string | null | undefined>[];
  series: SeriesSpec[];
  domain?: [number | "auto" | "dataMin", number | "auto" | "dataMax"];
  unit?: string;
  height?: number;
}) {
  return (
    <Panel title={title + (unit ? ` (${unit})` : "")}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 6, bottom: 0, left: -14 }}>
          <XAxis dataKey="t" tick={AXIS} tickLine={false} axisLine={{ stroke: "#232937" }}
                 minTickGap={50} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false}
                 domain={domain ?? [0, "auto"]} width={44} />
          <Tooltip {...TIP_STYLE} isAnimationActive={false} />
          {series.length > 1 && (
            <Legend wrapperStyle={{ fontSize: 11, color: "#97a0b3" }} iconType="plainline" />
          )}
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color}
                  strokeWidth={s.width ?? 2} dot={false} isAnimationActive={false}
                  connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  );
}
