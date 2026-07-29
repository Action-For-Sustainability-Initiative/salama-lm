export interface GpuInfo {
  source: string;
  error?: string;
  name?: string;
  driver?: string;
  cuda_driver?: string | null;
  util_pct?: number;
  mem_used_mib?: number;
  mem_total_mib?: number;
  temp_c?: number;
  power_w?: number;
  power_limit_w?: number | null;
  clock_sm_mhz?: number;
  fan_pct?: number | null;
  fan_note?: string | null;
  vram_note?: string | null;
  processes: { pid: number; name: string; vram_mib: number | null; kind: string }[];
}

export interface ProcRow {
  pid: number;
  name: string;
  attribution: "agent-tree" | "project" | "other-gpu";
  cpu_pct: number;
  rss_mb: number;
  started: number;
  cwd: string | null;
  cmd: string;
  connections: { total: number; established: number } | null;
}

export interface RunHistoryPoint {
  step: number;
  loss?: number;
  val_loss_en?: number;
  val_loss_sw?: number;
  tok_per_s?: number;
  lr?: number;
  vram_mib?: number;
  gpu_temp_c?: number;
}

export interface Run {
  name: string;
  active: boolean;
  log_age_s: number;
  latest: RunHistoryPoint;
  max_steps: number | null;
  eta_s: number | null;
  latest_checkpoint: string | null;
  checkpoint_mb: number | null;
  history: RunHistoryPoint[];
}

export interface ActivityEvent {
  ts: number;
  stage: string;
  status: string;
  command?: string | null;
  message?: string | null;
  exit_code?: number | null;
  duration_s?: number | null;
  cwd?: string;
}

export interface Sample {
  ts: number;
  session_started: number;
  gpu: GpuInfo;
  system: {
    cpu: {
      percent: number; per_core: number[]; freq_mhz: number | null;
      cores_physical: number; cores_logical: number; process_count: number;
      temp_c: number | null; temp_note: string;
    };
    mem: {
      used_gb: number; avail_gb: number; total_gb: number; percent: number;
      pagefile_used_gb: number; pagefile_total_gb: number; pagefile_pct: number;
    };
    disks: Record<string, { used_gb: number; total_gb: number; free_gb: number; used_pct: number }>;
    io: {
      disk_read_mbps: number | null; disk_write_mbps: number | null;
      net_up_mbps: number | null; net_down_mbps: number | null;
    };
    session: { started: number; net_sent_mb: number; net_recv_mb: number };
  };
  procs: {
    attribution_method: string;
    agent_totals: { cpu_pct: number; rss_mb: number; count: number };
    rows: ProcRow[];
  };
  runs: Run[];
  storage: {
    dirs_mb: Record<string, number>;
    recent_large_files: { path: string; size_mb: number; mtime: number }[];
  };
  activity: ActivityEvent[];
  recorder: { recording: boolean; path: string | null; samples: number };
  config: { refresh_interval_s: number; thresholds: Record<string, [number, number]> };
}

export type Level = "good" | "warn" | "crit";

export function level(value: number | null | undefined, pair?: [number, number]): Level {
  if (value == null || !pair) return "good";
  if (value >= pair[1]) return "crit";
  if (value >= pair[0]) return "warn";
  return "good";
}
