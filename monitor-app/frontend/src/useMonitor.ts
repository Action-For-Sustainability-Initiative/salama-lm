import { useCallback, useEffect, useRef, useState } from "react";
import type { Sample } from "./types";

const HISTORY_MAX = 240; // ~8 min at 2s

export type ConnState = "connecting" | "live" | "paused" | "ended" | "error";

export function useMonitor() {
  const [sample, setSample] = useState<Sample | null>(null);
  const [history, setHistory] = useState<Sample[]>([]);
  const [conn, setConn] = useState<ConnState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const stateRef = useRef<ConnState>("connecting");

  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    setConn("connecting");
    stateRef.current = "connecting";
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const s: Sample = JSON.parse(ev.data);
      setSample(s);
      setHistory((h) => [...h.slice(-HISTORY_MAX + 1), s]);
      setConn("live");
      stateRef.current = "live";
    };
    ws.onclose = () => {
      if (stateRef.current === "live" || stateRef.current === "connecting") {
        setConn("error");
        stateRef.current = "error";
        setTimeout(() => { if (stateRef.current === "error") connect(); }, 3000);
      }
    };
  }, []);

  useEffect(() => { connect(); return () => wsRef.current?.close(); }, [connect]);

  const pause = useCallback(() => {
    stateRef.current = "paused"; setConn("paused"); wsRef.current?.close();
  }, []);
  const resume = connect;
  const end = useCallback(() => {
    stateRef.current = "ended"; setConn("ended");
    wsRef.current?.close(); setHistory([]);
  }, []);
  const setInterval_ = useCallback((s: number) => {
    wsRef.current?.send(JSON.stringify({ interval: s }));
    void fetch("/api/config", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_interval_s: s }),
    });
  }, []);

  return { sample, history, conn, pause, resume, end, setInterval: setInterval_ };
}
