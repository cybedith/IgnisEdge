/**
 * useTelemetryWS — LIVE_HARDWARE channel from the LoRa gateway.
 *
 * 🔌 Expected payloads (when wired):
 *   { type: "nodes",   nodes: NodeLocation[] }
 *   { type: "tick",    ... }  (same shape as simulation tick)
 */
import { useEffect, useRef, useState } from "react";
import { ReconnectingWS } from "@/lib/ws";
import { useIgnisStore } from "@/store/useIgnisStore";
import type { NodeLocation, TickMessage } from "@/lib/types";

const WS_BASE =
  (import.meta.env.VITE_WS_BASE_URL as string | undefined) ?? "ws://localhost:8000";

export function useTelemetryWS(enabled: boolean) {
  const setNodes = useIgnisStore((s) => s.setNodes);
  const setLiveTick = useIgnisStore((s) => s.setLiveTick);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<ReconnectingWS | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const ws = new ReconnectingWS(`${WS_BASE}/ws/telemetry`, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (data) => {
        const msg = data as { type?: string; nodes?: NodeLocation[] };
        const t = msg?.type;
        if (t === "nodes" && Array.isArray(msg.nodes)) setNodes(msg.nodes);
        else if (t === "tick") setLiveTick(data as TickMessage);
      },
    });
    wsRef.current = ws;
    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [enabled, setNodes, setLiveTick]);

  return { connected };
}
