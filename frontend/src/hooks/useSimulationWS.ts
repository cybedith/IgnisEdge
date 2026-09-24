/**
 * useSimulationWS — connects to the backend's per-session WS.
 * Every "tick" message is pushed into the store.
 *
 * In Lovable preview (no backend) the connection will keep failing silently
 * and connected stays false. That's intentional.
 */
import { useEffect, useRef, useState } from "react";
import { ReconnectingWS } from "@/lib/ws";
import { useIgnisStore } from "@/store/useIgnisStore";
import type { TickMessage } from "@/lib/types";

const WS_BASE =
  (import.meta.env.VITE_WS_BASE_URL as string | undefined) ?? "ws://localhost:8000";

export function useSimulationWS(sessionId: string | null | undefined) {
  const setLiveTick = useIgnisStore((s) => s.setLiveTick);
  const [connected, setConnected] = useState(false);
  const [lastTickAt, setLastTickAt] = useState<number | null>(null);
  const wsRef = useRef<ReconnectingWS | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    const ws = new ReconnectingWS(`${WS_BASE}/ws/simulation/${sessionId}`, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (data) => {
        const msg = data as Partial<TickMessage> | undefined;
        if (msg?.type === "tick") {
          setLiveTick(msg as TickMessage);
          setLastTickAt(Date.now());
        }
      },
    });
    wsRef.current = ws;
    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [sessionId, setLiveTick]);

  return { connected, lastTickAt };
}
