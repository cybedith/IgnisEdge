import React, { useEffect, useRef, useState } from "react";
import { useIgnisStore } from "@/store/useIgnisStore";
import { SystemLogMessage } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TerminalSquare, PauseCircle, PlayCircle, Filter } from "lucide-react";
import { cn } from "@/lib/utils";

const SOURCE_COLORS: Record<SystemLogMessage["source"], string> = {
  LORA: "text-cyan-400",
  PIXHAWK: "text-emerald-400",
  JETSON: "text-amber-500",
  PERCEPTION: "text-purple-400",
};

export function SystemLogsPanel() {
  const [open, setOpen] = useState(false);
  const liveTick = useIgnisStore((s) => s.liveTick);
  const [logs, setLogs] = useState<SystemLogMessage[]>([]);
  const [filter, setFilter] = useState<"ALL" | SystemLogMessage["source"]>("ALL");
  const [autoScroll, setAutoScroll] = useState(true);

  const bottomRef = useRef<HTMLDivElement>(null);
  const seenEventsRef = useRef<Set<string>>(new Set());

  // 1. Logs reales desde el endpoint de estado de la Jetson
  useEffect(() => {
    let active = true;
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://192.168.0.9:8080/telemetry.json");
        if (!res.ok) return;
        const d = await res.json();
        if (!active) return;

        const incomingEvents: string[] = d.radio_logs || [];
        const newLogs: SystemLogMessage[] = [];

        for (const ev of incomingEvents) {
          if (!seenEventsRef.current.has(ev)) {
            seenEventsRef.current.add(ev);

            // Clasificar fuente real según el contenido del mensaje
            let src: SystemLogMessage["source"] = "JETSON";
            if (ev.includes("Nodo") || ev.includes("LoRa") || ev.includes("Heartbeat")) {
              src = "LORA";
            } else if (ev.includes("Pixhawk") || ev.includes("GUIDED") || ev.includes("NAV")) {
              src = "PIXHAWK";
            } else if (ev.includes("VEREDICTO") || ev.includes("FUEGO") || ev.includes("cámara")) {
              src = "PERCEPTION";
            }

            newLogs.push({
              id: `${Date.now()}-${Math.random()}`,
              timestamp_s: Date.now() / 1000,
              source: src,
              message: ev,
            });
          }
        }

        if (newLogs.length > 0) {
          setLogs((prev) => [...prev, ...newLogs].slice(-200));
        }
      } catch {
        // Jetson unreachable
      }
    }, 500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // 2. Logs reales desde el WebSocket si el backend los emite
  useEffect(() => {
    if (liveTick?.system_logs?.length) {
      setLogs((prev) => {
        const ids = new Set(prev.map((l) => l.id));
        const filtered = liveTick.system_logs!.filter((l) => !ids.has(l.id));
        return [...prev, ...filtered].slice(-200);
      });
    }
  }, [liveTick?.system_logs]);

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll, filter, open]);

  const filteredLogs = logs.filter((l) => filter === "ALL" || l.source === filter);

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="absolute left-3 top-20 z-10 flex items-center gap-2 rounded-md border border-border/60 bg-card/90 p-2 text-xs font-mono backdrop-blur-md hover:bg-card shadow-lg"
      >
        <TerminalSquare className="h-4 w-4 text-emerald-400 animate-pulse" />
        <span className="font-bold">AUDITORÍA HARDWARE</span>
        {logs.length > 0 && (
          <span className="px-1.5 py-0.2 bg-emerald-500/20 text-emerald-400 text-[10px] rounded-full">
            {logs.length}
          </span>
        )}
      </button>
    );
  }

  return (
    <div className="absolute left-3 top-20 z-20 flex h-[380px] w-[500px] flex-col rounded-md border border-border/60 bg-black/95 backdrop-blur-xl shadow-2xl overflow-hidden font-mono text-[11px]">
      {/* Barra superior */}
      <div className="flex h-8 items-center justify-between border-b border-border/40 bg-zinc-950 px-3 select-none">
        <div className="flex items-center gap-2 text-emerald-400 font-bold">
          <TerminalSquare className="h-4 w-4" />
          <span>TERMINAL DE AUDITORÍA HARDWARE (EN VIVO)</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={cn(
              "flex items-center gap-1 hover:text-emerald-400 transition-colors text-[10px]",
              autoScroll ? "text-emerald-500" : "text-muted-foreground"
            )}
          >
            {autoScroll ? <PauseCircle className="h-3.5 w-3.5" /> : <PlayCircle className="h-3.5 w-3.5" />}
            <span>{autoScroll ? "AUTO-SCROLL" : "PAUSADO"}</span>
          </button>
          <button
            onClick={() => setOpen(false)}
            className="text-muted-foreground hover:text-destructive font-bold text-xs"
          >
            [X]
          </button>
        </div>
      </div>

      {/* Filtros */}
      <div className="flex items-center gap-1 border-b border-border/40 bg-zinc-900/60 px-2 py-1 text-[10px]">
        <Filter className="h-3 w-3 text-muted-foreground mr-1" />
        {(["ALL", "LORA", "PIXHAWK", "JETSON", "PERCEPTION"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={cn(
              "px-2 py-0.5 rounded transition-colors font-bold",
              filter === s
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : "text-muted-foreground hover:bg-white/5"
            )}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Lista de Logs */}
      <ScrollArea className="flex-1 p-2">
        {filteredLogs.length === 0 ? (
          <div className="text-muted-foreground text-center mt-12 opacity-60 italic">
            Esperando eventos y paquetes de hardware...
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {filteredLogs.map((l) => {
              const timeStr = new Date(l.timestamp_s * 1000).toLocaleTimeString();
              return (
                <div key={l.id} className="flex gap-2 leading-relaxed hover:bg-white/5 px-1 rounded">
                  <span className="text-zinc-500 shrink-0 select-none">[{timeStr}]</span>
                  <span className={cn("font-bold shrink-0 w-[85px] select-none", SOURCE_COLORS[l.source])}>
                    [{l.source}]
                  </span>
                  <span className="text-zinc-200">{l.message}</span>
                </div>
              );
            })}
            <div ref={bottomRef} className="h-1" />
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
