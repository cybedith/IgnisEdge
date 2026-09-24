import { Flame, Zap, Radio, Settings, RotateCcw, Info, Wifi, WifiOff, ShieldAlert } from "lucide-react";
import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AbortMissionDialog } from "@/components/ignis/AbortMissionDialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useIgnisStore } from "@/store/useIgnisStore";
import { cn } from "@/lib/utils";

const STAGES = [
  { key: "perimeter", label: "Perímetro" },
  { key: "analysis", label: "Análisis" },
  { key: "mesh", label: "Optimización" },
  { key: "simulation", label: "Simulación" },
] as const;

interface TopBarProps {
  /** WebSocket connection state for the live indicator. */
  wsConnected?: boolean;
}

/**
 * Sticky top command bar.
 *
 * 🔌 FUTURE: the mode toggle will switch between SIMULATION (WS to backend
 * sim engine) and LIVE_HARDWARE (WS to LoRa gateway telemetry).
 */
export function TopBar({ wsConnected = false }: TopBarProps) {
  const systemMode = useIgnisStore((s) => s.systemMode);
  const setSystemMode = useIgnisStore((s) => s.setSystemMode);
  const perimeter = useIgnisStore((s) => s.perimeter);
  const fuelRisk = useIgnisStore((s) => s.fuelRisk);
  const meshResult = useIgnisStore((s) => s.meshResult);
  const session = useIgnisStore((s) => s.session);
  const liveTick = useIgnisStore((s) => s.liveTick);
  const reset = useIgnisStore((s) => s.reset);

  const isLive = systemMode === "LIVE_HARDWARE";

  // Determine the active stage (first one not yet completed).
  const activeStage = !perimeter.length
    ? "perimeter"
    : !fuelRisk
      ? "analysis"
      : !meshResult
        ? "mesh"
        : "simulation";

  const simTimeLabel = (() => {
    const s = liveTick?.sim_time_s ?? session?.sim_time_s ?? 0;
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    const sec = Math.floor(s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  })();

  const [abortOpen, setAbortOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 flex h-12 shrink-0 items-center justify-between border-b border-border/60 bg-background/80 px-4 backdrop-blur-md">
      {/* ── Left: brand ── */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 ring-1 ring-primary/30">
          <Flame className="h-4 w-4 text-primary" />
        </div>
        <div className="leading-tight">
          <h1 className="text-xs font-semibold tracking-wide text-foreground">Ignis Edge C2</h1>
          <p className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
            Sector Hualpén
          </p>
        </div>
      </div>

      {/* ── Center: breadcrumb stages ── */}
      <nav className="hidden items-center gap-1 md:flex">
        {STAGES.map((s, i) => {
          const active = s.key === activeStage;
          const done =
            (s.key === "perimeter" && perimeter.length > 0) ||
            (s.key === "analysis" && !!fuelRisk) ||
            (s.key === "mesh" && !!meshResult) ||
            (s.key === "simulation" && !!session);
          return (
            <div key={s.key} className="flex items-center gap-1">
              <span
                className={cn(
                  "rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider transition-colors",
                  active
                    ? "bg-primary/15 text-primary ring-1 ring-primary/40"
                    : done
                      ? "text-success"
                      : "text-muted-foreground/60",
                )}
              >
                {s.label}
              </span>
              {i < STAGES.length - 1 && (
                <span className="text-muted-foreground/40">›</span>
              )}
            </div>
          );
        })}
      </nav>

      {/* ── Right: mode + session + ws + menu ── */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Zap
            className={cn(
              "h-3.5 w-3.5",
              !isLive ? "text-warning" : "text-muted-foreground/40",
            )}
          />
          <Switch
            checked={isLive}
            onCheckedChange={(v) => setSystemMode(v ? "LIVE_HARDWARE" : "SIMULATION")}
          />
          <Radio
            className={cn(
              "h-3.5 w-3.5",
              isLive ? "text-success" : "text-muted-foreground/40",
            )}
          />
        </div>

        {session && (
          <Badge
            variant="outline"
            className="border-primary/40 bg-primary/10 font-mono text-[10px] text-primary"
          >
            {simTimeLabel} · ×{liveTick?.speed_multiplier ?? session.speed_multiplier}
          </Badge>
        )}

        {isLive && (
          <>
            <Button variant="destructive" size="sm" className="h-7 text-[10px] font-bold tracking-wider uppercase" onClick={() => setAbortOpen(true)}>
              <ShieldAlert className="mr-1.5 h-3.5 w-3.5" />
              ABORTAR
            </Button>
            <AbortMissionDialog open={abortOpen} onOpenChange={setAbortOpen} />
          </>
        )}

        <div
          className="flex items-center gap-1 text-[10px] text-muted-foreground"
          title={wsConnected ? "WebSocket conectado" : "WebSocket desconectado"}
        >
          {wsConnected ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/70" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
              </span>
              <Wifi className="h-3 w-3 text-success" />
            </>
          ) : (
            <>
              <span className="h-2 w-2 rounded-full bg-destructive/70" />
              <WifiOff className="h-3 w-3 text-destructive" />
            </>
          )}
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7">
              <Settings className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem onClick={reset}>
              <RotateCcw className="mr-2 h-3.5 w-3.5" /> Reiniciar misión
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              <Info className="mr-2 h-3.5 w-3.5" /> Acerca de Ignis
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
