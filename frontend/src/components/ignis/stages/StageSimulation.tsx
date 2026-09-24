/**
 * ETAPA 3 — Crear y controlar la simulación.
 * Payload alineado con SimulationCreateIn del backend (polygon, no perimeter).
 */
import { useState } from "react";
import { Play, Pause, Square, Loader2, Cloud, Wind } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useIgnisStore } from "@/store/useIgnisStore";
import {
  createSimulation,
  startSimulation,
  pauseSimulation,
  deleteSimulation,
} from "@/lib/api";
import type { GeoJSONPolygon } from "@/lib/types";

export function StageSimulation({ onContinue }: { onContinue: () => void }) {
  const perimeter = useIgnisStore((s) => s.perimeter);
  const perimeterName = useIgnisStore((s) => s.perimeterName);
  const nodes = useIgnisStore((s) => s.nodes);
  const session = useIgnisStore((s) => s.session);
  const setSession = useIgnisStore((s) => s.setSession);
  const liveTick = useIgnisStore((s) => s.liveTick);

  const [useRealWeather, setUseRealWeather] = useState(true);
  const [windSpeed, setWindSpeed] = useState(4);
  const [windDir, setWindDir] = useState(180);
  const [speedMul, setSpeedMul] = useState(30);
  const [busy, setBusy] = useState(false);

  const handleCreate = async () => {
    if (perimeter.length < 3 || nodes.length === 0) return;
    setBusy(true);
    const polygon: GeoJSONPolygon = {
      type: "Polygon",
      coordinates: [closeRing(perimeter)],
    };
    try {
      const created = await createSimulation({
        name: `${perimeterName} · ${new Date().toLocaleString()}`,
        polygon,
        nodes,
        use_real_weather: useRealWeather,
        manual_wind_speed_ms: useRealWeather ? undefined : windSpeed,
        manual_wind_direction_deg: useRealWeather ? undefined : windDir,
        speed_multiplier: speedMul,
      });
      const started = await startSimulation(created.session_id);
      setSession(started);
      toast.success(`Simulación iniciada · ${started.session_id.slice(0, 8)}`);
      onContinue();
    } catch (e) {
      toast.error(`Error: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handlePauseResume = async () => {
    if (!session) return;
    setBusy(true);
    try {
      const updated = session.running
        ? await pauseSimulation(session.session_id)
        : await startSimulation(session.session_id);
      setSession(updated);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    if (!session) return;
    setBusy(true);
    try {
      await deleteSimulation(session.session_id);
      setSession(null);
      toast.message("Simulación detenida");
    } finally {
      setBusy(false);
    }
  };

  if (!session) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-2.5 py-2">
          <Label className="flex items-center gap-1.5 text-xs">
            <Cloud className="h-3.5 w-3.5" /> Meteo real (NOAA GFS)
          </Label>
          <Switch checked={useRealWeather} onCheckedChange={setUseRealWeather} />
        </div>

        {!useRealWeather && (
          <div className="space-y-2 rounded-md border border-border/50 bg-background/40 px-2.5 py-2">
            <ManualSlider label="Viento" unit="m/s" min={0} max={30} step={0.5}
              value={windSpeed} onChange={setWindSpeed} icon={<Wind className="h-3 w-3" />} />
            <ManualSlider label="Dirección" unit="°" min={0} max={360} step={1}
              value={windDir} onChange={setWindDir} />
          </div>
        )}

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Velocidad simulación
            </Label>
            <span className="font-mono text-xs">×{speedMul}</span>
          </div>
          <Slider
            value={[speedMul]}
            min={1}
            max={300}
            step={1}
            onValueChange={([v]) => setSpeedMul(v)}
          />
        </div>

        <Button onClick={handleCreate} size="sm" className="w-full" disabled={busy || nodes.length === 0}>
          {busy ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Play className="mr-1.5 h-3.5 w-3.5" />}
          Crear y arrancar
        </Button>
      </div>
    );
  }

  const w = liveTick?.weather;
  return (
    <div className="space-y-3">
      <div className="rounded-md border border-border/50 bg-background/40 px-2.5 py-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                session.running ? "animate-pulse bg-success" : "bg-warning"
              }`}
            />
            <span className="font-mono">{session.running ? "RUNNING" : "PAUSED"}</span>
          </span>
          <span className="font-mono text-[10px] text-muted-foreground">
            ×{liveTick?.speed_multiplier ?? session.speed_multiplier}
          </span>
        </div>
        <div className="mt-1 font-mono text-[10px] text-muted-foreground">
          t = {(liveTick?.sim_time_s ?? session.sim_time_s).toFixed(0)} s
        </div>
      </div>

      {w && (
        <div className="rounded-md border border-border/50 bg-background/40 px-2.5 py-2 text-[11px]">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Meteo</div>
          <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
            <KV k="Viento" v={`${w.wind_speed_ms.toFixed(1)} m/s`} />
            <KV k="Dir" v={`${w.wind_direction_deg.toFixed(0)}°`} />
            <KV k="Temp" v={`${w.temperature_c.toFixed(1)}°C`} />
            <KV k="HR" v={`${w.relative_humidity_pct.toFixed(0)}%`} />
          </div>
          <div className="mt-1 text-[9px] text-muted-foreground/70">{w.source}</div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <Button onClick={handlePauseResume} size="sm" variant="outline" disabled={busy}>
          {session.running ? (
            <><Pause className="mr-1.5 h-3.5 w-3.5" /> Pausar</>
          ) : (
            <><Play className="mr-1.5 h-3.5 w-3.5" /> Reanudar</>
          )}
        </Button>
        <Button onClick={handleStop} size="sm" variant="destructive" disabled={busy}>
          <Square className="mr-1.5 h-3.5 w-3.5" /> Detener
        </Button>
      </div>

      <p className="text-[10px] text-muted-foreground">
        💡 Click en mapa para colocar foco de calor
      </p>
    </div>
  );
}

function ManualSlider({
  label, unit, min, max, step, value, onChange, icon,
}: {
  label: string; unit: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void; icon?: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px]">
        <span className="flex items-center gap-1 text-muted-foreground">{icon} {label}</span>
        <span className="font-mono">{value} {unit}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => onChange(v)} />
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{k}</span>
      <span>{v}</span>
    </div>
  );
}

function closeRing(coords: [number, number][]): [number, number][] {
  if (coords.length === 0) return coords;
  const [fx, fy] = coords[0];
  const [lx, ly] = coords[coords.length - 1];
  return fx === lx && fy === ly ? coords : [...coords, [fx, fy]];
}
