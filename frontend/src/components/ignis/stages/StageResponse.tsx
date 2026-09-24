/**
 * ETAPA 4 — Detección y respuesta (lista de nodos detectando, triangulación, drones).
 * launchDrone con campos `drone_id` + `safe_agl_m` alineados al backend.
 */
import { Flame, Plane, Radio, Loader2, Trash2, Eraser } from "lucide-react";
import { toast } from "sonner";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useIgnisStore } from "@/store/useIgnisStore";
import { launchDrone } from "@/lib/api";

const DETECTION_THRESHOLD_PPM = 50;

export function StageResponse() {
  const liveTick = useIgnisStore((s) => s.liveTick);
  const triangulation = useIgnisStore((s) => s.triangulation);
  const session = useIgnisStore((s) => s.session);
  const nodes = useIgnisStore((s) => s.nodes);
  const setToolMode = useIgnisStore((s) => s.setToolMode);
  const addDroneRoute = useIgnisStore((s) => s.addDroneRoute);
  const droneRoutes = useIgnisStore((s) => s.droneRoutes);
  const removeDroneRoute = useIgnisStore((s) => s.removeDroneRoute);
  const clearDroneRoutes = useIgnisStore((s) => s.clearDroneRoutes);
  const clearTriangulation = useIgnisStore((s) => s.clearTriangulation);
  const hotSpotsLocal = useIgnisStore((s) => s.hotSpotsLocal);
  const clearHotSpotsLocal = useIgnisStore((s) => s.clearHotSpotsLocal);

  const [launching, setLaunching] = useState(false);

  const detections = (liveTick?.node_status ?? [])
    .filter((n) => n.detected || n.smoke_ppm > 5)
    .sort((a, b) => (a.detection_time_s ?? Infinity) - (b.detection_time_s ?? Infinity));

  const drones = liveTick?.drones ?? [];

  const launch = async () => {
    if (!session || !triangulation) return;
    const gateway = nodes.find((n) => n.role === "GATEWAY") ?? nodes[0];
    if (!gateway) return;
    setLaunching(true);
    try {
      const droneId = `DRONE_${Date.now().toString(36)}`;
      const route = await launchDrone(session.session_id, {
        drone_id: droneId,
        start_lon: gateway.longitude,
        start_lat: gateway.latitude,
        start_alt_m: gateway.elevation_m,
        target_lon: triangulation.estimated_lon,
        target_lat: triangulation.estimated_lat,
        safe_agl_m: 80,
      });
      addDroneRoute(route);
      toast.success(`Dron ${route.drone_id.slice(0, 10)} · ETA ${Math.round(route.estimated_eta_s)}s`);
    } catch (e) {
      toast.error(`Error lanzamiento: ${(e as Error).message}`);
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
          <span className="flex items-center gap-1"><Radio className="h-3 w-3" /> Detecciones</span>
          <span className="font-mono">{detections.length}</span>
        </div>
        {detections.length === 0 ? (
          <div className="rounded-md border border-dashed border-border/60 bg-background/40 px-3 py-3 text-center text-[11px] text-muted-foreground">
            Sin detecciones de humo
          </div>
        ) : (
          <ul className="max-h-32 space-y-1 overflow-y-auto pr-1">
            {detections.map((d) => (
              <li
                key={d.node_id}
                className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-2 py-1 text-[11px]"
              >
                <span className="font-mono">{d.node_id}</span>
                <div className="flex items-center gap-2">
                  <span className={`font-mono ${d.smoke_ppm > DETECTION_THRESHOLD_PPM ? "text-destructive" : "text-warning"}`}>
                    {d.smoke_ppm.toFixed(0)} ppm
                  </span>
                  {d.detection_time_s != null && (
                    <span className="font-mono text-[9px] text-muted-foreground">
                      t={d.detection_time_s.toFixed(0)}s
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {triangulation && (
        <div className="space-y-2 rounded-md border border-primary/40 bg-primary/10 px-3 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-primary">
              <Flame className="h-4 w-4 animate-pulse" />
              <span>Origen triangulado</span>
            </div>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6 text-muted-foreground hover:text-destructive"
              title="Descartar triangulación"
              onClick={() => {
                clearTriangulation();
                toast.message("Triangulación descartada");
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="space-y-0.5 font-mono text-[11px]">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Lon</span>
              <span>{triangulation.estimated_lon.toFixed(5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Lat</span>
              <span>{triangulation.estimated_lat.toFixed(5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">±</span>
              <span>{triangulation.uncertainty_radius_m.toFixed(0)} m</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Conf</span>
              <span>{(triangulation.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
          <Button onClick={launch} size="sm" className="w-full" disabled={launching}>
            {launching ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Plane className="mr-1.5 h-3.5 w-3.5" />}
            Lanzar dron al objetivo
          </Button>
          <Button
            onClick={() => setToolMode("PLACE_DRONE_TARGET")}
            size="sm"
            variant="outline"
            className="w-full"
          >
            Override manual (click en mapa)
          </Button>
        </div>
      )}

      {(drones.length > 0 || droneRoutes.length > 0) && (
        <div>
          <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
            <span>Drones · rutas</span>
            <button
              onClick={() => {
                clearDroneRoutes();
                toast.message("Rutas de drones eliminadas");
              }}
              className="flex items-center gap-1 text-muted-foreground hover:text-destructive"
              title="Eliminar todas las rutas"
            >
              <Eraser className="h-3 w-3" /> Limpiar
            </button>
          </div>
          <ul className="space-y-1">
            {drones.map((d) => (
              <li key={d.drone_id} className="space-y-1 rounded-md border border-border/50 bg-background/40 px-2 py-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="flex items-center gap-1.5 font-mono">
                    <Plane className="h-3 w-3 text-cyan-300" />
                    {d.drone_id.slice(0, 10)}
                  </span>
                  <span className="font-mono text-[10px] text-muted-foreground">{d.status}</span>
                </div>
                <Progress value={d.progress * 100} className="h-1" />
              </li>
            ))}
            {droneRoutes
              .filter((r) => !drones.find((d) => d.drone_id === r.drone_id))
              .map((r) => (
                <li
                  key={r.drone_id}
                  className="flex items-center justify-between rounded-md border border-border/40 bg-background/30 px-2 py-1 text-[11px]"
                >
                  <span className="flex items-center gap-1.5 font-mono text-muted-foreground">
                    <Plane className="h-3 w-3" />
                    {r.drone_id.slice(0, 10)}
                  </span>
                  <button
                    onClick={() => removeDroneRoute(r.drone_id)}
                    className="text-muted-foreground hover:text-destructive"
                    title="Eliminar ruta"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </li>
              ))}
          </ul>
        </div>
      )}

      <div className="space-y-1.5 rounded-md border border-border/50 bg-background/30 px-2.5 py-2">
        <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
          Registros activos
        </div>
        <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
          <RecordRow
            label="Focos colocados"
            count={hotSpotsLocal.length}
            onClear={() => {
              clearHotSpotsLocal();
              toast.message("Focos eliminados");
            }}
          />
          <RecordRow
            label="Rutas de dron"
            count={droneRoutes.length}
            onClear={() => {
              clearDroneRoutes();
              toast.message("Rutas eliminadas");
            }}
          />
        </div>
      </div>
    </div>
  );
}

function RecordRow({
  label,
  count,
  onClear,
}: {
  label: string;
  count: number;
  onClear: () => void;
}) {
  return (
    <div className="flex items-center justify-between rounded border border-border/40 bg-background/40 px-1.5 py-1">
      <div className="leading-tight">
        <div className="text-[9px] uppercase text-muted-foreground">{label}</div>
        <div className="text-[11px]">{count}</div>
      </div>
      <button
        disabled={count === 0}
        onClick={onClear}
        className="text-muted-foreground hover:text-destructive disabled:opacity-30"
        title="Eliminar"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  );
}
