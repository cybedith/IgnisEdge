/**
 * ETAPA 2 — Optimización de nodos LoRa.
 */
import { useState } from "react";
import { Loader2, Radio, CheckCircle2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useIgnisStore } from "@/store/useIgnisStore";
import { optimizeMesh } from "@/lib/api";
import type { GeoJSONPolygon, NodeRole } from "@/lib/types";
import { cn } from "@/lib/utils";

const ROLE_BADGE: Record<NodeRole, string> = {
  GATEWAY: "border-primary/40 bg-primary/15 text-primary",
  RELAY: "border-blue-400/40 bg-blue-400/15 text-blue-300",
  EDGE: "border-cyan-400/40 bg-cyan-400/15 text-cyan-300",
};

function scoreColor(v: number) {
  if (v < 0.5) return "bg-destructive";
  if (v < 0.8) return "bg-warning";
  return "bg-success";
}

export function StageMesh({ onContinue }: { onContinue: () => void }) {
  const perimeter = useIgnisStore((s) => s.perimeter);
  const perimeterName = useIgnisStore((s) => s.perimeterName);
  const meshConfig = useIgnisStore((s) => s.meshConfig);
  const setMeshConfig = useIgnisStore((s) => s.setMeshConfig);
  const meshResult = useIgnisStore((s) => s.meshResult);
  const setMeshResult = useIgnisStore((s) => s.setMeshResult);

  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (perimeter.length < 3) return;
    setBusy(true);
    const polygon: GeoJSONPolygon = {
      type: "Polygon",
      coordinates: [closeRing(perimeter)],
    };
    try {
      const result = await optimizeMesh(
        perimeterName,
        polygon,
        meshConfig.n_nodes,
        meshConfig.detection_radius_m,
      );
      setMeshResult(result);
      toast.success(
        `${result.nodes.length} nodos optimizados · cobertura ${(result.coverage_score * 100).toFixed(0)}%`,
      );
    } catch (e) {
      toast.error(`Error optimización: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Cantidad de nodos
          </Label>
          <span className="font-mono text-xs">{meshConfig.n_nodes}</span>
        </div>
        <Slider
          value={[meshConfig.n_nodes]}
          min={3}
          max={12}
          step={1}
          onValueChange={([v]) => setMeshConfig({ n_nodes: v })}
        />
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Radio detección
          </Label>
          <span className="font-mono text-xs">{meshConfig.detection_radius_m} m</span>
        </div>
        <Slider
          value={[meshConfig.detection_radius_m]}
          min={200}
          max={3000}
          step={50}
          onValueChange={([v]) => setMeshConfig({ detection_radius_m: v })}
        />
      </div>

      <Button onClick={run} size="sm" className="w-full" disabled={busy || perimeter.length < 3}>
        {busy ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Radio className="mr-1.5 h-3.5 w-3.5" />}
        {meshResult ? "Re-optimizar" : "Optimizar mesh"}
      </Button>

      {meshResult && (
        <div className="space-y-2">
          <div className="space-y-1.5 rounded-md border border-border/50 bg-background/40 px-2.5 py-2">
            <ScoreBar label="Cobertura" v={meshResult.coverage_score} />
            <ScoreBar label="Conectividad" v={meshResult.connectivity_score} />
            <ScoreBar label="Fitness" v={meshResult.fitness} />
          </div>

          <ul className="max-h-44 space-y-1 overflow-y-auto pr-1">
            {meshResult.nodes.map((n) => (
              <li
                key={n.node_id}
                className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-2 py-1.5 text-[11px]"
              >
                <div className="flex items-center gap-1.5">
                  <Badge variant="outline" className={cn("h-4 px-1 text-[9px]", ROLE_BADGE[n.role])}>
                    {n.role}
                  </Badge>
                  <span className="font-mono">{n.node_id}</span>
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">
                  {n.elevation_m.toFixed(0)} m
                </span>
              </li>
            ))}
          </ul>

          <div className="grid grid-cols-2 gap-2">
            <Button onClick={run} size="sm" variant="outline" disabled={busy}>
              <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> Re-optimizar
            </Button>
            <Button onClick={onContinue} size="sm" variant="outline">
              <CheckCircle2 className="mr-1.5 h-3.5 w-3.5 text-success" /> Continuar
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreBar({ label, v }: { label: string; v: number }) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{(v * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/40">
        <div
          className={cn("h-full transition-all", scoreColor(v))}
          style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}
        />
      </div>
    </div>
  );
}

function closeRing(coords: [number, number][]): [number, number][] {
  if (coords.length === 0) return coords;
  const [fx, fy] = coords[0];
  const [lx, ly] = coords[coords.length - 1];
  return fx === lx && fy === ly ? coords : [...coords, [fx, fy]];
}
