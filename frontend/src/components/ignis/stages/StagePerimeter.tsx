/**
 * ETAPA 1 — Perímetro
 * Lets the user draw on the map, paste GeoJSON/JSON, run terrain analysis.
 */
import { useEffect, useState } from "react";
import * as turf from "@turf/turf";
import {
  Pencil, Upload, AlertTriangle, CheckCircle2, Activity, Loader2,
  Save, Trash2, FolderOpen, X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useIgnisStore } from "@/store/useIgnisStore";
import { analyzePerimeter, terrainStats } from "@/lib/api";
import type { GeoJSONPolygon, RiskLevel } from "@/lib/types";
import {
  listSavedPerimeters,
  savePerimeter,
  deleteSavedPerimeter,
  type SavedPerimeter,
} from "@/lib/perimeterStorage";

const RISK_COLOR: Record<RiskLevel, string> = {
  LOW: "border-success/40 bg-success/10 text-success",
  MODERATE: "border-warning/40 bg-warning/10 text-warning",
  HIGH: "border-primary/40 bg-primary/10 text-primary",
  EXTREME: "border-destructive/40 bg-destructive/10 text-destructive",
};

export function StagePerimeter({ onContinue }: { onContinue: () => void }) {
  const perimeter = useIgnisStore((s) => s.perimeter);
  const perimeterName = useIgnisStore((s) => s.perimeterName);
  const fuelRisk = useIgnisStore((s) => s.fuelRisk);
  const terrain = useIgnisStore((s) => s.terrainStats);
  const setPerimeter = useIgnisStore((s) => s.setPerimeter);
  const setFuelRisk = useIgnisStore((s) => s.setFuelRisk);
  const setTerrainStats = useIgnisStore((s) => s.setTerrainStats);
  const setToolMode = useIgnisStore((s) => s.setToolMode);
  const toolMode = useIgnisStore((s) => s.toolMode);

  const [raw, setRaw] = useState("");
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteErr, setPasteErr] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const [saved, setSaved] = useState<SavedPerimeter[]>([]);
  const [savedOpen, setSavedOpen] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState("");

  useEffect(() => {
    setSaved(listSavedPerimeters());
  }, []);

  const refreshSaved = () => setSaved(listSavedPerimeters());

  const handleSave = () => {
    if (perimeter.length < 3) return;
    const name = saveName.trim() || perimeterName;
    savePerimeter(name, perimeter);
    refreshSaved();
    setSaveOpen(false);
    setSaveName("");
    toast.success(`Perímetro guardado: ${name}`);
  };

  const handleLoadSaved = (item: SavedPerimeter) => {
    setPerimeter(item.coords);
    toast.success(`Cargado: ${item.name} · ${item.coords.length} vértices`);
    setSavedOpen(false);
  };

  const handleDeleteSaved = (item: SavedPerimeter) => {
    deleteSavedPerimeter(item.id);
    refreshSaved();
    toast.success(`Eliminado: ${item.name}`);
  };

  const areaHa = perimeter.length >= 3
    ? turf.area(turf.polygon([closeRing(perimeter)])) / 10_000
    : 0;

  const handlePaste = () => {
    setPasteErr(null);
    try {
      const data = JSON.parse(raw.trim());
      const arr = Array.isArray(data) ? data : data?.coordinates?.[0];
      if (!Array.isArray(arr) || arr.length < 3) {
        throw new Error("Se requiere un array con al menos 3 coordenadas.");
      }
      const coords: [number, number][] = arr.map((pt: unknown, i: number) => {
        if (Array.isArray(pt) && pt.length >= 2 && typeof pt[0] === "number" && typeof pt[1] === "number") {
          return [pt[0], pt[1]];
        }
        if (
          pt && typeof pt === "object" &&
          typeof (pt as { lat: number }).lat === "number" &&
          typeof (pt as { lng: number }).lng === "number"
        ) {
          return [(pt as { lng: number }).lng, (pt as { lat: number }).lat];
        }
        throw new Error(`Coordenada inválida en posición ${i}.`);
      });
      setPerimeter(coords);
      toast.success(`Perímetro cargado · ${coords.length} vértices`);
      setPasteOpen(false);
      setRaw("");
    } catch (e) {
      setPasteErr(e instanceof SyntaxError ? "JSON mal formateado." : (e as Error).message);
    }
  };

  const handleAnalyze = async () => {
    if (perimeter.length < 3) return;
    setAnalyzing(true);
    const polygon: GeoJSONPolygon = {
      type: "Polygon",
      coordinates: [closeRing(perimeter)],
    };
    try {
      const promise = Promise.all([
        analyzePerimeter(perimeterName, polygon),
        terrainStats(perimeterName, polygon),
      ]);
      toast.promise(promise, {
        loading: "Analizando combustible (NDVI/NDMI)…",
        success: ([r]) => `Riesgo: ${r.risk_level} · score ${r.risk_score.toFixed(2)}`,
        error: (e) => `Error análisis: ${(e as Error).message}`,
      });
      const [risk, terr] = await promise;
      setFuelRisk(risk);
      setTerrainStats(terr);
    } catch (e) {
      console.error(e);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        <Button
          variant={toolMode === "DRAW_PERIMETER" ? "default" : "outline"}
          size="sm"
          onClick={() =>
            setToolMode(toolMode === "DRAW_PERIMETER" ? "IDLE" : "DRAW_PERIMETER")
          }
        >
          <Pencil className="mr-1.5 h-3.5 w-3.5" /> Dibujar
        </Button>
        <Button variant="outline" size="sm" onClick={() => { setPasteOpen((o) => !o); setSavedOpen(false); setSaveOpen(false); }}>
          <Upload className="mr-1.5 h-3.5 w-3.5" /> JSON
        </Button>
        <Button variant="outline" size="sm" onClick={() => { setSavedOpen((o) => !o); setPasteOpen(false); setSaveOpen(false); }}>
          <FolderOpen className="mr-1.5 h-3.5 w-3.5" /> Guardados
          {saved.length > 0 && (
            <span className="ml-1 font-mono text-[10px] text-muted-foreground">({saved.length})</span>
          )}
        </Button>
      </div>

      {pasteOpen && (
        <div className="space-y-2">
          <Textarea
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
            placeholder={`[\n  [-73.065, -36.795],\n  [-73.055, -36.795],\n  [-73.055, -36.785]\n]`}
            className="h-28 resize-none border-border/60 bg-background/40 font-mono text-[11px] leading-tight"
          />
          {pasteErr && (
            <div className="flex items-start gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span>{pasteErr}</span>
            </div>
          )}
          <Button onClick={handlePaste} size="sm" className="w-full" disabled={!raw.trim()}>
            Cargar coordenadas
          </Button>
        </div>
      )}

      {savedOpen && (
        <div className="space-y-1.5 rounded-md border border-border/50 bg-background/40 p-2">
          <div className="flex items-center justify-between px-1">
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Perímetros guardados
            </span>
            <button
              onClick={() => setSavedOpen(false)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Cerrar"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
          {saved.length === 0 ? (
            <div className="px-1 py-2 text-[11px] text-muted-foreground">
              No hay perímetros guardados aún.
            </div>
          ) : (
            <ul className="max-h-48 space-y-1 overflow-y-auto">
              {saved.map((item) => (
                <li
                  key={item.id}
                  className="flex items-center gap-1.5 rounded border border-border/40 bg-background/60 px-2 py-1.5"
                >
                  <button
                    onClick={() => handleLoadSaved(item)}
                    className="flex-1 text-left"
                  >
                    <div className="truncate text-[11px] font-medium">{item.name}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">
                      {item.coords.length} vértices · {new Date(item.createdAt).toLocaleDateString()}
                    </div>
                  </button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-destructive hover:bg-destructive/10 hover:text-destructive"
                    onClick={() => handleDeleteSaved(item)}
                    aria-label={`Eliminar ${item.name}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {perimeter.length >= 3 && (
        <div className="rounded-md border border-border/50 bg-background/40 px-3 py-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Vértices</span>
            <span className="font-mono">{perimeter.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Área</span>
            <span className="font-mono">{areaHa.toFixed(1)} ha</span>
          </div>
        </div>
      )}

      {perimeter.length >= 3 && (
        <>
          {!saveOpen ? (
            <Button
              onClick={() => { setSaveOpen(true); setSaveName(perimeterName); }}
              size="sm"
              variant="outline"
              className="w-full"
            >
              <Save className="mr-1.5 h-3.5 w-3.5" /> Guardar perímetro
            </Button>
          ) : (
            <div className="space-y-2 rounded-md border border-border/50 bg-background/40 p-2">
              <Input
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder="Nombre del perímetro"
                className="h-8 text-xs"
              />
              <div className="grid grid-cols-2 gap-2">
                <Button onClick={handleSave} size="sm">
                  <Save className="mr-1.5 h-3.5 w-3.5" /> Guardar
                </Button>
                <Button
                  onClick={() => { setSaveOpen(false); setSaveName(""); }}
                  size="sm"
                  variant="outline"
                >
                  Cancelar
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      <Button
        onClick={handleAnalyze}
        size="sm"
        className="w-full"
        disabled={perimeter.length < 3 || analyzing}
      >
        {analyzing ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Activity className="mr-1.5 h-3.5 w-3.5" />}
        Analizar terreno
      </Button>

      {analyzing && !fuelRisk && (
        <div className="space-y-1.5">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      )}

      {fuelRisk && (
        <div className="space-y-2">
          <div className={`rounded-md border px-2.5 py-2 text-xs ${RISK_COLOR[fuelRisk.risk_level]}`}>
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-widest opacity-80">Riesgo</span>
              <Badge variant="outline" className="border-current bg-transparent text-current">
                {fuelRisk.risk_level}
              </Badge>
            </div>
            <div className="mt-1 grid grid-cols-3 gap-2 font-mono text-[11px]">
              <Stat label="NDVI" v={fuelRisk.mean_ndvi.toFixed(2)} />
              <Stat label="NDMI" v={fuelRisk.mean_ndmi.toFixed(2)} />
              <Stat label="Score" v={fuelRisk.risk_score.toFixed(2)} />
            </div>
          </div>

          {terrain && (
            <div className="rounded-md border border-border/50 bg-background/40 px-2.5 py-2 text-xs">
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Terreno</div>
              <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11px]">
                <Stat label="Elev media" v={`${terrain.elev_mean_m.toFixed(0)} m`} />
                <Stat label="Elev máx" v={`${terrain.elev_max_m.toFixed(0)} m`} />
                <Stat label="Pend media" v={`${terrain.slope_mean_deg.toFixed(1)}°`} />
                <Stat label="Pend máx" v={`${terrain.slope_max_deg.toFixed(1)}°`} />
              </div>
            </div>
          )}

          <Button onClick={onContinue} size="sm" variant="outline" className="w-full">
            <CheckCircle2 className="mr-1.5 h-3.5 w-3.5 text-success" /> Continuar
          </Button>
        </div>
      )}
    </div>
  );
}

function Stat({ label, v }: { label: string; v: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
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

