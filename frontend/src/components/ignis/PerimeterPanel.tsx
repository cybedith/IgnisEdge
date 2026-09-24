import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Upload, CheckCircle2, AlertTriangle, Radio } from "lucide-react";
import { useIgnisStore } from "@/store/useIgnisStore";

/**
 * Dynamic Perimeter import.
 * Accepts a JSON array of [lng, lat] pairs (GeoJSON convention) OR
 * an array of {lat, lng} objects. Validates and pushes to the store,
 * which triggers an automatic flyTo in the MapContainer.
 *
 * 🔌 FUTURE: same setPerimeter() will be invoked from a backend
 * fetch (e.g. cadastral service) when LIVE_HARDWARE is active.
 */
export function PerimeterPanel() {
  const setPerimeter = useIgnisStore((s) => s.setPerimeter);
  const deployNodes = useIgnisStore((s) => s.deployNodes);
  const perimeter = useIgnisStore((s) => s.perimeter);
  const nodes = useIgnisStore((s) => s.nodes);

  const [raw, setRaw] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);

  const parseCoords = (input: string): [number, number][] => {
    const data = JSON.parse(input);
    if (!Array.isArray(data) || data.length < 3) {
      throw new Error("Se requiere un array con al menos 3 coordenadas.");
    }

    return data.map((pt, i) => {
      // [lng, lat] tuple
      if (Array.isArray(pt) && pt.length === 2 && pt.every((n) => typeof n === "number")) {
        return [pt[0], pt[1]] as [number, number];
      }
      // {lat, lng} object
      if (
        pt &&
        typeof pt === "object" &&
        typeof pt.lat === "number" &&
        typeof pt.lng === "number"
      ) {
        return [pt.lng, pt.lat] as [number, number];
      }
      throw new Error(`Coordenada inválida en posición ${i}.`);
    });
  };

  const handleLoad = () => {
    setError(null);
    setSuccess(false);
    try {
      const coords = parseCoords(raw.trim());
      setPerimeter(coords);
      setSuccess(true);
    } catch (e) {
      const msg = e instanceof SyntaxError ? "JSON mal formateado." : (e as Error).message;
      setError(msg);
    }
  };

  return (
    <div className="space-y-2.5">
      <Textarea
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value);
          if (error) setError(null);
          if (success) setSuccess(false);
        }}
        placeholder={`[\n  [-73.065, -36.795],\n  [-73.055, -36.795],\n  [-73.055, -36.785],\n  [-73.065, -36.785]\n]`}
        className="h-32 resize-none border-border/60 bg-background/40 font-mono text-[11px] leading-tight"
      />

      {error && (
        <div className="flex items-start gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {success && !error && (
        <div className="flex items-center gap-1.5 rounded-md border border-success/40 bg-success/10 px-2 py-1.5 text-[11px] text-success">
          <CheckCircle2 className="h-3 w-3 shrink-0" />
          <span>Perímetro cargado · {perimeter.length} vértices</span>
        </div>
      )}

      <Button onClick={handleLoad} size="sm" className="w-full" disabled={!raw.trim()}>
        <Upload className="mr-2 h-3.5 w-3.5" />
        Cargar Perímetro
      </Button>

      {/* 🔌 FUTURE: when LIVE_HARDWARE is on, this button should request a real
          deployment plan from the backend instead of generating local nodes. */}
      <Button
        onClick={() => deployNodes(perimeter)}
        size="sm"
        variant="outline"
        className="w-full"
        disabled={perimeter.length < 3}
      >
        <Radio className="mr-2 h-3.5 w-3.5" />
        Desplegar Nodos LoRa
        {nodes.length > 0 && (
          <span className="ml-1 font-mono text-[10px] text-muted-foreground">
            ({nodes.length})
          </span>
        )}
      </Button>
    </div>
  );
}
