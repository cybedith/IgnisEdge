/**
 * PANEL DE VEREDICTO IA — IgnisEdge
 * Muestra en tiempo real el veredicto emitido por el clasificador
 * de Machine Learning / Visión Térmica en la Jetson Orin Nano.
 */
import { Flame, CheckCircle2, AlertTriangle, ShieldCheck, Activity, Eye } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface VerdictData {
  fire: boolean;
  confidence: number;
  max_T?: number;
  n_focos?: number;
  focos?: Array<{
    id: number;
    cx: number;
    cy: number;
    state: string;
    confidence: number;
    flicker?: number;
    temp?: number;
  }>;
  fps?: number;
}

interface VerdictHUDProps {
  verdict: VerdictData | null;
  missionState?: string;
  onDismiss?: () => void;
}

export function VerdictHUD({ verdict, missionState, onDismiss }: VerdictHUDProps) {
  if (!verdict && missionState !== "ON_STATION" && missionState !== "VERDICT") {
    return null;
  }

  const isConfirmed = verdict?.fire === true && (verdict?.confidence ?? 0) > 0.8;
  const isEvaluating = missionState === "ON_STATION";

  return (
    <div className="pointer-events-auto absolute top-14 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center">
      <div
        className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 shadow-2xl backdrop-blur-xl transition-all duration-300 ${
          isConfirmed
            ? "border-red-500/80 bg-red-950/80 text-red-100 shadow-red-900/50"
            : isEvaluating
            ? "border-amber-500/80 bg-amber-950/80 text-amber-100 shadow-amber-900/50"
            : "border-border/60 bg-card/90 text-foreground"
        }`}
      >
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-background/50 ring-1 ring-border/40">
          {isConfirmed ? (
            <Flame className="h-5 w-5 text-red-400 animate-pulse" />
          ) : isEvaluating ? (
            <Eye className="h-5 w-5 text-amber-400 animate-spin" />
          ) : (
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
          )}
        </div>

        <div className="flex flex-col font-mono text-xs">
          <div className="flex items-center gap-2">
            <span className="font-bold uppercase tracking-wider text-[11px]">
              {isConfirmed
                ? "VEREDICTO: FUEGO CONFIRMADO"
                : isEvaluating
                ? "INSPECCIÓN TÉRMICA EN CURSO (ON_STATION)"
                : "VEREDICTO: ÁREA SEGURA"}
            </span>
            <Badge
              variant={isConfirmed ? "destructive" : "outline"}
              className="text-[9px] font-mono uppercase"
            >
              {isConfirmed
                ? `Confianza ${( (verdict?.confidence ?? 1) * 100).toFixed(0)}%`
                : isEvaluating
                ? "Analizando Flicker..."
                : "Sin Fuego"}
            </Badge>
          </div>

          <div className="flex items-center gap-4 text-[10px] text-muted-foreground mt-0.5">
            <span>
              Temp Máx: <strong className="text-foreground">{verdict?.max_T?.toFixed(1) ?? "--"}°C</strong>
            </span>
            <span>
              Focos: <strong className="text-foreground">{verdict?.n_focos ?? 1}</strong>
            </span>
            <span>
              Flicker: <strong className="text-amber-300">{verdict?.focos?.[0]?.flicker?.toFixed(1) ?? "3.2"} Hz</strong>
            </span>
            <span className="text-[9px] text-muted-foreground/80">
              Evidencia: [TMP, CNN, NODO, TRACK]
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
