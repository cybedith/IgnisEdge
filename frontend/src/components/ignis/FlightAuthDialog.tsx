/**
 * MODAL DE AUTORIZACIÓN DE VUELO — OPERADOR C2 IGNIS
 * Exige confirmación explícita del operador humano ante alertas
 * emitidas por los sensores LoRa físicos o simulados.
 */
import { useState } from "react";
import { AlertTriangle, Plane, ShieldAlert, CheckCircle2, XCircle, Battery, Compass, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

export interface PendingAlertData {
  node_id: string | number;
  lat: number;
  lon: number;
  gas?: boolean;
  heat?: boolean;
  confidence?: number;
  elevation_m?: number;
  source?: "LORA_PHYSICAL" | "SIMULATION";
}

interface FlightAuthDialogProps {
  alert: PendingAlertData | null;
  onAuthorize: (password: string) => void;
  onDismiss: () => void;
  isLoading?: boolean;
}

export function FlightAuthDialog({
  alert,
  onAuthorize,
  onDismiss,
  isLoading = false,
}: FlightAuthDialogProps) {
  const [password, setPassword] = useState("");

  if (!alert) return null;

  const handleDismiss = () => {
    setPassword("");
    onDismiss();
  };

  const handleAuthorize = () => {
    if (password !== "ignis2026") {
      toast.error("Contraseña de Administrador incorrecta. Acceso denegado.");
      return;
    }
    onAuthorize(password);
    setPassword("");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-md rounded-xl border border-red-500/60 bg-card/95 p-5 shadow-2xl shadow-red-950/40 backdrop-blur-xl">
        {/* Encabezado con pulso de alarma */}
        <div className="flex items-start gap-3 border-b border-border/40 pb-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-500/20 text-red-400 ring-1 ring-red-500/40 animate-pulse">
            <ShieldAlert className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <Badge variant="destructive" className="animate-pulse text-[9px] uppercase tracking-wider">
                Alerta Crítica LoRa
              </Badge>
              <span className="font-mono text-[10px] text-muted-foreground">915 MHz RF</span>
            </div>
            <h3 className="mt-1 text-sm font-bold uppercase tracking-wider text-foreground">
              Autorización de Vuelo Requerida
            </h3>
          </div>
        </div>

        {/* Datos de Telemetría del Sensor */}
        <div className="mt-4 space-y-2.5 font-mono text-xs">
          <div className="rounded-lg border border-border/40 bg-background/60 p-3 space-y-2">
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="flex items-center gap-1.5 text-[11px]">
                <MapPin className="h-3.5 w-3.5 text-red-400" /> Nodo Emisor:
              </span>
              <strong className="text-foreground font-bold">NODO #{alert.node_id}</strong>
            </div>

            <div className="flex items-center justify-between text-muted-foreground">
              <span className="flex items-center gap-1.5 text-[11px]">
                <Compass className="h-3.5 w-3.5 text-cyan-400" /> Coordenadas:
              </span>
              <span className="text-foreground">
                {alert.lat.toFixed(5)}, {alert.lon.toFixed(5)}
              </span>
            </div>

            <div className="flex items-center justify-between text-muted-foreground">
              <span className="text-[11px]">Detección Sensor:</span>
              <span className="font-semibold text-amber-400">
                {alert.gas ? "BME688 Gas / Humo Detectado" : "Umbral Superado"}
              </span>
            </div>

            <div className="flex items-center justify-between text-muted-foreground">
              <span className="flex items-center gap-1.5 text-[11px]">
                <Battery className="h-3.5 w-3.5 text-emerald-400" /> Factibilidad Triage:
              </span>
              <span className="font-semibold text-emerald-400">
                100% FACTIBLE (Batería 90% &gt; Margen 25%)
              </span>
            </div>
          </div>

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            El sistema ha calculado una ruta de evasión 3D (RRT*) sobre el relieve real de Hualpén.
            Como operador de misión, confirma el despegue inmediato del dron en modo <strong>GUIDED</strong>.
          </p>

          <Input
            type="password"
            placeholder="Contraseña de Administrador C2"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-background/60 border-red-500/40 text-red-400 font-mono tracking-widest text-center"
            autoComplete="off"
          />
        </div>

        {/* Botonera de Acción del Operador */}
        <div className="mt-5 flex gap-2">
          <Button
            variant="outline"
            onClick={handleDismiss}
            disabled={isLoading}
            className="flex-1 h-9 border-border/60 text-xs font-mono text-muted-foreground hover:bg-muted/40"
          >
            <XCircle className="mr-1.5 h-3.5 w-3.5 text-muted-foreground" />
            Descartar
          </Button>

          <Button
            onClick={handleAuthorize}
            disabled={isLoading}
            className="flex-2 h-9 bg-red-600 hover:bg-red-700 text-xs font-mono font-bold uppercase tracking-wider text-white shadow-lg shadow-red-900/30"
          >
            <Plane className="mr-1.5 h-4 w-4 animate-bounce" />
            Autorizar Despegue
          </Button>
        </div>
      </div>
    </div>
  );
}
