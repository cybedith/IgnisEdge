/**
 * BANCO DE PRUEBAS HITL / PRODUCCIÓN — IgnisEdge
 * Control de vuelo MAVLink/Pixhawk, gestión y vinculación de nodos LoRa físicos,
 * autorización de despegue por operador humano y veredicto IA en vivo.
 */
import { useState, useEffect } from "react";
import {
  Plane,
  Radio,
  Flame,
  Activity,
  Cpu,
  RefreshCw,
  Send,
  Plus,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Pause,
  ArrowDownCircle,
  Navigation,
  Link2,
  Check,
  ShieldAlert,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { useIgnisStore } from "@/store/useIgnisStore";
import {
  triggerNodeAlert,
  addSimulationNode,
  sendDroneCommand,
} from "@/lib/api";
import type { NodeLocation } from "@/lib/types";

const JETSON_HOST = "http://192.168.0.9:8000";

interface JetsonState {
  connected: boolean;
  mission_state?: string;
  pix_link?: boolean;
  battery?: number | null;
  events?: string[];
  nodes?: any[];
  flight_progress?: number;
  inspection_progress?: number;
  pending_auth?: any;
}

export function StageWorkbench() {
  const session = useIgnisStore((s) => s.session);
  const liveTick = useIgnisStore((s) => s.liveTick);
  const nodes = useIgnisStore((s) => s.nodes);
  const addNode = useIgnisStore((s) => s.addNode);
  const removeNode = useIgnisStore((s) => s.removeNode);
  const setToolMode = useIgnisStore((s) => s.setToolMode);
  const addDroneRoute = useIgnisStore((s) => s.addDroneRoute);

  const pendingFlightAuth = useIgnisStore((s) => s.pendingFlightAuth);
  const setPendingFlightAuth = useIgnisStore((s) => s.setPendingFlightAuth);
  const boundPhysicalNodeId = useIgnisStore((s) => s.boundPhysicalNodeId);
  const setBoundPhysicalNodeId = useIgnisStore((s) => s.setBoundPhysicalNodeId);
  const latestVerdict = useIgnisStore((s) => s.latestVerdict);
  const setLatestVerdict = useIgnisStore((s) => s.setLatestVerdict);
  const setJetsonFsmState = useIgnisStore((s) => s.setJetsonFsmState);

  const [jetsonState, setJetsonState] = useState<JetsonState>({ connected: false });
  const [isPolling, setIsPolling] = useState(true);
  const [manualNodeLat, setManualNodeLat] = useState("");
  const [manualNodeLon, setManualNodeLon] = useState("");
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Poll Jetson Orin Nano hardware state
  useEffect(() => {
    if (!isPolling) return;
    const fetchJetson = async () => {
      try {
        const res = await fetch(`${JETSON_HOST}/state`, { signal: AbortSignal.timeout(1500) });
        if (res.ok) {
          const data = await res.json();
          setJetsonState({
            connected: true,
            mission_state: data.mission_state ?? "READY",
            pix_link: !!data.pix_link,
            battery: data.battery,
            events: data.events ?? [],
            nodes: data.nodes ?? [],
            flight_progress: data.flight_progress ?? 0,
            inspection_progress: data.inspection_progress ?? 0,
            pending_auth: data.pending_auth,
          });

          setJetsonFsmState(data.mission_state ?? "READY");

          if (data.verdict && data.verdict.fire !== undefined) {
            setLatestVerdict(data.verdict);
          }

          // Si la Jetson detectó alerta física por LoRa y está esperando autorización
          if (data.pending_auth && !pendingFlightAuth) {
            const authNode = nodes.find((n) => n.node_id === boundPhysicalNodeId) ?? nodes[0];
            setPendingFlightAuth({
              node_id: boundPhysicalNodeId,
              lat: authNode ? authNode.latitude : (data.pending_auth.lat ?? -36.8205),
              lon: authNode ? authNode.longitude : (data.pending_auth.lon ?? -73.0445),
              gas: data.pending_auth.gas ?? true,
              heat: data.pending_auth.heat ?? true,
              confidence: data.pending_auth.confidence ?? 220,
              source: "LORA_PHYSICAL",
            });
          }
        }
      } catch {
        setJetsonState((prev) => ({ ...prev, connected: false }));
      }
    };

    fetchJetson();
    const interval = setInterval(fetchJetson, 1500);
    return () => clearInterval(interval);
  }, [isPolling, session, pendingFlightAuth, boundPhysicalNodeId, nodes, setJetsonFsmState, setLatestVerdict, setPendingFlightAuth]);

  const activeDrone = liveTick?.drones?.[0];
  const nodeStatusMap = new Map((liveTick?.node_status ?? []).map((s) => [s.node_id, s]));

  // Disparar Alerta / Solicitar Autorización de Operador
  const handleTriggerAlert = (nodeId: string) => {
    const targetNode = nodes.find((n) => n.node_id === nodeId);
    if (!targetNode) return;
    setPendingFlightAuth({
      node_id: nodeId,
      lat: targetNode.latitude,
      lon: targetNode.longitude,
      gas: true,
      heat: true,
      confidence: 240,
      source: nodeId === boundPhysicalNodeId ? "LORA_PHYSICAL" : "SIMULATION",
    });
  };

  // Simular disparo físico desde el modem Heltec
  const handleSimulatePhysicalLoRa = async () => {
    setActionLoading("test-lora");
    try {
      await fetch(`${JETSON_HOST}/test_alert`);
      toast.info("Trama LoRa de prueba inyectada en Jetson. Esperando FSM Triage...");
    } catch (e) {
      toast.error(`Error inyectando alerta: ${(e as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Vincular Hardware Heltec Real #1 a un nodo
  const handleBindHardwareNode = async (nodeId: string) => {
    setBoundPhysicalNodeId(nodeId);
    const node = nodes.find((n) => n.node_id === nodeId);
    if (node) {
      try {
        await fetch(`${JETSON_HOST}/bind_node?node_id=1&lat=${node.latitude}&lon=${node.longitude}`);
        toast.success(`🔗 Heltec #1 vinculado a ${nodeId} (${node.latitude.toFixed(4)}, ${node.longitude.toFixed(4)})`);
      } catch {
        toast.info(`Vinculado a ${nodeId} localmente`);
      }
    }
  };

  // Comandos directos de vuelo al Dron
  const handleDroneCmd = async (cmd: "RTL" | "LOITER" | "LAND" | "GUIDED") => {
    if (!session || !activeDrone) {
      toast.error("Dron o sesión no disponible");
      return;
    }
    setActionLoading(`cmd-${cmd}`);
    try {
      await sendDroneCommand(session.session_id, activeDrone.drone_id, cmd);
      toast.success(`Comando ${cmd} ejecutado en ${activeDrone.drone_id}`);
    } catch (e) {
      toast.error(`Error enviando comando: ${(e as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Agregar nodo manual por coordenadas
  const handleAddManualNode = async () => {
    const lat = parseFloat(manualNodeLat);
    const lon = parseFloat(manualNodeLon);
    if (isNaN(lat) || isNaN(lon)) {
      toast.error("Coordenadas inválidas");
      return;
    }
    const nodeId = `NODE_${nodes.length + 1}`;
    const newNode: NodeLocation = {
      node_id: nodeId,
      longitude: lon,
      latitude: lat,
      elevation_m: 0,
      role: nodes.length === 0 ? "GATEWAY" : "EDGE",
      detection_radius_m: 1500,
    };
    addNode(newNode);
    if (session) {
      try {
        await addSimulationNode(session.session_id, newNode);
        toast.success(`Nodo #${nodeId} registrado`);
      } catch (err) {
        toast.error(`Error guardando en backend: ${(err as Error).message}`);
      }
    }
    setManualNodeLat("");
    setManualNodeLon("");
  };

  return (
    <div className="space-y-4 font-sans text-xs">
      {/* ── 1. ESTADO FÍSICO DE HARDWARE ── */}
      <div className="rounded-lg border border-border/60 bg-card/40 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-primary">
            <Cpu className="h-3.5 w-3.5" />
            <span>BANCO HITL JETSON ORIN NANO</span>
          </div>
          <Badge
            variant={jetsonState.connected ? "default" : "outline"}
            className={`text-[9px] uppercase ${
              jetsonState.connected
                ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                : "text-muted-foreground"
            }`}
          >
            {jetsonState.connected ? "ONLINE (192.168.0.9)" : "BENCH OFFLINE"}
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="rounded border border-border/40 bg-background/50 p-2">
            <div className="flex items-center justify-between text-muted-foreground">
              <span>Pixhawk 6C (/dev/ttyACM0)</span>
              <span className={`font-mono font-semibold ${jetsonState.pix_link ? "text-emerald-400" : "text-amber-400"}`}>
                {jetsonState.pix_link ? "MAVLink OK" : "LOITER SITL"}
              </span>
            </div>
            <div className="mt-1 flex items-center justify-between font-mono text-[9px] text-muted-foreground">
              <span>Estado FSM:</span>
              <span className="text-cyan-400 font-bold">
                {jetsonState.mission_state ?? "READY"}
              </span>
            </div>
          </div>

          <div className="rounded border border-border/40 bg-background/50 p-2">
            <div className="flex items-center justify-between text-muted-foreground">
              <span>Heltec LoRa (/dev/ttyACM1)</span>
              <span className="font-mono text-emerald-400 font-semibold">915 MHz</span>
            </div>
            <div className="mt-1 flex items-center justify-between font-mono text-[9px] text-muted-foreground">
              <span>Hardware Vinculado:</span>
              <span className="text-foreground font-bold">
                {boundPhysicalNodeId}
              </span>
            </div>
          </div>
        </div>

        {/* Botón de Inyección Rápida de Prueba */}
        <Button
          size="sm"
          variant="outline"
          disabled={actionLoading === "test-lora"}
          onClick={handleSimulatePhysicalLoRa}
          className="mt-2.5 h-7 w-full border-amber-500/40 text-amber-400 hover:bg-amber-500/10 text-[10px] font-mono"
        >
          <Flame className="mr-1.5 h-3.5 w-3.5 text-amber-400" />
          Disparar Alerta RF de Sensor Físico (Test Heltec)
        </Button>
      </div>

      {/* ── 2. CONTROL DE VUELO Y AUTORIZACIÓN DEL DRON (HITL) ── */}
      <div className="rounded-lg border border-border/60 bg-card/40 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-cyan-400">
            <Plane className="h-3.5 w-3.5" />
            <span>CONTROL DE VUELO PIXHAWK</span>
          </div>
          <Badge variant="outline" className="font-mono text-[9px] text-cyan-400 border-cyan-500/40">
            {jetsonState.mission_state ?? activeDrone?.flight_mode ?? "READY"}
          </Badge>
        </div>

        {activeDrone ? (
          <div className="space-y-2.5">
            <div className="grid grid-cols-3 gap-1.5 rounded border border-border/40 bg-background/40 p-2 font-mono text-[10px]">
              <div>
                <span className="text-[9px] text-muted-foreground">ALT AMSL</span>
                <p className="text-foreground font-bold">{activeDrone.alt_amsl_m?.toFixed(0) ?? activeDrone.altitude_m.toFixed(0)} m</p>
              </div>
              <div>
                <span className="text-[9px] text-muted-foreground">ALT AGL</span>
                <p className="text-cyan-400 font-bold">{activeDrone.alt_agl_m?.toFixed(0) ?? "80"} m</p>
              </div>
              <div>
                <span className="text-[9px] text-muted-foreground">RELIEVE DEM</span>
                <p className="text-foreground font-bold">{activeDrone.terrain_elev_m?.toFixed(0) ?? "--"} m</p>
              </div>
            </div>

            {/* Progreso del Waypoint actual */}
            <div className="space-y-1">
              <div className="flex justify-between font-mono text-[9px] text-muted-foreground">
                <span>Ruta RRT* 3D (Copernicus DEM)</span>
                <span>{Math.round(activeDrone.progress * 100)}%</span>
              </div>
              <Progress value={activeDrone.progress * 100} className="h-1 bg-muted/40" />
            </div>

            {/* Botonera de Vuelo Táctico */}
            <div className="grid grid-cols-3 gap-1.5 pt-1">
              <Button
                size="sm"
                variant="outline"
                disabled={actionLoading === "cmd-RTL"}
                onClick={() => handleDroneCmd("RTL")}
                className="h-8 border-amber-500/40 text-amber-400 hover:bg-amber-500/10 text-[10px]"
              >
                <RotateCcw className="mr-1 h-3 w-3" /> RTL
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={actionLoading === "cmd-LOITER"}
                onClick={() => handleDroneCmd("LOITER")}
                className="h-8 border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/10 text-[10px]"
              >
                <Pause className="mr-1 h-3 w-3" /> LOITER
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={actionLoading === "cmd-LAND"}
                onClick={() => handleDroneCmd("LAND")}
                className="h-8 border-red-500/40 text-red-400 hover:bg-red-500/10 text-[10px]"
              >
                <ArrowDownCircle className="mr-1 h-3 w-3" /> LAND
              </Button>
            </div>
          </div>
        ) : (
          <div className="rounded border border-dashed border-border/60 p-3 text-center text-muted-foreground">
            <p className="text-[11px]">Dron en hangar (Standby)</p>
            <Button
              size="sm"
              onClick={() => setToolMode("PLACE_DRONE_TARGET")}
              className="mt-2 w-full bg-cyan-600/80 hover:bg-cyan-600 text-[10px]"
            >
              <Navigation className="mr-1.5 h-3.5 w-3.5" /> Seleccionar destino en mapa
            </Button>
          </div>
        )}
      </div>

      {/* ── 3. GESTIÓN Y VINCULACIÓN DE NODOS LORA ── */}
      <div className="rounded-lg border border-border/60 bg-card/40 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-emerald-400">
            <Radio className="h-3.5 w-3.5" />
            <span>NODOS LORA ({nodes.length})</span>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setToolMode("PLACE_NODE")}
            className="h-6 px-2 text-[10px] text-emerald-400 hover:bg-emerald-500/10"
          >
            <Plus className="mr-1 h-3 w-3" /> Click Mapa
          </Button>
        </div>

        {nodes.length === 0 ? (
          <div className="rounded border border-dashed border-border/60 p-3 text-center text-muted-foreground">
            No hay nodos registrados. Haz click en el mapa o agrega coordenadas abajo.
          </div>
        ) : (
          <div className="max-h-56 space-y-1.5 overflow-y-auto pr-1">
            {nodes.map((node) => {
              const status = nodeStatusMap.get(node.node_id);
              const isAlerting = status?.detected || (status?.smoke_ppm ?? 0) > 50;
              const isBound = node.node_id === boundPhysicalNodeId;

              return (
                <div
                  key={node.node_id}
                  className={`flex flex-col gap-1.5 rounded border p-2 text-[10px] transition-colors ${
                    isAlerting
                      ? "border-red-500/60 bg-red-500/10"
                      : isBound
                      ? "border-emerald-500/50 bg-emerald-500/5"
                      : "border-border/40 bg-background/40"
                  }`}
                >
                  <div className="flex items-center justify-between font-mono">
                    <div className="flex items-center gap-1.5">
                      <span className="font-bold text-foreground">{node.node_id}</span>
                      <Badge variant="outline" className="text-[8px] uppercase">
                        {node.role}
                      </Badge>
                      {isBound && (
                        <Badge variant="default" className="h-4 bg-emerald-600 text-[8px] uppercase">
                          🔗 Físico Heltec #1
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] text-muted-foreground">
                        {node.latitude.toFixed(4)}, {node.longitude.toFixed(4)}
                      </span>
                      <button
                        onClick={() => removeNode(node.node_id)}
                        className="text-muted-foreground hover:text-red-400"
                        title="Eliminar nodo"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between font-mono text-[9px]">
                    <span className="text-muted-foreground">
                      Gas / Humo:{" "}
                      <strong className={isAlerting ? "text-red-400" : "text-foreground"}>
                        {status ? `${status.smoke_ppm.toFixed(0)} ppm` : "Limpio"}
                      </strong>
                    </span>
                    <span>Elev: {node.elevation_m.toFixed(0)}m DEM</span>
                  </div>

                  <div className="grid grid-cols-2 gap-1 mt-0.5">
                    {/* Botón de vinculación */}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleBindHardwareNode(node.node_id)}
                      className={`h-6 text-[9px] font-mono ${
                        isBound ? "border-emerald-500/60 text-emerald-400" : "text-muted-foreground"
                      }`}
                    >
                      <Link2 className="mr-1 h-2.5 w-2.5" />
                      {isBound ? "Sensor Vinculado" : "Vincular Heltec"}
                    </Button>

                    {/* Disparador de Alerta / Autorización */}
                    <Button
                      size="sm"
                      variant={isAlerting ? "destructive" : "outline"}
                      onClick={() => handleTriggerAlert(node.node_id)}
                      className="h-6 text-[9px] font-mono uppercase"
                    >
                      <Flame className="mr-1 h-2.5 w-2.5" />
                      {isAlerting ? "Alerta Activa" : "Simular Alerta"}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Input para agregar nodo por coordenadas */}
        <div className="mt-3 border-t border-border/40 pt-2 space-y-1.5">
          <span className="text-[9px] font-mono text-muted-foreground uppercase">Agregar nodo manual</span>
          <div className="flex gap-1.5">
            <Input
              placeholder="Lat (ej -36.795)"
              value={manualNodeLat}
              onChange={(e) => setManualNodeLat(e.target.value)}
              className="h-7 text-[10px] font-mono"
            />
            <Input
              placeholder="Lon (ej -73.065)"
              value={manualNodeLon}
              onChange={(e) => setManualNodeLon(e.target.value)}
              className="h-7 text-[10px] font-mono"
            />
            <Button
              size="sm"
              onClick={handleAddManualNode}
              className="h-7 px-2.5 bg-emerald-600/80 hover:bg-emerald-600 text-[10px]"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
