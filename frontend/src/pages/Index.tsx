import { useEffect } from "react";
import { Sidebar } from "@/components/ignis/Sidebar";
import { MapContainer } from "@/components/ignis/MapContainer";
import { TopBar } from "@/components/ignis/TopBar";
import { FloatingHUD } from "@/components/ignis/FloatingHUD";
import { ToolDock } from "@/components/ignis/ToolDock";
import { TimelinePanel } from "@/components/ignis/TimelinePanel";
import { ThermalHUD } from "@/components/ignis/ThermalHUD";
import { AvionicsHUD } from "@/components/ignis/AvionicsHUD";
import { SystemLogsPanel } from "@/components/ignis/SystemLogsPanel";
import { FlightAuthDialog } from "@/components/ignis/FlightAuthDialog";
import { VerdictHUD } from "@/components/ignis/VerdictHUD";
import { useIgnisStore } from "@/store/useIgnisStore";
import { useSimulationWS } from "@/hooks/useSimulationWS";
import { useTelemetryWS } from "@/hooks/useTelemetryWS";
import { useJetsonTelemetry } from "@/hooks/useJetsonTelemetry";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { toast } from "sonner";

import { createSimulation, startSimulation, triggerNodeAlert } from "@/lib/api";

const Index = () => {
  // Force dark mode (strict)
  useEffect(() => {
    document.documentElement.classList.add("dark");
    document.title = "Ignis Edge C2 — Hualpén";
  }, []);

  const systemMode = useIgnisStore((s) => s.systemMode);
  const session = useIgnisStore((s) => s.session);
  const sessionId = session?.session_id ?? null;
  const setSession = useIgnisStore((s) => s.setSession);
  const setNodes = useIgnisStore((s) => s.setNodes);
  const setPerimeter = useIgnisStore((s) => s.setPerimeter);

  // Auto-inicializar sesión táctica de vuelo si no hay una activa
  useEffect(() => {
    if (session) return;
    const defaultPoly: [number, number][] = [
      [-73.18, -36.75],
      [-73.02, -36.75],
      [-73.02, -36.85],
      [-73.18, -36.85],
      [-73.18, -36.75],
    ];
    setPerimeter(defaultPoly);
    const defaultNodes = [
      {
        node_id: "BASE_GATEWAY",
        longitude: -73.065,
        latitude: -36.795,
        elevation_m: 60,
        role: "GATEWAY" as const,
        detection_radius_m: 1500,
      },
      {
        node_id: "NODE_1",
        longitude: -73.0445,
        latitude: -36.8205,
        elevation_m: 85,
        role: "EDGE" as const,
        detection_radius_m: 1500,
      },
      {
        node_id: "NODE_2",
        longitude: -73.0820,
        latitude: -36.8110,
        elevation_m: 95,
        role: "EDGE" as const,
        detection_radius_m: 1500,
      },
    ];
    setNodes(defaultNodes);

    createSimulation({
      name: "Misión Táctica Hualpén",
      polygon: { type: "Polygon", coordinates: [defaultPoly] },
      nodes: defaultNodes,
      speed_multiplier: 1,
    })
      .then((created) => startSimulation(created.session_id))
      .then((started) => {
        setSession(started);
      })
      .catch((e) => {
        console.warn("Auto-bootstrap session notice:", e);
      });
  }, [session, setSession, setNodes, setPerimeter]);

  // WebSocket plumbing — only one channel active at a time depending on mode.
  const { connected: simConnected } = useSimulationWS(
    systemMode === "SIMULATION" ? sessionId : null,
  );
  // Reemplazamos la conexión maestra por nuestra API local de la Jetson AI Edge
  useJetsonTelemetry(systemMode === "LIVE_HARDWARE");
  const wsConnected = systemMode === "SIMULATION" ? simConnected : true;

  useKeyboardShortcuts();

  const pendingFlightAuth = useIgnisStore((s) => s.pendingFlightAuth);
  const setPendingFlightAuth = useIgnisStore((s) => s.setPendingFlightAuth);
  const latestVerdict = useIgnisStore((s) => s.latestVerdict);
  const jetsonFsmState = useIgnisStore((s) => s.jetsonFsmState);
  const addDroneRoute = useIgnisStore((s) => s.addDroneRoute);

  const handleAuthorizeFlight = async (password: string) => {
    if (!session || !pendingFlightAuth) return;
    const targetNodeId = String(pendingFlightAuth.node_id);
    setPendingFlightAuth(null);
    try {
      // Disparamos autorización local directo al cerebro del Dron Edge AI
      const jetsonRes = await fetch("http://192.168.0.9:8080/api/authorize", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      const jetsonAns = await jetsonRes.json();
      if (jetsonAns.status !== "ok") {
        toast.error(`Jetson rechazó autorización: ${jetsonAns.message}`);
        return;
      }
      const res = await triggerNodeAlert(session.session_id, targetNodeId);
      if (res.route) {
        addDroneRoute(res.route);
      }
      toast.success(`🚀 Vuelo AUTORIZADO por operador hacia ${targetNodeId}. Dron en despegue GUIDED.`);
    } catch (e) {
      toast.error(`Error autorizando vuelo: ${(e as Error).message}`);
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
      <TopBar wsConnected={wsConnected} />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="relative flex-1">
          <MapContainer />
          <FloatingHUD />
          <ToolDock />
          <TimelinePanel />
          <ThermalHUD />
          <AvionicsHUD />
          <SystemLogsPanel />
          <VerdictHUD verdict={latestVerdict} missionState={jetsonFsmState} />
          <FlightAuthDialog
            alert={pendingFlightAuth}
            onAuthorize={handleAuthorizeFlight}
            onDismiss={() => setPendingFlightAuth(null)}
          />
        </main>
      </div>
    </div>
  );
};

export default Index;
