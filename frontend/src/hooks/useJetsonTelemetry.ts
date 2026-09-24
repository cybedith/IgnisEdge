import { useEffect } from "react";
import { useIgnisStore } from "@/store/useIgnisStore";

// Cambia esto a "http://192.168.0.9:8080" cuando conectes la Jetson física por WiFi
const JETSON_URL = "http://192.168.0.9:8080";

export function useJetsonTelemetry(enabled: boolean) {
  const setLiveTick = useIgnisStore((s) => s.setLiveTick);
  const setJetsonFsmState = useIgnisStore((s) => s.setJetsonFsmState);
  const setLatestVerdict = useIgnisStore((s) => s.setLatestVerdict);
  const setPendingFlightAuth = useIgnisStore((s) => s.setPendingFlightAuth);

  const setTelemetryDown = useIgnisStore((s) => s.setTelemetryDown);

  useEffect(() => {
    if (!enabled) return;

    let inFlight = false;
    let seq = 0;

    const tick = async () => {
      if (inFlight) return;
      inFlight = true;
      const mySeq = ++seq;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 300);

      try {
        const res = await fetch(`${JETSON_URL}/telemetry.json`, { signal: controller.signal });
        if (!res.ok || mySeq !== seq) return;
        
        const d = await res.json();
        if (mySeq !== seq) return;

        // Recuperamos la bandera si el enlace acaba de volver
        setTelemetryDown(false);

        // 1. Estado FSM y Veredicto Térmico
        setJetsonFsmState(d.fsm_state);
        setLatestVerdict({
          fire: d.num_fires > 0,
          confidence: d.num_fires > 0 ? 0.98 : null, // Fix: no inventar score si es 0
          max_T: d.max_temp_c,
          min_T: d.min_temp_c,
          n_focos: d.num_fires,
          fps: d.fps,
          thermal_grid: d.thermal_grid // Matriz 16x12 comprimida
        } as any);

        // 2. Despliegue Automático del Modal de Autorización Humana
        if (d.fsm_state === "WAITING_AUTH" && !useIgnisStore.getState().pendingFlightAuth) {
          setPendingFlightAuth({
            node_id: "Jetson_AI_Target",
            threat_level: "HIGH",
            distance_m: d.max_radius_m || 120,
            recommended_action: "AUTHORIZE_GUIDED",
            lat: d.auth_target_lat || d.lat || 0,
            lon: d.auth_target_lon || d.lon || 0
          });
        } else if (d.fsm_state !== "WAITING_AUTH") {
          if (useIgnisStore.getState().pendingFlightAuth?.node_id === "Jetson_AI_Target") {
            setPendingFlightAuth(null);
          }
        }

        // 3. Mapear datos MAVLink para AvionicsHUD y el Mapa 3D
        setLiveTick({
          sim_time_s: Date.now() / 1000,
          node_status: [],
          fires: [],
          drones: [{
            drone_id: "IgnisEdge_Pixhawk",
            latitude: d.lat,
            longitude: d.lon,
            altitude_m: d.alt_m,
            heading_deg: null, // Fix: no forzar 0
            status: d.mode,
            battery_pct: d.battery_pct,
            motors: [d.motor_1_pct, d.motor_2_pct, d.motor_3_pct, d.motor_4_pct]
          }]
        });

      } catch (e) {
        // Enlace caído detectado por el frontend
        setTelemetryDown(true);
      } finally {
        clearTimeout(timeout);
        inFlight = false;
      }
    };

    const interval = setInterval(tick, 333); // 3 FPS para mantener la UI fluida

    return () => clearInterval(interval);
  }, [enabled, setJetsonFsmState, setLatestVerdict, setLiveTick, setPendingFlightAuth, setTelemetryDown]);
}
