/**
 * Ignis Edge — typed REST client.
 *
 * Rutas y payloads alineados 1:1 con el backend FastAPI:
 *   POST /perimeter/analyze       PerimeterIn         -> FuelRiskAnalysis
 *   POST /terrain/stats           PerimeterIn         -> TerrainStats
 *   GET  /weather/current?lon&lat                     -> WeatherSnapshot
 *   POST /mesh/optimize           MeshOptimizationIn  -> NodePlacementOut
 *   POST /simulation/create       SimulationCreateIn  -> SimulationStateOut
 *   POST /simulation/{id}/start                       -> SimulationStateOut
 *   POST /simulation/{id}/pause                       -> SimulationStateOut
 *   DELETE /simulation/{id}                           -> {deleted}
 *   POST /simulation/{id}/hotspot HotSpotIn           -> {hot_spot_id}
 *   POST /simulation/{id}/drone/launch DroneRouteRequest -> DroneRoute
 *
 * Behaviour:
 *   - Real fetch against VITE_API_BASE_URL (debe incluir /api/v1).
 *   - On network failure (backend down) we fall back to deterministic mock data
 *     so the Lovable preview stays demoable. Mocks are clearly tagged in console.
 */
import * as turf from "@turf/turf";
import type {
  CreateSimulationPayload,
  DroneLaunchRequest,
  DroneRoute,
  FuelRiskAnalysis,
  GeoJSONPolygon,
  HotSpotInput,
  NodePlacementOut,
  SimulationState,
  TerrainStats,
  WeatherSnapshot,
} from "./types";

/**
 * Base URL del backend. Por defecto incluye /api/v1; si el usuario configura
 * otra cosa la respetamos pero adjuntamos /api/v1 si falta.
 */
const RAW_API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000/api/v1";
const API_BASE = RAW_API_BASE.endsWith("/api/v1")
  ? RAW_API_BASE
  : RAW_API_BASE.replace(/\/$/, "") + "/api/v1";

const USE_MOCK_FALLBACK =
  (import.meta.env.VITE_USE_MOCK_FALLBACK as string | undefined) !== "false";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`[${status}] ${detail}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? JSON.stringify(body);
    } catch {
      /* keep status text */
    }
    throw new ApiError(res.status, String(detail));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Wrap a request with a mock fallback for offline / backend-down scenarios. */
async function withMock<T>(
  label: string,
  real: () => Promise<T>,
  mock: () => T,
): Promise<T> {
  try {
    return await real();
  } catch (err) {
    if (!USE_MOCK_FALLBACK) throw err;
    if (err instanceof ApiError) throw err; // real HTTP error → propagate
    // eslint-disable-next-line no-console
    console.warn(`[ignis-api] ${label} → backend unreachable, using MOCK.`, err);
    return mock();
  }
}

/* ───────────────── Endpoints ───────────────── */

export function analyzePerimeter(
  name: string,
  polygon: GeoJSONPolygon,
): Promise<FuelRiskAnalysis> {
  return withMock(
    "analyzePerimeter",
    () =>
      request<FuelRiskAnalysis>("/perimeter/analyze", {
        method: "POST",
        body: JSON.stringify({ name, polygon, crs: "EPSG:4326" }),
      }),
    () => ({
      mean_ndvi: 0.42,
      mean_ndmi: 0.18,
      risk_score: 0.74,
      risk_level: "HIGH",
      image_date: new Date().toISOString().slice(0, 10),
      cloud_cover: 8.3,
    }),
  );
}

export function terrainStats(
  name: string,
  polygon: GeoJSONPolygon,
): Promise<TerrainStats> {
  return withMock(
    "terrainStats",
    () =>
      request<TerrainStats>("/terrain/stats", {
        method: "POST",
        body: JSON.stringify({ name, polygon, crs: "EPSG:4326" }),
      }),
    () => {
      const bbox = turf.bbox(polygon) as [number, number, number, number];
      return {
        bbox,
        rows: 256,
        cols: 256,
        elev_min_m: 35,
        elev_max_m: 412,
        elev_mean_m: 187,
        slope_max_deg: 38.4,
        slope_mean_deg: 11.2,
      };
    },
  );
}

export function getCurrentWeather(
  lon: number,
  lat: number,
): Promise<WeatherSnapshot> {
  return withMock(
    "getCurrentWeather",
    () => request<WeatherSnapshot>(`/weather/current?lon=${lon}&lat=${lat}`),
    () => ({
      longitude: lon,
      latitude: lat,
      forecast_time: new Date().toISOString(),
      wind_speed_ms: 4.2,
      wind_direction_deg: 215,
      temperature_c: 26.4,
      relative_humidity_pct: 38,
      source: "MOCK",
    }),
  );
}

export function optimizeMesh(
  name: string,
  polygon: GeoJSONPolygon,
  n_nodes: number,
  detection_radius_m: number,
): Promise<NodePlacementOut> {
  return withMock(
    "optimizeMesh",
    () =>
      request<NodePlacementOut>("/mesh/optimize", {
        method: "POST",
        body: JSON.stringify({
          name,
          polygon,
          n_nodes,
          detection_radius_m,
        }),
      }),
    () => buildMockMesh(name, polygon, n_nodes, detection_radius_m),
  );
}

export function createSimulation(
  payload: CreateSimulationPayload,
): Promise<SimulationState> {
  return withMock(
    "createSimulation",
    () =>
      request<SimulationState>("/simulation/create", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    () => ({
      session_id: `mock-${Date.now()}`,
      name: payload.name,
      running: false,
      sim_time_s: 0,
      real_time_started: null,
      speed_multiplier: payload.speed_multiplier,
      n_hot_spots: 0,
      n_drones: 0,
    }),
  );
}

export function startSimulation(session_id: string): Promise<SimulationState> {
  return withMock(
    "startSimulation",
    () =>
      request<SimulationState>(`/simulation/${session_id}/start`, {
        method: "POST",
      }),
    () => ({
      session_id,
      name: "Mock Sim",
      running: true,
      sim_time_s: 0,
      real_time_started: new Date().toISOString(),
      speed_multiplier: 30,
      n_hot_spots: 0,
      n_drones: 0,
    }),
  );
}

export function pauseSimulation(session_id: string): Promise<SimulationState> {
  return withMock(
    "pauseSimulation",
    () =>
      request<SimulationState>(`/simulation/${session_id}/pause`, {
        method: "POST",
      }),
    () => ({
      session_id,
      name: "Mock Sim",
      running: false,
      sim_time_s: 0,
      real_time_started: null,
      speed_multiplier: 30,
      n_hot_spots: 0,
      n_drones: 0,
    }),
  );
}

export function deleteSimulation(session_id: string): Promise<void> {
  return withMock(
    "deleteSimulation",
    () =>
      request<void>(`/simulation/${session_id}`, { method: "DELETE" }),
    () => undefined,
  );
}

export function addHotSpot(
  session_id: string,
  hotspot: HotSpotInput,
): Promise<{ hot_spot_id: string }> {
  // Aseguramos los nombres de campo que espera el backend.
  const body = {
    longitude: hotspot.longitude,
    latitude: hotspot.latitude,
    temperature_c: hotspot.temperature_c,
    smoke_emission_g_per_s: hotspot.smoke_emission_g_per_s,
    radius_m: hotspot.radius_m ?? 5.0,
  };
  return withMock(
    "addHotSpot",
    () =>
      request<{ hot_spot_id: string }>(
        `/simulation/${session_id}/hotspot`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    () => ({ hot_spot_id: `hs-${Date.now()}` }),
  );
}

export function launchDrone(
  session_id: string,
  req: DroneLaunchRequest,
): Promise<DroneRoute> {
  const body = {
    drone_id: req.drone_id,
    start_lon: req.start_lon,
    start_lat: req.start_lat,
    start_alt_m: req.start_alt_m,
    target_lon: req.target_lon,
    target_lat: req.target_lat,
    safe_agl_m: req.safe_agl_m ?? 80.0,
  };
  return withMock(
    "launchDrone",
    () =>
      request<DroneRoute>(`/simulation/${session_id}/drone/launch`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    () => ({
      drone_id: req.drone_id,
      waypoints: [
        [req.start_lon, req.start_lat, req.start_alt_m ?? 100],
        [
          (req.start_lon + req.target_lon) / 2,
          (req.start_lat + req.target_lat) / 2,
          (req.start_alt_m ?? 100) + (req.safe_agl_m ?? 80),
        ],
        [req.target_lon, req.target_lat, (req.start_alt_m ?? 100) + (req.safe_agl_m ?? 80)],
      ],
      total_length_m: 1850,
      estimated_eta_s: 240,
      avoided_smoke_zones: 1,
      method: "MOCK_RRT*",
    }),
  );
}

export function addSimulationNode(
  session_id: string,
  node: import("./types").NodeLocation,
): Promise<{ status: string; node: import("./types").NodeLocation }> {
  return withMock(
    "addSimulationNode",
    () =>
      request<{ status: string; node: import("./types").NodeLocation }>(
        `/simulation/${session_id}/node`,
        { method: "POST", body: JSON.stringify(node) },
      ),
    () => ({ status: "ok", node }),
  );
}

export function triggerNodeAlert(
  session_id: string,
  node_id: string,
): Promise<{ status: string; node_id: string; route: DroneRoute }> {
  return withMock(
    "triggerNodeAlert",
    () =>
      request<{ status: string; node_id: string; route: DroneRoute }>(
        `/simulation/${session_id}/node/${node_id}/alert`,
        { method: "POST" },
      ),
    () => ({
      status: "alert_triggered",
      node_id,
      route: {
        drone_id: "DRONE_1",
        waypoints: [[-73.065, -36.795, 100], [-73.05, -36.79, 120]],
        total_length_m: 1200,
        estimated_eta_s: 160,
        avoided_smoke_zones: 0,
        method: "RRT*",
      },
    }),
  );
}

export function sendDroneCommand(
  session_id: string,
  drone_id: string,
  command: string,
): Promise<{ status: string; drone_id: string; command: string }> {
  return withMock(
    "sendDroneCommand",
    () =>
      request<{ status: string; drone_id: string; command: string }>(
        `/simulation/${session_id}/drone/${drone_id}/command`,
        { method: "POST", body: JSON.stringify({ command }) },
      ),
    () => ({ status: "ok", drone_id, command }),
  );
}

export { ApiError };

/* ───────────────── Mock helpers ───────────────── */

function buildMockMesh(
  name: string,
  polygon: GeoJSONPolygon,
  n_nodes: number,
  detection_radius_m: number,
): NodePlacementOut {
  const ring = polygon.coordinates[0];
  const cx = ring.reduce((a, [x]) => a + x, 0) / ring.length;
  const cy = ring.reduce((a, [, y]) => a + y, 0) / ring.length;
  const bbox = turf.bbox(polygon);
  const rx = (bbox[2] - bbox[0]) * 0.32;
  const ry = (bbox[3] - bbox[1]) * 0.32;

  const nodes = Array.from({ length: n_nodes }).map((_, i) => {
    const a = (i / n_nodes) * Math.PI * 2;
    const role = i === 0 ? "GATEWAY" : i % 3 === 0 ? "RELAY" : "EDGE";
    return {
      node_id: `N-${String(i + 1).padStart(2, "0")}`,
      longitude: cx + Math.cos(a) * rx,
      latitude: cy + Math.sin(a) * ry,
      elevation_m: 120 + Math.round(Math.random() * 80),
      role: role as "GATEWAY" | "RELAY" | "EDGE",
      detection_radius_m,
    };
  });

  const links = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const d =
        turf.distance(
          [nodes[i].longitude, nodes[i].latitude],
          [nodes[j].longitude, nodes[j].latitude],
        ) * 1000;
      if (d < detection_radius_m * 1.6) {
        links.push({
          from_node: nodes[i].node_id,
          to_node: nodes[j].node_id,
          distance_m: d,
          has_los: Math.random() > 0.2,
          fresnel_clearance_m: 8,
          margin_m: 4,
        });
      }
    }
  }

  return {
    perimeter_name: name,
    nodes,
    links,
    coverage_score: 0.78,
    connectivity_score: 0.86,
    fitness: 0.82,
  };
}
