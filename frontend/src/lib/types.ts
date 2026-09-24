/**
 * Ignis Edge — shared domain types.
 *
 * 1:1 mirror of the backend Pydantic schemas. Reuse these everywhere; do NOT
 * redefine domain types in components.
 */

export type RiskLevel = "LOW" | "MODERATE" | "HIGH" | "EXTREME";
export type NodeRole = "GATEWAY" | "RELAY" | "EDGE";
export type DroneStatus =
  | "IDLE"
  | "EN_ROUTE"
  | "ARRIVED"
  | "RETURNING"
  | "ABORTED";

export interface GeoJSONPolygon {
  type: "Polygon";
  coordinates: number[][][]; // [[[lng, lat], ...]]
}

export interface FuelRiskAnalysis {
  mean_ndvi: number;
  mean_ndmi: number;
  risk_score: number;
  risk_level: RiskLevel;
  image_date: string;
  cloud_cover: number;
}

export interface TerrainStats {
  bbox: [number, number, number, number];
  rows: number;
  cols: number;
  elev_min_m: number;
  elev_max_m: number;
  elev_mean_m: number;
  slope_max_deg: number;
  slope_mean_deg: number;
}

export interface WeatherSnapshot {
  longitude: number;
  latitude: number;
  forecast_time: string;
  wind_speed_ms: number;
  wind_direction_deg: number;
  temperature_c: number;
  relative_humidity_pct: number;
  source: string;
}

export interface NodeLocation {
  node_id: string;
  longitude: number;
  latitude: number;
  elevation_m: number;
  role: NodeRole;
  detection_radius_m: number;
}

export interface LoSLink {
  from_node: string;
  to_node: string;
  distance_m: number;
  has_los: boolean;
  fresnel_clearance_m: number;
  margin_m: number;
}

export interface NodePlacementOut {
  perimeter_name: string;
  nodes: NodeLocation[];
  links: LoSLink[];
  coverage_score: number;
  connectivity_score: number;
  fitness: number;
}

export interface FirePolygon {
  hot_spot_id: string;
  t_minutes: number;
  area_ha: number;
  perimeter_m: number;
  geojson: GeoJSONPolygon;
}

export interface NodeStatus {
  node_id: string;
  smoke_ppm: number;
  detected: boolean;
  detection_time_s: number | null;
}

export interface TriangulationResult {
  estimated_lon: number;
  estimated_lat: number;
  estimated_t0_s: number;
  uncertainty_radius_m: number;
  confidence: number;
  n_detections_used: number;
  method: string;
}

export interface DroneState {
  drone_id: string;
  longitude: number;
  latitude: number;
  altitude_m: number;
  heading_deg: number;
  status: DroneStatus;
  target_lon: number | null;
  target_lat: number | null;
  progress: number;
}

export interface DroneRoute {
  drone_id: string;
  waypoints: [number, number, number][];
  total_length_m: number;
  estimated_eta_s: number;
  avoided_smoke_zones: number;
  method: string;
}

export interface SimulationState {
  session_id: string;
  name: string;
  running: boolean;
  sim_time_s: number;
  real_time_started: string | null;
  speed_multiplier: number;
  n_hot_spots: number;
  n_drones: number;
}

export interface SystemLogMessage {
  id: string;
  timestamp_s: number;
  source: "LORA" | "PIXHAWK" | "JETSON" | "PERCEPTION";
  message: string;
}

export interface TickMessage {
  type: "tick";
  sim_time_s: number;
  speed_multiplier: number;
  weather: WeatherSnapshot | null;
  fires: FirePolygon[];
  node_status: NodeStatus[];
  triangulation: TriangulationResult | null;
  drones: DroneState[];
  system_logs?: SystemLogMessage[];
}

/* ───────────────── UI helpers ───────────────── */

export type ToolMode =
  | "IDLE"
  | "DRAW_PERIMETER"
  | "PLACE_HOTSPOT"
  | "PLACE_DRONE_TARGET"
  | "PLACE_NODE";

/**
 * Hot spot input — campo alineado con el backend (smoke_emission_g_per_s, radius_m).
 */
export interface HotSpotInput {
  longitude: number;
  latitude: number;
  temperature_c: number;
  smoke_emission_g_per_s: number;
  radius_m?: number;
}

/**
 * Payload to create a simulation. El backend espera `polygon` (no `perimeter`)
 * y `nodes` directamente, sin `perimeter_name` (que va dentro del polygon name).
 */
export interface CreateSimulationPayload {
  name: string;
  polygon: GeoJSONPolygon;
  nodes: NodeLocation[];
  use_real_weather: boolean;
  manual_wind_speed_ms?: number;
  manual_wind_direction_deg?: number;
  speed_multiplier: number;
}

/**
 * Drone launch — backend espera `safe_agl_m`, no `cruise_alt_m`.
 */
export interface DroneLaunchRequest {
  drone_id: string;
  start_lon: number;
  start_lat: number;
  start_alt_m?: number;
  target_lon: number;
  target_lat: number;
  safe_agl_m?: number;
}
