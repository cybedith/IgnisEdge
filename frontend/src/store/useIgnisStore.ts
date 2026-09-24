import { create } from "zustand";
import type {
  DroneRoute,
  FuelRiskAnalysis,
  HotSpotInput,
  LoSLink,
  NodeLocation,
  NodePlacementOut,
  SimulationState,
  TerrainStats,
  TickMessage,
  ToolMode,
  TriangulationResult,
} from "@/lib/types";

export type SystemMode = "SIMULATION" | "LIVE_HARDWARE";

export interface EnvironmentParams {
  windSpeed: number; // m/s
  windDirection: number; // degrees 0-360
  temperature: number; // °C
}

export interface MeshConfig {
  n_nodes: number;
  detection_radius_m: number;
}

export interface LocalHotSpot extends HotSpotInput {
  id: string;
}

/**
 * Ring buffer of recent ticks for the timeline charts (smoke ppm vs time).
 */
export interface TickSample {
  t: number;
  byNode: Record<string, number>;
  area_ha: number;
}
const TICK_BUFFER_MAX = 200;

interface IgnisState {
  /* ── Mode ── */
  systemMode: SystemMode;
  setSystemMode: (m: SystemMode) => void;

  /* ── Environment (manual sliders for SIMULATION fallback) ── */
  environmentParams: EnvironmentParams;
  setWindSpeed: (v: number) => void;
  setWindDirection: (v: number) => void;
  setTemperature: (v: number) => void;
  setEnvironmentParams: (p: Partial<EnvironmentParams>) => void;

  /* ── Perimeter & terrain ── */
  perimeterName: string;
  perimeter: [number, number][];
  fuelRisk: FuelRiskAnalysis | null;
  terrainStats: TerrainStats | null;
  setPerimeter: (p: [number, number][]) => void;
  setPerimeterName: (n: string) => void;
  setFuelRisk: (r: FuelRiskAnalysis | null) => void;
  setTerrainStats: (s: TerrainStats | null) => void;

  /* ── Mesh ── */
  meshConfig: MeshConfig;
  meshResult: NodePlacementOut | null;
  nodes: NodeLocation[];
  links: LoSLink[];
  setMeshConfig: (c: Partial<MeshConfig>) => void;
  setMeshResult: (r: NodePlacementOut | null) => void;
  /** @deprecated retro-compat with v0; prefer optimizeMesh() + setMeshResult(). */
  deployNodes: (perimeter: [number, number][]) => void;
  setNodes: (nodes: NodeLocation[]) => void;
  addNode: (node: NodeLocation) => void;
  removeNode: (node_id: string) => void;

  /* ── Simulation ── */
  session: SimulationState | null;
  liveTick: TickMessage | null;
  tickBuffer: TickSample[];
  hotSpotsLocal: LocalHotSpot[];
  triangulation: TriangulationResult | null;
  droneRoutes: DroneRoute[];
  setSession: (s: SimulationState | null) => void;
  setLiveTick: (t: TickMessage) => void;
  addHotSpotLocal: (h: LocalHotSpot) => void;
  removeHotSpotLocal: (id: string) => void;
  setTriangulation: (t: TriangulationResult | null) => void;
  clearTriangulation: () => void;
  addDroneRoute: (r: DroneRoute) => void;
  removeDroneRoute: (id: string) => void;
  clearDroneRoutes: () => void;
  clearHotSpotsLocal: () => void;

  /* ── Hardware & Flight Authorization ── */
  pendingFlightAuth: import("@/components/ignis/FlightAuthDialog").PendingAlertData | null;
  setPendingFlightAuth: (a: import("@/components/ignis/FlightAuthDialog").PendingAlertData | null) => void;
  boundPhysicalNodeId: string;
  setBoundPhysicalNodeId: (id: string) => void;
  latestVerdict: import("@/components/ignis/VerdictHUD").VerdictData | null;
  setLatestVerdict: (v: import("@/components/ignis/VerdictHUD").VerdictData | null) => void;
  jetsonFsmState: string;
  setJetsonFsmState: (s: string) => void;

  /* ── UI ── */
  toolMode: ToolMode;
  selectedNodeId: string | null;
  setToolMode: (m: ToolMode) => void;
  setSelectedNodeId: (id: string | null) => void;

  /* ── Layer visibility ── */
  layerVisibility: {
    nodes: boolean;
    links: boolean;
    fires: boolean;
    smoke: boolean;
    drones: boolean;
  };
  toggleLayer: (key: keyof IgnisState["layerVisibility"]) => void;
  isTelemetryDown: boolean;
  setTelemetryDown: (status: boolean) => void;
  reset: () => void;
}

const INITIAL_ENV: EnvironmentParams = {
  windSpeed: 0,
  windDirection: 0,
  temperature: 15,
};
const INITIAL_MESH: MeshConfig = { n_nodes: 6, detection_radius_m: 1500 };

export const useIgnisStore = create<IgnisState>((set, get) => ({
  systemMode: "SIMULATION",
  setSystemMode: (systemMode) => set({ systemMode }),

  environmentParams: INITIAL_ENV,
  setWindSpeed: (v) =>
    set((s) => ({ environmentParams: { ...s.environmentParams, windSpeed: v } })),
  setWindDirection: (v) =>
    set((s) => ({ environmentParams: { ...s.environmentParams, windDirection: v } })),
  setTemperature: (v) =>
    set((s) => ({ environmentParams: { ...s.environmentParams, temperature: v } })),
  setEnvironmentParams: (params) =>
    set((s) => ({ environmentParams: { ...s.environmentParams, ...params } })),

  perimeterName: "Sector Hualpén",
  perimeter: [],
  fuelRisk: null,
  terrainStats: null,
  setPerimeter: (perimeter) => set({ perimeter }),
  setPerimeterName: (perimeterName) => set({ perimeterName }),
  setFuelRisk: (fuelRisk) => set({ fuelRisk }),
  setTerrainStats: (terrainStats) => set({ terrainStats }),

  meshConfig: INITIAL_MESH,
  meshResult: null,
  nodes: [],
  links: [],
  setMeshConfig: (c) => set((s) => ({ meshConfig: { ...s.meshConfig, ...c } })),
  setMeshResult: (meshResult) =>
    set({
      meshResult,
      nodes: meshResult?.nodes ?? [],
      links: meshResult?.links ?? [],
    }),
  setNodes: (nodes) => set({ nodes }),
  addNode: (node) =>
    set((s) => ({
      nodes: [...s.nodes.filter((n) => n.node_id !== node.node_id), node],
    })),
  removeNode: (node_id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.node_id !== node_id),
      links: s.links.filter((l) => l.from_node !== node_id && l.to_node !== node_id),
    })),
  deployNodes: (perimeter) => {
    if (!perimeter || perimeter.length < 3) return;
    const cx = perimeter.reduce((a, [x]) => a + x, 0) / perimeter.length;
    const cy = perimeter.reduce((a, [, y]) => a + y, 0) / perimeter.length;
    let minLng = Infinity, maxLng = -Infinity, minLat = Infinity, maxLat = -Infinity;
    for (const [lng, lat] of perimeter) {
      if (lng < minLng) minLng = lng; if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat;
    }
    const offX = Math.max((maxLng - minLng) * 0.22, 0.0008);
    const offY = Math.max((maxLat - minLat) * 0.22, 0.0008);
    const layout: Array<[number, number]> = [
      [cx + offX, cy + offY],
      [cx - offX, cy + offY],
      [cx - offX, cy - offY],
      [cx + offX, cy - offY],
    ];
    const nodes: NodeLocation[] = layout.map(([lng, lat], i) => ({
      node_id: `Heltec-0${i + 1}`,
      longitude: lng,
      latitude: lat,
      elevation_m: 100 + Math.round(Math.random() * 80),
      role: i === 0 ? "GATEWAY" : i === 1 ? "RELAY" : "EDGE",
      detection_radius_m: get().meshConfig.detection_radius_m,
    }));
    set({ nodes });
  },

  session: null,
  liveTick: null,
  tickBuffer: [],
  hotSpotsLocal: [],
  triangulation: null,
  droneRoutes: [],
  setSession: (session) => set({ session }),
  setLiveTick: (t) =>
    set((s) => {
      const byNode: Record<string, number> = {};
      for (const n of t.node_status) byNode[n.node_id] = n.smoke_ppm;
      const area_ha = t.fires.reduce((a, f) => a + f.area_ha, 0);
      const sample: TickSample = { t: t.sim_time_s, byNode, area_ha };
      const buf = [...s.tickBuffer, sample].slice(-TICK_BUFFER_MAX);
      return {
        liveTick: t,
        tickBuffer: buf,
        triangulation: t.triangulation ?? s.triangulation,
      };
    }),
  addHotSpotLocal: (h) => set((s) => ({ hotSpotsLocal: [...s.hotSpotsLocal, h] })),
  removeHotSpotLocal: (id) =>
    set((s) => ({ hotSpotsLocal: s.hotSpotsLocal.filter((h) => h.id !== id) })),
  setTriangulation: (triangulation) => set({ triangulation }),
  clearTriangulation: () => set({ triangulation: null }),
  addDroneRoute: (r) => set((s) => ({ droneRoutes: [...s.droneRoutes, r] })),
  removeDroneRoute: (id) =>
    set((s) => ({ droneRoutes: s.droneRoutes.filter((r) => r.drone_id !== id) })),
  clearDroneRoutes: () => set({ droneRoutes: [] }),
  clearHotSpotsLocal: () => set({ hotSpotsLocal: [] }),

  pendingFlightAuth: null,
  setPendingFlightAuth: (pendingFlightAuth) => set({ pendingFlightAuth }),
  boundPhysicalNodeId: "NODE_1",
  setBoundPhysicalNodeId: (boundPhysicalNodeId) => set({ boundPhysicalNodeId }),
  latestVerdict: null,
  setLatestVerdict: (latestVerdict) => set({ latestVerdict }),
  jetsonFsmState: "READY",
  setJetsonFsmState: (jetsonFsmState) => set({ jetsonFsmState }),

  toolMode: "IDLE",
  selectedNodeId: null,
  setToolMode: (toolMode) => set({ toolMode }),
  setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),

  layerVisibility: {
    nodes: true,
    links: true,
    fires: true,
    smoke: true,
    drones: true,
  },
  toggleLayer: (key) =>
    set((s) => ({
      layerVisibility: { ...s.layerVisibility, [key]: !s.layerVisibility[key] },
    })),
  isTelemetryDown: false,
  setTelemetryDown: (status) => set({ isTelemetryDown: status }),

  reset: () =>
    set({
      perimeter: [],
      fuelRisk: null,
      terrainStats: null,
      meshResult: null,
      nodes: [],
      links: [],
      session: null,
      liveTick: null,
      tickBuffer: [],
      hotSpotsLocal: [],
      triangulation: null,
      droneRoutes: [],
      toolMode: "IDLE",
      selectedNodeId: null,
    }),
}));
