/**
 * deck.gl layer factory for Ignis Edge.
 *
 * Builds an array of Layer instances from the current store state.
 * Computed via useMemo with fine-grained deps to avoid full rebuilds on every tick.
 */
import { ScatterplotLayer, GeoJsonLayer, IconLayer, PathLayer, LineLayer, PolygonLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import type { Layer } from "@deck.gl/core";
import * as turf from "@turf/turf";
import type {
  DroneRoute,
  DroneState,
  FirePolygon,
  LoSLink,
  NodeLocation,
  NodeRole,
  NodeStatus,
  TickMessage,
  TriangulationResult,
} from "@/lib/types";

const ROLE_COLOR: Record<NodeRole, [number, number, number]> = {
  GATEWAY: [249, 115, 22], // primary orange
  RELAY: [59, 130, 246], // blue
  EDGE: [34, 211, 238], // cyan
};

export interface LayerVisibility {
  nodes: boolean;
  links: boolean;
  fires: boolean;
  smoke: boolean;
  drones: boolean;
}

export interface BuildLayersInput {
  nodes: NodeLocation[];
  links: LoSLink[];
  liveTick: TickMessage | null;
  triangulation: TriangulationResult | null;
  droneRoutes: DroneRoute[];
  visibility: LayerVisibility;
}

const DRONE_ICON_ATLAS = makeDroneIconAtlas();

export function buildLayers(input: BuildLayersInput): Layer[] {
  const { nodes, links, liveTick, triangulation, droneRoutes, visibility } = input;
  const layers: Layer[] = [];
  const nodeIndex = new Map(nodes.map((n) => [n.node_id, n] as const));
  const nodeStatus = new Map<string, NodeStatus>(
    (liveTick?.node_status ?? []).map((s) => [s.node_id, s]),
  );

  /* ── 1. Detection radius (translucent halo) ── */
  if (visibility.nodes && nodes.length) {
    layers.push(
      new ScatterplotLayer<NodeLocation>({
        id: "nodes-radius",
        data: nodes,
        getPosition: (n) => [n.longitude, n.latitude, n.elevation_m ?? 0],
        getRadius: (n) => n.detection_radius_m,
        radiusUnits: "meters",
        filled: true,
        stroked: true,
        getFillColor: (n) => [...ROLE_COLOR[n.role], 28] as [number, number, number, number],
        getLineColor: (n) => [...ROLE_COLOR[n.role], 140] as [number, number, number, number],
        lineWidthMinPixels: 1,
        pickable: false,
      }),
    );
  }

  /* ── 2. Node dots (clickable) ── */
  if (visibility.nodes && nodes.length) {
    layers.push(
      new ScatterplotLayer<NodeLocation>({
        id: "nodes-dots",
        data: nodes,
        getPosition: (n) => [n.longitude, n.latitude, n.elevation_m ?? 0],
        getRadius: (n) => (n.role === "GATEWAY" ? 130 : 90),
        radiusUnits: "meters",
        radiusMinPixels: 5,
        radiusMaxPixels: 14,
        getFillColor: (n) => [...ROLE_COLOR[n.role], 255] as [number, number, number, number],
        stroked: true,
        getLineColor: [10, 14, 20, 220],
        lineWidthMinPixels: 1.5,
        pickable: true,
      }),
    );
  }

  /* ── 3. LoS links ── */
  if (visibility.links && links.length) {
    layers.push(
      new LineLayer<LoSLink>({
        id: "links",
        data: links,
        getSourcePosition: (l) => {
          const n = nodeIndex.get(l.from_node);
          return n ? [n.longitude, n.latitude, n.elevation_m] : [0, 0, 0];
        },
        getTargetPosition: (l) => {
          const n = nodeIndex.get(l.to_node);
          return n ? [n.longitude, n.latitude, n.elevation_m] : [0, 0, 0];
        },
        getColor: (l) =>
          l.has_los ? [34, 197, 94, 180] : [239, 68, 68, 120],
        getWidth: (l) => (l.has_los ? 2 : 1),
        widthUnits: "pixels",
      }),
    );
  }

  /* ── 4. Fires ── */
  const fires = liveTick?.fires ?? [];
  if (visibility.fires && fires.length) {
    layers.push(
      new GeoJsonLayer({
        id: "fires",
        data: {
          type: "FeatureCollection",
          features: fires.map((f) => ({
            type: "Feature" as const,
            geometry: f.geojson,
            properties: { t: f.t_minutes, area: f.area_ha },
          })),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any,
        filled: true,
        stroked: true,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getFillColor: (f: any) => fireColor(f.properties?.t ?? 0),
        getLineColor: [255, 80, 0, 255],
        getLineWidth: 2,
        lineWidthUnits: "pixels",
        pickable: false,
      }),
    );
  }

  /* ── 5. Smoke heatmap (from node smoke_ppm readings) ── */
  if (visibility.smoke && nodeStatus.size && nodes.length) {
    const points = nodes
      .map((n) => {
        const s = nodeStatus.get(n.node_id);
        return s && s.smoke_ppm > 1
          ? { coord: [n.longitude, n.latitude] as [number, number], w: s.smoke_ppm }
          : null;
      })
      .filter((p): p is { coord: [number, number]; w: number } => !!p);
    if (points.length) {
      layers.push(
        new HeatmapLayer({
          id: "smoke-heat",
          data: points,
          getPosition: (p) => p.coord,
          getWeight: (p) => p.w,
          intensity: 1.2,
          radiusPixels: 90,
          aggregation: "SUM",
          opacity: 0.45,
        }),
      );
    }
  }

  /* ── 6. Triangulation marker + uncertainty disc + rays ── */
  if (triangulation) {
    const center: [number, number] = [
      triangulation.estimated_lon,
      triangulation.estimated_lat,
    ];
    const circle = turf.circle(center, triangulation.uncertainty_radius_m / 1000, {
      steps: 64,
      units: "kilometers",
    });
    layers.push(
      new PolygonLayer<number[][]>({
        id: "tri-uncertainty",
        data: [circle.geometry.coordinates[0]],
        getPolygon: (d) => d,
        filled: true,
        stroked: true,
        getFillColor: [249, 115, 22, 40],
        getLineColor: [249, 115, 22, 220],
        getLineWidth: 2,
        lineWidthUnits: "pixels",
      }),
    );
    layers.push(
      new ScatterplotLayer({
        id: "tri-center",
        data: [{ p: center }],
        getPosition: (d) => d.p,
        getRadius: 60,
        radiusUnits: "meters",
        radiusMinPixels: 6,
        getFillColor: [220, 38, 38, 255],
        stroked: true,
        getLineColor: [255, 200, 80, 255],
        lineWidthMinPixels: 2,
      }),
    );

    // Rays from detecting nodes
    const detecting = (liveTick?.node_status ?? []).filter((s) => s.detected);
    if (detecting.length) {
      layers.push(
        new LineLayer({
          id: "tri-rays",
          data: detecting,
          getSourcePosition: (s) => {
            const n = nodeIndex.get(s.node_id);
            return n ? [n.longitude, n.latitude] : center;
          },
          getTargetPosition: () => center,
          getColor: [249, 115, 22, 140],
          getWidth: 1,
          widthUnits: "pixels",
        }),
      );
    }
  }

  /* ── 7. Drones (paths + icon) ── */
  if (visibility.drones && (droneRoutes.length || (liveTick?.drones?.length ?? 0))) {
    if (droneRoutes.length) {
      layers.push(
        new PathLayer<DroneRoute>({
          id: "drone-paths",
          data: droneRoutes,
          getPath: (r) => r.waypoints,
          getColor: [34, 211, 238, 200],
          getWidth: 2,
          widthUnits: "pixels",
        }),
      );
    }
    const drones = liveTick?.drones ?? [];
    if (drones.length) {
      // Halo pulsante bajo el dron
      layers.push(
        new ScatterplotLayer<DroneState>({
          id: "drone-halos",
          data: drones,
          getPosition: (d) => [d.longitude, d.latitude, d.altitude_m],
          getRadius: 35,
          radiusUnits: "meters",
          radiusMinPixels: 12,
          filled: true,
          stroked: true,
          getFillColor: [34, 211, 238, 70],
          getLineColor: [34, 211, 238, 255],
          lineWidthMinPixels: 2,
        }),
      );

      layers.push(
        new IconLayer<DroneState>({
          id: "drone-icons",
          data: drones,
          getPosition: (d) => [d.longitude, d.latitude, d.altitude_m],
          getIcon: () => "drone",
          iconAtlas: DRONE_ICON_ATLAS.url,
          iconMapping: DRONE_ICON_ATLAS.mapping,
          getSize: 42,
          sizeUnits: "pixels",
          getAngle: (d) => -d.heading_deg,
          getColor: [0, 255, 230, 255],
          billboard: true,
        }),
      );
    }
  }

  return layers;
}

function fireColor(tMinutes: number): [number, number, number, number] {
  // 0 min → orange-ish; 60+ min → deep red
  const t = Math.max(0, Math.min(1, tMinutes / 60));
  const r = Math.round(255 - 35 * t);
  const g = Math.round(180 - 142 * t);
  const b = Math.round(80 - 80 * t);
  const a = Math.round(120 + 80 * t);
  return [r, g, b, a];
}

/* Tiny inline drone icon atlas (cyan triangle) generated as data URL. */
function makeDroneIconAtlas() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.translate(size / 2, size / 2);
  ctx.fillStyle = "rgba(34,211,238,1)";
  ctx.strokeStyle = "rgba(10,14,20,1)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, -22);
  ctx.lineTo(16, 18);
  ctx.lineTo(0, 8);
  ctx.lineTo(-16, 18);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  return {
    url: canvas.toDataURL(),
    mapping: { drone: { x: 0, y: 0, width: size, height: size, mask: false } },
  };
}
