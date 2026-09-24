import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import Map, { type MapRef } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import "@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { toast } from "sonner";
import { useIgnisStore } from "@/store/useIgnisStore";
import { addHotSpot, launchDrone, createSimulation, startSimulation, addSimulationNode } from "@/lib/api";
import type { GeoJSONPolygon, NodeLocation } from "@/lib/types";
import { buildLayers } from "./layers/buildLayers";
import { HotspotDialog, type HotspotDraft } from "./HotspotDialog";

const MAPBOX_TOKEN =
  (import.meta.env.VITE_MAPBOX_TOKEN as string | undefined) ??
  "";

const INITIAL_VIEW = {
  longitude: -73.06,
  latitude: -36.79,
  zoom: 13,
  pitch: 60,
  bearing: 0,
};

export function MapContainer() {
  const mapRef = useRef<MapRef>(null);
  const drawRef = useRef<MapboxDraw | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const perimeter = useIgnisStore((s) => s.perimeter);
  const setPerimeter = useIgnisStore((s) => s.setPerimeter);
  const nodes = useIgnisStore((s) => s.nodes);
  const links = useIgnisStore((s) => s.links);
  const liveTick = useIgnisStore((s) => s.liveTick);
  const triangulation = useIgnisStore((s) => s.triangulation);
  const droneRoutes = useIgnisStore((s) => s.droneRoutes);
  const visibility = useIgnisStore((s) => s.layerVisibility);
  const toolMode = useIgnisStore((s) => s.toolMode);
  const setToolMode = useIgnisStore((s) => s.setToolMode);
  const session = useIgnisStore((s) => s.session);
  const addDroneRoute = useIgnisStore((s) => s.addDroneRoute);
  const addHotSpotLocal = useIgnisStore((s) => s.addHotSpotLocal);
  const addNode = useIgnisStore((s) => s.addNode);

  const [hotspotDraft, setHotspotDraft] = useState<HotspotDraft | null>(null);

  /* ───────── deck.gl layers ───────── */
  const layers = useMemo(
    () =>
      buildLayers({
        nodes,
        links,
        liveTick,
        triangulation,
        droneRoutes,
        visibility,
      }),
    [nodes, links, liveTick, triangulation, droneRoutes, visibility],
  );

  /* ───────── Map onLoad ───────── */
  const handleLoad = useCallback(() => {
    const map = mapRef.current?.getMap();
    if (!map) return;

    if (!map.getSource("mapbox-dem")) {
      map.addSource("mapbox-dem", {
        type: "raster-dem",
        url: "mapbox://mapbox.mapbox-terrain-dem-v1",
        tileSize: 512,
        maxzoom: 14,
      });
      map.setTerrain({ source: "mapbox-dem", exaggeration: 1.5 });
    }

    if (!map.getSource("perimeter-src")) {
      map.addSource("perimeter-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      map.addLayer({
        id: "perimeter-fill",
        type: "fill",
        source: "perimeter-src",
        paint: { "fill-color": "hsl(142, 70%, 45%)", "fill-opacity": 0.18 },
      });
      map.addLayer({
        id: "perimeter-line",
        type: "line",
        source: "perimeter-src",
        paint: { "line-color": "hsl(28, 95%, 55%)", "line-width": 2.5, "line-blur": 0.5 },
      });
    }

    if (!drawRef.current) {
      drawRef.current = new MapboxDraw({
        displayControlsDefault: false,
        controls: { polygon: true, trash: true },
      });
    }

    if (!overlayRef.current) {
      overlayRef.current = new MapboxOverlay({ interleaved: true, layers });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.addControl(overlayRef.current as any);
    }
  }, [layers]);

  useEffect(() => {
    overlayRef.current?.setProps({ layers });
  }, [layers]);

  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map) return;
    const src = map.getSource("perimeter-src");
    if (!src || src.type !== "geojson") return;
    if (perimeter.length < 3) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (src as any).setData({ type: "FeatureCollection", features: [] });
      return;
    }
    const ring = closeRing(perimeter);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (src as any).setData({
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [ring] },
      properties: {},
    });
  }, [perimeter]);

  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map || !drawRef.current) return;
    const draw = drawRef.current;
    const drawing = toolMode === "DRAW_PERIMETER";

    const hasControl = (() => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return (map as any)._controls?.includes(draw);
      } catch {
        return false;
      }
    })();

    if (drawing && !hasControl) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.addControl(draw as any, "top-left");
      draw.changeMode("draw_polygon");
    } else if (!drawing && hasControl) {
      try { draw.deleteAll(); } catch { /* no-op */ }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.removeControl(draw as any);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const onCreate = (e: any) => {
      const feat = e.features?.[0];
      if (!feat) return;
      const ring = (feat.geometry?.coordinates?.[0] ?? []) as [number, number][];
      if (ring.length < 4) return;
      setPerimeter(ring);
      setToolMode("IDLE");
      toast.success(`Perímetro dibujado · ${ring.length - 1} vértices`);
    };
    const onDelete = () => setPerimeter([]);

    map.on("draw.create", onCreate);
    map.on("draw.delete", onDelete);
    return () => {
      map.off("draw.create", onCreate);
      map.off("draw.delete", onDelete);
    };
  }, [toolMode, setPerimeter, setToolMode]);

  useEffect(() => {
    if (perimeter.length < 2) return;
    const map = mapRef.current?.getMap();
    if (!map) return;
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
    for (const [lng, lat] of perimeter) {
      if (lng < minLng) minLng = lng; if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat;
    }
    map.fitBounds(
      [[minLng, minLat], [maxLng, maxLat]],
      { padding: 80, duration: 1800, pitch: 55, bearing: 0, essential: true },
    );
  }, [perimeter]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleClick = (e: any) => {
    const { lng, lat } = e.lngLat ?? {};
    if (lng == null || lat == null) return;

    if (toolMode === "PLACE_HOTSPOT" && session) {
      setHotspotDraft({ longitude: lng, latitude: lat });
    } else if (toolMode === "PLACE_DRONE_TARGET") {
      const baseLon = nodes.length > 0 ? (nodes.find((n) => n.role === "GATEWAY") ?? nodes[0]).longitude : -73.0650;
      const baseLat = nodes.length > 0 ? (nodes.find((n) => n.role === "GATEWAY") ?? nodes[0]).latitude : -36.7950;
      const baseAlt = nodes.length > 0 ? (nodes.find((n) => n.role === "GATEWAY") ?? nodes[0]).elevation_m : 60;
      const droneId = `DRONE_1`;

      const doLaunch = (sid: string) => {
        launchDrone(sid, {
          drone_id: droneId,
          start_lon: baseLon,
          start_lat: baseLat,
          start_alt_m: baseAlt,
          target_lon: lng,
          target_lat: lat,
          safe_agl_m: 80,
        })
          .then((r) => {
            addDroneRoute(r);
            toast.success(`✈️ Dron en vuelo hacia (${lat.toFixed(4)}, ${lng.toFixed(4)}) · ETA ${Math.round(r.estimated_eta_s)}s`);
            setToolMode("IDLE");
          })
          .catch((err) => toast.error(`Error de lanzamiento: ${err.message}`));
      };

      if (!session) {
        toast.info("Iniciando sesión de vuelo táctico...");
        const poly: GeoJSONPolygon = {
          type: "Polygon",
          coordinates: [[
            [-73.18, -36.75],
            [-73.02, -36.75],
            [-73.02, -36.85],
            [-73.18, -36.85],
            [-73.18, -36.75]
          ]],
        };
        createSimulation({
          name: "Misión Táctica Hualpén",
          polygon: poly,
          nodes: [{
            node_id: "BASE_GATEWAY",
            longitude: baseLon,
            latitude: baseLat,
            elevation_m: baseAlt,
            role: "GATEWAY",
            detection_radius_m: 1500,
          }],
          speed_multiplier: 1,
        })
          .then((created) => startSimulation(created.session_id))
          .then((started) => {
            setSession(started);
            doLaunch(started.session_id);
          })
          .catch((err) => toast.error(`Error al iniciar sesión: ${err.message}`));
      } else {
        doLaunch(session.session_id);
      }
    } else if (toolMode === "PLACE_NODE") {
      const nodeId = `NODE_${nodes.length + 1}`;
      const newNode: NodeLocation = {
        node_id: nodeId,
        longitude: lng,
        latitude: lat,
        elevation_m: 0,
        role: nodes.length === 0 ? "GATEWAY" : "EDGE",
        detection_radius_m: 1500,
      };
      addNode(newNode);
      if (session) {
        addSimulationNode(session.session_id, newNode)
          .then(() => toast.success(`Nodo #${nodeId} fijado en (${lat.toFixed(4)}, ${lng.toFixed(4)})`))
          .catch((err) => toast.error(`Error al registrar nodo: ${err.message}`));
      } else {
        toast.success(`Nodo #${nodeId} fijado en (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
      }
      setToolMode("IDLE");
    }
  };

  const confirmHotspot = async (h: {
    longitude: number;
    latitude: number;
    temperature_c: number;
    smoke_emission_g_per_s: number;
  }) => {
    if (!session) return;
    try {
      const res = await addHotSpot(session.session_id, h);
      addHotSpotLocal({ ...h, id: res.hot_spot_id });
      toast.success(`Foco encendido · ${res.hot_spot_id}`);
      setHotspotDraft(null);
      setToolMode("IDLE");
    } catch (e) {
      toast.error(`Error foco: ${(e as Error).message}`);
    }
  };

  const tokenMissing = !MAPBOX_TOKEN || (MAPBOX_TOKEN as string) === "INSERT_TOKEN_HERE";

  const cursor =
    toolMode === "PLACE_HOTSPOT" || toolMode === "PLACE_DRONE_TARGET" ? "crosshair" : undefined;

  return (
    <div className="relative h-full w-full" style={{ cursor }}>
      {tokenMissing && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="max-w-md rounded-lg border border-border/60 bg-card/80 p-6 text-center shadow-xl">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-foreground">
              Mapbox Token Requerido
            </h2>
            <p className="text-xs text-muted-foreground">
              Define{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
                VITE_MAPBOX_TOKEN
              </code>{" "}
              en tu archivo <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">.env.local</code>
            </p>
          </div>
        </div>
      )}

      <Map
        ref={mapRef}
        mapboxAccessToken={MAPBOX_TOKEN}
        initialViewState={INITIAL_VIEW}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        onLoad={handleLoad}
        onClick={handleClick}
        style={{ width: "100%", height: "100%" }}
      />

      <HotspotDialog
        draft={hotspotDraft}
        onCancel={() => setHotspotDraft(null)}
        onConfirm={confirmHotspot}
      />
    </div>
  );
}

function closeRing(coords: [number, number][]): [number, number][] {
  if (coords.length === 0) return coords;
  const [fx, fy] = coords[0];
  const [lx, ly] = coords[coords.length - 1];
  return fx === lx && fy === ly ? coords : [...coords, [fx, fy]];
}
