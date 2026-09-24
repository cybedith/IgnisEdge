import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from pyproj import Transformer
from shapely.geometry import shape, mapping, Polygon

from app.core.config import get_settings
from app.models.schemas import (
    SimulationCreateIn, SimulationStateOut, NodeLocation, HotSpotIn,
    NodeDetection, TriangulationResult, DroneState, DroneRoute, DroneRouteRequest,
    WeatherSnapshot, FirePolygon,
)
from app.services.terrain_service import terrain_service, TerrainTile
from app.services.weather_service import weather_service
from app.services.fire_propagation import EnhancedEllipticFire, utm_epsg_for
from app.services.smoke_dispersion import GaussianPlume, stability_class
from app.services.triangulation import TDoATriangulator
from app.services.drone_router import DroneRouter, HazardField

logger = logging.getLogger(__name__)


@dataclass
class HotSpot:
    hot_spot_id: str
    lon: float
    lat: float
    temperature_c: float
    smoke_emission_g_per_s: float
    radius_m: float
    ignited_at_sim_s: float
    active: bool = True

    def elapsed_min(self, sim_time_s: float) -> float:
        return max(0.0, (sim_time_s - self.ignited_at_sim_s) / 60.0)


@dataclass
class Drone:
    state: DroneState
    route: Optional[DroneRoute] = None
    leg_idx: int = 0
    leg_start_time_s: float = 0.0
    leg_duration_s: float = 0.0


@dataclass
class NodeSnapshot:
    node_id: str
    smoke_ppm: float
    detected: bool
    detection_time_s: Optional[float] = None


class SimulationSession:
    def __init__(self, payload: SimulationCreateIn):
        self.session_id = str(uuid.uuid4())
        self.name = payload.name
        self.polygon = shape(payload.polygon.model_dump())
        self.nodes: List[NodeLocation] = list(payload.nodes)
        self.speed_multiplier = payload.speed_multiplier
        self.use_real_weather = payload.use_real_weather
        self.manual_wind_speed = payload.manual_wind_speed_ms
        self.manual_wind_dir = payload.manual_wind_direction_deg

        self.running = False
        self.sim_time_s = 0.0
        self.real_time_started: Optional[datetime] = None
        self.hot_spots: List[HotSpot] = []
        self.drones: Dict[str, Drone] = {}
        self.node_first_detection: Dict[str, float] = {}
        self.last_triangulation: Optional[TriangulationResult] = None

        self._subscribers: Set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        self._system_logs: List[dict] = []

        self.tile: Optional[TerrainTile] = None
        self.weather: Optional[WeatherSnapshot] = None
        self.epsg_utm: int = utm_epsg_for(self.polygon.centroid.x, self.polygon.centroid.y)
        self.to_utm: Optional[Transformer] = None
        self.to_geo: Optional[Transformer] = None
        self.plume: Optional[GaussianPlume] = None

        self.settings = get_settings()
        self.tick_period_s = 1.0 / self.settings.SIM_TICK_HZ

    async def initialize(self):
        loop = asyncio.get_running_loop()

        def _load_tile():
            return terrain_service.get_tile_for_polygon(self.polygon)
        self.tile = await loop.run_in_executor(None, _load_tile)

        if self.use_real_weather:
            try:
                cx, cy = float(self.polygon.centroid.x), float(self.polygon.centroid.y)
                self.weather = await loop.run_in_executor(None, weather_service.get_current, cx, cy)
                logger.info(f"[{self.session_id}] Weather: u={self.weather.wind_speed_ms} dir={self.weather.wind_direction_deg}")
            except Exception as e:
                logger.warning(f"[{self.session_id}] Weather fetch failed, using manual: {e}")
                self.use_real_weather = False

        if not self.use_real_weather:
            self.weather = WeatherSnapshot(
                longitude=self.polygon.centroid.x,
                latitude=self.polygon.centroid.y,
                forecast_time=datetime.now(timezone.utc).isoformat(),
                wind_speed_ms=self.manual_wind_speed if self.manual_wind_speed is not None else 5.0,
                wind_direction_deg=self.manual_wind_dir if self.manual_wind_dir is not None else 270.0,
                temperature_c=20.0,
                relative_humidity_pct=40.0,
                source="manual",
            )

        self.to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_utm}", always_xy=True)
        self.to_geo = Transformer.from_crs(f"EPSG:{self.epsg_utm}", "EPSG:4326", always_xy=True)

        hour_now = datetime.now().hour
        cls = stability_class(self.weather.wind_speed_ms, hour_now)
        self.plume = GaussianPlume(stability=cls, source_height_m=10.0)
        logger.info(f"[{self.session_id}] Stability class={cls}")

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    async def _broadcast(self, message: dict):
        dead = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass
                try: q.put_nowait(message)
                except Exception: dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    async def start(self):
        if self.running:
            return
        if self.tile is None:
            await self.initialize()
        self.running = True
        self.real_time_started = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[{self.session_id}] Simulation started.")

    async def pause(self):
        self.running = False
        logger.info(f"[{self.session_id}] Simulation paused.")

    async def resume(self):
        if not self.running and self._task is not None and not self._task.done():
            self.running = True
        elif not self.running:
            await self.start()

    async def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        logger.info(f"[{self.session_id}] Simulation stopped.")

    async def _run_loop(self):
        try:
            while True:
                t0 = time.monotonic()
                if self.running:
                    async with self._lock:
                        await self._tick()
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, self.tick_period_s - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"[{self.session_id}] Sim loop crashed: {e}")
            self.running = False

    def add_system_log(self, source: str, level: str, message: str):
        now_str = datetime.now(timezone.utc).isoformat()
        self._system_logs.append({
            "timestamp": now_str,
            "source": source,
            "level": level,
            "message": message
        })
        if len(self._system_logs) > 100:
            self._system_logs = self._system_logs[-100:]

    async def _tick(self):
        dt_sim = self.tick_period_s * self.speed_multiplier
        self.sim_time_s += dt_sim
        


        fire_polys: List[FirePolygon] = []
        for hs in self.hot_spots:
            if not hs.active:
                continue
            elapsed_min = hs.elapsed_min(self.sim_time_s)
            if elapsed_min <= 0:
                continue
            slope = self.tile.slope_at(hs.lon, hs.lat)
            aspect = self.tile.aspect_at(hs.lon, hs.lat)
            fire_model = EnhancedEllipticFire(fuel_load=1.0, moisture=self._fuel_moisture())
            res = fire_model.compute_ellipse(
                ignition_lon=hs.lon, ignition_lat=hs.lat,
                elapsed_minutes=elapsed_min,
                wind_speed_ms=self.weather.wind_speed_ms,
                wind_direction_meteo_deg=self.weather.wind_direction_deg,
                slope_deg=slope, aspect_deg=aspect,
            )
            fire_polys.append(FirePolygon(
                hot_spot_id=hs.hot_spot_id,
                t_minutes=round(elapsed_min, 2),
                area_ha=round(res["area_m2"] / 10_000.0, 4),
                perimeter_m=round(res["perimeter_m"], 2),
                geojson=mapping(res["polygon"]),
            ))

        node_snaps: List[NodeSnapshot] = []
        u = self.weather.wind_speed_ms
        wd = self.weather.wind_direction_deg
        for nd in self.nodes:
            nx, ny = self.to_utm.transform(nd.longitude, nd.latitude)
            total_ppm = 0.0
            for hs in self.hot_spots:
                if not hs.active: continue
                if hs.ignited_at_sim_s > self.sim_time_s: continue
                fx, fy = self.to_utm.transform(hs.lon, hs.lat)
                xd, yc = self.plume.downwind_local_coords((fx, fy), (nx, ny), wd)
                if xd <= 0: continue
                c_g_m3 = self.plume.concentration_g_per_m3(xd, yc, hs.smoke_emission_g_per_s, u)
                total_ppm += self.plume.to_ppm(c_g_m3)

            detected = total_ppm >= self.settings.NODE_SMOKE_THRESHOLD_PPM
            first_time = self.node_first_detection.get(nd.node_id)
            if detected and first_time is None:
                self.node_first_detection[nd.node_id] = self.sim_time_s
                first_time = self.sim_time_s
            node_snaps.append(NodeSnapshot(
                node_id=nd.node_id,
                smoke_ppm=round(total_ppm, 3),
                detected=detected,
                detection_time_s=first_time,
            ))

        triangulation = None
        detections_for_tri = []
        for nd, snap in zip(self.nodes, node_snaps):
            if snap.detection_time_s is not None and snap.smoke_ppm > 0:
                detections_for_tri.append(NodeDetection(
                    node_id=nd.node_id,
                    longitude=nd.longitude, latitude=nd.latitude,
                    elevation_m=nd.elevation_m,
                    detection_time_s=snap.detection_time_s,
                    smoke_ppm=snap.smoke_ppm,
                ))
        if len(detections_for_tri) >= 3:
            try:
                tri = TDoATriangulator(u, wd)
                self.last_triangulation = tri.triangulate(detections_for_tri)
                triangulation = self.last_triangulation
            except Exception as e:
                logger.warning(f"[{self.session_id}] Triangulation failed: {e}")

        for drone in self.drones.values():
            self._advance_drone(drone, dt_sim)

        msg = {
            "type": "tick",
            "sim_time_s": round(self.sim_time_s, 2),
            "speed_multiplier": self.speed_multiplier,
            "weather": self.weather.model_dump() if self.weather else None,
            "fires": [fp.model_dump() for fp in fire_polys],
            "node_status": [{
                "node_id": s.node_id,
                "smoke_ppm": s.smoke_ppm,
                "detected": s.detected,
                "detection_time_s": s.detection_time_s,
            } for s in node_snaps],
            "triangulation": triangulation.model_dump() if triangulation else None,
            "drones": [d.state.model_dump() for d in self.drones.values()],
            "system_logs": list(self._system_logs)
        }
        await self._broadcast(msg)

    def _advance_drone(self, drone: Drone, dt_s: float):
        if not self.tile:
            return

        terrain_elev = self.tile.elevation_at(drone.state.longitude, drone.state.latitude)
        drone.state.terrain_elev_m = round(terrain_elev, 2)
        
        if drone.state.status == "IDLE":
            drone.state.flight_mode = "READY"
        elif drone.state.status == "ARRIVED":
            drone.state.flight_mode = "ON_STATION"
        elif drone.state.status == "EN_ROUTE":
            drone.state.flight_mode = "GUIDED"
        elif drone.state.status == "RETURNING":
            drone.state.flight_mode = "RTL"

        # Movement
        if drone.state.status in ("EN_ROUTE", "RETURNING"):
            if drone.route is None or drone.leg_idx >= len(drone.route.waypoints) - 1:
                drone.state.status = "ARRIVED"
                drone.state.progress = 1.0
                drone.state.flight_mode = "ON_STATION"
            else:
                cruise = self.settings.DRONE_CRUISE_SPEED_MS
                wps = drone.route.waypoints
                a = wps[drone.leg_idx]
                b = wps[drone.leg_idx + 1]
                ax, ay = self.to_utm.transform(a[0], a[1])
                bx, by = self.to_utm.transform(b[0], b[1])
                leg_len = math.sqrt((bx-ax)**2 + (by-ay)**2)
                
                if leg_len < 1e-3:
                    drone.leg_idx += 1
                else:
                    leg_duration = leg_len / max(1.0, cruise)
                    elapsed_in_leg = self.sim_time_s - drone.leg_start_time_s
                    t = min(1.0, elapsed_in_leg / leg_duration)
                    cur_x = ax + t * (bx - ax)
                    cur_y = ay + t * (by - ay)
                    
                    cur_lon, cur_lat = self.to_geo.transform(cur_x, cur_y)
                    drone.state.longitude = round(cur_lon, 7)
                    drone.state.latitude = round(cur_lat, 7)
                    
                    dx_e = bx - ax; dy_n = by - ay
                    if abs(dx_e) + abs(dy_n) > 1e-3:
                        heading = (math.degrees(math.atan2(dx_e, dy_n)) + 360.0) % 360.0
                        drone.state.heading_deg = round(heading, 1)

                    if t >= 1.0:
                        drone.leg_idx += 1
                        drone.leg_start_time_s = self.sim_time_s
                        if drone.leg_idx >= len(wps) - 1:
                            drone.state.status = "ARRIVED"
                            drone.state.progress = 1.0
                            drone.state.flight_mode = "ON_STATION"
                    else:
                        drone.state.progress = round((drone.leg_idx + t) / max(1, len(wps) - 1), 4)

        # Update terrain elevation after possible movement
        terrain_elev = self.tile.elevation_at(drone.state.longitude, drone.state.latitude)
        drone.state.terrain_elev_m = round(terrain_elev, 2)
        
        # Terrain Following Kinematics
        target_agl = 80.0
        target_amsl = terrain_elev + target_agl
        
        # Initial initialization
        if drone.state.alt_amsl_m == 0.0:
            drone.state.alt_amsl_m = target_amsl
            
        dz = target_amsl - drone.state.alt_amsl_m
        vz = dz * 0.2  # climb/descend proportional rate
        
        if abs(dz) > 0.5:
            drone.state.alt_amsl_m += vz * dt_s
        else:
            drone.state.alt_amsl_m = target_amsl
            
        drone.state.altitude_m = round(drone.state.alt_amsl_m, 2)
        drone.state.alt_agl_m = round(drone.state.alt_amsl_m - terrain_elev, 2)
        
        # Attitude and Motors
        import random
        base_pwm = 50.0
        pitch_deg = 0.0
        if drone.state.status in ("EN_ROUTE", "RETURNING"):
            pitch_deg = 15.0 + random.uniform(-2, 2)
            if vz > 1.0:
                base_pwm = 75.0 + random.uniform(-5, 5)
            elif vz < -1.0:
                base_pwm = 35.0 + random.uniform(-5, 5)
            else:
                base_pwm = 60.0 + random.uniform(-2, 2)
        else:
            pitch_deg = random.uniform(-1, 1)
            base_pwm = 50.0 + random.uniform(-2, 2)
            
        drone.state.attitude.pitch = round(pitch_deg, 2)
        drone.state.attitude.roll = round(random.uniform(-2, 2), 2)
        drone.state.attitude.yaw = round(drone.state.heading_deg + random.uniform(-1, 1), 2)
        drone.state.motors = [round(base_pwm, 1) for _ in range(4)]
        
        # Thermal & Hotspots
        if drone.state.status == "ON_STATION":
            drone.state.thermal_grid = [[round(random.uniform(20.0, 30.0), 1) for _ in range(32)] for _ in range(24)]
        else:
            drone.state.thermal_grid = None
            
        from app.models.schemas import HotSpotTelemetry
        drone.state.focos = []
        for hs in self.hot_spots:
            if not hs.active: continue
            dist = math.hypot(hs.lon - drone.state.longitude, hs.lat - drone.state.latitude)
            if dist < 0.006:  # roughly 600m
                drone.state.focos.append(HotSpotTelemetry(
                    hot_spot_id=hs.hot_spot_id,
                    temperature_c=round(hs.temperature_c + random.uniform(-10, 10), 1),
                    flicker=random.random() > 0.5
                ))

    def _fuel_moisture(self) -> float:
        rh = self.weather.relative_humidity_pct if self.weather else 40.0
        return max(0.04, min(0.40, 0.03 + 0.30 * (rh / 100.0)))

    async def add_hot_spot(self, hs_in: HotSpotIn) -> str:
        hs_id = f"HS_{len(self.hot_spots) + 1:03d}"
        async with self._lock:
            self.hot_spots.append(HotSpot(
                hot_spot_id=hs_id,
                lon=hs_in.longitude, lat=hs_in.latitude,
                temperature_c=hs_in.temperature_c,
                smoke_emission_g_per_s=hs_in.smoke_emission_g_per_s,
                radius_m=hs_in.radius_m,
                ignited_at_sim_s=self.sim_time_s,
            ))
        return hs_id

    async def remove_hot_spot(self, hs_id: str) -> bool:
        async with self._lock:
            for hs in self.hot_spots:
                if hs.hot_spot_id == hs_id and hs.active:
                    hs.active = False
    async def add_node(self, node: NodeLocation):
        async with self._lock:
            if node.elevation_m <= 0.0 and self.tile:
                node.elevation_m = round(self.tile.elevation_at(node.longitude, node.latitude), 2)
            self.nodes = [n for n in self.nodes if n.node_id != node.node_id]
            self.nodes.append(node)
            self.add_system_log("LORA", "INFO", f"Nuevo Nodo #{node.node_id} registrado en ({node.latitude:.4f}, {node.longitude:.4f}) Alt: {node.elevation_m}m")

    async def trigger_node_alert(self, node_id: str):
        target_node = next((n for n in self.nodes if n.node_id == node_id), None)
        if not target_node:
            return None

        async with self._lock:
            self.node_first_detection[node_id] = self.sim_time_s
            self.add_system_log("LORA", "WARN", f"ALERTA RF: Nodo #{node_id} reporta caída brusca gas (HUMO)")
            self.add_system_log("JETSON", "INFO", f"Triage FSM: Evaluada alerta Nodo #{node_id}. Despachando DRONE_1")

        base = next((n for n in self.nodes if n.role == "GATEWAY"), self.nodes[0])
        drone = self.drones.get("DRONE_1")
        start_lon = drone.state.longitude if drone else base.longitude
        start_lat = drone.state.latitude if drone else base.latitude
        start_alt = drone.state.alt_amsl_m if drone else (base.elevation_m + 80.0)

        req = DroneRouteRequest(
            drone_id="DRONE_1",
            start_lon=start_lon,
            start_lat=start_lat,
            start_alt_m=start_alt,
            target_lon=target_node.longitude,
            target_lat=target_node.latitude,
            safe_agl_m=80.0,
        )
        return await self.launch_drone(req)

    async def command_drone(self, drone_id: str, command: str) -> bool:
        async with self._lock:
            drone = self.drones.get(drone_id)
            if not drone:
                return False
            cmd = command.upper()
            if cmd == "RTL":
                base = next((n for n in self.nodes if n.role == "GATEWAY"), self.nodes[0] if self.nodes else None)
                target_lon = base.longitude if base else drone.state.longitude
                target_lat = base.latitude if base else drone.state.latitude
                drone.state.status = "RETURNING"
                drone.state.flight_mode = "RTL"
                self.add_system_log("PIXHAWK", "INFO", f"Comando RTL ejecutado para {drone_id}. Retornando a base ({target_lat:.4f}, {target_lon:.4f})")
                self.add_system_log("JETSON", "INFO", f"FSM transición a RTL para {drone_id}")
            elif cmd == "LOITER":
                drone.state.status = "ARRIVED"
                drone.state.flight_mode = "LOITER"
                drone.route = None
                self.add_system_log("PIXHAWK", "INFO", f"Comando LOITER ejecutado para {drone_id}. Manteniendo posición")
            elif cmd == "LAND":
                drone.state.status = "IDLE"
                drone.state.flight_mode = "LAND"
                drone.route = None
                self.add_system_log("PIXHAWK", "WARN", f"Comando LAND ejecutado para {drone_id}. Aterrizando")
            elif cmd == "GUIDED":
                drone.state.flight_mode = "GUIDED"
                self.add_system_log("PIXHAWK", "INFO", f"Modo GUIDED habilitado para {drone_id}")
            return True

    async def launch_drone(self, request: DroneRouteRequest) -> DroneRoute:
        hazard = self._build_hazard_field()
        router = DroneRouter()
        route = router.plan_route(request, hazard, self.settings.DRONE_CRUISE_SPEED_MS)

        ds = DroneState(
            drone_id=request.drone_id,
            longitude=request.start_lon, latitude=request.start_lat,
            altitude_m=request.start_alt_m or (self.tile.elevation_at(request.start_lon, request.start_lat) + request.safe_agl_m),
            status="EN_ROUTE",
            target_lon=request.target_lon, target_lat=request.target_lat,
            progress=0.0,
        )
        async with self._lock:
            self.drones[request.drone_id] = Drone(
                state=ds, route=route, leg_idx=0, leg_start_time_s=self.sim_time_s,
            )
            self.add_system_log("JETSON", "INFO", f"FSM transition to TRANSIT for drone {request.drone_id}")
            self.add_system_log("PIXHAWK", "INFO", f"Nav command sent for drone {request.drone_id}")
            
        return route

    def _build_hazard_field(self) -> HazardField:
        fires_xy = []; emissions = []
        for hs in self.hot_spots:
            if hs.active:
                fx, fy = self.to_utm.transform(hs.lon, hs.lat)
                fires_xy.append((fx, fy))
                emissions.append(hs.smoke_emission_g_per_s)
        return HazardField(
            fires_xy=fires_xy,
            fires_emission=emissions,
            wind_speed=self.weather.wind_speed_ms,
            wind_dir_meteo_deg=self.weather.wind_direction_deg,
            plume=self.plume,
            tile=self.tile,
            to_utm=self.to_utm,
            to_geo=self.to_geo,
        )

    def state_summary(self) -> SimulationStateOut:
        return SimulationStateOut(
            session_id=self.session_id,
            name=self.name,
            running=self.running,
            sim_time_s=round(self.sim_time_s, 2),
            real_time_started=self.real_time_started.isoformat() if self.real_time_started else None,
            speed_multiplier=self.speed_multiplier,
            n_hot_spots=sum(1 for h in self.hot_spots if h.active),
            n_drones=len(self.drones),
        )


class SimulationManager:
    def __init__(self):
        self._sessions: Dict[str, SimulationSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, payload: SimulationCreateIn) -> SimulationSession:
        sess = SimulationSession(payload)
        await sess.initialize()
        async with self._lock:
            self._sessions[sess.session_id] = sess
        return sess

    def get(self, session_id: str) -> Optional[SimulationSession]:
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        await sess.stop()
        return True

    def list_sessions(self) -> List[SimulationStateOut]:
        return [s.state_summary() for s in self._sessions.values()]


simulation_manager = SimulationManager()



