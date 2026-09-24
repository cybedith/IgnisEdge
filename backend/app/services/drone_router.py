import math
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from pyproj import Transformer

from app.services.terrain_service import TerrainTile
from app.services.smoke_dispersion import GaussianPlume
from app.models.schemas import DroneRoute, DroneRouteRequest

logger = logging.getLogger(__name__)


@dataclass
class HazardField:
    """
    Encapsula los hazards para el ruteo: pluma de humo + zonas calientes + terreno.
    Funciona en UTM local.
    """
    fires_xy: List[Tuple[float, float]]               # focos en UTM
    fires_emission: List[float]                        # g/s
    wind_speed: float
    wind_dir_meteo_deg: float
    plume: GaussianPlume
    tile: TerrainTile
    to_utm: Transformer
    to_geo: Transformer

    def smoke_ppm_at(self, x: float, y: float) -> float:
        total = 0.0
        for (fx, fy), Q in zip(self.fires_xy, self.fires_emission):
            xd, yc = self.plume.downwind_local_coords((fx, fy), (x, y), self.wind_dir_meteo_deg)
            if xd <= 0:
                continue
            c = self.plume.concentration_g_per_m3(xd, yc, Q, self.wind_speed)
            total += self.plume.to_ppm(c)
        return total

    def terrain_alt(self, x: float, y: float) -> float:
        lon, lat = self.to_geo.transform(x, y)
        return self.tile.elevation_at(lon, lat)


@dataclass
class RRTNode:
    x: float
    y: float
    z: float
    cost: float = 0.0
    parent: Optional[int] = None


class RRTStar3D:
    """
    RRT* en (x, y, z) UTM con costos por:
      - distancia euclídea
      - penalización por humo (ppm > umbral)
      - penalización por terreno (altura < terreno + safe_agl)
      - penalización por proximidad a fuegos

    Sampling con bias hacia goal (10%).
    Steering con paso máximo configurable.
    Rewiring estándar de RRT*.
    """

    MAX_ITER = 800
    STEP_SIZE_M = 60.0
    GOAL_BIAS = 0.10
    REWIRE_RADIUS_M = 120.0
    SMOKE_PENALTY_THRESHOLD_PPM = 80.0
    SMOKE_PENALTY_FACTOR = 6.0     # multiplica el costo por unidad de distancia atravesada
    GOAL_TOLERANCE_M = 40.0
    MAX_FIRE_PROXIMITY_M = 80.0    # no acercarse a menos de esto a un foco

    def __init__(self, hazard: HazardField, safe_agl_m: float = 80.0,
                 cruise_speed_ms: float = 15.0):
        self.hazard = hazard
        self.safe_agl = safe_agl_m
        self.cruise = cruise_speed_ms

    def plan(self, start_xyz: Tuple[float, float, float],
             goal_xyz: Tuple[float, float, float]) -> Tuple[List[Tuple[float, float, float]], float]:
        nodes: List[RRTNode] = [RRTNode(*start_xyz, cost=0.0, parent=None)]
        goal = RRTNode(*goal_xyz)

        # Bbox de exploración: bbox del tile en UTM
        bbox_xyz = self._exploration_bbox(start_xyz, goal_xyz)

        best_goal_idx: Optional[int] = None
        best_goal_cost = float("inf")

        for it in range(self.MAX_ITER):
            # Sampling
            if random.random() < self.GOAL_BIAS:
                sample = (goal.x, goal.y, goal.z)
            else:
                sample = self._random_sample(bbox_xyz)

            # Nearest
            i_near = self._nearest(nodes, sample)
            near = nodes[i_near]

            # Steer
            new_xyz = self._steer((near.x, near.y, near.z), sample)
            if not self._point_valid(new_xyz):
                continue
            if not self._segment_valid((near.x, near.y, near.z), new_xyz, samples=6):
                continue

            # Costo del segmento
            seg_cost = self._segment_cost((near.x, near.y, near.z), new_xyz)
            new_node = RRTNode(*new_xyz, cost=near.cost + seg_cost, parent=i_near)

            # Vecinos para rewiring
            neighbors_idx = self._neighbors(nodes, new_xyz, self.REWIRE_RADIUS_M)
            # Reasignar parent al de menor costo
            for j in neighbors_idx:
                cand = nodes[j]
                if not self._segment_valid((cand.x, cand.y, cand.z), new_xyz, samples=4):
                    continue
                c = cand.cost + self._segment_cost((cand.x, cand.y, cand.z), new_xyz)
                if c < new_node.cost:
                    new_node.cost = c
                    new_node.parent = j

            new_idx = len(nodes)
            nodes.append(new_node)

            # Rewire vecinos a través del nuevo nodo si baja su costo
            for j in neighbors_idx:
                cand = nodes[j]
                if not self._segment_valid((new_node.x, new_node.y, new_node.z),
                                           (cand.x, cand.y, cand.z), samples=4):
                    continue
                c = new_node.cost + self._segment_cost(
                    (new_node.x, new_node.y, new_node.z), (cand.x, cand.y, cand.z)
                )
                if c < cand.cost:
                    cand.parent = new_idx
                    cand.cost = c

            # ¿Llegamos a goal?
            d_goal = math.dist(new_xyz, (goal.x, goal.y, goal.z))
            if d_goal < self.GOAL_TOLERANCE_M:
                if new_node.cost < best_goal_cost:
                    best_goal_cost = new_node.cost
                    best_goal_idx = new_idx

        if best_goal_idx is None:
            # No llegamos. Devolver mejor aproximación al goal.
            best_goal_idx = self._nearest(nodes, (goal.x, goal.y, goal.z))
            logger.warning("RRT* did not reach goal; returning best-effort path.")

        path_idx = self._reconstruct(nodes, best_goal_idx)
        path_xyz = [(nodes[i].x, nodes[i].y, nodes[i].z) for i in path_idx]
        # Asegurar que el último punto sea exactamente el goal (si llegamos)
        if math.dist(path_xyz[-1], (goal.x, goal.y, goal.z)) < self.GOAL_TOLERANCE_M:
            path_xyz[-1] = (goal.x, goal.y, goal.z)

        # Suavizado (shortcut)
        path_xyz = self._shortcut(path_xyz)

        # Costo total
        total_cost = nodes[best_goal_idx].cost
        return path_xyz, total_cost

    # ============================================================
    # Geometría/colisión
    # ============================================================
    def _point_valid(self, xyz: Tuple[float, float, float]) -> bool:
        x, y, z = xyz
        # Altura mínima sobre el terreno
        ground = self.hazard.terrain_alt(x, y)
        if z < ground + 5.0:
            return False
        # No demasiado cerca de un foco activo
        for fx, fy in self.hazard.fires_xy:
            if math.hypot(x - fx, y - fy) < self.MAX_FIRE_PROXIMITY_M:
                return False
        return True

    def _segment_valid(self, a, b, samples: int = 6) -> bool:
        for i in range(1, samples + 1):
            t = i / (samples + 1)
            p = (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]), a[2] + t*(b[2]-a[2]))
            if not self._point_valid(p):
                return False
        return True

    def _segment_cost(self, a, b) -> float:
        d = math.dist(a, b)
        if d < 1e-3:
            return 0.0
        # Sample humo a lo largo del segmento
        n = max(2, int(d / 30.0))
        smoke_cost = 0.0
        for i in range(n + 1):
            t = i / n
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            ppm = self.hazard.smoke_ppm_at(x, y)
            if ppm > self.SMOKE_PENALTY_THRESHOLD_PPM:
                smoke_cost += (ppm - self.SMOKE_PENALTY_THRESHOLD_PPM) * (d / (n + 1)) * 0.001
        return d + self.SMOKE_PENALTY_FACTOR * smoke_cost

    def _steer(self, from_xyz, to_xyz) -> Tuple[float, float, float]:
        dx = to_xyz[0] - from_xyz[0]
        dy = to_xyz[1] - from_xyz[1]
        dz = to_xyz[2] - from_xyz[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d <= self.STEP_SIZE_M:
            return to_xyz
        s = self.STEP_SIZE_M / d
        new_xyz = (from_xyz[0] + s*dx, from_xyz[1] + s*dy, from_xyz[2] + s*dz)
        # Ajustar z para respetar safe_agl
        ground = self.hazard.terrain_alt(new_xyz[0], new_xyz[1])
        z_min = ground + self.safe_agl * 0.5
        return (new_xyz[0], new_xyz[1], max(new_xyz[2], z_min))

    @staticmethod
    def _nearest(nodes: List[RRTNode], target) -> int:
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(nodes):
            d = (n.x - target[0])**2 + (n.y - target[1])**2 + (n.z - target[2])**2
            if d < best_d:
                best_d = d; best_i = i
        return best_i

    @staticmethod
    def _neighbors(nodes: List[RRTNode], target, radius: float) -> List[int]:
        r2 = radius * radius
        return [i for i, n in enumerate(nodes)
                if (n.x - target[0])**2 + (n.y - target[1])**2 + (n.z - target[2])**2 <= r2]

    @staticmethod
    def _reconstruct(nodes: List[RRTNode], end_idx: int) -> List[int]:
        path = [end_idx]
        cur = nodes[end_idx]
        while cur.parent is not None:
            path.append(cur.parent)
            cur = nodes[cur.parent]
        path.reverse()
        return path

    def _shortcut(self, path: List[Tuple[float, float, float]], iters: int = 60) -> List[Tuple[float, float, float]]:
        if len(path) <= 2:
            return path
        out = list(path)
        for _ in range(iters):
            n = len(out)
            if n <= 2:
                break
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            if j - i <= 1:
                continue
            if self._segment_valid(out[i], out[j], samples=8):
                # reemplazar todo entre i y j
                out = out[:i+1] + [out[j]] + out[j+1:]
        return out

    def _exploration_bbox(self, start, goal):
        # Caja generosa alrededor de start-goal
        margin = max(500.0, math.dist(start, goal) * 0.5)
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        z_min = min(start[2], goal[2]) - 50.0
        z_max = max(start[2], goal[2]) + 200.0
        return (min(xs) - margin, max(xs) + margin,
                min(ys) - margin, max(ys) + margin,
                z_min, z_max)

    def _random_sample(self, bbox):
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return (random.uniform(xmin, xmax),
                random.uniform(ymin, ymax),
                random.uniform(zmin, zmax))


class DroneRouter:
    """Wrapper que arma el HazardField, llama a RRT* y devuelve DroneRoute."""

    def plan_route(
        self,
        request: DroneRouteRequest,
        hazard: HazardField,
        cruise_speed_ms: float,
    ) -> DroneRoute:
        # Convertir a UTM
        sx, sy = hazard.to_utm.transform(request.start_lon, request.start_lat)
        gx, gy = hazard.to_utm.transform(request.target_lon, request.target_lat)

        # Altura inicial: si no se da, asumir start_alt = terreno + safe_agl
        ground_s = hazard.terrain_alt(sx, sy)
        sz = request.start_alt_m if request.start_alt_m is not None else ground_s + request.safe_agl_m
        # Goal a la misma AGL
        ground_g = hazard.terrain_alt(gx, gy)
        gz = ground_g + request.safe_agl_m

        rrt = RRTStar3D(hazard, safe_agl_m=request.safe_agl_m, cruise_speed_ms=cruise_speed_ms)
        path_xyz, total_cost = rrt.plan((sx, sy, sz), (gx, gy, gz))

        # Convertir a lon/lat
        wp = []
        total_len = 0.0
        prev = None
        avoided = 0
        for x, y, z in path_xyz:
            lon, lat = hazard.to_geo.transform(x, y)
            wp.append((round(lon, 7), round(lat, 7), round(z, 2)))
            if prev is not None:
                total_len += math.dist((x, y, z), prev)
                # ¿pasamos cerca de zona con humo > umbral? cuenta como zona evitada
                mx = (prev[0] + x) / 2; my = (prev[1] + y) / 2
                if hazard.smoke_ppm_at(mx, my) > 80.0:
                    avoided += 1
            prev = (x, y, z)

        eta = total_len / max(1.0, cruise_speed_ms)

        return DroneRoute(
            drone_id=request.drone_id,
            waypoints=wp,
            total_length_m=round(total_len, 2),
            estimated_eta_s=round(eta, 2),
            avoided_smoke_zones=avoided,
        )