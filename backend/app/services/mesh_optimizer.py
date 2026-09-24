import logging
import math
import random
from typing import List, Tuple, Optional

import numpy as np
from shapely.geometry import shape, Point, Polygon

from app.core.config import get_settings
from app.models.schemas import (
    NodeLocation, LoSLink, NodePlacementOut, MeshOptimizationIn,
)
from app.services.terrain_service import TerrainTile, terrain_service

logger = logging.getLogger(__name__)

EARTH_R = 6_371_000.0
K_REFRACTION = 4.0 / 3.0


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*EARTH_R*math.asin(min(1.0, math.sqrt(a)))


def fresnel_radius_m(d1, d2, freq_mhz) -> float:
    if d1 <= 0 or d2 <= 0: return 0.0
    lam = 299.792458 / freq_mhz  # c/(MHz) -> m
    return math.sqrt(lam * d1 * d2 / (d1 + d2))


class MeshOptimizer:
    """
    Optimización combinatoria de N nodos sobre un polígono real con DEM.

    Función de fitness:
        F = 0.50 · coverage  +  0.35 · connectivity  +  0.15 · elevation_bonus

    coverage     = fracción del perímetro cubierta por al menos 1 nodo (radio R)
    connectivity = (nodos en componente conectada principal) / N,
                   con bonus por margen LoS positivo
    elevation_bonus = elevación promedio normalizada [0..1]

    Estrategia:
      1) Sampleo K candidatos en grilla dentro del polígono
      2) Inicialización greedy: el más alto + greedy por máximo ganancia de fitness
      3) Búsqueda local: para cada nodo, intentar swap por candidato cercano que mejore F
      4) Hasta MAX_ITER iteraciones sin mejora
    """

    GRID_TARGET = 900            # candidatos en la grilla
    LOCAL_SEARCH_ITER = 60       # iteraciones máximas de búsqueda local
    MIN_SEPARATION_FACTOR = 0.10  # separación mínima ~ 10% del diámetro

    def __init__(self):
        self.settings = get_settings()
        self.freq_mhz = self.settings.LORA_FREQ_MHZ
        self.antenna_h = self.settings.ANTENNA_HEIGHT_M

    # ============================================================
    # API pública
    # ============================================================
    def optimize(self, payload: MeshOptimizationIn) -> NodePlacementOut:
        poly = shape(payload.polygon.model_dump())
        if not isinstance(poly, Polygon) or poly.is_empty:
            raise ValueError("Polygon is empty or invalid")

        tile = terrain_service.get_tile_for_polygon(poly)
        candidates = self._sample_candidates(poly, tile, target=self.GRID_TARGET)
        if len(candidates) < payload.n_nodes:
            raise ValueError(f"Insufficient candidates ({len(candidates)}) for {payload.n_nodes} nodes")

        diag = self._diagonal_m(poly)
        min_sep = max(150.0, self.MIN_SEPARATION_FACTOR * diag)
        R = payload.detection_radius_m

        # Para cobertura, muestreamos puntos del polígono que sirven de "objetivo"
        coverage_targets = self._sample_coverage_targets(poly, n=400)

        # 1) Inicialización greedy
        selected = self._greedy_init(candidates, payload.n_nodes, min_sep, R, coverage_targets, tile)

        # 2) Búsqueda local con swaps
        selected = self._local_search(selected, candidates, min_sep, R, coverage_targets, tile)

        # 3) Construir output
        nodes = self._assign_roles_and_build(selected)
        links = self._build_links(nodes, tile)
        cov = self._coverage(selected, R, coverage_targets)
        conn = self._connectivity(selected, tile)
        fit = self._fitness(selected, R, coverage_targets, tile)

        return NodePlacementOut(
            perimeter_name=payload.name,
            nodes=nodes,
            links=links,
            coverage_score=round(cov, 4),
            connectivity_score=round(conn, 4),
            fitness=round(fit, 4),
        )

    # ============================================================
    # Sampling
    # ============================================================
    def _sample_candidates(self, poly: Polygon, tile: TerrainTile, target: int) -> List[Tuple[float, float, float]]:
        minlon, minlat, maxlon, maxlat = poly.bounds
        n = max(20, int(math.sqrt(target)))
        xs = np.linspace(minlon, maxlon, n)
        ys = np.linspace(minlat, maxlat, n)
        out = []
        for x in xs:
            for y in ys:
                if poly.contains(Point(x, y)):
                    e = tile.elevation_at(float(x), float(y))
                    out.append((float(x), float(y), float(e)))
        return out

    def _sample_coverage_targets(self, poly: Polygon, n: int = 400) -> List[Tuple[float, float]]:
        minlon, minlat, maxlon, maxlat = poly.bounds
        side = max(20, int(math.sqrt(n)))
        xs = np.linspace(minlon, maxlon, side)
        ys = np.linspace(minlat, maxlat, side)
        out = []
        for x in xs:
            for y in ys:
                if poly.contains(Point(x, y)):
                    out.append((float(x), float(y)))
        return out

    # ============================================================
    # LoS check
    # ============================================================
    def has_los(self, a: Tuple[float, float, float], b: Tuple[float, float, float],
                tile: TerrainTile, samples: int = 48) -> Tuple[bool, float]:
        """Devuelve (hay_los, margen_min_m)."""
        lon1, lat1, e1 = a
        lon2, lat2, e2 = b
        d_total = haversine_m(lon1, lat1, lon2, lat2)
        if d_total < 1.0:
            return True, 0.0
        ea = e1 + self.antenna_h
        eb = e2 + self.antenna_h
        min_margin = float("inf")

        for i in range(1, samples):
            t = i / samples
            lon = lon1 + t * (lon2 - lon1)
            lat = lat1 + t * (lat2 - lat1)
            ground = tile.elevation_at(lon, lat)
            d1 = t * d_total
            d2 = (1 - t) * d_total
            bulge = (d1 * d2) / (2.0 * EARTH_R * K_REFRACTION)
            los_alt = ea + t * (eb - ea)
            f = fresnel_radius_m(d1, d2, self.freq_mhz)
            required = ground + bulge + 0.6 * f
            margin = los_alt - required
            if margin < min_margin:
                min_margin = margin
        return (min_margin >= 0.0), min_margin

    # ============================================================
    # Métricas
    # ============================================================
    def _coverage(self, selected: List[Tuple[float, float, float]], R: float,
                  targets: List[Tuple[float, float]]) -> float:
        if not targets:
            return 0.0
        covered = 0
        for tx, ty in targets:
            for sx, sy, _ in selected:
                if haversine_m(sx, sy, tx, ty) <= R:
                    covered += 1
                    break
        return covered / len(targets)

    def _connectivity(self, selected: List[Tuple[float, float, float]], tile: TerrainTile) -> float:
        n = len(selected)
        if n <= 1:
            return 1.0
        # Construir grafo de adyacencias por LoS
        adj = [[] for _ in range(n)]
        margins = []
        for i in range(n):
            for j in range(i+1, n):
                ok, m = self.has_los(selected[i], selected[j], tile, samples=24)
                if ok:
                    adj[i].append(j); adj[j].append(i)
                    margins.append(m)
        # Componente conectada principal (BFS)
        visited = [False]*n
        stack = [0]; visited[0] = True; comp = 0
        while stack:
            u = stack.pop(); comp += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True; stack.append(v)
        ratio = comp / n
        margin_bonus = 0.0
        if margins:
            avg_margin = sum(margins) / len(margins)
            margin_bonus = max(0.0, min(1.0, avg_margin / 50.0))
        return 0.85 * ratio + 0.15 * margin_bonus

    def _fitness(self, selected, R, targets, tile) -> float:
        cov = self._coverage(selected, R, targets)
        conn = self._connectivity(selected, tile)
        elevs = [s[2] for s in selected]
        elev_norm = max(0.0, min(1.0, (np.mean(elevs) - 200) / 3000.0)) if elevs else 0.0
        return 0.50 * cov + 0.35 * conn + 0.15 * elev_norm

    # ============================================================
    # Greedy init
    # ============================================================
    def _greedy_init(self, candidates, n, min_sep, R, targets, tile):
        # Empezar por el de mayor elevación
        sorted_by_elev = sorted(candidates, key=lambda c: c[2], reverse=True)
        selected = [sorted_by_elev[0]]

        for _ in range(1, n):
            best_c, best_score = None, -float("inf")
            for c in candidates:
                if c in selected:
                    continue
                if any(haversine_m(c[0], c[1], s[0], s[1]) < min_sep for s in selected):
                    continue
                trial = selected + [c]
                f = self._fitness(trial, R, targets, tile)
                if f > best_score:
                    best_score = f; best_c = c
            if best_c is None:
                # relajar separación si nos quedamos sin candidatos
                for c in candidates:
                    if c not in selected:
                        best_c = c; break
            selected.append(best_c)
        return selected

    # ============================================================
    # Local search (swaps)
    # ============================================================
    def _local_search(self, selected, candidates, min_sep, R, targets, tile):
        current_f = self._fitness(selected, R, targets, tile)
        for it in range(self.LOCAL_SEARCH_ITER):
            improved = False
            # Para cada nodo, probar reemplazos cercanos
            for idx in range(len(selected)):
                anchor = selected[idx]
                # Top-K candidatos cercanos al anchor (búsqueda local)
                neighborhood = sorted(
                    candidates,
                    key=lambda c: haversine_m(c[0], c[1], anchor[0], anchor[1]),
                )[:30]
                for c in neighborhood:
                    if c in selected:
                        continue
                    # Verificar separación con los demás
                    others = [s for k, s in enumerate(selected) if k != idx]
                    if any(haversine_m(c[0], c[1], s[0], s[1]) < min_sep for s in others):
                        continue
                    trial = others + [c]
                    f = self._fitness(trial, R, targets, tile)
                    if f > current_f + 1e-4:
                        selected = trial
                        current_f = f
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                logger.info(f"Local search converged at iter {it}, fitness={current_f:.4f}")
                break
        return selected

    # ============================================================
    # Outputs
    # ============================================================
    def _assign_roles_and_build(self, selected: List[Tuple[float, float, float]]) -> List[NodeLocation]:
        # Gateway = el más alto (mejor visibilidad de salida del mesh)
        sorted_idx = sorted(range(len(selected)), key=lambda i: selected[i][2], reverse=True)
        roles = ["EDGE"] * len(selected)
        roles[sorted_idx[0]] = "GATEWAY"
        # 30% siguientes como RELAY
        n_relays = max(1, int(0.3 * len(selected)))
        for i in sorted_idx[1:1 + n_relays]:
            roles[i] = "RELAY"

        nodes = []
        for i, (lon, lat, elev) in enumerate(selected):
            nodes.append(NodeLocation(
                node_id=f"NODE_{i+1:02d}",
                longitude=round(lon, 7),
                latitude=round(lat, 7),
                elevation_m=round(elev, 2),
                role=roles[i],
                detection_radius_m=self.settings.NODE_DETECTION_RADIUS_M,
            ))
        return nodes

    def _build_links(self, nodes: List[NodeLocation], tile: TerrainTile) -> List[LoSLink]:
        links = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a = (nodes[i].longitude, nodes[i].latitude, nodes[i].elevation_m)
                b = (nodes[j].longitude, nodes[j].latitude, nodes[j].elevation_m)
                ok, margin = self.has_los(a, b, tile, samples=64)
                d = haversine_m(a[0], a[1], b[0], b[1])
                # El radio Fresnel reportado es el del punto medio (informativo)
                f_mid = fresnel_radius_m(d/2, d/2, self.freq_mhz)
                links.append(LoSLink(
                    from_node=nodes[i].node_id,
                    to_node=nodes[j].node_id,
                    distance_m=round(d, 2),
                    has_los=ok,
                    fresnel_clearance_m=round(f_mid, 2),
                    margin_m=round(margin, 2),
                ))
        return links

    @staticmethod
    def _diagonal_m(poly: Polygon) -> float:
        minx, miny, maxx, maxy = poly.bounds
        return haversine_m(minx, miny, maxx, maxy)


mesh_optimizer = MeshOptimizer()