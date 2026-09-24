import math
import logging
from typing import List, Tuple, Optional

import numpy as np
from pyproj import Transformer
from scipy.optimize import least_squares

from app.models.schemas import NodeDetection, TriangulationResult

logger = logging.getLogger(__name__)


class TDoATriangulator:
    """
    Triangulación TDoA del origen del incendio usando tiempos de detección de humo
    en N≥3 nodos, considerando el campo de viento.

    Modelo:
      Sea origen O=(x0, y0, t0) y nodo i en posición Pi detectando a tiempo ti.
      La pluma viaja a velocidad v_plume (componente downwind del viento) más
      una velocidad de difusión transversal pequeña.
      Para que un nodo "detecte" cuando la concentración cruza un umbral, asumimos
      tiempo de llegada del frente de pluma:

          ti_pred = t0 + (proyección_downwind(Pi - O)) / v_plume
                       + |proyección_crosswind| / v_diffusion

      Si la proyección downwind es negativa (nodo está upwind del origen),
      el tiempo es muy alto (la pluma no llega) -> penalización fuerte.

    Optimizamos (x0, y0, t0) minimizando residuos (ti - ti_pred).
    """

    V_DIFFUSION_MS = 0.6  # difusión lateral aproximada de pluma de humo

    def __init__(self, wind_speed_ms: float, wind_direction_meteo_deg: float):
        self.u = max(0.5, wind_speed_ms)
        self.wind_dir = wind_direction_meteo_deg
        # Vector downwind unitario (en UTM XY: x=Este, y=Norte)
        downwind_bearing = (wind_direction_meteo_deg + 180.0) % 360.0
        ang = math.radians(downwind_bearing)
        self.downwind_uv = (math.sin(ang), math.cos(ang))
        # Perpendicular (90° horario)
        self.crosswind_uv = (self.downwind_uv[1], -self.downwind_uv[0])

    def triangulate(self, detections: List[NodeDetection]) -> TriangulationResult:
        if len(detections) < 3:
            raise ValueError("Se requieren al menos 3 detecciones para triangular")

        # Trabajamos en UTM local (más estable numéricamente)
        ref_lon = float(np.mean([d.longitude for d in detections]))
        ref_lat = float(np.mean([d.latitude for d in detections]))
        epsg = self._utm_epsg(ref_lon, ref_lat)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        to_geo = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

        positions_xy = []
        times = []
        weights = []
        for d in detections:
            x, y = to_utm.transform(d.longitude, d.latitude)
            positions_xy.append((x, y))
            times.append(d.detection_time_s)
            # Peso ~ ppm (más concentración => más confiable)
            weights.append(max(0.5, math.sqrt(max(1.0, d.smoke_ppm))))
        positions_xy = np.array(positions_xy)
        times = np.array(times)
        weights = np.array(weights)

        # Estimación inicial: nodo con detección más temprana es el más cercano
        i_first = int(np.argmin(times))
        x_init = positions_xy[i_first, 0]
        y_init = positions_xy[i_first, 1]
        t_init = times[i_first] - 30.0  # ignición unos segundos antes

        x0 = np.array([x_init, y_init, t_init], dtype=float)

        result = least_squares(
            self._residuals,
            x0,
            args=(positions_xy, times, weights),
            method="lm",
            max_nfev=400,
        )

        x_est, y_est, t0_est = result.x
        # Incertidumbre: Cov ≈ (J^T J)^-1 · σ²
        residuals = result.fun
        n = len(residuals)
        if n > 3:
            sigma2 = float(np.sum(residuals**2) / (n - 3))
            try:
                JtJ = result.jac.T @ result.jac
                cov = np.linalg.inv(JtJ) * sigma2
                # Radio de incertidumbre = sqrt(traza espacial / 2) — radio promedio
                var_xy = (cov[0, 0] + cov[1, 1]) / 2.0
                radius = math.sqrt(max(0.0, var_xy))
            except np.linalg.LinAlgError:
                radius = 200.0
                cov = None
        else:
            radius = 500.0
            cov = None

        # Confianza: 1 / (1 + radius/100). 100m de radio => ~50% conf
        confidence = 1.0 / (1.0 + radius / 100.0)
        confidence = max(0.0, min(1.0, confidence))

        lon_est, lat_est = to_geo.transform(x_est, y_est)

        return TriangulationResult(
            estimated_lon=round(lon_est, 7),
            estimated_lat=round(lat_est, 7),
            estimated_t0_s=round(t0_est, 2),
            uncertainty_radius_m=round(radius, 2),
            confidence=round(confidence, 4),
            n_detections_used=len(detections),
        )

    def _residuals(self, x, positions_xy, times, weights):
        x0, y0, t0 = x
        ux, uy = self.downwind_uv
        px, py = self.crosswind_uv

        dx = positions_xy[:, 0] - x0
        dy = positions_xy[:, 1] - y0
        downwind = dx * ux + dy * uy
        crosswind = dx * px + dy * py

        # Tiempo predicho: si el nodo está upwind (downwind<0), el frente no llega.
        # Penalizamos con un costo lineal grande (distancia/V_min).
        t_pred = np.where(
            downwind > 0,
            t0 + downwind / self.u + np.abs(crosswind) / self.V_DIFFUSION_MS,
            t0 + (np.abs(downwind) + np.abs(crosswind)) / 0.3,  # difusión muy lenta upwind
        )
        return weights * (times - t_pred)

    @staticmethod
    def _utm_epsg(lon: float, lat: float) -> int:
        zone = int(math.floor((lon + 180.0) / 6.0) + 1)
        return (32600 if lat >= 0 else 32700) + zone