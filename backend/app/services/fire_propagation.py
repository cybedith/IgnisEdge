import math
import logging
from typing import List, Tuple, Optional

from shapely.geometry import Polygon, mapping
from shapely.affinity import rotate, translate
from pyproj import Transformer

from app.services.terrain_service import TerrainTile

logger = logging.getLogger(__name__)


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


class EnhancedEllipticFire:
    """
    Modelo elíptico de Anderson (1983) con corrección por pendiente.

    Forward ROS (m/min):
        R_f = R0 · fuel · (1 - moisture) · wind_factor · slope_factor

      wind_factor   = 1 + 0.16 · u^1.4         (u en m/s)
      slope_factor  = exp(3.533 · tan(α)^1.2)  (Rothermel, α ángulo upslope)

    L:B (Anderson):
        LB = 0.936·exp(0.2566 U_mph) + 0.461·exp(-0.1548 U_mph) - 0.397
        clamp ≥ 1

    Dirección efectiva de avance:
        El viento y la pendiente se combinan vectorialmente para definir
        el bearing real del eje mayor de la elipse.
    """

    R0_M_PER_MIN = 1.2

    def __init__(self, fuel_load: float = 1.0, moisture: float = 0.10):
        self.fuel = max(0.05, min(2.0, fuel_load))
        self.moisture = max(0.0, min(0.95, moisture))

    def compute_ellipse(
        self,
        ignition_lon: float,
        ignition_lat: float,
        elapsed_minutes: float,
        wind_speed_ms: float,
        wind_direction_meteo_deg: float,
        slope_deg: float = 0.0,
        aspect_deg: float = 0.0,
    ) -> dict:
        """
        Devuelve: {polygon: shapely.Polygon en EPSG:4326, area_m2, perimeter_m,
                   r_forward, r_back, lb, effective_bearing_deg}
        """
        epsg = utm_epsg_for(ignition_lon, ignition_lat)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        to_geo = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        ig_x, ig_y = to_utm.transform(ignition_lon, ignition_lat)

        # ---- Vectores: viento (downwind) + pendiente (upslope) ----
        downwind_bearing = (wind_direction_meteo_deg + 180.0) % 360.0
        # bearing 0=N => (0,1); 90=E => (1,0); ux=sin(bearing), uy=cos(bearing)
        wb = math.radians(downwind_bearing)
        wind_vx = math.sin(wb) * wind_speed_ms
        wind_vy = math.cos(wb) * wind_speed_ms

        # Aspect: bearing donde la ladera SUBE. El fuego acelera UPSLOPE.
        # Convertimos slope a "viento equivalente" m/s usando relación empírica:
        #   u_slope_equiv ~ tan(α) · 5.0  (calibrado para que slopes 30° aporten ~3 m/s)
        slope_rad = math.radians(min(60.0, max(0.0, slope_deg)))
        u_slope = math.tan(slope_rad) * 5.0
        ab = math.radians(aspect_deg % 360.0)
        slope_vx = math.sin(ab) * u_slope
        slope_vy = math.cos(ab) * u_slope

        eff_vx = wind_vx + slope_vx
        eff_vy = wind_vy + slope_vy
        u_eff = math.hypot(eff_vx, eff_vy)
        if u_eff < 1e-3:
            eff_bearing = 0.0
        else:
            # bearing del vector efectivo (el eje mayor apunta hacia eff_bearing)
            eff_bearing = (math.degrees(math.atan2(eff_vx, eff_vy)) + 360.0) % 360.0

        # ---- ROS ----
        wind_factor = 1.0 + 0.16 * (u_eff ** 1.4)
        slope_factor = math.exp(3.533 * (math.tan(slope_rad) ** 1.2)) if slope_deg > 0 else 1.0
        moisture_f = max(0.05, 1.0 - self.moisture)
        r_forward = self.R0_M_PER_MIN * self.fuel * moisture_f * wind_factor * slope_factor

        u_mph = u_eff * 2.23694
        if u_mph <= 1e-3:
            lb = 1.0
        else:
            lb = max(1.0,
                     0.936 * math.exp(0.2566 * u_mph)
                     + 0.461 * math.exp(-0.1548 * u_mph)
                     - 0.397)
        r_back = r_forward / (2.0 * lb)

        # ---- Geometría ----
        a = (r_forward + r_back) * elapsed_minutes / 2.0  # semieje mayor
        b_ax = a / lb                                      # semieje menor
        offset = (r_forward - r_back) * elapsed_minutes / 2.0

        ellipse = self._make_ellipse(a, b_ax, n=64)
        ellipse = translate(ellipse, xoff=offset, yoff=0.0)
        # Rotar al ángulo math = 90 - bearing
        math_angle = (90.0 - eff_bearing) % 360.0
        ellipse = rotate(ellipse, math_angle, origin=(0.0, 0.0), use_radians=False)
        ellipse_world = translate(ellipse, xoff=ig_x, yoff=ig_y)

        area_m2 = ellipse_world.area
        perim_m = ellipse_world.length

        # Convertir a WGS84 para el front
        coords = [to_geo.transform(x, y) for x, y in ellipse_world.exterior.coords]
        geo_poly = Polygon(coords)

        return {
            "polygon": geo_poly,
            "polygon_utm": ellipse_world,
            "epsg_utm": epsg,
            "ig_xy_utm": (ig_x, ig_y),
            "area_m2": area_m2,
            "perimeter_m": perim_m,
            "r_forward_m_min": r_forward,
            "r_back_m_min": r_back,
            "lb": lb,
            "effective_bearing_deg": eff_bearing,
            "u_effective_ms": u_eff,
        }

    @staticmethod
    def _make_ellipse(a: float, b: float, n: int = 64) -> Polygon:
        coords = []
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            coords.append((a * math.cos(theta), b * math.sin(theta)))
        coords.append(coords[0])
        return Polygon(coords)


class FireDynamicsHelper:
    """
    Helper para que el simulation_manager calcule pendiente/aspecto en el punto
    de ignición usando el TerrainTile cargado para el polígono de la simulación.
    """

    @staticmethod
    def slope_aspect_at(tile: TerrainTile, lon: float, lat: float) -> Tuple[float, float]:
        return tile.slope_at(lon, lat), tile.aspect_at(lon, lat)