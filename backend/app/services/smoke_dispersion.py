import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PasquillCoeffs:
    """Coeficientes para σy = a·x^b y σz = c·x^d en metros (terreno rural)."""
    a: float
    b: float
    c: float
    d: float


# Estabilidad atmosférica Pasquill-Gifford (Briggs rural)
# A=muy inestable, D=neutral (más común), F=muy estable
PG_COEFFS = {
    "A": PasquillCoeffs(0.22, 0.894, 0.20, 1.0),
    "B": PasquillCoeffs(0.16, 0.894, 0.12, 1.0),
    "C": PasquillCoeffs(0.11, 0.894, 0.08, 1.0),
    "D": PasquillCoeffs(0.08, 0.894, 0.06, 1.0),
    "E": PasquillCoeffs(0.06, 0.894, 0.03, 1.0),
    "F": PasquillCoeffs(0.04, 0.894, 0.016, 1.0),
}


def stability_class(wind_ms: float, hour: int = 12) -> str:
    """
    Estimación heurística simple. Para una versión productiva se usa también
    radiación solar / cobertura nubosa, pero con viento + hora obtenemos un
    estimador útil.
      Día (6-18h):    viento <2 -> A, <4 -> B, <6 -> C, sino D
      Noche:          viento <2 -> F, <3 -> E, sino D
    """
    if 6 <= hour <= 18:
        if wind_ms < 2: return "A"
        if wind_ms < 4: return "B"
        if wind_ms < 6: return "C"
        return "D"
    else:
        if wind_ms < 2: return "F"
        if wind_ms < 3: return "E"
        return "D"


class GaussianPlume:
    """
    Modelo de pluma gaussiana estacionaria a nivel del suelo (z=0) con fuente a altura H.
    Coordenadas locales: x = aguas abajo (downwind), y = transversal.

    C(x, y, 0) = (Q / (π·u·σy·σz)) · exp(-y² / (2σy²)) · exp(-H² / (2σz²))

    Donde:
      Q: tasa de emisión [g/s]
      u: velocidad de viento efectiva [m/s] (clamp inferior 0.5 para evitar singularidades)
      σy, σz: dispersión transversal/vertical [m] vs distancia x [m]
      H: altura efectiva de la fuente [m]
    """

    MIN_WIND_MS = 0.5

    def __init__(self, stability: str = "D", source_height_m: float = 5.0):
        self.coeffs = PG_COEFFS.get(stability, PG_COEFFS["D"])
        self.H = source_height_m

    def sigmas(self, x_downwind: float) -> Tuple[float, float]:
        x = max(1.0, x_downwind)
        c = self.coeffs
        sy = c.a * x**c.b
        sz = c.c * x**c.d
        return sy, sz

    def concentration_g_per_m3(self, x: float, y: float, Q_g_s: float, u_ms: float) -> float:
        """Concentración a nivel del suelo. x debe ser >0 (aguas abajo)."""
        if x <= 0.0:
            return 0.0
        u = max(self.MIN_WIND_MS, u_ms)
        sy, sz = self.sigmas(x)
        front = Q_g_s / (math.pi * u * sy * sz)
        cross = math.exp(-(y * y) / (2.0 * sy * sy))
        vert = math.exp(-(self.H * self.H) / (2.0 * sz * sz))
        return front * cross * vert

    @staticmethod
    def to_ppm(c_g_per_m3: float, M_g_per_mol: float = 28.0, T_k: float = 293.15) -> float:
        """
        Conversión aproximada de g/m³ a ppm volumétrico.
        ppm = (c[g/m³] · 24.45 · 1000) / M[g/mol]   (para 25°C, 1 atm; corrige por T)
        Usamos M=28 (mezcla CO + partículas en aerosol simplificada).
        """
        molar_vol = 22.414 * (T_k / 273.15)  # L/mol
        return c_g_per_m3 * (molar_vol * 1000.0) / M_g_per_mol

    def downwind_local_coords(self, fire_xy: Tuple[float, float],
                              point_xy: Tuple[float, float],
                              wind_dir_meteo_deg: float) -> Tuple[float, float]:
        """
        Convierte coords cartesianas (UTM) del punto a (x_downwind, y_crosswind)
        en el frame del incendio.
        wind_dir_meteo_deg: DE DÓNDE viene el viento (bearing meteorológico).
        Downwind bearing = wind + 180.
        """
        downwind_bearing = (wind_dir_meteo_deg + 180.0) % 360.0
        # Vector unitario downwind (en plano XY-UTM, X=este, Y=norte)
        # bearing 0=N => (0, 1); 90=E => (1, 0); por eso ux=sin, uy=cos
        ang = math.radians(downwind_bearing)
        ux, uy = math.sin(ang), math.cos(ang)
        # Vector perpendicular (90° horario) para y_crosswind: (uy, -ux)
        px, py = uy, -ux

        dx = point_xy[0] - fire_xy[0]
        dy = point_xy[1] - fire_xy[1]
        x = dx * ux + dy * uy
        y = dx * px + dy * py
        return x, y