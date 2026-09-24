import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from app.core.gee_client import gee_client
from app.models.schemas import WeatherSnapshot

logger = logging.getLogger(__name__)


class WeatherService:
    """
    Datos meteorológicos en tiempo casi-real desde NOAA GFS 0.25° vía GEE.
    Resolución ~25 km. Para áreas pequeñas el campo es prácticamente uniforme.
    Cachea por punto + hora del forecast.
    """

    GFS_ASSET = "NOAA/GFS0P25"
    LOOKBACK_HOURS = 12  # Buscamos hasta 12h hacia atrás si el forecast más reciente no está

    def __init__(self):
        self._cache: dict = {}

    def get_current(self, lon: float, lat: float) -> WeatherSnapshot:
        if not gee_client.available:
            raise RuntimeError("GEE no disponible. No se puede consultar NOAA GFS.")
        ee = gee_client.ee

        # Cache key: redondear coords y hora UTC para evitar saturar GEE
        key = (round(lon, 2), round(lat, 2), datetime.now(timezone.utc).strftime("%Y%m%d%H"))
        if key in self._cache:
            return self._cache[key]

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=self.LOOKBACK_HOURS)

        col = (
            ee.ImageCollection(self.GFS_ASSET)
            .filterDate(ee.Date(start), ee.Date(now))
            .filter(ee.Filter.eq("forecast_hours", 0))
            .sort("system:time_start", False)
        )
        size = col.size().getInfo()
        if size == 0:
            raise RuntimeError("No se encontraron pronósticos GFS recientes.")

        image = ee.Image(col.first())
        point = ee.Geometry.Point([lon, lat])
        bands = [
            "u_component_of_wind_10m_above_ground",
            "v_component_of_wind_10m_above_ground",
            "temperature_2m_above_ground",
            "relative_humidity_2m_above_ground",
        ]
        sample = (
            image.select(bands)
            .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=25000, bestEffort=True)
            .getInfo()
        )

        u = float(sample.get("u_component_of_wind_10m_above_ground") or 0.0)
        v = float(sample.get("v_component_of_wind_10m_above_ground") or 0.0)
        # GFS: la temperatura ya viene en °C en este asset
        t_c = float(sample.get("temperature_2m_above_ground") or 0.0)
        rh = float(sample.get("relative_humidity_2m_above_ground") or 0.0)

        speed = math.hypot(u, v)
        # u: viento hacia el ESTE (componente +x), v: hacia el NORTE (componente +y).
        # La dirección METEOROLÓGICA es DE DÓNDE viene el viento.
        # bearing del vector "hacia donde sopla" = atan2(u, v) (medido desde norte, sentido horario)
        # bearing meteorológico = bearing_hacia + 180
        if speed < 1e-3:
            wind_dir_meteo = 0.0
        else:
            bearing_to = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0
            wind_dir_meteo = (bearing_to + 180.0) % 360.0

        ts_millis = image.get("system:time_start").getInfo()
        forecast_time = datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc).isoformat()

        snapshot = WeatherSnapshot(
            longitude=lon,
            latitude=lat,
            forecast_time=forecast_time,
            wind_speed_ms=round(speed, 3),
            wind_direction_deg=round(wind_dir_meteo, 2),
            temperature_c=round(t_c, 2),
            relative_humidity_pct=max(0.0, min(100.0, round(rh, 2))),
        )
        self._cache[key] = snapshot
        return snapshot


weather_service = WeatherService()