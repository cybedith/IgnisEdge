import logging
import math
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from app.core.config import get_settings
from app.models.schemas import FuelRiskAnalysis

logger = logging.getLogger(__name__)

try:
    import ee
    EE_AVAILABLE = True
except Exception:  # pragma: no cover
    EE_AVAILABLE = False
    ee = None  # type: ignore


class GEEAnalyzer:
    """
    Análisis de combustible con Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED).
      - NDVI = (B8 - B4) / (B8 + B4)        Índice de vegetación
      - NDMI = (B8 - B11) / (B8 + B11)      Índice de humedad
    Si GEE no está disponible o falla, cae a una heurística determinística
    para no bloquear desarrollo.
    """

    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        if EE_AVAILABLE:
            self._init_ee()

    def _init_ee(self):
        try:
            if self.settings.GEE_SERVICE_ACCOUNT and self.settings.GEE_KEY_FILE:
                creds = ee.ServiceAccountCredentials(
                    self.settings.GEE_SERVICE_ACCOUNT,
                    self.settings.GEE_KEY_FILE,
                )
                ee.Initialize(credentials=creds, project=self.settings.GEE_PROJECT or None)
            else:
                ee.Initialize(project=self.settings.GEE_PROJECT or None)
            self._initialized = True
            logger.info("Earth Engine initialized.")
        except Exception as e:
            logger.warning(f"GEE init failed: {e}. Using heuristic fallback.")
            self._initialized = False

    # ---------- API ----------
    def analyze_perimeter(self, polygon_geojson: Dict[str, Any], days_back: int = 30) -> FuelRiskAnalysis:
        if not (EE_AVAILABLE and self._initialized):
            return self._fallback(polygon_geojson)
        try:
            geom = ee.Geometry(polygon_geojson)
            today = ee.Date(datetime.now(timezone.utc))
            start = today.advance(-days_back, "day")

            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geom)
                .filterDate(start, today)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .sort("system:time_start", False)
            )

            count = collection.size().getInfo()
            if not count:
                logger.info("No suitable Sentinel-2 imagery found, using fallback.")
                return self._fallback(polygon_geojson)

            image = ee.Image(collection.first())
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

            stats = (
                ndvi.addBands(ndmi)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=20,
                    bestEffort=True,
                    maxPixels=1e9,
                )
                .getInfo()
            )

            mean_ndvi = float(stats.get("NDVI") if stats.get("NDVI") is not None else 0.0)
            mean_ndmi = float(stats.get("NDMI") if stats.get("NDMI") is not None else 0.0)

            cloud = float(image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo() or 0.0)
            date_iso = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

            risk_score, level = self._compute_risk(mean_ndvi, mean_ndmi)
            return FuelRiskAnalysis(
                mean_ndvi=round(mean_ndvi, 4),
                mean_ndmi=round(mean_ndmi, 4),
                risk_score=risk_score,
                risk_level=level,
                image_date=date_iso,
                cloud_cover=round(cloud, 2),
            )
        except Exception as e:
            logger.exception(f"GEE analysis failed: {e}")
            return self._fallback(polygon_geojson)

    # ---------- Helpers ----------
    @staticmethod
    def _compute_risk(ndvi: float, ndmi: float) -> Tuple[float, str]:
        # Carga de combustible (clamp NDVI a [0.1, 0.8])
        fuel = max(0.0, min(1.0, (ndvi - 0.1) / 0.7))
        # Sequedad (NDMI alto = húmedo). Mapeamos NDMI [-0.2, 0.4] -> sequedad [1, 0]
        dryness = max(0.0, min(1.0, (0.4 - ndmi) / 0.6))
        risk = 0.55 * fuel + 0.45 * dryness
        risk = max(0.0, min(1.0, risk))
        if risk < 0.30:
            level = "LOW"
        elif risk < 0.55:
            level = "MODERATE"
        elif risk < 0.78:
            level = "HIGH"
        else:
            level = "EXTREME"
        return round(risk, 4), level

    def _fallback(self, polygon_geojson: Dict[str, Any]) -> FuelRiskAnalysis:
        try:
            from shapely.geometry import shape
            geom = shape(polygon_geojson)
            cx, cy = float(geom.centroid.x), float(geom.centroid.y)
        except Exception:
            cx, cy = 0.0, 0.0
        # NDVI/NDMI determinísticos a partir del centroide (suficiente para dev sin GEE)
        ndvi = 0.55 + 0.20 * math.sin(cx * 12.9898 + cy * 78.233)
        ndmi = 0.20 + 0.15 * math.cos(cx * 39.346 + cy * 11.135)
        ndvi = max(-0.1, min(0.9, ndvi))
        ndmi = max(-0.3, min(0.6, ndmi))
        score, level = self._compute_risk(ndvi, ndmi)
        return FuelRiskAnalysis(
            mean_ndvi=round(ndvi, 4),
            mean_ndmi=round(ndmi, 4),
            risk_score=score,
            risk_level=level,
            image_date="N/A (fallback)",
            cloud_cover=0.0,
        )


gee_analyzer = GEEAnalyzer()
