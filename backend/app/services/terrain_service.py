import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from app.core.config import get_settings
from app.core.gee_client import gee_client
from app.models.schemas import TerrainStats

logger = logging.getLogger(__name__)


@dataclass
class TerrainTile:
    """DEM + derivados (slope, aspect) sobre una grilla regular en lon/lat."""
    bbox: Tuple[float, float, float, float]   # minlon, minlat, maxlon, maxlat
    elevation: np.ndarray                     # shape (rows, cols), metros
    slope_deg: np.ndarray                     # shape (rows, cols), grados
    aspect_deg: np.ndarray                    # shape (rows, cols), grados (0=N, 90=E)
    cell_size_m_x: float
    cell_size_m_y: float

    @property
    def rows(self) -> int:
        return self.elevation.shape[0]

    @property
    def cols(self) -> int:
        return self.elevation.shape[1]

    def lonlat_to_idx(self, lon: float, lat: float) -> Tuple[int, int]:
        minlon, minlat, maxlon, maxlat = self.bbox
        col = int((lon - minlon) / (maxlon - minlon) * (self.cols - 1))
        # Filas: arriba (norte) = 0
        row = int((maxlat - lat) / (maxlat - minlat) * (self.rows - 1))
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return row, col

    def elevation_at(self, lon: float, lat: float) -> float:
        r, c = self.lonlat_to_idx(lon, lat)
        return float(self.elevation[r, c])

    def slope_at(self, lon: float, lat: float) -> float:
        r, c = self.lonlat_to_idx(lon, lat)
        return float(self.slope_deg[r, c])

    def aspect_at(self, lon: float, lat: float) -> float:
        r, c = self.lonlat_to_idx(lon, lat)
        return float(self.aspect_deg[r, c])

    def stats(self) -> TerrainStats:
        return TerrainStats(
            bbox=self.bbox,
            rows=self.rows,
            cols=self.cols,
            elev_min_m=float(np.nanmin(self.elevation)),
            elev_max_m=float(np.nanmax(self.elevation)),
            elev_mean_m=float(np.nanmean(self.elevation)),
            slope_max_deg=float(np.nanmax(self.slope_deg)),
            slope_mean_deg=float(np.nanmean(self.slope_deg)),
        )


class TerrainService:
    """
    Descarga DEM desde GEE (USGS/SRTMGL1_003) para un bbox dado y mantiene cache.
    Computa slope/aspect localmente (más confiable y rápido que pedirlos a GEE).
    """

    SRTM_ASSET = "USGS/SRTMGL1_003"
    DEFAULT_SCALE_M = 30.0
    MAX_PIXELS_PER_AXIS = 350  # Límite para no saturar getDownloadURL/sampleRectangle

    def __init__(self):
        self.settings = get_settings()
        self._cache: dict = {}
        self._lock = threading.Lock()

    # ---------- API pública ----------
    def get_tile_for_polygon(self, polygon: Polygon, pad_deg: float = 0.005) -> TerrainTile:
        minlon, minlat, maxlon, maxlat = polygon.bounds
        bbox = (minlon - pad_deg, minlat - pad_deg, maxlon + pad_deg, maxlat + pad_deg)
        return self.get_tile(bbox)

    def get_tile(self, bbox: Tuple[float, float, float, float]) -> TerrainTile:
        key = self._cache_key(bbox)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        tile = self._download_dem(bbox)
        with self._lock:
            self._cache[key] = tile
        logger.info(f"Cached DEM tile {tile.rows}x{tile.cols} for bbox={bbox}")
        return tile

    # ---------- GEE download ----------
    def _download_dem(self, bbox: Tuple[float, float, float, float]) -> TerrainTile:
        minlon, minlat, maxlon, maxlat = bbox
        
        if not gee_client.available:
            logger.warning("GEE no disponible. Retornando terreno plano simulado.")
            approx_w_m = self._haversine_m(minlon, minlat, maxlon, minlat)
            approx_h_m = self._haversine_m(minlon, minlat, minlon, maxlat)
            rows, cols = 100, 100
            elev = np.zeros((rows, cols), dtype=np.float32)
            slope = np.zeros_like(elev)
            aspect = np.zeros_like(elev)
            return TerrainTile(
                bbox=bbox,
                elevation=elev,
                slope_deg=slope,
                aspect_deg=aspect,
                cell_size_m_x=approx_w_m / cols,
                cell_size_m_y=approx_h_m / rows,
            )

        ee = gee_client.ee

        # Calculamos la resolución para no exceder MAX_PIXELS_PER_AXIS
        approx_w_m = self._haversine_m(minlon, minlat, maxlon, minlat)
        approx_h_m = self._haversine_m(minlon, minlat, minlon, maxlat)
        scale_x = max(self.DEFAULT_SCALE_M, approx_w_m / self.MAX_PIXELS_PER_AXIS)
        scale_y = max(self.DEFAULT_SCALE_M, approx_h_m / self.MAX_PIXELS_PER_AXIS)
        scale = max(scale_x, scale_y)

        region = ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat], proj="EPSG:4326",
                                       geodesic=False)
        image = ee.Image(self.SRTM_ASSET).select("elevation").clip(region)

        # sampleRectangle entrega un ndarray directo (mejor para tiles chicos/medianos)
        try:
            sample = image.sampleRectangle(region=region, defaultValue=0)
            elev = np.array(sample.get("elevation").getInfo(), dtype=np.float32)
        except Exception as e:
            logger.warning(f"sampleRectangle failed ({e}); falling back to getDownloadURL.")
            elev = self._fallback_geotiff(image, region, scale)

        # Estimación de tamaño de celda en metros
        rows, cols = elev.shape
        cell_size_m_x = approx_w_m / cols
        cell_size_m_y = approx_h_m / rows

        # Slope/aspect via Horn (1981)
        slope, aspect = self._compute_slope_aspect(elev, cell_size_m_x, cell_size_m_y)

        return TerrainTile(
            bbox=bbox,
            elevation=elev,
            slope_deg=slope,
            aspect_deg=aspect,
            cell_size_m_x=cell_size_m_x,
            cell_size_m_y=cell_size_m_y,
        )

    def _fallback_geotiff(self, image, region, scale: float) -> np.ndarray:
        """Si sampleRectangle falla, descargamos GeoTIFF y lo leemos con rasterio."""
        import io
        import requests
        import rasterio

        url = image.getDownloadURL({
            "scale": scale,
            "region": region,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with rasterio.open(io.BytesIO(resp.content)) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr = np.where(np.isnan(arr), 0.0, arr)
        return arr

    # ---------- Slope/Aspect ----------
    @staticmethod
    def _compute_slope_aspect(
        elev: np.ndarray, dx: float, dy: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algoritmo de Horn (3x3 kernel). Salida en grados.
        Aspect: 0 = Norte, 90 = Este (convención meteorológica de bearing).
        """
        rows, cols = elev.shape
        if rows < 3 or cols < 3:
            return np.zeros_like(elev), np.zeros_like(elev)

        # Kernels de Horn
        # dz/dx = ((c+2f+i) - (a+2d+g)) / (8 dx)
        z = elev
        a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
        d, _, f = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
        g, h, i = z[2:, :-2],  z[2:, 1:-1],  z[2:, 2:]

        dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
        # En arrays "imagen", las filas crecen hacia el sur. Para que +y apunte a norte
        # invertimos el signo, así dz/dy positivo = sube hacia el norte
        dzdy = -((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)

        slope_rad = np.arctan(np.hypot(dzdx, dzdy))
        slope_deg_inner = np.degrees(slope_rad)

        # Aspect: bearing del vector ascendente (apunta hacia donde la ladera sube).
        # En convención meteo: 0=N, 90=E.
        aspect_rad = np.arctan2(dzdx, dzdy)        # ángulo desde el norte hacia el este
        aspect_deg_inner = (np.degrees(aspect_rad) + 360.0) % 360.0
        # Donde la pendiente es ~0, aspect indefinido → 0
        aspect_deg_inner = np.where(slope_deg_inner < 1e-3, 0.0, aspect_deg_inner)

        # Padding para volver al tamaño original
        slope = np.zeros_like(elev)
        aspect = np.zeros_like(elev)
        slope[1:-1, 1:-1] = slope_deg_inner
        aspect[1:-1, 1:-1] = aspect_deg_inner
        # Bordes: copiar vecino interior
        slope[0, :] = slope[1, :]; slope[-1, :] = slope[-2, :]
        slope[:, 0] = slope[:, 1]; slope[:, -1] = slope[:, -2]
        aspect[0, :] = aspect[1, :]; aspect[-1, :] = aspect[-2, :]
        aspect[:, 0] = aspect[:, 1]; aspect[:, -1] = aspect[:, -2]
        return slope, aspect

    # ---------- Helpers ----------
    @staticmethod
    def _cache_key(bbox: Tuple[float, float, float, float]) -> str:
        return ",".join(f"{v:.5f}" for v in bbox)

    @staticmethod
    def _haversine_m(lon1, lat1, lon2, lat2) -> float:
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))


terrain_service = TerrainService()