import logging
import threading
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

try:
    import ee
    EE_IMPORTED = True
except Exception:
    EE_IMPORTED = False
    ee = None  # type: ignore


class GEEClient:
    """Singleton para inicializar GEE una sola vez por proceso."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._available = False
            return cls._instance

    def initialize(self) -> bool:
        if self._initialized:
            return self._available
        self._initialized = True
        if not EE_IMPORTED:
            logger.warning("earthengine-api not installed. GEE features disabled.")
            self._available = False
            return False

        s = get_settings()
        try:
            if s.GEE_SERVICE_ACCOUNT and s.GEE_KEY_FILE:
                creds = ee.ServiceAccountCredentials(s.GEE_SERVICE_ACCOUNT, s.GEE_KEY_FILE)
                ee.Initialize(credentials=creds, project=s.GEE_PROJECT or None)
                logger.info(f"GEE initialized via Service Account ({s.GEE_SERVICE_ACCOUNT})")
            else:
                ee.Initialize(project=s.GEE_PROJECT or None)
                logger.info(f"GEE initialized via user credentials (project={s.GEE_PROJECT})")
            self._available = True
        except Exception as e:
            logger.error(f"GEE initialization failed: {e}")
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        if not self._initialized:
            self.initialize()
        return self._available

    @property
    def ee(self):
        if not self.available:
            raise RuntimeError("GEE not available. Check credentials and network.")
        return ee


gee_client = GEEClient()