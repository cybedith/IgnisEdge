from typing import List
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Ignis Edge C2 Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ]

    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    MQTT_BASE_TOPIC: str = "ignis/heltec"
    MQTT_CLIENT_ID: str = "ignis-edge-backend"

    GEE_SERVICE_ACCOUNT: str = ""
    GEE_KEY_FILE: str = ""
    GEE_PROJECT: str = "ignis-edge-dev"

    LORA_FREQ_MHZ: float = 915.0
    ANTENNA_HEIGHT_M: float = 5.0

    NODE_DETECTION_RADIUS_M: float = 1500.0
    NODE_SMOKE_THRESHOLD_PPM: float = 50.0

    SIM_TICK_HZ: float = 2.0
    SIM_DEFAULT_SPEED_MULTIPLIER: float = 30.0
    SIM_GRID_RESOLUTION_M: float = 30.0

    DRONE_CRUISE_SPEED_MS: float = 15.0
    DRONE_SAFE_AGL_M: float = 80.0


@lru_cache()
def get_settings() -> Settings:
    return Settings()