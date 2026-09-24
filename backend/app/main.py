import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.mqtt_client import mqtt_manager
from app.core.gee_client import gee_client
from app.api import v1_routes, ws_routes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("ignis-edge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Booting {settings.APP_NAME}")
    # Inicializar GEE temprano para que falle rápido si está mal configurado
    if gee_client.initialize():
        logger.info("GEE ready.")
    else:
        logger.warning("GEE NOT available — terrain/weather endpoints will return 503.")
    loop = asyncio.get_running_loop()
    mqtt_manager.attach_loop(loop)
    mqtt_manager.start()
    try:
        yield
    finally:
        logger.info("Shutting down.")
        mqtt_manager.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version="2.0.0",
        description="Ignis Edge — Backend C2 con simulación tick-based, GEE, TDoA y RRT*-3D.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(v1_routes.router, prefix=settings.API_V1_PREFIX, tags=["v1"])
    app.include_router(ws_routes.router, tags=["websocket"])

    @app.get("/")
    async def root():
        return {"app": settings.APP_NAME, "docs": "/docs", "version": "2.0.0"}

    return app


app = create_app()