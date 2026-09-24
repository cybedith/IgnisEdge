import logging
from fastapi import APIRouter, HTTPException

from shapely.geometry import shape

from app.models.schemas import (
    PerimeterIn, FuelRiskAnalysis, MeshOptimizationIn, NodePlacementOut,
    SimulationCreateIn, SimulationStateOut, HotSpotIn,
    DroneRouteRequest, DroneRoute, WeatherSnapshot, TerrainStats,
    NodeLocation,
)
from app.services.gee_analyzer import gee_analyzer
from app.services.terrain_service import terrain_service
from app.services.weather_service import weather_service
from app.services.mesh_optimizer import mesh_optimizer
from app.core.simulation_manager import simulation_manager
from app.core.gee_client import gee_client
from app.core.mqtt_client import mqtt_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# Health
# ============================================================
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "gee": gee_client.available,
        "mqtt": mqtt_manager.is_connected,
        "active_sessions": len(simulation_manager.list_sessions()),
    }


# ============================================================
# Etapa 1-2: Perímetro y análisis del terreno
# ============================================================
@router.post("/perimeter/analyze", response_model=FuelRiskAnalysis)
async def analyze_perimeter(payload: PerimeterIn):
    try:
        return gee_analyzer.analyze_perimeter(payload.polygon.model_dump())
    except Exception as e:
        logger.exception("perimeter/analyze failed")
        raise HTTPException(500, f"Analysis failed: {e}")


@router.post("/terrain/stats", response_model=TerrainStats)
async def terrain_stats(payload: PerimeterIn):
    try:
        poly = shape(payload.polygon.model_dump())
        tile = terrain_service.get_tile_for_polygon(poly)
        return tile.stats()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("terrain/stats failed")
        raise HTTPException(500, str(e))


@router.get("/weather/current", response_model=WeatherSnapshot)
async def weather_current(lon: float, lat: float):
    try:
        return weather_service.get_current(lon, lat)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("weather/current failed")
        raise HTTPException(500, str(e))


# ============================================================
# Etapa 3: Optimización de nodos
# ============================================================
@router.post("/mesh/optimize", response_model=NodePlacementOut)
async def mesh_optimize(payload: MeshOptimizationIn):
    try:
        return mesh_optimizer.optimize(payload)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("mesh/optimize failed")
        raise HTTPException(500, str(e))


# ============================================================
# Etapas 4-6: Simulación / producción
# ============================================================
@router.post("/simulation/create", response_model=SimulationStateOut)
async def simulation_create(payload: SimulationCreateIn):
    try:
        sess = await simulation_manager.create(payload)
        return sess.state_summary()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("simulation/create failed")
        raise HTTPException(500, str(e))


@router.get("/simulation", response_model=list[SimulationStateOut])
async def simulation_list():
    return simulation_manager.list_sessions()


@router.get("/simulation/{session_id}", response_model=SimulationStateOut)
async def simulation_get(session_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess.state_summary()


@router.post("/simulation/{session_id}/start", response_model=SimulationStateOut)
async def simulation_start(session_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    await sess.start()
    return sess.state_summary()


@router.post("/simulation/{session_id}/pause", response_model=SimulationStateOut)
async def simulation_pause(session_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    await sess.pause()
    return sess.state_summary()


@router.post("/simulation/{session_id}/resume", response_model=SimulationStateOut)
async def simulation_resume(session_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    await sess.resume()
    return sess.state_summary()


@router.delete("/simulation/{session_id}")
async def simulation_delete(session_id: str):
    ok = await simulation_manager.delete(session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"deleted": session_id}


@router.post("/simulation/{session_id}/hotspot")
async def add_hotspot(session_id: str, hs: HotSpotIn):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    hs_id = await sess.add_hot_spot(hs)
    return {"hot_spot_id": hs_id}


@router.delete("/simulation/{session_id}/hotspot/{hot_spot_id}")
async def remove_hotspot(session_id: str, hot_spot_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    ok = await sess.remove_hot_spot(hot_spot_id)
    if not ok:
        raise HTTPException(404, "Hot spot not found or already removed")
    return {"hot_spot_id": hot_spot_id, "active": False}


@router.post("/simulation/{session_id}/drone/launch", response_model=DroneRoute)
async def launch_drone(session_id: str, req: DroneRouteRequest):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    try:
        return await sess.launch_drone(req)
    except Exception as e:
        logger.exception("drone launch failed")
        raise HTTPException(500, f"Drone launch failed: {e}")

from pydantic import BaseModel
class TargetCoordinates(BaseModel):
    drone_id: str
    target_lon: float
    target_lat: float

@router.post("/simulation/{session_id}/target", response_model=DroneRoute)
async def set_drone_target(session_id: str, target: TargetCoordinates):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if target.drone_id not in sess.drones:
        raise HTTPException(404, "Drone not found in session")
        
    drone = sess.drones[target.drone_id]
    req = DroneRouteRequest(
        drone_id=target.drone_id,
        start_lon=drone.state.longitude,
        start_lat=drone.state.latitude,
        start_alt_m=drone.state.alt_amsl_m,
        target_lon=target.target_lon,
        target_lat=target.target_lat
    )
    try:
        return await sess.launch_drone(req)
    except Exception as e:
        logger.exception("drone target routing failed")
        raise HTTPException(500, f"Drone target routing failed: {e}")


@router.post("/simulation/{session_id}/node")
async def add_simulation_node(session_id: str, node: NodeLocation):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    await sess.add_node(node)
    return {"status": "ok", "node": node}


@router.post("/simulation/{session_id}/node/{node_id}/alert")
async def trigger_node_alert(session_id: str, node_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    route = await sess.trigger_node_alert(node_id)
    if route is None:
        raise HTTPException(404, f"Node {node_id} not found in session")
    return {"status": "alert_triggered", "node_id": node_id, "route": route}


class DroneCommandIn(BaseModel):
    command: str


@router.post("/simulation/{session_id}/drone/{drone_id}/command")
async def drone_command(session_id: str, drone_id: str, payload: DroneCommandIn):
    sess = simulation_manager.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    ok = await sess.command_drone(drone_id, payload.command)
    if not ok:
        raise HTTPException(404, f"Drone {drone_id} not found in session")
    return {"status": "ok", "drone_id": drone_id, "command": payload.command}