import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.mqtt_client import mqtt_manager
from app.core.simulation_manager import simulation_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    queue = mqtt_manager.subscribe()
    try:
        await websocket.send_json({"type": "hello", "channel": "telemetry"})
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15.0)
                await websocket.send_json({"type": "telemetry", "data": item.model_dump(mode="json")})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info("Telemetry client disconnected")
    except Exception as e:
        logger.exception(f"Telemetry WS error: {e}")
    finally:
        mqtt_manager.unsubscribe(queue)


@router.websocket("/ws/simulation/{session_id}")
async def simulation_ws(websocket: WebSocket, session_id: str):
    sess = simulation_manager.get(session_id)
    if not sess:
        await websocket.close(code=4404, reason="Session not found")
        return
    await websocket.accept()
    queue = sess.subscribe()
    try:
        await websocket.send_json({
            "type": "hello",
            "channel": "simulation",
            "session_id": session_id,
            "state": sess.state_summary().model_dump(),
        })
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=10.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info(f"Simulation WS client disconnected from {session_id}")
    except Exception as e:
        logger.exception(f"Simulation WS error: {e}")
    finally:
        sess.unsubscribe(queue)