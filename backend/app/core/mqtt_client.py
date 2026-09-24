import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Set

import paho.mqtt.client as mqtt

from app.core.config import get_settings
from app.models.schemas import TelemetryData

logger = logging.getLogger(__name__)


class MqttManager:
    """
    Bridge entre paho-mqtt (callbacks en hilo de fondo) y asyncio (frontend WS).
    Cada conexión WebSocket se suscribe via subscribe() y obtiene su propia Queue.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[mqtt.Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: Set[asyncio.Queue] = set()
        self._connected: bool = False

    # ---------- Lifecycle ----------
    def attach_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def start(self):
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.settings.MQTT_CLIENT_ID,
                protocol=mqtt.MQTTv311,
            )
        except AttributeError:
            # paho-mqtt < 2.0 fallback
            client = mqtt.Client(client_id=self.settings.MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)

        if self.settings.MQTT_USERNAME:
            client.username_pw_set(self.settings.MQTT_USERNAME, self.settings.MQTT_PASSWORD)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=30)

        try:
            client.connect_async(self.settings.MQTT_BROKER, self.settings.MQTT_PORT, keepalive=60)
            client.loop_start()
            self.client = client
            logger.info(
                f"MQTT client started -> {self.settings.MQTT_BROKER}:{self.settings.MQTT_PORT}"
            )
        except Exception as e:
            logger.exception(f"MQTT failed to start: {e}")

    def stop(self):
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.warning(f"Error stopping MQTT client: {e}")
        self.client = None
        self._connected = False

    # ---------- Pub/Sub interno ----------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---------- Callbacks paho ----------
    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None):
        # paho v2: reason_code es ReasonCode (int-like). 0 = success.
        ok = (int(reason_code) == 0) if reason_code is not None else False
        if ok:
            self._connected = True
            topic = f"{self.settings.MQTT_BASE_TOPIC}/+/telemetry"
            client.subscribe(topic, qos=1)
            logger.info(f"MQTT connected. Subscribed to {topic}")
        else:
            self._connected = False
            logger.error(f"MQTT connect failed reason={reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        self._connected = False
        logger.warning(f"MQTT disconnected reason={reason_code}")

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8", errors="replace")
            data = json.loads(payload_str) if payload_str.strip().startswith("{") else {}

            parts = msg.topic.split("/")
            if "node_id" not in data and len(parts) >= 3:
                # ignis/heltec/<node_id>/telemetry
                data["node_id"] = parts[-2]
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()

            telemetry = TelemetryData(**data)
            self._dispatch(telemetry)
        except Exception as e:
            logger.exception(f"Failed to parse MQTT message on {msg.topic}: {e}")

    def _dispatch(self, telemetry: TelemetryData):
        if self._loop is None:
            return
        for q in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._safe_put, q, telemetry)

    @staticmethod
    def _safe_put(queue: asyncio.Queue, item):
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(item)
            except Exception:
                pass


mqtt_manager = MqttManager()
