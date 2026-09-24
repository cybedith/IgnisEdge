"""
ignis_link.py -- Proceso de Radio LoRa para Módem Heltec (915 MHz).
===================================================================

Responsabilidades:
  1. Enlace UART con el módem Heltec ( framing COBS + CRC16 ).
  2. Transmisión segura de alertas verificadas (ALERT v1 de 22 bytes).
  3. Recepción de alertas tempranas emitidas por nodos terrestres (ALERT_NODE).
  4. Hilo de recepción desacoplado con buffer de reconstrucción de paquetes.
  5. Soporta modo simulado / loopback para validación sin hardware conectado.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import serial  # pyserial

from edge_core.comms.ignis_proto import (
    AlertNodeMessage,
    AlertV1Message,
    MsgType,
    frame_packet,
    unframe_stream,
)

logger = logging.getLogger("ignis_link")


class IgnisLink:
    """Controlador de radio LoRa para comunicación con Heltec."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        sim_mode: bool = False,
    ):
        self.port = port
        self.baud = baud
        self.sim_mode = sim_mode

        self._ser: Optional[serial.Serial] = None
        self._rx_buffer = b""
        self._running = False
        self._rx_thread: Optional[threading.Thread] = None

        # Callbacks para eventos entrantes
        self.on_node_alert: Optional[Callable[[AlertNodeMessage], None]] = None
        self.on_node_heartbeat: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_packet_received: Optional[Callable[[bytes], None]] = None
        self.nodes_status: Dict[int, Dict[str, Any]] = {}

    def connect(self) -> bool:
        """Abre la interfaz serial hacia el módem Heltec."""
        if self.sim_mode:
            logger.info("Modo Simulación activo: IgnisLink operando sin hardware.")
            self._running = True
            return True

        try:
            logger.info(f"Conectando a módem Heltec en {self.port} ({self.baud} baud)...")
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1, write_timeout=0.2)
            self._ser.dtr = True
            self._running = True
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            logger.info("Enlace LoRa activo y a la escucha.")
            return True
        except Exception as e:
            logger.error(f"Fallo abriendo puerto serial {self.port}: {e}")
            return False

    def _rx_loop(self) -> None:
        """Loop de recepción continua y desempaquetado de stream UART."""
        while self._running and self._ser:
            try:
                chunk = self._ser.read(64)
                if chunk:
                    self._rx_buffer += chunk
                    payloads, self._rx_buffer = unframe_stream(self._rx_buffer)
                    for p in payloads:
                        self._dispatch_payload(p)

                    # Soporte fallback para tramas de texto ASCII / CSV / JSON
                    if b"\n" in self._rx_buffer:
                        lines = self._rx_buffer.split(b"\n")
                        self._rx_buffer = lines[-1]
                        for raw_line in lines[:-1]:
                            line_str = raw_line.decode(errors="replace").strip()
                            if line_str:
                                self._parse_ascii_line(line_str)
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.warning(f"Error en lectura serial LoRa: {e}")
                time.sleep(0.01)

    def _parse_ascii_line(self, text: str) -> None:
        """Parsea líneas de texto ASCII emitidas por la red LoRa (ALERT y HEARTBEAT en modo simple o multi-hop)."""
        logger.info(f"Trama LoRa de texto recibida: {text}")
        if self.on_packet_received:
            self.on_packet_received(text.encode("utf-8"))

        # 1. Separar por coma o punto y coma
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if not parts or len(parts) < 4:
            return

        hdr = parts[0].upper()

        # A. Trama ALERT
        if hdr.startswith("ALERT"):
            try:
                if len(parts) >= 7:
                    # Multi-hop: ALERT,src_id,seq,ttl,lat,lon,confidence
                    node_id = int(parts[1]) if parts[1].isdigit() else 1
                    seq = int(parts[2]) if parts[2].isdigit() else 1
                    ttl = int(parts[3]) if parts[3].isdigit() else 3
                    lat = float(parts[4])
                    lon = float(parts[5])
                    conf = int(float(parts[6]))
                elif len(parts) >= 5:
                    # Legacy: ALERT,src_id,lat,lon,confidence
                    node_id = int(parts[1]) if parts[1].isdigit() else 1
                    seq = 1
                    ttl = 3
                    lat = float(parts[2])
                    lon = float(parts[3])
                    conf = int(float(parts[4]))
                else:
                    return

                hops = max(0, 3 - ttl)
                alert = AlertNodeMessage(
                    node_id=node_id, seq=seq, lat=lat, lon=lon, gas_res_kohm=50, confidence=conf
                )
                logger.info(
                    f"Alerta de Nodo decodificada -> ID: {node_id}, Seq: {seq}, Saltos: {hops}, Lat: {lat:.6f}, Lon: {lon:.6f}, Conf: {conf}%"
                )
                if self.on_node_alert:
                    self.on_node_alert(alert)
                return
            except Exception as e:
                logger.warning(f"Error parseando ALERT ASCII: {e}")
                return

        # B. Trama HEARTBEAT
        if hdr.startswith("HEARTBEAT"):
            try:
                if len(parts) >= 9:
                    # Multi-hop: HEARTBEAT,src_id,seq,ttl,lat,lon,batt,temp,gas
                    node_id = int(parts[1]) if parts[1].isdigit() else 1
                    seq = int(parts[2]) if parts[2].isdigit() else 1
                    ttl = int(parts[3]) if parts[3].isdigit() else 3
                    lat = float(parts[4])
                    lon = float(parts[5])
                    batt = int(float(parts[6]))
                    temp = float(parts[7])
                    gas_kohm = float(parts[8])
                elif len(parts) >= 7:
                    # Legacy: HEARTBEAT,src_id,lat,lon,batt,temp,gas
                    node_id = int(parts[1]) if parts[1].isdigit() else 1
                    seq = 1
                    ttl = 3
                    lat = float(parts[2])
                    lon = float(parts[3])
                    batt = int(float(parts[4]))
                    temp = float(parts[5])
                    gas_kohm = float(parts[6])
                else:
                    return

                hops = max(0, 3 - ttl)
                hb_data = {
                    "node_id": node_id,
                    "seq": seq,
                    "hops": hops,
                    "lat": lat,
                    "lon": lon,
                    "batt": batt,
                    "temp": temp,
                    "gas_res_kohm": gas_kohm,
                    "last_seen": time.time(),
                }
                self.nodes_status[node_id] = hb_data
                logger.info(
                    f"Heartbeat Nodo #{node_id} (Saltos: {hops}) -> Bat: {batt}%, T: {temp:.1f}C, Gas: {gas_kohm:.1f}kOhm"
                )
                if self.on_node_heartbeat:
                    self.on_node_heartbeat(hb_data)
                return
            except Exception as e:
                logger.warning(f"Error parseando HEARTBEAT ASCII: {e}")
                return

        # C. Formato JSON fallback
        try:
            if text.startswith("{") and text.endswith("}"):
                import json
                d = json.loads(text)
                lat = float(d.get("lat", d.get("latitude", 0.0)))
                lon = float(d.get("lon", d.get("longitude", 0.0)))
                if lat != 0.0 and lon != 0.0:
                    node_id = int(d.get("node_id", d.get("node", 1)))
                    conf = int(d.get("conf", d.get("confidence", 80)))
                    alert = AlertNodeMessage(
                        node_id=node_id, seq=1, lat=lat, lon=lon, gas_res_kohm=50, confidence=conf
                    )
                    logger.info(f"Alerta de Nodo (JSON) decodificada -> ID: {node_id}, Lat: {lat}, Lon: {lon}")
                    if self.on_node_alert:
                        self.on_node_alert(alert)
        except Exception:
            pass

    def _dispatch_payload(self, payload: bytes) -> None:
        """Rutea el paquete según su MsgType."""
        if not payload:
            return

        msg_type = payload[0]
        if self.on_packet_received:
            self.on_packet_received(payload)

        if msg_type == MsgType.ALERT_NODE:
            try:
                node_alert = AlertNodeMessage.unpack(payload)
                logger.info(
                    f"Alerta de Nodo recibida -> ID: {node_alert.node_id}, "
                    f"Lat: {node_alert.lat:.5f}, Lon: {node_alert.lon:.5f}, Conf: {node_alert.confidence}%"
                )
                if self.on_node_alert:
                    self.on_node_alert(node_alert)
            except Exception as e:
                logger.warning(f"Error deserializando ALERT_NODE: {e}")

    def send_alert(self, alert: AlertV1Message) -> bool:
        """Empaqueta y transmite un mensaje ALERT v1 de 22 bytes por LoRa."""
        raw_msg = alert.pack()
        framed = frame_packet(raw_msg)

        if self.sim_mode:
            logger.info(f"[SIM] ALERT v1 transmitido por LoRa (22 B, frame: {len(framed)} B)")
            return True

        if not self._ser or not self._ser.is_open:
            logger.error("No se puede enviar alerta: puerto serial cerrado.")
            return False

        try:
            self._ser.write(framed)
            logger.info(f"ALERT v1 emitido por LoRa ({len(framed)} B framed) -> Confianza: {alert.confidence}%")
            return True
        except Exception as e:
            logger.error(f"Error transmitiendo alerta LoRa: {e}")
            return False

    def inject_simulated_packet(self, raw_packet: bytes) -> None:
        """Inyecta un paquete simulado (útil para pruebas de integración sin radio)."""
        self._dispatch_payload(raw_packet)

    def disconnect(self) -> None:
        """Cierra el puerto serial y finaliza el hilo de recepción."""
        self._running = False
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        logger.info("IgnisLink desconectado.")
