"""
ignis_mavlink.py -- Interfaz MAVLink Pura para Pixhawk 6C (ArduCopter).
========================================================================

Principios de Diseño Safety-Critical:
  1. El Pixhawk es la autoridad de vuelo; la Jetson sugiere, no manda.
  2. Watchdog de Heartbeat: si se pierde comunicación por > 2.0 s, se activa alarma.
  3. Comandos de guiado mediante SET_POSITION_TARGET_GLOBAL_INT en modo GUIDED.
  4. Lazo de recepción en hilo dedicado desacoplado (non-blocking).
  5. Soporta UART (/dev/ttyTHS1), USB (/dev/ttyACM0) y simulación UDP (SITL).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from pymavlink import mavutil

logger = logging.getLogger("ignis_mavlink")

def _pwm_to_pct(pwm_raw: int) -> int:
    """Normaliza PWM (1000-2000) a % (0-100) y satura ante PWM anómalo del ESC (NASA FDIR)."""
    pct = round((pwm_raw - 1000) * 100 / 1000)
    return max(0, min(100, pct))

@dataclass(slots=True)
class MavlinkTelemetry:
    lat: float = 0.0             # Grados decimales
    lon: float = 0.0             # Grados decimales
    alt_rel_m: float = 0.0       # Altitud sobre punto de despegue (m)
    alt_msl_m: float = 0.0       # Altitud sobre nivel del mar (m)
    heading_deg: float = 0.0     # 0..360°
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    voltage_v: float = 0.0
    current_a: float = 0.0
    battery_pct: int = -1        # -1 = desconocido
    mode: str = "UNKNOWN"
    armed: bool = False
    gps_fix_type: int = 0        # 0=sin fix, 3=3D fix, 4=DGPS/RTK
    satellites_visible: int = 0
    last_heartbeat: float = 0.0
    heartbeat_healthy: bool = False
    motor_1_pct: int = 0
    motor_2_pct: int = 0
    motor_3_pct: int = 0
    motor_4_pct: int = 0

class IgnisMavlink:
    """Cliente MAVLink puro para comunicación entre Jetson y Pixhawk."""

    def __init__(
        self,
        connection_string: str = "/dev/ttyTHS1",
        baud: int = 921600,
        source_system: int = 1,
        source_component: int = 191,  # MAV_COMP_ID_ONBOARD_COMPUTER
        heartbeat_timeout_s: float = 2.0,
    ):
        self.conn_str = connection_string
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component
        self.heartbeat_timeout_s = heartbeat_timeout_s

        self.telemetry = MavlinkTelemetry()
        self._lock = threading.Lock()
        self._master: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_heartbeat_lost: Optional[Callable[[], None]] = None

    def connect(self, timeout_s: float = 5.0) -> bool:
        """Establece conexión y espera el primer heartbeat del Pixhawk."""
        logger.info(f"Conectando a Pixhawk vía {self.conn_str} ({self.baud} baud)...")
        try:
            self._master = mavutil.mavlink_connection(
                self.conn_str,
                baud=self.baud,
                source_system=self.source_system,
                source_component=self.source_component,
            )
        except Exception as e:
            logger.error(f"Error abriendo puerto MAVLink {self.conn_str}: {e}")
            return False

        logger.info("Esperando heartbeat inicial del autopiloto...")
        t_start = time.time()
        while time.time() - t_start < timeout_s:
            try:
                msg = self._master.wait_heartbeat(timeout=0.5)
            except Exception:
                time.sleep(0.05)
                continue

            if msg:
                with self._lock:
                    self.telemetry.last_heartbeat = time.time()
                    self.telemetry.heartbeat_healthy = True
                    self.telemetry.mode = mavutil.mode_string_v10(msg)
                    self.telemetry.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                logger.info(f"Pixhawk detectado! Modo: {self.telemetry.mode} | Armed: {self.telemetry.armed}")

                self._running = True
                self._thread = threading.Thread(target=self._rx_loop, daemon=True)
                self._thread.start()
                return True

        logger.warning(f"Timeout ({timeout_s}s) esperando heartbeat del Pixhawk.")
        return False

    def _rx_loop(self) -> None:
        """Hilo de fondo para procesar mensajes MAVLink entrantes de forma continua."""
        last_hb_send = 0.0

        while self._running and self._master:
            now = time.time()

            # Enviar heartbeat periódico de la Jetson hacia el Pixhawk (1 Hz)
            if now - last_hb_send >= 1.0:
                self._master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                last_hb_send = now

            # Leer mensajes de forma no bloqueante
            try:
                msg = self._master.recv_match(blocking=False)
            except Exception as e:
                logger.warning(f"Excepción leyendo MAVLink: {e}")
                time.sleep(0.01)
                continue

            if msg is not None:
                msg_type = msg.get_type()
                with self._lock:
                    if msg_type == "HEARTBEAT":
                        self.telemetry.last_heartbeat = now
                        self.telemetry.heartbeat_healthy = True
                        self.telemetry.mode = mavutil.mode_string_v10(msg)
                        self.telemetry.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                    elif msg_type == "GLOBAL_POSITION_INT":
                        self.telemetry.lat = msg.lat / 1e7
                        self.telemetry.lon = msg.lon / 1e7
                        self.telemetry.alt_rel_m = msg.relative_alt / 1000.0
                        self.telemetry.alt_msl_m = msg.alt / 1000.0
                        self.telemetry.heading_deg = msg.hdg / 100.0

                    elif msg_type == "ATTITUDE":
                        self.telemetry.roll_deg = math.degrees(msg.roll)
                        self.telemetry.pitch_deg = math.degrees(msg.pitch)
                        self.telemetry.yaw_deg = math.degrees(msg.yaw)

                    elif msg_type == "SYS_STATUS":
                        self.telemetry.voltage_v = msg.voltage_battery / 1000.0
                        self.telemetry.current_a = msg.current_battery / 100.0
                        self.telemetry.battery_pct = msg.battery_remaining

                    elif msg_type == "GPS_RAW_INT":
                        self.telemetry.gps_fix_type = msg.fix_type
                        self.telemetry.satellites_visible = msg.satellites_visible

                    elif msg_type == "SERVO_OUTPUT_RAW":
                        self.telemetry.motor_1_pct = _pwm_to_pct(msg.servo1_raw)
                        self.telemetry.motor_2_pct = _pwm_to_pct(msg.servo2_raw)
                        self.telemetry.motor_3_pct = _pwm_to_pct(msg.servo3_raw)
                        self.telemetry.motor_4_pct = _pwm_to_pct(msg.servo4_raw)
            else:
                time.sleep(0.005)

            # Watchdog de Heartbeat
            with self._lock:
                time_since_hb = now - self.telemetry.last_heartbeat
                if time_since_hb > self.heartbeat_timeout_s and self.telemetry.heartbeat_healthy:
                    self.telemetry.heartbeat_healthy = False
                    logger.critical(
                        f"¡HEARTBEAT MAVLINK PERDIDO ({time_since_hb:.1f}s)! El Pixhawk pasará a LOITER."
                    )
                    if self._on_heartbeat_lost:
                        try:
                            self._on_heartbeat_lost()
                        except Exception:
                            pass

    def get_telemetry(self) -> MavlinkTelemetry:
        """Retorna una copia atómica del estado actual de telemetría."""
        with self._lock:
            return MavlinkTelemetry(
                lat=self.telemetry.lat,
                lon=self.telemetry.lon,
                alt_rel_m=self.telemetry.alt_rel_m,
                alt_msl_m=self.telemetry.alt_msl_m,
                heading_deg=self.telemetry.heading_deg,
                roll_deg=self.telemetry.roll_deg,
                pitch_deg=self.telemetry.pitch_deg,
                yaw_deg=self.telemetry.yaw_deg,
                voltage_v=self.telemetry.voltage_v,
                current_a=self.telemetry.current_a,
                battery_pct=self.telemetry.battery_pct,
                mode=self.telemetry.mode,
                armed=self.telemetry.armed,
                gps_fix_type=self.telemetry.gps_fix_type,
                satellites_visible=self.telemetry.satellites_visible,
                last_heartbeat=self.telemetry.last_heartbeat,
                heartbeat_healthy=self.telemetry.heartbeat_healthy,
                motor_1_pct=self.telemetry.motor_1_pct,
                motor_2_pct=self.telemetry.motor_2_pct,
                motor_3_pct=self.telemetry.motor_3_pct,
                motor_4_pct=self.telemetry.motor_4_pct,
            )

    def set_guided_target(
        self,
        lat: float,
        lon: float,
        alt_rel_m: float,
        yaw_deg: Optional[float] = None,
    ) -> bool:
        """
        Envía objetivo de posición 3D en modo GUIDED al Pixhawk.
        Usa SET_POSITION_TARGET_GLOBAL_INT con marco relativo al despegue.
        """
        if not self._master or not self.telemetry.heartbeat_healthy:
            logger.warning("No se puede enviar target: enlace MAVLink no saludable.")
            return False

        # Máscara de bits: solo posición activa (ignora velocidades y aceleraciones)
        # Bits 0..2 = pos X,Y,Z (0 = activa)
        # Bits 3..5 = vel X,Y,Z (1 = ignorar)
        # Bits 6..8 = acc X,Y,Z (1 = ignorar)
        # Bit 9 = forzar yaw (0 = activar si se proporciona yaw)
        type_mask = 0b0000111111111000 if yaw_deg is None else 0b0000101111111000

        yaw_rad = math.radians(yaw_deg) if yaw_deg is not None else 0.0

        self._master.mav.set_position_target_global_int_send(
            0,  # time_boot_ms (no usado)
            self._master.target_system,
            self._master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            int(round(lat * 1e7)),
            int(round(lon * 1e7)),
            float(alt_rel_m),
            0, 0, 0,  # vx, vy, vz
            0, 0, 0,  # afx, afy, afz
            yaw_rad,
            0.0,      # yaw_rate
        )
        logger.info(f"Comando GUIDED enviado -> Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt_rel_m:.1f}m")
        return True

    def set_mode(self, mode_name: str) -> bool:
        """Solicita cambio de modo de vuelo a PX4 o ArduPilot (ej. OFFBOARD, GUIDED, LOITER, RTL)."""
        if not self._master:
            return False
        
        target_mode = mode_name.upper()
        mapping = self._master.mode_mapping()
        if not mapping:
            return False

        # Mapeo inteligente de compatibilidad entre PX4 y ArduPilot
        if target_mode not in mapping:
            if target_mode == "GUIDED" and "OFFBOARD" in mapping:
                target_mode = "OFFBOARD"
            elif target_mode == "OFFBOARD" and "GUIDED" in mapping:
                target_mode = "GUIDED"
            elif target_mode == "AUTO" and "MISSION" in mapping:
                target_mode = "MISSION"

        mode_val = mapping.get(target_mode)
        if mode_val is None:
            logger.error(f"Modo desconocido o no soportado por este autopiloto: {mode_name}")
            return False

        try:
            if isinstance(mode_val, (tuple, list)):
                # Firmware PX4: (base_mode, custom_mode, custom_sub_mode)
                base_mode, custom_mode, custom_sub_mode = mode_val
                self._master.mav.command_long_send(
                    self._master.target_system,
                    self._master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    float(base_mode),
                    float(custom_mode),
                    float(custom_sub_mode),
                    0.0, 0.0, 0.0, 0.0,
                )
            else:
                # Firmware ArduPilot: integer mode id
                self._master.set_mode(mode_val)

            logger.info(f"Comando de modo {target_mode} enviado exitosamente al autopiloto.")
            return True
        except Exception as e:
            logger.error(f"Error enviando cambio de modo a {target_mode}: {e}")
            return False

    def request_rtl(self) -> bool:
        """Comando directo de emergencia para forzar Return-to-Launch."""
        logger.warning("EMERGENCIA / COMANDO: Solicitando RTL al Pixhawk...")
        self.send_statustext("IgnisEdge: Solicitando RTL", severity=mavutil.mavlink.MAV_SEVERITY_WARNING)
        return self.set_mode("RTL")

    def send_statustext(self, text: str, severity: int = 6) -> bool:
        """
        Envía un mensaje MAVLink STATUSTEXT que aparece directamente en pantalla en QGroundControl.
        severity: 0=EMERGENCY, 1=ALERT, 2=CRITICAL, 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG
        """
        if not self._master:
            return False
        try:
            # statustext acepta hasta 50 caracteres
            truncated = text[:50].encode("utf-8")
            self._master.mav.statustext_send(severity, truncated)
            return True
        except Exception as e:
            logger.warning(f"Error enviando statustext MAVLink: {e}")
            return False

    def disconnect(self) -> None:
        """Cierra el enlace MAVLink y finaliza el hilo de recepción."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._master:
            try:
                self._master.close()
            except Exception:
                pass
        logger.info("Enlace MAVLink desconectado.")
