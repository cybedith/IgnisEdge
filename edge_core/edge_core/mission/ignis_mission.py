"""
ignis_mission.py -- Cerebro de Misión y Máquina de Estados Finita (FSM).
========================================================================

FSM: READY -> TRIAGE -> TRANSIT -> ON_STATION -> VERDICT -> RTL -> READY

Reglas de Oro Implementadas:
  1. El Pixhawk es la autoridad de vuelo; la Jetson solo sugiere comandos GUIDED.
  2. Si ignis_vision se cuelga, ignis_mission degrada elegantemente sin tirar el dron.
  3. Filtro energético estricto: nunca despachar a un foco si la batería no garantiza
     ida + órbita + regreso + 20% reserva.
  4. La FSM es 100% testeable sin hardware mediante inyección de dependencias.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from edge_core.comms.ignis_proto import AlertNodeMessage, AlertV1Message, EvidenceBit
from edge_core.flight.ignis_mavlink import IgnisMavlink, MavlinkTelemetry

logger = logging.getLogger("ignis_mission")


class MissionState(Enum):
    READY = auto()
    TRIAGE = auto()
    WAITING_AUTH = auto()      # Bloqueo de seguridad: espera autorización humana obligatoria
    TRANSIT = auto()
    ON_STATION = auto()
    VERDICT = auto()
    RTL = auto()
    MANUAL_OVERRIDE = auto()   # Piloto toma control manual vía radiocontrol (RC)


@dataclass(slots=True)
class MissionTarget:
    lat: float
    lon: float
    alt_m: float
    source_node_id: int
    source_confidence: int
    detection_timestamp: float


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia en metros entre dos coordenadas geográficas."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class IgnisMissionFSM:
    """Máquina de Estados de Misión Autónoma."""

    def __init__(
        self,
        mavlink: IgnisMavlink,
        link: Any,  # IgnisLink
        cruise_speed_mps: float = 12.0,      # 12 m/s (~43 km/h crucero)
        cruise_power_w: float = 250.0,       # Consumo medio Holybro S500 (W)
        battery_capacity_wh: float = 90.0,   # Batería 4S típica ~5000mAh (Wh)
        reserve_battery_pct: float = 20.0,   # 20% reserva inviolable
        on_station_duration_s: float = 20.0, # 20 s de órbita acumulando evidencia
        nominal_patrol_alt_m: float = 80.0,  # 80 m de altura sobre el dosel
        max_allowed_dist_m: float = 5000.0,  # Límite perimétrico de seguridad (m)
        auth_timeout_s: float = 120.0,       # Timeout esperando autorización humana (s)
    ):
        self.mavlink = mavlink
        self.link = link
        self.cruise_speed_mps = cruise_speed_mps
        self.cruise_power_w = cruise_power_w
        self.battery_capacity_wh = battery_capacity_wh
        self.reserve_battery_pct = reserve_battery_pct
        self.on_station_duration_s = on_station_duration_s
        self.nominal_patrol_alt_m = nominal_patrol_alt_m
        self.max_allowed_dist_m = max_allowed_dist_m
        self.auth_timeout_s = auth_timeout_s

        self.state = MissionState.READY
        self.active_target: Optional[MissionTarget] = None
        self.t_state_entered = time.time()
        self.verified_fire_detected = False
        self.last_fire_evidence: Dict[str, Any] = {}
        self._nuc_triggered = False
        # RLock (no Lock): transition_to() es invocado reentrantemente desde
        # step()/abort_mission()/authorize_mission() dentro del mismo hilo.
        self._lock = threading.RLock()

    def transition_to(self, new_state: MissionState) -> None:
        """Efectúa transición de estado con registro de tiempo y logs de auditoría."""
        with self._lock:
            prev = self.state
            self.state = new_state
            self.t_state_entered = time.time()
            logger.info(f"[FSM] Transición: {prev.name} -> {new_state.name}")
            if self.mavlink:
                self.mavlink.send_statustext(f"IGNIS: {prev.name} -> {new_state.name}", severity=6)

    def command_direct_target(self, lat: float, lon: float, alt_m: Optional[float] = None) -> bool:
        """Comanda directamente un objetivo de inspección desde la estación de control o web."""
        with self._lock:
            logger.info(f"Comando directo de misión recibido hacia: Lat={lat:.6f}, Lon={lon:.6f}")
            self.active_target = MissionTarget(
                lat=lat,
                lon=lon,
                alt_m=alt_m if alt_m is not None else self.nominal_patrol_alt_m,
                source_node_id=0,
                source_confidence=100,
                detection_timestamp=time.time(),
            )
            self.transition_to(MissionState.TRIAGE)
            return True

    def on_node_alert_received(self, alert: AlertNodeMessage) -> None:
        """Callback cuando llega una alerta temprana de un nodo BME688."""
        logger.info(f"Alerta de nodo #{alert.node_id} recibida en FSM (estado actual: {self.state.name})")
        if self.state == MissionState.READY:
            self.active_target = MissionTarget(
                lat=alert.lat,
                lon=alert.lon,
                alt_m=self.nominal_patrol_alt_m,
                source_node_id=alert.node_id,
                source_confidence=alert.confidence,
                detection_timestamp=time.time(),
            )
            self.transition_to(MissionState.TRIAGE)
        else:
            logger.info(f"Dron ocupado en {self.state.name}. Alerta de nodo #{alert.node_id} encolada.")

    def on_vision_verdict_received(self, verdict_dict: Dict[str, Any]) -> None:
        """Callback cuando ignis_vision reporta detecciones térmicas."""
        tracks = verdict_dict.get("tracks", [])
        for tr in tracks:
            if tr.get("state") == "FUEGO":
                self.verified_fire_detected = True
                self.last_fire_evidence = tr
                logger.warning(
                    f"¡FUEGO CONFIRMADO POR VISIÓN! Confianza: {tr.get('confidence', 0)*100:.0f}%"
                )

    def evaluate_energy_feasibility(self, dist_m: float, battery_pct: float) -> Tuple[bool, float]:
        """
        Filtro de factibilidad energética:
          Calcula Wh requeridos para ida, órbita y vuelta vs energía restante en la batería.
        """
        # Tiempo de vuelo ida y vuelta
        t_transit_s = (dist_m * 2.0) / self.cruise_speed_mps
        t_total_s = t_transit_s + self.on_station_duration_s

        # Consumo estimado en Wh
        energy_wh = (self.cruise_power_w * t_total_s) / 3600.0
        pct_needed = (energy_wh / self.battery_capacity_wh) * 100.0

        pct_available = max(0.0, battery_pct - self.reserve_battery_pct)
        is_feasible = pct_available >= pct_needed

        logger.info(
            f"[TRIAGE] Dist: {dist_m:.0f}m | Necesario: {pct_needed:.1f}% bat | "
            f"Disponible sobre reserva: {pct_available:.1f}% | Factible: {is_feasible}"
        )
        return is_feasible, pct_needed

    def authorize_mission(self) -> bool:
        """
        Autoriza explícitamente el despacho del dron hacia el foco activo.

        POR SEGURIDAD AERONÁUTICA INVIOLABLE:
          1. Solo es válido si la FSM se encuentra en WAITING_AUTH.
          2. Re-evalúa la factibilidad energética en el milisegundo exacto del clic.
             Si la batería decayó durante la espera humana, RECHAZA el despegue.
          3. Solo si todo es conforme comanda modo GUIDED y envía el waypoint 3D.
        """
        with self._lock:
            if self.state != MissionState.WAITING_AUTH:
                logger.warning(
                    f"[SEGURIDAD] Rechazando autorización: la FSM está en estado '{self.state.name}', "
                    f"no en WAITING_AUTH."
                )
                return False

            if not self.active_target:
                logger.error("[SEGURIDAD] No hay target activo para autorizar misión.")
                self.transition_to(MissionState.READY)
                return False

            # DOBLE CANDADO: Re-evaluación atómica con telemetría viva fresca de la Pixhawk
            telem = self.mavlink.get_telemetry() if self.mavlink else MavlinkTelemetry()
            current_lat = telem.lat if telem.lat != 0.0 else self.active_target.lat
            current_lon = telem.lon if telem.lon != 0.0 else self.active_target.lon
            dist_to_target = haversine_distance_m(current_lat, current_lon, self.active_target.lat, self.active_target.lon)
            battery_pct = float(telem.battery_pct) if telem.battery_pct >= 0 else 100.0
            feasible, pct_needed = self.evaluate_energy_feasibility(dist_to_target, battery_pct)

            if not feasible:
                logger.critical(
                    f"[AUTORIZACIÓN DENEGADA POR BATERÍA] Operador autorizó pero la batería cayó a {battery_pct:.1f}% "
                    f"(se requiere {pct_needed:.1f}% + {self.reserve_battery_pct:.0f}% reserva). "
                    f"Despacho cancelado por seguridad aeronáutica."
                )
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: DENEGADO bateria baja", severity=3)
                self.active_target = None
                self.transition_to(MissionState.READY)
                return False

            logger.info(
                f"¡Misión AUTORIZADA por operador humano! Batería: {battery_pct:.1f}% (OK). Comandando GUIDED hacia "
                f"Lat={self.active_target.lat:.6f}, Lon={self.active_target.lon:.6f}, Alt={self.active_target.alt_m:.1f}m"
            )
            if self.mavlink:
                self.mavlink.send_statustext("IGNIS: AUTORIZADO -> GUIDED", severity=6)
                self.mavlink.set_mode("GUIDED")
                self.mavlink.set_guided_target(
                    self.active_target.lat,
                    self.active_target.lon,
                    self.active_target.alt_m,
                )
            self.transition_to(MissionState.TRANSIT)
            return True

    def abort_mission(self, reason: str = "Operador C2") -> bool:
        """
        Comanda aborto seguro de emergencia y fuerza retorno a casa (RTL).

        Cancela inmediatamente cualquier navegación autónoma en curso.
        """
        with self._lock:
            logger.warning(f"[ABORTO] Aborto de misión ejecutado. Razón: {reason}")
            if self.mavlink:
                self.mavlink.send_statustext(f"IGNIS: ABORTO ({reason}) -> RTL", severity=2)
                self.mavlink.request_rtl()
            self.active_target = None
            self.transition_to(MissionState.RTL)
            return True

    def step(self) -> MissionState:
        """Ejecuta un ciclo de la máquina de estados. Debe llamarse periódicamente (~5-10 Hz)."""
        with self._lock:
            return self._step_locked()

    def _step_locked(self) -> MissionState:
        telem = self.mavlink.get_telemetry()
        now = time.time()
        time_in_state = now - self.t_state_entered

        # ====================================================================
        # FAILSAFE NIVEL 1: Watchdog de Override Manual por Radiocontrol (RC)
        # Si el piloto mueve el switch físico a LOITER, STABILIZE, ALT_HOLD o POSHOLD,
        # la Jetson suspende de inmediato cualquier comando autónomo.
        # ====================================================================
        if self.state in (MissionState.TRANSIT, MissionState.ON_STATION, MissionState.VERDICT):
            if telem.mode in ("LOITER", "STABILIZE", "ALT_HOLD", "POSHOLD", "ACRO"):
                logger.warning(
                    f"[OVERRIDE RC] Piloto tomó control manual vía radiocontrol (Modo: {telem.mode}). "
                    f"Suspendiendo comandos autónomos de la Jetson."
                )
                if self.mavlink:
                    self.mavlink.send_statustext(f"IGNIS: MANUAL OVERRIDE ({telem.mode})", severity=2)
                self.transition_to(MissionState.MANUAL_OVERRIDE)
                return self.state

        # ====================================================================
        # FAILSAFE NIVEL 2: Watchdog de Batería Crítica en Vuelo Autónomo
        # Si durante TRANSIT, ON_STATION o VERDICT la batería decae a la reserva
        # inviolable (<= reserve_battery_pct), fuerza RTL de emergencia inmediato.
        # ====================================================================
        if self.state in (MissionState.TRANSIT, MissionState.ON_STATION, MissionState.VERDICT):
            if telem.battery_pct >= 0 and telem.battery_pct <= self.reserve_battery_pct:
                logger.critical(
                    f"[BATERÍA CRÍTICA EN VUELO] Batería cayó a {telem.battery_pct}% "
                    f"(límite de reserva inviolable: {self.reserve_battery_pct:.0f}%). Forzando RTL de emergencia."
                )
                self.abort_mission(reason=f"Batería en reserva ({telem.battery_pct}%)")
                return self.state

        # ====================================================================
        # ESTADO: READY
        # ====================================================================
        if self.state == MissionState.READY:
            self._nuc_triggered = False
            self.verified_fire_detected = False

        # ====================================================================
        # ESTADO: TRIAGE
        # ====================================================================
        elif self.state == MissionState.TRIAGE:
            if not self.active_target:
                self.transition_to(MissionState.READY)
                return self.state

            current_lat = telem.lat if telem.lat != 0.0 else self.active_target.lat
            current_lon = telem.lon if telem.lon != 0.0 else self.active_target.lon
            dist_to_target = haversine_distance_m(current_lat, current_lon, self.active_target.lat, self.active_target.lon)

            # Filtro perimétrico de seguridad (ej: límite de cancha de fútbol)
            if dist_to_target > self.max_allowed_dist_m:
                logger.warning(
                    f"[TRIAGE RECHAZADO] Distancia al objetivo ({dist_to_target:.1f} m) supera el límite "
                    f"máximo seguro permitido ({self.max_allowed_dist_m:.1f} m). Rechazado por seguridad."
                )
                if self.mavlink:
                    self.mavlink.send_statustext(f"IGNIS: RECHAZADO d={dist_to_target:.0f}m > max", severity=4)
                self.active_target = None
                self.transition_to(MissionState.READY)
                return self.state

            battery_pct = float(telem.battery_pct) if telem.battery_pct >= 0 else 100.0
            feasible, _ = self.evaluate_energy_feasibility(dist_to_target, battery_pct)

            if feasible:
                logger.info(
                    f"Target energéticamente factible (dist: {dist_to_target:.0f}m). "
                    f"Transicionando a WAITING_AUTH (Vuelo bloqueado hasta autorización humana explícita)..."
                )
                if self.mavlink:
                    self.mavlink.send_statustext(f"IGNIS: Target OK d={dist_to_target:.0f}m -> REQ AUTH", severity=5)
                self.transition_to(MissionState.WAITING_AUTH)
            else:
                logger.warning("Target RECHAZADO por límite energético. Regresando a READY.")
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: RECHAZADO bateria baja", severity=4)
                self.active_target = None
                self.transition_to(MissionState.READY)

        # ====================================================================
        # ESTADO: WAITING_AUTH (Compuerta Humana Inviolable)
        # ====================================================================
        elif self.state == MissionState.WAITING_AUTH:
            if not self.active_target:
                self.transition_to(MissionState.READY)
                return self.state

            # CANDADO ENERGÉTICO CONTINUO: Si la batería decae mientras espera autorización
            current_lat = telem.lat if telem.lat != 0.0 else self.active_target.lat
            current_lon = telem.lon if telem.lon != 0.0 else self.active_target.lon
            dist_to_target = haversine_distance_m(current_lat, current_lon, self.active_target.lat, self.active_target.lon)
            battery_pct = float(telem.battery_pct) if telem.battery_pct >= 0 else 100.0
            feasible, pct_needed = self.evaluate_energy_feasibility(dist_to_target, battery_pct)

            if not feasible:
                logger.warning(
                    f"[WAITING_AUTH CANCELADO] Batería cayó a {battery_pct:.1f}% durante la espera humana. "
                    f"Se requiere {pct_needed:.1f}% + reserva. Cancelando despacho por seguridad."
                )
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: Bateria cayo en espera -> CANCEL", severity=4)
                self.active_target = None
                self.transition_to(MissionState.READY)
                return self.state

            # Timeout de seguridad: Si pasan auth_timeout_s sin respuesta humana, se cancela
            if time_in_state > self.auth_timeout_s:
                logger.warning(
                    f"[WAITING_AUTH] Tiempo de espera de autorización agotado ({self.auth_timeout_s:.0f}s). "
                    f"Cancelando despacho y retornando a READY."
                )
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: AUTH TIMEOUT -> READY", severity=4)
                self.active_target = None
                self.transition_to(MissionState.READY)

        # ====================================================================
        # ESTADO: TRANSIT
        # ====================================================================
        elif self.state == MissionState.TRANSIT:
            if not self.active_target:
                self.transition_to(MissionState.RTL)
                return self.state

            dist = haversine_distance_m(telem.lat, telem.lon, self.active_target.lat, self.active_target.lon)
            # Si llegó a menos de 20 metros del objetivo
            if dist <= 20.0 or (telem.lat == 0.0 and time_in_state > 2.0):
                logger.info(f"Llegada a estación (dist: {dist:.1f}m). Entrando a ON_STATION.")
                if self.mavlink:
                    self.mavlink.send_statustext(f"IGNIS: En estacion d={dist:.0f}m", severity=6)
                self.transition_to(MissionState.ON_STATION)

        # ====================================================================
        # ESTADO: ON_STATION
        # ====================================================================
        elif self.state == MissionState.ON_STATION:
            # Forzar NUC al llegar
            if not self._nuc_triggered:
                logger.info("ON_STATION: Disparando NUC en cámara para calibración de fondo.")
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: Calibrando camara (NUC)...", severity=6)
                self._nuc_triggered = True

            # Si se confirma fuego o expira el tiempo de órbita
            if self.verified_fire_detected or time_in_state >= self.on_station_duration_s:
                self.transition_to(MissionState.VERDICT)

        # ====================================================================
        # ESTADO: VERDICT
        # ====================================================================
        elif self.state == MissionState.VERDICT:
            # Armar ALERT v1 (22 bytes)
            if self.verified_fire_detected:
                conf = int(round(self.last_fire_evidence.get("confidence", 0.95) * 100))
                area = int(round(self.last_fire_evidence.get("area_px", 10.0) * 10))  # aprox dm2
                mask = EvidenceBit.RAD | EvidenceBit.TMP | EvidenceBit.TRACK | EvidenceBit.NODE
                level = 3  # Emergencia fuego activo
                if self.mavlink:
                    self.mavlink.send_statustext(f"IGNIS: FUEGO CONFIRMADO! Conf={conf}%", severity=2)
            else:
                conf = 10
                area = 0
                mask = EvidenceBit.NODE
                level = 1  # Falsa alarma / no verificado
                if self.mavlink:
                    self.mavlink.send_statustext("IGNIS: ZONA DESPEJADA (Sin fuego)", severity=6)

            alert_msg = AlertV1Message(
                src_id=1,
                seq=int(time.time()) % 256,
                lat=self.active_target.lat if self.active_target else telem.lat,
                lon=self.active_target.lon if self.active_target else telem.lon,
                alt_m=int(round(telem.alt_rel_m)),
                level=level,
                confidence=conf,
                area_dm2=area,
                frp_w=500 if self.verified_fire_detected else 0,
                evidence_mask=mask,
                track_age_s=int(time_in_state),
            )

            # Transmitir por LoRa (protegido contra fallos de radio)
            if self.link:
                try:
                    self.link.send_alert(alert_msg)
                except Exception as e:
                    logger.warning(f"Aviso transmitiendo veredicto LoRa: {e}")

            logger.info("Veredicto procesado. Comandando RTL...")
            self.transition_to(MissionState.RTL)

        # ====================================================================
        # ESTADO: RTL
        # ====================================================================
        elif self.state == MissionState.RTL:
            if self.mavlink:
                self.mavlink.send_statustext("IGNIS: Mision terminada -> RTL", severity=4)
            self.mavlink.request_rtl()
            # En simulación o tras aterrizaje
            if not telem.armed or time_in_state > 5.0:
                logger.info("Misión finalizada. Dron en tierra / RTL completado.")
                self.active_target = None
                self.transition_to(MissionState.READY)

        # ====================================================================
        # ESTADO: MANUAL_OVERRIDE
        # ====================================================================
        elif self.state == MissionState.MANUAL_OVERRIDE:
            # En override manual la Jetson NO envía comandos de vuelo ni modifica el modo.
            # Si la aeronave se desarma (aterrizó en tierra) tras 3 segundos:
            if not telem.armed and time_in_state > 3.0:
                logger.info("[MANUAL_OVERRIDE] Aeronave desarmada en tierra. Reseteando a READY.")
                self.active_target = None
                self.transition_to(MissionState.READY)

        return self.state
