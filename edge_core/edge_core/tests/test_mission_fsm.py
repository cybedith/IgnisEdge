#!/usr/bin/env python3
"""
test_mission_fsm.py -- Validación Unitaria de la Máquina de Estados de Misión.
=============================================================================

Valida el ciclo completo de principio a fin (End-to-End FSM) con compuerta humana:
  1. READY -> Alerta de nodo BME688 recibida.
  2. TRIAGE -> Filtro energético (caso factible y caso rechazado).
  3. WAITING_AUTH -> Bloqueo estricto: por ningún motivo vuela sin autorización humana.
  4. TRANSIT -> Comando GUIDED hacia coordenadas tras autorización.
  5. MANUAL_OVERRIDE -> Failsafe Nivel 1 si el piloto toma el control con RC.
  6. ON_STATION -> Llegada a estación, NUC y recepción de veredicto térmico.
  7. VERDICT -> Empaquetado de ALERT v1 (22 bytes) y transmisión LoRa.
  8. RTL -> Retorno y transición segura a READY.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_proto import AlertNodeMessage, AlertV1Message, EvidenceBit
from edge_core.flight.ignis_mavlink import MavlinkTelemetry
from edge_core.mission.ignis_mission import IgnisMissionFSM, MissionState


class TestMissionFSM(unittest.TestCase):
    def setUp(self):
        self.mock_mav = MagicMock()
        self.mock_link = MagicMock()

        # Telemetría inicial: despegado, modo LOITER, batería al 85%
        self.telem = MavlinkTelemetry(
            lat=-36.820000,
            lon=-73.044000,
            alt_rel_m=80.0,
            mode="LOITER",
            armed=True,
            battery_pct=85,
            heartbeat_healthy=True,
        )
        self.mock_mav.get_telemetry.return_value = self.telem

        self.fsm = IgnisMissionFSM(
            mavlink=self.mock_mav,
            link=self.mock_link,
            on_station_duration_s=5.0,
            nominal_patrol_alt_m=80.0,
            auth_timeout_s=60.0,
        )

    def test_01_node_alert_triggers_triage(self):
        """Verifica que recibir alerta de un nodo cambie estado de READY a TRIAGE."""
        alert = AlertNodeMessage(
            node_id=101, seq=1, lat=-36.825000, lon=-73.045000, gas_res_kohm=350, confidence=90
        )
        self.fsm.on_node_alert_received(alert)
        self.assertEqual(self.fsm.state, MissionState.TRIAGE)
        self.assertIsNotNone(self.fsm.active_target)
        self.assertEqual(self.fsm.active_target.source_node_id, 101)

    def test_02_triage_energy_rejection(self):
        """Si la batería es insuficiente, TRIAGE debe rechazar la misión y volver a READY."""
        self.telem.battery_pct = 21  # Apenas 1% sobre la reserva del 20%
        # Target lejano (~4 km)
        alert = AlertNodeMessage(
            node_id=102, seq=2, lat=-36.860000, lon=-73.045000, gas_res_kohm=300, confidence=80
        )
        self.fsm.on_node_alert_received(alert)
        self.assertEqual(self.fsm.state, MissionState.TRIAGE)

        state = self.fsm.step()
        self.assertEqual(state, MissionState.READY, "Con batería insuficiente DEBE regresar a READY")
        self.assertIsNone(self.fsm.active_target)
        self.mock_mav.set_guided_target.assert_not_called()
        self.mock_mav.set_mode.assert_not_called()

    def test_03_triage_feasible_enters_waiting_auth_without_moving(self):
        """POR SEGURIDAD: Si el triage es factible, NUNCA debe volar directo a TRANSIT. Debe entrar a WAITING_AUTH."""
        alert = AlertNodeMessage(
            node_id=103, seq=3, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.assertEqual(self.fsm.state, MissionState.TRIAGE)

        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH, "Target factible DEBE esperar autorización")
        # Verificar que NO se haya intentado cambiar de modo ni mover el dron
        self.mock_mav.set_mode.assert_not_called()
        self.mock_mav.set_guided_target.assert_not_called()

        # Si corre otro ciclo de step() sin autorización, se mantiene en WAITING_AUTH
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)
        self.mock_mav.set_mode.assert_not_called()

    def test_04_waiting_auth_timeout_returns_to_ready(self):
        """Si expira el tiempo de espera de autorización humana, cancela y vuelve a READY."""
        alert = AlertNodeMessage(
            node_id=104, seq=4, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)

        # Simular que transcurrió el timeout de 60s
        self.fsm.t_state_entered = time.time() - 65.0
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.READY, "Timeout de auth DEBE abortar a READY")
        self.assertIsNone(self.fsm.active_target)

    def test_05_authorization_unlocks_flight_to_transit(self):
        """Solo al llamar authorize_mission(), se comanda GUIDED y se transiciona a TRANSIT."""
        alert = AlertNodeMessage(
            node_id=105, seq=5, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)

        # Operador humano autoriza
        res = self.fsm.authorize_mission()
        self.assertTrue(res)
        self.assertEqual(self.fsm.state, MissionState.TRANSIT)
        self.mock_mav.set_mode.assert_called_with("GUIDED")
        self.mock_mav.set_guided_target.assert_called_once_with(-36.821000, -73.044500, 80.0)

        # Intentar re-autorizar en TRANSIT debe ser rechazado
        self.assertFalse(self.fsm.authorize_mission())

    def test_06_manual_override_safety_pilot(self):
        """Failsafe Nivel 1: Si el piloto cambia a LOITER con su control remoto, se suspende la misión autónoma."""
        # Despachar dron a TRANSIT
        alert = AlertNodeMessage(
            node_id=106, seq=6, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()  # A WAITING_AUTH
        self.fsm.authorize_mission()  # A TRANSIT
        self.assertEqual(self.fsm.state, MissionState.TRANSIT)

        # El piloto de seguridad detecta un riesgo y mueve el switch de su transmisor a LOITER
        self.telem.mode = "LOITER"
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.MANUAL_OVERRIDE)

        # Si la aeronave aterriza y se desarma, vuelve a READY
        self.telem.armed = False
        self.fsm.t_state_entered = time.time() - 4.0
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.READY)

    def test_07_emergency_abort_mission(self):
        """Operador C2 puede abortar la misión en cualquier momento forzando RTL."""
        alert = AlertNodeMessage(
            node_id=107, seq=7, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()  # WAITING_AUTH
        self.fsm.authorize_mission()  # TRANSIT

        # Ordenar aborto
        res = self.fsm.abort_mission(reason="Mal clima / Viento")
        self.assertTrue(res)
        self.assertEqual(self.fsm.state, MissionState.RTL)
        self.mock_mav.request_rtl.assert_called_once()
        self.assertIsNone(self.fsm.active_target)

    def test_08_full_mission_happy_path(self):
        """Valida el ciclo completo exitoso con compuerta humana: Alerta -> TRIAGE -> WAITING_AUTH -> AUTH -> GUIDED -> ON_STATION -> FUEGO -> ALERT v1 -> RTL."""
        # 1. Alerta de nodo a 200m
        alert = AlertNodeMessage(
            node_id=108, seq=8, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.assertEqual(self.fsm.state, MissionState.TRIAGE)

        # 2. Triage evalúa factible y pasa a WAITING_AUTH (Vuelo BLOQUEADO)
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)
        self.mock_mav.set_mode.assert_not_called()

        # 3. Operador humano autoriza formalmente
        ok = self.fsm.authorize_mission()
        self.assertTrue(ok)
        self.assertEqual(self.fsm.state, MissionState.TRANSIT)
        self.mock_mav.set_mode.assert_called_with("GUIDED")
        self.mock_mav.set_guided_target.assert_called_once()

        # 4. Simular llegada a la coordenada objetivo
        self.telem.lat = -36.821000
        self.telem.lon = -73.044500
        self.telem.mode = "GUIDED"
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.ON_STATION)

        # 5. En ON_STATION, llega veredicto térmico confirmando FUEGO
        self.fsm.on_vision_verdict_received({
            "tracks": [
                {
                    "id": 1,
                    "state": "FUEGO",
                    "confidence": 0.98,
                    "area_px": 35.0,
                    "flicker": 42.0,
                }
            ]
        })
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.VERDICT)

        # 6. En VERDICT, emite ALERT v1 de 22B y solicita RTL
        self.fsm.step()
        self.mock_link.send_alert.assert_called_once()
        sent_alert: AlertV1Message = self.mock_link.send_alert.call_args[0][0]
        self.assertEqual(len(sent_alert.pack()), 22, "El mensaje LoRa DEBE medir 22 bytes")
        self.assertEqual(sent_alert.level, 3)
        self.assertEqual(sent_alert.confidence, 98)
        self.assertTrue(bool(sent_alert.evidence_mask & EvidenceBit.TMP))
        self.assertEqual(self.fsm.state, MissionState.RTL)

        # 7. En RTL, ejecuta step() que solicita RTL al Pixhawk y finaliza al desarmar
        self.telem.armed = False
        self.fsm.step()
        self.mock_mav.request_rtl.assert_called_once()
        self.assertEqual(self.fsm.state, MissionState.READY)

    def test_09_battery_drops_during_waiting_auth_auto_cancels(self):
        """Si la batería cae mientras la FSM espera la autorización humana, cancela el despacho automáticamente."""
        alert = AlertNodeMessage(
            node_id=109, seq=9, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()  # A WAITING_AUTH
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)

        # La aeronave consume batería mientras espera (ej: en Loiter cae al 21%)
        self.telem.battery_pct = 21
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.READY, "Batería baja en espera DEBE abortar a READY")
        self.assertIsNone(self.fsm.active_target)
        self.mock_mav.set_mode.assert_not_called()

    def test_10_atomic_battery_recheck_on_authorize_mission(self):
        """Si el operador autoriza pero la batería cayó justo antes del clic, rechaza y no vuela."""
        alert = AlertNodeMessage(
            node_id=110, seq=10, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()  # A WAITING_AUTH
        self.assertEqual(self.fsm.state, MissionState.WAITING_AUTH)

        # La batería cae abruptamente a 21% justo antes de autorizar
        self.telem.battery_pct = 21
        res = self.fsm.authorize_mission()
        self.assertFalse(res, "authorize_mission() DEBE retornar False si la batería decayó")
        self.assertEqual(self.fsm.state, MissionState.READY)
        self.assertIsNone(self.fsm.active_target)
        self.mock_mav.set_guided_target.assert_not_called()

    def test_11_critical_battery_in_flight_triggers_rtl(self):
        """Watchdog en vuelo: si en TRANSIT la batería decae <= 20% de reserva, fuerza RTL inmediato."""
        alert = AlertNodeMessage(
            node_id=111, seq=11, lat=-36.821000, lon=-73.044500, gas_res_kohm=250, confidence=95
        )
        self.fsm.on_node_alert_received(alert)
        self.fsm.step()
        self.fsm.authorize_mission()
        self.assertEqual(self.fsm.state, MissionState.TRANSIT)
        self.telem.mode = "GUIDED"

        # Simular consumo en vuelo con viento en contra: batería llega al 20% de reserva
        self.telem.battery_pct = 20
        self.fsm.step()
        self.assertEqual(self.fsm.state, MissionState.RTL, "Batería <= 20% en vuelo DEBE activar RTL inmediato")
        self.mock_mav.request_rtl.assert_called()


if __name__ == "__main__":
    unittest.main()
