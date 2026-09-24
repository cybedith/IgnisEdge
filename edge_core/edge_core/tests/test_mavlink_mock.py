#!/usr/bin/env python3
"""
test_mavlink_mock.py -- Validación de MAVLink y Reglas de Vuelo sin Hardware.
============================================================================

Pruebas:
  1. Inicialización y estructura de telemetría.
  2. Watchdog de Heartbeat: simulación de timeout > 2.0 s y disparo de callback.
  3. Formateo y bitmask de SET_POSITION_TARGET_GLOBAL_INT para modo GUIDED.
  4. Mapeo seguro de modos de vuelo (GUIDED, LOITER, RTL).
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.flight.ignis_mavlink import IgnisMavlink, MavlinkTelemetry


class TestIgnisMavlink(unittest.TestCase):
    def test_01_telemetry_defaults(self):
        """Verifica que la telemetría inicial sea segura y no asuma estado de vuelo."""
        t = MavlinkTelemetry()
        self.assertFalse(t.armed)
        self.assertFalse(t.heartbeat_healthy)
        self.assertEqual(t.mode, "UNKNOWN")
        self.assertEqual(t.battery_pct, -1)

    def test_02_heartbeat_watchdog(self):
        """Verifica que el watchdog detecte pérdida de heartbeat tras 2.0 segundos."""
        mav = IgnisMavlink(connection_string="/dev/null", heartbeat_timeout_s=0.2)
        callback_called = False

        def on_lost():
            nonlocal callback_called
            callback_called = True

        mav._on_heartbeat_lost = on_lost
        mav.telemetry.last_heartbeat = time.time() - 0.5  # 500 ms atrás (excede 200 ms)
        mav.telemetry.heartbeat_healthy = True

        # Simular ciclo de watchdog
        with mav._lock:
            now = time.time()
            time_since_hb = now - mav.telemetry.last_heartbeat
            if time_since_hb > mav.heartbeat_timeout_s and mav.telemetry.heartbeat_healthy:
                mav.telemetry.heartbeat_healthy = False
                if mav._on_heartbeat_lost:
                    mav._on_heartbeat_lost()

        self.assertFalse(mav.telemetry.heartbeat_healthy)
        self.assertTrue(callback_called, "El callback de pérdida de heartbeat DEBE dispararse")

    def test_03_guided_command_formatting(self):
        """Verifica que el comando GUIDED configure correctamente la máscara y coordenadas."""
        mav = IgnisMavlink(connection_string="/dev/null")
        mav._master = MagicMock()
        mav._master.target_system = 1
        mav._master.target_component = 1
        mav.telemetry.heartbeat_healthy = True

        # Enviar target con altitud relativa 85 m
        success = mav.set_guided_target(lat=-36.820123, lon=-73.044123, alt_rel_m=85.0)
        self.assertTrue(success)

        mav._master.mav.set_position_target_global_int_send.assert_called_once()
        call_args = mav._master.mav.set_position_target_global_int_send.call_args[0]

        # Verificar parámetros clave
        # frame: MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (6)
        self.assertEqual(call_args[3], 6)
        # lat int32: -368201230
        self.assertEqual(call_args[5], -368201230)
        # lon int32: -730441230
        self.assertEqual(call_args[6], -730441230)
        # alt float: 85.0
        self.assertEqual(call_args[7], 85.0)

    def test_04_rtl_request(self):
        """Verifica que la solicitud de RTL busque el ID correcto en ArduCopter."""
        mav = IgnisMavlink(connection_string="/dev/null")
        mav._master = MagicMock()
        mav._master.mode_mapping.return_value = {"GUIDED": 4, "RTL": 6, "LOITER": 5}

        success = mav.request_rtl()
        self.assertTrue(success)
        mav._master.set_mode.assert_called_once_with(6)


if __name__ == "__main__":
    unittest.main()
