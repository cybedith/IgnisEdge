import unittest
import json
from unittest.mock import MagicMock
from http.server import BaseHTTPRequestHandler
from io import BytesIO

# Importar el manejador desde el modulo
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge_core.tools import web_stream

class MockRequest:
    def makefile(self, *args, **kwargs):
        return BytesIO(b"")

class TestWebEndpoints(unittest.TestCase):
    def setUp(self):
        self.mock_fsm = MagicMock()
        self.mock_mavlink = MagicMock()
        
        # Inyectar mocks al modulo
        web_stream.fsm_instance = self.mock_fsm
        web_stream.mavlink_instance = self.mock_mavlink
        web_stream.latest_raw_frame = None
        web_stream.latest_verdict = None
        web_stream.recorder_instance = None
        
        self.handler_class = web_stream.WebMissionHandler

    def make_request(self, method, path, body_dict=None):
        """Helper para emular un Request HTTP en memoria."""
        req_body = json.dumps(body_dict).encode("utf-8") if body_dict else b""
        
        class TestHandler(self.handler_class):
            def __init__(self, *args, **kwargs):
                self.rfile = BytesIO(req_body)
                self.wfile = BytesIO()
                # Mock properties that BaseHTTPRequestHandler expects
                self.path = path
                self.headers = {"Content-Length": str(len(req_body))} if req_body else {}
                self.command = method
                
            def send_response(self, code, message=None):
                self.response_code = code
                
            def send_header(self, keyword, value):
                if not hasattr(self, 'response_headers'):
                    self.response_headers = {}
                self.response_headers[keyword] = value
                
            def end_headers(self):
                pass
                
        handler = TestHandler(MockRequest(), ("0.0.0.0", 8080), None)
        
        if method == "OPTIONS":
            handler.do_OPTIONS()
        elif method == "POST":
            handler.do_POST()
            
        return handler

    def test_01_cors_preflight(self):
        """Verifica que do_OPTIONS retorna 204 y los headers CORS correctos."""
        h = self.make_request("OPTIONS", "/api/authorize")
        self.assertEqual(h.response_code, 204)
        self.assertEqual(h.response_headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("OPTIONS", h.response_headers.get("Access-Control-Allow-Methods", ""))

    def test_02_authorize_success(self):
        """Prueba /api/authorize cuando la FSM lo permite."""
        self.mock_fsm.authorize_mission.return_value = True
        self.mock_fsm.state.name = "TRANSIT"
        
        telem_mock = MagicMock()
        telem_mock.battery_pct = 85
        self.mock_mavlink.get_telemetry.return_value = telem_mock

        h = self.make_request("POST", "/api/authorize", {"password": "ignis2026"})
        
        self.assertEqual(h.response_code, 200)
        self.assertEqual(h.response_headers.get("Access-Control-Allow-Origin"), "*")
        
        resp = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["fsm_state"], "TRANSIT")
        self.assertEqual(resp["battery_pct"], 85)
        self.mock_fsm.authorize_mission.assert_called_once()

    def test_03_authorize_rejected(self):
        """Prueba /api/authorize cuando la FSM lo rechaza (batería o mal estado)."""
        self.mock_fsm.authorize_mission.return_value = False
        self.mock_fsm.state.name = "READY"

        h = self.make_request("POST", "/api/authorize", {"password": "ignis2026"})
        
        self.assertEqual(h.response_code, 200)  # HTTP siempre 200, la logica esta en el JSON
        resp = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertEqual(resp["status"], "rejected")
        self.assertEqual(resp["fsm_state"], "READY")

    def test_04_abort_mission(self):
        """Prueba /api/abort."""
        self.mock_fsm.state.name = "RTL"
        
        h = self.make_request("POST", "/api/abort", {"reason": "Test de integracion", "password": "ignis2026"})
        
        self.assertEqual(h.response_code, 200)
        resp = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["fsm_state"], "RTL")
        self.mock_fsm.abort_mission.assert_called_once_with(reason="Test de integracion")

if __name__ == "__main__":
    unittest.main()
