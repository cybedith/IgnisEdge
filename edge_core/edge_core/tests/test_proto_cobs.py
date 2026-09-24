#!/usr/bin/env python3
"""
test_proto_cobs.py -- Validación Unitario del Protocolo Binario y Framing LoRa.
=============================================================================

Pruebas sin hardware:
  1. Serialización y deserialización idéntica de ALERT v1 (exactamente 22 bytes).
  2. Serialización y deserialización de ALERT_NODE.
  3. COBS encode/decode con payloads que contienen 0x00 embebidos.
  4. Integridad de CRC16 y descarte de paquetes con 1 bit corrupto.
  5. Des-framing continuo desde stream UART ruidoso o fragmentado.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_proto import (
    ALERT_NODE_SIZE,
    ALERT_V1_SIZE,
    AlertNodeMessage,
    AlertV1Message,
    EvidenceBit,
    cobs_decode,
    cobs_encode,
    crc16_ccitt,
    frame_packet,
    unframe_stream,
)


class TestIgnisProto(unittest.TestCase):
    def test_01_alert_v1_size_and_roundtrip(self):
        """Verifica que ALERT v1 mida exactamente 22 bytes y se serialice sin pérdidas."""
        msg = AlertV1Message(
            src_id=1,
            seq=42,
            lat=-36.8201345,
            lon=-73.0443912,
            alt_m=95,
            level=3,
            confidence=98,
            area_dm2=450,
            frp_w=1250,
            evidence_mask=EvidenceBit.RAD | EvidenceBit.TMP | EvidenceBit.TRACK,
            track_age_s=14,
        )
        packed = msg.pack()
        self.assertEqual(len(packed), 22, f"El mensaje ALERT v1 DEBE medir 22 bytes, pero mide {len(packed)}")
        self.assertEqual(len(packed), ALERT_V1_SIZE)

        unpacked = AlertV1Message.unpack(packed)
        self.assertEqual(unpacked.src_id, 1)
        self.assertEqual(unpacked.seq, 42)
        self.assertAlmostEqual(unpacked.lat, -36.8201345, places=6)
        self.assertAlmostEqual(unpacked.lon, -73.0443912, places=6)
        self.assertEqual(unpacked.alt_m, 95)
        self.assertEqual(unpacked.level, 3)
        self.assertEqual(unpacked.confidence, 98)
        self.assertEqual(unpacked.area_dm2, 450)
        self.assertEqual(unpacked.frp_w, 1250)
        self.assertEqual(unpacked.evidence_mask, EvidenceBit.RAD | EvidenceBit.TMP | EvidenceBit.TRACK)
        self.assertEqual(unpacked.track_age_s, 14)

    def test_02_alert_node_roundtrip(self):
        """Verifica la serialización de alertas de nodos BME688."""
        node_msg = AlertNodeMessage(
            node_id=105,
            seq=12,
            lat=-36.8190000,
            lon=-73.0450000,
            gas_res_kohm=480,
            confidence=85,
        )
        packed = node_msg.pack()
        self.assertEqual(len(packed), ALERT_NODE_SIZE)

        unpacked = AlertNodeMessage.unpack(packed)
        self.assertEqual(unpacked.node_id, 105)
        self.assertAlmostEqual(unpacked.lat, -36.8190000, places=6)
        self.assertEqual(unpacked.gas_res_kohm, 480)
        self.assertEqual(unpacked.confidence, 85)

    def test_03_cobs_with_zeros(self):
        """Verifica que COBS elimine completamente todos los bytes 0x00."""
        payload_with_zeros = bytes([0x00, 0x12, 0x00, 0x00, 0xFF, 0x00, 0x42])
        encoded = cobs_encode(payload_with_zeros)
        self.assertNotIn(0x00, encoded, "COBS encode NUNCA debe contener bytes 0x00 en el cuerpo")

        decoded = cobs_decode(encoded)
        self.assertEqual(decoded, payload_with_zeros)

    def test_04_crc16_corruption_detection(self):
        """Verifica que un paquete con un solo bit corrupto sea rechazado."""
        msg = AlertV1Message(
            src_id=1, seq=1, lat=-36.8, lon=-73.0, alt_m=100,
            level=1, confidence=50, area_dm2=10, frp_w=50,
            evidence_mask=0, track_age_s=1
        )
        framed = frame_packet(msg.pack())
        self.assertTrue(framed.endswith(b"\x00"), "Todo paquete enmarcado debe terminar en 0x00")

        # Corromper 1 byte en el payload codificado
        corrupt_list = bytearray(framed)
        corrupt_list[4] ^= 0x01
        corrupt_framed = bytes(corrupt_list)

        # Unframe no debe retornar paquetes válidos
        valid_pkts, remainder = unframe_stream(corrupt_framed)
        self.assertEqual(len(valid_pkts), 0, "Paquete corrupto DEBE ser descartado por CRC16")

    def test_05_unframe_stream_fragmentation(self):
        """Simula stream UART llegando en fragmentos arbitrarios con múltiples paquetes."""
        m1 = AlertV1Message(src_id=1, seq=10, lat=-36.8, lon=-73.0, alt_m=80, level=2, confidence=90, area_dm2=100, frp_w=300, evidence_mask=1, track_age_s=5)
        m2 = AlertV1Message(src_id=1, seq=11, lat=-36.81, lon=-73.01, alt_m=82, level=3, confidence=99, area_dm2=200, frp_w=500, evidence_mask=7, track_age_s=6)

        stream = frame_packet(m1.pack()) + frame_packet(m2.pack())

        # Simular llegada en fragmentos de 7 bytes (típico buffer UART en microcontroladores)
        buffer = b""
        collected = []
        for i in range(0, len(stream), 7):
            chunk = stream[i : i + 7]
            buffer += chunk
            pkts, buffer = unframe_stream(buffer)
            collected.extend(pkts)

        self.assertEqual(len(collected), 2, "Deben haberse recibido exactamente 2 paquetes completos")
        unp1 = AlertV1Message.unpack(collected[0])
        unp2 = AlertV1Message.unpack(collected[1])
        self.assertEqual(unp1.seq, 10)
        self.assertEqual(unp2.seq, 11)


if __name__ == "__main__":
    unittest.main()
