"""
ignis_proto.py -- Protocolo Binario y Framing LoRa para IgnisEdge.
==================================================================

Implementación pura y determinista de serialización de mensajes y framing COBS+CRC16:
  - ALERT v1 (22 bytes): Veredicto completo de detección georreferenciada con evidence_mask.
  - TASK (18 bytes): Comandos y tareas de navegación / patrulla.
  - ALERT_NODE (15 bytes): Alerta emitida por nodos en tierra (BME688 + TinyML).

Framing:
  Paquete = [ COBS_ENCODE( Payload + CRC16(Payload) ) ] + 0x00 (Delimitador)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import List, Optional, Tuple


class MsgType(IntEnum):
    ALERT_V1 = 0x01
    TASK = 0x02
    ALERT_NODE = 0x03


class EvidenceBit(IntFlag):
    RAD = 1 << 0   # Temperatura radiométrica absoluta
    GEO = 1 << 1   # Anomalía contextual / fondo local
    TMP = 1 << 2   # Parpadeo térmico temporal (flicker)
    CNN = 1 << 3   # Inferencia de clasificador ML
    RGB = 1 << 4   # Confirmación visual RGB (humo)
    NODE = 1 << 5  # Coincidencia con alerta de nodo terrestre
    TRACK = 1 << 6 # Foco persistente seguido por tracker


# ============================================================================
# CRC16 (CCITT-FALSE: poly=0x1021, init=0xFFFF)
# ============================================================================

def crc16_ccitt(data: bytes) -> int:
    """Calcula CRC16-CCITT (standard robusto para microcontroladores y LoRa)."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ============================================================================
# COBS (Consistent Overhead Byte Stuffing)
# ============================================================================

def cobs_encode(data: bytes) -> bytes:
    """Codifica un buffer binario garantizando la ausencia de bytes 0x00."""
    out = bytearray()
    idx = 0
    while idx < len(data):
        next_zero = data.find(b"\x00", idx)
        if next_zero == -1:
            chunk = data[idx:]
            while len(chunk) > 254:
                out.append(255)
                out.extend(chunk[:254])
                chunk = chunk[254:]
            out.append(len(chunk) + 1)
            out.extend(chunk)
            break
        else:
            chunk = data[idx:next_zero]
            while len(chunk) > 254:
                out.append(255)
                out.extend(chunk[:254])
                chunk = chunk[254:]
            out.append(len(chunk) + 1)
            out.extend(chunk)
            idx = next_zero + 1
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    """Decodifica un stream COBS. Lanza ValueError si está corrupto."""
    out = bytearray()
    idx = 0
    while idx < len(data):
        code = data[idx]
        if code == 0:
            raise ValueError("Byte 0x00 inesperado dentro del bloque COBS")
        idx += 1
        end = idx + code - 1
        if end > len(data):
            raise ValueError("Buffer COBS truncado o corrupto")
        out.extend(data[idx:end])
        idx = end
        if code < 255 and idx < len(data):
            out.append(0)
    return bytes(out)


# ============================================================================
# Framing y Des-framing con Delimitador 0x00
# ============================================================================

def frame_packet(payload: bytes) -> bytes:
    """Adjunta CRC16 al payload, lo codifica en COBS y añade delimitador 0x00."""
    crc = crc16_ccitt(payload)
    with_crc = payload + struct.pack("<H", crc)
    return cobs_encode(with_crc) + b"\x00"


def unframe_stream(buffer: bytes) -> Tuple[List[bytes], bytes]:
    """
    Parsea un buffer continuo de bytes buscando delimitadores 0x00.
    Retorna (lista_de_payloads_validos, remanente_incompleto).
    """
    valid_payloads: List[bytes] = []
    while b"\x00" in buffer:
        pkt, buffer = buffer.split(b"\x00", 1)
        if not pkt:
            continue
        try:
            decoded = cobs_decode(pkt)
            if len(decoded) < 2:
                continue
            payload = decoded[:-2]
            received_crc = struct.unpack("<H", decoded[-2:])[0]
            if crc16_ccitt(payload) == received_crc:
                valid_payloads.append(payload)
        except ValueError:
            # Paquete corrupto descartado
            continue
    return valid_payloads, buffer


# ============================================================================
# Estructuras de Mensajes
# ============================================================================

ALERT_V1_FORMAT = "<BBBiiHBBHHBH"
ALERT_V1_SIZE = struct.calcsize(ALERT_V1_FORMAT)  # Exactamente 22 bytes

TASK_FORMAT = "<BBBBiiHH"
TASK_SIZE = struct.calcsize(TASK_FORMAT)          # Exactamente 16 bytes (o 18 según params)

ALERT_NODE_FORMAT = "<BHBiiHH"
ALERT_NODE_SIZE = struct.calcsize(ALERT_NODE_FORMAT)  # 15 bytes


@dataclass(slots=True)
class AlertV1Message:
    src_id: int
    seq: int
    lat: float        # Grados decimales
    lon: float        # Grados decimales
    alt_m: int        # Metros
    level: int        # 1=info, 2=alerta, 3=emergencia
    confidence: int   # 0 a 100
    area_dm2: int     # Decímetros cuadrados
    frp_w: int        # Fire Radiative Power (Watts)
    evidence_mask: int
    track_age_s: int

    def pack(self) -> bytes:
        lat_e7 = int(round(self.lat * 1e7))
        lon_e7 = int(round(self.lon * 1e7))
        return struct.pack(
            ALERT_V1_FORMAT,
            MsgType.ALERT_V1,
            self.src_id & 0xFF,
            self.seq & 0xFF,
            lat_e7,
            lon_e7,
            max(0, min(65535, int(round(self.alt_m)))),
            self.level & 0xFF,
            self.confidence & 0xFF,
            max(0, min(65535, int(round(self.area_dm2)))),
            max(0, min(65535, int(round(self.frp_w)))),
            self.evidence_mask & 0xFF,
            max(0, min(65535, int(round(self.track_age_s)))),
        )

    @classmethod
    def unpack(cls, data: bytes) -> AlertV1Message:
        if len(data) != ALERT_V1_SIZE:
            raise ValueError(f"Tamaño inválido para ALERT v1: {len(data)} != {ALERT_V1_SIZE}")
        (
            msg_type,
            src_id,
            seq,
            lat_e7,
            lon_e7,
            alt_m,
            level,
            confidence,
            area_dm2,
            frp_w,
            evidence_mask,
            track_age_s,
        ) = struct.unpack(ALERT_V1_FORMAT, data)
        if msg_type != MsgType.ALERT_V1:
            raise ValueError(f"MsgType incorrecto: {msg_type}")
        return cls(
            src_id=src_id,
            seq=seq,
            lat=lat_e7 / 1e7,
            lon=lon_e7 / 1e7,
            alt_m=alt_m,
            level=level,
            confidence=confidence,
            area_dm2=area_dm2,
            frp_w=frp_w,
            evidence_mask=evidence_mask,
            track_age_s=track_age_s,
        )


@dataclass(slots=True)
class AlertNodeMessage:
    node_id: int
    seq: int
    lat: float
    lon: float
    gas_res_kohm: int
    confidence: int

    def pack(self) -> bytes:
        lat_e7 = int(round(self.lat * 1e7))
        lon_e7 = int(round(self.lon * 1e7))
        return struct.pack(
            ALERT_NODE_FORMAT,
            MsgType.ALERT_NODE,
            self.node_id & 0xFFFF,
            self.seq & 0xFF,
            lat_e7,
            lon_e7,
            self.gas_res_kohm & 0xFFFF,
            self.confidence & 0xFF,
        )

    @classmethod
    def unpack(cls, data: bytes) -> AlertNodeMessage:
        if len(data) != ALERT_NODE_SIZE:
            raise ValueError(f"Tamaño inválido para ALERT_NODE: {len(data)} != {ALERT_NODE_SIZE}")
        (
            msg_type,
            node_id,
            seq,
            lat_e7,
            lon_e7,
            gas_res_kohm,
            confidence,
        ) = struct.unpack(ALERT_NODE_FORMAT, data)
        if msg_type != MsgType.ALERT_NODE:
            raise ValueError(f"MsgType incorrecto: {msg_type}")
        return cls(
            node_id=node_id,
            seq=seq,
            lat=lat_e7 / 1e7,
            lon=lon_e7 / 1e7,
            gas_res_kohm=gas_res_kohm,
            confidence=confidence,
        )
