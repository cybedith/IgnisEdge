#!/usr/bin/env python3
"""
heltec_sniffer.py -- Monitor y Sniffer de Radio LoRa Heltec (915 MHz).
======================================================================

Permite verificar la comunicación de radio LoRa entre la Heltec en tierra
(con sensor de humo/gas) y la Heltec a bordo del dron:
  1. Escucha en tiempo real todo lo que entra por el puerto serial de la Heltec.
  2. Muestra paquetes binarios COBS decodificados y tramas ASCII/JSON.
  3. Opción de emitir una alerta de prueba (ALERT_NODE) para verificar recepción.

Uso:
  python3 edge_core/tools/heltec_sniffer.py
  python3 edge_core/tools/heltec_sniffer.py --port /dev/ttyACM1
  python3 edge_core/tools/heltec_sniffer.py --send --lat -36.8205 --lon -73.0445
"""

import argparse
import glob
import os
import struct
import sys
import time
from pathlib import Path

import serial

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_proto import (
    AlertNodeMessage,
    AlertV1Message,
    MsgType,
    frame_packet,
    unframe_stream,
)

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def auto_detect_port() -> str:
    """Busca puertos ttyACM o ttyUSB disponibles."""
    candidates = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    # Preferir ttyACM1 si existe (en Jetson ttyACM0 suele ser Pixhawk)
    if "/dev/ttyACM1" in candidates:
        return "/dev/ttyACM1"
    if candidates:
        return candidates[0]
    return "/dev/ttyUSB0"


def main():
    parser = argparse.ArgumentParser(description="IgnisEdge Heltec LoRa Sniffer & Tester")
    parser.add_argument("--port", default=None, help="Puerto serial (def: auto-detect)")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate (def: 115200)")
    parser.add_argument("--send", action="store_true", help="Enviar paquete ALERT_NODE de prueba")
    parser.add_argument("--lat", type=float, default=-36.8205, help="Latitud para prueba")
    parser.add_argument("--lon", type=float, default=-73.0445, help="Longitud para prueba")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    print(f"\n{BOLD}===================================================={RESET}")
    print(f"{BOLD}   IgnisEdge -- Monitor de Radio LoRa (Heltec 915)  {RESET}")
    print(f"{BOLD}===================================================={RESET}")
    print(f"Puerto: {CYAN}{port}{RESET} | Baudrate: {CYAN}{args.baud}{RESET}")

    if not os.path.exists(port):
        print(f"{RED}[ERROR] El puerto {port} no existe.{RESET}")
        print(f"Conecta el módem Heltec por USB y verifica con: ls /dev/ttyACM* /dev/ttyUSB*")
        return

    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as e:
        print(f"{RED}[ERROR] No se pudo abrir {port}: {e}{RESET}")
        return

    if args.send:
        alert = AlertNodeMessage(
            node_id=1,
            seq=1,
            lat=args.lat,
            lon=args.lon,
            gas_res_kohm=45,
            confidence=95,
        )
        payload = alert.pack()
        framed = frame_packet(payload)
        ser.write(framed)
        ser.flush()
        print(f"{GREEN}[TRANSMITIDO] Alerta de nodo enviada por {port}:{RESET}")
        print(f"  Lat: {args.lat}, Lon: {args.lon}, Confianza: 95%")
        print(f"  Trama binaria ({len(framed)} bytes): {framed.hex()}")
        ser.close()
        return

    print(f"{GREEN}[CONECTADO] Escuchando tráfico LoRa en tiempo real...{RESET}")
    print(f"{YELLOW}Presiona Ctrl+C para salir.{RESET}\n")

    rx_buf = b""

    try:
        while True:
            chunk = ser.read(64)
            if chunk:
                rx_buf += chunk

                # 1. Intentar desempacar COBS binario
                payloads, rx_buf = unframe_stream(rx_buf)
                for p in payloads:
                    msg_type = p[0]
                    if msg_type == MsgType.ALERT_NODE:
                        try:
                            msg = AlertNodeMessage.unpack(p)
                            print(f"{RED}{BOLD}🚨 [ALERT_NODE RECIBIDA]{RESET} Nodo #{msg.node_id} | "
                                  f"Lat: {msg.lat:.6f}, Lon: {msg.lon:.6f} | Gas: {msg.gas_res_kohm}kΩ | Conf: {msg.confidence}%")
                        except Exception as ex:
                            print(f"{YELLOW}[BINARIO CORRUPTO] {ex}{RESET}")
                    elif msg_type == MsgType.ALERT_V1:
                        try:
                            msg = AlertV1Message.unpack(p)
                            print(f"{GREEN}{BOLD}🔥 [ALERT_V1 DRON RECIBIDA]{RESET} Src: {msg.src_id} | "
                                  f"Lat: {msg.lat:.6f}, Lon: {msg.lon:.6f}, Alt: {msg.alt_m}m | Conf: {msg.confidence}% | FRP: {msg.frp_w}W")
                        except Exception as ex:
                            print(f"{YELLOW}[BINARIO CORRUPTO] {ex}{RESET}")
                    else:
                        print(f"{CYAN}[PAQUETE DESCONOCIDO]{RESET} Tipo: {msg_type:#04x} | Hex: {p.hex()}")

                # 2. Intentar desempacar texto ASCII
                if b"\n" in rx_buf:
                    lines = rx_buf.split(b"\n")
                    rx_buf = lines[-1]
                    for raw in lines[:-1]:
                        txt = raw.decode(errors="replace").strip()
                        if txt:
                            print(f"{CYAN}[TEXTO / SENSOR]{RESET} {txt}")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Cerrando monitor Heltec...{RESET}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
