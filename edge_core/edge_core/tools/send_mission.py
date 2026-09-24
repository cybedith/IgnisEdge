#!/usr/bin/env python3
"""
send_mission.py -- Herramienta de Envío de Misiones Autónomas por Coordenadas GPS.
================================================================================

Permite despachar misiones al dron definiendo:
  - Punto A: Coordenada del Dron (Origen / Despegue)
  - Punto B: Coordenada Objetivo (Foco / Zona a inspeccionar)

Calcula automáticamente distancia geodésica (Haversine), rumbo en grados,
verifica límites perimétricos y comanda el vuelo vía MAVLink / API.

Uso:
  # Modo Interactivo:
  python3 edge_core/tools/send_mission.py

  # Misión directa de 2 Coordenadas:
  python3 edge_core/tools/send_mission.py --mission-ab \
      --drone-lat -36.820100 --drone-lon -73.044100 \
      --target-lat -36.820250 --target-lon -73.044150 \
      --alt 8.0

  # Comandos de emergencia:
  python3 edge_core/tools/send_mission.py --rtl
  python3 edge_core/tools/send_mission.py --loiter
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.flight.ignis_mavlink import IgnisMavlink

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def haversine_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula rumbo en grados (0..360) de (lat1, lon1) hacia (lat2, lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def try_api_mission(lat_a: float, lon_a: float, lat_b: float, lon_b: float, alt_m: float) -> bool:
    """Intenta despachar la misión vía API Web si web_stream.py está corriendo."""
    url = "http://localhost:8080/api/mission_ab"
    payload = json.dumps({
        "drone_lat": lat_a,
        "drone_lon": lon_a,
        "target_lat": lat_b,
        "target_lon": lon_b,
        "alt_m": alt_m,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "ok":
                print(f"{GREEN}[OK] Misión aceptada por el servidor de vuelo web_stream:{RESET}")
                print(f"     {data.get('message')}")
                return True
    except Exception:
        pass
    return False


def connect_pixhawk(port: str = "/dev/ttyTHS1", baud: int = 921600) -> IgnisMavlink:
    candidates = [port, "/dev/ttyACM0"]
    for p in candidates:
        if p and os.path.exists(p):
            b = 921600 if "THS" in p else 115200
            print(f"{CYAN}Conectando a Pixhawk en {p} ({b} baud)...{RESET}")
            mav = IgnisMavlink(connection_string=p, baud=b)
            if mav.connect(timeout_s=3.0):
                return mav
            mav.disconnect()
    print(f"{YELLOW}[AVISO] Pixhawk físico no detectado. Modo visualización/simulado activo.{RESET}")
    return IgnisMavlink(connection_string="/dev/null")


def interactive_menu(mav: IgnisMavlink) -> None:
    while True:
        telem = mav.get_telemetry()
        print(f"\n{BOLD}===================================================={RESET}")
        print(f"{BOLD}   IgnisEdge -- Misión por Dos Coordenadas GPS    {RESET}")
        print(f"{BOLD}===================================================={RESET}")
        status_color = GREEN if telem.heartbeat_healthy else YELLOW
        print(f"Estado Pixhawk: {status_color}{telem.mode}{RESET} | "
              f"Armed: {telem.armed} | "
              f"Bat: {telem.battery_pct}% | "
              f"GPS Fix: {telem.gps_fix_type} (Sats: {telem.satellites_visible})")
        print(f"Posición actual: Lat: {telem.lat:.6f}, Lon: {telem.lon:.6f}, Alt Rel: {telem.alt_rel_m:.1f}m")
        print(f"----------------------------------------------------")
        print("  1) 📍 Despachar Misión de 2 Coordenadas (Dron A -> Objetivo B)")
        print("  2) 🛑 Solicitar modo LOITER (Frenar en el lugar)")
        print("  3) 🏠 Solicitar modo RTL (Retorno a casa y aterrizaje)")
        print("  4) Salir")
        print(f"----------------------------------------------------")

        try:
            choice = input(f"{BOLD}Selecciona una opción (1-4): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            try:
                # Punto A (Dron)
                def_lat_a = telem.lat if telem.lat != 0.0 else -36.820100
                def_lon_a = telem.lon if telem.lon != 0.0 else -73.044100
                print(f"\n{BOLD}[1/2] PUNTO A — Posición del Dron (Origen / Base):{RESET}")
                lat_a_in = input(f"  Latitud A [{def_lat_a:.6f}]: ").strip() or str(def_lat_a)
                lon_a_in = input(f"  Longitud A [{def_lon_a:.6f}]: ").strip() or str(def_lon_a)
                lat_a = float(lat_a_in)
                lon_a = float(lon_a_in)

                # Punto B (Objetivo)
                def_lat_b = lat_a + (12.0 / 111320.0)
                def_lon_b = lon_a
                print(f"\n{BOLD}[2/2] PUNTO B — Objetivo a Inspeccionar (Foco):{RESET}")
                lat_b_in = input(f"  Latitud B [{def_lat_b:.6f}]: ").strip() or str(def_lat_b)
                lon_b_in = input(f"  Longitud B [{def_lon_b:.6f}]: ").strip() or str(def_lon_b)
                alt_in = input(f"  Altitud relativa en metros [8.0]: ").strip() or "8.0"
                lat_b = float(lat_b_in)
                lon_b = float(lon_b_in)
                alt = float(alt_in)

                # Cálculo de Distancia y Rumbo
                dist = haversine_dist_m(lat_a, lon_a, lat_b, lon_b)
                bearing = calculate_bearing_deg(lat_a, lon_a, lat_b, lon_b)

                print(f"\n{CYAN}--- Resumen Geodésico ---{RESET}")
                print(f"  * Origen A: ({lat_a:.6f}, {lon_a:.6f})")
                print(f"  * Destino B: ({lat_b:.6f}, {lon_b:.6f})")
                print(f"  * Distancia calculada: {BOLD}{dist:.1f} metros{RESET}")
                print(f"  * Rumbo / Azimut: {BOLD}{bearing:.0f}°{RESET}")
                print(f"  * Altitud de patrulla: {alt} m")

                if dist > 35.0:
                    print(f"\n{YELLOW}[ALERTA PERÍMETRO] La distancia ({dist:.1f}m) supera los 35m seguros para la cancha.{RESET}")
                    confirm = input("¿Confirmas despachar el dron hacia allá? (s/N): ").strip().lower()
                    if confirm != "s":
                        print("Misión cancelada.")
                        continue

                # Intentar enviar vía API Web si está activa
                if not try_api_mission(lat_a, lon_a, lat_b, lon_b, alt):
                    # Envío MAVLink directo
                    mav.set_mode("GUIDED")
                    mav.set_guided_target(lat_b, lon_b, alt)
                    mav.send_statustext(f"IGNIS: Mision A->B d={dist:.0f}m GUIDED", severity=6)
                    print(f"{GREEN}[OK] Comando GUIDED enviado directamente por MAVLink.{RESET}")

            except ValueError:
                print(f"{RED}Entrada inválida. Debe ser número decimal.{RESET}")

        elif choice == "2":
            mav.set_mode("LOITER")
            mav.send_statustext("IGNIS: Solicitado modo LOITER", severity=4)
            print(f"{GREEN}[OK] Modo LOITER comandado.{RESET}")

        elif choice == "3":
            mav.request_rtl()
            mav.send_statustext("IGNIS: Solicitado modo RTL", severity=4)
            print(f"{GREEN}[OK] Modo RTL comandado.{RESET}")

        elif choice == "4":
            break

    mav.disconnect()
    print("Desconectado.")


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisEdge - Envío de Misiones por Dos Coordenadas")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="Puerto UART Pixhawk (def: /dev/ttyTHS1)")
    parser.add_argument("--baud", type=int, default=921600, help="Baudrate (def: 921600)")
    parser.add_argument("--mission-ab", action="store_true", help="Despachar misión de 2 coordenadas")
    parser.add_argument("--drone-lat", type=float, default=None, help="Latitud Punto A (Dron)")
    parser.add_argument("--drone-lon", type=float, default=None, help="Longitud Punto A (Dron)")
    parser.add_argument("--target-lat", type=float, default=None, help="Latitud Punto B (Objetivo)")
    parser.add_argument("--target-lon", type=float, default=None, help="Longitud Punto B (Objetivo)")
    parser.add_argument("--alt", type=float, default=8.0, help="Altitud relativa en metros (def: 8.0)")
    parser.add_argument("--rtl", action="store_true", help="Solicitar modo RTL inmediato")
    parser.add_argument("--loiter", action="store_true", help="Solicitar modo LOITER inmediato")
    args = parser.parse_args()

    if args.mission_ab and args.target_lat is not None and args.target_lon is not None:
        lat_a = args.drone_lat if args.drone_lat is not None else -36.820100
        lon_a = args.drone_lon if args.drone_lon is not None else -73.044100
        dist = haversine_dist_m(lat_a, lon_a, args.target_lat, args.target_lon)
        bearing = calculate_bearing_deg(lat_a, lon_a, args.target_lat, args.target_lon)
        print(f"{CYAN}Misión A ➔ B:{RESET} Distancia={dist:.1f}m, Rumbo={bearing:.0f}°, Alt={args.alt}m")

        if try_api_mission(lat_a, lon_a, args.target_lat, args.target_lon, args.alt):
            return

        mav = connect_pixhawk(port=args.port, baud=args.baud)
        mav.set_mode("GUIDED")
        mav.set_guided_target(args.target_lat, args.target_lon, args.alt)
        mav.send_statustext(f"IGNIS: Mision A->B d={dist:.0f}m GUIDED", severity=6)
        mav.disconnect()
        return

    mav = connect_pixhawk(port=args.port, baud=args.baud)

    if args.rtl:
        mav.request_rtl()
        mav.disconnect()
        return

    if args.loiter:
        mav.set_mode("LOITER")
        mav.disconnect()
        return

    interactive_menu(mav)


if __name__ == "__main__":
    main()
