#!/usr/bin/env python3
"""
check_hardware.py -- Diagnóstico de Placas y Puertos de IgnisEdge.
=================================================================

Escanea y valida el estado físico de conexión de los 3 subsistemas:
  1. Cámara Térmica P3 (USB VID=0x3474, PID=0x45a2).
  2. Autopiloto Pixhawk 6C (UART /dev/ttyTHS1 o USB /dev/ttyACM0).
  3. Módem LoRa Heltec (UART /dev/ttyUSB0).

Uso:
  python3 edge_core/tools/check_hardware.py
"""

import glob
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_status(name: str, ok: bool, details: str) -> None:
    status = f"{GREEN}[OK]{RESET}" if ok else f"{RED}[FALLO]{RESET}"
    print(f"  {status} {BOLD}{name:<25}{RESET} : {details}")


def check_p3_camera() -> bool:
    print(f"\n{CYAN}--- Verificando Cámara Térmica P3 (InfiRay) ---{RESET}")
    # 1. Chequeo por libusb / pyusb
    try:
        import usb.core
        dev = usb.core.find(idVendor=0x3474, idProduct=0x45A2)
        if dev is None:
            print_status("Detección USB", False, "No detectada en bus USB (VID: 0x3474, PID: 0x45a2)")
            print(f"    {YELLOW}Solución: Reconecta el cable USB-C de la cámara.{RESET}")
            return False
        print_status("Detección USB", True, f"Dispositivo P3 encontrado en Bus {dev.bus:03d} Dispositivo {dev.address:03d}")
    except ImportError:
        print_status("Detección USB", False, "Falta pyusb en el venv (pip install pyusb)")
        return False
    except Exception as e:
        print_status("Detección USB", False, f"Error accediendo a USB: {e}")
        return False

    # 2. Permisos USB y driver P3
    cam = None
    try:
        import usb.util
        from p3_camera import P3Camera, GainMode

        # Liberar posibles recursos huérfanos
        try:
            usb.util.dispose_resources(dev)
        except Exception:
            pass

        cam = P3Camera()
        cam.connect()
        print_status("Permisos udev", True, "Lectura USB sin necesidad de sudo")
        cam.init()
        cam.set_gain_mode(GainMode.LOW)
        cam.start_streaming()
        time.sleep(0.3)
        _, raw = cam.read_frame_both()
        if raw is not None:
            print_status("Streaming Térmico", True, f"Frame térmico recibido con éxito ({raw.shape[1]}x{raw.shape[0]} px)")
            return True
        else:
            print_status("Streaming Térmico", False, "Frame vacío recibido de la cámara")
            return False
    except Exception as e:
        print_status("Driver P3", False, f"Error al inicializar cámara: {e}")
        return False
    finally:
        if cam is not None:
            try:
                cam.stop_streaming()
            except Exception:
                pass
            if cam.dev is not None:
                try:
                    import usb.util
                    usb.util.dispose_resources(cam.dev)
                except Exception:
                    pass
            try:
                cam.disconnect()
            except Exception:
                pass


def check_pixhawk() -> bool:
    print(f"\n{CYAN}--- Verificando Autopiloto Pixhawk 6C ---{RESET}")
    # Buscar posibles puertos
    candidates = ["/dev/ttyTHS1", "/dev/ttyACM0", "/dev/ttyACM1"]
    found_ports = [p for p in candidates if os.path.exists(p)]

    if not found_ports:
        print_status("Puerto Serial", False, "No se encontró ni /dev/ttyTHS1 (UART) ni /dev/ttyACM* (USB)")
        print(f"    {YELLOW}Para pruebas en escritorio: Conecta el Pixhawk por USB-C a la Jetson/PC.{RESET}")
        print(f"    {YELLOW}Para vuelo: Conecta TELEM2 (TX/RX/GND cruzados) a los pines UART de la Jetson.{RESET}")
        return False

    print_status("Puerto Serial", True, f"Puertos encontrados: {', '.join(found_ports)}")

    # Probar conexión MAVLink en los puertos encontrados
    from edge_core.flight.ignis_mavlink import IgnisMavlink

    for port in found_ports:
        baud = 921600 if "THS" in port else 115200
        print(f"  Probando MAVLink en {port} a {baud} baud (esperando heartbeat 3s)...")
        mav = IgnisMavlink(connection_string=port, baud=baud)
        if mav.connect(timeout_s=3.0):
            telem = mav.get_telemetry()
            print_status("MAVLink Heartbeat", True, f"Pixhawk respondiendo! Modo: {telem.mode} | Armed: {telem.armed}")
            mav.disconnect()
            return True
        mav.disconnect()

    print_status("MAVLink Heartbeat", False, "No se recibió respuesta en ningún puerto")
    print(f"    {YELLOW}Revisar: Parámetros SERIAL2_PROTOCOL=2 y SERIAL2_BAUD=921 en ArduCopter.{RESET}")
    return False


def check_heltec() -> bool:
    print(f"\n{CYAN}--- Verificando Módem LoRa Heltec (915 MHz) ---{RESET}")
    # Heltec V2 usa /dev/ttyUSB* (CP2102); Heltec V3 (ESP32-S3) usa /dev/ttyACM*
    all_ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    # Excluir /dev/ttyACM0 si pertenece a Pixhawk
    heltec_candidates = [p for p in all_ports if p != "/dev/ttyACM0" or not os.path.exists("/dev/ttyACM1")]
    if not heltec_candidates and "/dev/ttyACM1" in all_ports:
        heltec_candidates = ["/dev/ttyACM1"]

    if not heltec_candidates:
        print_status("Puerto Serial", False, "No se encontraron puertos /dev/ttyUSB* ni /dev/ttyACM* secundarios")
        print(f"    {YELLOW}Solución: Conectar el módem Heltec por cable USB.{RESET}")
        return False

    print_status("Puerto Serial", True, f"Módem Heltec detectado en: {', '.join(heltec_candidates)}")
    return True


def main() -> None:
    print(f"{BOLD}===================================================={RESET}")
    print(f"{BOLD}   IgnisEdge -- Diagnóstico de Hardware y Placas   {RESET}")
    print(f"{BOLD}===================================================={RESET}")

    p3_ok = check_p3_camera()
    pix_ok = check_pixhawk()
    hel_ok = check_heltec()

    print(f"\n{BOLD}===================================================={RESET}")
    print(f"{BOLD}   Resumen de Disponibilidad                        {RESET}")
    print(f"{BOLD}===================================================={RESET}")
    print_status("Cámara Térmica P3", p3_ok, "Listo" if p3_ok else "No lista")
    print_status("Pixhawk 6C", pix_ok, "Listo" if pix_ok else "No listo")
    print_status("Heltec LoRa", hel_ok, "Listo" if hel_ok else "No listo")
    print()

    if p3_ok and pix_ok and hel_ok:
        print(f"{GREEN}{BOLD}¡TODO EL HARDWARE ESTÁ LISTO PARA MODO VUELO!{RESET}")
    else:
        print(f"{YELLOW}Puedes ejecutar en modo simulación: python3 edge_core/launcher.py --sim{RESET}")


if __name__ == "__main__":
    main()
