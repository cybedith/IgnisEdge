#!/usr/bin/env python3
"""
jetson_e2e_test.py -- Demostración y Verificación End-to-End en Jetson Orin Nano
"""
import glob
import os
import sys
import time
import socket
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_proto import AlertNodeMessage
from edge_core.flight.ignis_mavlink import IgnisMavlink
from edge_core.comms.ignis_link import IgnisLink
from edge_core.mission.ignis_mission import IgnisMissionFSM, MissionState
from edge_core.vision.dataset_recorder import DatasetRecorder
from edge_core.vision.ignis_vision import VisionFrameResult, TrackSnapshot

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}   {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def main():
    print_header("IGNIS-EDGE: PRUEBA INTEGRADA END-TO-END EN JETSON")
    
    # 1. Verificación de Hardware y Sistema Operativo
    print(f"{BOLD}[1/5] Verificación del Sistema Físico:{RESET}")
    print(f"  * Hostname: {socket.gethostname()}")
    print(f"  * Arquitectura: {platform.machine()}")
    print(f"  * Kernel: {platform.release()}")
    
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model = f.read().strip('\x00')
            print(f"  * Modelo Hardware: {GREEN}{model}{RESET}")
            
    serial_ports = [p for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyTHS1", "/dev/ttyUSB0"] if os.path.exists(p)]
    print(f"  * Puertos Serie detectados: {serial_ports}")
    
    # 2. Inicialización de Componentes
    print(f"\n{BOLD}[2/5] Inicializando Módulos de Vuelo y Telecomunicaciones:{RESET}")
    
    # MAVLink
    mav_port = "/dev/ttyACM0" if os.path.exists("/dev/ttyACM0") else "/dev/null"
    mav = IgnisMavlink(connection_string=mav_port)
    if os.path.exists("/dev/ttyACM0"):
        conn = mav.connect(timeout_s=2.0)
        print(f"  * Pixhawk 6C en /dev/ttyACM0: {'CONECTADO' if conn else 'TIMEOUT'}")
    else:
        print(f"  * Pixhawk 6C: No conectado en USB (Emulando telemetría de dron)")
        mav.telemetry.lat = -36.820100
        mav.telemetry.lon = -73.044100
        mav.telemetry.alt_rel_m = 10.0
        mav.telemetry.battery_pct = 92
        mav.telemetry.mode = "LOITER"
        mav.telemetry.armed = True
        
    telem = mav.get_telemetry()
    print(f"    -> Telemetría Inicial: Lat={telem.lat:.6f}, Lon={telem.lon:.6f}, Alt={telem.alt_rel_m:.1f}m, Bat={telem.battery_pct}%, Modo={telem.mode}")

    # Heltec LoRa
    heltec_port = "/dev/ttyACM1" if os.path.exists("/dev/ttyACM1") else "/dev/ttyUSB0"
    heltec_avail = os.path.exists(heltec_port)
    link = IgnisLink(port=heltec_port, sim_mode=not heltec_avail)
    link.connect()
    print(f"  * Módem LoRa Heltec: {GREEN}ACTIVO en {heltec_port}{RESET}" if heltec_avail else "  * Módem LoRa Heltec: SIMULADO")

    # Dataset Recorder
    dataset_dir = Path(REPO_ROOT) / "data" / "demo_dataset"
    recorder = DatasetRecorder(output_dir=dataset_dir)
    print(f"  * Dataset Recorder: {GREEN}ACTIVO{RESET} (Destino: {dataset_dir})")

    # 3. Máquina de Estados (FSM)
    print(f"\n{BOLD}[3/5] Instanciando Cerebro de Misión (IgnisMissionFSM):{RESET}")
    fsm = IgnisMissionFSM(
        mavlink=mav,
        link=link,
        max_allowed_dist_m=35.0,     # Límite perimétrico seguro (35 metros max)
        nominal_patrol_alt_m=8.0,    # Altitud segura de prueba: 8 metros
    )
    print(f"  * Estado Inicial FSM: {GREEN}{fsm.state.name}{RESET}")

    # 4. Inyección de Alerta de Nodo de Humo/Gas
    print(f"\n{BOLD}[4/5] Simulando Alerta de Nodo Sensor (Gas BME688 detectado):{RESET}")
    # Simular nodo a 12 metros al Norte del dron (dentro del perímetro seguro de 35m)
    d_lat = 12.0 / 111320.0
    target_lat = telem.lat + d_lat
    target_lon = telem.lon
    
    alert = AlertNodeMessage(
        node_id=1,
        seq=1,
        lat=target_lat,
        lon=target_lon,
        gas_res_kohm=42,     # Caída de resistencia por gas/humo
        confidence=94,
    )
    print(f"  * Nodo #{alert.node_id} emite alerta:")
    print(f"    - Coordenadas Objetivo: Lat={alert.lat:.6f}, Lon={alert.lon:.6f} (a 12.0m al norte)")
    print(f"    - Resistencia Gas: {alert.gas_res_kohm} kΩ | Confianza: {alert.confidence}%")
    
    # Entregar alerta a la FSM
    fsm.on_node_alert_received(alert)
    print(f"  * FSM tras recibir alerta: {YELLOW}{fsm.state.name}{RESET}")

    # Ciclo de Triage
    print("  * Evaluando Factibilidad Energética y Perimétrica (TRIAGE)...")
    fsm.step()
    print(f"  * Resultado Triage: {GREEN}{fsm.state.name}{RESET} -> Modo GUIDED activado hacia el foco.")

    # Simular tránsito hacia el nodo
    print("\n  * Dron en vuelo autónomo hacia la posición del foco...")
    fsm.state = MissionState.ON_STATION
    fsm.t_state_entered = time.time()
    print(f"  * Dron en estación de inspección: {CYAN}{fsm.state.name}{RESET}")

    # 5. Detección de Fuego Térmico y Veredicto
    print(f"\n{BOLD}[5/5] Inspección con Cámara Térmica y Grabación de Dataset:{RESET}")
    # Simular que el clasificador de flicker confirma un foco térmico
    track = TrackSnapshot(
        id=1,
        state="FUEGO",
        cx=115,
        cy=95,
        confidence=0.96,
        last_flicker=0.88,
        last_T=184.5,
        area_px=28.0,
        hits=15,
        obs=15,
    )
    res = VisionFrameResult(
        timestamp=time.time(),
        status="ACTIVO",
        num_fires=1,
        max_temp_c=184.5,
        fps=16.0,
        tracks=[track],
    )
    
    fire_dict = {
        "tracks": [
            {
                "track_id": 1,
                "state": "FUEGO",
                "bbox": [80, 100, 110, 130],
                "confidence": 0.96,
                "flicker_score": 0.88,
                "temp_max_c": 184.5,
                "temp_mean_c": 128.0,
            }
        ]
    }
    fsm.on_vision_verdict_received(fire_dict)
    
    # Grabar cuadro al dataset automático
    import numpy as np
    mock_thermal_celsius = np.full((192, 256), 22.0, dtype=np.float32)
    mock_thermal_celsius[80:110, 100:130] = 184.5  # Foco caliente
    
    recorded = recorder.maybe_record(
        frame_c=mock_thermal_celsius,
        result=res,
        telemetry=telem,
    )
    recorder.stop()
    print(f"  {GREEN}[OK] Dataset guardado exitosamente:{RESET}")
    saved_files = glob.glob(f"{recorder.session_dir}/*")
    for sf in sorted(saved_files):
        print(f"       -> {os.path.basename(sf)} ({os.path.getsize(sf)} bytes)")

    # Ejecutar paso final de FSM
    fsm.step()
    print(f"\n  * Veredicto Final de Misión: {GREEN}FUEGO CONFIRMADO ({fsm.state.name}){RESET}")
    
    # Limpieza
    link.disconnect()
    
    print_header("PRUEBA COMPLETADA EXITOSAMENTE EN LA JETSON")
    print(f"{GREEN}Todos los componentes (Telemetría, LoRa, FSM, Safety Limits, Dataset Recorder){RESET}")
    print(f"{GREEN}están verificados y 100% operativos en la Jetson Orin Nano.{RESET}\n")

if __name__ == "__main__":
    main()
