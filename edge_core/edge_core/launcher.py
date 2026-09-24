"""
launcher.py -- Orquestador de Arranque del Sistema de Vuelo IgnisEdge.
====================================================================

Lanza y supervisa los componentes a bordo de la Jetson Orin Nano:
  1. ignis_vision: Percepción térmica con cámara P3 (Headless en vuelo o GUI en banco).
  2. ignis_mavlink: Enlace MAVLink con Pixhawk (Auto-detecta UART /dev/ttyTHS1 o USB /dev/ttyACM0).
  3. ignis_link: Radio LoRa 915 MHz con módem Heltec.
  4. ignis_mission: Cerebro FSM que ejecuta el triage y coordina la misión.

Uso:
  python3 edge_core/launcher.py --flight
  python3 edge_core/launcher.py --sim
  python3 edge_core/launcher.py --bench
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_link import IgnisLink
from edge_core.flight.ignis_mavlink import IgnisMavlink
from edge_core.mission.ignis_mission import IgnisMissionFSM
from edge_core.vision.dataset_recorder import DatasetRecorder
from edge_core.vision.ignis_vision import P3CaptureThread, VisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [launcher] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launcher")


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisEdge Flight Launcher")
    parser.add_argument("--sim", action="store_true", help="Ejecutar en modo simulación sin hardware")
    parser.add_argument("--bench", action="store_true", help="Modo banco de pruebas con GUI")
    parser.add_argument("--flight", action="store_true", help="Modo vuelo a bordo (hardware real, headless)")
    parser.add_argument("--pixhawk-port", default=None, help="Puerto Pixhawk forzado (def: auto-detect)")
    parser.add_argument("--heltec-port", default="/dev/ttyUSB0", help="Puerto UART Heltec (def: /dev/ttyUSB0)")
    parser.add_argument("--record-dataset", action="store_true", help="Capturar dataset de entrenamiento cuando detecte fuego o posibles fuegos")
    parser.add_argument("--dataset-dir", default="data/fire_dataset", help="Directorio destino para el dataset (def: data/fire_dataset)")
    parser.add_argument("--max-radius", type=float, default=5000.0, help="Radio máximo seguro permitido hacia un objetivo en metros (def: 5000m)")
    parser.add_argument("--patrol-alt", type=float, default=80.0, help="Altitud relativa de patrullaje / inspección en metros (def: 80m)")
    args = parser.parse_args()

    mode_str = "SIMULACIÓN" if args.sim else ("BANCO" if args.bench else "VUELO A BORDO")
    logger.info("==================================================")
    logger.info(f"Iniciando IgnisEdge Core en modo: {mode_str}")
    logger.info("==================================================")

    # 1. Enlace MAVLink con Pixhawk (con auto-detección de puerto)
    if args.sim:
        mavlink = IgnisMavlink(connection_string="/dev/null")
    else:
        # Prioridad de puertos: especificado -> /dev/ttyTHS1 -> /dev/ttyACM0
        candidates = [args.pixhawk_port] if args.pixhawk_port else ["/dev/ttyTHS1", "/dev/ttyACM0"]
        mavlink = None
        for port in candidates:
            if port and os.path.exists(port):
                baud = 921600 if "THS" in port else 115200
                logger.info(f"Intentando conectar Pixhawk en {port} ({baud} baud)...")
                candidate_mav = IgnisMavlink(connection_string=port, baud=baud)
                if candidate_mav.connect(timeout_s=3.0):
                    mavlink = candidate_mav
                    logger.info(f"Pixhawk conectado exitosamente en {port}!")
                    break
                else:
                    candidate_mav.disconnect()
        if mavlink is None:
            logger.warning("No se pudo conectar al Pixhawk. Continuando en modo degradado...")
            mavlink = IgnisMavlink(connection_string="/dev/null")

    # 2. Enlace Radio LoRa con Heltec (Auto-detección USB/ACM)
    heltec_port = args.heltec_port
    if not os.path.exists(heltec_port):
        for candidate in ["/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]:
            if os.path.exists(candidate) and (candidate != getattr(mavlink, "conn_str", "")):
                heltec_port = candidate
                break
    link = IgnisLink(port=heltec_port, sim_mode=args.sim or not os.path.exists(heltec_port))
    link.connect()

    # 3. Cerebro FSM (con límites perimétricos de seguridad)
    fsm = IgnisMissionFSM(
        mavlink=mavlink,
        link=link,
        max_allowed_dist_m=args.max_radius,
        nominal_patrol_alt_m=args.patrol_alt,
    )
    link.on_node_alert = fsm.on_node_alert_received

    # 4. Motor de Visión
    model_path = REPO_ROOT / "scripts" / "ignis_fire_classifier.joblib"
    engine = VisionEngine(model_path=model_path)

    # 5. Hilo de Captura Cámara Térmica P3
    camera_thread: Optional[P3CaptureThread] = None
    if not args.sim:
        try:
            camera_thread = P3CaptureThread(gain="low")
            camera_thread.start()
            logger.info("Hilo de captura P3 conectado y streaming.")
        except Exception as e:
            logger.warning(f"Cámara P3 no disponible en streaming: {e}")

    # 6. Grabador de Datasets Térmicos (Opcional)
    recorder: Optional[DatasetRecorder] = None
    if args.record_dataset:
        recorder = DatasetRecorder(output_dir=args.dataset_dir, max_fps=5.0)
        logger.info(f"Grabación de dataset habilitada en: {args.dataset_dir}")

    running = True

    def sig_handler(_sig: int, _frame: Any) -> None:
        nonlocal running
        logger.info("Interrupción recibida. Iniciando parada segura del sistema...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    logger.info("Sistema inicializado completamente. Lazo de vuelo activo a ~25 Hz.")

    last_log_time = 0.0

    try:
        while running:
            now = time.time()

            # A. Procesar frame de visión si la cámara está activa
            if camera_thread:
                frame_c = camera_thread.get_latest_frame(timeout=0.01)
                if frame_c is not None:
                    verdict = engine.process_frame(frame_c)
                    fsm.on_vision_verdict_received({
                        "status": verdict.status,
                        "num_fires": verdict.num_fires,
                        "tracks": [asdict(t) for t in verdict.tracks],
                    })

                    # Grabación selectiva cuando hay fuego o posibles fuegos
                    if recorder:
                        recorder.maybe_record(frame_c, verdict, telemetry=mavlink.get_telemetry())

                    # Log periódico de salud cada 2 segundos
                    if now - last_log_time >= 2.0:
                        telem = mavlink.get_telemetry()
                        fire_str = f"FUEGO: {verdict.num_fires}" if verdict.num_fires > 0 else "SIN FUEGO"
                        logger.info(
                            f"[TELEMETRÍA] {verdict.fps:.1f} fps | Max T: {verdict.max_temp_c:.1f}°C | "
                            f"{fire_str} | Modo: {telem.mode} | Bat: {telem.battery_pct}% | FSM: {fsm.state.name}"
                        )
                        last_log_time = now

            # B. Ciclo de la FSM de misión
            current_state = fsm.step()
            time.sleep(0.04)  # ~25 Hz

    finally:
        if recorder:
            recorder.stop()
        if camera_thread:
            camera_thread.stop()
        link.disconnect()
        mavlink.disconnect()
        logger.info("Todos los subsistemas desconectados ordenadamente. IgnisEdge apagado.")


if __name__ == "__main__":
    main()
