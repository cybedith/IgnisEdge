#!/usr/bin/env python3
"""
Versión con DEBUG del sistema híbrido (CORREGIDA).
Muestra TODAS las detecciones del clasificador morfológico, no solo WILDFIRE.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from p3_camera import P3Camera, raw_to_celsius, GainMode
from preprocess import preprocess_radiometric, hysteresis_mask, SEED_TEMP_C, EXTEND_TEMP_C
from morphological_classifier import (
    detect_and_classify,
    ThreatClass,
    AlertLevel,
    MIN_REGION_AREA_PX,
    SEED_TEMP_C as CLASS_SEED,
    EXTEND_TEMP_C as CLASS_EXTEND
)
from ultralytics import YOLO

# ----------------- CONFIGURACIÓN -----------------
MODEL_PATH = Path.home() / "IgnisEdge/models/thermal_v3_s/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEBUG_MODE = True
# -------------------------------------------------

def draw_detailed_tracker(image, detections, celsius):
    """Dibuja TODAS las detecciones del clasificador con etiquetas detalladas."""
    for det in detections:
        x, y, w, h = det.features.bbox
        # Color según ThreatClass
        if det.threat_class == ThreatClass.WILDFIRE:
            color = (0, 0, 255)  # Rojo
        elif det.threat_class == ThreatClass.EXTENDED_HOT:
            color = (0, 255, 255)  # Amarillo
        elif det.threat_class == ThreatClass.POINT_SOURCE:
            color = (255, 0, 255)  # Magenta
        elif det.threat_class == ThreatClass.AMBIGUOUS:
            color = (255, 255, 0)  # Cian
        else:
            color = (128, 128, 128)  # Gris

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{det.threat_class.value} A={det.features.area_px} T={det.features.max_temp_c:.0f}C"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return image


def main():
    print("=" * 70)
    print("IgnisEdge - Sistema Híbrido MODO DEBUG")
    print("=" * 70)
    print(f"Thresholds clasificador: SEED={CLASS_SEED}°C, EXTEND={CLASS_EXTEND}°C")
    print(f"Área mínima: {MIN_REGION_AREA_PX} px")

    model = YOLO(str(MODEL_PATH))
    cam = P3Camera()
    cam.connect()
    cam.init()
    time.sleep(0.5)
    cam.set_gain_mode(GainMode.LOW)
    cam.start_streaming()

    for _ in range(5):
        cam.read_frame_both()
        time.sleep(0.04)

    cv2.namedWindow("Hybrid DEBUG", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid DEBUG", 1400, 800)

    print("\nLeyenda:")
    print("  - ROJO: Clasificador dice WILDFIRE")
    print("  - AMARILLO: EXTENDED_HOT")
    print("  - MAGENTA: POINT_SOURCE")
    print("  - CIAN: AMBIGUOUS")
    print("  - VERDE (caja): YOLO wildfire")
    print("  - AZUL (caja): YOLO false_positive\n")

    try:
        while True:
            _, thermal_raw = cam.read_frame_both()
            celsius = raw_to_celsius(thermal_raw)
            max_temp = np.max(celsius)

            # Clasificador morfológico
            tracker_detections = detect_and_classify(celsius)

            # YOLO
            yolo_input = preprocess_radiometric(celsius)
            results = model.predict(yolo_input, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

            # Visualización base (imagen preprocesada)
            display = cv2.cvtColor(yolo_input[:, :, 0], cv2.COLOR_GRAY2BGR)

            # Dibujar TODAS las detecciones del clasificador
            display = draw_detailed_tracker(display, tracker_detections, celsius)

            # Dibujar YOLO
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for i in range(len(r.boxes)):
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                        cls_id = int(r.boxes.cls[i].item())
                        conf = float(r.boxes.conf[i].item())
                        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                        label = f"YOLO: {model.names[cls_id]} ({conf:.2f})"
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Info general
            n_tracker = len(tracker_detections)
            fire_trackers = [d for d in tracker_detections if d.threat_class == ThreatClass.WILDFIRE]
            yolo_fires = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                yolo_fires = sum(1 for cls in results[0].boxes.cls if int(cls) == 0)

            cv2.putText(display, f"Max: {max_temp:.1f}C | Tracker detections: {n_tracker} (WILDFIRE: {len(fire_trackers)})",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"YOLO wildfire: {yolo_fires}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Máscara de histéresis (para referencia) - insertar de manera segura
            mask = hysteresis_mask(celsius).astype(np.uint8) * 255
            mask_small = cv2.resize(mask, (256, 192))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

            # Insertar en esquina superior derecha, asegurando que cabe
            h_disp, w_disp = display.shape[:2]
            if h_disp >= 202 and w_disp >= 266:
                display[10:10+192, w_disp-266:w_disp-10] = mask_color
                cv2.putText(display, "Hyst.Mask", (w_disp-266, 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                # Si la imagen es muy pequeña, no mostramos la miniatura
                cv2.putText(display, "Hyst.Mask (no room)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            cv2.imshow("Hybrid DEBUG", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop_streaming()
        cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()