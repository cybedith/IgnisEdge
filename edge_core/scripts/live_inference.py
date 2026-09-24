#!/usr/bin/env python3
"""
Prueba en vivo del sistema híbrido completo (YOLO + Clasificador Morfológico).
Muestra detecciones de ambos pipelines y el veredicto del árbitro.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from p3_camera import P3Camera, raw_to_celsius, GainMode
from preprocess import preprocess_radiometric
from morphological_classifier import detect_and_classify, ThreatClass, AlertLevel
from ultralytics import YOLO

# ----------------- CONFIGURACIÓN -----------------
MODEL_PATH = Path.home() / "IgnisEdge/models/thermal_v3_s/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# -------------------------------------------------

class HybridArbiter:
    """Árbitro simplificado para pruebas en vivo."""
    
    @staticmethod
    def arbitrate(tracker_detections, yolo_detections, frame_shape):
        """
        Retorna: (alert_level, reasoning)
        """
        H, W = frame_shape
        
        # Filtrar trackers confirmados como WILDFIRE
        fire_trackers = [d for d in tracker_detections if d.threat_class == ThreatClass.WILDFIRE]
        
        # Filtrar detecciones YOLO
        yolo_fires = [y for y in yolo_detections if y['class_id'] == 0]  # wildfire
        yolo_fps = [y for y in yolo_detections if y['class_id'] == 1]    # false_positive
        
        # Regla 1: RED (Tracker + YOLO wildfire coinciden)
        for tracker in fire_trackers:
            cx, cy = tracker.features.centroid
            for yf in yolo_fires:
                if HybridArbiter._point_in_bbox(cx, cy, yf['bbox_px']):
                    return "RED", f"Tracker WILDFIRE + YOLO wildfire (conf={yf['confidence']:.2f})"
        
        # Regla 2: WHITE (Tracker WILDFIRE + YOLO false_positive)
        for tracker in fire_trackers:
            cx, cy = tracker.features.centroid
            for yfp in yolo_fps:
                if HybridArbiter._point_in_bbox(cx, cy, yfp['bbox_px']):
                    return "WHITE", f"CONFLICTO: Tracker WILDFIRE vs YOLO false_positive"
        
        # Regla 3: ORANGE (Solo tracker WILDFIRE)
        if fire_trackers:
            return "ORANGE", f"Tracker WILDFIRE (física), sin confirmación YOLO"
        
        # Regla 4: YELLOW (Solo YOLO wildfire)
        if yolo_fires:
            best = max(yolo_fires, key=lambda y: y['confidence'])
            return "YELLOW", f"YOLO wildfire (conf={best['confidence']:.2f}), tracker no confirma"
        
        return "NONE", "Sin detecciones de incendio"
    
    @staticmethod
    def _point_in_bbox(x, y, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2


def parse_yolo_results(results, frame_shape):
    """Convierte resultados de YOLO a formato simple."""
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            H, W = frame_shape
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'class_id': int(r.boxes.cls[i].item()),
                    'confidence': float(r.boxes.conf[i].item()),
                    'bbox_norm': (x1/W, y1/H, x2/W, y2/H),
                    'bbox_px': (int(x1), int(y1), int(x2), int(y2))
                })
    return detections


def draw_tracker_detections(image, tracker_detections):
    """Dibuja las detecciones del clasificador morfológico."""
    for det in tracker_detections:
        x, y, w, h = det.features.bbox
        color = (255, 0, 0) if det.threat_class == ThreatClass.WILDFIRE else (0, 255, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        label = f"Tracker: {det.threat_class.value}"
        cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def draw_yolo_detections(image, yolo_detections):
    """Dibuja las detecciones de YOLO."""
    for det in yolo_detections:
        x1, y1, x2, y2 = det['bbox_px']
        color = (0, 255, 0) if det['class_id'] == 0 else (0, 0, 255)
        label = f"YOLO: {'wildfire' if det['class_id']==0 else 'false_pos'} ({det['confidence']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def main():
    print("=" * 60)
    print("IgnisEdge - Sistema Híbrido en Vivo")
    print("=" * 60)
    
    # Cargar YOLO
    print(f"Cargando modelo: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Clases: {model.names}")
    
    # Conectar cámara
    print("Conectando P3...")
    cam = P3Camera()
    cam.connect()
    cam.init()
    time.sleep(0.5)
    cam.set_gain_mode(GainMode.LOW)
    cam.start_streaming()
    
    # Estabilizar
    for _ in range(5):
        cam.read_frame_both()
        time.sleep(0.04)
    
    print("\nSistema listo. Presiona 'q' para salir.\n")
    cv2.namedWindow("IgnisEdge Hybrid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IgnisEdge Hybrid", 1280, 720)
    
    arbiter = HybridArbiter()
    
    try:
        while True:
            # Leer frame
            _, thermal_raw = cam.read_frame_both()
            celsius = raw_to_celsius(thermal_raw)
            max_temp = np.max(celsius)
            
            # Pipeline físico (clasificador morfológico)
            tracker_detections = detect_and_classify(celsius)
            
            # Pipeline ML (YOLO)
            yolo_input = preprocess_radiometric(celsius)
            results = model.predict(yolo_input, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            yolo_detections = parse_yolo_results(results, celsius.shape)
            
            # Arbitraje
            alert_level, reasoning = arbiter.arbitrate(tracker_detections, yolo_detections, celsius.shape)
            
            # Visualización
            display = cv2.cvtColor(yolo_input[:,:,0], cv2.COLOR_GRAY2BGR)
            display = draw_tracker_detections(display, tracker_detections)
            display = draw_yolo_detections(display, yolo_detections)
            
            # Overlay de estado
            cv2.putText(display, f"Max: {max_temp:.1f} C", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Color del veredicto
            alert_colors = {
                "RED": (0, 0, 255),
                "ORANGE": (0, 140, 255),
                "YELLOW": (0, 255, 255),
                "WHITE": (200, 200, 200),
                "NONE": (100, 100, 100)
            }
            color = alert_colors.get(alert_level, (255, 255, 255))
            cv2.putText(display, f"ARBITRO: {alert_level}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, reasoning[:60], (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("IgnisEdge Hybrid", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cam.stop_streaming()
        cam.disconnect()
        cv2.destroyAllWindows()
        print("\nPrueba finalizada.")


if __name__ == "__main__":
    main()