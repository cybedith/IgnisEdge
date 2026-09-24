import cv2
import numpy as np
import onnxruntime as ort

# 1. Cargar el modelo ONNX y definir clases
session = ort.InferenceSession("ignisedge_v1.onnx")
CLASSES = ["fire", "zinc", "engine", "sun_metal", "person", "veg", "other"]

# 2. Inicializar la cámara (0 es la webcam por defecto; si tu P3 entra por USB/capturadora, prueba con 1 o la ruta de video)
cap = cv2.VideoCapture(0)

print("[INFO] Iniciando transmisión en vivo. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el fotograma de la cámara.")
        break

    # Obtener dimensiones del fotograma real
    h, w, _ = frame.shape
    
    # Tomar el centro del cuadro como región de interés (ROI) simulando la detección del dron (64x64)
    # En producción real, aquí entra tu algoritmo de búsqueda de anomalías térmicas.
    cx, cy = w // 2, h // 2
    crop_size = 64
    x1, y1 = max(0, cx - crop_size // 2), max(0, cy - crop_size // 2)
    x2, y2 = min(w, x1 + crop_size), min(h, y1 + crop_size)
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        # Preprocesamiento idéntico al entrenamiento: redimensionar a 64x64 y normalizar canales
        roi_resized = cv2.resize(roi, (64, 64))
        img_input = roi_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1)) # (3, 64, 64)
        img_input = np.expand_dims(img_input, axis=0)  # (1, 3, 64, 64)
        
        # Inferencia con ONNX Runtime
        logits = session.run(None, {"crop": img_input})[0][0]
        class_idx = np.argmax(logits)
        pred_label = CLASSES[class_idx]
        confidence = float(np.max(np.exp(logits) / np.sum(np.exp(logits)))) # Softmax aproximado
        
        # Definir color del borde según la predicción (Rojo = Fuego, Verde = Seguro/Other)
        color = (0, 0, 255) if pred_label == "fire" else (0, 255, 0)
        text = f"{pred_label.upper()} ({confidence:.2f})"
        
        # Dibujar rectángulo y etiqueta en el visor
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar ventana en tiempo real
    cv2.imshow("IgnisEdge - Live Test", frame)

    # Romper ciclo con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()