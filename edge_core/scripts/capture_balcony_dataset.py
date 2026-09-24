import sys
import cv2
import time
from pathlib import Path
import numpy as np

# Asegurar que el entorno reconozca los módulos locales
sys.path.insert(0, str(Path.home() / "IgnisEdge/scripts"))

try:
    from p3_camera import P3Camera, raw_to_celsius, GainMode
    from preprocess import preprocess_radiometric
except ImportError:
    print("Error: Asegúrate de estar en el venv y que p3_camera esté instalado.")
    sys.exit(1)

def main():
    output_dir = Path.home() / "IgnisEdge/datasets/yolo_fire_thermal_v3_raw/images"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = P3Camera()
    
    try:
        camera.connect()
        camera.init()
        camera.set_gain_mode(GainMode.LOW) 
        camera.start_streaming()
        
        print("--- Iniciando Captura Balcón IgnisEdge (Visor Doble) ---")
        print("Apuntando... Presiona 's' para guardar un frame.")
        print("Presiona 'q' para salir.")

        frame_count = 0

        while True:
            try:
                # 1. Leer datos crudos
                _, thermal_raw = camera.read_frame_both()
                celsius = raw_to_celsius(thermal_raw)
                max_temp = np.max(celsius)
                min_temp = np.min(celsius)
                
                # ---------------------------------------------------------
                # VENTANA 1: VISOR DE APUNTADO (Para el humano)
                # ---------------------------------------------------------
                # Normalizamos las temperaturas a 0-255 para poder verlas
                norm_celsius = cv2.normalize(celsius, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # Aplicamos un mapa de colores térmico (INFERNO) para visualizar la escena
                color_map = cv2.applyColorMap(norm_celsius, cv2.COLORMAP_INFERNO)
                display_color = cv2.resize(color_map, (512, 384), interpolation=cv2.INTER_NEAREST)
                
                cv2.putText(display_color, f"Max: {max_temp:.1f} C", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_color, "VISOR DE APUNTADO", (10, 370), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # ---------------------------------------------------------
                # VENTANA 2: VISOR IGNISEDGE (Para la IA)
                # ---------------------------------------------------------
                processed_img = preprocess_radiometric(celsius)
                display_processed = cv2.resize(processed_img, (512, 384), interpolation=cv2.INTER_NEAREST)
                
                cv2.putText(display_processed, f"Max: {max_temp:.1f} C", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_processed, "LO QUE VE YOLO (S PARA GUARDAR)", (10, 370), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Mostrar ambas ventanas
                cv2.imshow("1. Apuntado (Humano)", display_color)
                cv2.imshow("2. Pipeline IgnisEdge (IA)", display_processed)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    timestamp = int(time.time())
                    
                    # Guardamos SOLO lo que ve YOLO (y el respaldo NPY)
                    save_img = cv2.resize(processed_img, (640, 512), interpolation=cv2.INTER_NEAREST)
                    jpg_filename = output_dir / f"balcony_capture_{timestamp}.jpg"
                    cv2.imwrite(str(jpg_filename), save_img)
                    
                    npy_filename = output_dir / f"balcony_capture_{timestamp}.npy"
                    np.save(str(npy_filename), celsius)
                    
                    frame_count += 1
                    print(f"[{frame_count}] Guardado: {jpg_filename.name} (Max: {max_temp:.1f}°C)")
                    
                    # Efecto visual en la ventana procesada para confirmar
                    flash = np.ones_like(display_processed) * 255
                    cv2.imshow("2. Pipeline IgnisEdge (IA)", flash)
                    cv2.waitKey(50)
                    
                elif key == ord('q'):
                    print("Saliendo de la captura...")
                    break
                    
            except Exception as e:
                print(f"Error procesando frame: {e}")
                continue

    finally:
        print("Cerrando conexión con P3...")
        camera.stop_streaming()
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()