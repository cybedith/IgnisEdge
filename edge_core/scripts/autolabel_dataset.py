import cv2
import os
from pathlib import Path
import numpy as np

def main():
    # Definición de rutas según tu estructura actual
    base_dir = Path.home() / "IgnisEdge/datasets/yolo_fire_thermal_v3_raw"
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # Crear carpeta de labels si no existe
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Búsqueda recursiva de JPGs (entra automáticamente en balcon/ y horno/)
    image_paths = list(images_dir.rglob("*.jpg"))
    print(f"--- Iniciando Procesamiento ---")
    print(f"Encontradas {len(image_paths)} imágenes en las subcarpetas.")
    
    labeled_count = 0
    background_count = 0

    for img_path in image_paths:
        # El nombre del label debe ser igual al de la imagen pero .txt
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        # Leer imagen en escala de grises
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error leyendo: {img_path.name}")
            continue
            
        height, width = img.shape
        
        # Detectar píxeles que no sean negros (el objeto caliente)
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos para segmentación (YOLO-seg)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_polygons = []
        for contour in contours:
            # Filtrar ruido menor a 5 píxeles
            if cv2.contourArea(contour) < 5:
                continue
                
            # Simplificar el polígono (DP algorithm)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # --- PARCHE DE SEGURIDAD YOLO ---
            # Un polígono DEBE tener al menos 3 vértices para existir
            if len(approx) < 3:
                continue
            # --------------------------------
            
            # Formato segmentación YOLO: class x1 y1 x2 y2 ... (normalizado 0-1)
            # Clase 1 = false_positive (maquinaria/horno/cocina)
            polygon_str = "1 " 
            for point in approx:
                x = point[0][0] / width
                y = point[0][1] / height
                polygon_str += f"{x:.6f} {y:.6f} "
            
            valid_polygons.append(polygon_str.strip())
            
        # Escritura de los archivos
        if valid_polygons:
            with open(label_path, 'w') as f:
                f.write("\n".join(valid_polygons))
            labeled_count += 1
        else:
            # Si la imagen es negra (balcón) o puro ruido, creamos un archivo vacío
            # Esto es vital para el Negative Mining en YOLO
            open(label_path, 'w').close()
            background_count += 1

    print("\n--- Resumen de Auto-Etiquetado ---")
    print(f"Imágenes de 'Horno/Cocina' etiquetadas (Clase 1): {labeled_count}")
    print(f"Imágenes de 'Balcón' marcadas como fondo: {background_count}")
    print(f"Total archivos .txt generados en: {labels_dir}")

if __name__ == "__main__":
    main()