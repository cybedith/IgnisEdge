"""
IgnisEdge - YOLOv8s Training v3 with Class Imbalance Handling

Trains a 2-class YOLOv8s-seg model on the fused dataset:
  Class 0: wildfire        (~800 images from FLAME 3, zero-masked radiometric)
  Class 1: false_positive  (~100 images captured with P3, urban/domestic hot objects)

Imbalance strategy: PHYSICAL OVERSAMPLING.
  Ultralytics does not support class weights in its API (neither 8.4.x nor later).
  Workarounds that DON'T work:
    - Increasing `cls` loss globally → affects all classes equally
    - Setting `weights=` arg → not supported in YOLO.train()
    - Focal loss flag → also affects all classes equally
  The only reliable solution with current Ultralytics is to oversample the
  minority class physically by duplicating its label files into a separate
  train folder before training.

Augmentation strategy (justified for radiometric zero-masked grayscale):
  - hsv_h=0, hsv_s=0:  grayscale images have no meaningful hue/saturation
  - hsv_v=0.1:         minimal brightness jitter (simulates sensor gain variations)
  - mosaic=1.0:        combines 4 images per batch (diversifies context)
  - mixup=0.15:        slight blending (regularization; kept low because blending
                       two zero-masked images can create non-physical patterns)
  - degrees=15:        drone rotation invariance
  - fliplr=0.5,        valid for nadir aerial view
    flipud=0.3
  - scale=0.5:         objects
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Dict

import yaml
from ultralytics import YOLO

class TrainConfig:
    def __init__(
        self,
        flame3_dataset_root: Path,
        balcony_dataset_root: Path,
        output_dataset_root: Path,
        output_model_dir: Path,
        run_name: str,
        oversample_ratio: float,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str,
    ):
        self.flame3_dataset_root = flame3_dataset_root
        self.balcony_dataset_root = balcony_dataset_root
        self.output_dataset_root = output_dataset_root
        self.output_model_dir = output_model_dir
        self.run_name = run_name
        self.oversample_ratio = oversample_ratio
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="IgnisEdge YOLOv8 v3 Training")
    # Usa Path.home() para resolver ~ correctamente y apuntar a tus datos
    home_dir = Path.home()
    p.add_argument("--flame3-source", type=Path, default=home_dir / "IgnisEdge/datasets/yolo_fire_thermal_v2",
                   help="Path to FLAME 3 dataset (Class 0)")
    p.add_argument("--balcony-source", type=Path, default=home_dir / "IgnisEdge/datasets/yolo_fire_thermal_v3_raw",
                   help="Path to new balcony dataset (Class 1)")
    p.add_argument("--output-dataset", type=Path, default=home_dir / "IgnisEdge/datasets/yolo_fire_thermal_v3_fused",
                   help="Path to generated balanced dataset")
    p.add_argument("--output-models", type=Path, default=home_dir / "IgnisEdge/models",
                   help="Path to save model weights")
    p.add_argument("--name", type=str, default="thermal_v3_s")
    # Calculamos el ratio: 800 (FLAME) / ~150 (Balcon Clase 1) = ~5.3. Usamos 6 por seguridad.
    p.add_argument("--oversample", type=float, default=6.0,
                   help="Multiplier for minority class duplication")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--skip-oversample", action="store_true",
                   help="Use existing output-dataset without regenerating")
    args = p.parse_args()

    return TrainConfig(
        flame3_dataset_root=args.flame3_source,
        balcony_dataset_root=args.balcony_source,
        output_dataset_root=args.output_dataset,
        output_model_dir=args.output_models,
        run_name=args.name,
        oversample_ratio=args.oversample,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

def prepare_fused_dataset(cfg: TrainConfig) -> None:
    print(f"--- Iniciando Fusión de Datasets y Oversampling ---")
    
    # Limpiar y crear estructura
    if cfg.output_dataset_root.exists():
        shutil.rmtree(cfg.output_dataset_root)
    
    for split in ["train", "val"]:
        (cfg.output_dataset_root / split / "images").mkdir(parents=True)
        (cfg.output_dataset_root / split / "labels").mkdir(parents=True)

    # 1. Copiar FLAME 3 (Asumiendo que ya tiene estructura train/val)
    print("Copiando dataset base FLAME 3 (Clase 0)...")
    flame_splits = ["train", "val", "valid"] # Maneja 'valid' o 'val'
    for split_dir in cfg.flame3_dataset_root.iterdir():
        if split_dir.is_dir() and split_dir.name in flame_splits:
            target_split = "val" if split_dir.name in ["val", "valid"] else "train"
            
            img_src_dir = split_dir / "images"
            lbl_src_dir = split_dir / "labels"
            
            if img_src_dir.exists():
                for img_file in img_src_dir.glob("*.jpg"):
                    shutil.copy(img_file, cfg.output_dataset_root / target_split / "images" / img_file.name)
            if lbl_src_dir.exists():
                for lbl_file in lbl_src_dir.glob("*.txt"):
                     shutil.copy(lbl_file, cfg.output_dataset_root / target_split / "labels" / lbl_file.name)

    # 2. Procesar Dataset Balcón (Clase 1 y Background)
    print("Procesando y oversampleando capturas de balcón (Clase 1)...")
    balcony_img_dir = cfg.balcony_dataset_root / "images"
    balcony_lbl_dir = cfg.balcony_dataset_root / "labels"
    
    # Obtener todas las imágenes generadas por el script de auto-etiquetado
    all_balcony_images = list(balcony_img_dir.rglob("*.jpg"))
    
    import random
    random.shuffle(all_balcony_images)
    
    # Split simple: 80% train, 20% val
    split_idx = int(len(all_balcony_images) * 0.8)
    train_imgs = all_balcony_images[:split_idx]
    val_imgs = all_balcony_images[split_idx:]
    
    def process_split(img_list, split_name, is_train):
        added_count = 0
        for img_path in img_list:
            lbl_name = img_path.stem + ".txt"
            lbl_path = balcony_lbl_dir / lbl_name
            
            # Verificar si existe el label
            if not lbl_path.exists():
                continue
                
            # Leer el label para saber si es background (vacío) o Clase 1
            with open(lbl_path, 'r') as f:
                content = f.read().strip()
            
            is_background = len(content) == 0
            
            # Determinar cuántas copias hacer (Oversampling solo aplica a train y a Clase 1)
            num_copies = int(cfg.oversample_ratio) if (is_train and not is_background) else 1
            
            for i in range(num_copies):
                # Modificar el nombre para evitar colisiones si se duplica
                suffix = f"_copy{i}" if num_copies > 1 else ""
                new_img_name = f"{img_path.stem}{suffix}.jpg"
                new_lbl_name = f"{img_path.stem}{suffix}.txt"
                
                target_img = cfg.output_dataset_root / split_name / "images" / new_img_name
                target_lbl = cfg.output_dataset_root / split_name / "labels" / new_lbl_name
                
                shutil.copy(img_path, target_img)
                shutil.copy(lbl_path, target_lbl)
                added_count += 1
        return added_count

    train_added = process_split(train_imgs, "train", is_train=True)
    val_added = process_split(val_imgs, "val", is_train=False)
    
    print(f"  Agregadas {train_added} imágenes a train (Oversampling ratio: {cfg.oversample_ratio})")
    print(f"  Agregadas {val_added} imágenes a val")

    # 3. Generar el dataset.yaml
    yaml_path = cfg.output_dataset_root / "dataset.yaml"
    yaml_content = {
        "path": str(cfg.output_dataset_root.absolute()),
        "train": "train/images",
        "val": "val/images",
        "names": {
            0: "wildfire",
            1: "false_positive"
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"--- Fusión Completada. Archivo YAML en: {yaml_path} ---")


def main() -> None:
    cfg = parse_args()

    if not cfg.flame3_dataset_root.exists():
        print(f"ERROR: No se encuentra el dataset FLAME 3 en: {cfg.flame3_dataset_root}")
        print("Asegúrate de que la carpeta yolo_fire_thermal_v2 esté ahí.")
        sys.exit(1)
        
    if not cfg.balcony_dataset_root.exists():
         print(f"ERROR: No se encuentra el dataset del balcón en: {cfg.balcony_dataset_root}")
         sys.exit(1)

    # Phase 1: prepare balanced dataset
    skip = "--skip-oversample" in sys.argv
    if skip and (cfg.output_dataset_root / "dataset.yaml").exists():
        print("Saltando oversampling, usando el dataset fusionado existente.")
    else:
        prepare_fused_dataset(cfg)

    yaml_path = cfg.output_dataset_root / "dataset.yaml"
    
    print("\n--- Iniciando Entrenamiento YOLOv8 ---")
    # Initialize YOLOv8s-seg model
    model = YOLO("yolov8s-seg.pt")
    
    # Ensure output directory exists
    cfg.output_model_dir.mkdir(parents=True, exist_ok=True)

    # Train the model with specific augmentations for radiometric data
    results = model.train(
        data=str(yaml_path),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        project=str(cfg.output_model_dir),
        name=cfg.run_name,
        
        # Augmentations (Customized for Zero-Masked Radiometric Data)
        hsv_h=0.0,    # Hue jitter no tiene sentido en B/N
        hsv_s=0.0,    # Saturation jitter no tiene sentido en B/N
        hsv_v=0.1,    # Pequeño jitter de brillo
        mosaic=1.0,   # Ayuda a encontrar objetos pequeños
        mixup=0.15,   # Regularización
        degrees=15.0, # El dron no va a volar de cabeza, pero +-15 grados es realista
        fliplr=0.5,   # Asimetría natural
        flipud=0.3,   # Válido para tomas cenitales (nadir)
        scale=0.5,
        
        # Opciones extra sugeridas para segmentación
        overlap_mask=True, 
        mask_ratio=4,
    )
    
    print("\n--- Entrenamiento Finalizado Exitosamente ---")
    print(f"Los pesos del modelo se encuentran en: {cfg.output_model_dir / cfg.run_name / 'weights'}")


if __name__ == "__main__":
    main()