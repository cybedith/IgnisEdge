#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def dilate_crop(crop, factor):
    """Simula la pérdida de resolución espacial y mezcla sub-píxel a distancia."""
    h, w = crop.shape[:2]
    # Comprimir para perder información de alta frecuencia
    small = cv2.resize(crop, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    # Expandir de vuelta al tamaño ROI de 64x64
    diluted = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # Atenuar el valor radiométrico (el calor se mezcla con el aire circundante)
    diluted = diluted * (1.0 / (factor * 0.5))
    return diluted.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al dataset de fuego")
    args = ap.parse_args()

    root = Path(args.data)
    manifest_path = root / "manifest.csv"
    df = pd.read_csv(manifest_path)

    fire_crops = df[df["label"] == "fire"].copy()
    if len(fire_crops) == 0:
        raise SystemExit("[error] No hay recortes etiquetados como 'fire' para diluir.")

    new_rows = []
    # Factores de simulación (ej. 2x, 4x y 8x la distancia original)
    factors = [2, 4, 8]

    for _, row in fire_crops.iterrows():
        crop_path = root / row["crop_path"]
        crop_data = np.load(crop_path)

        for f in factors:
            diluted_crop = dilate_crop(crop_data, f)
            new_name = str(row["crop_path"]).replace(".npy", f"_diluted_{f}.npy")
            new_path = root / new_name
            np.save(new_path, diluted_crop)

            new_row = row.copy()
            new_row["crop_path"] = new_name
            # La temperatura aparente registrada cae a mayor distancia
            new_row["T_peak_C"] = row["T_peak_C"] / (f * 0.5) 
            new_rows.append(new_row)

    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)
        df.to_csv(manifest_path, index=False)
        print(f"[ok] Creados {len(new_rows)} recortes de fuego diluido sintético.")
        print(f"[ok] Manifiesto actualizado con {len(df)} recortes totales.")

if __name__ == "__main__":
    main()