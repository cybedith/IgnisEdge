#!/usr/bin/env python3
"""
autolabel_negatives.py -- Etiquetado masivo de confusores calientes
Ignora el fondo frío y etiqueta los recortes calientes con una clase negativa específica.
"""

import argparse
import pandas as pd
from pathlib import Path

# Clases negativas válidas según train_classifier.py
NEGATIVE_CLASSES = ["zinc", "engine", "sun_metal", "person", "veg", "other"]

def main():
    ap = argparse.ArgumentParser(description="Auto-etiqueta confusores calientes como negativos.")
    ap.add_argument("--data", required=True, help="Carpeta del dataset (ej. ../dataset_neg)")
    ap.add_argument("--label", required=True, choices=NEGATIVE_CLASSES, help="Clase a asignar")
    ap.add_argument("--min-temp", type=float, default=60.0, help="Temp mínima para etiquetar (ej. 70)")
    ap.add_argument("--dry-run", action="store_true", help="Muestra el cálculo sin guardar")
    args = ap.parse_args()

    manifest_path = Path(args.data) / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"[error] No se encontró {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Filtrar solo los recortes que superen la temperatura mínima del confusor
    hot_mask = df["T_peak_C"] >= args.min_temp

    if args.dry_run:
        print("\n--- MODO PRUEBA (dry-run) ---")
        print(f"Dataset      : {manifest_path}")
        print(f"Clase        : {args.label}")
        print(f"Filtro Temp  : >= {args.min_temp} °C")
        print(f"Total crops  : {len(df)}")
        print(f"A etiquetar  : {hot_mask.sum()} (calientes)")
        print(f"Ignorados    : {(~hot_mask).sum()} (fondo frío)")
    else:
        # Asignar la etiqueta solo a la máscara caliente
        df.loc[hot_mask, "label"] = args.label
        df.to_csv(manifest_path, index=False)
        print(f"[ok] {hot_mask.sum()} recortes calientes etiquetados como '{args.label}'.")
        print(f"[ok] Manifiesto actualizado: {manifest_path}")

if __name__ == "__main__":
    main()