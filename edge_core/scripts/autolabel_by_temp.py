#!/usr/bin/env python3
"""
autolabel_by_temp.py  --  IgnisEdge / Auto-etiquetado por temperatura
=====================================================================

Para sesiones donde el FUEGO es CLARAMENTE lo mas caliente del cuadro
(ej. una vela sola, con a lo mas cosas tibias de fondo como un notebook).

Que hace:
  - Recortes con T_pico >= --fire-temp   ->  etiqueta 'fire'  (la vela)
  - Recortes con T_pico <= --cold-temp   ->  MUESTRA como 'other' (ambiente/negativos)
  - Recortes en el medio (zona gris)     ->  se DESCARTAN (borran del manifest)
  - Deja una relacion sana negativos:positivos (no 13.000 tibios contra 1.000 velas)

Asi conviertes 14.000 recortes sin etiquetar en un dataset limpio y balanceado
SIN etiquetar a mano. Los recortes descartados tambien se borran del disco.

IMPORTANTE: usa esto SOLO cuando el fuego es lo unico caliente en la sesion.
Si hubiera otro foco caliente real (otra llama, un motor ardiendo), este metodo
lo etiquetaria mal. En ese caso usa label_crops.py a mano.

Requisitos: numpy, pandas

Uso tipico (sobre una sesion de vela ya cosechada):
  python3 autolabel_by_temp.py --data ../dataset_vela15 --fire-temp 120 --cold-temp 45

  # controlar cuantos negativos guardar (por defecto 3x los positivos):
  python3 autolabel_by_temp.py --data ../dataset_vela15 --neg-ratio 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Auto-etiqueta por temperatura (fuego = lo mas caliente).")
    ap.add_argument("--data", required=True, help="Carpeta de la sesion cosechada (con manifest.csv y crops/)")
    ap.add_argument("--fire-temp", type=float, default=120.0,
                    help="T_pico (°C) >= esto -> 'fire'. Sube si se cuela algo tibio como fuego.")
    ap.add_argument("--cold-temp", type=float, default=45.0,
                    help="T_pico (°C) <= esto -> candidato a negativo 'other'. El resto (zona gris) se descarta.")
    ap.add_argument("--neg-ratio", type=float, default=3.0,
                    help="Cuantos negativos guardar por cada positivo (muestra aleatoria). Ej. 3 = 3x.")
    ap.add_argument("--neg-label", default="other", help="Etiqueta para los negativos de ambiente")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Solo muestra que haria, sin escribir ni borrar")
    args = ap.parse_args()

    root = Path(args.data)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"[error] no encuentro {manifest}")

    df = pd.read_csv(manifest)
    n0 = len(df)

    fire = df[df["T_peak_C"] >= args.fire_temp].copy()
    cold = df[df["T_peak_C"] <= args.cold_temp].copy()
    gray = df[(df["T_peak_C"] > args.cold_temp) & (df["T_peak_C"] < args.fire_temp)].copy()

    n_fire = len(fire)
    if n_fire == 0:
        raise SystemExit(f"[error] ningun recorte supera {args.fire_temp}°C. "
                         f"Baja --fire-temp o revisa que la sesion tenga fuego caliente.")

    # Muestra de negativos: neg_ratio x positivos (o todos si hay menos)
    n_neg_target = int(round(n_fire * args.neg_ratio))
    rng = np.random.default_rng(args.seed)
    if len(cold) > n_neg_target:
        keep_idx = rng.choice(cold.index.values, size=n_neg_target, replace=False)
        neg = cold.loc[keep_idx].copy()
    else:
        neg = cold.copy()

    fire["label"] = "fire"
    neg["label"] = args.neg_label

    kept = pd.concat([fire, neg], ignore_index=True)
    kept_paths = set(kept["crop_path"].values)

    print(f"[resumen] recortes totales      : {n0}")
    print(f"[resumen] FIRE (>= {args.fire_temp:.0f}°C)   : {n_fire}")
    print(f"[resumen] negativos guardados   : {len(neg)}  (de {len(cold)} tibios <= {args.cold_temp:.0f}°C)")
    print(f"[resumen] zona gris descartada  : {len(gray)}  (entre {args.cold_temp:.0f} y {args.fire_temp:.0f}°C)")
    print(f"[resumen] descartados totales   : {n0 - len(kept)}")
    print(f"[resumen] dataset final         : {len(kept)}  ({n_fire} fire + {len(neg)} {args.neg_label})")

    if args.dry_run:
        print("\n[dry-run] no escribi nada. Quita --dry-run para aplicar.")
        return

    # Borrar del disco los crops que no se conservan
    removed = 0
    for p in df["crop_path"].values:
        if p not in kept_paths:
            fp = root / p
            if fp.exists():
                fp.unlink()
                removed += 1

    kept.to_csv(manifest, index=False)
    print(f"\n[ok] manifest reescrito con {len(kept)} recortes etiquetados")
    print(f"[ok] borrados del disco: {removed} recortes descartados")
    print(f"[ok] listo para mezclar/entrenar")


if __name__ == "__main__":
    main()
