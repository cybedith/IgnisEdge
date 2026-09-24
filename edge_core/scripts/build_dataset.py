#!/usr/bin/env python3
"""
build_dataset.py  --  IgnisEdge / Reconstruccion LIMPIA del dataset completo
============================================================================

Arma el dataset de entrenamiento desde CERO, desde las capturas originales,
en UN solo comando. Reemplaza los comandos sueltos de pandas que se cruzaron.

Que hace y por que:
  1. Cosecha cada sesion con su GAIN real (velas HIGH, confusores LOW) para que
     la temperatura salga bien, PERO fuerza un RANGO DE ch0 COMUN (--t-lo/--t-hi)
     para que el gain no sea una pista falsa. (Fix del gain mezclado sin regrabar.)
  2. Auto-etiqueta: velas -> 'fire' por temperatura;  confusores -> 'other'.
  3. Regenera FLAME 3 desde su carpeta original (fuego real de altura).
  4. BALANCEA: limita negativos para no tener 15.000 tibios contra 300 fuegos.
  5. Junta todo respetando las SESIONES REALES (sin bloques inventados), para que
     el split por sesion sea honesto y la validacion tenga fuego Y confusores.

Este script NO toca tus capturas (.npy). Solo reconstruye ../dataset_master.

Requisitos: numpy, scipy, pandas, tifffile   (todo ya instalado en tu venv)
Debe correr desde ~/IgnisEdge/scripts (junto a harvest_crops.py y harvest_flame3.py).

Uso:
  python3 build_dataset.py
  # o ajustando cosas:
  python3 build_dataset.py --neg-per-fire 1.5 --fire-temp 110
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # ~/IgnisEdge/scripts
ROOT = HERE.parent                               # ~/IgnisEdge
CAP = ROOT / "capturas"
FLAME = ROOT / "datasets" / "flame3"

# --- Define aqui tus fuentes reales -----------------------------------------
# (sesion de captura, gain con que se grabo, etiqueta destino)
# TODO grabado en LOW, camara quieta, rafagas largas (flicker vivo).
VELAS = [("vela_fondofrio", "low"), ("vela_fondotibio", "low"),
         ("vela_con_horno", "low"), ("velas_multiples", "low")]      # positivos
CONFUSORES = [("horno", "low"), ("hervidor", "low"), ("persona", "low")]  # negativos
# FLAME se maneja aparte (TIFF), abajo.


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        sys.exit(f"[error] fallo: {' '.join(str(c) for c in cmd)}")
    return r.stdout


def harvest_session(session, gain, out_dir, t_lo, t_hi, extra=None):
    """Cosecha una sesion de captura forzando el rango comun de ch0."""
    inp = CAP / session
    if not inp.is_dir():
        print(f"[aviso] no existe {inp}, la salto")
        return False
    cmd = ["python3", str(HERE / "harvest_crops.py"),
           "--input", str(inp), "--out", str(out_dir),
           "--gain", gain, "--t-lo", str(t_lo), "--t-hi", str(t_hi)]
    if extra:
        cmd += extra
    run(cmd)
    return True


def main():
    ap = argparse.ArgumentParser(description="Reconstruye el dataset master limpio.")
    ap.add_argument("--out", default=str(ROOT / "dataset_master"))
    ap.add_argument("--t-lo", type=float, default=0.0, help="Rango comun de ch0: piso (°C)")
    ap.add_argument("--t-hi", type=float, default=300.0, help="Rango comun de ch0: techo (°C)")
    ap.add_argument("--fire-temp", type=float, default=110.0, help="T_pico >= esto en velas -> candidato a 'fire'")
    ap.add_argument("--flicker-min", type=float, default=5.0,
                    help="Parpadeo (flicker_peak_C) minimo para que un foco caliente sea 'fire'. "
                         "Debajo de esto, caliente-pero-quieto = confusor. (vela ~45C, horno ~0.5C)")
    ap.add_argument("--cold-temp", type=float, default=45.0, help="T_pico <= esto en velas -> negativo ambiente")
    ap.add_argument("--neg-per-fire", type=float, default=2.0, help="Negativos por cada positivo (balance)")
    ap.add_argument("--flame-every", type=int, default=8, help="1 de cada N TIFF de FLAME")
    ap.add_argument("--flame-max", type=int, default=1200, help="Tope de recortes de FLAME")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[info] borro dataset previo {out}")
        shutil.rmtree(out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    tmp = ROOT / "_build_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    rng = np.random.default_rng(args.seed)
    all_rows = []

    # ---------------------------------------------------------------------
    # 1. VELAS (positivos) -> auto-etiqueta por temperatura
    # ---------------------------------------------------------------------
    print("\n=== VELAS (positivos) — fuego = caliente Y parpadea ===")
    for session, gain in VELAS:
        sdir = tmp / f"v_{session}"
        if not harvest_session(session, gain, sdir, args.t_lo, args.t_hi):
            continue
        df = pd.read_csv(sdir / "manifest.csv")
        # FUEGO = caliente Y parpadea (flicker alto). En vela_con_horno, esto
        # deja la vela como 'fire' y el horno (caliente pero QUIETO) como 'other'.
        fire = df[(df["T_peak_C"] >= args.fire_temp) & (df["flicker_peak_C"] >= args.flicker_min)].copy()
        hot_static = df[(df["T_peak_C"] >= args.fire_temp) & (df["flicker_peak_C"] < args.flicker_min)].copy()
        cold = df[df["T_peak_C"] <= args.cold_temp].copy()
        fire["label"] = "fire"
        hot_static["label"] = "other"     # caliente pero no parpadea = NO fuego
        cold["label"] = "other"
        fire["session"] = session
        hot_static["session"] = f"{session}_hotstatic"
        cold["session"] = f"{session}_bg"
        for _, r in pd.concat([fire, hot_static, cold]).iterrows():
            src = sdir / r["crop_path"]
            newname = f"{session}_{Path(r['crop_path']).name}"
            shutil.copy(src, out / "crops" / newname)
            r = r.copy(); r["crop_path"] = f"crops/{newname}"
            all_rows.append(r)
        print(f"  {session}: {len(fire)} fire (parpadea) + {len(hot_static)} caliente-quieto(other) + {len(cold)} ambiente")

    # ---------------------------------------------------------------------
    # 2. CONFUSORES (negativos) -> todo 'other'
    # ---------------------------------------------------------------------
    print("\n=== CONFUSORES (negativos) ===")
    for session, gain in CONFUSORES:
        sdir = tmp / f"c_{session}"
        if not harvest_session(session, gain, sdir, args.t_lo, args.t_hi):
            continue
        df = pd.read_csv(sdir / "manifest.csv")
        df["label"] = "other"
        df["session"] = session
        for _, r in df.iterrows():
            src = sdir / r["crop_path"]
            newname = f"{session}_{Path(r['crop_path']).name}"
            shutil.copy(src, out / "crops" / newname)
            r = r.copy(); r["crop_path"] = f"crops/{newname}"
            all_rows.append(r)
        print(f"  {session}: {len(df)} negativos ('other')")

    # ---------------------------------------------------------------------
    # 3. FLAME 3 (positivos de altura) -> regenera desde la carpeta original
    # ---------------------------------------------------------------------
    print("\n=== FLAME 3 (positivos de altura) ===")
    if FLAME.is_dir():
        fdir = tmp / "flame"
        run(["python3", str(HERE / "harvest_flame3.py"),
             "--flame", str(FLAME), "--out", str(fdir),
             "--every", str(args.flame_every), "--max-area", "60",
             "--min-peak-temp", "150", "--min-sigma", "8",
             "--max-crops", str(args.flame_max),
             "--t-lo", str(args.t_lo), "--t-hi", str(args.t_hi)])
        df = pd.read_csv(fdir / "manifest.csv")
        df["label"] = "fire"
        # sesiones ya vienen como flame3_plot1 / flame3_plot2
        for _, r in df.iterrows():
            src = fdir / r["crop_path"]
            newname = Path(r["crop_path"]).name
            shutil.copy(src, out / "crops" / newname)
            r = r.copy(); r["crop_path"] = f"crops/{newname}"
            all_rows.append(r)
        print(f"  FLAME: {len(df)} fire de altura")
    else:
        print(f"[aviso] no encuentro {FLAME}, sigo sin FLAME")

    # ---------------------------------------------------------------------
    # 4. Balancear negativos y escribir manifest
    # ---------------------------------------------------------------------
    df = pd.DataFrame(all_rows).reset_index(drop=True)
    fire = df[df["label"] == "fire"]
    other = df[df["label"] == "other"]
    n_fire = len(fire)
    n_neg_target = int(round(n_fire * args.neg_per_fire))

    if len(other) > n_neg_target:
        # muestra balanceada PERO conservando variedad de sesiones de confusor
        keep = other.sample(n=n_neg_target, random_state=args.seed)
        # borra del disco los negativos que no se conservan
        drop = other.drop(keep.index)
        for p in drop["crop_path"]:
            fp = out / p
            if fp.exists():
                fp.unlink()
        final = pd.concat([fire, keep], ignore_index=True)
    else:
        final = df

    final.to_csv(out / "manifest.csv", index=False)
    shutil.rmtree(tmp)

    # ---------------------------------------------------------------------
    # Resumen
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET MASTER RECONSTRUIDO")
    print("=" * 60)
    print(f"total recortes : {len(final)}")
    print(f"  fire  : {int((final.label=='fire').sum())}")
    print(f"  other : {int((final.label=='other').sum())}")
    print(f"\nsesiones (para el split por sesion):")
    print(final.session.value_counts().to_string())
    print(f"\nfire por sesion:")
    print(final[final.label=='fire'].session.value_counts().to_string())
    print(f"other por sesion:")
    print(final[final.label=='other'].session.value_counts().to_string())
    print(f"\n[ok] listo en {out}")
    print(f"[ok] entrena con:  python3 train_classifier.py --data {out} --epochs 40")


if __name__ == "__main__":
    main()
