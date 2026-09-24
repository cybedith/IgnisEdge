#!/usr/bin/env python3
"""
label_crops.py  --  IgnisEdge / Etiquetado interactivo de crops termicos
========================================================================

Por que esta herramienta y no LabelImg / CVAT / Roboflow:
  Esas son para dibujar CAJAS (bounding boxes) sobre imagenes. Tu no
  necesitas eso: la ETAPA 1 (harvest_crops.py) YA localizo cada mancha
  caliente y la recorto centrada. Lo que te queda es CLASIFICAR cada
  recorte con UNA etiqueta -> un click por crop. Mas rapido y mas simple.
  Ademas tus crops son matrices .npy (3 canales), no .png, asi que las
  herramientas de imagen normales ni siquiera los abren bien.

Que hace:
  - Recorre los crops NO etiquetados del manifest.csv.
  - Muestra cada uno en 2 paneles:
        izquierda  = temperatura absoluta (se ve "como una foto termica")
        derecha    = anomalia local en sigmas (cuan raro vs. su entorno)
    con las metricas arriba (T pico, sigmas, area, parpadeo).
  - Aprietas una tecla y queda etiquetado; avanza solo al siguiente.
  - Guarda de vuelta en el mismo manifest.csv (incremental).

Teclas:
  1 fire        2 zinc        3 engine      4 sun_metal
  5 person      6 veg         7 other
  espacio = saltar   u = deshacer ultimo   q = guardar y salir

Requisitos:  numpy, pandas, matplotlib
  (si no abre ventana en Windows:  pip install pyqt5   y reintenta)

Uso:
  python3 label_crops.py --data ./dataset/
  python3 label_crops.py --data ./dataset/ --relabel   # revisar TODOS, no solo vacios
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYMAP = {
    "1": "fire", "2": "zinc", "3": "engine", "4": "sun_metal",
    "5": "person", "6": "veg", "7": "other",
}
LEGEND = ("[1]fire  [2]zinc  [3]engine  [4]sun_metal  [5]person  [6]veg  [7]other"
          "\n[espacio]saltar   [u]deshacer   [q]guardar y salir")


def main():
    ap = argparse.ArgumentParser(description="Etiquetado interactivo de crops termicos.")
    ap.add_argument("--data", required=True, help="Carpeta del dataset (con manifest.csv y crops/)")
    ap.add_argument("--relabel", action="store_true", help="Revisar todos los crops, no solo los sin etiqueta")
    ap.add_argument("--save-every", type=int, default=20, help="Autoguardar cada N etiquetas")
    args = ap.parse_args()

    root = Path(args.data)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"[error] no encuentro {manifest}. Corre harvest_crops.py primero.")

    df = pd.read_csv(manifest)
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("")

    # Cola de trabajo: indices a etiquetar
    if args.relabel:
        queue = list(df.index)
    else:
        queue = list(df.index[df["label"] == ""])

    total = len(df)
    done = int((df["label"] != "").sum())
    if not queue:
        raise SystemExit(f"[ok] no hay crops pendientes. {done}/{total} ya etiquetados. "
                         f"Usa --relabel para revisarlos de nuevo.")

    print(f"[info] {len(queue)} crops por etiquetar  ({done}/{total} ya listos)")
    print("[info] enfoca la ventana y usa el teclado. Cierra con 'q'.")

    state = {"pos": 0, "history": [], "n_labeled": 0}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(bottom=0.22, top=0.86)

    def save():
        df.to_csv(manifest, index=False)

    def show():
        i = queue[state["pos"]]
        row = df.loc[i]
        crop = np.load(root / row["crop_path"])       # (64,64,3): ch0 abs, ch1 sigma, ch2 temporal
        axL.clear(); axR.clear()
        axL.imshow(crop[..., 0], cmap="inferno", vmin=0, vmax=1)
        axL.set_title("Temp. absoluta"); axL.axis("off")
        axR.imshow(crop[..., 1], cmap="magma", vmin=0, vmax=1)
        axR.set_title("Anomalia (sigmas)"); axR.axis("off")
        cur = row["label"] if row["label"] else "—"
        remaining = len(queue) - state["pos"]
        fig.suptitle(
            f"{row['crop_path'].split('/')[-1]}   |   T_pico {row.get('T_peak_C','?')}°C   "
            f"sigmas {row.get('sigma_peak','?')}   area {row.get('area_px','?')}px   "
            f"parpadeo(std) {row.get('temporal_std_C','?')}°C\n"
            f"sesion: {row.get('session','?')}   |   etiqueta actual: {cur}   |   "
            f"quedan: {remaining}", fontsize=9)
        fig.text(0.5, 0.06, LEGEND, ha="center", fontsize=9, family="monospace")
        fig.canvas.draw_idle()

    def advance():
        state["pos"] += 1
        if state["pos"] >= len(queue):
            print(f"[ok] terminaste la cola. Etiquetados esta sesion: {state['n_labeled']}.")
            save()
            plt.close(fig)
        else:
            show()

    def on_key(event):
        k = event.key
        if k == "q":
            save()
            print(f"[ok] guardado. Etiquetados esta sesion: {state['n_labeled']}. "
                  f"Total: {int((df['label']!='').sum())}/{total}.")
            plt.close(fig)
            return
        if k == "u":  # deshacer
            if state["history"]:
                prev_pos, prev_idx, prev_val = state["history"].pop()
                df.at[prev_idx, "label"] = prev_val
                state["pos"] = prev_pos
                state["n_labeled"] = max(0, state["n_labeled"] - 1)
                print(f"[undo] revertido crop {prev_idx}")
                show()
            return
        if k == " ":  # saltar sin etiquetar
            advance()
            return
        if k in KEYMAP:
            i = queue[state["pos"]]
            state["history"].append((state["pos"], i, df.at[i, "label"]))
            df.at[i, "label"] = KEYMAP[k]
            state["n_labeled"] += 1
            if state["n_labeled"] % args.save_every == 0:
                save()
                print(f"[autosave] {state['n_labeled']} etiquetados")
            advance()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()

    # Resumen final
    save()
    counts = df[df["label"] != ""]["label"].value_counts().to_dict()
    print("\n[resumen] crops por clase:")
    for c in ["fire", "zinc", "engine", "sun_metal", "person", "veg", "other"]:
        print(f"   {c:10s}: {counts.get(c, 0)}")
    print(f"[resumen] total etiquetados: {int((df['label']!='').sum())}/{total}")


if __name__ == "__main__":
    main()
