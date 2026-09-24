#!/usr/bin/env python3
"""
test_simetria.py  --  IgnisEdge / Test de compacidad y simetria por sesion
==========================================================================

Verifica la idea de Pablo: la SIMETRIA/COHERENCIA de la mancha caliente separa
la LLAMA (compacta, coherente) del VAPOR/CONVECCION (difuso, caotico)?

Mide, sobre la mascara caliente de cada recorte, varias features de forma:
  - compacidad   : que tan "redonda/coherente" es (area / area_del_bounding_box).
                   Llama compacta -> alto; vapor disperso -> bajo.
  - fill_ratio   : area caliente / area del recuadro que la contiene.
  - simetria_h   : parecido entre mitad izquierda y derecha (0-1, 1=simetrico).
  - simetria_v   : parecido entre mitad superior e inferior.
  - solidez      : area / area del casco convexo (1 = sin huecos ni brazos).

Compara esos numeros entre las sesiones de FUEGO (velas) y las de CONFUSOR
(hervidor, horno). Si las velas puntuan alto y el hervidor bajo, la idea SIRVE.

NO re-cosecha nada: usa los recortes que ya estan en dataset_master.

Uso:  python3 test_simetria.py --data ../dataset_master --n 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


def shape_features(crop):
    """Calcula features de forma sobre la mancha caliente del recorte (canal ch1)."""
    # ch1 = anomalia; la mancha caliente son los pixeles con ch1 alto.
    ch1 = crop[..., 1]
    ch0 = crop[..., 0]
    # mascara de la mancha: pixeles claramente por encima del fondo
    thr = max(0.15, ch1.mean() + 2 * ch1.std())
    mask = ch1 > thr
    if mask.sum() < 3:
        # fallback: usar los pixeles mas calientes en temperatura
        thr0 = np.percentile(ch0, 99)
        mask = ch0 >= thr0
    if mask.sum() < 3:
        return None

    # quedarse con la componente conexa mas grande (la mancha principal)
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        biggest = 1 + int(np.argmax(sizes))
        mask = lab == biggest

    ys, xs = np.where(mask)
    area = mask.sum()
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    bbox_area = h * w

    fill_ratio = area / bbox_area            # que tan lleno esta su recuadro
    aspect = min(h, w) / max(h, w)           # 1 = cuadrado, ~0 = alargado

    # simetria: recortar al bounding box y comparar mitades
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    # horizontal (izq vs der espejada)
    left = sub[:, :w // 2]
    right = sub[:, w - w // 2:][:, ::-1]
    sim_h = 1.0 - np.abs(left - right).mean() if w >= 2 else 1.0
    # vertical (arriba vs abajo espejada)
    top = sub[:h // 2, :]
    bot = sub[h - h // 2:, :][::-1]
    sim_v = 1.0 - np.abs(top - bot).mean() if h >= 2 else 1.0

    # solidez: area / area del casco convexo (aprox con relleno morfologico)
    filled = ndimage.binary_fill_holes(mask)
    solidity = area / max(filled.sum(), 1)

    return dict(area=int(area), fill_ratio=fill_ratio, aspect=aspect,
                sim_h=sim_h, sim_v=sim_v, solidity=solidity)


def main():
    ap = argparse.ArgumentParser(description="Test de simetria/compacidad por sesion.")
    ap.add_argument("--data", default="../dataset_master")
    ap.add_argument("--n", type=int, default=300, help="Recortes por sesion a muestrear")
    args = ap.parse_args()

    root = Path(args.data)
    d = pd.read_csv(root / "manifest.csv")

    # sesiones de interes: velas (fuego) y los confusores dificiles
    focus = ["vela_fondofrio", "vela_fondotibio", "velas_multiples", "vela_con_horno",
             "hervidor", "horno", "persona"]

    rows = []
    for ses in focus:
        sub = d[d.session == ses]
        if len(sub) == 0:
            continue
        sub = sub.sample(min(args.n, len(sub)), random_state=0)
        feats = []
        for p in sub.crop_path:
            try:
                c = np.load(root / p)
            except Exception:
                continue
            f = shape_features(c)
            if f:
                feats.append(f)
        if not feats:
            continue
        fdf = pd.DataFrame(feats)
        lab = d[d.session == ses].label.iloc[0]
        rows.append((ses, lab, len(feats), fdf))

    # Imprimir promedios por sesion
    print(f"{'sesion':22s} {'tipo':6s} {'fill':>6s} {'aspect':>7s} {'sim_h':>6s} {'sim_v':>6s} {'solidez':>8s} {'area':>6s}")
    print("-" * 74)
    for ses, lab, n, fdf in rows:
        print(f"{ses:22s} {lab:6s} "
              f"{fdf.fill_ratio.mean():6.2f} {fdf.aspect.mean():7.2f} "
              f"{fdf.sim_h.mean():6.2f} {fdf.sim_v.mean():6.2f} "
              f"{fdf.solidity.mean():8.2f} {fdf.area.mean():6.0f}")

    # Resumen: comparar FUEGO vs los dos confusores dificiles
    print()
    fire = pd.concat([f for s, l, n, f in rows if l == "fire"], ignore_index=True) if any(l=="fire" for _,l,_,_ in rows) else None
    herv = next((f for s, l, n, f in rows if s == "hervidor"), None)
    horn = next((f for s, l, n, f in rows if s == "horno"), None)
    if fire is not None:
        print("PROMEDIOS CLAVE (mas alto = mas compacto/coherente = mas 'llama'):")
        for feat in ["fill_ratio", "aspect", "solidity"]:
            line = f"  {feat:12s}: FUEGO(velas)={fire[feat].mean():.2f}"
            if herv is not None: line += f"   hervidor={herv[feat].mean():.2f}"
            if horn is not None: line += f"   horno={horn[feat].mean():.2f}"
            print(line)
        print()
        print("Si FUEGO puntua CLARAMENTE distinto al hervidor/horno en alguna columna,")
        print("esa feature de forma SIRVE para separarlos y la metemos al clasificador.")


if __name__ == "__main__":
    main()
