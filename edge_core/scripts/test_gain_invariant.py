#!/usr/bin/env python3
"""
test_gain_invariant.py  --  IgnisEdge / Viabilidad de features independientes del gain
=======================================================================================

Objetivo: ver si podemos clasificar fuego vs no-fuego con features RELATIVAS
(razones y formas) que NO dependen de la escala absoluta del gain, para que el
mismo modelo funcione en LOW (dataset) y en HIGH (dron) SIN regrabar.

Calcula, desde el manifest + recortes que YA tienes, estas features:
  RELATIVAS (sobreviven al cambio de gain):
    flicker_ratio  = flicker_peak_C / (delta_peak_C + eps)   (parpadeo vs sobre-fondo)
    sigma_peak     = cuantas sigmas sobresale del fondo local (ya en el manifest)
    fill_ratio     = compacidad de la mancha (forma, en pixeles)
    aspect         = relacion de aspecto de la mancha
  Y para comparar, la ABSOLUTA:
    T_peak_C       = temperatura absoluta (DEPENDE del gain)

Mide el poder discriminante de cada feature (AUC: 1.0=separa perfecto, 0.5=inutil)
y entrena un mini-clasificador SOLO con las relativas, validado por sesion.

Si las relativas dan AUC alto y el clasificador valida bien, el Camino B es viable.

Requisitos: numpy, pandas, scikit-learn

Uso:  python3 test_gain_invariant.py --data ../dataset_master
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("[error] falta scikit-learn:  pip install scikit-learn")


def shape_feats(crop):
    """Compacidad y aspecto de la mancha (ch1), en pixeles -> independiente del gain."""
    ch1 = crop[..., 1]
    thr = max(0.15, ch1.mean() + 2 * ch1.std())
    mask = ch1 > thr
    if mask.sum() < 3:
        return 0.0, 0.0
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    fill = mask.sum() / (h * w)
    aspect = min(h, w) / max(h, w)
    return float(fill), float(aspect)


def main():
    ap = argparse.ArgumentParser(description="Test de features independientes del gain.")
    ap.add_argument("--data", default="../dataset_master")
    ap.add_argument("--per-session", type=int, default=400, help="Recortes por sesion a muestrear")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    df = pd.read_csv(root / "manifest.csv")
    df = df[df["label"].notna() & (df["label"] != "")].copy()
    # excluir FLAME (fuego sin flicker) y hervidor (confusor irreal)
    df = df[~df["session"].str.startswith("flame3")]
    df = df[df["session"] != "hervidor"]
    # excluir la sesion de etiquetas sucias
    df = df[df["session"] != "vela_con_horno"]
    df["flicker_peak_C"] = df["flicker_peak_C"].fillna(0.0)

    # muestrear para que sea rapido (leer recortes es lo lento)
    parts = []
    for s, g in df.groupby("session"):
        parts.append(g.sample(min(args.per_session, len(g)), random_state=args.seed))
    df = pd.concat(parts, ignore_index=True)

    # calcular features de forma leyendo los recortes
    fills, aspects = [], []
    for p in df["crop_path"]:
        try:
            c = np.load(root / p)
            f, a = shape_feats(c)
        except Exception:
            f, a = 0.0, 0.0
        fills.append(f); aspects.append(a)
    df["fill_ratio"] = fills
    df["aspect"] = aspects

    eps = 1e-3
    df["flicker_ratio"] = df["flicker_peak_C"] / (df["delta_peak_C"].abs() + eps)

    y = (df["label"] == "fire").astype(int).values
    groups = df["session"].values

    # ---- poder discriminante individual (AUC) de cada feature
    print("=== PODER DISCRIMINANTE por feature (AUC: 1.0=perfecto, 0.5=inutil) ===")
    cand = {
        "flicker_peak_C  (RELAT-ish)": df["flicker_peak_C"].values,
        "flicker_ratio   (RELATIVA) ": df["flicker_ratio"].values,
        "sigma_peak      (RELATIVA) ": df["sigma_peak"].values,
        "fill_ratio      (RELATIVA) ": df["fill_ratio"].values,
        "aspect          (RELATIVA) ": df["aspect"].values,
        "delta_peak_C    (semi-rel) ": df["delta_peak_C"].values,
        "T_peak_C        (ABSOLUTA) ": df["T_peak_C"].values,
    }
    for name, v in cand.items():
        v = np.nan_to_num(v)
        try:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)  # direccion no importa
        except Exception:
            auc = float("nan")
        print(f"  {name}: AUC {auc:.3f}")

    # ---- clasificador SOLO con features relativas (sin T absoluta), val por sesion
    REL = ["flicker_peak_C", "flicker_ratio", "sigma_peak", "fill_ratio", "aspect"]
    X = np.nan_to_num(df[REL].values.astype(np.float64))
    n_groups = len(set(groups))
    folds = min(5, n_groups)
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=folds)
    for tr, va in gkf.split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300,
                                            l2_regularization=1.0, min_samples_leaf=40,
                                            random_state=args.seed).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]

    print("\n=== CLASIFICADOR SOLO features RELATIVAS (independiente del gain) ===")
    print("   (validacion por sesion, out-of-fold)")
    try:
        print(f"   AUC global: {roc_auc_score(y, oof):.3f}")
    except Exception:
        pass
    for thr in [0.3, 0.5, 0.7]:
        pred = oof >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        print(f"   umbral {thr}: recall {rec:.3f}  FP-rate {fpr:.3f}")

    print("\nSi las RELATIVAS dan AUC alto (>0.9) y el clasificador recall alto/FP bajo,")
    print("el Camino B es viable: mismo modelo en LOW y HIGH sin regrabar.")


if __name__ == "__main__":
    main()