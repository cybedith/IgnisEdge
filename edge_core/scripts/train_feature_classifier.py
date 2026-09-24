#!/usr/bin/env python3
"""
train_feature_classifier.py  --  IgnisEdge / Clasificador de PRODUCCION
=======================================================================

Por que este y no el CNN:
  Los datos lo demostraron: una regla trivial de flicker ya acierta 95%. El
  CNN colapsaba porque con pocas sesiones memorizaba fondos. Este clasificador
  aprende la COMBINACION optima de las features FISICAS del fuego, que son
  robustas al fondo y a la sesion:
      flicker_peak_C  (parpadeo)         <- la mas fuerte
      T_peak_C        (temperatura)
      delta_peak_C    (cuanto sobresale del fondo local)
      area_px         (tamano del foco)
  Ventajas de produccion: interpretable, corre en microsegundos sin GPU en
  la Jetson, no colapsa, y se puede explicar en la memoria feature por feature.

Validacion HONESTA: GroupKFold por SESION (nunca mezcla una sesion entre
train y val), reporte de acierto por sesion, y curva de operacion (umbral vs
recall vs falsos positivos) para elegir el punto de trabajo.

Exporta:  ignis_fire_classifier.joblib  (modelo)
          ignis_fire_classifier.json    (features, umbral, metricas)
que carga live_classify.py y, despues, la Jetson.

Requisitos: numpy, pandas, scikit-learn, joblib   (pip install scikit-learn)

Uso:
  python3 train_feature_classifier.py --data ../dataset_master
  # FLAME se excluye por defecto (es fuego SIN flicker; regimen de altura aparte)
  python3 train_feature_classifier.py --data ../dataset_master --keep-flame
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    import joblib
except ImportError:
    raise SystemExit("[error] falta scikit-learn. Instala:  pip install scikit-learn")

# FEATURES por defecto: SOLO independientes del gain de verdad.
# 'delta_peak_C' se QUITO: es una diferencia de temperaturas ABSOLUTAS, depende
# del gain (fue un error incluirla). Las genuinamente invariantes son:
#   flicker_peak_C : parpadeo (std temporal) -> igual en cualquier gain
#   sigma_peak     : contraste en SIGMAS sobre el fondo -> adimensional, invariante
FEATURES = ["flicker_peak_C", "sigma_peak"]


def make_model(seed):
    # Arboles poco profundos + regularizacion: aprende combinaciones sin memorizar.
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=300,
        l2_regularization=1.0, min_samples_leaf=40,
        random_state=seed)


def metrics(y_true, p, thr):
    pred = p >= thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return recall, fpr, prec, tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser(description="Clasificador de produccion sobre features fisicas.")
    ap.add_argument("--data", default="../dataset_master")
    ap.add_argument("--keep-flame", action="store_true",
                    help="Incluir FLAME (fuego SIN flicker). Por defecto se excluye.")
    ap.add_argument("--drop-sessions", default="",
                    help="Sesiones a excluir, separadas por coma (ej. hervidor)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--flicker-floor", type=float, default=10.0,
                    help="COMPUERTA FISICA: si el parpadeo < esto, NO es fuego (prob=0) aunque el "
                         "modelo diga otra cosa. Rechaza objetos calientes QUIETOS nunca vistos.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ignis_fire_classifier")
    ap.add_argument("--features", default="",
                    help="Features separadas por coma (ej. flicker_peak_C,sigma_peak). Vacio = default.")
    args = ap.parse_args()

    global FEATURES
    if args.features:
        FEATURES = [f.strip() for f in args.features.split(",") if f.strip()]
    print(f"[features] usando: {FEATURES}")

    root = Path(args.data)
    df = pd.read_csv(root / "manifest.csv")
    df = df[df["label"].notna() & (df["label"] != "")].copy()

    if not args.keep_flame:
        before = len(df)
        df = df[~df["session"].str.startswith("flame3")].copy()
        print(f"[info] FLAME excluido (fuego sin flicker, regimen de altura): {before} -> {len(df)}")

    if args.drop_sessions:
        drop = {x.strip() for x in args.drop_sessions.split(",") if x.strip()}
        before = len(df)
        df = df[~df["session"].isin(drop)].copy()
        print(f"[info] excluidas {sorted(drop)}: {before} -> {len(df)}")

    # Features bien comportadas; NaN de flicker -> 0 (sin historia = sin parpadeo)
    df["flicker_peak_C"] = df["flicker_peak_C"].fillna(0.0)
    X = df[FEATURES].values.astype(np.float64)
    y = (df["label"] == "fire").astype(int).values
    groups = df["session"].values

    print(f"[data] recortes: {len(df)}  |  fire: {int(y.sum())}  |  other: {int((1-y).sum())}")
    print(f"[data] sesiones: {sorted(set(groups))}")

    # ------------------------------------------------------------------
    # Validacion honesta: GroupKFold por sesion (out-of-fold predictions)
    # ------------------------------------------------------------------
    n_groups = len(set(groups))
    folds = min(args.folds, n_groups)
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=folds)
    # COMPUERTA FISICA: sin parpadeo no hay fuego. Se aplica al sistema completo
    # (modelo + regla), asi que la evaluacion refleja lo que corre en produccion.
    gate = X[:, 0] >= args.flicker_floor        # X[:,0] = flicker_peak_C
    print(f"[gate] compuerta fisica: flicker < {args.flicker_floor}C => nunca fuego "
          f"({int((~gate).sum())} recortes rechazados de plano, {int(((~gate)&(y==1)).sum())} eran fuego)")

    print(f"\n[cv] GroupKFold por sesion, {folds} folds  (metricas ya con la compuerta)")
    for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = make_model(args.seed).fit(X[tr], y[tr])
        oof[va] = np.where(gate[va], m.predict_proba(X[va])[:, 1], 0.0)
        r, f, p, *_ = metrics(y[va], oof[va], 0.5)
        val_sessions = sorted(set(groups[va]))
        print(f"  fold {i+1}: val={val_sessions}  recall={r:.3f}  FP={f:.3f}  prec={p:.3f}")

    # Metricas globales out-of-fold (honestas)
    print("\n=== RESULTADO HONESTO (out-of-fold, nunca vio la sesion que evalua) ===")
    print(f"{'umbral':>7s} {'recall':>7s} {'FP-rate':>8s} {'prec':>6s}   (tp/fp/fn/tn)")
    best_thr, best_score = 0.5, -1
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r, f, p, tp, fp, fn, tn = metrics(y, oof, thr)
        print(f"{thr:7.2f} {r:7.3f} {f:8.3f} {p:6.3f}   ({tp}/{fp}/{fn}/{tn})")
        # punto de trabajo: maximiza recall - 2*FP (prioriza no perder fuegos)
        score = r - 2 * f
        if score > best_score:
            best_score, best_thr = score, thr
    print(f"\n[operacion] umbral sugerido = {best_thr}  (recall alto con FP bajo)")

    # Acierto por sesion (con el umbral sugerido)
    print("\n=== ACIERTO POR SESION (umbral sugerido) ===")
    pred = (oof >= best_thr).astype(int)
    for s in sorted(set(groups)):
        idx = groups == s
        acc = (pred[idx] == y[idx]).mean()
        lab = "fire " if y[idx][0] == 1 else "other"
        print(f"  {s:26s} [{lab}]: {acc*100:5.1f}%  (n={int(idx.sum())})")

    # ------------------------------------------------------------------
    # Modelo FINAL: entrenado con TODO, e importancia de features
    # ------------------------------------------------------------------
    final = make_model(args.seed).fit(X, y)
    # importancia por permutacion (interpretable): cuanto cae el acierto sin cada feature
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(final, X, y, n_repeats=5, random_state=args.seed, scoring="accuracy")
    print("\n=== IMPORTANCIA DE FEATURES (que usa el modelo para decidir) ===")
    order = np.argsort(pi.importances_mean)[::-1]
    for k in order:
        print(f"  {FEATURES[k]:16s}: {pi.importances_mean[k]:.4f}")

    r, f, p, *_ = metrics(y, oof, best_thr)
    joblib.dump(final, f"{args.out}.joblib")
    meta = {
        "features": FEATURES,
        "threshold": float(best_thr),
        "flicker_floor": float(args.flicker_floor),
        "oof_recall": float(r), "oof_fp_rate": float(f), "oof_precision": float(p),
        "n_train": int(len(df)), "sessions": sorted(set(groups)),
        "flame_included": bool(args.keep_flame),
        "note": "fuego = (flicker_peak_C >= flicker_floor) AND (prob >= threshold). "
                "Features en el MISMO orden que 'features'.",
    }
    with open(f"{args.out}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[ok] modelo exportado: {args.out}.joblib  +  {args.out}.json")
    print(f"[ok] usar en vivo:  python3 live_classify.py --model {args.out}.joblib")


if __name__ == "__main__":
    main()
