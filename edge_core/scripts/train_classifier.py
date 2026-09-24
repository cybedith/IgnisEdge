#!/usr/bin/env python3
"""
train_classifier.py  --  IgnisEdge / Etapa 2: clasificador ROI fuego vs. no-fuego
=================================================================================

Entrena el clasificador que decide, por cada crop caliente que entrego la
Etapa 1, si es FUEGO o no (techo de zinc, motor, roca al sol, etc.).

Lo NO negociable que este script hace bien:
  * SPLIT POR SESION, jamas aleatorio. Si crops del mismo vuelo caen en train
    y val, el modelo memoriza la escena y las metricas mienten. Aca sesiones
    completas van a train o a val, nunca partidas.
  * Entrena MULTI-CLASE (fire/zinc/engine/...) pero evalua BINARIO (fire vs
    resto), que es lo que importa en vuelo. Multi-clase ayuda a que aprenda a
    separar los confusores.
  * Baseline CNN chica (~150k params) primero. Haz que funcione punta a punta
    ANTES de subir a MobileNetV3. Un baseline que corre gana a un SOTA que no.

Requisitos:  torch, numpy, pandas   (pip install torch numpy pandas)

Uso:
  python3 train_classifier.py --data ./dataset/ --epochs 40 --val-frac 0.25
  # exporta ONNX al terminar:
  python3 train_classifier.py --data ./dataset/ --export modelo.onnx
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Clases: se ENTRENA con estas; se colapsa a binario (fire=1, resto=0) al medir.
CLASSES = ["fire", "zinc", "engine", "sun_metal", "person", "veg", "other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FIRE_IDX = CLASS_TO_IDX["fire"]


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class CropDataset(Dataset):
    def __init__(self, df, root, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        crop = np.load(self.root / row["crop_path"]).astype(np.float32)   # (64,64,3)
        if self.augment:
            # Flips y rotaciones: rompen la memorizacion de posicion/forma del fondo.
            if np.random.rand() < 0.5:
                crop = crop[:, ::-1, :]
            if np.random.rand() < 0.5:
                crop = crop[::-1, :, :]
            crop = np.rot90(crop, np.random.randint(0, 4), axes=(0, 1))
            # Ruido por canal: impide apoyarse en el patron EXACTO del fondo (ch0/ch1).
            # El flicker (ch2) es robusto a esto -> empuja al modelo a usarlo.
            crop = crop + np.random.normal(0, 0.04, crop.shape).astype(np.float32)
            crop = np.clip(crop, 0.0, 1.0)
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).contiguous()
        y = CLASS_TO_IDX[row["label"]]
        return x, y


# ----------------------------------------------------------------------------
# Baseline: CNN 6 capas, ~150k params, entrada 3x64x64
# ----------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(3, 16),    # 64 -> 32
            block(16, 32),   # 32 -> 16
            block(32, 64),   # 16 -> 8
            block(64, 64),   # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------
# Split POR SESION  (lo mas importante del archivo)
# ----------------------------------------------------------------------------
def split_by_session(df, val_frac, seed):
    sessions = sorted(df["session"].unique())
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val = max(1, int(round(len(sessions) * val_frac)))
    val_sessions = set(sessions[:n_val])
    train_df = df[~df["session"].isin(val_sessions)].copy()
    val_df = df[df["session"].isin(val_sessions)].copy()
    return train_df, val_df, sorted(val_sessions)


# ----------------------------------------------------------------------------
# Evaluacion BINARIA (fire vs. resto): recall de fuego y falsos positivos
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = tn = fn = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y = y.numpy()
        pred_fire = pred == FIRE_IDX
        true_fire = y == FIRE_IDX
        tp += int(np.sum(pred_fire & true_fire))
        fp += int(np.sum(pred_fire & ~true_fire))
        tn += int(np.sum(~pred_fire & ~true_fire))
        fn += int(np.sum(~pred_fire & true_fire))
    recall = tp / (tp + fn) if (tp + fn) else 0.0      # % de fuegos detectados
    precision = tp / (tp + fp) if (tp + fp) else 0.0   # de las alarmas, % reales
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0     # % de no-fuegos mal marcados
    return dict(recall=recall, precision=precision, fp_rate=fp_rate,
                tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    ap = argparse.ArgumentParser(description="Entrena clasificador ROI fuego vs. no-fuego.")
    ap.add_argument("--data", required=True, help="Carpeta del dataset (con manifest.csv y crops/)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25, help="Fraccion de SESIONES para validacion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-sessions", default="",
                    help="Sesiones a excluir separadas por coma (ej. flame3_plot1,flame3_plot2)")
    ap.add_argument("--no-augment", action="store_true",
                    help="Desactivar augmentation (por defecto esta ACTIVA)")
    ap.add_argument("--out", default="best_model.pt")
    ap.add_argument("--export", default=None, help="Ruta .onnx para exportar el mejor modelo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data)
    df = pd.read_csv(root / "manifest.csv")

    # Filtra crops sin etiquetar y etiquetas fuera del set.
    df = df[df["label"].notna() & (df["label"] != "")]
    unknown = set(df["label"].unique()) - set(CLASSES)
    if unknown:
        raise SystemExit(f"[error] etiquetas desconocidas en manifest: {unknown}\n"
                         f"        usa solo: {CLASSES}")
    if len(df) == 0:
        raise SystemExit("[error] no hay crops etiquetados. Rellena la columna 'label' primero.")

    # Excluir sesiones pedidas (ej. FLAME, que es fuego SIN flicker y ensucia la señal)
    if args.drop_sessions:
        drop = set(x.strip() for x in args.drop_sessions.split(",") if x.strip())
        before = len(df)
        df = df[~df["session"].isin(drop)].copy()
        print(f"[info] excluidas sesiones {sorted(drop)}: {before} -> {len(df)} recortes")

    train_df, val_df, val_sessions = split_by_session(df, args.val_frac, args.seed)
    print(f"[split] sesiones val: {val_sessions}")
    print(f"[split] train crops: {len(train_df)}  |  val crops: {len(val_df)}")
    print(f"[split] fuego en train: {int((train_df['label']=='fire').sum())}  "
          f"|  fuego en val: {int((val_df['label']=='fire').sum())}")
    if len(val_df) == 0 or (val_df["label"] == "fire").sum() == 0:
        print("[aviso] la validacion no tiene fuego o esta vacia. Captura mas sesiones "
              "con fuego y reparte para que train y val tengan positivos.")

    # Pesos de clase: el fuego suele ser minoria -> penaliza mas no detectarlo.
    counts = train_df["label"].value_counts().to_dict()
    weights = torch.tensor(
        [1.0 / max(counts.get(c, 0), 1) for c in CLASSES], dtype=torch.float32)
    weights = (weights / weights.sum() * len(CLASSES)).to(device)

    train_loader = DataLoader(CropDataset(train_df, root, augment=not args.no_augment),
                              batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(CropDataset(val_df, root), batch_size=args.batch,
                            shuffle=False, num_workers=2)

    model = BaselineCNN(len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[modelo] BaselineCNN  params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_df)

        m = evaluate(model, val_loader, device) if len(val_df) else {}
        msg = f"epoch {epoch:03d}  loss {train_loss:.4f}"
        if m:
            msg += (f"  | val recall {m['recall']:.3f}  prec {m['precision']:.3f}  "
                    f"FP-rate {m['fp_rate']:.3f}  (tp{m['tp']} fp{m['fp']} fn{m['fn']})")
        print(msg)

        # Guarda por mejor RECALL de fuego (no queremos perder incendios).
        if m and m["recall"] > best_recall:
            best_recall = m["recall"]
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "val_metrics": m}, args.out)

    print(f"[ok] mejor modelo guardado en {args.out}  (recall fuego = {best_recall:.3f})")

    # Export ONNX (opset 17). Luego en la JETSON:  trtexec --onnx=... --int8
    if args.export:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.randn(1, 3, 64, 64, device=device)
        torch.onnx.export(model, dummy, args.export, opset_version=17,
                          input_names=["crop"], output_names=["logits"],
                          dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}})
        print(f"[ok] ONNX exportado: {args.export}")
        print("     En la Jetson:  trtexec --onnx={} --int8 --saveEngine=modelo.trt".format(args.export))


if __name__ == "__main__":
    main()
