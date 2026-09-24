#!/usr/bin/env python3
"""
harvest_crops.py  --  IgnisEdge / Etapa 1 + cosecha de dataset
================================================================

Qué hace:
  1. Lee frames TERMICOS grabados del Thermal Master P3 (uint16 crudo o °C).
  2. Corre la ETAPA 1 del pipeline: el detector CONTEXTUAL de candidatos
     calientes  ->  esto es MATEMATICA PURA, no se entrena, no usa dataset.
        candidato si:  (T - fondo_local) > k * sigma_robusto   OR   T > T_abs
  3. Por cada mancha caliente, recorta una ventana 64x64 CON contexto y
     calcula los 3 canales relativos que alimentan al clasificador (Etapa 2):
        ch0 = temperatura absoluta escalada   (cuan caliente)
        ch1 = anomalia local en sigmas         (cuan raro vs. su entorno)  <- canal estrella
        ch2 = cambio temporal corto            (parpadeo / crecimiento)
  4. Guarda cada crop como .npy (64,64,3) y escribe un manifiesto CSV con
     una columna 'label' VACIA, lista para que la rellenes a mano.

Por que asi:
  - Recortes uint16 con canales relativos, NO imagenes .png. El canal ch1
    (anomalia en sigmas) cancela el domain gap entre camaras/alturas.
  - La Etapa 1 corre HOY sin datos: apunta el P3 a una vela + una plancha
    metalica al sol y verifica que marca el fuego. Ese es el test de humo.

Requisitos:  numpy, scipy   (pip install numpy scipy)

Uso tipico:
  # frames guardados como .npy sueltos dentro de carpetas-por-sesion:
  python3 harvest_crops.py --input ./capturas/ --out ./dataset/ --gain high

  # o un unico stack .npy (N, 192, 256):
  python3 harvest_crops.py --input sesion01.npy --out ./dataset/ --gain high

Formato de entrada aceptado:
  - Carpeta con .npy por-frame (cada uno HxW).  La SESION = nombre de la subcarpeta.
  - Un archivo .npy apilado (N,H,W).             La SESION = nombre del archivo.
  Detecta solo si es crudo (uint16) o °C (float) por el rango de valores;
  puedes forzar con --raw / --celsius.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------
# Conversion crudo -> °C  (segun spec del P3: raw/64 - 273.15, cuant. 15.6 mK)
# ----------------------------------------------------------------------------
def raw_to_celsius(frame_raw):
    return frame_raw.astype(np.float32) / 64.0 - 273.15


def looks_raw(frame):
    """Heuristica: crudo P3 ~ 273*64 = 17472 para 0°C. °C anda en decenas."""
    return np.issubdtype(frame.dtype, np.integer) and np.median(frame) > 5000


# ----------------------------------------------------------------------------
# Carga de frames (generador): entrega (session, frame_idx, frame_celsius)
#   Acepta .npy (crudo o °C) y .tif/.tiff (FLAME 3 entrega TIFF termico en °C).
# ----------------------------------------------------------------------------
def _load_thermal_file(path):
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf in (".tif", ".tiff"):
        try:
            import tifffile
            return np.asarray(tifffile.imread(str(path)))
        except ImportError:
            from PIL import Image
            return np.asarray(Image.open(str(path)))
    raise ValueError(f"extension no soportada: {path}")


def iter_frames(input_path, force_raw=None):
    p = Path(input_path)
    exts = ("*.npy", "*.tif", "*.tiff")

    def to_c(fr):
        fr = np.asarray(fr)
        if fr.ndim == 3:                 # TIFF multibanda (H,W,C): quedarse 1 canal
            fr = fr[..., 0]
        is_raw = force_raw if force_raw is not None else looks_raw(fr)
        return raw_to_celsius(fr) if is_raw else fr.astype(np.float32)

    if p.is_dir():
        # Carpeta: cada subcarpeta = una sesion; frames (.npy/.tif) ordenados.
        # Para FLAME 3: pon la carpeta "Fire" o "No Fire" y sus TIFF adentro.
        subdirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
        groups = subdirs if subdirs else [p]  # sin subcarpetas -> la carpeta es la sesion
        for g in groups:
            session = g.name
            files = sorted(f for ext in exts for f in g.glob(ext))
            for i, f in enumerate(files):
                yield session, i, to_c(_load_thermal_file(f))
    elif p.suffix.lower() in (".npy", ".tif", ".tiff"):
        session = p.stem
        arr = np.asarray(_load_thermal_file(p))
        if p.suffix.lower() == ".npy" and arr.ndim == 3 and arr.shape[-1] > 4:
            for i in range(arr.shape[0]):     # stack .npy (N,H,W)
                yield session, i, to_c(arr[i])
        else:
            yield session, 0, to_c(arr)       # un frame (2D o TIFF multibanda)
    else:
        sys.exit(f"[error] --input debe ser carpeta, .npy o .tif, no: {input_path}")


# ----------------------------------------------------------------------------
# ETAPA 1: detector contextual de candidatos  (nucleo matematico, sin entrenar)
# ----------------------------------------------------------------------------
def stage1_candidates(frame_c, k, t_abs, bg_kernel):
    """
    Devuelve (mask, bg, delta, sigma).
      bg     = fondo local (mediana en ventana grande)
      delta  = T - fondo
      sigma  = escala robusta (1.4826 * MAD global)  ~ desviacion estandar
      mask   = pixeles anomalos (contextual) O sobre umbral absoluto
    """
    # Fondo local: mediana de ventana grande -> robusto a gradientes de escena.
    bg = ndimage.median_filter(frame_c, size=bg_kernel, mode="nearest")
    delta = frame_c - bg

    # Escala robusta global via MAD (Median Absolute Deviation).
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    sigma = 1.4826 * mad
    sigma = max(sigma, 0.05)  # piso: evita division por ~0 en escenas planas

    mask = (delta > k * sigma) | (frame_c > t_abs)
    return mask, bg, delta, sigma


# ----------------------------------------------------------------------------
# Recorte 64x64 con padding por borde
# ----------------------------------------------------------------------------
def crop_window(arr, cy, cx, size):
    h, w = arr.shape
    half = size // 2
    r0, r1 = cy - half, cy + half
    c0, c1 = cx - half, cx + half
    r0c, r1c = max(r0, 0), min(r1, h)
    c0c, c1c = max(c0, 0), min(c1, w)
    win = arr[r0c:r1c, c0c:c1c]
    pad = ((r0c - r0, r1 - r1c), (c0c - c0, c1 - c1c))
    if any(sum(pr) for pr in pad):
        win = np.pad(win, pad, mode="edge")
    return win


# ----------------------------------------------------------------------------
# Construccion de los 3 canales de un crop (compartida con harvest_flame3.py
# y live_classify.py para que TODOS los recortes se generen igual)
#   ch0 = temperatura absoluta normalizada al rango [t_lo, t_hi]
#   ch1 = anomalia local en sigmas robustos (canal que cancela el domain gap)
#   ch2 = PARPADEO: magnitud de la oscilacion temporal del pixel (0=estatico,
#         1=parpadea fuerte). EL discriminador fuego vs. objeto caliente quieto.
#         Se pasa una imagen de desviacion estandar temporal; 0 si no hay historia.
# ----------------------------------------------------------------------------
def build_channels(T_win, d_win, flicker_win, sigma, t_lo, t_hi, flicker_scale=15.0):
    ch0 = np.clip((T_win - t_lo) / (t_hi - t_lo), 0, 1)
    ch1 = np.clip(d_win / sigma, -3, 12) / 12.0
    ch2 = np.clip(flicker_win / flicker_scale, 0, 1)   # 0=quieto, ~1=parpadeo del fuego
    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cosecha crops termicos (Etapa 1 + canales relativos).")
    ap.add_argument("--input", required=True, help="Carpeta con .npy por-frame o un stack .npy (N,H,W)")
    ap.add_argument("--out", required=True, help="Carpeta de salida del dataset")
    ap.add_argument("--gain", choices=["high", "low"], default="high",
                    help="high (patrulla, -20..150°C) o low (confirmacion, 0..550°C)")
    ap.add_argument("--k", type=float, default=4.0, help="Umbral contextual en sigmas robustos (baja=mas recall)")
    ap.add_argument("--t-abs", type=float, default=None, help="Umbral absoluto °C (default segun gain)")
    ap.add_argument("--bg-kernel", type=int, default=41, help="Tamano de ventana del fondo local (px)")
    ap.add_argument("--crop", type=int, default=64, help="Lado del crop en px")
    ap.add_argument("--min-area", type=int, default=2, help="Area minima de mancha en px (chico! el fuego incipiente es chico)")
    ap.add_argument("--flicker-window", type=int, default=10,
                    help="Frames para medir el parpadeo (std temporal). REQUIERE camara quieta. "
                         "A 25fps, 10 frames = 0.4s, captura varios ciclos de llama.")
    ap.add_argument("--flicker-scale", type=float, default=15.0,
                    help="Escala de normalizacion del parpadeo (°C de std -> 1.0). Baja=mas sensible.")
    ap.add_argument("--raw", action="store_true", help="Forzar interpretacion como crudo uint16")
    ap.add_argument("--celsius", action="store_true", help="Forzar interpretacion como °C float")
    ap.add_argument("--t-lo", type=float, default=None,
                    help="Forzar piso de temp para ch0 (por defecto segun gain). Usar para que velas "
                         "(HIGH) y confusores (LOW) tengan el MISMO rango de ch0 y el gain no sea pista falsa.")
    ap.add_argument("--t-hi", type=float, default=None, help="Forzar techo de temp para ch0")
    args = ap.parse_args()

    if args.raw and args.celsius:
        sys.exit("[error] --raw y --celsius son mutuamente excluyentes")
    force_raw = True if args.raw else (False if args.celsius else None)

    # Umbral absoluto por defecto segun ganancia. En HIGH los focos diluidos
    # leen 60-130°C, asi que el t_abs no debe ser alto: la deteccion fina la
    # hace el test contextual, t_abs es solo un respaldo para lo obvio.
    t_abs = args.t_abs if args.t_abs is not None else (80.0 if args.gain == "high" else 150.0)

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    # Escalas para ch0. Por defecto segun gain; pero se pueden FORZAR con --t-lo/--t-hi
    # para que todo el dataset comparta rango (clave si mezclas gains entre sesiones).
    if args.t_lo is not None and args.t_hi is not None:
        t_lo, t_hi = args.t_lo, args.t_hi
    else:
        t_lo, t_hi = (-20.0, 150.0) if args.gain == "high" else (0.0, 550.0)

    fields = ["crop_path", "session", "frame_idx", "blob_id", "cx", "cy",
              "x0", "y0", "x1", "y1", "area_px", "T_peak_C", "T_bg_C",
              "delta_peak_C", "sigma_C", "sigma_peak", "flicker_peak_C", "label"]

    n_frames = 0
    n_crops = 0
    # Buffer temporal por sesion: guarda ultimos frames para el canal de parpadeo.
    ring = {}

    with open(manifest_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()

        for session, idx, frame in iter_frames(args.input, force_raw=force_raw):
            n_frames += 1
            mask, bg, delta, sigma = stage1_candidates(frame, args.k, t_abs, args.bg_kernel)

            # Canal de PARPADEO (flicker): desviacion estandar temporal por pixel
            # sobre una ventana de W frames. El fuego oscila -> std alto; un objeto
            # caliente QUIETO -> std ~0. Esto REQUIERE camara quieta en la rafaga.
            buf = ring.setdefault(session, [])
            buf.append(frame)
            if len(buf) > args.flicker_window:
                buf.pop(0)
            flicker = np.stack(buf, axis=0).std(axis=0) if len(buf) >= 3 else np.zeros_like(frame)

            # Componentes conexas de la mascara
            labels, n = ndimage.label(mask)
            if n == 0:
                continue
            areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            coms = ndimage.center_of_mass(np.ones_like(labels), labels, index=np.arange(1, n + 1))

            for bid in range(1, n + 1):
                area = int(areas[bid - 1])
                if area < args.min_area:
                    continue
                cy, cx = coms[bid - 1]
                cy, cx = int(round(cy)), int(round(cx))

                blob = (labels == bid)
                t_peak = float(frame[blob].max())
                d_peak = float(delta[blob].max())
                t_bg = float(bg[blob].mean())
                flicker_peak = float(flicker[blob].max())

                # Recortes de los 3 canales
                T_win = crop_window(frame, cy, cx, args.crop)
                d_win = crop_window(delta, cy, cx, args.crop)
                flick_win = crop_window(flicker, cy, cx, args.crop)

                crop = build_channels(T_win, d_win, flick_win, sigma, t_lo, t_hi, args.flicker_scale)

                fname = f"{session}_f{idx:05d}_b{bid:03d}.npy"
                np.save(crops_dir / fname, crop)
                n_crops += 1

                half = args.crop // 2
                wr.writerow({
                    "crop_path": f"crops/{fname}",
                    "session": session, "frame_idx": idx, "blob_id": bid,
                    "cx": cx, "cy": cy,
                    "x0": cx - half, "y0": cy - half, "x1": cx + half, "y1": cy + half,
                    "area_px": area,
                    "T_peak_C": round(t_peak, 2), "T_bg_C": round(t_bg, 2),
                    "delta_peak_C": round(d_peak, 2), "sigma_C": round(sigma, 3),
                    "sigma_peak": round(d_peak / sigma, 2),
                    "flicker_peak_C": round(flicker_peak, 3),
                    "label": "",  # <- RELLENAR A MANO: fire / zinc / engine / sun_metal / person / veg / other
                })

    print(f"[ok] frames procesados : {n_frames}")
    print(f"[ok] crops extraidos   : {n_crops}")
    print(f"[ok] manifiesto        : {manifest_path}")
    print(f"[ok] crops en          : {crops_dir}")
    print()
    print("Siguiente paso: abre manifest.csv y rellena la columna 'label' por cada crop.")
    print("Etiquetas sugeridas: fire / zinc / engine / sun_metal / person / veg / other")
    print("(multi-clase para entrenar; se colapsa a binario fire-vs-resto al inferir)")


if __name__ == "__main__":
    main()
