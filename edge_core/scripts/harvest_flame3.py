#!/usr/bin/env python3
"""
harvest_flame3.py  --  IgnisEdge / Cosecha automatica de FLAME 3 (subconjunto NADIR)
====================================================================================

Escarba la carpeta de FLAME 3 que YA tienes descargada (la version cenital,
con estructura  plot N/duringburn/geo_thermal_tiff_celsius/*.TIFF), saca lo
mejor para entrenar, y lo deja en tu dataset listo para train_classifier.py.

Que hace, en un comando:
  1. Recorre recursivamente tu carpeta FLAME 3.
  2. Agarra SOLO los TIFF termicos en °C de "duringburn" (el fuego). Ignora
     los RGB, los jpg de paleta y las carpetas pre/post burn.
  3. Corre la MISMA Etapa 1 que tus velas (detector contextual) sobre cada
     TIFF y recorta las manchas calientes (los recortes 64x64, 3 canales).
  4. Pre-etiqueta esos recortes como 'fire'  -> NO tienes que etiquetar a mano.
  5. Filtra por lo INCIPIENTE: descarta manchas gigantes (fuego ya desarrollado)
     y manchas tibias (suelo quemado enfriandose), para quedarse con lo que se
     parece a lo que vera tu dron a 100 m.
  6. Controla CUANTOS recortes guardar (para no llenarte de miles casi iguales).
  7. Los ESCRIBE en tu dataset/, mezclados con tus velas (APENDE al manifest.csv).

Perspectiva: este subconjunto es NADIR (mirando recto hacia abajo) = la vista
de tu dron. Por eso es el subconjunto ideal de FLAME 3 para ti.

Ojo honesto:
  - Este set es casi puro FUEGO (solo hay termica en 'duringburn'). Tus
    NEGATIVOS calientes (calefactor, sarten, parrilla...) salen de tus propias
    capturas con la vela. FLAME 3 aporta los positivos reales de altura.
  - El pre-etiquetado como 'fire' es best-effort: dentro de una imagen de quema,
    ademas de la llama activa puede haber suelo recien quemado aun caliente. El
    filtro --min-peak-temp sesga hacia la llama, pero IGUAL conviene revisar al
    voleo con:   python3 label_crops.py --data ./dataset/ --relabel
  - El canal temporal (parpadeo) queda NEUTRO en los recortes de FLAME (son fotos
    sueltas cenitales, no video). El parpadeo lo aprende de tus videos de vela.

Requisitos:  numpy, scipy, pandas, tifffile
  (pip install tifffile   si no lo tienes)
  Debe estar en la MISMA carpeta que harvest_crops.py (importa sus funciones).

Uso tipico:
  python3 harvest_flame3.py --flame ./datasets/flame3 --out ./dataset/
  # con mas control:
  python3 harvest_flame3.py --flame ./datasets/flame3 --out ./dataset/ \
          --every 2 --max-crops 2500 --min-peak-temp 90 --max-area 500
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# Reusar EXACTAMENTE la misma logica de recortes que las velas -----------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from harvest_crops import stage1_candidates, crop_window, build_channels
except ImportError:
    sys.exit("[error] no encuentro harvest_crops.py. Pon este script en la MISMA "
             "carpeta que harvest_crops.py.")


def load_thermal_tiff(path):
    """Carga un TIFF termico float en °C. Devuelve (frame, valid_mask)."""
    try:
        import tifffile
        arr = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except ImportError:
        sys.exit("[error] falta 'tifffile'. Instala:  pip install tifffile")
    if arr.ndim == 3:                       # por si viniera multibanda
        arr = arr[..., 0]
    # Los TIFF georreferenciados traen bordes 'nodata' (NaN o valores absurdos).
    # Los marcamos invalidos para que no generen falsas manchas calientes.
    valid = np.isfinite(arr) & (arr > -40.0) & (arr < 1500.0)
    if valid.sum() == 0:
        return None, None
    fill = float(np.median(arr[valid]))     # rellenar invalidos con el fondo
    arr = np.where(valid, arr, fill).astype(np.float32)
    return arr, valid


def main():
    ap = argparse.ArgumentParser(description="Cosecha automatica de FLAME 3 (NADIR) a tu dataset.")
    ap.add_argument("--flame", required=True, help="Carpeta raiz de FLAME 3 (la que tiene 'plot 1', 'plot 2'...)")
    ap.add_argument("--out", required=True, help="Tu carpeta dataset/ (se APENDE al manifest.csv existente)")
    ap.add_argument("--tiff-subdir", default="geo_thermal_tiff_celsius",
                    help="Subcarpeta con los TIFF en °C (por defecto la georreferenciada)")
    ap.add_argument("--phase", default="duringburn", help="Fase con fuego (donde hay termica)")
    ap.add_argument("--every", type=int, default=1, help="Tomar 1 de cada N TIFF (adelgaza casi-duplicados)")
    ap.add_argument("--max-crops", type=int, default=3000, help="Tope global de recortes a guardar")
    # Filtros Etapa 1 (mismos nombres que harvest_crops)
    ap.add_argument("--k", type=float, default=4.0, help="Umbral contextual en sigmas robustos")
    ap.add_argument("--t-abs", type=float, default=150.0, help="Umbral absoluto °C (fuego real supera 150)")
    ap.add_argument("--bg-kernel", type=int, default=41, help="Ventana del fondo local (px)")
    ap.add_argument("--crop", type=int, default=64, help="Lado del recorte (px)")
    ap.add_argument("--min-area", type=int, default=2, help="Area minima de mancha (px)")
    ap.add_argument("--max-area", type=int, default=600,
                    help="Area MAXIMA de mancha (px): descarta fuego ya grande, deja lo incipiente")
    ap.add_argument("--min-peak-temp", type=float, default=80.0,
                    help="Temp pico minima (°C) para aceptar la mancha: sesga a llama vs. suelo tibio")
    ap.add_argument("--min-sigma", type=float, default=5.0,
                    help="Anomalia minima en sigmas: el foco debe sobresalir de su entorno "
                         "(descarta artefactos y masas calientes uniformes). Sube a 8-10 para mas estricto.")
    # Escala de ch0 para FLAME (fuego real llega a ~500°C -> usar rango amplio)
    ap.add_argument("--t-lo", type=float, default=0.0, help="Piso de temp para ch0")
    ap.add_argument("--t-hi", type=float, default=550.0, help="Techo de temp para ch0")
    args = ap.parse_args()

    flame = Path(args.flame)
    if not flame.is_dir():
        sys.exit(f"[error] no existe la carpeta {flame}")

    # Encontrar todas las carpetas de TIFF termicos de la fase con fuego --------
    tiff_dirs = sorted(d for d in flame.rglob(args.tiff_subdir)
                       if d.is_dir() and args.phase in str(d).lower())
    if not tiff_dirs:
        sys.exit(f"[error] no encontre subcarpetas '{args.tiff_subdir}' bajo '{args.phase}'.\n"
                 f"        Revisa con:  find {flame} -type d -iname '{args.tiff_subdir}'")

    print(f"[info] carpetas de TIFF termico encontradas: {len(tiff_dirs)}")
    for d in tiff_dirs:
        print(f"        {d.relative_to(flame)}")

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    fields = ["crop_path", "session", "frame_idx", "blob_id", "cx", "cy",
              "x0", "y0", "x1", "y1", "area_px", "T_peak_C", "T_bg_C",
              "delta_peak_C", "sigma_C", "sigma_peak", "temporal_std_C", "label"]

    # Abrir manifest en modo APEND (crea encabezado solo si es nuevo) -----------
    new_file = not manifest_path.exists()
    fh = open(manifest_path, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields)
    if new_file:
        wr.writeheader()

    n_tiffs = 0
    n_crops = 0
    stop = False

    for d in tiff_dirs:
        # session = plotX  (nombre de la parcela, dos niveles arriba del tiff-subdir)
        plot_name = d.parent.parent.name.replace(" ", "").lower()
        session = f"flame3_{plot_name}"
        # rglob: busca recursivo, por si los TIFF estan en subcarpetas (ej. plot 2/.../part2/)
        seen = set()
        tiffs = []
        for pat in ("*.TIFF", "*.tif", "*.tiff"):
            for f in d.rglob(pat):
                if f not in seen:
                    seen.add(f)
                    tiffs.append(f)
        tiffs = sorted(tiffs)
        tiffs = tiffs[::args.every]
        print(f"\n[plot] {session}: {len(tiffs)} TIFF (cada {args.every})")

        for idx, tpath in enumerate(tiffs):
            if stop:
                break
            frame, valid = load_thermal_tiff(tpath)
            if frame is None:
                continue
            n_tiffs += 1

            mask, bg, delta, sigma = stage1_candidates(frame, args.k, args.t_abs, args.bg_kernel)
            mask = mask & valid                      # nunca detectar en bordes nodata

            labels, n = ndimage.label(mask)
            if n == 0:
                continue
            areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            coms = ndimage.center_of_mass(np.ones_like(labels), labels, index=np.arange(1, n + 1))

            for bid in range(1, n + 1):
                area = int(areas[bid - 1])
                if area < args.min_area or area > args.max_area:   # descarta gigantes e inutiles
                    continue
                blob = (labels == bid)
                t_peak = float(frame[blob].max())
                if t_peak < args.min_peak_temp:                    # sesga a llama vs. suelo tibio
                    continue
                cy, cx = coms[bid - 1]
                cy, cx = int(round(cy)), int(round(cx))
                d_peak = float(delta[blob].max())
                t_bg = float(bg[blob].mean())
                sigma_peak = d_peak / sigma
                # Filtro por ANOMALIA: el foco debe sobresalir de su entorno.
                # Descarta artefactos (bordes nodata, masas calientes uniformes)
                # que leen caliente en absoluto pero NO son un punto distinto del fondo.
                if sigma_peak < args.min_sigma:
                    continue

                T_win = crop_window(frame, cy, cx, args.crop)
                d_win = crop_window(delta, cy, cx, args.crop)
                tmp_win = np.zeros((args.crop, args.crop), dtype=np.float32)  # temporal neutro
                crop = build_channels(T_win, d_win, tmp_win, sigma, args.t_lo, args.t_hi)

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
                    "sigma_peak": round(sigma_peak, 2),
                    "temporal_std_C": 0.0,
                    "label": "fire",              # <- pre-etiquetado. Revisar con label_crops --relabel
                })

                if n_crops >= args.max_crops:
                    print(f"\n[info] alcance el tope de {args.max_crops} recortes.")
                    stop = True
                    break
        if stop:
            break

    fh.close()
    print(f"\n[ok] TIFF procesados : {n_tiffs}")
    print(f"[ok] recortes FLAME  : {n_crops}  (etiquetados 'fire')")
    print(f"[ok] guardados en    : {crops_dir}")
    print(f"[ok] manifest        : {manifest_path}  (apendido)")
    print()
    print("Recomendado: revisa al voleo que sean llama y no suelo quemado:")
    print("   python3 label_crops.py --data " + str(out) + " --relabel")
    print("Luego entrena mezclando con tus velas:")
    print("   python3 train_classifier.py --data " + str(out) + " --epochs 40")


if __name__ == "__main__":
    main()
