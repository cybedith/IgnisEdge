#!/usr/bin/env python3
"""
capture_p3.py  --  IgnisEdge / Captura de frames del P3 para el dataset
=======================================================================

Muestra el video termico del P3 en vivo (para que apuntes y ENFOQUES el lente
manual) y, al apretar ESPACIO, guarda una RAFAGA de frames CRUDOS (uint16) como
.npy en una carpeta-por-sesion, lista para harvest_crops.py.

Por que rafaga y no una sola foto:
  - Frames seguidos (ms de diferencia) capturan el PARPADEO de la llama -> eso
    llena el canal temporal (ch2) que FLAME 3 no te dio.
  - Entre rafaga y rafaga repositionas la vela / cambias el fondo -> variedad.
  Para CONFUSORES (que no parpadean) puedes usar rafagas cortas o de 1 frame y
  apretar ESPACIO en muchas posiciones distintas.

Guarda el frame CRUDO uint16 (no °C): harvest_crops.py lo detecta y convierte
solo con raw/64-273.15. Asi no se pierde nada.

Requisitos:
  - El driver del P3 instalado en tu venv:
        cd ~/IgnisEdge/p3-ir-camera && pip install -e .
    (eso trae numpy, opencv, pyusb, etc.)
  - La regla udev del README puesta (permiso USB sin sudo).

Uso:
  # Positivos (vela) a 6 m, gain HIGH (el que patrulla el dron):
  python3 capture_p3.py --session vela_6m --gain high --out ../capturas --burst-frames 30

  # Confusor caliente (calefactor), rafagas cortas, muchas posiciones:
  python3 capture_p3.py --session calefactor --gain high --out ../capturas --burst-frames 6

Controles (en la ventana de video):
  ESPACIO = guardar una rafaga         n = disparar NUC (calibracion del sensor)
  g       = cambiar gain high/low      q = salir
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from p3_camera import P3Camera, GainMode, raw_to_celsius
except ImportError:
    sys.exit("[error] no encuentro 'p3_camera'. Instala el driver dentro de tu venv:\n"
             "        cd ~/IgnisEdge/p3-ir-camera && pip install -e .")
try:
    import cv2
except ImportError:
    sys.exit("[error] falta OpenCV. Deberia venir con el driver (pip install -e .).\n"
             "        Si no: pip install opencv-python")


def colorize(thermal_c, scale):
    """Normaliza por percentiles y aplica paleta de calor para el preview."""
    lo = float(np.percentile(thermal_c, 1))
    hi = float(np.percentile(thermal_c, 99))
    if hi - lo < 1.0:
        hi = lo + 1.0
    norm = np.clip((thermal_c - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    h, w = color.shape[:2]
    return cv2.resize(color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def put(img, text, org, color=(255, 255, 255)):
    """Texto con contorno negro para que se lea sobre cualquier fondo."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Captura frames del P3 como .npy para el dataset.")
    ap.add_argument("--session", required=True, help="Nombre de la sesion (ej. vela_6m, calefactor)")
    ap.add_argument("--out", default="./capturas", help="Carpeta base (se crea out/session/)")
    ap.add_argument("--gain", choices=["high", "low"], default="high",
                    help="high (-20..150°C, patrulla) o low (0..550°C, fuego caliente cercano)")
    ap.add_argument("--burst-frames", type=int, default=20, help="Frames por rafaga (ESPACIO)")
    ap.add_argument("--scale", type=int, default=3, help="Factor de zoom del preview")
    args = ap.parse_args()

    sess_dir = Path(args.out) / args.session
    sess_dir.mkdir(parents=True, exist_ok=True)

    gain = GainMode.HIGH if args.gain == "high" else GainMode.LOW

    print("[info] conectando al P3...")
    camera = P3Camera()
    camera.connect()
    camera.init()
    camera.set_gain_mode(gain)
    camera.start_streaming()

    # Warmup: descarta unos frames para que el streaming se estabilice antes
    # de pedir el primer NUC (evita timeouts si la camara recien se libero).
    for _ in range(10):
        try:
            camera.read_frame_both()
        except Exception:
            pass
        time.sleep(0.05)

    # NUC inicial tolerante: si falla (camara ocupada/no lista), seguimos igual;
    # se puede recalibrar despues con la tecla 'n'.
    try:
        camera.trigger_shutter()
    except Exception as e:
        print(f"[aviso] NUC inicial no respondio ({type(e).__name__}). "
              f"Sigo igual; aprieta 'n' para calibrar cuando veas imagen.")
    print(f"[info] listo. sesion='{args.session}'  gain={args.gain}  rafaga={args.burst_frames} frames")
    print("[info] ESPACIO=rafaga  n=NUC  g=gain  q=salir")

    win = f"P3 captura [{args.session}]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    burst = 0
    total = 0
    last_nuc = time.time()

    try:
        while True:
            ir, raw = camera.read_frame_both()
            if raw is None:
                continue
            tc = raw_to_celsius(raw)
            disp = colorize(tc, args.scale)

            # Estadisticas para que confirmes que la camara ve el objeto caliente
            tmax = float(tc.max())
            tmin = float(tc.min())
            ij = np.unravel_index(int(np.argmax(tc)), tc.shape)
            hy, hx = ij[0] * args.scale, ij[1] * args.scale
            cv2.circle(disp, (hx, hy), 8, (255, 255, 255), 1, cv2.LINE_AA)   # marca lo mas caliente

            put(disp, f"max {tmax:5.1f}C  min {tmin:5.1f}C   gain={args.gain}", (10, 24))
            put(disp, f"sesion: {args.session}   rafagas: {burst}   frames: {total}", (10, 46))
            put(disp, "ESPACIO=rafaga  n=NUC  g=gain  q=salir", (10, disp.shape[0] - 14),
                color=(0, 255, 255))

            # Aviso suave si hace rato no calibras (el sensor deriva)
            if time.time() - last_nuc > 90:
                put(disp, "conviene NUC (aprieta n)", (10, 68), color=(0, 200, 255))

            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            elif k == ord("n"):
                try:
                    camera.trigger_shutter()
                    last_nuc = time.time()
                    print("[nuc] shutter disparado (recalibrado)")
                except Exception as e:
                    print(f"[aviso] NUC no respondio ({type(e).__name__})")
            elif k == ord("g"):
                gain = GainMode.LOW if gain == GainMode.HIGH else GainMode.HIGH
                args.gain = "low" if gain == GainMode.LOW else "high"
                camera.set_gain_mode(gain)
                try:
                    camera.trigger_shutter()
                except Exception:
                    pass
                last_nuc = time.time()
                print(f"[gain] cambiado a {args.gain}")
            elif k == 32:   # ESPACIO -> rafaga
                burst += 1
                saved = 0
                for i in range(args.burst_frames):
                    _, r = camera.read_frame_both()
                    if r is None:
                        continue
                    np.save(sess_dir / f"{args.session}_b{burst:03d}_f{i:03d}.npy", r)  # CRUDO uint16
                    saved += 1
                total += saved
                print(f"[rafaga {burst:03d}] {saved} frames guardados  (total {total})")
    except KeyboardInterrupt:
        print("\n[info] interrumpido")
    finally:
        camera.stop_streaming()
        camera.disconnect()
        cv2.destroyAllWindows()

    print(f"\n[ok] {total} frames guardados en {sess_dir}")
    print(f"[ok] cuando termines todas las sesiones, cosecha con:")
    print(f"     python3 harvest_crops.py --input {args.out} --out ../dataset/ --gain high")


if __name__ == "__main__":
    main()
