#!/usr/bin/env python3
"""
live_classify.py  --  IgnisEdge / Detector EN VIVO de PRODUCCION (veredicto robusto)
====================================================================================

Camara P3 + clasificador de flicker + LOGICA DE DECISION CONSERVADORA. Detecta
fuego en tiempo real, fluido, independiente del gain, y -clave- SIN falsos verdes:
un objeto caliente (horno, hervidor) NUNCA dispara "fuego" ni por un instante.

Filosofia (correcta para despachar brigadas): un fuego real a 100 m permanece en
vista mucho tiempo, asi que confirmar con calma no cuesta nada; pero un falso
positivo manda recursos al vado. Por eso el sistema NACE DESCONFIADO y solo
declara fuego con evidencia ALTA y SOSTENIDA.

Capas:
  1. WARM-UP global: hasta llenar la ventana de flicker, el sistema solo OBSERVA,
     no emite ningun veredicto (mata la inestabilidad del arranque).
  2. DETECTOR rapido + union de manchas (un fuego = un foco).
  3. TRACKER con periodo de GRACIA: un foco recien nacido es "evaluando"; no puede
     ser fuego hasta ser observado min_obs frames.
  4. COMPUERTA SOSTENIDA: exige flicker alto en VARIOS frames seguidos (no un pico).
  5. EVIDENCIA (log-odds) + HISTERESIS ASIMETRICA: cuesta mucho ENCENDER (umbral
     alto), es facil mantener; el fuego real sube parejo, un pico de confusor no.

Sin torch. Corre en CPU en tu PC y en la Jetson. Esta logica es la que vivira en
ignis_mission: acumular evidencia hasta un veredicto robusto.

Requisitos: numpy, scipy, opencv, scikit-learn, joblib, driver p3_camera.
Junto a harvest_crops.py y al modelo ignis_fire_classifier.joblib/.json.

Uso:
  python3 live_classify.py --model ignis_fire_classifier.joblib --gain low --t-floor 45
Controles:  n = NUC (recalibra + reinicia warm-up)   +/- = sensibilidad   q = salir
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import cv2
except ImportError:
    sys.exit("[error] falta opencv (viene con el driver).")
try:
    import joblib
except ImportError:
    sys.exit("[error] falta joblib/scikit-learn:  pip install scikit-learn")
try:
    from p3_camera import P3Camera, GainMode, raw_to_celsius
except ImportError:
    sys.exit("[error] no encuentro p3_camera. Instala el driver: pip install -e . en su repo.")


def detect(frame_c, k, t_abs, t_floor, sigma_floor, merge_px):
    h, w = frame_c.shape
    small = cv2.resize(frame_c, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bg_s = ndimage.median_filter(small, size=11, mode="nearest")
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    delta = frame_c - bg
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    sigma = max(1.4826 * mad, sigma_floor)
    mask = ((delta > k * sigma) | (frame_c > t_abs)) & (frame_c >= t_floor)
    if merge_px > 0 and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=merge_px)
    return mask, delta, sigma


def colorize(tc, scale, lo, hi):
    if hi - lo < 1:
        hi = lo + 1
    u8 = (np.clip((tc - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    h, w = color.shape[:2]
    return cv2.resize(color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def put(img, text, org, color=(255, 255, 255), sc=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, 1, cv2.LINE_AA)


class Track:
    """Foco seguido en el tiempo, con periodo de gracia, compuerta sostenida,
    acumulacion de evidencia (log-odds) e histeresis asimetrica."""
    _next_id = 0

    def __init__(self, cx, cy, flicker_hist):
        self.id = Track._next_id; Track._next_id += 1
        self.cx, self.cy = cx, cy
        self.logodds = 0.0
        self.state = "eval"          # eval | FUEGO | no
        self.missed = 0
        self.hits = 1                # detecciones consecutivas
        self.obs = 1                 # total de observaciones (periodo de gracia)
        self.flick_ok = deque(maxlen=flicker_hist)   # historial: flicker alto? (compuerta sostenida)
        self.last_flicker = 0.0
        self.last_T = 0.0
        self.area = 4

    def update(self, cx, cy, p, feats, cfg, flicker_ok):
        self.cx = int(0.6 * self.cx + 0.4 * cx)
        self.cy = int(0.6 * self.cy + 0.4 * cy)
        self.missed = 0
        self.hits += 1
        self.obs += 1
        self.last_flicker = feats.get("flicker_peak_C", 0.0)
        self.last_T = feats.get("T_peak_C", 0.0)
        self.area = feats.get("area_px", self.area)
        self.flick_ok.append(1 if flicker_ok else 0)

        # COMPUERTA SOSTENIDA: solo se acumula evidencia de fuego si el flicker
        # ha estado alto en la MAYORIA de los ultimos frames (no un pico suelto).
        sostenido = (len(self.flick_ok) >= cfg["flick_need"]
                     and sum(self.flick_ok) >= cfg["flick_need"])
        if not sostenido:
            p = min(p, 0.2)          # sin flicker sostenido, la evidencia de fuego se capa

        p = min(max(p, 1e-4), 1 - 1e-4)
        step = np.log(p / (1 - p)) * cfg["ev_gain"]
        self.logodds = np.clip(self.logodds * cfg["decay"] + step, -14, 14)

        # PERIODO DE GRACIA: no puede ser FUEGO hasta ser observado lo suficiente.
        grace_ok = self.obs >= cfg["min_obs"]

        # HISTERESIS ASIMETRICA: encender cuesta (on alto + gracia); mantener es facil.
        if self.state != "FUEGO":
            if grace_ok and sostenido and self.logodds >= cfg["on"]:
                self.state = "FUEGO"
            elif self.logodds <= cfg["off"]:
                self.state = "no"
        else:  # ya es FUEGO
            if self.logodds <= cfg["off"]:
                self.state = "no"

    def miss(self, cfg):
        self.missed += 1
        self.hits = 0
        self.flick_ok.append(0)
        self.logodds = np.clip(self.logodds * cfg["decay"], -14, 14)

    def confidence(self):
        return 1.0 / (1.0 + np.exp(-self.logodds))


def main():
    ap = argparse.ArgumentParser(description="Detector en vivo de produccion con veredicto robusto.")
    ap.add_argument("--model", default=str(HERE / "ignis_fire_classifier.joblib"))
    ap.add_argument("--gain", choices=["high", "low"], default="low")
    ap.add_argument("--flicker-window", type=int, default=10)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--t-abs", type=float, default=80.0)
    ap.add_argument("--t-floor", type=float, default=45.0)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--merge-px", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=2)
    ap.add_argument("--max-area", type=int, default=4000)
    ap.add_argument("--match-dist", type=float, default=25.0)
    ap.add_argument("--keep-missed", type=int, default=4)
    ap.add_argument("--min-hits", type=int, default=3, help="detecciones consecutivas para DIBUJAR")
    # --- endurecimiento anti-falso-verde (conservador) ---
    ap.add_argument("--min-obs", type=int, default=20,
                    help="frames de GRACIA: un foco debe observarse esto antes de poder ser fuego (~0.8s@25fps)")
    ap.add_argument("--flick-need", type=int, default=8,
                    help="frames recientes con flicker alto REQUERIDOS (compuerta sostenida, no un pico)")
    ap.add_argument("--flick-hist", type=int, default=12, help="ventana del historial de flicker sostenido")
    ap.add_argument("--flicker-gate", type=float, default=None,
                    help="flicker minimo por frame para contar como 'parpadea' (default: el del modelo)")
    ap.add_argument("--on", type=float, default=6.0, help="log-odds para ENCENDER (alto=conservador, sin falsos verdes)")
    ap.add_argument("--off", type=float, default=-1.5, help="log-odds para APAGAR")
    ap.add_argument("--ev-gain", type=float, default=0.6, help="evidencia por frame")
    ap.add_argument("--decay", type=float, default=0.92, help="inercia (1=nunca olvida)")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--display-lo", type=float, default=15.0)
    ap.add_argument("--display-hi", type=float, default=200.0)
    args = ap.parse_args()

    mpath = Path(args.model)
    if not mpath.exists():
        sys.exit(f"[error] no encuentro el modelo {mpath}")
    model = joblib.load(mpath)
    meta_path = mpath.with_suffix(".json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    features = meta.get("features", ["flicker_peak_C"])
    model_floor = float(meta.get("flicker_floor", 10.0))
    flicker_gate = args.flicker_gate if args.flicker_gate is not None else model_floor
    print(f"[info] modelo cargado. features={features}")
    print(f"[info] CONSERVADOR: gracia {args.min_obs}f  flicker sostenido {args.flick_need}/{args.flick_hist}f  "
          f"enciende en {args.on}  (nace desconfiado, sin falsos verdes)")

    cfg = dict(min_obs=args.min_obs, flick_need=args.flick_need, on=args.on, off=args.off,
               ev_gain=args.ev_gain, decay=args.decay)

    gain = GainMode.HIGH if args.gain == "high" else GainMode.LOW
    print("[info] conectando al P3...")
    cam = P3Camera()
    cam.connect(); cam.init(); cam.set_gain_mode(gain); cam.start_streaming()
    for _ in range(10):
        try: cam.read_frame_both()
        except Exception: pass
        time.sleep(0.05)
    try: cam.trigger_shutter()
    except Exception: pass

    win = "IgnisEdge - deteccion en vivo (produccion)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("[info] camara quieta = flicker fiable.  n=NUC  +/-=sensibilidad  q=salir")

    fbuf = deque(maxlen=args.flicker_window)
    tracks = []
    t_last = time.time(); fps = 0.0
    warmup = args.flicker_window     # frames de observacion antes de emitir veredictos

    try:
        while True:
            ir, raw = cam.read_frame_both()
            if raw is None:
                continue
            tc = raw_to_celsius(raw)
            fbuf.append(tc)
            flicker = np.stack(fbuf, axis=0).std(axis=0) if len(fbuf) >= 3 else np.zeros_like(tc)
            if warmup > 0:
                warmup -= 1

            mask, delta, sigma = detect(tc, args.k, args.t_abs, args.t_floor, args.sigma_floor, args.merge_px)
            labels, n = ndimage.label(mask)
            dets = []
            if n > 0:
                areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
                coms = ndimage.center_of_mass(np.ones_like(labels), labels, index=np.arange(1, n + 1))
                for bid in range(1, n + 1):
                    area = int(areas[bid - 1])
                    if area < args.min_area or area > args.max_area:
                        continue
                    blob = labels == bid
                    cy, cx = coms[bid - 1]
                    d_peak = float(delta[blob].max())
                    fk = float(flicker[blob].max())
                    feats = {
                        "flicker_peak_C": fk, "T_peak_C": float(tc[blob].max()),
                        "delta_peak_C": d_peak, "sigma_peak": d_peak / sigma, "area_px": float(area),
                    }
                    dets.append((int(round(cx)), int(round(cy)), feats, fk))

            if dets:
                Xf = np.array([[d[2].get(f, 0.0) for f in features] for d in dets], dtype=np.float64)
                probs = model.predict_proba(Xf)[:, 1]
                fkv = np.array([d[3] for d in dets])
                probs = np.where(fkv >= model_floor, probs, 0.0)
            else:
                probs = np.array([])

            # asociar y actualizar (durante warm-up NO se emiten veredictos: solo se observa)
            for tr in tracks:
                tr._matched = False
            for (cx, cy, feats, fk), p in zip(dets, probs):
                best, bd = None, args.match_dist
                for tr in tracks:
                    if getattr(tr, "_matched", False):
                        continue
                    dd = np.hypot(tr.cx - cx, tr.cy - cy)
                    if dd < bd:
                        best, bd = tr, dd
                if best is None:
                    best = Track(cx, cy, args.flick_hist); tracks.append(best)
                best._matched = True
                if warmup > 0:
                    # en warm-up solo se registra presencia, sin acumular veredicto
                    best.cx, best.cy = cx, cy; best.missed = 0; best.hits += 1; best.obs += 1
                    best.last_flicker = fk; best.last_T = feats["T_peak_C"]; best.area = feats["area_px"]
                else:
                    best.update(cx, cy, p, feats, cfg, fk >= flicker_gate)
            for tr in tracks:
                if not getattr(tr, "_matched", False):
                    tr.miss(cfg)
            tracks = [t for t in tracks if t.missed <= args.keep_missed]

            disp = colorize(tc, args.scale, args.display_lo, args.display_hi)
            n_fire = 0
            for tr in tracks:
                if tr.missed > 0 or tr.hits < args.min_hits:
                    continue
                x, y = tr.cx * args.scale, tr.cy * args.scale
                r = max(12, int(np.sqrt(tr.area) * args.scale))
                if tr.state == "FUEGO":
                    col = (0, 255, 0); tag = "FUEGO"; n_fire += 1
                elif tr.state == "eval":
                    col = (0, 200, 255); tag = "evaluando"
                else:
                    col = (255, 128, 0); tag = "no-fuego"
                cv2.rectangle(disp, (x - r, y - r), (x + r, y + r), col, 2)
                put(disp, f"{tag} {tr.confidence()*100:.0f}%", (x - r, y - r - 6), color=col, sc=0.5)
                put(disp, f"{tr.last_T:.0f}C flk{tr.last_flicker:.0f}", (x - r, y + r + 14), color=col, sc=0.42)

            now = time.time(); dt = now - t_last; t_last = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
            status = "OBSERVANDO..." if warmup > 0 else f"FUEGO: {n_fire}"
            put(disp, f"{status}   focos: {len(tracks)}   max {tc.max():.0f}C   {fps:.0f}fps", (10, 24))
            put(disp, "verde=FUEGO (evidencia sostenida)  amarillo=evaluando  azul=no   n=NUC  q=salir",
                (10, disp.shape[0] - 12), color=(0, 255, 255), sc=0.45)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("n"):
                try:
                    cam.trigger_shutter(); fbuf.clear(); tracks.clear()
                    warmup = args.flicker_window
                    print("[nuc] recalibrado + warm-up reiniciado")
                except Exception: pass
            elif k in (ord("+"), ord("=")):
                cfg["on"] = max(1.0, cfg["on"] - 1.0); print(f"[sensibilidad] enciende en {cfg['on']} (mas sensible)")
            elif k == ord("-"):
                cfg["on"] += 1.0; print(f"[sensibilidad] enciende en {cfg['on']} (mas conservador)")
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop_streaming(); cam.disconnect(); cv2.destroyAllWindows()
    print("[ok] cerrado")


if __name__ == "__main__":
    main()
