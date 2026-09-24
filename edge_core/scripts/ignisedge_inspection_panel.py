#!/usr/bin/env python3
"""
IgnisEdge — Panel de Inspección Industrial (v1.0)
==================================================

Visualización 2×2 en tiempo real de todas las etapas del pipeline híbrido de
detección de incendios forestales. Pensado para validación en campo: cada panel
muestra una transformación distinta para poder auditar inline dónde falla una
detección (¿mala histéresis? ¿mala normalización radiométrica? ¿desacuerdo
morph/YOLO?).

Layout
------
    ┌──────────────────────┬──────────────────────┐
    │ 1. Raw Thermal (cmap)│ 2. Hysteresis Mask   │
    ├──────────────────────┼──────────────────────┤
    │ 3. YOLO Input        │ 4. Detections + Ver. │
    ├──────────────────────┴──────────────────────┤
    │          Status bar (FPS, temps, conteos)   │
    └─────────────────────────────────────────────┘

Controles
---------
    q / ESC     Salir
    p           Pausar / reanudar (congela el frame mostrado)
    s           Screenshot completo (PNG timestamped)
    d           Dump de arrays crudos en dumps/ (thermal_raw, celsius, mask, ...)
    r           Reiniciar contadores y estado (FPS, frame_count, errores)
    c           Ciclar colormap del panel térmico
    + / =       Zoom in (paso 0.1, máx 2.0)
    - / _       Zoom out (paso 0.1, mín 0.5)

Uso
---
    python ignisedge_inspection_panel.py

Árbitro híbrido (lógica resumida)
---------------------------------
    RED    ← morph.WILDFIRE/RED    ∧  YOLO.wildfire
    ORANGE ← morph.WILDFIRE        ∧  YOLO.wildfire  (consenso)
    ORANGE ← YOLO.wildfire(conf≥0.85) ∧ ¬morph.POINT_SOURCE/EXTENDED_HOT
             (escalado neuronal: YOLO muy seguro, morph no veta)
    ORANGE ← morph.WILDFIRE/RED    ∧  ¬YOLO.wildfire
    YELLOW ← morph.WILDFIRE        xor YOLO.wildfire  (evidencia unilateral)
    WHITE  ← desacuerdo explícito (morph.WILDFIRE ∧ YOLO.false_positive)
    NONE   ← ninguna evidencia
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label as ndi_label

# ---- Imports del proyecto --------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from p3_camera import P3Camera, raw_to_celsius, GainMode
from preprocess import (
    preprocess_radiometric,
    hysteresis_mask,
    SEED_TEMP_C,
    EXTEND_TEMP_C,
    NORMALIZE_MAX_C,
)
from morphological_classifier import (
    detect_and_classify,
    ThreatClass,
    AlertLevel,
    MIN_REGION_AREA_PX,
)
from ultralytics import YOLO


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_PATH = Path.home() / "IgnisEdge/models/thermal_v3_s/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Umbral de escalado neuronal en el árbitro:
# si YOLO detecta wildfire con confianza > este valor y morph no veta con una
# clase artificial explícita, YELLOW se escala a ORANGE. Cierra el "gap de
# régimen medio" (fuegos 400-5000 px que el clasificador morfológico v4 no
# tipifica por falta de halo amplio o multi-foco, pero que YOLO reconoce
# con alta confianza gracias al entrenamiento en FLAME 3).
YOLO_HIGH_CONF_ESCALATION = 0.85

# Tamaño base de cada panel (antes de zoom).
PANEL_W = 512
PANEL_H = 384
HEADER_H = 28
STATUS_H = 110
BORDER = 2
BG = (18, 18, 18)           # Fondo oscuro (BGR)
BORDER_COL = (60, 60, 60)

WINDOW_NAME = "IgnisEdge — Inspection Panel"

DUMP_DIR = Path.cwd() / "dumps"
SHOT_DIR = Path.cwd() / "screenshots"

# Ciclo de colormaps para Panel 1.
COLORMAPS = [
    ("inferno", cv2.COLORMAP_INFERNO),
    ("magma",   cv2.COLORMAP_MAGMA),
    ("plasma",  cv2.COLORMAP_PLASMA),
    ("hot",     cv2.COLORMAP_HOT),
    ("jet",     cv2.COLORMAP_JET),
    ("viridis", cv2.COLORMAP_VIRIDIS),
]

# ---- Paleta de colores (BGR) ----------------------------------------------
COL_WILDFIRE  = (0, 0, 255)      # Rojo
COL_EXTENDED  = (0, 255, 255)    # Amarillo
COL_POINT     = (255, 0, 255)    # Magenta
COL_AMBIGUOUS = (255, 255, 0)    # Cian
COL_NOISE     = (128, 128, 128)  # Gris
COL_YOLO_FIRE = (0, 255, 0)      # Verde
COL_YOLO_FP   = (255, 80, 80)    # Azul claro

THREAT_COLOR = {
    ThreatClass.WILDFIRE:     COL_WILDFIRE,
    ThreatClass.EXTENDED_HOT: COL_EXTENDED,
    ThreatClass.POINT_SOURCE: COL_POINT,
    ThreatClass.AMBIGUOUS:    COL_AMBIGUOUS,
    ThreatClass.NOISE:        COL_NOISE,
}

VERDICT_COLOR = {
    AlertLevel.RED:    (0, 0, 255),
    AlertLevel.ORANGE: (0, 128, 255),
    AlertLevel.YELLOW: (0, 255, 255),
    AlertLevel.WHITE:  (230, 230, 230),
    AlertLevel.NONE:   (90, 90, 90),
}


# ============================================================================
# ESTADO DEL PANEL
# ============================================================================

@dataclass
class PanelState:
    paused: bool = False
    colormap_idx: int = 0
    zoom: float = 1.0
    frame_count: int = 0
    fps: float = 0.0
    t_last: float = field(default_factory=time.time)
    last_error: str = ""
    error_count: int = 0
    camera_status: str = "init"
    gain_mode: str = "?"
    # Caché del último frame procesado (para repintar en pausa / screenshots).
    cache: dict = field(default_factory=dict)


# ============================================================================
# FUNCIONES DE DIBUJO — una por etapa
# ============================================================================

def _put(image, text, xy, scale=0.45, color=(230, 230, 230), thick=1):
    """Helper de texto con antialias consistente."""
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _panel_canvas(title: str) -> np.ndarray:
    """Canvas vacío de un panel con su cabecera."""
    canvas = np.full((PANEL_H + HEADER_H, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_W - 1, HEADER_H - 1),
                  (34, 34, 34), -1)
    _put(canvas, title, (8, 19), scale=0.5,
         color=(220, 220, 220), thick=1)
    return canvas


def _fit_into(dst_canvas: np.ndarray, content: np.ndarray) -> None:
    """Inserta `content` (BGR) redimensionado al área útil de `dst_canvas`."""
    resized = cv2.resize(content, (PANEL_W, PANEL_H),
                         interpolation=cv2.INTER_NEAREST)
    dst_canvas[HEADER_H:HEADER_H + PANEL_H, 0:PANEL_W] = resized


def raw_to_colormap(celsius: np.ndarray, colormap_id: int,
                    vmax_cap: float = 600.0) -> tuple[np.ndarray, float, float]:
    """
    Convierte datos radiométricos en °C a imagen BGR con falso color.

    Normaliza con un rango dinámico acotado (vmax_cap hace clamp a 600 °C para
    evitar que un píxel muy caliente colapse toda la paleta). Devuelve también
    los límites usados para poder anotarlos en pantalla.
    """
    vmin = float(np.min(celsius))
    vmax = float(min(vmax_cap, max(vmin + 1.0, np.max(celsius))))
    norm = np.clip((celsius - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap_id)
    return colored, vmin, vmax


def generate_mask_overlay(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Máscara binaria → BGR blanco/negro + conteos (píxeles totales, componentes).
    """
    mask_bool = mask.astype(bool)
    u8 = mask_bool.astype(np.uint8) * 255
    bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    total_px = int(mask_bool.sum())
    if total_px == 0:
        return bgr, 0, 0
    _, n_components = ndi_label(mask_bool)
    return bgr, total_px, int(n_components)


def yolo_input_to_bgr(yolo_input: np.ndarray) -> np.ndarray:
    """
    El preprocesador produce (H,W,3) uint8 con los 3 canales replicados
    (zero-masked + normalizado). Tomamos el canal 0 y lo expandimos a BGR.
    """
    if yolo_input.ndim == 3:
        gray = yolo_input[:, :, 0]
    else:
        gray = yolo_input
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_detections(base_bgr: np.ndarray,
                    morph_detections,
                    yolo_results,
                    verdict: AlertLevel,
                    verdict_reason: str,
                    yolo_names: dict) -> np.ndarray:
    """
    Dibuja sobre una copia de `base_bgr`:
      - Cajas del clasificador morfológico (color por ThreatClass).
      - Cajas de YOLO (verde/azul según clase).
      - Banner de veredicto (grande, esquina inferior-izquierda).
    """
    img = base_bgr.copy()
    h, w = img.shape[:2]

    # --- Cajas del clasificador morfológico -------------------------------
    for det in morph_detections:
        x, y, bw, bh = det.features.bbox
        color = THREAT_COLOR.get(det.threat_class, COL_NOISE)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        label = (f"{det.threat_class.value.upper()} "
                 f"A={det.features.area_px} "
                 f"T={det.features.max_temp_c:.0f}C "
                 f"[{det.alert_level.value}]")
        # Texto con fondo para legibilidad, clampeado al ancho del frame.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ty = max(th + 3, y - 4)
        tx0 = max(0, min(x, w - tw - 6))
        cv2.rectangle(img, (tx0, ty - th - 3), (tx0 + tw + 4, ty + 2),
                      (0, 0, 0), -1)
        _put(img, label, (tx0 + 2, ty - 1), scale=0.35, color=color, thick=1)

    # --- Cajas YOLO -------------------------------------------------------
    if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
        r = yolo_results[0]
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            color = COL_YOLO_FIRE if cls_id == 0 else COL_YOLO_FP
            # Dibujamos las cajas YOLO con línea discontinua "simulada"
            # (doble rectángulo) para distinguirlas visualmente del morph.
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)
            label = f"YOLO:{yolo_names.get(cls_id, cls_id)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            by = min(h - 2, y2 + th + 4)
            cv2.rectangle(img, (x1, by - th - 3), (x1 + tw + 4, by + 2),
                          (0, 0, 0), -1)
            _put(img, label, (x1 + 2, by - 1), scale=0.38,
                 color=color, thick=1)

    # --- Banner de veredicto ---------------------------------------------
    vcolor = VERDICT_COLOR.get(verdict, (120, 120, 120))
    vtext = f"VERDICT: {verdict.value.upper()}"
    # Escala adaptativa: si el texto excede el ancho, lo reducimos.
    v_scale = 0.9
    (tw, th), _ = cv2.getTextSize(vtext, cv2.FONT_HERSHEY_DUPLEX, v_scale, 2)
    while tw + 20 > w and v_scale > 0.4:
        v_scale -= 0.1
        (tw, th), _ = cv2.getTextSize(vtext, cv2.FONT_HERSHEY_DUPLEX, v_scale, 2)
    pad = 8
    bx1 = 6
    by2 = h - 8
    by1 = by2 - th - 2 * pad
    bx2 = min(w - 6, bx1 + tw + 2 * pad)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), vcolor, 2)
    cv2.putText(img, vtext, (bx1 + pad, by2 - pad - 2),
                cv2.FONT_HERSHEY_DUPLEX, v_scale, vcolor, 2, cv2.LINE_AA)
    # Razón debajo del banner (truncada al ancho del frame).
    if verdict_reason:
        reason_y = by1 - 4
        if reason_y > 12:
            max_chars = max(20, (w - 8) // 6)
            _put(img, verdict_reason[:max_chars], (bx1 + 2, reason_y),
                 scale=0.35, color=(200, 200, 200), thick=1)

    return img


# ============================================================================
# ÁRBITRO HÍBRIDO
# ============================================================================

def _morph_has_artificial_veto(morph_detections) -> bool:
    """
    Veto morfológico para el escalado por confianza YOLO.

    Devuelve True si el clasificador morfológico *explícitamente* etiquetó
    alguna región del frame como fuente artificial (POINT_SOURCE) o caliente
    extendida industrial (EXTENDED_HOT). En ese caso bloqueamos el escalado
    YELLOW→ORANGE por confianza YOLO, porque hay evidencia morfológica
    directa de que algo en el frame es artificial.

    NOTA: AMBIGUOUS no veta — por definición significa "no pude clasificar",
    no "no es fuego". NOISE tampoco veta (área <30 px, ruido del sensor).
    El veto es conservador: sólo clases que explícitamente dicen "artificial".
    """
    return any(
        d.threat_class in (ThreatClass.POINT_SOURCE, ThreatClass.EXTENDED_HOT)
        for d in morph_detections
    )


def arbitrate(morph_detections,
              yolo_results) -> tuple[AlertLevel, str]:
    """
    Fusiona evidencia morfológica + YOLO en un único veredicto.

    Lógica de consenso:
      - Consenso positivo fuerte:  morph.WILDFIRE(RED|ORANGE) ∧ yolo.wildfire
        → RED si morph es RED, si no ORANGE.
      - Desacuerdo explícito:      morph.WILDFIRE ∧ yolo.false_positive
                                   (sin yolo.wildfire)
        → WHITE (ambiguo, no escalar ni descartar).
      - Morph-only (sin YOLO):     morph.WILDFIRE ∧ ¬yolo
        → ORANGE si morph=RED, si no YELLOW.
      - YOLO-only (sin morph):     yolo.wildfire ∧ ¬morph.WILDFIRE
        → ORANGE si conf ≥ YOLO_HIGH_CONF_ESCALATION ∧ no hay veto artificial
          en morph (i.e. morph no vio POINT_SOURCE / EXTENDED_HOT).
        → YELLOW en caso contrario.
      - Nada:                      NONE.

    El escalado YOLO-only por confianza existe para cerrar el "régimen medio":
    fuegos de 400–5000 px que no disparan Rule 0.5 (incipient, tope en 400 px)
    ni Rule 4/5 (established, requieren halo grande o multi-foco). YOLO sí los
    reconoce semánticamente; si su confianza es alta y morph no contradice,
    aceptamos la evidencia neuronal como suficiente para ORANGE.
    """
    morph_wildfires = [
        d for d in morph_detections if d.threat_class == ThreatClass.WILDFIRE
    ]
    morph_red = any(d.alert_level == AlertLevel.RED for d in morph_wildfires)
    morph_orange = any(d.alert_level == AlertLevel.ORANGE
                       for d in morph_wildfires)

    yolo_fire = 0
    yolo_fp = 0
    max_yolo_fire_conf = 0.0
    if (yolo_results and len(yolo_results) > 0
            and yolo_results[0].boxes is not None):
        for i, cls in enumerate(yolo_results[0].boxes.cls):
            cid = int(cls.item())
            conf = float(yolo_results[0].boxes.conf[i].item())
            if cid == 0:
                yolo_fire += 1
                max_yolo_fire_conf = max(max_yolo_fire_conf, conf)
            else:
                yolo_fp += 1

    has_morph_fire = len(morph_wildfires) > 0
    has_yolo_fire = yolo_fire > 0
    has_yolo_fp = yolo_fp > 0

    # Consenso positivo
    if has_morph_fire and has_yolo_fire:
        if morph_red:
            return AlertLevel.RED, (
                f"Consensus: morph RED + YOLO wildfire (conf={max_yolo_fire_conf:.2f})"
            )
        if morph_orange:
            return AlertLevel.ORANGE, (
                f"Consensus: morph ORANGE + YOLO wildfire (conf={max_yolo_fire_conf:.2f})"
            )
        return AlertLevel.ORANGE, (
            f"Consensus: morph wildfire + YOLO wildfire (conf={max_yolo_fire_conf:.2f})"
        )

    # Desacuerdo explícito morph-YOLO
    if has_morph_fire and has_yolo_fp and not has_yolo_fire:
        return AlertLevel.WHITE, (
            "Disagreement: morph says WILDFIRE, YOLO says false_positive"
        )

    # Una sola vía detecta
    if has_morph_fire and not has_yolo_fire:
        # Si morph dice RED sin YOLO, subimos a ORANGE (no RED sin consenso).
        lvl = AlertLevel.ORANGE if morph_red else AlertLevel.YELLOW
        return lvl, "Morph-only evidence (YOLO silent)"
    if has_yolo_fire and not has_morph_fire:
        # --- Escalado por confianza YOLO (opción B) -----------------------
        # Si YOLO está muy seguro y morph no marcó nada artificial en el
        # frame, subimos YELLOW→ORANGE. AMBIGUOUS y NOISE no cuentan como
        # veto (son "no sé", no "no es fuego").
        if (max_yolo_fire_conf >= YOLO_HIGH_CONF_ESCALATION
                and not _morph_has_artificial_veto(morph_detections)):
            return AlertLevel.ORANGE, (
                f"YOLO high-conf escalation (conf={max_yolo_fire_conf:.2f} "
                f">= {YOLO_HIGH_CONF_ESCALATION:.2f}), no morph veto"
            )
        # Si hay veto de morph, lo decimos explícitamente en la razón.
        if _morph_has_artificial_veto(morph_detections):
            return AlertLevel.YELLOW, (
                f"YOLO-only (conf={max_yolo_fire_conf:.2f}) but morph "
                f"saw artificial source in frame - no escalation"
            )
        return AlertLevel.YELLOW, (
            f"YOLO-only evidence (morph silent, conf={max_yolo_fire_conf:.2f} "
            f"< {YOLO_HIGH_CONF_ESCALATION:.2f})"
        )

    return AlertLevel.NONE, "No evidence"


# ============================================================================
# COMPOSICIÓN DEL GRID
# ============================================================================

def compose_grid(panels: list[np.ndarray],
                 status_lines: list[str]) -> np.ndarray:
    """
    Ensambla los 4 paneles en una rejilla 2×2 + barra de estado.
    `panels` debe tener exactamente 4 imágenes de igual tamaño.
    """
    assert len(panels) == 4, "Se esperan 4 paneles"
    ph, pw = panels[0].shape[:2]

    grid_h = ph * 2 + BORDER * 3 + STATUS_H
    grid_w = pw * 2 + BORDER * 3
    out = np.full((grid_h, grid_w, 3), BG, dtype=np.uint8)

    # Coloca paneles.
    positions = [
        (BORDER, BORDER),
        (BORDER, BORDER * 2 + pw),
        (BORDER * 2 + ph, BORDER),
        (BORDER * 2 + ph, BORDER * 2 + pw),
    ]
    for (y0, x0), panel in zip(positions, panels):
        out[y0:y0 + ph, x0:x0 + pw] = panel

    # Líneas separadoras.
    cv2.line(out, (0, ph + BORDER), (grid_w, ph + BORDER), BORDER_COL, 1)
    cv2.line(out, (pw + BORDER, 0), (pw + BORDER, ph * 2 + BORDER * 2),
             BORDER_COL, 1)

    # Barra de estado.
    status_y0 = ph * 2 + BORDER * 3
    cv2.rectangle(out, (0, status_y0),
                  (grid_w, status_y0 + STATUS_H - 1),
                  (28, 28, 28), -1)
    cv2.line(out, (0, status_y0), (grid_w, status_y0), BORDER_COL, 1)

    # Texto multi-línea en el status bar.
    line_h = 20
    y = status_y0 + line_h
    for line in status_lines[:5]:    # Máx 5 líneas en el status bar
        _put(out, line, (12, y), scale=0.5,
             color=(220, 220, 220), thick=1)
        y += line_h

    return out


def apply_zoom(image: np.ndarray, zoom: float) -> np.ndarray:
    """Reescala la composición final según el zoom actual."""
    if abs(zoom - 1.0) < 1e-3:
        return image
    new_w = int(image.shape[1] * zoom)
    new_h = int(image.shape[0] * zoom)
    interp = cv2.INTER_LINEAR if zoom > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


# ============================================================================
# PERSISTENCIA (screenshots / dumps)
# ============================================================================

def save_screenshot(shot_dir: Path, composed: np.ndarray) -> Path:
    shot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = shot_dir / f"ignisedge_panel_{ts}.png"
    cv2.imwrite(str(path), composed)
    return path


def save_dump(dump_dir: Path,
              thermal_raw: np.ndarray,
              celsius: np.ndarray,
              mask: np.ndarray,
              yolo_input: np.ndarray,
              morph_detections,
              verdict: AlertLevel,
              verdict_reason: str) -> Path:
    """
    Guarda todos los arrays crudos del frame actual en una subcarpeta
    timestamped. Permite re-analizar offline con los mismos datos que vio
    el pipeline en vivo.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    sub = dump_dir / f"dump_{ts}"
    sub.mkdir(parents=True, exist_ok=True)

    np.save(sub / "thermal_raw.npy", thermal_raw)
    np.save(sub / "celsius.npy", celsius)
    np.save(sub / "mask.npy", mask.astype(np.uint8))
    np.save(sub / "yolo_input.npy", yolo_input)

    # Resumen humano-legible.
    with (sub / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Timestamp: {ts}\n")
        fh.write(f"Verdict: {verdict.value}\n")
        fh.write(f"Reason: {verdict_reason}\n")
        fh.write(f"Max temp: {float(np.max(celsius)):.2f} C\n")
        fh.write(f"Min temp: {float(np.min(celsius)):.2f} C\n")
        fh.write(f"Mask pixels: {int(mask.sum())}\n")
        fh.write(f"\n--- Morphological detections ({len(morph_detections)}) ---\n")
        for d in morph_detections:
            f = d.features
            fh.write(
                f"  [{d.threat_class.value}/{d.alert_level.value}] "
                f"bbox={f.bbox} area={f.area_px} "
                f"max_t={f.max_temp_c:.1f} mean_t={f.mean_temp_c:.1f} "
                f"grad={f.temp_gradient_c:.1f} circ={f.circularity:.2f} "
                f"halo_ext={f.halo_extent:.2f} n_sub={f.n_subregions}\n"
                f"    reason: {d.reasoning}\n"
            )
    return sub


# ============================================================================
# CONEXIÓN DE CÁMARA CON REINTENTOS
# ============================================================================

def _connect_camera(max_retries: int = 3) -> Optional[P3Camera]:
    """Intenta abrir la cámara con backoff corto."""
    for attempt in range(1, max_retries + 1):
        try:
            cam = P3Camera()
            cam.connect()
            cam.init()
            time.sleep(0.5)
            cam.set_gain_mode(GainMode.LOW)
            cam.start_streaming()
            # Warmup
            for _ in range(5):
                cam.read_frame_both()
                time.sleep(0.04)
            return cam
        except Exception as e:
            print(f"[cam] intento {attempt}/{max_retries} falló: {e}")
            time.sleep(0.8 * attempt)
    return None


# ============================================================================
# LOOP PRINCIPAL
# ============================================================================

def handle_key(key: int, state: PanelState) -> bool:
    """
    Procesa una tecla y actualiza el estado. Devuelve False si hay que salir.
    """
    if key in (ord('q'), 27):  # ESC
        return False
    if key == ord('p'):
        state.paused = not state.paused
        print(f"[ui] paused={state.paused}")
    elif key == ord('c'):
        state.colormap_idx = (state.colormap_idx + 1) % len(COLORMAPS)
        print(f"[ui] colormap = {COLORMAPS[state.colormap_idx][0]}")
    elif key in (ord('+'), ord('=')):
        state.zoom = min(2.0, round(state.zoom + 0.1, 2))
        print(f"[ui] zoom = {state.zoom}")
    elif key in (ord('-'), ord('_')):
        state.zoom = max(0.5, round(state.zoom - 0.1, 2))
        print(f"[ui] zoom = {state.zoom}")
    elif key == ord('r'):
        state.frame_count = 0
        state.fps = 0.0
        state.last_error = ""
        state.error_count = 0
        state.t_last = time.time()
        print("[ui] state reset")
    # 's' y 'd' se procesan en el loop porque requieren datos del frame.
    return True


def main() -> int:
    print("=" * 70)
    print("IgnisEdge — Inspection Panel v1.0")
    print("=" * 70)
    print(f"Modelo  : {MODEL_PATH}")
    print(f"Conf    : {CONF_THRESHOLD} | IoU: {IOU_THRESHOLD}")
    print(f"Hyst    : seed={SEED_TEMP_C}°C, extend={EXTEND_TEMP_C}°C, "
          f"norm_max={NORMALIZE_MAX_C}°C")
    print(f"Min área: {MIN_REGION_AREA_PX} px")
    print()
    print("Controles: q=salir  p=pausa  s=screenshot  d=dump  "
          "r=reset  c=colormap  +/-=zoom")
    print()

    # --- Carga de modelo y cámara ----------------------------------------
    try:
        model = YOLO(str(MODEL_PATH))
        yolo_names = dict(model.names)
    except Exception as e:
        print(f"[fatal] no pude cargar YOLO: {e}")
        return 1

    cam = _connect_camera()
    if cam is None:
        print("[fatal] no pude conectar la cámara tras varios intentos")
        return 1

    state = PanelState()
    state.camera_status = "streaming"
    state.gain_mode = "LOW"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,
                     PANEL_W * 2 + BORDER * 3,
                     (PANEL_H + HEADER_H) * 2 + BORDER * 3 + STATUS_H)

    last_composed: Optional[np.ndarray] = None

    try:
        while True:
            loop_t0 = time.time()

            # ---- Captura de frame (o reutilizar caché si pausa) ---------
            if not state.paused or not state.cache:
                try:
                    _, thermal_raw = cam.read_frame_both()
                    celsius = raw_to_celsius(thermal_raw)
                    state.camera_status = "streaming"
                except Exception as e:
                    state.error_count += 1
                    state.last_error = f"camera: {e}"
                    state.camera_status = "ERROR"
                    print(f"[cam] lectura falló: {e}")
                    # Intentar recuperar cámara.
                    try:
                        cam.stop_streaming()
                        cam.disconnect()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    new_cam = _connect_camera(max_retries=2)
                    if new_cam is not None:
                        cam = new_cam
                        state.camera_status = "recovered"
                    # Mostrar último frame conocido y seguir.
                    if last_composed is not None:
                        cv2.imshow(WINDOW_NAME, apply_zoom(last_composed,
                                                          state.zoom))
                    key = cv2.waitKey(200) & 0xFF
                    if key != 255 and not handle_key(key, state):
                        break
                    continue

                # ---- Pipeline completo ---------------------------------
                try:
                    mask = hysteresis_mask(celsius).astype(bool)
                    morph_detections = detect_and_classify(celsius)
                    yolo_input = preprocess_radiometric(celsius)
                    yolo_results = model.predict(
                        yolo_input, conf=CONF_THRESHOLD,
                        iou=IOU_THRESHOLD, verbose=False
                    )
                    verdict, verdict_reason = arbitrate(morph_detections,
                                                        yolo_results)
                except Exception as e:
                    state.error_count += 1
                    state.last_error = f"pipeline: {e}"
                    print(f"[pipe] error: {e}")
                    traceback.print_exc()
                    key = cv2.waitKey(50) & 0xFF
                    if key != 255 and not handle_key(key, state):
                        break
                    continue

                # Cachear para repintar en pausa / dumps / screenshots.
                state.cache = dict(
                    thermal_raw=thermal_raw,
                    celsius=celsius,
                    mask=mask,
                    yolo_input=yolo_input,
                    morph_detections=morph_detections,
                    yolo_results=yolo_results,
                    verdict=verdict,
                    verdict_reason=verdict_reason,
                )
            else:
                # En pausa: usamos caché.
                celsius = state.cache["celsius"]
                mask = state.cache["mask"]
                yolo_input = state.cache["yolo_input"]
                morph_detections = state.cache["morph_detections"]
                yolo_results = state.cache["yolo_results"]
                verdict = state.cache["verdict"]
                verdict_reason = state.cache["verdict_reason"]

            # ---- Construcción de los 4 paneles -------------------------
            cmap_name, cmap_id = COLORMAPS[state.colormap_idx]
            thermal_bgr, vmin, vmax = raw_to_colormap(celsius, cmap_id)
            mask_bgr, mask_px, mask_components = generate_mask_overlay(mask)
            yolo_in_bgr = yolo_input_to_bgr(yolo_input)
            detections_bgr = draw_detections(
                yolo_in_bgr, morph_detections, yolo_results,
                verdict, verdict_reason, yolo_names
            )

            # Panel 1 — Raw thermal
            p1 = _panel_canvas(
                f"1. Raw Thermal  [{cmap_name}]  "
                f"min={vmin:.1f}C  max={vmax:.1f}C"
            )
            _fit_into(p1, thermal_bgr)

            # Panel 2 — Hysteresis mask
            p2 = _panel_canvas(
                f"2. Hysteresis Mask  "
                f"seed={SEED_TEMP_C:.0f}C extend={EXTEND_TEMP_C:.0f}C  "
                f"px={mask_px}  comps={mask_components}"
            )
            _fit_into(p2, mask_bgr)

            # Panel 3 — YOLO input
            p3 = _panel_canvas(
                f"3. YOLO Input  zero-masked + norm->{NORMALIZE_MAX_C:.0f}C"
            )
            _fit_into(p3, yolo_in_bgr)

            # Panel 4 — Detections + verdict
            n_wf = sum(1 for d in morph_detections
                       if d.threat_class == ThreatClass.WILDFIRE)
            n_yolo_f = n_yolo_fp = 0
            if (yolo_results and len(yolo_results) > 0
                    and yolo_results[0].boxes is not None):
                for cls in yolo_results[0].boxes.cls:
                    if int(cls) == 0:
                        n_yolo_f += 1
                    else:
                        n_yolo_fp += 1
            p4 = _panel_canvas(
                f"4. Detections  morph={len(morph_detections)}"
                f"(WF:{n_wf})  yolo=fire:{n_yolo_f}/fp:{n_yolo_fp}"
            )
            _fit_into(p4, detections_bgr)

            # ---- FPS ---------------------------------------------------
            now = time.time()
            dt = now - state.t_last
            state.t_last = now
            if dt > 0:
                # Exponential moving average para suavizar.
                inst_fps = 1.0 / dt
                state.fps = (0.85 * state.fps + 0.15 * inst_fps
                             if state.frame_count > 3 else inst_fps)
            state.frame_count += 1

            # ---- Status bar --------------------------------------------
            max_t = float(np.max(celsius))
            min_t = float(np.min(celsius))
            status_lines = [
                (f"FPS: {state.fps:5.1f}  |  frame #{state.frame_count}  "
                 f"|  {'PAUSED' if state.paused else 'LIVE'}  "
                 f"|  zoom x{state.zoom:.1f}"),
                (f"Temp:  min={min_t:6.1f}C   max={max_t:6.1f}C   "
                 f"mask_px={mask_px}   components={mask_components}"),
                (f"Morph: total={len(morph_detections)}  "
                 f"WILDFIRE={n_wf}  "
                 f"|  YOLO: wildfire={n_yolo_f}  false_positive={n_yolo_fp}"),
                (f"Verdict: {verdict.value.upper()}  -  {verdict_reason[:80]}"),
                (f"Camera: {state.camera_status}  gain={state.gain_mode}  "
                 f"errors={state.error_count}  "
                 f"last_err={state.last_error[:40] if state.last_error else '-'}"),
            ]

            # ---- Composición final -------------------------------------
            composed = compose_grid([p1, p2, p3, p4], status_lines)
            last_composed = composed

            # Pinta banner "PAUSED" sobreimpreso si aplica.
            display = apply_zoom(composed, state.zoom)
            if state.paused:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], 28),
                              (0, 0, 140), -1)
                display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
                _put(display,
                     "PAUSED - press 'p' to resume, 'd' to dump, 's' to snap",
                     (12, 19), scale=0.55, color=(255, 255, 255), thick=1)

            cv2.imshow(WINDOW_NAME, display)

            # ---- Teclado -----------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == ord('s'):
                try:
                    p = save_screenshot(SHOT_DIR, composed)
                    print(f"[io] screenshot → {p}")
                except Exception as e:
                    state.last_error = f"screenshot: {e}"
                    print(f"[io] screenshot falló: {e}")
                continue

            if key == ord('d'):
                if state.cache:
                    try:
                        p = save_dump(
                            DUMP_DIR,
                            state.cache["thermal_raw"],
                            state.cache["celsius"],
                            state.cache["mask"],
                            state.cache["yolo_input"],
                            state.cache["morph_detections"],
                            state.cache["verdict"],
                            state.cache["verdict_reason"],
                        )
                        print(f"[io] dump → {p}")
                    except Exception as e:
                        state.last_error = f"dump: {e}"
                        print(f"[io] dump falló: {e}")
                continue

            if not handle_key(key, state):
                break

            # Evitar busy-loop si el frame fue muy rápido.
            elapsed = time.time() - loop_t0
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

    except KeyboardInterrupt:
        print("\n[main] interrupción por teclado")
    finally:
        try:
            cam.stop_streaming()
            cam.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"[main] salida limpia  |  frames={state.frame_count}  "
              f"errors={state.error_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())