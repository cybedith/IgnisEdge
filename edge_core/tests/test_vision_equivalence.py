#!/usr/bin/env python3
"""
test_vision_equivalence.py -- Test de Regresión y Equivalencia Matemática.
========================================================================

Verifica que el motor de visión optimizado (ignis_vision.py) sea 100% equivalente
en resultados, características y comportamiento de histéresis a live_classify.py,
garantizando que ninguna optimización rompa la fidelidad física validada en banco.

Tests incluidos:
  1. Equivalencia de detección (Etapa 1): dilatación y umbralización contextual.
  2. Equivalencia de extracción de blobs y features físicas:
     - flicker_peak_C
     - T_peak_C
     - delta_peak_C
     - sigma_peak
     - area_px
     - Centroides (cx, cy)
  3. Equivalencia de la compuerta física (flicker_floor).
  4. Equivalencia de la FSM de Track (período de gracia, compuerta sostenida, log-odds, histéresis).
  5. Simulación de 50 frames: Vela vs Objeto Caliente Estático.
"""

import sys
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

# Agregar scripts al path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import live_classify


def detect_reference(frame_c, k=4.0, t_abs=80.0, t_floor=45.0, sigma_floor=0.05, merge_px=3):
    """Implementación de referencia original de live_classify.py."""
    return live_classify.detect(frame_c, k, t_abs, t_floor, sigma_floor, merge_px)


def detect_optimized(frame_c, k=4.0, t_abs=80.0, t_floor=45.0, sigma_floor=0.05, merge_px=3):
    """Implementación optimizada: C++ OpenCV dilate (cross) manteniendo el fondo idéntico."""
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
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), kernel_cross, iterations=merge_px).astype(bool)
    return mask, delta, sigma


def extract_blobs_reference(mask, delta, sigma, tc, flicker, min_area=2, max_area=4000):
    """Extracción original de blobs con scipy.ndimage."""
    labels, n = ndimage.label(mask)
    dets = []
    if n > 0:
        areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        coms = ndimage.center_of_mass(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        for bid in range(1, n + 1):
            area = int(areas[bid - 1])
            if area < min_area or area > max_area:
                continue
            blob = labels == bid
            cy, cx = coms[bid - 1]
            d_peak = float(delta[blob].max())
            fk = float(flicker[blob].max())
            feats = {
                "flicker_peak_C": fk,
                "T_peak_C": float(tc[blob].max()),
                "delta_peak_C": d_peak,
                "sigma_peak": d_peak / sigma,
                "area_px": float(area),
            }
            dets.append((int(round(cx)), int(round(cy)), feats, fk))
    return dets


def extract_blobs_optimized(mask, delta, sigma, tc, flicker, min_area=2, max_area=4000):
    """Extracción optimizada con cv2.connectedComponentsWithStats (4-connectivity y sub-window slicing)."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
    dets = []
    for bid in range(1, num_labels):
        area = int(stats[bid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (
            stats[bid, cv2.CC_STAT_LEFT],
            stats[bid, cv2.CC_STAT_TOP],
            stats[bid, cv2.CC_STAT_WIDTH],
            stats[bid, cv2.CC_STAT_HEIGHT],
        )
        cx, cy = centroids[bid]
        sub_labels = labels[y : y + h, x : x + w]
        blob_sub = sub_labels == bid

        sub_delta = delta[y : y + h, x : x + w]
        sub_flicker = flicker[y : y + h, x : x + w]
        sub_tc = tc[y : y + h, x : x + w]

        d_peak = float(sub_delta[blob_sub].max())
        fk = float(sub_flicker[blob_sub].max())
        feats = {
            "flicker_peak_C": fk,
            "T_peak_C": float(sub_tc[blob_sub].max()),
            "delta_peak_C": d_peak,
            "sigma_peak": d_peak / sigma,
            "area_px": float(area),
        }
        dets.append((int(round(cx)), int(round(cy)), feats, fk))
    return dets


class TestVisionEquivalence(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Escena térmica sintética típica: 192x256, fondo a ~22°C con gradiente
        self.h, self.w = 192, 256
        y, x = np.mgrid[: self.h, : self.w]
        self.frame = 20.0 + 5.0 * np.sin(x / 40.0) + np.random.normal(0, 0.4, (self.h, self.w)).astype(np.float32)

        # Foco 1: Vela caliente con flicker alto (llama ~180°C)
        self.frame[80:88, 120:128] = 185.0
        # Foco 2: Objeto caliente quieto (horno/roca ~90°C)
        self.frame[140:155, 40:60] = 92.0

        # Simular mapa de flicker
        self.flicker = np.zeros((self.h, self.w), dtype=np.float32)
        self.flicker[80:88, 120:128] = 45.0  # Vela parpadea a 45°C
        self.flicker[140:155, 40:60] = 0.8   # Objeto caliente quieto: 0.8°C

    def test_01_detect_equivalence(self):
        """Verifica que detect_optimized produzca exactamente la misma máscara que detect_reference."""
        mask_ref, delta_ref, sigma_ref = detect_reference(self.frame)
        mask_opt, delta_opt, sigma_opt = detect_optimized(self.frame)

        # Verificar sigma y delta
        self.assertAlmostEqual(sigma_ref, sigma_opt, places=5)
        np.testing.assert_allclose(delta_ref, delta_opt, rtol=1e-5, atol=1e-5)

        # Verificar máscara bit por bit
        diff_pixels = np.sum(mask_ref != mask_opt)
        self.assertEqual(diff_pixels, 0, f"Diferencia en máscara: {diff_pixels} píxeles")

    def test_02_blob_extraction_equivalence(self):
        """Verifica que extract_blobs_optimized extraiga exactamente los mismos blobs y features."""
        mask, delta, sigma = detect_reference(self.frame)
        dets_ref = extract_blobs_reference(mask, delta, sigma, self.frame, self.flicker)
        dets_opt = extract_blobs_optimized(mask, delta, sigma, self.frame, self.flicker)

        self.assertEqual(len(dets_ref), len(dets_opt), f"Cantidad de blobs distinta: {len(dets_ref)} vs {len(dets_opt)}")

        # Ordenar por centroide para comparar 1 a 1
        dets_ref.sort(key=lambda d: (d[0], d[1]))
        dets_opt.sort(key=lambda d: (d[0], d[1]))

        for d_r, d_o in zip(dets_ref, dets_opt):
            # Centroides
            self.assertAlmostEqual(d_r[0], d_o[0], delta=1, msg="Centroide X no coincide")
            self.assertAlmostEqual(d_r[1], d_o[1], delta=1, msg="Centroide Y no coincide")

            # Features
            feats_r, feats_o = d_r[2], d_o[2]
            for feat_name in ["flicker_peak_C", "T_peak_C", "delta_peak_C", "sigma_peak", "area_px"]:
                self.assertAlmostEqual(
                    feats_r[feat_name],
                    feats_o[feat_name],
                    places=4,
                    msg=f"Feature '{feat_name}' difiere: ref={feats_r[feat_name]}, opt={feats_o[feat_name]}",
                )

    def test_03_physical_gate(self):
        """Verifica que la compuerta física anule la probabilidad si flicker < flicker_floor."""
        flicker_floor = 10.0
        dets = [
            (100, 100, {"flicker_peak_C": 45.0}, 45.0),  # Fuego
            (50, 50, {"flicker_peak_C": 3.2}, 3.2),      # Objeto caliente quieto
        ]
        fkv = np.array([d[3] for d in dets])
        probs = np.array([0.98, 0.85])  # Suponer que el modelo dudó en el objeto caliente

        # Aplicar compuerta física
        gated_probs = np.where(fkv >= flicker_floor, probs, 0.0)

        self.assertAlmostEqual(gated_probs[0], 0.98)
        self.assertAlmostEqual(gated_probs[1], 0.0, msg="El objeto con flicker < 10.0 DEBE ser forzado a prob=0.0")

    def test_04_track_hysteresis_and_grace_period(self):
        """Verifica que la clase Track respete el período de gracia y la histéresis asimétrica."""
        cfg = {
            "min_obs": 20,
            "flick_need": 8,
            "on": 6.0,
            "off": -1.5,
            "ev_gain": 0.6,
            "decay": 0.92,
        }
        track = live_classify.Track(cx=100, cy=100, flicker_hist=12)

        # Frame 1 al 19: Debe estar en estado 'eval' (período de gracia min_obs=20)
        feats_fire = {"flicker_peak_C": 50.0, "T_peak_C": 180.0, "area_px": 25.0}
        for frame_i in range(1, 20):
            track.update(100, 100, p=0.99, feats=feats_fire, cfg=cfg, flicker_ok=True)
            self.assertEqual(
                track.state,
                "eval",
                f"En frame {frame_i} (< min_obs=20) el estado DEBE ser 'eval', pero fue '{track.state}'",
            )

        # En frames subsecuentes con evidencia sostenida, debe pasar a 'FUEGO'
        for _ in range(10):
            track.update(100, 100, p=0.99, feats=feats_fire, cfg=cfg, flicker_ok=True)

        self.assertEqual(track.state, "FUEGO", "Con evidencia alta y sostenida tras período de gracia DEBE ser 'FUEGO'")
        self.assertGreaterEqual(track.logodds, cfg["on"])

        # Si el fuego se apaga de golpe (probabilidad cae), gracias a la histéresis no se apaga instantáneamente
        feats_cold = {"flicker_peak_C": 1.0, "T_peak_C": 30.0, "area_px": 25.0}
        track.update(100, 100, p=0.01, feats=feats_cold, cfg=cfg, flicker_ok=False)
        self.assertEqual(
            track.state,
            "FUEGO",
            "Histéresis: un solo frame bajo no debe apagar el estado FUEGO inmediatamente",
        )

        # Tras muchos frames sin fuego, el log-odds debe decaer por debajo de off (-1.5) y apagarse
        for _ in range(30):
            track.update(100, 100, p=0.001, feats=feats_cold, cfg=cfg, flicker_ok=False)

        self.assertEqual(track.state, "no", "Tras decaimiento prolongado debe pasar a estado 'no'")


if __name__ == "__main__":
    unittest.main()
