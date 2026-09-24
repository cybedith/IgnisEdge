#!/usr/bin/env python3
"""
test_vision_engine.py -- Pruebas Unitarias del Motor de Visión Optimizado.
========================================================================

Verifica:
  1. Carga correcta del modelo y metadatos en VisionEngine.
  2. Detección y extracción rápida de blobs.
  3. Comportamiento en simulación de 35 frames:
     - Warm-up inicial
     - Identificación de foco caliente con flicker
     - Declaración de FUEGO tras período de gracia e histéresis
  4. Reset por comando NUC.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.vision.ignis_vision import VisionEngine, detect_fast, extract_blobs_fast


class TestVisionEngine(unittest.TestCase):
    def setUp(self):
        self.model_path = REPO_ROOT / "scripts" / "ignis_fire_classifier.joblib"
        self.assertTrue(self.model_path.exists(), f"Modelo no encontrado en {self.model_path}")
        self.engine = VisionEngine(
            model_path=self.model_path,
            flicker_window=5,
            min_obs=10,
            flick_need=5,
            flick_hist=8,
            on_thresh=5.0,
            t_floor=40.0,
        )

    def test_01_engine_initialization(self):
        """Verifica que el modelo y features se carguen correctamente."""
        self.assertEqual(self.engine.features, ["flicker_peak_C"])
        self.assertEqual(self.engine.model_floor, 10.0)
        self.assertEqual(self.engine.warmup, 5)

    def test_02_detect_and_blob_extraction(self):
        """Verifica la detección de un foco térmico sintético."""
        frame = np.full((192, 256), 22.0, dtype=np.float32)
        frame[90:98, 120:128] = 175.0  # Foco caliente a 175°C
        flicker = np.zeros((192, 256), dtype=np.float32)
        flicker[90:98, 120:128] = 40.0

        mask, delta, sigma = detect_fast(frame, t_floor=40.0)
        self.assertTrue(mask[94, 124], "El centro del foco caliente DEBE estar activo en la máscara")

        blobs = extract_blobs_fast(mask, delta, sigma, frame, flicker)
        self.assertEqual(len(blobs), 1, "Debe extraerse exactamente 1 foco caliente")
        self.assertAlmostEqual(blobs[0].features["T_peak_C"], 175.0, places=1)
        self.assertAlmostEqual(blobs[0].flicker_peak, 40.0, places=1)

    def test_03_lifecycle_simulation(self):
        """Simula 25 frames con un fuego que parpadea y verifica la transición a FUEGO."""
        np.random.seed(42)
        h, w = 192, 256

        for frame_idx in range(25):
            # Fondo base con leve ruido térmico
            frame = 22.0 + np.random.normal(0, 0.3, (h, w)).astype(np.float32)

            # Inyectar llama oscilante (flicker real: 120°C a 210°C)
            temp_osc = 160.0 + 50.0 * np.sin(frame_idx * 1.2)
            frame[80:88, 120:128] = temp_osc

            res = self.engine.process_frame(frame)

            if frame_idx < 4:
                self.assertEqual(res.status, "OBSERVANDO", "Durante warm-up debe estar OBSERVANDO")

        # Al frame 25, con evidencia sostenida, el foco debe estar confirmado como FUEGO
        fire_tracks = [t for t in res.tracks if t.state == "FUEGO"]
        self.assertGreaterEqual(len(fire_tracks), 1, "Debe haberse declarado al menos 1 foco en estado FUEGO")
        self.assertGreater(fire_tracks[0].confidence, 0.85)

    def test_04_nuc_reset(self):
        """Verifica que el reset de NUC limpie los tracks y reinicie el warm-up."""
        frame = np.full((192, 256), 25.0, dtype=np.float32)
        self.engine.process_frame(frame)
        self.engine.reset_warmup()
        self.assertEqual(self.engine.warmup, self.engine.flicker_window)
        self.assertEqual(len(self.engine.tracks), 0)


if __name__ == "__main__":
    unittest.main()
