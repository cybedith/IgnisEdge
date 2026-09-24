"""
test_dataset_recorder.py -- Tests unitarios para el grabador selectivo de datasets.
"""

import json
import shutil
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge_core.vision.dataset_recorder import DatasetRecorder


@dataclass
class MockTrack:
    id: int = 1
    cx: float = 120.0
    cy: float = 90.0
    state: str = "eval"
    confidence: float = 0.85
    T_peak_C: float = 95.0
    flicker_peak: float = 7.5
    area_px: float = 12.0


@dataclass
class MockVisionResult:
    timestamp: float = time.time()
    status: str = "ACTIVO"
    num_fires: int = 0
    max_temp_c: float = 25.0
    fps: float = 16.5
    tracks: list = None

    def __post_init__(self):
        if self.tracks is None:
            self.tracks = []


class TestDatasetRecorder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recorder = DatasetRecorder(
            output_dir=self.temp_dir,
            min_free_disk_gb=0.01,
            max_fps=10.0,
            save_npy=True,
            save_jpg=True,
        )

    def tearDown(self):
        self.recorder.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_recording_on_normal_frame(self):
        """No debe grabar si la temperatura es normal y no hay tracks ni fuegos."""
        frame = np.full((192, 256), 22.0, dtype=np.float32)
        res = MockVisionResult(num_fires=0, max_temp_c=25.0, tracks=[])

        recorded = self.recorder.maybe_record(frame, res)
        self.assertFalse(recorded)

        # Esperar y verificar que no haya archivos
        time.sleep(0.3)
        files = list(Path(self.recorder.session_dir).glob("*"))
        self.assertEqual(len(files), 0)

    def test_record_on_fire(self):
        """Debe grabar cuando num_fires > 0."""
        frame = np.full((192, 256), 22.0, dtype=np.float32)
        frame[90, 120] = 125.0
        track = MockTrack(state="FUEGO", confidence=0.98, T_peak_C=125.0)
        res = MockVisionResult(num_fires=1, max_temp_c=125.0, tracks=[track])

        recorded = self.recorder.maybe_record(frame, res)
        self.assertTrue(recorded)

        # Esperar a que el worker guarde en disco
        time.sleep(0.5)

        npy_files = list(Path(self.recorder.session_dir).glob("*.npy"))
        jpg_files = list(Path(self.recorder.session_dir).glob("*.jpg"))
        json_files = list(Path(self.recorder.session_dir).glob("*.json"))

        self.assertEqual(len(npy_files), 1)
        self.assertEqual(len(jpg_files), 1)
        self.assertEqual(len(json_files), 1)

        # Verificar contenido del npy
        loaded_npy = np.load(str(npy_files[0]))
        self.assertEqual(loaded_npy.shape, (192, 256))
        self.assertAlmostEqual(float(loaded_npy.max()), 125.0, places=1)

        # Verificar contenido del JSON
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["label"], "FUEGO")
        self.assertEqual(data["num_fires"], 1)
        self.assertEqual(len(data["tracks"]), 1)
        self.assertEqual(data["tracks"][0]["state"], "FUEGO")

    def test_record_on_eval_candidate(self):
        """Debe grabar cuando hay un track en estado 'eval' (posible fuego)."""
        frame = np.full((192, 256), 24.0, dtype=np.float32)
        frame[80, 100] = 85.0
        track = MockTrack(state="eval", confidence=0.75, T_peak_C=85.0)
        res = MockVisionResult(num_fires=0, max_temp_c=85.0, tracks=[track])

        recorded = self.recorder.maybe_record(frame, res)
        self.assertTrue(recorded)

        time.sleep(0.5)
        json_files = list(Path(self.recorder.session_dir).glob("*.json"))
        self.assertEqual(len(json_files), 1)
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["label"], "EVAL")


if __name__ == "__main__":
    unittest.main()
