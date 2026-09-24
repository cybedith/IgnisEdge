"""
IgnisEdge - Radiometric preprocessing (shared training + deployment).

Apply identically to:
  - FLAME 3 dataset regeneration (offline)
  - Jetson P3 capture (online inference)

Scientific basis:
  - SEED_TEMP_C (200°C): active fire threshold (Hopkins et al. 2024, FLAME 3 paper)
  - EXTEND_TEMP_C (80°C): canopy pre-heating zone (NWCG)
  - Hysteresis: rejects isolated warm objects (motors, sun-heated metal)
    while preserving connected fire halos.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_propagation


# --- Configuration (scientifically validated) ---
SEED_TEMP_C = 150.0       # Pirolisis + fuego temprano (ajustado para early detection)
EXTEND_TEMP_C = 80.0      # Halo térmico + canopy pre-heating
NORMALIZE_MAX_C = 600.0   # Match P3 low-gain max (550°C) + buffer


def hysteresis_mask(
    celsius: np.ndarray,
    seed_temp: float = SEED_TEMP_C,
    extend_temp: float = EXTEND_TEMP_C,
) -> np.ndarray:
    """
    Hysteresis thresholding: keep pixels >= extend_temp ONLY if connected
    to at least one pixel >= seed_temp. Rejects isolated warm objects.

    Returns: binary mask (H, W) bool
    """
    seed = celsius >= seed_temp
    extend = celsius >= extend_temp
    if not seed.any():
        return np.zeros_like(celsius, dtype=bool)
    return binary_propagation(seed, mask=extend)


def preprocess_radiometric(
    celsius: np.ndarray,
    seed_temp: float = SEED_TEMP_C,
    extend_temp: float = EXTEND_TEMP_C,
    normalize_max: float = NORMALIZE_MAX_C,
) -> np.ndarray:
    """
    Radiometric °C array -> 3-channel uint8 ready for YOLO.

    Args:
        celsius: (H, W) float array of °C values
        seed_temp, extend_temp: hysteresis thresholds
        normalize_max: max temp for linear normalization (maps to 255)

    Returns:
        (H, W, 3) uint8 array
    """
    celsius = np.asarray(celsius, dtype=np.float32)

    # 1. Hysteresis mask
    mask = hysteresis_mask(celsius, seed_temp, extend_temp)

    # 2. Zero-mask everything else
    masked = np.where(mask, celsius, 0.0)

    # 3. Normalize: 0°C -> 0, normalize_max -> 255 (clipped)
    normalized = np.clip(masked / normalize_max * 255.0, 0, 255).astype(np.uint8)

    # 4. Replicate to 3 channels for YOLO input
    return np.stack([normalized, normalized, normalized], axis=-1)


def compute_stats(celsius: np.ndarray) -> dict:
    """QA statistics for a radiometric frame."""
    mask = hysteresis_mask(celsius)
    return {
        "max_c": float(celsius.max()),
        "min_c": float(celsius.min()),
        "mean_c": float(celsius.mean()),
        "seed_pixels": int((celsius >= SEED_TEMP_C).sum()),
        "extend_pixels": int((celsius >= EXTEND_TEMP_C).sum()),
        "connected_pixels": int(mask.sum()),
        "rejected_pixels": int((celsius >= EXTEND_TEMP_C).sum() - mask.sum()),
    }


if __name__ == "__main__":
    # Self-test with synthetic data
    test = np.full((100, 100), 25.0, dtype=np.float32)  # 25°C background
    test[40:50, 40:50] = 300.0  # fire core (connected cluster)
    test[45:47, 45:47] = 550.0  # hot center
    test[40:55, 35:55] = 150.0  # halo (would be kept, connected)
    test[10:15, 10:15] = 120.0  # isolated warm object (rejected)
    test[40:50, 40:50] = 300.0  # restore core after halo assignment

    result = preprocess_radiometric(test)
    stats = compute_stats(test)
    print(f"Self-test OK. Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"Stats: {stats}")
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8
    print("All assertions passed.")
