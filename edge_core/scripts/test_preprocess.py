"""
Sanity check: process 3 FLAME 3 thermal TIFFs through the preprocessing
pipeline and save side-by-side visualizations for manual inspection.

Output: ~/IgnisEdge/qa_preprocess/*.png
  - *_qa.png: [raw thermal | hysteresis mask overlay | YOLO input] side by side
  - *_yolo_input.png: pure grayscale output (exactly what YOLO will see)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

# Import preprocess module from same directory
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    preprocess_radiometric,
    compute_stats,
    hysteresis_mask,
    SEED_TEMP_C,
    EXTEND_TEMP_C,
)

FLAME3_BASE = Path.home() / "IgnisEdge/datasets/flame3"
OUT_DIR = Path.home() / "IgnisEdge/qa_preprocess"
OUT_DIR.mkdir(exist_ok=True)

# 3 test cases spanning burn intensity
TEST_FILES = [
    ("plot 1/duringburn/raw_thermal_tiff/IRX_0529.TIFF", "IRX_0529_early"),
    ("plot 1/duringburn/raw_thermal_tiff/IRX_0700.TIFF", "IRX_0700_peak"),
    ("plot 1/duringburn/raw_thermal_tiff/IRX_0800.TIFF", "IRX_0800_late"),
]


def dn_to_celsius(dn):
    """FLAME 3 raw DN -> Celsius. Formula from paper: celsius = DN * 0.1 - 275"""
    return dn.astype(np.float32) * 0.1 - 275.0


def visualize(celsius, processed, stats, title):
    """Side-by-side: raw thermal | hysteresis mask | YOLO input."""
    h, w = celsius.shape

    # Panel 1: raw thermal with inferno colormap (full range)
    raw_norm = np.clip(
        (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
        0, 255
    ).astype(np.uint8)
    raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)

    # Panel 2: hysteresis mask as red overlay on raw
    mask = hysteresis_mask(celsius)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask] = [0, 0, 255]  # BGR red
    mask_overlay = cv2.addWeighted(raw_colored, 0.5, mask_rgb, 0.5, 0)

    # Panel 3: YOLO input (grayscale from channel 0)
    processed_gray = processed[:, :, 0]
    processed_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

    # Stack horizontally
    combined = np.hstack([raw_colored, mask_overlay, processed_bgr])

    # Labels
    labels = ["RAW THERMAL", f"MASK (seed>{SEED_TEMP_C:.0f}C ext>{EXTEND_TEMP_C:.0f}C)", "YOLO INPUT"]
    for i, label in enumerate(labels):
        x = i * w + 10
        cv2.putText(combined, label, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Stats footer
    footer = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    line1 = f"{title}  |  max={stats['max_c']:.0f}C  mean={stats['mean_c']:.1f}C"
    line2 = f"seed px={stats['seed_pixels']}  extend px={stats['extend_pixels']}  connected={stats['connected_pixels']}  rejected={stats['rejected_pixels']}"
    cv2.putText(footer, line1, (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([combined, footer])


def main():
    print("=" * 70)
    print("IgnisEdge Preprocess Sanity Check")
    print("=" * 70)
    print(f"FLAME 3 base: {FLAME3_BASE}")
    print(f"Output dir:   {OUT_DIR}")
    print(f"Thresholds:   seed={SEED_TEMP_C}°C  extend={EXTEND_TEMP_C}°C")
    print()

    processed_count = 0
    for rel_path, out_name in TEST_FILES:
        tiff_path = FLAME3_BASE / rel_path
        if not tiff_path.exists():
            print(f"SKIP: {tiff_path} not found")
            continue

        print(f"Processing {rel_path}")

        # Read radiometric TIFF
        with rasterio.open(tiff_path) as src:
            dn = src.read(1)

        celsius = dn_to_celsius(dn)
        stats = compute_stats(celsius)

        print(f"  Shape:          {celsius.shape}")
        print(f"  Temp range:     [{stats['min_c']:.1f}, {stats['max_c']:.1f}] °C")
        print(f"  Seed pixels:    {stats['seed_pixels']}")
        print(f"  Extend pixels:  {stats['extend_pixels']}")
        print(f"  Connected:      {stats['connected_pixels']}")
        print(f"  Rejected:       {stats['rejected_pixels']} (isolated warm objects)")

        # Apply preprocessing
        processed = preprocess_radiometric(celsius)

        # Save QA visualization
        viz = visualize(celsius, processed, stats, out_name)
        qa_path = OUT_DIR / f"{out_name}_qa.png"
        cv2.imwrite(str(qa_path), viz)

        # Save pure YOLO input
        yolo_path = OUT_DIR / f"{out_name}_yolo_input.png"
        cv2.imwrite(str(yolo_path), processed[:, :, 0])

        print(f"  Saved: {qa_path.name}")
        print(f"  Saved: {yolo_path.name}")
        print()
        processed_count += 1

    print("=" * 70)
    print(f"Done. Processed {processed_count}/{len(TEST_FILES)} files")
    print(f"Review QA images at: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
