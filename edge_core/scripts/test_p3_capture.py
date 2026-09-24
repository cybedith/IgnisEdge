"""
Test 1: Capture single frame from P3, apply preprocessing, save QA image.
NO model inference yet - just verifying the pipeline end-to-end.

Output: ~/IgnisEdge/qa_p3_capture/frame_TIMESTAMP_qa.png
  Side-by-side: [raw thermal colored | hysteresis mask overlay | YOLO input]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    preprocess_radiometric,
    hysteresis_mask,
    compute_stats,
    SEED_TEMP_C,
    EXTEND_TEMP_C,
)

from p3_camera import P3Camera, raw_to_celsius, GainMode

OUT_DIR = Path.home() / "IgnisEdge/qa_p3_capture"
OUT_DIR.mkdir(exist_ok=True)


def visualize(celsius, processed, stats, title):
    """Side-by-side QA: raw | mask | YOLO input."""
    h, w = celsius.shape

    # Panel 1: thermal with inferno colormap (full dynamic range)
    raw_norm = np.clip(
        (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
        0, 255,
    ).astype(np.uint8)
    raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)

    # Panel 2: hysteresis mask in red over raw
    mask = hysteresis_mask(celsius)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask] = [0, 0, 255]
    mask_overlay = cv2.addWeighted(raw_colored, 0.5, mask_rgb, 0.5, 0)

    # Panel 3: YOLO input
    processed_bgr = cv2.cvtColor(processed[:, :, 0], cv2.COLOR_GRAY2BGR)

    combined = np.hstack([raw_colored, mask_overlay, processed_bgr])

    labels = [
        "P3 RAW THERMAL",
        f"HYST MASK ({EXTEND_TEMP_C:.0f}-{SEED_TEMP_C:.0f}C)",
        "YOLO INPUT",
    ]
    for i, label in enumerate(labels):
        cv2.putText(combined, label, (i * w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    footer = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    line1 = f"{title}  |  shape={celsius.shape}  range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  mean={stats['mean_c']:.1f}C"
    line2 = f"seed_px(>={SEED_TEMP_C:.0f}C)={stats['seed_pixels']}  extend_px(>={EXTEND_TEMP_C:.0f}C)={stats['extend_pixels']}  connected={stats['connected_pixels']}  rejected={stats['rejected_pixels']}"
    cv2.putText(footer, line1, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([combined, footer])


def main():
    print("=" * 70)
    print("IgnisEdge - P3 Capture Pipeline Test")
    print("=" * 70)

    print("\n[1/4] Connecting to P3 camera...")
    camera = P3Camera()
    camera.connect()
    print("      Connected.")

    print("\n[2/4] Initializing camera...")
    camera.init()
    time.sleep(0.5)

    print("\n[3/4] Setting LOW GAIN (range 0-550C, required for fire detection)...")
    camera.set_gain_mode(GainMode.LOW)
    time.sleep(0.5)

    camera.start_streaming()
    print("      Streaming started.")

    # Discard first 5 frames (camera stabilization)
    print("\n[4/4] Stabilizing (discarding first 5 frames)...")
    for i in range(5):
        camera.read_frame_both()
        time.sleep(0.04)

    # Capture 3 frames spaced 1 second apart
    print("\nCapturing 3 frames (1s apart)...\n")
    for frame_num in range(1, 4):
        ir_brightness, thermal_raw = camera.read_frame_both()
        celsius = raw_to_celsius(thermal_raw)

        stats = compute_stats(celsius)
        processed = preprocess_radiometric(celsius)

        ts = time.strftime("%H%M%S")
        title = f"frame{frame_num}_{ts}"
        viz = visualize(celsius, processed, stats, title)

        out_path = OUT_DIR / f"{title}_qa.png"
        cv2.imwrite(str(out_path), viz)

        print(f"  Frame {frame_num}: range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  "
              f"connected={stats['connected_pixels']}px  "
              f"saved={out_path.name}")

        if frame_num < 3:
            time.sleep(1.0)

    print("\nStopping stream...")
    camera.stop_streaming()
    camera.disconnect()

    print(f"\nDone. QA images saved to: {OUT_DIR}")
    print("\nTo view:")
    print(f"  xdg-open {OUT_DIR}/$(ls -t {OUT_DIR} | head -1)")


if __name__ == "__main__":
    main()
