"""
Test P3 + IgnisEdge model inference end-to-end.
Captures 5 frames, runs the v1 model on each, saves QA visualizations.
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
from ultralytics import YOLO

MODEL_PATH = Path.home() / "IgnisEdge/models/production/ignisedge_thermal_v1_best.pt"
OUT_DIR = Path.home() / "IgnisEdge/qa_p3_inference"
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES = 5


def visualize_with_detection(celsius, processed, detections, stats, title):
    """Side-by-side: raw thermal | YOLO input | model detections overlay."""
    h, w = celsius.shape

    # Panel 1: thermal with inferno colormap
    raw_norm = np.clip(
        (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
        0, 255,
    ).astype(np.uint8)
    raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)

    # Panel 2: YOLO input (what the model sees)
    yolo_input_bgr = cv2.cvtColor(processed[:, :, 0], cv2.COLOR_GRAY2BGR)

    # Panel 3: detections drawn on raw thermal
    detection_overlay = raw_colored.copy()
    n_det = 0
    if detections is not None and len(detections) > 0:
        result = detections[0]
        if result.masks is not None:
            for i, mask_obj in enumerate(result.masks.data):
                mask = mask_obj.cpu().numpy()
                # Resize mask to image size
                mask = cv2.resize(mask, (w, h)) > 0.5
                # Draw cyan overlay
                overlay = np.zeros_like(detection_overlay)
                overlay[mask] = [255, 255, 0]  # BGR cyan
                detection_overlay = cv2.addWeighted(
                    detection_overlay, 1.0, overlay, 0.4, 0
                )
                conf = float(result.boxes.conf[i]) if result.boxes is not None else 0.0
                n_det += 1
            # Draw bounding boxes
            if result.boxes is not None:
                for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cv2.rectangle(detection_overlay, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(detection_overlay, f"fire {conf:.2f}",
                                (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    combined = np.hstack([raw_colored, yolo_input_bgr, detection_overlay])

    labels = ["P3 RAW THERMAL", "YOLO INPUT (zero-masked)", f"DETECTIONS ({n_det})"]
    for i, label in enumerate(labels):
        cv2.putText(combined, label, (i * w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    footer = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    line1 = f"{title}  |  range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  mean={stats['mean_c']:.1f}C  connected={stats['connected_pixels']}px"
    line2 = f"detections={n_det}  model={MODEL_PATH.name}"
    cv2.putText(footer, line1, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([combined, footer])


def main():
    print("=" * 70)
    print("IgnisEdge - P3 + Model Inference Test")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"\nLoading model: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    print(f"  Classes: {model.names}")

    camera = P3Camera()

    try:
        print("\n[1/4] Connecting...")
        camera.connect()

        print("[2/4] Initializing...")
        camera.init()
        time.sleep(0.5)

        print("[3/4] Setting LOW GAIN...")
        camera.set_gain_mode(GainMode.LOW)
        time.sleep(0.5)

        camera.start_streaming()
        print("      Streaming.")

        print("[4/4] Stabilizing...")
        for _ in range(5):
            camera.read_frame_both()
            time.sleep(0.04)

        print(f"\nCapturing {N_FRAMES} frames + running inference...\n")
        for frame_num in range(1, N_FRAMES + 1):
            ir_brightness, thermal_raw = camera.read_frame_both()
            celsius = raw_to_celsius(thermal_raw)

            stats = compute_stats(celsius)
            processed = preprocess_radiometric(celsius)

            # Run inference
            detections = model.predict(
                processed,
                conf=0.25,
                iou=0.45,
                verbose=False,
            )

            n_det = 0
            if detections and detections[0].boxes is not None:
                n_det = len(detections[0].boxes)

            ts = time.strftime("%H%M%S")
            title = f"frame{frame_num}_{ts}"
            viz = visualize_with_detection(celsius, processed, detections, stats, title)

            out_path = OUT_DIR / f"{title}.png"
            cv2.imwrite(str(out_path), viz)

            print(f"  Frame {frame_num}: max={stats['max_c']:.1f}C  "
                  f"connected={stats['connected_pixels']}px  "
                  f"detections={n_det}  saved={out_path.name}")

            if frame_num < N_FRAMES:
                time.sleep(0.8)

    finally:
        print("\nCleaning up...")
        try:
            camera.stop_streaming()
        except Exception as e:
            print(f"  stop_streaming: {e}")
        try:
            camera.disconnect()
        except Exception as e:
            print(f"  disconnect: {e}")

    print(f"\nDone. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
