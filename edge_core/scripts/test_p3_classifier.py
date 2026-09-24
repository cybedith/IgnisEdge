"""
P3 + Morphological Classifier end-to-end test.
Captures frames, classifies thermal regions, saves QA visualizations
with alert level and reasoning per region.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from morphological_classifier import (
    detect_and_classify,
    AlertLevel,
    ThreatClass,
    SEED_TEMP_C,
    EXTEND_TEMP_C,
)

from p3_camera import P3Camera, raw_to_celsius, GainMode

OUT_DIR = Path.home() / "IgnisEdge/qa_p3_classifier"
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES = 5

# Color per alert level (BGR)
ALERT_COLORS = {
    AlertLevel.RED: (0, 0, 255),
    AlertLevel.ORANGE: (0, 140, 255),
    AlertLevel.YELLOW: (0, 255, 255),
    AlertLevel.WHITE: (255, 255, 255),
    AlertLevel.NONE: (128, 128, 128),
}

ALERT_LABEL = {
    AlertLevel.RED: "ROJA",
    AlertLevel.ORANGE: "NARANJA",
    AlertLevel.YELLOW: "AMARILLA",
    AlertLevel.WHITE: "REVISAR",
    AlertLevel.NONE: "DESCARTE",
}


def visualize(celsius, detections, title):
    """Side-by-side: raw thermal | classifications overlay | text report."""
    h, w = celsius.shape

    # Panel 1: thermal inferno
    raw_norm = np.clip(
        (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
        0, 255,
    ).astype(np.uint8)
    raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)

    # Panel 2: detections drawn
    overlay = raw_colored.copy()
    for det in detections:
        x, y, bw, bh = det.features.bbox
        color = ALERT_COLORS[det.alert_level]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        label = f"{det.threat_class.value} [{ALERT_LABEL[det.alert_level]}]"
        cv2.putText(overlay, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Panel 3: text report
    report = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(report, "REPORTE", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if not detections:
        cv2.putText(report, "Sin regiones", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        y_pos = 40
        for i, det in enumerate(detections):
            color = ALERT_COLORS[det.alert_level]
            cv2.putText(report, f"#{i+1} {det.threat_class.value}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += 14
            cv2.putText(report, f"  alert: {ALERT_LABEL[det.alert_level]}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 12
            cv2.putText(report, f"  area={det.features.area_px}px",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            y_pos += 11
            cv2.putText(report, f"  max={det.features.max_temp_c:.0f}C grad={det.features.temp_gradient_c:.0f}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            y_pos += 11
            cv2.putText(report, f"  circ={det.features.circularity:.2f} halo={det.features.halo_extent:.2f}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            y_pos += 11
            cv2.putText(report, f"  asp={det.features.aspect_ratio:.1f} sub={det.features.n_subregions}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            y_pos += 18
            if y_pos > h - 20:
                break

    combined = np.hstack([raw_colored, overlay, report])

    labels_top = ["P3 RAW THERMAL", "CLASSIFICATION", "REPORT"]
    for i, label in enumerate(labels_top):
        cv2.putText(combined, label, (i * w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    footer = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    summary = f"{title}  |  range=[{celsius.min():.1f}, {celsius.max():.1f}]C  regions={len(detections)}"
    cv2.putText(footer, summary, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return np.vstack([combined, footer])


def main():
    print("=" * 70)
    print("IgnisEdge - P3 + Morphological Classifier Test")
    print("=" * 70)

    camera = P3Camera()

    try:
        print("\nConnecting...")
        camera.connect()
        camera.init()
        time.sleep(0.5)
        camera.set_gain_mode(GainMode.LOW)
        time.sleep(0.5)
        camera.start_streaming()

        print("Stabilizing...")
        for _ in range(5):
            camera.read_frame_both()
            time.sleep(0.04)

        print(f"\nCapturing {N_FRAMES} frames + classifying...\n")
        for frame_num in range(1, N_FRAMES + 1):
            _, thermal_raw = camera.read_frame_both()
            celsius = raw_to_celsius(thermal_raw)

            detections = detect_and_classify(celsius)

            ts = time.strftime("%H%M%S")
            title = f"frame{frame_num}_{ts}"
            viz = visualize(celsius, detections, title)
            out_path = OUT_DIR / f"{title}.png"
            cv2.imwrite(str(out_path), viz)

            # Print summary
            summary = f"  Frame {frame_num}: max={celsius.max():.1f}C  regions={len(detections)}"
            for det in detections:
                summary += f"\n    -> {det.threat_class.value} [{ALERT_LABEL[det.alert_level]}]"
                summary += f"  ({det.reasoning})"
            print(summary)
            print(f"    saved={out_path.name}")

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

    print(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
