#!/usr/bin/env python3
"""
IgnisEdge - Auto-Labeling Pipeline v3 (Production Final)
==========================================================
Generates YOLOv8-seg labels from FLAME 3 radiometric thermal TIFFs.

Key improvements over v2 (from internal stress testing):
  1. Uses BOTH plot 1 (296) AND plot 2 (522) = ~818 thermal images (3x more data)
  2. Hysteresis thresholding for active_fire: 200°C seed, 120°C extension
     (captures gradual fire boundaries, scientifically more accurate)
  3. Visual QA output for human verification of label quality
  4. Negative samples from early cold frames (auto-detected)
  5. Deployment-aware: documents P3 resolution gap (640x512 train → 512x384 deploy)
  6. Class imbalance monitoring with warnings
  7. Full traceability JSON with per-image stats

Bottlenecks identified and addressed:
  - Dataset size: 296 → 818 images by including plot 2 raw thermal
  - Sharp thresholds: Hysteresis preserves thin fire fronts at boundaries
  - Domain gap (Arizona→Chile): Documented; thermal fire signatures are universal,
    but thermal_anomaly class needs Chilean calibration data for production
  - Resolution mismatch: Training at 640x512, deploying at 512x384 (P3 upscaled).
    YOLOv8 internal resize handles this, but small fires (<10px) may be missed.
  - False positives: Early cold frames serve as hard negatives
  - Label quality: QA images generated for human spot-check

Scientific basis:
  - Active fire seed >200°C: Hopkins et al. 2024 (FLAME 3, arXiv:2412.02831)
  - Active fire extension >120°C: Pyrolysis begins ~230°C, pre-heating zone
    reaches 100-150°C ahead of flame front (NWCG, Wikipedia combustion science)
  - Thermal anomaly >3σ + >50°C: NWCG Fire Weather ch.2, canopy temp variation
  - No-fire <80°C: Hopkins et al. 2024

Classes:
  0 = active_fire       Hysteresis: >200°C seed, >120°C connected extension
  1 = thermal_anomaly   >3σ above local median AND >50°C absolute

Hardware:
  Training:   HP Omen, i7-13th, RTX 4060 8GB VRAM, 16GB RAM
  Deployment: Jetson Orin Nano 8GB + Thermal Master P3 (256x192 native)

Author: Pablo Silva / IgnisEdge / Visioncore Technologies SpA
"""

import os
import sys
import glob
import shutil
import random
import json
import warnings
import numpy as np
import rasterio
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import median_filter, label as ndlabel

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ============================================================
# SCIENTIFIC THRESHOLDS
# ============================================================

# Active fire: hysteresis thresholding
# Seed (high confidence): >200°C — confirmed active fire (Hopkins et al., 2024)
# Extension (connected): >120°C — pre-heating zone ahead of flame front
#   Scientific basis: pyrolysis begins ~230°C, radiant pre-heating reaches
#   100-150°C in the convective column (NWCG). Extension only applies to
#   pixels CONNECTED to a seed region, preventing isolated warm spots.
FIRE_SEED_THRESHOLD = 200.0    # °C — high confidence fire seed
FIRE_EXTEND_THRESHOLD = 120.0  # °C — connected extension zone

# Thermal anomaly: statistical (sub-canopy fire detection)
# >3σ above local median AND >50°C absolute
# Basis: NWCG canopy temp varies ~10°C; in Chile (5-25°C ambient),
# >50°C absolute eliminates sun-heated surfaces
ANOMALY_SIGMA = 3.0
ANOMALY_ABSOLUTE_MIN = 50.0
ANOMALY_WINDOW_SIZE = 31  # Must be odd for median_filter. 31px ≈ 5% of 640

# No-fire: max temp below 80°C (Hopkins et al., 2024)
NO_FIRE_THRESHOLD = 80.0

# Sensor saturation (AUTEL Evo II 640T)
SENSOR_SATURATION = 500.0

# ============================================================
# MORPHOLOGICAL PARAMETERS (stress-tested)
# ============================================================

FIRE_MORPH_KERNEL = 5
FIRE_MIN_AREA = 60       # ~0.018% of 640x512 — preserves small ignition points
ANOMALY_MORPH_KERNEL = 3
ANOMALY_MIN_AREA = 100   # Larger to avoid scattered noise in anomaly class
MIN_POLYGON_POINTS = 6

# ============================================================
# DATASET CONFIG
# ============================================================

YOLO_IMG_SIZE = 640
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

FLAME3_BASE = os.path.expanduser("~/IgnisEdge/datasets/flame3")
YOLO_BASE = os.path.expanduser("~/IgnisEdge/datasets/yolo_fire_thermal")
QA_DIR = os.path.expanduser("~/IgnisEdge/datasets/yolo_fire_thermal/qa_review")

CLASS_NAMES = {0: "active_fire", 1: "thermal_anomaly"}
DN_FORMULA = "celsius = DN * 0.1 - 275"


# ============================================================
# CORE FUNCTIONS
# ============================================================

def dn_to_celsius(dn_array):
    """Convert raw DN to Celsius. Source: FLAME 3 README."""
    return dn_array.astype(np.float32) * 0.1 - 275.0


def hysteresis_threshold(celsius, seed_thresh, extend_thresh):
    """
    Hysteresis thresholding for fire detection.

    Like Canny edge detection but for temperature:
    1. Seed pixels (>200°C): definitely fire
    2. Extension pixels (>120°C): fire ONLY if connected to a seed

    Why: fire boundaries are gradual. A pixel at 150°C adjacent to
    a 400°C pixel is clearly part of the fire, but a lone 150°C pixel
    could be sun-heated rock. Hysteresis captures this distinction.

    Returns binary mask (uint8, 0 or 255).
    """
    seed_mask = (celsius > seed_thresh).astype(np.uint8)
    extend_mask = (celsius > extend_thresh).astype(np.uint8)

    # If no seeds, no fire
    if seed_mask.sum() == 0:
        return np.zeros_like(seed_mask, dtype=np.uint8)

    # Label connected components in the extension mask
    labeled_extend, n_components = ndlabel(extend_mask)

    # Find which components contain at least one seed pixel
    seed_labels = set(np.unique(labeled_extend[seed_mask > 0]))
    seed_labels.discard(0)  # Remove background

    # Keep only components connected to seeds
    result = np.zeros_like(seed_mask, dtype=np.uint8)
    for lbl in seed_labels:
        result[labeled_extend == lbl] = 255

    return result


def compute_thermal_anomaly_mask(celsius):
    """
    Statistical anomaly detection for sub-canopy fire.

    Method:
    1. Compute local median in window (robust to outliers)
    2. Compute local standard deviation
    3. Flag pixels > median + 3σ AND > 50°C absolute

    Why: in dense canopy, fire below trees heats the crown to
    40-60°C while surroundings are 10-20°C. This is invisible
    to absolute thresholds but clear as a statistical outlier.
    """
    if celsius.max() <= ANOMALY_ABSOLUTE_MIN:
        return np.zeros(celsius.shape, dtype=np.uint8)

    c32 = celsius.astype(np.float32)
    ws = ANOMALY_WINDOW_SIZE

    # Local median (scipy handles float32)
    local_med = median_filter(c32, size=ws).astype(np.float32)

    # Local std via E[X²] - E[X]²
    local_mean = cv2.blur(c32.astype(np.float64), (ws, ws))
    local_sq_mean = cv2.blur((c32.astype(np.float64)) ** 2, (ws, ws))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0)).astype(np.float32)
    local_std = np.maximum(local_std, 1.0)  # Floor to avoid div-by-zero

    # Statistical + absolute criterion
    stat_mask = c32 > (local_med + ANOMALY_SIGMA * local_std)
    abs_mask = c32 > ANOMALY_ABSOLUTE_MIN
    combined = (stat_mask & abs_mask).astype(np.uint8) * 255

    return combined


def clean_mask(mask, kernel_size, min_area):
    """Morphological cleaning: open (denoise) → close (fill gaps) → area filter."""
    if mask.sum() == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Area filter
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, -1)

    return result


def mask_to_yolo_polygons(mask, class_id):
    """Convert binary mask → YOLO-seg polygon format (normalized coords)."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 25:
            continue

        eps = 0.006 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2)
        coords = []
        for px, py in pts:
            coords.append(str(round(np.clip(px / w, 0, 1), 6)))
            coords.append(str(round(np.clip(py / h, 0, 1), 6)))

        if len(coords) >= MIN_POLYGON_POINTS:
            lines.append(f"{class_id} " + " ".join(coords))

    return lines


def process_image(tiff_path):
    """
    Full pipeline for one thermal TIFF.

    Returns:
        yolo_lines: list of YOLO label strings
        stats: dict with image statistics
        fire_mask, anomaly_mask: for QA visualization
    """
    with rasterio.open(tiff_path) as src:
        dn = src.read(1)

    celsius = dn_to_celsius(dn)
    h, w = celsius.shape
    total_px = h * w

    stats = {
        'temp_min': round(float(celsius.min()), 1),
        'temp_max': round(float(celsius.max()), 1),
        'temp_mean': round(float(celsius.mean()), 1),
    }

    # --- Class 0: Active Fire (hysteresis) ---
    fire_mask = hysteresis_threshold(celsius, FIRE_SEED_THRESHOLD, FIRE_EXTEND_THRESHOLD)
    fire_mask = clean_mask(fire_mask, FIRE_MORPH_KERNEL, FIRE_MIN_AREA)
    fire_lines = mask_to_yolo_polygons(fire_mask, 0)

    stats['fire_px'] = int((fire_mask > 0).sum())
    stats['fire_pct'] = round(stats['fire_px'] / total_px * 100, 2)
    stats['fire_polygons'] = len(fire_lines)

    # --- Class 1: Thermal Anomaly (statistical) ---
    anomaly_mask = compute_thermal_anomaly_mask(celsius)
    anomaly_mask = cv2.subtract(anomaly_mask, fire_mask)  # No overlap
    anomaly_mask = clean_mask(anomaly_mask, ANOMALY_MORPH_KERNEL, ANOMALY_MIN_AREA)
    anomaly_lines = mask_to_yolo_polygons(anomaly_mask, 1)

    stats['anomaly_px'] = int((anomaly_mask > 0).sum())
    stats['anomaly_pct'] = round(stats['anomaly_px'] / total_px * 100, 2)
    stats['anomaly_polygons'] = len(anomaly_lines)

    # Classification
    if stats['fire_px'] > 0:
        stats['class'] = 'fire'
    elif stats['temp_max'] < NO_FIRE_THRESHOLD:
        stats['class'] = 'no_fire'
    elif stats['anomaly_px'] > 0:
        stats['class'] = 'anomaly_only'
    else:
        stats['class'] = 'ambiguous'

    return fire_lines + anomaly_lines, stats, fire_mask, anomaly_mask


def generate_qa_image(thermal_jpg_path, fire_mask, anomaly_mask, output_path, stats):
    """
    Generate QA overlay image for human review.
    Red = active_fire, Yellow = thermal_anomaly, text overlay with stats.
    """
    img = cv2.imread(thermal_jpg_path)
    if img is None:
        return

    overlay = img.copy()

    # Red overlay for fire
    if fire_mask is not None and fire_mask.sum() > 0:
        fire_colored = np.zeros_like(img)
        fire_colored[:, :, 2] = fire_mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, fire_colored, 0.3, 0)

    # Yellow overlay for anomaly
    if anomaly_mask is not None and anomaly_mask.sum() > 0:
        anom_colored = np.zeros_like(img)
        anom_colored[:, :, 1] = anomaly_mask  # Green channel
        anom_colored[:, :, 2] = anomaly_mask  # Red channel → Yellow
        overlay = cv2.addWeighted(overlay, 0.85, anom_colored, 0.15, 0)

    # Text overlay
    texts = [
        f"Max: {stats['temp_max']}C | Class: {stats['class']}",
        f"Fire: {stats['fire_pct']}% ({stats['fire_polygons']} poly)",
        f"Anomaly: {stats['anomaly_pct']}% ({stats['anomaly_polygons']} poly)",
    ]
    for i, txt in enumerate(texts):
        cv2.putText(overlay, txt, (10, 20 + i * 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])


def discover_pairs(base_dir):
    """
    Auto-discover all TIFF ↔ thermal JPG pairs across plots.
    Handles both plot 1 and plot 2, with any naming convention.
    """
    all_pairs = []

    for plot_dir_name in ["plot 1", "plot 2"]:
        tiff_dir = os.path.join(base_dir, plot_dir_name, "duringburn", "raw_thermal_tiff")
        jpg_dir = os.path.join(base_dir, plot_dir_name, "duringburn", "raw_thermal_jpg")

        if not os.path.exists(tiff_dir) or not os.path.exists(jpg_dir):
            continue

        plot_id = plot_dir_name.replace(" ", "")
        tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "IRX_*.TIFF")))

        for tiff_path in tiff_files:
            fname = os.path.basename(tiff_path)
            num = fname.split("_")[1].split(".")[0]

            jpg_path = os.path.join(jpg_dir, f"IRX_{num}.jpg")
            if not os.path.exists(jpg_path):
                # Try uppercase
                jpg_path = os.path.join(jpg_dir, f"IRX_{num}.JPG")

            if os.path.exists(jpg_path):
                all_pairs.append({
                    'plot': plot_id,
                    'num': num,
                    'tiff': tiff_path,
                    'jpg': jpg_path,
                    'name': f"{plot_id}_{num}",
                })

    return all_pairs


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="IgnisEdge FLAME3 Auto-Labeling v3 (Production Final)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    parser.add_argument("--qa-samples", type=int, default=50,
                        help="Number of QA images to generate (default: 50)")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA image generation")
    args = parser.parse_args()

    ts = datetime.now()

    print()
    print("=" * 70)
    print("  IgnisEdge — FLAME 3 Auto-Labeling v3 (Production)")
    print("=" * 70)
    print(f"  Timestamp:        {ts.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Fire detection:   Hysteresis — seed >{FIRE_SEED_THRESHOLD}°C, "
          f"extend >{FIRE_EXTEND_THRESHOLD}°C")
    print(f"  Anomaly detect:   >{ANOMALY_SIGMA}σ local (window {ANOMALY_WINDOW_SIZE}px) "
          f"+ >{ANOMALY_ABSOLUTE_MIN}°C abs")
    print(f"  No-fire:          <{NO_FIRE_THRESHOLD}°C max")
    print(f"  Output:           {YOLO_BASE}")
    print("=" * 70)

    # --- Phase 1: Discover ---
    print("\n[1/5] Discovering thermal pairs...")
    pairs = discover_pairs(FLAME3_BASE)
    print(f"  Found {len(pairs)} matched TIFF↔JPG pairs")

    plot_counts = {}
    for p in pairs:
        plot_counts[p['plot']] = plot_counts.get(p['plot'], 0) + 1
    for pname, cnt in sorted(plot_counts.items()):
        print(f"    {pname}: {cnt} pairs")

    if not pairs:
        print("ERROR: No pairs found.")
        sys.exit(1)

    # --- Phase 2: Process ---
    print(f"\n[2/5] Processing {len(pairs)} thermal images...")

    results = []
    for pair in tqdm(pairs, desc="  Labeling"):
        try:
            lines, stats, fire_m, anom_m = process_image(pair['tiff'])
            results.append({
                **pair,
                'lines': lines,
                'stats': stats,
                'fire_mask': fire_m,
                'anomaly_mask': anom_m,
            })
        except Exception as e:
            print(f"\n  ERROR {pair['name']}: {e}")

    # --- Phase 3: Analyze ---
    print(f"\n[3/5] Analyzing distribution...")

    classes = {}
    for r in results:
        c = r['stats']['class']
        classes[c] = classes.get(c, 0) + 1

    n_fire = classes.get('fire', 0)
    n_anom = classes.get('anomaly_only', 0)
    n_nofire = classes.get('no_fire', 0)
    n_ambig = classes.get('ambiguous', 0)
    n_total = len(results)

    temps = [r['stats']['temp_max'] for r in results]

    print(f"  Total processed:      {n_total}")
    print(f"  With active_fire:     {n_fire} ({n_fire/n_total*100:.1f}%)")
    print(f"  Anomaly only:         {n_anom} ({n_anom/n_total*100:.1f}%)")
    print(f"  No fire (<80°C):      {n_nofire} ({n_nofire/n_total*100:.1f}%)")
    print(f"  Ambiguous (80-200°C): {n_ambig} ({n_ambig/n_total*100:.1f}%)")
    print(f"  Temp range:           {min(temps):.1f}°C — {max(temps):.1f}°C")

    # Class imbalance warning
    if n_fire > 0 and n_anom > 0:
        ratio = max(n_fire, n_anom) / max(min(n_fire, n_anom), 1)
        if ratio > 5:
            print(f"\n  ⚠ CLASS IMBALANCE WARNING: ratio {ratio:.1f}:1")
            print(f"    Consider training active_fire only for v1, "
                  f"add thermal_anomaly with Chilean field data in v2")

    if args.dry_run:
        print("\n  DRY RUN — no files written.")
        return

    # --- Phase 4: Split & Write ---
    print(f"\n[4/5] Writing dataset...")

    random.seed(RANDOM_SEED)
    random.shuffle(results)

    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": results[:n_train],
        "val": results[n_train:n_train + n_val],
        "test": results[n_train + n_val:],
    }

    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_BASE, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(YOLO_BASE, s, "labels"), exist_ok=True)

    split_info = {}
    for sname, samples in splits.items():
        img_d = os.path.join(YOLO_BASE, sname, "images")
        lbl_d = os.path.join(YOLO_BASE, sname, "labels")
        n_labeled = 0

        for r in tqdm(samples, desc=f"  {sname}"):
            name = r['name']
            shutil.copy2(r['jpg'], os.path.join(img_d, f"{name}.jpg"))

            content = "\n".join(r['lines']) if r['lines'] else ""
            with open(os.path.join(lbl_d, f"{name}.txt"), "w") as f:
                f.write(content)

            if r['lines']:
                n_labeled += 1

        split_info[sname] = {'total': len(samples), 'labeled': n_labeled}

    # --- QA Images ---
    if not args.no_qa:
        os.makedirs(QA_DIR, exist_ok=True)
        qa_candidates = [r for r in results if r['stats']['class'] in ('fire', 'anomaly_only', 'ambiguous')]
        qa_sample = qa_candidates[:args.qa_samples]
        print(f"\n  Generating {len(qa_sample)} QA images...")
        for r in tqdm(qa_sample, desc="  QA"):
            qa_path = os.path.join(QA_DIR, f"qa_{r['name']}.jpg")
            generate_qa_image(r['jpg'], r['fire_mask'], r['anomaly_mask'], qa_path, r['stats'])

    # --- Phase 5: YAML & Metadata ---
    print(f"\n[5/5] Writing configuration...")

    yaml_text = f"""# IgnisEdge Fire Detection Dataset (Thermal Model v1)
# Generated: {ts.strftime('%Y-%m-%d %H:%M')}
# Source: FLAME 3 Hanna Hammock (AUTEL Evo II 640T, 640x512)
# Pipeline: autolabel_flame3_v3.py (hysteresis + statistical anomaly)
#
# Thresholds:
#   active_fire:     hysteresis seed>{FIRE_SEED_THRESHOLD}°C extend>{FIRE_EXTEND_THRESHOLD}°C
#   thermal_anomaly: >{ANOMALY_SIGMA}σ local + >{ANOMALY_ABSOLUTE_MIN}°C absolute
#
# Stats: {n_total} images | fire:{n_fire} anomaly:{n_anom} nofire:{n_nofire}
# Split: train:{split_info['train']['total']} val:{split_info['val']['total']} test:{split_info['test']['total']}
#
# DEPLOYMENT NOTE:
#   Trained on 640x512 (AUTEL), deploying on 512x384 (P3 upscaled).
#   Test inference at both resolutions before flight.

path: {YOLO_BASE}
train: train/images
val: val/images
test: test/images

names:
  0: active_fire
  1: thermal_anomaly
"""

    with open(os.path.join(YOLO_BASE, "dataset.yaml"), "w") as f:
        f.write(yaml_text)

    # Metadata JSON
    meta = {
        'version': 'v3_production_final',
        'timestamp': ts.isoformat(),
        'thresholds': {
            'fire_seed_celsius': FIRE_SEED_THRESHOLD,
            'fire_extend_celsius': FIRE_EXTEND_THRESHOLD,
            'anomaly_sigma': ANOMALY_SIGMA,
            'anomaly_abs_min_celsius': ANOMALY_ABSOLUTE_MIN,
            'anomaly_window_px': ANOMALY_WINDOW_SIZE,
            'no_fire_max_celsius': NO_FIRE_THRESHOLD,
        },
        'dataset': {
            'total': n_total,
            'classes': classes,
            'splits': split_info,
            'temp_range': [min(temps), max(temps)],
        },
        'per_image': [{
            'name': r['name'],
            **r['stats']
        } for r in results],
        'references': [
            'Hopkins et al. 2024, FLAME 3 (arXiv:2412.02831) — fire/no-fire thresholds',
            'Amici 2022, JGR Biogeosciences — smoldering ~327°C, flaming ~727°C',
            'NWCG Fire Weather ch.2 — canopy temperature variation ~10°C',
            'Wikipedia Wildfire — pyrolysis 230°C, smoldering 380°C, ignition 590°C',
        ],
        'known_limitations': [
            'FLAME 3 is open pine forest (FL/AZ); deployment is dense native forest (Chile)',
            'thermal_anomaly class has limited training variety for sub-canopy scenarios',
            'Resolution gap: train 640x512 → deploy 512x384 (P3 upscaled) or 256x192 (native)',
            'No smoke labels in thermal model (smoke is near-ambient temp; use RGB model)',
        ],
        'training_recommendations': {
            'model': 'yolov8n-seg.pt',
            'epochs': 200,
            'batch': 16,
            'imgsz': 640,
            'patience': 50,
            'device': 0,
            'note_augmentation': 'Reduce HSV augmentation for thermal (hsv_h=0, hsv_s=0.2, hsv_v=0.2)',
            'note_deployment': 'Export to TensorRT FP16 on Jetson for real-time inference',
        },
    }

    with open(os.path.join(YOLO_BASE, "labeling_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Summary ---
    print()
    print("=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    for sname, si in split_info.items():
        print(f"  {sname:6s}: {si['total']} images ({si['labeled']} labeled)")
    print(f"\n  Dataset YAML: {YOLO_BASE}/dataset.yaml")
    print(f"  QA images:    {QA_DIR}/ ({args.qa_samples} samples)")
    print(f"  Metadata:     {YOLO_BASE}/labeling_meta.json")

    print(f"\n  Train command:")
    print(f"  cd ~/IgnisEdge && source venv/bin/activate")
    print(f"  yolo segment train \\")
    print(f"    data={YOLO_BASE}/dataset.yaml \\")
    print(f"    model=yolov8n-seg.pt \\")
    print(f"    epochs=200 imgsz=640 batch=16 \\")
    print(f"    patience=50 device=0 \\")
    print(f"    hsv_h=0 hsv_s=0.2 hsv_v=0.2 \\")
    print(f"    project=~/IgnisEdge/models \\")
    print(f"    name=ignisedge_thermal_v1")
    print()


if __name__ == "__main__":
    main()
