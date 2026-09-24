"""
IgnisEdge - Regenerate FLAME 3 dataset with zero-masked radiometric preprocessing.

Reads raw thermal TIFFs from plot 1 + plot 2, applies preprocess_radiometric()
(same pipeline used at deployment), generates YOLO-seg labels from the hysteresis
mask, and writes the full train/val/test split.

Output: ~/IgnisEdge/datasets/yolo_fire_thermal_v2/
  ├── dataset.yaml
  ├── train/images/*.jpg  labels/*.txt
  ├── val/images/*.jpg    labels/*.txt
  ├── test/images/*.jpg   labels/*.txt
  └── metadata.json       (full reproducibility record)
"""

import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    preprocess_radiometric,
    hysteresis_mask,
    compute_stats,
    SEED_TEMP_C,
    EXTEND_TEMP_C,
    NORMALIZE_MAX_C,
)

# --- Paths ---
FLAME3_BASE = Path.home() / "IgnisEdge/datasets/flame3"
OUT_BASE = Path.home() / "IgnisEdge/datasets/yolo_fire_thermal_v2"

# --- Dataset config ---
PLOTS = {
    "plot1": FLAME3_BASE / "plot 1/duringburn/raw_thermal_tiff",
    "plot2": FLAME3_BASE / "plot 2/duringburn/raw_thermal_tiff",
}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
SEED = 42

# --- Segmentation label config ---
# Two classes: class 0 = active_fire (seed-connected >= 150°C)
#              class 1 = thermal_anomaly (extended halo, >= 80°C connected to fire)
# For this iteration we start with ONE unified class "fire_region" that captures
# the full connected hysteresis mask — aligns with the physical reality that the
# halo is what's detectable at the P3's GSD.
MIN_CONTOUR_AREA = 30      # pixels; reject noise
MIN_POLYGON_POINTS = 6     # min (x,y) pairs for YOLO polygon
CONTOUR_APPROX_EPS = 0.005 # contour simplification aggressiveness


def dn_to_celsius(dn):
    """FLAME 3 raw DN -> Celsius."""
    return dn.astype(np.float32) * 0.1 - 275.0


def mask_to_yolo_polygons(mask, class_id=0):
    """Binary mask -> list of YOLO polygon strings."""
    h, w = mask.shape
    mask_u8 = mask.astype(np.uint8) * 255

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        epsilon = CONTOUR_APPROX_EPS * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(approx) < MIN_POLYGON_POINTS // 2:
            continue

        coords = []
        for x, y in approx:
            nx = max(0.0, min(1.0, x / w))
            ny = max(0.0, min(1.0, y / h))
            coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

        if len(coords) >= MIN_POLYGON_POINTS:
            lines.append(f"{class_id} " + " ".join(coords))

    return lines


def collect_samples():
    """Find all TIFF files across plots."""
    samples = []
    for plot_name, tiff_dir in PLOTS.items():
        if not tiff_dir.exists():
            print(f"  SKIP: {tiff_dir} not found")
            continue
        tiffs = sorted(tiff_dir.glob("IRX_*.TIFF"))
        print(f"  {plot_name}: {len(tiffs)} TIFFs at {tiff_dir}")
        for tiff in tiffs:
            samples.append((plot_name, tiff))
    return samples


def process_sample(plot_name, tiff_path, img_out_dir, lbl_out_dir):
    """Process one TIFF -> write image + label file. Returns stats dict."""
    with rasterio.open(tiff_path) as src:
        dn = src.read(1)

    celsius = dn_to_celsius(dn)

    # Preprocessing (IDENTICAL to deployment)
    processed = preprocess_radiometric(celsius)  # (H, W, 3) uint8

    # Segmentation mask = the hysteresis mask itself
    mask = hysteresis_mask(celsius)

    # Build base filename
    base_name = f"{plot_name}_{tiff_path.stem}"
    img_path = img_out_dir / f"{base_name}.jpg"
    lbl_path = lbl_out_dir / f"{base_name}.txt"

    # Write image (JPG quality 95 keeps gradients without bloat)
    cv2.imwrite(str(img_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Write label (possibly empty = background sample)
    yolo_lines = mask_to_yolo_polygons(mask, class_id=0)
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_lines))

    return {
        "base_name": base_name,
        "plot": plot_name,
        "max_c": float(celsius.max()),
        "connected_px": int(mask.sum()),
        "n_polygons": len(yolo_lines),
    }


def main():
    print("=" * 70)
    print("IgnisEdge - FLAME 3 Dataset Regeneration (zero-masked radiometric)")
    print("=" * 70)
    print(f"Source:          {FLAME3_BASE}")
    print(f"Output:          {OUT_BASE}")
    print(f"Preprocessing:   seed={SEED_TEMP_C}°C  extend={EXTEND_TEMP_C}°C  norm_max={NORMALIZE_MAX_C}°C")
    print(f"Split ratios:    train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}")
    print(f"Random seed:     {SEED}")
    print()

    # Clean previous output if exists
    if OUT_BASE.exists():
        print(f"WARNING: {OUT_BASE} already exists. Removing.")
        shutil.rmtree(OUT_BASE)

    # Create directory structure
    for split in ["train", "val", "test"]:
        (OUT_BASE / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_BASE / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all samples
    print("Scanning source directories...")
    samples = collect_samples()
    if not samples:
        print("ERROR: No TIFF files found.")
        return
    print(f"Total samples found: {len(samples)}")
    print()

    # Shuffle with fixed seed, then split
    random.seed(SEED)
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for name, s in splits.items():
        print(f"  {name}: {len(s)} samples")
    print()

    # Process each split
    t0 = time.time()
    all_stats = []
    split_summary = {}

    for split_name, split_samples in splits.items():
        img_dir = OUT_BASE / split_name / "images"
        lbl_dir = OUT_BASE / split_name / "labels"

        n_with_polys = 0
        n_empty = 0
        max_temps = []

        for plot_name, tiff_path in tqdm(split_samples, desc=f"  {split_name}", ncols=80):
            try:
                st = process_sample(plot_name, tiff_path, img_dir, lbl_dir)
                all_stats.append(st)
                max_temps.append(st["max_c"])
                if st["n_polygons"] > 0:
                    n_with_polys += 1
                else:
                    n_empty += 1
            except Exception as e:
                print(f"\n  ERROR {tiff_path.name}: {e}")

        split_summary[split_name] = {
            "total": len(split_samples),
            "with_polygons": n_with_polys,
            "empty_background": n_empty,
            "max_temp_range": [float(min(max_temps)), float(max(max_temps))] if max_temps else None,
        }

    elapsed = time.time() - t0

    # Write dataset.yaml
    yaml_content = f"""# IgnisEdge Fire Detection Dataset v2
# Auto-generated from FLAME 3 - Hanna Hammock prescribed burn
# Preprocessing: zero-masked radiometric (hysteresis {EXTEND_TEMP_C}-{SEED_TEMP_C}°C)

path: {OUT_BASE}
train: train/images
val: val/images
test: test/images

names:
  0: fire_region
"""
    (OUT_BASE / "dataset.yaml").write_text(yaml_content)

    # Write reproducibility metadata
    metadata = {
        "version": "v2-zeromasked",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(FLAME3_BASE),
        "preprocessing": {
            "seed_temp_c": SEED_TEMP_C,
            "extend_temp_c": EXTEND_TEMP_C,
            "normalize_max_c": NORMALIZE_MAX_C,
            "method": "hysteresis_zero_masking",
        },
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": SEED,
        },
        "label_config": {
            "classes": {"0": "fire_region"},
            "min_contour_area_px": MIN_CONTOUR_AREA,
            "min_polygon_points": MIN_POLYGON_POINTS,
            "contour_approx_epsilon": CONTOUR_APPROX_EPS,
        },
        "split_summary": split_summary,
        "elapsed_seconds": round(elapsed, 2),
        "total_samples": len(all_stats),
    }
    (OUT_BASE / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Final report
    print()
    print("=" * 70)
    print("REGENERATION COMPLETE")
    print("=" * 70)
    for split_name, s in split_summary.items():
        ratio = s["with_polygons"] / max(s["total"], 1) * 100
        print(f"{split_name:>6}: {s['total']:>4} samples  |  "
              f"with labels: {s['with_polygons']:>4} ({ratio:.1f}%)  |  "
              f"empty: {s['empty_background']:>3}")
        if s["max_temp_range"]:
            print(f"         temp range: [{s['max_temp_range'][0]:.1f}, {s['max_temp_range'][1]:.1f}] °C")

    print()
    print(f"Total:           {len(all_stats)} samples in {elapsed:.1f}s")
    print(f"Dataset YAML:    {OUT_BASE}/dataset.yaml")
    print(f"Metadata:        {OUT_BASE}/metadata.json")
    print()
    print("Next: verify a random sample and start training.")


if __name__ == "__main__":
    main()
