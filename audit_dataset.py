"""
audit_dataset.py
Seraph Zenith Mark 1 — Dataset Quality Audit Tool

Performs:
  1. Per-class bounding-box distribution count
  2. Image-level annotation coverage statistics
  3. Hard-negative quality check (detects flat/synthetic images)
  4. Cross-split leakage detection (same image filename in train+val)
  5. Box geometry sanity check (zero-area, out-of-bounds boxes)
  6. Recommendation report

Usage:
    python audit_dataset.py
    python audit_dataset.py --dataset C:\path\to\your\dataset
    python audit_dataset.py --show-flat-negatives
"""

import os
import glob
import argparse
import hashlib
import collections
import cv2
import numpy as np

DATASET_DIR  = r"C:\Users\esakk\dataset\unified_hazards_v2"
SPLITS       = ["train", "val", "test"]
CLASS_NAMES  = {
    0: "fire",
    1: "flood",
    2: "damaged_structure",
    3: "exposed_powerline",
    4: "debris",
    5: "landslide",
}
# Target minimum boxes per class for a reasonably balanced dataset
# These are guidance values, not hard requirements.
BALANCE_TARGET = {cls: 3000 for cls in CLASS_NAMES.values()}

# Std-dev threshold below which an image is considered "flat" (synthetic uniform colour)
FLAT_IMAGE_STDDEV_THRESH = 8.0


def load_split(dataset_dir: str, split: str):
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")) +
                  glob.glob(os.path.join(img_dir, "*.tif")))
    lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    return imgs, lbls, img_dir, lbl_dir


def is_flat_image(img_path: str) -> tuple:
    """
    Return (is_flat: bool, mean_stddev: float).
    Flat = synthetic solid-colour canvas (std < FLAT_IMAGE_STDDEV_THRESH).
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, -1.0
    stddev = float(np.std(img))
    return stddev < FLAT_IMAGE_STDDEV_THRESH, stddev


def audit_labels(lbl_dir: str):
    """Audit all label files in a directory for geometry issues."""
    issues = []
    class_counts = collections.Counter()
    total_boxes  = 0
    empty_files  = 0
    files        = glob.glob(os.path.join(lbl_dir, "*.txt"))

    for fpath in files:
        lines = [l.strip() for l in open(fpath).readlines() if l.strip()]
        if not lines:
            empty_files += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"BAD_FORMAT {fpath}: '{line}'")
                continue
            cls_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            class_counts[cls_id] += 1
            total_boxes += 1

            if cls_id not in CLASS_NAMES:
                issues.append(f"UNKNOWN_CLASS {cls_id} in {fpath}")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                issues.append(f"BAD_WH cls={cls_id} w={w:.3f} h={h:.3f} in {fpath}")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(f"BAD_XY cls={cls_id} x={x:.3f} y={y:.3f} in {fpath}")
            if w * h < 0.0001:
                issues.append(f"TINY_BOX cls={cls_id} area={w*h:.6f} in {fpath}")
            if w * h > 0.90:
                issues.append(f"HUGE_BOX cls={cls_id} area={w*h:.2f} in {fpath}")

    return class_counts, total_boxes, empty_files, issues


def check_leakage(train_imgs, val_imgs):
    """Detect same base filename appearing in both train and val."""
    train_names = {os.path.basename(p) for p in train_imgs}
    val_names   = {os.path.basename(p) for p in val_imgs}
    overlap = train_names & val_names
    return sorted(overlap)


def main(dataset_dir, show_flat_negatives=False):
    print("=" * 65)
    print("  Seraph Zenith — Dataset Quality Audit")
    print(f"  Dataset: {dataset_dir}")
    print("=" * 65)

    all_class_counts = collections.Counter()
    total_all        = 0
    flat_images      = []

    split_data = {}
    for split in SPLITS:
        imgs, lbls, img_dir, lbl_dir = load_split(dataset_dir, split)
        if not imgs and not lbls:
            print(f"\n[SKIP] Split '{split}' not found.")
            continue
        cls_counts, total_boxes, empty_files, issues = audit_labels(lbl_dir)
        split_data[split] = {"imgs": imgs, "cls_counts": cls_counts}

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Images: {len(imgs)} | Label files: {len(lbls)}")
        print(f"  Total bboxes: {total_boxes} | Empty label files: {empty_files}")
        print(f"  Label geometry issues: {len(issues)}")
        if issues:
            for iss in issues[:10]:
                print(f"    [ISSUE] {iss}")
            if len(issues) > 10:
                print(f"    ... and {len(issues)-10} more")

        all_class_counts += cls_counts
        total_all        += total_boxes

    print(f"\n{'='*65}")
    print(f"  OVERALL BOUNDING BOX DISTRIBUTION (train + val + test)")
    print(f"  Total bboxes: {total_all}")
    print(f"{'='*65}")
    max_cls = max(all_class_counts.values()) if all_class_counts else 1
    for cls_id, cls_name in CLASS_NAMES.items():
        cnt = all_class_counts.get(cls_id, 0)
        pct = cnt / total_all * 100 if total_all > 0 else 0
        bar = "#" * int(40 * cnt / max_cls)
        flag = " *** UNDER-REPRESENTED ***" if cnt < BALANCE_TARGET.get(cls_name, 0) else ""
        print(f"  Cls {cls_id} ({cls_name:22s}): {cnt:6d} ({pct:5.1f}%)  {bar}{flag}")

    # --- Cross-split leakage check ---
    if "train" in split_data and "val" in split_data:
        leaks = check_leakage(split_data["train"]["imgs"], split_data["val"]["imgs"])
        print(f"\n  Train/Val filename leakage: {len(leaks)} shared filenames")
        if leaks:
            for f in leaks[:5]:
                print(f"    [LEAK] {f}")

    # --- Hard-negative quality check ---
    print(f"\n{'='*65}")
    print("  HARD-NEGATIVE QUALITY CHECK")
    print(f"  (checking for flat synthetic images with stddev < {FLAT_IMAGE_STDDEV_THRESH})")
    print(f"{'='*65}")
    flat_count  = 0
    total_checked = 0
    for split in SPLITS:
        if split not in split_data:
            continue
        imgs = split_data[split]["imgs"]
        for img_path in imgs:
            if "bg_hard_negative" in os.path.basename(img_path):
                is_flat, stddev = is_flat_image(img_path)
                total_checked += 1
                if is_flat:
                    flat_count += 1
                    if show_flat_negatives:
                        print(f"  [FLAT] stddev={stddev:.1f} {img_path}")
    if total_checked > 0:
        print(f"  Synthetic negative images found: {total_checked}")
        print(f"  Flat (uniform-colour) of those:  {flat_count} "
              f"({flat_count/total_checked*100:.0f}%)")
        print(f"  -> These should be replaced with real aerial drone frames")
    else:
        print("  No bg_hard_negative images found (or none checked).")

    # --- Recommendations ---
    print(f"\n{'='*65}")
    print("  RECOMMENDATIONS")
    print(f"{'='*65}")
    for cls_id, cls_name in CLASS_NAMES.items():
        cnt = all_class_counts.get(cls_id, 0)
        target = BALANCE_TARGET.get(cls_name, 0)
        if cnt < target:
            needed = target - cnt
            print(f"  [{cls_name}] Needs ~{needed} more bboxes "
                  f"(current {cnt}, target {target})")
    if flat_count > 0:
        print(f"  [hard_negatives] Replace {flat_count} flat synthetic images with "
              f"real aerial drone frames (no hazard confirmed).")
    print(f"  [landslide] Verify that Jiuzhaigou TIF training images are visually "
          f"similar to IMX296 drone footage before retraining.")
    print(f"  [general] Run per-class mAP evaluation on val set BEFORE retraining.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--show-flat-negatives", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.show_flat_negatives)
