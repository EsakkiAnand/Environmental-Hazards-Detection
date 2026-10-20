import os
import glob
import json
import cv2
import math
import numpy as np
from PIL import Image

UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards_v2"
VISUAL_DIR = r"C:\Users\esakk\.gemini\antigravity-ide\brain\3e443671-9322-4212-b0d4-e31c49f7474a\visual_samples"

CLASS_NAMES = {
    0: "fire",
    1: "flood",
    2: "damaged_structure",
    3: "exposed_powerline",
    4: "debris",
    5: "landslide"
}

def analyze_dataset_stats():
    print("="*70)
    print("PHASE 1-3 ANALYZER: CLASS IMBALANCE & HARD NEGATIVES")
    print("="*70)

    splits = ["train", "val", "test"]
    class_instances = {k: 0 for k in CLASS_NAMES.keys()}
    class_images = {k: set() for k in CLASS_NAMES.keys()}
    total_imgs = 0
    bg_imgs = []
    annotated_imgs = 0

    for split in splits:
        img_dir = os.path.join(UNIFIED_DIR, split, "images")
        lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")

        imgs = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        total_imgs += len(imgs)

        for ipath in imgs:
            stem = os.path.splitext(os.path.basename(ipath))[0]
            lpath = os.path.join(lbl_dir, stem + ".txt")

            if not os.path.exists(lpath) or os.path.getsize(lpath) == 0:
                bg_imgs.append(ipath)
                continue

            with open(lpath, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                bg_imgs.append(ipath)
                continue

            annotated_imgs += 1
            file_classes = set()
            for l in lines:
                parts = l.split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid in CLASS_NAMES:
                        class_instances[cid] += 1
                        file_classes.add(cid)

            for cid in file_classes:
                class_images[cid].add(ipath)

    print("\n--- CLASS INSTANCE & IMAGE BREAKDOWN ---")
    print(f"{'Class ID':<10}{'Class Name':<20}{'Images':<12}{'Instances':<15}{'Inst/Img':<12}")
    print("-" * 69)
    
    for cid in sorted(CLASS_NAMES.keys()):
        cname = CLASS_NAMES[cid]
        num_imgs = len(class_images[cid])
        num_inst = class_instances[cid]
        inst_per_img = (num_inst / float(num_imgs)) if num_imgs > 0 else 0.0
        print(f"{cid:<10}{cname:<20}{num_imgs:<12}{num_inst:<15}{inst_per_img:<12.2f}")

    max_inst = max(class_instances.values())
    min_inst = min(class_instances.values())
    imbalance_ratio = max_inst / float(min_inst) if min_inst > 0 else 0
    print(f"\nMax Class Instances: {max_inst} (Class 0: fire)")
    print(f"Min Class Instances: {min_inst} (Class 4: debris)")
    print(f"Effective Imbalance Ratio: {imbalance_ratio:.2f}x")

    print("\n--- PROPOSED REPEAT FACTOR SAMPLING (RFS) CONFIGURATION ---")
    total_annotated = float(annotated_imgs)
    for cid in sorted(CLASS_NAMES.keys()):
        f_c = len(class_images[cid]) / total_annotated if total_annotated > 0 else 0.0
        r_t = 0.15
        r_c = max(1.0, math.sqrt(r_t / f_c)) if f_c > 0 and f_c < r_t else 1.0
        print(f"  Class {cid} ({CLASS_NAMES[cid]}): category_freq={f_c:.4f}, RFS multiplier={r_c:.2f}x")

    print("\n--- HARD NEGATIVES AUDIT ---")
    print(f"Total Background Images: {len(bg_imgs)}")
    print(f"Total Dataset Images: {total_imgs}")
    print(f"Overall Background Ratio: {len(bg_imgs)/float(total_imgs)*100.0:.2f}%")
    print(f"Training Split Background Ratio: {len(bg_imgs)/7636.0*100.0:.2f}%")

def calculate_sahi_tiling(cam_w=1456, cam_h=1088, tile_size=640, overlap_pct=0.20):
    print("\n" + "="*70)
    print("PHASE 5: SAHI EXACT TILING CALCULATOR FOR ARDUCAM (1456x1088)")
    print("="*70)

    overlap_px = int(tile_size * overlap_pct) # 640 * 0.20 = 128 px
    stride = tile_size - overlap_px          # 640 - 128 = 512 px

    # Compute tiles along X
    x_positions = []
    x = 0
    while x + tile_size <= cam_w:
        x_positions.append(x)
        x += stride
    if x_positions[-1] + tile_size < cam_w:
        x_positions.append(cam_w - tile_size)

    # Compute tiles along Y
    y_positions = []
    y = 0
    while y + tile_size <= cam_h:
        y_positions.append(y)
        y += stride
    if y_positions[-1] + tile_size < cam_h:
        y_positions.append(cam_h - tile_size)

    total_tiles = len(x_positions) * len(y_positions)
    print(f"Camera Resolution: {cam_w} x {cam_h}")
    print(f"Tile Input Size: {tile_size} x {tile_size}")
    print(f"Overlap: {overlap_pct*100:.0f}% ({overlap_px} px overlap, {stride} px stride)")
    print(f"X-axis Tile Origin Offsets: {x_positions} ({len(x_positions)} tiles wide)")
    print(f"Y-axis Tile Origin Offsets: {y_positions} ({len(y_positions)} tiles high)")
    print(f"Exact Total Sliced Tiles per Camera Frame: {total_tiles} tiles/frame")

    target_frame_fps = 30.0
    required_tile_fps = total_tiles * target_frame_fps
    print(f"\nTarget Engineering Workload:")
    print(f"  Target Full-Frame Rate: {target_frame_fps} FPS")
    print(f"  Derived Tile Inference Target: {required_tile_fps:.1f} tile-FPS (on Hailo-8)")

if __name__ == "__main__":
    analyze_dataset_stats()
    calculate_sahi_tiling()
