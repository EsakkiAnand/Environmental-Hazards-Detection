import os
import glob
import cv2
import json
import yaml
import shutil
import random
import numpy as np
from tqdm import tqdm

RAW_DATASET_DIR = r"C:\Users\esakk\dataset\New Hazards"
UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards_v2"

CLASS_NAMES = {
    0: "fire",
    1: "flood",
    2: "damaged_structure",
    3: "exposed_powerline",
    4: "debris",
    5: "landslide"
}

MIN_BOX_PX = 12
MAX_AREA_RATIO = 0.95

def sanitize_bbox(x, y, w, h, img_w, img_h, min_px=MIN_BOX_PX, min_area_px=144):
    px_w = w * img_w
    px_h = h * img_h
    px_area = px_w * px_h
    area_ratio = w * h

    if px_w < min_px or px_h < min_px or px_area < min_area_px or area_ratio > MAX_AREA_RATIO:
        return None

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"{x:.6f} {y:.6f} {w:.6f} {h:.6f}"

def extract_connected_component_bboxes(binary_mask, target_class_id, min_px=MIN_BOX_PX, min_area_px=144):
    img_h, img_w = binary_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):
        x_min, y_min, w, h, area = stats[i]
        x_center = (x_min + w / 2.0) / float(img_w)
        y_center = (y_min + h / 2.0) / float(img_h)
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)

        cleaned_box = sanitize_bbox(x_center, y_center, norm_w, norm_h, img_w, img_h, min_px=min_px, min_area_px=min_area_px)
        if cleaned_box:
            bboxes.append(f"{target_class_id} {cleaned_box}")

    return bboxes

def test_dataset_counts():
    print("=== Testing dataset file counts and structure in C:\\Users\\esakk\\dataset\\New Hazards ===")
    
    # 1. Damaged Structures & Debris
    ds_masks = glob.glob(os.path.join(RAW_DATASET_DIR, "Damaged Structures", "**", "*_lab.png"), recursive=True)
    print(f"Damaged Structures mask files found: {len(ds_masks)}")

    # 2. Debris
    deb_files = os.listdir(os.path.join(RAW_DATASET_DIR, "Debris"))
    print(f"Debris files in folder: {deb_files}")

    # 3. Exposed Electrical Lines
    el_jsons = glob.glob(os.path.join(RAW_DATASET_DIR, "Exposed Electrical Lines", "**", "*.json"), recursive=True)
    print(f"Exposed Electrical Lines JSON files: {len(el_jsons)}")

    # 4. Fire
    fire_imgs = glob.glob(os.path.join(RAW_DATASET_DIR, "Fire", "images", "*.jpg"))
    fire_yolo_txts = glob.glob(os.path.join(RAW_DATASET_DIR, "Fire", "annotations", "YOLO_UAV", "*.txt"))
    print(f"Fire images: {len(fire_imgs)}, Fire YOLO txt splits: {len(fire_yolo_txts)}")

    # 5. Flood
    fl_masks = glob.glob(os.path.join(RAW_DATASET_DIR, "Flood", "**", "*_lab.png"), recursive=True)
    print(f"FloodNet mask files found: {len(fl_masks)}")

    # 6. Landslide
    ls_imgs = glob.glob(os.path.join(RAW_DATASET_DIR, "Landslide", "img", "*.tif"))
    ls_masks = glob.glob(os.path.join(RAW_DATASET_DIR, "Landslide", "mask", "*.tif"))
    print(f"Landslide images: {len(ls_imgs)}, masks: {len(ls_masks)}")

if __name__ == "__main__":
    test_dataset_counts()
