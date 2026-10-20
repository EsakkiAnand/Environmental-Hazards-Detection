import os
import glob
import cv2
import json
import yaml
import shutil
import random
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Source & Target Paths
RAW_DATASET_DIR = r"C:\Users\esakk\dataset\Hazards dataset"
UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards"

# Final 7-Class Mapping
CLASS_NAMES = {
    0: "fire",
    1: "flood",
    2: "damaged_structure",
    3: "exposed_powerline",
    4: "debris",
    5: "landslide",
    6: "chemical_spill"
}

CLASS_MAPPING = {v: k for k, v in CLASS_NAMES.items()}

# Spatial Scale & Sanitation Thresholds (for 15m Drone Perspective)
MIN_BOX_PX = 12       # Minimum box dimension in pixels (width or height >= 12px)
MAX_AREA_RATIO = 0.95 # Maximum relative box area (drop background frame artifacts)

def sanitize_bbox(x, y, w, h, img_w, img_h, min_px=MIN_BOX_PX, min_area_px=144):
    """
    Validates and clamps bounding box in normalized YOLO format: (x_center, y_center, norm_w, norm_h).
    Returns None if box fails minimum pixel size or maximum area criteria.
    """
    px_w = w * img_w
    px_h = h * img_h
    px_area = px_w * px_h
    area_ratio = w * h

    if px_w < min_px or px_h < min_px or px_area < min_area_px:
        return None
    if area_ratio > MAX_AREA_RATIO:
        return None

    # Clamp normalized coordinates to [0.0, 1.0]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"{x:.6f} {y:.6f} {w:.6f} {h:.6f}"

def extract_connected_component_bboxes(binary_mask, target_class_id, min_px=MIN_BOX_PX, min_area_px=144):
    """
    Extracts individual bounding boxes from a binary mask (uint8 0/255) using connected components analysis.
    """
    img_h, img_w = binary_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    bboxes = []
    for i in range(1, num_labels): # Skip background label 0
        x_min, y_min, w, h, area = stats[i]
        
        # Calculate center and normalized coordinates
        x_center = (x_min + w / 2.0) / float(img_w)
        y_center = (y_min + h / 2.0) / float(img_h)
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)

        cleaned_box = sanitize_bbox(x_center, y_center, norm_w, norm_h, img_w, img_h, min_px=min_px, min_area_px=min_area_px)
        if cleaned_box:
            bboxes.append(f"{target_class_id} {cleaned_box}")

    return bboxes

def distribute_samples(samples, output_dir, prefix, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Distributes collected (image_path, bboxes) samples into 70% Train, 20% Val, 10% Test splits.
    """
    if not samples:
        return 0, 0, 0

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    def write_split(split_name, sample_list):
        count = 0
        for idx, item in enumerate(sample_list):
            img_path = item['img_path']
            bboxes = item['bboxes']
            out_name = f"{prefix}_{idx:06d}_{os.path.basename(img_path)}"
            base_no_ext = os.path.splitext(out_name)[0]

            target_img = os.path.join(output_dir, split_name, "images", f"{base_no_ext}.jpg")
            target_lbl = os.path.join(output_dir, split_name, "labels", f"{base_no_ext}.txt")

            if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg"):
                shutil.copy2(img_path, target_img)
            else:
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.imwrite(target_img, img)

            with open(target_lbl, "w") as f:
                f.write("\n".join(bboxes) + "\n")
            count += 1
        return count

    c_train = write_split("train", train_samples)
    c_val = write_split("val", val_samples)
    c_test = write_split("test", test_samples)

    return c_train, c_val, c_test

# ==================== DATASET CONVERTERS ====================

def process_fire_flamevision(output_dir):
    """ Class 0: Fire from FlameVision XML files """
    print("[*] Processing Class 0: Fire (FlameVision)...")
    target_cls = 0
    xml_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Flame", "**", "annotations", "*.xml"), recursive=True)
    if not xml_files:
        xml_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Flame", "**", "*.xml"), recursive=True)

    samples = []
    for xml_path in tqdm(xml_files):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size_elem = root.find("size")
            if size_elem is None:
                continue
            img_w = float(size_elem.find("width").text)
            img_h = float(size_elem.find("height").text)
            if img_w <= 0 or img_h <= 0:
                continue
                
            xml_dir = os.path.dirname(xml_path)
            img_dir = xml_dir.replace("annotations", "images")
            base_no_ext = os.path.splitext(os.path.basename(xml_path))[0]
            
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                cand = os.path.join(img_dir, base_no_ext + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
                cand_same = os.path.join(xml_dir, base_no_ext + ext)
                if os.path.exists(cand_same):
                    img_path = cand_same
                    break
                    
            if not img_path:
                continue

            bboxes = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                w = xmax - xmin
                h = ymax - ymin
                xc = (xmin + w / 2.0) / img_w
                yc = (ymin + h / 2.0) / img_h
                nw = w / img_w
                nh = h / img_h

                box_str = sanitize_bbox(xc, yc, nw, nh, img_w, img_h)
                if box_str:
                    bboxes.append(f"{target_cls} {box_str}")

            if bboxes:
                samples.append({"img_path": img_path, "bboxes": bboxes})
        except Exception:
            continue

    tr, val, ts = distribute_samples(samples, output_dir, prefix="flame")
    print(f"[+] Converted Fire (FlameVision): {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def process_flood_floodnet(output_dir):
    """ Class 1: Flood from FloodNet Challenge (Mask values 2: Building Flooded & 3: Road Flooded) """
    print("[*] Processing Class 1: Flood (FloodNet)...")
    target_cls = 1
    
    mask_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Flood", "**", "mask", "*.png"), recursive=True)
    if not mask_files:
        mask_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Flood", "**", "*.png"), recursive=True)

    samples = []
    for mask_path in tqdm(mask_files):
        mask_dir = os.path.dirname(mask_path)
        img_dir = mask_dir.replace("mask", "image")
        base_stem = os.path.basename(mask_path).replace("_lab.png", "").replace(".png", "")
        
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = os.path.join(img_dir, base_stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
                
        if not img_path:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        binary_flood = np.zeros_like(mask, dtype=np.uint8)
        binary_flood[(mask == 2) | (mask == 3)] = 255

        if np.count_nonzero(binary_flood) == 0:
            continue

        bboxes = extract_connected_component_bboxes(binary_flood, target_cls)
        if bboxes:
            samples.append({"img_path": img_path, "bboxes": bboxes})

    tr, val, ts = distribute_samples(samples, output_dir, prefix="floodnet")
    print(f"[+] Converted Flood (FloodNet): {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def process_rescuenet_hazards(output_dir):
    """
    Class 2: Damaged Structure (RescueNet mask values 3: Minor, 4: Major, 5: Total Destruction)
    Class 4: Debris (RescueNet mask value 8: Road-Blocked/Debris)
    """
    print("[*] Processing Class 2 (Damaged Structure) & Class 4 (Debris) from RescueNet...")
    mask_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Damaged structures", "**", "*label*/*.png"), recursive=True)

    samples = []
    for mask_path in tqdm(mask_files):
        base_name = os.path.basename(mask_path).replace("_lab", "").replace(".png", "")
        mask_dir = os.path.dirname(mask_path)
        org_dir = mask_dir.replace("label-img", "org-img")
        img_path = os.path.join(org_dir, f"{base_name}.jpg")

        if not os.path.exists(img_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        bboxes = []

        # 1. Damaged Structure (Values 3, 4, 5)
        bin_struct = np.zeros_like(mask, dtype=np.uint8)
        bin_struct[(mask >= 3) & (mask <= 5)] = 255
        if np.count_nonzero(bin_struct) > 0:
            b_struct = extract_connected_component_bboxes(bin_struct, target_class_id=2)
            bboxes.extend(b_struct)

        # 2. Debris (Value 8: Road Blocked / Rubble)
        bin_debris = np.zeros_like(mask, dtype=np.uint8)
        bin_debris[mask == 8] = 255
        if np.count_nonzero(bin_debris) > 0:
            b_debris = extract_connected_component_bboxes(bin_debris, target_class_id=4)
            bboxes.extend(b_debris)

        if bboxes:
            samples.append({"img_path": img_path, "bboxes": bboxes})

    tr, val, ts = distribute_samples(samples, output_dir, prefix="rescuenet")
    print(f"[+] Converted RescueNet Hazards: {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def process_powerlines_mpid(output_dir):
    """ Class 3: Exposed Powerlines / Insulators from MPID dataset """
    print("[*] Processing Class 3: Exposed Powerlines (MPID)...")
    target_cls = 3
    
    mpid_root = os.path.join(RAW_DATASET_DIR, "Electrical Signals")
    img_files = glob.glob(os.path.join(mpid_root, "**", "images", "*.*"), recursive=True)

    samples = []
    for img_path in tqdm(img_files):
        img_dir = os.path.dirname(img_path)
        lbl_dir = img_dir.replace("images", "labels")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")

        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        bboxes = []
        with open(txt_path, "r") as rf:
            for line in rf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        xc, yc, w, h = map(float, parts[1:5])
                        cleaned = sanitize_bbox(xc, yc, w, h, img_w, img_h)
                        if cleaned:
                            bboxes.append(f"{target_cls} {cleaned}")
                    except Exception:
                        continue

        if bboxes:
            samples.append({"img_path": img_path, "bboxes": bboxes})

    tr, val, ts = distribute_samples(samples, output_dir, prefix="mpid")
    print(f"[+] Converted Exposed Powerlines (MPID): {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def process_landslide_cas(output_dir):
    """ Class 5: Landslide from CAS Landslide dataset (.tif images & masks) with Morphological Consolidation """
    print("[*] Processing Class 5: Landslide (CAS Landslide with Morphological Consolidation)...")
    target_cls = 5
    
    img_files = glob.glob(os.path.join(RAW_DATASET_DIR, "Landslide", "img", "*.tif"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

    samples = []
    for img_path in tqdm(img_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(RAW_DATASET_DIR, "Landslide", "mask", f"{base_name}.tif")

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        binary_landslide = np.zeros_like(mask, dtype=np.uint8)
        binary_landslide[mask > 0] = 255

        # Morphological Closing & Dilation to consolidate fragmented mask speckles into cohesive landslide region bounding boxes
        binary_closed = cv2.morphologyEx(binary_landslide, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.dilate(binary_closed, kernel, iterations=1)

        bboxes = extract_connected_component_bboxes(binary_clean, target_cls, min_px=25, min_area_px=625)
        if bboxes:
            samples.append({"img_path": img_path, "bboxes": bboxes})

    tr, val, ts = distribute_samples(samples, output_dir, prefix="cas")
    print(f"[+] Converted Landslide (CAS Consolidated): {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def process_chemical_spill_oilspills(output_dir):
    """ Class 6: Chemical / Oil Spill from OilSpills Ground-Truth RGB Mask dataset """
    print("[*] Processing Class 6: Chemical Spill (OilSpills)...")
    target_cls = 6
    
    oil_root = os.path.join(RAW_DATASET_DIR, "OilSpills")
    mask_files = glob.glob(os.path.join(oil_root, "**", "masks", "*.png"), recursive=True)

    samples = []
    for mask_path in tqdm(mask_files):
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        img_dir = os.path.dirname(mask_path).replace("masks", "images")
        
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = os.path.join(img_dir, base_name + ext)
            if os.path.exists(cand):
                img_path = cand
                break

        if not img_path:
            continue

        img = cv2.imread(img_path)
        mask_bgr = cv2.imread(mask_path)
        if img is None or mask_bgr is None:
            continue

        # Oil Spill ground truth BGR color: (124, 0, 255)
        lower_bgr = np.array([110, 0, 240], dtype=np.uint8)
        upper_bgr = np.array([140, 20, 255], dtype=np.uint8)
        binary_oil = cv2.inRange(mask_bgr, lower_bgr, upper_bgr)

        bboxes = extract_connected_component_bboxes(binary_oil, target_cls)
        if bboxes:
            samples.append({"img_path": img_path, "bboxes": bboxes})

    tr, val, ts = distribute_samples(samples, output_dir, prefix="oil")
    print(f"[+] Converted Chemical Spill (OilSpills): {len(samples)} total samples (Train: {tr}, Val: {val}, Test: {ts}).")

def create_data_yaml(output_dir):
    yaml_content = {
        "path": output_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"[+] Generated data.yaml at: {yaml_path}")

def clean_output_dir(output_dir):
    if os.path.exists(output_dir):
        print(f"[*] Cleaning previous output directory at {output_dir}...")
        shutil.rmtree(output_dir)

def main():
    print("=" * 70)
    print("  Unified 7-Class Hazard Dataset Preparation Pipeline")
    print("  Optimized for 15m Drone Altitude & Hailo-8 YOLOv8 Detection")
    print("=" * 70)

    clean_output_dir(UNIFIED_DIR)

    # Initialize directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(UNIFIED_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(UNIFIED_DIR, split, "labels"), exist_ok=True)

    process_fire_flamevision(UNIFIED_DIR)
    process_flood_floodnet(UNIFIED_DIR)
    process_rescuenet_hazards(UNIFIED_DIR)
    process_powerlines_mpid(UNIFIED_DIR)
    process_landslide_cas(UNIFIED_DIR)
    process_chemical_spill_oilspills(UNIFIED_DIR)

    create_data_yaml(UNIFIED_DIR)

    print("\n" + "=" * 70)
    print("  [SUCCESS] Unified Dataset Creation Complete!")
    print(f"  Target Location: {UNIFIED_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
