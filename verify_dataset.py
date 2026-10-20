import os
import glob
import cv2
import yaml
import numpy as np

UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards"
YAML_PATH = os.path.join(UNIFIED_DIR, "data.yaml")

def verify_dataset():
    print("=" * 70)
    print("  Unified Hazards Dataset Verification & Integrity Auditor")
    print("=" * 70)

    if not os.path.exists(YAML_PATH):
        print(f"[!] ERROR: data.yaml not found at {YAML_PATH}")
        return

    with open(YAML_PATH, "r") as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get("names", {})
    print(f"[+] Loaded {len(class_names)} classes from data.yaml:")
    for cid, cname in class_names.items():
        print(f"    Class {cid}: {cname}")

    total_images_all = 0
    total_labels_all = 0
    overall_class_counts = {cid: 0 for cid in class_names.keys()}

    sample_visual_dir = os.path.join(UNIFIED_DIR, "verification_samples")
    os.makedirs(sample_visual_dir, exist_ok=True)
    visual_saved = 0

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(UNIFIED_DIR, split, "images")
        lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")

        images = glob.glob(os.path.join(img_dir, "*.*"))
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")

        total_images_all += len(images)
        total_labels_all += len(labels)

        split_class_counts = {cid: 0 for cid in class_names.keys()}
        invalid_coords_count = 0

        for img_path in images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            if not os.path.exists(lbl_path):
                print(f"  [!] Missing label file for image: {os.path.basename(img_path)}")
                continue

            with open(lbl_path, "r") as lf:
                lines = lf.readlines()

            bboxes_for_draw = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        invalid_coords_count += 1

                    if cls_id in split_class_counts:
                        split_class_counts[cls_id] += 1
                        overall_class_counts[cls_id] += 1

                    bboxes_for_draw.append((cls_id, xc, yc, w, h))

            # Save visual debug sample
            if visual_saved < 10 and bboxes_for_draw:
                img = cv2.imread(img_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    for cid, xc, yc, w, h in bboxes_for_draw:
                        xmin = int((xc - w / 2.0) * iw)
                        ymin = int((yc - h / 2.0) * ih)
                        xmax = int((xc + w / 2.0) * iw)
                        ymax = int((yc + h / 2.0) * ih)

                        cname = class_names.get(cid, str(cid))
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, cname, (xmin, max(15, ymin - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    out_vis = os.path.join(sample_visual_dir, f"sample_{visual_saved}_{base_name}.jpg")
                    cv2.imwrite(out_vis, img)
                    visual_saved += 1

        print("  Class Distribution:")
        for cid, cnt in split_class_counts.items():
            print(f"    {class_names.get(cid, cid)}: {cnt} boxes")
        if invalid_coords_count > 0:
            print(f"  [!] WARNING: Found {invalid_coords_count} out-of-bounds coordinates!")
        else:
            print("  [+] All coordinates are strictly within valid range [0.0, 1.0].")

    print("\n" + "=" * 70)
    print("  OVERALL DATASET AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Total Dataset Images: {total_images_all}")
    print(f"  Total Dataset Labels: {total_labels_all}")
    print("  Total Box Instances Per Class:")
    for cid, cnt in overall_class_counts.items():
        print(f"    [{cid}] {class_names.get(cid, cid)}: {cnt} instances")
    print(f"  Visual Verification Samples Saved To: {sample_visual_dir}")
    print("=" * 70)

if __name__ == "__main__":
    verify_dataset()
