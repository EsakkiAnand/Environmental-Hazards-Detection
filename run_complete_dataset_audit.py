import os
import glob
import json
import cv2
import hashlib
import numpy as np
import math
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards_v2"
OUTPUT_VISUAL_DIR = r"C:\Users\esakk\.gemini\antigravity-ide\brain\3e443671-9322-4212-b0d4-e31c49f7474a\visual_samples"

CLASS_NAMES = {
    0: "fire",
    1: "flood",
    2: "damaged_structure",
    3: "exposed_powerline",
    4: "debris",
    5: "landslide"
}

CLASS_COLORS = {
    0: (0, 0, 255),      # Red for fire (BGR)
    1: (255, 128, 0),    # Blue for flood
    2: (0, 165, 255),    # Orange for damaged structure
    3: (255, 0, 255),    # Magenta for powerline
    4: (0, 255, 255),    # Yellow for debris
    5: (0, 255, 0)       # Green for landslide
}

def get_file_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def run_audit():
    os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)
    report = {}

    print("\n" + "="*80)
    print("      COMPREHENSIVE 8-PHASE AUDIT: C:\\Users\\esakk\\dataset\\unified_hazards_v2")
    print("="*80 + "\n")

    # ----------------------------------------------------
    # PHASE 1: DATASET STRUCTURE AUDIT
    # ----------------------------------------------------
    print("--- PHASE 1: DATASET STRUCTURE AUDIT ---")
    splits = ["train", "val", "test"]
    p1_data = {}
    total_imgs_all = 0
    total_lbls_all = 0

    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")

        imgs = glob.glob(os.path.join(s_img_dir, "*.*"))
        lbls = glob.glob(os.path.join(s_lbl_dir, "*.txt"))

        # Check background images (0 bytes or 0 lines)
        bg_count = 0
        for l in lbls:
            if os.path.getsize(l) == 0:
                bg_count += 1
            else:
                with open(l, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        bg_count += 1

        p1_data[split] = {
            "image_count": len(imgs),
            "label_count": len(lbls),
            "background_count": bg_count,
            "annotated_count": len(imgs) - bg_count
        }
        total_imgs_all += len(imgs)
        total_lbls_all += len(lbls)

    for split in splits:
        pct = (p1_data[split]["image_count"] / float(total_imgs_all) * 100.0) if total_imgs_all > 0 else 0
        p1_data[split]["split_percentage"] = f"{pct:.2f}%"
        print(f"Split [{split.upper()}]: Images={p1_data[split]['image_count']}, Labels={p1_data[split]['label_count']}, Background={p1_data[split]['background_count']}, Share={p1_data[split]['split_percentage']}")

    print(f"Total Images: {total_imgs_all}, Total Label Files: {total_lbls_all}\n")
    report["phase_1"] = p1_data

    # ----------------------------------------------------
    # PHASE 2: YOLO LABEL INTEGRITY AUDIT
    # ----------------------------------------------------
    print("--- PHASE 2: YOLO LABEL INTEGRITY AUDIT ---")
    total_rows = 0
    invalid_rows = 0
    invalid_class_ids = 0
    out_of_range_coords = 0
    zero_area_boxes = 0
    malformed_files = 0
    orphan_labels = 0
    orphan_images = 0
    empty_label_images = 0

    all_image_paths = []
    all_label_paths = []

    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")

        imgs = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(s_img_dir, "*.*"))}
        lbls = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(s_lbl_dir, "*.txt"))}

        all_image_paths.extend(imgs.values())
        all_label_paths.extend(lbls.values())

        # Check orphan labels / orphan images
        for stem in lbls:
            if stem not in imgs:
                orphan_labels += 1
        for stem in imgs:
            if stem not in lbls:
                orphan_images += 1

        for stem, lpath in lbls.items():
            try:
                with open(lpath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    empty_label_images += 1
                    continue
                for line in lines:
                    parts = line.split()
                    total_rows += 1
                    if len(parts) != 5:
                        invalid_rows += 1
                        continue
                    try:
                        cid = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                    except ValueError:
                        invalid_rows += 1
                        continue

                    if cid not in CLASS_NAMES:
                        invalid_class_ids += 1
                        invalid_rows += 1

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        out_of_range_coords += 1
                        invalid_rows += 1

                    if w <= 0.0 or h <= 0.0 or math.isnan(xc) or math.isnan(yc) or math.isnan(w) or math.isnan(h):
                        zero_area_boxes += 1
                        invalid_rows += 1

            except Exception:
                malformed_files += 1

    p2_data = {
        "total_annotation_rows": total_rows,
        "invalid_rows": invalid_rows,
        "invalid_class_ids": invalid_class_ids,
        "out_of_range_coords": out_of_range_coords,
        "zero_area_boxes": zero_area_boxes,
        "malformed_files": malformed_files,
        "orphan_labels": orphan_labels,
        "orphan_images": orphan_images,
        "empty_label_images": empty_label_images
    }

    print(json.dumps(p2_data, indent=2))
    report["phase_2"] = p2_data

    # ----------------------------------------------------
    # PHASE 3: CLASS DISTRIBUTION
    # ----------------------------------------------------
    print("\n--- PHASE 3: CLASS DISTRIBUTION ---")
    class_instance_counts = defaultdict(int)
    class_image_counts = defaultdict(set)
    total_instances = 0

    for split in splits:
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")
        lbls = glob.glob(os.path.join(s_lbl_dir, "*.txt"))
        for lpath in lbls:
            stem = os.path.splitext(os.path.basename(lpath))[0]
            with open(lpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                        if cid in CLASS_NAMES:
                            class_instance_counts[cid] += 1
                            class_image_counts[cid].add(stem)
                            total_instances += 1
                    except ValueError:
                        pass

    p3_table = []
    print(f"{'Class ID':<10}{'Class Name':<20}{'Images Count':<15}{'Images %':<12}{'Instance Count':<18}{'Instance %':<12}")
    print("-" * 87)
    for cid in sorted(CLASS_NAMES.keys()):
        cname = CLASS_NAMES[cid]
        icnt = class_instance_counts[cid]
        img_cnt = len(class_image_counts[cid])
        inst_pct = (icnt / float(total_instances) * 100.0) if total_instances > 0 else 0
        img_pct = (img_cnt / float(total_imgs_all) * 100.0) if total_imgs_all > 0 else 0
        print(f"{cid:<10}{cname:<20}{img_cnt:<15}{img_pct:<12.2f}%{icnt:<18}{inst_pct:<12.2f}%")
        p3_table.append({
            "class_id": cid,
            "class_name": cname,
            "images_count": img_cnt,
            "images_percentage": f"{img_pct:.2f}%",
            "instance_count": icnt,
            "instance_percentage": f"{inst_pct:.2f}%"
        })

    report["phase_3"] = p3_table

    # ----------------------------------------------------
    # PHASE 4: CRITICAL CLASS-MAPPING VISUAL VERIFICATION
    # ----------------------------------------------------
    print("\n--- PHASE 4: CRITICAL CLASS-MAPPING VISUAL VERIFICATION ---")
    # Save visual validation sample images for each class with bounding boxes drawn
    class_samples = defaultdict(list)
    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")
        lbls = glob.glob(os.path.join(s_lbl_dir, "*.txt"))
        for lpath in lbls:
            stem = os.path.splitext(os.path.basename(lpath))[0]
            ipath = os.path.join(s_img_dir, stem + ".jpg")
            if not os.path.exists(ipath):
                ipath = os.path.join(s_img_dir, stem + ".png")
            if not os.path.exists(ipath):
                continue
            with open(lpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            cids_in_file = set()
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cids_in_file.add(int(parts[0]))
            for cid in cids_in_file:
                if len(class_samples[cid]) < 5:
                    class_samples[cid].append((ipath, lpath))

    saved_visuals = []
    for cid in sorted(CLASS_NAMES.keys()):
        cname = CLASS_NAMES[cid]
        samples = class_samples[cid]
        print(f"Generating visual validation sample for Class {cid} ({cname})... ({len(samples)} samples)")
        for idx, (ipath, lpath) in enumerate(samples):
            img = cv2.imread(ipath)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            with open(lpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    l_cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    bx = int((xc - w/2.0) * img_w)
                    by = int((yc - h/2.0) * img_h)
                    bw = int(w * img_w)
                    bh = int(h * img_h)
                    color = CLASS_COLORS.get(l_cid, (0, 255, 0))
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 3)
                    label_str = f"{CLASS_NAMES.get(l_cid, str(l_cid))}"
                    cv2.putText(img, label_str, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            out_filename = f"class_{cid}_{cname}_sample_{idx+1}.jpg"
            out_path = os.path.join(OUTPUT_VISUAL_DIR, out_filename)
            cv2.imwrite(out_path, img)
            saved_visuals.append(out_path)

    print(f"[+] Visual samples saved at: {OUTPUT_VISUAL_DIR}\n")
    report["phase_4"] = {"saved_visual_samples_count": len(saved_visuals), "sample_directory": OUTPUT_VISUAL_DIR}

    # ----------------------------------------------------
    # PHASE 5: BOUNDING BOX QUALITY
    # ----------------------------------------------------
    print("--- PHASE 5: BOUNDING BOX QUALITY ---")
    widths_px = []
    heights_px = []
    areas_pct = []
    tiny_boxes_cnt = 0
    huge_boxes_cnt = 0
    border_touch_cnt = 0

    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")
        lbls = glob.glob(os.path.join(s_lbl_dir, "*.txt"))
        for lpath in lbls:
            stem = os.path.splitext(os.path.basename(lpath))[0]
            ipath = os.path.join(s_img_dir, stem + ".jpg")
            if not os.path.exists(ipath):
                continue
            # Get dimensions fast without decoding full image if possible
            with Image.open(ipath) as im:
                img_w, img_h = im.size

            with open(lpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    xc, yc, w, h = map(float, parts[1:])
                    px_w = w * img_w
                    px_h = h * img_h
                    area_pct = w * h
                    widths_px.append(px_w)
                    heights_px.append(px_h)
                    areas_pct.append(area_pct)

                    if px_w < 12 or px_h < 12:
                        tiny_boxes_cnt += 1
                    if area_pct > 0.95:
                        huge_boxes_cnt += 1
                    
                    xmin = xc - w/2.0
                    ymin = yc - h/2.0
                    xmax = xc + w/2.0
                    ymax = yc + h/2.0
                    if xmin <= 0.01 or ymin <= 0.01 or xmax >= 0.99 or ymax >= 0.99:
                        border_touch_cnt += 1

    p5_data = {
        "total_boxes_analyzed": len(widths_px),
        "min_width_px": float(np.min(widths_px)) if widths_px else 0,
        "median_width_px": float(np.median(widths_px)) if widths_px else 0,
        "max_width_px": float(np.max(widths_px)) if widths_px else 0,
        "min_height_px": float(np.min(heights_px)) if heights_px else 0,
        "median_height_px": float(np.median(heights_px)) if heights_px else 0,
        "max_height_px": float(np.max(heights_px)) if heights_px else 0,
        "boxes_below_12x12_px": tiny_boxes_cnt,
        "boxes_above_95_percent_area": huge_boxes_cnt,
        "boxes_touching_borders": border_touch_cnt
    }
    print(json.dumps(p5_data, indent=2))
    report["phase_5"] = p5_data

    # ----------------------------------------------------
    # PHASE 6: HARD NEGATIVES
    # ----------------------------------------------------
    print("\n--- PHASE 6: HARD NEGATIVES ---")
    bg_images = []
    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        s_lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")
        lbls = glob.glob(os.path.join(s_lbl_dir, "*.txt"))
        for lpath in lbls:
            with open(lpath, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                stem = os.path.splitext(os.path.basename(lpath))[0]
                ipath = os.path.join(s_img_dir, stem + ".jpg")
                bg_images.append(ipath)

    bg_ratio = (len(bg_images) / float(total_imgs_all) * 100.0) if total_imgs_all > 0 else 0
    p6_data = {
        "hard_negative_count": len(bg_images),
        "total_images": total_imgs_all,
        "hard_negative_percentage": f"{bg_ratio:.2f}%",
        "sample_hard_negative_paths": [os.path.basename(p) for p in bg_images[:5]]
    }
    print(json.dumps(p6_data, indent=2))
    report["phase_6"] = p6_data

    # ----------------------------------------------------
    # PHASE 7: DATA LEAKAGE / SPLIT AUDIT
    # ----------------------------------------------------
    print("\n--- PHASE 7: DATA LEAKAGE / SPLIT AUDIT ---")
    split_hashes = defaultdict(dict)
    duplicate_count_within_splits = 0
    leakage_cross_splits = 0

    for split in splits:
        s_img_dir = os.path.join(UNIFIED_DIR, split, "images")
        imgs = glob.glob(os.path.join(s_img_dir, "*.*"))
        for ipath in imgs:
            h = get_file_md5(ipath)
            if h in split_hashes[split]:
                duplicate_count_within_splits += 1
            else:
                split_hashes[split][h] = ipath

    # Check cross-split leakage
    train_hashes = set(split_hashes["train"].keys())
    val_hashes = set(split_hashes["val"].keys())
    test_hashes = set(split_hashes["test"].keys())

    train_val_leak = train_hashes.intersection(val_hashes)
    train_test_leak = train_hashes.intersection(test_hashes)
    val_test_leak = val_hashes.intersection(test_hashes)

    leakage_cross_splits = len(train_val_leak) + len(train_test_leak) + len(val_test_leak)

    p7_data = {
        "duplicates_within_splits": duplicate_count_within_splits,
        "train_val_image_leaks": len(train_val_leak),
        "train_test_image_leaks": len(train_test_leak),
        "val_test_image_leaks": len(val_test_leak),
        "total_cross_split_leaks": leakage_cross_splits
    }
    print(json.dumps(p7_data, indent=2))
    report["phase_7"] = p7_data

    # ----------------------------------------------------
    # PHASE 8: DATASET SUMMARY & STATUS
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("--- PHASE 8: DATASET SUMMARY REPORT ---")
    print("="*80)

    status_phase1 = "PASS" if total_imgs_all > 0 else "FAIL"
    status_phase2 = "PASS" if invalid_rows == 0 and invalid_class_ids == 0 and out_of_range_coords == 0 else "FAIL"
    status_phase3 = "PASS" if len(class_instance_counts) == 6 else "WARNING"
    status_phase4 = "PASS" if len(saved_visuals) >= 6 else "FAIL"
    status_phase5 = "PASS" if p5_data["boxes_below_12x12_px"] == 0 else "WARNING"
    status_phase6 = "PASS" if 10.0 <= bg_ratio <= 25.0 else "WARNING"
    status_phase7 = "PASS" if leakage_cross_splits == 0 else "FAIL"

    summary_table = {
        "Phase 1: Dataset Structure": status_phase1,
        "Phase 2: YOLO Label Integrity": status_phase2,
        "Phase 3: Class Distribution": status_phase3,
        "Phase 4: Class-Mapping Visual Verification": status_phase4,
        "Phase 5: Bounding Box Quality": status_phase5,
        "Phase 6: Hard Negatives": status_phase6,
        "Phase 7: Data Leakage / Split Audit": status_phase7
    }

    print("\nAudit Category Status Summary:")
    for cat, stat in summary_table.items():
        print(f"  {cat:<45}: [{stat}]")

    report["phase_8_summary"] = summary_table

    # Save full json audit report
    report_path = os.path.join(OUTPUT_VISUAL_DIR, "audit_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit complete! Full JSON report written to: {report_path}")

    all_passed = all(v == "PASS" for v in summary_table.values())
    if all_passed:
        print("\n" + "*"*70)
        print("  DATASET AUDIT PASSED — READY FOR EARLY HAILO-8 FEASIBILITY GATE")
        print("*"*70)
    else:
        print("\n" + "!"*70)
        print("  AUDIT COMPLETED WITH WARNINGS/ISSUES — REVIEW SUMMARY ABOVE")
        print("!"*70)

if __name__ == "__main__":
    run_audit()
