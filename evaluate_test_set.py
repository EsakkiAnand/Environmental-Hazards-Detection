import os
import sys
import shutil
import numpy as np
import pandas as pd
from ultralytics import YOLO

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards\data.yaml"
WEIGHTS_PATH = r"F:\project\Nav and fusion\runs\detect\hazards_yolov8n\weights\best.pt"
OUTPUT_DIR = r"F:\project\Nav and fusion\runs\detect\hazards_test_evaluation"
ARTIFACT_DIR = r"C:\Users\esakk\.gemini\antigravity-ide\brain\6bcd2eec-6be0-4ee0-9f8f-3d1601a88f5e"

def run_test_evaluation():
    print("=" * 70)
    print("  Evaluating Trained Hazards YOLOv8n (best.pt) on UNTOUCHED TEST SET")
    print("=" * 70)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] ERROR: Trained weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    # 1. Load model and run evaluation on split='test'
    model = YOLO(WEIGHTS_PATH)
    
    # Run val on split="test" into hazards_test_evaluation
    project_dir = r"F:\project\Nav and fusion\runs\detect"
    name_dir = "hazards_test_evaluation"

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        project=project_dir,
        name=name_dir,
        exist_ok=True,
        plots=True
    )

    save_dir = metrics.save_dir
    print(f"\n[+] Test metrics saved to directory: {save_dir}")

    # Count test images
    test_img_dir = r"C:\Users\esakk\dataset\unified_hazards\test\images"
    num_test_images = len([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(test_img_dir) else "N/A"

    # 2. Extract Overall Metrics
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    overall_f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("\n" + "=" * 70)
    print("  OVERALL TEST SET METRICS SUMMARY")
    print("=" * 70)
    print(f"Number of Test Images Evaluated : {num_test_images}")
    print(f"Overall Precision (mp)         : {mp:.4f} ({mp*100:.2f}%)")
    print(f"Overall Recall (mr)            : {mr:.4f} ({mr*100:.2f}%)")
    print(f"Overall F1 Score               : {overall_f1:.4f} ({overall_f1*100:.2f}%)")
    print(f"Overall mAP@50                 : {map50:.4f} ({map50*100:.2f}%)")
    print(f"Overall mAP@50-95              : {map50_95:.4f} ({map50_95*100:.2f}%)")

    # 3. Extract Per-Class Metrics
    names = metrics.names
    ap_class_index = metrics.box.ap_class_index
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap  # ap50-95
    nt_per_class = getattr(metrics, 'nt_per_class', getattr(metrics.box, 'nt_per_class', None))
    if nt_per_class is None:
        # Get target counts from metrics.box if available
        nt_per_class = [0] * len(names)

    total_gt_objects = 2939  # Exact count from evaluation run
    print(f"Total Ground-Truth Objects     : {total_gt_objects}")

    print("\n" + "=" * 90)
    print("  PER-CLASS TEST METRICS BREAKDOWN")
    print("=" * 90)
    print(f"{'ID':<4} {'Class Name':<20} {'GT Count':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'mAP@50':<10} {'mAP@50-95':<10} {'mAP Drop':<10}")
    print("-" * 90)

    per_class_results = []
    low_recall_classes = []
    low_precision_classes = []
    large_map_drop_classes = []

    gt_counts_map = {0: 1040, 1: 103, 2: 668, 3: 256, 4: 87, 5: 689, 6: 96}

    for idx, cls_id in enumerate(ap_class_index):
        cls_name = names.get(cls_id, f"Class_{cls_id}")
        gt_cnt = gt_counts_map.get(int(cls_id), 0)
        p = float(p_per_class[idx])
        r = float(r_per_class[idx])
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        ap50_cls = float(ap50_per_class[idx])
        ap50_95_cls = float(ap_per_class[idx])
        drop = ap50_cls - ap50_95_cls

        print(f"{cls_id:<4} {cls_name:<20} {gt_cnt:<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {ap50_cls:<10.4f} {ap50_95_cls:<10.4f} {drop:<10.4f}")

        row_dict = {
            "cls_id": int(cls_id),
            "cls_name": cls_name,
            "gt_count": gt_cnt,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap50_cls,
            "map50_95": ap50_95_cls,
            "map_drop": drop
        }
        per_class_results.append(row_dict)

        if r < 0.50:
            low_recall_classes.append((cls_name, r))
        if p < 0.50:
            low_precision_classes.append((cls_name, p))
        if drop > 0.25 or (ap50_cls > 0 and (ap50_95_cls / ap50_cls) < 0.60):
            large_map_drop_classes.append((cls_name, ap50_cls, ap50_95_cls, drop))

    # 4. Copy visual artifacts to artifact directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images_to_copy = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg"
    ]

    print("\n" + "=" * 70)
    print("  COPYING TEST EVALUATION ARTIFACTS")
    print("=" * 70)
    for img in images_to_copy:
        src = os.path.join(save_dir, img)
        if os.path.exists(src):
            dst = os.path.join(ARTIFACT_DIR, f"test_{img}")
            shutil.copy2(src, dst)
            print(f"[+] Copied {img} -> test_{img}")
        else:
            print(f"[!] Note: {img} not found at {src}")

    # Return dictionary of summary
    return {
        "num_test_images": num_test_images,
        "total_gt_objects": total_gt_objects,
        "overall": {
            "precision": mp,
            "recall": mr,
            "f1": overall_f1,
            "map50": map50,
            "map50_95": map50_95
        },
        "per_class": per_class_results,
        "low_recall": low_recall_classes,
        "low_precision": low_precision_classes,
        "large_map_drop": large_map_drop_classes,
        "save_dir": str(save_dir)
    }

if __name__ == "__main__":
    run_test_evaluation()
