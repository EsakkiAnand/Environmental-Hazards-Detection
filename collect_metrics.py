import os
import sys
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards\data.yaml"
WEIGHTS_PATH = r"runs\detect\hazards_yolov8n\weights\best.pt"
RUN_DIR = r"runs\detect\hazards_yolov8n"
ARTIFACT_DIR = r"C:\Users\esakk\.gemini\antigravity-ide\brain\6bcd2eec-6be0-4ee0-9f8f-3d1601a88f5e"

def collect():
    print("=" * 65)
    print("  Collecting Comprehensive Metrics & Visual Artifacts")
    print("=" * 65)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] ERROR: Weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    # 1. Generate results.png from results.csv if missing
    results_csv = os.path.join(RUN_DIR, "results.csv")
    results_png = os.path.join(RUN_DIR, "results.png")
    if os.path.exists(results_csv):
        print("[+] Generating results.png curve plot from results.csv...")
        try:
            plot_results(results_csv)
            print("[+] results.png created successfully.")
        except Exception as e:
            print(f"[!] Warning plotting results.png: {e}")

    # 2. Evaluate model metrics
    model = YOLO(WEIGHTS_PATH)
    metrics = model.val(data=DATA_YAML, split="val", imgsz=640, plots=True)

    print("\n" + "=" * 65)
    print("  OVERALL METRICS SUMMARY")
    print("=" * 65)
    print(f"Overall Precision (mp):    {metrics.box.mp:.4f}")
    print(f"Overall Recall (mr):       {metrics.box.mr:.4f}")
    # Overall F1 score calculation
    mp, mr = metrics.box.mp, metrics.box.mr
    overall_f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0
    print(f"Overall F1 Score:          {overall_f1:.4f}")
    print(f"Overall mAP@50:            {metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95:         {metrics.box.map:.4f}")

    print("\n" + "=" * 65)
    print("  PER-CLASS METRICS BREAKDOWN")
    print("=" * 65)
    print(f"{'Class ID':<8} {'Class Name':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'mAP@50':<10} {'mAP@50-95':<10}")
    print("-" * 80)

    names = metrics.names
    per_class_summary = []

    for i, cls_id in enumerate(metrics.box.ap_class_index):
        cls_name = names.get(cls_id, f"Class_{cls_id}")
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        ap50 = metrics.box.ap50[i]
        ap50_95 = metrics.box.ap[i]  # mean AP across IoUs 0.5:0.95 for this class
        
        print(f"{cls_id:<8} {cls_name:<20} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {ap50:<10.4f} {ap50_95:<10.4f}")
        per_class_summary.append({
            "cls_id": cls_id,
            "cls_name": cls_name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap50,
            "map50_95": ap50_95
        })

    # 3. Copy visual artifacts to Artifact Directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images_to_copy = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg"
    ]

    print("\n" + "=" * 65)
    print("  COPYING IMAGE ARTIFACTS TO CONVERSATION ARTIFACT DIR")
    print("=" * 65)
    for img_name in images_to_copy:
        src = os.path.join(RUN_DIR, img_name)
        if os.path.exists(src):
            dst = os.path.join(ARTIFACT_DIR, img_name)
            shutil.copy2(src, dst)
            print(f"[+] Copied {img_name} -> {dst}")
        else:
            print(f"[!] Warning: Source image {src} does not exist.")

if __name__ == "__main__":
    collect()
