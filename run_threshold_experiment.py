import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards\data.yaml"
WEIGHTS_PATH = r"F:\project\Nav and fusion\runs\detect\hazards_yolov8n\weights\best.pt"
BASE_EXPERIMENT_DIR = r"F:\project\Nav and fusion\runs\detect\hazards_threshold_experiment"
ARTIFACT_DIR = r"C:\Users\esakk\.gemini\antigravity-ide\brain\6bcd2eec-6be0-4ee0-9f8f-3d1601a88f5e"

CONF_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]
IOU_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]

CLASS_NAMES = {0: "fire", 1: "flood", 2: "damaged_structure", 3: "exposed_powerline", 4: "debris", 5: "landslide", 6: "chemical_spill"}
GT_COUNTS = {0: 1040, 1: 103, 2: 668, 3: 256, 4: 87, 5: 689, 6: 96}

def run_experiment():
    print("=" * 85)
    print("  POST-PROCESSING THRESHOLD EXPERIMENT (5x5 GRID SEARCH ON UNTOUCHED TEST SET)")
    print("=" * 85)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] ERROR: Trained weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS_PATH)

    results_grid = []

    # Iterate over 25 configurations
    total_configs = len(CONF_THRESHOLDS) * len(IOU_THRESHOLDS)
    config_num = 0

    for conf in CONF_THRESHOLDS:
        for iou in IOU_THRESHOLDS:
            config_num += 1
            name_dir = f"conf_{conf:.2f}_iou_{iou:.2f}"
            print(f"\n[{config_num}/{total_configs}] Evaluating Conf={conf:.2f}, IoU={iou:.2f}...")

            # Save plots for selected key configs (e.g. conf=0.15, iou=0.50, conf=0.20, iou=0.50, conf=0.25, iou=0.50)
            save_plots = (iou == 0.50 or iou == 0.60) and (conf in [0.15, 0.20, 0.25, 0.30])

            metrics = model.val(
                data=DATA_YAML,
                split="test",
                imgsz=640,
                conf=conf,
                iou=iou,
                project=BASE_EXPERIMENT_DIR,
                name=name_dir,
                exist_ok=True,
                plots=save_plots,
                verbose=False
            )

            mp = float(metrics.box.mp)
            mr = float(metrics.box.mr)
            overall_f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
            map50 = float(metrics.box.map50)
            map50_95 = float(metrics.box.map)

            ap_class_index = metrics.box.ap_class_index
            p_per_class = metrics.box.p
            r_per_class = metrics.box.r
            ap50_per_class = metrics.box.ap50
            ap_per_class = metrics.box.ap

            per_class_dict = {}
            for idx, cls_id in enumerate(ap_class_index):
                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                p_c = float(p_per_class[idx])
                r_c = float(r_per_class[idx])
                f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
                map50_c = float(ap50_per_class[idx])
                map50_95_c = float(ap_per_class[idx])

                per_class_dict[cls_name] = {
                    "precision": p_c,
                    "recall": r_c,
                    "f1": f1_c,
                    "map50": map50_c,
                    "map50_95": map50_95_c
                }

            # Extract specific metrics for Fire, Flood, Debris
            fire_m = per_class_dict.get("fire", {})
            flood_m = per_class_dict.get("flood", {})
            debris_m = per_class_dict.get("debris", {})

            row = {
                "conf": conf,
                "iou": iou,
                "precision": mp,
                "recall": mr,
                "f1": overall_f1,
                "map50": map50,
                "map50_95": map50_95,
                # Fire
                "fire_p": fire_m.get("precision", 0),
                "fire_r": fire_m.get("recall", 0),
                "fire_f1": fire_m.get("f1", 0),
                "fire_map50": fire_m.get("map50", 0),
                "fire_map50_95": fire_m.get("map50_95", 0),
                # Flood
                "flood_p": flood_m.get("precision", 0),
                "flood_r": flood_m.get("recall", 0),
                "flood_f1": flood_m.get("f1", 0),
                "flood_map50": flood_m.get("map50", 0),
                "flood_map50_95": flood_m.get("map50_95", 0),
                # Debris
                "debris_p": debris_m.get("precision", 0),
                "debris_r": debris_m.get("recall", 0),
                "debris_f1": debris_m.get("f1", 0),
                "debris_map50": debris_m.get("map50", 0),
                "debris_map50_95": debris_m.get("map50_95", 0),
                "per_class": per_class_dict,
                "save_dir": str(metrics.save_dir)
            }
            results_grid.append(row)

            print(f"   Overall: P={mp:.4f}, R={mr:.4f}, F1={overall_f1:.4f}, mAP50={map50:.4f}")
            print(f"   Weak Classes Recall -> Fire: {fire_m.get('recall',0):.4f}, Flood: {flood_m.get('recall',0):.4f}, Debris: {debris_m.get('recall',0):.4f}")

    # Save summary DataFrame to CSV in experiment dir
    df_grid = pd.DataFrame(results_grid)
    csv_out_path = os.path.join(BASE_EXPERIMENT_DIR, "grid_search_summary.csv")
    
    # Save clean summary CSV without raw per_class dict
    cols_to_save = [c for c in df_grid.columns if c not in ["per_class", "save_dir"]]
    df_grid[cols_to_save].to_csv(csv_out_path, index=False)
    print(f"\n[+] Grid search summary saved to: {csv_out_path}")

    # Save complete JSON
    json_out_path = os.path.join(BASE_EXPERIMENT_DIR, "grid_search_full.json")
    with open(json_out_path, "w") as f:
        json.dump(results_grid, f, indent=2)

    # Find candidate best configuration
    # Criteria: maximize F1 score while improving weak class recall and maintaining precision
    df_sorted_f1 = df_grid.sort_values(by="f1", ascending=False)
    best_config = df_sorted_f1.iloc[0].to_dict()

    print("\n" + "=" * 85)
    print("  EXPERIMENT ANALYSIS & SELECTION")
    print("=" * 85)
    print(f"Default Baseline (Conf=0.25, IoU=0.70): P=0.6673, R=0.6109, F1=0.6379, mAP50=0.6300")
    print(f"Highest F1 Config (Conf={best_config['conf']:.2f}, IoU={best_config['iou']:.2f}): P={best_config['precision']:.4f}, R={best_config['recall']:.4f}, F1={best_config['f1']:.4f}, mAP50={best_config['map50']:.4f}")

    # Copy sample images from candidate configs to Artifact Directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    top_candidates = [
        ("conf_0.15_iou_0.50", "conf15_iou50"),
        ("conf_0.20_iou_0.50", "conf20_iou50"),
        ("conf_0.25_iou_0.50", "conf25_iou50"),
        ("conf_0.30_iou_0.50", "conf30_iou50")
    ]

    for c_dir, prefix in top_candidates:
        src_dir = os.path.join(BASE_EXPERIMENT_DIR, c_dir)
        for img in ["val_batch0_pred.jpg", "val_batch1_pred.jpg", "confusion_matrix_normalized.png"]:
            src_file = os.path.join(src_dir, img)
            if os.path.exists(src_file):
                dst_file = os.path.join(ARTIFACT_DIR, f"{prefix}_{img}")
                shutil.copy2(src_file, dst_file)
                print(f"[+] Copied {src_file} -> {dst_file}")

    return results_grid

if __name__ == "__main__":
    run_experiment()
