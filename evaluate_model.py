import os
import sys
from ultralytics import YOLO

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards\data.yaml"
WEIGHTS_PATH = r"runs\detect\hazards_yolov8n\weights\best.pt"

def evaluate():
    print("=" * 65)
    print("  Evaluating Hazards Model on Aerial Validation Set")
    print("=" * 65)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] ERROR: Trained weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    model = YOLO(WEIGHTS_PATH)
    metrics = model.val(data=DATA_YAML, split="val", imgsz=640)

    print("\n" + "=" * 65)
    print("  Per-Class Evaluation Metrics (Aerial Val Split)")
    print("=" * 65)
    print(f"Overall mAP@50:    {metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95: {metrics.box.map:.4f}\n")

    names = metrics.names
    weak_classes = []
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        ap50 = metrics.box.ap50[i]
        cls_name = names.get(cls_id, f"Class_{cls_id}")
        print(f"  [{cls_id}] {cls_name:<20}: mAP50 = {ap50:.4f}")
        if ap50 < 0.50:
            weak_classes.append((cls_name, ap50))

    if weak_classes:
        print("\n[!] WEAK CLASSES IDENTIFIED (<0.50 mAP50):")
        for cls_name, ap50 in weak_classes:
            print(f"   - {cls_name}: {ap50:.4f}")
        print("[!] Recommendation: Increase data augmentation / sampling weight for these classes.")
    else:
        print("\n[+] All 7 hazard classes achieved mAP50 >= 0.50!")

if __name__ == "__main__":
    evaluate()
