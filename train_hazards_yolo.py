import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import gc
import cv2
import torch

# Disable cuDNN benchmarking to prevent CUDNN_STATUS_EXECUTION_FAILED on Windows GPUs
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# Limit OpenCV internal thread buffer allocation to prevent host RAM memory leaks
cv2.setNumThreads(0)

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards\data.yaml"
MODEL_PRETRAINED = "yolov8n.pt"

def train_model(epochs=100, batch_size=4, imgsz=640, workers=0):
    print("=" * 65)
    print("  Starting Seraph Zenith Mark 1 - Hazards YOLOv8n Fine-Tuning")
    print("=" * 65)

    if not os.path.exists(DATA_YAML):
        print(f"[!] ERROR: data.yaml not found at {DATA_YAML}. Run build_unified_dataset.py first.")
        sys.exit(1)

    # Auto-detect GPU device vs CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Selected training device: {device} (CUDA Available: {torch.cuda.is_available()})", flush=True)

    gc.collect()
    time.sleep(3)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[!] CUDA sync note: {e}")

    # Patch BaseTrainer._load_checkpoint_state to safely handle optimizer param group mismatch during resume
    _orig_load_checkpoint_state = BaseTrainer._load_checkpoint_state
    def _safe_load_checkpoint_state(self, ckpt):
        if ckpt and "optimizer" in ckpt:
            ckpt.pop("optimizer", None)
        return _orig_load_checkpoint_state(self, ckpt)
    BaseTrainer._load_checkpoint_state = _safe_load_checkpoint_state

    # Check for existing checkpoint to resume training
    checkpoint_path = os.path.abspath(os.path.join("runs", "detect", "hazards_yolov8n", "weights", "last.pt"))
    if os.path.exists(checkpoint_path):
        print(f"[+] Existing checkpoint found: {checkpoint_path}")
        print(f"[+] Attempting to resume training from last checkpoint...")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
    else:
        print(f"[+] Starting new training run with {MODEL_PRETRAINED}...")
        model = YOLO(MODEL_PRETRAINED)

        # Train model
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            device=device,
            name="hazards_yolov8n",
            project="runs/detect",
            exist_ok=True,
            # Data augmentation hyperparameters optimized for aerial hazard scenes
            mosaic=1.0,
            mixup=0.15,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            fliplr=0.5,
            val=True,
            save=True
        )

    print("\n" + "=" * 65)
    print("  Training Complete!")
    print(f"  Best Weights Saved At: runs/detect/hazards_yolov8n/weights/best.pt")
    print("=" * 65)

if __name__ == "__main__":
    train_model()

