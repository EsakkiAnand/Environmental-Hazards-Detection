import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import gc
import cv2
import torch
import torch.serialization as torch_ser
import pathlib
from PIL import Image
import numpy as np

# Disable cuDNN benchmarking to prevent CUDNN_STATUS_EXECUTION_FAILED on Windows GPUs
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# Monkey-patch cv2.resize with PIL fallback for 100% memory safety resilience
_orig_cv2_resize = cv2.resize

def safe_cv2_resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR):
    try:
        return _orig_cv2_resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)
    except Exception:
        # Robust PIL fallback if OpenCV C++ memory allocator raises error
        pil_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        resized_pil = pil_img.resize(dsize, Image.BILINEAR)
        return cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

cv2.resize = safe_cv2_resize

# Deep monkey-patch of torch.save for PyTorch 2.5.1 on Windows
_orig_torch_save = torch_ser.save

def safe_torch_save(obj, f, *args, **kwargs):
    if hasattr(f, "__fspath__") or isinstance(f, (str, pathlib.Path)):
        f = str(f)
        parent_dir = os.path.dirname(f)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    for i in range(5):
        try:
            return _orig_torch_save(obj, f, *args, **kwargs)
        except (RuntimeError, ValueError, PermissionError, OSError) as e:
            if i == 4:
                raise e
            time.sleep((2 ** i) * 0.5)

torch.save = safe_torch_save
torch_ser.save = safe_torch_save

import ultralytics.utils.patches as patches
patches.torch_save = safe_torch_save

import ultralytics.engine.trainer as trainer_mod
if hasattr(trainer_mod, "torch"):
    trainer_mod.torch.save = safe_torch_save

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# Limit OpenCV internal thread buffer allocation
cv2.setNumThreads(0)

DATA_YAML = r"C:\Users\esakk\dataset\unified_hazards_v2\data.yaml"
MODEL_PRETRAINED = "yolov8s.pt"

# Ultralytics callback to flush garbage collection after validation passes
def on_val_end_cleanup(validator):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_yolov8s_hazards(epochs=50, batch_size=4, imgsz=640, workers=0):
    print("=" * 65)
    print("  Starting Training - Unified 6-Class Hazards YOLOv8s Model")
    print("  Dataset: C:\\Users\\esakk\\dataset\\unified_hazards_v2\\data.yaml")
    print("=" * 65)

    if not os.path.exists(DATA_YAML):
        print(f"[!] ERROR: data.yaml not found at {DATA_YAML}.")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Selected training device: {device} (CUDA Available: {torch.cuda.is_available()})", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[!] CUDA sync note: {e}")

    _orig_load_checkpoint_state = BaseTrainer._load_checkpoint_state
    def _safe_load_checkpoint_state(self, ckpt):
        if ckpt and "optimizer" in ckpt:
            ckpt.pop("optimizer", None)
        return _orig_load_checkpoint_state(self, ckpt)
    BaseTrainer._load_checkpoint_state = _safe_load_checkpoint_state

    checkpoint_path = os.path.abspath(os.path.join("runs", "detect", "hazards_yolov8s", "weights", "last.pt"))
    nested_checkpoint = os.path.abspath(os.path.join("runs", "detect", "runs", "detect", "hazards_yolov8s", "weights", "last.pt"))
    if not os.path.exists(checkpoint_path) and os.path.exists(nested_checkpoint):
        checkpoint_path = nested_checkpoint

    if os.path.exists(checkpoint_path):
        print(f"[+] Existing checkpoint found: {checkpoint_path}")
        print(f"[+] Resuming training from last checkpoint...")
        model = YOLO(checkpoint_path)
        model.add_callback("on_val_end", on_val_end_cleanup)
        results = model.train(resume=True)
    else:
        print(f"[+] Initializing new training run with {MODEL_PRETRAINED}...")
        model = YOLO(MODEL_PRETRAINED)
        model.add_callback("on_val_end", on_val_end_cleanup)

        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            cache=False,
            device=device,
            name="hazards_yolov8s",
            project="runs/detect",
            exist_ok=True,
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
            rect=False,
            val=True,
            plots=False,
            save=True
        )

    print("\n" + "=" * 65)
    print("  YOLOv8s Hazard Model Training Complete!")
    print(f"  Best Weights: runs/detect/hazards_yolov8s/weights/best.pt")
    print("=" * 65)

if __name__ == "__main__":
    train_yolov8s_hazards()
