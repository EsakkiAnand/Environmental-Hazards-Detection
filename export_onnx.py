import os
import sys
from ultralytics import YOLO

def get_weights_path():
    candidate_paths = [
        os.path.abspath(os.path.join("runs", "detect", "runs", "detect", "hazards_yolov8s", "weights", "best.pt")),
        os.path.abspath(os.path.join("runs", "detect", "runs", "detect", "hazards_yolov8s", "weights", "last.pt")),
        os.path.abspath(os.path.join("runs", "detect", "hazards_yolov8s", "weights", "best.pt")),
        os.path.abspath(os.path.join("runs", "detect", "hazards_yolov8n", "weights", "best.pt")),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return candidate_paths[0]

def export():
    weights_path = get_weights_path()
    print("=" * 65)
    print("  Exporting Trained Hazard Detection Checkpoint to ONNX")
    print(f"  Source Weights: {weights_path}")
    print("=" * 65)

    if not os.path.exists(weights_path):
        print(f"[!] ERROR: Checkpoint not found at {weights_path}")
        sys.exit(1)

    model = YOLO(weights_path)
    exported_file = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,
        imgsz=640
    )

    # Make target output copy in weights/ and workspace root for convenience
    weights_dir = os.path.abspath("weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    target_onnx_name = "hazards_yolov8s_best.onnx"
    onnx_weights_path = os.path.join(weights_dir, target_onnx_name)
    onnx_root_path = os.path.abspath(target_onnx_name)

    import shutil
    shutil.copy2(exported_file, onnx_weights_path)
    shutil.copy2(exported_file, onnx_root_path)

    print("\n" + "=" * 65)
    print(f"  ONNX Export Complete Successfully!")
    print(f"  Exported File: {exported_file}")
    print(f"  Saved File in weights/: {onnx_weights_path}")
    print(f"  Saved File in root: {onnx_root_path}")
    print("=" * 65)

    try:
        import onnx
        onnx_model = onnx.load(onnx_root_path)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model integrity check passed successfully!")
    except Exception as e:
        print(f"[!] Note: ONNX validation check: {e}")

if __name__ == "__main__":
    export()

