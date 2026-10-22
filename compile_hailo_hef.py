"""
Hailo-8 HEF Compilation Pipeline (Requires Hailo Dataflow Compiler in WSL2/Ubuntu)
Seraph Zenith Mark 1 — v2 with Explicit NMS Postprocessing

Run inside WSL2 Ubuntu environment with Hailo DFC installed:
    python compile_hailo_hef.py --onnx hazards_yolov8s_best.onnx --calib-dir /mnt/c/Users/esakk/dataset/unified_hazards_v2/val/images
"""

import os
import sys

ALLS_OPTIMIZATION_SCRIPT = """# Hailo Model Optimization Script (.alls)
# Seraph Zenith Mark 1 — 6-class YOLOv8 hazard model on Hailo-8
model_optimization_flavor(optimization_level=1, compression_level=0, batch_size=2)
performance_param(fps=30)

# Explicit hardware-level NMS post-processing configuration
# Prevents low-confidence noise from entering the output tensor (6, 5, 100)
nms_postprocess(meta_arch=yolov8, engine=cpu, nms_scores_th=0.40, nms_iou_th=0.45, classes=6)
"""

def generate_alls_script(out_path="yolov8n_hazards.alls"):
    with open(out_path, "w") as f:
        f.write(ALLS_OPTIMIZATION_SCRIPT.strip() + "\n")
    print(f"[+] Generated Hailo optimization script: {out_path}")
    return out_path

def print_hailo_instructions():
    print("=" * 65)
    print("  Hailo-8 HEF Compilation Guide (Phase 5)")
    print("=" * 65)
    print("""
1. Prerequisites:
   - WSL2 Ubuntu 20.04/22.04 or Native Ubuntu x86_64
   - Hailo Dataflow Compiler (DFC) installed from Hailo Developer Zone:
     https://hailo.ai/developer-zone/
   
2. Execution Commands (in WSL2):
   
   # Step A: Generate optimization script
   python compile_hailo_hef.py

   # Step B: Parse ONNX model (6-class YOLOv8s)
   hailomz parse yolov8s --ckpt hazards_yolov8s_best.onnx --hw-arch hailo8

   # Step C: Compile to HEF with calibration and explicit NMS filtering
   hailomz compile --ckpt hazards_yolov8s_best.onnx --calib-path /mnt/c/Users/esakk/dataset/unified_hazards_v2/val/images --yaml yolov8s.yaml --alls yolov8n_hazards.alls --hw-arch hailo8

3. Transfer compiled HEF to Raspberry Pi 5:
   Move hazards_yolov8s.hef to /home/pi/models/hazards_yolov8n.hef
""")
    print("=" * 65)

if __name__ == "__main__":
    generate_alls_script()
    print_hailo_instructions()
