import os
import sys

"""
Hailo-8 HEF Compilation Pipeline (Requires Hailo Dataflow Compiler in WSL2/Ubuntu)

Run inside WSL2 Ubuntu environment with Hailo DFC installed:
    python compile_hailo_hef.py --onnx best.onnx --calib-dir /mnt/c/Users/esakk/dataset/unified_hazards/val/images
"""

ALLS_OPTIMIZATION_SCRIPT = """
# Hailo Model Optimization Script (.alls)
# Configured specifically for low-class YOLOv8 models on Hailo-8
model_optimization_flavor(optimization_level=1, compression_level=0, batch_size=2)
performance_param(fps=30)
"""

def generate_alls_script(out_path="yolov8n_hazards.alls"):
    with open(out_path, "w") as f:
        f.write(ALLS_OPTIMIZATION_SCRIPT.strip() + "\n")
    print(f"[+] Generated Hailo optimization script: {out_path}")
    return out_path

def print_hailo_instructions():
    print("=" * 65)
    print("  Hailo-8 HEF Compilation Guide (Step 8)")
    print("=" * 65)
    print("""
1. Prerequisites:
   - WSL2 Ubuntu 20.04/22.04 or Native Ubuntu x86_64
   - Hailo Dataflow Compiler (DFC) installed from Hailo Developer Zone:
     https://hailo.ai/developer-zone/
   
2. Execution Commands (in WSL2):
   
   # Step A: Generate optimization script
   python compile_hailo_hef.py

   # Step B: Parse ONNX model
   hailomz parse yolov8n --ckpt best.onnx --hw-arch hailo8 --start-node-names images --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv

   # Step C: Compile to HEF with calibration
   hailomz compile --ckpt best.onnx --calib-path /mnt/c/Users/esakk/dataset/unified_hazards/val/images --yaml yolov8n.yaml --alls yolov8n_hazards.alls --hw-arch hailo8

3. Transfer hazards_yolov8n.hef to Raspberry Pi 5 AI HAT+:
   Move compiled .hef file to /home/pi/seraph_zenith/models/hazards_yolov8n.hef
""")
    print("=" * 65)

if __name__ == "__main__":
    generate_alls_script()
    print_hailo_instructions()
