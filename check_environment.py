import sys
import os

def check_env():
    print("=" * 60)
    print("  Seraph Zenith Mark 1 — Environment Verification")
    print("=" * 60)
    
    print(f"[+] Python Executable: {sys.executable}")
    print(f"[+] Python Version:    {sys.version.split()[0]}")
    
    # 1. PyTorch & CUDA Check
    try:
        import torch
        print(f"[+] PyTorch Version:   {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[+] CUDA Available:    {cuda_avail}")
        if cuda_avail:
            print(f"[+] GPU Device Count:  {torch.cuda.device_count()}")
            print(f"[+] Active GPU Name:   {torch.cuda.get_device_name(0)}")
            print(f"[+] CUDA Version:      {torch.version.cuda}")
            # Quick VRAM check
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)
            print(f"[+] Total GPU VRAM:    {vram_gb} GB")
        else:
            print("[!] WARNING: CUDA is NOT available to PyTorch. Training will default to CPU (slow).")
    except ImportError as e:
        print(f"[!] ERROR: PyTorch import failed: {e}")
        return False

    # 2. Ultralytics YOLO Check
    try:
        import ultralytics
        print(f"[+] Ultralytics Ver:   {ultralytics.__version__}")
    except ImportError as e:
        print(f"[!] ERROR: Ultralytics import failed: {e}")
        return False

    # 3. OpenCV Check
    try:
        import cv2
        print(f"[+] OpenCV Version:    {cv2.__version__}")
    except ImportError as e:
        print(f"[!] ERROR: OpenCV import failed: {e}")
        return False

    print("=" * 60)
    print("[SUCCESS] Environment check passed cleanly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = check_env()
    if not success:
        sys.exit(1)
