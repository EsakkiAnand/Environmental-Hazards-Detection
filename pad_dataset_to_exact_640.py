import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards_v2"
TARGET_SIZE = 640

def letterbox_image_and_labels(img, bboxes, target_size=TARGET_SIZE):
    """
    Pads image to exact (target_size x target_size) with letterbox padding.
    Updates normalized YOLO bounding box coordinates (x, y, w, h) with 100% mathematical precision.
    """
    h_orig, w_orig = img.shape[:2]
    
    scale = float(target_size) / float(max(w_orig, h_orig))
    new_w = int(round(w_orig * scale))
    new_h = int(round(h_orig * scale))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create target canvas
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # Update normalized bboxes
    updated_bboxes = []
    for bbox_str in bboxes:
        parts = bbox_str.strip().split()
        if len(parts) == 5:
            cid = parts[0]
            xc, yc, w, h = map(float, parts[1:])

            # Convert to absolute original pixels
            abs_xc = xc * w_orig
            abs_yc = yc * h_orig
            abs_w = w * w_orig
            abs_h = h * h_orig

            # Apply scale and padding offsets
            new_abs_xc = abs_xc * scale + pad_w
            new_abs_yc = abs_yc * scale + pad_h
            new_abs_w = abs_w * scale
            new_abs_h = abs_h * scale

            # Convert to new normalized coordinates (relative to 640x640)
            norm_xc = new_abs_xc / float(target_size)
            norm_yc = new_abs_yc / float(target_size)
            norm_w = new_abs_w / float(target_size)
            norm_h = new_abs_h / float(target_size)

            # Clamp coordinates to [0.0, 1.0]
            norm_xc = max(0.0, min(1.0, norm_xc))
            norm_yc = max(0.0, min(1.0, norm_yc))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            updated_bboxes.append(f"{cid} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")

    return canvas, updated_bboxes

def process_dataset():
    print(f"[*] Padding all images & labels in {UNIFIED_DIR} to exact ({TARGET_SIZE}x{TARGET_SIZE})...")
    splits = ["train", "val", "test"]
    processed_count = 0

    for split in splits:
        img_dir = os.path.join(UNIFIED_DIR, split, "images")
        lbl_dir = os.path.join(UNIFIED_DIR, split, "labels")

        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        print(f"\nProcessing split [{split.upper()}]: {len(img_files)} images...")

        for ipath in tqdm(img_files):
            stem = os.path.splitext(os.path.basename(ipath))[0]
            lpath = os.path.join(lbl_dir, stem + ".txt")

            img = cv2.imread(ipath)
            if img is None:
                continue

            h, w = img.shape[:2]
            if (h, w) == (TARGET_SIZE, TARGET_SIZE):
                continue

            bboxes = []
            if os.path.exists(lpath):
                with open(lpath, "r") as f:
                    bboxes = [line.strip() for line in f if line.strip()]

            padded_img, new_bboxes = letterbox_image_and_labels(img, bboxes, target_size=TARGET_SIZE)

            cv2.imwrite(ipath, padded_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lpath, "w") as f:
                f.write("\n".join(new_bboxes) + ("\n" if new_bboxes else ""))

            processed_count += 1

    print("\n" + "="*70)
    print(f"  [SUCCESS] Dataset Exact ({TARGET_SIZE}x{TARGET_SIZE}) Padding Complete!")
    print(f"  Total Images Padded: {processed_count}")
    print("="*70)

if __name__ == "__main__":
    process_dataset()
