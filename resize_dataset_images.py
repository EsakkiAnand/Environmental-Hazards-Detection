import os
import glob
import cv2
from tqdm import tqdm
from PIL import Image

UNIFIED_DIR = r"C:\Users\esakk\dataset\unified_hazards_v2"
MAX_DIM = 1280 # Standardize high-res UAV images to max 1280px (preserves quality while 10x memory reduction)

def resize_images():
    print(f"[*] Standardizing image resolutions in {UNIFIED_DIR} to max dimension {MAX_DIM}px...")
    
    splits = ["train", "val", "test"]
    resized_count = 0
    total_scanned = 0

    for split in splits:
        img_dir = os.path.join(UNIFIED_DIR, split, "images")
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

        print(f"\nProcessing split [{split.upper()}]: {len(img_files)} images...")
        for ipath in tqdm(img_files):
            total_scanned += 1
            try:
                # Fast header check using PIL without loading full image into uncompressed array
                with Image.open(ipath) as im:
                    w, h = im.size

                if max(w, h) > MAX_DIM:
                    img = cv2.imread(ipath)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        scale = float(MAX_DIM) / float(max(img_w, img_h))
                        new_w = int(img_w * scale)
                        new_h = int(img_h * scale)

                        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(ipath, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        resized_count += 1
            except Exception as e:
                print(f"Error processing {ipath}: {e}")

    print("\n" + "="*70)
    print(f"  [SUCCESS] Resolution Standardization Complete!")
    print(f"  Total Scanned Images: {total_scanned}")
    print(f"  Total Resized High-Res Images (> {MAX_DIM}px): {resized_count}")
    print("="*70)

if __name__ == "__main__":
    resize_images()
