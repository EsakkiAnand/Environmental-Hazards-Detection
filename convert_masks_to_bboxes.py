import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def convert_mask_image_to_yolo_bboxes(mask_path, target_class_id, min_area=100):
    """
    Reads a binary/grayscale segmentation mask image and extracts YOLO bounding boxes.
    Normalized format: class_id x_center y_center width height
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    height, width = mask.shape
    # Threshold mask to binary (non-zero pixels are foreground)
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Find contours around target regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate normalized YOLO center coordinates and dimensions
        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        norm_w = w / float(width)
        norm_h = h / float(height)

        # Clamp values between 0.0 and 1.0
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        bboxes.append(f"{target_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return bboxes

def convert_polygon_json_to_yolo_bboxes(polygons, img_width, img_height, target_class_id):
    """
    Converts polygon vertex lists [[x1, y1], [x2, y2], ...] into YOLO bounding box format.
    """
    bboxes = []
    for poly in polygons:
        poly_np = np.array(poly, dtype=np.float32)
        x_min = np.min(poly_np[:, 0])
        x_max = np.max(poly_np[:, 0])
        y_min = np.min(poly_np[:, 1])
        y_max = np.max(poly_np[:, 1])

        w = x_max - x_min
        h = y_max - y_min

        x_center = (x_min + w / 2.0) / img_width
        y_center = (y_min + h / 2.0) / img_height
        norm_w = w / float(img_width)
        norm_h = h / float(img_height)

        bboxes.append(f"{target_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return bboxes

def process_rescuenet_masks(raw_rescuenet_dir, output_labels_dir, debris_class_id=3):
    """
    Batch converts RescueNet debris masks into YOLO format label files.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    mask_files = glob.glob(os.path.join(raw_rescuenet_dir, "**", "*mask*.png"), recursive=True)
    print(f"[*] Processing {len(mask_files)} RescueNet mask files...")

    converted_count = 0
    for mask_path in tqdm(mask_files):
        bboxes = convert_mask_image_to_yolo_bboxes(mask_path, debris_class_id)
        if bboxes:
            base_name = os.path.splitext(os.path.basename(mask_path))[0].replace("_mask", "")
            out_txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")
            with open(out_txt_path, "w") as f:
                f.write("\n".join(bboxes) + "\n")
            converted_count += 1

    print(f"[+] Converted {converted_count} RescueNet masks to YOLO labels.")

if __name__ == "__main__":
    print("Mask-to-Bbox Conversion Utility ready.")
