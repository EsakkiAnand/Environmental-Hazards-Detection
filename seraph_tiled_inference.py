#!/usr/bin/env python3
"""
seraph_tiled_inference.py
Seraph Zenith Mark 1 — Raspberry Pi 5 / Hailo-8 Tiled Inference Script

This script is designed to REPLACE the current undocumented Pi-side inference
script that generated run3.log. It implements the same 6-tile inference
strategy but with:
  - Full Hailo (6,5,100) tensor decoding
  - Cross-tile NMS in full-frame coordinate space
  - Configurable confidence threshold
  - Structured per-frame logging (class, score, tile, bbox)

USAGE (on Raspberry Pi 5):
  python seraph_tiled_inference.py \
      --hef /home/pi/seraph_zenith/models/hazards_yolov8n.hef \
      --conf 0.40 \
      --iou  0.45 \
      --output flight4.mp4 \
      --log   run4.log

IMPORTANT BEFORE DEPLOYING:
  1. Verify that HAILO_COORD_ORDER matches your DFC version:
       'yxyx' = [y1, x1, y2, x2, score]  (Hailo DFC <= 3.27 default)
       'xyxy' = [x1, y1, x2, y2, score]  (some DFC versions)
     Run verify_coord_order() on a known-good frame to confirm.
  2. Verify that classes.txt on the Pi has EXACTLY 6 lines (no chemical_spill).
  3. Copy seraph_nms.py to the same directory as this script.
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# seraph_nms is copied from dashboard/ai/seraph_nms.py
# On the Pi, place it in the same directory as this script.
# ---------------------------------------------------------------------------
try:
    from seraph_nms import (
        HAILO_CLASS_NAMES,
        decode_hailo_nms_output,
        cross_tile_nms,
        apply_conf_filter,
        build_tile_grid,
        log_detection_summary,
    )
except ImportError:
    print("[ERROR] seraph_nms.py not found. Copy it from dashboard/ai/seraph_nms.py")
    sys.exit(1)

try:
    import hailo_platform as hailo
    HAILO_OK = True
except ImportError:
    print("[WARN] hailo_platform not available — exiting (Pi-only script)")
    HAILO_OK = False

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Coordinate order for the Hailo NMS output tensor axis-1.
# Confirmed value depends on the DFC version used to compile the HEF.
# Options: 'yxyx'  →  [y1_norm, x1_norm, y2_norm, x2_norm, score]
#          'xyxy'  →  [x1_norm, y1_norm, x2_norm, y2_norm, score]
# DEFAULT: 'yxyx' (standard for Hailo DFC 3.x / RPi AI HAT+)
HAILO_COORD_ORDER = "yxyx"

# Tile grid: 2 rows x 3 columns = 6 tiles (matching run3.log tiles=6)
TILE_GRID_ROWS = 2
TILE_GRID_COLS = 3
TILE_OVERLAP   = 0.15   # 15% overlap between adjacent tiles

# Camera capture resolution (IMX296 on RPi 5)
CAMERA_W = 1456
CAMERA_H = 1088

# Model input resolution
MODEL_W = 640
MODEL_H = 640

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seraph.inference")


def load_hailo_engine(hef_path: str):
    """Load Hailo VDevice and configure network group."""
    target = hailo.VDevice()
    hef    = hailo.Hef(hef_path)
    cfg    = hailo.ConfigureParams.create_from_hef(
        hef, interface=hailo.HailoStreamInterface.PCIe
    )
    ng = target.configure(hef, cfg)[0]
    in_info  = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()

    logger.info(f"HEF input : {[(i.name, i.shape) for i in [in_info]]}")
    logger.info(f"HEF output: {[(o.name, o.shape) for o in out_info]}")
    return target, ng, in_info, out_info


def infer_tile(
    tile_crop: np.ndarray,
    ng, in_info, out_info,
    conf_thresh: float,
    tile_meta: Dict,
) -> List[Dict[str, Any]]:
    """
    Run HEF inference on a single tile crop and return detections
    in FULL-FRAME pixel coordinates.

    Parameters
    ----------
    tile_crop  : BGR crop of the tile (frame_tile_w x frame_tile_h)
    tile_meta  : dict from build_tile_grid(): {tile_id, x, y, w, h}
    """
    # Resize crop to model input
    resized = cv2.resize(tile_crop, (MODEL_W, MODEL_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp     = {in_info.name: np.expand_dims(rgb, axis=0)}

    with hailo.InferVStreams(ng, in_info, out_info) as pipe:
        raw = pipe.infer(inp)

    tensor = list(raw.values())[0]   # shape (1,6,5,100) or (6,5,100)

    # decode_hailo_nms_output maps from tile pixel space → full-frame space
    dets = decode_hailo_nms_output(
        raw_output      = tensor,
        frame_w         = CAMERA_W,
        frame_h         = CAMERA_H,
        conf_thresh     = conf_thresh,
        tile_id         = tile_meta["tile_id"],
        tile_x_offset   = tile_meta["x"],
        tile_y_offset   = tile_meta["y"],
        frame_tile_w    = tile_meta["w"],
        frame_tile_h    = tile_meta["h"],
    )
    return dets


def run_frame(
    frame: np.ndarray,
    ng, in_info, out_info,
    tile_grid: List[Dict],
    conf_thresh: float,
    iou_thresh: float,
) -> List[Dict[str, Any]]:
    """
    Full tiled inference pipeline for one frame.

    Returns final deduplicated detections in full-frame pixel coordinates.
    """
    all_raw: List[Dict] = []

    for tile_meta in tile_grid:
        x, y, w, h = tile_meta["x"], tile_meta["y"], tile_meta["w"], tile_meta["h"]
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        tile_dets = infer_tile(crop, ng, in_info, out_info, conf_thresh, tile_meta)
        all_raw.extend(tile_dets)

    # --- Cross-tile NMS (all tiles, full-frame coordinates) ---
    # IMPORTANT: tile_dets are already in full-frame coords (done in infer_tile)
    kept, n_suppressed = cross_tile_nms(all_raw, iou_threshold=iou_thresh)

    log_detection_summary(
        raw_count      = len(all_raw),
        after_conf     = len(all_raw),   # conf filter already in decode_hailo_nms_output
        after_nms      = len(kept),
        n_suppressed   = n_suppressed,
        n_rejected_conf= 0,
    )

    return kept


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    COLOR_MAP = {
        "fire":               (0,   69,  255),
        "flood":              (255, 128,   0),
        "damaged_structure":  (0,   165, 255),
        "exposed_powerline":  (0,   255, 255),
        "debris":             (128, 128, 128),
        "landslide":          (19,   69, 139),
    }
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        color = COLOR_MAP.get(d["class_name"], (0, 0, 255))
        label = f"{d['class_name'].upper()} {d['score']:.2f} t{d.get('tile_id','?')}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return annotated


def verify_coord_order(frame: np.ndarray, ng, in_info, out_info, hef_path: str):
    """
    Helper: run one inference and print raw tensor values so you can manually
    verify whether axis-1 is [y1,x1,y2,x2,score] or [x1,y1,x2,y2,score].
    Run this once on a frame that contains a clearly visible object.
    """
    resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp = {in_info.name: np.expand_dims(rgb, axis=0)}
    with hailo.InferVStreams(ng, in_info, out_info) as pipe:
        raw = pipe.infer(inp)
    tensor = list(raw.values())[0]
    if tensor.ndim == 4:
        tensor = tensor[0]
    print("\n=== RAW HEF OUTPUT (first 5 non-zero detections) ===")
    found = 0
    for cls_id in range(tensor.shape[0]):
        for det_idx in range(tensor.shape[2]):
            score = tensor[cls_id, 4, det_idx]
            if score > 0.1:
                vals = tensor[cls_id, :, det_idx]
                print(f"  cls={cls_id} ({HAILO_CLASS_NAMES[cls_id]}) "
                      f"fields={[f'{v:.4f}' for v in vals]}")
                found += 1
            if found >= 5:
                break
        if found >= 5:
            break
    print("If field[0] and field[1] look like (y, x) coords: set HAILO_COORD_ORDER='yxyx'")
    print("If field[0] and field[1] look like (x, y) coords: set HAILO_COORD_ORDER='xyxy'")


def main():
    parser = argparse.ArgumentParser(description="Seraph Zenith Mark 1 — Tiled Inference")
    parser.add_argument("--hef",    required=True,  help="Path to .hef file")
    parser.add_argument("--source", default="0",    help="Camera index or video file path")
    parser.add_argument("--conf",   type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.45, help="Cross-tile NMS IoU threshold")
    parser.add_argument("--output", default=None,   help="Output video file path")
    parser.add_argument("--log",    default=None,   help="Log file path (default: stdout)")
    parser.add_argument("--verify", action="store_true",
                        help="Run coord-order verification on first frame then exit")
    args = parser.parse_args()

    if args.log:
        fh = logging.FileHandler(args.log)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
        logging.getLogger().addHandler(fh)

    if not HAILO_OK:
        sys.exit(1)

    logger.info(f"Loading HEF: {args.hef}")
    target, ng, in_info, out_info = load_hailo_engine(args.hef)

    # Build 2x3 tile grid for 1456x1088 frame
    tile_grid = build_tile_grid(
        frame_h   = CAMERA_H,
        frame_w   = CAMERA_W,
        grid_rows = TILE_GRID_ROWS,
        grid_cols = TILE_GRID_COLS,
        overlap   = TILE_OVERLAP,
    )
    logger.info(f"Tile grid: {TILE_GRID_ROWS}x{TILE_GRID_COLS} = {len(tile_grid)} tiles "
                f"(overlap={TILE_OVERLAP*100:.0f}%)")
    for t in tile_grid:
        logger.info(f"  tile {t['tile_id']}: x={t['x']} y={t['y']} w={t['w']} h={t['h']}")

    # Open video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {src}")
        sys.exit(1)

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, 22.0, (CAMERA_W, CAMERA_H))

    fps_counter, fps_timer = 0, time.time()
    current_fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            frame_count += 1

            # --- Optional coord-order verification ---
            if args.verify and frame_count == 1:
                verify_coord_order(frame, ng, in_info, out_info, args.hef)
                break

            # --- Tiled inference + cross-tile NMS ---
            detections = run_frame(
                frame, ng, in_info, out_info,
                tile_grid, args.conf, args.iou,
            )

            # --- FPS tracking ---
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            # --- Console log (mirrors run3.log format) ---
            cls_counts = Counter(d["class_name"] for d in detections)
            det_str = str(dict(cls_counts)) if cls_counts else "no detections"
            print(f"{current_fps:.1f} FPS  tiles={len(tile_grid)}  {det_str}")

            # --- Annotate and write ---
            annotated = draw_detections(frame, detections)
            cv2.putText(annotated,
                        f"FPS:{current_fps:.1f} CONF>={args.conf} NMS_IOU={args.iou}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if out_writer:
                out_writer.write(annotated)

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
            logger.info(f"saved video: {args.output}")


if __name__ == "__main__":
    main()
