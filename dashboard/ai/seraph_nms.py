"""
seraph_nms.py — Cross-Tile NMS and Detection Utilities
Seraph Zenith Mark 1

This module is intentionally standalone (no dashboard imports) so it can be
copied to the Raspberry Pi inference script without modification.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("seraph.nms")


# ---------------------------------------------------------------------------
# Detection dict schema
# ---------------------------------------------------------------------------
# {
#   "class_id"   : int,           # 0-5, matching HEF output axis-0 index
#   "class_name" : str,           # resolved from HAILO_CLASS_NAMES[class_id]
#   "score"      : float,         # post-NMS confidence from HEF (0.0–1.0)
#   "x1"         : int,           # full-frame pixel coordinates (top-left)
#   "y1"         : int,
#   "x2"         : int,           # bottom-right
#   "y2"         : int,
#   "tile_id"    : int | str,     # originating tile index or "full_frame"
# }

HAILO_CLASS_NAMES: List[str] = [
    "fire",               # 0
    "flood",              # 1
    "damaged_structure",  # 2
    "exposed_powerline",  # 3
    "debris",             # 4
    "landslide",          # 5
]


def tile_coords_to_frame(
    det_x1: int, det_y1: int, det_x2: int, det_y2: int,
    tile_x_offset: int, tile_y_offset: int,
    tile_w: int, tile_h: int,
    model_input_w: int = 640, model_input_h: int = 640,
    frame_tile_w: int = None, frame_tile_h: int = None,
) -> Tuple[int, int, int, int]:
    """
    Convert bounding-box coordinates from model-input space (640×640) back to
    full-frame pixel space.

    The pipeline is:
      full_frame → crop tile (frame_tile_w × frame_tile_h) → resize to 640×640 → HEF

    So the inverse is:
      model coords → divide by 640 → multiply by frame_tile size → add tile offset

    Parameters
    ----------
    det_x1, det_y1, det_x2, det_y2 : int
        Bounding box in 640×640 model-input pixel coordinates.
    tile_x_offset, tile_y_offset : int
        Top-left corner of the tile crop in the full frame (pixels).
    tile_w, tile_h : int
        Width and height of the tile crop in the full frame before resizing.
        If frame_tile_w/frame_tile_h are None these are used.
    model_input_w, model_input_h : int
        Model input size (default 640×640).
    frame_tile_w, frame_tile_h : int or None
        Actual pixel size of the tile crop in the full frame.
        Use these when tile_w/tile_h already equal model_input_w/h.

    Returns
    -------
    (x1, y1, x2, y2) in full-frame pixel coordinates.
    """
    ftw = frame_tile_w if frame_tile_w is not None else tile_w
    fth = frame_tile_h if frame_tile_h is not None else tile_h

    scale_x = ftw / model_input_w
    scale_y = fth / model_input_h

    x1 = int(det_x1 * scale_x) + tile_x_offset
    y1 = int(det_y1 * scale_y) + tile_y_offset
    x2 = int(det_x2 * scale_x) + tile_x_offset
    y2 = int(det_y2 * scale_y) + tile_y_offset
    return x1, y1, x2, y2


def _iou(a: Dict, b: Dict) -> float:
    """Intersection-over-Union for two detection dicts (full-frame coords)."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cross_tile_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Class-aware Non-Maximum Suppression across all tiles.

    All detections MUST already be in full-frame pixel coordinates.
    Detections from different classes are never suppressed against each other.

    Parameters
    ----------
    detections : list of detection dicts (schema above)
    iou_threshold : float
        Boxes with IoU >= this value are considered duplicates.
        The higher-scoring box is kept.

    Returns
    -------
    (kept, n_suppressed)
        kept        : filtered list of detection dicts
        n_suppressed: number of boxes removed
    """
    if not detections:
        return [], 0

    # Validate that all detections are in full-frame space
    # (simple sanity: reject zero-area boxes silently)
    valid = [d for d in detections
             if d["x2"] > d["x1"] and d["y2"] > d["y1"] and d["score"] > 0.0]
    if len(valid) < len(detections):
        dropped = len(detections) - len(valid)
        logger.warning(f"NMS: dropped {dropped} zero-area or zero-score boxes before NMS")

    # Group by class_id
    by_class: Dict[int, List[Dict]] = {}
    for d in valid:
        by_class.setdefault(d["class_id"], []).append(d)

    kept: List[Dict] = []
    total_suppressed = 0

    for cls_id, boxes in by_class.items():
        # Sort descending by score — highest confidence is kept first
        boxes.sort(key=lambda b: b["score"], reverse=True)
        suppressed = [False] * len(boxes)

        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            for j in range(i + 1, len(boxes)):
                if suppressed[j]:
                    continue
                iou_val = _iou(boxes[i], boxes[j])
                if iou_val >= iou_threshold:
                    suppressed[j] = True
                    total_suppressed += 1
                    logger.debug(
                        f"NMS: suppressed {boxes[j]['class_name']} "
                        f"(tile={boxes[j].get('tile_id','?')} "
                        f"score={boxes[j]['score']:.3f}) "
                        f"IoU={iou_val:.3f} vs kept "
                        f"(tile={boxes[i].get('tile_id','?')} "
                        f"score={boxes[i]['score']:.3f})"
                    )

    return kept, total_suppressed


def apply_conf_filter(
    detections: List[Dict[str, Any]],
    default_thresh: float = 0.40,
    per_class_thresh: Dict[str, float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter detections by per-class (or global) confidence threshold.

    Returns
    -------
    (accepted, rejected) — both as lists of detection dicts
    """
    per_class_thresh = per_class_thresh or {}
    accepted, rejected = [], []
    for d in detections:
        thresh = per_class_thresh.get(d["class_name"], default_thresh)
        if d["score"] >= thresh:
            accepted.append(d)
        else:
            rejected.append(d)
            logger.debug(
                f"CONF_FILTER: rejected {d['class_name']} score={d['score']:.3f} "
                f"< thresh={thresh:.2f} tile={d.get('tile_id','?')} "
                f"bbox=({d['x1']},{d['y1']},{d['x2']},{d['y2']})"
            )
    return accepted, rejected


def decode_hailo_nms_output(
    raw_output,           # numpy array shape (6, 5, 100) or (1, 6, 5, 100)
    frame_w: int,
    frame_h: int,
    conf_thresh: float = 0.40,
    tile_id: Any = "full_frame",
    tile_x_offset: int = 0,
    tile_y_offset: int = 0,
    frame_tile_w: int = None,
    frame_tile_h: int = None,
) -> List[Dict[str, Any]]:
    """
    Decode the Hailo yolov8_nms_postprocess output tensor into detection dicts.

    Tensor layout (Hailo DFC YOLOv8 NMS output):
      axis 0 : class_id          (nc = 6)
      axis 1 : [y1, x1, y2, x2, score]   (5 fields, coordinates normalised 0–1)
      axis 2 : detection index   (up to 100 per class)

    IMPORTANT — axis-1 field order:
      Hailo DFC uses (y1, x1, y2, x2) in normalised form, not (x1, y1, x2, y2).
      Verify this against the Pi-side script output before deployment.
      If coordinates are swapped, swap the assignment below accordingly.

    Parameters
    ----------
    raw_output : np.ndarray
        Raw tensor from Hailo InferVStreams. Shape (6,5,100) or (1,6,5,100).
    frame_w, frame_h : int
        Full-frame dimensions (used to scale normalised coords to pixels).
        For tiled inference, pass the TILE crop dimensions and set tile offsets.
    conf_thresh : float
        Minimum score to include a detection. Applied here as an additional
        filter on top of whatever threshold was baked into the HEF at compile time.
    tile_id : int or str
        Identifier for the source tile (logged with each detection).
    tile_x_offset, tile_y_offset : int
        Pixel offset of this tile's top-left corner in the full frame.
        Use 0 for full-frame inference.
    frame_tile_w, frame_tile_h : int or None
        Actual pixel size of the tile crop in the full frame.
        Used for coord scaling when tile is not square or != 640.

    Returns
    -------
    List of detection dicts in full-frame pixel coordinates.
    """
    import numpy as np
    out = np.asarray(raw_output)
    if out.ndim == 4:
        out = out[0]   # remove batch dim → (6, 5, 100)

    if out.shape != (6, 5, 100):
        logger.error(f"Unexpected HEF output shape: {out.shape}. Expected (6,5,100).")
        return []

    detections: List[Dict] = []

    for cls_id in range(out.shape[0]):
        cls_name = HAILO_CLASS_NAMES[cls_id]
        for det_idx in range(out.shape[2]):
            score = float(out[cls_id, 4, det_idx])
            if score < conf_thresh:
                continue  # skip low-confidence (includes zero-padded slots)

            # Hailo normalised coordinate order: y1, x1, y2, x2
            y1n = float(out[cls_id, 0, det_idx])
            x1n = float(out[cls_id, 1, det_idx])
            y2n = float(out[cls_id, 2, det_idx])
            x2n = float(out[cls_id, 3, det_idx])

            # Scale to tile-crop pixel space
            tile_px_w = frame_tile_w if frame_tile_w is not None else frame_w
            tile_px_h = frame_tile_h if frame_tile_h is not None else frame_h
            det_x1 = int(x1n * tile_px_w)
            det_y1 = int(y1n * tile_px_h)
            det_x2 = int(x2n * tile_px_w)
            det_y2 = int(y2n * tile_px_h)

            # Map from tile space to full-frame space
            x1 = det_x1 + tile_x_offset
            y1 = det_y1 + tile_y_offset
            x2 = det_x2 + tile_x_offset
            y2 = det_y2 + tile_y_offset

            # Sanity-check
            if x2 <= x1 or y2 <= y1:
                logger.debug(
                    f"DECODE: skipped degenerate box cls={cls_name} "
                    f"tile={tile_id} det_idx={det_idx} "
                    f"coords=({x1},{y1},{x2},{y2})"
                )
                continue

            det = {
                "class_id":   cls_id,
                "class_name": cls_name,
                "score":      score,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "tile_id":    tile_id,
            }
            logger.info(
                f"DECODE: accepted {cls_name} score={score:.3f} "
                f"tile={tile_id} bbox=({x1},{y1},{x2},{y2})"
            )
            detections.append(det)

    return detections


def build_tile_grid(
    frame_h: int,
    frame_w: int,
    grid_rows: int = 2,
    grid_cols: int = 3,
    overlap: float = 0.15,
) -> List[Dict[str, int]]:
    """
    Divide a frame into a grid of overlapping tiles.

    Parameters
    ----------
    frame_h, frame_w : int
        Full-frame dimensions.
    grid_rows, grid_cols : int
        Tile grid layout. Default 2×3 = 6 tiles (matches run3.log tiles=6).
    overlap : float
        Fractional overlap between adjacent tiles (0.0–0.5).
        A value of 0.15 means each tile overlaps 15% of its width/height
        with its neighbour, ensuring objects at tile edges are not missed.

    Returns
    -------
    List of tile dicts:
        {"tile_id": int, "x": int, "y": int, "w": int, "h": int}
        where (x, y) is the top-left corner, w/h are pixel dimensions.
    """
    base_h = frame_h / grid_rows
    base_w = frame_w / grid_cols
    ov_h = int(base_h * overlap)
    ov_w = int(base_w * overlap)

    tiles = []
    tile_id = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = max(0, int(col * base_w) - ov_w)
            y = max(0, int(row * base_h) - ov_h)
            x2 = min(frame_w, int((col + 1) * base_w) + ov_w)
            y2 = min(frame_h, int((row + 1) * base_h) + ov_h)
            tiles.append({
                "tile_id": tile_id,
                "x": x, "y": y,
                "w": x2 - x, "h": y2 - y,
            })
            tile_id += 1
    return tiles


def log_detection_summary(
    raw_count: int,
    after_conf: int,
    after_nms: int,
    n_suppressed: int,
    n_rejected_conf: int,
    fps: float = 0.0,
):
    """Log per-frame detection pipeline summary."""
    logger.info(
        f"FRAME_SUMMARY: raw={raw_count} "
        f"after_conf_filter={after_conf} (rejected={n_rejected_conf}) "
        f"after_nms={after_nms} (suppressed={n_suppressed}) "
        f"fps={fps:.1f}"
    )
