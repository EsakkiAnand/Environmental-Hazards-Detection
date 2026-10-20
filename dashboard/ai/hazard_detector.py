"""
Environmental Hazard Detector Engine (7 Classes)
"""

import time
import numpy as np
import cv2
from typing import List, Tuple, Optional
from dashboard.ai.detector import BaseDetector, BoundingBox, DetectionResult
from dashboard.ai.hailo_runner import HailoInferenceRunner
from dashboard.config import (
    HAZARD_CLASSES,
    HAZARD_COLOR_MAP,
    DEFAULT_HAZARD_COLOR_HEX,
    DEFAULT_CONF_THRESHOLD,
    PYTORCH_HAZARD_MODEL_PATH
)

class HazardDetector(BaseDetector):
    """Detector for 7 Environmental Hazard Classes."""

    def __init__(self, model_path: str = PYTORCH_HAZARD_MODEL_PATH):
        self.runner = HailoInferenceRunner(pytorch_fallback_path=model_path)
        self.last_fps = 0.0
        self.last_latency = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def detect(self, frame: np.ndarray, conf_thresh: float = DEFAULT_CONF_THRESHOLD) -> DetectionResult:
        annotated = frame.copy()
        h, w, _ = frame.shape
        
        raw_dets, latency = self.runner.run_inference(frame, conf_thresh=conf_thresh)
        self.last_latency = round(latency, 1)
        
        boxes: List[BoundingBox] = []
        for det in raw_dets:
            cname = det['class_name'].lower().replace(" ", "_")
            if cname in HAZARD_CLASSES or "fire" in cname or "flood" in cname or "structure" in cname or "power" in cname or "spill" in cname or "debris" in cname or "landslide" in cname:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Match class name or fallback
                matched_class = cname
                if matched_class not in HAZARD_CLASSES:
                    for hz in HAZARD_CLASSES:
                        if hz in cname:
                            matched_class = hz
                            break
                
                box = BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_name=matched_class,
                    confidence=conf,
                    detection_type="HAZARD"
                )
                boxes.append(box)
                
                # Get color scheme
                color_meta = HAZARD_COLOR_MAP.get(matched_class, {"bgr": (0, 0, 255)})
                bgr_color = color_meta["bgr"]
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)
                label = f"{matched_class.upper()} {conf:.2f}"
                
                # Background text pill
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - h_txt - 8), (x1 + w_txt + 6, y1), bgr_color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS calculation
        self._frame_count += 1
        dt = time.time() - self._fps_timer
        if dt >= 1.0:
            self.last_fps = round(self._frame_count / dt, 1)
            self._frame_count = 0
            self._fps_timer = time.time()

        return DetectionResult(
            boxes=boxes,
            frame_annotated=annotated,
            inference_time_ms=self.last_latency,
            fps=self.last_fps,
            model_status=self.get_status()
        )

    def get_status(self) -> str:
        return f"HAZARD_DETECTOR [{self.runner.status}]"
