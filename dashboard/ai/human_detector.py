"""
Human / Stranded Person Detector Engine
"""

import time
import numpy as np
import cv2
from typing import List, Tuple, Optional
from dashboard.ai.detector import BaseDetector, BoundingBox, DetectionResult
from dashboard.ai.hailo_runner import HailoInferenceRunner
from dashboard.config import (
    HUMAN_CLASSES,
    COLOR_HUMAN_BGR,
    DEFAULT_CONF_THRESHOLD,
    PYTORCH_HUMAN_MODEL_PATH
)

class HumanDetector(BaseDetector):
    """Detector for human / stranded person recognition."""

    def __init__(self, model_path: str = PYTORCH_HUMAN_MODEL_PATH):
        self.runner = HailoInferenceRunner(pytorch_fallback_path=model_path)
        self.last_fps = 0.0
        self.last_latency = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def detect(self, frame: np.ndarray, conf_thresh: float = DEFAULT_CONF_THRESHOLD) -> DetectionResult:
        start_t = time.time()
        annotated = frame.copy()
        h, w, _ = frame.shape
        
        raw_dets, latency = self.runner.run_inference(frame, conf_thresh=conf_thresh)
        self.last_latency = round(latency, 1)
        
        boxes: List[BoundingBox] = []
        for det in raw_dets:
            cname = det['class_name'].lower()
            if cname in HUMAN_CLASSES or cname == "person":
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                box = BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_name="person",
                    confidence=conf,
                    detection_type="HUMAN"
                )
                boxes.append(box)
                
                # Draw bounding box (Cyan)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_HUMAN_BGR, 2)
                label = f"PERSON {conf:.2f}"
                
                # Background text pill
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - h_txt - 8), (x1 + w_txt + 6, y1), COLOR_HUMAN_BGR, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
        return f"HUMAN_DETECTOR [{self.runner.status}]"
