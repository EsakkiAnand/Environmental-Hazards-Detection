"""
Abstract Base Detector & Unified Detection Result Model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class BoundingBox:
    """Bounding box coordinates (x1, y1, x2, y2) normalized or pixel."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    detection_type: str  # "HUMAN" or "HAZARD"

@dataclass
class DetectionResult:
    """Inference output result container."""
    boxes: List[BoundingBox]
    frame_annotated: np.ndarray
    inference_time_ms: float
    fps: float
    model_status: str

class BaseDetector(ABC):
    """Abstract interface for AI detection engines."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray, conf_thresh: float = 0.40) -> DetectionResult:
        """Run detection on an BGR OpenCV frame."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Return model runtime status string."""
        pass
