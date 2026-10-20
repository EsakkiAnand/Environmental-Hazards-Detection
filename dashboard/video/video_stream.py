"""
Multi-Source Video Stream & AI Inference Pipeline Manager
"""

import threading
import time
import base64
import random
import uuid
import numpy as np
import cv2
from typing import Optional, Tuple, List
from dashboard.ai.human_detector import HumanDetector
from dashboard.ai.hazard_detector import HazardDetector
from dashboard.ai.detector import BoundingBox, DetectionResult
from dashboard.data.event_store import EventStore
from dashboard.data.models import DetectionEvent
from dashboard.telemetry.telemetry_state import TelemetryState
from dashboard.config import HAZARD_CLASSES, COLOR_HUMAN_BGR, HAZARD_COLOR_MAP

class VideoStreamPipeline:
    """Manages video frame capture, AI inference, and detection logging."""

    def __init__(
        self,
        event_store: EventStore,
        video_source: str = "SYNTHETIC",
        conf_thresh: float = 0.40
    ):
        self.event_store = event_store
        self.video_source = str(video_source)
        self.conf_thresh = conf_thresh
        self.telemetry_state = TelemetryState()
        
        self.human_detector = HumanDetector()
        self.hazard_detector = HazardDetector()
        
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._current_raw_frame: Optional[np.ndarray] = None
        self._current_annotated_frame: Optional[np.ndarray] = None
        self._is_camera_online = False
        self._fps = 0.0
        self._inference_ms = 0.0
        self._latest_detections: List[BoundingBox] = []

    def start(self):
        """Start the video capture & inference worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background worker and release video capture."""
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def set_source(self, source: str):
        """Switch video input source (camera ID, file path, or 'SYNTHETIC')."""
        with self._lock:
            self.video_source = str(source)
            if self._cap:
                self._cap.release()
                self._cap = None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], bool, float, float, List[BoundingBox]]:
        """Return latest annotated BGR frame, camera online status, FPS, latency, and current boxes."""
        with self._lock:
            if self._current_annotated_frame is not None:
                return (
                    self._current_annotated_frame.copy(),
                    self._is_camera_online,
                    self._fps,
                    self._inference_ms,
                    list(self._latest_detections)
                )
            else:
                offline_frame = self._generate_offline_graphic("CAMERA OFFLINE")
                return (offline_frame, False, 0.0, 0.0, [])

    def _processing_loop(self):
        """Main processing thread loop."""
        frame_count = 0
        fps_timer = time.time()
        
        while self._running:
            start_t = time.time()
            frame = self._grab_frame()
            
            if frame is None:
                with self._lock:
                    self._is_camera_online = False
                    self._current_annotated_frame = self._generate_offline_graphic("CAMERA OFFLINE")
                time.sleep(0.5)
                continue
            
            with self._lock:
                self._is_camera_online = True
                self._current_raw_frame = frame
            
            # Run AI Inference
            res_human = self.human_detector.detect(frame, conf_thresh=self.conf_thresh)
            res_hazard = self.hazard_detector.detect(frame, conf_thresh=self.conf_thresh)
            
            # Combine annotated overlays
            annotated = frame.copy()
            combined_boxes = res_human.boxes + res_hazard.boxes
            
            # Render combined bounding boxes on frame
            for box in combined_boxes:
                if box.detection_type == "HUMAN":
                    color = COLOR_HUMAN_BGR
                    label = f"PERSON {box.confidence:.2f}"
                else:
                    color_meta = HAZARD_COLOR_MAP.get(box.class_name, {"bgr": (0, 0, 255)})
                    color = color_meta["bgr"]
                    label = f"{box.class_name.upper()} {box.confidence:.2f}"
                
                cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (box.x1, box.y1 - th - 6), (box.x1 + tw + 6, box.y1), color, -1)
                cv2.putText(annotated, label, (box.x1 + 3, box.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Measure performance stats
            inference_time = res_human.inference_time_ms + res_hazard.inference_time_ms
            
            # FPS calculation
            frame_count += 1
            dt = time.time() - fps_timer
            if dt >= 1.0:
                current_fps = round(frame_count / dt, 1)
                frame_count = 0
                fps_timer = time.time()
            else:
                current_fps = self._fps
            
            # Render HUD overlay on video (FPS, Latency, Target counts)
            cv2.putText(annotated, f"FPS: {current_fps} | LATENCY: {inference_time:.1f} ms", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(annotated, f"HUMAN DETECTED: {len(res_human.boxes)} | HAZARDS: {len(res_hazard.boxes)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2)

            # Store detection events with drone GPS snapshot
            self._log_detection_events(combined_boxes, annotated)

            with self._lock:
                self._current_annotated_frame = annotated
                self._fps = current_fps
                self._inference_ms = inference_time
                self._latest_detections = combined_boxes

            # Frame rate throttling (~25 FPS max)
            elapsed = time.time() - start_t
            if elapsed < 0.04:
                time.sleep(0.04 - elapsed)

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Fetch frame from video source or synthetic generator."""
        if self.video_source == "SYNTHETIC":
            return self._generate_synthetic_drone_frame()

        if self._cap is None or not self._cap.isOpened():
            try:
                # Int or video path
                src = int(self.video_source) if self.video_source.isdigit() else self.video_source
                self._cap = cv2.VideoCapture(src)
            except Exception:
                return None

        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
            else:
                # Reopen for recorded video loop
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self._cap.read()
                return frame2 if ret2 else None
        
        return None

    def _log_detection_events(self, boxes: List[BoundingBox], frame: np.ndarray):
        """Convert bounding boxes to DetectionEvent objects with telemetry lat/lon/alt."""
        if not boxes:
            return
            
        telemetry = self.telemetry_state.get_snapshot()
        lat = telemetry.latitude
        lon = telemetry.longitude
        alt = telemetry.relative_altitude

        for box in boxes:
            # Throttle repetitive logging: only log high confidence or distinct events
            if box.confidence >= self.conf_thresh:
                event_id = f"EV-{uuid.uuid4().hex[:6].upper()}"
                now = time.time()
                formatted_time = time.strftime("%H:%M:%S", time.localtime(now))
                
                # Encode small thumbnail for event preview
                thumb = cv2.resize(frame, (320, 180))
                _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_img = base64.b64encode(buffer).decode('utf-8')
                
                ev = DetectionEvent(
                    event_id=event_id,
                    timestamp=now,
                    formatted_time=formatted_time,
                    detection_type=box.detection_type,
                    class_name=box.class_name,
                    confidence=box.confidence,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    frame_image_base64=b64_img
                )
                self.event_store.add_event(ev)

    def _generate_synthetic_drone_frame(self) -> np.ndarray:
        """Create a synthetic high-resolution UAV camera feed with environmental elements."""
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        
        # Ground texture gradient (aerial landscape simulation)
        t = time.time()
        shift = int((t * 20) % 960)
        
        # Base terrain background
        img[:, :] = (35, 55, 30)  # Forest green
        cv2.rectangle(img, (0, 300), (960, 540), (45, 75, 40), -1)
        
        # Draw simulated river/flood line
        pts = np.array([
            [0, 200], [300, 250], [600, 220], [960, 280],
            [960, 340], [600, 290], [300, 320], [0, 260]
        ], np.int32)
        cv2.fillPoly(img, [pts], (180, 100, 20))
        
        # Occasional synthetic detection objects for demonstration
        cycle = int(t % 15)
        if cycle < 5:
            # Person synthetic bbox simulation
            x1, y1 = 400, 220
            x2, y2 = 460, 310
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_HUMAN_BGR, 2)
            cv2.putText(img, "PERSON 0.94", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUMAN_BGR, 2)
        elif cycle < 10:
            # Fire hazard synthetic bbox simulation
            x1, y1 = 650, 150
            x2, y2 = 780, 260
            color = HAZARD_COLOR_MAP["fire"]["bgr"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "FIRE 0.91", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif cycle < 14:
            # Damaged structure simulation
            x1, y1 = 150, 320
            x2, y2 = 320, 450
            color = HAZARD_COLOR_MAP["damaged_structure"]["bgr"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "DAMAGED STRUCTURE 0.88", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw Crosshair Telemetry HUD
        cx, cy = 480, 270
        cv2.line(img, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
        cv2.line(img, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
        cv2.circle(img, (cx, cy), 40, (0, 255, 0), 1)
        
        return img

    def _generate_offline_graphic(self, text: str) -> np.ndarray:
        """Create a clean dark placeholder frame when video stream is offline."""
        img = np.zeros((480, 720, 3), dtype=np.uint8)
        img[:, :] = (18, 22, 28)  # Deep slate gray
        
        # Border
        cv2.rectangle(img, (10, 10), (710, 470), (45, 55, 72), 2)
        
        # Center Camera Offline text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cx, cy = 360, 230
        cv2.putText(img, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 2)
        
        sub_text = "Check camera hardware connection / RTSP stream URL"
        (stw, _), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, sub_text, (cx - stw // 2, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 165), 1)
        
        return img
