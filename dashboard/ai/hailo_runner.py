"""
Hailo-8 HEF Inference Engine Runner
Seraph Zenith Mark 1 — v2 (full tensor decoding + confidence filtering)

Changes from v1:
  - Hailo code path now fully decodes the (6, 5, 100) NMS output tensor.
  - conf_thresh is actually applied to Hailo detections (was a stub before).
  - Uses HAILO_CLASS_NAMES from seraph_nms (6 classes, no chemical_spill).
  - Structured logging: ACCEPT / REJECT / DECODE per detection.
  - PyTorch fallback path unchanged.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional

from dashboard.config import (
    HAILO_HEF_PATH,
    PYTORCH_HAZARD_MODEL_PATH,
    DEFAULT_CONF_THRESHOLD,
    get_conf_threshold,
)
from dashboard.ai.seraph_nms import (
    HAILO_CLASS_NAMES,
    decode_hailo_nms_output,
)

logger = logging.getLogger("seraph.hailo_runner")

# ---------------------------------------------------------------------------
# Platform availability checks
# ---------------------------------------------------------------------------
HAILO_AVAILABLE = False
try:
    import hailo_platform as hailo
    HAILO_AVAILABLE = True
    logger.info("hailo_platform import: OK")
except ImportError:
    logger.info("hailo_platform not available — will use PyTorch fallback")

ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass


class HailoInferenceRunner:
    """
    Wrapper for running YOLOv8s HEF inference on Hailo-8 hardware.

    Inference path priority:
      1. Hailo-8 via hailo_platform (if hw present and HEF exists)
      2. PyTorch / Ultralytics YOLO fallback
      3. No-op (returns empty detections)

    Output format: list of dicts with keys:
        bbox       : (x1, y1, x2, y2) full-frame pixel coords
        confidence : float
        class_name : str  (one of HAILO_CLASS_NAMES)
        tile_id    : "full_frame" (tiling is handled upstream)
    """

    def __init__(
        self,
        hef_path: str = HAILO_HEF_PATH,
        pytorch_fallback_path: str = PYTORCH_HAZARD_MODEL_PATH,
    ):
        self.hef_path = hef_path
        self.pytorch_path = pytorch_fallback_path
        self.is_hailo_active = False
        self.is_pytorch_active = False
        self.hailo_target = None
        self.pytorch_model = None
        self.status = "UNINITIALIZED"

        # Keep a reference to Hailo objects needed for inference
        self._network_group = None
        self._input_vstream_info = None
        self._output_vstream_info = None

        self._init_engine()

    # ------------------------------------------------------------------
    def _init_engine(self):
        """Initialise Hailo-8 hardware or fall back to PyTorch."""
        if HAILO_AVAILABLE and os.path.exists(self.hef_path):
            try:
                self.hailo_target = hailo.VDevice()
                hef = hailo.Hef(self.hef_path)
                cfg_params = hailo.ConfigureParams.create_from_hef(
                    hef, interface=hailo.HailoStreamInterface.PCIe
                )
                network_groups = self.hailo_target.configure(hef, cfg_params)
                self._network_group = network_groups[0]
                self._input_vstream_info  = hef.get_input_vstream_infos()[0]
                self._output_vstream_info = hef.get_output_vstream_infos()
                self.is_hailo_active = True
                self.status = "HAILO8_HEF_ONLINE"
                logger.info(f"Hailo-8 initialised. HEF: {self.hef_path}")
                logger.info(
                    f"HEF input:  {[(i.name, i.shape) for i in [self._input_vstream_info]]}"
                )
                logger.info(
                    f"HEF output: {[(o.name, o.shape) for o in self._output_vstream_info]}"
                )
                return
            except Exception as exc:
                self.status = f"HAILO_INIT_ERROR ({exc})"
                logger.warning(f"Hailo-8 init failed: {exc}")

        # PyTorch fallback
        if ULTRALYTICS_AVAILABLE and os.path.exists(self.pytorch_path):
            try:
                self.pytorch_model = YOLO(self.pytorch_path)
                self.is_pytorch_active = True
                self.status = "PYTORCH_YOLO_FALLBACK"
                logger.info(f"PyTorch fallback loaded: {self.pytorch_path}")
                return
            except Exception as exc:
                self.status = f"PYTORCH_INIT_ERROR ({exc})"
                logger.warning(f"PyTorch fallback init failed: {exc}")

        self.status = "NO_INFERENCE_ENGINE"
        logger.warning("No inference engine available. Returning empty detections.")

    # ------------------------------------------------------------------
    def run_inference(
        self,
        frame: np.ndarray,
        conf_thresh: float = DEFAULT_CONF_THRESHOLD,
    ) -> Tuple[List[dict], float]:
        """
        Execute inference on a single BGR frame.

        Parameters
        ----------
        frame      : BGR numpy array (any resolution — resized internally to 640x640)
        conf_thresh: minimum confidence to include a detection.
                     Applied as a post-HEF runtime filter regardless of the
                     threshold baked into the HEF at compile time.

        Returns
        -------
        (detections, latency_ms)
            detections : list of detection dicts (see class docstring)
            latency_ms : end-to-end inference latency in milliseconds
        """
        start_t = time.time()
        detections: List[dict] = []

        # ---- Hailo-8 path ------------------------------------------------
        if self.is_hailo_active:
            try:
                img_h, img_w = frame.shape[:2]
                img_resized = cv2.resize(frame, (640, 640))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                input_data = {
                    self._input_vstream_info.name: np.expand_dims(img_rgb, axis=0)
                }

                with hailo.InferVStreams(
                    self._network_group,
                    self._input_vstream_info,
                    self._output_vstream_info,
                ) as pipeline:
                    raw_outputs = pipeline.infer(input_data)

                # raw_outputs is a dict keyed by output vstream name
                # Expected shape: (1, 6, 5, 100) or (6, 5, 100)
                raw_tensor = list(raw_outputs.values())[0]

                # decode_hailo_nms_output handles both (6,5,100) and (1,6,5,100)
                # For full-frame inference tile_x_offset=0, tile_y_offset=0
                detections = decode_hailo_nms_output(
                    raw_output=raw_tensor,
                    frame_w=img_w,
                    frame_h=img_h,
                    conf_thresh=conf_thresh,
                    tile_id="full_frame",
                    tile_x_offset=0,
                    tile_y_offset=0,
                )

                # Convert seraph_nms dict format to legacy bbox-tuple format
                # expected by hazard_detector.py
                legacy = []
                for d in detections:
                    legacy.append({
                        "bbox":       (d["x1"], d["y1"], d["x2"], d["y2"]),
                        "confidence": d["score"],
                        "class_name": d["class_name"],
                    })

                latency = (time.time() - start_t) * 1000.0
                logger.info(
                    f"Hailo inference: {len(legacy)} detections "
                    f"(conf>={conf_thresh}) latency={latency:.1f}ms"
                )
                return legacy, latency

            except Exception as exc:
                logger.error(f"Hailo inference error: {exc} — falling through to PyTorch")

        # ---- PyTorch fallback path ---------------------------------------
        if self.is_pytorch_active and self.pytorch_model:
            results = self.pytorch_model.predict(
                frame, conf=conf_thresh, verbose=False
            )
            latency = (time.time() - start_t) * 1000.0

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy    = box.xyxy[0].cpu().numpy().astype(int)
                    conf    = float(box.conf[0].cpu().numpy())
                    cls_id  = int(box.cls[0].cpu().numpy())
                    cls_name = (
                        HAILO_CLASS_NAMES[cls_id]
                        if cls_id < len(HAILO_CLASS_NAMES)
                        else f"cls_{cls_id}"
                    )
                    logger.info(
                        f"PyTorch ACCEPT: {cls_name} conf={conf:.3f} "
                        f"bbox=({xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]})"
                    )
                    detections.append({
                        "bbox":       (int(xyxy[0]), int(xyxy[1]),
                                       int(xyxy[2]), int(xyxy[3])),
                        "confidence": conf,
                        "class_name": cls_name,
                    })
            return detections, latency

        # ---- No engine ---------------------------------------------------
        latency = (time.time() - start_t) * 1000.0
        return [], latency

    # ------------------------------------------------------------------
    def get_status(self) -> str:
        return self.status
