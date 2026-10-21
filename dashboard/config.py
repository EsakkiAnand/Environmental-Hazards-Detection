"""
Centralized Configuration for UAV Ground Station Dashboard
Seraph Zenith Mark 1 — v2 (false-positive remediation)
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# System Operating Modes
MODE_SIMULATION = "SIMULATION"       # ArduPilot SITL / Synthetic Telemetry
MODE_RECORDED   = "RECORDED_DATA"   # Replay recorded video & test telemetry
MODE_REAL_DRONE = "REAL_DRONE"       # Live Raspberry Pi 5 + Hailo-8 + ArduPilot

# MAVLink Default Settings
DEFAULT_MAVLINK_CONNECTION_STRING = "udp:127.0.0.1:14550"  # SITL default
DEFAULT_MAVLINK_BAUD               = 57600
MAVLINK_TIMEOUT_SEC                = 3.0
MAVLINK_HEARTBEAT_INTERVAL_SEC     = 1.0

# Telemetry History Ring Buffer Size
TELEMETRY_HISTORY_MAX_LEN = 1000

# ---------------------------------------------------------------------------
# Hazard Classes — exactly 6, matching training data.yaml and HEF output
#
# IMPORTANT: These indices MUST match the trained model and compiled HEF:
#   0 = fire          (15 678 training bboxes, 42.0%)
#   1 = flood         ( 3 768 training bboxes, 10.1%)
#   2 = damaged_structure (3 863, 10.4%)
#   3 = exposed_powerline (10 038, 26.9%)
#   4 = debris        (   515 training bboxes,  1.4%)  ← under-represented
#   5 = landslide     ( 3 446 training bboxes,  9.2%)
#
# chemical_spill was listed in classes.txt but was NEVER included in the
# training dataset (unified_hazards_v2/data.yaml has nc=6).
# It has been removed here to prevent index mismatches with the 6-class HEF.
# ---------------------------------------------------------------------------
HAZARD_CLASSES = [
    "fire",               # class 0
    "flood",              # class 1
    "damaged_structure",  # class 2
    "exposed_powerline",  # class 3
    "debris",             # class 4
    "landslide",          # class 5
]

# Single global confidence threshold (applied as post-HEF runtime filter)
DEFAULT_CONF_THRESHOLD = 0.40

# IoU threshold for cross-tile NMS
DEFAULT_IOU_THRESHOLD = 0.45

# Per-class confidence overrides.
# Set to None to use DEFAULT_CONF_THRESHOLD for that class.
# Only set these AFTER per-class evaluation on labelled data — do not
# guess values without validation evidence.
PER_CLASS_CONF_THRESHOLD = {
    "fire":               None,   # use DEFAULT
    "flood":              None,   # use DEFAULT
    "damaged_structure":  None,   # use DEFAULT
    "exposed_powerline":  None,   # use DEFAULT
    "debris":             None,   # use DEFAULT (under-represented — lower recall expected)
    "landslide":          None,   # use DEFAULT — DO NOT raise blindly without evaluation
}

def get_conf_threshold(class_name: str) -> float:
    """Return the effective confidence threshold for a given class name."""
    override = PER_CLASS_CONF_THRESHOLD.get(class_name)
    return override if override is not None else DEFAULT_CONF_THRESHOLD

# Human Detection Classes
HUMAN_CLASSES = ["person", "stranded_person", "human"]

# ---------------------------------------------------------------------------
# AI Model Paths
# NOTE: The compiled HEF was generated from the YOLOv8s ONNX export.
#       The file was historically named hazards_yolov8n.hef but the
#       embedded network name in the HEF is 'yolov8s'. These names are
#       inconsistent; do not use the filename to infer architecture.
# ---------------------------------------------------------------------------
HAILO_HEF_PATH            = os.path.join(PROJECT_ROOT, "models", "hazards_yolov8n.hef")
PYTORCH_HAZARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "yolov8n_hazards.pt")
PYTORCH_HUMAN_MODEL_PATH  = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# Color Schemes for UI & Video Overlays (BGR for OpenCV, Hex/RGB for Streamlit/PyDeck)
COLOR_HUMAN_BGR = (255, 204, 0)
COLOR_HUMAN_HEX = "#00E5FF"
COLOR_HUMAN_RGB = [0, 229, 255]

HAZARD_COLOR_MAP = {
    "fire":               {"bgr": (0,   69,  255), "hex": "#FF4500", "rgb": [255, 69,   0]},
    "flood":              {"bgr": (255, 128,   0), "hex": "#0080FF", "rgb": [0,   128, 255]},
    "damaged_structure":  {"bgr": (0,   165, 255), "hex": "#FFA500", "rgb": [255, 165,   0]},
    "exposed_powerline":  {"bgr": (0,   255, 255), "hex": "#FFFF00", "rgb": [255, 255,   0]},
    "debris":             {"bgr": (128, 128, 128), "hex": "#A0A0A0", "rgb": [160, 160, 160]},
    "landslide":          {"bgr": (19,   69, 139), "hex": "#8B4513", "rgb": [139,  69,  19]},
    # chemical_spill removed — class not present in trained model
}
DEFAULT_HAZARD_COLOR_HEX = "#FF3366"
DEFAULT_HAZARD_COLOR_RGB = [255, 51, 102]

# Map Center Defaults
DEFAULT_HOME_LAT = 37.7749
DEFAULT_HOME_LON = -122.4194
DEFAULT_HOME_ALT = 10.0

# UI Refresh Settings
UI_AUTOREFRESH_MS = 1000
