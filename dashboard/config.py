"""
Centralized Configuration for UAV Ground Station Dashboard
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# System Operating Modes
MODE_SIMULATION = "SIMULATION"       # ArduPilot SITL / Synthetic Telemetry
MODE_RECORDED = "RECORDED_DATA"     # Replay recorded video & test telemetry
MODE_REAL_DRONE = "REAL_DRONE"       # Live Raspberry Pi 5 + Hailo-8 + ArduPilot

# MAVLink Default Settings
DEFAULT_MAVLINK_CONNECTION_STRING = "udp:127.0.0.1:14550"  # SITL default
DEFAULT_MAVLINK_BAUD = 57600
MAVLINK_TIMEOUT_SEC = 3.0
MAVLINK_HEARTBEAT_INTERVAL_SEC = 1.0

# Telemetry History Ring Buffer Size
TELEMETRY_HISTORY_MAX_LEN = 1000

# Environmental Hazard Classes (from classes.txt)
HAZARD_CLASSES = [
    "fire",
    "flood",
    "damaged_structure",
    "exposed_powerline",
    "debris",
    "landslide",
    "chemical_spill"
]

# Human Detection Classes
HUMAN_CLASSES = ["person", "stranded_person", "human"]

# AI Inference Defaults
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45

# AI Model Paths
HAILO_HEF_PATH = os.path.join(PROJECT_ROOT, "models", "hazards_yolov8n.hef")
PYTORCH_HAZARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "yolov8n_hazards.pt")
PYTORCH_HUMAN_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# Color Schemes for UI & Video Overlays (BGR for OpenCV, Hex/RGB for Streamlit/PyDeck)
COLOR_HUMAN_BGR = (255, 204, 0)      # Bright Cyan / Yellow-Blue
COLOR_HUMAN_HEX = "#00E5FF"
COLOR_HUMAN_RGB = [0, 229, 255]

HAZARD_COLOR_MAP = {
    "fire": {"bgr": (0, 69, 255), "hex": "#FF4500", "rgb": [255, 69, 0]},            # Red-Orange
    "flood": {"bgr": (255, 128, 0), "hex": "#0080FF", "rgb": [0, 128, 255]},          # Blue
    "damaged_structure": {"bgr": (0, 165, 255), "hex": "#FFA500", "rgb": [255, 165, 0]}, # Orange
    "exposed_powerline": {"bgr": (0, 255, 255), "hex": "#FFFF00", "rgb": [255, 255, 0]}, # Yellow
    "debris": {"bgr": (128, 128, 128), "hex": "#A0A0A0", "rgb": [160, 160, 160]},     # Gray
    "landslide": {"bgr": (19, 69, 139), "hex": "#8B4513", "rgb": [139, 69, 19]},       # Saddle Brown
    "chemical_spill": {"bgr": (128, 0, 128), "hex": "#9370DB", "rgb": [147, 112, 219]} # Medium Purple
}
DEFAULT_HAZARD_COLOR_HEX = "#FF3366"
DEFAULT_HAZARD_COLOR_RGB = [255, 51, 102]

# Map Center Defaults (San Francisco SITL default or customizable)
DEFAULT_HOME_LAT = 37.7749
DEFAULT_HOME_LON = -122.4194
DEFAULT_HOME_ALT = 10.0

# UI Refresh Settings
UI_AUTOREFRESH_MS = 1000
