"""
Autonomous UAV Ground Station Monitor — Real-Time Telemetry & Hailo-8 AI Dashboard
Author: Senior UAV Software Architect & Full-Stack Python Engineer
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path for module resolution
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="UAV Ground Station Monitor | ArduPilot & Hailo-8",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Professional Ground Station CSS Theme
CUSTOM_CSS = """
<style>
    /* Dark Ground Station Background */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }
    
    /* Header Container */
    .stMainHeader {
        background-color: #111722;
        border-bottom: 2px solid #00e5ff;
        padding: 10px;
    }
    
    /* Custom Card Containers */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #00e5ff !important;
        font-weight: 700;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111722;
        border-radius: 6px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e5ff !important;
        color: #0f172a !important;
        font-weight: bold;
    }
    
    /* Status Badge Styling */
    .status-badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

from dashboard.config import (
    MODE_SIMULATION,
    MODE_RECORDED,
    MODE_REAL_DRONE,
    DEFAULT_MAVLINK_CONNECTION_STRING,
    DEFAULT_CONF_THRESHOLD,
    HAZARD_CLASSES
)
from dashboard.data.telemetry_store import TelemetryStore
from dashboard.data.event_store import EventStore
from dashboard.telemetry.telemetry_state import TelemetryState
from dashboard.telemetry.mavlink_client import MAVLinkClient
from dashboard.video.video_stream import VideoStreamPipeline
from dashboard.ui.telemetry_panel import render_top_status_bar, render_telemetry_metrics_panel
from dashboard.ui.detection_panel import render_active_detections_summary, render_recent_alerts_stream, render_detection_event_table
from dashboard.ui.map_panel import render_realtime_map
from dashboard.ui.charts import render_telemetry_history_charts

# Initialize Shared Application Singleton Services in Streamlit Session State
if "telemetry_store" not in st.session_state:
    st.session_state.telemetry_store = TelemetryStore()

if "event_store" not in st.session_state:
    st.session_state.event_store = EventStore()

if "telemetry_state" not in st.session_state:
    st.session_state.telemetry_state = TelemetryState()

if "mavlink_client" not in st.session_state:
    client = MAVLinkClient(telemetry_store=st.session_state.telemetry_store)
    client.start(use_synthetic=True)
    st.session_state.mavlink_client = client

if "video_pipeline" not in st.session_state:
    pipeline = VideoStreamPipeline(
        event_store=st.session_state.event_store,
        video_source="SYNTHETIC",
        conf_thresh=DEFAULT_CONF_THRESHOLD
    )
    pipeline.start()
    st.session_state.video_pipeline = pipeline

# ==============================================================================
# SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/drone.png", width=64)
st.sidebar.title("🛸 UAV Ground Control")
st.sidebar.caption("ArduPilot + Pi 5 + Hailo-8 AI Dashboard")

st.sidebar.divider()

# 1. Operating Mode Selector
st.sidebar.subheader("🕹️ Operating Mode")
op_mode = st.sidebar.radio(
    "Select System Mode",
    [MODE_SIMULATION, MODE_RECORDED, MODE_REAL_DRONE],
    index=0,
    help="SIMULATION: ArduPilot SITL / Synthetic. RECORDED: Video/log replay. REAL_DRONE: Physical UAV."
)

st.session_state.telemetry_state.set_mode(op_mode)

st.sidebar.divider()

# 2. Connection Settings
st.sidebar.subheader("📡 MAVLink Connection")
conn_str = st.sidebar.text_input(
    "Connection String",
    value=DEFAULT_MAVLINK_CONNECTION_STRING if op_mode != MODE_REAL_DRONE else "/dev/ttyAMA0",
    help="SITL: udp:127.0.0.1:14550 | Pi Serial: /dev/ttyAMA0 | UDP Broadcast: udp:0.0.0.0:14550"
)

if st.sidebar.button("🔄 Reconnect MAVLink"):
    st.session_state.mavlink_client.connection_string = conn_str
    st.session_state.mavlink_client.stop()
    use_synth = (op_mode == MODE_SIMULATION and "udp" not in conn_str)
    st.session_state.mavlink_client.start(use_synthetic=use_synth)
    st.sidebar.success("Reconnecting...")

st.sidebar.divider()

# 3. AI & Video Stream Controls
st.sidebar.subheader("🎥 Video & AI Inference")
video_src = st.sidebar.selectbox(
    "Video Source",
    ["SYNTHETIC", "0 (USB Webcam)", "RTSP Stream URL", "Test Video File (.mp4)"],
    index=0
)

if video_src == "0 (USB Webcam)":
    real_src = "0"
elif video_src == "RTSP Stream URL":
    real_src = st.sidebar.text_input("Enter RTSP URL", "rtsp://192.168.1.100:8554/live")
elif video_src == "Test Video File (.mp4)":
    real_src = st.sidebar.text_input("Enter Video Path", "sample_flight.mp4")
else:
    real_src = "SYNTHETIC"

st.session_state.video_pipeline.set_source(real_src)

conf_thresh = st.sidebar.slider(
    "AI Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=DEFAULT_CONF_THRESHOLD,
    step=0.05
)
st.session_state.video_pipeline.conf_thresh = conf_thresh

st.sidebar.divider()
st.sidebar.caption("⚡ **System Health**")
st.sidebar.write(f"• Hailo-8 AI Accelerator: `{st.session_state.video_pipeline.human_detector.runner.status}`")
st.sidebar.write(f"• Telemetry State: `{st.session_state.telemetry_state.get_snapshot().mavlink_status}`")

# ==============================================================================
# MAIN DASHBOARD AREA
# ==============================================================================

# Fetch active snapshots
telemetry = st.session_state.telemetry_state.get_snapshot()

# Top Status Bar
render_top_status_bar(telemetry, op_mode)

st.divider()

# MAIN ROW: Live Video AI Feed (Left) & Telemetry Metrics Panel (Right)
col_video, col_telemetry = st.columns([1.6, 1.0])

with col_video:
    st.markdown("### 🎥 Live Video & AI Inference Feed")
    frame, camera_online, fps, latency, current_boxes = st.session_state.video_pipeline.get_latest_frame()
    
    if frame is not None:
        # Convert BGR to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
    else:
        st.warning("Video stream unavailable.")

    # Model status & overlay metrics
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        st.caption(f"⚡ **Inference FPS**: `{fps} Hz`")
    with c_f2:
        st.caption(f"⏱️ **Latency**: `{latency:.1f} ms`")
    with c_f3:
        status_color = "🟢" if camera_online else "🔴"
        st.caption(f"📷 **Camera**: {status_color} `{'ONLINE' if camera_online else 'OFFLINE'}`")

with col_telemetry:
    render_telemetry_metrics_panel(telemetry)

st.divider()

# SECOND ROW: Live PyDeck Map (Left) & Active AI Detections Summary (Right)
col_map, col_alerts = st.columns([1.5, 1.0])

with col_map:
    render_realtime_map(telemetry, st.session_state.event_store, st.session_state.telemetry_store)

with col_alerts:
    render_active_detections_summary(st.session_state.event_store)
    st.divider()
    render_recent_alerts_stream(st.session_state.event_store, limit=4)

st.divider()

# BOTTOM SECTION: Detection Event Table & Historical Telemetry Charts
tab_events, tab_charts = st.tabs(["📋 Real-Time Detection Event Log", "📈 Bounded Telemetry History"])

with tab_events:
    render_detection_event_table(st.session_state.event_store)

with tab_charts:
    render_telemetry_history_charts(st.session_state.telemetry_store)

# Auto-refresh UI (1 second polling interval)
time.sleep(1.0)
st.rerun()
