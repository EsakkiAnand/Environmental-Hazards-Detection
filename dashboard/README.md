# Autonomous UAV Ground Station Monitor & AI Telemetry Dashboard

A professional, real-time ground station dashboard built in Python using **Streamlit**, integrated with **ArduPilot/MAVLink** telemetry and **Hailo-8 AI Acceleration** (with fallback support for PyTorch/OpenCV/Simulation).

---

## 1. System Architecture & Data Flow

```
Drone (ArduPilot FCU + RPi 5 + Hailo-8 AI HAT+ + Camera)
   │
   ├─► ArduPilot Telemetry ──(MAVLink UDP/Serial)──► MAVLinkClient ──► TelemetryState
   │                                                                     │
   └─► Camera Stream ─────────► HailoInferenceRunner ──► EventStore ──────┤
                               (YOLO HEF / PyTorch)                      │
                                                                         ▼
                                                             Streamlit Dashboard UI
                                                             (Top Bar + Camera + Map + Logs)
```

---

## 2. Directory Structure

```
dashboard/
├── app.py                      # Main Streamlit web application entrypoint
├── config.py                   # System configuration & default thresholds
├── requirements.txt            # Dependency specification
├── README.md                   # Operational guide & troubleshooting
│
├── telemetry/
│   ├── mavlink_client.py       # Threaded pymavlink receiver with reconnect loop
│   ├── telemetry_parser.py     # MAVLink message unpacker (ATTITUDE, GPS, BATTERY, etc.)
│   └── telemetry_state.py      # Thread-safe telemetry state container
│
├── ai/
│   ├── detector.py             # Abstract AI detector base & bounding box models
│   ├── human_detector.py       # Human / Stranded Person Detector module
│   ├── hazard_detector.py      # Environmental Hazard Detector (7 classes)
│   └── hailo_runner.py         # Hailo-8 HEF inference runner with PyTorch fallback
│
├── video/
│   └── video_stream.py         # Multi-source video pipeline (USB, RTSP, Video, Synthetic)
│
├── data/
│   ├── models.py               # TelemetryData, DetectionEvent, SystemMetrics dataclasses
│   ├── event_store.py          # Real-time filterable detection event storage
│   └── telemetry_store.py      # Bounded ring-buffer telemetry history logger
│
└── ui/
    ├── telemetry_panel.py      # Top status bar & telemetry sidebar cards
    ├── detection_panel.py      # Active detection summary & alert stream
    ├── map_panel.py            # PyDeck 2D/3D interactive map
    └── charts.py               # Historical telemetry Plotly analytics
```

---

## 3. Required Python Packages & Installation

Ensure Python 3.10+ is installed.

```bash
cd dashboard
pip install -r requirements.txt
```

Key dependencies: `streamlit`, `pymavlink`, `opencv-python`, `plotly`, `pydeck`, `pandas`, `psutil`, `ultralytics`.

---

## 4. How to Run the Dashboard

To launch the dashboard locally:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 5. System Operating Modes & Testing Instructions

### MODE 1: SIMULATION (ArduPilot SITL / Synthetic Telemetry)
- **Use Case**: Testing dashboard without physical UAV hardware.
- **Option A (Synthetic Default)**: Select `SIMULATION` mode in sidebar. Synthetic telemetry and camera frames will auto-generate.
- **Option B (ArduPilot SITL)**:
  1. Launch ArduPilot SITL simulator (e.g. `sim_vehicle.py -v ArduCopter --console --map`).
  2. Set connection string in sidebar to `udp:127.0.0.1:14550`.
  3. Click **🔄 Reconnect MAVLink**.

### MODE 2: RECORDED DATA REPLAY
- **Use Case**: Post-flight analysis or video demonstration.
- **Instructions**:
  1. Select `RECORDED_DATA` mode in sidebar.
  2. Choose `Test Video File (.mp4)` under Video Source and enter path to your recorded flight MP4 video.
  3. The video pipeline will play the recorded flight video and perform real-time AI detection overlay.

### MODE 3: REAL DRONE (Raspberry Pi 5 + Hailo-8 + ArduPilot)
- **Use Case**: Operational flight demonstration on physical UAV.
- **Instructions**:
  1. Select `REAL_DRONE` mode in sidebar.
  2. Connect Raspberry Pi 5 to ArduPilot Flight Controller via UART serial (e.g. `/dev/ttyAMA0` or `/dev/ttyUSB0`) or telemetry UDP socket.
  3. Enter `/dev/ttyAMA0` and baud rate `57600` (or `115200`).
  4. Ensure Hailo-8 AI HAT+ is configured and `hazards_yolov8n.hef` compiled file is placed in `models/hazards_yolov8n.hef`.

---

## 6. Raspberry Pi 5 Deployment Guide

1. **Install Raspberry Pi OS 64-bit (Bookworm)**.
2. **Install Hailo-8 Drivers & Hailo RT API**:
   ```bash
   sudo apt update && sudo apt install -y hailo-all
   hailortcli fw-control identify
   ```
3. **Configure UART Serial for MAVLink**:
   Enable UART in `/boot/firmware/config.txt` (`enable_uart=1`) and disable Linux serial console on `/dev/ttyAMA0`.
4. **Clone Dashboard & Run**:
   ```bash
   cd dashboard
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

---

## 7. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **"MAVLink DISCONNECTED / TIMEOUT"** | Incorrect connection port or missing heartbeat | Check UART wiring `/dev/ttyAMA0` or ensure SITL is broadcasting on `udp:127.0.0.1:14550`. |
| **"CAMERA OFFLINE"** | Camera index invalid or RTSP URL unreachable | Check camera index in sidebar (0, 1, or RTSP stream). Switch to `SYNTHETIC` mode to verify UI. |
| **"HAILO_INIT_ERROR"** | Hailo RT API not installed on non-Pi laptop | Expected behavior on dev laptops. System automatically falls back to PyTorch/OpenCV runner. |
| **"GPS Unavailable"** | Flight controller lacks 3D GPS fix | Wait for satellite lock. Dashboard displays default fallback map coordinates without crashing. |
| **High Latency / Lag** | Heavy UI rerun loop | Telemetry collection runs in background threads; UI reruns every 1.0s without blocking AI. |
