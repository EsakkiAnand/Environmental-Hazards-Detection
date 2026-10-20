"""
Data Models for UAV Telemetry, AI Detections, and System Metrics
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import time

@dataclass
class TelemetryData:
    """Snapshot of UAV Flight Telemetry & System Health"""
    timestamp: float = field(default_factory=time.time)
    
    # Flight Status
    armed: bool = False
    flight_mode: str = "N/A"
    system_status: str = "N/A"
    safety_status: str = "N/A"
    ekf_status: str = "N/A"
    prearm_status: str = "N/A"
    
    # GPS / Position
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    relative_altitude: Optional[float] = None
    absolute_altitude: Optional[float] = None
    gps_fix_type: str = "N/A"
    satellites_visible: int = 0
    hdop: Optional[float] = None
    ground_speed: Optional[float] = None
    air_speed: Optional[float] = None
    heading: Optional[float] = None
    
    # Attitude
    roll: Optional[float] = None          # Degrees
    pitch: Optional[float] = None         # Degrees
    yaw: Optional[float] = None           # Degrees
    roll_rate: Optional[float] = None     # Deg/s
    pitch_rate: Optional[float] = None    # Deg/s
    yaw_rate: Optional[float] = None      # Deg/s
    
    # Velocity
    vertical_speed: Optional[float] = None  # Climb rate m/s
    vx: Optional[float] = None              # North velocity m/s
    vy: Optional[float] = None              # East velocity m/s
    vz: Optional[float] = None              # Down velocity m/s
    
    # Battery
    battery_percentage: Optional[int] = None
    voltage: Optional[float] = None         # Volts
    current: Optional[float] = None         # Amperes
    battery_remaining_mah: Optional[int] = None
    battery_health: str = "N/A"
    
    # Mission
    mission_state: str = "IDLE"             # IDLE, ACTIVE, PAUSED
    current_waypoint: int = 0
    target_waypoint: int = 0
    total_waypoints: int = 0
    mission_progress: float = 0.0           # Percentage 0 - 100
    distance_to_target: Optional[float] = None
    home_latitude: Optional[float] = None
    home_longitude: Optional[float] = None
    home_altitude: Optional[float] = None
    target_latitude: Optional[float] = None
    target_longitude: Optional[float] = None
    
    # Connection / System Health
    mavlink_status: str = "DISCONNECTED"   # CONNECTED, DISCONNECTED, TIMEOUT
    heartbeat_received: bool = False
    message_rate_hz: float = 0.0
    latency_ms: float = 0.0
    cpu_usage_pct: float = 0.0
    ram_usage_pct: float = 0.0
    hailo_status: str = "OFFLINE"           # ONLINE, OFFLINE, SIMULATED

@dataclass
class DetectionEvent:
    """Individual AI Detection Record"""
    event_id: str
    timestamp: float
    formatted_time: str
    detection_type: str                     # "HUMAN" or "HAZARD"
    class_name: str                         # e.g., "fire", "person", "flood"
    confidence: float
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    frame_image_base64: Optional[str] = None  # JPEG/PNG encoded for UI display
    extra_meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """Raspberry Pi 5 & Hailo-8 Resource Metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    temp_celsius: Optional[float] = None
    hailo_fps: float = 0.0
    hailo_latency_ms: float = 0.0
    hailo_temp_celsius: Optional[float] = None
    mavlink_hz: float = 0.0

@dataclass
class Waypoint:
    """Mission Waypoint Item"""
    seq: int
    latitude: float
    longitude: float
    altitude: float
    command: str = "NAV_WAYPOINT"
    is_current: bool = False
    is_target: bool = False
