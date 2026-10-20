"""
Threaded MAVLink Connection Client & Synthetic Telemetry Generator
"""

import threading
import time
import math
import random
import psutil
from typing import Optional
from dashboard.telemetry.telemetry_state import TelemetryState
from dashboard.telemetry.telemetry_parser import TelemetryParser
from dashboard.data.telemetry_store import TelemetryStore
from dashboard.config import (
    DEFAULT_MAVLINK_CONNECTION_STRING,
    DEFAULT_MAVLINK_BAUD,
    MAVLINK_TIMEOUT_SEC,
    MODE_SIMULATION,
    MODE_RECORDED,
    MODE_REAL_DRONE,
    DEFAULT_HOME_LAT,
    DEFAULT_HOME_LON,
    DEFAULT_HOME_ALT
)

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False

class MAVLinkClient:
    """Threaded MAVLink Receiver Client."""

    def __init__(
        self,
        connection_string: str = DEFAULT_MAVLINK_CONNECTION_STRING,
        baud: int = DEFAULT_MAVLINK_BAUD,
        telemetry_store: Optional[TelemetryStore] = None
    ):
        self.connection_string = connection_string
        self.baud = baud
        self.telemetry_store = telemetry_store
        self.telemetry_state = TelemetryState()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sim_thread: Optional[threading.Thread] = None
        self._mav_conn = None
        self._use_synthetic = False

    def start(self, use_synthetic: bool = False):
        """Start the background telemetry thread."""
        if self._running:
            return
        
        self._running = True
        self._use_synthetic = use_synthetic
        
        if use_synthetic or not HAS_PYMAVLINK:
            self._sim_thread = threading.Thread(target=self._synthetic_loop, daemon=True)
            self._sim_thread.start()
        else:
            self._thread = threading.Thread(target=self._mavlink_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop background threads."""
        self._running = False
        if self._mav_conn:
            try:
                self._mav_conn.close()
            except Exception:
                pass
            self._mav_conn = None

    def _mavlink_loop(self):
        """Main background MAVLink reception loop with reconnect logic."""
        while self._running:
            mode = self.telemetry_state.get_mode()
            if mode != MODE_REAL_DRONE and mode != MODE_SIMULATION:
                # If set to synthetic/recorded, fallback to synthetic loop
                self._run_synthetic_step()
                time.sleep(0.2)
                continue

            if self._mav_conn is None:
                self.telemetry_state.set_connection_status("CONNECTING...", heartbeat=False)
                try:
                    self._mav_conn = mavutil.mavlink_connection(
                        self.connection_string,
                        baud=self.baud,
                        autoreconnect=True
                    )
                    # Request data streams
                    self._mav_conn.wait_heartbeat(timeout=3)
                    self.telemetry_state.set_connection_status("CONNECTED", heartbeat=True)
                    self._request_data_streams()
                except Exception as e:
                    self.telemetry_state.set_connection_status("DISCONNECTED", heartbeat=False)
                    self._mav_conn = None
                    time.sleep(2.0)
                    continue

            try:
                msg = self._mav_conn.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    # Timeout check
                    if not self.telemetry_state.is_connected():
                        self.telemetry_state.set_connection_status("TIMEOUT", heartbeat=False)
                    continue
                
                updates = TelemetryParser.parse_message(msg)
                if updates:
                    # Add Raspberry Pi 5 CPU/RAM stats
                    updates['cpu_usage_pct'] = round(psutil.cpu_percent(), 1)
                    updates['ram_usage_pct'] = round(psutil.virtual_memory().percent, 1)
                    
                    self.telemetry_state.update_telemetry(updates)
                    
                    if self.telemetry_store:
                        snapshot = self.telemetry_state.get_snapshot()
                        self.telemetry_store.add_snapshot(snapshot)

            except Exception as e:
                self.telemetry_state.set_connection_status("DISCONNECTED", heartbeat=False)
                if self._mav_conn:
                    try:
                        self._mav_conn.close()
                    except Exception:
                        pass
                    self._mav_conn = None
                time.sleep(2.0)

    def _request_data_streams(self):
        """Request standard telemetry data streams from ArduPilot."""
        if not self._mav_conn:
            return
        try:
            self._mav_conn.mav.request_data_stream_send(
                self._mav_conn.target_system,
                self._mav_conn.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10 Hz rate
                1
            )
        except Exception:
            pass

    def _synthetic_loop(self):
        """Synthetic telemetry generator loop for simulation & offline testing."""
        while self._running:
            self._run_synthetic_step()
            time.sleep(0.2)  # 5 Hz synthetic rate

    def _run_synthetic_step(self):
        """Generate smooth, realistic synthetic flight telemetry."""
        t = time.time()
        
        # Simulate circular flight orbit around Home position
        radius = 0.0015  # ~150 meters radius
        speed = 0.05
        angle = t * speed
        
        lat = DEFAULT_HOME_LAT + radius * math.cos(angle)
        lon = DEFAULT_HOME_LON + radius * math.sin(angle)
        heading = round((math.degrees(angle) + 90) % 360, 1)
        rel_alt = round(25.0 + 3.0 * math.sin(t * 0.2), 2)
        abs_alt = round(DEFAULT_HOME_ALT + rel_alt, 2)
        
        roll = round(5.0 * math.sin(t * 0.5), 1)
        pitch = round(3.0 * math.cos(t * 0.3), 1)
        
        ground_speed = round(8.5 + 0.5 * math.sin(t), 2)
        vertical_speed = round(0.2 * math.cos(t * 0.2), 2)
        
        # Battery depletion simulation
        battery_pct = max(15, int(98 - ((t % 1800) / 1800.0) * 80))
        voltage = round(14.8 * (battery_pct / 100.0) + 1.2, 2)
        current = round(12.5 + random.uniform(-0.5, 0.5), 2)
        
        updates = {
            'mavlink_status': 'CONNECTED',
            'heartbeat_received': True,
            'armed': True,
            'flight_mode': 'GUIDED',
            'system_status': 'ACTIVE',
            'safety_status': 'SAFE',
            'ekf_status': 'OK',
            'prearm_status': 'PASSED',
            'latitude': lat,
            'longitude': lon,
            'relative_altitude': rel_alt,
            'absolute_altitude': abs_alt,
            'gps_fix_type': '3D Fix',
            'satellites_visible': 16,
            'hdop': 0.8,
            'ground_speed': ground_speed,
            'air_speed': round(ground_speed + 0.4, 2),
            'heading': heading,
            'roll': roll,
            'pitch': pitch,
            'yaw': heading,
            'roll_rate': round(random.uniform(-1.0, 1.0), 2),
            'pitch_rate': round(random.uniform(-1.0, 1.0), 2),
            'yaw_rate': round(random.uniform(-0.5, 0.5), 2),
            'vertical_speed': vertical_speed,
            'vx': round(ground_speed * math.cos(math.radians(heading)), 2),
            'vy': round(ground_speed * math.sin(math.radians(heading)), 2),
            'vz': round(-vertical_speed, 2),
            'battery_percentage': battery_pct,
            'voltage': voltage,
            'current': current,
            'battery_remaining_mah': battery_pct * 40,
            'battery_health': 'GOOD',
            'mission_state': 'ACTIVE',
            'current_waypoint': 3,
            'target_waypoint': 4,
            'total_waypoints': 8,
            'mission_progress': 42.5,
            'distance_to_target': round(84.2 - (t % 10), 1),
            'home_latitude': DEFAULT_HOME_LAT,
            'home_longitude': DEFAULT_HOME_LON,
            'home_altitude': DEFAULT_HOME_ALT,
            'target_latitude': DEFAULT_HOME_LAT + 0.002,
            'target_longitude': DEFAULT_HOME_LON + 0.002,
            'cpu_usage_pct': round(psutil.cpu_percent(), 1),
            'ram_usage_pct': round(psutil.virtual_memory().percent, 1),
            'hailo_status': 'ONLINE'
        }
        
        self.telemetry_state.update_telemetry(updates)
        
        if self.telemetry_store:
            snapshot = self.telemetry_state.get_snapshot()
            self.telemetry_store.add_snapshot(snapshot)
