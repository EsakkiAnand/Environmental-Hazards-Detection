"""
Thread-Safe Active Telemetry State Manager
"""

import threading
import time
from typing import Optional, Dict, Any
from dashboard.data.models import TelemetryData
from dashboard.config import MODE_SIMULATION

class TelemetryState:
    """Thread-safe active telemetry state container."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryState, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance
    
    def _init_state(self):
        self._data = TelemetryData()
        self._active_mode = MODE_SIMULATION
        self._last_update_time = time.time()
        self._msg_count = 0
        self._calc_rate_timer = time.time()
        self._calculated_rate = 0.0

    def update_telemetry(self, updates: Dict[str, Any]):
        """Bulk update telemetry fields thread-safely."""
        with self._lock:
            now = time.time()
            for key, val in updates.items():
                if hasattr(self._data, key) and val is not None:
                    setattr(self._data, key, val)
            
            self._data.timestamp = now
            self._last_update_time = now
            self._msg_count += 1
            
            # Calculate update rate in Hz
            dt = now - self._calc_rate_timer
            if dt >= 1.0:
                self._calculated_rate = round(self._msg_count / dt, 1)
                self._data.message_rate_hz = self._calculated_rate
                self._msg_count = 0
                self._calc_rate_timer = now

    def get_snapshot(self) -> TelemetryData:
        """Return a copy of current telemetry data."""
        with self._lock:
            # Create a shallow copy of TelemetryData attributes
            snap = TelemetryData()
            for key in self._data.__dataclass_fields__:
                setattr(snap, key, getattr(self._data, key))
            return snap

    def set_mode(self, mode: str):
        """Set active dashboard mode."""
        with self._lock:
            self._active_mode = mode

    def get_mode(self) -> str:
        """Get active dashboard mode."""
        with self._lock:
            return self._active_mode

    def set_connection_status(self, status: str, heartbeat: bool = True):
        """Set MAVLink status."""
        with self._lock:
            self._data.mavlink_status = status
            self._data.heartbeat_received = heartbeat

    def is_connected(self) -> bool:
        """Check if MAVLink is actively connected."""
        with self._lock:
            return self._data.mavlink_status == "CONNECTED" and (time.time() - self._last_update_time < 3.5)
