"""
Bounded Ring-Buffer Store for Telemetry Analytics & Charts
"""

import threading
from collections import deque
from typing import List, Dict, Any, Optional
import pandas as pd
import time
from dashboard.data.models import TelemetryData
from dashboard.config import TELEMETRY_HISTORY_MAX_LEN

class TelemetryStore:
    """Thread-safe bounded ring buffer for historical telemetry snapshots."""
    
    def __init__(self, max_len: int = TELEMETRY_HISTORY_MAX_LEN):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=max_len)
        self._start_time = time.time()
    
    def add_snapshot(self, data: TelemetryData):
        """Add a telemetry snapshot to history."""
        with self._lock:
            self._buffer.append(data)
    
    def get_history(self) -> List[TelemetryData]:
        """Return list copy of all stored telemetry snapshots."""
        with self._lock:
            return list(self._buffer)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert telemetry history into a Pandas DataFrame for Plotly charts."""
        with self._lock:
            snapshots = list(self._buffer)
        
        if not snapshots:
            return pd.DataFrame()
        
        rows = []
        for snap in snapshots:
            elapsed_sec = round(snap.timestamp - self._start_time, 1)
            time_str = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))
            
            rows.append({
                "timestamp": snap.timestamp,
                "elapsed_sec": elapsed_sec,
                "time_str": time_str,
                "armed": snap.armed,
                "mode": snap.flight_mode,
                "latitude": snap.latitude,
                "longitude": snap.longitude,
                "rel_altitude": snap.relative_altitude,
                "abs_altitude": snap.absolute_altitude,
                "ground_speed": snap.ground_speed,
                "air_speed": snap.air_speed,
                "vertical_speed": snap.vertical_speed,
                "roll": snap.roll,
                "pitch": snap.pitch,
                "yaw": snap.yaw,
                "battery_pct": snap.battery_percentage,
                "voltage": snap.voltage,
                "current": snap.current,
                "satellites": snap.satellites_visible,
                "hdop": snap.hdop,
                "message_rate_hz": snap.message_rate_hz,
                "cpu_usage": snap.cpu_usage_pct,
                "ram_usage": snap.ram_usage_pct
            })
        
        return pd.DataFrame(rows)

    def clear(self):
        """Clear historical snapshots."""
        with self._lock:
            self._buffer.clear()
            self._start_time = time.time()
