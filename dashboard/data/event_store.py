"""
Thread-Safe AI Detection Event Store
"""

import threading
from typing import List, Optional, Dict, Any
import pandas as pd
from dashboard.data.models import DetectionEvent

class EventStore:
    """Stores and manages detection events with thread safety."""
    
    def __init__(self, max_events: int = 500):
        self._lock = threading.Lock()
        self._max_events = max_events
        self._events: List[DetectionEvent] = []
    
    def add_event(self, event: DetectionEvent):
        """Append a new detection event."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)
    
    def get_all_events(self) -> List[DetectionEvent]:
        """Return shallow copy of all stored events."""
        with self._lock:
            return list(self._events)
    
    def get_latest_events(self, limit: int = 10) -> List[DetectionEvent]:
        """Return the N most recent detection events."""
        with self._lock:
            return list(reversed(self._events[-limit:]))
    
    def filter_events(
        self,
        det_type: Optional[str] = None,
        class_name: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[DetectionEvent]:
        """Filter events by type, class, and confidence."""
        with self._lock:
            filtered = []
            for ev in reversed(self._events):
                if det_type and det_type != "ALL" and ev.detection_type != det_type:
                    continue
                if class_name and class_name != "ALL" and ev.class_name != class_name:
                    continue
                if ev.confidence < min_confidence:
                    continue
                filtered.append(ev)
            return filtered

    def get_summary_counts(self) -> Dict[str, int]:
        """Return counts by type and class."""
        with self._lock:
            counts = {
                "total": len(self._events),
                "human": 0,
                "hazard": 0
            }
            class_counts: Dict[str, int] = {}
            for ev in self._events:
                if ev.detection_type == "HUMAN":
                    counts["human"] += 1
                elif ev.detection_type == "HAZARD":
                    counts["hazard"] += 1
                
                class_counts[ev.class_name] = class_counts.get(ev.class_name, 0) + 1
            
            counts["by_class"] = class_counts
            return counts

    def to_dataframe(self, events: Optional[List[DetectionEvent]] = None) -> pd.DataFrame:
        """Convert events to Pandas DataFrame for Streamlit UI displaying."""
        target_list = events if events is not None else self.get_all_events()
        
        rows = []
        for ev in target_list:
            lat_str = f"{ev.latitude:.6f}" if ev.latitude is not None else "GPS N/A"
            lon_str = f"{ev.longitude:.6f}" if ev.longitude is not None else "GPS N/A"
            alt_str = f"{ev.altitude:.1f} m" if ev.altitude is not None else "N/A"
            
            rows.append({
                "Event ID": ev.event_id,
                "Timestamp": ev.formatted_time,
                "Type": ev.detection_type,
                "Class": ev.class_name.upper().replace("_", " "),
                "Confidence": f"{ev.confidence:.2f}",
                "Latitude": lat_str,
                "Longitude": lon_str,
                "Altitude": alt_str
            })
        
        if not rows:
            return pd.DataFrame(columns=[
                "Event ID", "Timestamp", "Type", "Class", "Confidence", "Latitude", "Longitude", "Altitude"
            ])
            
        return pd.DataFrame(rows)

    def clear(self):
        """Clear all stored events."""
        with self._lock:
            self._events.clear()
