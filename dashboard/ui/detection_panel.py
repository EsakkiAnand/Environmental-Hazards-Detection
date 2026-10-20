"""
Streamlit UI Component: AI Detections & Event Table Panel
"""

import streamlit as st
import pandas as pd
from typing import List, Optional
from dashboard.data.event_store import EventStore
from dashboard.config import HAZARD_CLASSES

def render_active_detections_summary(event_store: EventStore):
    """Render summary card of AI detections."""
    st.markdown("### 🧠 AI Detections Summary")
    
    counts = event_store.get_summary_counts()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Events", counts["total"])
    with c2:
        st.metric("👤 Human Detections", counts["human"])
    with c3:
        st.metric("⚠️ Hazards Detected", counts["hazard"])

    st.markdown("#### 🚨 Breakdown by Class")
    class_counts = counts.get("by_class", {})
    if not class_counts:
        st.info("No AI detections recorded yet. Scanner active.")
    else:
        for cls_name, count in class_counts.items():
            st.write(f"• **{cls_name.upper().replace('_', ' ')}**: `{count}` detections")

def render_recent_alerts_stream(event_store: EventStore, limit: int = 5):
    """Render live stream of latest detection alerts."""
    st.markdown("### 🔔 Recent AI Alerts")
    latest = event_store.get_latest_events(limit=limit)
    
    if not latest:
        st.caption("No recent alerts.")
        return

    for ev in latest:
        icon = "👤" if ev.detection_type == "HUMAN" else "⚠️"
        lat_str = f"{ev.latitude:.5f}" if ev.latitude is not None else "N/A"
        lon_str = f"{ev.longitude:.5f}" if ev.longitude is not None else "N/A"
        alt_str = f"{ev.altitude:.1f}m" if ev.altitude is not None else "N/A"
        
        with st.expander(f"{icon} **{ev.class_name.upper()}** (Conf: {ev.confidence:.2f}) @ `{ev.formatted_time}`", expanded=False):
            st.write(f"**Event ID**: `{ev.event_id}`")
            st.write(f"**Coordinates**: Lat `{lat_str}`, Lon `{lon_str}`, Alt `{alt_str}`")
            if ev.frame_image_base64:
                st.image(f"data:image/jpeg;base64,{ev.frame_image_base64}", caption=f"Detection Frame: {ev.class_name}")

def render_detection_event_table(event_store: EventStore):
    """Render full filterable detection table with CSV download capability."""
    st.markdown("### 📋 Real-Time Detection Event Log")
    
    f1, f2, f3 = st.columns([1.5, 1.5, 2.0])
    with f1:
        type_filter = st.selectbox("Filter Type", ["ALL", "HUMAN", "HAZARD"], index=0)
    with f2:
        class_options = ["ALL", "person"] + HAZARD_CLASSES
        class_filter = st.selectbox("Filter Class", class_options, index=0)
    with f3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.40, step=0.05)

    filtered_events = event_store.filter_events(
        det_type=type_filter,
        class_name=class_filter,
        min_confidence=min_conf
    )

    df = event_store.to_dataframe(filtered_events)
    
    if df.empty:
        st.info("No events match current filter criteria.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # CSV Export
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Detection Log (CSV)",
            data=csv_bytes,
            file_name="uav_detection_log.csv",
            mime="text/csv"
        )
