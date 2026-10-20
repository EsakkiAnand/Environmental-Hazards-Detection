"""
Streamlit UI Component: PyDeck Real-Time Interactive UAV Map
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
from typing import List, Optional
from dashboard.data.models import TelemetryData
from dashboard.data.event_store import EventStore
from dashboard.data.telemetry_store import TelemetryStore
from dashboard.config import COLOR_HUMAN_RGB, HAZARD_COLOR_MAP, DEFAULT_HOME_LAT, DEFAULT_HOME_LON

def render_realtime_map(telemetry: TelemetryData, event_store: EventStore, telemetry_store: TelemetryStore):
    """Render PyDeck interactive ground station map."""
    st.markdown("### 🗺️ Live Mission Map & Target Locations")
    
    lat = telemetry.latitude
    lon = telemetry.longitude
    
    if lat is None or lon is None:
        st.warning("⚠️ GPS Unavailable — Waiting for 3D Lock")
        # Display fallback location
        lat = DEFAULT_HOME_LAT
        lon = DEFAULT_HOME_LON

    # Build layers
    layers = []
    
    # 1. Drone Current Location Layer
    drone_df = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "heading": telemetry.heading if telemetry.heading is not None else 0,
        "alt": telemetry.relative_altitude if telemetry.relative_altitude is not None else 0,
        "label": f"DRONE (Alt: {telemetry.relative_altitude}m, Spd: {telemetry.ground_speed}m/s)"
    }])
    
    drone_layer = pdk.Layer(
        "ScatterplotLayer",
        drone_df,
        get_position=["lon", "lat"],
        get_color=[0, 255, 255, 255],  # Bright Cyan
        get_radius=15,
        radius_min_pixels=10,
        radius_max_pixels=25,
        pickable=True
    )
    layers.append(drone_layer)

    # 2. Flight Trajectory Path Layer
    hist_df = telemetry_store.get_dataframe()
    if not hist_df.empty and "latitude" in hist_df.columns and "longitude" in hist_df.columns:
        valid_coords = hist_df.dropna(subset=["latitude", "longitude"])
        if len(valid_coords) > 1:
            path_coords = valid_coords[["longitude", "latitude"]].values.tolist()
            path_df = pd.DataFrame([{"path": path_coords}])
            
            path_layer = pdk.Layer(
                "PathLayer",
                path_df,
                get_path="path",
                get_color=[0, 229, 255, 180],
                width_min_pixels=3,
                pickable=False
            )
            layers.append(path_layer)

    # 3. Detection Event Markers Layer
    events = event_store.get_all_events()
    det_rows = []
    for ev in events:
        if ev.latitude is not None and ev.longitude is not None:
            if ev.detection_type == "HUMAN":
                color = COLOR_HUMAN_RGB + [220]
            else:
                color_meta = HAZARD_COLOR_MAP.get(ev.class_name, {"rgb": [255, 0, 0]})
                color = color_meta["rgb"] + [220]
                
            det_rows.append({
                "lat": ev.latitude,
                "lon": ev.longitude,
                "type": ev.detection_type,
                "class": ev.class_name.upper(),
                "confidence": ev.confidence,
                "time": ev.formatted_time,
                "color": color,
                "label": f"{ev.class_name.upper()} ({ev.confidence:.2f}) @ {ev.formatted_time}"
            })
            
    if det_rows:
        det_df = pd.DataFrame(det_rows)
        det_layer = pdk.Layer(
            "ScatterplotLayer",
            det_df,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius=18,
            radius_min_pixels=8,
            radius_max_pixels=20,
            pickable=True
        )
        layers.append(det_layer)

    # View state centered on drone
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=16,
        pitch=35,
        bearing=0
    )

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{label}"}
    )

    st.pydeck_chart(deck, use_container_width=True)
