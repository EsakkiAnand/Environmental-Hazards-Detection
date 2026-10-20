"""
Streamlit UI Component: Historical Telemetry Analytics & Plotly Charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.data.telemetry_store import TelemetryStore

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(14, 17, 23, 1)',
    plot_bgcolor='rgba(22, 27, 34, 1)',
    font=dict(color='#E0E6ED', family='Inter, sans-serif'),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(gridcolor='#262C36', showgrid=True),
    yaxis=dict(gridcolor='#262C36', showgrid=True),
    legend=dict(orientation='h', y=1.1)
)

def render_telemetry_history_charts(telemetry_store: TelemetryStore):
    """Render bounded historical telemetry charts."""
    st.markdown("### 📈 Historical Telemetry Analytics")
    
    df = telemetry_store.get_dataframe()
    if df.empty or len(df) < 2:
        st.info("Gathering historical telemetry data...")
        return

    c1, c2 = st.columns(2)
    
    # 1. Altitude vs Time
    with c1:
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(
            x=df["time_str"], y=df["rel_altitude"],
            mode='lines', name='Rel Alt (m)',
            line=dict(color='#00E5FF', width=2)
        ))
        fig_alt.add_trace(go.Scatter(
            x=df["time_str"], y=df["abs_altitude"],
            mode='lines', name='Abs Alt (m)',
            line=dict(color='#FF9900', width=1.5, dash='dash')
        ))
        fig_alt.update_layout(title="Altitude vs Time", yaxis_title="Altitude (m)", **DARK_LAYOUT)
        st.plotly_chart(fig_alt, use_container_width=True)

    # 2. Speed vs Time
    with c2:
        fig_spd = go.Figure()
        fig_spd.add_trace(go.Scatter(
            x=df["time_str"], y=df["ground_speed"],
            mode='lines', name='Ground Speed (m/s)',
            line=dict(color='#00FF66', width=2)
        ))
        fig_spd.add_trace(go.Scatter(
            x=df["time_str"], y=df["vertical_speed"],
            mode='lines', name='Vertical Speed (m/s)',
            line=dict(color='#FF3366', width=1.5)
        ))
        fig_spd.update_layout(title="Speed vs Time", yaxis_title="Speed (m/s)", **DARK_LAYOUT)
        st.plotly_chart(fig_spd, use_container_width=True)

    c3, c4 = st.columns(2)

    # 3. Battery vs Time
    with c3:
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Scatter(
            x=df["time_str"], y=df["battery_pct"],
            mode='lines', name='Battery %',
            line=dict(color='#FFCC00', width=2)
        ))
        fig_bat.add_trace(go.Scatter(
            x=df["time_str"], y=df["voltage"],
            mode='lines', name='Voltage (V)', yaxis='y2',
            line=dict(color='#FF5500', width=1.5, dash='dot')
        ))
        layout_kwargs = DARK_LAYOUT.copy()
        layout_kwargs.pop('yaxis', None)
        fig_bat.update_layout(
            title="Battery & Voltage vs Time",
            yaxis=dict(title="Battery %", gridcolor='#262C36', showgrid=True),
            yaxis2=dict(title="Voltage (V)", overlaying='y', side='right', showgrid=False),
            **layout_kwargs
        )
        st.plotly_chart(fig_bat, use_container_width=True)

    # 4. Attitude vs Time
    with c4:
        fig_att = go.Figure()
        fig_att.add_trace(go.Scatter(x=df["time_str"], y=df["roll"], mode='lines', name='Roll (°)', line=dict(color='#00E5FF')))
        fig_att.add_trace(go.Scatter(x=df["time_str"], y=df["pitch"], mode='lines', name='Pitch (°)', line=dict(color='#FF00FF')))
        fig_att.add_trace(go.Scatter(x=df["time_str"], y=df["yaw"], mode='lines', name='Yaw (°)', line=dict(color='#FFFF00')))
        fig_att.update_layout(title="Attitude Angles vs Time", yaxis_title="Degrees (°)", **DARK_LAYOUT)
        st.plotly_chart(fig_att, use_container_width=True)
