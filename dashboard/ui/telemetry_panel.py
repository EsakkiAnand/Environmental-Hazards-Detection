"""
Streamlit UI Component: Telemetry Panels & Top Status Bar
"""

import streamlit as st
from dashboard.data.models import TelemetryData

def render_top_status_bar(telemetry: TelemetryData, active_mode: str):
    """Render top ground station status bar."""
    
    # Status badges
    conn_color = "🟢" if telemetry.mavlink_status == "CONNECTED" else "🔴"
    arm_badge = "🔴 ARMED" if telemetry.armed else "⚪ DISARMED"
    mode_str = f"🛸 {telemetry.flight_mode}"
    
    gps_str = f"🛰️ {telemetry.satellites_visible} SAT ({telemetry.gps_fix_type})" if telemetry.satellites_visible > 0 else "🛰️ GPS N/A"
    
    bat_pct = telemetry.battery_percentage
    if bat_pct is not None:
        bat_icon = "🔋" if bat_pct > 30 else "🪫"
        bat_str = f"{bat_icon} {bat_pct}% ({telemetry.voltage}V)"
    else:
        bat_str = "🔋 BAT N/A"
        
    ai_status = f"🧠 AI: {telemetry.hailo_status}"

    c1, c2, c3, c4, c5, c6 = st.columns([1.8, 1.2, 1.2, 1.8, 1.8, 1.4])
    
    with c1:
        st.markdown(f"**MAVLink**: {conn_color} `{telemetry.mavlink_status}`")
    with c2:
        st.markdown(f"**Status**: `{arm_badge}`")
    with c3:
        st.markdown(f"**Mode**: `{mode_str}`")
    with c4:
        st.markdown(f"**GPS**: `{gps_str}`")
    with c5:
        st.markdown(f"**Battery**: `{bat_str}`")
    with c6:
        st.markdown(f"**System**: `{ai_status}`")

    st.caption(f"⚙️ **ACTIVE DASHBOARD MODE**: `{active_mode}` | Update Rate: `{telemetry.message_rate_hz:.1f} Hz` | RPi CPU: `{telemetry.cpu_usage_pct}%` | RAM: `{telemetry.ram_usage_pct}%`")

def render_telemetry_metrics_panel(telemetry: TelemetryData):
    """Render detailed right sidebar telemetry cards."""
    st.markdown("### 📊 Flight Telemetry")
    
    t1, t2 = st.tabs(["🚀 Navigation", "🔋 Power & Health"])
    
    with t1:
        # Position & Altitude
        st.markdown("#### 📍 Position & Altitude")
        c1, c2, c3 = st.columns(3)
        with c1:
            rel_alt_str = f"{telemetry.relative_altitude:.1f} m" if telemetry.relative_altitude is not None else "N/A"
            st.metric("Rel Altitude", rel_alt_str)
        with c2:
            abs_alt_str = f"{telemetry.absolute_altitude:.1f} m" if telemetry.absolute_altitude is not None else "N/A"
            st.metric("Abs Altitude", abs_alt_str)
        with c3:
            spd_str = f"{telemetry.ground_speed:.1f} m/s" if telemetry.ground_speed is not None else "N/A"
            st.metric("Ground Speed", spd_str)

        st.caption(
            f"Lat: `{telemetry.latitude if telemetry.latitude is not None else 'N/A'}` | "
            f"Lon: `{telemetry.longitude if telemetry.longitude is not None else 'N/A'}` | "
            f"Heading: `{telemetry.heading if telemetry.heading is not None else 'N/A'}°`"
        )
        
        st.divider()
        
        # Attitude
        st.markdown("#### 🔄 Attitude & Rates")
        a1, a2, a3 = st.columns(3)
        with a1:
            roll_str = f"{telemetry.roll:.1f}°" if telemetry.roll is not None else "N/A"
            st.metric("Roll", roll_str)
        with a2:
            pitch_str = f"{telemetry.pitch:.1f}°" if telemetry.pitch is not None else "N/A"
            st.metric("Pitch", pitch_str)
        with a3:
            yaw_str = f"{telemetry.yaw:.1f}°" if telemetry.yaw is not None else "N/A"
            st.metric("Yaw", yaw_str)

    with t2:
        # Power
        st.markdown("#### ⚡ Power System")
        p1, p2, p3 = st.columns(3)
        with p1:
            v_str = f"{telemetry.voltage:.2f} V" if telemetry.voltage is not None else "N/A"
            st.metric("Voltage", v_str)
        with p2:
            i_str = f"{telemetry.current:.1f} A" if telemetry.current is not None else "N/A"
            st.metric("Current", i_str)
        with p3:
            b_str = f"{telemetry.battery_percentage}%" if telemetry.battery_percentage is not None else "N/A"
            st.metric("Remaining", b_str)

        st.divider()
        
        # Mission
        st.markdown("#### 🎯 Mission Progress")
        m1, m2 = st.columns(2)
        with m1:
            st.write(f"**State**: `{telemetry.mission_state}`")
            st.write(f"**Current Waypoint**: `{telemetry.current_waypoint}` / `{telemetry.total_waypoints}`")
        with m2:
            dist_str = f"{telemetry.distance_to_target:.1f} m" if telemetry.distance_to_target is not None else "N/A"
            st.write(f"**Dist to Target**: `{dist_str}`")
            st.progress(min(1.0, max(0.0, telemetry.mission_progress / 100.0)))
