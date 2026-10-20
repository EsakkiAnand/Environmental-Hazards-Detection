"""
MAVLink Message Parser for ArduPilot Telemetry
"""

import math
from typing import Dict, Any, Optional

# ArduPilot Custom Flight Modes Mapping
ARDUPILOT_MODE_MAP = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE"
}

# GPS Fix Type Mapping
GPS_FIX_TYPE_MAP = {
    0: "No GPS",
    1: "No Fix",
    2: "2D Fix",
    3: "3D Fix",
    4: "DGPS",
    5: "RTK Float",
    6: "RTK Fixed",
    7: "Static",
    8: "PPP"
}

# System Status Mapping
SYS_STATUS_MAP = {
    0: "UNINIT",
    1: "BOOT",
    2: "CALIBRATING",
    3: "STANDBY",
    4: "ACTIVE",
    5: "CRITICAL",
    6: "EMERGENCY",
    7: "POWEROFF",
    8: "TERMINATION"
}

class TelemetryParser:
    """Decodes MAVLink message objects into telemetry dictionary updates."""
    
    @staticmethod
    def parse_message(msg) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        msg_type = msg.get_type()
        
        if msg_type == 'HEARTBEAT':
            # Armed state check (base_mode bit 128 = MAV_MODE_FLAG_SAFETY_ARMED)
            armed = bool(msg.base_mode & 128)
            updates['armed'] = armed
            
            # Flight mode
            custom_mode = msg.custom_mode
            flight_mode = ARDUPILOT_MODE_MAP.get(custom_mode, f"MODE_{custom_mode}")
            updates['flight_mode'] = flight_mode
            
            # System status
            system_status = SYS_STATUS_MAP.get(msg.system_status, "UNKNOWN")
            updates['system_status'] = system_status
            updates['mavlink_status'] = "CONNECTED"
            updates['heartbeat_received'] = True

        elif msg_type == 'GLOBAL_POSITION_INT':
            # Lat/Lon in deg * 1e7
            updates['latitude'] = msg.lat / 1e7
            updates['longitude'] = msg.lon / 1e7
            updates['relative_altitude'] = round(msg.relative_alt / 1000.0, 2)  # mm to m
            updates['absolute_altitude'] = round(msg.alt / 1000.0, 2)           # mm to m
            
            # Ground velocities in cm/s to m/s
            vx = msg.vx / 100.0
            vy = msg.vy / 100.0
            vz = msg.vz / 100.0
            updates['vx'] = round(vx, 2)
            updates['vy'] = round(vy, 2)
            updates['vz'] = round(vz, 2)
            updates['vertical_speed'] = round(-vz, 2)  # Negative z is UP in NED frame
            updates['ground_speed'] = round(math.sqrt(vx*vx + vy*vy), 2)
            
            # Heading (cdeg to deg)
            if msg.hdg != 65535:
                updates['heading'] = round(msg.hdg / 100.0, 1)

        elif msg_type == 'GPS_RAW_INT':
            updates['gps_fix_type'] = GPS_FIX_TYPE_MAP.get(msg.fix_type, f"FIX_{msg.fix_type}")
            updates['satellites_visible'] = msg.satellites_visible
            if msg.eph != 65535:
                updates['hdop'] = round(msg.eph / 100.0, 2)

        elif msg_type == 'ATTITUDE':
            # Radians to Degrees
            updates['roll'] = round(math.degrees(msg.roll), 1)
            updates['pitch'] = round(math.degrees(msg.pitch), 1)
            updates['yaw'] = round(math.degrees(msg.yaw) % 360, 1)
            updates['roll_rate'] = round(math.degrees(msg.rollspeed), 2)
            updates['pitch_rate'] = round(math.degrees(msg.pitchspeed), 2)
            updates['yaw_rate'] = round(math.degrees(msg.yawspeed), 2)

        elif msg_type == 'VFR_HUD':
            updates['air_speed'] = round(msg.airspeed, 2)
            updates['ground_speed'] = round(msg.groundspeed, 2)
            updates['heading'] = round(msg.heading, 1)
            updates['relative_altitude'] = round(msg.alt, 2)
            updates['vertical_speed'] = round(msg.climb, 2)

        elif msg_type == 'SYS_STATUS':
            # Battery voltage (mV to V), current (cA to A), percentage
            if msg.voltage_battery != 65535:
                updates['voltage'] = round(msg.voltage_battery / 1000.0, 2)
            if msg.current_battery != -1:
                updates['current'] = round(msg.current_battery / 100.0, 2)
            if msg.battery_remaining != -1:
                updates['battery_percentage'] = msg.battery_remaining
                
        elif msg_type == 'BATTERY_STATUS':
            if msg.battery_remaining != -1:
                updates['battery_percentage'] = msg.battery_remaining
            if msg.current_battery != -1:
                updates['current'] = round(msg.current_battery / 100.0, 2)
            if hasattr(msg, 'voltages') and len(msg.voltages) > 0 and msg.voltages[0] != 65535:
                total_v = sum(v for v in msg.voltages if v != 65535) / 1000.0
                updates['voltage'] = round(total_v, 2)

        elif msg_type == 'HOME_POSITION':
            updates['home_latitude'] = msg.latitude / 1e7
            updates['home_longitude'] = msg.longitude / 1e7
            updates['home_altitude'] = msg.altitude / 1000.0

        elif msg_type == 'MISSION_CURRENT':
            updates['current_waypoint'] = msg.seq

        elif msg_type == 'NAV_CONTROLLER_OUTPUT':
            updates['target_waypoint'] = msg.wp_dist
            updates['distance_to_target'] = round(msg.wp_dist, 1)

        elif msg_type == 'EKF_STATUS_REPORT':
            # EKF flags summary
            flags = msg.flags
            if flags > 0:
                updates['ekf_status'] = "OK"
            else:
                updates['ekf_status'] = "WARN"

        return updates
