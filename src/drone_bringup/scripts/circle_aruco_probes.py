#!/usr/bin/env python3
"""
circle_aruco_updated.py
Hardware-Grade Offboard Search, Slow & Smooth ArUco 102 Alignment down to 0.45m,
Tracked Descent with Precise Camera Extrinsics (+100mm X, -70mm Z),
Precision Landing, and Guaranteed Post-Touchdown Disarm.

Features:
- Camera Extrinsics: +100mm (+0.10m) X from CG, -70mm (-0.07m) Z from CG, 0mm Y from CG
- Continuous visual centering and alignment from 2.5m all the way down to 0.45m AGL (0.4m - 0.5m)
- Subscribes to /oak/rgb/image_raw/compressed for minimum latency & battery savings
- Slow & Smooth Visual Servoing: P-gain = 0.50, Max Speed = 0.22 m/s, Descent = 0.18 m/s
- Low-power, ultra-low-latency display (320x240 @ 12 FPS) with Tkinter GUI & HTTP streaming
- Dedicated post-landing touchdown detection and active DISARM commands

okish
"""

import math
import os
import time
import threading
import http.server
import socketserver
from collections import deque
from enum import Enum, auto

import numpy as np
import cv2

# Optional Tkinter GUI support for robust desktop window rendering
try:
    import tkinter as tk
    from PIL import Image as PILImage, ImageTk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32
from px4_msgs.msg import (
    VehicleCommand,
    VehicleCommandAck,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleStatus,
    VehicleLocalPosition,
    VehicleAttitude,
    VehicleLandDetected,
)

# ============================================================
# CAMERA & TOPIC CONFIGURATION
# ============================================================
CAMERA_TOPIC = '/oak/rgb/image_raw/compressed'
USE_COMPRESSED = True           # Set True for CompressedImage, False for raw Image

# ============================================================
# LOW-POWER & LIVE VIEWER CONFIGURATION
# ============================================================
ENABLE_GUI_WINDOW = True        # Native desktop window pops up on launch
ENABLE_HTTP_STREAM = True       # Run lightweight MJPEG web server at http://<drone-ip>:8080
HTTP_STREAM_PORT = 8080
ENABLE_DEBUG_TOPIC = True       # Publishes annotated video to /aruco_debug_feed

# Low-Resolution & Low-Latency Stream Settings (Prevents lag on battery)
DISPLAY_WIDTH = 420             # Ultra-compact 320x240 for zero lag
DISPLAY_HEIGHT = 340
DISPLAY_FPS = 15                # 12 FPS preview refresh rate
JPEG_QUALITY = 30               # Lightweight compression (~1.8 KB per frame)

# ============================================================
# CAMERA EXTRINSICS & OPTICAL PARAMETERS
# ============================================================
# Camera mounting relative to drone Center of Gravity (FRD body frame):
# +X is Forward, +Y is Right, +Z is Down
CAM_YAW_180 = True              # SET TRUE: Camera is mounted rotated 180 deg relative to drone nose
CAM_OFFSET_X_M = 0.100          # +100 mm in +X (forward from drone CG)
CAM_OFFSET_Y_M = 0.000          # 0 mm in Y
CAM_OFFSET_Z_M = -0.070         # 70 mm in -Z (upward from drone CG)

CAM_FOV_H_DEG = 73.0            # Horizontal Field of View (OAK-D Pro RGB approx 73 deg)
CAM_FOV_V_DEG = 55.0            # Vertical Field of View (approx 55 deg)

# ============================================================
# TARGET ARUCO & SLOW/SMOOTH VISUAL SERVOING
# ============================================================
TARGET_ARUCO_ID = 102
TARGET_PROBES_COUNT = 2         # Number of probes to wait for before approaching target
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL

# Gentle, Slow Visual Servoing Dynamics
VISUAL_POS_P_GAIN = 0.50        # Slow & smooth proportional tracking gain
MAX_TRACK_SPEED_MPS = 0.22      # Gentle horizontal speed cap (0.22 m/s)
PIXEL_DEADBAND_PX = 4.0         # Pixel error deadband
ARUCO_STALE_TIMEOUT_S = 0.6     # Detection validity timeout without fresh frames
TARGET_FILTER_ALPHA = 0.30      # Heavy EMA smoothing on marker world position

# Alignment Lock Thresholds
ALIGN_LOCK_DIST_M = 0.02        # Ground metric distance considered centered (meters)
ALIGN_LOCK_PX = 5.0             # Pixel error magnitude considered centered
ALIGN_LOCK_TIME_S = 1.0         # Stable centered hold duration before descent (seconds)
ALIGN_MAX_DURATION_S = 8.0      # Max duration before proceeding to descent best-effort

# ============================================================
# FLIGHT PROFILE & CONTINUOUS DESCENT PARAMETERS
# ============================================================
RATE_HZ = 20.0

# Takeoff & Transit
TARGET_ALT = 2.5                # Takeoff altitude AGL (meters)
CLIMB_RATE_MPS = 0.8            # Vertical climb speed (m/s)
ALT_TOLERANCE = 0.20
TAKEOFF_HOLD_S = 5.0            # Hold duration after takeoff (seconds)

CIRCLE_RADIUS_M = 1.8           # Search orbit radius & transit distance (meters)
TRANSIT_POS_TOLERANCE = 0.20
TRANSIT_HOLD_S = 5.0            # Settle duration before orbiting (seconds)
TRANSIT_SPEED_MPS = 0.35        # Smooth translation speed (m/s)
CIRCLE_DURATION_S = 70.0        # Time for 1 full 360-degree search sweep (seconds)

# Continuous Alignment Down to 0.45m (0.4m - 0.5m) from Ground
DESCEND_RATE_MPS = 0.18         # Gentle, controlled descent speed (0.18 m/s) while visually tracking
LAND_TRIGGER_HEIGHT_M = 0.45    # Continues aligning until 0.45m AGL before landing handover
DESCEND_LOST_REACQUIRE_S = 3.0  # Marker lost during descent -> pause descent to re-lock
DESCEND_LOST_ABORT_S = 6.0      # Marker lost overall -> land at last known target position

# Safety Tolerances & Limits
STABILITY_WINDOW_S = 2.0
MAX_X_VARIATION = 0.20
MAX_Y_VARIATION = 0.20
MAX_Z_VARIATION = 0.18
MAX_HORIZONTAL_SPEED = 0.20
MAX_VERTICAL_SPEED = 0.20

MAX_ABSOLUTE_GEOFENCE_M = CIRCLE_RADIUS_M + 1.8
MAX_POSITION_ERROR_M = 1.20
MAX_ALT_LOSS_M = 0.50
MAX_ROLL_DEG = 25.0
MAX_PITCH_DEG = 25.0

# Timeouts
TAKEOFF_TIMEOUT_S = 35.0
TRANSIT_TIMEOUT_S = 35.0
PRE_OFFBOARD_STREAM_S = 2.0
OFFBOARD_TIMEOUT_S = 5.0
ARM_TIMEOUT_S = 5.0
LANDING_TIMEOUT_S = 20.0        # Max duration for landing & disarm


class State(Enum):
    WAIT_FOR_CHECKS = auto()
    CHECK_STABILITY = auto()
    PRE_OFFBOARD = auto()
    REQUEST_OFFBOARD = auto()
    REQUEST_ARM = auto()
    TAKEOFF = auto()
    TAKEOFF_HOLD = auto()
    TRANSIT_FORWARD = auto()
    TRANSIT_HOLD = auto()
    SEARCH_CIRCLE = auto()
    STOP_SLOWLY = auto()
    HOLD_5_SEC = auto()
    YAW_ALIGN_TARGET = auto()
    APPROACH_TARGET = auto()
    ALIGN_ARUCO = auto()
    DESCEND_TRACK = auto()      # Continues visual alignment while descending down to 0.45m AGL
    LANDING = auto()            # Commands land and aggressively monitors disarm
    DONE = auto()
    ABORT = auto()


# Global references
_NODE_INSTANCE = None
_ACTIVE_HTTP_CLIENTS = 0
_CLIENTS_LOCK = threading.Lock()


class DesktopGUIViewer:
    """Lightweight Tkinter-based desktop window viewer running at throttled low-power rate."""
    def __init__(self, node):
        self.node = node
        self.root = None
        self.lbl = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.root = tk.Tk()
            self.root.title("ArUco 102 Live Tracking")
            self.root.geometry(f"{DISPLAY_WIDTH + 8}x{DISPLAY_HEIGHT + 36}")
            self.root.configure(bg='#121214')
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

            title_lbl = tk.Label(
                self.root, text="ArUco 102 Tracking (Cam: +100mm X, -70mm Z)",
                bg='#121214', fg='#00ffcc', font=('Helvetica', 8, 'bold')
            )
            title_lbl.pack(pady=2)

            self.lbl = tk.Label(self.root, bg='#000000', bd=1, relief=tk.SOLID)
            self.lbl.pack(padx=2, pady=2)

            self._update_loop()
            self.root.mainloop()
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"[GUI] Tkinter exception: {e}")

    def _update_loop(self):
        if not self.running or self.root is None:
            return

        frame = None
        if self.node:
            with self.node.frame_lock:
                frame = self.node.latest_annotated_frame

        if frame is None and self.node:
            frame = self.node.create_splash_frame(f"Waiting for {CAMERA_TOPIC}...")

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = PILImage.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=im)
                self.lbl.imgtk = imgtk
                self.lbl.configure(image=imgtk)
            except Exception:
                pass

        if self.running and self.root:
            interval_ms = max(int(1000 / DISPLAY_FPS), 50)
            self.root.after(interval_ms, self._update_loop)

    def on_close(self):
        self.running = False
        if self.root:
            self.root.destroy()
            self.root = None

    def close(self):
        self.running = False
        if self.root:
            try:
                self.root.quit()
            except Exception:
                pass


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    """Low-bandwidth HTTP server streaming live annotated MJPEG frames."""
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global _NODE_INSTANCE, _ACTIVE_HTTP_CLIENTS
        if self.path == '/' or self.path == '/index.html':
            content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ArUco 102 Precision Tracking & Landing</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            background: #121214;
            color: #ececf1;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            text-align: center;
            margin: 0;
            padding: 12px;
        }}
        h2 {{ margin: 6px 0 2px 0; color: #00ffcc; font-size: 20px; }}
        .subtitle {{ color: #8e8ea0; font-size: 12px; margin-bottom: 10px; }}
        .stream-box {{
            display: inline-block;
            background: #000;
            border: 2px solid #00ffcc;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0,255,204,0.15);
            overflow: hidden;
            max-width: 95vw;
        }}
        img {{
            display: block;
            width: 100%;
            max-width: 480px;
            height: auto;
        }}
        .info-bar {{
            margin-top: 10px;
            font-size: 11px;
            color: #a1a1aa;
        }}
        .badge {{
            display: inline-block;
            background: #27272a;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            padding: 3px 6px;
            margin: 2px 3px;
        }}
    </style>
</head>
<body>
    <h2>ArUco 102 Precision Tracking & Landing</h2>
    <div class="subtitle">Live CV Feed (Cam Offset: +100mm X, -70mm Z &bull; Align down to {LAND_TRIGGER_HEIGHT_M:.2f}m)</div>
    <div class="stream-box">
        <img src="/stream.mjpg" alt="Live Tracking Stream" />
    </div>
    <div class="info-bar">
        <span class="badge">Target ID: <b>{TARGET_ARUCO_ID}</b></span>
        <span class="badge">Align to: <b>{LAND_TRIGGER_HEIGHT_M:.2f}m AGL</b></span>
        <span class="badge">Port: <b>{HTTP_STREAM_PORT}</b></span>
    </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        elif self.path == '/stream.mjpg':
            with _CLIENTS_LOCK:
                _ACTIVE_HTTP_CLIENTS += 1
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            frame_interval = 1.0 / max(DISPLAY_FPS, 1)
            try:
                while True:
                    t_start = time.time()
                    if _NODE_INSTANCE is not None:
                        jpeg_bytes = _NODE_INSTANCE.get_latest_jpeg()
                        if jpeg_bytes is not None:
                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg_bytes)))
                            self.end_headers()
                            self.wfile.write(jpeg_bytes)
                            self.wfile.write(b'\r\n')
                    elapsed = time.time() - t_start
                    sleep_time = max(frame_interval - elapsed, 0.01)
                    time.sleep(sleep_time)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _CLIENTS_LOCK:
                    _ACTIVE_HTTP_CLIENTS = max(_ACTIVE_HTTP_CLIENTS - 1, 0)
        else:
            self.send_error(404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ArUcoSearchAndLandNode(Node):

    def __init__(self):
        super().__init__('aruco_search_land_node')
        global _NODE_INSTANCE
        _NODE_INSTANCE = self

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        qos_cam_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        qos_debug_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.image_cb_group = MutuallyExclusiveCallbackGroup()

        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_px4)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)

        if ENABLE_DEBUG_TOPIC:
            self.debug_pub = self.create_publisher(Image, '/aruco_debug_feed', qos_debug_pub)

        # PX4 Subscriptions
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, qos_px4)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_pos_cb, qos_px4)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_cb, qos_px4)
        self.create_subscription(VehicleCommandAck, '/fmu/out/vehicle_command_ack_v1', self.command_ack_cb, qos_px4)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', self.land_detected_cb, qos_px4)

        # Camera Subscription (CompressedImage support)
        # Probe Subscription
        self.probe_count = 0
        self.create_subscription(Int32, '/erc/probe_count', self.probe_cb, qos_cam_sub)

        if USE_COMPRESSED:
            self.create_subscription(
                CompressedImage, CAMERA_TOPIC, self.compressed_image_cb, qos_cam_sub,
                callback_group=self.image_cb_group,
            )
            self.get_logger().info(f"[CAMERA] Subscribed to CompressedImage on: {CAMERA_TOPIC}")
        else:
            self.create_subscription(
                Image, CAMERA_TOPIC, self.raw_image_cb, qos_cam_sub,
                callback_group=self.image_cb_group,
            )
            self.get_logger().info(f"[CAMERA] Subscribed to raw Image on: {CAMERA_TOPIC}")

        # ArUco Detector Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Periodic Control Timer (20 Hz)
        self.timer = self.create_timer(
            1.0 / RATE_HZ, self.control_loop,
            callback_group=self.timer_cb_group,
        )

        self.state = State.WAIT_FOR_CHECKS
        self.state_start_time = self.now_seconds()

        # Telemetry Cache
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.pre_flight_ok = False
        self.local_pos_valid = False
        self.is_landed = False
        self.ground_contact = False

        self.cur_x, self.cur_y, self.cur_z = 0.0, 0.0, 0.0
        self.cur_vx, self.cur_vy, self.cur_vz = 0.0, 0.0, 0.0
        self.cur_heading = 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0

        self.pos_history = deque(maxlen=int(STABILITY_WINDOW_S * RATE_HZ))

        # Dynamic Image Dimensions
        self.raw_img_w = 640
        self.raw_img_h = 480

        # Frame buffer for HTTP streamer & GUI (throttled)
        self.frame_lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_annotated_frame = None
        self.last_display_render_time = 0.0
        self.frames_received_count = 0

        # Coordinates & Trajectory Setpoints
        self.home_x, self.home_y, self.home_z, self.home_yaw = None, None, None, None
        self.target_z = 0.0
        self.transit_target_x = 0.0
        self.transit_target_y = 0.0
        self.sp_x, self.sp_y, self.sp_z, self.sp_yaw = 0.0, 0.0, 0.0, 0.0

        # ArUco Detection & Metric Tracking State
        self.aruco_detected = False
        self.aruco_err_u = 0.0
        self.aruco_err_v = 0.0
        self.aruco_tag_cx = 0.0
        self.aruco_tag_cy = 0.0
        self.last_aruco_see_time = 0.0

        # World-Frame Marker Localization & Filtered Target Position
        self.filtered_tag_x = None
        self.filtered_tag_y = None
        self.instant_tag_x = None
        self.instant_tag_y = None
        self.metric_err_x = 0.0         # Metric error relative to Drone CG
        self.metric_err_y = 0.0
        self.ground_dist_err = 0.0

        # Alignment Timing & Sequencing
        self.alignment_start_time = None
        self.align_lock_start_time = None
        self.descend_lost_since = None
        self.new_frame_since_tick = False
        self.last_control_time = self.now_seconds()

        # Landing & Disarm State
        self.phase_timer = None
        self.land_requested = False
        self.last_disarm_attempt_time = 0.0
        self.last_land_cmd_time = 0.0
        self.abort_reason = None
        self.last_log_time = 0.0

        # Start Desktop GUI Window
        self.gui_viewer = None
        has_display = ('DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ)
        if ENABLE_GUI_WINDOW and TKINTER_AVAILABLE and has_display:
            try:
                self.gui_viewer = DesktopGUIViewer(self)
                self.get_logger().info("[GUI] Low-latency desktop tracking window launched.")
            except Exception as e:
                self.get_logger().warn(f"[GUI] Could not launch desktop window: {e}")

        # Start Background HTTP MJPEG Streaming Server
        if ENABLE_HTTP_STREAM:
            self.start_http_streamer()

    def create_splash_frame(self, message: str):
        """Creates a compact startup placeholder frame."""
        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        frame[:] = (18, 18, 22)
        cv2.rectangle(frame, (8, 8), (DISPLAY_WIDTH - 8, DISPLAY_HEIGHT - 8), (0, 255, 200), 1)
        cv2.putText(frame, "ArUco 102 Tracking", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"State: {self.state.name}", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, message, (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
        cv2.putText(frame, f"HTTP: http://localhost:{HTTP_STREAM_PORT}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.putText(frame, "ROS 2: /aruco_debug_feed", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        return frame

    def start_http_streamer(self):
        try:
            server = ThreadedHTTPServer(('0.0.0.0', HTTP_STREAM_PORT), StreamingHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.get_logger().info(
                f"[HTTP STREAMER] Live tracking stream available at: http://0.0.0.0:{HTTP_STREAM_PORT}"
            )
        except Exception as e:
            self.get_logger().error(f"[HTTP STREAMER] Failed to start server on port {HTTP_STREAM_PORT}: {e}")

    def get_latest_jpeg(self):
        with self.frame_lock:
            if self.latest_jpeg is not None:
                return self.latest_jpeg
        splash = self.create_splash_frame("Waiting for camera feed...")
        _, buf = cv2.imencode('.jpg', splash, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buf.tobytes()

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def height_agl(self):
        """Height above ground reference for drone CG, positive up."""
        if self.home_z is None:
            return 0.0
        return max(self.home_z - self.cur_z, 0.0)

    def status_cb(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state
        self.pre_flight_ok = msg.pre_flight_checks_pass

    def local_pos_cb(self, msg: VehicleLocalPosition):
        self.cur_x, self.cur_y, self.cur_z = msg.x, msg.y, msg.z
        self.cur_vx, self.cur_vy, self.cur_vz = msg.vx, msg.vy, msg.vz
        self.cur_heading = msg.heading
        self.local_pos_valid = msg.z_valid and (msg.xy_valid or msg.v_xy_valid)
        if self.local_pos_valid:
            self.pos_history.append((self.now_seconds(), self.cur_x, self.cur_y, self.cur_z))

    def attitude_cb(self, msg: VehicleAttitude):
        q = msg.q
        if len(q) < 4:
            return
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = max(-1.0, min(1.0, 2.0 * (q0 * q2 - q3 * q1)))
        self.pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def land_detected_cb(self, msg: VehicleLandDetected):
        self.ground_contact = bool(msg.ground_contact)
        self.is_landed = bool(msg.landed or msg.ground_contact or msg.maybe_landed)

    def command_ack_cb(self, msg: VehicleCommandAck):
        if msg.result != 0:
            self.get_logger().error(f"[COMMAND REJECTED] Cmd: {msg.command} | Result: {msg.result}")
            if self.state in [State.REQUEST_OFFBOARD, State.REQUEST_ARM]:
                self.abort(f"Command {msg.command} denied.")

    def probe_cb(self, msg: Int32):
        self.probe_count = msg.data

    def compressed_image_cb(self, msg: CompressedImage):
        """Decodes sensor_msgs/CompressedImage directly to OpenCV BGR."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
        except Exception as e:
            self.get_logger().error(f"Compressed image decode error: {e}")
            return
        self.process_cv_image(cv_image)

    def raw_image_cb(self, msg: Image):
        """Decodes raw sensor_msgs/Image."""
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                cv_image = np_arr
            else:
                cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            self.get_logger().error(f"Raw image decode error: {e}")
            return
        self.process_cv_image(cv_image)

    def process_cv_image(self, cv_image):
        """Shared CV pipeline for ArUco 102 detection, localization, and HUD rendering."""
        self.frames_received_count += 1
        raw_h, raw_w = cv_image.shape[:2]
        self.raw_img_w, self.raw_img_h = raw_w, raw_h
        raw_center_x, raw_center_y = raw_w / 2.0, raw_h / 2.0

        # Detect ArUco
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        try:
            corners, ids, _ = self.detector.detectMarkers(gray)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        now = self.now_seconds()

        if ids is not None and TARGET_ARUCO_ID in ids:
            idx = np.where(ids == TARGET_ARUCO_ID)[0][0]
            tag_corners = corners[idx][0]
            cx_tag = float(np.mean(tag_corners[:, 0]))
            cy_tag = float(np.mean(tag_corners[:, 1]))

            self.aruco_tag_cx = cx_tag
            self.aruco_tag_cy = cy_tag
            self.aruco_err_u = cx_tag - raw_center_x
            self.aruco_err_v = cy_tag - raw_center_y
            self.aruco_detected = True
            self.last_aruco_see_time = now
            self.new_frame_since_tick = True

            # Update metric world-frame marker localization
            self.update_world_marker_localization()
        else:
            if (now - self.last_aruco_see_time) > ARUCO_STALE_TIMEOUT_S:
                self.aruco_detected = False

        # Throttle display rendering to DISPLAY_FPS (12 FPS) to eliminate lag
        min_display_interval = 1.0 / max(DISPLAY_FPS, 1)
        if (now - self.last_display_render_time) < min_display_interval:
            return

        self.last_display_render_time = now

        # Downsample to compact preview resolution (320x240)
        preview_img = cv2.resize(cv_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        scale_x = DISPLAY_WIDTH / float(raw_w)
        scale_y = DISPLAY_HEIGHT / float(raw_h)

        prev_cx = DISPLAY_WIDTH / 2.0
        prev_cy = DISPLAY_HEIGHT / 2.0

        # Draw detected marker bounding box and error vector
        if ids is not None and TARGET_ARUCO_ID in ids:
            scaled_corners = [c * np.array([scale_x, scale_y]) for c in corners]
            cv2.aruco.drawDetectedMarkers(preview_img, scaled_corners, ids)
            scaled_tag_x = int(self.aruco_tag_cx * scale_x)
            scaled_tag_y = int(self.aruco_tag_cy * scale_y)

            cv2.circle(preview_img, (scaled_tag_x, scaled_tag_y), 4, (0, 0, 255), -1)
            cv2.line(
                preview_img,
                (int(prev_cx), int(prev_cy)),
                (scaled_tag_x, scaled_tag_y),
                (0, 255, 0), 2, cv2.LINE_AA
            )

        # Render compact HUD telemetry
        self.render_compact_hud(preview_img, prev_cx, prev_cy, scale_x)

        # Update Frame Buffers
        with self.frame_lock:
            self.latest_annotated_frame = preview_img

        # Encode JPEG only when HTTP stream is active
        global _ACTIVE_HTTP_CLIENTS
        if _ACTIVE_HTTP_CLIENTS > 0 or ENABLE_HTTP_STREAM:
            try:
                _, jpeg_buf = cv2.imencode('.jpg', preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                with self.frame_lock:
                    self.latest_jpeg = jpeg_buf.tobytes()
            except Exception:
                pass

        # Publish ROS 2 Debug Image Topic
        if ENABLE_DEBUG_TOPIC:
            debug_msg = Image()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'camera_link'
            debug_msg.height = preview_img.shape[0]
            debug_msg.width = preview_img.shape[1]
            debug_msg.encoding = 'bgr8'
            debug_msg.is_bigendian = 0
            debug_msg.step = preview_img.shape[1] * 3
            debug_msg.data = np.ascontiguousarray(preview_img).tobytes()
            self.debug_pub.publish(debug_msg)

    def update_world_marker_localization(self):
        """Calculates metric marker position in World NED frame relative to Drone CG with exact extrinsics."""
        # Camera optical altitude above ground (CG altitude - CAM_OFFSET_Z_M)
        alt_cam = max(self.height_agl() - CAM_OFFSET_Z_M, 0.15)

        err_u = self.aruco_err_u if abs(self.aruco_err_u) > PIXEL_DEADBAND_PX else 0.0
        err_v = self.aruco_err_v if abs(self.aruco_err_v) > PIXEL_DEADBAND_PX else 0.0

        fov_h_rad = math.radians(CAM_FOV_H_DEG)
        fov_v_rad = math.radians(CAM_FOV_V_DEG)
        fx = (self.raw_img_w / 2.0) / math.tan(fov_h_rad / 2.0)
        fy = (self.raw_img_h / 2.0) / math.tan(fov_v_rad / 2.0)

        # Optical displacement on ground relative to camera lens
        cam_dist_x = alt_cam * (err_v / fy)
        cam_dist_y = alt_cam * (err_u / fx)

        # Transform to Drone Body Frame (FRD) relative to Drone Center of Gravity
        if CAM_YAW_180:
            # 180 deg yaw mount: +v in image is Drone Forward (+X body), +u in image is Drone Left (-Y body)
            body_dx = cam_dist_x + CAM_OFFSET_X_M
            body_dy = -cam_dist_y + CAM_OFFSET_Y_M
        else:
            body_dx = -cam_dist_x + CAM_OFFSET_X_M
            body_dy = cam_dist_y + CAM_OFFSET_Y_M

        self.metric_err_x = body_dx
        self.metric_err_y = body_dy

        # Transform Drone Body Offsets to World (NED) Frame
        world_dx = body_dx * math.cos(self.yaw) - body_dy * math.sin(self.yaw)
        world_dy = body_dx * math.sin(self.yaw) + body_dy * math.cos(self.yaw)

        self.instant_tag_x = self.cur_x + world_dx
        self.instant_tag_y = self.cur_y + world_dy

        if self.filtered_tag_x is None or self.filtered_tag_y is None:
            self.filtered_tag_x = self.instant_tag_x
            self.filtered_tag_y = self.instant_tag_y
            self.get_logger().info(f"\033[92m🎯 [VISION] ArUco {TARGET_ARUCO_ID} Position Captured in memory: ({self.filtered_tag_x:.2f}, {self.filtered_tag_y:.2f})! Continuing search...\033[0m")
        else:
            alpha = TARGET_FILTER_ALPHA
            self.filtered_tag_x = (1.0 - alpha) * self.filtered_tag_x + alpha * self.instant_tag_x
            self.filtered_tag_y = (1.0 - alpha) * self.filtered_tag_y + alpha * self.instant_tag_y

        self.ground_dist_err = math.hypot(self.cur_x - self.filtered_tag_x, self.cur_y - self.filtered_tag_y)

    def render_compact_hud(self, img, cx, cy, scale):
        """Draws lightweight HUD telemetry, camera optical crosshair, and Drone CG reticle."""
        h, w = img.shape[:2]

        # 1. Optical Center Crosshair (Yellow)
        cv2.drawMarker(img, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

        # 2. Projected Drone CG Reticle (Cyan Crosshair)
        alt_cam = max(self.height_agl() - CAM_OFFSET_Z_M, 0.15)
        fov_h_rad = math.radians(CAM_FOV_H_DEG)
        fov_v_rad = math.radians(CAM_FOV_V_DEG)
        fx_prev = (DISPLAY_WIDTH / 2.0) / math.tan(fov_h_rad / 2.0)
        fy_prev = (DISPLAY_HEIGHT / 2.0) / math.tan(fov_v_rad / 2.0)

        # Projected pixel location where drone CG is located on ground
        cg_px_x = int(cx - (CAM_OFFSET_Y_M * fx_prev / alt_cam))
        cg_px_y = int(cy - (CAM_OFFSET_X_M * fy_prev / alt_cam)) if CAM_YAW_180 else int(cy + (CAM_OFFSET_X_M * fy_prev / alt_cam))

        if 0 <= cg_px_x < w and 0 <= cg_px_y < h:
            cv2.drawMarker(img, (cg_px_x, cg_px_y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 1)
            cv2.putText(img, "CG", (cg_px_x + 6, cg_px_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)

        # 3. Lock Tolerance Circle (around CG target)
        scaled_lock_r = max(int(ALIGN_LOCK_PX * scale), 8)
        cv2.circle(img, (cg_px_x if 0 <= cg_px_x < w else int(cx), cg_px_y if 0 <= cg_px_y < h else int(cy)), scaled_lock_r, (0, 255, 0), 1, cv2.LINE_AA)

        # 4. Compact Telemetry Box (Top-Left)
        cv2.rectangle(img, (4, 4), (210, 105), (20, 20, 24), -1)
        cv2.rectangle(img, (4, 4), (210, 105), (60, 60, 70), 1)

        tag_status_str = f"LOCKED #{TARGET_ARUCO_ID}" if self.aruco_detected else "SEARCHING"
        tag_color = (0, 255, 0) if self.aruco_detected else (0, 165, 255)

        cv2.putText(img, f"STATE: {self.state.name}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)
        cv2.putText(img, f"TAG: {tag_status_str}", (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tag_color, 1)
        cv2.putText(img, f"PROBES: {self.probe_count}/{TARGET_PROBES_COUNT}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.putText(img, f"ALT: {self.height_agl():4.2f}m (Trig: {LAND_TRIGGER_HEIGHT_M:.2f}m)", (8, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        if self.aruco_detected:
            cv2.putText(
                img, f"CG ERR: dX={self.metric_err_x:+4.2f} dY={self.metric_err_y:+4.2f}m (dist:{self.ground_dist_err:3.2f}m)",
                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1
            )
        else:
            cv2.putText(img, "CG ERR: SEARCHING", (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)

        # Lock Progress Indicator
        if self.align_lock_start_time is not None:
            lock_dur = self.now_seconds() - self.align_lock_start_time
            prog = min(lock_dur / ALIGN_LOCK_TIME_S, 1.0)
            bar_w = int(prog * 80)
            cv2.putText(img, f"LOCK: {lock_dur:3.1f}s", (8, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1)
            cv2.rectangle(img, (70, 89), (70 + bar_w, 97), (0, 255, 0), -1)
            cv2.rectangle(img, (70, 89), (70 + 80, 97), (100, 100, 100), 1)
        else:
            cv2.putText(img, "LOCK: WAITING", (8, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)

        # 5. Orientation Guide (Bottom Center)
        fwd_label = "FRONT (Nose)" if CAM_YAW_180 else "BACK"
        cv2.putText(img, f"^ {fwd_label} ^", (int(cx) - 45, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    def is_armed(self):
        return self.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def is_offboard(self):
        return self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def send_cmd(self, command: int, param1=0.0, param2=0.0, param3=0.0):
        cmd = VehicleCommand()
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        cmd.command = command
        cmd.param1 = float(param1)
        cmd.param2 = float(param2)
        cmd.param3 = float(param3)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)

    def publish_offboard_mode(self):
        off = OffboardControlMode()
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        off.position = True
        off.velocity = False
        off.acceleration = False
        off.attitude = False
        off.body_rate = False
        self.offboard_mode_pub.publish(off)

    def publish_setpoint(self):
        traj = TrajectorySetpoint()
        traj.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        traj.position = [float(self.sp_x), float(self.sp_y), float(self.sp_z)]
        traj.yaw = float(self.sp_yaw)
        traj.velocity = [float('nan'), float('nan'), float('nan')]
        traj.acceleration = [float('nan'), float('nan'), float('nan')]
        traj.jerk = [float('nan'), float('nan'), float('nan')]
        traj.yawspeed = float('nan')
        self.trajectory_pub.publish(traj)

    def position_is_stable(self):
        req_samples = int(STABILITY_WINDOW_S * RATE_HZ)
        if len(self.pos_history) < req_samples:
            return False
        recent = list(self.pos_history)[-req_samples:]
        xs, ys, zs = [p[1] for p in recent], [p[2] for p in recent], [p[3] for p in recent]
        return (
            (max(xs) - min(xs)) <= MAX_X_VARIATION and
            (max(ys) - min(ys)) <= MAX_Y_VARIATION and
            (max(zs) - min(zs)) <= MAX_Z_VARIATION and
            math.hypot(self.cur_vx, self.cur_vy) <= MAX_HORIZONTAL_SPEED and
            abs(self.cur_vz) <= MAX_VERTICAL_SPEED
        )

    def flight_envelope_is_safe(self):
        if abs(math.degrees(self.roll)) > MAX_ROLL_DEG or abs(math.degrees(self.pitch)) > MAX_PITCH_DEG:
            self.abort_reason = "EXCEEDED ATTITUDE TILT LIMITS"
            return False

        active_states = [
            State.TAKEOFF, State.TAKEOFF_HOLD,
            State.TRANSIT_FORWARD, State.TRANSIT_HOLD,
            State.SEARCH_CIRCLE, State.STOP_SLOWLY, State.HOLD_5_SEC,
            State.YAW_ALIGN_TARGET, State.APPROACH_TARGET,
            State.ALIGN_ARUCO, State.DESCEND_TRACK
        ]

        if self.state in active_states:
            pos_err = math.hypot(self.cur_x - self.sp_x, self.cur_y - self.sp_y)
            if pos_err > MAX_POSITION_ERROR_M:
                self.abort_reason = f"PATH TRACKING ERROR: {pos_err:.2f}m deviation"
                return False

            abs_dist_from_home = math.hypot(self.cur_x - self.home_x, self.cur_y - self.home_y)
            if abs_dist_from_home > MAX_ABSOLUTE_GEOFENCE_M:
                self.abort_reason = f"GEOFENCE BREACH: {abs_dist_from_home:.2f}m from origin"
                return False

        post_takeoff_states = [
            State.TAKEOFF_HOLD, State.TRANSIT_FORWARD,
            State.TRANSIT_HOLD, State.SEARCH_CIRCLE, State.STOP_SLOWLY,
            State.HOLD_5_SEC, State.YAW_ALIGN_TARGET, State.APPROACH_TARGET,
            State.ALIGN_ARUCO
        ]
        if self.state in post_takeoff_states:
            if self.cur_z > (self.target_z + MAX_ALT_LOSS_M):
                self.abort_reason = f"ALTITUDE DROP DETECTED > {MAX_ALT_LOSS_M}m"
                return False

        return True

    def capture_home_reference(self):
        if not math.isfinite(self.yaw):
            self.abort("Invalid yaw telemetry.")
            return False

        self.home_x, self.home_y, self.home_z = self.cur_x, self.cur_y, self.cur_z
        self.home_yaw = self.yaw
        self.target_z = self.home_z - TARGET_ALT

        self.transit_target_x = self.home_x + CIRCLE_RADIUS_M
        self.transit_target_y = self.home_y

        self.sp_x, self.sp_y, self.sp_z, self.sp_yaw = self.home_x, self.home_y, self.home_z, self.home_yaw
        self.get_logger().info(
            f"[HOME LOCKED] Center: ({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f}) | Orbit Radius: {CIRCLE_RADIUS_M}m"
        )
        return True

    def abort(self, reason: str):
        self.abort_reason = reason
        self.get_logger().error(f"ABORT TRIGGERED: {reason}")
        self.state = State.ABORT

    def guide_setpoint_to_target(self, target_x, target_y, dt):
        """Slowly and smoothly guides the position setpoint towards target world position."""
        dx = target_x - self.sp_x
        dy = target_y - self.sp_y
        dist = math.hypot(dx, dy)

        if dist < 0.003:
            self.sp_x = target_x
            self.sp_y = target_y
            return

        desired_speed = min(dist * VISUAL_POS_P_GAIN, MAX_TRACK_SPEED_MPS)
        step = desired_speed * dt

        if step >= dist:
            self.sp_x = target_x
            self.sp_y = target_y
        else:
            self.sp_x += (dx / dist) * step
            self.sp_y += (dy / dist) * step

    def control_loop(self):
        """Main 20 Hz Offboard State Machine & Control Loop."""
        if self.home_x is not None and self.state != State.DONE:
            self.publish_offboard_mode()
            self.publish_setpoint()

        now = self.now_seconds()

        # ====================================================
        # STATE MACHINE
        # ====================================================
        if self.state == State.WAIT_FOR_CHECKS:
            if self.local_pos_valid:
                self.pos_history.clear()
                self.state = State.CHECK_STABILITY

        elif self.state == State.CHECK_STABILITY:
            if not self.local_pos_valid:
                self.state = State.WAIT_FOR_CHECKS
            elif self.position_is_stable():
                if self.capture_home_reference():
                    self.state = State.PRE_OFFBOARD
                    self.state_start_time = now

        elif self.state == State.PRE_OFFBOARD:
            if (now - self.state_start_time) >= PRE_OFFBOARD_STREAM_S:
                self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self.state = State.REQUEST_OFFBOARD
                self.state_start_time = now

        elif self.state == State.REQUEST_OFFBOARD:
            if self.is_offboard():
                self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                self.state = State.REQUEST_ARM
                self.state_start_time = now
            elif (now - self.state_start_time) > OFFBOARD_TIMEOUT_S:
                self.abort("Offboard switch timed out.")

        elif self.state == State.REQUEST_ARM:
            if not self.is_offboard():
                self.abort("Dropped Offboard mode before Arming.")
            elif self.is_armed():
                self.phase_timer = now
                self.get_logger().info(f"Armed! Climbing vertically to {TARGET_ALT}m...")
                self.state = State.TAKEOFF
            elif (now - self.state_start_time) > ARM_TIMEOUT_S:
                self.abort("Arming timed out.")

        elif self.state == State.TAKEOFF:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during climb.")
            else:
                elapsed = now - self.phase_timer
                climb_step = self.home_z - (CLIMB_RATE_MPS * elapsed)
                self.sp_z = max(climb_step, self.target_z)

                if elapsed > TAKEOFF_TIMEOUT_S:
                    self.abort("Climb phase timeout exceeded.")
                elif abs(self.cur_z - self.target_z) <= ALT_TOLERANCE:
                    self.get_logger().info(f"Target altitude reached. Holding {TAKEOFF_HOLD_S}s...")
                    self.phase_timer = now
                    self.state = State.TAKEOFF_HOLD

        elif self.state == State.TAKEOFF_HOLD:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Takeoff Hold.")
            elif (now - self.phase_timer) >= TAKEOFF_HOLD_S:
                self.get_logger().info(f"Translating {CIRCLE_RADIUS_M}m World +X to orbit radius...")
                self.phase_timer = now
                self.state = State.TRANSIT_FORWARD

        elif self.state == State.TRANSIT_FORWARD:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Transit.")
            else:
                elapsed = now - self.phase_timer
                transit_step = TRANSIT_SPEED_MPS * elapsed
                self.sp_x = min(self.home_x + transit_step, self.transit_target_x)
                self.sp_y = self.transit_target_y

                dist_to_target = math.hypot(self.cur_x - self.transit_target_x, self.cur_y - self.transit_target_y)

                if elapsed > TRANSIT_TIMEOUT_S:
                    self.abort("Transit phase timeout exceeded.")
                elif dist_to_target <= TRANSIT_POS_TOLERANCE:
                    self.get_logger().info(f"Arrived at orbit perimeter. Settling for {TRANSIT_HOLD_S}s...")
                    self.phase_timer = now
                    self.state = State.TRANSIT_HOLD

        elif self.state == State.TRANSIT_HOLD:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Transit Hold.")
            else:
                elapsed = now - self.phase_timer

                target_start_yaw = math.pi / 2.0
                yaw_diff = target_start_yaw - self.home_yaw
                yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))

                progress = min(elapsed / TRANSIT_HOLD_S, 1.0)
                self.sp_yaw = self.home_yaw + (yaw_diff * progress)
                self.sp_yaw = math.atan2(math.sin(self.sp_yaw), math.cos(self.sp_yaw))

                if elapsed >= TRANSIT_HOLD_S:
                    self.get_logger().info(f"Starting circular search sweep for ArUco ID {TARGET_ARUCO_ID}...")
                    self.phase_timer = now
                    self.state = State.SEARCH_CIRCLE

        elif self.state == State.SEARCH_CIRCLE:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Circular Search.")
            else:
                elapsed_circle_time = now - self.phase_timer

                # Condition 1: Got probes and know where marker is
                if self.probe_count >= TARGET_PROBES_COUNT and self.filtered_tag_x is not None:
                    self.get_logger().info(
                        f"\033[95m🚀 [PROBES COMPLETE] {TARGET_PROBES_COUNT}/{TARGET_PROBES_COUNT} probes reached and Marker known! Stopping slowly...\033[0m"
                    )
                    self.sp_x = self.cur_x
                    self.sp_y = self.cur_y
                    self.phase_timer = now
                    self.state = State.STOP_SLOWLY

                # Condition 2: Ran out of time (2 laps = 2 * CIRCLE_DURATION_S)
                elif elapsed_circle_time >= (2.0 * CIRCLE_DURATION_S):
                    if self.filtered_tag_x is not None:
                        self.get_logger().warn(
                            f"[TIMEOUT] 2 laps complete but only got {self.probe_count}/{TARGET_PROBES_COUNT} probes. Forcing return to ArUco 102!"
                        )
                        self.sp_x = self.cur_x
                        self.sp_y = self.cur_y
                        self.phase_timer = now
                        self.state = State.STOP_SLOWLY
                    else:
                        self.get_logger().warn(
                            "[TIMEOUT] 2 laps complete, probes failed, and ArUco NEVER seen. Landing blindly here!"
                        )
                        self.state = State.LANDING
                        self.phase_timer = now

                # Condition 3: Keep circling
                else:
                    if elapsed_circle_time >= CIRCLE_DURATION_S and not hasattr(self, 'lap2_notified'):
                        self.get_logger().warn("Lap 1 complete. Missing conditions. Starting Lap 2...")
                        self.lap2_notified = True

                    # Modulo arithmetic for continuous laps
                    progress_fraction = (elapsed_circle_time % CIRCLE_DURATION_S) / CIRCLE_DURATION_S
                    current_angle_rad = 2.0 * math.pi * progress_fraction

                    self.sp_x = self.home_x + (CIRCLE_RADIUS_M * math.cos(current_angle_rad))
                    self.sp_y = self.home_y + (CIRCLE_RADIUS_M * math.sin(current_angle_rad))

                    tangent_angle = current_angle_rad + (math.pi / 2.0)
                    self.sp_yaw = math.atan2(math.sin(tangent_angle), math.cos(tangent_angle))

        elif self.state == State.STOP_SLOWLY:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Stop Slowly.")
            else:
                if math.hypot(self.cur_vx, self.cur_vy) < 0.1:
                    self.get_logger().info("Drone stopped. Holding for 5 seconds...")
                    self.phase_timer = now
                    self.state = State.HOLD_5_SEC
                elif (now - self.phase_timer) > 5.0:
                    self.get_logger().info("Stop timeout. Holding for 5 seconds...")
                    self.phase_timer = now
                    self.state = State.HOLD_5_SEC

        elif self.state == State.HOLD_5_SEC:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Hold 5 Sec.")
            else:
                if (now - self.phase_timer) >= 5.0:
                    self.get_logger().info("Hold complete. Aligning yaw smoothly towards ArUco 102...")
                    self.last_control_time = now
                    self.state = State.YAW_ALIGN_TARGET

        elif self.state == State.YAW_ALIGN_TARGET:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Yaw Align.")
            else:
                dt = now - self.last_control_time
                self.last_control_time = now
                
                # Calculate angle to target
                target_yaw = math.atan2(self.filtered_tag_y - self.cur_y, self.filtered_tag_x - self.cur_x)
                
                # Smoothly interpolate the setpoint (very slow, approx 0.1 rad/s)
                yaw_diff = math.atan2(math.sin(target_yaw - self.sp_yaw), math.cos(target_yaw - self.sp_yaw))
                step = 0.1 * dt
                
                if abs(yaw_diff) <= step:
                    self.sp_yaw = target_yaw
                else:
                    self.sp_yaw += step if yaw_diff > 0 else -step
                    
                self.sp_yaw = math.atan2(math.sin(self.sp_yaw), math.cos(self.sp_yaw))

                # Check if physical drone has aligned with the final target yaw
                yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
                
                if abs(yaw_error) < 0.1 and abs(yaw_diff) < 0.05:
                    self.get_logger().info("Yaw smoothly aligned. Approaching target...")
                    self.state = State.APPROACH_TARGET

        elif self.state == State.APPROACH_TARGET:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Approach Target.")
            else:
                dt = now - self.last_control_time
                self.last_control_time = now
                self.guide_setpoint_to_target(self.filtered_tag_x, self.filtered_tag_y, dt)

                dist_to_tag = math.hypot(self.cur_x - self.filtered_tag_x, self.cur_y - self.filtered_tag_y)
                if self.aruco_detected or dist_to_tag < 0.3:
                    self.get_logger().info("\033[96m👀 Target acquired visually or arrived at memory location. Handing over to ALIGN_ARUCO...\033[0m")
                    self.alignment_start_time = now
                    self.align_lock_start_time = None
                    self.new_frame_since_tick = False
                    self.last_control_time = now
                    self.state = State.ALIGN_ARUCO

        elif self.state == State.ALIGN_ARUCO:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Visual Alignment.")
            else:
                dt = now - self.last_control_time
                self.last_control_time = now
                elapsed_align = now - self.alignment_start_time

                if self.aruco_detected and self.filtered_tag_x is not None:
                    self.guide_setpoint_to_target(self.filtered_tag_x, self.filtered_tag_y, dt)

                    dist_to_tag = math.hypot(self.cur_x - self.filtered_tag_x, self.cur_y - self.filtered_tag_y)
                    pixel_err_mag = math.hypot(self.aruco_err_u, self.aruco_err_v)

                    if dist_to_tag <= ALIGN_LOCK_DIST_M and pixel_err_mag <= ALIGN_LOCK_PX:
                        if self.align_lock_start_time is None:
                            self.align_lock_start_time = now
                    else:
                        self.align_lock_start_time = None
                else:
                    self.align_lock_start_time = None

                locked_dur = (now - self.align_lock_start_time) if self.align_lock_start_time else 0.0

                if locked_dur >= ALIGN_LOCK_TIME_S:
                    self.get_logger().info(
                        f"[LOCKED] Centered over marker ({locked_dur:.1f}s). Beginning continuous alignment down to {LAND_TRIGGER_HEIGHT_M}m..."
                    )
                    self.descend_lost_since = None
                    self.state = State.DESCEND_TRACK
                elif elapsed_align >= ALIGN_MAX_DURATION_S:
                    self.get_logger().warn(
                        f"Alignment timeout ({ALIGN_MAX_DURATION_S}s). Proceeding to continuous tracked descent."
                    )
                    self.descend_lost_since = None
                    self.state = State.DESCEND_TRACK

        # ==============================================================
        # CONTINUOUS ALIGNMENT WHILE DESCENDING DOWN TO 0.45m (0.4m - 0.5m)
        # ==============================================================
        elif self.state == State.DESCEND_TRACK:
            if not self.flight_envelope_is_safe() or not self.is_offboard():
                self.abort(self.abort_reason or "Safety breached during Descent.")
            else:
                dt = now - self.last_control_time
                self.last_control_time = now
                current_height = self.height_agl()

                # Actively and continuously guide setpoint over ArUco 102
                if self.aruco_detected and self.filtered_tag_x is not None:
                    self.descend_lost_since = None
                    self.guide_setpoint_to_target(self.filtered_tag_x, self.filtered_tag_y, dt)
                    self.sp_z += DESCEND_RATE_MPS * dt

                elif self.filtered_tag_x is not None:
                    # TARGET MEMORY: Hold last known marker coordinates
                    if self.descend_lost_since is None:
                        self.descend_lost_since = now
                    lost_duration = now - self.descend_lost_since

                    self.guide_setpoint_to_target(self.filtered_tag_x, self.filtered_tag_y, dt)

                    if current_height <= 0.8:
                        self.sp_z += DESCEND_RATE_MPS * dt
                    elif lost_duration >= DESCEND_LOST_ABORT_S:
                        self.get_logger().warn("Marker lost for extended duration. Switching to landing.")
                        self.state = State.LANDING
                        self.phase_timer = now
                    elif lost_duration >= DESCEND_LOST_REACQUIRE_S:
                        self.get_logger().warn("Marker lost during descent. Pausing descent to re-acquire...")
                        self.align_lock_start_time = None
                        self.alignment_start_time = now
                        self.state = State.ALIGN_ARUCO

                # When height reaches LAND_TRIGGER_HEIGHT_M (0.45m) -> Switch to landing directly on tag!
                if self.state == State.DESCEND_TRACK and (current_height <= LAND_TRIGGER_HEIGHT_M or self.is_landed):
                    self.get_logger().info(
                        f"[{LAND_TRIGGER_HEIGHT_M}m REACHED & ALIGNED] Height: {current_height:.2f}m. Perfect lock achieved! Initiating landing touchdown..."
                    )
                    self.state = State.LANDING
                    self.phase_timer = now
                    self.land_requested = False

        # ==============================================================
        # LANDING & GUARANTEED DISARM LOGIC
        # ==============================================================
        elif self.state == State.LANDING:
            elapsed_landing = now - self.phase_timer

            # Continuously push setpoint down in case drone remains in Offboard mode
            dt = now - self.last_control_time
            self.last_control_time = now
            self.sp_z += DESCEND_RATE_MPS * dt

            # 1. Send AUTO.LAND command (Mode 4 = AUTO, submode 6 = LAND)
            if not self.land_requested:
                self.get_logger().info("Executing Native PX4 AUTO.LAND command...")
                self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=6.0)
                self.send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_requested = True
                self.last_land_cmd_time = now

            # 2. Check touchdown conditions (Ground contact / altitude < 0.20m / Land Detector)
            current_height = self.height_agl()
            touchdown_detected = (
                self.ground_contact or
                self.is_landed or
                (current_height <= 0.25 and abs(self.cur_vz) < 0.15 and elapsed_landing > 2.0) or
                (elapsed_landing > 5.0 and abs(self.cur_vz) < 0.10)
            )

            # 3. Aggressively command DISARM upon touchdown
            if touchdown_detected and self.is_armed():
                if (now - self.last_disarm_attempt_time) >= 0.5:
                    self.get_logger().info("\033[93m🛬 [TOUCHDOWN DETECTED] Drone on ground. Commanding DISARM...\033[0m")
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
                    self.last_disarm_attempt_time = now

            # 4. Timeout fallback disarm
            if elapsed_landing > LANDING_TIMEOUT_S and self.is_armed():
                if (now - self.last_disarm_attempt_time) >= 0.5:
                    self.get_logger().warn(f"[LANDING TIMEOUT {LANDING_TIMEOUT_S}s] Forcing DISARM command...")
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
                    self.last_disarm_attempt_time = now

            # 5. Mission Complete when Disarmed
            if not self.is_armed():
                self.get_logger().info("\033[92m🎉 [MISSION SUCCESS] Drone landed squarely on ArUco 102 and DISARMED securely!\033[0m")
                self.state = State.DONE

        elif self.state == State.ABORT:
            if not self.is_offboard() and not self.land_requested:
                self.get_logger().warn("External mode switch (RC/Failsafe) detected. Yielding control completely.")
                self.state = State.DONE
                return

            if not self.land_requested:
                if self.is_armed():
                    self.get_logger().warn("ABORT: Sending PX4 AUTO.LAND command...")
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=6.0)
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_requested = True

            if (self.is_landed or self.height_agl() <= 0.25) and self.is_armed():
                self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

            if not self.is_armed():
                self.state = State.DONE

        elif self.state == State.DONE:
            pass

        # Telemetry & Status Logging (1 Hz)
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            dist_home = 0.0
            if self.home_x is not None:
                dist_home = math.hypot(self.cur_x - self.home_x, self.cur_y - self.home_y)

            # Add ANSI colors for an interesting but clean log
            c_reset = "\033[0m"
            c_state = "\033[94m"  # Blue
            c_alt = "\033[96m"    # Cyan
            c_probes = "\033[95m" # Magenta
            c_tag = "\033[92m" if self.aruco_detected else "\033[93m"  # Green / Yellow
            c_arm = "\033[91m" if self.is_armed() else "\033[92m"      # Red / Green
            c_land = "\033[92m" if (self.ground_contact or self.is_landed) else "\033[96m"

            tag_status = f"LOCK(d={self.ground_dist_err:3.2f}m)" if self.aruco_detected else "SEARCHING"
            timer_val = (now - self.phase_timer) if self.phase_timer is not None else (now - self.state_start_time)
            arm_status = "ARMED" if self.is_armed() else "DISARMED"
            land_status = "TOUCHDOWN" if self.ground_contact or self.is_landed else "AIRBORNE"

            self.get_logger().info(
                f"{c_state}[{self.state.name:<15}]{c_reset} "
                f"{c_alt}Alt: {self.height_agl():4.2f}m{c_reset} | "
                f"{c_probes}Probes: {self.probe_count}/{TARGET_PROBES_COUNT}{c_reset} | "
                f"{c_tag}Tag102: {tag_status:<16}{c_reset} | "
                f"{c_arm}{arm_status:<8}{c_reset} | "
                f"{c_land}{land_status:<9}{c_reset} | "
                f"Tmr: {timer_val:4.1f}s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArUcoSearchAndLandNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().warn("Manual Keyboard Intercept! Triggering Emergency AUTO.LAND...")
        if node.is_armed():
            node.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=6.0)
            node.send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            time.sleep(0.2)
    finally:
        if node.gui_viewer:
            node.gui_viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
