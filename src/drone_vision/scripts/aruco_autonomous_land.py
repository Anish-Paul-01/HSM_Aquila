#!/usr/bin/env python3
"""
ERC 2026 Droning Sub-Task — Orbit Probe Scan & Precision Land
================================================================
Mission flow:
  1. WARMING       -> wait for pre-flight checks
  2. TAKEOFF       -> arm, switch offboard, climb to target altitude
  3. MOVE_OUT      -> fly outward radially to the orbit radius
  4. CIRCLE_SCAN   -> fly a circular path, detect ArUco 102 (remember its
                      world position if seen) and detect probes (dedup +
                      confirm, write to CSV with unique non-repeating ID). 
                      Stops as soon as 3 unique probes are confirmed (see 
                      "Option A" note below) OR after max_scan_loops full 
                      revolutions as a hard safety cap.
  5. ALIGN_TO_TARGET -> slowly rotate yaw to face the remembered ArUco 102
                      position (no translation yet)
  6. RETURN_HOME   -> fly straight to the remembered ArUco 102 position
  7. PRECISION_LAND -> descend while visually re-centering over ArUco 102
  8. DISARM        -> command NAV_LAND, then explicitly force disarm
  9. FINISHED      -> shut down node

Scan-exit policy ("Option A")
-------------------------------
CIRCLE_SCAN ends as soon as 3 probes are CONFIRMED, even if ArUco 102 has
not been confirmed yet. This is a deliberate trade-off requested for this
mission: it means alignment/return/landing may proceed using whatever the
current best estimate of the 102 position is -- which could still be the
TAKEOFF-SPOT FALLBACK if 102 was never actually seen during the orbit. A
clear WARNING is logged at the CIRCLE_SCAN -> ALIGN_TO_TARGET transition
if aruco_102_confirmed is still False at that point, so this is visible in
the flight log rather than failing silently. ArUco 102 detection keeps
running (and can still upgrade the estimate) all the way through
ALIGN_TO_TARGET; it is only frozen once RETURN_HOME begins.

Probe dedup / anti-double-counting
-----------------------------------
Instead of trusting the YOLO tracker's internal ID across frames (track IDs
can reset/split if a probe is briefly occluded or leaves frame during the
orbit), every detection is projected to world (x, y) and matched against:
  a) already CONFIRMED probes (self.tracked_probes)   -> if within
     DEDUP_RADIUS, it's the same physical probe, ignored.
  b) pending CANDIDATES (self.probe_candidates)        -> if within
     DEDUP_RADIUS, running-average the position and bump its confirmation
     count. Only once a candidate is seen CONFIRM_THRESHOLD times is it
     promoted to a logged probe (with a permanent, non-repeating
     next_probe_id that only increments at the moment of confirmation).
The distance check is plain Euclidean distance from a coordinate
subtraction: dx = x2-x1, dy = y2-y1, dist = sqrt(dx**2 + dy**2)
(via math.hypot). DEDUP_RADIUS is 1.5 m to absorb realistic frame-to-frame
projection jitter without still under-merging real, separate probes.

Camera projection (downward / nadir-mounted camera)
-------------------------------------------------------
The camera looks straight down at the ground. Both ArUco-102 tracking and
probe detection now share ONE consistent nadir-camera convention:
    image "up"    (smaller v, negative Yc) = drone body FORWARD
    image "right" (larger u,  positive Xc) = drone body RIGHT
This matches the convention the ArUco tvec-based tracking already used
successfully (err_body_x, err_body_y = -cam_y, cam_x). The PREVIOUS probe
ray-cast used a different, forward-facing-camera-style formula, which for
a nadir camera made its z-denominator sit near zero -- causing the
world-position "scale" term to blow up and drift wildly frame to frame.
That's what was making a single physical probe look like several
different probes tens of meters apart. Probe projection is now done as a
simple pinhole ground-plane projection at the known flight altitude
(no ray-cast division), using the exact same up=forward/right=right
mapping as ArUco, plus:
  - a `camera_mount_yaw_offset` (radians, default 0.0) in case your
    camera's physical rotation relative to the drone body isn't exactly
    "image-up = drone-nose" -- tune this single number if projected
    probe/landing positions still look rotated/offset from reality.
  - a plausibility clamp: a probe can never legitimately be farther from
    the drone than the orbit radius plus a margin. Anything beyond that
    is discarded before it ever reaches dedup, as a second line of
    defense against any residual projection glitches.

ArUco ID-misread protection
-----------------------------
DICT_ARUCO_ORIGINAL has weak bit redundancy, so at range/angle/motion-blur
a single frame can misdecode marker 101's bits as ID 102 (or vice versa).
To stop that from ever corrupting the remembered landing position:
  1. The detector is tightened (stricter error-correction, subpixel corner
     refinement) so ambiguous reads are more likely rejected outright
     rather than "corrected" to the wrong nearby ID.
  2. Any "102" reading that lands within ARUCO_LIFTOFF_EXCLUSION_RADIUS of
     the known lift-off spot (where 101 physically is) is discarded --
     102 can never legitimately be there.
  3. Readings are clustered and must repeat ARUCO_CONFIRM_THRESHOLD times
     at roughly the same place before they're trusted enough to move the
     remembered landing position (same style as probe dedup). A single
     stray misread can no longer relocate the target.
"""

import os
import sys
import math
import csv
import warnings
from datetime import datetime
import numpy as np
import cv2

os.environ["QT_LOGGING_RULES"] = "*=false;qt.qpa.fonts.warning=false"
warnings.filterwarnings("ignore")

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleStatus, VehicleLocalPosition
)
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class OrbitProbeScanAndLand(Node):

    def __init__(self):
        super().__init__('orbit_probe_scan_and_land')

        # ── Mission Parameters ───────────────────────────────────────────────
        self.target_aruco_id = 102
        self.liftoff_aruco_id = 101
        self.target_altitude = -2.3
        self.search_radius_a = 1.8
        self.marker_size = 0.15
        self.required_probes = 3
        self.max_scan_loops = 4        # hard cap: stop orbiting after this many full revolutions
                                        # regardless of probe count, so the mission can't loop forever

        # ── Motion Profile Tuning ────────────────────────────────────────────
        self.move_out_speed = 0.015
        self.orbit_speed_theta = 0.008
        self.max_yaw_step = 0.015
        self.tracking_gain = 0.010
        self.yaw_align_tolerance = 0.05        # rad, ~3 deg
        self.arrival_tolerance = 0.25          # m, for RETURN_HOME
        self.radius_arrival_tolerance = 0.15   # m, MOVE_OUT must physically reach this close to radius
        self.return_speed = 0.02               # m per tick (~0.4 m/s at 20Hz) -- ramp, don't jump

        # ── Probe Dedup / Confirmation Tuning ────────────────────────────────
        self.DEDUP_RADIUS = 1.5        # m: detections closer than this = same probe
        self.CONFIRM_THRESHOLD = 3     # frames a candidate must be seen before logging

        # ── ArUco 102 Misread Protection Tuning ──────────────────────────────
        self.ARUCO_CLUSTER_RADIUS = 0.4          # m: cluster readings within this as "same spot"
        self.ARUCO_CONFIRM_THRESHOLD = 5         # consistent reads needed before trusting a NEW spot
        self.ARUCO_LIFTOFF_EXCLUSION_RADIUS = 0.7  # m: reject "102" reads this close to 101's spot
        self._aruco_candidates = []              # [{'x':.., 'y':.., 'count':..}] -- pending, unconfirmed
        self.aruco_102_confirmed = False         # True only once 102 is actually confirmed (not the fallback)

        # ── Camera Mounting / Projection Tuning ──────────────────────────────
        # Downward (nadir) camera. Default assumes image-up = drone-forward,
        # image-right = drone-right (matches the proven ArUco mapping). If
        # projected positions look rotated relative to reality, tune this
        # single offset (radians) rather than touching the projection math.
        self.camera_mount_yaw_offset = 0.0

        # ── State Machine ─────────────────────────────────────────────────────
        self.STATE_WARMING = "WARMING"
        self.STATE_TAKEOFF = "TAKEOFF"
        self.STATE_MOVE_OUT = "MOVE_OUT"
        self.STATE_CIRCLE_SCAN = "CIRCLE_SCAN"
        self.STATE_ALIGN_TO_TARGET = "ALIGN_TO_TARGET"
        self.STATE_RETURN_HOME = "RETURN_HOME"
        self.STATE_PRECISION_LAND = "PRECISION_LAND"
        self.STATE_DISARM = "DISARM"
        self.STATE_FINISHED = "FINISHED"
        self.current_state = self.STATE_WARMING

        # ── CSV & Probe Tracking Setup ───────────────────────────────────────
        self.csv_filename = "probes_location.csv"
        self.tracked_probes = {}       # {probe_id: (x, y, z)}  -- CONFIRMED probes
        self.probe_candidates = []     # [{'x':.., 'y':.., 'count':..}] -- pending
        self.next_probe_id = 1         # only incremented at confirmation -> never repeats
        self.init_csv()

        # ── YOLO Model Setup ────────────────────────────────────────────────
        self.yolo_model_path = '/home/anish1234/drone_ws/src/drone_vision/detection_model/best.pt'
        self.yolo_model = YOLO(self.yolo_model_path)

        # ── Telemetry State ──────────────────────────────────────────────────
        self._arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self._nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self._pre_flight_ok = False

        self._cur_x, self._cur_y, self._cur_z, self._cur_yaw = 0.0, 0.0, 0.0, 0.0
        self._sp_x, self._sp_y, self._sp_z, self._sp_yaw = 0.0, 0.0, 0.0, 0.0

        self.center_x = 0.0
        self.center_y = 0.0
        self.current_radius = 0.0
        self.orbit_theta = 0.0
        self._scan_loops = 0

        # Trusted world estimate of ArUco 102 (see ArUco misread protection
        # in the module docstring). Frozen once RETURN_HOME starts.
        self.remembered_landing_pos = None  # (x_world, y_world)
        # Live-tracked world position of ArUco 101 (lift-off marker). Used
        # as the exclusion zone for 102-misread protection -- refined from
        # actual sightings rather than trusting the takeoff snapshot alone.
        self.known_liftoff_pos = None       # (x_world, y_world), refined on each 101 sighting

        # ── Vision Engine ────────────────────────────────────────────────────
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = self.cx = self.cy = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Tighten decoding so an ambiguous/blurred/distant read is more
        # likely to be REJECTED outright than "error-corrected" to the
        # wrong nearby ID (this is what let 101 get misread as 102).
        self.aruco_params.errorCorrectionRate = 0.3       # default ~0.6; stricter bit tolerance
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.minMarkerPerimeterRate = 0.05    # ignore tiny/far/unreliable detections
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        s = self.marker_size / 2.0
        self.object_points = np.array([
            [-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]
        ], dtype=np.float32)

        # ── Subscriptions & Publishers ───────────────────────────────────────
        self._offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self._cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v4', self._status_cb, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._local_pos_cb, PX4_QOS)

        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self._camera_info_cb, 10)
        self.create_subscription(Image, '/oak/rgb/image_raw', self._image_cb, 10)

        self.window_name = "Drone Mission Control Feed"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)

        self._heartbeat_count = 0
        self._disarm_attempts = 0
        self._dt = 1.0 / 20.0
        self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(
            "Mission Node initialized: Orbit Search -> Probe Logging -> Align -> Return -> Land -> Disarm"
        )

    # ── CSV ────────────────────────────────────────────────────────────────
    def init_csv(self):
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Probe_ID", "Timestamp", "X_Odom_m", "Y_Odom_m", "Z_Odom_m"])
            self.get_logger().info(f"Initialized CSV log file: {self.csv_filename}")

    def log_probe_to_csv(self, probe_id: int, x: float, y: float, z: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([probe_id, timestamp, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
        self.get_logger().info(
            f"Logged Probe #{probe_id} -> X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m"
        )

    # ── ArUco 102 Position Confirmation (misread protection) ────────────────
    def _update_landing_estimate(self, wx: float, wy: float):
        """Feed one ArUco-102 world reading through misread protection.
        See module docstring 'ArUco ID-misread protection' for the full
        rationale. Never blindly trusts a single frame."""

        # Guard 1: 102 (landing target) can never legitimately be where
        # 101 (lift-off) physically is. A reading here means the detector
        # almost certainly misread 101's bits as ID 102 -- discard it.
        # Prefer the LIVE-tracked 101 position (from actual sightings) over
        # the one-time takeoff snapshot, since that's more accurate.
        liftoff_x, liftoff_y = self.known_liftoff_pos if self.known_liftoff_pos else (self.center_x, self.center_y)
        if math.hypot(wx - liftoff_x, wy - liftoff_y) < self.ARUCO_LIFTOFF_EXCLUSION_RADIUS:
            return

        # Guard 2: if we already trust a CONFIRMED position, a nearby
        # reading just refines it (light low-pass blend). A reading far
        # from the confirmed position does NOT get to move it immediately
        # -- it has to prove itself as a repeated, consistent candidate.
        if self.aruco_102_confirmed and self.remembered_landing_pos is not None:
            tx, ty = self.remembered_landing_pos
            if math.hypot(wx - tx, wy - ty) < self.ARUCO_CLUSTER_RADIUS:
                self.remembered_landing_pos = (tx + 0.05 * (wx - tx), ty + 0.05 * (wy - ty))
                return

        # Guard 3: cluster into candidates; only promote after repeated,
        # consistent confirmation at roughly the same spot.
        for cand in self._aruco_candidates:
            if math.hypot(wx - cand['x'], wy - cand['y']) < self.ARUCO_CLUSTER_RADIUS:
                n = cand['count']
                cand['x'] = (cand['x'] * n + wx) / (n + 1)
                cand['y'] = (cand['y'] * n + wy) / (n + 1)
                cand['count'] += 1
                if cand['count'] >= self.ARUCO_CONFIRM_THRESHOLD:
                    self.remembered_landing_pos = (cand['x'], cand['y'])
                    self.aruco_102_confirmed = True
                    self.get_logger().info(
                        f"ArUco 102 position confirmed -> ({cand['x']:.2f}, {cand['y']:.2f})"
                    )
                    self._aruco_candidates.remove(cand)
                return

        self._aruco_candidates.append({'x': wx, 'y': wy, 'count': 1})

    # ── Probe Dedup / Confirmation ───────────────────────────────────────────
    def _register_probe_detection(self, wx: float, wy: float):
        """Feed one detection (world x,y) through dedup+confirmation.
        Logs a new probe to CSV only once it's confirmed CONFIRM_THRESHOLD
        times and isn't within DEDUP_RADIUS of an already-logged probe.

        Distance is plain Euclidean distance from a coordinate subtraction:
        dx = wx - lx, dy = wy - ly, dist = sqrt(dx**2 + dy**2) via
        math.hypot(dx, dy). Anything closer than DEDUP_RADIUS (1.5 m) is
        treated as the same physical probe rather than a new one."""

        # 1) Already confirmed & logged? Same physical probe -> ignore.
        for (lx, ly, _lz) in self.tracked_probes.values():
            dx = wx - lx
            dy = wy - ly
            dist = math.hypot(dx, dy)
            if dist < self.DEDUP_RADIUS:
                return

        # 2) Matches a pending candidate? Update running average + count.
        for cand in self.probe_candidates:
            dx = wx - cand['x']
            dy = wy - cand['y']
            dist = math.hypot(dx, dy)
            if dist < self.DEDUP_RADIUS:
                n = cand['count']
                cand['x'] = (cand['x'] * n + wx) / (n + 1)
                cand['y'] = (cand['y'] * n + wy) / (n + 1)
                cand['count'] += 1

                if cand['count'] >= self.CONFIRM_THRESHOLD:
                    probe_id = self.next_probe_id      # only source of IDs -> never repeats
                    self.next_probe_id += 1
                    self.tracked_probes[probe_id] = (cand['x'], cand['y'], 0.0)
                    self.log_probe_to_csv(probe_id, cand['x'], cand['y'], 0.0)
                    self.probe_candidates.remove(cand)
                return

        # 3) Brand new candidate.
        self.probe_candidates.append({'x': wx, 'y': wy, 'count': 1})

    # ── Telemetry Callbacks ──────────────────────────────────────────────────
    def _status_cb(self, msg: VehicleStatus):
        self._arming_state = msg.arming_state
        self._nav_state = msg.nav_state
        self._pre_flight_ok = msg.pre_flight_checks_pass

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        self._cur_x, self._cur_y, self._cur_z, self._cur_yaw = msg.x, msg.y, msg.z, msg.heading

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info("Camera intrinsics successfully configured.")

    def _pixel_to_world_nadir(self, u: float, v: float, cam_x_metric=None, cam_y_metric=None):
        """Convert an image observation to a world (x, y) point, using the
        single nadir-camera convention shared by both ArUco tracking and
        probe detection (see module docstring 'Camera projection').

        Either pass a metric camera-frame offset directly (cam_x_metric,
        cam_y_metric -- e.g. from solvePnP's tvec, already in meters), OR
        leave those None and pass pixel coordinates (u, v); in that case
        the offset is estimated via a pinhole ground-plane projection
        using the drone's current altitude AGL (flat-ground assumption).

        Returns (world_x, world_y) or None if geometry isn't usable
        (e.g. camera not yet calibrated, or altitude too close to zero).
        """
        if cam_x_metric is None or cam_y_metric is None:
            if self.camera_matrix is None:
                return None
            h = -self._cur_z  # altitude AGL (cur_z is NED: negative = above ground)
            if h <= 0.1:
                return None
            cam_x_metric = (u - self.cx) / self.fx * h
            cam_y_metric = (v - self.cy) / self.fy * h

        # Optional correction if the camera's physical yaw on the frame
        # isn't exactly "image-up = drone-nose". Default 0.0 = no-op.
        if self.camera_mount_yaw_offset != 0.0:
            co, so = math.cos(self.camera_mount_yaw_offset), math.sin(self.camera_mount_yaw_offset)
            rx = cam_x_metric * co - cam_y_metric * so
            ry = cam_x_metric * so + cam_y_metric * co
            cam_x_metric, cam_y_metric = rx, ry

        # image-up = forward, image-right = right (matches proven ArUco mapping)
        err_body_x, err_body_y = -cam_y_metric, cam_x_metric

        cos_y, sin_y = math.cos(self._cur_yaw), math.sin(self._cur_yaw)
        world_x = self._cur_x + (err_body_x * cos_y - err_body_y * sin_y)
        world_y = self._cur_y + (err_body_x * sin_y + err_body_y * cos_y)
        return world_x, world_y

    # ── Vision Pipeline ───────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        if self.camera_matrix is None or self.current_state in [self.STATE_WARMING, self.STATE_FINISHED]:
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding.lower() in ('rgb8', 'rgb'):
            # OAK-D / depthai-ros commonly publishes RGB, but OpenCV (and
            # cv2.imshow) expects BGR -- without this swap, reds and blues
            # get flipped and the feed looks blue-tinted.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotated_frame = frame.copy()

        # ---------------------------------------------------------------
        # 1. ARUCO 102 (landing target) — update running-average estimate
        #    in every state except PRECISION_LAND (where raw feedback is
        #    used directly for fine centering) and RETURN_HOME (frozen).
        # ---------------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            for index, marker_id in enumerate(ids.flatten()):
                if marker_id not in (self.target_aruco_id, self.liftoff_aruco_id):
                    continue

                success, rvec, tvec = cv2.solvePnP(
                    self.object_points, corners[index][0].astype(np.float32),
                    self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not success:
                    continue

                cv2.drawFrameAxes(annotated_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                cam_x, cam_y = tvec[0][0], tvec[1][0]

                world = self._pixel_to_world_nadir(0, 0, cam_x_metric=cam_x, cam_y_metric=cam_y)
                if world is None:
                    continue
                world_x, world_y = world
                err_body_x, err_body_y = -cam_y, cam_x  # kept for precision-land feedback below

                if marker_id == self.liftoff_aruco_id:
                    # Confidently read 101: refine our live knowledge of
                    # where it actually is, for the misread-exclusion guard.
                    if self.known_liftoff_pos is None:
                        self.known_liftoff_pos = (world_x, world_y)
                    else:
                        lx, ly = self.known_liftoff_pos
                        self.known_liftoff_pos = (lx + 0.1 * (world_x - lx), ly + 0.1 * (world_y - ly))
                    continue  # nothing further to do with 101

                # marker_id == self.target_aruco_id (102) from here on
                if self.current_state == self.STATE_PRECISION_LAND:
                    # Fine descent: react to the raw, most-recent reading.
                    self._process_landing_feedback(err_body_x, err_body_y)
                elif self.current_state != self.STATE_RETURN_HOME:
                    # Still searching/orbiting/aligning: run every reading
                    # through misread protection before trusting it.
                    self._update_landing_estimate(world_x, world_y)

        # ---------------------------------------------------------------
        # 2. PROBE DETECTION — only while actively orbiting the circular
        #    scan path, per the required search pattern, and only until
        #    3 unique probes are confirmed (Option A: this alone ends the
        #    scan -- see module docstring "Scan-exit policy").
        # ---------------------------------------------------------------
        if self.current_state == self.STATE_CIRCLE_SCAN and len(self.tracked_probes) < self.required_probes:
            results = self.yolo_model.predict(frame, verbose=False)
            for r_det in results:
                if r_det.boxes is None:
                    continue
                boxes = r_det.boxes.xyxy.cpu().numpy()

                for box in boxes:
                    u = (box[0] + box[2]) / 2.0
                    v = (box[1] + box[3]) / 2.0

                    world = self._pixel_to_world_nadir(u, v)
                    if world is None:
                        continue
                    p_x, p_y = world

                    # Plausibility clamp: a probe can't legitimately be
                    # farther from the drone than the orbit radius + a
                    # margin. Discards any residual bad projection before
                    # it can pollute dedup / get logged as a "new" probe.
                    max_plausible_dist = self.search_radius_a + 1.0
                    if math.hypot(p_x - self._cur_x, p_y - self._cur_y) > max_plausible_dist:
                        continue

                    self._register_probe_detection(p_x, p_y)

                    cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Status overlay
        status_txt = (
            f"State: {self.current_state} | "
            f"Probes: {len(self.tracked_probes)}/{self.required_probes} | "
            f"Candidates: {len(self.probe_candidates)} | "
            f"102: {'CONFIRMED' if self.aruco_102_confirmed else 'searching'}"
        )
        cv2.putText(annotated_frame, status_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        display_img = cv2.resize(annotated_frame, (960, 540))
        cv2.imshow(self.window_name, display_img)
        cv2.waitKey(1)

    def _process_landing_feedback(self, err_x, err_y):
        """Precision centering control over ArUco 102 during final descent."""
        cos_y, sin_y = math.cos(self._cur_yaw), math.sin(self._cur_yaw)
        error_ned_x = err_x * cos_y - err_y * sin_y
        error_ned_y = err_x * sin_y + err_y * cos_y

        target_world_x = self._cur_x + error_ned_x
        target_world_y = self._cur_y + error_ned_y

        self._sp_x += self.tracking_gain * (target_world_x - self._sp_x)
        self._sp_y += self.tracking_gain * (target_world_y - self._sp_y)

    # ── Main Flight Execution Loop ───────────────────────────────────────────
    def _control_loop(self):
        if self.current_state not in [self.STATE_DISARM, self.STATE_FINISHED]:
            self._publish_offboard_mode()

        self._heartbeat_count += 1
        target_yaw = self._sp_yaw

        # ------------------------------------------------------------
        # 1. WARMING
        # ------------------------------------------------------------
        if self.current_state == self.STATE_WARMING:
            if self._heartbeat_count >= 20 and self._pre_flight_ok:
                self.current_state = self.STATE_TAKEOFF
                self.get_logger().info("Warmup complete. Initiating takeoff.")

        # ------------------------------------------------------------
        # 2. TAKEOFF
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_TAKEOFF:
            if not self._is_armed():
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            elif not self._is_offboard():
                self._sp_x, self._sp_y, self._sp_z = self._cur_x, self._cur_y, self.target_altitude
                self._sp_yaw = self._cur_yaw
                self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            else:
                if abs(self._cur_z - self.target_altitude) < 0.2:
                    self.current_state = self.STATE_MOVE_OUT
                    self.center_x = self._cur_x
                    self.center_y = self._cur_y
                    # Default fallback target in case ArUco 102 is never seen
                    self.remembered_landing_pos = (self._cur_x, self._cur_y)
                    self.current_radius = 0.0
                    self.get_logger().info(f"Target altitude reached. Moving out to radius: {self.search_radius_a}m")

        # ------------------------------------------------------------
        # 3. MOVE_OUT
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_MOVE_OUT:
            self._sp_z = self.target_altitude
            if self.current_radius < self.search_radius_a:
                self.current_radius += self.move_out_speed

            self._sp_x = self.center_x + self.current_radius
            self._sp_y = self.center_y
            target_yaw = 0.0

            # Don't start orbiting just because the SETPOINT ramp finished --
            # the position controller lags behind the setpoint, so wait until
            # the drone has actually, physically reached the radius too.
            if self.current_radius >= self.search_radius_a:
                dist_from_center = math.hypot(self._cur_x - self.center_x, self._cur_y - self.center_y)
                if dist_from_center >= (self.search_radius_a - self.radius_arrival_tolerance):
                    self.current_state = self.STATE_CIRCLE_SCAN
                    self.orbit_theta = 0.0
                    self._scan_loops = 0
                    self.get_logger().info("Orbit radius achieved. Beginning circular probe scan...")

        # ------------------------------------------------------------
        # 4. CIRCLE_SCAN — Option A: exits as soon as 3 probes are
        #    confirmed, regardless of whether ArUco 102 has been
        #    confirmed yet (see module docstring "Scan-exit policy").
        #    max_scan_loops is kept as a hard safety cap so the mission
        #    can't orbit forever if probes are never found.
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_CIRCLE_SCAN:
            self._sp_z = self.target_altitude
            self.orbit_theta += self.orbit_speed_theta

            self._sp_x = self.center_x + (self.search_radius_a * math.cos(self.orbit_theta))
            self._sp_y = self.center_y + (self.search_radius_a * math.sin(self.orbit_theta))
            target_yaw = self.orbit_theta + (math.pi / 2.0)

            wrapped_this_tick = False
            if self.orbit_theta >= (2.0 * math.pi):
                self.orbit_theta -= 2.0 * math.pi  # wrap and keep going
                self._scan_loops += 1
                wrapped_this_tick = True
                if not self.aruco_102_confirmed:
                    self.get_logger().warn(
                        f"Completed {self._scan_loops} orbit(s), ArUco 102 still not confirmed -- "
                        f"continuing to search for probes..."
                    )

            probes_done = len(self.tracked_probes) >= self.required_probes
            hit_loop_cap = self._scan_loops >= self.max_scan_loops and wrapped_this_tick

            if probes_done or hit_loop_cap:
                if not self.aruco_102_confirmed:
                    self.get_logger().warn(
                        "Leaving CIRCLE_SCAN without a CONFIRMED ArUco 102 position. "
                        "Aligning/returning to best current estimate (may be the takeoff-spot fallback)."
                    )
                self.get_logger().info(
                    f"Scan complete ({len(self.tracked_probes)}/{self.required_probes} probes). "
                    f"Aligning toward target..."
                )
                self.current_state = self.STATE_ALIGN_TO_TARGET

        # ------------------------------------------------------------
        # 5. ALIGN_TO_TARGET — rotate slowly to face ArUco 102, hold position.
        #    ArUco detection keeps running here, so a late-confirmed 102
        #    can still improve remembered_landing_pos before RETURN_HOME
        #    freezes it.
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_ALIGN_TO_TARGET:
            self._sp_z = self.target_altitude
            self._sp_x = self._cur_x
            self._sp_y = self._cur_y

            target_x, target_y = self.remembered_landing_pos
            target_yaw = math.atan2(target_y - self._cur_y, target_x - self._cur_x)

            yaw_err = (target_yaw - self._cur_yaw + math.pi) % (2.0 * math.pi) - math.pi
            if abs(yaw_err) < self.yaw_align_tolerance:
                self.current_state = self.STATE_RETURN_HOME
                self.get_logger().info("Alignment complete. Returning to ArUco 102 target...")

        # ------------------------------------------------------------
        # 6. RETURN_HOME — fly to the remembered ArUco 102 location.
        #    Setpoint is RAMPED (not jumped) so PX4's position controller
        #    doesn't try to close the whole gap at max accel/vel.
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_RETURN_HOME:
            self._sp_z = self.target_altitude
            target_x, target_y = self.remembered_landing_pos

            dx = target_x - self._sp_x
            dy = target_y - self._sp_y
            dist_remaining = math.hypot(dx, dy)
            if dist_remaining > 1e-6:
                step = min(self.return_speed, dist_remaining)
                self._sp_x += step * dx / dist_remaining
                self._sp_y += step * dy / dist_remaining

            target_yaw = math.atan2(target_y - self._cur_y, target_x - self._cur_x)

            dist_to_target = math.hypot(self._cur_x - target_x, self._cur_y - target_y)
            if dist_to_target < self.arrival_tolerance:
                self.current_state = self.STATE_PRECISION_LAND
                self.get_logger().info("Arrived over ArUco 102 target location. Commencing precision descent.")

        # ------------------------------------------------------------
        # 7. PRECISION_LAND
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_PRECISION_LAND:
            self._sp_z += 0.007  # ~0.14 m/s descent

            if self._cur_z >= -0.05:
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.get_logger().info("Touchdown threshold reached. Commanding NAV_LAND.")
                self.current_state = self.STATE_DISARM
                self._disarm_attempts = 0

        # ------------------------------------------------------------
        # 8. DISARM — explicitly force disarm to guarantee mission end state
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_DISARM:
            if self._is_armed():
                self._disarm_attempts += 1
                # Give PX4's own post-land auto-disarm a moment, then force it.
                if self._disarm_attempts > 40:  # ~2s at 20Hz
                    self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
            else:
                self.get_logger().info("Vehicle disarmed. Mission complete.")
                self.current_state = self.STATE_FINISHED

        # ------------------------------------------------------------
        # 9. FINISHED
        # ------------------------------------------------------------
        elif self.current_state == self.STATE_FINISHED:
            rclpy.shutdown()
            return

        # Smooth Heading Limiter (shared by all active states)
        if self.current_state not in [self.STATE_WARMING, self.STATE_DISARM, self.STATE_FINISHED]:
            yaw_error = target_yaw - self._sp_yaw
            yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
            clipped_step = np.clip(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            self._sp_yaw = (self._sp_yaw + clipped_step + math.pi) % (2.0 * math.pi) - math.pi

        if self.current_state not in [self.STATE_DISARM, self.STATE_FINISHED]:
            self._publish_setpoint()

    # ── PX4 Helper Functions ────────────────────────────────────────────────
    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._ts()
        msg.position = True
        self._offboard_pub.publish(msg)

    def _publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self._ts()
        msg.position = [self._sp_x, self._sp_y, self._sp_z]
        msg.yaw = self._sp_yaw
        self._setpoint_pub.publish(msg)

    def _send_cmd(self, command: int, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp, msg.command, msg.param1, msg.param2 = self._ts(), command, param1, param2
        msg.target_system, msg.target_component = 1, 1
        msg.source_system, msg.source_component = 1, 1
        msg.from_external = True
        self._cmd_pub.publish(msg)

    def _is_armed(self):
        return self._arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _is_offboard(self):
        return self._nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _ts(self):
        return self.get_clock().now().nanoseconds // 1000

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OrbitProbeScanAndLand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()