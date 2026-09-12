#!/usr/bin/env python3
"""
odom_tf_broadcaster.py
══════════════════════════════════════════════════════════════════════════════
PX4 ↔ RViz2 Live Bridge — Hochschule Schmalkalden ERC Drone Project
══════════════════════════════════════════════════════════════════════════════

PURPOSE
───────
PX4 publishes all pose data in NED / FRD conventions (px4_msgs).
RViz2 / ROS2 nav stack expect ENU / FLU conventions (standard msgs).
This node is the single conversion layer between the two worlds.

WHAT IT DOES
────────────
  1. Subscribes to /fmu/out/vehicle_local_position_v1  (NED position + velocity)
  2. Subscribes to /fmu/out/vehicle_attitude            (FRD quaternion)
  3. Converts position:    NED  → ENU
       x_enu =  y_ned
       y_enu =  x_ned
       z_enu = -z_ned
  4. Converts quaternion:  FRD body frame → FLU body frame
       Step A: rotate body FRD → FLU  (+π around X axis)
       Step B: rotate world NED → ENU (+π/2 around Z, then +π around X)
  5. Broadcasts dynamic TF:  odom → base_link          (RViz sees drone move)
  6. Publishes static TF:    map  → odom               (single identity)
  7. Publishes nav_msgs/Odometry on /drone/odom         (for Nav2 / future use)
  8. Publishes geometry_msgs/PoseStamped on /drone/pose (lightweight RViz overlay)

FRAME CONVENTIONS (official PX4 docs reference)
────────────────────────────────────────────────
  PX4 World  : NED  — X North,  Y East,  Z Down
  PX4 Body   : FRD  — X Forward, Y Right, Z Down
  ROS2 World : ENU  — X East,   Y North,  Z Up
  ROS2 Body  : FLU  — X Forward, Y Left,  Z Up

QoS
───
Matches PX4 uXRCE-DDS publisher profile exactly:
  BEST_EFFORT | VOLATILE | KEEP_LAST depth=1
Using RELIABLE here would cause the subscriber to never match and receive nothing.

TOPICS PRODUCED (all in ENU/FLU)
─────────────────────────────────
  /tf                  — dynamic: odom → base_link   @ px4 attitude rate (~50 Hz)
  /tf_static           — static:  map  → odom        (published once on startup)
  /drone/odom          — nav_msgs/Odometry
  /drone/pose          — geometry_msgs/PoseStamped

HOW TO USE
──────────
  # Direct run (for testing):
  ros2 run drone_bringup odom_tf_broadcaster.py

  # Via launch file (already wired in drone_gazebo.launch_px4.py, uncomment):
  #   px4_tf_bridge node

  # RViz2 Fixed Frame must be set to:  odom
  # Add displays: RobotModel, TF, Odometry (/drone/odom)

AUTHOR  : Hochschule Schmalkalden — ERC Drone Team
STACK   : ROS2 Humble | PX4 v1.14+ | micro-ROS XRCE-DDS
"""

import math
import numpy as np

import rclpy
from rclpy.node        import Node
from rclpy.qos         import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
)

# TF2
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# ROS2 standard message types (what RViz/Nav2 understand)
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg      import Odometry

# PX4 message types (what micro-ROS XRCE-DDS delivers)
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

# ──────────────────────────────────────────────────────────────────────────────
# QoS profile — must match PX4 uXRCE-DDS publisher exactly
# ──────────────────────────────────────────────────────────────────────────────
PX4_QOS = QoSProfile(
    reliability = ReliabilityPolicy.BEST_EFFORT,
    durability  = DurabilityPolicy.VOLATILE,
    history     = HistoryPolicy.KEEP_LAST,
    depth       = 1,
)

# ──────────────────────────────────────────────────────────────────────────────
# Frame ID constants
# ──────────────────────────────────────────────────────────────────────────────
FRAME_MAP       = 'map'        # global fixed world frame
FRAME_ODOM      = 'odom'       # local odometry origin (ENU)
FRAME_BASE_LINK = 'base_link'  # drone body root link (FLU, from URDF)


# ══════════════════════════════════════════════════════════════════════════════
# Quaternion helpers — pure Python/NumPy, no scipy dependency
# ══════════════════════════════════════════════════════════════════════════════

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.
    Convention: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (radians) to quaternion [x, y, z, w].
    Rotation order: Z → Y → X (extrinsic), i.e. RPY intrinsic.
    """
    cr = math.cos(roll  * 0.5);  sr = math.sin(roll  * 0.5)
    cp = math.cos(pitch * 0.5);  sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw   * 0.5);  sy = math.sin(yaw   * 0.5)
    return np.array([
        sr*cp*cy - cr*sp*sy,   # x
        cr*sp*cy + sr*cp*sy,   # y
        cr*cp*sy - sr*sp*cy,   # z
        cr*cp*cy + sr*sp*sy,   # w
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-computed rotation constants (computed once at import time)
# ──────────────────────────────────────────────────────────────────────────────

# Step A: FRD → FLU body frame rotation
#   +π (180°) around X axis
#   [x, y, z, w] = [sin(π/2), 0, 0, cos(π/2)] = [1, 0, 0, 0]
Q_FRD_TO_FLU = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Step B: NED → ENU world frame rotation
#   +π/2 around Z then +π around X  (official PX4 px4_ros_com convention)
#   Equivalent single quaternion computed analytically:
#   q_z90  = euler_to_quat(0, 0,    π/2)
#   q_x180 = euler_to_quat(π, 0,    0  )
#   Q_NED_TO_ENU = q_x180 * q_z90   (apply z90 first, then x180)
_q_z90  = euler_to_quat(0.0,        0.0, math.pi / 2.0)
_q_x180 = euler_to_quat(math.pi,    0.0, 0.0)
Q_NED_TO_ENU = quat_normalize(quat_multiply(_q_x180, _q_z90))


def ned_to_enu_position(x_ned: float, y_ned: float, z_ned: float):
    """
    Convert NED position to ENU position.
      ENU_x =  NED_y  (East  ← North)
      ENU_y =  NED_x  (North ← East)   wait — correct mapping:
      ENU_x =  NED_y  (East  = NED East = NED_y)
      ENU_y =  NED_x  (North = NED North = NED_x)
      ENU_z = -NED_z  (Up    = -Down)
    """
    return float(y_ned), float(x_ned), float(-z_ned)


def ned_frd_to_enu_flu_quaternion(q_frd_ned: np.ndarray) -> np.ndarray:
    """
    Convert a PX4 quaternion (body=FRD, world=NED) to
    ROS2 quaternion (body=FLU, world=ENU).

    Two-step rotation following official PX4 frame_transforms.h:
      1. Rotate body frame: FRD → FLU  (Q_FRD_TO_FLU applied on right)
      2. Rotate world frame: NED → ENU (Q_NED_TO_ENU applied on left)

    Result: q_flu_enu = Q_NED_TO_ENU ⊗ q_frd_ned ⊗ Q_FRD_TO_FLU
    """
    q = quat_normalize(q_frd_ned)
    q = quat_multiply(q, Q_FRD_TO_FLU)       # body: FRD → FLU
    q = quat_multiply(Q_NED_TO_ENU, q)        # world: NED → ENU
    q_z180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    q = quat_multiply(q_z180, q)
    return quat_normalize(q)


# ══════════════════════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════════════════════

class OdomTfBroadcaster(Node):
    """
    Converts PX4 pose telemetry (NED/FRD) into ROS2-standard
    TF transforms and Odometry messages (ENU/FLU) for RViz2.

    Subscribes
    ──────────
      /fmu/out/vehicle_local_position_v1  px4_msgs/VehicleLocalPosition
      /fmu/out/vehicle_attitude           px4_msgs/VehicleAttitude

    Publishes
    ─────────
      /tf             dynamic TF:  odom → base_link
      /tf_static      static TF:   map  → odom
      /drone/odom     nav_msgs/Odometry
      /drone/pose     geometry_msgs/PoseStamped
    """

    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        # ── TF broadcasters ───────────────────────────────────────────────────
        self._tf_broadcaster        = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ── Publishers ────────────────────────────────────────────────────────
        self._odom_pub = self.create_publisher(
            Odometry,      '/drone/odom',  10)
        self._pose_pub = self.create_publisher(
            PoseStamped,   '/drone/pose',  10)

        # ── Subscribers (PX4 QoS — BEST_EFFORT / VOLATILE) ───────────────────
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._local_pos_cb,
            PX4_QOS,
        )
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._attitude_cb,
            PX4_QOS,
        )

        # ── Internal state cache ──────────────────────────────────────────────
        # Position (ENU) — initialised to origin
        self._enu_x: float = 0.0
        self._enu_y: float = 0.0
        self._enu_z: float = 0.0

        # Velocity (ENU)
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0
        self._vel_z: float = 0.0

        # Quaternion (ENU/FLU) [x, y, z, w] — identity on startup
        self._q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # Attitude timestamp (nanoseconds, ROS clock)
        self._attitude_stamp = None

        # Flags — only publish once BOTH topics have been received
        self._has_position: bool = False
        self._has_attitude: bool = False

        # ── Publish static map → odom identity transform once ─────────────────
        self._publish_static_map_odom()

        self.get_logger().info(
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
            '  OdomTfBroadcaster started\n'
            '  Waiting for PX4 telemetry on:\n'
            '    /fmu/out/vehicle_local_position_v1\n'
            '    /fmu/out/vehicle_attitude\n'
            '  Publishing to:\n'
            '    /tf          (odom → base_link)\n'
            '    /tf_static   (map  → odom)\n'
            '    /drone/odom  (nav_msgs/Odometry)\n'
            '    /drone/pose  (geometry_msgs/PoseStamped)\n'
            '  RViz2 Fixed Frame → set to: odom\n'
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _local_pos_cb(self, msg: VehicleLocalPosition) -> None:
        """
        Receives NED local position from PX4 EKF.
        Converts to ENU and caches.  Triggers a TF+odom publish if
        attitude is already available (attitude drives the publish rate).
        """
        # Guard: only use valid position estimates
        if not (msg.xy_valid and msg.z_valid):
            self.get_logger().warn(
                'Position not yet valid (xy_valid=%s z_valid=%s) — skipping',
                msg.xy_valid, msg.z_valid,
                throttle_duration_sec=5.0,
            )
            return

        # Convert NED → ENU
        self._enu_x, self._enu_y, self._enu_z = ned_to_enu_position(
            msg.x, msg.y, msg.z
        )

        # Velocity NED → ENU  (same swap, negate Z)
        self._vel_x =  float(msg.vy)   # ENU East  = NED vy
        self._vel_y =  float(msg.vx)   # ENU North = NED vx
        self._vel_z = -float(msg.vz)   # ENU Up    = -NED vz

        self._has_position = True

        # Publish immediately if attitude is ready so position updates
        # are not held back by attitude message timing
        if self._has_attitude:
            self._publish_all()

    def _attitude_cb(self, msg: VehicleAttitude) -> None:
        """
        Receives FRD quaternion from PX4 attitude estimator.
        Converts to ENU/FLU and triggers the main publish cycle.

        PX4 VehicleAttitude.q is [w, x, y, z] (scalar-first).
        We re-order to [x, y, z, w] (scalar-last) for internal use.
        """
        # PX4 quaternion layout: q[0]=w, q[1]=x, q[2]=y, q[3]=z
        q_frd_ned = np.array([
            msg.q[1],   # x
            msg.q[2],   # y
            msg.q[3],   # z
            msg.q[0],   # w  ← scalar moved to last position
        ], dtype=np.float64)

        self._q = ned_frd_to_enu_flu_quaternion(q_frd_ned)
        self._attitude_stamp = self.get_clock().now()
        self._has_attitude   = True

        # Attitude arrives faster (~250 Hz IMU → ~50 Hz EKF output).
        # Drive the publish cycle here so TF matches attitude rate.
        if self._has_position:
            self._publish_all()

    # ──────────────────────────────────────────────────────────────────────────
    # Publishers
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_all(self) -> None:
        """Publish TF, Odometry and PoseStamped in one atomic call."""
        #stamp = self._attitude_stamp or self.get_clock().now()
        stamp = self.get_clock().now()
        ros_stamp = stamp.to_msg()

        self._broadcast_odom_to_base_link(ros_stamp)
        self._publish_odometry(ros_stamp)
        self._publish_pose_stamped(ros_stamp)

    def _broadcast_odom_to_base_link(self, stamp) -> None:
        """
        Broadcast dynamic TF:  odom (ENU) → base_link (FLU).
        This is the transform RViz2 uses to place the RobotModel in 3D space.
        """
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = FRAME_ODOM
        t.child_frame_id  = FRAME_BASE_LINK

        # Position (ENU)
        t.transform.translation.x = self._enu_x
        t.transform.translation.y = self._enu_y
        t.transform.translation.z = self._enu_z

        # Orientation (ENU/FLU quaternion) [x, y, z, w]
        t.transform.rotation.x = float(self._q[0])
        t.transform.rotation.y = float(self._q[1])
        t.transform.rotation.z = float(self._q[2])
        t.transform.rotation.w = float(self._q[3])

        self._tf_broadcaster.sendTransform(t)

    def _publish_static_map_odom(self) -> None:
        """
        Publish a static identity transform: map → odom.

        In simulation the map and odometry frames are identical
        (no drift, no SLAM correction needed).  This static TF
        allows Nav2 and any map-frame displays to work correctly
        without requiring a localization node.
        """
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = FRAME_MAP
        t.child_frame_id  = FRAME_ODOM

        # Identity — map == odom in simulation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x    = 0.0
        t.transform.rotation.y    = 0.0
        t.transform.rotation.z    = 0.0
        t.transform.rotation.w    = 1.0

        self._static_tf_broadcaster.sendTransform(t)
        self.get_logger().info('Static TF published: map → odom (identity)')

    def _publish_odometry(self, stamp) -> None:
        """
        Publish nav_msgs/Odometry on /drone/odom.
        Required by Nav2 and useful for path recording in RViz2.
        """
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = FRAME_ODOM
        msg.child_frame_id  = FRAME_BASE_LINK

        # Pose (ENU/FLU)
        msg.pose.pose.position.x    = self._enu_x
        msg.pose.pose.position.y    = self._enu_y
        msg.pose.pose.position.z    = self._enu_z
        msg.pose.pose.orientation.x = float(self._q[0])
        msg.pose.pose.orientation.y = float(self._q[1])
        msg.pose.pose.orientation.z = float(self._q[2])
        msg.pose.pose.orientation.w = float(self._q[3])

        # Twist — linear velocity in ENU world frame
        # (expressed in child frame = base_link for Nav2 compatibility)
        msg.twist.twist.linear.x = self._vel_x
        msg.twist.twist.linear.y = self._vel_y
        msg.twist.twist.linear.z = self._vel_z

        # Covariance — diagonal, modest uncertainty
        # These are placeholders; tune when connecting real sensors
        pose_cov  = [0.0] * 36
        twist_cov = [0.0] * 36
        for i in (0, 7, 14):           # x, y, z position variance
            pose_cov[i]  = 0.01
        for i in (21, 28, 35):         # roll, pitch, yaw variance
            pose_cov[i]  = 0.001
        for i in (0, 7, 14):           # vx, vy, vz variance
            twist_cov[i] = 0.01
        msg.pose.covariance  = pose_cov
        msg.twist.covariance = twist_cov

        self._odom_pub.publish(msg)

    def _publish_pose_stamped(self, stamp) -> None:
        """
        Publish geometry_msgs/PoseStamped on /drone/pose.
        Lightweight alternative for simple RViz2 Pose display.
        """
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = FRAME_ODOM

        msg.pose.position.x    = self._enu_x
        msg.pose.position.y    = self._enu_y
        msg.pose.position.z    = self._enu_z
        msg.pose.orientation.x = float(self._q[0])
        msg.pose.orientation.y = float(self._q[1])
        msg.pose.orientation.z = float(self._q[2])
        msg.pose.orientation.w = float(self._q[3])

        self._pose_pub.publish(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('OdomTfBroadcaster stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
