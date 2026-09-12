#!/usr/bin/env python3
"""
Offboard Position Controller — PID with anti-windup.
Uses TF tree (odom → base_link) for real 3D position feedback.
Uses /odom twist for XY velocity damping only.

Topics:
  SUB  /odom               (nav_msgs/Odometry) — XY velocity only
  TF   odom → base_link    — real X, Y, Z position
  PUB  /drone/cmd_vel      (geometry_msgs/Twist)
  PUB  /drone/enable       (std_msgs/Bool)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import math
import tf2_ros
from tf2_ros import TransformException


class PIDController:
    """Single-axis PID with integral anti-windup (clamping)."""

    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def compute(self, error, derivative, dt):
        """
        error      — setpoint − measurement
        derivative — pre-computed d/dt of the controlled variable (velocity)
                     passed in directly so we can use cleaner sensor data
        dt         — time step in seconds
        """
        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))

        return (self.kp * error
                + self.ki * self._integral
                - self.kd * derivative)

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


class OffboardController(Node):

    def __init__(self):
        super().__init__('offboard_controller')

        # ── Tunable gains ──────────────────────────────────────────────────
        self.declare_parameter('kp_z',    0.6)
        self.declare_parameter('ki_z',    0.05)
        self.declare_parameter('kd_z',    0.4)

        self.declare_parameter('kp_xy',   0.9)
        self.declare_parameter('ki_xy',   0.03)
        self.declare_parameter('kd_xy',   0.5)

        self.declare_parameter('kp_yaw',  1.0)
        self.declare_parameter('ki_yaw',  0.01)
        self.declare_parameter('kd_yaw',  0.1)

        # ── Integral limits (anti-windup) ──────────────────────────────────
        self.declare_parameter('int_limit_z',   0.5)
        self.declare_parameter('int_limit_xy',  0.5)
        self.declare_parameter('int_limit_yaw', 0.3)

        # ── Hover setpoint ─────────────────────────────────────────────────
        self.declare_parameter('target_x',    0.0)
        self.declare_parameter('target_y',    0.0)
        self.declare_parameter('target_z',    1.2)
        self.declare_parameter('target_yaw',  0.0)

        # ── Output clamps ──────────────────────────────────────────────────
        self.declare_parameter('max_vz',   1.0)
        self.declare_parameter('max_vxy',  2.0)
        self.declare_parameter('max_vyaw', 0.8)

        self.declare_parameter('arm_delay', 5.0)

        # ── State ──────────────────────────────────────────────────────────
        self.pos_x  = 0.0
        self.pos_y  = 0.0
        self.pos_z  = 0.0
        self.yaw    = 0.0

        self.vel_x  = 0.0
        self.vel_y  = 0.0
        self.vel_z  = 0.0
        self.prev_z = 0.0

        self.prev_yaw_err = 0.0

        self.dt       = 0.02   # 50 Hz
        self.tf_ready = False
        self.armed    = False

        # ── PID controllers ────────────────────────────────────────────────
        self.pid_z = PIDController(
            kp=self.get_parameter('kp_z').value,
            ki=self.get_parameter('ki_z').value,
            kd=self.get_parameter('kd_z').value,
            integral_limit=self.get_parameter('int_limit_z').value)

        self.pid_x = PIDController(
            kp=self.get_parameter('kp_xy').value,
            ki=self.get_parameter('ki_xy').value,
            kd=self.get_parameter('kd_xy').value,
            integral_limit=self.get_parameter('int_limit_xy').value)

        self.pid_y = PIDController(
            kp=self.get_parameter('kp_xy').value,
            ki=self.get_parameter('ki_xy').value,
            kd=self.get_parameter('kd_xy').value,
            integral_limit=self.get_parameter('int_limit_xy').value)

        self.pid_yaw = PIDController(
            kp=self.get_parameter('kp_yaw').value,
            ki=self.get_parameter('ki_yaw').value,
            kd=self.get_parameter('kd_yaw').value,
            integral_limit=self.get_parameter('int_limit_yaw').value)

        # ── TF listener ────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Publishers ─────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool,  '/drone/enable',  10)

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ── Timers ─────────────────────────────────────────────────────────
        arm_delay = self.get_parameter('arm_delay').value
        self.arm_timer  = self.create_timer(arm_delay, self.arm_drone)
        self.ctrl_timer = self.create_timer(self.dt,   self.control_loop)

        self.get_logger().info(
            f'Offboard PID controller ready — arming in {arm_delay}s')
        self.get_logger().info(
            'Reading Z height from TF (odom → base_link)')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        # XY velocity only — Z is always 0 in this drone's odom
        self.vel_x = msg.twist.twist.linear.x
        self.vel_y = msg.twist.twist.linear.y

    def read_tf(self):
        """Read real 3D position from TF tree. Returns True on success."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time())

            self.pos_x = tf.transform.translation.x
            self.pos_y = tf.transform.translation.y
            self.pos_z = tf.transform.translation.z

            # Numerical Z velocity (odom doesn't give us this)
            self.vel_z = (self.pos_z - self.prev_z) / self.dt
            self.prev_z = self.pos_z

            # Yaw from quaternion
            q = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny, cosy)

            if not self.tf_ready:
                self.tf_ready = True
                self.get_logger().info(
                    f'TF ready — initial Z = {self.pos_z:.3f} m')
            return True

        except TransformException as e:
            if not self.tf_ready:
                self.get_logger().warn(f'Waiting for TF odom→base_link: {e}')
            return False

    # ── Arm ────────────────────────────────────────────────────────────────

    def arm_drone(self):
        if not self.read_tf():
            return

        if not hasattr(self, '_arm_stable_count'):
            self._arm_stable_count = 0

        self._arm_stable_count += 1

        if self._arm_stable_count < 10:
            self.get_logger().info(
                f'Waiting for stable TF... ({self._arm_stable_count}/10) '
                f'pos=({self.pos_x:.2f}, {self.pos_y:.2f}, {self.pos_z:.2f})')
            return

        # Capture spawn XY as hold target; reset all integrators cleanly
        self.set_parameters([
            rclpy.parameter.Parameter('target_x',
                rclpy.Parameter.Type.DOUBLE, self.pos_x),
            rclpy.parameter.Parameter('target_y',
                rclpy.Parameter.Type.DOUBLE, self.pos_y),
        ])
        self.pid_z.reset()
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_yaw.reset()
        self.prev_yaw_err = 0.0

        msg = Bool()
        msg.data = True
        self.enable_pub.publish(msg)
        self.armed = True
        self.arm_timer.cancel()   # one-shot — don't re-arm
        self.get_logger().info(
            f'ARMED → holding x={self.pos_x:.2f} y={self.pos_y:.2f}')

    # ── Gain sync ──────────────────────────────────────────────────────────

    def _sync_gains(self):
        """Pull live ROS params into PID objects (allows runtime ros2 param set tuning)."""
        self.pid_z.set_gains(
            self.get_parameter('kp_z').value,
            self.get_parameter('ki_z').value,
            self.get_parameter('kd_z').value)

        kp_xy = self.get_parameter('kp_xy').value
        ki_xy = self.get_parameter('ki_xy').value
        kd_xy = self.get_parameter('kd_xy').value
        self.pid_x.set_gains(kp_xy, ki_xy, kd_xy)
        self.pid_y.set_gains(kp_xy, ki_xy, kd_xy)

        self.pid_yaw.set_gains(
            self.get_parameter('kp_yaw').value,
            self.get_parameter('ki_yaw').value,
            self.get_parameter('kd_yaw').value)

    # ── Control loop ───────────────────────────────────────────────────────

    def control_loop(self):
        if not self.armed:
            return
        if not self.read_tf():
            return

        self._sync_gains()

        tx  = self.get_parameter('target_x').value
        ty  = self.get_parameter('target_y').value
        tz  = self.get_parameter('target_z').value
        tyw = self.get_parameter('target_yaw').value

        max_vz   = self.get_parameter('max_vz').value
        max_vxy  = self.get_parameter('max_vxy').value
        max_vyaw = self.get_parameter('max_vyaw').value

        # Errors
        ez  = tz - self.pos_z
        ex  = tx - self.pos_x
        ey  = ty - self.pos_y
        # Shortest-path yaw error
        eyw = math.atan2(math.sin(tyw - self.yaw),
                         math.cos(tyw - self.yaw))

        # Yaw derivative from error difference (avoids derivative kick)
        dyaw = (eyw - self.prev_yaw_err) / self.dt
        self.prev_yaw_err = eyw

        # PID outputs
        # X/Y/Z: derivative term uses measured velocity (not error diff)
        #        → no spike when setpoint changes mid-flight
        vz  = self.pid_z.compute(ez,  self.vel_z,  self.dt)
        vx  = self.pid_x.compute(ex,  self.vel_x,  self.dt)
        vy  = self.pid_y.compute(ey,  self.vel_y,  self.dt)
        wz  = self.pid_yaw.compute(eyw, -dyaw,     self.dt)

        # Clamp outputs
        vz  = max(-max_vz,   min(max_vz,   vz))
        vx  = max(-max_vxy,  min(max_vxy,  vx))
        vy  = max(-max_vxy,  min(max_vxy,  vy))
        wz  = max(-max_vyaw, min(max_vyaw, wz))

        cmd = Twist()
        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.linear.z  = vz
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

        # Log every 1 s (50 cycles × 0.02 s)
        if not hasattr(self, '_tick'):
            self._tick = 0
        self._tick += 1
        if self._tick % 50 == 0:
            self.get_logger().info(
                f'POS  x:{self.pos_x:6.2f} y:{self.pos_y:6.2f} z:{self.pos_z:6.2f} | '
                f'TGT  x:{tx:6.2f} y:{ty:6.2f} z:{tz:6.2f} | '
                f'ERR  x:{ex:5.2f} y:{ey:5.2f} z:{ez:5.2f} | '
                f'INT  z:{self.pid_z._integral:5.3f} '
                f'x:{self.pid_x._integral:5.3f} '
                f'y:{self.pid_y._integral:5.3f}')

    def disarm(self):
        msg = Bool()
        msg.data = False
        self.enable_pub.publish(msg)
        self.armed = False
        self.get_logger().info('DISARMED')


def main(args=None):
    rclpy.init(args=args)
    node = OffboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.disarm()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()