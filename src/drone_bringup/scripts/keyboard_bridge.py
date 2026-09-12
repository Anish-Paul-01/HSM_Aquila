#!/usr/bin/env python3
"""
keyboard_bridge.py  —  PX4 SITL keyboard tele-op (replaces joystick_bridge.py)

Requires:  pip install readchar --break-system-packages

Controls
--------
  W / S       Forward / Backward  (NED +X / -X)
  A / D       Strafe Left / Right (NED -Y / +Y)
  R / F       Up / Down           (NED -Z / +Z)
  Q / E       Yaw Left / Right
  ──────────────────────────────────────────────
  I           ARM
  K           LAND + auto-disarm
  O           OFFBOARD mode  (snaps setpoint to current pos)
  P           POSCTL mode
  X / Ctrl-C  Quit
"""

import math
import threading
import sys

import readchar
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleStatus, VehicleLocalPosition,
)

# ── QoS ──────────────────────────────────────────────────────────────────────
PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── Limits ───────────────────────────────────────────────────────────────────
Z_MIN = -30.0
Z_MAX = -0.3
XY_MAX = 100.0
HEARTBEAT_REQUIRED = 20


class KeyboardBridge(Node):

    def __init__(self):
        super().__init__('keyboard_bridge')

        # ── Publishers ───────────────────────────────────────────────────────
        self._offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self._setpoint_pub = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  PX4_QOS)
        self._cmd_pub = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',      PX4_QOS)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            VehicleStatus,        '/fmu/out/vehicle_status_v4',
            self._status_cb,   PX4_QOS)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self._local_pos_cb, PX4_QOS)

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('xy_speed',  2.0)
        self.declare_parameter('z_speed',   1.0)
        self.declare_parameter('yaw_speed', 0.8)
        self.declare_parameter('loop_hz',  20.0)

        # ── State ────────────────────────────────────────────────────────────
        self._arming_state  = VehicleStatus.ARMING_STATE_DISARMED
        self._nav_state     = VehicleStatus.NAVIGATION_STATE_MAX
        self._pre_flight_ok = False

        self._cur_x   = 0.0
        self._cur_y   = 0.0
        self._cur_z   = -1.5
        self._cur_yaw = 0.0

        self._sp_x   = 0.0
        self._sp_y   = 0.0
        self._sp_z   = -1.5
        self._sp_yaw = 0.0

        # Keys currently held down
        self._held: set[str] = set()
        self._held_lock = threading.Lock()

        self._heartbeat_count = 0
        self._dt = 1.0 / self.get_parameter('loop_hz').value
        self.create_timer(self._dt, self._timer_cb)

        # ── Start keyboard reader thread ──────────────────────────────────
        self._running = True
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

        self._print_banner()

    # ── Subscribers ──────────────────────────────────────────────────────────

    def _status_cb(self, msg: VehicleStatus):
        self._arming_state  = msg.arming_state
        self._nav_state     = msg.nav_state
        self._pre_flight_ok = msg.pre_flight_checks_pass

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        self._cur_x   = msg.x
        self._cur_y   = msg.y
        self._cur_z   = msg.z
        self._cur_yaw = msg.heading

    # ── Keyboard thread ───────────────────────────────────────────────────────

    def _keyboard_loop(self):
        """
        Reads key events in a tight loop.
        readchar.readkey() is blocking but returns immediately on each keypress.
        We use a simple press-and-release model: a movement key contributes to
        the setpoint for exactly one timer tick per press.  Hold the key for
        continuous movement (OS key-repeat will re-fire it).
        """
        while self._running:
            try:
                key = readchar.readkey().lower()
            except Exception:
                break

            if key in ('x', readchar.key.CTRL_C):
                self.get_logger().info('Quit key pressed — shutting down')
                self._running = False
                rclpy.shutdown()
                sys.exit(0)

            # One-shot commands
            if key == 'i':
                self._cmd_arm()
            elif key == 'k':
                self._cmd_land()
            elif key == 'o':
                self._cmd_offboard()
            elif key == 'p':
                self._cmd_posctl()
            else:
                # Movement keys: add to held set; timer will consume them
                with self._held_lock:
                    self._held.add(key)

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        self._publish_offboard_mode()
        self._heartbeat_count += 1
        self._update_setpoint()
        self._publish_setpoint()

        if self._heartbeat_count % 40 == 0:
            armed = 'ARMED'    if self._is_armed()    else 'DISARMED'
            mode  = 'OFFBOARD' if self._is_offboard() else f'nav={self._nav_state}'
            pf    = 'PF:OK'    if self._pre_flight_ok else 'PF:FAIL'
            ready = 'READY'    if self._heartbeat_count >= HEARTBEAT_REQUIRED else 'WARMING'
            self.get_logger().info(
                f'[{armed}][{mode}][{pf}][{ready}] '
                f'sp x:{self._sp_x:.2f} y:{self._sp_y:.2f} z:{self._sp_z:.2f} | '
                f'cur x:{self._cur_x:.2f} y:{self._cur_y:.2f} z:{self._cur_z:.2f}'
            )

    # ── Setpoint update ───────────────────────────────────────────────────────

    def _update_setpoint(self):
        xy  = self.get_parameter('xy_speed').value
        z   = self.get_parameter('z_speed').value
        yaw = self.get_parameter('yaw_speed').value
        dt  = self._dt

        with self._held_lock:
            held = set(self._held)
            self._held.clear()   # consume; OS key-repeat will re-add held keys

        if 'w' in held: self._sp_x += xy  * dt
        if 's' in held: self._sp_x -= xy  * dt
        if 'd' in held: self._sp_y += xy  * dt
        if 'a' in held: self._sp_y -= xy  * dt
        if 'r' in held: self._sp_z -= z   * dt   # up   → more negative NED
        if 'f' in held: self._sp_z += z   * dt   # down → less negative NED
        if 'q' in held: self._sp_yaw -= yaw * dt
        if 'e' in held: self._sp_yaw += yaw * dt

        self._sp_z   = max(Z_MIN, min(Z_MAX,  self._sp_z))
        self._sp_x   = max(-XY_MAX, min(XY_MAX, self._sp_x))
        self._sp_y   = max(-XY_MAX, min(XY_MAX, self._sp_y))
        self._sp_yaw = math.atan2(math.sin(self._sp_yaw), math.cos(self._sp_yaw))

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._ts()
        msg.position  = True
        self._offboard_pub.publish(msg)

    def _publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp    = self._ts()
        msg.position     = [self._sp_x, self._sp_y, self._sp_z]
        msg.yaw          = self._sp_yaw
        msg.velocity     = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        self._setpoint_pub.publish(msg)

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_arm(self):
        if self._heartbeat_count < HEARTBEAT_REQUIRED:
            self.get_logger().warn(
                f'ARM denied: warming up ({self._heartbeat_count}/{HEARTBEAT_REQUIRED})')
            return
        if not self._pre_flight_ok:
            self.get_logger().warn('ARM denied: preflight FAIL — run "commander check" in pxh>')
            return
        if self._is_armed():
            self.get_logger().warn('Already ARMED')
            return
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('ARM command sent')

    def _cmd_land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND command sent — will auto-disarm on touchdown')

    def _cmd_offboard(self):
        if not self._is_armed():
            self.get_logger().warn('OFFBOARD denied: ARM first')
            return
        self._sp_x   = self._cur_x
        self._sp_y   = self._cur_y
        self._sp_z   = self._cur_z
        self._sp_yaw = self._cur_yaw if not math.isnan(self._cur_yaw) else self._sp_yaw
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(
            f'OFFBOARD mode sent — holding '
            f'x:{self._sp_x:.2f} y:{self._sp_y:.2f} z:{self._sp_z:.2f}')

    def _cmd_posctl(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=2.0)
        self.get_logger().info('POSCTL mode sent')

    def _send_cmd(self, command: int, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp        = self._ts()
        msg.command          = command
        msg.param1           = param1
        msg.param2           = param2
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self._cmd_pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_armed(self):    return self._arming_state == VehicleStatus.ARMING_STATE_ARMED
    def _is_offboard(self): return self._nav_state    == VehicleStatus.NAVIGATION_STATE_OFFBOARD
    def _ts(self):          return self.get_clock().now().nanoseconds // 1000

    def _print_banner(self):
        banner = """
╔══════════════════════════════════════════════════╗
║           PX4 Keyboard Bridge  (NED)             ║
╠══════════════════════════════════════════════════╣
║  W / S   Forward  / Backward                     ║
║  A / D   Strafe Left / Right                     ║
║  R / F   Up / Down                               ║
║  Q / E   Yaw Left / Right                        ║
╠══════════════════════════════════════════════════╣
║  I       ARM                                     ║
║  K       LAND (auto-disarm)                      ║
║  O       OFFBOARD mode                           ║
║  P       POSCTL  mode                            ║
║  X       Quit                                    ║
╚══════════════════════════════════════════════════╝
Terminal must have focus for key input to register.
"""
        print(banner, flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
