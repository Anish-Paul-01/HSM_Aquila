#!/usr/bin/env python3
"""
ps5_joystick_bridge.py
======================
ROS 2 node — maps a PS5 DualSense controller (Bluetooth / joy_node) to
PX4 Offboard position setpoints.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Joy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Axis Indices (Linux hid-playstation driver)
AXIS_LEFT_X   = 0   # Yaw (CW / CCW)
AXIS_LEFT_Y   = 1   # Altitude Z (+Up / -Down)
AXIS_RIGHT_X  = 3   # Pitch/Roll Y (+Left / -Right)
AXIS_RIGHT_Y  = 4   # Pitch/Roll X (+Forward / -Back)

# Button Indices
BTN_CROSS     = 0   # ✕  → ARM
BTN_CIRCLE    = 1   # ○  → LAND
BTN_SQUARE    = 2   # □  → OFFBOARD
BTN_TRIANGLE  = 3   # △  → POSCTL

# Flight limits
Z_MIN          = -50.0    # max altitude (NED, negative = up)
Z_MAX          = -0.3     # min altitude
XY_MAX         = 200.0    # max range in meters
HEARTBEAT_REQ  = 20       # ticks before state actions allowed


class PS5JoystickBridge(Node):
    def __init__(self):
        super().__init__('ps5_joystick_bridge')

        # Publishers
        self._offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self._setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self._cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)

        # Subscribers
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self._status_cb, PX4_QOS)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._local_pos_cb, PX4_QOS)
        self.create_subscription(
            Joy, '/joy', self._joy_cb, 10)

        # Parameters
        self.declare_parameter('xy_speed',  3.0)   # m/s
        self.declare_parameter('z_speed',   1.5)   # m/s
        self.declare_parameter('yaw_speed', 1.0)   # rad/s
        self.declare_parameter('deadzone',  0.08)  # stick deadzone
        self.declare_parameter('loop_hz',  20.0)

        # State Variables
        self._axes:         list[float] = []
        self._buttons:      list[int]   = []
        self._prev_buttons: list[int]   = []

        self._arming_state  = VehicleStatus.ARMING_STATE_DISARMED
        self._nav_state     = VehicleStatus.NAVIGATION_STATE_MAX
        self._pre_flight_ok = False
        self._pos_received  = False

        # EKF position (NED)
        self._cur_x   = 0.0
        self._cur_y   = 0.0
        self._cur_z   = 0.0
        self._cur_yaw = 0.0

        # Target Setpoint (NED)
        self._sp_x   = 0.0
        self._sp_y   = 0.0
        self._sp_z   = -1.5
        self._sp_yaw = 0.0

        self._heartbeat_count = 0
        loop_hz = self.get_parameter('loop_hz').value
        self._dt = 1.0 / loop_hz
        self.create_timer(self._dt, self._timer_cb)

        self.get_logger().info('PS5 Joystick Bridge initialised.')

    def _status_cb(self, msg: VehicleStatus):
        self._arming_state  = msg.arming_state
        self._nav_state     = msg.nav_state
        self._pre_flight_ok = msg.pre_flight_checks_pass

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        self._cur_x   = msg.x
        self._cur_y   = msg.y
        self._cur_z   = msg.z
        self._cur_yaw = msg.heading
        self._pos_received = True

    def _joy_cb(self, msg: Joy):
        self._axes    = list(msg.axes)
        self._buttons = list(msg.buttons)

    def _timer_cb(self):
        self._publish_offboard_mode()
        self._heartbeat_count += 1
        self._process_buttons()

        # Update and lock setpoint to current position if disarmed or not in OFFBOARD mode
        if not self._is_offboard() or not self._is_armed():
            if self._pos_received:
                self._sp_x   = self._cur_x
                self._sp_y   = self._cur_y
                self._sp_z   = self._cur_z
                self._sp_yaw = self._cur_yaw if not math.isnan(self._cur_yaw) else 0.0
        else:
            self._update_setpoint()

        self._publish_setpoint()

        if self._heartbeat_count % 40 == 0:
            armed  = 'ARMED'    if self._is_armed()    else 'DISARMED'
            mode   = 'OFFBOARD' if self._is_offboard() else f'NAV={self._nav_state}'
            pf     = 'PF:OK'    if self._pre_flight_ok else 'PF:FAIL'
            self.get_logger().info(
                f'[{armed}][{mode}][{pf}] '
                f'SP: x:{self._sp_x:5.2f} y:{self._sp_y:5.2f} z:{self._sp_z:5.2f} | '
                f'CUR: x:{self._cur_x:5.2f} y:{self._cur_y:5.2f} z:{self._cur_z:5.2f}'
            )

    def _process_buttons(self):
        if not self._buttons:
            return

        while len(self._prev_buttons) < len(self._buttons):
            self._prev_buttons.append(0)

        def pressed(idx: int) -> bool:
            return idx < len(self._buttons) and self._buttons[idx] == 1 and self._prev_buttons[idx] == 0

        if pressed(BTN_CROSS):    self._cmd_arm()
        if pressed(BTN_CIRCLE):   self._cmd_land()
        if pressed(BTN_SQUARE):   self._cmd_offboard()
        if pressed(BTN_TRIANGLE): self._cmd_posctl()

        self._prev_buttons = list(self._buttons)

    def _update_setpoint(self):
        if not self._axes:
            return

        xy_spd  = self.get_parameter('xy_speed').value
        z_spd   = self.get_parameter('z_speed').value
        yaw_spd = self.get_parameter('yaw_speed').value
        dz      = self.get_parameter('deadzone').value

        def ax(idx: int) -> float:
            if idx >= len(self._axes):
                return 0.0
            v = self._axes[idx]
            return 0.0 if abs(v) < dz else float(v)

        # Map stick commands (NED frame)
        self._sp_x   += ax(AXIS_RIGHT_Y) * xy_spd  * self._dt
        self._sp_y   -= ax(AXIS_RIGHT_X) * xy_spd  * self._dt
        self._sp_z   -= ax(AXIS_LEFT_Y)  * z_spd   * self._dt
        self._sp_yaw -= ax(AXIS_LEFT_X)  * yaw_spd * self._dt

        # Clamp parameters
        self._sp_z   = max(Z_MIN, min(Z_MAX, self._sp_z))
        self._sp_x   = max(-XY_MAX, min(XY_MAX, self._sp_x))
        self._sp_y   = max(-XY_MAX, min(XY_MAX, self._sp_y))
        self._sp_yaw = math.atan2(math.sin(self._sp_yaw), math.cos(self._sp_yaw))

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._ts()
        msg.position  = True
        msg.velocity  = False
        msg.acceleration = False
        msg.attitude  = False
        msg.body_rate = False
        self._offboard_pub.publish(msg)

    def _publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp    = self._ts()
        msg.position     = [self._sp_x, self._sp_y, self._sp_z]
        msg.yaw          = self._sp_yaw
        msg.velocity     = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        self._setpoint_pub.publish(msg)

    def _cmd_arm(self):
        if self._heartbeat_count < HEARTBEAT_REQ:
            self.get_logger().warn('ARM denied — streaming warmup in progress')
            return
        if not self._pre_flight_ok:
            self.get_logger().warn('ARM denied — preflight checks FAILED')
            return
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('ARM command sent')

    def _cmd_land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND command sent')

    def _cmd_offboard(self):
        if self._pos_received:
            self._sp_x   = self._cur_x
            self._sp_y   = self._cur_y
            self._sp_z   = self._cur_z
            self._sp_yaw = self._cur_yaw if not math.isnan(self._cur_yaw) else self._sp_yaw

        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('OFFBOARD mode command sent')

    def _cmd_posctl(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=2.0)
        self.get_logger().info('POSCTL mode command sent')

    def _send_cmd(self, command: int, param1: float = 0.0, param2: float = 0.0):
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

    def _is_armed(self) -> bool:
        return self._arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _is_offboard(self) -> bool:
        return self._nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _ts(self) -> int:
        # Fixed: Microsecond Epoch time matching PX4 SITL host clock
        return int(time.time() * 1e6)


def main(args=None):
    rclpy.init(args=args)
    node = PS5JoystickBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()