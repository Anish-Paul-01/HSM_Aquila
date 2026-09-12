#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Joy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleStatus, VehicleLocalPosition,
)

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

AXIS_LEFT_X  = 0
AXIS_LEFT_Y  = 1
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
BTN_A = 0  # ARM
BTN_B = 1  # LAND + DISARM
BTN_X = 2  # OFFBOARD (holds current position)
BTN_Y = 3  # POSCTL

Z_MIN = -30.0
Z_MAX = -0.3
XY_MAX = 100.0
HEARTBEAT_REQUIRED = 20


class JoystickBridge(Node):

    def __init__(self):
        super().__init__('joystick_bridge')

        self._offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint,   '/fmu/in/trajectory_setpoint',  PX4_QOS)
        self._cmd_pub      = self.create_publisher(VehicleCommand,        '/fmu/in/vehicle_command',      PX4_QOS)

        self.create_subscription(VehicleStatus,        '/fmu/out/vehicle_status_v4',       self._status_cb,   PX4_QOS)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._local_pos_cb, PX4_QOS)
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        self.declare_parameter('xy_speed',  2.0)
        self.declare_parameter('z_speed',   1.0)
        self.declare_parameter('yaw_speed', 0.8)
        self.declare_parameter('deadzone',  0.08)
        self.declare_parameter('loop_hz',  20.0)

        self._axes: list[float] = []
        self._buttons: list[int] = []
        self._prev_buttons: list[int] = []
        self._arming_state  = VehicleStatus.ARMING_STATE_DISARMED
        self._nav_state     = VehicleStatus.NAVIGATION_STATE_MAX
        self._pre_flight_ok = False

        # Current position from EKF (NED)
        self._cur_x   = 0.0
        self._cur_y   = 0.0
        self._cur_z   = -1.5
        self._cur_yaw = 0.0

        # Setpoint (NED) — starts at home hover
        self._sp_x   = 0.0
        self._sp_y   = 0.0
        self._sp_z   = -1.5
        self._sp_yaw = 0.0

        self._heartbeat_count = 0
        dt = 1.0 / self.get_parameter('loop_hz').value
        self._dt = dt
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info('Joystick Bridge ready | A=ARM  B=LAND+DISARM  X=OFFBOARD  Y=POSCTL')

    def _status_cb(self, msg: VehicleStatus):
        self._arming_state  = msg.arming_state
        self._nav_state     = msg.nav_state
        self._pre_flight_ok = msg.pre_flight_checks_pass

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        self._cur_x   = msg.x
        self._cur_y   = msg.y
        self._cur_z   = msg.z
        self._cur_yaw = msg.heading

    def _joy_cb(self, msg: Joy):
        self._axes    = list(msg.axes)
        self._buttons = list(msg.buttons)

    def _timer_cb(self):
        self._publish_offboard_mode()
        self._heartbeat_count += 1
        self._process_buttons()
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

    def _process_buttons(self):
        if not self._buttons:
            return
        while len(self._prev_buttons) < len(self._buttons):
            self._prev_buttons.append(0)

        def pressed(idx):
            if idx >= len(self._buttons): return False
            return self._buttons[idx] == 1 and self._prev_buttons[idx] == 0

        if pressed(BTN_A): self._cmd_arm()
        if pressed(BTN_B): self._cmd_land()
        if pressed(BTN_X): self._cmd_offboard()
        if pressed(BTN_Y): self._cmd_posctl()
        self._prev_buttons = list(self._buttons)

    def _update_setpoint(self):
        if not self._axes:
            return
        xy  = self.get_parameter('xy_speed').value
        z   = self.get_parameter('z_speed').value
        yaw = self.get_parameter('yaw_speed').value
        dz  = self.get_parameter('deadzone').value

        def ax(idx):
            if idx >= len(self._axes): return 0.0
            v = self._axes[idx]
            return 0.0 if abs(v) < dz else float(v)

        self._sp_x   += ax(AXIS_RIGHT_Y) * xy  * self._dt
        self._sp_y   -= ax(AXIS_RIGHT_X) * xy  * self._dt
        self._sp_z   -= ax(AXIS_LEFT_Y)  * z   * self._dt
        self._sp_yaw -= ax(AXIS_LEFT_X)  * yaw * self._dt

        self._sp_z   = max(Z_MIN, min(Z_MAX, self._sp_z))
        self._sp_x   = max(-XY_MAX, min(XY_MAX, self._sp_x))
        self._sp_y   = max(-XY_MAX, min(XY_MAX, self._sp_y))
        self._sp_yaw = math.atan2(math.sin(self._sp_yaw), math.cos(self._sp_yaw))

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

    def _cmd_arm(self):
        if self._heartbeat_count < HEARTBEAT_REQUIRED:
            self.get_logger().warn(f'ARM denied: warming up ({self._heartbeat_count}/{HEARTBEAT_REQUIRED})')
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
        # Switch to AUTO.LAND — PX4 lands and auto-disarms safely
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        #self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=9.0)
        self.get_logger().info('LAND command sent — will auto-disarm on touchdown')

    def _cmd_offboard(self):
        if not self._is_armed():
            self.get_logger().warn('OFFBOARD denied: ARM first')
            return
        # Snap setpoint to current position so drone holds in place
        self._sp_x   = self._cur_x
        self._sp_y   = self._cur_y
        self._sp_z   = self._cur_z
        self._sp_yaw = self._cur_yaw if not math.isnan(self._cur_yaw) else self._sp_yaw
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(
            f'OFFBOARD mode sent — holding x:{self._sp_x:.2f} y:{self._sp_y:.2f} z:{self._sp_z:.2f}'
        )

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

    def _is_armed(self):    return self._arming_state == VehicleStatus.ARMING_STATE_ARMED
    def _is_offboard(self): return self._nav_state    == VehicleStatus.NAVIGATION_STATE_OFFBOARD
    def _ts(self):          return self.get_clock().now().nanoseconds // 1000


def main(args=None):
    rclpy.init(args=args)
    node = JoystickBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
