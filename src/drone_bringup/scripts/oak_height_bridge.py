#!/usr/bin/env python3
"""
gz_rangefinder_bridge.py

SITL equivalent of oak_height_bridge.py. Reads the simulated single-beam
lidar (from your model.sdf's gpu_lidar on height_link, relayed into ROS2
by ros_gz_bridge per gazebo_bridge_oak_px4.yaml) and republishes it as
px4_msgs/DistanceSensor on /fmu/in/distance_sensor, exactly matching the
real hardware pipeline's downstream shape.

RUN ORDER:
  1. PX4 SITL + Gazebo (drone_cage world, my_drone model)
  2. ros_gz_bridge with gazebo_bridge_oak_px4.yaml
  3. this script
  4. takeoff_land_range.py

Requires: sensor.lidar.range.min/max in model.sdf currently 0.10 / 12.0
          (matches MIN_VALID_M/MAX_VALID_M below - keep these in sync if
          you change the SDF).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from px4_msgs.msg import DistanceSensor


# ============================================================
# CONFIGURATION
# ============================================================

SCAN_TOPIC = '/drone/lidar_1d/range'   # matches gazebo_bridge_oak_px4.yaml ros_topic_name

# Must match model.sdf's <range><min>/<max> for the lidar_1d sensor.
MIN_VALID_M = 0.001
MAX_VALID_M = 12.0

PUBLISH_ORIENTATION = 25  # ROTATION_DOWNWARD_FACING
DEVICE_ID = 0x53494C31    # arbitrary fixed non-zero ID ("SIL1" in hex-ish)


class GzRangefinderBridge(Node):
    def __init__(self):
        super().__init__('gz_rangefinder_bridge')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.scan_sub = self.create_subscription(
            LaserScan, SCAN_TOPIC, self.scan_cb, sensor_qos
        )
        self.range_pub = self.create_publisher(
            DistanceSensor, '/fmu/in/distance_sensor', px4_qos
        )

        self.frame_count = 0
        self.reject_count = 0
        self.last_log_time = self.now_seconds()

        self.get_logger().info(
            f"GZ rangefinder bridge started. Listening on {SCAN_TOPIC}, "
            f"valid band [{MIN_VALID_M}, {MAX_VALID_M}] m."
        )

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def scan_cb(self, msg: LaserScan):
        self.frame_count += 1

        if not msg.ranges:
            self.reject_count += 1
            self._maybe_log()
            return

        distance_m = float(msg.ranges[0])

        # inf/nan or out-of-band means "no valid return" - don't publish
        if not (MIN_VALID_M < distance_m < MAX_VALID_M):
            self.reject_count += 1
            self._maybe_log()
            return

        self._publish(distance_m, msg)
        self._maybe_log()

    def _publish(self, distance_m, src_msg: LaserScan):
        out = DistanceSensor()

        # Use the scan's own capture timestamp, matching the delay-
        # compensation pattern PX4 expects (same as oak_height_bridge.py).
        stamp = src_msg.header.stamp
        out.timestamp = int(stamp.sec * 1_000_000 + stamp.nanosec / 1000)

        out.device_id = DEVICE_ID
        out.min_distance = MIN_VALID_M
        out.max_distance = MAX_VALID_M
        out.current_distance = distance_m

        # SDF's gaussian noise stddev is 0.02m -> variance = 0.02^2.
        # This is the simulated sensor's known noise, not a guess.
        out.variance = 0.0004

        out.signal_quality = 100  # SITL: no dropout modeling, treat every valid return as clean
        out.type = DistanceSensor.MAV_DISTANCE_SENSOR_LASER
        out.h_fov = 0.0  # single-beam sensor, negligible beam width
        out.v_fov = 0.0
        out.orientation = PUBLISH_ORIENTATION
        out.mode = DistanceSensor.MODE_ENABLED

        self.range_pub.publish(out)

    def _maybe_log(self):
        now = self.now_seconds()
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            self.get_logger().info(
                f"Frames: {self.frame_count} | Rejected: {self.reject_count}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GzRangefinderBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
