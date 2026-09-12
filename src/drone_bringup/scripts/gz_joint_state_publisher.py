#!/usr/bin/env python3
"""
gz_joint_state_publisher.py
══════════════════════════════════════════════════════════════════════════════
Subscribes to /my_drone/joint_states (gz.msgs.Model) directly via the
gz-transport Python API and republishes as sensor_msgs/msg/JointState on
/joint_states so robot_state_publisher can animate the rotors in RViz.

Why this node exists:
  ros_gz_bridge on ROS2 Humble has NO supported conversion pair for
  gz.msgs.Model → sensor_msgs/msg/JointState. Verified with --print-pairs.
  This node does the conversion manually.

How to run standalone (for testing):
  ros2 run drone_bringup gz_joint_state_publisher.py

Topics:
  SUB  /my_drone/joint_states   gz-transport  gz.msgs.Model
  PUB  /joint_states            ROS2           sensor_msgs/msg/JointState
══════════════════════════════════════════════════════════════════════════════
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# gz-transport Python bindings — ships with Gazebo Harmonic
from gz.transport13 import Node as GzNode
from gz.msgs10.model_pb2 import Model


class GzJointStatePublisher(Node):

    def __init__(self):
        super().__init__('gz_joint_state_publisher')

        # ROS2 publisher — robot_state_publisher listens here
        self._ros_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # gz-transport subscriber
        self._gz_node = GzNode()
        self._gz_node.subscribe(
            Model,
            '/my_drone/joint_states',
            self._gz_callback
        )

        self.get_logger().info(
            'gz_joint_state_publisher started — '
            'bridging /my_drone/joint_states → /joint_states'
        )

    def _gz_callback(self, gz_model: Model):
        """
        Convert gz.msgs.Model → sensor_msgs/JointState.
        gz.msgs.Model contains a repeated joint field, each with:
          - name        : joint name string
          - axis1.position : current angle (rad)
          - axis1.velocity : current velocity (rad/s)
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        for joint in gz_model.joint:
            msg.name.append(joint.name)
            msg.position.append(joint.axis1.position)
            msg.velocity.append(joint.axis1.velocity)
            msg.effort.append(0.0)

        self._ros_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GzJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
