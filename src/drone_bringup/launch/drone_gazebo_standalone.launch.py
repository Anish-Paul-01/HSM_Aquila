"""
drone_gazebo_standalone.launch.py
══════════════════════════════════════════════════════════════════════════════
Standalone Gazebo launch — YOUR Gazebo, YOUR world, YOUR model.sdf.
PX4 attaches to the already-running Gazebo instance.

Boot order (critical — do not reorder):
  1. Gazebo starts with default.sdf  (world + plugins)
  2. my_drone spawned via gz service  (your model.sdf, not PX4's copy)
  3. PX4 SITL starts with PX4_GZ_STANDALONE=1
  4. XRCE-DDS agent bridges PX4 → ROS2
  5. ros_gz_bridge  (clock + camera topics)
  6. robot_state_publisher + RViz
  7. Custom nodes & bridges
  8. ArUco autonomous land (at t=60s)
"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node

# ── Paths ──────────────────────────────────────────────────────────────────────
PX4_DIR        = os.path.expanduser('~/PX4-Autopilot')
PX4_BIN        = os.path.join(PX4_DIR, 'build', 'px4_sitl_default', 'bin', 'px4')
PX4_BUILD_DIR  = os.path.join(PX4_DIR, 'build', 'px4_sitl_default')
PX4_GZ_MODELS  = os.path.join(PX4_DIR, 'Tools', 'simulation', 'gz', 'models')
PX4_GZ_WORLDS  = os.path.join(PX4_DIR, 'Tools', 'simulation', 'gz', 'worlds')
PX4_ROOTFS     = os.path.join(PX4_DIR, 'ROMFS', 'px4fmu_common')


def generate_launch_description():
    pkg_desc    = get_package_share_directory('drone_description')
    pkg_bringup = get_package_share_directory('drone_bringup')

    # ── File paths ──────────────────────────────────────────────────────────────
    xacro_file    = os.path.join(pkg_desc,    'urdf',   'drone.urdf.xacro')
    rviz_config   = os.path.join(pkg_desc,    'rviz',   'drone_urdf_config.rviz')
    world_file    = os.path.join(pkg_bringup, 'worlds', 'drone_cage.sdf')
    bridge_config = os.path.join(pkg_bringup, 'config', 'gazebo_bridge_oak_px4.yaml')

    pkg_sdf_dir = os.path.join(pkg_desc, 'sdf')
    pkg_models_dir = os.path.join(pkg_bringup, 'models')

    gz_resource_path = ':'.join([
        pkg_sdf_dir,
        pkg_models_dir,
        PX4_GZ_MODELS,
        PX4_GZ_WORLDS,
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ])

    set_gz_path = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path)
    set_gz_ip = SetEnvironmentVariable('GZ_IP', '127.0.0.1')

    # ── Process URDF/xacro ─────────────────────────────────────────────────────
    robot_description_xml = xacro.process_file(xacro_file).toxml()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Start Gazebo with YOUR world
    # ══════════════════════════════════════════════════════════════════════════
    gazebo = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'GZ_SIM_RESOURCE_PATH={gz_resource_path} '
            f'gz sim -r {world_file}'
        ],
        output='screen',
        name='gazebo',
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Spawn YOUR model.sdf into Gazebo (t=6s)
    # ══════════════════════════════════════════════════════════════════════════
    spawn_drone = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-c',
                    f'GZ_SIM_RESOURCE_PATH={gz_resource_path} '
                    'gz service -s /world/drone_cage/create '
                    '--reqtype gz.msgs.EntityFactory '
                    '--reptype gz.msgs.Boolean '
                    '--timeout 5000 '
                    '--req \'name: "my_drone_0" '
                    'sdf_filename: "model://my_drone/model.sdf" '
                    'pose: {position: {x: 0, y: 0, z: 0.24}}\''
                ],
                output='screen',
                name='spawn_drone',
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — PX4 SITL in standalone mode (t=12s)
    # ══════════════════════════════════════════════════════════════════════════
    px4_sitl = TimerAction(
        period=12.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-c',
                    f'cd {PX4_DIR} && '
                    f'GZ_SIM_RESOURCE_PATH={gz_resource_path} '
                    'PX4_GZ_STANDALONE=1 '
                    'PX4_SIM_MODEL=my_drone '
                    'PX4_GZ_MODEL_NAME=my_drone_0 '
                    'PX4_GZ_WORLD=drone_cage '
                    'PX4_GZ_MODEL_POSE="0,0,0.24,0,0,0" '
                    'PX4_SYS_AUTOSTART=4900 '
                    f'{PX4_BIN} -w sitl_my_drone -s etc/init.d-posix/rcS '
                    f'-d {PX4_ROOTFS}'
                ],
                output='screen',
                name='px4_sitl',
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — XRCE-DDS agent (t=15s)
    # ══════════════════════════════════════════════════════════════════════════
    xrce_agent = TimerAction(
        period=15.0,
        actions=[
            ExecuteProcess(
                cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
                output='screen',
                name='xrce_agent',
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — ros_gz_bridge (t=18s)
    # ══════════════════════════════════════════════════════════════════════════
    bridge = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='ros_gz_bridge',
                output='screen',
                parameters=[{'config_file': bridge_config}],
                additional_env={'GZ_SIM_RESOURCE_PATH': gz_resource_path},
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6a — robot_state_publisher (t=20s)
    # ══════════════════════════════════════════════════════════════════════════
    rsp = TimerAction(
        period=20.0,
        actions=[
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{
                    'robot_description': robot_description_xml,
                    'use_sim_time': True
                }],
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6b — RViz (t=22s)
    # ══════════════════════════════════════════════════════════════════════════
    rviz = TimerAction(
        period=22.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Custom Nodes & Camera Bridges
    # ══════════════════════════════════════════════════════════════════════════
    px4_tf_bridge = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='drone_bringup',
                executable='odom_tf_broadcaster.py',
                name='px4_tf_axis_bridge',
                output='screen',
            )
        ],
    )

    gz_joint_bridge = TimerAction(
        period=30.0,
        actions=[
            Node(
                package='drone_bringup',
                executable='gz_joint_state_publisher.py',
                name='gz_joint_state_publisher',
                output='screen',
            )
        ],
    )

    camera_bridge = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='camera_bridge',
                output='screen',
                parameters=[{'lazy': False, 'use_sim_time': True}],
                additional_env={'GZ_IP': '127.0.0.1'},
                arguments=[
                    '/drone/camera/rgb/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                    '/drone/camera/rgb/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                    '/drone/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                    '/drone/camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                    '/drone/camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                ],
            )
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — ArUco Autonomous Land (t=60s)
    # ══════════════════════════════════════════════════════════════════════════
    aruco_land_node = TimerAction(
        period=60.0,
        actions=[
            Node(
                package='drone_vision',
                executable='aruco_autonomous_land.py',
                name='aruco_autonomous_land',
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        set_gz_path,
        set_gz_ip,
        gazebo,
        spawn_drone,
        px4_sitl,
        xrce_agent,
        bridge,
        rsp,
        rviz,
        px4_tf_bridge,
        gz_joint_bridge,
        camera_bridge,
        aruco_land_node
    ])