#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    oak_params_file_arg = DeclareLaunchArgument(
        'oak_params_file',
        default_value='/home/sar/drone_ws/config/oak_all.yaml',
        description='Path to the DepthAI camera parameters YAML file'
    )

    # Micro XRCE-DDS Agent - bridges PX4 <-> ROS2 over serial.
    # This is a plain binary, not a ROS node, so it goes through ExecuteProcess.
    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'serial', '--dev', '/dev/ttyCH341USB0', '-b', '921600'],
        output='screen',
        name='micro_xrce_agent',
    )

    # OAK-D camera driver
    depthai_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver_v3'),
                'launch',
                'driver.launch.py'
            ])
        ),
        launch_arguments={
            'params_file': LaunchConfiguration('oak_params_file')
        }.items()
    )

    # MATLAB <-> ROS2 bridge
    matlab_ros_node = Node(
        package='drone_bringup',
        executable='circle_aruco_probes.py',
        name='matlab_ros_bridge',
        output='screen',
    )

    return LaunchDescription([
        oak_params_file_arg,
        xrce_agent,
        TimerAction(
            period=5.0,
            actions=[depthai_driver]
        ),
        TimerAction(
            period=25.0,
            actions=[matlab_ros_node]
        ),
    ])