cmake_minimum_required(VERSION 3.8)
project(drone_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY 
  launch 
  config 
  worlds 
  scripts
  models
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/circle_aruco_probes.py
  scripts/gz_joint_state_publisher.py
  scripts/joystick_bridge.py
  scripts/keyboard_bridge.py
  scripts/oak_height_bridge.py
  scripts/odom_tf_broadcaster.py
  scripts/offboard_controller.py
  scripts/ps5_joystick_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
