# HSM_Aquila — ERC 2026 Droning Sub-Task

ROS 2 workspace for **HSM Aries**'s autonomous quadrotor, built for the European Rover Challenge (ERC) Droning Sub-Task. The stack flies a search pattern over an unknown field, detects ground probes and an ArUco landing marker, then returns and lands precisely on the marker — in Gazebo simulation and on the real Jetson/Pixhawk hardware.

## Media & Flight Demonstrations

### 1. SITL Simulation (Gazebo)
*Demonstration of the drone executing the search circle, locating the ArUco marker, and performing the continuous precision descent sequence in a simulated environment.*

[Watch: SITL Flight Demo](https://youtu.be/6rYdZYjD0zA)

### 2. Real-World Hardware Flight (ERC)
*Actual hardware flight footage demonstrating the visual servoing script running on the Jetson Orin companion computer and Luxonis OAK-D Pro camera.*

[Watch: Real World ERC Flight](https://youtu.be/70YiU4rVjNg)

---

## What's in this repository

This repo *is* a colcon workspace: it already contains the `src/` folder colcon expects at its root, alongside a PX4 airframe file used by the simulation.

| Path | Type | Purpose |
|---|---|---|
| `src/drone_description` | ament_cmake | URDF/xacro + Gazebo SDF model of the quadrotor (`my_drone`), meshes, and an RViz config |
| `src/drone_bringup` | ament_cmake | Launch files, the Gazebo world and marker/probe models, and the ROS 2 nodes that glue PX4, Gazebo, and the vehicle together (TF bridge, joint-state bridge, rangefinder bridge, teleop bridges, and the real-hardware mission script) |
| `src/drone_vision` | ament_cmake | The **simulation** mission node — YOLO-based probe detection + ArUco-102 detection + orbit/return/land state machine |
| `4900_gz_my_drone` (repo root) | PX4 airframe file | Custom PX4 parameter set for the quadrotor (autostart ID `4900`) — gets installed into your PX4-Autopilot checkout, not built by colcon |

## What the mission actually does

There are two separate mission "brains" in this repo, for two different targets:

- **Simulation** — `drone_vision/aruco_autonomous_land.py`: takeoff, fly out to an orbit radius, circle while a YOLO model looks for ground probes and OpenCV looks for ArUco marker 102, log each *confirmed* probe (deduplicated, position-averaged) to `probes_location.csv`, then align, fly straight back, and land — either once 3 probes are confirmed or after 4 full orbits as a hard safety cap.
- **Real hardware** — `drone_bringup/scripts/circle_aruco_probes.py`: takeoff → transit → search circle → slow, continuous visual centering on ArUco 102 all the way from 2.5 m down to 0.45 m AGL → touchdown → forced disarm. It streams a live annotated MJPEG feed at `http://<drone-ip>:8080` and expects a probe count on `/erc/probe_count`, published externally (in this project, by a MATLAB-based detector).

## Hardware (real-drone target platform)

Only relevant if you're deploying to the physical drone — the simulation path below needs none of this.

- Flight controller: Pixhawk 6C
- Companion computer: NVIDIA Jetson Orin
- Vision: Luxonis OAK-D Pro, mounted +100 mm forward / −70 mm up relative to the drone's centre of gravity
- Frame: custom quadrotor, ~3.05 kg all-up, T-Motor MN4110 400KV motors, 15×5" props, 6S battery

## Software prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- PX4-Autopilot (`main` branch is easiest — just make sure `px4_msgs` below is the matching branch/tag)
- Gazebo Harmonic (installed automatically by PX4's setup script, see step 2)
- Micro XRCE-DDS Agent

ROS packages:

```bash
sudo apt install ros-humble-robot-state-publisher ros-humble-rviz2 \
  ros-humble-ros-gz-bridge ros-humble-ros-gz-sim ros-humble-xacro \
  ros-humble-cv-bridge
```

Python packages:

```bash
pip install --break-system-packages opencv-python opencv-contrib-python numpy \
  ultralytics pillow readchar
```

`ultralytics` (which pulls in PyTorch) is needed even for the simulation run, since the sim mission node runs a YOLO model for probe detection. `pillow` and `readchar` are only needed for the optional desktop GUI viewer and keyboard teleop, respectively.

## Step-by-step setup

### 1. Install ROS 2 Humble

Follow the official guide: https://docs.ros.org/en/humble/Installation.html

### 2. Install PX4-Autopilot

```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
```

This also installs Gazebo Harmonic. Reboot, or at least log out and back in, so the user-group changes from the setup script take effect.

### 3. Install the Micro XRCE-DDS Agent

This bridges PX4's internal uORB topics to ROS 2.

```bash
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

### 4. Clone this repository as your workspace

Clone it directly as `drone_ws` (don't nest it inside another `src/`, since it already has its own):

```bash
cd ~
git clone https://github.com/Anish-Paul-01/HSM_Aquila.git drone_ws
cd drone_ws/src
git clone https://github.com/PX4/px4_msgs.git
```

### 5. Install the custom PX4 airframe

```bash
cp ~/drone_ws/4900_gz_my_drone ~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/
```

This registers autostart ID 4900 — the motor layout, EKF2 sensor config, and PID gains the simulation launch file starts PX4 with.

### 6. Build PX4 SITL once

```bash
cd ~/PX4-Autopilot
make px4_sitl_default
```

You only need the `px4` binary this produces — the workspace's own launch file starts Gazebo and PX4 itself, so you don't need to launch any simulator from here. If a simulator window pops up anyway once the build finishes, just close it.

### 7. Build the ROS 2 workspace

```bash
cd ~/drone_ws
colcon build --symlink-install
echo "source ~/drone_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### About the Gazebo↔ROS bridge config

No action needed here — `src/drone_bringup/config/gazebo_bridge_oak_px4.yaml` is included in the repo and gets picked up automatically by `colcon build` in step 7. It bridges `/clock`, the OAK-D RGB camera (`/oak/rgb/image_raw` + `camera_info`, used for ArUco detection), the stereo pair and IMU under `/oak/...` (for OpenVINS), `/tf`, and the downward rangefinder on `/drone/lidar_1d/range`.

## Running it

### Simulation

One command brings up everything — Gazebo, PX4 SITL, the XRCE-DDS agent, the ROS↔Gazebo bridges, RViz, and the mission node, staged with timers so each piece comes up in order:

```bash
ros2 launch drone_bringup drone_gazebo_standalone.launch.py
```

Gazebo opens with the `drone_cage` world, the quadrotor spawns a few seconds in, and roughly a minute later — once PX4, the bridges, and TF are all up — the probe-scan-and-land node takes over.

### Real hardware

On the companion computer, wired to the Pixhawk 6C over `/dev/ttyCH341USB0`:

```bash
source ~/drone_ws/install/setup.bash
ros2 launch drone_bringup final_launch.py
```

This starts the XRCE-DDS agent over serial at 921600 baud, brings up the OAK-D driver, and after a 25 s settling delay starts the mission script. Before this will work you also need:

- the `depthai_ros_driver_v3` ROS 2 package built in your workspace (a DepthAI-v3-API camera driver — not included in this repo)
- something publishing ground-probe counts to `/erc/probe_count` (`std_msgs/Int32`) — in this project, a separate MATLAB-based detector
- the `oak_params_file` launch argument pointed at your actual camera-params YAML — it currently defaults to `/home/sar/drone_ws/config/oak_all.yaml`, e.g. `ros2 launch drone_bringup final_launch.py oak_params_file:=/your/path/oak_all.yaml`

### Manual teleop (optional)

A few standalone nodes let you fly the SITL drone by hand, independent of the autonomous mission — useful for sanity-checking the offboard link, TF, and camera feed first:

```bash
ros2 run drone_bringup keyboard_bridge.py     # WASD + QE yaw, arm/land/offboard from the keyboard
ros2 run drone_bringup joystick_bridge.py     # generic joystick via joy_node
ros2 run drone_bringup ps5_joystick_bridge.py # DualSense over Bluetooth
```

## Output

Each simulation run writes confirmed probe detections to `probes_location.csv` (probe ID, timestamp, world-frame X/Y/Z) in whichever directory you launched from.

## Known gaps (read before your first run)

This is an active competition codebase, not a polished release — two things still need attention on a fresh clone:

1. **Hard-coded YOLO weights path.** `drone_vision/aruco_autonomous_land.py` loads its model from `/home/anish1234/drone_ws/src/drone_vision/detection_model/best.pt`. If your username or workspace path differs, edit `self.yolo_model_path` in that file — the weights themselves are already committed at `src/drone_vision/detection_model/best.pt`.
2. **`xacro` isn't declared as a package dependency**, even though the Gazebo launch file imports it — installing `ros-humble-xacro` manually (covered above) works around this rather than relying on `rosdep install`.

No LICENSE file is currently included.
