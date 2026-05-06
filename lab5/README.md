# Lab 5 — Vision-Guided Pick and Place with UR7e

Closed-loop pick-and-place on a Universal Robots UR7e using a wrist-mounted Intel RealSense camera. A fine-tuned YOLOv8 model detects objects in the RGB stream, their 3D centroids are extracted from the depth point cloud, and MoveIt 2 plans and executes a grasp trajectory.

---

## System Architecture

```
RealSense Camera
    │
    ├─ /camera/.../color/image_raw       ──► DetectionNode
    ├─ /camera/.../depth/color/points    ──► DetectionNode
    └─ /camera/.../color/camera_info     ──► DetectionNode
                                               │
                              /detected_pick_point (PointStamped, base_link)
                              /detected_class    (String)
                                               │
                                         UR7e_CubeGrasp
                                               │
                                         IKPlanner ──► MoveIt 2 ──► Robot
```

**Static TF:** `wrist_3_link → camera_link` is broadcast at startup so the entire TF tree is connected (`base_link → ... → wrist_3_link → camera_link → camera optical frames`).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| ROS 2 (Humble or later) | Tested on Ubuntu 22.04 |
| `ur_robot_driver` + `ur_moveit_config` | For UR7e bringup and MoveIt 2 |
| `realsense2_camera` | RealSense ROS 2 wrapper |
| `ultralytics` | YOLOv8 inference (`pip install ultralytics`) |
| `cv_bridge`, `tf2_ros`, `tf2_geometry_msgs` | ROS image/TF utilities |
| `scipy` | Quaternion conversion in the TF broadcaster |
| `best.pt` | Fine-tuned YOLO weights (place at repo root or set `model_path` param) |

---

## Quickstart

### 1. Build

```bash
cd <workspace_root>
colcon build --packages-select perception planning
source install/setup.bash
```

### 2. Start the RealSense camera (separate terminal)

```bash
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=true \
    rgb_camera.color_profile:=1280x720x30
```

> The camera is launched separately because MoveIt 2 initialisation takes longer; starting it first avoids TF timing races.

### 3. Launch everything else

```bash
ros2 launch planning lab5_bringup.launch.py \
    robot_ip:=192.168.1.102 \
    target_class:=apple
```

| Argument | Default | Description |
|---|---|---|
| `robot_ip` | `192.168.1.102` | IP of the UR7e |
| `ur_type` | `ur7e` | Robot model passed to `ur_moveit_config` |
| `launch_rviz` | `true` | Open RViz with MoveIt 2 |
| `target_class` | *(empty)* | YOLO class to pick; empty = highest-confidence detection |
| `shutdown_on_exit` | `true` | Kill the whole launch if any node exits |

### 4. Watch it go

- The detection window (`YOLO Detections`) pops up showing bounding boxes.
- Once a pick point is published the arm moves through: **home → hover → descend → grip → lift → release**.
- Set `show_image:=false` on the `detection_node` if running headless.

---

## Outputs

| Topic | Type | Description |
|---|---|---|
| `/detected_pick_point` | `geometry_msgs/PointStamped` | 3D centroid of the best detection in `base_link` frame |
| `/detected_class` | `std_msgs/String` | Class label of the selected detection |

The arm trajectory is sent directly to `/scaled_joint_trajectory_controller/follow_joint_trajectory` (FollowJointTrajectory action). Gripper open/close goes through the `/toggle_gripper` service (Trigger).

---

## Package & File Reference

### `perception` package

#### [`detection_node.py`](src/perception/perception/detection_node.py) — `DetectionNode`

Main ROS 2 node that ties perception together.

- Subscribes to the RGB image, depth point cloud, and camera info.
- Runs YOLO on every RGB frame via `YOLODetector`.
- For each detection that passes the class filter, calls `get_centroid_from_cloud_bbox` to get a 3D point in camera frame, then `transform_point` to convert it to `base_link`.
- Publishes the highest-confidence valid detection as `/detected_pick_point` and `/detected_class`.
- Displays an annotated live window when `show_image:=true`.
- Includes a 5-second watchdog that warns if no images or point clouds are arriving.

**Key parameters:** `model_path`, `conf_threshold`, `target_class`, `target_frame`, `show_image`.

---

#### [`yolo_detector.py`](src/perception/perception/yolo_detector.py) — `YOLODetector`

Stateless wrapper around an Ultralytics YOLO model.

| Method | What it does |
|---|---|
| `__init__(model_path, conf_threshold)` | Loads YOLO weights from `model_path`. |
| `detect(image)` | Runs inference on a BGR OpenCV image; returns a list of dicts with `class_id`, `class_name`, `confidence`, `bbox [x1,y1,x2,y2]`, and `center [cx,cy]`. |
| `draw_detections(image, detections)` | Draws green bounding boxes and red centroid dots on a copy of the image. |

Can be run standalone (`python yolo_detector.py`) to test on images in `test_images/`.

---

#### [`pixel_to_world.py`](src/perception/perception/pixel_to_world.py)

Two pure functions, no ROS node.

| Function | What it does |
|---|---|
| `get_centroid_from_cloud_bbox(cloud_msg, camera_info_msg, bbox)` | Reads all XYZ points from a `PointCloud2`, reprojects them into image coordinates using the camera intrinsics, keeps only those that fall inside the YOLO bounding box, and returns their mean as a `PointStamped` in the cloud's frame. Works with both organized and unorganized clouds. Returns `None` if no valid depth points exist in the box. |
| `transform_point(tf_buffer, point_stamped, target_frame)` | Looks up the TF transform and converts a `PointStamped` to `target_frame` (typically `base_link`). |

---

### `planning` package

#### [`static_tf_transform.py`](src/planning/planning/static_tf_transform.py) — `ConstantTransformPublisher`

Broadcasts a single static TF from `wrist_3_link` → `camera_link` at startup using a hand-measured 4×4 homogeneous transform matrix `G`. This is what connects the camera TF sub-tree to the robot's kinematic chain so depth measurements can be expressed in `base_link`.

The rotation matrix in `G` is converted to a quaternion via `scipy.spatial.transform.Rotation`.

---

#### [`ik.py`](src/planning/planning/ik.py) — `IKPlanner`

ROS 2 node that wraps the MoveIt 2 IK and motion-planning services.

| Method | What it does |
|---|---|
| `compute_ik(current_joint_state, x, y, z, qx, qy, qz, qw)` | Calls `/compute_ik` to find joint angles that put `tool0` at `(x, y, z)` with the given orientation (default: gripper pointing straight down, `qy=1`). Returns a `JointState` or `None` on failure. |
| `plan_to_joints(target_joint_state, start_joint_state)` | Calls `/plan_kinematic_path` (RRTConnect) to generate a collision-free trajectory from the current state to the target joint configuration. Returns the trajectory or `None`. |

---

#### [`main.py`](src/planning/planning/main.py) — `UR7e_CubeGrasp`

Top-level pick-and-place state machine.

**Pick sequence (queued as jobs):**

1. Move to **home** joint pose on first joint state received.
2. On each `/detected_pick_point` message (when not busy):
   - Compute IK for **pre-grasp hover** (`z + pre_grasp_z_offset`)
   - Compute IK for **grasp descent** (`z + grasp_z_offset`)
   - **Toggle gripper** (close)
   - Compute IK for **lift** (`z + lift_z_offset`)
   - **Toggle gripper** (open) — drop

Per-class grasp offsets are defined in the `PICK_OFFSETS` dict at the top of the file. Add a new entry to tune offsets for a new object class without touching any other logic.

| Method | What it does |
|---|---|
| `cube_callback` | Receives detected 3D point, builds the job queue, kicks off execution. |
| `execute_jobs` | Pops and runs one job at a time; each trajectory result triggers the next via async callbacks. |
| `_toggle_gripper` | Calls the `/toggle_gripper` Trigger service synchronously (2 s timeout). |
| `_execute_joint_trajectory` | Sends a `FollowJointTrajectory` goal to the scaled joint trajectory controller. |

---

#### [`lab5_bringup.launch.py`](src/planning/launch/lab5_bringup.launch.py)

Launches in order:

1. `perception/detection_node`
2. `planning/tf` (static camera TF)
3. `ur_moveit_config/ur_moveit.launch.py` (MoveIt 2 + RViz)
4. `planning/ik` (IKPlanner service node)
5. `planning/main` (UR7e_CubeGrasp pick-and-place node)

---

## Tuning Grasp Offsets

Edit `PICK_OFFSETS` in [main.py](src/planning/planning/main.py):

```python
PICK_OFFSETS = {
    "apple": {
        "x_offset":           0.02,   # lateral correction (m)
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,   # hover height above centroid (m)
        "grasp_z_offset":     0.12,   # descent height above centroid (m)
        "lift_z_offset":      0.185,  # lift height after grip (m)
    },
    # add more classes here ...
}
```

Objects not listed fall back to `DEFAULT_OFFSETS`.
