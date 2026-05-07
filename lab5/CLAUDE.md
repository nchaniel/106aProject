# Lab 5 — Vision-Guided Pick and Place (UR7e)

## Current Objective
Pick up a **user-specified object class** (e.g. "apple") from the table using YOLO detection and place it onto a **detected plate**. The plate is detected in the same YOLO pipeline and its 3D centroid is published separately. The full run sequence is:
1. **Arm circler phase** (optional): orbit the arm around the plate centroid for a 2-pass inspection (40 waypoints, photos saved). Skip with `skip_circler:=true`.
2. **Mode switch**: user presses Enter in the commander terminal to activate pick-and-place.
3. Move to a **pre-pregrasp position** directly above the initial centroid (`cz + 0.5 m`) so the camera gets an undistorted centered view.
4. Recalculate the object centroid from this better vantage point.
5. Execute a 7-step grasp sequence using the refined centroid (hover → descend → grip → lift → move over plate → lower → release).
6. Return to home automatically and wait for the next pick command.
7. **Between picks**: type the next class name in the commander terminal to update the target.

The `target_class` launch argument (default: empty = highest-confidence detection) selects which class to pick initially. The plate is excluded from pick candidates by a radius filter in `detection_node.py`.

---

## System Architecture

```
RealSense Camera (started separately)
    ├─ /camera/.../color/image_raw       ──► DetectionNode
    ├─ /camera/.../depth/color/points    ──► DetectionNode
    └─ /camera/.../color/camera_info     ──► DetectionNode
                                              │
                  /detected_class (String)    │
                  /detected_pick_point ───────┤
                  /detected_plate_point ──────┤
                                              ▼
                  /set_target_class ─────► DetectionNode (runtime class switch)

Commander (commander.py) ──/start_pick_place──► UR7e_CubeGrasp (main.py)
    (separate terminal)   ──/set_target_class──► DetectionNode

ArmCircler (arm_circler.py)
    ├─ /detected_plate_point  (orbit center)
    └─ /joint_states

                                       UR7e_CubeGrasp (main.py)
                                              │
                                         IKPlanner (ik.py)
                                              ├─ /compute_ik   ──► MoveIt 2
                                              └─ /plan_kinematic_path ──► MoveIt 2
                                                                    │
                                              /scaled_joint_trajectory_controller ──► Robot
                                              /toggle_gripper ──► Gripper
```

**Static TF:** `wrist_3_link → camera_link` is broadcast at startup by `static_tf_transform.py` so the camera depth points can be transformed into `base_link`.

---

## Package & File Map

### `perception` package — `src/perception/perception/`

| File | Class/Role | Key detail |
|---|---|---|
| `detection_node.py` | `DetectionNode` | Main perception node. Runs YOLO on each RGB frame, extracts 3D centroid from depth cloud, transforms to `base_link`, publishes pick point + class. Has a plate-radius exclusion filter (0.14 m). Subscribes to `/set_target_class` to switch target at runtime. |
| `yolo_detector.py` | `YOLODetector` | Stateless Ultralytics YOLO wrapper. `detect(image)` returns list of `{class_id, class_name, confidence, bbox, center}`. |
| `pixel_to_world.py` | (functions) | `get_centroid_from_cloud_bbox()` — reprojects pointcloud into image coords, averages points inside bbox. `transform_point()` — TF lookup to convert camera frame → `base_link`. |

**Key parameters on `detection_node`:** `model_path`, `conf_threshold`, `target_class`, `target_frame`, `show_image`.

### `planning` package — `src/planning/planning/`

| File | Class/Role | Key detail |
|---|---|---|
| `main.py` | `UR7e_CubeGrasp` | Top-level state machine. Owns `busy` flag, `job_queue`, and all pick/place logic. Starts locked (`pick_place_enabled=False`) until `/start_pick_place` is received (or `skip_circler:=true`). |
| `arm_circler.py` | `ArmCircler` | Pre-inspection node. Orbits arm around plate centroid (2 rows × 20 waypoints, look-at orientation). Takes a photo at each waypoint, saves `poses.npy`. Sets `_orbit_done=True` when complete. Launched by bringup unless `skip_circler:=true`. |
| `commander.py` | `Commander` | Interactive controller run in a **separate terminal**. Blocking `input()` in main thread, ROS spin in background thread. First Enter → publishes `/start_pick_place`. Subsequent inputs → publishes `/set_target_class`. |
| `ik.py` | `IKPlanner` | Wraps `/compute_ik` (MoveIt IK) and `/plan_kinematic_path` (RRTConnect). Default orientation `qy=1` = gripper pointing straight down. |
| `static_tf_transform.py` | `ConstantTransformPublisher` | Broadcasts `wrist_3_link → camera_link` via a hard-coded 4×4 matrix `G`. |

**Launch file:** `src/planning/launch/lab5_bringup.launch.py`

---

## Key Design Patterns

### 1. `busy` semaphore
`self.busy = True` is set as soon as `cube_callback` accepts a detection. While True, all further detection callbacks return immediately. Cleared only after the arm returns home.

### 2. Job queue
`self.job_queue` is a list of either `JointState` (move to joints) or `'toggle_grip'` (service call). Jobs are executed one at a time; each async completion callback calls `execute_jobs()` to pop the next job.

### 3. IK chaining
Each `compute_ik()` call uses the *previous step's joint solution* as the seed, not the live arm pose. This keeps the solver in a locally consistent region and prevents elbow flips between adjacent waypoints.

### 4. Two-phase centroid refinement
Objects at the edge of the camera's field of view produce distorted depth centroids. To correct this:
- **Phase 1** (`cube_callback`): compute IK for `(cx, cy, cz + 0.5)` — directly above the rough centroid, no offsets. Queue only that one waypoint and set `_refining = True`.
- **Phase 2** (`_on_refined_detection`): once the arm arrives (`_at_pre_pregrasp = True`), the next detection fires with a better-centered camera view. That centroid is used with full offsets to compute and queue the remaining 5 IK steps.

Offsets in `PICK_OFFSETS` are applied **only to the refined centroid** from phase 2, not the initial detection.

### 5. `_refining` / `_at_pre_pregrasp` sentinels
`execute_jobs()` now has three empty-queue states:
- `_refining=True, _at_pre_pregrasp=False` → still moving to pre-pregrasp; detections ignored
- `_refining=True, _at_pre_pregrasp=True` → arm arrived; next `cube_callback` fires `_on_refined_detection`
- `_going_home=False` → pick/place done; chain into `_go_home()`
- `_going_home=True` → home reached; clear `busy`

### 6. Async trajectory execution
`_execute_joint_trajectory` → `_on_goal_sent` → `_on_exec_done` → `execute_jobs()`. Non-blocking — the ROS spin loop is never stalled waiting for a trajectory.

### 7. `_going_home` sentinel
When the job queue empties, `execute_jobs()` checks `_going_home`:
- `False` → pick/place just finished; automatically chain into `_go_home()`
- `True` → home reached; clear `busy` and wait for next detection

### 8. Topic ordering race condition
`/detected_class` **must** arrive before `/detected_pick_point` or the per-class offsets in `PICK_OFFSETS` silently fall back to `DEFAULT_OFFSETS`.

### 9. Plate always published regardless of `target_class`
In `detection_node.py`, the `target_class` filter explicitly exempts `"plate"` so the plate centroid is always published on `/detected_plate_point` even when a specific object class is targeted.

### 10. `pick_place_enabled` gate
`UR7e_CubeGrasp.cube_callback` returns immediately if `self.pick_place_enabled` is False. Set to True either by receiving `/start_pick_place Bool(True)` (from commander) or at startup when `skip_circler:=true` is passed as a launch argument.

### 11. Arm circler orbit
`ArmCircler` generates 2 rows × 20 waypoints around the plate centroid using look-at quaternions (`rot_z * rot_y * rot_x`). Row 1 at `height=0.3 m`, row 2 reversed at `height=0.2 m`. Orbit is triggered once on the first `/detected_plate_point`; subsequent messages are ignored (`_orbit_triggered` guard).

---

## ROS Topics & Services

| Topic / Service | Type | Direction | Purpose |
|---|---|---|---|
| `/detected_pick_point` | `PointStamped` | perception → planning | 3D centroid of target object in `base_link` |
| `/detected_class` | `String` | perception → planning | YOLO class label of the target object |
| `/detected_plate_point` | `PointStamped` | perception → planning/circler | 3D centroid of the plate in `base_link` |
| `/set_target_class` | `String` | commander → perception | Switches YOLO target class at runtime |
| `/start_pick_place` | `Bool` | commander → planning | Activates pick-and-place mode after orbit |
| `/joint_states` | `JointState` | robot → planning | Current arm configuration; seeds IK |
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Action | planning → robot | Executes planned joint trajectory |
| `/toggle_gripper` | `Trigger` service | planning → gripper | Opens/closes gripper (toggle) |
| `/compute_ik` | `GetPositionIK` service | planning → MoveIt | Computes joint angles for Cartesian target |
| `/plan_kinematic_path` | `GetMotionPlan` service | planning → MoveIt | Plans collision-free trajectory |

---

## Per-Class Grasp Offsets (`main.py`)

```python
PICK_OFFSETS = {
    "apple":      { "x_offset": 0.01,  "y_offset": 0.005, "pre_grasp_z_offset": 0.16, "grasp_z_offset": 0.125, "lift_z_offset": 0.185 },
    "tomato":     { "x_offset": 0.015, "y_offset": 0.005, "pre_grasp_z_offset": 0.16, "grasp_z_offset": 0.14,  "lift_z_offset": 0.185 },
    "cake":       { "x_offset": 0.02,  "y_offset": 0.005, "pre_grasp_z_offset": 0.20, "grasp_z_offset": 0.15,  "lift_z_offset": 0.20  },
    "strawberry": { "x_offset": 0.02,  "y_offset": 0.005, "pre_grasp_z_offset": 0.20, "grasp_z_offset": 0.14,  "lift_z_offset": 0.20  },
    "cherry":     { "x_offset": 0.02,  "y_offset": 0.005, "pre_grasp_z_offset": 0.20, "grasp_z_offset": 0.14,  "lift_z_offset": 0.20  },
    "grape":      { "x_offset": 0.02,  "y_offset": 0.005, "pre_grasp_z_offset": 0.20, "grasp_z_offset": 0.14,  "lift_z_offset": 0.20  },
    "blueberry":  { "x_offset": 0.02,  "y_offset": 0.005, "pre_grasp_z_offset": 0.20, "grasp_z_offset": 0.14,  "lift_z_offset": 0.20  },
}
DEFAULT_OFFSETS = { "x_offset": 0.02, "y_offset": 0.005, "pre_grasp_z_offset": 0.16, "grasp_z_offset": 0.14, "lift_z_offset": 0.185 }
```

All offsets are applied to the **refined centroid** — the one captured after the arm moves to pre-pregrasp above the object. Tune `grasp_z_offset` if the gripper misses (too high) or hits the table (too low).

Drop pose: `drop_z + 0.2` for clearance hover, `drop_z + 0.15` for the release point.

---

## Home Pose (joint angles, radians)

```python
[4.723, -1.576, -2.161, -0.993, 1.580, -3.143]
# order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
```

To update: move arm to desired pose manually, run `ros2 topic echo /joint_states --once`, copy the 6 values into `_home_joints` in `main.py`.

---

## Build & Run

```bash
# Build (from lab5/ root)
colcon build --packages-select perception planning
source install/setup.bash

# Terminal 1: RealSense camera (always start first)
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=true \
    rgb_camera.color_profile:=1280x720x30

# Terminal 2: Bringup (with arm circler)
ros2 launch planning lab5_bringup.launch.py \
    robot_ip:=192.168.1.102 \
    target_class:=apple

# Terminal 2 (alt): Skip arm circler, go straight to pick-and-place
ros2 launch planning lab5_bringup.launch.py \
    robot_ip:=192.168.1.102 \
    target_class:=apple \
    skip_circler:=true

# Terminal 3: Commander (required for mode switch and class changes)
ros2 run planning commander
```

**Commander workflow:**
1. Wait for orbit to finish, then press Enter → pick-and-place activates
2. Type a class name (e.g. `cake`) + Enter → detection node switches target immediately
3. Repeat after each pick cycle

YOLO model weights: `best.pt` (fine-tuned), `updated.pt`, `yolov8n.pt` — all at the repo root. `model_path` parameter selects which one.

---

## Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `robot_ip` | `192.168.1.102` | UR7e IP address |
| `target_class` | `""` | Initial YOLO target class (empty = highest confidence) |
| `skip_circler` | `false` | Skip arm circler and enable pick-and-place immediately |
| `launch_rviz` | `true` | Launch RViz with MoveIt |
| `shutdown_on_exit` | `true` | Kill all nodes if any process exits |

---

## Known Issues / Watch-outs
- Camera launch is **commented out** in `lab5_bringup.launch.py` — must be started separately.
- `drop_pre_joints` and `drop_joints` IK failures are **not checked** before enqueueing (`None` would be passed to `plan_to_joints`). If IK fails for a drop waypoint, the arm will error mid-sequence.
- Gripper state (`self.gripper_open`) is assumed to start open; gets out of sync if the gripper is physically closed at startup.
- If the object moves or is removed after the arm reaches pre-pregrasp, the refined detection will use its new/absent position. The arm will wait indefinitely if no detection arrives after pre-pregrasp.
- Commander must be run in a separate terminal — stdin is not forwarded to nodes launched via `ros2 launch`.
- When `skip_circler:=true`, the commander is still useful for switching target class between pick cycles, but the initial Enter press is not needed.
