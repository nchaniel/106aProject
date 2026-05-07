# Lab 5 — Vision-Guided Pick and Place (UR7e)

## Current Objective

**Active goal: wire the 6D pose pipeline's pre-computed positions into the pick-and-place executor and compute per-object drop destinations on the plate.**

The full end-to-end sequence is now:
1. **Arm circler phase**: orbit arm 2 rows × 20 waypoints around the plate, saving photos → `captured_images_2/` and EE poses → `poses.npy`.
2. **Offline processing** (triggered automatically by commander when `/orbit_done` fires):
   - Run YOLO + SAM2 segmentation on all captured images → `segmented/` masks.
   - Run 6D pose estimation (multi-view triangulation) → `results/object_poses.npy` with a `position_base_link_m` per object.
3. Commander displays the detected objects + positions and presents a planned pick order.
4. User presses Enter → pick-and-place activates. The arm executes the planned sequence **using the pre-computed 6D positions** (not live detection) for source positions, and **computed drop destinations** on the plate for placement.
5. Each pick/place uses the same 7-step grasp sequence (hover → descend → grip → lift → move over plate → lower → release), but the source `(cx, cy, cz)` comes from `position_base_link_m` and the drop `(dx, dy, dz)` comes from a pre-computed layout on the plate.

### What is NOT yet implemented
- **Planning algorithm**: compute pick order from 6D pose results (e.g. nearest-first, or by object type priority).
- **Per-object drop positions**: calculate distinct placement spots on the plate for each object (e.g. arranged in a grid/arc around the plate centroid) so objects don't stack on each other.
- **Integration**: commander currently just lets the user type class names; it does not feed `position_base_link_m` from `object_poses.npy` to `main.py` as the pick source. `main.py` still relies on live `/detected_pick_point`. A new mechanism (topic, service, or parameter file) is needed to pass pre-computed positions to the executor.

The plate centroid (from live `/detected_plate_point`) sets the centre of the drop zone; per-object offsets are computed around it.

The `target_class` launch argument (default: empty = highest-confidence detection) selects which class to pick initially when running in manual/live mode. The plate is excluded from pick candidates by a radius filter in `detection_node.py`.

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

ArmCircler (arm_circler.py)
    ├─ /detected_plate_point  (orbit center)
    ├─ /joint_states
    ├─ saves captured_images_2/ + poses.npy during orbit
    └─ /orbit_done ──────────────────────────────────────► Commander

Commander (commander.py) — runs in separate terminal
    ├─ on /orbit_done: subprocess → segment_batch.py (YOLO+SAM2 → segmented/)
    ├─ on /orbit_done: subprocess → 6D_poses/run.py  (triangulation → results/object_poses.npy)
    ├─ displays detected objects + positions
    ├─ [TODO] compute pick order + per-object drop positions from object_poses.npy
    ├─ [TODO] publish pre-computed task list to main.py
    ├─ /start_pick_place ────────────────────────────────► UR7e_CubeGrasp (main.py)
    └─ /set_target_class ────────────────────────────────► DetectionNode

                                       UR7e_CubeGrasp (main.py)
                                              │  (currently: uses live /detected_pick_point)
                                              │  (TODO: consume pre-computed positions)
                                         IKPlanner (ik.py)
                                              ├─ /compute_ik   ──► MoveIt 2
                                              └─ /plan_kinematic_path ──► MoveIt 2
                                                                    │
                                              /scaled_joint_trajectory_controller ──► Robot
                                              /toggle_gripper ──► Gripper
```

**Static TF:** `wrist_3_link → camera_link` is broadcast at startup by `static_tf_transform.py` so the camera depth points can be transformed into `base_link`.

---

---

## 6D Pose Pipeline (Offline)

This runs entirely in the background after the arm circler orbit finishes, before the user activates pick-and-place.

### Data flow

```
poses.npy                    captured_images_2/
  [x,y,z,qx,qy,qz,qw]         captured_image_{1..N}.jpg
        │                              │
        └──────────┬───────────────────┘
                   ▼
         segment_batch.py  (YOLO + SAM2)
                   │
                   ▼ segmented/<stem>/<stem>_<class>_<i>_mask.png
                   │
                   ▼
         6D_poses/run.py  ::  run_pose_estimation()
           │  _pose7_to_ee_T_world()  →  4×4 ee_T_world per view
           │  cam_T_world_from_ee()   →  4×4 cam_T_world  (T_CAM_FROM_EE applied)
           │  PoseEstimator.process_multi_view()
           │    └─ _multi_view_position_only()  (position_only=True, default)
           │         ├─ per view: mask centroid (u,v) + apparent-size depth
           │         │    z = fx × (diameter_m/2) / r_px
           │         └─ RANSAC + weighted DLT triangulation across views
           │              triangulate_ransac() → triangulate_dlt()
           │              → triangulated_position_m  (world/base_link frame)
           │  _add_base_link_poses()  →  position_base_link_m
           │
           ▼
     results/object_poses.npy
       [{object_name, instance_idx, position_base_link_m, confidence}, ...]
```

### Key files — `src/planning/planning/6D_poses/`

| File | Role |
|---|---|
| `run.py` | Entry point. `run_pose_estimation()` loads poses/images/masks, builds `PoseEstimator`, runs multi-view or single-view estimation, saves `object_poses.npy`. |
| `pose_estimator.py` | `PoseEstimator` — full pipeline. `process_multi_view()` → RANSAC triangulation. `_multi_view_position_only()` for position-only mode. Also contains `triangulate_dlt()` and `triangulate_ransac()`. |
| `constants.py` | `OBJECTS` list (`ObjectConfig` entries with `name`, `stl_path`, `color_rgb`, `diameter_m`), `CAMERA_FX/FY`, `T_CAM_FROM_EE` (4×4 camera-from-EE offset), `SEG_NAME_MAP`. |
| `object_config.py` | `ObjectConfig` dataclass — `name`, `stl_path`, `color_rgb`, `diameter_m`, `is_symmetric`. |

### Camera-from-EE transform (`T_CAM_FROM_EE` in `constants.py`)
```
+3 cm forward (+x), -13.5 cm above (-z), same orientation as EE.
```
Used by `cam_T_world_from_ee(ee_T_world)` in `run.py` to convert each arm pose into a camera-frame pose for triangulation.

### `position_only=True` mode (current default)
Skips DINOv2 feature extraction and STL rendering. Depth is estimated from apparent mask size:
```
z = fx × (diameter_m / 2) / r_px
```
where `r_px = sqrt(mask_area / π)`. Then the (x, y) offset from image centre is back-projected. Multi-view positions are triangulated via RANSAC + weighted DLT.

### Output: `results/object_poses.npy`
List of dicts. Key fields:
- `object_name` — YOLO class label
- `position_base_link_m` — `(3,)` float array, object centroid in `base_link` frame (metres)
- `confidence` — fraction of views that detected the object (0–1)
- `instance_idx` — 0-indexed for multiple instances of the same class

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
| `commander.py` | `Commander` | Subscribes to `/orbit_done`. On orbit done: runs segmentation subprocess then 6D pose estimation subprocess, loads `results/object_poses.npy`, prints detected objects + positions. User Enter → publishes `/start_pick_place`. Subsequent inputs → publishes `/set_target_class`. |
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

## Planning Algorithm & Final Destination Positions

### Goal
After 6D pose estimation produces `results/object_poses.npy`, the commander must:
1. **Order the picks** — sort detected objects into a pick sequence (e.g. nearest to home first, or by object class priority).
2. **Compute drop positions** — assign a specific `(dx, dy, dz)` in `base_link` to each object so they land at distinct spots on the plate rather than all dropping on the same point.
3. **Feed positions to `main.py`** — pass pre-computed source positions to the executor so the arm doesn't rely on live detection for objects whose positions are already known.

### Proposed drop position layout
The plate centroid from `/detected_plate_point` is the centre of the drop zone. Individual drop positions are arranged around it, e.g.:
```python
# Simple radial layout — N objects equally spaced on a circle of radius r
angle = 2 * pi * i / N
dx = plate_x + r * cos(angle)
dy = plate_y + r * sin(angle)
dz = plate_z   # same height as plate centroid
```
Radius `r` should be ~0.04–0.06 m (small enough to land on the plate, large enough objects don't collide).

### Proposed pick order
Sort `object_poses.npy` entries by distance from the home joint position projected to the table, or simply by descending confidence. The commander builds an ordered list of `(object_name, position_base_link_m, drop_position)` tuples.

### Integration path (not yet built)
**Option A — new topic**: commander publishes pre-computed pick+drop pairs on a new topic, e.g. `/planned_pick_task` (`geometry_msgs/PoseArray` or custom msg). `main.py` subscribes and queues them instead of waiting for live detection.

**Option B — parameter file**: commander writes a JSON/npy file of ordered tasks; `main.py` reads it at startup.

**Option C — extend `/start_pick_place`**: replace the Bool with a custom action/service that carries the full task list.

The cleanest approach is **Option A**: publish a task list once when the user presses Enter, and have `main.py` consume it in order without needing live YOLO at all for the pick sources.

### Current state
`commander.py` loads `object_poses.npy` and prints it but does **not** yet compute an order or drop destinations, and does **not** pass positions to `main.py`. `main.py` still uses live `/detected_pick_point` and `/detected_plate_point`.

---

## Commander Path & Subprocess Configuration

### Path resolution (`commander.py`)
`_find_project_root()` walks up from `__file__` until it finds a parent containing `armcircler/`. This works from both the source tree and the installed package (which lives 7 levels deep inside `install/`). The old hardcoded "walk up 4 dirs" only worked from source.

Resolved constants:
```
_PROJECT_ROOT = ros_workspaces/                          (contains armcircler/)
_SIXD_DIR     = ros_workspaces/6D_poses/
_SEG_SCRIPT   = ros_workspaces/armcircler/segment_batch.py
_YOLO_WEIGHTS = ros_workspaces/lab5/updated.pt           ← better model, already in lab5/
_SAM2_PYTHON  = ros_workspaces/sam2/sam2_env/bin/python  ← venv with sam2 + trimesh etc.
```

Both segmentation and pose estimation subprocesses use `_SAM2_PYTHON` (not the system `/usr/bin/python3`). The sam2 venv has: `sam2`, `ultralytics`, `torch`, `scipy`, `trimesh`, `pyrender`, `scikit-learn` (trimesh/pyrender/sklearn were installed this session via `pip install` into the venv).

### `skip_circler` + `/orbit_done` QoS fix
When `skip_circler:=true`, the arm circler never runs so `/orbit_done` was never published — commander waited forever. Fix:
- `main.py`: creates a **TRANSIENT_LOCAL** publisher for `/orbit_done` and publishes `Bool(True)` immediately when `skip_circler=True`.
- `arm_circler.py`: changed its `/orbit_done` publisher to **TRANSIENT_LOCAL** (required for QoS compatibility).
- `commander.py`: changed `/orbit_done` subscription to **TRANSIENT_LOCAL** so it receives the retained message even if it starts after `main.py`.

### IMPORTANT: source vs install
`colcon build` copies Python files — it does **not** symlink. Every change to `src/planning/planning/*.py` must also be manually mirrored to `install/planning/lib/python3.10/site-packages/planning/*.py` until the next `colcon build`. Files changed this session: `commander.py`, `main.py`, `arm_circler.py`.

---

## Known Issues / Watch-outs
- Camera launch is **commented out** in `lab5_bringup.launch.py` — must be started separately.
- `drop_pre_joints` and `drop_joints` IK failures are **not checked** before enqueueing (`None` would be passed to `plan_to_joints`). If IK fails for a drop waypoint, the arm will error mid-sequence.
- Gripper state (`self.gripper_open`) is assumed to start open; gets out of sync if the gripper is physically closed at startup.
- If the object moves or is removed after the arm reaches pre-pregrasp, the refined detection will use its new/absent position. The arm will wait indefinitely if no detection arrives after pre-pregrasp.
- Commander must be run in a separate terminal — stdin is not forwarded to nodes launched via `ros2 launch`.
- When `skip_circler:=true`, commander still runs the full segmentation + pose pipeline on whatever images are already in `captured_images/`, then prompts the user to press Enter.
