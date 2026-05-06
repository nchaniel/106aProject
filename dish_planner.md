# Integration Plan: 6d_poses → Lab 5 Pick-and-Place Dish Recreator

## Context

The goal is to use the `6d_poses` package (standalone Python, no ROS) to analyze a **reference dish** of 3D-printed food objects, extract each object's 3D centroid in the camera frame, transform those positions into the robot's `base_link` frame, and then replay the layout onto a new plate whose position is given by an AR tag. The robot should pick matching objects from the workspace and place them in relative positions that match the reference dish, centered on the AR-tagged plate.

---

## Key Facts from Code Inspection

### 6d_poses outputs (per detected object)
- `result["object_name"]` — string label (e.g. `"apple"`, `"strawberry"`)
- `result["position_m"]` — `np.ndarray (3,)` — **3D centroid in camera frame** (metres), available when `diameter_m` is set in `ObjectConfig`
- `result["translation_m"]` — `np.ndarray (3,)` — translation scaled to metres
- `result["diameter_m"]` — float, real-world diameter from config (proxy for size)
- `result["confidence"]` — float [0,1]
- `result["mask"]` — `np.ndarray (H,W)` binary mask (fallback for size if `diameter_m` absent)
- Source frame: **`camera_color_optical_frame`** (same frame the RealSense depth cloud uses)

### Lab 5 existing utilities (reuse these)
- `pixel_to_world.transform_point(tf_buffer, point_stamped, target_frame)` — `lab5/src/perception/perception/pixel_to_world.py:61`
  - Already handles TF2 lookup + apply; reuse directly to convert `position_m` → `base_link`
- `IKPlanner.compute_ik(joint_state, x, y, z, qx, qy, qz, qw)` — `lab5/src/planning/planning/ik.py:47`
- `IKPlanner.plan_to_joints(target_js, start_js)` — `lab5/src/planning/planning/ik.py:92`
- `UR7e_CubeGrasp.execute_jobs()` / `._execute_joint_trajectory()` / `._toggle_gripper()` — `lab5/src/planning/planning/main.py`
- `PICK_OFFSETS` dict — per-class grasp Z offsets — `lab5/src/planning/planning/main.py:25`

### TF frame chain (already live in Lab 5)
```
base_link → ... → wrist_3_link → camera_link → camera_color_optical_frame
```
`static_tf_transform.py` broadcasts `wrist_3_link → camera_link`. The RealSense broadcasts `camera_link → camera_color_optical_frame`. Both are already in the TF tree.

---

## Files to Modify

| File | What Changes |
|---|---|
| `lab5/src/planning/planning/main.py` | Replace single-object reactive loop with dish-recreation state machine |
| `lab5/src/planning/planning/ik.py` | Add `plan_to_pose()` convenience wrapper (optional but useful) |
| `lab5/src/planning/launch/lab5_bringup.launch.py` | Add AR tag node + dish_runner node to launch |

## New Files to Create

| File | Purpose |
|---|---|
| `lab5/src/planning/planning/dish_planner.py` | Core data structures + layout math |
| `lab5/src/planning/planning/dish_runner.py` | New ROS node: orchestrates the full dish-recreation sequence |
| `lab5/src/planning/planning/ar_tag_detector.py` | Thin ROS node: listens to AR tag TF, publishes plate pose |

---

## Step-by-Step Implementation Plan

---

### Step 1 — Data Representation (`dish_planner.py`)

Create `lab5/src/planning/planning/dish_planner.py`.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DishObject:
    name: str               # e.g. "apple"
    instance_idx: int       # 0-based (for multi-instance)
    position_base: np.ndarray   # shape (3,) in base_link frame, metres
    diameter_m: float           # real-world diameter; used for size sort
    layer: int = 0              # assigned during planning (0=bottom)
    relative_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # relative_offset = position_base minus dish centroid (XY only; Z kept absolute)
```

**Why this structure:** keeps raw base_link position separate from the computed relative offset so the planner can recompute the offset when the dish centroid changes.

---

### Step 2 — Transform 6d_poses Output to base_link

In `dish_planner.py`, add:

```python
from geometry_msgs.msg import PointStamped

def pose_results_to_base_link(results: list[dict], tf_buffer, source_frame="camera_color_optical_frame") -> list[DishObject]:
    """
    Convert 6d_poses process_scene() output list into DishObject list
    with positions expressed in base_link.
    Reuses transform_point() from pixel_to_world.py.
    """
    from perception.pixel_to_world import transform_point   # existing utility

    dish_objects = []
    for r in results:
        if "error" in r:
            continue
        pos = r.get("position_m")
        if pos is None:
            continue

        ps = PointStamped()
        ps.header.frame_id = source_frame
        ps.point.x, ps.point.y, ps.point.z = float(pos[0]), float(pos[1]), float(pos[2])

        ps_base = transform_point(tf_buffer, ps, "base_link")
        if ps_base is None:
            continue

        dish_objects.append(DishObject(
            name=r["object_name"],
            instance_idx=r["instance_idx"],
            position_base=np.array([ps_base.point.x, ps_base.point.y, ps_base.point.z]),
            diameter_m=r.get("diameter_m") or _mask_diameter(r["mask"]),
        ))
    return dish_objects


def _mask_diameter(mask) -> float:
    """Fallback size estimate from mask area when diameter_m is absent."""
    area_px = float((mask > 0).sum())
    return max(0.01, np.sqrt(area_px) * 0.001)   # rough metres proxy
```

**Reuses:** `transform_point` from `lab5/src/perception/perception/pixel_to_world.py:61` — no new TF code needed.

---

### Step 3 — Layer Assignment

In `dish_planner.py`, add:

```python
def assign_layers(objects: list[DishObject], z_bin_size: float = 0.015) -> list[DishObject]:
    """
    Bin objects into layers by their Z coordinate in base_link.
    Lower Z → lower layer index (place first).
    z_bin_size: height tolerance to consider two objects on the same layer (metres).
    """
    if not objects:
        return objects
    zs = np.array([o.position_base[2] for o in objects])
    z_min = zs.min()
    for o in objects:
        o.layer = int((o.position_base[2] - z_min) / z_bin_size)
    return objects
```

---

### Step 4 — Placement Order Planner

In `dish_planner.py`, add:

```python
def sort_placement_order(objects: list[DishObject]) -> list[DishObject]:
    """
    Primary sort: layer ascending (bottom first).
    Secondary sort within same layer: diameter_m ascending (smaller first).
    """
    return sorted(objects, key=lambda o: (o.layer, o.diameter_m))
```

---

### Step 5 — Compute Relative Layout

In `dish_planner.py`, add:

```python
def compute_relative_layout(objects: list[DishObject]) -> tuple[np.ndarray, list[DishObject]]:
    """
    Compute each object's XY offset from the dish centroid.
    Z offset is kept absolute (relative to lowest object).
    Returns: (dish_centroid_base, objects with relative_offset filled in)
    """
    positions = np.array([o.position_base for o in objects])
    centroid = positions.mean(axis=0)
    z_base = positions[:, 2].min()

    for o in objects:
        dx = o.position_base[0] - centroid[0]
        dy = o.position_base[1] - centroid[1]
        dz = o.position_base[2] - z_base   # height above lowest object
        o.relative_offset = np.array([dx, dy, dz])

    return centroid, objects
```

---

### Step 6 — Convert to Target Poses Using AR Tag

In `dish_planner.py`, add:

```python
def get_target_poses(objects: list[DishObject], plate_position_base: np.ndarray) -> list[tuple[DishObject, np.ndarray]]:
    """
    Translate relative offsets so the dish is centered on the AR-tagged plate.
    plate_position_base: (3,) XYZ of the plate center in base_link.
    Returns list of (DishObject, target_xyz_in_base_link).
    """
    targets = []
    for o in objects:
        target = np.array([
            plate_position_base[0] + o.relative_offset[0],
            plate_position_base[1] + o.relative_offset[1],
            plate_position_base[2] + o.relative_offset[2],
        ])
        targets.append((o, target))
    return targets
```

---

### Step 7 — AR Tag Detector Node (`ar_tag_detector.py`)

Create `lab5/src/planning/planning/ar_tag_detector.py`.

**Dependency:** Add `apriltag_ros` (ROS 2) to the launch. It publishes TF frames named `tag_<id>` (or similar, depending on config).

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import tf2_ros

class ARTagDetector(Node):
    """
    Polls TF for the AR plate tag and publishes its position in base_link
    on /plate_position as PointStamped.
    """
    TAG_FRAME = "tag_0"   # apriltag_ros default for tag ID 0

    def __init__(self):
        super().__init__("ar_tag_detector")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._pub = self.create_publisher(PointStamped, "/plate_position", 10)
        self.create_timer(0.5, self._poll)

    def _poll(self):
        try:
            t = self._tf_buffer.lookup_transform("base_link", self.TAG_FRAME, rclpy.time.Time())
            ps = PointStamped()
            ps.header.frame_id = "base_link"
            ps.point.x = t.transform.translation.x
            ps.point.y = t.transform.translation.y
            ps.point.z = t.transform.translation.z
            self._pub.publish(ps)
        except Exception:
            pass   # tag not visible yet; silent until detected
```

**Topics:**
- Subscribes: TF tree (implicit via tf2)
- Publishes: `/plate_position` (`PointStamped`, frame=`base_link`)

---

### Step 8 — Dish Runner Node (`dish_runner.py`)

Create `lab5/src/planning/planning/dish_runner.py`. This is the new top-level orchestrator, replacing `main.py`'s reactive single-object loop with a planned multi-object sequence.

**Subscriptions:**
- `/detected_pick_point` (`PointStamped`) — workspace object position (existing)
- `/detected_class` (`String`) — workspace object class (existing)
- `/joint_states` (`JointState`) — robot state (existing)
- `/plate_position` (`PointStamped`) — AR tag plate pose (new)

**State machine phases:**
```
INIT → SCAN_REFERENCE → PLAN → WAIT_FOR_PLATE → EXECUTE(loop) → DONE
```

**Pseudocode for core execute loop:**

```python
def _run_placement_loop(self):
    """
    Called once after plan is complete and plate position is known.
    Iterates over sorted placement targets and executes pick-then-place for each.
    """
    targets = get_target_poses(self._sorted_objects, self._plate_position)

    for dish_obj, target_xyz in targets:
        # 1. Ask YOLO to find a matching object in the workspace
        self._request_detection(dish_obj.name)   # sets target_class param
        workspace_pos = self._wait_for_detection(dish_obj.name, timeout=10.0)
        if workspace_pos is None:
            self.get_logger().warn(f"Could not find {dish_obj.name}, skipping")
            continue

        # 2. Pick from workspace (reuse existing pick sequence from main.py)
        self._execute_pick(workspace_pos, dish_obj.name)

        # 3. Place at target pose
        self._execute_place(target_xyz, dish_obj.name)
```

**Pick sequence** (`_execute_pick`): copy the existing job-queue logic from `UR7e_CubeGrasp.cube_callback()` (`main.py:119–222`). Reuse `PICK_OFFSETS` dict as-is.

**Place sequence** (`_execute_place`): analogous to pick but without gripper close:
1. Pre-place hover: IK to `(tx, ty, tz + pre_grasp_z_offset)`
2. Descend: IK to `(tx, ty, tz + grasp_z_offset)`
3. Gripper open (release)
4. Retreat: IK to `(tx, ty, tz + lift_z_offset)`

**Important:** Use the same `IKPlanner.compute_ik()` + `IKPlanner.plan_to_joints()` + `_execute_joint_trajectory()` pattern already in `main.py`. Do not add a new motion interface.

---

### Step 9 — Modifications to `main.py`

`lab5/src/planning/planning/main.py` needs minimal changes:

1. **Extract `PICK_OFFSETS`** to a shared constants file (or `dish_planner.py`) so `dish_runner.py` can import it without re-defining.
2. **Add a `place_sequence()` method** (or move the job-queue pick logic into a helper) so `dish_runner.py` can call it directly rather than duplicating code.
3. **Option:** Keep `main.py` for standalone single-object demos and run `dish_runner.py` as a separate node for dish recreation. The launch file decides which to start.

---

### Step 10 — Launch File Changes (`lab5_bringup.launch.py`)

Add to `generate_launch_description()`:

```python
# AR tag detection (apriltag_ros)
Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
     parameters=[{"tag_family": "tag36h11", "tag_ids": [0]}]),

# AR tag → /plate_position bridge
Node(package="planning", executable="ar_tag", name="ar_tag_detector"),

# Dish recreation orchestrator
Node(package="planning", executable="dish_runner", name="dish_runner"),
```

And register `dish_runner` and `ar_tag` as entry points in `setup.py`.

---

## Data Flow Summary

```
6d_poses.run_pose_estimation(rgb_image)
    → list[dict]  (position_m in camera_color_optical_frame)
    ↓
pose_results_to_base_link(results, tf_buffer)
    → list[DishObject]  (position_base in base_link)
    ↓
assign_layers(objects)         ← bins by Z height
sort_placement_order(objects)  ← bottom-up, small-before-large
compute_relative_layout(objects)  ← XY offset from centroid, Z from floor
    ↓
AR tag TF lookup → plate_position in base_link
    ↓
get_target_poses(objects, plate_position)
    → [(DishObject, target_xyz)]  ordered by placement priority
    ↓
For each (dish_obj, target_xyz):
    YOLO detects matching dish_obj.name in workspace
    → workspace_pos in base_link
    pick(workspace_pos)   ← existing job-queue logic
    place(target_xyz)     ← new, symmetric to pick
```

---

## ROS Topics & Message Types

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/detected_pick_point` | `PointStamped` | sub | Workspace object centroid (existing) |
| `/detected_class` | `String` | sub | Workspace object class (existing) |
| `/joint_states` | `JointState` | sub | Robot state (existing) |
| `/plate_position` | `PointStamped` | sub | AR tag plate center (new) |
| `/toggle_gripper` | `Trigger` srv | call | Gripper (existing) |

TF frames used: `base_link`, `wrist_3_link`, `camera_link`, `camera_color_optical_frame`, `tag_0`

---

## Minimal V1 Implementation

Do these in order — each adds one capability:

1. **V1a (transform only):** Run `6d_poses` offline on a saved image. Call `pose_results_to_base_link()`. Print the `DishObject` list. Verify positions look correct relative to the robot.

2. **V1b (planner only):** Call `assign_layers`, `sort_placement_order`, `compute_relative_layout`. Print sorted placement order with relative offsets. No robot motion yet.

3. **V1c (place one object):** Hardcode a plate position. Pick one object from workspace (existing code). Call `_execute_place()` at `plate_position + relative_offset` for that object. Verify it lands in roughly the right spot.

4. **V1d (AR tag):** Add `ar_tag_detector.py`. Replace hardcoded plate position with live `/plate_position` topic. Re-run V1c.

5. **V1e (full loop):** Loop over all objects in sorted order. Run the full pick-then-place sequence for each.

---

## Verification

- **Unit test the planner:** Call `assign_layers` + `sort_placement_order` + `compute_relative_layout` on mock data; verify `relative_offset` sums to near-zero mean (centroid is at origin of offsets).
- **TF sanity check:** After transform, `position_base.z` should be ~0.05–0.25 m (table height). Negative or >0.5 m indicates wrong frame.
- **Pick test:** Run existing Lab 5 pick on a single object; confirm unchanged behavior.
- **Place test:** Manually drive robot to `target_xyz` with `compute_ik()`; verify it hovers at correct height above plate.
- **Full integration:** Run dish recreation with 3 objects; measure XY distances between placed objects and compare to reference dish layout.
