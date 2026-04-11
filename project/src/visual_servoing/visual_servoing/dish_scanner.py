#!/usr/bin/env python3
"""
dish_scanner.py — Plating robot dish scanner
=============================================
Orbits the UR7e around a plated dish and captures images for 3D reconstruction.

Builds directly on the Lab 7 Checkpoint 1 pipeline:
  • AR tag detection via TF (same lookup as main.py)
  • ScanOrbitTrajectory for the orbital path
  • UR7e trajectory controller for execution (same as default controller)
  • Image capture triggered at angularly-uniform waypoints

Usage
-----
    ros2 run visual_servoing dish_scanner \\
        --ar_marker 0 \\
        --radius 0.18 \\
        --height 0.28 \\
        --total_time 20.0 \\
        --n_images 36 \\
        --output_dir ~/scan_output

Then feed the saved images to COLMAP or Meshroom for 3D reconstruction.
See the bottom of this file for a ready-to-run COLMAP command.

Architecture
------------
DishScanner (Node)
  ├── TF listener          — AR tag lookup (same as VisualServo)
  ├── MoveIt IK client     — Cartesian → joint space
  ├── UR7eTrajectoryController — joint trajectory execution
  ├── CameraCapture        — saves images at computed capture times
  └── ScanOrbitTrajectory  — generates the orbital path

Author: adapted from EECS C106A Lab 7 starter code, Spring 2026
"""

import os
import sys
import time
import select
import threading
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

import cv2
from cv_bridge import CvBridge

# Import from the visual_servoing package
from .trajectories import LinearTrajectory, ScanOrbitTrajectory
from .controller import UR7eTrajectoryController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

CAMERA_TOPIC = '/camera/camera/color/image_raw'


# ---------------------------------------------------------------------------
# Camera capture helper
# ---------------------------------------------------------------------------

class CameraCapture:
    """
    Subscribes to the camera topic and saves images on demand.

    Call mark_capture_times() once before the trajectory starts.
    During trajectory execution, call tick(elapsed_time) every control
    cycle — it automatically saves an image when the elapsed time crosses
    the next scheduled capture time.
    """

    def __init__(self, node: Node, output_dir: str, n_images: int):
        self.node = node
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self._lock = threading.Lock()

        self._capture_times: np.ndarray = np.array([])
        self._next_capture_idx: int = 0
        self._saved_count: int = 0

        self._sub = node.create_subscription(
            Image, CAMERA_TOPIC, self._image_callback, 5
        )
        node.get_logger().info(f"CameraCapture: listening on {CAMERA_TOPIC}")
        node.get_logger().info(f"CameraCapture: saving images to {self.output_dir}")

    def _image_callback(self, msg: Image):
        with self._lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def mark_capture_times(self, capture_times: np.ndarray):
        """
        Set the scheduled capture times (seconds from trajectory start).
        Must be called before the trajectory begins.
        """
        self._capture_times = np.sort(capture_times)
        self._next_capture_idx = 0
        self._saved_count = 0
        self.node.get_logger().info(
            f"CameraCapture: scheduled {len(capture_times)} captures at "
            f"t = {capture_times.round(1).tolist()} s"
        )

    def tick(self, elapsed_time: float):
        """
        Call this every control cycle with the current elapsed time.
        Saves an image whenever a scheduled capture time is passed.
        """
        while (self._next_capture_idx < len(self._capture_times) and
               elapsed_time >= self._capture_times[self._next_capture_idx]):
            self._save_image(self._next_capture_idx)
            self._next_capture_idx += 1

    def _save_image(self, idx: int):
        with self._lock:
            if self.latest_image is None:
                self.node.get_logger().warn(
                    f"CameraCapture: no image available for capture {idx}, skipping"
                )
                return
            img = self.latest_image.copy()

        angle_deg = round(360.0 * idx / len(self._capture_times))
        filename = self.output_dir / f"frame_{idx:03d}_deg{angle_deg:03d}.jpg"
        cv2.imwrite(str(filename), img)
        self._saved_count += 1
        self.node.get_logger().info(
            f"CameraCapture: saved image {self._saved_count} → {filename.name}"
        )

    @property
    def saved_count(self) -> int:
        return self._saved_count


# ---------------------------------------------------------------------------
# Main scanner node
# ---------------------------------------------------------------------------

class DishScanner(Node):
    """
    ROS2 node that:
      1. Detects the AR tag and builds a ScanOrbitTrajectory around it
      2. Moves to the trajectory start position
      3. Executes the orbit while capturing images at uniform angular intervals
      4. Reports where images were saved
    """

    def __init__(self, args):
        super().__init__('dish_scanner_node')
        self.args = args

        # TF listener (same pattern as VisualServo)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint state subscriber
        self.current_joint_state: JointState | None = None
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 1)

        # MoveIt IK client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Still waiting for /compute_ik...')

        # Trajectory executor (uses the same UR7eTrajectoryController as main.py)
        self.trajectory_controller = UR7eTrajectoryController(self)

        # Camera capture helper
        output_dir = args.output_dir or self._default_output_dir()
        self.camera_capture = CameraCapture(self, output_dir, args.n_images)

        self.get_logger().info("DishScanner initialised")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg

    # ------------------------------------------------------------------
    # IK (identical to VisualServo.compute_ik)
    # ------------------------------------------------------------------

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        if self.current_joint_state is None:
            self.get_logger().error("No joint state available for IK")
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'ur_manipulator'
        req.ik_request.robot_state.joint_state = self.current_joint_state
        req.ik_request.ik_link_name = 'wrist_3_link'
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = Duration(sec=2)
        req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service call failed')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        return result.solution.joint_state

    # ------------------------------------------------------------------
    # AR tag lookup (identical to VisualServo.lookup_ar_tag)
    # ------------------------------------------------------------------

    def lookup_ar_tag(self, marker_id, timeout=5.0):
        source_frame = f'ar_marker_{marker_id}'
        start = time.time()

        while rclpy.ok() and (time.time() - start) < timeout:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link', source_frame, rclpy.time.Time()
                )
                t = tf.transform.translation
                return np.array([t.x, t.y, t.z])
            except Exception:
                self.get_logger().info(
                    f"Waiting for AR tag {marker_id}... ({time.time()-start:.1f}s)"
                )
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(f"Could not find AR tag {marker_id} after {timeout}s")
        return None

    # ------------------------------------------------------------------
    # Trajectory execution with image capture
    # ------------------------------------------------------------------

    def _move_to_start(self, trajectory: ScanOrbitTrajectory):
        """
        Move the robot to the first waypoint of the scan orbit using
        a short linear trajectory. This avoids a sudden jump.
        """
        start_pose = trajectory.target_pose(0.0)
        sx, sy, sz = start_pose[:3]
        qx, qy, qz, qw = start_pose[3:]

        self.get_logger().info(
            f"Moving to scan start: ({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )

        # Get current wrist position for the linear move
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'wrist_3_link', rclpy.time.Time()
            )
            wt = tf.transform.translation
            current_pos = np.array([wt.x, wt.y, wt.z])
        except Exception as e:
            self.get_logger().warn(f"Could not get current wrist pos: {e}")
            return

        approach = LinearTrajectory(current_pos, np.array([sx, sy, sz]), total_time=5.0)
        self.trajectory_controller.execute_trajectory(
            approach,
            compute_ik_fn=self.compute_ik,
            num_waypoints=50,
        )
        self.get_logger().info("Reached scan start position")

    def execute_scan(self, trajectory: ScanOrbitTrajectory):
        """
        Execute the scan orbit.

        Hooks into UR7eTrajectoryController.execute_trajectory the same way
        main.py does, but additionally calls camera_capture.tick() at each
        waypoint so images are saved at the right angles.

        If UR7eTrajectoryController exposes an on_waypoint callback, use it.
        Otherwise we fall back to a parallel timer thread that polls elapsed
        time and ticks the capture logic independently.
        """
        # Schedule image captures at angularly-uniform times
        capture_times = trajectory.capture_times(self.args.n_images)
        self.camera_capture.mark_capture_times(capture_times)

        self.get_logger().info(
            f"Starting scan orbit: {self.args.n_images} images over "
            f"{self.args.total_time:.0f}s "
            f"(radius={self.args.radius}m, height={self.args.height}m)"
        )

        # --- Approach A: controller supports an on_waypoint hook -------
        # Uncomment this block and remove Approach B if your
        # UR7eTrajectoryController accepts an on_waypoint_fn callback.
        #
        # self.trajectory_controller.execute_trajectory(
        #     trajectory,
        #     compute_ik_fn=self.compute_ik,
        #     num_waypoints=200,
        #     on_waypoint_fn=lambda elapsed: self.camera_capture.tick(elapsed),
        # )

        # --- Approach B: parallel capture thread -----------------------
        # Runs a background thread that ticks the capture logic while the
        # trajectory controller blocks the main thread.
        start_event = threading.Event()
        stop_event = threading.Event()

        def capture_thread():
            start_event.wait()   # wait until trajectory actually starts
            t0 = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - t0
                self.camera_capture.tick(elapsed)
                time.sleep(0.05)   # 20 Hz polling — fast enough for any orbit speed

        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
        start_event.set()   # signal thread to start the clock

        try:
            self.trajectory_controller.execute_trajectory(
                trajectory,
                compute_ik_fn=self.compute_ik,
                num_waypoints=200,
            )
        finally:
            stop_event.set()
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self):
        # 1. Detect dish position from AR tag
        self.get_logger().info(f"Looking up AR tag {self.args.ar_marker}...")
        dish_pos = self.lookup_ar_tag(self.args.ar_marker, timeout=10.0)

        if dish_pos is None:
            self.get_logger().error("Aborting: AR tag not found")
            return

        self.get_logger().info(f"Dish position (base_link): {dish_pos.round(3)}")

        # 2. Build scan orbit around the detected dish position
        trajectory = ScanOrbitTrajectory(
            dish_position=dish_pos,
            radius=self.args.radius,
            scan_height=self.args.height,
            total_time=self.args.total_time,
        )

        # 3. Confirm before moving
        self.get_logger().info(
            f"\n=== Scan plan ===\n"
            f"  Dish:       {dish_pos.round(3)}\n"
            f"  Orbit at:   {trajectory.orbit_center.round(3)}\n"
            f"  Radius:     {self.args.radius} m\n"
            f"  Height:     {self.args.height} m\n"
            f"  Images:     {self.args.n_images}\n"
            f"  Duration:   {self.args.total_time} s\n"
            f"  Output dir: {self.camera_capture.output_dir}\n"
            f"\nPress ENTER to execute (Ctrl+C to abort)"
        )
        while rclpy.ok():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        # 4. Move to orbit start position
        self._move_to_start(trajectory)

        # 5. Execute the scan orbit with image capture
        self.execute_scan(trajectory)

        # 6. Report results
        self.get_logger().info(
            f"\n=== Scan complete ===\n"
            f"  Images saved:  {self.camera_capture.saved_count}\n"
            f"  Output dir:    {self.camera_capture.output_dir}\n"
        )
        self._print_colmap_instructions()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _default_output_dir() -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.expanduser(f'~/dish_scans/scan_{timestamp}')

    def _print_colmap_instructions(self):
        out = self.camera_capture.output_dir
        self.get_logger().info(
            f"\n=== 3D Reconstruction ===\n"
            f"\nOption 1 — COLMAP (CLI):\n"
            f"  colmap automatic_reconstructor \\\n"
            f"      --workspace_path {out}/colmap \\\n"
            f"      --image_path {out}\n"
            f"\nOption 2 — Meshroom (GUI):\n"
            f"  Open Meshroom, drag-and-drop all images from:\n"
            f"      {out}\n"
            f"  then click 'Start'.\n"
            f"\nOption 3 — Open3D (Python, simple point cloud):\n"
            f"  See the commented script at the bottom of dish_scanner.py\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description='Dish scanner for 3D plating reconstruction')
    parser.add_argument('--ar_marker', type=int, default=0,
                        help='AR marker ID placed at/near the dish center')
    parser.add_argument('--radius', type=float, default=0.18,
                        help='Orbit radius in meters (default: 0.18)')
    parser.add_argument('--height', type=float, default=0.28,
                        help='Orbit height above the dish in meters (default: 0.28)')
    parser.add_argument('--total_time', type=float, default=20.0,
                        help='Total scan duration in seconds (default: 20.0). '
                             'Slower is better — gives the camera time to settle.')
    parser.add_argument('--n_images', type=int, default=36,
                        help='Number of images to capture (default: 36, i.e. every 10°)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save images. Defaults to ~/dish_scans/scan_<timestamp>')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = DishScanner(parsed_args)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Scan aborted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Optional: quick Open3D point cloud from saved images
# Run this separately after the scan, not during ROS execution.
# ---------------------------------------------------------------------------
#
# import open3d as o3d
# from pathlib import Path
#
# def build_point_cloud(image_dir: str):
#     """
#     Minimal example using Open3D's tensor-based pipeline.
#     For best results use COLMAP or Meshroom instead.
#     """
#     image_dir = Path(image_dir)
#     images = sorted(image_dir.glob("*.jpg"))
#     print(f"Found {len(images)} images in {image_dir}")
#
#     # Open3D RGBD reconstruction requires pre-computed camera poses.
#     # The easiest path is to run COLMAP first to get poses, then use
#     # Open3D for meshing / colorisation.
#     #
#     # Quick sanity check — just display the first image:
#     import cv2
#     img = cv2.imread(str(images[0]))
#     cv2.imshow("First scan image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()