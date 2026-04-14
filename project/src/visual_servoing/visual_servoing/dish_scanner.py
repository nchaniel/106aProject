#!/usr/bin/env python3
"""
dish_scanner.py — Plating robot dish scanner
=============================================
Orbits the UR7e around a plated dish and captures images for 3D reconstruction.

Fixed: IK seeding with previous solution, joint angle unwrapping,
       increased waypoint density, same fixes as main.py.

Usage
-----
    ros2 run visual_servoing dish_scanner \
        --ar_marker 0 \
        --radius 0.18 \
        --height 0.28 \
        --total_time 20.0 \
        --n_images 36 \
        --output_dir ~/scan_output
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

import cv2
from cv_bridge import CvBridge

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
# Joint-space utilities (same as fixed main.py)
# ---------------------------------------------------------------------------

def unwrap_joint_angles(prev_positions, new_positions):
    """
    Unwrap joint angles so transitions never jump by more than pi.
    """
    diff = np.array(new_positions) - np.array(prev_positions)
    adjusted = np.array(new_positions) - np.round(diff / (2 * np.pi)) * (2 * np.pi)
    return adjusted


def extract_joint_positions(joint_state_msg):
    """Extract the 6 UR joint positions in canonical order."""
    d = {name: pos for name, pos in zip(joint_state_msg.name, joint_state_msg.position)}
    return np.array([d[n] for n in JOINT_NAMES])


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
        self.latest_image = None
        self._lock = threading.Lock()

        self._capture_times = np.array([])
        self._next_capture_idx = 0
        self._saved_count = 0

        self._sub = node.create_subscription(
            Image, CAMERA_TOPIC, self._image_callback, 5
        )
        node.get_logger().info(f"CameraCapture: listening on {CAMERA_TOPIC}")
        node.get_logger().info(f"CameraCapture: saving images to {self.output_dir}")

    def _image_callback(self, msg: Image):
        with self._lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def mark_capture_times(self, capture_times: np.ndarray):
        self._capture_times = np.sort(capture_times)
        self._next_capture_idx = 0
        self._saved_count = 0
        self.node.get_logger().info(
            f"CameraCapture: scheduled {len(capture_times)} captures"
        )

    def tick(self, elapsed_time: float):
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
            f"CameraCapture: saved image {self._saved_count} -> {filename.name}"
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 1)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Still waiting for /compute_ik...')

        self.trajectory_controller = UR7eTrajectoryController(self)

        output_dir = args.output_dir or self._default_output_dir()
        self.camera_capture = CameraCapture(self, output_dir, args.n_images)

        self.get_logger().info("DishScanner initialised")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg

    # ------------------------------------------------------------------
    # IK  (FIXED: accepts seed_joint_state)
    # ------------------------------------------------------------------

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0, seed_joint_state=None):
        """
        Compute IK with optional seed state.

        The seed_joint_state parameter is the key fix: seeding with the
        previous waypoint's IK solution keeps the solver in the same
        joint-space branch throughout the orbit.
        """
        seed = seed_joint_state if seed_joint_state is not None else self.current_joint_state

        if seed is None:
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
        req.ik_request.robot_state.joint_state = seed
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
    # AR tag lookup
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
    # Build joint trajectory from Cartesian trajectory (FIXED)
    # ------------------------------------------------------------------

    def _build_joint_trajectory(self, trajectory, num_waypoints=80):
        """
        Convert a Cartesian trajectory into a JointTrajectory with:
        - IK seeded from previous solution
        - Joint angle unwrapping
        - Finite-difference velocities

        Returns
        -------
        JointTrajectory or None
        """
        times = np.linspace(0, trajectory.total_time, num_waypoints)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(JOINT_NAMES)

        prev_ik_solution = None
        prev_positions = None
        all_positions = []

        for i, t in enumerate(times):
            desired_pose = trajectory.target_pose(t)
            x, y, z = desired_pose[0:3]
            qx, qy, qz, qw = desired_pose[3:7]

            joint_solution = self.compute_ik(
                x, y, z, qx, qy, qz, qw,
                seed_joint_state=prev_ik_solution,
            )

            if joint_solution is None:
                self.get_logger().error(
                    f"IK failed at waypoint {i+1}/{num_waypoints} "
                    f"(t={t:.2f}s)"
                )
                return None

            raw_positions = extract_joint_positions(joint_solution)

            if prev_positions is not None:
                positions = unwrap_joint_angles(prev_positions, raw_positions)
            else:
                positions = raw_positions

            # Warn on large jumps
            if prev_positions is not None:
                max_jump = np.max(np.abs(positions - prev_positions))
                if max_jump > 0.5:
                    self.get_logger().warn(
                        f"  Large joint jump at waypoint {i}: "
                        f"max delta = {max_jump:.3f} rad"
                    )

            all_positions.append(positions)

            # Update seed with unwrapped positions
            seed_msg = JointState()
            seed_msg.name = list(JOINT_NAMES)
            seed_msg.position = positions.tolist()
            prev_ik_solution = seed_msg
            prev_positions = positions

            if (i + 1) % 10 == 0 or i == 0:
                self.get_logger().info(
                    f"  Waypoint {i+1}/{num_waypoints} OK "
                    f"(j6={positions[5]:.3f} rad)"
                )

        # Skip the t=0 waypoint and offset times so the controller
        # has time to start tracking before the first real waypoint.
        TIME_OFFSET = 0.5

        for i in range(1, len(times)):
            t = times[i] + TIME_OFFSET
            positions = all_positions[i]

            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            point.velocities = [0.0] * 6
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            joint_traj.points.append(point)

        # Compute velocities via finite differences
        for i in range(len(joint_traj.points)):
            if i == 0 and len(joint_traj.points) > 1:
                dt = self._time_diff(joint_traj.points[1], joint_traj.points[0])
                if dt > 0:
                    dp = np.array(joint_traj.points[1].positions) - \
                         np.array(joint_traj.points[0].positions)
                    joint_traj.points[i].velocities = list(dp / dt)
            elif i == len(joint_traj.points) - 1:
                dt = self._time_diff(joint_traj.points[i], joint_traj.points[i-1])
                if dt > 0:
                    dp = np.array(joint_traj.points[i].positions) - \
                         np.array(joint_traj.points[i-1].positions)
                    joint_traj.points[i].velocities = list(dp / dt)
            elif 0 < i < len(joint_traj.points) - 1:
                dt = self._time_diff(joint_traj.points[i+1], joint_traj.points[i-1])
                if dt > 0:
                    dp = np.array(joint_traj.points[i+1].positions) - \
                         np.array(joint_traj.points[i-1].positions)
                    joint_traj.points[i].velocities = list(dp / dt)

        max_vel = max(max(abs(v) for v in pt.velocities) for pt in joint_traj.points)
        self.get_logger().info(f"Max joint velocity: {max_vel:.3f} rad/s")

        return joint_traj

    @staticmethod
    def _time_diff(point_a, point_b):
        return (point_a.time_from_start.sec - point_b.time_from_start.sec) + \
               (point_a.time_from_start.nanosec - point_b.time_from_start.nanosec) * 1e-9

    # ------------------------------------------------------------------
    # Trajectory execution with image capture
    # ------------------------------------------------------------------

    def _move_to_start(self, trajectory: ScanOrbitTrajectory):
        """Move to the first waypoint of the scan orbit via a linear move."""
        start_pose = trajectory.target_pose(0.0)
        sx, sy, sz = start_pose[:3]

        self.get_logger().info(
            f"Moving to scan start: ({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )

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

        # Build the approach joint trajectory with the same fixed IK pipeline
        approach_traj = self._build_joint_trajectory(approach, num_waypoints=20)
        if approach_traj is not None:
            self.trajectory_controller.execute_joint_trajectory(approach_traj)
            self.get_logger().info("Reached scan start position")
        else:
            self.get_logger().error("Failed to build approach trajectory")

    def execute_scan(self, trajectory: ScanOrbitTrajectory):
        """
        Execute the scan orbit with image capture.
        """
        # Schedule image captures at angularly-uniform times
        capture_times = trajectory.capture_times(self.args.n_images)
        self.camera_capture.mark_capture_times(capture_times)

        self.get_logger().info(
            f"Starting scan orbit: {self.args.n_images} images over "
            f"{self.args.total_time:.0f}s "
            f"(radius={self.args.radius}m, height={self.args.height}m)"
        )

        # Build the orbit joint trajectory
        self.get_logger().info("Computing orbit joint trajectory...")
        joint_traj = self._build_joint_trajectory(trajectory, num_waypoints=80)
        if joint_traj is None:
            self.get_logger().error("Failed to build orbit trajectory — aborting scan")
            return

        # Parallel capture thread (same approach as original)
        start_event = threading.Event()
        stop_event = threading.Event()

        def capture_thread():
            start_event.wait()
            t0 = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - t0
                self.camera_capture.tick(elapsed)
                time.sleep(0.05)

        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
        start_event.set()

        try:
            self.trajectory_controller.execute_joint_trajectory(joint_traj)
        finally:
            stop_event.set()
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self):
        self.get_logger().info(f"Looking up AR tag {self.args.ar_marker}...")
        dish_pos = self.lookup_ar_tag(self.args.ar_marker, timeout=10.0)

        if dish_pos is None:
            self.get_logger().error("Aborting: AR tag not found")
            return

        self.get_logger().info(f"Dish position (base_link): {dish_pos.round(3)}")

        trajectory = ScanOrbitTrajectory(
            dish_position=dish_pos,
            radius=self.args.radius,
            scan_height=self.args.height,
            total_time=self.args.total_time,
        )

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

        self._move_to_start(trajectory)
        self.execute_scan(trajectory)

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
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description='Dish scanner for 3D plating reconstruction')
    parser.add_argument('--ar_marker', type=int, default=0)
    parser.add_argument('--radius', type=float, default=0.18)
    parser.add_argument('--height', type=float, default=0.28)
    parser.add_argument('--total_time', type=float, default=20.0)
    parser.add_argument('--n_images', type=int, default=36)
    parser.add_argument('--output_dir', type=str, default=None)
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