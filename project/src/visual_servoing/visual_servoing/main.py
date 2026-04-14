#!/usr/bin/env python3
"""
Main script to run visual servoing
Author: Daniel Municio, Spring 2026

Fixed: IK seeding with previous solution, joint angle unwrapping,
       increased waypoint density for scan orbit trajectories.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from tf2_ros import TransformListener, Buffer
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
import argparse
import sys
import select
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .trajectories import LinearTrajectory, CircularTrajectory, ScanOrbitTrajectory
from .controller import UR7eTrajectoryController, PIDJointVelocityController


# ---------------------------------------------------------------------------
# Joint-space utilities
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def unwrap_joint_angles(prev_positions, new_positions):
    """
    Unwrap joint angles so that the transition from prev to new never jumps
    by more than pi.  This fixes the +/-pi discontinuity that causes velocity
    spikes and trajectory controller oscillations.

    Parameters
    ----------
    prev_positions : np.ndarray  (6,)
        Joint angles from the previous waypoint.
    new_positions  : np.ndarray  (6,)
        Joint angles from the current IK solution.

    Returns
    -------
    np.ndarray (6,)
        Adjusted joint angles (may lie outside [-pi, pi] but are continuous
        with prev_positions).
    """
    diff = np.array(new_positions) - np.array(prev_positions)
    adjusted = np.array(new_positions) - np.round(diff / (2 * np.pi)) * (2 * np.pi)
    return adjusted


def extract_joint_positions(joint_state_msg):
    """Extract the 6 UR joint positions in canonical order from a JointState."""
    d = {name: pos for name, pos in zip(joint_state_msg.name, joint_state_msg.position)}
    return np.array([d[n] for n in JOINT_NAMES])


class VisualServo(Node):
    def __init__(self, args):
        super().__init__('visual_servo_node')

        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trajectory = None
        self.trajectory_start_time = None
        self.current_joint_state = None

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 1
        )

        # MoveIt IK client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.controller_type = args.controller
        self.trajectory_controller = UR7eTrajectoryController(self)

        if self.controller_type == 'pid':
            Kp = 12 * np.array([0.4, 2, 1.7, 1.5, 2, 2])
            Kd = 1.5 * np.array([2, 1, 2, 0.5, 0.8, 0.8])
            Ki = 0.05 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6])
            self.velocity_controller = PIDJointVelocityController(self, Kp, Ki, Kd)

        self.path_pub = self.create_publisher(Path, '/trajectory_path', 1)

        self.viz_timer = None
        self.ar_tag_detected = False
        self.ar_tag_position = None

        self.get_logger().info("Visual Servo Node initialized")
        self.get_logger().info(f"Task: {args.task}")
        self.get_logger().info(f"AR Marker ID: {args.ar_marker}")
        self.get_logger().info(f"Controller: {args.controller}")

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0, seed_joint_state=None):
        """
        Compute IK for a given workspace pose using MoveIt.

        Parameters
        ----------
        x, y, z : float
            Target position in base_link frame.
        qx, qy, qz, qw : float
            Target orientation as quaternion.
        seed_joint_state : JointState or None
            ** KEY FIX ** — If provided, seeds the IK solver with this state
            instead of the live robot state. Seeding with the previous waypoint's
            solution keeps the solver in the same joint-space branch, preventing
            the wrist-flip discontinuity at ~270 degrees.

        Returns
        -------
        JointState or None
        """
        seed = seed_joint_state if seed_joint_state is not None else self.current_joint_state

        if seed is None:
            self.get_logger().error("No joint state available for IK seed")
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

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'ur_manipulator'
        ik_req.ik_request.robot_state.joint_state = seed
        ik_req.ik_request.ik_link_name = 'wrist_3_link'
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.timeout = Duration(sec=2)
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        return result.solution.joint_state

    def lookup_ar_tag(self, marker_id, timeout=5.0):
        target_frame = 'base_link'
        source_frame = f'ar_marker_{marker_id}'
        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < timeout:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time()
                )
                translation = transform.transform.translation
                return np.array([translation.x, translation.y, translation.z])
            except Exception:
                self.get_logger().info(
                    f"Waiting for AR tag {marker_id}... ({time.time() - start_time:.1f}s)"
                )
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(f"Could not find AR tag {marker_id} after {timeout}s")
        return None

    def create_trajectory(self):
        ar_position = self.lookup_ar_tag(self.args.ar_marker)
        if ar_position is None:
            self.get_logger().error(f"Could not find AR tag {self.args.ar_marker}")
            return None

        hover_height = 0.25
        goal_position = ar_position + np.array([0, 0, hover_height])

        if self.args.task == 'line':
            try:
                wrist_transform = self.tf_buffer.lookup_transform(
                    'base_link', 'wrist_3_link', rclpy.time.Time()
                )
                wrist_pos = wrist_transform.transform.translation
                start_position = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])
            except Exception as e:
                self.get_logger().warn(f"Could not get current wrist position: {e}")
                return None

            self.get_logger().info(f"  Task: LINEAR")
            self.get_logger().info(f"  Start: {start_position}")
            trajectory = LinearTrajectory(
                start_position=start_position,
                goal_position=goal_position,
                total_time=self.args.total_time,
            )

        elif self.args.task == 'circle':
            self.get_logger().info(f"  Task: CIRCULAR")
            self.get_logger().info(f"  Center: {goal_position}")
            trajectory = CircularTrajectory(
                center_position=goal_position,
                radius=self.args.circle_radius,
                total_time=self.args.total_time,
            )

        elif self.args.task == 'scanorbit':
            self.get_logger().info(f"  Task: SCAN ORBIT")
            self.get_logger().info(f"  Dish position: {ar_position}")
            self.get_logger().info(f"  Radius: {self.args.circle_radius} m")
            trajectory = ScanOrbitTrajectory(
                dish_position=ar_position,
                radius=self.args.circle_radius,
                scan_height=0.28,
                total_time=self.args.total_time,
            )
        else:
            self.get_logger().error(f"Unknown task: {self.args.task}")
            return None

        return trajectory

    def publish_trajectory_visualization(self):
        if self.trajectory is None:
            return

        num_vis_points = 100
        times = np.linspace(0, self.trajectory.total_time, num_vis_points)

        path_msg = Path()
        path_msg.header.frame_id = 'base_link'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for t in times:
            desired_pose = self.trajectory.target_pose(t)
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = desired_pose[0]
            pose_stamped.pose.position.y = desired_pose[1]
            pose_stamped.pose.position.z = desired_pose[2]
            pose_stamped.pose.orientation.x = desired_pose[3]
            pose_stamped.pose.orientation.y = desired_pose[4]
            pose_stamped.pose.orientation.z = desired_pose[5]
            pose_stamped.pose.orientation.w = desired_pose[6]
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def visualization_callback(self):
        self.publish_trajectory_visualization()

    def start_visualization_timer(self):
        if self.viz_timer is not None:
            return
        self.viz_timer = self.create_timer(1.0, self.visualization_callback)
        self.get_logger().info("Started trajectory visualization timer (1 Hz)")

    # ------------------------------------------------------------------
    # Trajectory execution  (FIXED)
    # ------------------------------------------------------------------

    def execute_trajectory(self):
        """
        Execute the trajectory by computing waypoints and sending to robot.

        Key fixes vs. original:
        1. IK seeded with previous IK solution, not live robot state.
        2. Joint angles unwrapped so wrist crossing +/-pi stays continuous.
        3. More waypoints for scan orbit (80) and circle (50).
        4. Max joint jump detection with warnings.
        """
        if self.trajectory is None:
            self.get_logger().error("No trajectory to execute")
            return

        self.get_logger().info("Starting trajectory execution...")

        # Scale waypoint count to trajectory complexity
        if self.args.task == 'scanorbit':
            num_waypoints = 80
        elif self.args.task == 'circle':
            num_waypoints = 50
        else:
            num_waypoints = 20

        times = np.linspace(0, self.trajectory.total_time, num_waypoints)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(JOINT_NAMES)

        self.get_logger().info(f"Computing IK for {num_waypoints} waypoints...")

        prev_ik_solution = None
        prev_positions = None

        # We'll collect all waypoints first, then adjust timing
        all_positions = []

        for i, t in enumerate(times):
            desired_pose = self.trajectory.target_pose(t)
            x, y, z = desired_pose[0:3]
            qx, qy, qz, qw = desired_pose[3:7]

            joint_solution = self.compute_ik(
                x, y, z, qx, qy, qz, qw,
                seed_joint_state=prev_ik_solution,
            )

            if joint_solution is None:
                self.get_logger().error(
                    f"IK failed at waypoint {i+1}/{num_waypoints} "
                    f"(t={t:.2f}s, pos=[{x:.3f},{y:.3f},{z:.3f}])"
                )
                return

            raw_positions = extract_joint_positions(joint_solution)

            if prev_positions is not None:
                positions = unwrap_joint_angles(prev_positions, raw_positions)
            else:
                positions = raw_positions

            # Warn on large jumps (helps debug remaining issues)
            if prev_positions is not None:
                max_jump = np.max(np.abs(positions - prev_positions))
                if max_jump > 0.5:
                    self.get_logger().warn(
                        f"  Large joint jump at waypoint {i}: "
                        f"max delta = {max_jump:.3f} rad"
                    )

            all_positions.append(positions)

            # Update seed for next iteration with UNWRAPPED positions
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

        # --- FIX: Skip the first waypoint (t=0) and offset times ---
        # The robot is already at the start position from _move_to_start.
        # Starting the trajectory at t=0 means the controller expects to
        # already be there perfectly at time zero — any tiny settling error
        # trips the path tolerance and aborts.  Instead, skip the t=0
        # waypoint and give a small offset so the first real waypoint
        # starts at t=0.5s, giving the controller time to begin tracking.
        TIME_OFFSET = 0.5  # seconds of slack before the first waypoint

        for i in range(1, len(times)):  # skip i=0
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
            if i == 0:
                if len(joint_traj.points) > 1:
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
            else:
                dt = self._time_diff(joint_traj.points[i+1], joint_traj.points[i-1])
                if dt > 0:
                    dp = np.array(joint_traj.points[i+1].positions) - \
                         np.array(joint_traj.points[i-1].positions)
                    joint_traj.points[i].velocities = list(dp / dt)

        # Log max velocity for sanity check
        max_vel = 0.0
        for pt in joint_traj.points:
            max_vel = max(max_vel, max(abs(v) for v in pt.velocities))
        self.get_logger().info(f"Max joint velocity in trajectory: {max_vel:.3f} rad/s")

        self.get_logger().info("Moving to trajectory start position...")
        self._move_to_start(joint_traj.points[0].positions)

        switch_controllers(self.controller_type)
        self._execute_joint_trajectory(joint_traj)

    @staticmethod
    def _time_diff(point_a, point_b):
        return (point_a.time_from_start.sec - point_b.time_from_start.sec) + \
               (point_a.time_from_start.nanosec - point_b.time_from_start.nanosec) * 1e-9

    def _move_to_start(self, start_positions):
        if self.args.task == 'line':
            self.get_logger().info('Line task: skipping move to start')
            return

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(start_positions)
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        joint_traj.points.append(point)

        success = self.trajectory_controller.execute_joint_trajectory(joint_traj)
        if success:
            self.get_logger().info('Reached trajectory start position')
        else:
            self.get_logger().error('Failed to reach trajectory start position')

    def _execute_joint_trajectory(self, joint_traj):
        if self.controller_type == 'default':
            success = self.trajectory_controller.execute_joint_trajectory(joint_traj)
            if not success:
                self.get_logger().error('Trajectory execution failed')
        else:
            self._execute_velocity_control(joint_traj)

    def _execute_velocity_control(self, joint_traj):
        self._control_joint_traj = joint_traj
        self._control_current_index = 0
        self._control_max_index = len(joint_traj.points) - 1
        self._control_iteration = 0
        self._control_start_time = self.get_clock().now()
        self._control_done = False

        self._velocity_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10
        )

        self.get_logger().info(
            f'Executing with {self.velocity_controller.get_name()} controller...'
        )

        self._control_timer = self.create_timer(0.1, self._velocity_control_callback)

        while rclpy.ok() and not self._control_done:
            rclpy.spin_once(self, timeout_sec=0.1)

        vel_msg = Float64MultiArray()
        vel_msg.data = [0.0] * 6
        self._velocity_pub.publish(vel_msg)
        self._control_timer.cancel()
        self.get_logger().info('Velocity control execution complete!')

    def _velocity_control_callback(self):
        elapsed = (self.get_clock().now() - self._control_start_time).nanoseconds / 1e9

        if self.current_joint_state is None:
            return

        current_joint_dict = {}
        for i, name in enumerate(self.current_joint_state.name):
            if i < 6:
                current_joint_dict[name] = (
                    self.current_joint_state.position[i],
                    self.current_joint_state.velocity[i],
                )

        current_position = np.array([current_joint_dict[n][0] for n in JOINT_NAMES])
        current_velocity = np.array([current_joint_dict[n][1] for n in JOINT_NAMES])

        target_position, target_velocity, self._control_current_index = \
            self._interpolate_trajectory(
                self._control_joint_traj, elapsed, self._control_current_index
            )

        commanded_velocity = self.velocity_controller.step_control(
            target_position, target_velocity, current_position, current_velocity
        )

        vel_msg = Float64MultiArray()
        vel_msg.data = commanded_velocity.tolist()
        self._velocity_pub.publish(vel_msg)

        if self._control_current_index >= self._control_max_index:
            self.get_logger().info(f'Reached max index at t={elapsed:.2f}s')
            self._control_done = True

        self._control_iteration += 1

    def _interpolate_trajectory(self, joint_traj, t, current_index=0):
        epsilon = 0.0001
        max_index = len(joint_traj.points) - 1

        current_time = joint_traj.points[current_index].time_from_start.sec + \
                       joint_traj.points[current_index].time_from_start.nanosec * 1e-9
        if current_time > t:
            current_index = 0

        while (current_index < max_index and
               joint_traj.points[current_index + 1].time_from_start.sec +
               joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9
               < t + epsilon):
            current_index += 1

        if current_index < max_index:
            time_low = joint_traj.points[current_index].time_from_start.sec + \
                       joint_traj.points[current_index].time_from_start.nanosec * 1e-9
            time_high = joint_traj.points[current_index + 1].time_from_start.sec + \
                        joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9

            target_position_low = np.array(joint_traj.points[current_index].positions)
            target_velocity_low = np.array(joint_traj.points[current_index].velocities)
            target_position_high = np.array(joint_traj.points[current_index + 1].positions)
            target_velocity_high = np.array(joint_traj.points[current_index + 1].velocities)

            alpha = (t - time_low) / (time_high - time_low) if time_high != time_low else 0
            target_position = target_position_low + alpha * (target_position_high - target_position_low)
            target_velocity = target_velocity_low + alpha * (target_velocity_high - target_velocity_low)
        else:
            target_position = np.array(joint_traj.points[current_index].positions)
            target_velocity = np.array(joint_traj.points[current_index].velocities)

        return target_position, target_velocity, current_index

    def run(self):
        self.trajectory = self.create_trajectory()
        if self.trajectory is None:
            self.get_logger().error("Failed to create trajectory")
            return

        self.get_logger().info("Trajectory created! Publishing visualization to RViz...")
        self.start_visualization_timer()

        self.get_logger().info("\nPress ENTER to execute trajectory (Ctrl+C to cancel)")
        while rclpy.ok():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.execute_trajectory()
        self.get_logger().info("=== Execution Complete ===")


def switch_controllers(controller_type):
    if controller_type == 'default':
        cmd = [
            'ros2', 'control', 'switch_controllers',
            '--deactivate', 'forward_velocity_controller',
            '--activate', 'scaled_joint_trajectory_controller',
        ]
    else:
        cmd = [
            'ros2', 'control', 'switch_controllers',
            '--deactivate', 'scaled_joint_trajectory_controller',
            '--activate', 'forward_velocity_controller',
        ]

    logger.info(f"Switching controllers for '{controller_type}' mode...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("Switching controllers successful")
        else:
            logger.warning(f"Controller switch warning: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Controller switch timed out")
    except Exception as e:
        logger.error(f"Controller switch failed: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(description='Visual Servoing')
    parser.add_argument('--task', '-t', type=str, default='line',
                        choices=['line', 'circle', 'scanorbit'],
                        help='Type of trajectory')
    parser.add_argument('--ar_marker', type=int, default=0)
    parser.add_argument('--total_time', type=float, default=10.0)
    parser.add_argument('--circle_radius', type=float, default=0.1)
    parser.add_argument('--controller', '-c', type=str, default='default',
                        choices=['default', 'pid'])

    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = VisualServo(parsed_args)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        logger.info("Cleaning up: switching back to default controller...")
        switch_controllers('default')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()