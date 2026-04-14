#!/usr/bin/env python3
"""
Merged Lab 5 + Lab 7: Point Cloud Pick & Place to AR Tag
=========================================================

This node combines:
  - Lab 7's AR tag detection (via TF lookup) to get the PLACE target
  - Lab 5's point cloud pipeline to detect objects to PICK
  - Lab 5's MoveIt IK + FollowJointTrajectory action server for execution

Usage:
  # Terminal 1: enable comms
  ros2 run ur7e_utils enable_comms

  # Terminal 2: launch the merged bringup (see merged_bringup.launch.py)
  ros2 launch planning merged_bringup.launch.py

  # Terminal 3: run this node
  ros2 run planning main_merged --ar_marker 0

The robot will:
  1. Wait for the AR tag to be detected (place location)
  2. Subscribe to /cube_pose from the point cloud pipeline (pick location)
  3. Pick up each detected object and place it at the AR tag location
  4. Re-enable point cloud listening to pick the next object (--continuous mode)

Author: Auto-generated merge of Labs 5 & 7
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

import numpy as np
import argparse
import time
import sys

# ── If installed as a ROS2 package, use relative imports ──
# Adjust these if your package structure differs.
try:
    from planning.ik import IKPlanner
except ImportError:
    # Fallback: assume ik.py is in the same directory
    from ik import IKPlanner


class PickAndPlaceToAR(Node):
    """
    Combined perception→planning→control node.

    Perception
    ----------
    * AR tag pose  – looked up from the TF tree (Lab 7 style)
    * Object pose  – received on /cube_pose (published by process_pointcloud.py from Lab 5)

    Planning
    --------
    * IK via MoveIt (/compute_ik + /plan_kinematic_path)

    Control
    -------
    * FollowJointTrajectory action server on the UR controller
    * Gripper toggle via /toggle_gripper service
    """

    # ── Grasp offsets (metres) ──
    PRE_GRASP_Z_OFFSET   = 0.185   # hover height above object
    GRASP_Z_OFFSET       = 0.092   # actual grasp height (don't go lower than 0.14 per lab safety)
    PLACE_Z_OFFSET       = 0.185   # hover height above AR tag for placement
    PLACE_LOWER_Z_OFFSET = 0.10    # lower to release (above AR tag)
    X_FUDGE              = -0.015  # small x correction you had in lab 5

    def __init__(self, args):
        super().__init__('pick_and_place_to_ar')

        self.args = args

        # ── TF ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscriptions ──
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 1)
        self.cube_pose_sub = self.create_subscription(
            PointStamped, '/cube_pose', self._cube_pose_cb, 1)

        # ── Action client (UR controller) ──
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        # ── Gripper service client ──
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # ── IK Planner (reuse Lab 5's IKPlanner node) ──
        self.ik_planner = IKPlanner()

        # ── State ──
        self.joint_state = None
        self.ar_tag_position = None   # np.ndarray [x, y, z] in base_link
        self.cube_pose = None         # PointStamped
        self.busy = False             # True while executing a pick-place cycle
        self.pick_count = 0

        self.get_logger().info('─── Pick-and-Place to AR Tag node started ───')
        self.get_logger().info(f'  AR marker ID : {args.ar_marker}')
        self.get_logger().info(f'  Continuous   : {args.continuous}')

    # ================================================================== #
    #                       CALLBACKS                                      #
    # ================================================================== #

    def _joint_state_cb(self, msg: JointState):
        self.joint_state = msg

    def _cube_pose_cb(self, msg: PointStamped):
        """Called every time process_pointcloud publishes a new object pose."""
        if self.busy:
            return  # ignore while we're executing
        if self.joint_state is None:
            self.get_logger().info('Waiting for joint state before acting on cube pose...')
            return

        self.cube_pose = msg
        self.get_logger().info(
            f'Object detected at ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})')

    # ================================================================== #
    #                     AR TAG LOOKUP  (Lab 7 style)                     #
    # ================================================================== #

    def lookup_ar_tag(self, marker_id, timeout=10.0):
        """
        Block until the AR tag is visible in the TF tree, then return its
        position in base_link.

        Returns
        -------
        np.ndarray or None
            [x, y, z] of the AR tag in base_link, or None on timeout.
        """
        target_frame = 'base_link'
        source_frame = f'ar_marker_{marker_id}'
        start = time.time()

        while rclpy.ok() and (time.time() - start) < timeout:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
                t = tf.transform.translation
                pos = np.array([t.x, t.y, t.z])
                self.get_logger().info(
                    f'AR tag {marker_id} found at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
                return pos
            except Exception:
                self.get_logger().info(
                    f'Waiting for AR tag {marker_id}... ({time.time()-start:.1f}s)')
                rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().error(f'AR tag {marker_id} not found after {timeout}s')
        return None

    # ================================================================== #
    #                     JOB QUEUE EXECUTION  (Lab 5 style)               #
    # ================================================================== #

    def _build_pick_place_queue(self, pick_xyz, place_xyz):
        """
        Build a job queue (list of JointState | 'toggle_grip') that:
          1. Pre-grasp above the object
          2. Lower to grasp
          3. Close gripper
          4. Retreat to pre-grasp
          5. Move to pre-place above AR tag
          6. Lower to place
          7. Open gripper
          8. Retreat above AR tag

        Returns list or None on IK failure.
        """
        px, py, pz = pick_xyz
        ax, ay, az = place_xyz
        queue = []

        steps = [
            ('Pre-grasp',       px + self.X_FUDGE, py, pz + self.PRE_GRASP_Z_OFFSET),
            ('Grasp',           px + self.X_FUDGE, py, pz + self.GRASP_Z_OFFSET),
            ('close_grip',      None, None, None),
            ('Retreat',         px + self.X_FUDGE, py, pz + self.PRE_GRASP_Z_OFFSET),
            ('Pre-place',       ax, ay, az + self.PLACE_Z_OFFSET),
            ('Place lower',     ax, ay, az + self.PLACE_LOWER_Z_OFFSET),
            ('open_grip',       None, None, None),
            ('Place retreat',   ax, ay, az + self.PLACE_Z_OFFSET),
        ]

        for label, x, y, z in steps:
            if label in ('close_grip', 'open_grip'):
                queue.append('toggle_grip')
                self.get_logger().info(f'  Queue: {label}')
            else:
                js = self.ik_planner.compute_ik(self.joint_state, x, y, z)
                if js is None:
                    self.get_logger().error(f'IK failed for step "{label}" at ({x:.3f},{y:.3f},{z:.3f})')
                    return None
                queue.append(js)
                self.get_logger().info(f'  Queue: {label} → IK ok')

        return queue

    def execute_jobs(self, job_queue):
        """Execute a list of (JointState | 'toggle_grip') sequentially."""
        while job_queue:
            remaining = len(job_queue)
            self.get_logger().info(f'Jobs remaining: {remaining}')
            job = job_queue.pop(0)

            if isinstance(job, JointState):
                traj = self.ik_planner.plan_to_joints(job)
                if traj is None:
                    self.get_logger().error('Motion planning failed, aborting queue.')
                    return False
                self._execute_trajectory_blocking(traj.joint_trajectory)

            elif job == 'toggle_grip':
                self._toggle_gripper()

            else:
                self.get_logger().error(f'Unknown job type: {job}')

        self.get_logger().info('All jobs completed.')
        return True

    # ================================================================== #
    #                     LOW-LEVEL EXECUTION HELPERS                       #
    # ================================================================== #

    def _execute_trajectory_blocking(self, joint_traj: JointTrajectory):
        """Send trajectory to the UR action server and block until done."""
        self.get_logger().info('Waiting for trajectory action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory...')
        future = self.exec_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected!')
            return

        self.get_logger().info('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Trajectory execution complete.')

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            return
        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().info('Gripper toggled.')

    # ================================================================== #
    #                     MAIN RUN LOOP                                    #
    # ================================================================== #

    def run(self):
        """
        Main blocking loop:
          1. Look up AR tag → place location
          2. Wait for point-cloud object detection → pick location
          3. Execute pick-and-place
          4. If --continuous, loop back to step 2
        """

        # ── Step 1: find the AR tag ──
        self.get_logger().info('═══ Step 1: Looking up AR tag ═══')
        self.ar_tag_position = self.lookup_ar_tag(self.args.ar_marker, timeout=15.0)
        if self.ar_tag_position is None:
            self.get_logger().error('Cannot proceed without AR tag. Exiting.')
            return

        place_xyz = self.ar_tag_position
        self.get_logger().info(
            f'Place target (AR tag): ({place_xyz[0]:.3f}, {place_xyz[1]:.3f}, {place_xyz[2]:.3f})')

        while rclpy.ok():
            # ── Step 2: wait for an object to be detected ──
            self.busy = False
            self.cube_pose = None
            self.get_logger().info('═══ Step 2: Waiting for object detection via point cloud ═══')

            while rclpy.ok() and self.cube_pose is None:
                rclpy.spin_once(self, timeout_sec=0.2)

            if not rclpy.ok():
                break

            self.busy = True
            pick_xyz = np.array([
                self.cube_pose.point.x,
                self.cube_pose.point.y,
                self.cube_pose.point.z,
            ])
            self.get_logger().info(
                f'Pick target (point cloud): ({pick_xyz[0]:.3f}, {pick_xyz[1]:.3f}, {pick_xyz[2]:.3f})')

            # ── Step 3: plan & execute ──
            self.get_logger().info('═══ Step 3: Building pick-and-place queue ═══')
            queue = self._build_pick_place_queue(pick_xyz, place_xyz)
            if queue is None:
                self.get_logger().error('Failed to build job queue. Retrying...')
                continue

            self.get_logger().info('═══ Step 4: Executing pick-and-place ═══')
            success = self.execute_jobs(queue)
            self.pick_count += 1
            self.get_logger().info(f'Pick-and-place #{self.pick_count} {"succeeded" if success else "FAILED"}')

            if not self.args.continuous:
                self.get_logger().info('Single-shot mode. Done.')
                break

            self.get_logger().info('Continuous mode — looping for next object...')
            # Small delay to let the point cloud settle after the arm moves
            time.sleep(2.0)

        self.get_logger().info('═══ Finished ═══')


# ==================================================================== #
#                            ENTRY POINT                                #
# ==================================================================== #

def main(args=None):
    parser = argparse.ArgumentParser(
        description='Pick objects (point cloud) and place them at an AR tag')
    parser.add_argument('--ar_marker', type=int, default=0,
                        help='AR marker ID for the place target (default: 0)')
    parser.add_argument('--continuous', action='store_true',
                        help='Keep picking objects until none are left')
    parsed = parser.parse_args()

    rclpy.init(args=args)
    node = PickAndPlaceToAR(parsed)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
