# NOTE: realsense_launch is commented out in lab5_bringup.launch.py.
# Start the camera separately before running bringup:
#   ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true rgb_camera.color_profile:=1280x720x30
# To re-enable camera launch from bringup, uncomment realsense_launch in lab5_bringup.launch.py.

import json

# ROS Libraries
import tf2_ros
from std_srvs.srv import Trigger
from std_msgs.msg import String, Bool
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

from planning.ik import IKPlanner

# Image taking
import numpy as np
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import sys
import select
from scipy.spatial.transform import Rotation as R

# Per-class grasp offsets. Tune each entry so the gripper clears the object
# without hitting the table.
#   x_offset / y_offset      : lateral correction applied to the centroid
#   pre_grasp_z_offset        : height above centroid for the hover waypoint
#   grasp_z_offset            : height above centroid where the gripper closes
#   lift_z_offset             : height above centroid after grasping
PICK_OFFSETS = {
    "apple": {
        "x_offset":           0.01,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,
        "grasp_z_offset":     0.13,
        "lift_z_offset":      0.185,
    },
    "tomato": {
        "x_offset":           0.015,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,
        "grasp_z_offset":     0.145,
        "lift_z_offset":      0.185,
    },
    "cake": {
        "x_offset":           0.015,
        "y_offset":           0.00,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.20,
    },
    "strawberry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.145,
        "lift_z_offset":      0.20,
    },
    "cherry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.20,
    },
    "grape": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.144,
        "lift_z_offset":      0.20,
    },
    "blueberry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.145,
        "lift_z_offset":      0.20,
    },
    "chocolate": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.143,
        "lift_z_offset":      0.20,
    },
    
}

DEFAULT_OFFSETS = {
    "x_offset":           0.02,
    "y_offset":           0.005,
    "pre_grasp_z_offset": 0.16,
    "grasp_z_offset":     0.14,
    "lift_z_offset":      0.185,
}

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/detected_pick_point', self.cube_callback, 1)
        self.class_sub = self.create_subscription(String, '/detected_class', self.class_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        self.plate_sub = self.create_subscription(PointStamped, '/detected_plate_point', self.plate_callback, 10)
        self._tasks_sub = self.create_subscription(String, '/planned_pick_tasks', self._on_tasks, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.declare_parameter('skip_circler', False)
        self.pick_place_enabled = self.get_parameter('skip_circler').value
        self._start_sub = self.create_subscription(Bool, '/start_pick_place', self._on_start_pick_place, 1)
        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._orbit_done_pub = self.create_publisher(Bool, '/orbit_done', _latched)
        if self.pick_place_enabled:
            self.get_logger().info("skip_circler=true: pick-and-place active immediately.")
            _msg = Bool()
            _msg.data = True
            self._orbit_done_pub.publish(_msg)

        self.busy = False
        self.cube_pose = None
        self.plate_pose = None
        self.detected_class = ""
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.job_queue = [] # Entries should be of type either JointState or String('toggle_grip')

        # Joint-space observation pose.
        # TODO: move the arm to the desired observation position manually, then run:
        #   ros2 topic echo /joint_states --once
        # and fill in the 6 joint angles below in the order:
        #   shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        self._home_joints = [4.750492095947266, -1.4821723264506836, -2.0345261096954346, -1.2388786238482972, 1.5857458114624023, -3.075918738042013]
        self._going_home = False
        self._refining = False        # True from pre-pregrasp dispatch until refined centroid is used
        self._at_pre_pregrasp = False # True only after the arm has physically arrived at pre-pregrasp
        self.gripper_open = True  # assume physical gripper starts open

        self._task_queue = []  # ordered list from /planned_pick_tasks
        self._task_idx   = 0   # index of next task to execute

        # Image Taking
        self.subscription = self.create_subscription(Image,'/camera/camera/color/image_raw',self.photo_callback,10)
        self.bridge = CvBridge()
        self.image_count = 0
        self.frame = None



    def photo_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def take_photo(self):
        cv2.imwrite(f'captured_images1/captured_image_{self.image_count+1}.jpg', self.frame)
        self.get_logger().info(f'Saved image {self.image_count+1}')
        self.image_count +=1

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        if not hasattr(self, '_home_triggered'):
            self._home_triggered = True
            self._go_home()

    def _on_start_pick_place(self, msg: Bool):
        if msg.data and not self.pick_place_enabled:
            self.pick_place_enabled = True
            self.get_logger().info("Pick-and-place mode activated — moving to observation pose.")
            self._go_home()

    def class_callback(self, msg: String):
        self.detected_class = msg.data

    def _go_home(self):
        self._going_home = True
        self.busy = True
        goal = JointState()
        goal.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        goal.position = self._home_joints
        self.job_queue.append(goal)
        self.execute_jobs()
    def plate_callback(self, msg: PointStamped):
        self.plate_pose = msg
        self.get_logger().info("Plate updated")


    def cube_callback(self, pick_pose):
        if not self.pick_place_enabled:
            return

        self.get_logger().info(
            f"cube_callback fired | busy={self.busy} refining={self._refining} plate={self.plate_pose is not None}"
        )

        # Phase 2: arm has physically arrived at pre-pregrasp — use the fresh centroid.
        if self._refining and self._at_pre_pregrasp:
            self._on_refined_detection(pick_pose)
            return

        # Still moving to pre-pregrasp — ignore detections.
        if self._refining:
            return

        if self.busy:
            return

        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, cannot proceed")
            return

        if self.plate_pose is None:
            self.get_logger().error("No plate detected yet!")
            return

        self.busy = True
        self.cube_pose = pick_pose

        cx = pick_pose.point.x
        cy = pick_pose.point.y
        cz = pick_pose.point.z

        self.get_logger().info(
            f"Initial pick point: x={cx:.3f}, y={cy:.3f}, z={cz:.3f} "
            f"[class='{self.detected_class}'] — moving to pre-pregrasp"
        )

        # Move directly above the object so the camera gets a centered, undistorted
        # view before committing to the final grasp trajectory.
        pre_pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state, cx, cy, cz + 0.5
        )

        if pre_pre_grasp_joints is None:
            self.get_logger().error(
                f"IK failed for pre-pregrasp ({cx:.3f}, {cy:.3f}, {cz + 0.5:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        self._refining = True
        self.job_queue.append(pre_pre_grasp_joints)
        self.execute_jobs()

    def _on_refined_detection(self, pick_pose):
        """Called once the arm has reached pre-pregrasp and a fresh centroid is available."""
        self._refining = False
        self._at_pre_pregrasp = False

        cx = pick_pose.point.x
        cy = pick_pose.point.y
        cz = pick_pose.point.z

        drop_x = self.plate_pose.point.x
        drop_y = self.plate_pose.point.y
        drop_z = self.plate_pose.point.z

        offsets = PICK_OFFSETS.get(self.detected_class, DEFAULT_OFFSETS)
        self.get_logger().info(
            f"Refined pick point: x={cx:.3f}, y={cy:.3f}, z={cz:.3f} "
            f"[class='{self.detected_class}']"
        )
        self.get_logger().info(
            f"Place point: x={drop_x:.3f}, y={drop_y:.3f}, z={drop_z:.3f}"
        )

        x_offset           = offsets["x_offset"]
        y_offset           = offsets["y_offset"]
        pre_grasp_z_offset = offsets["pre_grasp_z_offset"]
        grasp_z_offset     = offsets["grasp_z_offset"]
        lift_z_offset      = offsets["lift_z_offset"]

        # Seed from current joint state (arm is now at pre-pregrasp position).
        pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state, cx + x_offset, cy + y_offset, cz + pre_grasp_z_offset
        )
        grasp_joints = self.ik_planner.compute_ik(
            pre_grasp_joints, cx + x_offset, cy + y_offset, cz + grasp_z_offset
        )
        lift_joints = self.ik_planner.compute_ik(
            grasp_joints, cx + x_offset, cy + y_offset, cz + lift_z_offset
        )
        drop_pre_joints = self.ik_planner.compute_ik(
            lift_joints, drop_x, drop_y, drop_z + 0.2
        )
        drop_joints = self.ik_planner.compute_ik(
            drop_pre_joints, drop_x, drop_y, drop_z + 0.15
        )

        if pre_grasp_joints is None:
            self.get_logger().error(
                f"IK failed for pre-grasp ({cx + x_offset:.3f}, {cy + y_offset:.3f}, {cz + pre_grasp_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        if grasp_joints is None:
            self.get_logger().error(
                f"IK failed for grasp ({cx + x_offset:.3f}, {cy + y_offset:.3f}, {cz + grasp_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        if lift_joints is None:
            self.get_logger().error(
                f"IK failed for lift ({cx + x_offset:.3f}, {cy + y_offset:.3f}, {cz + lift_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        self.job_queue.append(pre_grasp_joints)
        self.job_queue.append(grasp_joints)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(lift_joints)
        self.job_queue.append(drop_pre_joints)
        self.job_queue.append(drop_joints)
        self.job_queue.append('toggle_grip')
        self.execute_jobs()


    def _on_tasks(self, msg: String):
        try:
            self._task_queue = json.loads(msg.data)
            self._task_idx   = 0
            self.get_logger().info(f"Received {len(self._task_queue)} pre-computed pick tasks.")
        except Exception as e:
            self.get_logger().error(f"Failed to parse /planned_pick_tasks: {e}")

    def _start_precomputed_task(self):
        """Execute the next task from the pre-computed list (no live-detection refinement needed)."""
        task = self._task_queue[self._task_idx]
        self._task_idx += 1

        name = task['object_name']
        pos  = task['position']
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

        self.detected_class = name
        self.busy = True

        self.get_logger().info(
            f"Pre-computed task [{self._task_idx}/{len(self._task_queue)}]: "
            f"{name} at ({cx:.3f}, {cy:.3f}, {cz:.3f})"
        )

        if self.plate_pose is None:
            self.get_logger().error("No plate pose — cannot execute task.")
            self.busy = False
            return

        drop_x = self.plate_pose.point.x
        drop_y = self.plate_pose.point.y
        drop_z = self.plate_pose.point.z

        offsets            = PICK_OFFSETS.get(name, DEFAULT_OFFSETS)
        x_offset           = offsets["x_offset"]
        y_offset           = offsets["y_offset"]
        pre_grasp_z_offset = offsets["pre_grasp_z_offset"]
        grasp_z_offset     = offsets["grasp_z_offset"]
        lift_z_offset      = offsets["lift_z_offset"]

        pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state, cx + x_offset, cy + y_offset, cz + pre_grasp_z_offset)
        grasp_joints = self.ik_planner.compute_ik(
            pre_grasp_joints, cx + x_offset, cy + y_offset, cz + grasp_z_offset)
        lift_joints = self.ik_planner.compute_ik(
            grasp_joints, cx + x_offset, cy + y_offset, cz + lift_z_offset)
        drop_pre_joints = self.ik_planner.compute_ik(
            lift_joints, drop_x, drop_y, drop_z + 0.2)
        drop_joints = self.ik_planner.compute_ik(
            drop_pre_joints, drop_x, drop_y, drop_z + 0.15)

        if pre_grasp_joints is None or grasp_joints is None or lift_joints is None:
            self.get_logger().error(f"IK failed for pre-computed task '{name}' — skipping.")
            self.busy = False
            # Try the next task rather than hanging
            if self._task_idx < len(self._task_queue):
                self._start_precomputed_task()
            return

        self.job_queue.extend([
            pre_grasp_joints,
            grasp_joints,
            'toggle_grip',
            lift_joints,
            drop_pre_joints,
            drop_joints,
            'toggle_grip',
        ])
        self.execute_jobs()

    def execute_jobs(self):
        if not self.job_queue:
            if self._refining:
                self._at_pre_pregrasp = True
                self.get_logger().info("Pre-pregrasp reached. Waiting for refined centroid...")
                return

            if not self._going_home:
                self.get_logger().info("Pick/place complete → going home")
                self.cube_pose = None
                self._going_home = True
                self._go_home()
                return

            # We arrived home
            self._going_home = False
            self.busy = False
            self.cube_pose = None

            # If there are more pre-computed tasks, start the next one automatically
            if self._task_idx < len(self._task_queue):
                self.get_logger().info(
                    f"Home reached. Starting next task ({self._task_idx + 1}/{len(self._task_queue)})."
                )
                self._start_precomputed_task()
                return

            self.get_logger().info("Home reached. Ready for detection.")
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):

            traj = self.ik_planner.plan_to_joints(next_job, start_joint_state=self.joint_state)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                self.busy = False
                self.cube_pose = None
                self.job_queue.clear()
                return

            self.get_logger().info("Planned to position")

            self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.gripper_open = not self.gripper_open
        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
