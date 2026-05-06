# NOTE: realsense_launch is commented out in lab5_bringup.launch.py.
# Start the camera separately before running bringup:
#   ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true rgb_camera.color_profile:=1280x720x30
# To re-enable camera launch from bringup, uncomment realsense_launch in lab5_bringup.launch.py.

# ROS Libraries
import tf2_ros
from std_srvs.srv import Trigger
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

from planning.ik import IKPlanner

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
        "grasp_z_offset":     0.125,
        "lift_z_offset":      0.185,
    },
    "tomato": {
        "x_offset":           0.015,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.185,
    },
    "cake": {
        "x_offset":           -0.03,
        "y_offset":           -0.05,
        "pre_grasp_z_offset": 0.15,
        "grasp_z_offset":     0.1,
        "lift_z_offset":      0.20,
    },
    #not tested
    "strawberry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.20,
    },
    #not gonna use for now
    "cherry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.20,
    },
    #not tested
    "blueberry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
        "lift_z_offset":      0.20,
    },
    #not tested
    "grape": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.14,
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
    """
    Top-level ROS 2 node and pick-and-place state machine for the UR7e arm.

    Subscribes to vision detections (object centroid, object class, plate centroid),
    computes a full IK plan upfront, then drives the robot through a fixed 7-step
    grasp sequence: hover → descend → grip → lift → move over plate → lower → release.

    The `busy` flag acts as a binary semaphore: once a pick cycle starts, incoming
    detection callbacks are silently dropped until the arm returns to home.
    """
    def __init__(self):
        super().__init__('cube_grasp')

        # /detected_pick_point  : 3D centroid of the target object in base_link frame;
        #                         QoS=1 so we always take the most recent detection.
        #                         IMPORTANT: /detected_class must be published first, or
        #                         per-class offsets silently fall back to DEFAULT_OFFSETS.
        # /detected_class       : YOLO class label string (e.g. "apple") for offset lookup.
        # /detected_plate_point : 3D centroid of the drop plate; required before any pick.
        # /joint_states         : current arm configuration; seeds IK and triggers home on startup.
        self.cube_pub = self.create_subscription(PointStamped, '/detected_pick_point', self.cube_callback, 1)
        self.class_sub = self.create_subscription(String, '/detected_class', self.class_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        self.plate_sub = self.create_subscription(PointStamped,'/detected_plate_point',self.plate_callback,10)

        # Sends MoveIt-planned joint trajectories to the UR hardware controller.
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        
        # busy: True while the arm is executing a pick/place or returning home.
        #       Prevents cube_callback from queueing a second cycle mid-motion.
        self.busy = False
        self.cube_pose = None
        self.plate_pose = None
        self.detected_class = ""
        self.current_plan = None  # unused — vestigial from earlier iteration
        self.joint_state = None

        self.ik_planner = IKPlanner()

        # Sequential job queue. Each entry is either:
        #   JointState  → plan and execute a trajectory to those joint angles
        #   'toggle_grip' → call the gripper service to open/close
        # Jobs are popped one at a time; each async completion callback fires the next.
        self.job_queue = []

        # Joint-space observation pose.
        # TODO: move the arm to the desired observation position manually, then run:
        #   ros2 topic echo /joint_states --once
        # and fill in the 6 joint angles below in the order:
        #   shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        self._home_joints = [4.723223686218262, -1.5760652027525843, -2.1608810424804688, -0.9934399288943787, 1.579869031906128, -3.142892901097433]
        self._going_home = False
        self._refining = False        # True from pre-pregrasp dispatch until refined centroid is used
        self._at_pre_pregrasp = False # True only after the arm has physically arrived at pre-pregrasp
        self.gripper_open = True  # assume physical gripper starts open
    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        # _home_triggered is created lazily on first callback, so _go_home() fires
        # exactly once at startup. This ensures the arm reaches the observation pose
        # before any detection callbacks can start a pick cycle.
        if not hasattr(self, '_home_triggered'):
            self._home_triggered = True
            self._go_home()

    def class_callback(self, msg: String):
        self.detected_class = msg.data

    def _go_home(self):
        # _going_home=True tells execute_jobs() that the upcoming empty-queue event
        # means "we arrived home" rather than "pick/place just finished" — prevents
        # an infinite loop of home→done→home→done.
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

        # Guard: drop this detection if the arm is already moving.
        if self.busy:
            return

        # Guard: IK needs the current joint configuration as a seed; can't compute
        # a solution without knowing where the arm currently is.
        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, cannot proceed")
            return

        # Guard: the plate is mandatory — if it hasn't been detected yet, abort rather
        # than place the object somewhere arbitrary on the table.
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

        # 7-step pick-and-place sequence using the refined centroid:
        #   pre_grasp_joints  → hover above object
        #   grasp_joints      → descend to grasp height
        #   'toggle_grip'     → close gripper
        #   lift_joints       → lift object clear of table
        #   drop_pre_joints   → move above plate (clearance)
        #   drop_joints       → lower onto plate
        #   'toggle_grip'     → open gripper (release)
        self.job_queue.append(pre_grasp_joints)
        self.job_queue.append(grasp_joints)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(lift_joints)
        self.job_queue.append(drop_pre_joints)
        self.job_queue.append(drop_joints)
        self.job_queue.append('toggle_grip')
        self.execute_jobs()


    def execute_jobs(self):
        # Empty queue has three meanings, distinguished by _refining / _going_home:
        #   _refining       → pre-pregrasp reached; wait for cube_callback to deliver refined centroid
        #   not _going_home → pick/place just finished; automatically chain into _go_home()
        #   _going_home     → arm has arrived at the home pose; clear busy and wait for detections
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

            # Arm is home — clear the busy flag so cube_callback can accept new detections.
            self._going_home = False
            self.busy = False
            self.cube_pose = None

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
        # Blocking wait: gripper state must be confirmed before the next motion job runs.
        # We can't fire the next trajectory while the gripper is still mid-toggle.
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.gripper_open = not self.gripper_open
        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj):
        # Async callback chain — none of these calls block the ROS spin loop:
        #   send_goal_async  →  _on_goal_sent   (fired when controller accepts/rejects goal)
        #   get_result_async →  _on_exec_done   (fired when trajectory finishes executing)
        #   _on_exec_done    →  execute_jobs()  (advances to the next job in the queue)
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        # Step 2 of the async chain: confirm the controller accepted the goal,
        # then register the completion callback to wait for execution to finish.
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        # Step 3: trajectory finished — advance the job queue.
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
