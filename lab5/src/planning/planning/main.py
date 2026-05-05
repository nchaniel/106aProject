# NOTE: realsense_launch is commented out in lab5_bringup.launch.py.
# Start the camera separately before running bringup:
#   ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true rgb_camera.color_profile:=1280x720x30
# To re-enable camera launch from bringup, uncomment realsense_launch in lab5_bringup.launch.py.

# ROS Libraries
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
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,
        "grasp_z_offset":     0.12,
        "lift_z_offset":      0.185,
    },
    "tomato": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.16,
        "grasp_z_offset":     0.12,
        "lift_z_offset":      0.185,
    },
    "cake": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.20,
        "grasp_z_offset":     0.15,
        "lift_z_offset":      0.20,
    },
    "strawberry": {
        "x_offset":           0.02,
        "y_offset":           0.005,
        "pre_grasp_z_offset": 0.21,
        "grasp_z_offset":     0.15,
        "lift_z_offset":      0.20,
    },
}

DEFAULT_OFFSETS = {
    "x_offset":           0.02,
    "y_offset":           0.005,
    "pre_grasp_z_offset": 0.16,
    "grasp_z_offset":     0.12,
    "lift_z_offset":      0.185,
}

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/detected_pick_point', self.cube_callback, 1)
        self.class_sub = self.create_subscription(String, '/detected_class', self.class_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        
        self.busy = False
        self.cube_pose = None
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
        self._home_joints = [4.723223686218262, -1.5760652027525843, -2.1608810424804688, -0.9934399288943787, 1.579869031906128, -3.142892901097433]
        self._going_home = False
        self.gripper_open = True  # assume physical gripper starts open

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        if not hasattr(self, '_home_triggered'):
            self._home_triggered = True
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

    def cube_callback(self, pick_pose):
        if self.busy or self.cube_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, cannot proceed")
            return

        self.busy = True
        self.cube_pose = pick_pose

        # Detected point in base_link frame
        cx = self.cube_pose.point.x
        cy = self.cube_pose.point.y
        cz = self.cube_pose.point.z

        offsets = PICK_OFFSETS.get(self.detected_class, DEFAULT_OFFSETS)
        self.get_logger().info(
            f"Using detected pick point: x={cx:.3f}, y={cy:.3f}, z={cz:.3f} "
            f"[class='{self.detected_class}']"
        )

        x_offset           = offsets["x_offset"]
        y_offset           = offsets["y_offset"]
        pre_grasp_z_offset = offsets["pre_grasp_z_offset"]
        grasp_z_offset     = offsets["grasp_z_offset"]
        lift_z_offset      = offsets["lift_z_offset"]

        # Fixed drop-off location for v1
        drop_x = 0.45
        drop_y = 0.40
        drop_z = 0.20

        
        # 1. Move above detected object
        pre_grasp_joints = self.ik_planner.compute_ik(self.joint_state, cx + x_offset, cy + y_offset, cz + pre_grasp_z_offset)

        # 2. Move down to grasp object
        grasp_joints = self.ik_planner.compute_ik(
             self.joint_state,
             cx + x_offset,
             cy,
             cz + grasp_z_offset
        )

        # 3. Lift after grasping
        lift_joints = self.ik_planner.compute_ik(
            self.joint_state,
            cx + x_offset,
            cy,
            cz + lift_z_offset
        )

        # 4. Move above drop location
        drop_pre_joints = self.ik_planner.compute_ik(
            self.joint_state,
            drop_x,
            drop_y,
            drop_z + 0.12
        )

        # 5. Move down to drop location
        drop_joints = self.ik_planner.compute_ik(
            self.joint_state,
            drop_x,
            drop_y,
            drop_z
        )

        if pre_grasp_joints is None:
            self.get_logger().error(
                f"IK failed for pre-grasp target "
                f"({cx + x_offset:.3f}, {cy:.3f}, {cz + pre_grasp_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        if grasp_joints is None:
            self.get_logger().error(
                f"IK failed for grasp target "
                f"({cx + x_offset:.3f}, {cy:.3f}, {cz + grasp_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        if lift_joints is None:
            self.get_logger().error(
                f"IK failed for lift target "
                f"({cx + x_offset:.3f}, {cy:.3f}, {cz + lift_z_offset:.3f})"
            )
            self.busy = False
            self.cube_pose = None
            return

        self.job_queue.append(pre_grasp_joints)
        self.job_queue.append(grasp_joints)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(lift_joints)
        #self.job_queue.append(drop_pre_joints)
        #self.job_queue.append(drop_joints)
        self.job_queue.append('toggle_grip')
        self.execute_jobs()


    def execute_jobs(self):
        if not self.job_queue:
            if self._going_home:
                self._going_home = False
                self.busy = False
                self.cube_pose = None
                self.get_logger().info("Home reached. Ready for detection.")
            else:
                self.get_logger().info("Pick complete. Staying at final position.")
                self.busy = False
                self.cube_pose = None
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
