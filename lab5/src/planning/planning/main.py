# ROS Libraries
from std_srvs.srv import Trigger
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

from planning.ik import IKPlanner

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/detected_pick_point', self.cube_callback, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        
        self.busy = False
        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.job_queue = [] # Entries should be of type either JointState or String('toggle_grip')
        self.obs_pose_done = False

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        if not self.obs_pose_done and not hasattr(self, '_obs_triggered'):
            self._obs_triggered = True
            obs_joints = self.ik_planner.compute_ik(
                self.joint_state, 0.13, 0.474, 0.619,
                qx=0.002, qy=1.00, qz=-0.003, qw=-0.011
            )
            if obs_joints:
                self.job_queue.append(obs_joints)
                self.execute_jobs()
            else:
                self.get_logger().error("IK failed for observation pose")

    def cube_callback(self, pick_pose):
        # Ignore new detections while already executing a pick/place
        if self.cube_pose is not None or getattr(self, "busy", False):
            return

        if not self.obs_pose_done:
            self.get_logger().info("Observation pose not complete, skipping detection")
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.busy = True
        self.cube_pose = pick_pose

        # Detected point in base_link frame
        cx = pick_pose.point.x
        cy = pick_pose.point.y
        cz = pick_pose.point.z

        self.get_logger().info(
            f"Using detected pick point: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}"
        )

        # Safety offsets
        x_offset = -0.015
        pre_grasp_z_offset = 0.185
        grasp_z_offset = 0.092
        lift_z_offset = 0.185

        # Fixed drop-off location for v1
        drop_x = 0.45
        drop_y = 0.20
        drop_z = 0.20

        # 1. Move above detected object
        pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state,
            cx + x_offset,
            cy,
            cz + pre_grasp_z_offset
        )

        # # 2. Move down to grasp object
        # grasp_joints = self.ik_planner.compute_ik(
        #     self.joint_state,
        #     cx + x_offset,
        #     cy,
        #     cz + grasp_z_offset
        # )

        # # 3. Lift after grasping
        # lift_joints = self.ik_planner.compute_ik(
        #     self.joint_state,
        #     cx + x_offset,
        #     cy,
        #     cz + lift_z_offset
        # )

        # # 4. Move above drop location
        # drop_pre_joints = self.ik_planner.compute_ik(
        #     self.joint_state,
        #     drop_x,
        #     drop_y,
        #     drop_z + 0.12
        # )

        # # 5. Move down to drop location
        # drop_joints = self.ik_planner.compute_ik(
        #     self.joint_state,
        #     drop_x,
        #     drop_y,
        #     drop_z
        # )

        # # Check IK success before queueing
        # if not all([pre_grasp_joints, grasp_joints, lift_joints, drop_pre_joints, drop_joints]):
        #     self.get_logger().error("IK failed for one or more pick/place poses")
        #     self.busy = False
        #     self.cube_pose = None
        #     return

        self.job_queue.append(pre_grasp_joints)
        # self.job_queue.append(grasp_joints)
        # self.job_queue.append("toggle_grip")
        # self.job_queue.append(lift_joints)
        # self.job_queue.append(drop_pre_joints)
        # self.job_queue.append(drop_joints)
        # self.job_queue.append("toggle_grip")
        # self.job_queue.append(drop_pre_joints)

        self.execute_jobs()


    def execute_jobs(self):
        if not self.job_queue:
            if not self.obs_pose_done:
                self.obs_pose_done = True
                self.get_logger().info("Observation pose reached. Waiting for cube detection...")
                return
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):

            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
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

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
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
