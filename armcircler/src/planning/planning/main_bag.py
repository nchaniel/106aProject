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
from sensor_msgs.msg import JointState, Image
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np


from planning.ik import IKPlanner

import subprocess
import signal

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/cube_pose', self.cube_callback, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.job_queue = [] # Entries should be of type either JointState or String('toggle_grip')
        
        self.bag_process = None
        self.bag_started = False

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def cube_callback(self, cube_pose):
        if self.cube_pose is not None:
            return
        if self.joint_state is None:
            return

        #old_poses = np.load("poses.npy")
        #self.get_logger().info(f"Poses: {old_poses}")

        self.cube_pose = cube_pose
        cx, cy, cz = cube_pose.point.x, cube_pose.point.y, cube_pose.point.z
        cx -= 0.02  # tweak sign after testing
        cy += 0
        
        # --- Orbit Parameters ---
        tilt_distance = 0.12
        radius = 0.15     # Distance from the cube in meters
        height = 0.3      # Height above the cube
        num_points = 20    # Number of waypoints for a smooth arc
        
        self.get_logger().info(f"Generating 180-degree orbit around: {cx}, {cy}, {cz}")
        #pose_data = [cx, cy, cz, 0, 0, 0, 1]

        # Generate angles from 0 to Pi (180 degrees)
        for row in range(2):
            angles = np.linspace(-np.pi/4, np.pi+np.pi/8, num_points)
            if row == 1:
                angles = np.flip(angles)
            for theta in angles:
                # 1. Calculate Cartesian Position (Orbiting in the XY plane)
                tx = cx + radius * np.cos(theta)
                ty = cy + radius * np.sin(theta)
                tz = cz + height


                # 2. Calculate "Look-At" Orientation
                # We want the camera's Z-axis (optical axis) to point at the cube.
                # A simple approach for a top-down orbit:
                # Rotate around Z by (theta + pi) to face inward, 
                # then tilt down (pi around Y) to look at the table.
                
                # This creates a rotation matrix representing the camera facing the center
                rot_z = R.from_euler('z', theta-np.pi/2) 
                rot_y = R.from_euler('y', np.pi) # Points the gripper/camera down
                tilt_angle = np.arctan2(radius + tilt_distance, height)
                rot_x = R.from_euler('x', tilt_angle)
                combined_rot = rot_z * rot_y * rot_x
                q = combined_rot.as_quat() # [x, y, z, w]'''


                # 3. Compute IK for this waypoint
                self.get_logger().info(f"Pose: \nPosition:{tx}, {ty}, {tz}\nOrientation:{q[0]},{q[1]},{q[2]},{q[3]}")
                current_pose = np.array([tx, ty, tz, q[0], q[1], q[2], q[3]])
                #pose_data.append(current_pose)

                ik_sol = self.ik_planner.compute_ik(
                    self.joint_state, 
                    tx, ty, tz, 
                    qx=q[0], qy=q[1], qz=q[2], qw=q[3]
                )

                if ik_sol:
                    self.job_queue.append(ik_sol)
                else:
                    self.get_logger().warn(f"IK failed for theta {theta}")
            height = 0.2
            tilt_distance += 0.05

        ik_sol = self.ik_planner.compute_ik(self.joint_state, cx, cy-0.15, cz+0.35)
        if ik_sol:
            self.job_queue.append(ik_sol)

        #final_poses = np.array(pose_data)
        #np.save('poses.npy', final_poses)

        # Start execution
        if self.job_queue:
            self.execute_jobs()



    def execute_jobs(self):
        if not self.bag_started:
            self.get_logger().info("Starting rosbag recording")

            self.bag_process = subprocess.Popen([
                "ros2", "bag", "record",
                "/camera/camera/color/image_raw",
                "/camera/camera/depth/image_raw",
                "/joint_states",
                "/cube_pose",
                "/tf",
                "/tf_static"
            ])

            self.bag_started = True
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")

            if self.bag_process is not None:
                self.get_logger().info("Stopping rosbag")
                self.bag_process.send_signal(signal.SIGINT)
                try:
                    self.bag_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.get_logger().warn("Forcing bag shutdown")
                    self.bag_process.kill()

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