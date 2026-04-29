# ROS Libraries
from std_srvs.srv import Trigger
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped,TransformStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

from cv_bridge import CvBridge
import cv2

from planning.ik import IKPlanner

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/cube_pose', self.cube_callback, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        
        # -------------------------------------
        self.bridge = CvBridge()
        self.image_count = 0
        self.frame = None
        self.first_position_done = 0

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.photo_callback,
            10
        )
    def get_tool_camera_transform(self):
        R_flip = R.from_euler('z', np.pi).as_matrix()

        T_tool_camera = np.eye(4)
        T_tool_camera[:3, :3] = R_flip
        T_tool_camera[:3, 3] = np.array([-0.025, 0.13, 0.0])
        return T_tool_camera
    def photo_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    def take_photo(self):
        cv2.imwrite(f'captured_image_{self.image_count+1}.jpg', self.frame)
        self.get_logger().info(f'Saved image {self.image_count+1}')
        self.image_count +=1


    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def cube_callback(self, cube_pose):
        if self.cube_pose is not None:
            return
        if self.joint_state is None:
            return
        '''
        tx0, ty0, tz0 = 0.4, 0.0, 0.5  # example safe point
        qx0, qy0, qz0, qw0 = 0, 1, 0, 0  # pointing downward
        
        if self.first_position_done == 0:
            print("trying to solve first pose")
            ik_start = self.ik_planner.compute_ik(
                self.joint_state,
                tx0, ty0, tz0,
                qx=qx0, qy=qy0, qz=qz0, qw=qw0
            )
            self.job_queue.append(ik_start)
            print("first pose sent")
            self.first_position_done = 1
        '''

        self.cube_pose = cube_pose
        cx, cy, cz = cube_pose.point.x, cube_pose.point.y, cube_pose.point.z
        
        # --- Orbit Parameters ---
        radius = 0.25      # Distance from the cube in meters
        height = 0.3      # Height above the cube
        num_points = 10    # Number of waypoints for a smooth arc
        
        self.get_logger().info(f"Generating 180-degree orbit around: {cx}, {cy}, {cz}")


        # Generate angles from 0 to Pi (180 degrees)
        for theta in np.linspace(-4*np.pi/5, 6*np.pi/5, num_points):
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
            '''
            rot_z = R.from_euler('z', theta-np.pi/2) 
            tilt_angle = np.arctan2(radius, height)
            rot_y = R.from_euler('y', np.pi) # Points the gripper/camera down
            rot_x = R.from_euler('x', tilt_angle)
            combined_rot = rot_z * rot_y * rot_x
            q = combined_rot.as_quat() # [x, y, z, w]
            # --- Step 1: camera position in world ---
            '''
            camera_pos = np.array([tx, ty, tz])

            # --- Step 2: build camera orientation (look-at cube) ---
            target = np.array([cx, cy, cz])

            z_axis = target - camera_pos
            z_axis = z_axis / np.linalg.norm(z_axis)

            up = np.array([0, 0, 1])

            x_axis = np.cross(up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            y_axis = np.cross(z_axis, x_axis)

            R_camera_world = np.vstack([x_axis, y_axis, z_axis]).T

            # --- Step 3: build full camera pose in world ---
            T_camera_world = np.eye(4)
            T_camera_world[:3, :3] = R_camera_world
            T_camera_world[:3, 3] = camera_pos

            T_tool_camera = self.get_tool_camera_transform()
            T_tool_world = T_camera_world @ np.linalg.inv(T_tool_camera)

            position = T_tool_world[:3, 3]
            rotation_matrix = T_tool_world[:3, :3]

            q = R.from_matrix(rotation_matrix).as_quat()




            # 3. Compute IK for this waypoint
            ik_sol = self.ik_planner.compute_ik(
                self.joint_state,
                position[0], position[1], position[2],
                qx=q[0], qy=q[1], qz=q[2], qw=q[3]
            )

            if ik_sol:
                self.job_queue.append(ik_sol)
            else:
                self.get_logger().warn(f"IK failed for theta {theta}")
        

        # Start execution
        if self.job_queue:
            self.execute_jobs()



    def execute_jobs(self):
        if not self.job_queue:
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
            self.take_photo()
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