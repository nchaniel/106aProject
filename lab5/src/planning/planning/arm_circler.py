import sys
import select
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2

from planning.ik import IKPlanner


class ArmCircler(Node):
    def __init__(self):
        super().__init__('arm_circler')

        self._plate_sub = self.create_subscription(
            PointStamped, '/detected_plate_point', self._plate_callback, 1
        )
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 1
        )
        self._img_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._photo_callback, 10
        )

        self._start_pub = self.create_publisher(Bool, '/start_pick_place', 1)

        self._exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.ik_planner = IKPlanner()
        self.job_queue = []

        self.joint_state = None
        self._orbit_triggered = False
        self._orbit_done = False

        self._bridge = CvBridge()
        self._frame = None
        self._image_count = 0

        # Non-blocking stdin check — fires after orbit is done
        self._stdin_timer = self.create_timer(0.1, self._check_stdin)

    def _joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def _photo_callback(self, msg: Image):
        self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _take_photo(self):
        if self._frame is not None:
            cv2.imwrite(f'captured_images/circler_{self._image_count + 1}.jpg', self._frame)
            self.get_logger().info(f'Saved circler image {self._image_count + 1}')
            self._image_count += 1

    def _plate_callback(self, msg: PointStamped):
        if self._orbit_triggered:
            return
        if self.joint_state is None:
            return

        self._orbit_triggered = True

        cx = msg.point.x
        cy = msg.point.y
        cz = msg.point.z

        self.get_logger().info(
            f"Plate centroid: x={cx:.3f}, y={cy:.3f}, z={cz:.3f} — building orbit"
        )

        tilt_distance = 0.12
        radius = 0.15
        height = 0.3
        num_points = 20

        pose_data = [np.array([cx, cy, cz, 0, 0, 0, 1])]

        for row in range(2):
            angles = np.linspace(-np.pi / 4, np.pi + np.pi / 8, num_points)
            if row == 1:
                angles = np.flip(angles)
            for theta in angles:
                tx = cx + radius * np.cos(theta)
                ty = cy + radius * np.sin(theta)
                tz = cz + height

                rot_z = R.from_euler('z', theta - np.pi / 2)
                rot_y = R.from_euler('y', np.pi)
                tilt_angle = np.arctan2(radius + tilt_distance, height)
                rot_x = R.from_euler('x', tilt_angle)
                combined_rot = rot_z * rot_y * rot_x
                q = combined_rot.as_quat()  # [x, y, z, w]

                self.get_logger().info(
                    f"Waypoint: pos=({tx:.3f},{ty:.3f},{tz:.3f}) "
                    f"quat=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})"
                )
                pose_data.append(np.array([tx, ty, tz, q[0], q[1], q[2], q[3]]))

                ik_sol = self.ik_planner.compute_ik(
                    self.joint_state, tx, ty, tz,
                    qx=q[0], qy=q[1], qz=q[2], qw=q[3]
                )
                if ik_sol:
                    self.job_queue.append(ik_sol)
                else:
                    self.get_logger().warn(f"IK failed for theta={theta:.2f}")

            height = 0.2
            tilt_distance += 0.05

        final_ik = self.ik_planner.compute_ik(
            self.joint_state, cx, cy - 0.15, cz + 0.35
        )
        if final_ik:
            self.job_queue.append(final_ik)

        np.save('poses.npy', np.array(pose_data))
        self.get_logger().info(f"Orbit has {len(self.job_queue)} waypoints. Starting execution.")

        if self.job_queue:
            self._execute_jobs()

    def _execute_jobs(self):
        if not self.job_queue:
            self._orbit_done = True
            self.get_logger().info("Orbit complete. Press Enter to begin pick-and-place.")
            return

        self.get_logger().info(f"Executing waypoint ({len(self.job_queue)} remaining).")
        next_job = self.job_queue.pop(0)

        traj = self.ik_planner.plan_to_joints(next_job, start_joint_state=self.joint_state)
        if traj is None:
            self.get_logger().error("Failed to plan to waypoint — skipping.")
            self._execute_jobs()
            return

        self._execute_joint_trajectory(traj.joint_trajectory)

    def _execute_joint_trajectory(self, joint_traj):
        self._exec_ac.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        send_future = self._exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected by controller.')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            future.result()
            self._take_photo()
            self._execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')

    def _check_stdin(self):
        if not self._orbit_done:
            return
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            self.get_logger().info("Enter pressed — switching to pick-and-place mode.")
            msg = Bool()
            msg.data = True
            self._start_pub.publish(msg)
            self._stdin_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ArmCircler()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
