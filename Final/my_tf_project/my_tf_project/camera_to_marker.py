import rclpy
from rclpy.node import Node

import tf2_ros
import tf2_geometry_msgs
import math
from moveit.planning import MoveItPy

from geometry_msgs.msg import PoseStamped


class CameraToMarker(Node):

    def __init__(self):
        super().__init__('camera_to_marker')

        # ---------------- TF setup ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- MoveIt ----------------
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'
            ],
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator"
        )

        # ---------------- control ----------------
        self.goal_sent = False
        self.timer = self.create_timer(1.0, self.run)

    # ------------------------------------------------------------
    # Desired camera pose relative to marker
    # ------------------------------------------------------------
    def desired_camera_pose_in_marker(self):
        pose = PoseStamped()
        pose.header.frame_id = "ar_marker_8"

        # 20 cm above marker
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.2

        # look down

        pose.pose.orientation.x = 1.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

        return pose

    # ------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------
    def run(self):

        if self.goal_sent:
            return

        try:
            # ---------------------------
            # 1. marker → base_link
            # ---------------------------
            base_to_marker = self.tf_buffer.lookup_transform(
                "base_link",
                "ar_marker_8",
                rclpy.time.Time()
            )

            # ---------------------------
            # 2. desired pose in marker frame
            # ---------------------------
            marker_goal = self.desired_camera_pose_in_marker()

            # ---------------------------
            # 3. convert to base frame
            # ---------------------------
            base_goal = tf2_geometry_msgs.do_transform_pose(
                marker_goal,
                base_to_marker
            )

            # ---------------------------
            # 4. convert base → tool0 (FINAL STEP)
            # ---------------------------
            tool0_goal = self.tf_buffer.transform(
                base_goal,
                "tool0",
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # ---------------------------
            # 5. send to MoveIt
            # ---------------------------
            self.move_to_pose(
                position=tool0_goal.pose.position,
                quat_xyzw=(
                    tool0_goal.pose.orientation.x,
                    tool0_goal.pose.orientation.y,
                    tool0_goal.pose.orientation.z,
                    tool0_goal.pose.orientation.w
                )
            )

            self.goal_sent = True
            self.get_logger().info("Goal sent to MoveIt")

        except Exception as e:
            self.get_logger().warn(f"TF not ready or failed: {e}")


def main():
    rclpy.init()
    node = CameraToMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()