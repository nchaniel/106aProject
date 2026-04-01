#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import transforms3d.euler as euler
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist, PointStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from plannedcntrl.trajectory import plan_curved_trajectory  # Your existing Bezier planner
import time

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')

        # Publisher and TF setup
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Controller gains
        self.Kp = np.diag([0.8, 1.5])
        self.Ki = np.diag([0.0, 0.0])
        self.Kd = np.diag([0.0, 0.1])

        # Subscriber
        self.create_subscription(PointStamped, '/goal_point', self.planning_callback, 10)
        self.timer = self.create_timer(0.5, self.control_loop)

        self.trajectory = None
        self.traj_index = 0
        self.x_i_err = 0.0
        self.yaw_i_err = 0.0
        self.prev_x_err = None
        self.prev_yaw_err = None

        self.get_logger().info('TurtleBot controller node initialized.')

    # ------------------------------------------------------------------
    # Main waypoint controller
    # ------------------------------------------------------------------

    def control_loop(self):
        if not self.trajectory:
            return

        if self.traj_index >= len(self.trajectory):
            self.trajectory = None
            self.traj_index = 0
            self.x_i_err = 0.0
            self.yaw_i_err = 0.0
            self.prev_x_err = None
            self.prev_yaw_err = None
            self.pub.publish(Twist())
            return

        waypoint = self.trajectory[self.traj_index]
        waypoint_pose = PoseStamped()
        # TODO: Fill in waypoint pose message using docs for PoseStamped 
        # (recall what frame this trajectory point is in from your trajectory.py code)
        # NOTE: The staticmethod below may be helpful


        # TODO: Find tf and transform waypoint to base_link
        # NOTE: do_transform_pose takes in and outputs a pose message type not PoseStamped (this is contrary to online documentation)
        odom_to_base = ...
        waypoint_base = ...

        # TODO: Calculate proportional error terms including x_err and y_err
        

        if abs(x_err) < 0.03 and abs(y_err) < 0.03:
            self.traj_index += 1
            print("Waypoint Reached, Now going to waypoint ", self.traj_index)
            self.prev_x_err = None
            self.prev_yaw_err = None
            return

        # TODO: Update derivative and integral error terms (refer to class variables defined in init)
        
        # TODO: Generate control command from error terms 
        control_cmd = Twist()
        
        self.pub.publish(control_cmd)
  

    # ------------------------------------------------------------------
    # Callback when goal point is published
    # ------------------------------------------------------------------
    def planning_callback(self, msg: PointStamped):
        if self.trajectory:
            return
        self.trajectory = plan_curved_trajectory(...)
        self.traj_index = 0
        self.x_i_err = 0.0
        self.yaw_i_err = 0.0
        self.prev_x_err = None
        self.prev_yaw_err = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_from_yaw(yaw):
        """Return quaternion (x, y, z, w) from yaw angle."""
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
