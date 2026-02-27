#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys
import tty
import termios
import threading
import select

class JointPosController(Node):
    def __init__(self, angles):
        super().__init__('joint_pos_controller')
        
        self.joint_names = [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint',
        ]

        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)
    
        self.send_trajectory(angles)

    # same send_trajectory logic from keyboard_controller
    def send_trajectory(self, angles):
        #setup
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        # points 
        point = JointTrajectoryPoint()
        point.positions = angles
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 5 # Set to 5 for safety, as said in the instructions
    
        traj.points.append(point)
        
        self.pub.publish(traj)
        self.get_logger().info(f"Published target: {angles}")

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 7:
        print(
            "Usage:\n"
            "ros2 run joint_control joint_pos_controller "
            "lift elbow w1 w2 w3 pan"
        )
        return

    # We use float() here to ensure they are numbers, not strings
    try:
        angles = [float(a) for a in sys.argv[1:]]
        node = JointPosController(angles)
        
        # We spin once 
        rclpy.spin(node)
    except ValueError:
        print("Error: All arguments must be numbers (radians).")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
