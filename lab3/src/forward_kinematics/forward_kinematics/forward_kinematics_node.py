#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from forward_kinematics import ur7e_forward_kinematics_from_angles

class ForwardKinematicsNode(Node):
    def __init__(self):
        super().__init__('forward_kinematics_node')
        
        # Subscriber to the /joint_states topic
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10 
        )
        self.subscription  # prevent unused variable warning

    def joint_state_callback(self, msg):
        """
        Callback function that is called when a new /joint_states message is received.
        It extracts joint angles from the message, computes the forward kinematics, and prints the resulting transformation matrix.
        """
        joint_angles = msg.position  # Joint angles from the /joint_states message

        # Ensure we have exactly 6 joint angles (as expected for UR7e)
        if len(joint_angles) == 6:
            # Compute the transformation matrix
            transformation_matrix = ur7e_forward_kinematics_from_angles(np.array(joint_angles))

            # Print the resulting transformation matrix
            self.get_logger().info(f"Transformation Matrix (for End-Effector):\n{transformation_matrix}")
        else:
            self.get_logger().warn("Expected 6 joint angles.")

def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
