#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from forward_kinematics.forward_kinematics import ur7e_forward_kinematics_from_angles

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
        joint_dict = dict(zip(msg.name, msg.position))

        try:
            # re-order the joints explicitly as /joint_states topic typically publishes joint data in alphabetical order, apparently
            ordered_angles = np.array([
                joint_dict['shoulder_pan_joint'],
                joint_dict['shoulder_lift_joint'],
                joint_dict['elbow_joint'],
                joint_dict['wrist_1_joint'],
                joint_dict['wrist_2_joint'],
                joint_dict['wrist_3_joint']
            ])

            # calculate transformation matrix
            transformation_matrix = ur7e_forward_kinematics_from_angles(ordered_angles)

            # print resulting transformation matrix
            self.get_logger().info(f"Transformation Matrix:\n{transformation_matrix}")

        except KeyError as e:
            self.get_logger().warn(f"Missing joint in /joint_states: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
