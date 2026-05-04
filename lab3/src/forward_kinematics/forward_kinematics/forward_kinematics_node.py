<<<<<<< HEAD
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

=======
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import forward_kinematics.forward_kinematics as fk
from rclpy.node import Node

from sensor_msgs.msg import JointState


class ForwardKinematicsSubscriber(Node):

    def __init__(self):
        super().__init__('forward_kinematics_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        G = fk.ur7e_forward_kinematics_from_joint_state(msg)
        self.get_logger().info('I heard: \n"%s"' % str(G))


def main(args=None):
    rclpy.init(args=args)

    forward_kinematics_subscriber = ForwardKinematicsSubscriber()

    rclpy.spin(forward_kinematics_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


>>>>>>> 0445bcc (Complete lab3)
if __name__ == '__main__':
    main()
