#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        #new transform
        G = np.array([[0,  -1,  0, -0.025],
                      [0,  0,  -1,  0.13],
                      [1,  0,  0,  0.0],
                      [0,  0,  0,  1.0]
        ])

        # Create TransformStamped
        self.transform = TransformStamped()
        

        # Frame IDs tell ROS which part of the robot the camera is attached to.
        # IMPORTANT: child_frame_id must be `camera_link` (the ROOT of the RealSense
        # TF sub-tree), NOT an optical frame. RealSense already publishes
        # `camera_link -> camera_{color,depth}_optical_frame` as static transforms,
        # and every frame can only have ONE parent. Publishing to an optical frame
        # here creates a parent conflict and TF2 ends up with two disconnected trees
        # (the UR tree rooted at base_link, and the camera tree rooted at camera_link).
        self.transform.header.frame_id = "wrist_3_link"
        self.transform.child_frame_id = "camera_link"

        #Translate the offset between the camera and the end effector
        self.transform.transform.translation.x = G[0, 3]
        self.transform.transform.translation.y = G[1, 3]
        self.transform.transform.translation.z = G[2, 3]

        # Convert rotation matrix to quaternion (x, y, z, w)
        rotation_matrix = G[:3, :3]
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()

        # Populate TransformStamped
        self.transform.transform.rotation.x = q[0]
        self.transform.transform.rotation.y = q[1]
        self.transform.transform.rotation.z = q[2]
        self.transform.transform.rotation.w = q[3]

        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

        self.get_logger().info(f"Broadcasting transform:\n{G}\nQuaternion: {q}")

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
