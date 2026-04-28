#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class StaticTransformPublisher(Node):
    def __init__(self):
        super().__init__('tf_node')
        self.br = StaticTransformBroadcaster(self)

        # Homogeneous transform G wrist_3_link -> camera_color_optical_frame
        G = np.array([[1, 0, 0, -0.025],
                      [0, 1, 0, 0.13],
                      [0, 0, 1, 0.0],
                      [0, 0, 0, 1.0]
        ])

        # Create TransformStamped
        self.transform = TransformStamped()

        rotation_matrix = G[:3, :3]
        translation = G[:3, 3]

        q = R.from_matrix(rotation_matrix).as_quat()

        self.transform.header.frame_id = 'wrist_3_link'
        self.transform.child_frame_id = 'camera_color_optical_frame'

        self.transform.transform.translation.x = float(translation[0])
        self.transform.transform.translation.y = float(translation[1])
        self.transform.transform.translation.z = float(translation[2])

        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.get_logger().info(f"Broadcasting transform:\n{G}\nQuaternion: {q}")

        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = StaticTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
