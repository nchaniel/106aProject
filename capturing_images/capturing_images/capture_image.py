import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageCapture(Node):
    def __init__(self):
        super().__init__('image_capture')

        self.bridge = CvBridge()
        self.done = False

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.callback,
            10
        )

        print("init done")

    def callback(self, msg):
        if self.done:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("frame found")

        cv2.imwrite('captured_image.jpg', frame)
        self.get_logger().info('Saved image')

        self.done = True


def main():
    rclpy.init()
    node = ImageCapture()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()