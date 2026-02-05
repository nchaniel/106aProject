
import rclpy
from rclpy.node import Node
from my_chatter_msgs.msg import TimestampString


class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            TimestampString,
            '/user_messages',
            self.callback,
            10
        )
        self.get_logger().info('Listener started.')

    def callback(self, msg):
        recv_time = self.get_clock().now().nanoseconds

        self.get_logger().info(
            f'Received: "{msg.text}"\n'
            f'  Sent time:     {msg.timestamp}\n'
            f'  Receive time:  {recv_time}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

