import rclpy
from rclpy.node import Node
from my_chatter_msgs.msg import TimestampString


class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(
            TimestampString,
            '/user_messages',
            10
        )
        self.get_logger().info('Talker started.')

    def run(self):
        try:
            while rclpy.ok():
                text = input("Please enter a line of text and press <Enter>: ")

                msg = TimestampString()
                msg.text = text
                msg.timestamp = self.get_clock().now().nanoseconds

                self.publisher_.publish(msg)
                self.get_logger().info(
                    f'Published: "{msg.text}" at {msg.timestamp}'
                )
        except KeyboardInterrupt:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = Talker()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

