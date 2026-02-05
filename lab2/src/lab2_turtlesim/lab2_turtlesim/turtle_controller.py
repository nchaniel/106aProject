import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys

class TurtleController(Node):
    def __init__(self, turtle_name):
        super().__init__(f'turtle_controller_{turtle_name}')
        self.turtle_name = turtle_name

        self.publisher_ = self.create_publisher(
            Twist,
            f'/{turtle_name}/cmd_vel',
            10
        )

        self.get_logger().info(f'Controlling turtle: {turtle_name}')

    def run(self):
        try:
            while rclpy.ok():
                key = input("w/a/s/d to move, q to quit: ").lower()

                twist = Twist()

                if key == 'w':
                    twist.linear.x = 2.0
                elif key == 's':
                    twist.linear.x = -2.0
                elif key == 'a':
                    twist.angular.z = 2.0
                elif key == 'd':
                    twist.angular.z = -2.0
                elif key == 'q':
                    break
                else:
                    continue

                self.publisher_.publish(twist)
        finally:
            self.publisher_.publish(Twist())
            self.get_logger().info('Controller stopped')


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run lab2_turtlesim turtle_controller <turtle_name>")
        return

    rclpy.init(args=args)
    node = TurtleController(sys.argv[1])
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
