import rclpy
<<<<<<< HEAD
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
=======
import sys
import math
from rclpy.node import Node

# This line imports the built-in string message type that our node will use to structure its data to pass on our topic
from geometry_msgs.msg import Twist

# We're creating a class called Talker, which is a subclass of Node
class TurtleController(Node):

    # Here, we define the constructor
    def __init__(self, name):
        self.name = name
        super().__init__('turtle_controller')
        
         # Here, we set that the node publishes message of type String (where did this type come from?), over a topic called "chatter_talk", and with queue size 10. The queue size limits the amount of queued messages if a subscriber doesn't receive them quickly enough.
        self.publisher_ = self.create_publisher(Twist, '/'+ name + '/cmd_vel', 10)
        
        # We create a timer with a callback (a function that runs automatically when something happens so you don't have to constantly check if something has happened) 
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
    
    # Here we create a message with the counter value appended and publish it
    def timer_callback(self):
        msg = Twist()
        command = input("Enter command:").lower()
        if command == 'w':
            msg.linear.x = 2.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
        if command == 'a':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = math.pi/4
        if command == 's':
            msg.linear.x = -2.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
        if command == 'd':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = -math.pi/4
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % command)


def main(args=None):
    name = sys.argv[1]
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    minimal_publisher = TurtleController(name)
    # Spin the node so its callbacks are called
    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
>>>>>>> 82b000e (lab2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
<<<<<<< HEAD
=======

>>>>>>> 82b000e (lab2)
