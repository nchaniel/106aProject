import rclpy
from rclpy.node import Node
from turtle_patrol_interface.srv import MultiPatrol
import sys


class MultiTurtlePatrolClient(Node):
    def __init__(self, turtle_name, x, y, theta, vel, omega):
        super().__init__(f'multi_turtle_patrol_client_{turtle_name}') #allows multiple turtles to run as different clients

        self._client = self.create_client(MultiPatrol, '/turtle_patrol')
        self._client.wait_for_service()

        req = MultiPatrol.Request() #makes a request object to send to server
        # info to be used, matches srv file
        req.turtle_name = turtle_name
        req.x = float(x)
        req.y = float(y)
        req.theta = float(theta)
        req.vel = float(vel)
        req.omega = float(omega)

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Response: {future.result().message}")
        else:
            self.get_logger().error("Service call failed")

        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 7:
        print(
            "Usage:\n"
            "ros2 run turtle_patrol multi_patrol_client "
            "turtle_name x y theta vel omega"
        )
        return

    MultiTurtlePatrolClient(*sys.argv[1:]) #parses through command sent on terminal to the above 
    rclpy.shutdown()


if __name__ == '__main__':
    main()
