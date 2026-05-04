import rclpy
import sys
from rclpy.node import Node

# Import our custom service
from turtle_patrol_interface.srv import Patrol


class TurtlePatrolClient(Node):

    def __init__(self,turtle_name,x,y,theta,velocity,omega):
        super().__init__('turtle1_patrol_client')


        self._service_name = '/patrol'

        # Create a client for our Patrol service type
        self._client = self.create_client(Patrol, self._service_name)

        # Wait until the server is up (polling loop; logs once per second)
        self.get_logger().info(f"Waiting for service {self._service_name} ...")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Service {self._service_name} not available, waiting...")

        # Build request
        req = Patrol.Request()
        req.vel = float(velocity)
        req.omega = float(omega)
        req.turtle_name = turtle_name
        req.x = float(x)
        req.y = float(y)
        req.theta = float(theta)

        # Send request (async under the hood)
        self._future = self._client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    argv = sys.argv
    turtle_name,x,y,theta,velocity,omega = argv[1:7]
    node = TurtlePatrolClient(turtle_name,x,y,theta,velocity,omega)

    # Block here until the service responds (simple for teaching)
    rclpy.spin_until_future_complete(node, node._future)

    if node._future.done():
        result = node._future.result()
        if result is not None:
            node.get_logger().info(
                f"Service response {result.success}"
            )
        else:
            node.get_logger().error("Service call failed: no result returned.")
    else:
        node.get_logger().error("Service call did not complete.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


