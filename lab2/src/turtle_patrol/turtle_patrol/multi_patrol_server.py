import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from turtlesim.srv import TeleportAbsolute
from turtle_patrol_interface.srv import MultiPatrol


class MultiTurtlePatrolServer(Node):
    def __init__(self):
        super().__init__('multi_turtle_patrol_server')

        self.create_service(
            MultiPatrol,
            '/turtle_patrol',
            self.patrol_callback
        )

        # turtle_name -> { pub, teleport_client, vel, omega }
        self._turtles = {}

        # Timer publishes cmd_vel for all turtles
        self.create_timer(0.1, self.publish_all_cmds)

        self.get_logger().info("MultiTurtlePatrolServer ready.")

    # --------------------------------------------------
    # Timer callback
    # --------------------------------------------------
    def publish_all_cmds(self):
        for data in self._turtles.values():
            msg = Twist()
            msg.linear.x = data['vel']
            msg.angular.z = data['omega']
            data['pub'].publish(msg)

    # --------------------------------------------------
    # Service callback (FIXED)
    # --------------------------------------------------
    def patrol_callback(self, request, response):
        name = request.turtle_name

        self.get_logger().info(
            f"{name}: teleport to ({request.x}, {request.y}, {request.theta}), "
            f"vel={request.vel}, omega={request.omega}"
        )

        # First time this turtle is seen
        if name not in self._turtles:
            pub = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)

            teleport_client = self.create_client(
                TeleportAbsolute,
                f'/{name}/teleport_absolute'
            )

            if not teleport_client.wait_for_service(timeout_sec=2.0):
                response.success = False
                response.message = f"Teleport service not available for {name}"
                return response

            self._turtles[name] = {
                'pub': pub,
                'teleport_client': teleport_client,
                'vel': 0.0,
                'omega': 0.0
            }

        # -------------------------------
        # Teleport asynchronously
        # -------------------------------
        teleport_req = TeleportAbsolute.Request()
        teleport_req.x = request.x
        teleport_req.y = request.y
        teleport_req.theta = request.theta

        self._turtles[name]['teleport_client'].call_async(teleport_req)

        # -------------------------------
        # Store patrol parameters
        # -------------------------------
        self._turtles[name]['vel'] = request.vel
        self._turtles[name]['omega'] = request.omega

        response.success = True
        response.message = f"{name} teleported and patrolling."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurtlePatrolServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
