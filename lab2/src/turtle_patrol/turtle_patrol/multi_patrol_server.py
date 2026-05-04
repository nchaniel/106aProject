import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
<<<<<<< HEAD
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

        # turtle_name -> { pub, teleport_client, vel, omega }; stores state of each turtle  
        self._turtles = {}

        # Timer publishes cmd_vel for all turtles
        self.create_timer(0.1, self.publish_all_cmds)

    # --------------------------------------------------
    # Timer publishes current Twist
    # --------------------------------------------------
    def publish_all_cmds(self):
        for turtle in self._turtles.values(): 
            msg = Twist()
            msg.linear.x = turtle['vel']
            msg.angular.z = turtle['omega']
            turtle['pub'].publish(msg) #actually moves turtle by publishing to cmd_vel topic

    # --------------------------------------------------
    # Service callback: update speeds and teleport; if new turtle, set up publisher and teleport client for it
    # --------------------------------------------------
    def patrol_callback(self, request, response): #recieves request from client, sends response back to client
        name = request.turtle_name

        # if first time this turtle is seen
        if name not in self._turtles:
            pub = self.create_publisher(Twist, f'/{name}/cmd_vel', 10) #publisher for this turtle
            
            #creates teleport client to call the teleport service
            teleport_client = self.create_client(
                TeleportAbsolute,
                f'/{name}/teleport_absolute'
            ) 

            if not teleport_client.wait_for_service(timeout_sec=2.0): #if turtle doesn't exist, return failure response
                response.success = False
                response.message = f"Teleport service not available for {name}"
                return response     

            self._turtles[name] = {  #store info needed to control turtle 
                'pub': pub,
                'teleport_client': teleport_client,
                'vel': 0.0,
                'omega': 0.0
            }

        # creates the teleport request
        teleport_req = TeleportAbsolute.Request()
        # put info for teleport request
        teleport_req.x = request.x
        teleport_req.y = request.y
        teleport_req.theta = request.theta

        self._turtles[name]['teleport_client'].call_async(teleport_req) #send teleport request

        #update turtle with requested values
        self._turtles[name]['vel'] = request.vel
        self._turtles[name]['omega'] = request.omega

        response.success = True
        response.message = f"{name} teleported and patrolling."
=======
from std_srvs.srv import Empty
from turtlesim.srv import TeleportAbsolute
from turtle_patrol_interface.srv import Patrol


class Turtle1PatrolServer(Node):
    def __init__(self):
        super().__init__('multi_patrol_server')
        self.cmd_dict = {}

        # Publisher: actually drives turtle1
        #self._cmd_pub = self.create_publisher(Twist, '/' +turtle_name+'/cmd_vel', 10)
        self._srv = self.create_service(Patrol, '/patrol', self.patrol_callback)

        # Current commanded speeds (what timer publishes)
        #self._lin = 0.0
        #self._ang = 0.0

        # Timer: publish current speeds at 10 Hz
        self._pub_timer = self.create_timer(0.1, self._publish_current_cmds)

        self.get_logger().info('Turtle1PatrolServer ready (continuous publish mode).')

    # -------------------------------------------------------
    # Timer publishes current Twist
    # -------------------------------------------------------
    def _publish_current_cmds(self):
        for command in self.cmd_dict:
            if len(self.cmd_dict[command]) == 2:
                self.cmd_dict[command][0].publish(self.cmd_dict[command][1])

        #self._cmd_pub.publish(msg)

    # -------------------------------------------------------
    # Service callback: update speeds
    # -------------------------------------------------------
    def patrol_callback(self, request: Patrol.Request, response: Patrol.Response):
        self.get_logger().info(
            f"Patrol request: vel={request.vel:.2f}, omega={request.omega:.2f}"
        )
        turtle_name = request.turtle_name

        if turtle_name not in self.cmd_dict:
            cmd_pub = self.create_publisher(Twist, '/' + turtle_name +'/cmd_vel', 10)
            cmd = Twist()
            cmd.linear.x = request.vel
            cmd.angular.z = request.omega
            self.cmd_dict[request.turtle_name] = [cmd_pub, cmd]
            tele_client = self.create_client(TeleportAbsolute, '/' +turtle_name+'/teleport_absolute')
            tele_req = TeleportAbsolute.Request()
            tele_req.x = request.x
            tele_req.y = request.y
            tele_req.theta = request.theta
            future = tele_client.call_async(tele_req)

        self.get_logger().info(
            f"Streaming cmd_vel: lin.x={request.vel:.2f}, ang.z={request.omega:.2f} (10 Hz)"
        )
        response.success = True
>>>>>>> 82b000e (lab2)
        return response


def main(args=None):
    rclpy.init(args=args)
<<<<<<< HEAD
    node = MultiTurtlePatrolServer()
=======
    node = Turtle1PatrolServer()
>>>>>>> 82b000e (lab2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
