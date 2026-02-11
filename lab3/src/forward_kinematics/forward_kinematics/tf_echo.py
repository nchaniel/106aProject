import rclpy
import sys
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TFListenerNode(Node):
    def __init__(self):
        super().__init__('tf_echo_node')
        
        # Create a tf2 buffer and listener
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        
        # Extract the target and source frames from command line arguments
        if len(sys.argv) != 3:
            self.get_logger().error("Please provide target and source frames as command line arguments.")
            sys.exit(1)

        self.target_frame = sys.argv[1]
        self.source_frame = sys.argv[2]

        # Start a timer to check for the transform every 1 second
        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        try:
            # Look up the transform from source_frame to target_frame
            trans = self.tfBuffer.lookup_transform(self.target_frame, self.source_frame, rclpy.time.Time())

            # If successful, print the transformation
            self.get_logger().info(f"Transform from {self.source_frame} to {self.target_frame} is:\n{trans}")
        
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Could not find transform: {e}")
        except tf2_ros.ConnectivityException as e:
            self.get_logger().warn(f"Connectivity issue: {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"Extrapolation issue: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TFListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
