import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class Commander(Node):
    def __init__(self):
        super().__init__('commander')
        self._start_pub = self.create_publisher(Bool, '/start_pick_place', 1)
        self._class_pub = self.create_publisher(String, '/set_target_class', 1)


def main(args=None):
    rclpy.init(args=args)
    node = Commander()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    input("Orbit running... Press Enter when ready to start pick-and-place: ")
    msg = Bool()
    msg.data = True
    node._start_pub.publish(msg)
    print("Pick-and-place mode activated!")

    while rclpy.ok():
        try:
            class_name = input("Enter object class to pick (e.g. apple, cake): ").strip()
        except EOFError:
            break
        if class_name:
            msg = String()
            msg.data = class_name
            node._class_pub.publish(msg)
            print(f"Target set to: {class_name}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
