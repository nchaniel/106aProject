import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from perception.yolo_detector import YOLODetector


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        # Parameters you may want to change later
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("show_image", True)

        image_topic = self.get_parameter("image_topic").value
        model_path = self.get_parameter("model_path").value
        conf_threshold = self.get_parameter("conf_threshold").value
        self.show_image = self.get_parameter("show_image").value

        self.bridge = CvBridge()
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Detection node started.")
        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Using model: {model_path}")

    def image_callback(self, msg):
        try:
            # Convert ROS image -> OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Run YOLO
        detections = self.detector.detect(cv_image)

        # Print detections
        self.get_logger().info(f"Number of detections: {len(detections)}")
        for det in detections:
            self.get_logger().info(
                f"class={det['class_name']}, "
                f"conf={det['confidence']:.2f}, "
                f"center={det['center']}, "
                f"bbox={det['bbox']}"
            )

        # Draw and display detections
        if self.show_image:
            vis_image = self.detector.draw_detections(cv_image, detections)

            max_width = 1000
            max_height = 700
            h, w = vis_image.shape[:2]
            scale = min(max_width / w, max_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)

            display_image = cv2.resize(vis_image, (new_width, new_height))

            cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO Detections", display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()