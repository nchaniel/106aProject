import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2

import tf2_ros

from perception.yolo_detector import YOLODetector
from perception.pixel_to_world import (
    get_centroid_from_cloud_bbox,
    transform_point,
)


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("cloud_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("model_path", "updated.pt")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("show_image", True)
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("target_class", "")

        image_topic = self.get_parameter("image_topic").value
        cloud_topic = self.get_parameter("cloud_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        model_path = self.get_parameter("model_path").value
        conf_threshold = self.get_parameter("conf_threshold").value
        self.show_image = self.get_parameter("show_image").value
        self.target_frame = self.get_parameter("target_frame").value
        self.target_class = self.get_parameter("target_class").value

        # ----------------------------
        # Core objects
        # ----------------------------
        self.bridge = CvBridge()
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Latest data buffers
        self.cloud = None
        self.camera_info = None

        self.plate_position = None
        self.plate_radius = 0.14  # tune this (meters)

        # ----------------------------
        # Subscribers
        # ----------------------------
        self.rgb_subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback, 10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )

        # ----------------------------
        # Publisher
        # ----------------------------
        self.pick_point_pub = self.create_publisher(PointStamped, "/detected_pick_point", 10)
        self.plate_point_pub = self.create_publisher(PointStamped, "/detected_plate_point", 10)
        self.class_pub = self.create_publisher(String, "/detected_class", 10)

        # ----------------------------
        # Logging / watchdog
        # ----------------------------
        self.get_logger().info("Detection node started.")
        self.get_logger().info(f"RGB topic:    {image_topic}")
        self.get_logger().info(f"Cloud topic:  {cloud_topic}")
        self.get_logger().info(f"Using model:  {model_path}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"Target class: '{self.target_class}' (empty = any)")

        self._image_topic = image_topic
        self._cloud_topic = cloud_topic
        self._frames_received = 0
        self._cloud_received = 0
        self._watchdog = self.create_timer(5.0, self._watchdog_cb)

    def _watchdog_cb(self):
        msgs = []
        if self._frames_received == 0:
            msgs.append(f"No RGB images received on '{self._image_topic}'")
        if self._cloud_received == 0:
            msgs.append(f"No PointCloud2 received on '{self._cloud_topic}'")
        if msgs:
            for m in msgs:
                self.get_logger().warn(m)
            self.get_logger().warn(
                "Run 'ros2 topic list' and override topics with "
                "--ros-args -p image_topic:=... -p cloud_topic:=..."
            )
        else:
            self._watchdog.cancel()

    def cloud_callback(self, msg):
        self._cloud_received += 1
        self.cloud = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        self._frames_received += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        detections = self.detector.detect(cv_image)

        if self.cloud is None or self.camera_info is None:
            self.get_logger().debug("No cloud/camera_info yet; skipping 3D")

        best_pick_point = None

        for det in detections:
            class_name = det["class_name"]
            if self.target_class and class_name != self.target_class and class_name != "plate":
                continue

            if self.cloud is None or self.camera_info is None:
                continue

            try:
                pt_cam = get_centroid_from_cloud_bbox(
                    self.cloud, self.camera_info, det["bbox"]
                )

                if pt_cam is None:
                    self.get_logger().warn(
                        f"No valid depth for {det['class_name']} in bbox {det['bbox']}"
                    )
                    continue

                pt_base = transform_point(self.tf_buffer, pt_cam, self.target_frame)

                if class_name == "plate":
                    self.plate_point_pub.publish(pt_base)
                    self.plate_position = pt_base
                    self.get_logger().info("Published PLATE")
                    continue

                if self.plate_position is not None:
                    dx = pt_base.point.x - self.plate_position.point.x
                    dy = pt_base.point.y - self.plate_position.point.y
                    dist = (dx**2 + dy**2) ** 0.5

                    if dist < self.plate_radius:
                        self.get_logger().info(f"Ignoring {class_name} inside plate")
                        continue

                
                if best_pick_point is None or det["confidence"] > best_pick_point[0]:
                    best_pick_point = (det["confidence"], pt_base, det["class_name"])

            except Exception as e:
                self.get_logger().warn(f"Failed 3D for {det['class_name']}: {e}")

        if best_pick_point is not None:
            _, pt, class_name = best_pick_point
            self.class_pub.publish(String(data=class_name))
            self.pick_point_pub.publish(pt)
            self.get_logger().info(
                f"Published pick point [{class_name}] in {self.target_frame}: "
                f"({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f})"
            )

        if self.show_image:
            vis_image = self.detector.draw_detections(cv_image, detections)
            h, w = vis_image.shape[:2]
            scale = min(1000 / w, 700 / h)
            display_image = cv2.resize(vis_image, (int(w * scale), int(h * scale)))
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
