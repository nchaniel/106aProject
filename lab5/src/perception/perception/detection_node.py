import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
import cv2

import tf2_ros

from perception.yolo_detector import YOLODetector
from perception.pixel_to_world import (
    bbox_center,
    get_depth_at_pixel_window,
    pixel_to_camera_xyz,
    make_point_stamped,
    transform_point,
)


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("show_image", True)
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_scale", 0.001)   # uint16 mm -> meters
        self.declare_parameter("depth_window_size", 5)

        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        model_path = self.get_parameter("model_path").value
        conf_threshold = self.get_parameter("conf_threshold").value
        self.show_image = self.get_parameter("show_image").value
        self.target_frame = self.get_parameter("target_frame").value
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_window_size = int(self.get_parameter("depth_window_size").value)

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
        self.depth_image = None
        self.camera_info = None

        # ----------------------------
        # Subscribers
        # ----------------------------
        self.rgb_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        # ----------------------------
        # Publisher
        # ----------------------------
        self.pick_point_pub = self.create_publisher(
            PointStamped,
            "/detected_pick_point",
            10
        )

        # ----------------------------
        # Logging
        # ----------------------------
        self.get_logger().info("Detection node started.")
        self.get_logger().info(f"RGB topic: {image_topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        self.get_logger().info(f"Camera info topic: {camera_info_topic}")
        self.get_logger().info(f"Using model: {model_path}")
        self.get_logger().info(f"Target frame: {self.target_frame}")

        # Watchdogs
        self._image_topic = image_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic

        self._frames_received = 0
        self._depth_received = 0
        self._camera_info_received = 0

        self._watchdog = self.create_timer(5.0, self._watchdog_cb)

    def _watchdog_cb(self):
        msgs = []

        if self._frames_received == 0:
            msgs.append(f"No RGB images received on '{self._image_topic}'")
        if self._depth_received == 0:
            msgs.append(f"No depth images received on '{self._depth_topic}'")
        if self._camera_info_received == 0:
            msgs.append(f"No CameraInfo received on '{self._camera_info_topic}'")

        if len(msgs) > 0:
            for msg in msgs:
                self.get_logger().warn(msg)
            self.get_logger().warn(
                "Run 'ros2 topic list' and override topics with "
                "--ros-args -p image_topic:=... -p depth_topic:=... -p camera_info_topic:=..."
            )
        else:
            self._watchdog.cancel()

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self._camera_info_received += 1

    def depth_callback(self, msg):
        self._depth_received += 1
        try:
            # Keep native encoding for depth
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def image_callback(self, msg):
        self._frames_received += 1

        try:
            # Convert ROS image -> OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        # Run YOLO
        detections = self.detector.detect(cv_image)

        self.get_logger().info(f"Number of detections: {len(detections)}")

        if self.camera_info is None:
            self.get_logger().warn("No CameraInfo yet; cannot convert detections to 3D")
        if self.depth_image is None:
            self.get_logger().warn("No depth image yet; cannot convert detections to 3D")

        best_pick_point = None

        for det in detections:
            self.get_logger().info(
                f"class={det['class_name']}, "
                f"conf={det['confidence']:.2f}, "
                f"center={det['center']}, "
                f"bbox={det['bbox']}"
            )

            # Only continue if we have what we need for 3D
            if self.camera_info is None or self.depth_image is None:
                continue

            try:
                # Compute bbox center
                u, v = bbox_center(det["bbox"])

                # Get robust depth estimate
                depth_m = get_depth_at_pixel_window(
                    self.depth_image,
                    u,
                    v,
                    window_size=self.depth_window_size,
                    depth_scale=self.depth_scale,
                )

                if depth_m is None:
                    self.get_logger().warn(
                        f"No valid depth for detection {det['class_name']} at pixel ({u}, {v})"
                    )
                    continue

                # Pixel -> camera 3D
                x_cam, y_cam, z_cam = pixel_to_camera_xyz(
                    u,
                    v,
                    depth_m,
                    self.camera_info
                )

                # Make PointStamped in camera frame
                pt_cam = make_point_stamped(
                    x_cam,
                    y_cam,
                    z_cam,
                    frame_id=self.camera_info.header.frame_id,
                    stamp=msg.header.stamp
                )

                # Camera frame -> target/base frame
                pt_base = transform_point(
                    self.tf_buffer,
                    pt_cam,
                    self.target_frame
                )

                self.get_logger().info(
                    f"[3D] {det['class_name']} | "
                    f"pixel=({u}, {v}) | "
                    f"depth={depth_m:.3f} m | "
                    f"camera=({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) | "
                    f"{self.target_frame}=({pt_base.point.x:.3f}, "
                    f"{pt_base.point.y:.3f}, {pt_base.point.z:.3f})"
                )

                # For v1: choose the highest-confidence valid detection
                if best_pick_point is None:
                    best_pick_point = (det["confidence"], pt_base)
                else:
                    if det["confidence"] > best_pick_point[0]:
                        best_pick_point = (det["confidence"], pt_base)

            except Exception as e:
                self.get_logger().warn(
                    f"Failed 3D conversion for detection {det['class_name']}: {e}"
                )

        # Publish best valid pick point
        if best_pick_point is not None:
            _, pt = best_pick_point
            self.pick_point_pub.publish(pt)
            self.get_logger().info(
                f"Published pick point in {self.target_frame}: "
                f"({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f})"
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