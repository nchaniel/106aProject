import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


def bbox_center(bbox):
    """
    Compute the pixel center of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (u, v): integer pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    u = int((x1 + x2) / 2.0)
    v = int((y1 + y2) / 2.0)
    return u, v


def get_depth_at_pixel_window(depth_image, u, v, window_size=5, depth_scale=0.001):
    """
    Get a robust depth estimate near pixel (u, v) using a small window.

    Args:
        depth_image: numpy depth image
        u, v: pixel coordinates
        window_size: odd integer window size
        depth_scale:
            0.001 if depth image is uint16 in millimeters
            1.0   if depth image is already float32 in meters

    Returns:
        depth_meters or None if no valid depth found
    """
    if depth_image is None:
        return None

    h, w = depth_image.shape[:2]

    if u < 0 or u >= w or v < 0 or v >= h:
        return None

    half = window_size // 2

    u_min = max(0, u - half)
    u_max = min(w, u + half + 1)
    v_min = max(0, v - half)
    v_max = min(h, v + half + 1)

    patch = depth_image[v_min:v_max, u_min:u_max]

    # Only keep valid positive depth values
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    depth_raw = np.median(valid)
    depth_meters = float(depth_raw) * depth_scale
    return depth_meters


def pixel_to_camera_xyz(u, v, depth, camera_info):
    """
    Back-project a pixel with depth into 3D camera coordinates.

    Using CameraInfo.k:
        [fx,  0, cx,
         0, fy, cy,
         0,  0,  1]

    Args:
        u, v: pixel coordinates
        depth: depth in meters
        camera_info: sensor_msgs.msg.CameraInfo

    Returns:
        (x, y, z) in camera frame
    """
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    if fx == 0.0 or fy == 0.0:
        raise ValueError("Camera intrinsics invalid: fx or fy is zero")

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return x, y, z


def make_point_stamped(x, y, z, frame_id, stamp):
    """
    Create a PointStamped message.
    """
    pt = PointStamped()
    pt.header.frame_id = frame_id
    pt.header.stamp = stamp
    pt.point.x = float(x)
    pt.point.y = float(y)
    pt.point.z = float(z)
    return pt


def transform_point(tf_buffer, point_stamped, target_frame):
    """
    Transform a PointStamped into target_frame.

    Args:
        tf_buffer: tf2_ros.Buffer
        point_stamped: geometry_msgs.msg.PointStamped
        target_frame: target frame name, e.g. 'base'

    Returns:
        transformed PointStamped
    """
    transform = tf_buffer.lookup_transform(
        target_frame,
        point_stamped.header.frame_id,
        rclpy.time.Time()
    )
    return do_transform_point(point_stamped, transform)