import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


def get_centroid_from_cloud_bbox(cloud_msg, camera_info_msg, bbox):
    """
    Filter an unorganized PointCloud2 to the points that project into the YOLO
    bounding box, then return their centroid.

    Works with both organized and unorganized clouds since it reprojects each
    3D point to image coords rather than indexing by pixel.
    """
    if cloud_msg is None or camera_info_msg is None:
        return None

    fx = camera_info_msg.k[0]
    fy = camera_info_msg.k[4]
    cx = camera_info_msg.k[2]
    cy = camera_info_msg.k[5]

    x1, y1, x2, y2 = [int(v) for v in bbox]

    n = cloud_msg.width * cloud_msg.height
    if n == 0:
        return None

    # Read all points into numpy — x/y/z are the first 3 float32s in each point
    stride = cloud_msg.point_step // 4  # number of float32s per point
    data = np.frombuffer(bytes(cloud_msg.data), dtype=np.float32)
    xyz = data.reshape(n, stride)[:, :3]  # (N, 3)

    # Drop NaN and zero-depth points
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz = xyz[valid]

    if len(xyz) == 0:
        print(f"[pixel_to_world] cloud has no valid points at all")
        return None

    # Project to image coordinates using color camera intrinsics
    u = fx * xyz[:, 0] / xyz[:, 2] + cx
    v = fy * xyz[:, 1] / xyz[:, 2] + cy

    in_bbox = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    xyz_in = xyz[in_bbox]

    if len(xyz_in) == 0:
        print(f"[pixel_to_world] no cloud points project into bbox {bbox}")
        return None

    pt = PointStamped()
    pt.header = cloud_msg.header
    pt.point.x = float(np.mean(xyz_in[:, 0]))
    pt.point.y = float(np.mean(xyz_in[:, 1]))
    pt.point.z = float(np.mean(xyz_in[:, 2]))
    return pt


def transform_point(tf_buffer, point_stamped, target_frame):
    transform = tf_buffer.lookup_transform(
        target_frame,
        point_stamped.header.frame_id,
        rclpy.time.Time()
    )
    return do_transform_point(point_stamped, transform)
