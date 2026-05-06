from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

import rosbag2_py

bridge = CvBridge()

# open bag
reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri='rosbag2_2026_05_04-12_22_21', storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('', '')
)

writer = cv2.VideoWriter('output1.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         15,
                         (640, 480))

while reader.has_next():
    topic, data, t = reader.read_next()

    if topic == "/camera/camera/color/image_raw":
        msg = deserialize_message(data, Image)
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        writer.write(img)

writer.release()