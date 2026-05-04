import rclpy
from rclpy.node import Node
import open3d as o3d
import trimesh




class PointcloudFitting(Node):

    def __init__(self):
        super().__init__('pointcloud_fitting')

        self.load_mesh()

    def load_mesh(self):
        mesh_path = "/home/cc/ee106a/sp26/class/ee106a-abs/ros_workspaces/pointcloud_fitting/Halved_Strawberry_OBJ.obj"

        mesh = trimesh.load("/home/cc/ee106a/sp26/class/ee106a-abs/ros_workspaces/pointcloud_fitting/Halved_Strawberry_OBJ.obj")

        print(mesh)


def main(args=None):
    rclpy.init(args=args)

    node = PointcloudFitting()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()