from setuptools import find_packages, setup
from glob import glob
import os # Don't forget this!

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line handles all .py and .launch.py files in your launch folder
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-tah',
    maintainer_email='danielmunicio360@gmail.com',
    description='Lab 7 Planning Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = planning.main:main',
            'tf = planning.static_tf_transform:main',
            'ik = planning.ik:main',
            'main_merged = planning.main_merged:main',
            'transform_cube_pose = planning.transform_cube_pose:main',
            'gripper = planning.gripper:main'
        ],
    },
)
