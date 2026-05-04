from setuptools import find_packages, setup

package_name = 'my_tf_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-abs',
    maintainer_email='christopher-mich_g@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_to_marker = my_tf_project.camera_to_marker:main',
            'FollowJointTrajectory = my_tf_project.FollowJointTrajectory:main'
        ],
    },
)
