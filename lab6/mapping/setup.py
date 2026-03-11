from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Corrected: Using the package_name variable and standard quotes
        ('share/' + package_name + '/launch', glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-abu',
    maintainer_email='nathanielchan@berkeley.edu',
    description='Lab 6 Occupancy Grid Mapping',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mapping_node = mapping.mapping_node:main',
            'occupancy_grid = mapping.occupancy_grid_2d:main',
        ],
    },
)
