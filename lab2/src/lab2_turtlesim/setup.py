from setuptools import find_packages, setup

package_name = 'lab2_turtlesim'

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
<<<<<<< HEAD
    maintainer='ee106a-abm',
    maintainer_email='nathan.k.lam04@gmail.com',
=======
    maintainer='ee106a-abs',
    maintainer_email='christopher-mich_g@berkeley.edu',
>>>>>>> 82b000e (lab2)
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
<<<<<<< HEAD
        'turtle_controller = lab2_turtlesim.turtle_controller:main',
=======
            'turtle_controller = lab2_turtlesim.turtle_controller:main',
>>>>>>> 82b000e (lab2)
        ],
    },
)
