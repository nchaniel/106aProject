from setuptools import find_packages, setup

package_name = 'forward_kinematics'

setup(
    name=package_name,
    version='0.0.0',
<<<<<<< HEAD
    packages=[package_name],
=======
    packages=find_packages(exclude=['test']),
>>>>>>> 0445bcc (Complete lab3)
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
>>>>>>> 0445bcc (Complete lab3)
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
            'tf_echo = forward_kinematics.tf_echo:main',
            'forward_kinematics_node = forward_kinematics.forward_kinematics_node:main',
=======
            'listener = forward_kinematics.forward_kinematics_node:main',
            'echoer = forward_kinematics.tf_echo:main',
>>>>>>> 0445bcc (Complete lab3)
        ],
    },
)
