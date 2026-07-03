from setuptools import find_packages, setup

package_name = 'obstacle_detection_package'

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
    maintainer='wego',
    maintainer_email='wego@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolov8n_node = obstacle_detection_package.yolov8n_node:main',
            'cam_lidar_fusion_node = obstacle_detection_package.cam_lidar_fusion_node:main',
            'traffic_light_perception_node = obstacle_detection_package.traffic_light_perception_node:main',
        ],
    },
)
