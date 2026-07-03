from setuptools import find_packages, setup

package_name = 'test_pkg'

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
            'yolov8n_node = test_pkg.yolov8n_node:main',
            'lane_extractor_node = test_pkg.lane_extractor_node:main',
            'lane_fitting_node = test_pkg.lane_fitting_node:main',
            'stanley_planner = test_pkg.stanley_planner:main',
            'pure_pursuit_planner = test_pkg.pure_pursuit_planner:main',
            'extended_stanley_planner = test_pkg.extended_stanley_planner:main',
            'dh_stanley_planner = test_pkg.dh_stanley_planner:main',
            'real_stanley_planner = test_pkg.real_stanley_planner:main',
            'mpc_planner = test_pkg.mpc_planner:main',
        ],
    },
)
