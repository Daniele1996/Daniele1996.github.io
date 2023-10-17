import os
from glob import glob
from setuptools import setup

package_name = 'task3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='piter',
    maintainer_email='piter@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
    'task3 = task3.task3:main',
    'service_caller = task3.service_caller:main'
        ],
    },
)
