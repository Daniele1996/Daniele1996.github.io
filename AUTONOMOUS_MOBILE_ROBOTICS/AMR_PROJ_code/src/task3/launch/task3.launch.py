#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
 
 
def generate_launch_description():
 
  return LaunchDescription([
    Node(
            package= 'task3',
            executable='task3',
            name = 'task3',
            output = 'screen'
        )
  ])

