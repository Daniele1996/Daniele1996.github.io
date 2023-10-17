# Formation Control

## Table of contents:
- Installation - Building
- Usage

## Installation - Building

1. Navigate to the das_proj_ws directory
2. Build the package via the terminal command:
    colcon build --symlink-install
3. Set up the environment variables and paths required to word via the command:
    . install/setup.bash

## Usage

1. Run the following command to start the program:
    ros2 launch formation_control ***.launch.py
2. Here's the list of possible launch files:
    - cube.launch.py
    - pentagon.launch.py
    - hexagon.launch.py
    - letterD.launch.py
    - letterA.launch.py
    - letterS.launch.py
3. In case of problems with the execution, try increase the timer_period in the launch file.