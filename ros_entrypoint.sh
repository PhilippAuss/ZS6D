#!/bin/bash
set -e

# Setup ros environment
source "/opt/ros/noetic/setup.bash"
source "/root/catkin_ws/devel/setup.bash"

export ROS_MASTER_URI=http://10.0.0.143:11311
export ROS_IP=10.0.0.232

exec "$@"