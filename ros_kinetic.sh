sudo apt-get remove ros-kinetic-desktop-full
sudo apt-get remove ros-kinetic-gazebo*
sudo apt-get upgrade
# Installation
# Install the core of the ROS and rest of the packages can be added manually
sudo apt-get install ros-kinetic-ros-base
# Run $ roscore to confirm ROS installation. Install Gazebo 8 or 9 using commands:
# sudo apt-get install ros-kinetic-gazebo8-ros-pkgs ros-kinetic-gazebo8-ros-control ros-kinetic-gazebo8*
sudo apt-get install ros-kinetic-gazebo9-ros-pkgs ros-kinetic-gazebo9-ros-control ros-kinetic-gazebo9*
# Run $ gazebo to confirm Gazebo installation.
# Adding other ROS packages and dependencies
sudo apt-get install ros-kinetic-catkin
sudo apt-get install rviz
sudo apt-get install ros-kinetic-controller-manager ros-kinetic-joint-state-controller ros-kinetic-joint-trajectory-controller ros-kinetic-rqt ros-kinetic-rqt-controller-manager ros-kinetic-rqt-joint-trajectory-controller ros-kinetic-ros-control ros-kinetic-rqt-gui
sudo apt-get install ros-kinetic-rqt-plot ros-kinetic-rqt-graph ros-kinetic-rqt-rviz ros-kinetic-rqt-tf-tree
sudo apt-get install ros-kinetic-gazebo8-ros ros-kinetic-kdl-conversions ros-kinetic-kdl-parser ros-kinetic-forward-command-controller ros-kinetic-tf-conversions ros-kinetic-xacro ros-kinetic-joint-state-publisher ros-kinetic-robot-state-publisher
sudo apt-get install ros-kinetic-ros-control ros-kinetic-ros-controllers
