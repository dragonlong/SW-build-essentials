######################################
# originally contributed by Hailin Ren, Virginia Tech
#####################################
virtualenv -p python3 rostf                 # Create a virtual environment
source ~/rostf/bin/activate
cd /opt/ros/kinetic/lib/python2.7/dist-packages/
sudo nautilus .
## do it using GUI
# remove or rename and check
ls | grep cv*
# install opencv for python3
pip3 install opencv-python
# install following dependencies
pip3 install gym
pip3 install rospkg catkin_pkg
sudo apt-get install python3-pyqt4
sudo apt-get install cmake gcc g++ qt4-qmake libqt4-dev libusb-dev libftdi-dev python3-defusedxml python3-vcstool ros-kinetic-octomap-msgs        ros-kinetic-joy                 ros-kinetic-geodesy             ros-kinetic-octomap-ros         ros-kinetic-control-toolbox     ros-kinetic-pluginlib       ros-kinetic-trajectory-msgs     ros-kinetic-control-msgs       ros-kinetic-std-srvs        ros-kinetic-nodelet       ros-kinetic-urdf       ros-kinetic-rviz       ros-kinetic-kdl-conversions     ros-kinetic-eigen-conversions   ros-kinetic-tf2-sensor-msgs     ros-kinetic-pcl-ros ros-kinetic-navigation
sudo apt-get install ros-kinetic-ar-track-alvar-msgs
sudo apt-get install ros-kinetic-sophus
sudo apt-get install git
# install gym-gazebo
cd ~
git clone https://github.com/erlerobot/gym-gazebo
cd gym-gazebo/
pip3 install -e .
pip3 install h5py
sudo apt-get install python3-skimage
# install tensorflow without gpu, you can install tensorflow with gpu later
sudo pip3 install --upgrade tensorflow
sudo pip3 install keras
deactivate
# compile gym_gazebo
cd ~/gym_gazebo/gym_gazebo/envs/installation
# folowing to gazebo.repos 
geometry:
   type: git
   url: https://github.com/ros/geometry
   version: indigo-devel
 geometry2:
   type: git
   url: https://github.com/ros/geometry2
   version: indigo-devel
 vision_opencv:
   type: git
   url: https://github.com/ros-perception/vision_opencv.git
   version: kineticym_gazebo/gym_gazebo/envs/installation
# try to compile, it will fail, I know that
bash setup_kinetic.bash
# Ok you got a problem at ~50%, you need modify some stuff
cd ~
touch gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/joystick_drivers/spacenav_node/CATKIN_IGNORE
touch gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/joystick_drivers/wiimote/CATKIN_IGNORE
touch gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/kobuki_desktop/kobuki_qtestsuite/CATKIN_IGNORE
cd ~/gym-gazebo/gym_gazebo/envs/installation
# try again
echo "# For Gym-Gazebo" >> ~/.bashrc
bash setup_kinetic.bash
bash turtlebot_setup.bash
bash turtlebot_nn_setup.bash
# It should finish
# put this into .bashrc
echo "# For Gym-Gazebo ROS Interface" >> ~/.bashrc
echo "export ROS_PORT_SIM=11311" >> ~/.bashrc
# modify kobuki.launch.xml
gym_gazebo/envs/installation/catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/launch/includes
nano kobuki.launch.xml
# modify sixth line to following
<arg name="urdf_file" default="$(find xacro)/xacro.py '$(find turtlebot_description)/robots/$(arg base)_$(arg stacks)_$(arg 3d_sensor).urdf.xacro'"/>
