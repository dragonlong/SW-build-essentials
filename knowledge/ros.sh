source /opt/ros/kinetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
echo $ROS_PACKAGE_PATH


roslaunch urdf_tutorial display.launch model:='$(find urdf_tutorial)/urdf/my_laptop.urdf' gui:True
# can also launch xacro
# check_urdf command
urdf_to_graphiz /tmp/laptop_whole.urdf
#
<collision>
  <geometry>
    <box size="0.6 0.1 0.2"/>
  </geometry>
  <origin rpy="0 1.57075 0" xyz="0 0 -0.3"/>
</collision>
<inertial>
  <mass value="10"/>
  <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
</inertial>
