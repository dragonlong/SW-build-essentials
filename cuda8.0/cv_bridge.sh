# https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3
# https://github.com/erlerobot/gym-gazebo/issues/90
# `python-catkin-tools` is needed for catkin tool
# `python3-dev` and `python3-catkin-pkg-modules` is needed to build cv_bridge
# `python3-numpy` and `python3-yaml` is cv_bridge dependencies
# `ros-kinetic-cv-bridge` is needed to install a lot of cv_bridge deps. Probaply you already have it installed.
sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge
# Create catkin workspace
cd ~
mkdir catkin_workspace
cd catkin_workspace
catkin init
# Instruct catkin to set cmake variables
catkin config -DPYTHON_EXECUTABLE=/home/dragonx/anaconda3/envs/tensorflow/bin/python -DPYTHON_INCLUDE_DIR=/home/dragonx/anaconda3/envs/tensorflow/include/python3.6m -DPYTHON_LIBRARY=/home/dragonx/anaconda3/envs/tensorflow/lib/libpython3.6m.so
# Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
catkin config --install
# Clone cv_bridge src
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
# Find version of cv_bridge in your repository
apt-cache show ros-kinetic-cv-bridge | grep Version
# Version: 1.12.8-0xenial-20180416-143935-0800
# Checkout right version in git repo. In our case it is 1.12.8
cd src/vision_opencv/
git checkout 1.12.8
cd ../../
# Build
catkin build cv_bridge
# Extend environment with new package
source install/setup.bash --extend


cmake  ..

copying python/cv_bridge/core.py -> build/lib/cv_bridge
copying python/cv_bridge/__init__.py -> build/lib/cv_bridge
running install_lib
creating /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge
copying build/lib/cv_bridge/core.py -> /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge
copying build/lib/cv_bridge/__init__.py -> /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge
byte-compiling /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge/core.py to core.cpython-36.pyc
byte-compiling /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge/__init__.py to __init__.cpython-36.pyc
running install_egg_info
Writing /home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages/cv_bridge-1.12.8-py3.6.egg-info

echo "source ~/catkin_workspace/devel/setup.bash" >> ~/.bashrc
