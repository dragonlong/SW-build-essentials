#source ~/venv3/bin/activate
#sudo pip3 uninstall tensorflow
#sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp35-cp35m-linux_x86_64.whl
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --set show_channel_urls yes

# create py36 env
conda create -n tensorflow python=3.6
source activate tensorflow
pip install --upgrade tensorflow-gpu==1.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/linux-64/ opencv3
pip install matplotlib
pip install pillow
pip install scikit-image
python -m pip install opencv-python
python -m pip install opencv-contrib-python
# problems on rospy
pip install rospkg catkin_pkg -i https://pypi.tuna.tsinghua.edu.cn/simple
# about the rospkg import issues: https://www.bugshoot.cn/thread-5791647.htm

#conda create -n tf_cc python=3.6
pip3 install tensorflow-gpu==1.6.0
pip3 install keras==2.1.6
pip3 install ipython
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl
pip install matplotlib
pip install pillow
pip install scikit-image==0.13.1
pip install future
pip install pytest>=3.1
pip install scipy
pip install keras
python -m pip install opencv-python
python -m pip install opencv-contrib-python
# using opencv2, conflict with ROS and tensorflow
# how to solve OpenCV compatibility issues:
# 1. rm anaconda3 from python path
# 2. rm python2.7/cv2.so
# 3. link
# problem: mportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type
# https://stackoverflow.com/questions/43019951/after-install-ros-kinetic-cannot-import-opencv
cd ~/venv3/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
/home/dragonx/anaconda3/lib/python3.6/site-packages/
/opt/ros/kinetic/lib/python2.7/dist-packages/cv_bridge/
/home/dragonx/anaconda3/envs/tensorflow/lib/python3.6/site-packages
ln -sf /usr/local/lib/python3.5/site-packages/cv2.so cv2.so

echo $PYTHONPATH
unset PYTHONPATH
export PYTHONPATH=/home/dragonx/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
source activate tensorflow
cd ~/catkin_ws
rosrun mask_rcnn_ros mask_rcnn_node /mask_rcnn/input:=/camera/rgb/image_color
