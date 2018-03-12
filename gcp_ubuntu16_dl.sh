#!/usr/bin/env bash

# This is the set-up script for Google Cloud.

# Add cuda to path
echo -e "export PATH=/usr/local/cuda-8.0/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
source ~/.bashrc

cd ~
wget http://dl.dropboxusercontent.com/s/o4ptwrkxzi88mpx/cudnn-8.0-linux-x64-v6.0.tgz # Download Cudnn 6.0 for Cuda 8.0
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz

# Copy cudnn files
cd cuda/
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

cd ~

# Set locale
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
sudo locale-gen
sudo dpkg-reconfigure locales

# Update and Upgrade
sudo apt-get -y update
sudo apt-get -y upgrade

# Install Dependencies
sudo apt-get install -y libncurses5-dev
sudo apt-get install -y python-pip
sudo apt-get install -y python3-pip
sudo apt-get install -y python2.7-dev python3.5-dev
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y zip unzip
sudo apt-get install -y unrar-free
sudo apt-get install -y p7zip-full
sudo apt-get install -y libjpeg8-dev
sudo apt-get install -y python-lxml
sudo apt-get install -y python3-lxml
sudo apt-get install -y python-tk
sudo apt-get install -y python3-tk
sudo apt-get install -y htop
sudo apt-get install -y cython
sudo apt-get remove -y unattended-upgrades  # remove automatic updates
sudo apt-get install -y libcupti-dev

sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
sudo apt-get build-dep python-imaging python3-imaging
sudo apt-get install -y libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev

sudo pip install virtualenv     # Install virtual environment
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip   # Download OpenCV 3.3.0
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip   # Download OpenCV 3.3.0 Contrib
unzip opencv_contrib.zip


###################################
##		VENV for Python 3		 ##
###################################
virtualenv -p python3 venv3                 # Create a virtual environment
source ~/venv3/bin/activate                  # Activate the virtual environment
pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install -U scikit-learn
pip install -U nltk
pip install pillow
pip install h5py
pip install kaggle-cli
pip install scikit-image
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip install torchvision
python -m pip install pymongo
pip install tqdm
pip install Click
pip install --upgrade tensorflow-gpu
# Install keras from source
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py install
cd ..
rm -rf keras

# Download OpenCV source and install
cd ~/opencv-3.3.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
      -D PYTHON_EXECUTABLE=~/venv3/bin/python \
      -D BUILD_EXAMPLES=ON ..
make -j16
sudo make install
sudo ldconfig
cd /usr/local/lib/python3.5/site-packages/
sudo mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
cd ~/venv3/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
deactivate

rm -rf ~/opencv-3.3.0/build

jupyter notebook --generate-config
echo -e "c = get_config()" >> ~/.jupyter/jupyter_notebook_config.py
echo -e "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo -e "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo -e "c.NotebookApp.port = 7000" >> ~/.jupyter/jupyter_notebook_config.py

cd ~

###################################
##		VENV for Python 2		 ##
###################################
virtualenv -p python venv2
source ~/venv2/bin/activate                  # Activate the virtual environment
pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install -U scikit-learn
pip install -U nltk
pip install pillow
pip install h5py
pip install kaggle-cli
pip install scikit-image

pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip install torchvision
python -m pip install pymongo
pip install tqdm
pip install Click
pip install --upgrade tensorflow-gpu
# Install keras from source
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py install
cd ..
rm -rf keras


# Download OpenCV source and install
cd ~/opencv-3.3.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
      -D PYTHON_EXECUTABLE=~/venv2/bin/python \
      -D BUILD_EXAMPLES=ON ..
make -j16
sudo make install
sudo ldconfig
cd ~/venv2/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
deactivate

# Clean Up
cd ~
rm *.zip
rm cudnn*

echo "********************************************************"
echo "*****  End of Google Cloud Set-up Script  	  ********"
echo "*****  If there is no errors following    	  ********"
echo "*****  Python Learning libraries are installed. ********"
echo "*****  OpenCV 3.3.0, Tensorflow, pytorch, Keras ********"
echo "********************************************************"
echo ""
echo "You can check if everything were installed successfully."
echo "1- First run 'source ~/venv2/bin/activate' for python 2.7 or 'source ~/venv3/bin/activate' for python 3.x"
echo "on Terminal to activate virtualenv which have the libraries installed."
echo "                                                        "
echo "2- run 'ipython' on Terminal to enter ipython."
echo "                                                        "
echo "3- If you can import following libraries everything was fine."
echo "	- In the ipython run 'import tensorflow', 'import keras', 'import cv2', 'import torch'."
echo "	- If there is no errors, libraries were installed."
echo "                                                        "
echo "4- To exit from ipython run 'exit', and to deactivate venv run 'deactivate' on Terminal."
echo "                                                        "
echo "5- You can also run Jupyter notebook on created virtualenv. To run it follow the instruction."
echo "	- On Terminal run the followin command."
echo "                                                        "
echo " 	'jupyter notebook --no-browser --port 7000' "
echo "                                                        "
echo "	- You will be provided a link with token. Copy the link to your browser address bar and replace 'localhost' with your"
echo "   google cloud machine external ip address."
