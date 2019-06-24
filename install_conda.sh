# cd ~/Downloads
# wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# chmod a+x miniconda.sh
# ./miniconda.sh  # you will be prompted to agree on terms and to configure the installation
# rm miniconda.sh
source ~/.bashrc
conda update --yes conda
conda create -n py36 -y python=3.6
conda install -n py36 -c menpo -y opencv3
conda install -n py36 -y cython scikit-image scikit-learn matplotlib bokeh ipython h5py nose pandas pyyaml jupyter pillow scipy opencv-contrib-python
source activate py36
conda install -c anaconda scikit-image
conda install -c conda-forge jupyter_contrib_nbextensions opencv-python
conda install -c conda-forge ipywidgets
conda install -c conda-forge ipyevents
pip install pybullet
pip install https://github.com/majimboo/py-mathutils/archive/2.78a.zip
pip install pycollada
pip install PyQt5
pip install mayavi
# eth0      Link encap:Ethernet  HWaddr f4:6d:04:4f:d9:06
#           inet addr:192.168.1.4  Bcast:192.168.1.255  Mask:255.255.255.0
#           inet6 addr: fe80::f66d:4ff:fe4f:d906/64 Scope:Link
#           UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
#           RX packets:42689838 errors:0 dropped:0 overruns:0 frame:0
#           TX packets:34678047 errors:0 dropped:0 overruns:0 carrier:0
#           collisions:0 txqueuelen:1000
#           RX bytes:38806246852 (38.8 GB)  TX bytes:16126960569 (16.1 GB)
#           Interrupt:18 Memory:f9100000-f9120000
#
# lo        Link encap:Local Loopback
#           inet addr:127.0.0.1  Mask:255.0.0.0
#           inet6 addr: ::1/128 Scope:Host
#           UP LOOPBACK RUNNING  MTU:65536  Metric:1
#           RX packets:1021503 errors:0 dropped:0 overruns:0 frame:0
#           TX packets:1021503 errors:0 dropped:0 overruns:0 carrier:0
#           collisions:0 txqueuelen:1
#           RX bytes:717490417 (717.4 MB)  TX bytes:717490417 (717.4 MB)
