source ~/.bashrc
conda update --yes conda
conda create -n py36 -y python=3.6
source activate py36
conda install -c menpo -y opencv3
conda install -y cython scikit-image scikit-learn matplotlib bokeh ipython h5py nose pandas pyyaml jupyter pillow scipy opencv-contrib-python
conda install -c anaconda scikit-image
conda install -c conda-forge jupyter_contrib_nbextensions opencv-python
conda install -c conda-forge ipywidgets
conda install -c conda-forge ipyevents
pip install pybullet
pip install https://github.com/majimboo/py-mathutils/archive/2.78a.zip
pip install pycollada
pip install PyQt5
pip install mayavi
pip install pywavefront

# export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tensorflow/lib/python3.6/site-packages
#           inet addr:192.168.1.4  Bcast:192.168.1.255  Mask:255.255.255.0
