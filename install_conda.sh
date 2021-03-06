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
pip install pybullet trimesh pandas h5py Pyyaml descartes yacs
pip install scikit-learn
pip install https://github.com/majimboo/py-mathutils/archive/2.78a.zip
pip install pycollada
pip install PyQt5
pip install mayavi
pip install pywavefront

# export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tensorflow/lib/python3.6/site-packages
#           inet addr:192.168.1.4  Bcast:192.168.1.255  Mask:255.255.255.0
pip install scikit-learn pyquaternion opencv-python cachetools matplotlib vispy pillow
conda install numba
pip install vispy==0.5.3 opencv_python==4.1.0.25 opencv_contrib_python==4.1.0.25 Pillow==6.1.0 PyYAML==5.1.1 scipy==0.19.1

source activate benchmarks
module load cuda/10.1.
pip install torch torchvision # default 1.4.1
export CUDA=cu100
git clone https://github.com/rusty1s/pytorch_geometric.git
cd pytorch_geometric
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
python setup.py install #or pip install torch-geometric

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
im /home/lxiaol9/anaconda3/envs/benchmarks/lib/python3.6/site-packages/torch_points_kernels/torchpoints.py
vim /home/lxiaol9/anaconda3/envs/benchmarks/lib/python3.6/site-packages/torch_points_kernels/knn.py
export PYTHONPATH=$PYTHONPATH:/groups/CESCA-CV/src/etw-pytorch-utils

python train.py task=segmentation model_type=pointnet2 model_name=pointnet2_charlesssg dataset=shapenet
python train.py task=segmentation model_type=kpconv model_name=KPConvPaper dataset=semantickitti
