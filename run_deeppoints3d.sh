source activate benchmarks
module load cuda/10.1.
pip install torch torchvision # default 1.4.1
export CUDA=cu101
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
python train.py task=segmentation model_type=kpconv model_name=KPDeformableConvPaper dataset=shapenet
python train.py task=segmentation model_type=kpconv model_name=KPConvPaper dataset=semantickitti
working on
#>>>>>>>>> tf version >>>>>>>>
vim datasets/SemanticKitti.py # change line 127
cp ~/4DAutoSeg/fuse_1.1d/config/labels/semantic-kitti* .
