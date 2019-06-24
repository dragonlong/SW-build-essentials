module purge
module load gcc/5.2.0
module load cuda/8.0.44
module load cudnn/6.0
conda create -n pytf_nr python=2.7
source activate pytf_nr

pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl

conda install future pybind11 pytest requests scipy
python -c "import tensorflow as tf; print(tf.__version__)"
# build
cd roi-pooling
python setup.py install
# test
from roi_pooling.roi_pooling_ops import roi_pooling
roi_pooling_module = tf.load_op_library('/home/lxiaol9/jobs/install/roi-pooling/roi_pooling/roi_pooling.so')

# download installer for Linux, take Anaconda3-5.3.1 for example
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# run the installer, the same procedure as local installation, say 'yes' for PATH appending
bash Anaconda3-5.3.1-Linux-x86_64.sh


# install in login node(take TF 1.12 for example, it works quite well from my test)
conda create -n pytf_nr python=3.6
source activate pytf_nr
conda install tensorflow-gpu


# then you run your test in interactive session on GPU nodes by:
source activate pytf_nr
module load cuda/9.0.176
module load cudnn/7.1
python -c 'import tensorflow as tf ; print(tf. __version__)'
# install in login node(take TF 1.10 for example, it works quite well from my test)
# since we already have anaconda installed
conda create -n pytf_nr python=3.6
source activate pytf_nr
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl
---------------------------------
# during usage in GPU node
module purge

module load gcc cmake
module load cuda/9.0.176 cudnn/7.1
source activate pytf_nr
# could run your own code, usually we debug code in interactive GPU job
python -c "import tensorflow as tf; print(tf.__version__)"


#####################################################################
# download installer for Linux, take Anaconda3-5.3.1 for example
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# run the installer, the same procedure as local installation, say 'yes' for PATH appending
bash Anaconda3-5.3.1-Linux-x86_64.sh
#append the path for Anaconda
source ~/.bashrc
# test conda installation
which conda
# install in login node(take TF 1.12 for example, it works quite well from my test)
conda create -n pytf_cc python=3.6
source activate pytf_cc
conda install tensorflow-gpu
# for Keras
conda install -c conda-forge keras
conda install -c conda-forge/label/cf201901 keras # this will install 2.2.4 by default

# then you run your test in interactive session on GPU nodes by:
source activate pytf_cc
module load gcc cmake
module load cuda/9.0.176
module load cudnn/7.1
python -c 'import tensorflow as tf ; print(tf. __version__)'
