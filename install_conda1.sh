conda create -n py37 python=3.7 anaconda
source activate py37
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
pip install --upgrade pip
pip install fire tensorboardX

pip install numba numpy scipy six tqdm plyfile h5py
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
