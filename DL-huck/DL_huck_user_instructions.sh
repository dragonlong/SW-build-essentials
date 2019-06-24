# Note:

# 1. both Option 1 and Option 2 provide methods to install new packages
# the recommanded way is: pip install --user package_name

# 2. please go to check folders in Google Drive for more examples
# https://drive.google.com/open?id=1n3aEGnQdM3NU6XUyDHEAd5HQM0v5tfvl

#>>>>>>>>>>>>>>>>>>>>-------Option 1. Use powerai_36 -------<<<<<<<<<<<<<<<<<<<<<<#
# make sure you're in GPU nodes via interactive job session
# In the default powerai_36 virtual environment, you would have access to TF-gpu/1.13.1, Pytorch 1.0.1, Keras

# step 1: load all necessary modules
module load gcc cuda Anaconda3 jdk
# step 2: activate the virtual environment
source activate powerai_36
# step 3: test with simple code examples
python test_pytorch.py
python test_TF_multiGPUs.py
python test_keras.py

# example of installing new packages, add --user flag
pip install --user beautifulsoup4
#>>>>>>>>>>>>>>>>>>>>-------Option 2. Clone Python Environment-----------<<<<<<<<<<<<<<<<<<<<<#
# In this part, you're instructed to clone python virtual environment into your own home directory
# and you would be able to install new python packages using pip or conda
cd ~
cp /opt/apps/Anaconda3/2019.03/powerai36.tar.gz .
mkdir -p .conda/envs/
tar -xzvf powerai36.tar.gz ~/.conda/envs/
sed -i -e 's|/home/lxiaol9/.conda/envs/py36ibm_1|/home/$USER/.conda/envs/powerai36|g' /home/$USER/.conda/envs/powerai36/site.cfg
sed -i -e 's|/home/lxiaol9/.conda/envs/py36ibm_1|/home/$USER/.conda/envs/powerai36|g' /home/$USER/.conda/envs/powerai36/bin/pip
module load gcc cuda Anaconda3 jdk
source activate powerai36
# Now you could run your python code

# example of installing new packages,
pip install beautifulsoup4


#>>>>>>>>>>>>>>>>>>>>---------job submission example------------<<<<<<<<<<<<<<<<<<<<<#
# Please refer to https://www.arc.vt.edu/computing/huckleberry-user-guide/
# more examples refer to slurm official site: https://slurm.schedmd.com/quickstart.html
salloc -N 1 --gres=gpu:pascal:1 --partition=normal_q --account=openpower


#>>>>>>>>>>>>>>>>>>>>---------TF/Pytorch/Keras test example------------<<<<<<<<<<<<<<<<<<<<<#
