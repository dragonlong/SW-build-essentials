#>>>>>>>>>>>>>>>>>>>>---------Installation----------<<<<<<<<<<<<<<<<<<<#
# reference here: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm
ssh hu014
module load gcc cuda Anaconda3 jdk
java -version
export https_proxy=http://uaserve.cc.vt.edu:8080
conda create -n py36ibm_1 python==3.6
source activate py36ibm_1
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --add default_channels https://repo.anaconda.com/pkgs/main
conda config --add default_channels https://repo.anaconda.com/pkgs/r

export PATH=$PATH:/opt/apps/Anaconda3/2019.03/bin
source activate powerai16_ibm
export https_proxy=http://uaserve.cc.vt.edu:8080
pip install jupyter-tensorboard

conda install powerai
# keep enter and then enter 1 for license acceptance
export IBM_POWERAI_LICENSE_ACCEPT=yes
# move to Anaconda3 module system

#>>>>>>>>>>>>>>>>>>>>---------Usages----------<<<<<<<<<<<<<<<<<<<#
salloc -N 1 --gres=gpu:pascal:1 --partition=normal_q --account=openpower
module load gcc cuda Anaconda3 jdk
source activate py36ibm_1
python test_pytorch.py
python test_TF_multiGPUs.py
python test_keras.py

#>>>>>>>>>>>>>>>>>>>>----------Test------------<<<<<<<<<<<<<<<<<<<<<#
cd ~
cp /opt/apps/Anaconda3/2019.03/powerai36.tar.gz .
mkdir -p .conda/envs/
tar -xzvf powerai36.tar.gz ~/.conda/envs/
module load gcc cuda Anaconda3 jdk
source activate powerai36

sed -i -e 's|/home/lxiaol9/.conda/envs/py36ibm_1|/opt/apps/Anaconda3/2019.03/envs/powerai_36|g'
find /opt/apps/Anaconda3/2019.03/envs/powerai_36/ -type f -exec sed -i -e 's|/home/lxiaol9/.conda/envs/py36ibm_1|/opt/apps/Anaconda3/2019.03/envs/powerai_36|g' {} \;
