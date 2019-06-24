#>>>>>>>>>>>>>>>>>>>>---------Part 1. Installation----------<<<<<<<<<<<<<<<<<<<#
# reference here: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm
# first make sure you have logined into hulogin1/hulogin2
module load gcc cuda Anaconda3 jdk
java -version
conda create -n powerai36 python==3.6 # create a virtual environment
source activate powerai36             # activate virtual environment
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
# if things don't work, add two channels and run commands showing below
conda config --add default_channels https://repo.anaconda.com/pkgs/main
conda config --add default_channels https://repo.anaconda.com/pkgs/r
# install ibm powerai meta-package via conda
conda install powerai
# keep type 'enter' and then enter 1 for license acceptance
export IBM_POWERAI_LICENSE_ACCEPT=yes

#>>>>>>>>>>>>>>>>>>>>---------Part 2. DL Library Usages----------<<<<<<<<<<<<<<<<<<<#
# step 1: request for GPU nodes
# salloc -N 1 --gres=gpu:pascal:1 --partition=normal_q --account=openpower
# step 2: load all necessary modules
module load gcc cuda Anaconda3 jdk
# step 3: activate the virtual environment
source activate powerai36
# step 4: test with simple code examples
python test_pytorch.py
python test_TF_multiGPUs.py
python test_keras.py

# test with your own codes and begin your AI projects!
