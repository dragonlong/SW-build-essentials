#!/bin/bash
################# 1. check ENV infos
1. update the user with ARC website, update instructions;
2. draft the email for Huckleberry;
3. CUDA/9.0,
uname -r
cat /usr/local/cuda/version.txt
cat /proc/driver/nvidia/version
nvcc -V
ls /usr/lib/powerpc64le-linux-gnu/libcudnn*
java -version
################# 2. prepare environment
export https_proxy=http://uaserve.cc.vt.edu:8080
module load gcc/5.4.0
module load openblas
module load cuda
module load anaconda3
# my_root                  /home/lxiaol9/.conda/envs/my_root
# py27                     /home/lxiaol9/.conda/envs/py27
# pytf_huck             *  /home/lxiaol9/.conda/envs/pytf_huck
# python36                 /home/lxiaol9/.conda/envs/python36
# pytorch_py3              /home/lxiaol9/.conda/envs/pytorch_py3
#                          /home/lxiaol9/anaconda3/envs/tf_cc
# base                     /opt/apps/anaconda3/5.2.0
# conda config --add channels conda-forge
# conda install -c conda-forge shapely
# . /opt/DL/bazel/bin/bazel-activate
# bazel shutdown
################# 3. start building bazel
cd ~
wget https://github.com/bazelbuild/bazel/releases/download/0.15.0/bazel-0.15.0-dist.zip
mkdir -p TF/bazel
cd TF/bazel
unzip ../../bazel-0.15.0-dist.zip
export EXTRA_BAZEL_ARGS="--jobs 8"
./compile.sh
export PATH=~/jobs/TF/bazel_0.15/output:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
################ 4. Get TensorFlow
# use bazel binary
cd /home/lxiaol9/jobs/TF/bazel_0.15/
git clone https://github.com/tensorflow/tensorflow
git checkout r1.10
# git checkout tags/v1.12.0-rc0
# git checkout tags/v1.12.0-rc1
# git checkout tags/v1.12.0-rc2
git submodule update --init
cd /home/lxiaol9/jobs/TF/bazel_0.15/
# for cuda lib issues, refer to https://github.com/tensorflow/tensorflow/issues/17801
# Solution 1
# cd /usr/local/cuda-8.0/nvvm/libdevice
# sudo ln -s libdevice.compute_50.10.bc libdevice.10.bc
# Solution 2
sed -i 's/libdevice.10/libdevice.compute_50.10/g' third_party/gpus/cuda_configure.bzl
./configure
################## 5. Build TF
########### updated ###########
cat /home/lxiaol9/jobs/TF/1.12/tensorflow/.tf_configure.bazelrc
bazel `@com_google_absl//absl/strings`
# I found the archive stuff
# https://chromium.googlesource.com/external/github.com/abseil/abseil-cpp/+/308ce31528a7edfa39f5f6d36142278a0ae1bf45
# bazel build --local_resources 2048,.5,1.0  --config=opt //tensorflow/tools/pip_package:build_pip_package
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /home/mcclurej/huckleberry/tensorflow_pkg/
bazel build //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
# Now the *.whl file is in ../tensorflow_pkg
conda create -n tf_testhuck python=3.6
source activate tf_testhuck
pip install ../tensorflow_pkg/*.whl
python -c "import tensorflow as tf; print(tf.__version__)"
#
# lxiaol9@hu013:~/jobs/TF/tensorflow$ ./configure
# WARNING: Output base '/home/lxiaol9/.cache/bazel/_bazel_lxiaol9/c109a875a51e1598b064a4d0d994ce0c' is on NFS. This may lead to surprising failures and undetermined behavior.
# You have bazel 0.14.0- (@non-git) installed.
# Please specify the location of python. [Default is /opt/apps/anaconda3/5.2.0/bin/python]:
#
#
# Found possible Python library paths:
#   /opt/apps/anaconda3/5.2.0/lib/python3.6/site-packages
# Please input the desired Python library path to use.  Default is [/opt/apps/anaconda3/5.2.0/lib/python3.6/site-packages]
#
# Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
# jemalloc as malloc support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
# No Google Cloud Platform support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
# No Hadoop File System support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
# No Amazon S3 File System support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
# No Apache Kafka Platform support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
# No XLA JIT support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with GDR support? [y/N]: n
# No GDR support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with VERBS support? [y/N]: n
# No VERBS support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
# No OpenCL SYCL support will be enabled for TensorFlow.
#
# Do you wish to build TensorFlow with CUDA support? [y/N]: y
# CUDA support will be enabled for TensorFlow.
#
# Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 8.0
#
#
# Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
#
#
# Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 6
#
#
# Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/usr/lib/powerpc64le-linux-gnu
#
#
# Do you wish to build TensorFlow with TensorRT support? [y/N]: n
# No TensorRT support will be enabled for TensorFlow.
#
# Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]:
#
#
# Please specify a list of comma-separated Cuda compute capabilities you want to build with.
# You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
# Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]3.5,3.7,5.2,6.0
#
#
# Do you want to use clang as CUDA compiler? [y/N]: n
# nvcc will be used as CUDA compiler.
#
# Please specify which gcc should be used by nvcc as the host compiler. [Default is /opt/apps/gcc/5.4.0/bin/gcc]:
#
#
# Do you wish to build TensorFlow with MPI support? [y/N]: n
# No MPI support will be enabled for TensorFlow.
#
# Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -mcpu=native]: -mcpu=power8 -mtune=power8
#
#
# Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
# Not configuring the WORKSPACE for Android builds.
#
# Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
# 	--config=mkl         	# Build with MKL support.
# 	--config=monolithic  	# Config for mostly static monolithic build.
# Configuration finished
# lxiaol9@hu013:~/jobs/TF/tensorflow$
