#!/bin/bash
# refer to https://stackoverflow.com/questions/38137828/how-do-i-update-cudnn-to-a-newer-version?answertab=oldest#tab-top
# wget http://dl.dropboxusercontent.com/s/o4ptwrkxzi88mpx/cudnn-8.0-linux-x64-v6.0.tgz # Download Cudnn 6.0 for Cuda 8.0
# tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
# Copy cudnn files
# cd cuda/
# sudo cp -P include/cudnn.h /usr/include
# sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/

# remove
rm -f /usr/include/cudnn.h
rm -f /usr/lib/x86_64-linux-gnu/*libcudnn*
rm -f /usr/local/cuda-*/lib64/*libcudnn*
rm -rf ~/cuda/

cd 
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget -c ${CUDNN_URL}
tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda/
cp -P include/cudnn.h /usr/include
cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

rm -rf ~/cuda/


