# Install cuDNN 5.1
ubuntu_version="$(lsb_release -r)"
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget -c ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig


# http://developer.download.nvidia.com/compute/redist/cudnn/v7.0/cudnn-9.0-linux-x64-v7.0.tgz
# curl -O http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.0-linux-x64-v7.tgz
