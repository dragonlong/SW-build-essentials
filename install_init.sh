#!/usr/bin/bash
# https://realpython.com/setting-up-sublime-text-3-for-full-stack-python-development/
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
sudo apt install sshfs
sudo apt install gcc
curl https://sh.rustup.rs -sSf | sh
cargo install urdf-viz
sudo apt-get install cmake xorg-dev libglu1-mesa-dev
