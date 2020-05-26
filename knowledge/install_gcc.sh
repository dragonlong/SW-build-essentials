#!/usr/bin/env bash

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update

sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
sudo apt-get install -y gcc-4.8 g++-4.8 gcc-4.9 g++-4.9 gcc-5 g++-5 gcc-6 g++-6 gcc-7 g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 30
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 30
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 40
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50

sudo update-alternatives --set cc /usr/bin/gcc
sudo update-alternatives --set c++ /usr/bin/g++

sudo update-alternatives --config gcc
sudo update-alternatives --config g++

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 --slave /usr/bin/g++ g++ /usr/bin/g++-5 --slave /usr/bin/gcov gcov /usr/bin/gcov-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7
