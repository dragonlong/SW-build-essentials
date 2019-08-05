#!/usr/bin/bash
# MYDIR="/usr/local/google/home/xiaolongli/Downloads/data/6DPOSE/shape2motion/urdf/bike"
MYDIR="/home/xiaolongli/data/6DPOSE/arms"
DIRS=$(ls $MYDIR)
arr=($DIRS)
for ind in "${arr[@]}"; do   # The quotes are necessary here
    # urdf-viz /usr/local/google/home/xiaolongli/Downloads/data/6DPOSE/shape2motion/urdf/bike/$ind/syn.urdf
    echo $ind
    urdf-viz /home/xiaolongli/data/6DPOSE/arms/urdf/$ind
done
