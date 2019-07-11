#!/usr/bin/bash
MYDIR="/usr/local/google/home/xiaolongli/Downloads/data/6DPOSE/shape2motion/urdf/bike"
DIRS=$(ls $MYDIR)
arr=($DIRS)
for ind in "${arr[@]}"; do   # The quotes are necessary here
    urdf-viz /usr/local/google/home/xiaolongli/Downloads/data/6DPOSE/shape2motion/urdf/bike/$ind/syn.urdf
done
