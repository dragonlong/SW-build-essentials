!/bin/bash
cd ~/Downloads/remo3d_v2.8_64_linux
# for converting the data;
MYDIR="/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb"
DIRS=$(ls $MYDIR)
arr=($DIRS)
echo ${arr[0]}
for DIR in $DIRS
do
   echo "processing ${DIR} data"
   mkdir -p /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/output/$DIR
   ./remo3d samples/scripts/convert.lua ~/Downloads/data/Motion\ Dataset\ v0/laptop/0001/0001.flt ~/Downloads/data/m2s/example.obj
   ./remo3d samples/scripts/flt2txt.lua ~/Downloads/data/Motion\ Dataset\ v0/bike/0001/0001.flt ~/Downloads/data/m2s/example.txt
done
# for attributes, where is the pivot origin and axis
# remo.getAttributes(name1, name2, ...)
# DOF Attributes:
#
