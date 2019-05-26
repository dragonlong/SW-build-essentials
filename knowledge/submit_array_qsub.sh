#!/bin/bash
#PBS -l walltime=15:00:00
#PBS -l nodes=1:ppn=4
#PBS -W group_list=newriver
#PBS -q normal_q
#PBS -A DeepText
#PBS -t 0-4
pwd; hostname; date
######################
# Begin work section #
# 1. car: 02958343 chair: 03001627
######################
cd /home/lxiaol9/6DPose2019/blender-2.79b
MYDIR="/work/cascades/lxiaol9/6DPOSE/shapenet/ShapeNetCore.v2/02958343"
obj="02958343"
DIRS=$(ls $MYDIR)
arr=($DIRS)
for i in {0..599}
do
	echo "My ARRAY_TASK_ID: " ${PBS_ARRAYID}
	instance=${arr[${PBS_ARRAYID}*600+${i}]}
	mkdir -p /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/shapenet/$obj/$instance
	./blender -b --python /home/lxiaol9/3DGenNet2019/src/tools/render_ae3d_shapenet.py \
	-- -m /work/cascades/lxiaol9/6DPOSE/shapenet/ShapeNetCore.v2/$obj/$instance/models/model_normalized.obj \
	-rotate -o /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/shapenet/$obj/$instance \
	-s 128 -n 100 -fov 5
	#python main.py --model_dir=/home/lxiaol9/6DPose2019/keypointnet/checkpoints/ --input=/home/lxiaol9/6DPose2019/keypointnet/dataset/YCB_benchmarks/004_sugar_box/test_images/ --predict
done
# Do some work based on the SLURM_ARRAY_TASK_ID
# For example:
# ./my_process $SLURM_ARRAY_TASK_ID
#
# where my_process is you executable
# https://slurm.schedmd.com/job_array.html
exit
