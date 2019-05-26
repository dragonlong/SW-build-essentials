#!/bin/bash

#SBATCH -q normal_q
#SBATCH --account=DeepText
#SBATCH -t 10:00
#SBATCH -N 1

#SBATCH --mail-user lxiaol9@vt.edu
#SBATCH -J blender
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=1-99

# use --partition=broadwl if you want to submit your jobs to Midway2
# #SBATCH --partition=sandyb
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4

srun hostname
pwd; hostname; date
######################
# Begin work section #
######################
cd /home/lxiaol9/6DPose2019/blender-2.79b
MYDIR="/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb"
DIRS=$(ls $MYDIR)
arr=($DIRS)
echo ${arr[$SLURM_ARRAY_TASK_ID-1]}
DIR=${arr[$SLURM_ARRAY_TASK_ID-1]}
# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
mkdir -p /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/output/$DIR
./blender -b --python /home/lxiaol9/6DPose2019/keypointnet/tools/render.py -- \
-m /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/$DIR/tsdf/textured.obj \
-o /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/output/$DIR \
-t /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/$DIR/tsdf/textured.png \
-s 128 -n 120 -fov 5
# Do some work based on the SLURM_ARRAY_TASK_ID
# For example:
# ./my_process $SLURM_ARRAY_TASK_ID
#
# where my_process is you executable
# https://slurm.schedmd.com/job_array.html
