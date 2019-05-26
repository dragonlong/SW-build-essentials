#!/bin/bash
#SBATCH --partition=v100_normal_q
#SBATCH --account=DeepText
#SBATCH -t 48:00:00
#SBATCH -N 1

#SBATCH --mail-type=FAIL
#SBATCH --mail-user lxiaol9@vt.edu
#SBATCH -J Dragon

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
pwd; hostname; date
# # refer to https://ubccr.freshdesk.com/support/solutions/articles/5000688140-submitting-a-slurm-job-script
# # https://slurm.schedmd.com/squeue.html
######################
# Begin work section #
######################
module purge
module load gcc cmake
module load cuda/9.0.176 cudnn/7.1
source activate dlubu36
module list
unset LANG
export LANG=en_GB.UTF-8
######################
# Begin work section #
######################

# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
# mkdir -p /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/output/$DIR
cd /home/lxiaol9/3DGenNet2019/AE3D/src
# using CXT_size 3
python train_dense5.py --model_type=ASPP --model_dir='/work/cascades/lxiaol9/6DPOSE/checkpoints/tmp/0.81' --adam_lr_alpha=0.001 --log_steps=20 --random_shuffle
# Do some work based on the SLURM_ARRAY_TASK_ID
# For example:
# ./my_process $SLURM_ARRAY_TASK_ID
#
# where my_process is you executable
# https://slurm.schedmd.com/job_array.html
