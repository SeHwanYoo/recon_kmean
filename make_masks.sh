#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=64:ngpus=1:mem=440gb 
#PBS -l walltime=10:00:00
#PBS -P personal-sehwan00
#PBS -N log_train_gan
# export NCCL_SOCKET_IFNAME=enp8s0
export NCCL_DEBUG=INFO
# Commands start here
module load pytorch/1.11.0-py3-gpu
cd ${PBS_O_WORKDIR}
python ./make_masks.py