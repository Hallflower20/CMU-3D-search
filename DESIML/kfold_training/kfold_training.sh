#!/bin/bash
#SBATCH -A m4237
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --image=nersc/tensorflow:24.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --constraint=gpu

export SLURM_CPU_BIND="cores"
module load tensorflow
cd /global/homes/x/xhall/GitHub/CMU-3D-search/DESIML/kfold_training
srun shifter python kfold_training.py