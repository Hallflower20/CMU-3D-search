#!/bin/bash
#SBATCH -N 1   
#SBATCH -c 32
#SBATCH -G 1 
#SBATCH -t 12:00:00   
#SBATCH -A phy220048p
#SBATCH -p HENON-GPU


    

cd /hildafs/home/xhall/GitHub/CMU-3D-search/DESIML/kfold_training
module load anaconda3/2020.07
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate /hildafs/projects/phy220048p/share/envs/tf_gpu

srun python kfold_training.py