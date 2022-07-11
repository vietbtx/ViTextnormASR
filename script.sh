#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem 64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=researchqostien
#SBATCH --job-name=VRP

module purge
module load Python/3.9.6
module load CUDA/11.3.1

source /common/home/users/t/tvbui/myenv/bin/activate

srun --gres=gpu:1 python -u run_train.py
