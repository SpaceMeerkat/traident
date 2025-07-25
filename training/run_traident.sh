#!/bin/bash

#SBATCH --job-name='Traident-PCA'
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB    # Adjust memory per CPU if needed
#SBATCH --ntasks=96
#SBATCH --output=logs/Traident-%j-stdout.log
#SBATCH --error=logs/Traident-%j-stderr.log
#SBATCH --time=01:00:00

echo "Submitting Slurm job"

source /idia/users/jdawson/torchenv/bin/activate

echo PyTorch initialised

cd /idia/users/jdawson/transient/traident/training/

mpirun -np 96 python3 PCA_test.py

echo "Job complete"