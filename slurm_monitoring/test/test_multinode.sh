#!/bin/bash
#SBATCH --job-name=maml-mpi-test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32  # total 8 MPI ranks
#SBATCH --time=24:00:00
#SBATCH --output=/home/cmacmahon/slurm/test_datagen-%j.std
#SBATCH --error=/home/cmacmahon/slurm/test_datagen-%j.err

#! Load conda and activate the environment
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

# Run the script with MPI
srun --mpi=pmix python /home/cmacmahon/CosyMAML/generate_MAML_data_MPI.py --n_tasks 2 --n_samples 500 --seed 1337
