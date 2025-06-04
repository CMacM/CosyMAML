#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name=TEST_DATAGEN
#SBATCH --time=24:00:00
#SBATCH --output=/home/cmacmahon/slurm/test_datagen-%j.std
#SBATCH --error=/home/cmacmahon/slurm/test_datagen-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-user=cmacmahon@ddn.com
#SBATCH --mail-type=END
# --mail-type=BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, and STAGE_OUT

#! Load conda and activate the environment
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

#! Load conda and activate the environment for the application
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

cd /home/cmacmahon/CosyMAML
python /home/cmacmahon/CosyMAML/test_datagen.py \
--n_samples 10 \
--n_threads 16 \
