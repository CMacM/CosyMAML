#!/bin/bash
#SBATCH --output=/home/cmacmahon/slurm/test_datagen-%j.std
#SBATCH --error=/home/cmacmahon/slurm/test_datagen-%j.err
#SBATCH --job-name=cosymaml_monitor
#SBATCH --nodelist=x440-03
#SBATCH --time=01:00:00

# # Retrieve the SLURM_JOB_ID
# env | grep -i job
# SLURM_JOB_ID=$(env | grep -i SLURM_JOB_ID | cut -d '=' -f2)
 
# # Check if SLURM_JOB_ID is defined
# if [ -z "$SLURM_JOB_ID" ]; then
#     echo "Error: SLURM_JOB_ID not defined."
#     exit 1
# fi

# # Export SLURM job ID in container environment
# export SLURM_JOB_ID=$SLURM_JOB_ID
# export SLURM_JOBID=$SLURM_JOB_ID

#! Load conda and activate the environment for the application
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

echo "Starting monitoring"
# Start glljobstat monitoring in the background
# Runs in the base environment job is submitted from
python3 /home/cmacmahon/glljobstat.py -i 5 -f $SLURM_JOB_ID -fm -r > /home/cmacmahon/CosyMAML/logs/data/log_${SLURM_JOB_ID}_rate.log &
PID1=$!

# -f filters job, -fm tells glljobstat to only include this job
python3 /home/cmacmahon/glljobstat.py -i 5 -f $SLURM_JOB_ID -fm -hi > /home/cmacmahon/CosyMAML/logs/data/log_${SLURM_JOB_ID}_hist.log &
PID2=$!
 
echo "Monitoring processes started"

srun python3 /home/cmacmahon/CosyMAML/test_datagen.py \
--n_samples 100 \
--n_threads 32

sleep 10

echo "Killing monitoring processes..."
kill $PID1 $PID2
 
echo "Ending batch script"
exit 0