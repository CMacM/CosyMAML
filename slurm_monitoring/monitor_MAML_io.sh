#!/bin/bash
#SBATCH --output=/home/cmacmahon/slurm/maml_pipe-%j.std
#SBATCH --error=/home/cmacmahon/slurm/maml_pipe-%j.err
#SBATCH --job-name=maml_pipe_monitor
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=10:00:00

# Retrieve the SLURM_JOB_ID
env | grep -i job
SLURM_JOB_ID=$(env | grep -i SLURM_JOB_ID | cut -d '=' -f2)
 
# Check if SLURM_JOB_ID is defined
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: SLURM_JOB_ID not defined."
    exit 1
fi

# Export SLURM job ID in container environment
export SLURM_JOB_ID=$SLURM_JOB_ID
export SLURM_JOBID=$SLURM_JOB_ID

#! Load conda and activate the environment for the application
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

echo "Starting monitoring"
# Start glljobstat monitoring in the background
# Runs in the base environment job is submitted from
python3 /home/cmacmahon/glljobstat.py -i 10 -f $SLURM_JOB_ID -fm -r > /home/cmacmahon/CosyMAML/logs/data/MAML_pipe_log_${SLURM_JOB_ID}_rate.log &
PID1=$!

# -f filters job, -fm tells glljobstat to only include this job
python3 /home/cmacmahon/glljobstat.py -i 10 -f $SLURM_JOB_ID -fm -hi > /home/cmacmahon/CosyMAML/logs/data/MAML_pipe_log_${SLURM_JOB_ID}_hist.log &
PID2=$!
 
echo "Monitoring processes started"

sleep 5

srun python3 /home/cmacmahon/CosyMAML/full_MAML_pipeline.py \
--force_stop 5 \
--n_tasks 20 \
--n_samples 500 \
--batch_size 5 \
--max_iter 100 \
--trainfile /exafs/400NVX2/cmacmahon/spectra_data/cl_ee_200tasks_5000samples_seed456.h5 \
--mcmc_trainfile /exafs/400NVX2/cmacmahon/spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5

sleep 5

echo "Killing monitoring processes..."
kill $PID1 $PID2
 
echo "Ending batch script"
exit 0