#!/bin/bash
#SBATCH --output=/home/cmacmahon/slurm/maml_pipe-%j.std
#SBATCH --error=/home/cmacmahon/slurm/maml_pipe-%j.err
#SBATCH --job-name=maml_pipe_monitor
#SBATCH --nodes=5
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
source /home/cmacmahon/software/miniconda3/etc/profile.d/conda.sh
conda activate gllmon

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

conda activate cosymaml

echo "Starting MAML training"
srun --nodes=1 --exclusive \
    python3 /home/cmacmahon/CosyMAML/train_MAML_model.py \
    --force_stop 5 \

echo "Launching 5 parallel chains"
for CHAIN_ID in {0..4}; do
    srun --nodes=1 --exclusive \
        python3 /home/cmacmahon/CosyMAML/run_MAML_BMH.py \
        --max_iter 100 \
        --chain_id $CHAIN_ID &
done

# Wait for all chains to finish
wait

echo "Killing monitoring processes..."
kill $PID1 $PID2

echo "Ending batch script"
exit 0