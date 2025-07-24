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

# Automatically detect number of chains = number of nodes
NUM_LEARNERS=${NUM_LEARNERS:-$SLURM_JOB_NUM_NODES}
echo "Running $NUM_LEARNERS chains across $SLURM_JOB_NUM_NODES node(s)"

# --- CONFIG GENERATION ---
CONF_ACTUAL="$HOME/CosyMAML/slurm_monitoring/parallel_chains.conf"
> "$CONF_ACTUAL"
for i in $(seq 0 $((NUM_CHAINS - 1))); do
    echo "$i python3 $HOME/CosyMAML/train_MAML_model.py \
    --device cpu \
    --trainfile /exafs/400NVX2/cmacmahon/spectra_data/cl_ee_200tasks_5000samples_seed456.h5 \
    --model_dir /exafs/400NVX2/cmacmahon/model_weights/ \
    --learner_id=$i" >> "$CONF_ACTUAL"
done

#! Load conda and activate the environment for the application
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

echo "Starting monitoring"
# Start glljobstat monitoring in the background
# Runs in the base environment job is submitted from
python3 /home/cmacmahon/glljobstat.py -i 1 -f $SLURM_JOB_ID -fm -r > /home/cmacmahon/CosyMAML/logs/io/MAML_pipe_log_${SLURM_JOB_ID}_rate.log &
PID1=$!

# -f filters job, -fm tells glljobstat to only include this job
python3 /home/cmacmahon/glljobstat.py -i 1 -f $SLURM_JOB_ID -fm -hi > /home/cmacmahon/CosyMAML/logs/io/MAML_pipe_log_${SLURM_JOB_ID}_hist.log &
PID2=$!
 
echo "Monitoring processes started"

sleep 5

# --- LAUNCH CHAINS ---
echo "Launching $NUM_CHAINS chains (1 per node)"
START_TIME=$(date +%s)

srun --multi-prog "$CONF_ACTUAL"
SRUN_EXIT_CODE=$?

END_TIME=$(date +%s)
MAKESPAN=$((END_TIME - START_TIME))
echo "Total makespan: $MAKESPAN seconds"

# --- CLEANUP ---
sleep 5
echo "Killing monitoring processes..."
kill $PID1 $PID2

echo "Ending batch script."
exit 0