#!/bin/bash
#SBATCH --output=/home/cmacmahon/slurm/maml_pipe-%j.std
#SBATCH --error=/home/cmacmahon/slurm/maml_pipe-%j.err
#SBATCH --job-name=maml_pipe_monitor
#SBATCH --nodes=14
#SBATCH --ntasks=14             # 1 task per node
#SBATCH --time=10:00:00

# --- ENV SETUP ---
SLURM_JOB_ID=${SLURM_JOB_ID:-$(env | grep -i SLURM_JOB_ID | cut -d '=' -f2)}
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: SLURM_JOB_ID not defined."
    exit 1
fi
export SLURM_JOBID=$SLURM_JOB_ID

# Automatically detect number of chains = number of nodes
NUM_CHAINS=${NUM_CHAINS:-$SLURM_JOB_NUM_NODES}
echo "Running $NUM_CHAINS chains across $SLURM_JOB_NUM_NODES node(s)"

# --- CONFIG GENERATION ---
CONF_ACTUAL="$HOME/CosyMAML/slurm_monitoring/parallel_chains.conf"
> "$CONF_ACTUAL"
for i in $(seq 0 $((NUM_CHAINS - 1))); do
    echo "$i python3 /home/cmacmahon/CosyMAML/run_MAML_emcee.py --device=cpu --chain_id=$i" >> "$CONF_ACTUAL"
done

# --- ENVIRONMENT ---
source /home/cmacmahon/software/miniconda3/bin/activate
conda activate cosymaml

# --- MONITORING ---
echo "Starting monitoring..."
python3 /home/cmacmahon/glljobstat.py -c $NUM_CHAINS -i 1 -f $SLURM_JOB_ID -fm -r > /home/cmacmahon/CosyMAML/logs/io/MAML_pipe_log_${SLURM_JOB_ID}_rate.log &
PID1=$!
python3 /home/cmacmahon/glljobstat.py -c $NUM_CHAINS -i 1 -f $SLURM_JOB_ID -fm -hi > /home/cmacmahon/CosyMAML/logs/io/MAML_pipe_log_${SLURM_JOB_ID}_hist.log &
PID2=$!

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
