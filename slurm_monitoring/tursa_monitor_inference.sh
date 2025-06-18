#!/bin/bash
#SBATCH --account=dp364
#SBATCH --time=00:10:00
#SBATCH --job-name=maml_gpu_monitor
#SBATCH --qos=dev

#SBATCH --partition=gpu-a100-40
#SBATCH --nodes=2                      # Adjust based on GPUs per node
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

#SBATCH --output=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.std
#SBATCH --error=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.err

# Load conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cosymaml

# SLURM job ID
JOBID="${SLURM_JOB_ID}"
if [ -z "$JOBID" ]; then
    echo "Error: SLURM_JOB_ID not defined."
    exit 1
fi

# Number of chains to run (can override with sbatch --export=NUM_CHAINS=...)
NUM_CHAINS=${NUM_CHAINS:-$((SLURM_JOB_NUM_NODES * 4))} # Default to 4 chains per node

# Output config path
CONF_ACTUAL="$HOME/CosyMAML/slurm_monitoring/parallel_chains.conf"
> "$CONF_ACTUAL"

# Generate config
for i in $(seq 0 $((NUM_CHAINS - 1))); do
    echo "$i python3 /home/dp364/dp364/dc-macm1/CosyMAML/run_MAML_emcee.py --device=cuda --data_dir=/home/dp364/dp364/dc-macm1/ --chain_id=$i" >> "$CONF_ACTUAL"
done

echo "Launching $NUM_CHAINS chains..."
START_TIME=$(date +%s)

srun --ntasks=$NUM_CHAINS --multi-prog --cpu-bind=none "$CONF_ACTUAL"

END_TIME=$(date +%s)
MAKESPAN=$((END_TIME - START_TIME))
echo "Total makespan: $MAKESPAN seconds"
