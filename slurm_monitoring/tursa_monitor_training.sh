#!/bin/bash
#SBATCH --account=dp364
#SBATCH --time=00:20:00
#SBATCH --job-name=maml_gpu_monitor
#SBATCH --qos=standard

#SBATCH --partition=gpu-a100-40
#SBATCH --nodes=1                     # Adjust based on GPUs per node
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

# Number of learners to run (can override with sbatch --export=NUM_LEARNERS=...)
NUM_LEARNERS=${NUM_LEARNERS:-$((SLURM_JOB_NUM_NODES * 4))} # Default to 4 learners per node

# Output config path
CONF_ACTUAL="$HOME/CosyMAML/slurm_monitoring/parallel_train.conf"
> "$CONF_ACTUAL"

# Generate config
for i in $(seq 0 $((NUM_LEARNERS - 1))); do
    echo "$i python3 /home/dp364/dp364/dc-macm1/CosyMAML/train_MAML_model.py \
    --device=cuda \
    --trainfile=/home/dp364/dp364/dc-macm1/spectra_data/cl_ee_200tasks_5000samples_seed456.h5 \
    --model_dir=/home/dp364/dp364/dc-macm1/model_weights/ \
    --learner_id=$i \
    --seed=$((SLURM_JOB_ID * 1000 + i))" >> "$CONF_ACTUAL"
done

echo "Launching $NUM_LEARNERS learners..."
START_TIME=$(date +%s)

srun --ntasks=$NUM_LEARNERS --multi-prog --cpu-bind=none "$CONF_ACTUAL"

END_TIME=$(date +%s)
MAKESPAN=$((END_TIME - START_TIME))
echo "Total makespan: $MAKESPAN seconds"
