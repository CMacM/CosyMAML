#!/bin/bash
#SBATCH --account=dp364
#SBATCH --time=00:10:00
#SBATCH --job-name=maml_gpu_monitor
#SBATCH --qos=dev

# Use 32 tasks for A100-40 or unspecified, 48 for A100-80
#SBATCH --partition=gpu-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1 
#SBATCH --gres=gpu:4

#SBATCH --output=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.std
#SBATCH --error=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.err

# Get job ID
SLURM_JOB_ID=$(echo $SLURM_JOB_ID)

# Check if SLURM_JOB_ID is defined
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: SLURM_JOB_ID not defined."
    exit 1
fi

MY_PID=$$

# Load conda and activate the environment for the application
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cosymaml

# Add a header to the nvidia-smi log file
echo \
    "gpu.id,timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm" \
    > $HOME/CosyMAML/logs/gpu/gpu_metrics_$SLURM_JOB_ID.csv

# Then start logging values only
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
    --format=csv,noheader,nounits \
    --loop=1 >> $HOME/CosyMAML/logs/gpu/gpu_metrics_$SLURM_JOB_ID.csv &
GPU_MONITOR_PID=$!

# Start IOSTAT monitoring
iostat -xt 1 > $HOME/CosyMAML/logs/io/iostat_$SLURM_JOB_ID.csv &
IOSTAT_PID=$!

# Run MAML training
srun --ntasks=1 --gres=gpu:1 \
    python3 $HOME/CosyMAML/train_MAML_model.py \
    --device cuda \
    --trainfile $HOME/spectra_data/cl_ee_200tasks_5000samples_seed456.h5 \
    --model_dir $HOME/model_weights/ \
    --log_dir $HOME/CosyMAML/logs/io/

# Run batched MH chains
echo "Launching 4 parallel chains (1 per GPU)"
srun --multi-prog --cpu-bind=none $HOME/CosyMAML/slurm_monitoring/parallel_chains.conf

#Stop GPU monitoring
kill $GPU_MONITOR_PID

# Stop IOSTAT monitoring
kill $IOSTAT_PID