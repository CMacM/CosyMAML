#!/bin/bash
#SBATCH --account=dp364
#SBATCH --time=00:10:00
#SBATCH --job-name=maml_gpu_monitor
#SBATCH --qos=dev

# Use 32 tasks for A100-40 or unspecified, 48 for A100-80
#SBATCH --partition=gpu
#SBATCH --nodes=1

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

# Start glljobstat monitoring in the background
# Runs in the base environment job is submitted from
# python3 $HOME/glljobstat.py -i 5 -f $SLURM_JOB_ID -fm -r > $HOME/CosyMAML/logs/data/MAML_pipe_log_${SLURM_JOB_ID}_rate.log &
# GLL_RATE_PID=$!

# # -f filters job, -fm tells glljobstat to only include this job
# python3 $HOME/glljobstat.py -i 5 -f $SLURM_JOB_ID -fm -hi > $HOME/CosyMAML/logs/data/MAML_pipe_log_${SLURM_JOB_ID}_hist.log &
# GLL_HIST_PID=$!

# sleep 5

# Add a header to the log file
echo \
    "timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm" \
    > $HOME/CosyMAML/logs/gpu/gpu_metrics_$SLURM_JOB_ID.csv

# Then start logging values only
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
    --format=csv,noheader,nounits \
    --loop=5 >> $HOME/CosyMAML/logs/gpu/gpu_metrics_$SLURM_JOB_ID.csv &
GPU_MONITOR_PID=$!

# Run your AI pipeline
srun python3 $HOME/CosyMAML/train_MAML_model.py \
    --device cuda \
    --trainfile $HOME/spectra_data/cl_ee_200tasks_5000samples_seed456.h5 \
    --mcmc_trainfile $HOME/spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5 \
    --model_dir $HOME/model_weights/ \
    --log_dir $HOME/CosyMAML/logs/io/

# Stop GPU monitoring
kill $GPU_MONITOR_PID

# Stop glljobstat monitoring
# kill $GLL_RATE_PID, $GLL_HIST_PID