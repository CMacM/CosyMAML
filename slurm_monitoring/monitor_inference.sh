#!/bin/bash
#SBATCH --account=dp364
#SBATCH --time=00:10:00
#SBATCH --job-name=maml_gpu_monitor
#SBATCH --qos=dev

#SBATCH --partition=gpu-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1 
#SBATCH --gres=gpu:4

#SBATCH --output=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.std
#SBATCH --error=/home/dp364/dp364/dc-macm1/slurm_logs/JOB_%j.err

# Get job ID
SLURM_JOB_ID=$(echo $SLURM_JOB_ID)

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: SLURM_JOB_ID not defined."
    exit 1
fi

# Load conda and activate the environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cosymaml

# Create a runtime multi-prog config with the actual job ID
CONF_TEMPLATE="$HOME/CosyMAML/slurm_monitoring/parallel_chains.template"
CONF_ACTUAL="$HOME/CosyMAML/slurm_monitoring/parallel_chains.conf"

sed "s/JOBID/$SLURM_JOB_ID/g" "$CONF_TEMPLATE" > "$CONF_ACTUAL"

# Start time
START_TIME=$(date +%s)

# Run 4 parallel chains
echo "Launching 4 parallel chains (1 per GPU)"
srun --multi-prog --cpu-bind=none $HOME/CosyMAML/slurm_monitoring/parallel_chains.conf

# End time
END_TIME=$(date +%s)
MAKESPAN=$(($END_TIME - $START_TIME))
echo "Total makespan: $MAKESPAN seconds"

echo "Parsing detailed strace logs for I/O bytes..."

TOTAL_BYTES=0
for i in {0..3}; do
    STRACE_LOG="/home/dp364/dp364/dc-macm1/slurm_logs/strace_chain_${i}_${SLURM_JOB_ID}.log"
    if [[ -f "$STRACE_LOG" ]]; then
        BYTES=$(awk '
            $0 ~ /read\(|write\(/ {
                match($0, /= ([0-9]+)/, m);
                if (m[1] ~ /^[0-9]+$/) sum += m[1];
            }
            END {print sum}
        ' "$STRACE_LOG")
        echo "Chain $i I/O: $BYTES bytes"
        TOTAL_BYTES=$((TOTAL_BYTES + BYTES))
    else
        echo "Missing strace log for chain $i."
    fi
done

TOTAL_MB=$(echo "scale=2; $TOTAL_BYTES / 1024 / 1024" | bc)
echo "Total I/O across all chains: $TOTAL_BYTES bytes (~$TOTAL_MB MB)"

