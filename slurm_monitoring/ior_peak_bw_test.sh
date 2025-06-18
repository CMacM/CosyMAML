#!/bin/bash -l
#SBATCH --job-name=ior_peak_test
#SBATCH --output=slurm/ior_peak_test.%j.out
#SBATCH --error=slurm/ior_peak_test.%j.err
#SBATCH --nodes=14                    # Adjust based on available nodes
#SBATCH --ntasks-per-node=1          # Total tasks = nodes * tasks-per-node
#SBATCH --time=00:10:00              # Wall time
#SBATCH --exclusive                  # Optional: ensure full-node usage

module load ior/Ubuntu/24.04/gcc-13.3.0/openmpi/4.0.0                      # Or specify the path to `ior` manually

# IOR parameters
IOR_OUTPUT="/exafs/400NVX2/cmacmahon/ior_testfile"  # Set to a Lustre-mounted directory
BLOCK_SIZE="1408M"                             # Per process file size
TRANSFER_SIZE="16M"                          # Per write size
ITERATIONS=3

echo "==== Starting IOR Write Test ===="
srun ior -a MPIIO -b $BLOCK_SIZE -t $TRANSFER_SIZE -F -rw -i $ITERATIONS -o ${IOR_OUTPUT}
echo "==== Completed Test ===="