#!/bin/bash
#SBATCH --job-name=aq4_rf_benchmark # Job name
#SBATCH --output=output.log # Standard output log file
#SBATCH --error=error.log # Standard error log file
#SBATCH --time=0:01:00 # Time limit (hh:mm:ss)
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of total tasks (MPI processes)
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --partition=normal # Partition to submit to
#SBATCH --mem=4G # Memory per node (4GB)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

make clean
make Comparison

srun ./Comparison "./test/magic04.data" -t 100 -m 100