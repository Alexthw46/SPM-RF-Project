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

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output1.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output2.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output3.log 2>&1

for N in 1 2 4 8; do
    echo "Running with $N nodes"
    srun --nodes=$N --ntasks-per-node=1 --time=00:2:00 \
                --mpi=pmix ./my_mpi_program ./RandomForestMPI "./test/magic04.data" -t 100 -md 100 -m t > mpi_output_${N}_nodes.log 2>&1
done