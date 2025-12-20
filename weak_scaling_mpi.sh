#!/bin/bash

make clean
make RandomForestMPI

for N in 1 2 4 6 8; do
    echo "Running with $N nodes"
    trees=$((100 * N))
    srun --nodes=$N --ntasks-per-node=1 --time=00:3:00 \
                --mpi=pmix ./RandomForestMPI "./test/covertype.csv" -t $trees -md 100 > mpi_output_${N}_nodes_${trees}_trees.log 2>&1
done