#!/bin/bash

make clean
make Comparison

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output1.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output2.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:1:00 ./Comparison "./test/magic04.data" -t 100 -m 100 > output3.log 2>&1

make RandomForestMPI

for N in 1 2 4 8; do
    echo "Running with $N nodes"
    srun --nodes=$N --ntasks-per-node=1 --time=00:2:00 \
                --mpi=pmix ./my_mpi_program ./RandomForestMPI "./test/magic04.data" -t 100 -md 100 -m t > mpi_output_${N}_nodes.log 2>&1
done