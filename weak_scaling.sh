#!/bin/bash

make clean

# Shared Memory Version with different tree and thread counts
make Comparison
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 40 -m 100 -p > output_2_threads_40_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 80 -m 100 -p > output_4_threads_80_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 160 -m 100 -p > output_8_threads_160_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 320 -m 100 -p > output_16_threads_320_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 640 -m 100 -p > output_32_threads_640_trees.log 2>&1 &