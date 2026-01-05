#!/bin/bash

make clean

# Shared Memory Version with different tree and thread counts
make Comparison
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:20:00 ./Comparison "./test/covertype.csv" -t 40 -m 100 -p > covertype_2_threads_40_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:20:00 ./Comparison "./test/covertype.csv" -t 80 -m 100 -p > covertype_4_threads_80_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 160 -m 100 -p > covertype_8_threads_160_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 320 -m 100 -p > covertype_16_threads_320_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 640 -m 100 -p > covertype_32_threads_640_trees.log 2>&1 &

# Shared Memory Version with different thread counts, test set size varies according to number of threads
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:30:00 ./Comparison "./test/covertype.csv" -t 64 -m 100 -p > covertype_2_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:20:00 ./Comparison "./test/covertype.csv" -t 64 -m 100 -p > covertype_4_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:15:00 ./Comparison "./test/covertype.csv" -t 64 -m 100 -p > covertype_8_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 64 -m 100 -p > covertype_16_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/covertype.csv" -t 64 -m 100 -p > covertype_32_threads_wsp.log 2>&1 &

# Using magic04.data
# Shared Memory Version with different tree and thread counts
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:20:00 ./Comparison "./test/magic04.data" -t 40 -m 100 -p > magic_2_threads_40_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:20:00 ./Comparison "./test/magic04.data" -t 80 -m 100 -p > magic_4_threads_80_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:10:00 ./Comparison "./test/magic04.data" -t 160 -m 100 -p > magic_8_threads_160_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/magic04.data" -t 320 -m 100 -p > magic_16_threads_320_trees.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/magic04.data" -t 640 -m 100 -p > magic_32_threads_640_trees.log 2>&1 &

# Shared Memory Version with different thread counts, test set size varies according to number of threads
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:30:00 ./Comparison "./test/magic04.data" -t 64 -m 100 -p > magic_2_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:20:00 ./Comparison "./test/magic04.data" -t 64 -m 100 -p > magic_4_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:15:00 ./Comparison "./test/magic04.data" -t 64 -m 100 -p > magic_8_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/magic04.data" -t 64 -m 100 -p > magic_16_threads_wsp.log 2>&1 &
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/magic04.data" -t 64 -m 100 -p > magic_32_threads_wsp.log 2>&1 &