#!/bin/bash

make clean
#make Comparison
make RandomForestMPI

# Magic04 dataset benchmarks
# Shared Memory Version with different thread counts
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_2_magic.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_4_magic.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_8_magic.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_16_magic.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_24_magic.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/magic04.data" -p -t 100 -m 100 > output_sh_32_magic.log 2>&1

# Distributed Memory Version with different node counts
for N in 1 2 3 4 5 6 7 8; do
    echo "Running with $N nodes"
    srun --nodes=$N --ntasks-per-node=1 --time=00:3:00 \
    --mpi=pmix ./RandomForestMPI "./test/magic04.data" -t 256 -md 100 > mpi_output_${N}_nodes_magic.log 2>&1
done

# Covertype dataset benchmarks
# Shared Memory Version with different thread countsù
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_2_covertype.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_4_covertype.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_8_covertype.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_16_covertype.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_24_covertype.log 2>&1
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=00:10:00 ./Comparison "./test/covertype.data" -p -t 100 -m 100 > output_sh_32_covertype.log 2>&1
# Distributed Memory Version with different node counts
for N in 1 2 3 4 5 6 7 8; do
    echo "Running with $N nodes"
    srun --nodes=$N --ntasks-per-node=1 --time=00:3:00 \
    --mpi=pmix ./RandomForestMPI "./test/covertype.data" -t 256 -md 100 > mpi_output_${N}_nodes_covertype.log 2>&1
done