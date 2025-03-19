#!/bin/bash

# dpu-upmem-dpurte-clang   -DNR_TASKLETS=1 -O2 -o device device.c
# #dpu-upmem-dpurte-clang  -g -O0  -DNR_TASKLETS=1 -o device device.c
# gcc --std=c99   -D_POSIX_C_SOURCE=199309L  min_sum.c Common/results.c Common/ldpc_encode.c Common/channel.c Common/rand.c -o min_sum `dpu-pkg-config --cflags --libs dpu` -lm
# #gcc --std=c99 -g0 -D_POSIX_C_SOURCE=199309L  min_sum.c Common/results.c Common/ldpc_encode.c Common/channel.c Common/rand.c -o min_sum `dpu-pkg-config --cflags --libs dpu` -lm

# #dpu-lldb
# #file driver
# #process launch
# #exit

rm chrometf*
for ((i=1; i<=5; i++))
do
    echo "Iteration $i"
    dpu-profiling functions -o chrometf$i.json -A -- ./host

done