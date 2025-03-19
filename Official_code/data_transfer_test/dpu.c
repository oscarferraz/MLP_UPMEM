#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

// #include "host.c"


__host __dma_aligned T d_X[SIZE];

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

int main(){
    for(int i = 0; i < SIZE; i++){
        d_X[i]+=1;
    }
}

