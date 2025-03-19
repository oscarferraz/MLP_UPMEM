#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

// #include "host.c"


__mram T d_X[SIZE];

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

int main(){
    T d_X_W[SIZE];
    /* for(int i = 0; i < SIZE; i++){
        printf("before mram[%d]=%d\n",i , d_X[i]);
    } */
    mram_read(d_X, d_X_W, SIZE*sizeof(T));

    for(int i = 0; i < SIZE; i++){
        d_X_W[i]=2;
    }

    mram_write(d_X_W, d_X, SIZE*sizeof(T));

    /* for(int i = 0; i < SIZE; i++){
        printf("after mram[%d]=%d\n",i , d_X[i]);
    } */
}

