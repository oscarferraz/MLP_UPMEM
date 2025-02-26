#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram int d_X[NR_ELEM_PER_DPU];
__mram int d_W0[NR_ELEM_PER_DPU];
__mram int d_layer_1[NR_ELEM_PER_DPU];


__mram int prediction[NR_ELEM_PER_DPU];

__dma_aligned int cache1[CACHE_SIZE]; // cache in WRAM to perform transfer
__dma_aligned int cache2[CACHE_SIZE]; // cache in WRAM to perform transfer


__mram_ptr int* aux[NR_ELEM_PER_DPU];
//int X_res[NR_ELEM_PER_DPU * 1];

int main()
{
	barrier_t *barr_addr = &my_barrier;
	//me() gets the id of the thread that is running
	if(me() == 0)
	{
		printf("Nr of Tasklets is: %d\n", NR_TASKLETS);
		perfcounter_config(COUNT_CYCLES, true);
	}
	barrier_wait(barr_addr);
	
	/*for(int i = 0; i < 16; i++)
	{
		printf("Value is : %d \n", d_X[i]);
		//printf("Value is : %d \n", d_W0[i]);
		//printf("Value is : %d \n", d_layer_1[i]);

	}*/

	*aux = kDot(NR_TASKLETS, CACHE_SIZE, d_X, d_W0, d_layer_1, cache1, cache2, TRAINING_SIZE, TRAINING_DIM, 8, barr_addr);

	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_ELEM_PER_DPU);
	}
	return 0;
}