#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels2.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_test1[60*50];
__mram float d_test2[60*50];
__mram float d_res[60*50];


__mram_ptr float* aux[60*50];



__dma_aligned float cache1[60*50]; // cache in WRAM to perform transfer
__dma_aligned float cache2[60*50]; // cache in WRAM to perform transfer


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
	
	//test kernels
	*aux = kMartixByMatrixElementwise(NR_TASKLETS, 60, 50, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kMartixSubstractMatrix(NR_TASKLETS, 60, 60, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kSigmoid(NR_TASKLETS, 60, 60, d_test1, d_res, cache1, barr_addr);
	//*aux = kSigmoid_d(NR_TASKLETS, 60, 50, d_test1, d_res, cache1, barr_addr);
	//*aux = kDot(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1_m2T(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1T_m2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	
	/*for(int i = 0; i < 10; i++)
	{
		printf("DPUnr: Result: %f\n", d_res[i]);
	}*/

	/*
		kFit(NR_TASKLETS, cache1, cache2, barr_addr, d_X,
		TRAINING_SIZE, TRAINING_DIM, d_Y, 1, d_layer_1, 8, d_layer_1_delta,
		d_pred, d_pred_delta, d_W0, d_W1, d_buffer);
	*/	


	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_ELEM_PER_DPU);
	}
	return 0;
}
