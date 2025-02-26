#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_X[NR_ELEM_PER_DPU];
__mram float d_W0[NR_ELEM_PER_DPU];
__mram float d_layer_1[NR_ELEM_PER_DPU];
__mram float d_buffer[NR_ELEM_PER_DPU];
__mram float d_layer_1_delta[NR_ELEM_PER_DPU];
__mram float d_W1[NR_ELEM_PER_DPU];
__mram float d_Y[NR_ELEM_PER_DPU];
__mram float d_pred[NR_ELEM_PER_DPU];
__mram float d_pred_delta[NR_ELEM_PER_DPU];

__mram float prediction[NR_ELEM_PER_DPU];

__dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE]; // cache in WRAM to perform transfer
__dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE]; // cache in WRAM to perform transfer




//__host float X_res[NR_ELEM_PER_DPU * 1];
__mram float X_res[NR_ELEM_PER_DPU];

__host float *aux;
//float X_res[NR_ELEM_PER_DPU * 1];

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
	//kMartixByMatrixElementwise(NR_TASKLETS, TRAINING_SIZE, TRAINING_DIM, CACHE_SIZE, d_X, d_X, X_res, cache1, cache2, barr_addr);
	//kMartixSubstractMatrix(NR_ELEM_PER_TASKLETS, TRAINING_SIZE, TRAINING_DIM, CACHE_SIZE, d_X, d_X, X_res, cache1, cache2, barr_addr); 
	//aux = kSigmoid(NR_ELEM_PER_TASKLETS, CACHE_SIZE, d_X, X_res, cache1, barr_addr);
	//aux = kSigmoid_d(NR_ELEM_PER_TASKLETS, CACHE_SIZE, d_X, X_res, cache1, barr_addr);
	//aux = kDot(NR_ELEM_PER_TASKLETS, CACHE_SIZE, d_X, d_X, X_res, cache1, cache2, 4, 4, 4, barr_addr);
	//kDot_m1_m2T(NR_TASKLETS, CACHE_SIZE, d_X, d_X, X_res, cache1, cache2,4, 4, 4, barr_addr);
	//kDot_m1T_m2(NR_TASKLETS, CACHE_SIZE, d_X, d_X, X_res, cache1, cache2, 4, 4, 4, barr_addr);
	
	//for(int i = 0; i < TRAINING_SIZE*TRAINING_DIM; i++)
	//{
	//	printf("Result: %f\n", d_X[i]);
	//}


		kFit(NR_TASKLETS,CACHE_SIZE, cache1, cache2, barr_addr, d_X,
		TRAINING_SIZE, TRAINING_DIM, d_Y, 1, d_layer_1, 8, d_layer_1_delta,
		d_pred, d_pred_delta, d_W0, d_W1, d_buffer);
		
	//barrier_wait(barr_addr);
	/*kTest(NR_ELEM_PER_TASKLETS,CACHE_SIZE, cache1, cache2, barr_addr, [5.1, 3.5, 1.4, 0.2],
						TRAINING_SIZE, 1, 0, 1, d_layer_1, 8, d_layer_1_delta,
						d_pred, d_pred_delta, d_W0, d_W1, d_buffer, prediction);*/


	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_ELEM_PER_DPU);
	}
	return 0;
}