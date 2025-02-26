#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "GEMM_kernel.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_test1[M1_ROWS*M1_COLS];
__mram float d_test2[M2_ROWS*M2_COLS];
__mram float d_res[M1_ROWS*M2_COLS];


__mram_ptr float* aux;



__dma_aligned float cache1[BLOCK_SIZE]; // cache in WRAM to perform transfer
__dma_aligned float cache2[BLOCK_SIZE]; // cache in WRAM to perform transfer


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
	//*aux = kMartixByMatrixElementwise(NR_TASKLETS, 8, 8, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kMartixSubstractMatrix(NR_TASKLETS, 60, 60, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kSigmoid(NR_TASKLETS, 60, 60, d_test1, d_res, cache1, barr_addr);
	//*aux = kSigmoid_d(NR_TASKLETS, 60, 50, d_test1, d_res, cache1, barr_addr);
	//*aux = kDot(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1_m2T(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1T_m2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	
	/*for(int i = 0; i < 2048; i++)
	{
		printf("Result: %f\n", d_test1[i]);
	}*/

	/*
		kFit(NR_TASKLETS, cache1, cache2, barr_addr, d_X,
		TRAINING_SIZE, TRAINING_DIM, d_Y, 1, d_layer_1, 8, d_layer_1_delta,
		d_pred, d_pred_delta, d_W0, d_W1, d_buffer);
	*/	


	aux = kDot_WRAM2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, M1_ROWS, M1_COLS, M2_COLS, barr_addr);
	//*aux = kDot_MRAM(NR_TASKLETS, d_test1, d_test2, d_res, M1_ROWS, M1_COLS, M2_COLS, barr_addr);
	for(int i = 0; i < 400; i++)
	{
		printf("Pos: %d, Result: %f\n",i, d_res[i]);
	}
	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_TASKLETS);
	}
	return 0;
}
