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


__mram_ptr float* aux[M1_ROWS*M2_COLS];

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
	


	//*aux = kDot_WRAM2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, M1_ROWS, M1_COLS, M2_COLS, barr_addr);
	*aux = kDot_MRAM(NR_TASKLETS, d_test1, d_test2, d_res, M1_ROWS, M1_COLS, M2_COLS, barr_addr);
	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_TASKLETS);
	}
	return 0;
}
