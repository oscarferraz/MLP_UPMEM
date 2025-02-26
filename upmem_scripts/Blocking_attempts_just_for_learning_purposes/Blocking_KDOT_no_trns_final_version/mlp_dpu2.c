#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels2_mram.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

__mram T d_test1[(test1_rows/NR_DPUS+1)*test1_cols];
__mram T d_test2[test2_rows*(test2_cols/NR_DPUS+2)];
__mram T d_res[(test1_rows/NR_DPUS+1)*(test2_cols/NR_DPUS+2)];

__mram_ptr float* aux;

__dma_aligned T cache1[100*100]; // cache in WRAM to perform transfer
__dma_aligned T cache2[100*100]; // cache in WRAM to perform transfer


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
	

	aux = kDot(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, Ceil((float)test1_rows/NR_DPUS), test1_cols, Ceil(((float)(test2_cols/NR_DPUS)/2))*2, barr_addr);

	
	printf("\n");
	/*for(int i = 0; i < (test2_rows); i++)
	{
		for(int j = 0; j < Ceil(((float)(test2_cols/NR_DPUS)/2))*2; j++)
		{
			printf("%lf ", d_test2[i*Ceil(((float)(test2_cols/NR_DPUS)/2))*2+j]);
		}
		printf("\n");
		//printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}*/

	/*for(int i = 0; i < Ceil(test1_rows/NR_DPUS); i++)
	{
		for(int j = 0; j < test1_cols; j++)
		{
			printf("%.1f ", d_test1[i*test1_cols+j]);
		}
		printf("\n");
		//printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}*/

	/*for(int i = 0; i < Ceil(test1_rows/NR_DPUS); i++)
	{
		for(int j = 0; j < Ceil(((float)(test2_cols/NR_DPUS)/2))*2; j++)
		{
			printf("%.1f ", d_res[i*Ceil(((float)(test2_cols/NR_DPUS)/2))*2+j]);
		}
		printf("\n");
		//printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}*/

	return 0;
}
