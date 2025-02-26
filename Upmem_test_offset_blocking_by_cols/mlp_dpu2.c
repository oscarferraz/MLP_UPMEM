#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels2_mram.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram T d_test2[test2_rows*(test2_cols/NR_DPUS+2)];
//__mram float d_res[(test1_rows/NR_DPUS+1)*(test2_cols/NR_DPUS+1)];

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
	



	
	printf("\n");
	for(int i = 0; i < (test2_rows); i++)
	{
		for(int j = 0; j < Ceil(((float)(test2_cols/NR_DPUS)/2))*2; j++)
		{
			printf("%lf ", d_test2[i*Ceil(((float)(test2_cols/NR_DPUS)/2))*2+j]);
		}
		printf("\n");
		//printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}

	return 0;
}
