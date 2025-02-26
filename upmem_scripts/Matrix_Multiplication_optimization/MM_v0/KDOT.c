#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "kdot_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

//since block's dimentions might round up due to ceil we have
//to allocate based on that assumption
__mram T d_m1[(M1_ROWS/NR_DPUS+1)*M1_COLS];
__mram T d_m2[(M1_COLS)*(M2_COLS/NR_DPUS+2)];
__mram T d_res[(M1_ROWS/NR_DPUS+1)*(M2_COLS/NR_DPUS+2)];

__mram_ptr T* aux;



int main()
{
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);
	
	//run kdot (matrix multiplication) in DPUs
	aux = kDot(NR_TASKLETS, d_m1, d_m2, d_res, Ceil((float)M1_ROWS/NR_DPUS), M1_COLS, Ceil(((float)(M2_COLS/NR_DPUS)/2)*2), barr_addr);
	
	//print stuff for debugging purposes
	/*for(int i = 0; i < (Ceil(M1_ROWS/NR_DPUS)); i++)
	{
		for(int j = 0; j < (M1_COLS); j++)
		{
			printf("%1.f ", d_m1[i*(M1_COLS)+j]);
		}
		printf("\n");
	}*/

}
