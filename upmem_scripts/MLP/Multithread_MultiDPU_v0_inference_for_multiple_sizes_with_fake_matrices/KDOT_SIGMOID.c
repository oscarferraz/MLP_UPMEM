#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

__mram T d_X[(TRAINING_SIZE/NR_DPUS+1)*TRAINING_DIM];
__mram T d_W0[(TRAINING_DIM)*(L1_SIZE/NR_DPUS+2)];
__mram T d_l1[(TRAINING_SIZE/NR_DPUS+1)*(L1_SIZE/NR_DPUS+2)];

__mram_ptr T* aux;

int main()
{
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);
	
	aux = kDot(NR_TASKLETS, d_X, d_W0, d_l1, Ceil((float)TRAINING_SIZE/NR_DPUS), TRAINING_DIM, Ceil(((float)(L1_SIZE/NR_DPUS)/2)*2), barr_addr);
	aux = kSigmoid(NR_TASKLETS,Ceil((float)TRAINING_SIZE/NR_DPUS), Ceil(((float)(L1_SIZE/NR_DPUS)/2)*2), d_l1, d_l1, barr_addr);
	
	/*for(int i = 0; i < (Ceil(TRAINING_SIZE/NR_DPUS)); i++)
	{
		for(int j = 0; j < (Ceil((float)(L1_SIZE/NR_DPUS)/2)*2); j++)
		{
			printf("%f ", d_l1[i*(Ceil((float)(L1_SIZE/NR_DPUS)/2)*2)+j]);
		}
		printf("\n");
	}*/
}
