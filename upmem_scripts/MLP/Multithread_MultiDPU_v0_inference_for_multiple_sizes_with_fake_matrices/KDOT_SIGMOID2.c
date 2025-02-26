#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

__mram T d_l1[(TRAINING_SIZE/(NR_DPUS*NR_DPUS)+2)*(L1_SIZE)];
__mram T d_W1[L1_SIZE];
__mram T d_pred[(TRAINING_SIZE/(NR_DPUS*NR_DPUS)+2)*(1)];

__mram_ptr T* aux;

int main()
{
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);

	aux = kDot(NR_TASKLETS, d_l1, d_W1, d_pred, Ceil(((float)TRAINING_SIZE/(NR_DPUS*NR_DPUS))/2)*2, L1_SIZE, 1, barr_addr);
	aux = kSigmoid(NR_TASKLETS,Ceil(((float)TRAINING_SIZE/(NR_DPUS*NR_DPUS))/2)*2, 1, d_pred, d_pred, barr_addr);

	/*for(int i = 0; i < Ceil(((float)TRAINING_SIZE/(NR_DPUS*NR_DPUS))/2)*2; i++)
	{
		for(int j = 0; j < L1_SIZE; j++)
		{
			printf("%f ", d_l1[i*L1_SIZE+j]);
		}
		printf("\n");
	}*/

	/*printf("W1 starts: \n");
	for(int i = 0; i < L1_SIZE; i++)
	{
		for(int j = 0; j < 1; j++)
		{
			printf("%f ", d_W1[i*1+j]);
		}
		printf("\n");
	}*/

	/*printf("prediction starts: \n");
	for(int i = 0; i < Ceil(((float)TRAINING_SIZE/(NR_DPUS*NR_DPUS))/2)*2; i++)
	{
		for(int j = 0; j < 1; j++)
		{
			printf("%f ", d_pred[i*1+j]);
		}
		printf("\n");
	}*/


	/*printf("\n");
	for(int i = 0; i < (test2_rows/NR_DPUS+1)*test2_cols; i++)
	{
		printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}*/

}
