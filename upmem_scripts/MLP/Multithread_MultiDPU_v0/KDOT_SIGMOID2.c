#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels2_mram.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_l1[(TRAINING_SIZE/NR_DPUS+1)*(L1_SIZE)];
__mram float d_W1[(L1_SIZE)*(1/NR_DPUS+2)];
__mram float d_pred[(TRAINING_SIZE/NR_DPUS+1)*(1/NR_DPUS+2)];

__mram_ptr float* aux;





int main()
{
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);
	
	//test kernels
	//*aux = kMartixByMatrixElementwise(NR_TASKLETS, 60, 50, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kMartixSubstractMatrix(NR_TASKLETS, 60, 60, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kSigmoid(NR_TASKLETS, 60, 60, d_test1, d_res, cache1, barr_addr);
	//*aux = kSigmoid_d(NR_TASKLETS, 60, 50, d_test1, d_res, cache1, barr_addr);
	aux = kDot(NR_TASKLETS, d_l1, d_W1, d_pred, Ceil((float)TRAINING_SIZE/NR_DPUS), L1_SIZE, Ceil(((float)(1/NR_DPUS)/2))*2, barr_addr);
	aux = kSigmoid(NR_TASKLETS,Ceil((float)TRAINING_SIZE/NR_DPUS), Ceil(((float)(1/NR_DPUS)/2))*2, d_pred, d_pred, barr_addr);
	//*aux = kDot_m1_m2T(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1T_m2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);

	printf("fodasse\n\n\n\n\n");
	/*for(int i = 0; i < Ceil((float)(TRAINING_SIZE/NR_DPUS)); i++)
	{
		for(int j = 0; j < L1_SIZE; j++)
		{
			printf("%f ", d_l1[i*L1_SIZE+j]);
		}
		printf("\n");
	}*/


	for(int i = 0; i < (Ceil((float)(TRAINING_SIZE/NR_DPUS))); i++)
	{
		for(int j = 0; j < (Ceil((float)(1/NR_DPUS)/2)*2); j++)
		{
			printf("%f ", d_pred[i*(Ceil((float)(1/NR_DPUS)/2)*2)+j]);
		}
		printf("\n");
	}

	
	/*printf("\n");
	for(int i = 0; i < (test2_rows/NR_DPUS+1)*test2_cols; i++)
	{
		printf("test2 nr: %d Result: %f\n",i, d_test2[i]);
	}*/

}
