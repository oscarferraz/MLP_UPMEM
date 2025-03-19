#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"


__mram T d_l2[(TRAINING_SIZE/NR_DPUS)*L2_SIZE];
__mram T d_W2[L2_SIZE*OUTPUT_SIZE];
__mram T d_pred[NR_DPUS*OUTPUT_SIZE*8];



#if OUTPUT_SIZE < NR_TASKLETS
	#if GEMM == 1
		__dma_aligned T sum[NR_TASKLETS];
	#else
		#if IS_FLOAT == 1
			__dma_aligned T sum[NR_TASKLETS]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
		#endif
		#if IS_INT == 1
			__dma_aligned T sum[NR_TASKLETS]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		#endif
	#endif
#endif



int main(){


	// d_l2[0]=1;
	// d_W1[0]=1;

	#if OUTPUT_SIZE >= NR_TASKLETS
		kDot_m1_m2T_Output(d_l2, d_W2, d_pred);
	#else
		kDot_m1_m2T_Output(d_l2, d_W2, d_pred, sum);
	#endif


	/*for(int i = 0; i < Ceil(((float)TRAINING_SIZE/(NR_DPUS*NR_DPUS))/2)*2; i++)
	{
		for(int j = 0; j < L1_SIZE; j++)
		{
			printf("%f ", d_l1[i*L1_SIZE+j]);
		}
		printf("\n");
	}*/

}
