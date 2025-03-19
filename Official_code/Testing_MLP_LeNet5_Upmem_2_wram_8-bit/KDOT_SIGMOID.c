#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"


__host T d_X[(TRAINING_SIZE/NR_DPUS)*TRAINING_DIM];
__host T d_W0[TRAINING_DIM*L1_SIZE];
__host T d_l1[(TRAINING_SIZE/NR_DPUS)*L1_SIZE];



int main(){

	// printf("OK\n");
	// d_X[0]=1;
	// d_W0[0]=1;

	kDot_m1_m2T_L1(d_X, d_W0, d_l1);

	/* for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			printf("d_W0[%d]=%f ",i*TRAINING_DIM+j, d_W0[i*TRAINING_DIM+j]);
		}
		printf("\n");
	} */
}
