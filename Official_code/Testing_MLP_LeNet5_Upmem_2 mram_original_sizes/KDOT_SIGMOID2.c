#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"


__mram T d_l1[(TRAINING_SIZE/NR_DPUS)*L1_SIZE];
__mram T d_W1[L1_SIZE*L2_SIZE];
__mram T d_l2[(TRAINING_SIZE/NR_DPUS)*L2_SIZE];


int main(){

	// d_l1[0]=1;
	// d_W1[0]=1;
	

	kDot_m1_m2T_L2(d_l1, d_W1, d_l2);

	/*for(int i = 0; i < (Ceil(TRAINING_SIZE/NR_DPUS)); i++)
	{
		for(int j = 0; j < (Ceil((float)(L1_SIZE/NR_DPUS)/2)*2); j++)
		{
			printf("%f ", d_l1[i*(Ceil((float)(L1_SIZE/NR_DPUS)/2)*2)+j]);
		}
		printf("\n");
	}*/
}
