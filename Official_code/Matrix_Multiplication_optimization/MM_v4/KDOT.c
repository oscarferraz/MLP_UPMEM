#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "kdot_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
	__host float d_m1[M1_COLS];
	__host float d_m2[M2_ROWS*WORKLOAD_CONFIG];
#endif
#if WRAM_CONFIG == 2
	__host float d_m1[M1_COLS*WORKLOAD_CONFIG];
	__host float d_m2[M2_ROWS];
#endif

__host float d_res[WORKLOAD_CONFIG];


// __dma_aligned float* aux;



int main()
{

	// printf("(M1_ROWS/NR_DPUS+1)*M1_COLS = %d\n", (M1_ROWS/NR_DPUS+1)*M1_COLS);
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);

	/* for(int i = 0; i < M2_ROWS*WORKLOAD_CONFIG; i++){
		printf("m2[%d]=%f", i, d_m2[i]);
    }

	printf("\n"); */
	
	//test kernels
	//*aux = kMartixByMatrixElementwise(NR_TASKLETS, 60, 50, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kMartixSubstractMatrix(NR_TASKLETS, 60, 60, d_test1, d_test2, d_res, cache1, cache2, barr_addr);
	//*aux = kSigmoid(NR_TASKLETS, 60, 60, d_test1, d_res, cache1, barr_addr);
	//*aux = kSigmoid_d(NR_TASKLETS, 60, 50, d_test1, d_res, cache1, barr_addr);
	kDot_m1_m2T( d_m1, d_m2, d_res);
	//*aux = kDot_m1_m2T(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1T_m2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	
	//printf("\n\nceil val is: %d\n", Ceil((((T)M2_ROWS/NR_DPUS)/2))*2);
	/* for(int i = 0; i < WORKLOAD_CONFIG; i++){
		printf("res[%d]=%f\n", i, d_res[i]);
	} */



	// return 0;

}
