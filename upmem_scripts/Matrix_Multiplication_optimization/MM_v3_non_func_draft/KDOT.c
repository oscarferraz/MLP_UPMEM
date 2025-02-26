#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "kdot_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_m1[(M1_ROWS/NR_DPUS+1)*M1_COLS];
__mram float d_m2[(M2_ROWS/NR_DPUS+2)*(M1_COLS)];
__mram float d_res[(M1_ROWS/NR_DPUS+1)*(M2_ROWS/NR_DPUS+2)];



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
	aux = kDot_m1_m2T(NR_TASKLETS, d_m1, d_m2, d_res, Ceil((float)M1_ROWS/NR_DPUS), M1_COLS, Ceil((((T)M2_ROWS/NR_DPUS)/2))*2, barr_addr);
	//*aux = kDot_m1_m2T(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	//*aux = kDot_m1T_m2(NR_TASKLETS, d_test1, d_test2, d_res, cache1, cache2, 60, 50, 40, barr_addr);
	
	//printf("\n\nceil val is: %d\n", Ceil((((T)M2_ROWS/NR_DPUS)/2))*2);
	/*for(int i = 0; i < (Ceil((float)M1_ROWS/NR_DPUS)); i++)
	{
		for(int j = 0; j < (Ceil((((T)M2_ROWS/NR_DPUS)/2))*2); j++)
		{
			printf("%.1f ", d_res[i*(Ceil((((T)M2_ROWS/NR_DPUS)/2))*2)+j]);
		}
		printf("\n");
	}*/

}
