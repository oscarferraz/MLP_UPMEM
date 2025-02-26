#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"


static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}



static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(1, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}


int main()
{
	srand(1);
	//const int TRAINING_SIZE = 4;
	//const int TRAINING_DIM = 4;
	const int L1_SIZE = 8;

	// X, the first 4 lines from Iris dataset
	int h_X[TRAINING_SIZE*TRAINING_DIM] = {5, 3, 1, 0,
												4, 3, 1, 0,
												6, 3, 5, 2,
												5, 3, 5, 1};


	const signed int X_size = sizeof(h_X);
	int *d_X;

	//WEIGHTS_0
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(int);
	int *h_W0 = (int*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = i;
	    //printf("%.10f ", h_W0[i]);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(int);

	int* h_layer_1 = (int*)malloc(L1_size);
	int* h_layer_1_delta = (int*)malloc(L1_size);
	int* h_buffer = (int*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	}

	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

    
    DPU_ASSERT(dpu_copy_to(set, "d_X", 0, h_X, sizeof(int) * TRAINING_SIZE*TRAINING_DIM));
    DPU_ASSERT(dpu_copy_to(set, "d_W0", 0, h_W0, W0_size));

    DPU_ASSERT(dpu_copy_to(set, "d_layer_1", 0, h_layer_1, L1_size));
 
    //Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu)
	{
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	return 0;
}
