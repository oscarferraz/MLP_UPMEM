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
	//const int TRAINING_SIZE = 4;
	//const int TRAINING_DIM = 4;
	const int L1_SIZE = 8;

	// X, the first 4 lines from Iris dataset
	float h_X[TRAINING_SIZE*TRAINING_DIM] = {5.1, 3.5, 1.4, 0.2,
												4.9, 3.0, 1.4, 0.2,
												6.2, 3.4, 5.4, 2.3,
												5.9, 3.0, 5.1, 1.8};

	float test_X[TRAINING_SIZE*TRAINING_DIM] = {0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0};


	const signed int X_size = sizeof(h_X);
	float *d_X;

	//WEIGHTS_0
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(float);
	float *h_W0 = (float*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    //printf("%.10f ", h_W0[i]);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(float);

	float* h_layer_1 = (float*)malloc(L1_size);
	float* h_layer_1_delta = (float*)malloc(L1_size);
	float* h_buffer = (float*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	//WEIGHTS_1
	const long signed int W1_size = L1_SIZE*sizeof(float);
	float *h_W1 = (float*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	    //printf("%f ", h_W1[i]);
	}

	//Y
	const int y_dim = 4;
	float h_Y[y_dim] = {	0,
						0,
						1,
						1 };
	const signed int y_size = sizeof(h_Y);

	//PRED AND PRED_DELTA
	float* h_pred = (float*)malloc(y_size);
	float* h_pred_delta = (float*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}

	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

    
    DPU_ASSERT(dpu_copy_to(set, "d_X", 0, h_X, sizeof(float) * TRAINING_SIZE*TRAINING_DIM));
    DPU_ASSERT(dpu_copy_to(set, "d_W0", 0, h_W0, W0_size));

    DPU_ASSERT(dpu_copy_to(set, "d_layer_1", 0, h_layer_1, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_buffer", 0, h_buffer, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_layer_1_delta", 0, h_layer_1_delta, L1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_W1", 0, h_W1, W1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_Y", 0, h_Y, y_size));

    DPU_ASSERT(dpu_copy_to(set, "d_pred", 0, h_pred, y_size));
	DPU_ASSERT(dpu_copy_to(set, "d_pred_delta", 0, h_pred_delta, y_size));

    //Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu)
	{
		//DPU_ASSERT(dpu_copy_from(dpu, "X_res", 0, test_X, sizeof(float) * TRAINING_SIZE*TRAINING_DIM));
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	/*for(int i = 1; i<=TRAINING_SIZE*TRAINING_DIM; i++)
	{
		printf("%f ", test_X[i-1]);
		if(i%4  == 0)
		{
			printf("\n");
		}
	}*/

}
