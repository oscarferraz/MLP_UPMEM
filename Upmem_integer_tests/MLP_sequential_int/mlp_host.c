#include <stdio.h>
#include <stdlib.h>
#include "common/common.h"
#include "upmem_kernels2.c"
#include <time.h>

#define test_size 4*28
#define test_pred_size 28


static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}


int main()
{
	srand(1);
	const int L1_SIZE = 8;

	// X, the first 4 lines from Iris dataset
	/*int h_X[TRAINING_SIZE*TRAINING_DIM] = {5.1, 3.5, 1.4, 0.2,
												4.9, 3.0, 1.4, 0.2,
												6.2, 3.4, 5.4, 2.3,
												5.9, 3.0, 5.1, 1.8};
	*/

	int h_X[TRAINING_SIZE*TRAINING_DIM] = {
5, 4, 1, 0,
5, 4, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 4, 1, 0,
5, 4, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 2, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
7, 3, 4, 1,
6, 3, 4, 1,
6, 3, 4, 1,
5, 2, 4, 1,
6, 2, 4, 1,
5, 2, 4, 1,
6, 3, 4, 1,
4, 2, 3, 1,
6, 2, 4, 1,
5, 2, 3, 1,
5, 2, 3, 1,
5, 3, 4, 1,
6, 2, 4, 1,
6, 2, 4, 1,
5, 2, 3, 1,
6, 3, 4, 1,
5, 3, 4, 1,
5, 2, 4, 1,
6, 2, 4, 1,
5, 2, 3, 1,
5, 3, 4, 1,
6, 2, 4, 1,
6, 2, 4, 1,
6, 2, 4, 1,
6, 2, 4, 1,
6, 3, 4, 1,
6, 2, 4, 1,
6, 3, 5, 1,
6, 2, 4, 1,
5, 2, 3, 1,
5, 2, 3, 1,
5, 2, 3, 1,
5, 2, 3, 1,
6, 2, 5, 1,
5, 3, 4, 1,
6, 3, 4, 1,
6, 3, 4, 1,
6, 2, 4, 1,
5, 3, 4, 1,
5, 2, 4, 1,
5, 2, 4, 1,
6, 3, 4, 1,
5, 2, 4, 1,
5, 2, 3, 1,
5, 2, 4, 1,
5, 3, 4, 1,
5, 2, 4, 1,
6, 2, 4, 1,
5, 2, 3, 1,
5, 2, 4, 1,
6, 3, 6, 2,
5, 2, 5, 1,
7, 3, 5, 2,
6, 2, 5, 1,
6, 3, 5, 2,
7, 3, 6, 2,
4, 2, 4, 1,
7, 2, 6, 1,
6, 2, 5, 1,
7, 3, 6, 2,
6, 3, 5, 2,
6, 2, 5, 1,
6, 3, 5, 2,
5, 2, 5, 2,
5, 2, 5, 2,
6, 3, 5, 2,
6, 3, 5, 1,
7, 3, 6, 2,
7, 2, 6, 2,
6, 2, 5, 1,
6, 3, 5, 2,
5, 2, 4, 2,
7, 2, 6, 2,
6, 2, 4, 1,
6, 3, 5, 2,
7, 3, 6, 1,
6, 2, 4, 1,
6, 3, 4, 1,
6, 2, 5, 2,
7, 3, 5, 1,
7, 2, 6, 1,
7, 3, 6, 2,
6, 2, 5, 2,
6, 2, 5, 1,
6, 2, 5, 1,
7, 3, 6, 2};

	/*int test_X[TRAINING_SIZE*TRAINING_DIM] = {0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.0};
	*/

	const signed int X_size = sizeof(h_X);
	int *d_X;

	//WEIGHTS_0
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(int);
	int *h_W0 = (int*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    //printf("%.10f ", h_W0[i]);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(int);

	int* h_layer_1 = (int*)malloc(L1_size);
	int* h_layer_1_delta = (int*)malloc(L1_size);
	int* h_buffer = (int*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	//WEIGHTS_1
	const long signed int W1_size = L1_SIZE*sizeof(int);
	int *h_W1 = (int*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	    //printf("%f ", h_W1[i]);
	}

	//Y
	/*int h_Y[4] = {	0,
						0,
						1,
						1 };*/
	int h_Y[122] = {
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1};
	const signed int y_size = sizeof(h_Y);

	//PRED AND PRED_DELTA
	int* h_pred = (int*)malloc(y_size);
	int* h_pred_delta = (int*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}



//test vector for KTest function
	int h_test[test_size] = {5, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
4, 2, 1, 0,
4, 3, 1, 0,
5, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
4, 3, 1, 0,
6, 3, 5, 2,
6, 3, 5, 1,
6, 3, 4, 1,
6, 3, 5, 2,
6, 3, 5, 2,
6, 3, 5, 2,
5, 2, 5, 1,
6, 3, 5, 2,
6, 3, 5, 2,
6, 3, 5, 2,
6, 2, 5, 1,
6, 3, 5, 2,
6, 3, 5, 2,
5, 3, 5, 1};

	
	int h_test_pred[test_pred_size] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	int h_test_l1[28*L1_SIZE];
	for(int i = 0; i < 28*L1_SIZE; i++)
	{
		h_test_l1[i] = 0;
	}
	 clock_t start, end;
     double cpu_time_used;
     
     start = clock();

	kFit(h_X,
		TRAINING_DIM, TRAINING_SIZE, h_Y, 1, h_layer_1, 8, h_layer_1_delta,
		h_pred, h_pred_delta, h_W0, h_W1, h_buffer, 0.1);


	kTest(h_test,
	4, 28, 1, h_test_l1, 8,
	h_test_pred, h_W0, h_W1);

	end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("MLP %f seconds to execute \n", cpu_time_used);

	return 0;
}
